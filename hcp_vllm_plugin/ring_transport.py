"""P2P ring TCP transport for the decode Q-ring.

Each ring node maintains:
  * a persistent TCP connection to its ring SUCCESSOR (send side)
  * a listener socket accepting exactly ONE connection from its PREDECESSOR
    (recv side)

Framing: 8-byte uint64 little-endian length prefix + payload (safetensors).

The transport is deliberately minimal — a clean send/recv abstraction that
can later be replaced by RDMA / CXL-class interconnect without touching the
decode logic above it.

Ring order convention:
  ring_addrs[0] -> ring_addrs[1] -> ... -> ring_addrs[N-1] -> ring_addrs[0].
The owner is typically ring_addrs[0] (the node that initiates decode packets).
"""

from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default TCP port for the decode ring transport (distinct from HTTP serve_port).
DEFAULT_DECODE_PORT = 8950

_HEADER = struct.Struct("<Q")  # 8-byte uint64 LE


def _decode_transport() -> str:
    return os.environ.get("HCP_RING_DECODE_TRANSPORT", "ring")


def _decode_port() -> int:
    return int(os.environ.get("HCP_RING_DECODE_PORT", str(DEFAULT_DECODE_PORT)))


class RingTransport:
    """P2P ring TCP transport: persistent successor connection + predecessor
    listener thread.

    Parameters
    ----------
    ring_addrs : list of (host, port)
        Ring-order addresses of ALL nodes (including self).
    my_index : int
        This node's position in ring_addrs.
    connect_timeout_s : float
        Timeout for connecting to successor (retried).
    """

    def __init__(
        self,
        ring_addrs: list[tuple[str, int]],
        my_index: int,
        connect_timeout_s: float = 120,
    ) -> None:
        self._ring_addrs = list(ring_addrs)
        self._n = len(ring_addrs)
        self._my_index = my_index
        self._connect_timeout = connect_timeout_s

        self._succ_addr = ring_addrs[(my_index + 1) % self._n]
        self._pred_addr = ring_addrs[(my_index - 1) % self._n]

        self._send_sock: socket.socket | None = None
        self._recv_conn: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._recv_queue: queue.Queue[bytes] = queue.Queue()
        self._running = False
        self._listener_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._started = threading.Event()

        # Stats
        self.packets_sent = 0
        self.packets_recv = 0

    @property
    def ring_size(self) -> int:
        return self._n

    @property
    def my_index(self) -> int:
        return self._my_index

    @property
    def is_owner(self) -> bool:
        """Convention: index 0 is the owner (decode loop driver)."""
        return self._my_index == 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bind listener, accept predecessor connection, connect to successor."""
        if self._running:
            return
        self._running = True

        # 1. Bind listener on my address
        my_host, my_port = self._ring_addrs[self._my_index]
        bind_host = "0.0.0.0" if my_host in ("0.0.0.0", "") else my_host
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((bind_host, my_port))
        self._listener.listen(1)
        self._listener.settimeout(1.0)
        logger.info(
            "RingTransport[%d]: listening on %s:%d (ring_size=%d)",
            self._my_index, bind_host, my_port, self._n,
        )

        # 2. Accept predecessor connection (background thread so we can
        #    simultaneously connect to successor — avoids deadlock when
        #    all nodes start at the same time).
        self._listener_thread = threading.Thread(
            target=self._accept_predecessor, daemon=True
        )
        self._listener_thread.start()

        # 3. Connect to successor (with retries — successor may not be up yet)
        self._connect_successor()

        # 4. Start recv pump thread
        self._recv_thread = threading.Thread(target=self._recv_pump, daemon=True)
        self._recv_thread.start()

        self._started.set()
        logger.info(
            "RingTransport[%d]: started (succ=%s:%d)",
            self._my_index, *self._succ_addr,
        )

    def wait_ready(self, timeout_s: float = 300) -> bool:
        return self._started.wait(timeout_s)

    def stop(self) -> None:
        self._running = False
        for s in (self._send_sock, self._recv_conn, self._listener):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._send_sock = None
        self._recv_conn = None
        self._listener = None

    # ------------------------------------------------------------------
    # Send / Recv
    # ------------------------------------------------------------------
    def send_packet(self, payload: bytes) -> None:
        """Send a length-prefixed packet to the ring successor."""
        header = _HEADER.pack(len(payload))
        with self._send_lock:
            sock = self._send_sock
            if sock is None:
                raise RuntimeError(
                    f"RingTransport[{self._my_index}]: no successor connection"
                )
            sock.sendall(header + payload)
        self.packets_sent += 1

    def recv_packet(self, timeout_s: float = 600) -> bytes:
        """Receive one packet from the predecessor (blocking with timeout)."""
        try:
            return self._recv_queue.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"RingTransport[{self._my_index}]: recv_packet timed out "
                f"after {timeout_s}s"
            )

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------
    def _accept_predecessor(self) -> None:
        """Accept exactly one connection from the ring predecessor."""
        while self._running:
            try:
                conn, addr = self._listener.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._recv_conn = conn
                logger.info(
                    "RingTransport[%d]: predecessor connected from %s:%d",
                    self._my_index, addr[0], addr[1],
                )
                return
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    time.sleep(0.1)
                return

    def _connect_successor(self) -> None:
        """Connect to ring successor with retries."""
        t0 = time.time()
        last_err = None
        while self._running and time.time() - t0 < self._connect_timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(10)
                sock.connect(self._succ_addr)
                sock.settimeout(None)
                self._send_sock = sock
                logger.info(
                    "RingTransport[%d]: connected to successor %s:%d",
                    self._my_index, *self._succ_addr,
                )
                return
            except (OSError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.5)
        raise RuntimeError(
            f"RingTransport[{self._my_index}]: failed to connect to successor "
            f"{self._succ_addr} within {self._connect_timeout}s: {last_err}"
        )

    def _recv_pump(self) -> None:
        """Continuously read length-prefixed packets from predecessor and
        enqueue them."""
        while self._running:
            conn = self._recv_conn
            if conn is None:
                time.sleep(0.05)
                continue
            try:
                header = self._recv_exact(conn, _HEADER.size)
                if header is None:
                    break
                (length,) = _HEADER.unpack(header)
                payload = self._recv_exact(conn, length)
                if payload is None:
                    break
                self._recv_queue.put(payload)
                self.packets_recv += 1
            except (OSError, ConnectionResetError):
                if self._running:
                    logger.warning(
                        "RingTransport[%d]: predecessor connection lost",
                        self._my_index,
                    )
                break

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        """Read exactly n bytes from sock; None on EOF/error."""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(min(n - len(buf), 65536))
            except (OSError, ConnectionResetError):
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)


# ---------------------------------------------------------------------------
# Module-level singleton (one ring transport per vLLM process)
# ---------------------------------------------------------------------------
_TRANSPORT: RingTransport | None = None


def get_ring_transport() -> RingTransport | None:
    return _TRANSPORT


def init_ring_transport(
    ring_addrs: list[tuple[str, int]], my_index: int
) -> RingTransport:
    """Initialize and start the module-level ring transport singleton."""
    global _TRANSPORT
    if _TRANSPORT is not None:
        return _TRANSPORT
    t = RingTransport(ring_addrs, my_index)
    t.start()
    _TRANSPORT = t
    return t


def shutdown_ring_transport() -> None:
    global _TRANSPORT
    if _TRANSPORT is not None:
        _TRANSPORT.stop()
        _TRANSPORT = None
