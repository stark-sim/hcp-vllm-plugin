"""Memory-split decode for the HCP ring: accumulator-ring distributed attention.

Design (decision-memory-split-decode-20260725, extended by
decision-decode-growth-shard-20260725): decode must preserve the same
memory-splitting semantics as prefill — NO node may hold the full KV, not
even transiently, and that includes the DECODE GROWTH: each newly generated
token's KV is assigned to a ring node (default policy: global position
round-robin, p mod N; HCP_RING_DECODE_GROWTH=owner keeps the legacy
all-on-owner behavior for comparison).  The owner of the tail chunk drives
the decode loop; per decode step, per layer:

  1. the owner computes the new token's Q and the LOCAL partial attention
     over its own chunk + its owned growth share + the current token
     (causal) -> accumulator (o, lse);
  2. the accumulator + Q travel along the ring (one RPC hop per peer);
     each peer computes the partial attention of Q against ITS OWN chunk
     snapshot + its growth share (non-causal) and folds it into the
     accumulator with the online-softmax merge;
  3. growth KV owed to a peer is piggybacked on the NEXT step's RPC to that
     peer (append-then-serve keeps the ordering with zero extra hops);
  4. after the last peer the accumulator is the globally exact attention
     output.  Hop cost is L x (P-1) RTT per token — slow by design in the
     experiment phase (direct evidence for the CXL-class interconnect
     requirement).

This module contains both sides:
  * PartialAttentionService — runs on EVERY ring node: loads this node's own
    chunk KV snapshot from its connector store (safetensors written during
    prefill) into GPU once, keeps an in-memory growth buffer for the decode
    tokens assigned to this node, then serves POST /partial_attn requests
    that merge a remote accumulator with the local partial attention.
  * merge_remote_partial — the owner-side client: one RPC hop of the
    accumulator ring (optionally carrying one growth token's KV for append).

Transport payload is safetensors over HTTP (consistent with the KV store);
messages are KB-sized (Q + accumulator + at most one token of growth KV), so
latency is dominated by pure link RTT — exactly the quantity the experiment
wants to expose.
"""

from __future__ import annotations

import os
import threading
import urllib.request

import safetensors.torch
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PARTIAL_PATH = "/partial_attn"

# Live services in this process (validation inspects growth stats here; the
# plugin logger does not reliably reach captured driver logs).
SERVICES: list["PartialAttentionService"] = []


def growth_assignee(pos: int, ring_size: int, policy: str = "rr") -> int:
    """Which ring node durably holds the KV of the decode token at global
    position `pos`.  Nodes are indexed in ring order: 0..N-2 are the owner's
    prefix peers (in peer_chunks order), N-1 is the owner itself.

    Policies (the planning-layer seam for future alternatives):
      * "rr"    — round-robin by global position, p mod N (default): growth
                  memory pressure splits evenly across the ring.
      * "owner" — everything stays on the owner (legacy owner-collapse;
                  kept for A/B comparison of locality vs. split).
    """
    if policy == "owner":
        return ring_size - 1
    return pos % ring_size


def _merge_pair(
    o1: torch.Tensor, lse1: torch.Tensor, o2: torch.Tensor, lse2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax merge of two partials.

    vLLM's merge_attn_states computes only the merged OUTPUT; the accumulator
    ring must also carry the merged LSE forward, computed explicitly here as
    log(exp(lse1) + exp(lse2)) in max-shifted form.
    """
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    out = torch.empty_like(o1)
    merge_attn_states(out, o1, lse1, o2, lse2)
    m = torch.maximum(lse1, lse2)
    lse = m + torch.log(torch.exp(lse1 - m) + torch.exp(lse2 - m))
    return out, lse


class PartialAttentionService:
    """Serves partial-attention merges over this node's own chunk snapshot(s).

    The chunk KV is loaded from the connector store (per-layer safetensors
    written during prefill) into device memory ONCE; the node therefore holds
    exactly its own chunk(s) during the whole decode phase.
    """

    def __init__(
        self,
        store_dir: str,
        chunk_keys: list[str],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._chunk_keys = list(chunk_keys)
        self._device = device
        self._dtype = dtype
        # (chunk_key, layer_name) -> (k, v)
        self._layers: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        # layer_name -> (k, v): decode-growth tokens assigned to THIS node,
        # appended via /partial_attn piggyback (durable for the decode phase;
        # PoC scope: one decode-ring request per service lifetime).
        self._growth: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._growth_appends = 0
        self._ready = threading.Event()
        self._calls = 0
        self._store_dir = store_dir
        SERVICES.append(self)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        t0 = __import__("time").time()
        for chunk_key in self._chunk_keys:
            chunk_dir = os.path.join(self._store_dir, chunk_key)
            # The chunk's store files appear only after this node's own
            # prefill completes (_READY marker); wait for it, then load.
            marker = os.path.join(chunk_dir, "_READY")
            waited = 0.0
            while not os.path.exists(marker) and waited < 3600:
                __import__("time").sleep(2)
                waited += 2
            for fname in sorted(os.listdir(chunk_dir)):
                if not fname.endswith(".safetensors"):
                    continue
                layer = fname[: -len(".safetensors")]
                t = safetensors.torch.load_file(
                    os.path.join(chunk_dir, fname), device=self._device
                )
                self._layers[(chunk_key, layer)] = (
                    t["k"].to(self._dtype).contiguous(),
                    t["v"].to(self._dtype).contiguous(),
                )
        total = sum(k.numel() * k.element_size() * 2 for k, _ in self._layers.values())
        logger.info(
            "PartialAttentionService: chunks %s loaded (%d layers total, %.1f MiB, %.1fs)",
            self._chunk_keys,
            len(self._layers),
            total / 2**20,
            __import__("time").time() - t0,
        )
        self._ready.set()

    def wait_ready(self, timeout_s: float = 600) -> bool:
        return self._ready.wait(timeout_s)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def growth_appends(self) -> int:
        return self._growth_appends

    def growth_tokens(self) -> int:
        """Decode-growth tokens held per layer (0 when no growth arrived)."""
        if not self._growth:
            return 0
        return next(iter(self._growth.values()))[0].shape[0]

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def handle(self, payload: bytes) -> bytes:
        """Append any piggybacked growth KV, then merge a remote accumulator
        with this node's partial (chunk snapshot + growth share)."""
        tensors = safetensors.torch.load(payload)
        chunk = bytes(tensors["chunk"].tolist()).decode()
        layer = bytes(tensors["layer"].tolist()).decode()
        q = tensors["q"].to(self._device, self._dtype)
        o_acc = tensors["o"].to(self._device, self._dtype)
        lse_acc = tensors["lse"].to(self._device, torch.float32)
        scale = float(tensors["scale"].item())
        if "growth_k" in tensors:
            gk = tensors["growth_k"].to(self._device, self._dtype)
            gv = tensors["growth_v"].to(self._device, self._dtype)
            prev = self._growth.get(layer)
            self._growth[layer] = (
                gk if prev is None else torch.cat([prev[0], gk], dim=0),
                gv if prev is None else torch.cat([prev[1], gv], dim=0),
            )
            self._growth_appends += 1
        k, v = self._layers[(chunk, layer)]
        g = self._growth.get(layer)
        if g is not None:
            k = torch.cat([k, g[0]], dim=0)
            v = torch.cat([v, g[1]], dim=0)
        from hcp_vllm_plugin.ring_triton_attn import ring_attn_with_lse

        o_loc, lse_loc = ring_attn_with_lse(q.contiguous(), k, v, scale, None)
        o, lse = _merge_pair(o_acc, lse_acc, o_loc, lse_loc)
        self._calls += 1
        if self._calls % 50 == 0:
            logger.info(
                "PartialAttentionService: served %d partial merges", self._calls
            )
        return safetensors.torch.save(
            {"o": o.contiguous(), "lse": lse.contiguous()}
        )


def make_handler(partial_service: "PartialAttentionService | None", directory: str):
    """HTTP handler: GET serves store files (like SimpleHTTPRequestHandler),
    POST /partial_attn merges an accumulator with the local chunk partial."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_POST(self):
            if self.path != PARTIAL_PATH or partial_service is None:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = self.rfile.read(length)
                out = partial_service.handle(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except Exception as e:  # noqa: BLE001
                logger.warning("partial_attn failed: %s", e)
                self.send_error(500, str(e))

    return Handler


def merge_remote_partial(
    peer_url: str,
    chunk: str,
    layer: str,
    q: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    growth_k: torch.Tensor | None = None,
    growth_v: torch.Tensor | None = None,
    timeout_s: float = 600,
    retries: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One accumulator-ring hop: ask `peer_url` to fold its chunk's partial
    attention of q into the (o, lse) accumulator; returns the merged pair.
    When growth_k/growth_v are given, the peer durably appends that decode
    token's KV to its growth buffer BEFORE serving (append-then-serve: the
    piggyback keeps the ordering with zero extra hops)."""
    payload_dict = {
        "chunk": torch.tensor(list(chunk.encode()), dtype=torch.uint8),
        "layer": torch.tensor(list(layer.encode()), dtype=torch.uint8),
        "q": q.detach().contiguous().cpu(),
        "o": o.detach().contiguous().cpu(),
        "lse": lse.detach().contiguous().cpu(),
        "scale": torch.tensor([scale], dtype=torch.float32),
    }
    if growth_k is not None and growth_v is not None:
        payload_dict["growth_k"] = growth_k.detach().contiguous().cpu()
        payload_dict["growth_v"] = growth_v.detach().contiguous().cpu()
    payload = safetensors.torch.save(payload_dict)
    req = urllib.request.Request(
        peer_url.rstrip("/") + PARTIAL_PATH,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                out = safetensors.torch.load(resp.read())
            return (
                out["o"].to(q.device, q.dtype),
                out["lse"].to(q.device, torch.float32),
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            __import__("time").sleep(0.2 * (attempt + 1))
    raise RuntimeError(
        f"merge_remote_partial failed after {retries} attempts "
        f"(peer={peer_url}, chunk={chunk}, layer={layer}): {last_err}"
    )


def peer_partial_ready(peer_url: str, timeout_s: float = 10) -> bool:
    """Readiness probe for a peer's partial-attention service (store server
    doubles as the transport; a 200 on / means the server is up)."""
    try:
        req = urllib.request.Request(peer_url.rstrip("/") + "/", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout_s):
            return True
    except Exception:
        return False


# ===========================================================================
# P2P Ring Decode (HCP_RING_DECODE_TRANSPORT=ring)
#
# True ring topology: Q + accumulator (O, LSE) travel hop-by-hop along the
# P2P ring.  Each node computes its local partial attention against its own
# chunk KV snapshot + growth share, merges into the accumulator, and forwards
# the updated packet to its ring successor.  The owner initiates the packet
# and receives the final globally-exact result after one full ring traversal.
#
# This replaces the star-HTTP PartialAttentionService for decode while
# preserving identical math (online softmax merge) and memory-splitting
# semantics (each node holds only its own chunk + growth share).
# ===========================================================================

RING_NODES: list["RingDecodeNode"] = []


class RingDecodeNode:
    """A ring node for P2P decode: holds this node's chunk KV snapshot +
    growth buffer, processes incoming accumulator packets (compute partial
    attention + merge), and forwards to the ring successor.

    For NON-OWNER nodes: a background worker thread continuously receives
    packets from the predecessor, processes them, and forwards — pure relay,
    no vLLM engine involvement.

    For the OWNER node: the worker thread is NOT started; instead the owner
    calls `ring_decode_step()` synchronously from within the vLLM forward
    pass (initiate packet, block until it returns from the ring).
    """

    def __init__(
        self,
        store_dir: str,
        chunk_keys: list[str],
        ring_index: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._chunk_keys = list(chunk_keys)
        self._device = device
        self._dtype = dtype
        self._ring_index = ring_index
        # (chunk_key, layer_name) -> (k, v)
        self._layers: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        # layer_name -> (k, v): decode-growth tokens assigned to THIS node
        self._growth: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._growth_appends = 0
        self._ready = threading.Event()
        self._calls = 0
        self._store_dir = store_dir
        self._worker_thread: threading.Thread | None = None
        self._running = False
        RING_NODES.append(self)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        t0 = __import__("time").time()
        for chunk_key in self._chunk_keys:
            chunk_dir = os.path.join(self._store_dir, chunk_key)
            marker = os.path.join(chunk_dir, "_READY")
            waited = 0.0
            while not os.path.exists(marker) and waited < 3600:
                __import__("time").sleep(2)
                waited += 2
            for fname in sorted(os.listdir(chunk_dir)):
                if not fname.endswith(".safetensors"):
                    continue
                layer = fname[: -len(".safetensors")]
                t = safetensors.torch.load_file(
                    os.path.join(chunk_dir, fname), device=self._device
                )
                self._layers[(chunk_key, layer)] = (
                    t["k"].to(self._dtype).contiguous(),
                    t["v"].to(self._dtype).contiguous(),
                )
        total = sum(k.numel() * k.element_size() * 2 for k, _ in self._layers.values())
        logger.info(
            "RingDecodeNode[%d]: chunks %s loaded (%d layers, %.1f MiB, %.1fs)",
            self._ring_index,
            self._chunk_keys,
            len(self._layers),
            total / 2**20,
            __import__("time").time() - t0,
        )
        self._ready.set()

    def wait_ready(self, timeout_s: float = 600) -> bool:
        return self._ready.wait(timeout_s)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def growth_appends(self) -> int:
        return self._growth_appends

    def growth_tokens(self) -> int:
        if not self._growth:
            return 0
        return next(iter(self._growth.values()))[0].shape[0]

    def is_ready(self) -> bool:
        return self._ready.is_set()

    # ------------------------------------------------------------------
    # Packet processing (shared by owner and relay paths)
    # ------------------------------------------------------------------
    def handle_packet(self, payload: bytes) -> bytes:
        """Process one accumulator packet: append piggyback growth, compute
        local partial attention, merge into accumulator, return updated
        packet payload."""
        tensors = safetensors.torch.load(payload)
        chunk = bytes(tensors["chunk"].tolist()).decode()
        layer = bytes(tensors["layer"].tolist()).decode()
        q = tensors["q"].to(self._device, self._dtype)
        o_acc = tensors["o"].to(self._device, self._dtype)
        lse_acc = tensors["lse"].to(self._device, torch.float32)
        scale = float(tensors["scale"].item())

        # Append piggyback growth KV ONLY if this node is the target.
        # The packet carries growth_target (transport ring index of the
        # assignee); intermediate nodes forward it unchanged.
        growth_consumed = False
        if "growth_k" in tensors:
            target = int(tensors["growth_target"].item()) if "growth_target" in tensors else self._ring_index
            if target == self._ring_index:
                gk = tensors["growth_k"].to(self._device, self._dtype)
                gv = tensors["growth_v"].to(self._device, self._dtype)
                prev = self._growth.get(layer)
                self._growth[layer] = (
                    gk if prev is None else torch.cat([prev[0], gk], dim=0),
                    gv if prev is None else torch.cat([prev[1], gv], dim=0),
                )
                self._growth_appends += 1
                growth_consumed = True

        # Local partial attention: chunk snapshot + growth (non-causal).
        # In ring mode each node uses its OWN chunk key (the packet's chunk
        # field is the first peer's key — irrelevant for other nodes).
        my_chunk = self._chunk_keys[0] if self._chunk_keys else chunk
        k, v = self._layers[(my_chunk, layer)]
        g = self._growth.get(layer)
        if g is not None:
            k = torch.cat([k, g[0]], dim=0)
            v = torch.cat([v, g[1]], dim=0)

        from hcp_vllm_plugin.ring_triton_attn import ring_attn_with_lse

        o_loc, lse_loc = ring_attn_with_lse(q.contiguous(), k, v, scale, None)
        o, lse = _merge_pair(o_acc, lse_acc, o_loc, lse_loc)
        self._calls += 1
        if self._calls % 50 == 0:
            logger.info(
                "RingDecodeNode[%d]: served %d partial merges",
                self._ring_index, self._calls,
            )

        # Build updated packet (preserve q, scale, chunk, layer for next hop)
        out_dict = {
            "chunk": tensors["chunk"],
            "layer": tensors["layer"],
            "q": q.detach().contiguous().cpu(),
            "o": o.detach().contiguous().cpu(),
            "lse": lse.detach().contiguous().cpu(),
            "scale": tensors["scale"],
        }
        # Forward growth if it was NOT consumed by this node (destined for
        # a downstream ring neighbor).
        if "growth_k" in tensors and not growth_consumed:
            out_dict["growth_k"] = tensors["growth_k"]
            out_dict["growth_v"] = tensors["growth_v"]
            if "growth_target" in tensors:
                out_dict["growth_target"] = tensors["growth_target"]
        return safetensors.torch.save(out_dict)

    # ------------------------------------------------------------------
    # Relay worker thread (non-owner nodes only)
    # ------------------------------------------------------------------
    def start_relay(self, transport) -> None:
        """Start the background relay loop: recv -> handle -> forward."""
        if self._running:
            return
        self._running = True
        self._transport = transport
        self._worker_thread = threading.Thread(
            target=self._relay_loop, daemon=True
        )
        self._worker_thread.start()
        logger.info("RingDecodeNode[%d]: relay worker started", self._ring_index)

    def stop_relay(self) -> None:
        self._running = False

    def _relay_loop(self) -> None:
        """Wait for the chunk snapshot, then continuously receive packets,
        process, and forward.  The store readiness wait happens in THIS
        thread (never in connector init): the store files appear only after
        this node's own prefill, long after engine construction; packets
        arrive even later (owner decode), so blocking here is safe."""
        if not self.wait_ready(timeout_s=3600):
            logger.warning(
                "RingDecodeNode[%d]: chunk store never became ready; relay "
                "exiting", self._ring_index,
            )
            return
        transport = self._transport
        while self._running:
            try:
                payload = transport.recv_packet(timeout_s=1.0)
            except TimeoutError:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(
                        "RingDecodeNode[%d]: relay recv error: %s",
                        self._ring_index, e,
                    )
                break
            try:
                out = self.handle_packet(payload)
                transport.send_packet(out)
            except Exception as e:
                logger.warning(
                    "RingDecodeNode[%d]: relay handle/forward error: %s",
                    self._ring_index, e,
                )


def ring_decode_step(
    transport,
    q: torch.Tensor,
    o_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    scale: float,
    layer: str,
    chunk: str,
    growth_k: torch.Tensor | None = None,
    growth_v: torch.Tensor | None = None,
    growth_target: int = -1,
    timeout_s: float = 600,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owner-side entry: initiate one accumulator packet into the ring and
    block until it returns after traversing all peers.

    Returns the final (O, LSE) — the globally exact attention output for
    this layer's decode step.
    """
    payload_dict = {
        "chunk": torch.tensor(list(chunk.encode()), dtype=torch.uint8),
        "layer": torch.tensor(list(layer.encode()), dtype=torch.uint8),
        "q": q.detach().contiguous().cpu(),
        "o": o_acc.detach().contiguous().cpu(),
        "lse": lse_acc.detach().contiguous().cpu(),
        "scale": torch.tensor([scale], dtype=torch.float32),
    }
    if growth_k is not None and growth_v is not None:
        payload_dict["growth_k"] = growth_k.detach().contiguous().cpu()
        payload_dict["growth_v"] = growth_v.detach().contiguous().cpu()
        # growth_target: transport ring index of the assignee node.
        # Caller passes it via the growth_target kwarg (default: -1 means
        # "first node that matches" — but we always set it explicitly).
        payload_dict["growth_target"] = torch.tensor(
            [growth_target], dtype=torch.int64
        )

    payload = safetensors.torch.save(payload_dict)
    transport.send_packet(payload)
    result_bytes = transport.recv_packet(timeout_s=timeout_s)
    out = safetensors.torch.load(result_bytes)
    return (
        out["o"].to(q.device, q.dtype),
        out["lse"].to(q.device, torch.float32),
    )
