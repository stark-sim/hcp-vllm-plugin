"""Memory-split decode for the HCP ring: accumulator-ring distributed attention.

Design (blueprint-two-phase-ring-20260727, decision-no-star-transport-20260727):
decode preserves the same memory-splitting semantics as prefill — NO node may
hold the full KV, not even transiently, and that includes the DECODE GROWTH
(each generated token's KV is assigned round-robin by global position).
Q + accumulator (O, LSE) travel hop-by-hop along the P2P ring; every node
connects only to its ring predecessor/successor (linear topology, works on
partially-reachable networks).  Star HTTP decode was removed: it required
owner-to-all-peers connectivity, which breaks for N > 3.

This module contains the ring node (RingDecodeNode) and the owner-side
entry point (ring_decode_step).  The prefill KV store remains HTTP
(neighbor accumulate-forward); decode uses ONLY the P2P TCP ring.
"""

from __future__ import annotations

import os
import threading
import urllib.request

import safetensors.torch
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

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
