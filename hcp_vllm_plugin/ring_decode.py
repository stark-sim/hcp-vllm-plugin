"""Memory-split decode for the HCP ring: accumulator-ring distributed attention.

Design (decision-memory-split-decode-20260725): decode must preserve the same
memory-splitting semantics as prefill — NO node may hold the full KV, not
even transiently.  The owner of the tail chunk drives the decode loop; per
decode step, per layer:

  1. the owner computes the new token's Q and the LOCAL partial attention
     over its own chunk + decode growth (causal) -> accumulator (o, lse);
  2. the accumulator + Q travel along the ring (one RPC hop per peer);
     each peer computes the partial attention of Q against ITS OWN chunk
     snapshot (non-causal) and folds it into the accumulator with the
     online-softmax merge;
  3. after the last peer the accumulator is the globally exact attention
     output.  Hop cost is L x (P-1) RTT per token — slow by design in the
     experiment phase (direct evidence for the CXL-class interconnect
     requirement).

This module contains both sides:
  * PartialAttentionService — runs on EVERY ring node: loads this node's own
    chunk KV snapshot from its connector store (safetensors written during
    prefill) into GPU once, then serves POST /partial_attn requests that
    merge a remote accumulator with the local chunk's partial attention.
  * merge_remote_partial — the owner-side client: one RPC hop of the
    accumulator ring.

Transport payload is safetensors over HTTP (consistent with the KV store);
messages are KB-sized (Q + accumulator), so latency is dominated by pure
link RTT — exactly the quantity the experiment wants to expose.
"""

from __future__ import annotations

import io
import os
import threading
import urllib.request

import safetensors.torch
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PARTIAL_PATH = "/partial_attn"


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
        self._ready = threading.Event()
        self._calls = 0
        self._store_dir = store_dir
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

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def handle(self, payload: bytes) -> bytes:
        """Merge a remote accumulator with this chunk's partial attention."""
        tensors = safetensors.torch.load(io.BytesIO(payload))
        chunk = bytes(tensors["chunk"].tolist()).decode()
        layer = bytes(tensors["layer"].tolist()).decode()
        q = tensors["q"].to(self._device, self._dtype)
        o_acc = tensors["o"].to(self._device, self._dtype)
        lse_acc = tensors["lse"].to(self._device, torch.float32)
        scale = float(tensors["scale"].item())
        k, v = self._layers[(chunk, layer)]
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
    timeout_s: float = 600,
    retries: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One accumulator-ring hop: ask `peer_url` to fold its chunk's partial
    attention of q into the (o, lse) accumulator; returns the merged pair."""
    payload = safetensors.torch.save(
        {
            "chunk": torch.tensor(list(chunk.encode()), dtype=torch.uint8),
            "layer": torch.tensor(list(layer.encode()), dtype=torch.uint8),
            "q": q.detach().contiguous().cpu(),
            "o": o.detach().contiguous().cpu(),
            "lse": lse.detach().contiguous().cpu(),
            "scale": torch.tensor([scale], dtype=torch.float32),
        }
    )
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
                out = safetensors.torch.load(io.BytesIO(resp.read()))
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
