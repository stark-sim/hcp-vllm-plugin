"""HCP ring-KV connector for vLLM 0.23 (V1 engine) — transient peer-KV staging.

This connector implements the HCP *memory-splitting* context-parallel pattern,
which is deliberately DIFFERENT from the stock disaggregated-prefill semantics
(``HcpCpConnector`` / standard P/D KV transfer):

  * Stock semantics: a request's whole KV is computed on one node and copied
    wholesale into the other node's PERMANENT paged cache (full-KV transfer).
  * HCP ring semantics (this connector): each worker permanently holds ONLY
    its own chunk's KV.  The scheduler side of this connector marks the
    earlier chunk as "externally computed" via ``get_num_new_matched_tokens``
    (giving the later chunk global RoPE positions and stopping this worker
    from recomputing it), but the peer chunk's KV is NEVER loaded into the
    paged pool / block table.  Instead the worker side fetches it over HTTP
    and writes it into ``hcp_vllm_plugin.ring_backend.PEER_KV_STAGING`` — a
    transient staging dict that ``HcpRingAttentionImpl`` reads during
    attention (online-softmax merge) and that can be discarded after use.

Roles (``kv_connector_extra_config``):
  - ``ring_role``: "producer" | "consumer"
      producer: prompt is its own chunk; computes it, saves per-layer K/V to
                its store and serves the store over HTTP (worker side only).
      consumer: prompt is the FULL sequence; the prefix (earlier chunk) is
                marked external so only the suffix is computed; peer KV is
                fetched from the producer into the ring backend's staging.

Config keys (``kv_connector_extra_config``):
  - ``ring_role``             : "producer" | "consumer"
  - ``ring_shared_path``      : local store dir (this instance's side)
  - ``ring_run_id``           : session id namespacing the store
  - ``ring_chunk_id``         : this instance's own chunk key (producer)
  - ``ring_prefix_chunk_ids`` : comma-separated prefix chunk keys (consumer;
                                PoC supports exactly one)
  - ``ring_prefix_len``       : number of prefix (peer chunk) tokens (consumer)
  - ``ring_peer_url``         : producer base URL, e.g. "http://127.0.0.1:8901"
                                (consumer; empty => local shared-path read)
  - ``ring_serve_port``       : HTTP port to serve the store on (producer)
  - ``ring_load_timeout_s``   : HTTP fetch timeout (consumer)

Per-request overrides (multi-request / continuous batching):
  Each request may carry ``SamplingParams(extra_args={"kv_transfer_params":
  {"hcp_ring": {...}}})`` with keys ``chunk_id``, ``prefix_len``, ``peer_url``.
  These take precedence over the global extra-config values, so concurrent
  requests can reference DIFFERENT peer chunks.  The worker stages each chunk
  under its own key and binds it to the request via the request's first
  block-table block id (see ``ring_backend.PEER_REQ_MAP``); staged KV and the
  mapping are freed when the request finishes.

Both sides are expected to run with ``--attention-backend CUSTOM`` so that
``HcpRingAttentionBackend`` performs the local(causal) + peer(transient,
non-causal) merge.  ``HCP_RING_SPLIT_TOKENS`` is only a fallback on the
consumer; the staged dict takes precedence.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from hcp_vllm_plugin.ring_backend import (
    drop_chunk_kv,
    map_request_peer,
    stage_peer_kv,
    unmap_request_peer,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _align_down(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


def _ring_params_from_sampling(sampling_params: Any) -> dict[str, Any]:
    """Extract per-request ring params from SamplingParams.extra_args."""
    if sampling_params is None:
        return {}
    extra = getattr(sampling_params, "extra_args", None) or {}
    kv_params = extra.get("kv_transfer_params") or {}
    return dict(kv_params.get("hcp_ring") or {})


@dataclass
class RingReqMeta:
    is_store: bool
    chunk_key: str
    # Store side only: paged-pool slots of this instance's own chunk tokens.
    slot_mapping: torch.Tensor | None = None
    # Load side only: request binding + per-request fetch parameters.
    req_id: str = ""
    first_block_id: int = -1
    peer_url: str = ""
    prefix_len: int = 0


@dataclass
class HcpRingConnectorMetadata(KVConnectorMetadata):
    requests: list[RingReqMeta] = field(default_factory=list)


class HcpRingKvConnector(KVConnectorBase_V1):
    """Ring-KV connector: external-prefix scheduling + transient KV staging."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        cfg = self._kv_transfer_config
        self._block_size = vllm_config.cache_config.block_size
        self._ring_role = cfg.get_from_extra_config("ring_role", "producer")
        self._shared_path = cfg.get_from_extra_config(
            "ring_shared_path", "/tmp/hcp_ring_kv"
        )
        self._run_id = cfg.get_from_extra_config("ring_run_id", "run")
        self._chunk_id = cfg.get_from_extra_config("ring_chunk_id", "chunk0")
        prefix_ids = cfg.get_from_extra_config("ring_prefix_chunk_ids", "")
        self._prefix_chunk_ids = [s for s in prefix_ids.split(",") if s]
        self._prefix_len = int(cfg.get_from_extra_config("ring_prefix_len", 0))
        self._load_timeout_s = float(
            cfg.get_from_extra_config("ring_load_timeout_s", 600)
        )
        self._peer_url = cfg.get_from_extra_config("ring_peer_url", "").rstrip("/")
        self._serve_port = int(cfg.get_from_extra_config("ring_serve_port", 0))

        self._requests_need_load: dict[str, "Request"] = {}
        self._saved_tokens: dict[str, int] = {}
        self._saved_layers: dict[str, int] = {}
        # Worker-side live-request bookkeeping for cleanup:
        #   req_id -> (chunk_key, first_block_id); chunk_key -> refcount
        self._live: dict[str, tuple[str, int]] = {}
        self._chunk_refs: dict[str, int] = {}
        os.makedirs(self._run_dir(), exist_ok=True)
        # Only the producer's WORKER-side connector holds/serves KV; the
        # scheduler-side instance must not bind the HTTP port.
        if (
            self._ring_role == "producer"
            and self._serve_port > 0
            and self._role == KVConnectorRole.WORKER
        ):
            self._start_http_server()
        logger.info(
            "HcpRingKvConnector init: role=%s chunk=%s prefix=%s(%d) path=%s "
            "peer=%s serve=%d",
            self._ring_role,
            self._chunk_id,
            self._prefix_chunk_ids,
            self._prefix_len,
            self._run_dir(),
            self._peer_url,
            self._serve_port,
        )

    # ------------------------------------------------------------------
    # HTTP transport (same pattern as HcpCpConnector)
    # ------------------------------------------------------------------
    def _start_http_server(self) -> None:
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        handler = partial(SimpleHTTPRequestHandler, directory=self._shared_path)
        server = ThreadingHTTPServer(("0.0.0.0", self._serve_port), handler)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self._http_server = server
        logger.info(
            "HcpRingKvConnector serving KV store on 0.0.0.0:%d", self._serve_port
        )

    def _fetch(self, rel_path: str, dest_path: str, peer_url: str = "") -> bool:
        peer = (peer_url or self._peer_url).rstrip("/")
        url = f"{peer}/{rel_path}"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        last_err = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(
                    url, timeout=self._load_timeout_s
                ) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
                return True
            except Exception as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        logger.warning("fetch %s failed after retries: %s", url, last_err)
        return False

    def _remote_exists(self, rel_path: str, peer_url: str = "") -> bool:
        peer = (peer_url or self._peer_url).rstrip("/")
        url = f"{peer}/{rel_path}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10):
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _run_dir(self) -> str:
        return os.path.join(self._shared_path, self._run_id)

    def _chunk_dir(self, chunk_key: str) -> str:
        return os.path.join(self._run_dir(), chunk_key)

    def _layer_file(self, chunk_key: str, layer_name: str) -> str:
        return os.path.join(self._chunk_dir(chunk_key), f"{layer_name}.safetensors")

    def _ready_marker(self, chunk_key: str) -> str:
        return os.path.join(self._chunk_dir(chunk_key), "_READY")

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self._ring_role != "consumer":
            return 0, False

        # Per-request params (kv_transfer_params.hcp_ring) override globals,
        # so concurrent requests may reference different peer chunks.
        rp = (getattr(request, "kv_transfer_params", None) or {}).get(
            "hcp_ring"
        ) or {}
        prefix_len = int(rp.get("prefix_len") or self._prefix_len)
        if prefix_len == 0:
            return 0, False
        chunk_key = rp.get("chunk_id") or (
            self._prefix_chunk_ids[0] if self._prefix_chunk_ids else ""
        )
        if not chunk_key:
            return 0, False
        peer_url = (rp.get("peer_url") or "").rstrip("/")

        # Wait until the peer chunk's KV is fully written by the producer.
        if not self._prefix_ready(chunk_key, peer_url):
            # None => scheduler asks again later (async stall).
            return None, True

        external = _align_down(prefix_len, self._block_size)
        external = max(external - num_computed_tokens, 0)
        # "Synchronous load": the scheduler may compute the suffix this step.
        # NOTE: unlike stock connectors, nothing is loaded into the paged
        # pool — the worker side stages the peer KV transiently instead.
        return external, False

    def _prefix_ready(self, chunk_key: str, peer_url: str = "") -> bool:
        if peer_url or self._peer_url:
            rel = os.path.join(self._run_id, chunk_key, "_READY")
            return self._remote_exists(rel, peer_url)
        return os.path.exists(self._ready_marker(chunk_key))

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = HcpRingConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                # Consumer: stage the peer chunk KV (no pool slot mapping —
                # we deliberately do NOT load into the block table).
                rp = _ring_params_from_sampling(new_req.sampling_params)
                chunk_key = rp.get("chunk_id") or (
                    self._prefix_chunk_ids[0] if self._prefix_chunk_ids else ""
                )
                meta.requests.append(
                    RingReqMeta(
                        is_store=False,
                        chunk_key=chunk_key,
                        req_id=new_req.req_id,
                        first_block_id=new_req.block_ids[0][0],
                        peer_url=(rp.get("peer_url") or "").rstrip("/"),
                        prefix_len=int(rp.get("prefix_len") or self._prefix_len),
                    )
                )
            elif self._ring_role == "producer":
                # Producer: save this instance's own chunk KV after compute.
                token_ids = new_req.prompt_token_ids or []
                block_ids = list(new_req.block_ids[0])
                rp = _ring_params_from_sampling(new_req.sampling_params)
                chunk_key = rp.get("chunk_id") or self._chunk_id
                meta.requests.append(
                    RingReqMeta(
                        is_store=True,
                        chunk_key=chunk_key,
                        slot_mapping=self._make_slot_mapping(token_ids, block_ids),
                    )
                )
        self._requests_need_load.clear()
        return meta

    def _make_slot_mapping(
        self, token_ids: list[int], block_ids: list[int]
    ) -> torch.Tensor:
        n = _align_down(len(token_ids), self._block_size)
        block_ids_t = torch.tensor(block_ids[: (n // self._block_size)])
        num_blocks = block_ids_t.shape[0]
        offsets = torch.arange(0, self._block_size)
        slot_mapping = (
            offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()[:n]
        return slot_mapping

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Fetch each request's peer chunk per-layer KV into the ring backend's
        TRANSIENT staging dict (keyed by chunk), bind the request to its chunk
        via its first block-table block id, and track it for cleanup on finish.
        Never writes to the paged pool/block table."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HcpRingConnectorMetadata)
        need = [r for r in metadata.requests if not r.is_store]
        if not need:
            return
        for req in need:
            expected = _align_down(
                req.prefix_len or self._prefix_len, self._block_size
            )
            if req.chunk_key not in self._chunk_refs:
                # First live request referencing this chunk: stage all layers.
                staged = 0
                total_bytes = 0
                for layer_name, layer in forward_context.no_compile_layers.items():
                    kvc = getattr(layer, "kv_cache", None)
                    device = (
                        str(kvc.device)
                        if torch.is_tensor(kvc) and kvc.numel() > 0
                        else "cuda"
                    )
                    tensors = self._load_layer(
                        req.chunk_key, layer_name, device, req.peer_url
                    )
                    if tensors is None:
                        raise RuntimeError(
                            "HcpRingKvConnector: peer KV fetch failed for "
                            f"{req.chunk_key}/{layer_name} "
                            f"(peer={req.peer_url or self._peer_url})"
                        )
                    k, v = tensors
                    if k.shape[0] != expected:
                        raise RuntimeError(
                            f"HcpRingKvConnector: staged peer KV length "
                            f"{k.shape[0]} != expected external prefix "
                            f"{expected} for {layer_name}"
                        )
                    stage_peer_kv(req.chunk_key, layer_name, k, v)
                    staged += 1
                    total_bytes += k.numel() * k.element_size() * 2
                logger.info(
                    "HcpRingKvConnector: staged peer chunk %s KV for %d layers "
                    "(%d tokens/layer, %.1f MiB over %s) into TRANSIENT staging "
                    "(paged pool untouched)",
                    req.chunk_key,
                    staged,
                    expected,
                    total_bytes / 2**20,
                    req.peer_url or self._peer_url or "local store",
                )
            # Bind this request to its chunk and track it for cleanup.
            map_request_peer(req.first_block_id, req.chunk_key)
            self._live[req.req_id] = (req.chunk_key, req.first_block_id)
            self._chunk_refs[req.chunk_key] = (
                self._chunk_refs.get(req.chunk_key, 0) + 1
            )

    def _load_layer(
        self, chunk_key: str, layer_name: str, device: str, peer_url: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        fname = self._layer_file(chunk_key, layer_name)
        if peer_url or self._peer_url:
            rel = os.path.join(self._run_id, chunk_key, f"{layer_name}.safetensors")
            if not self._fetch(rel, fname, peer_url):
                return None
        if not os.path.exists(fname):
            return None
        t = safetensors.torch.load_file(fname, device=device)
        return t["k"].contiguous(), t["v"].contiguous()

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Bulk staging happens in start_load_kv (before the forward pass).
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HcpRingConnectorMetadata):
            return
        for req in metadata.requests:
            if not req.is_store:
                continue
            fname = self._layer_file(req.chunk_key, layer_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            sm = req.slot_mapping
            block_idxs = sm // self._block_size
            offsets = sm % self._block_size
            kv = kv_layer[block_idxs, :, offsets]  # [n, 2, HKV, D]
            safetensors.torch.save_file(
                {
                    "k": kv[:, 0].detach().cpu().contiguous(),
                    "v": kv[:, 1].detach().cpu().contiguous(),
                },
                fname,
            )
            self._saved_tokens[req.chunk_key] = sm.shape[0]
            self._saved_layers[req.chunk_key] = (
                self._saved_layers.get(req.chunk_key, 0) + 1
            )

    def wait_for_save(self) -> None:
        # Mark this instance's chunk as fully written so the consumer can
        # start fetching.
        metadata = self._get_connector_metadata()
        if isinstance(metadata, HcpRingConnectorMetadata):
            for req in metadata.requests:
                if req.is_store:
                    os.makedirs(self._chunk_dir(req.chunk_key), exist_ok=True)
                    with open(self._ready_marker(req.chunk_key), "w") as f:
                        json.dump(
                            {
                                "tokens": self._saved_tokens.get(req.chunk_key, 0),
                                "layers": self._saved_layers.get(req.chunk_key, 0),
                                "ts": time.time(),
                            },
                            f,
                        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Loads/saves are synchronous within the step; nothing outstanding.
        # Free transient staged KV of finished requests (refcounted per chunk).
        for req_id in finished_req_ids:
            live = self._live.pop(req_id, None)
            if live is None:
                continue
            chunk_key, first_block_id = live
            unmap_request_peer(first_block_id)
            refs = self._chunk_refs.get(chunk_key, 0) - 1
            if refs <= 0:
                self._chunk_refs.pop(chunk_key, None)
                dropped = drop_chunk_kv(chunk_key)
                logger.info(
                    "HcpRingKvConnector: freed staged chunk %s (%d layers)",
                    chunk_key,
                    dropped,
                )
            else:
                self._chunk_refs[chunk_key] = refs
        return None, None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return None

    def shutdown(self) -> None:
        return
