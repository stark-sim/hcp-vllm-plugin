"""HCP context-parallel block-KV ring connector for vLLM (V1 engine).

This connector lets several vLLM instances cooperate on one long sequence via
*context-passing* prefill: each instance computes only its own contiguous
chunk of the prompt and loads the earlier chunks' KV from the previous
instance.  It is deliberately shaped after vLLM's ``ExampleConnector`` so it
tracks the stable ``KVConnectorBase_V1`` API instead of patching vLLM.

Roles (set via ``kv_connector_extra_config``):
- ``producer``: computes its own chunk and stores its KV so the next instance
  can load it.  The producer's prompt is just its chunk (no prefix).
- ``consumer``: its prompt is ``prefix + own_chunk``; the ``prefix`` tokens are
  marked externally-computed (loaded from the producer) so the model only
  computes ``own_chunk``.

Config keys (``kv_connector_extra_config``):
- ``cp_role``: "producer" | "consumer" | "both"
- ``cp_shared_path``: directory of the shared KV store (single-machine transport)
- ``cp_run_id``: session id namespacing the store
- ``cp_chunk_id``: this instance's own chunk key (producer save key)
- ``cp_prefix_chunk_ids``: comma-separated chunk keys that form this instance's
  prefix, in order (consumer load keys)
- ``cp_prefix_len``: total number of prefix tokens (consumer)

The shared-path transport is for single-machine validation.  A network
transport (TCP) for cross-node runs plugs into the same save/load helpers.
"""

from __future__ import annotations

import os
import time
import threading
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

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _align_down(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


@dataclass
class ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    chunk_key: str


@dataclass
class HcpCpConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)


class HcpCpConnector(KVConnectorBase_V1):
    """Context-passing CP connector (shared-path transport)."""

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
        self._cp_role = cfg.get_from_extra_config("cp_role", "producer")
        self._shared_path = cfg.get_from_extra_config("cp_shared_path", "/tmp/hcp_cp")
        self._run_id = cfg.get_from_extra_config("cp_run_id", "run")
        self._chunk_id = cfg.get_from_extra_config("cp_chunk_id", "chunk0")
        prefix_ids = cfg.get_from_extra_config("cp_prefix_chunk_ids", "")
        self._prefix_chunk_ids = [s for s in prefix_ids.split(",") if s]
        self._prefix_len = int(cfg.get_from_extra_config("cp_prefix_len", 0))
        self._load_timeout_s = float(cfg.get_from_extra_config("cp_load_timeout_s", 600))
        # Network transport: consumer pulls prefix KV from the producer's HTTP
        # server (cp_peer_url, e.g. "http://100.118.253.68:8899"); producer
        # serves its store on cp_serve_port.  Empty peer_url => local shared path.
        self._peer_url = cfg.get_from_extra_config("cp_peer_url", "").rstrip("/")
        self._serve_port = int(cfg.get_from_extra_config("cp_serve_port", 0))

        self._requests_need_load: dict[str, Request] = {}
        os.makedirs(self._run_dir(), exist_ok=True)
        if self._cp_role == "producer" and self._serve_port > 0:
            self._start_http_server()
        logger.info(
            "HcpCpConnector init: role=%s chunk=%s prefix=%s(%d) path=%s peer=%s serve=%d",
            self._cp_role, self._chunk_id, self._prefix_chunk_ids,
            self._prefix_len, self._run_dir(), self._peer_url, self._serve_port,
        )

    def _start_http_server(self) -> None:
        """Serve the shared store over HTTP so remote consumers can pull KV."""
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        from functools import partial

        handler = partial(SimpleHTTPRequestHandler, directory=self._shared_path)
        server = ThreadingHTTPServer(("0.0.0.0", self._serve_port), handler)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self._http_server = server
        logger.info("HcpCpConnector serving KV store on 0.0.0.0:%d", self._serve_port)

    def _fetch(self, rel_path: str, dest_path: str) -> bool:
        """Download rel_path from the peer store to dest_path. Returns success."""
        url = f"{self._peer_url}/{rel_path}"
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with urllib.request.urlopen(url, timeout=self._load_timeout_s) as resp, \
                    open(dest_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception as e:
            logger.warning("fetch %s failed: %s", url, e)
            return False

    def _remote_exists(self, rel_path: str) -> bool:
        url = f"{self._peer_url}/{rel_path}"
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
        if self._cp_role == "producer" or self._prefix_len == 0:
            return 0, False

        # Wait until all prefix chunks' KV are ready (all layers written).
        if not self._prefix_ready():
            # None => scheduler asks again later (async stall).
            return None, True

        external = _align_down(self._prefix_len, self._block_size)
        external = max(external - num_computed_tokens, 0)
        # Synchronous load: prefix KV is already in the shared store, so the
        # scheduler can compute the suffix this step (load_kv_async=False).
        return external, False

    def _prefix_ready(self) -> bool:
        for chunk_key in self._prefix_chunk_ids:
            if self._peer_url:
                rel = os.path.join(self._run_id, chunk_key, "_READY")
                if not self._remote_exists(rel):
                    return False
            elif not os.path.exists(self._ready_marker(chunk_key)):
                return False
        return True

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = HcpCpConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            block_ids = list(new_req.block_ids[0])
            if new_req.req_id in self._requests_need_load:
                # Load prefix tokens into the first prefix blocks.
                n_load = _align_down(self._prefix_len, self._block_size)
                load_ids = token_ids[:n_load]
                meta.requests.append(
                    self._make_meta(load_ids, block_ids, is_store=False,
                                    chunk_key=self._prefix_chunk_ids[0])
                )
            else:
                # Store this instance's own chunk KV.
                meta.requests.append(
                    self._make_meta(token_ids, block_ids, is_store=True,
                                    chunk_key=self._chunk_id)
                )
        self._requests_need_load.clear()
        return meta

    def _make_meta(
        self, token_ids: list[int], block_ids: list[int], is_store: bool, chunk_key: str
    ) -> ReqMeta:
        n = _align_down(len(token_ids), self._block_size)
        token_ids_t = torch.tensor(token_ids)[:n]
        block_ids_t = torch.tensor(block_ids[: (n // self._block_size)])
        num_blocks = block_ids_t.shape[0]
        offsets = torch.arange(0, self._block_size)
        slot_mapping = (
            offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()[:n]
        return ReqMeta(token_ids=token_ids_t, slot_mapping=slot_mapping,
                       is_store=is_store, chunk_key=chunk_key)

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HcpCpConnectorMetadata)
        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                fname = self._layer_file(request.chunk_key, layer_name)
                if self._peer_url:
                    rel = os.path.join(self._run_id, request.chunk_key,
                                       f"{layer_name}.safetensors")
                    if not self._fetch(rel, fname):
                        logger.warning("prefix KV fetch failed: %s", rel)
                        continue
                elif not os.path.exists(fname):
                    logger.warning("prefix KV missing: %s", fname)
                    continue
                kv = safetensors.torch.load_file(
                    fname, device=str(kv_cache_layer.device)
                )["kv_cache"]
                block_idxs = request.slot_mapping // self._block_size
                offsets = request.slot_mapping % self._block_size
                kv_cache_layer[block_idxs, :, offsets] = kv

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HcpCpConnectorMetadata)
        for request in metadata.requests:
            if not request.is_store:
                continue
            fname = self._layer_file(request.chunk_key, layer_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            block_idxs = request.slot_mapping // self._block_size
            offsets = request.slot_mapping % self._block_size
            kv = kv_layer[block_idxs, :, offsets]
            safetensors.torch.save_file({"kv_cache": kv.detach().cpu()}, fname)

    def wait_for_save(self) -> None:
        # Mark this instance's chunk as fully written so consumers can load it.
        metadata = self._get_connector_metadata()
        if isinstance(metadata, HcpCpConnectorMetadata):
            for request in metadata.requests:
                if request.is_store:
                    os.makedirs(self._chunk_dir(request.chunk_key), exist_ok=True)
                    with open(self._ready_marker(request.chunk_key), "w") as f:
                        f.write(str(time.time()))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        # Shared-path load/save is synchronous within the step; there are no
        # background sends/recvs to report, so return no finished ids.
        return None, None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return None

    def shutdown(self) -> None:
        return
