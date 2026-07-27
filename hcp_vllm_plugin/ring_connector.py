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
  - ``ring_role``: "ring" | "producer" | "consumer" | "relay"
      ring    : the UNIFIED role of a true ring node.  Every node consumes
                its ordered prefix chunks (empty for position 0) and produces
                its own chunk, i.e. ring = consumer-side + producer-side in
                one instance.  "relay" is kept as a backward-compatible alias
                of the same semantics; "producer" is the degenerate case with
                no prefix; "consumer" is the degenerate case with no
                downstream.  A staged chunk is re-served (its _READY marker
                is written locally right after staging), so the ring
                transport is NEIGHBOR accumulate-forward: a node pulls its
                whole accumulated prefix from its physical predecessor
                (i-1+N)%N only, and its successor pulls from it in turn —
                producer N's consumer is literally (N+1)%N.
      producer: prompt is its own chunk; computes it, saves per-layer K/V to
                its store and serves the store over HTTP (worker side only).
      consumer: prompt is the FULL sequence; the prefix (earlier chunk) is
                marked external so only the suffix is computed; peer KV is
                fetched from the producer into the ring backend's staging.
      relay   : legacy alias of "ring" (kept for existing drivers).

Ring-order shorthand (single-request deployments):
  - ``ring_order``      : comma-separated ``chunk@url`` specs in ring order,
                          e.g. "c0@http://a:8901,c1@http://b:8901,c2@http://c:8901"
  - ``ring_my_chunk``   : this node's own chunk id within ``ring_order``
  When set and a request carries no per-request hcp_ring params, the prefix
  chunk list derives from the order (all chunks before mine) and the peer
  url defaults to my physical predecessor (neighbor accumulate-forward).

Config keys (``kv_connector_extra_config``):
  - ``ring_role``             : "producer" | "consumer" | "relay"
  - ``ring_shared_path``      : local store dir (this instance's side)
  - ``ring_run_id``           : session id namespacing the store
  - ``ring_chunk_id``         : this instance's own chunk key (producer/relay)
  - ``ring_prefix_chunk_ids`` : comma-separated prefix chunk keys, ordered
                                (consumer/relay; any number of chunks)
  - ``ring_prefix_len``       : total number of prefix (peer chunk) tokens
                                (consumer/relay)
  - ``ring_peer_url``         : producer base URL, e.g. "http://127.0.0.1:8901"
                                (consumer/relay; empty => local shared-path read)
  - ``ring_serve_port``       : HTTP port to serve the store on (producer/relay)
  - ``ring_load_timeout_s``   : HTTP fetch timeout (consumer/relay)

Per-request overrides (multi-request / continuous batching):
  Each request may carry ``SamplingParams(extra_args={"kv_transfer_params":
  {"hcp_ring": {...}}})`` with keys ``chunk_ids`` (list or comma string,
  ordered prefix chunks), ``peer_urls`` (list or comma string, aligned with
  ``chunk_ids``; missing entries fall back to the global peer url), and
  ``prefix_len`` (total prefix tokens).  The singular ``chunk_id`` /
  ``peer_url`` forms are still accepted and treated as one-element lists.
  These take precedence over the global extra-config values, so concurrent
  requests can reference DIFFERENT peer chunk sets.  The worker stages each
  chunk under its own key and binds it to the request via the request's first
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
import re
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
    PEER_KV_STAGING,
    clear_decode_ring,
    drop_chunk_kv,
    map_request_peer,
    register_decode_ring,
    set_staging_release_fn,
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


def _decode_ring_on() -> bool:
    """Decode-ring (memory-split decode + streaming prefill) is the DEFAULT.
    HCP_RING_DECODE_RING=0 explicitly selects the legacy owner-collapse
    comparison path (bulk staging, NOT memory-split)."""
    return os.environ.get("HCP_RING_DECODE_RING", "1") == "1"


_STREAM_WINDOW = int(os.environ.get("HCP_RING_STREAM_WINDOW", "2"))


def _layer_index(layer_name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", layer_name)
    return int(m.group(1)) if m else 0


class _PrefillStream:
    """One request's streaming-prefill pipeline state.

    A background worker thread fetches per-layer prefix KV in model order
    into the ring backend's TRANSIENT staging, never running more than
    _STREAM_WINDOW layers ahead of the consumption watermark advanced by
    wait_for_layer_load().  GPU transient staging is therefore bounded to
    ~(window+1) layers x num_chunks at ALL times (disk store untouched) —
    the fix for the bulk full-prefix transient staging (ISSUE-003).
    """

    def __init__(
        self,
        req_id: str,
        layers: list[str],
        chunk_urls: list[tuple[str, str]],
        expected_total: int,
    ) -> None:
        self.req_id = req_id
        self.layers = layers
        self.chunk_urls = chunk_urls
        self.expected_total = expected_total
        self.events = {ln: threading.Event() for ln in layers}
        self.consumed_idx = -1
        self.error: Exception | None = None
        self.done = threading.Event()


def _ring_params_from_sampling(sampling_params: Any) -> dict[str, Any]:
    """Extract per-request ring params from SamplingParams.extra_args."""
    if sampling_params is None:
        return {}
    extra = getattr(sampling_params, "extra_args", None) or {}
    kv_params = extra.get("kv_transfer_params") or {}
    return dict(kv_params.get("hcp_ring") or {})


def _as_str_list(value: Any) -> list[str]:
    """Accept a list/tuple or comma-separated string; return a clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s for s in value.split(",") if s]
    return [str(s) for s in value]


def _chunks_and_urls(
    rp: dict[str, Any], global_ids: list[str], global_urls: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Ordered prefix chunk keys + per-chunk peer urls ("" = global fallback).

    Plural ``chunk_ids`` / ``peer_urls`` win; the singular ``chunk_id`` /
    ``peer_url`` forms are treated as one-element lists (backward compat).
    Without any per-request values, the global ids/urls (ring_order-derived)
    are used.
    """
    ids = _as_str_list(rp.get("chunk_ids"))
    if not ids and rp.get("chunk_id"):
        ids = [str(rp["chunk_id"])]
    if not ids:
        ids = list(global_ids)
    urls = _as_str_list(rp.get("peer_urls"))
    if not urls and rp.get("peer_url"):
        urls = [str(rp["peer_url"])]
    if not urls and global_urls:
        urls = list(global_urls)
    urls = [u.rstrip("/") for u in urls]
    if urls and len(urls) < len(ids):
        # Neighbor accumulate-forward: one predecessor serves the WHOLE
        # accumulated prefix, so a shorter url list is extended with its
        # last entry (NOT with "", which would wrongly fall back to the
        # local store / global peer and stall the readiness check forever).
        urls += [urls[-1]] * (len(ids) - len(urls))
    else:
        urls += [""] * (len(ids) - len(urls))
    return ids, urls


@dataclass
class RingReqMeta:
    is_store: bool
    # Store side: this instance's own chunk key.  Load side: unused ("").
    chunk_key: str
    # Store side only: paged-pool slots of this instance's own chunk tokens.
    slot_mapping: torch.Tensor | None = None
    # Load side only: request binding + per-request fetch parameters.
    req_id: str = ""
    first_block_id: int = -1
    # Load side only: ordered prefix chunk keys + aligned per-chunk peer urls
    # ("" = use the connector's global peer url / local store).
    chunk_keys: list[str] = field(default_factory=list)
    peer_urls: list[str] = field(default_factory=list)
    prefix_len: int = 0


@dataclass
class HcpRingConnectorMetadata(KVConnectorMetadata):
    requests: list[RingReqMeta] = field(default_factory=list)
    # Requests finished between the previous and current steps (copied from
    # SchedulerOutput, which is populated before build_connector_meta runs).
    # The worker cleans their staged KV in start_load_kv — BEFORE this step's
    # forward — because their freed blocks may already have been re-allocated
    # to requests scheduled in THIS step.  Cleaning only in get_finished
    # (post-forward) would leave a stale first-block -> chunk binding that a
    # recycled block id could wrongly attach to a NEW request.
    finished_req_ids: list[str] = field(default_factory=list)


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
        if self._ring_role not in ("ring", "producer", "consumer", "relay"):
            raise ValueError(
                f"HcpRingKvConnector: unknown ring_role {self._ring_role!r} "
                "(expected ring|producer|consumer|relay)"
            )
        # Side flags: "ring"/"relay" are both consumer-side and producer-side.
        self._consumer_side = self._ring_role in ("consumer", "relay", "ring")
        self._producer_side = self._ring_role in ("producer", "relay", "ring")
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
        # Ring-order shorthand: derive my prefix chunks and my physical
        # predecessor (neighbor accumulate-forward) from "c0@url,c1@url,...".
        # Per-request hcp_ring params always take precedence over this.
        self._prefix_peer_urls: list[str] = []
        order = cfg.get_from_extra_config("ring_order", "")
        my_chunk = cfg.get_from_extra_config("ring_my_chunk", self._chunk_id)
        if order:
            specs = [s for s in order.split(",") if s]
            ids = [s.split("@")[0] for s in specs]
            urls = [
                (s.split("@", 1)[1].rstrip("/") if "@" in s else "")
                for s in specs
            ]
            if my_chunk in ids:
                i = ids.index(my_chunk)
                self._chunk_id = my_chunk
                self._prefix_chunk_ids = ids[:i]
                pred = urls[i - 1] if i > 0 else ""
                self._prefix_peer_urls = [pred] * i

        self._requests_need_load: dict[str, "Request"] = {}
        self._saved_tokens: dict[str, int] = {}
        self._saved_layers: dict[str, int] = {}
        # Readiness poll hygiene: a chunk that reported _READY stays ready
        # forever (cache hits skip polling entirely); a failed check is
        # backed off per chunk.  Without this, a scheduler spinning on
        # stalled requests fires thousands of HEAD requests per second at
        # the peer and exhausts ephemeral ports (observed: 6M polls, then
        # urlopen failures -> permanent livelock).
        self._ready_ok: set[str] = set()
        self._ready_next_poll: dict[str, float] = {}
        self._ready_poll_interval = float(
            cfg.get_from_extra_config("ring_ready_poll_interval_s", 0.5)
        )
        # Worker-side live-request bookkeeping for cleanup:
        #   req_id -> (first_block_id, [chunk_keys]); chunk_key -> refcount;
        #   chunk_key -> staged token count (per layer)
        self._live: dict[str, tuple[int, list[str]]] = {}
        self._chunk_refs: dict[str, int] = {}
        self._chunk_lens: dict[str, int] = {}
        # req_id -> first_block_id, for finish-time decode-ring clearing
        # (the _live entry may already be released at decode start).
        self._ring_blocks: dict[str, int] = {}
        # req_id -> streaming prefill pipeline state (decode-ring mode only).
        self._streams: dict[str, "_PrefillStream"] = {}
        os.makedirs(self._run_dir(), exist_ok=True)
        # Producer-side WORKER connectors hold/serve KV; the scheduler-side
        # instance must not bind the HTTP port.
        if (
            self._producer_side
            and self._serve_port > 0
            and self._role == KVConnectorRole.WORKER
        ):
            self._start_http_server()
        # P2P ring decode transport (independent of HTTP store server).
        self._ring_transport = None
        self._ring_node = None
        if (
            os.environ.get("HCP_RING_DECODE_RING", "1") == "1"
            and os.environ.get("HCP_RING_DECODE_TRANSPORT", "ring") == "ring"
            and self._role == KVConnectorRole.WORKER
        ):
            self._start_ring_transport()
        # Decode-ring: let the backend release staged prefix KV (refcount
        # aware) when a request enters the decode phase.  Only the WORKER-side
        # instance owns _live (start_load_kv); EngineCore builds the model
        # executor BEFORE the scheduler, so the scheduler-side instance is
        # constructed last and must not overwrite the callback with its own
        # (its _live is always empty -> release would be a silent no-op).
        if self._role == KVConnectorRole.WORKER:
            set_staging_release_fn(self._release_request_staging)
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
        from http.server import ThreadingHTTPServer

        partial_service = None
        decode_transport_mode = os.environ.get(
            "HCP_RING_DECODE_TRANSPORT", "ring"
        )
        if (
            os.environ.get("HCP_RING_DECODE_RING", "1") == "1"
            and decode_transport_mode == "http"
        ):
            # Legacy HTTP star mode: PartialAttentionService on the same port.
            from hcp_vllm_plugin.ring_decode import (
                PartialAttentionService,
                make_handler,
            )

            partial_chunks = [
                s
                for s in self._kv_transfer_config.get_from_extra_config(
                    "ring_partial_chunks", self._chunk_id
                ).split(",")
                if s
            ]
            partial_service = PartialAttentionService(
                self._run_dir(), partial_chunks
            )
            handler = make_handler(partial_service, self._shared_path)
        else:
            from functools import partial
            from http.server import SimpleHTTPRequestHandler

            handler = partial(
                SimpleHTTPRequestHandler, directory=self._shared_path
            )
        server = ThreadingHTTPServer(("0.0.0.0", self._serve_port), handler)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self._http_server = server
        self._partial_service = partial_service
        logger.info(
            "HcpRingKvConnector serving KV store on 0.0.0.0:%d (partial_attn=%s)",
            self._serve_port,
            partial_service is not None,
        )

    def _start_ring_transport(self) -> None:
        """Initialize P2P ring TCP transport + RingDecodeNode for decode.

        Config keys:
          ring_decode_order : comma-separated "host:port" in ring order
                              (index 0 = owner).  If empty, derived from
                              ring_order URLs + ring_decode_port.
          ring_decode_port  : TCP port for this node (default 8950).
          ring_partial_chunks : comma-separated chunk keys this node serves.
        """
        from hcp_vllm_plugin.ring_transport import init_ring_transport
        from hcp_vllm_plugin.ring_decode import RingDecodeNode

        cfg = self._kv_transfer_config
        decode_port = int(cfg.get_from_extra_config("ring_decode_port", "8950"))
        order_str = cfg.get_from_extra_config("ring_decode_order", "")

        if order_str:
            # Explicit "host:port,host:port,..." ring order
            ring_addrs = []
            for spec in order_str.split(","):
                spec = spec.strip()
                host, port_s = spec.rsplit(":", 1)
                ring_addrs.append((host, int(port_s)))
        else:
            # Derive from ring_order: extract hosts from URLs, use decode_port
            order = cfg.get_from_extra_config("ring_order", "")
            if not order:
                logger.warning(
                    "HcpRingKvConnector: ring_decode_order and ring_order both "
                    "empty; cannot start ring transport"
                )
                return
            specs = [s for s in order.split(",") if s]
            ring_addrs = []
            for s in specs:
                if "@" in s:
                    url = s.split("@", 1)[1].rstrip("/")
                    # Extract host from URL like "http://host:port"
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname or "127.0.0.1"
                else:
                    host = "127.0.0.1"
                ring_addrs.append((host, decode_port))

        # Determine my index in the transport ring.
        # Priority: ring_decode_index (explicit) > match by port in
        # ring_decode_order > derive from ring_order prefill position.
        explicit_idx = cfg.get_from_extra_config("ring_decode_index", "")
        if explicit_idx:
            my_index = int(explicit_idx)
        elif order_str:
            # Match my decode port against the ring_decode_order entries
            my_index = 0
            for i, (h, p) in enumerate(ring_addrs):
                if p == decode_port:
                    my_index = i
                    break
        else:
            my_chunk = cfg.get_from_extra_config("ring_my_chunk", self._chunk_id)
            order = cfg.get_from_extra_config("ring_order", "")
            my_index = 0
            if order:
                specs = [s for s in order.split(",") if s]
                ids = [s.split("@")[0] for s in specs]
                if my_chunk in ids:
                    my_index = ids.index(my_chunk)

        # Initialize transport via the module singleton so in-process
        # consumers (validation probes) can find it via get_ring_transport().
        self._ring_transport = init_ring_transport(ring_addrs, my_index)

        # Initialize RingDecodeNode (loads chunk KV snapshot for serving)
        partial_chunks = [
            s
            for s in cfg.get_from_extra_config(
                "ring_partial_chunks", self._chunk_id
            ).split(",")
            if s
        ]
        self._ring_node = RingDecodeNode(
            self._run_dir(), partial_chunks, my_index
        )

        # Non-owner nodes start the relay worker thread immediately; the
        # thread itself waits for the chunk store to become ready (the
        # store appears only after this node's prefill, which cannot
        # happen until engine init completes — waiting HERE would
        # self-deadlock the engine construction).
        # Owner does NOT run relay — it initiates packets synchronously
        # from within the vLLM forward pass.
        if my_index != 0:
            self._ring_node.start_relay(self._ring_transport)

        logger.info(
            "HcpRingKvConnector: ring transport started (index=%d, "
            "addrs=%s, chunks=%s, relay=%s)",
            my_index, ring_addrs, partial_chunks, my_index != 0,
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
        if not self._consumer_side:
            return 0, False

        # Per-request params (kv_transfer_params.hcp_ring) override globals,
        # so concurrent requests may reference different peer chunk sets.
        rp = (getattr(request, "kv_transfer_params", None) or {}).get(
            "hcp_ring"
        ) or {}
        # Presence (not truthiness) decides override, so a request can
        # explicitly opt out with prefix_len=0 even when globals are set.
        prefix_len = int(rp["prefix_len"]) if "prefix_len" in rp else self._prefix_len
        if prefix_len == 0:
            return 0, False
        chunk_ids, peer_urls = _chunks_and_urls(
            rp, self._prefix_chunk_ids, self._prefix_peer_urls
        )
        if not chunk_ids:
            return 0, False

        # Wait until ALL prefix chunks' KV are fully written by their
        # producers (ready state cascades down the ring).
        for chunk_key, peer_url in zip(chunk_ids, peer_urls):
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
        cache_key = f"{peer_url or self._peer_url}/{self._run_id}/{chunk_key}"
        if cache_key in self._ready_ok:
            return True
        now = time.monotonic()
        if now < self._ready_next_poll.get(cache_key, 0.0):
            return False
        if peer_url or self._peer_url:
            rel = os.path.join(self._run_id, chunk_key, "_READY")
            ok = self._remote_exists(rel, peer_url)
        else:
            ok = os.path.exists(self._ready_marker(chunk_key))
        if ok:
            self._ready_ok.add(cache_key)
        else:
            self._ready_next_poll[cache_key] = now + self._ready_poll_interval
        return ok

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = HcpRingConnectorMetadata()
        meta.finished_req_ids = sorted(scheduler_output.finished_req_ids)
        for new_req in scheduler_output.scheduled_new_reqs:
            rp = _ring_params_from_sampling(new_req.sampling_params)
            if new_req.req_id in self._requests_need_load:
                # Consumer side: stage the prefix chunks' KV (no pool slot
                # mapping — we deliberately do NOT load into the block table).
                chunk_ids, peer_urls = _chunks_and_urls(
                    rp, self._prefix_chunk_ids, self._prefix_peer_urls
                )
                meta.requests.append(
                    RingReqMeta(
                        is_store=False,
                        chunk_key="",
                        req_id=new_req.req_id,
                        first_block_id=new_req.block_ids[0][0],
                        chunk_keys=chunk_ids,
                        peer_urls=peer_urls,
                        prefix_len=int(rp["prefix_len"]) if "prefix_len" in rp else self._prefix_len,
                    )
                )
            if self._producer_side:
                # Producer side: save this instance's own chunk KV after
                # compute.  On a ring/relay node the own chunk is the
                # request's suffix: the external prefix is skipped in the
                # slot mapping.
                token_ids = new_req.prompt_token_ids or []
                block_ids = list(new_req.block_ids[0])
                chunk_key = rp.get("chunk_id") or self._chunk_id
                skip = 0
                if self._consumer_side:
                    plen = int(rp["prefix_len"]) if "prefix_len" in rp else self._prefix_len
                    skip = _align_down(plen, self._block_size)
                slot_mapping = self._make_slot_mapping(token_ids, block_ids, skip)
                if slot_mapping.shape[0] > 0:
                    meta.requests.append(
                        RingReqMeta(
                            is_store=True,
                            chunk_key=chunk_key,
                            slot_mapping=slot_mapping,
                        )
                    )
        self._requests_need_load.clear()
        return meta

    def _make_slot_mapping(
        self, token_ids: list[int], block_ids: list[int], skip: int = 0
    ) -> torch.Tensor:
        n = _align_down(len(token_ids), self._block_size)
        block_ids_t = torch.tensor(block_ids[skip // self._block_size : n // self._block_size])
        num_blocks = block_ids_t.shape[0]
        if num_blocks == 0:
            return torch.empty(0, dtype=torch.long)
        offsets = torch.arange(0, self._block_size)
        slot_mapping = (
            offsets.reshape((1, self._block_size))
            + block_ids_t.reshape((num_blocks, 1)) * self._block_size
        ).flatten()[: n - skip]
        return slot_mapping

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Fetch each request's prefix chunks' per-layer KV into the ring
        backend's TRANSIENT staging dict (keyed by chunk), bind the request to
        its ordered chunk list via its first block-table block id, and track
        them for cleanup on finish.  Never writes to the paged pool/block
        table."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HcpRingConnectorMetadata)
        # Clean finished requests' staging BEFORE this step's forward (their
        # recycled blocks may serve new requests scheduled in this step).
        self._cleanup_finished(metadata.finished_req_ids)
        need = [r for r in metadata.requests if not r.is_store]
        if not need:
            return
        layer_order = list(forward_context.no_compile_layers.keys())
        device = "cuda"
        for _ln, _layer in forward_context.no_compile_layers.items():
            kvc = getattr(_layer, "kv_cache", None)
            if torch.is_tensor(kvc) and kvc.numel() > 0:
                device = str(kvc.device)
            break
        streaming = _decode_ring_on()
        if not streaming:
            logger.warning(
                "HcpRingKvConnector: HCP_RING_DECODE_RING=0 — legacy "
                "owner-collapse path: BULK full-prefix staging (NOT "
                "memory-split; comparison only)"
            )
        for req in need:
            expected_total = _align_down(
                req.prefix_len or self._prefix_len, self._block_size
            )
            # Registration is synchronous (the backend binds the request to
            # its chunk set from layer 0 of the prefill forward).
            map_request_peer(req.first_block_id, req.chunk_keys)
            self._live[req.req_id] = (req.first_block_id, list(req.chunk_keys))
            for chunk_key in req.chunk_keys:
                self._chunk_refs[chunk_key] = (
                    self._chunk_refs.get(chunk_key, 0) + 1
                )
            # Decode-ring: register the accumulator-ring endpoints for this
            # request ((chunk, url) pairs in prefix order).
            pairs = [
                (ck, (u or self._peer_url).rstrip("/"))
                for ck, u in zip(req.chunk_keys, req.peer_urls)
            ]
            register_decode_ring(
                req.first_block_id, pairs, expected_total,
                transport=getattr(self, "_ring_transport", None),
            )
            self._ring_blocks[req.req_id] = req.first_block_id
            if streaming:
                # True-ring prefill: fetch per-layer KV in the background,
                # bounded window ahead of the attention consumption
                # watermark — never the whole prefix at once.
                st = _PrefillStream(
                    req.req_id, layer_order, pairs, expected_total
                )
                self._streams[req.req_id] = st
                threading.Thread(
                    target=self._stream_worker,
                    args=(st, device),
                    daemon=True,
                ).start()
                continue
            # Legacy bulk staging (explicit HCP_RING_DECODE_RING=0 only).
            staged_total = 0
            for chunk_key, peer_url in zip(req.chunk_keys, req.peer_urls):
                if chunk_key in self._chunk_refs and self._chunk_refs[chunk_key] > 1:
                    # Already staged by another live request.
                    staged_total += self._chunk_lens.get(chunk_key, 0)
                    continue
                staged = 0
                total_bytes = 0
                for layer_name in layer_order:
                    tensors = self._load_layer(
                        chunk_key, layer_name, device, peer_url
                    )
                    if tensors is None:
                        raise RuntimeError(
                            "HcpRingKvConnector: peer KV fetch failed for "
                            f"{chunk_key}/{layer_name} "
                            f"(peer={peer_url or self._peer_url})"
                        )
                    k, v = tensors
                    stage_peer_kv(chunk_key, layer_name, k, v)
                    staged += 1
                    total_bytes += k.numel() * k.element_size() * 2
                self._chunk_lens[chunk_key] = k.shape[0]
                staged_total += k.shape[0]
                self._write_ready_marker(chunk_key, k.shape[0], staged)
            if staged_total != expected_total:
                raise RuntimeError(
                    f"HcpRingKvConnector: staged prefix length {staged_total} "
                    f"!= expected external prefix {expected_total} "
                    f"(chunks={req.chunk_keys})"
                )

    def _write_ready_marker(
        self, chunk_key: str, tokens: int, layers: int
    ) -> None:
        """Re-serve (neighbor accumulate-forward): mark the chunk ready in
        THIS node's store so its ring successor can pull the accumulated
        prefix from here alone."""
        with open(self._ready_marker(chunk_key), "w") as f:
            json.dump(
                {
                    "tokens": tokens,
                    "layers": layers,
                    "ts": time.time(),
                    "re_served": True,
                },
                f,
            )

    def _stream_worker(self, st: _PrefillStream, device: str) -> None:
        """Background per-layer fetch pipeline for one request (streaming
        prefill).  Stays at most _STREAM_WINDOW layers ahead of the
        consumption watermark; sets a per-layer event as each layer's
        staging becomes usable."""
        lens: dict[str, int] = {}
        staged_layers = 0
        try:
            for i, layer_name in enumerate(st.layers):
                while (
                    not st.done.is_set()
                    and st.consumed_idx < i - _STREAM_WINDOW
                ):
                    time.sleep(0.001)
                if st.done.is_set():
                    return
                for chunk_key, peer_url in st.chunk_urls:
                    tensors = self._load_layer(
                        chunk_key, layer_name, device, peer_url
                    )
                    if tensors is None:
                        raise RuntimeError(
                            "HcpRingKvConnector: peer KV fetch failed for "
                            f"{chunk_key}/{layer_name} "
                            f"(peer={peer_url or self._peer_url})"
                        )
                    k, v = tensors
                    stage_peer_kv(chunk_key, layer_name, k, v)
                    lens[chunk_key] = k.shape[0]
                staged_layers += 1
                st.events[layer_name].set()
            staged_total = sum(lens.values())
            for chunk_key, n in lens.items():
                self._chunk_lens[chunk_key] = n
                self._write_ready_marker(chunk_key, n, staged_layers)
            if staged_total != st.expected_total:
                raise RuntimeError(
                    f"HcpRingKvConnector: staged prefix length {staged_total} "
                    f"!= expected external prefix {st.expected_total} "
                    f"(chunks={[c for c, _ in st.chunk_urls]})"
                )
        except Exception as e:  # noqa: BLE001
            st.error = e
            for ev in st.events.values():
                ev.set()

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
        # Streaming prefill: block until this layer's prefix KV is staged
        # (the background pipeline runs up to _STREAM_WINDOW layers ahead),
        # then advance the consumption watermark and release GPU staging of
        # all earlier layers.  Transient staging stays bounded to
        # ~(window+1) layers x num_chunks at all times.
        if not self._streams:
            return
        idx = _layer_index(layer_name)
        for st in list(self._streams.values()):
            ev = st.events.get(layer_name)
            if ev is not None:
                ev.wait()
            if st.error is not None:
                raise RuntimeError(
                    f"HcpRingKvConnector: prefill stream for req "
                    f"{st.req_id} failed: {st.error}"
                )
            st.consumed_idx = max(st.consumed_idx, idx)
        # GPU staging only; the disk store (re-serve / partial services)
        # is untouched.
        old = [
            key
            for key in PEER_KV_STAGING
            if _layer_index(key[1]) < idx
        ]
        for key in old:
            del PEER_KV_STAGING[key]

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
        # Belt-and-suspenders: cleanup already ran pre-forward via metadata;
        # this also covers any path that bypasses connector metadata.
        self._cleanup_finished(finished_req_ids)
        return None, None

    def _release_live_entry(self, req_id: str, clear_ring: bool = True) -> None:
        """Release one live request's staged KV (refcounted per chunk) and
        unbind its first-block -> chunks mapping.  Idempotent.  clear_ring
        also drops the decode-ring registration (only at request finish —
        decode steps need it for the whole decode phase)."""
        st = self._streams.pop(req_id, None)
        if st is not None:
            st.done.set()
        live = self._live.pop(req_id, None)
        if live is not None:
            first_block_id, chunk_keys = live
            unmap_request_peer(first_block_id)
            for chunk_key in chunk_keys:
                refs = self._chunk_refs.get(chunk_key, 0) - 1
                if refs <= 0:
                    self._chunk_refs.pop(chunk_key, None)
                    self._chunk_lens.pop(chunk_key, None)
                    dropped = drop_chunk_kv(chunk_key)
                    logger.info(
                        "HcpRingKvConnector: freed staged chunk %s (%d layers)",
                        chunk_key,
                        dropped,
                    )
                else:
                    self._chunk_refs[chunk_key] = refs
        if clear_ring:
            fb = self._ring_blocks.pop(req_id, None)
            if fb is not None:
                clear_decode_ring(fb)

    def _release_request_staging(self, first_block_id: int) -> None:
        """Decode-ring callback: free a request's staged prefix KV at decode
        start (looked up by first block id; the decode-ring registration is
        kept for the decode phase)."""
        req_id = next(
            (rid for rid, (fb, _) in self._live.items() if fb == first_block_id),
            None,
        )
        if req_id is not None:
            self._release_live_entry(req_id, clear_ring=False)

    def _cleanup_finished(self, finished_req_ids) -> None:
        """Free transient staged KV of finished requests (refcounted per
        chunk) and unbind their first-block -> chunks mapping.  Idempotent."""
        for req_id in finished_req_ids:
            self._release_live_entry(req_id)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return None

    def shutdown(self) -> None:
        # Engine teardown: drop any staged KV still referenced.  (During
        # continuous serving, cleanup happens one step after a request
        # finishes via get_finished; anything left at shutdown is the tail.)
        for first_block_id, _ in list(self._live.values()):
            unmap_request_peer(first_block_id)
        for chunk_key in list(self._chunk_refs):
            drop_chunk_kv(chunk_key)
        for fb in list(self._ring_blocks.values()):
            clear_decode_ring(fb)
        self._live.clear()
        self._chunk_refs.clear()
        self._chunk_lens.clear()
        self._ring_blocks.clear()
