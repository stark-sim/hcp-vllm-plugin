"""HCP ring-attention backend for vLLM 0.23 (V1 engine) — PoC.

Memory-splitting "ring attention" (online-softmax) custom attention backend.

Semantics (2-chunk causal split, positions are global within the request):
  * tokens with position <  HCP_RING_SPLIT_TOKENS  = "chunk A" (the peer chunk)
  * tokens with position >= HCP_RING_SPLIT_TOKENS  = "chunk B" (the local chunk)

For a query in chunk B, attention is computed as the online-softmax merge of
  (a) LOCAL attention over chunk-B KV only (causal), and
  (b) PEER  attention over chunk-A KV only (non-causal, transient).
For a query in chunk A, plain causal attention is used.

This reproduces full attention over all KV exactly (up to float rounding),
while the math only ever touches chunk-B KV as "resident" state; chunk-A KV
is consumed transiently.  On a real deployment chunk-A KV would arrive via
the KV connector into a staging buffer; this PoC reads it either from the
module-level staging dict below or from the local paged cache (which in the
single-process validation happens to hold the whole prompt's KV — the
attention math deliberately never reads chunk-A KV on the local path).

ROCm notes (validated on pearl, gfx1200, torch 2.13+rocm7.13, vllm 0.23.1):
  * upstream `flash_attn` is NOT installed and
    `vllm.v1.attention.backends.fa_utils.is_flash_attn_varlen_func_available()`
    returns False (vendored FA is CUDA/XPU-only; ROCm RDNA's native path is
    Triton), so FlashAttentionImpl cannot run here.  Attention therefore runs
    through the plugin's own Triton kernel (ring_triton_attn, flash-style
    with LSE output, validated on gfx1200: max|diff| ~1e-3 fp16 vs the fp32
    PyTorch reference, LSE ~1e-6); the fp32 plain-PyTorch path remains as a
    debug fallback (HCP_RING_IMPL=torch).
  * vLLM's triton `merge_attn_states` is the default LSE merge (validated on
    gfx1200 vs plain merge: 6.1e-5); plain PyTorch merge remains as fallback
    (HCP_RING_MERGE=torch).

Configuration (env vars):
  HCP_RING_SPLIT_TOKENS : int, position boundary between chunk A and chunk B.
                          0 (default) disables the ring merge (plain local
                          causal attention, i.e. vanilla behavior).
  HCP_RING_ENABLED      : "1" (default) / "0" master switch.
  HCP_RING_IMPL         : "triton" (default) / "torch" — attention impl for
                          the local/peer passes.  "triton" uses
                          ring_triton_attn (flash-style, no score-matrix
                          materialization, works on CUDA and ROCm); "torch"
                          is the original fp32 einsum path (debug fallback).
  HCP_RING_MERGE        : "triton" (default) / "torch" — LSE merge impl
                          (vLLM merge_attn_states vs plain PyTorch).
  IMPL_STATS            : dispatch counters (validation evidence of which
                          path actually ran).

Peer KV staging API (used by HcpRingKvConnector):
  stage_peer_kv(chunk_key, layer_name, k, v) / drop_chunk_kv / clear_peer_kv
  map_request_peer(first_block_id, chunk_keys) / unmap_request_peer
  k/v: [num_tokens, num_kv_heads, head_dim] contiguous, post-RoPE K.
  Staging is keyed by (chunk_key, layer_name); each request is bound to an
  ORDERED list of prefix chunk keys via PEER_REQ_MAP[first_block_id], so
  multiple requests with different peer chunk sets can batch concurrently.
  The ordered chunks are a globally contiguous prefix: forward() concatenates
  them into one peer KV segment and runs a single non-causal peer pass.
  When a request's row has staged peer KV it is used as the peer (chunk-A) KV
  and the first `k.shape[0]` positions of the local cache are excluded from
  the local path.  Without staged KV, HCP_RING_SPLIT_TOKENS splits the local
  paged cache instead (single-process validation path).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Peer KV staging (module level).  The KV connector writes here; the
# AttentionImpl reads (never stores) it during forward.
#
# Multi-request design:
#   PEER_KV_STAGING is keyed by (chunk_key, layer_name) so that several
#   requests — each with its own peer chunk set — can be in flight
#   concurrently.  PEER_REQ_MAP maps a request's FIRST block-table block id
#   (stable for the request's whole lifetime) to its ORDERED list of prefix
#   chunk keys (the chunks form a globally contiguous prefix of the request).
#   The connector registers the mapping when it stages the chunks and
#   unregisters it when the request finishes, so block-id reuse by the
#   allocator is safe.
# ---------------------------------------------------------------------------
PEER_KV_STAGING: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
PEER_REQ_MAP: dict[int, list[str]] = {}


def stage_peer_kv(
    chunk_key: str, layer_name: str, k: torch.Tensor, v: torch.Tensor
) -> None:
    """Stage a peer chunk's K/V for one attention layer.

    k, v: [num_tokens, num_kv_heads, head_dim] — K must be post-RoPE.
    """
    PEER_KV_STAGING[(chunk_key, layer_name)] = (k, v)
    _note_staging_locked(chunk_key, k)


def drop_chunk_kv(chunk_key: str) -> int:
    """Drop all staged layers of one chunk.  Returns layers dropped."""
    keys = [key for key in PEER_KV_STAGING if key[0] == chunk_key]
    for key in keys:
        del PEER_KV_STAGING[key]
    return len(keys)


def map_request_peer(first_block_id: int, chunk_keys) -> None:
    """Bind a request (by first block id) to its ordered prefix chunk keys.

    Accepts a single chunk key (str) or an ordered list/tuple of keys; a
    single key is treated as a one-element list (backward compat).
    """
    if isinstance(chunk_keys, str):
        chunk_keys = [chunk_keys]
    PEER_REQ_MAP[first_block_id] = list(chunk_keys)


def unmap_request_peer(first_block_id: int) -> None:
    PEER_REQ_MAP.pop(first_block_id, None)


def request_peer_chunk(first_block_id: int) -> list[str] | None:
    return PEER_REQ_MAP.get(first_block_id)


def clear_peer_kv() -> None:
    PEER_KV_STAGING.clear()
    PEER_REQ_MAP.clear()


# ---------------------------------------------------------------------------
# Memory-split decode (accumulator ring).  When HCP_RING_DECODE_RING=1, a
# request that has registered decode-ring info (its prefix chunks' peer
# endpoints + prefix length) switches at decode time: its transiently staged
# prefix KV is DROPPED, and each decode step's attention is computed as the
# online-softmax accumulator merge of the owner's local partial (own chunk +
# owned growth share + current token, causal) plus each peer's remote
# partial (its chunk snapshot + its growth share, non-causal) via one RPC
# hop per peer.  Decode GROWTH is itself sharded across the ring (default:
# global position round-robin; HCP_RING_DECODE_GROWTH selects the policy) —
# every node durably holds only its own share of ALL KV, prefill and decode
# alike.  Semantics stay exactly memory-split; latency is L x (P-1) RTT per
# token (experiment-phase evidence for CXL-class interconnect).
# ---------------------------------------------------------------------------
DECODE_RING_MAP: dict[int, dict] = {}
# The connector registers its staging-release callback here so the backend
# can free a request's staged prefix KV (refcount-aware) at decode start.
_STAGING_RELEASE_FN = None


def _decode_ring_enabled() -> bool:
    # Decode-ring is the DEFAULT (memory-split decode + streaming prefill).
    # HCP_RING_DECODE_RING=0 explicitly selects the legacy owner-collapse
    # comparison path (bulk staging, NOT memory-split).
    return os.environ.get("HCP_RING_DECODE_RING", "1") == "1"


def _growth_policy() -> str:
    return os.environ.get("HCP_RING_DECODE_GROWTH", "rr")


def set_staging_release_fn(fn) -> None:
    global _STAGING_RELEASE_FN
    _STAGING_RELEASE_FN = fn


def register_decode_ring(
    first_block_id: int, peer_chunks: list[tuple[str, str]], prefix_len: int,
    transport=None,
) -> None:
    """peer_chunks: ordered (chunk_key, peer_url) pairs of the prefix.
    transport: RingTransport instance for P2P ring mode (None = HTTP star)."""
    peers = [(ck, u) for ck, u in peer_chunks if u]
    DECODE_RING_MAP[first_block_id] = {
        "peer_chunks": peers,
        "prefix_len": prefix_len,
        # Ring indexing for growth assignment: 0..N-2 = peers in prefix
        # order, N-1 = the owner itself.
        "ring_size": len(peers) + 1,
        "own_index": len(peers),
        # Filled lazily at the first decode step (tk = prompt_len + 1 there).
        "own_chunk_len": None,
        # layer_name -> (k, v): decode tokens assigned to the owner.
        "growth": {},
        # layer_name -> (assignee, k, v): this step's token, shipped on the
        # next step's RPC to its assignee (one-step pipeline lag).
        "pending_growth": {},
        "growth_policy": _growth_policy(),
        # P2P ring transport (None = legacy HTTP star mode).
        "transport": transport,
    }


def clear_decode_ring(first_block_id: int) -> None:
    DECODE_RING_MAP.pop(first_block_id, None)


def _release_staged_for_request(first_block_id: int) -> None:
    """Decode-ring transition: free this request's transiently staged prefix
    KV via the connector's refcount-aware callback (falls back to a direct
    idempotent drop when no callback is registered)."""
    if first_block_id not in PEER_REQ_MAP:
        return
    if _STAGING_RELEASE_FN is not None:
        _STAGING_RELEASE_FN(first_block_id)
    else:
        ck_list = PEER_REQ_MAP.pop(first_block_id, None) or []
        for ck in ck_list:
            drop_chunk_kv(ck)


# High-water marks for validation: staging is freed when a request finishes,
# so post-run checks cannot inspect live staging — use these counters instead.
STAGING_STATS: dict = {
    "max_concurrent_chunks": 0,
    "max_staged_layers": 0,
    "last_chunk_len": 0,
}


def reset_staging_stats() -> None:
    STAGING_STATS["max_concurrent_chunks"] = 0
    STAGING_STATS["max_staged_layers"] = 0
    STAGING_STATS["last_chunk_len"] = 0


def _note_staging_locked(chunk_key: str, k: torch.Tensor) -> None:
    STAGING_STATS["max_staged_layers"] = max(
        STAGING_STATS["max_staged_layers"], len(PEER_KV_STAGING)
    )
    chunks = {key[0] for key in PEER_KV_STAGING}
    STAGING_STATS["max_concurrent_chunks"] = max(
        STAGING_STATS["max_concurrent_chunks"], len(chunks)
    )
    STAGING_STATS["last_chunk_len"] = k.shape[0]


# ---------------------------------------------------------------------------
# Debug write-tracking (validation only).  When enabled, do_kv_cache_update
# records which paged-pool slots this worker wrote, and forward() cross-checks
# that no slot belonging to the peer (chunk-A) region was written locally —
# i.e. the consumer's permanent pool provably never holds chunk-A KV.
# ---------------------------------------------------------------------------
WRITE_TRACK: dict = {"enabled": False, "slots": set(), "overlap": 0, "skipped": 0}


def reset_write_tracking() -> None:
    WRITE_TRACK["enabled"] = True
    WRITE_TRACK["slots"] = set()
    WRITE_TRACK["overlap"] = 0
    WRITE_TRACK["skipped"] = 0


# Continuous-batching evidence (validation only): largest number of requests
# seen in one attention step.  Updated unconditionally; cost is negligible.
BATCH_STATS: dict = {"max_reqs": 0}


def reset_batch_stats() -> None:
    BATCH_STATS["max_reqs"] = 0


def _ring_enabled() -> bool:
    return os.environ.get("HCP_RING_ENABLED", "1") == "1"


def _split_tokens() -> int:
    return int(os.environ.get("HCP_RING_SPLIT_TOKENS", "0"))


def _ring_impl() -> str:
    return os.environ.get("HCP_RING_IMPL", "triton")


def _merge_impl() -> str:
    return os.environ.get("HCP_RING_MERGE", "triton")


# Implementation dispatch counters (validation evidence: which path ran).
IMPL_STATS: dict = {
    "attn_triton": 0,
    "attn_torch": 0,
    "merge_triton": 0,
    "merge_torch": 0,
}


def reset_impl_stats() -> None:
    for key in IMPL_STATS:
        IMPL_STATS[key] = 0


def _attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    q_pos0_in_kv: int | None,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention + LSE, dispatched to the Triton ring kernel by default.

    Falls back to the plain-PyTorch path (fp32) when the Triton impl is
    disabled, unavailable, or errors, or when a sliding window is requested
    (the ring kernel does not implement windowing).
    """
    if _ring_impl() == "triton" and window is None and q.is_cuda:
        try:
            from hcp_vllm_plugin.ring_triton_attn import ring_attn_with_lse

            o, lse = ring_attn_with_lse(
                q.contiguous(), k.contiguous(), v.contiguous(), scale,
                q_pos0_in_kv,
            )
            IMPL_STATS["attn_triton"] += 1
            return o, lse
        except Exception as e:
            logger.warning("ring triton attn failed (%s); using torch path", e)
    IMPL_STATS["attn_torch"] += 1
    return _attn_with_lse(q, k, v, scale, q_pos0_in_kv, window)


def _merge(
    o_loc: torch.Tensor,
    lse_loc: torch.Tensor,
    o_peer: torch.Tensor,
    lse_peer: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge of local + peer partial attention, dispatched to
    vLLM's merge_attn_states (triton) by default; torch LSE merge fallback."""
    if _merge_impl() == "triton" and o_loc.is_cuda:
        try:
            from vllm.v1.attention.ops.merge_attn_states import (
                merge_attn_states,
            )

            out = torch.empty_like(o_loc)
            merge_attn_states(out, o_peer, lse_peer, o_loc, lse_loc)
            IMPL_STATS["merge_triton"] += 1
            return out
        except Exception as e:
            logger.warning("triton merge failed (%s); using torch merge", e)
    IMPL_STATS["merge_torch"] += 1
    return _lse_merge(o_loc, lse_loc, o_peer, lse_peer)


# ---------------------------------------------------------------------------
# Plain-PyTorch attention helpers (fp32 accumulation, natural-log LSE)
# ---------------------------------------------------------------------------
def _attn_with_lse(
    q: torch.Tensor,  # [Tq, H, D]
    k: torch.Tensor,  # [Tk, HKV, D]
    v: torch.Tensor,  # [Tk, HKV, D]
    scale: float,
    q_pos0_in_kv: int | None,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention with explicit logsumexp.

    q_pos0_in_kv: absolute position of query token 0 measured in this KV
    block's coordinate frame, or None for non-causal (all keys visible).
    window: optional sliding-window size (keys j visible iff j > qpos - w).

    Returns (out [Tq, H, D] fp32, lse [H, Tq] fp32, natural log).
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    kf = k.float()
    vf = v.float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale  # [H, Tq, Tk]
    if q_pos0_in_kv is not None:
        tq, tk = q.shape[0], k.shape[0]
        i = torch.arange(tq, device=q.device)[:, None]
        j = torch.arange(tk, device=q.device)[None, :]
        qpos = q_pos0_in_kv + i
        mask = j <= qpos
        if window is not None:
            mask &= j > qpos - window
        scores = scores.masked_fill(~mask, float("-inf"))
    m = scores.amax(dim=-1)  # [H, Tq]
    p = torch.exp(scores - m[..., None])
    denom = p.sum(dim=-1)  # [H, Tq]
    out = torch.einsum("hqk,khd->qhd", p, vf)
    out = out / denom.permute(1, 0)[..., None]
    lse = m + torch.log(denom)
    return out, lse


def _lse_merge(
    o1: torch.Tensor,  # [Tq, H, D]
    lse1: torch.Tensor,  # [H, Tq]
    o2: torch.Tensor,
    lse2: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge of two partial attentions over disjoint KV sets."""
    m = torch.maximum(lse1, lse2)  # [H, Tq]
    w1 = torch.exp(lse1 - m)
    w2 = torch.exp(lse2 - m)
    w1e = w1.permute(1, 0)[..., None]  # [Tq, H, 1]
    w2e = w2.permute(1, 0)[..., None]
    return (o1 * w1e + o2 * w2e) / (w1e + w2e)


# ---------------------------------------------------------------------------
# Backend class (registered as AttentionBackendEnum.CUSTOM)
# ---------------------------------------------------------------------------
class HcpRingAttentionBackend(FlashAttentionBackend):
    """Ring-attention backend: FlashAttention cache layout + metadata builder,

    but a plain-PyTorch AttentionImpl with online-softmax peer merge.

    Note: get_name() must stay "CUSTOM" — vllm's Attention layer does
    `AttentionBackendEnum[self.attn_backend.get_name()]` and CUSTOM is the
    enum member this backend is registered under.
    """

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["HcpRingAttentionImpl"]:
        return HcpRingAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return FlashAttentionMetadataBuilder


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------
class HcpRingAttentionImpl(AttentionImpl):
    """Plain-PyTorch attention + online-softmax merge with peer chunk KV.

    Reads/writes the standard vLLM paged KV cache
    ([num_blocks, 2, block_size, num_kv_heads, head_dim]).  The peer chunk's
    KV is used transiently (staging dict or split of the local cache) and is
    never written back.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.supports_quant_query_input = False
        if alibi_slopes is not None:
            raise NotImplementedError("HCP ring backend: ALiBi not supported")
        if logits_soft_cap:
            raise NotImplementedError("HCP ring backend: soft cap not supported")
        # Qwen2-0.5B ships use_sliding_window=False, so this is normally None.
        # If a model does set a window we honor it on the causal/local path.
        self.sliding_window = sliding_window
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        logger.info_once(
            "HcpRingAttentionImpl: plain-PyTorch attention with online-softmax "
            "peer merge (HCP_RING_SPLIT_TOKENS=%s)",
            os.environ.get("HCP_RING_SPLIT_TOKENS", "0"),
        )

    # -- KV cache update -----------------------------------------------------
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,  # [num_tokens(padded), HKV, D]
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, HKV, D]
        slot_mapping: torch.Tensor,  # [num_actual_tokens]
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        n = slot_mapping.shape[0]
        # Memory-split decode with sharded growth: a decode-ring decode step
        # (single-token write while a decode-ring registration is live) does
        # NOT write the pool — the token's KV is routed by the growth
        # assignment policy (owner's compact buffer, or shipped to its
        # assignee).  The pool therefore durably holds only the owner's own
        # chunk.  PoC scope: decode-ring runs are single-request.  The skip
        # is counted (slots alone cannot prove it: freed slots are recycled
        # by later requests and collapse in the slot-id set).
        if n == 1 and _decode_ring_enabled() and DECODE_RING_MAP:
            WRITE_TRACK["skipped"] += n
            return
        key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        block_idx = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_off = slot_mapping % block_size
        key_cache[block_idx, block_off] = key[:n]
        value_cache[block_idx, block_off] = value[:n]
        if WRITE_TRACK["enabled"]:
            WRITE_TRACK["slots"].update(slot_mapping.tolist())

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # [num_tokens, H, D]
        key: torch.Tensor,  # [num_tokens, HKV, D]
        value: torch.Tensor,  # [num_tokens, HKV, D]
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, HKV, D]
        attn_metadata,  # FlashAttentionMetadata
        output: torch.Tensor,  # [num_tokens, H, D]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quant not supported")
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        assert self.attn_type == AttentionType.DECODER, (
            "HCP ring backend PoC supports decoder attention only"
        )

        num_actual = attn_metadata.num_actual_tokens
        query = query[:num_actual]
        output = output[:num_actual]

        key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        # Small tensors; .tolist() costs one GPU sync — fine for the PoC.
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        block_table = attn_metadata.block_table
        BATCH_STATS["max_reqs"] = max(BATCH_STATS["max_reqs"], len(seq_lens))

        split = _split_tokens() if _ring_enabled() else 0

        for r in range(len(seq_lens)):
            qs, qe = qsl[r], qsl[r + 1]
            tq = qe - qs
            tk = seq_lens[r]
            if tq == 0:
                continue
            nb = (tk + block_size - 1) // block_size
            blocks = block_table[r, :nb].long()
            k_all = key_cache[blocks].reshape(-1, self.num_kv_heads, self.head_size)[
                :tk
            ]
            v_all = value_cache[blocks].reshape(
                -1, self.num_kv_heads, self.head_size
            )[:tk]

            q_r = query[qs:qe]
            qpos0 = tk - tq  # global position of this request's first query

            # Memory-split decode (accumulator ring): decode steps only.
            # The staged prefix KV is released at the transition; attention =
            # local partial (own chunk + owned growth share + current token,
            # causal) merged with each peer's remote partial (its chunk
            # snapshot + its growth share, non-causal).  Decode growth is
            # sharded across the ring by the assignment policy: owned tokens
            # go to a compact per-layer buffer, the rest are piggybacked on
            # the NEXT step's RPC to their assignee (one-step pipeline lag;
            # the current token always participates locally this step).
            dr = None
            if _decode_ring_enabled() and tq == 1:
                dr = DECODE_RING_MAP.get(int(block_table[r, 0]))
            if dr is not None:
                _release_staged_for_request(int(block_table[r, 0]))
                plen = dr["prefix_len"]
                if dr["own_chunk_len"] is None:
                    # First decode step: tk = prompt_len + 1 (current token).
                    dr["own_chunk_len"] = tk - 1 - plen
                own_len = dr["own_chunk_len"]
                layer_name = layer.layer_name
                pos = tk - 1  # global position of the current (query) token

                from hcp_vllm_plugin.ring_decode import growth_assignee

                assignee = growth_assignee(
                    pos, dr["ring_size"], dr["growth_policy"]
                )
                # Last step's token for THIS layer rides this step's RPC to
                # its assignee (append-then-serve keeps the ordering).
                ship = dr["pending_growth"].pop(layer_name, None)

                # Current token K/V (post-RoPE), [1, HKV, D].
                k_cur = key[qs:qe].contiguous()
                v_cur = value[qs:qe].contiguous()
                # Local segment: own chunk (pool) + owned growth + current.
                k_loc = k_all[plen : plen + own_len]
                v_loc = v_all[plen : plen + own_len]
                g = dr["growth"].get(layer_name)
                if g is not None:
                    k_loc = torch.cat([k_loc, g[0]], dim=0)
                    v_loc = torch.cat([v_loc, g[1]], dim=0)
                k_loc = torch.cat([k_loc, k_cur], dim=0)
                v_loc = torch.cat([v_loc, v_cur], dim=0)
                o_acc, lse_acc = _attn(
                    q_r,
                    k_loc,
                    v_loc,
                    self.scale,
                    k_loc.shape[0] - 1,
                    self.sliding_window,
                )
                if assignee == dr["own_index"]:
                    dr["growth"][layer_name] = (
                        k_cur if g is None else torch.cat([g[0], k_cur], 0),
                        v_cur if g is None else torch.cat([g[1], v_cur], 0),
                    )
                else:
                    dr["pending_growth"][layer_name] = (assignee, k_cur, v_cur)
                transport = dr.get("transport")
                if transport is None:
                    raise RuntimeError(
                        "HCP decode-ring requires the P2P ring transport; "
                        "star HTTP decode was removed (decision-no-star-"
                        "transport-20260727: N>3 networks are not fully "
                        "connected; ring keeps topology linear)."
                    )
                # One ring_decode_step call replaces the entire peer
                # for-loop: the packet traverses all peers and returns the
                # globally merged result.
                from hcp_vllm_plugin.ring_decode import ring_decode_step

                # Growth piggyback: ship last step's pending growth to
                # its assignee via the ring packet.
                gk = gv = None
                growth_target = -1
                if ship is not None:
                    gk, gv = ship[1], ship[2]
                    # Map growth assignee index to transport ring index:
                    # growth 0..N-2 = peers -> transport 1..N-1
                    growth_target = ship[0] + 1
                # The first peer's chunk key identifies the KV segment
                # for the ring nodes (all peers share the same layer
                # name; chunk key routes to the correct snapshot).
                first_chunk = dr["peer_chunks"][0][0] if dr["peer_chunks"] else ""
                o_acc, lse_acc = ring_decode_step(
                    transport,
                    q_r,
                    o_acc,
                    lse_acc,
                    self.scale,
                    layer_name,
                    first_chunk,
                    growth_k=gk,
                    growth_v=gv,
                    growth_target=growth_target,
                )
                output[qs:qe] = o_acc.to(output.dtype)
                continue
            # Determine the peer (prefix) KV for THIS request.  Lookup is
            # per-row: the request's first block id identifies its ORDERED
            # staged prefix chunks (registered by the KV connector).  The
            # chunks form a globally contiguous prefix, so they are
            # concatenated into one contiguous peer KV segment and consumed
            # with a single non-causal peer pass (linear copy, mathematically
            # equivalent to N-1 separate passes).  PoC convenience: with no
            # mapping, a SINGLE-request batch, and exactly ONE staged chunk
            # (no-connector runs such as validate_ring_backend customst),
            # use that chunk.
            staged = None
            if _ring_enabled():
                ck_list = PEER_REQ_MAP.get(int(block_table[r, 0]))
                if ck_list is None and len(seq_lens) == 1:
                    # Single-request fallback ONLY (no-connector runs such as
                    # validate_ring_backend customst).  In a multi-request
                    # batch an unmapped request must NOT adopt another
                    # request's staged chunk — the fallback would bind the
                    # wrong peer KV (and trip the WRITE_TRACK peer-region
                    # check on the request's own slots).
                    chunks = {key[0] for key in PEER_KV_STAGING}
                    if len(chunks) == 1:
                        ck_list = [next(iter(chunks))]
                if ck_list:
                    parts = []
                    for ck in ck_list:
                        kv = PEER_KV_STAGING.get((ck, layer.layer_name))
                        if kv is None:
                            parts = None
                            break
                        parts.append(kv)
                    if parts is None:
                        logger.warning(
                            "HCP ring: request mapped to chunks %s but staging "
                            "incomplete at layer %s; falling back to local path",
                            ck_list,
                            layer.layer_name,
                        )
                    elif len(parts) == 1:
                        staged = parts[0]
                    else:
                        staged = (
                            torch.cat([p[0] for p in parts], dim=0),
                            torch.cat([p[1] for p in parts], dim=0),
                        )
            if staged is not None:
                k_peer, v_peer = staged
                peer_end = min(k_peer.shape[0], tk)
            elif split > 0:
                peer_end = min(split, tk)
                k_peer, v_peer = k_all[:peer_end], v_all[:peer_end]
            else:
                peer_end = 0
                k_peer = v_peer = None

            if peer_end == 0:
                # Vanilla path: plain causal attention over all local KV.
                o, _ = _attn(
                    q_r, k_all, v_all, self.scale, qpos0, self.sliding_window
                )
                output[qs:qe] = o.to(output.dtype)
                continue

            # Ring path.  Queries still inside chunk A (qpos < peer_end) get
            # plain causal attention; queries in chunk B get the merged
            # local(causal, chunk B) + peer(non-causal, chunk A) attention.
            if WRITE_TRACK["enabled"]:
                # Debug: prove the peer chunk's pool slots were never written
                # by this worker (memory-splitting evidence).
                nb_a = (peer_end + block_size - 1) // block_size
                blk_a = block_table[r, :nb_a].long()
                slots_a = (
                    blk_a[:, None] * block_size
                    + torch.arange(block_size, device=blk_a.device)[None, :]
                ).flatten()[:peer_end]
                overlap = WRITE_TRACK["slots"].intersection(slots_a.tolist())
                if overlap:
                    WRITE_TRACK["overlap"] += len(overlap)
                    logger.warning(
                        "HCP ring: %d chunk-A pool slots were written locally "
                        "(expected 0 in a memory-splitting worker)",
                        len(overlap),
                    )
            n_a = max(0, min(tq, peer_end - qpos0))
            if n_a > 0:
                o_a, _ = _attn(
                    q_r[:n_a], k_all, v_all, self.scale, qpos0,
                    self.sliding_window,
                )
                output[qs : qs + n_a] = o_a.to(output.dtype)
            if n_a < tq:
                q_b = q_r[n_a:]
                o_loc, lse_loc = _attn(
                    q_b,
                    k_all[peer_end:],
                    v_all[peer_end:],
                    self.scale,
                    qpos0 + n_a - peer_end,
                    self.sliding_window,
                )
                o_peer, lse_peer = _attn(
                    q_b, k_peer, v_peer, self.scale, None
                )
                o_b = _merge(o_loc, lse_loc, o_peer, lse_peer)
                output[qs + n_a : qe] = o_b.to(output.dtype)

        return output
