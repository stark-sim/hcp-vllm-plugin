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
    returns False, so FlashAttentionImpl cannot run here at all.  This file
    therefore implements attention in plain PyTorch (fp32 accumulation) and
    computes the logsumexp (LSE, natural log) explicitly.
  * vLLM's triton `merge_attn_states` matched a PyTorch reference in a toy
    probe here (4.8e-7), but the HCP team has seen it return `inf` in other
    ROCm runs — so the merge is implemented in plain PyTorch instead.
  * The online-softmax merge math itself was verified on this GPU:
    merged two-chunk attention vs full attention, max|diff| = 3.0e-7 (fp32).

Configuration (env vars):
  HCP_RING_SPLIT_TOKENS : int, position boundary between chunk A and chunk B.
                          0 (default) disables the ring merge (plain local
                          causal attention, i.e. vanilla behavior).
  HCP_RING_ENABLED      : "1" (default) / "0" master switch.

Peer KV staging API (for the future KV connector):
  stage_peer_kv(layer_name, k, v) / clear_peer_kv()
  k/v: [num_tokens, num_kv_heads, head_dim] contiguous, post-RoPE K.
  When a layer has staged peer KV it is used as the peer (chunk-A) KV and the
  first `k.shape[0]` positions of the local cache are excluded from the
  local path.  Without staged KV, HCP_RING_SPLIT_TOKENS splits the local
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
# Peer KV staging (module level).  The KV connector will write here; the
# AttentionImpl reads (never stores) it during forward.
# ---------------------------------------------------------------------------
PEER_KV_STAGING: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def stage_peer_kv(layer_name: str, k: torch.Tensor, v: torch.Tensor) -> None:
    """Stage a peer chunk's K/V for one attention layer.

    k, v: [num_tokens, num_kv_heads, head_dim] — K must be post-RoPE.
    """
    PEER_KV_STAGING[layer_name] = (k, v)


def clear_peer_kv() -> None:
    PEER_KV_STAGING.clear()


# ---------------------------------------------------------------------------
# Debug write-tracking (validation only).  When enabled, do_kv_cache_update
# records which paged-pool slots this worker wrote, and forward() cross-checks
# that no slot belonging to the peer (chunk-A) region was written locally —
# i.e. the consumer's permanent pool provably never holds chunk-A KV.
# ---------------------------------------------------------------------------
WRITE_TRACK: dict = {"enabled": False, "slots": set(), "overlap": 0}


def reset_write_tracking() -> None:
    WRITE_TRACK["enabled"] = True
    WRITE_TRACK["slots"] = set()
    WRITE_TRACK["overlap"] = 0


def _ring_enabled() -> bool:
    return os.environ.get("HCP_RING_ENABLED", "1") == "1"


def _split_tokens() -> int:
    return int(os.environ.get("HCP_RING_SPLIT_TOKENS", "0"))


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
        key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        n = slot_mapping.shape[0]
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

        split = _split_tokens() if _ring_enabled() else 0
        staged = (
            PEER_KV_STAGING.get(layer.layer_name) if _ring_enabled() else None
        )

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

            # Determine the peer (chunk-A) KV for this request.
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
                o, _ = _attn_with_lse(
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
                o_a, _ = _attn_with_lse(
                    q_r[:n_a], k_all, v_all, self.scale, qpos0,
                    self.sliding_window,
                )
                output[qs : qs + n_a] = o_a.to(output.dtype)
            if n_a < tq:
                q_b = q_r[n_a:]
                o_loc, lse_loc = _attn_with_lse(
                    q_b,
                    k_all[peer_end:],
                    v_all[peer_end:],
                    self.scale,
                    qpos0 + n_a - peer_end,
                    self.sliding_window,
                )
                o_peer, lse_peer = _attn_with_lse(
                    q_b, k_peer, v_peer, self.scale, None
                )
                o_b = _lse_merge(o_loc, lse_loc, o_peer, lse_peer)
                output[qs + n_a : qe] = o_b.to(output.dtype)

        return output
