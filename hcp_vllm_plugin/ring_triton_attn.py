"""HCP ring-attention Triton kernel: flash-style attention with LSE output.

Forked from vLLM's triton_prefill_attention._fwd_kernel (itself adapted from
SGLang/lightllm), with two changes the ring backend needs:

  1. LSE output: the kernel already tracks the online-softmax running max
     (m_i, exp2 domain) and denominator (l_i); we additionally store the
     natural-log logsumexp  lse = m_i * ln2 + ln(l_i)  per (head, query),
     which is exactly what cascade-style merging (merge_attn_states) needs.
     vLLM's stock triton kernels do NOT emit LSE (and TRITON_ATTN asserts
     use_cascade is False), hence this fork.
  2. Q_OFFSET: causal masking with the query block sitting at an arbitrary
     offset inside the KV range (ring semantics: chunk-B queries start at
     global position >= 0 within the local KV).  The stock kernel assumes
     query i aligns with key i.

One kernel serves both platforms (Triton compiles for CUDA and ROCm gfx):
  - local pass: causal with Q_OFFSET over the local KV
  - peer pass:  non-causal (Q_OFFSET unused) over the staged peer KV

Shapes (no batch dim; the ring backend calls it per request):
  q: [Tq, H, D] fp16/bf16 contiguous
  k, v: [Tk, HKV, D] same dtype contiguous
  out: [Tq, H, D] same dtype as q
  lse: [H, Tq] fp32, natural log
"""

import torch

from vllm.triton_utils import tl, triton

RCP_LN2 = 1.4426950408889634  # 1 / ln(2)


@triton.jit
def _ring_fwd_kernel(
    Q,
    K,
    V,
    LSE,
    Out,
    sm_scale,  # already multiplied by RCP_LN2 (exp2 domain)
    stride_qs,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_lh,
    stride_os,
    stride_oh,
    Q_LEN,
    K_LEN,
    Q_OFFSET,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_head = tl.program_id(0)
    start_m = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_q = (
        offs_m[:, None] * stride_qs + cur_head * stride_qh + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < Q_LEN) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < Q_LEN, 1, 0)

    end_n = K_LEN
    # Causal pruning: query block covers global positions
    # [Q_OFFSET + start_m*BLOCK_M, Q_OFFSET + (start_m+1)*BLOCK_M)
    if IS_CAUSAL:
        end_n = tl.minimum(end_n, Q_OFFSET + (start_m + 1) * BLOCK_M)
    end_n_limit = block_mask * end_n

    for start_n in range(0, end_n_limit, BLOCK_N):
        pos_q = Q_OFFSET + offs_m[:, None]  # [BLOCK_M, 1]
        pos_k = start_n + offs_n[None, :]  # [1, BLOCK_N]

        mask = pos_k < K_LEN
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + start_n * stride_ks,
            mask=(pos_k < K_LEN) & (mask_d[:, None]),
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + start_n * stride_vs,
            mask=(pos_k < K_LEN) & (mask_d[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]

    # lse in natural log: exp2-domain max * ln2 + ln(denominator)
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    tl.store(
        LSE + cur_head * stride_lh + offs_m,
        lse,
        mask=offs_m < Q_LEN,
    )

    off_o = offs_m[:, None] * stride_os + cur_head * stride_oh + offs_d[None, :]
    tl.store(
        Out + off_o,
        acc,
        mask=(offs_m[:, None] < Q_LEN) & (mask_d[None, :]),
    )


def ring_attn_with_lse(
    q: torch.Tensor,  # [Tq, H, D]
    k: torch.Tensor,  # [Tk, HKV, D]
    v: torch.Tensor,  # [Tk, HKV, D]
    scale: float,
    causal_offset: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash-style attention with LSE output.

    causal_offset: absolute position of query token 0 within the KV range
    (None => non-causal, all keys visible).

    Returns (out [Tq, H, D] same dtype as q, lse [H, Tq] fp32 natural log).
    """
    tq, num_heads, head_dim = q.shape
    tk = k.shape[0]
    num_kv_heads = k.shape[1]
    assert k.shape == v.shape and k.stride(-1) == 1 and q.stride(-1) == 1

    out = torch.empty_like(q)
    lse = torch.empty((num_heads, tq), dtype=torch.float32, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (num_heads, triton.cdiv(tq, BLOCK_M))
    _ring_fwd_kernel[grid](
        q,
        k,
        v,
        lse,
        out,
        scale * RCP_LN2,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        lse.stride(0),
        out.stride(0),
        out.stride(1),
        tq,
        tk,
        causal_offset if causal_offset is not None else 0,
        kv_group_num=num_heads // num_kv_heads,
        BLOCK_M=BLOCK_M,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal_offset is not None,
        Lk=head_dim,
        num_warps=4,
    )
    return out, lse
