#!/usr/bin/env python
"""Numerics probe for the HCP ring Triton kernel + merge_attn_states.

Checks (random data, Qwen2-0.5B shapes H=14 HKV=2 D=64 fp16):
  1. ring_attn_with_lse vs plain-PyTorch _attn_with_lse:
     causal-with-offset, non-causal, GQA, Tq in {1, 37, 512}, Tk up to 2048.
  2. vLLM merge_attn_states (triton) vs plain _lse_merge.
  3. End-to-end ring math: kernel local+peer merged vs plain full attention.

Usage (on a GPU host with vllm env):
  python validate_ring_triton_kernel.py
"""

import sys

import torch

from hcp_vllm_plugin.ring_backend import _attn_with_lse, _lse_merge
from hcp_vllm_plugin.ring_triton_attn import ring_attn_with_lse

H, HKV, D = 14, 2, 64
SCALE = 1.0 / (D**0.5)

failures = []


def check(name: str, got: torch.Tensor, want: torch.Tensor, tol: float) -> None:
    diff = (got.float() - want.float()).abs().max().item()
    ok = diff <= tol
    print(f"  {name:<44} max|diff|={diff:.3e} {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (CUDA or ROCm)"
    dev = "cuda"
    torch.manual_seed(0)

    print("== 1. kernel vs plain torch ==")
    for tq, tk, offset in [
        (1, 1, 0),
        (1, 512, 511),
        (37, 37, 0),
        (37, 512, 475),
        (512, 512, 0),
        (512, 2048, 1536),
    ]:
        q = torch.randn(tq, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(tk, HKV, D, device=dev, dtype=torch.float16)
        v = torch.randn(tk, HKV, D, device=dev, dtype=torch.float16)

        o_t, lse_t = _attn_with_lse(q, k, v, SCALE, offset, None)
        o_k, lse_k = ring_attn_with_lse(q, k, v, SCALE, offset)
        check(f"causal off={offset} Tq={tq} Tk={tk} out", o_k, o_t.to(o_k.dtype), 2e-3)
        check(f"causal off={offset} Tq={tq} Tk={tk} lse", lse_k, lse_t, 2e-3)

        o_t2, lse_t2 = _attn_with_lse(q, k, v, SCALE, None, None)
        o_k2, lse_k2 = ring_attn_with_lse(q, k, v, SCALE, None)
        check(f"noncausal       Tq={tq} Tk={tk} out", o_k2, o_t2.to(o_k2.dtype), 2e-3)
        check(f"noncausal       Tq={tq} Tk={tk} lse", lse_k2, lse_t2, 2e-3)

    print("== 2. merge_attn_states vs plain _lse_merge ==")
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    tq, tk_a, tk_b = 512, 1024, 1024
    q = torch.randn(tq, H, D, device=dev, dtype=torch.float16)
    ka = torch.randn(tk_a, HKV, D, device=dev, dtype=torch.float16)
    va = torch.randn(tk_a, HKV, D, device=dev, dtype=torch.float16)
    kb = torch.randn(tk_b, HKV, D, device=dev, dtype=torch.float16)
    vb = torch.randn(tk_b, HKV, D, device=dev, dtype=torch.float16)
    o_a, lse_a = ring_attn_with_lse(q, ka, va, SCALE, None)
    o_b, lse_b = ring_attn_with_lse(q, kb, vb, SCALE, 0)

    m_plain = _lse_merge(o_b, lse_b, o_a, lse_a)  # local + peer
    m_tri = torch.empty_like(o_b)
    merge_attn_states(m_tri, o_a, lse_a, o_b, lse_b)  # prefix=peer, suffix=local
    check("merge_attn_states out", m_tri, m_plain.to(m_tri.dtype), 2e-3)

    print("== 3. end-to-end: two-chunk merged vs full attention ==")
    tk_full = 2048
    split = 1024
    q3 = torch.randn(512, H, D, device=dev, dtype=torch.float16)
    k3 = torch.randn(tk_full, HKV, D, device=dev, dtype=torch.float16)
    v3 = torch.randn(tk_full, HKV, D, device=dev, dtype=torch.float16)
    # queries are the last 512 tokens: local causal over chunk B + peer chunk A
    o_full, _ = _attn_with_lse(q3, k3, v3, SCALE, tk_full - 512, None)
    o_loc, lse_loc = ring_attn_with_lse(q3, k3[split:], v3[split:], SCALE, tk_full - 512 - split)
    o_peer, lse_peer = ring_attn_with_lse(q3, k3[:split], v3[:split], SCALE, None)
    o_merged = _lse_merge(o_loc, lse_loc, o_peer, lse_peer)
    check("ring merged (plain merge) vs full", o_merged, o_full, 3e-3)
    o_merged2 = torch.empty_like(o_loc)
    merge_attn_states(o_merged2, o_peer, lse_peer, o_loc, lse_loc)
    check("ring merged (triton merge) vs full", o_merged2, o_full.to(o_merged2.dtype), 3e-3)

    print("---")
    if failures:
        print(f"FAIL: {len(failures)} checks failed: {failures}")
        sys.exit(1)
    print("PASS: all numerics checks")


if __name__ == "__main__":
    main()
