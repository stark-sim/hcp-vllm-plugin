#!/usr/bin/env python
"""Validate the HCP ring-attention CUSTOM backend against single-node vLLM.

Modes:
  ref     : full prompt, default backend (auto -> Triton on ROCm). Baseline.
  custom0 : full prompt, CUSTOM ring backend, HCP_RING_SPLIT_TOKENS=0
            (sanity: custom backend without the merge == vanilla attention)
  custom  : full prompt, CUSTOM ring backend, HCP_RING_SPLIT_TOKENS=split
            (chunk-A queries: plain causal; chunk-B queries: online-softmax
            merge of local chunk-B attention + transient chunk-A attention)
  customst: like custom, but the peer (chunk-A) KV is provided through the
            module-level staging dict (ring_backend.stage_peer_kv), captured
            from a separate chunk-A-only prefill in the same process — this
            exercises exactly the API the KV connector will use.
  compare : compare saved outputs of ref vs custom0 / custom / customst
  all     : run ref, custom0, custom, customst as subprocesses, then compare

PASS criteria: sampled next token identical, top-5 token set identical, and
max|logit diff| below tolerance (fp16 numerics; reported either way).

Run (with the vllm-rocm conda env + ROCm LD_LIBRARY_PATH set):
  python validate_ring_backend.py --mode all --total 2048 --split 1024
"""

import argparse
import os
import subprocess
import sys

# In-process engine core so the logits hook (same process) actually fires.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/home/stark/models/Qwen2-0.5B-1M"


def build_prompt_ids(total: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = " ".join(
        f"Paragraph {i}: the quick brown fox jumps over the lazy dog, "
        f"then counts {i} clouds drifting over mountain {i % 7}."
        for i in range(512)
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= total, f"prompt too short: {len(ids)} < {total}"
    return ids[:total]


def find_model(llm):
    """Walk LLM -> in-proc EngineCore -> model (defensive, vLLM 0.23)."""
    eng = llm.llm_engine
    core = getattr(eng, "engine_core", None)
    ec = getattr(core, "engine_core", core)  # InprocClient.engine_core
    me = getattr(ec, "model_executor", None)
    w = getattr(me, "driver_worker", None)
    mr = getattr(w, "model_runner", None)
    model = getattr(mr, "model", None)
    if model is None:
        raise RuntimeError("could not locate model inside the engine")
    return model


def run_one(mode: str, total: int, split: int, out_prefix: str) -> None:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if mode == "custom":
        os.environ["HCP_RING_SPLIT_TOKENS"] = str(split)
    elif mode == "customst":
        # Fallback only; the staged dict (populated below) takes precedence.
        os.environ["HCP_RING_SPLIT_TOKENS"] = str(split)
    elif mode == "custom0":
        os.environ["HCP_RING_SPLIT_TOKENS"] = "0"

    ids = build_prompt_ids(total)
    print(f"[{mode}] prompt tokens: {len(ids)} (split={split})")

    llm_kwargs = dict(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=4096,
        disable_hybrid_kv_cache_manager=True,
        seed=0,
    )
    if mode in ("custom", "custom0", "customst"):
        llm_kwargs["attention_backend"] = "CUSTOM"

    llm = LLM(**llm_kwargs)
    model = find_model(llm)

    kv_hooks = []
    capture_on = {"on": False}
    if mode == "customst":
        # Phase-0 hook: capture per-layer chunk-A K/V (post-RoPE K) into the
        # ring backend's module-level staging dict during a chunk-A prefill.
        import hcp_vllm_plugin.ring_backend as rb
        from vllm.model_executor.layers.attention import Attention

        rb.clear_peer_kv()

        def make_pre_hook():
            def pre(mod, args):
                if not capture_on["on"] or len(args) < 3 or args[1] is None:
                    return
                k, v = args[1], args[2]
                t = k.shape[0]
                rb.stage_peer_kv(
                    "chunkA",
                    mod.layer_name,
                    k.view(t, mod.num_kv_heads, mod.head_size).detach().clone(),
                    v.view(t, mod.num_kv_heads, mod.head_size).detach().clone(),
                )

            return pre

        for m in model.modules():
            if isinstance(m, Attention):
                kv_hooks.append(m.register_forward_pre_hook(make_pre_hook()))

    # Capture last-token logits via a hook on the LogitsProcessor.
    lp = getattr(model, "logits_processor", None)
    if lp is None:
        raise RuntimeError("model has no logits_processor to hook")
    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            captured.append(t.detach().float().cpu())

    h = lp.register_forward_hook(hook)

    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=8)
    captured.clear()  # ignore warmup/profile captures

    if mode == "customst":
        # Phase 1: chunk-A-only prefill -> stage peer KV per layer.
        capture_on["on"] = True
        llm.generate([TokensPrompt(prompt_token_ids=ids[:split])], sp)
        capture_on["on"] = False
        n_staged = len(rb.PEER_KV_STAGING)
        print(f"[{mode}] staged peer KV for {n_staged} layers "
              f"(chunk A = {split} tokens)")
        assert n_staged > 0, "no peer KV was staged"
        captured.clear()  # drop phase-1 logits captures

    outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    h.remove()
    for hh in kv_hooks:
        hh.remove()

    comp = outs[0].outputs[0]
    token = comp.token_ids[0]
    top_logprobs = {
        tid: lp_.logprob for tid, lp_ in (comp.logprobs[0].items() if comp.logprobs else [])
    }
    if not captured:
        raise RuntimeError("logits hook captured nothing")
    logits = captured[-1].reshape(-1)  # [vocab]

    out_path = f"{out_prefix}_{mode}.pt"
    torch.save(
        {
            "mode": mode,
            "token": token,
            "logits": logits,
            "top_logprobs": top_logprobs,
            "n_prompt_tokens": len(ids),
            "split": split,
        },
        out_path,
    )
    top5 = torch.topk(logits, 5)
    print(f"[{mode}] sampled token: {token}")
    print(f"[{mode}] top-5: ids={top5.indices.tolist()} "
          f"vals={[round(v, 4) for v in top5.values.tolist()]}")
    print(f"[{mode}] saved -> {out_path}")


def compare(out_prefix: str, modes: list[str]) -> bool:
    import torch

    ref = torch.load(f"{out_prefix}_ref.pt", weights_only=False)
    ok_all = True
    for mode in modes:
        cur = torch.load(f"{out_prefix}_{mode}.pt", weights_only=False)
        d = (ref["logits"] - cur["logits"]).abs()
        ref_top5 = set(torch.topk(ref["logits"], 5).indices.tolist())
        cur_top5 = set(torch.topk(cur["logits"], 5).indices.tolist())
        token_match = ref["token"] == cur["token"]
        top5_match = ref_top5 == cur_top5
        max_diff = d.max().item()
        argmax_diff = d[ref["logits"].argmax()].item()
        ok = token_match and top5_match and max_diff < 0.1
        ok_all = ok_all and ok
        print(f"--- ref vs {mode} ---")
        print(f"  sampled token : ref={ref['token']} {mode}={cur['token']} "
              f"-> {'MATCH' if token_match else 'MISMATCH'}")
        print(f"  top-5 set     : {'MATCH' if top5_match else 'MISMATCH'} "
              f"(ref={sorted(ref_top5)} {mode}={sorted(cur_top5)})")
        print(f"  max|logit diff| = {max_diff:.6f}   "
              f"|diff| at ref argmax = {argmax_diff:.6f}")
        print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["ref", "custom0", "custom", "customst",
                             "compare", "all"])
    ap.add_argument("--total", type=int, default=2048)
    ap.add_argument("--split", type=int, default=1024)
    ap.add_argument("--out-prefix", default="/tmp/hcp_ring")
    args = ap.parse_args()

    if args.mode in ("ref", "custom0", "custom", "customst"):
        run_one(args.mode, args.total, args.split, args.out_prefix)
    elif args.mode == "compare":
        modes = ["custom0", "custom"]
        if os.path.exists(f"{args.out_prefix}_customst.pt"):
            modes.append("customst")
        ok = compare(args.out_prefix, modes)
        sys.exit(0 if ok else 1)
    else:  # all
        for mode in ("ref", "custom0", "custom", "customst"):
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--mode", mode,
                   "--total", str(args.total),
                   "--split", str(args.split),
                   "--out-prefix", args.out_prefix]
            print(f"=== running {' '.join(cmd)} ===", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"mode {mode} FAILED with exit code {r.returncode}")
                sys.exit(r.returncode)
        ok = compare(args.out_prefix, ["custom0", "custom", "customst"])
        print("=== OVERALL:", "PASS" if ok else "FAIL", "===")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
