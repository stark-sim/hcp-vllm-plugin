#!/usr/bin/env python
"""Decode-phase validation of the HCP ring attention backend (vLLM 0.23, ROCm).

Proves that the CUSTOM `HcpRingAttentionBackend` preserves normal vLLM
baseline capabilities:

  nopeer : degenerate/no-peer case — CUSTOM backend with HCP_RING_SPLIT_TOKENS=0
           on a single contiguous prompt must match the default backend
           (multi-step greedy decode, all tokens).
  batch  : continuous batching — several prompts of different lengths in ONE
           generate() call on the CUSTOM backend (no peer); every request's
           output must match its single-node reference.  Also reports the
           baseline (default backend batched vs single-run) for context.
  cp     : multi-step decode over the 2-process ring-connector CP path
           (producer + consumer), via validate_ring_connector.py with
           --decode 8 and --decode 16.
  all    : nopeer + batch + cp.

Note on CP + batching: PEER_KV_STAGING is keyed by layer name only, so the
CP-with-peer path is restricted to ONE in-flight request (the connector
validation uses max_num_seqs=1).  The no-peer batched case has no such
limitation (no staged state).  See the final report for details.

Run (with vllm-rocm env + ROCm LD_LIBRARY_PATH, cd /tmp):
  python validate_decode.py --mode all
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/home/stark/models/Qwen2-0.5B-1M"
CONNECTOR_SCRIPT = "/home/stark/hetero-cp-ringattn/hcp_vllm_plugin/validate_ring_connector.py"


# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------
def _token_buffer(min_tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = " ".join(
        f"Report {i}: analyst number {i} reviews shipment {(i * 3) % 29} "
        f"across region {i % 11} and records {i} anomalies."
        for i in range(800)
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= min_tokens, f"buffer too short: {len(ids)}"
    return ids


def build_prompt(total: int) -> list[int]:
    return _token_buffer(total)[:total]


# (start, length) windows — disjoint, so no shared prefixes between prompts.
BATCH_WINDOWS = [(0, 64), (2000, 200), (3000, 350), (4000, 700),
                 (5000, 1000), (7000, 1500)]


def build_prompt_set() -> list[list[int]]:
    need = max(s + n for s, n in BATCH_WINDOWS)
    buf = _token_buffer(need)
    return [buf[s : s + n] for s, n in BATCH_WINDOWS]


# ---------------------------------------------------------------------------
# engine helpers
# ---------------------------------------------------------------------------
def find_model(llm):
    eng = llm.llm_engine
    core = getattr(eng, "engine_core", None)
    ec = getattr(core, "engine_core", core)
    me = getattr(ec, "model_executor", None)
    w = getattr(me, "driver_worker", None)
    mr = getattr(w, "model_runner", None)
    model = getattr(mr, "model", None)
    if model is None:
        raise RuntimeError("could not locate model inside the engine")
    return model


def run_capture(llm, prompts: list[list[int]], decode: int):
    """Greedy-generate `decode` tokens for each prompt (one generate call).

    Returns (last_logits [R, vocab] fp32 cpu, tokens_per_req [R, decode]).
    """
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    model = find_model(llm)
    lp = getattr(model, "logits_processor", None)
    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            captured.append(t.detach().float().cpu())

    h = lp.register_forward_hook(hook)
    sp = SamplingParams(temperature=0.0, max_tokens=decode)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    h.remove()
    if not captured:
        raise RuntimeError("logits hook captured nothing")
    last = captured[-1]
    last = last.reshape(last.shape[0], -1)  # [R, vocab]
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    return last, tokens


def make_llm(custom: bool, max_num_seqs: int, gpu_mem: float):
    from vllm import LLM

    kwargs = dict(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        max_num_seqs=max_num_seqs,
        disable_hybrid_kv_cache_manager=True,
        seed=0,
    )
    if custom:
        kwargs["attention_backend"] = "CUSTOM"
    return LLM(**kwargs)


def shutdown(llm) -> None:
    import gc

    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    import torch

    torch.cuda.empty_cache()


def compare_runs(ref_logits, ref_tokens, cur_logits, cur_tokens, label: str):
    ok_all = True
    for i in range(len(ref_tokens)):
        tok_match = ref_tokens[i] == cur_tokens[i]
        d = (ref_logits[i] - cur_logits[i]).abs().max().item()
        ok = tok_match and d < 0.1
        ok_all = ok_all and ok
        print(f"  [{label} req{i}] tokens {'MATCH' if tok_match else 'MISMATCH'} "
              f"(ref[:6]={ref_tokens[i][:6]} cur[:6]={cur_tokens[i][:6]}), "
              f"max|logit diff|={d:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def mode_nopeer(args) -> bool:
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"
    ids = build_prompt(args.total)
    print(f"[nopeer] single prompt {len(ids)} tokens, decode {args.decode}",
          flush=True)

    ref = make_llm(custom=False, max_num_seqs=1, gpu_mem=args.gpu_mem)
    ref_logits, ref_tokens = run_capture(ref, [ids], args.decode)
    print(f"[nopeer] ref tokens: {ref_tokens[0]}", flush=True)
    shutdown(ref)

    cus = make_llm(custom=True, max_num_seqs=1, gpu_mem=args.gpu_mem)
    cus_logits, cus_tokens = run_capture(cus, [ids], args.decode)
    print(f"[nopeer] custom tokens: {cus_tokens[0]}", flush=True)
    shutdown(cus)

    ok = compare_runs(ref_logits, ref_tokens, cus_logits, cus_tokens, "nopeer")
    print(f"[nopeer] verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def mode_batch(args) -> bool:
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"
    prompts = build_prompt_set()
    lens = [len(p) for p in prompts]
    print(f"[batch] {len(prompts)} prompts, lengths={lens}, "
          f"decode {args.decode}", flush=True)

    # Reference engine: batched run + per-request single runs.
    ref = make_llm(custom=False, max_num_seqs=len(prompts),
                   gpu_mem=args.gpu_mem)
    ref_b_logits, ref_b_tokens = run_capture(ref, prompts, args.decode)
    ref_s_logits, ref_s_tokens = [], []
    for p in prompts:
        lg, tk = run_capture(ref, [p], args.decode)
        ref_s_logits.append(lg[0])
        ref_s_tokens.append(tk[0])
    shutdown(ref)

    print("[batch] baseline (default backend): batched vs single-run",
          flush=True)
    base_ok = True
    for i in range(len(prompts)):
        m = ref_b_tokens[i] == ref_s_tokens[i]
        base_ok = base_ok and m
        if not m:
            print(f"  [baseline req{i}] MISMATCH batched={ref_b_tokens[i][:6]} "
                  f"single={ref_s_tokens[i][:6]}")
    print(f"[batch] baseline tokens all match: {base_ok}", flush=True)

    # CUSTOM engine: one batched generate call.
    import hcp_vllm_plugin.ring_backend as rb

    rb.reset_batch_stats()
    cus = make_llm(custom=True, max_num_seqs=len(prompts),
                   gpu_mem=args.gpu_mem)
    cus_logits, cus_tokens = run_capture(cus, prompts, args.decode)
    max_reqs = rb.BATCH_STATS["max_reqs"]
    shutdown(cus)
    print(f"[batch] CUSTOM backend max requests in one attention step: "
          f"{max_reqs} (continuous batching evidence)", flush=True)

    import torch

    ref_s_logits_t = torch.stack(ref_s_logits)
    ok = compare_runs(ref_s_logits_t, ref_s_tokens, cus_logits, cus_tokens,
                      "custom-batch")
    ok = ok and max_reqs == len(prompts)
    print(f"[batch] verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def mode_cp(args) -> bool:
    ok_all = True
    for dec, port, run_id in ((8, 8911, "d8"), (16, 8912, "d16")):
        cmd = [sys.executable, CONNECTOR_SCRIPT, "--mode", "all",
               "--total", str(args.total), "--split", str(args.split),
               "--decode", str(dec), "--port", str(port),
               "--run-id", run_id,
               "--done-file", f"/tmp/hcp_ring_conn_done_{run_id}"]
        print(f"[cp] running {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        print(f"[cp] decode={dec} exit={r.returncode} "
              f"({'PASS' if r.returncode == 0 else 'FAIL'})", flush=True)
        ok_all = ok_all and r.returncode == 0
    print(f"[cp] verdict: {'PASS' if ok_all else 'FAIL'}", flush=True)
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["nopeer", "batch", "cp", "all"])
    ap.add_argument("--total", type=int, default=2048)
    ap.add_argument("--split", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    args = ap.parse_args()

    ok = True
    if args.mode in ("nopeer", "all"):
        ok = mode_nopeer(args) and ok
    if args.mode in ("batch", "all"):
        ok = mode_batch(args) and ok
    if args.mode in ("cp", "all"):
        ok = mode_cp(args) and ok
    print(f"=== validate_decode OVERALL: {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
