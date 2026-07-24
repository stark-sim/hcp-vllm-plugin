#!/usr/bin/env python
"""3-instance (N=3) validation of the generic-N HCP ring-KV connector.

Topology (single machine, three OS processes, loopback HTTP transport):
  * A = producer (chunk c0): prefills ids[:s0], serves its KV store on :portA.
  * B = relay    (chunk c1): gets ids[:s0+s1]; its connector marks c0 (s0
    tokens) external, stages c0 over HTTP from A (TRANSIENT staging, never in
    B's paged pool), prefills c1, and saves c1's per-layer KV to B's own
    store, served on :portB.  B is consumer(c0) + producer(c1) in one
    instance; the ready state cascades (B's _READY only after A's).
  * C = consumer (chunk c2): gets the full prompt ids[:s0+s1+s2]; marks
    c0+c1 (s0+s1 tokens) external via the PLURAL per-request params
    chunk_ids=[c0,c1] / peer_urls=[A,B], stages both prefix chunks (from two
    different peers), and the ring backend concatenates them into one
    contiguous peer KV segment for a single non-causal peer pass.

Checks (consumer mode):
  1. C's greedy tokens match a single-node reference; last-step logits close.
  2. memory-splitting: WRITE_TRACK proves no prefix-region pool slot was
     written by C; STAGING_STATS proves TWO chunks were staged concurrently.
  3. relay store correctness is proven transitively (C's tokens depend on
     B's c1 KV).

Modes: producer | relay | consumer | all (orchestrates all three).

Run:
  python validate_ring_relay.py --mode all --total 1536 --split0 512 --split1 512
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

# In-process engine core so hooks / WRITE_TRACK run in this process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "/home/stark/models/Qwen2-0.5B-1M"
NUM_LAYERS = 24  # Qwen2-0.5B


def build_prompt_ids(total: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = " ".join(
        f"Paragraph {i}: the quick brown fox jumps over the lazy dog, "
        f"then counts {i} clouds drifting over mountain {i % 7}."
        for i in range(1024)
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= total, f"prompt too short: {len(ids)} < {total}"
    return ids[:total]


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


def run_capture(llm, ids: list[int], decode: int):
    """Greedy-generate `decode` tokens; return (last-step logits [vocab], tokens)."""
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
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    h.remove()
    if not captured:
        raise RuntimeError("logits hook captured nothing")
    return captured[-1].reshape(-1), list(outs[0].outputs[0].token_ids)


def _base_cfg(args, ring_cfg: dict, kv_role: str) -> dict:
    return {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": kv_role,
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": ring_cfg,
    }


def mode_producer(args) -> None:
    """Instance A: chunk c0 producer."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = build_prompt_ids(args.total)[: args.split0]
    print(f"[A/producer] c0 = {len(ids)} tokens; store {args.store_a} "
          f"on :{args.port_a}", flush=True)
    cfg = _base_cfg(args, {
        "ring_role": "producer",
        "ring_chunk_id": "c0",
        "ring_shared_path": args.store_a,
        "ring_run_id": args.run_id,
        "ring_serve_port": args.port_a,
    }, "kv_producer")
    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_ab, max_model_len=args.max_model_len,
              max_num_seqs=1, disable_hybrid_kv_cache_manager=True,
              attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=1))
    print(f"[A/producer] c0 prefilled, sample {out[0].outputs[0].token_ids[0]}; "
          f"KV served on :{args.port_a}", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    print("[A/producer] exiting", flush=True)


def mode_relay(args) -> None:
    """Instance B: chunk c1 relay — consumes c0 (external prefix), produces c1."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = build_prompt_ids(args.total)[: args.split0 + args.split1]
    print(f"[B/relay] prompt = {len(ids)} (prefix c0={args.split0} external, "
          f"own c1={args.split1}); peer A={args.url_a}; store {args.store_b} "
          f"on :{args.port_b}", flush=True)
    cfg = _base_cfg(args, {
        "ring_role": "relay",
        "ring_chunk_id": "c1",
        "ring_prefix_chunk_ids": "c0",
        "ring_prefix_len": args.split0,
        "ring_peer_url": args.url_a,
        "ring_shared_path": args.store_b,
        "ring_run_id": args.run_id,
        "ring_serve_port": args.port_b,
    }, "kv_both")
    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_ab, max_model_len=args.max_model_len,
              max_num_seqs=1, disable_hybrid_kv_cache_manager=True,
              enable_prefix_caching=False,
              attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=1))
    print(f"[B/relay] c1 prefilled (c0 staged transiently from A), sample "
          f"{out[0].outputs[0].token_ids[0]}; c1 KV served on :{args.port_b}",
          flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    print("[B/relay] exiting", flush=True)


def mode_consumer(args) -> None:
    """Instance C: chunk c2 consumer — stages c0 (from A) + c1 (from B)."""
    import torch
    from vllm import LLM

    s2 = args.total - args.split0 - args.split1
    ids = build_prompt_ids(args.total)
    print(f"[C/consumer] full prompt = {len(ids)} (prefix c0+c1 = "
          f"{args.split0 + args.split1} external, own c2 = {s2}); "
          f"peers A={args.url_a} B={args.url_b}", flush=True)

    # ---- single-node reference (default backend) ----
    ref = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_c, max_model_len=args.max_model_len,
              max_num_seqs=1, disable_hybrid_kv_cache_manager=True, seed=0)
    ref_logits, ref_tokens = run_capture(ref, ids, args.decode)
    print(f"[ref] tokens: {ref_tokens}", flush=True)
    try:
        ref.llm_engine.shutdown()
    except Exception:
        pass
    del ref
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- consumer instance (CUSTOM backend + ring connector) ----
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"
    cfg = _base_cfg(args, {
        "ring_role": "consumer",
        "ring_prefix_chunk_ids": "c0,c1",
        "ring_prefix_len": args.split0 + args.split1,
        "ring_shared_path": args.store_c,
        "ring_run_id": args.run_id,
    }, "kv_consumer")
    cons = LLM(model=MODEL, dtype="float16", enforce_eager=True,
               gpu_memory_utilization=args.gpu_mem_c, max_model_len=args.max_model_len,
               max_num_seqs=1, disable_hybrid_kv_cache_manager=True,
               enable_prefix_caching=False,
               attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    import hcp_vllm_plugin.ring_backend as rb

    rb.reset_write_tracking()
    rb.reset_staging_stats()
    rb.reset_impl_stats()
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    model = find_model(cons)
    lp = getattr(model, "logits_processor", None)
    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            captured.append(t.detach().float().cpu())

    h = lp.register_forward_hook(hook)
    # PLURAL per-request params: ordered prefix chunks, each with its own peer.
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.decode,
        extra_args={"kv_transfer_params": {"hcp_ring": {
            "chunk_ids": ["c0", "c1"],
            "peer_urls": [args.url_a, args.url_b],
            "prefix_len": args.split0 + args.split1,
        }}},
    )
    outs = cons.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    h.remove()
    cons_tokens = list(outs[0].outputs[0].token_ids)
    cons_logits = captured[-1].reshape(-1)
    print(f"[consumer] tokens: {cons_tokens}", flush=True)

    # Trigger the next schedule so finished-req cleanup runs before checks.
    cons.generate([TokensPrompt(prompt_token_ids=ids[:32])],
                  SamplingParams(temperature=0.0, max_tokens=1,
                                 extra_args={"kv_transfer_params":
                                             {"hcp_ring": {"prefix_len": 0}}}))

    # ---- correctness ----
    token_match = cons_tokens == ref_tokens
    max_diff = (ref_logits - cons_logits).abs().max().item()
    argmax_diff = (ref_logits - cons_logits).abs()[ref_logits.argmax()].item()

    # ---- memory-splitting / generic-N evidence ----
    n_staged = rb.STAGING_STATS["max_staged_layers"]
    n_chunks = rb.STAGING_STATS["max_concurrent_chunks"]
    staged_len = rb.STAGING_STATS["last_chunk_len"]
    leftover = len(rb.PEER_KV_STAGING)
    leftover_map = len(rb.PEER_REQ_MAP)
    overlap = rb.WRITE_TRACK["overlap"]
    n_written = len(rb.WRITE_TRACK["slots"])
    mem_ok = (
        n_staged == 2 * NUM_LAYERS
        and n_chunks == 2
        and staged_len in (args.split0, args.split1)
        and overlap == 0
        and leftover == 0
        and leftover_map == 0
    )
    impl_ok = (
        rb.IMPL_STATS["attn_triton"] > 0 and rb.IMPL_STATS["attn_torch"] == 0
    )
    print(f"[impl] attn triton/torch: {rb.IMPL_STATS['attn_triton']}/"
          f"{rb.IMPL_STATS['attn_torch']}, merge triton/torch: "
          f"{rb.IMPL_STATS['merge_triton']}/{rb.IMPL_STATS['merge_torch']}")
    print(f"[memsplit] {n_chunks} prefix chunks staged concurrently, "
          f"{n_staged} layer entries (2x{NUM_LAYERS} expected)")
    print(f"[memsplit] consumer wrote {n_written} pool slots (own chunk only); "
          f"prefix-region slots written locally: {overlap}")
    print(f"[memsplit] post-run transient staging freed: "
          f"{leftover == 0 and leftover_map == 0}")

    ok = token_match and max_diff < 0.1 and mem_ok and impl_ok
    print("--- result ---")
    print(f"  tokens match        : {token_match} (ref={ref_tokens} "
          f"cons={cons_tokens})")
    print(f"  max|logit diff|     : {max_diff:.6f} (at ref argmax: "
          f"{argmax_diff:.6f})")
    print(f"  memory-splitting    : {'OK' if mem_ok else 'VIOLATED'}")
    print(f"  triton kernel path  : {'OK' if impl_ok else 'NOT USED'}")
    print(f"  verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def _wait_ready(path: str, proc: subprocess.Popen, name: str, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            print(f"{name} ready ({time.time() - t0:.0f}s)", flush=True)
            return
        if proc.poll() is not None:
            print(f"{name} died early (exit {proc.returncode})", flush=True)
            sys.exit(1)
        time.sleep(2)
    proc.terminate()
    print(f"{name} never became ready", flush=True)
    sys.exit(1)


def mode_all(args) -> None:
    for port in (args.port_a, args.port_b):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                print(f"port {port} already in use; aborting")
                sys.exit(2)
    for d in (args.store_a, args.store_b, args.store_c):
        shutil.rmtree(d, ignore_errors=True)
    if os.path.exists(args.done_file):
        os.remove(args.done_file)

    script = os.path.abspath(__file__)
    common = ["--total", str(args.total), "--split0", str(args.split0),
              "--split1", str(args.split1), "--run-id", args.run_id,
              "--max-model-len", str(args.max_model_len),
              "--store-a", args.store_a, "--store-b", args.store_b,
              "--store-c", args.store_c, "--done-file", args.done_file,
              "--port-a", str(args.port_a), "--port-b", str(args.port_b),
              "--gpu-mem-ab", str(args.gpu_mem_ab), "--gpu-mem-c", str(args.gpu_mem_c)]

    pa = subprocess.Popen([sys.executable, script, "--mode", "producer", *common],
                          stdout=open("/tmp/ring_relay_a.log", "w"),
                          stderr=subprocess.STDOUT)
    print(f"A/producer pid={pa.pid}, log=/tmp/ring_relay_a.log", flush=True)
    _wait_ready(os.path.join(args.store_a, args.run_id, "c0", "_READY"), pa, "A")

    pb = subprocess.Popen([sys.executable, script, "--mode", "relay", *common,
                           "--url-a", f"http://127.0.0.1:{args.port_a}"],
                          stdout=open("/tmp/ring_relay_b.log", "w"),
                          stderr=subprocess.STDOUT)
    print(f"B/relay pid={pb.pid}, log=/tmp/ring_relay_b.log", flush=True)
    _wait_ready(os.path.join(args.store_b, args.run_id, "c1", "_READY"), pb, "B")

    # Relay store sanity: c1 must hold exactly split1 tokens x NUM_LAYERS.
    with open(os.path.join(args.store_b, args.run_id, "c1", "_READY")) as f:
        ready_b = json.load(f)
    assert ready_b["tokens"] == args.split1 and ready_b["layers"] == NUM_LAYERS, (
        f"relay c1 store wrong: {ready_b}")
    print(f"B/relay store OK: {ready_b}", flush=True)

    cons = subprocess.run(
        [sys.executable, script, "--mode", "consumer", *common,
         "--url-a", f"http://127.0.0.1:{args.port_a}",
         "--url-b", f"http://127.0.0.1:{args.port_b}",
         "--decode", str(args.decode)],
        stdout=open("/tmp/ring_relay_c.log", "w"), stderr=subprocess.STDOUT,
    )
    with open(args.done_file, "w") as f:
        f.write("done")
    for p in (pa, pb):
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.terminate()
    print(f"consumer exit={cons.returncode}; log=/tmp/ring_relay_c.log", flush=True)
    sys.exit(cons.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["producer", "relay", "consumer", "all"])
    ap.add_argument("--total", type=int, default=1536)
    ap.add_argument("--split0", type=int, default=512)
    ap.add_argument("--split1", type=int, default=512)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--port-a", type=int, default=8901)
    ap.add_argument("--port-b", type=int, default=8902)
    ap.add_argument("--url-a", default="")
    ap.add_argument("--url-b", default="")
    ap.add_argument("--store-a", default="/tmp/hcp_ring_relay_store_a")
    ap.add_argument("--store-b", default="/tmp/hcp_ring_relay_store_b")
    ap.add_argument("--store-c", default="/tmp/hcp_ring_relay_store_c")
    ap.add_argument("--done-file", default="/tmp/hcp_ring_relay_done")
    ap.add_argument("--hold-secs", type=int, default=900)
    ap.add_argument("--gpu-mem-ab", type=float, default=0.18)
    ap.add_argument("--gpu-mem-c", type=float, default=0.2)
    args = ap.parse_args()

    if args.mode == "producer":
        mode_producer(args)
    elif args.mode == "relay":
        mode_relay(args)
    elif args.mode == "consumer":
        mode_consumer(args)
    else:
        mode_all(args)


if __name__ == "__main__":
    main()
