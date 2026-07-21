#!/usr/bin/env python
"""2-instance validation of the HCP ring-KV connector + ring attention backend.

Topology (single machine, two OS processes, loopback HTTP transport):
  * producer instance: vLLM (CUSTOM attention backend + HcpRingKvConnector,
    role=producer) prefills chunk A only, saves its per-layer KV to a store,
    and serves the store over HTTP.  Its paged pool holds ONLY chunk-A KV.
  * consumer instance: vLLM (CUSTOM backend + HcpRingKvConnector,
    role=consumer) gets the FULL prompt.  The connector's scheduler side marks
    chunk A as externally computed (global RoPE positions, no recompute); the
    worker side fetches chunk-A KV over HTTP into the ring backend's
    TRANSIENT PEER_KV_STAGING dict — never into the consumer's paged pool.
    HcpRingAttentionImpl merges local(chunk B, causal) + peer(chunk A,
    transient, non-causal) with online softmax.

Checks (consumer mode):
  1. consumer's next tokens (greedy, --decode tokens) match a single-node
     reference run in the same process; last-step logits are close.
  2. memory-splitting: ring_backend.WRITE_TRACK proves no paged-pool slot of
     the chunk-A region was ever written by the consumer, and the staged peer
     KV really came over HTTP from the producer.

Modes: producer | consumer | all (orchestrates both as subprocesses).

Run (with vllm-rocm env + ROCm LD_LIBRARY_PATH):
  python validate_ring_connector.py --mode all --total 2048 --split 1024
"""

import argparse
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
        for i in range(512)
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= total, f"prompt too short: {len(ids)} < {total}"
    return ids[:total]


def find_model(llm):
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


def mode_producer(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = build_prompt_ids(args.total)[: args.split]
    print(f"[producer] chunk A = {len(ids)} tokens; serving store "
          f"{args.producer_store} on :{args.port}", flush=True)

    cfg = {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": {
            "ring_role": "producer",
            "ring_chunk_id": args.chunk_id,
            "ring_shared_path": args.producer_store,
            "ring_run_id": args.run_id,
            "ring_serve_port": args.port,
        },
    }
    llm = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.producer_gpu_mem,
        max_model_len=4096,
        max_num_seqs=1,
        disable_hybrid_kv_cache_manager=True,
        attention_backend="CUSTOM",
        kv_transfer_config=cfg,
        seed=0,
    )
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=ids)],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    print(f"[producer] chunk A prefilled, sample token "
          f"{out[0].outputs[0].token_ids[0]}; KV served on :{args.port}",
          flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    print("[producer] exiting", flush=True)


def mode_consumer(args) -> None:
    import torch
    from vllm import LLM

    ids = build_prompt_ids(args.total)
    print(f"[consumer] full prompt = {len(ids)} tokens; chunk A (peer) = "
          f"{args.split}; peer={args.peer_url}", flush=True)

    # ---- single-node reference (default backend) ----
    ref = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        max_num_seqs=1,
        disable_hybrid_kv_cache_manager=True,
        seed=0,
    )
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
    os.environ["HCP_RING_SPLIT_TOKENS"] = str(args.split)  # fallback only
    cfg = {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": {
            "ring_role": "consumer",
            "ring_prefix_chunk_ids": args.chunk_id,
            "ring_prefix_len": args.split,
            "ring_shared_path": args.consumer_store,
            "ring_run_id": args.run_id,
            "ring_peer_url": args.peer_url,
        },
    }
    cons = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        max_num_seqs=1,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        attention_backend="CUSTOM",
        kv_transfer_config=cfg,
        seed=0,
    )
    import hcp_vllm_plugin.ring_backend as rb

    rb.reset_write_tracking()
    rb.reset_staging_stats()
    cons_logits, cons_tokens = run_capture(cons, ids, args.decode)
    print(f"[consumer] tokens: {cons_tokens}", flush=True)

    # Cleanup is one step late BY DESIGN: finished_req_ids are shipped in the
    # NEXT SchedulerOutput.  Submit one tiny non-CP request to trigger that
    # schedule so the staged chunk is freed before we check below.
    from vllm.inputs import TokensPrompt as _TP
    from vllm import SamplingParams as _SP

    cons.generate([_TP(prompt_token_ids=ids[:32])],
                  _SP(temperature=0.0, max_tokens=1,
                      extra_args={"kv_transfer_params":
                                  {"hcp_ring": {"prefix_len": 0}}}))

    # ---- correctness ----
    token_match = cons_tokens == ref_tokens
    max_diff = (ref_logits - cons_logits).abs().max().item()
    argmax_diff = (ref_logits - cons_logits).abs()[ref_logits.argmax()].item()

    # ---- memory-splitting evidence ----
    # Staging is freed when the request finishes, so use the backend's
    # high-water marks; live staging must be empty now (cleanup proof).
    n_staged = rb.STAGING_STATS["max_staged_layers"]
    staged_len = rb.STAGING_STATS["last_chunk_len"]
    leftover = len(rb.PEER_KV_STAGING)
    leftover_map = len(rb.PEER_REQ_MAP)
    overlap = rb.WRITE_TRACK["overlap"]
    n_written = len(rb.WRITE_TRACK["slots"])
    mem_ok = (
        n_staged == NUM_LAYERS
        and staged_len == args.split
        and overlap == 0
        and leftover == 0
        and leftover_map == 0
    )
    print(f"[memsplit] peer KV staged for {n_staged}/{NUM_LAYERS} layers, "
          f"{staged_len} tokens/layer (fetched over HTTP from producer)")
    print(f"[memsplit] consumer wrote {n_written} pool slots (its own chunk "
          f"only); chunk-A pool slots written locally: {overlap}")
    print(f"[memsplit] post-run transient staging freed: {leftover == 0 and leftover_map == 0}")

    ok = (
        token_match
        and max_diff < 0.1
        and mem_ok
    )
    print("--- result ---")
    print(f"  tokens match        : {token_match} (ref={ref_tokens} "
          f"cons={cons_tokens})")
    print(f"  max|logit diff|     : {max_diff:.6f} (at ref argmax: "
          f"{argmax_diff:.6f})")
    print(f"  memory-splitting    : {'OK' if mem_ok else 'VIOLATED'}")
    print(f"  verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def mode_all(args) -> None:
    # Port must be free.
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"port {args.port} already in use; aborting")
            sys.exit(2)
    shutil.rmtree(args.producer_store, ignore_errors=True)
    shutil.rmtree(args.consumer_store, ignore_errors=True)
    if os.path.exists(args.done_file):
        os.remove(args.done_file)

    script = os.path.abspath(__file__)
    common = ["--total", str(args.total), "--split", str(args.split),
              "--run-id", args.run_id, "--chunk-id", args.chunk_id,
              "--port", str(args.port),
              "--producer-store", args.producer_store,
              "--consumer-store", args.consumer_store,
              "--done-file", args.done_file]

    prod_log = open("/tmp/ring_conn_producer.log", "w")
    prod = subprocess.Popen(
        [sys.executable, script, "--mode", "producer", *common],
        stdout=prod_log, stderr=subprocess.STDOUT,
    )
    print(f"producer pid={prod.pid}, log=/tmp/ring_conn_producer.log")

    ready = os.path.join(args.producer_store, args.run_id, args.chunk_id, "_READY")
    t0 = time.time()
    while time.time() - t0 < 600:
        if os.path.exists(ready):
            break
        if prod.poll() is not None:
            print(f"producer died early (exit {prod.returncode}); "
                  f"see /tmp/ring_conn_producer.log")
            sys.exit(1)
        time.sleep(2)
    else:
        prod.terminate()
        print("producer never became ready")
        sys.exit(1)
    print(f"producer ready ({time.time() - t0:.0f}s); launching consumer")

    cons_log = open("/tmp/ring_conn_consumer.log", "w")
    cons = subprocess.run(
        [sys.executable, script, "--mode", "consumer", *common,
         "--peer-url", f"http://127.0.0.1:{args.port}",
         "--decode", str(args.decode)],
        stdout=cons_log, stderr=subprocess.STDOUT,
    )
    with open(args.done_file, "w") as f:
        f.write("done")
    try:
        prod.wait(timeout=90)
    except subprocess.TimeoutExpired:
        prod.terminate()
    print(f"consumer exit={cons.returncode}; log=/tmp/ring_conn_consumer.log")
    sys.exit(cons.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["producer", "consumer", "all"])
    ap.add_argument("--total", type=int, default=2048)
    ap.add_argument("--split", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--chunk-id", default="chunk0")
    ap.add_argument("--peer-url", default="")
    ap.add_argument("--producer-store", default="/tmp/hcp_ring_store_producer")
    ap.add_argument("--consumer-store", default="/tmp/hcp_ring_store_consumer")
    ap.add_argument("--done-file", default="/tmp/hcp_ring_conn_done")
    ap.add_argument("--hold-secs", type=int, default=900)
    ap.add_argument("--gpu-mem", type=float, default=0.3,
                    help="gpu_memory_utilization for ref and consumer engines")
    ap.add_argument("--producer-gpu-mem", type=float, default=0.25)
    args = ap.parse_args()

    if args.mode == "producer":
        mode_producer(args)
    elif args.mode == "consumer":
        mode_consumer(args)
    else:
        mode_all(args)


if __name__ == "__main__":
    main()
