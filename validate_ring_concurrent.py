#!/usr/bin/env python
"""Multi-request (continuous-batching) validation of the HCP ring-KV connector.

Same topology as validate_ring_connector.py (single machine, two OS processes,
loopback HTTP), but with TWO concurrent requests, each carrying its own peer
chunk via SamplingParams(extra_args={"kv_transfer_params": {"hcp_ring": ...}}):

  * producer: prefills chunk A of TWO different prompts as two requests,
    saving each under its own chunk key (c0 / c1), serves the store over HTTP.
  * consumer: gets BOTH full prompts in ONE generate call (max_num_seqs=2).
    The scheduler marks each request's prefix external; the worker stages each
    chunk under its own key and binds request -> chunk via first block id.
    HcpRingAttentionImpl merges per-request local + peer KV in the same batch.

Checks (consumer mode):
  1. both requests' greedy tokens match a single-node reference run;
  2. STAGING_STATS: 2 chunks staged concurrently (2 x 24 layers) — true
     multi-request CP, and BATCH_STATS.max_reqs >= 2 — the CP path really
     ran inside a continuous batch;
  3. memory-splitting: WRITE_TRACK overlap == 0 (no chunk-A pool slot written
     locally for either request);
  4. lifecycle: after generate returns, PEER_KV_STAGING / PEER_REQ_MAP are
     empty (connector freed staged KV on request finish).

Modes: producer | consumer | all (orchestrates both as subprocesses).

Run (with vllm-rocm env + ROCm LD_LIBRARY_PATH):
  python validate_ring_concurrent.py --mode all --total 1024 --split 512
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
N_REQ = 2
CHUNK_KEYS = [f"c{i}" for i in range(N_REQ)]


def build_prompt_ids(total: int, variant: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if variant == 0:
        text = " ".join(
            f"Paragraph {i}: the quick brown fox jumps over the lazy dog, "
            f"then counts {i} clouds drifting over mountain {i % 7}."
            for i in range(512)
        )
    else:
        text = " ".join(
            f"Chapter {i}: a curious hedgehog sails across the quiet river, "
            f"then collects {i} pebbles near harbor {i % 5}."
            for i in range(512)
        )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= total, f"prompt too short: {len(ids)} < {total}"
    return ids[:total]


def ring_params(chunk_key: str, split: int, peer_url: str = "") -> dict:
    p = {"chunk_id": chunk_key, "prefix_len": split}
    if peer_url:
        p["peer_url"] = peer_url
    return {"kv_transfer_params": {"hcp_ring": p}}


def sampling(extra: dict | None, decode: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=decode, extra_args=extra)


def free_llm(llm) -> None:
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    import gc

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mode_producer(args) -> None:
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    prompts = [build_prompt_ids(args.total, v)[: args.split] for v in range(N_REQ)]
    print(f"[producer] {N_REQ} chunks of {args.split} tokens each; serving "
          f"{args.producer_store} on :{args.port}", flush=True)

    cfg = {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_producer",
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": {
            "ring_role": "producer",
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
        max_num_seqs=N_REQ,
        disable_hybrid_kv_cache_manager=True,
        attention_backend="CUSTOM",
        kv_transfer_config=cfg,
        seed=0,
    )
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts],
        [sampling(ring_params(CHUNK_KEYS[i], args.split), 1) for i in range(N_REQ)],
    )
    print(f"[producer] chunks prefilled, sample tokens "
          f"{[o.outputs[0].token_ids[0] for o in outs]}; KV served on :{args.port}",
          flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    print("[producer] exiting", flush=True)


def mode_consumer(args) -> None:
    import torch
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    full = [build_prompt_ids(args.total, v) for v in range(N_REQ)]
    print(f"[consumer] {N_REQ} full prompts of {args.total} tokens; "
          f"peer chunks at split={args.split}; peer={args.peer_url}", flush=True)

    # ---- single-node reference (default backend, batched) ----
    ref = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        max_num_seqs=N_REQ,
        disable_hybrid_kv_cache_manager=True,
        seed=0,
    )
    ref_outs = ref.generate(
        [TokensPrompt(prompt_token_ids=p) for p in full],
        sampling(None, args.decode),
    )
    ref_tokens = [list(o.outputs[0].token_ids) for o in ref_outs]
    print(f"[ref] tokens: {ref_tokens}", flush=True)
    free_llm(ref)

    # ---- consumer instance (CUSTOM backend + ring connector) ----
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"  # staged dict path only
    cfg = {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": {
            "ring_role": "consumer",
            "ring_shared_path": args.consumer_store,
            "ring_run_id": args.run_id,
        },
    }
    cons = LLM(
        model=MODEL,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        max_num_seqs=N_REQ,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        attention_backend="CUSTOM",
        kv_transfer_config=cfg,
        seed=0,
    )
    import hcp_vllm_plugin.ring_backend as rb

    rb.reset_write_tracking()
    rb.reset_staging_stats()
    rb.reset_batch_stats()
    outs = cons.generate(
        [TokensPrompt(prompt_token_ids=p) for p in full],
        [sampling(ring_params(CHUNK_KEYS[i], args.split, args.peer_url),
                  args.decode) for i in range(N_REQ)],
    )
    cons_tokens = [list(o.outputs[0].token_ids) for o in outs]
    print(f"[consumer] tokens: {cons_tokens}", flush=True)

    # Cleanup is one step late BY DESIGN: finished_req_ids are shipped in the
    # NEXT SchedulerOutput (scheduler.py ships the accumulated set and clears
    # it).  In continuous serving the next step arrives with the next request;
    # here we submit one tiny non-CP request to trigger that schedule.
    cons.generate([TokensPrompt(prompt_token_ids=full[0][:32])],
                  sampling({"kv_transfer_params":
                            {"hcp_ring": {"prefix_len": 0}}}, 1))

    # ---- correctness ----
    token_match = cons_tokens == ref_tokens

    # ---- memory-splitting + concurrency evidence ----
    max_chunks = rb.STAGING_STATS["max_concurrent_chunks"]
    max_layers = rb.STAGING_STATS["max_staged_layers"]
    staged_len = rb.STAGING_STATS["last_chunk_len"]
    max_reqs = rb.BATCH_STATS["max_reqs"]
    overlap = rb.WRITE_TRACK["overlap"]
    leftover = len(rb.PEER_KV_STAGING)
    leftover_map = len(rb.PEER_REQ_MAP)
    mem_ok = (
        max_chunks == N_REQ
        and max_layers == N_REQ * NUM_LAYERS
        and staged_len == args.split
        and overlap == 0
        and leftover == 0
        and leftover_map == 0
        and max_reqs >= N_REQ
    )
    print(f"[memsplit] concurrent chunks staged: {max_chunks}/{N_REQ}, "
          f"{max_layers} layer entries (2x{NUM_LAYERS}), {staged_len} tokens/layer")
    print(f"[memsplit] pool overlap (chunk-A slots written locally): {overlap}")
    print(f"[batch] max requests in one attention step: {max_reqs}")
    print(f"[lifecycle] staging freed after finish: "
          f"{leftover == 0 and leftover_map == 0}")

    ok = token_match and mem_ok
    print("--- result ---")
    print(f"  tokens match        : {token_match} (ref={ref_tokens} "
          f"cons={cons_tokens})")
    print(f"  multi-req CP + batch: {'OK' if mem_ok else 'VIOLATED'}")
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
              "--run-id", args.run_id, "--port", str(args.port),
              "--producer-store", args.producer_store,
              "--consumer-store", args.consumer_store,
              "--done-file", args.done_file]

    prod_log = open("/tmp/ring_conc_producer.log", "w")
    prod = subprocess.Popen(
        [sys.executable, script, "--mode", "producer", *common],
        stdout=prod_log, stderr=subprocess.STDOUT,
    )
    print(f"producer pid={prod.pid}, log=/tmp/ring_conc_producer.log")

    ready = [os.path.join(args.producer_store, args.run_id, ck, "_READY")
             for ck in CHUNK_KEYS]
    t0 = time.time()
    while time.time() - t0 < 600:
        if all(os.path.exists(r) for r in ready):
            break
        if prod.poll() is not None:
            print(f"producer died early (exit {prod.returncode}); "
                  f"see /tmp/ring_conc_producer.log")
            sys.exit(1)
        time.sleep(2)
    else:
        prod.terminate()
        print("producer never became ready")
        sys.exit(1)
    print(f"producer ready ({time.time() - t0:.0f}s); launching consumer")

    cons_log = open("/tmp/ring_conc_consumer.log", "w")
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
    print(f"consumer exit={cons.returncode}; log=/tmp/ring_conc_consumer.log")
    sys.exit(cons.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["producer", "consumer", "all"])
    ap.add_argument("--total", type=int, default=1024)
    ap.add_argument("--split", type=int, default=512)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--peer-url", default="")
    ap.add_argument("--producer-store", default="/tmp/hcp_ring_conc_producer")
    ap.add_argument("--consumer-store", default="/tmp/hcp_ring_conc_consumer")
    ap.add_argument("--done-file", default="/tmp/hcp_ring_conc_done")
    ap.add_argument("--hold-secs", type=int, default=900)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
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
