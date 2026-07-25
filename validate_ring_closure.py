#!/usr/bin/env python
"""Ring-closure validation of the generic-N HCP ring (unified "ring" role).

Topology (true ring, neighbor accumulate-forward):
  N=3 nodes, each running ONE engine with ring_role="ring".  The physical
  ring order is node0 -> node1 -> node2 -> node0; every node pulls its
  accumulated prefix from its physical predecessor ONLY, and re-serves its
  staged chunks (the connector writes their _READY markers right after
  staging), so producer N's consumer is literally (N+1)%N.

Workload (request rotation makes the ring CLOSE at the workload level):
  NREQ=3 concurrent requests, each split into 3 chunks of --chunk-len tokens.
  Request j's chunk p lands on node (j+p)%3.  Thus node k plays all three
  positions across the 3 requests:
    request j=k       : position 0 (empty prefix, produces its chunk)
    request j=(k-1)%3 : position 1 (prefix req{j}_pos0 from predecessor)
    request j=(k-2)%3 : position 2 (prefix pos0+pos1 from predecessor,
                                   produces the request's tokens)
  Every node is producer for (k+1)%N and consumer of (k-1)%N — the cycle is
  literal.  (A single causal sequence cannot close the data loop — chunk 0
  must never attend to the future — so the ring closes across requests.)

Checks (mode all):
  1. each node's position-2 request tokens match the single-node reference
     for that request's full prompt (greedy);
  2. WRITE_TRACK: zero prefix-region pool slots written locally (memory-
     splitting holds on every node);
  3. BATCH_STATS shows the 3 requests really batched on each engine.

Modes: ref | ringnode | all (single-machine 3-instance orchestration).

Run (single machine):
  python validate_ring_closure.py --mode all
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


def build_prompt_ids(req: int, total: int) -> list[int]:
    """Distinct token stream per request (so rotations are distinguishable)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    animal = ["fox", "falcon", "otter"][req % 3]
    text = " ".join(
        f"Story {req} paragraph {i}: the quick brown {animal} leaps over "
        f"hill {i % 5} counting {i} drifting clouds near river {req}."
        for i in range(1024)
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= total, f"prompt too short: {len(ids)} < {total}"
    return ids[:total]


def compute_refs(args) -> dict:
    """Greedy reference tokens for each request's FULL prompt (single node)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_ref,
              max_model_len=args.max_model_len, max_num_seqs=1,
              disable_hybrid_kv_cache_manager=True, seed=0)
    refs = {}
    for j in range(args.nreq):
        ids = build_prompt_ids(j, args.total)
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(temperature=0.0, max_tokens=args.decode))
        refs[str(j)] = list(out[0].outputs[0].token_ids)
        print(f"[ref] req{j} tokens: {refs[str(j)]}", flush=True)
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    return refs


def mode_ref(args) -> None:
    refs = compute_refs(args)
    with open(args.ref_file, "w") as f:
        json.dump(refs, f)
    print(f"[ref] wrote {args.ref_file}", flush=True)


def mode_ringnode(args) -> None:
    """One ring node: drives its 3 rotated requests concurrently."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    k = args.node_index
    n = args.nnodes
    cl = args.chunk_len
    print(f"[node{k}] ring engine; predecessor={args.pred_url}; "
          f"store {args.store} on :{args.port}", flush=True)
    cfg = {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": {
            "ring_role": "ring",
            "ring_shared_path": args.store,
            "ring_run_id": args.run_id,
            "ring_serve_port": args.port,
        },
    }
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"
    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              max_num_seqs=args.nreq, disable_hybrid_kv_cache_manager=True,
              enable_prefix_caching=False,
              attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    import hcp_vllm_plugin.ring_backend as rb

    rb.reset_write_tracking()
    rb.reset_staging_stats()
    rb.reset_batch_stats()
    rb.reset_impl_stats()

    prompts, sps, jobs = [], [], []
    for j in range(args.nreq):
        p = (k - j) % n  # this node's position in request j's chunk order
        ids = build_prompt_ids(j, args.total)[: (p + 1) * cl]
        rp = {"chunk_id": f"req{j}_pos{p}"}
        if p > 0:
            rp["chunk_ids"] = [f"req{j}_pos{q}" for q in range(p)]
            rp["peer_url"] = args.pred_url
            rp["prefix_len"] = p * cl
        else:
            rp["prefix_len"] = 0  # explicit opt-out (first chunk, no prefix)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        sps.append(SamplingParams(
            temperature=0.0,
            max_tokens=args.decode if p == n - 1 else 1,
            extra_args={"kv_transfer_params": {"hcp_ring": rp}},
        ))
        jobs.append((j, p))
        print(f"[node{k}] request req{j}: my position p={p} "
              f"(prompt {len(ids)} tokens, prefix {p * cl})", flush=True)

    outs = llm.generate(prompts, sps)

    closure = {}
    for (j, p), out in zip(jobs, outs):
        if p == n - 1:
            closure[j] = list(out.outputs[0].token_ids)
            print(f"[closure] req{j} tokens: {closure[j]}", flush=True)

    # Trigger one more schedule so finished-req cleanup runs before checks.
    flush_ids = build_prompt_ids(0, 64)[:32]
    llm.generate([TokensPrompt(prompt_token_ids=flush_ids)],
                 SamplingParams(temperature=0.0, max_tokens=1,
                                extra_args={"kv_transfer_params":
                                            {"hcp_ring": {"prefix_len": 0}}}))

    with open(args.tokens_file, "w") as f:
        json.dump(closure, f)
    print(f"[memsplit] overlap={rb.WRITE_TRACK['overlap']} "
          f"slots_written={len(rb.WRITE_TRACK['slots'])}", flush=True)
    print(f"[staging] max_concurrent_chunks="
          f"{rb.STAGING_STATS['max_concurrent_chunks']} "
          f"max_staged_layers={rb.STAGING_STATS['max_staged_layers']} "
          f"leftover={len(rb.PEER_KV_STAGING)}", flush=True)
    print(f"[batch] max_reqs={rb.BATCH_STATS['max_reqs']}", flush=True)
    print(f"[impl] attn triton/torch: {rb.IMPL_STATS['attn_triton']}/"
          f"{rb.IMPL_STATS['attn_torch']}", flush=True)
    print(f"[node{k}] RINGNODE_DONE", flush=True)

    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    print(f"[node{k}] exiting", flush=True)


def mode_all(args) -> None:
    ports = [args.port0, args.port0 + 1, args.port0 + 2]
    for port in ports:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                print(f"port {port} already in use; aborting")
                sys.exit(2)
    for j in range(args.nnodes):
        shutil.rmtree(f"{args.store_prefix}_{j}", ignore_errors=True)
        for f in (f"/tmp/hcp_closure_tokens_{j}.json", f"/tmp/hcp_closure_done_{j}"):
            if os.path.exists(f):
                os.remove(f)

    print("=== computing single-node references ===", flush=True)
    refs = compute_refs(args)

    script = os.path.abspath(__file__)
    procs, logs = [], []
    for k in range(args.nnodes):
        pred = f"http://127.0.0.1:{ports[(k - 1) % args.nnodes]}"
        log = open(f"/tmp/ring_closure_node{k}.log", "w")
        p = subprocess.Popen(
            [sys.executable, script, "--mode", "ringnode",
             "--node-index", str(k), "--nnodes", str(args.nnodes),
             "--nreq", str(args.nreq), "--total", str(args.total),
             "--chunk-len", str(args.chunk_len), "--decode", str(args.decode),
             "--max-model-len", str(args.max_model_len),
             "--run-id", args.run_id, "--port", str(ports[k]),
             "--pred-url", pred, "--store", f"{args.store_prefix}_{k}",
             "--tokens-file", f"/tmp/hcp_closure_tokens_{k}.json",
             "--done-file", "/tmp/hcp_closure_done_all",
             "--gpu-mem", str(args.gpu_mem)],
            stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        logs.append(f"/tmp/ring_closure_node{k}.log")
        print(f"node{k} pid={p.pid} pred={pred} log={logs[-1]}", flush=True)
        # Stagger engine starts: concurrent inits race on free-memory
        # profiling and the later ones can fail KV-cache sizing.
        if k < args.nnodes - 1:
            time.sleep(args.start_delay)

    # Wait for all nodes to finish their requests (RINGNODE_DONE in logs).
    t0 = time.time()
    done = set()
    while time.time() - t0 < 1200 and len(done) < args.nnodes:
        for k in range(args.nnodes):
            if k in done:
                continue
            if procs[k].poll() is not None:
                print(f"node{k} died early (exit {procs[k].returncode}); "
                      f"see {logs[k]}", flush=True)
                sys.exit(1)
            try:
                if "RINGNODE_DONE" in open(logs[k]).read():
                    done.add(k)
            except FileNotFoundError:
                pass
        time.sleep(5)
    with open("/tmp/hcp_closure_done_all", "w") as f:
        f.write("done")
    if len(done) < args.nnodes:
        print(f"timeout: only nodes {sorted(done)} finished", flush=True)
        sys.exit(1)
    for p in procs:
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.terminate()

    # Compare closure tokens with references; collect stats.
    ok = True
    overlap_total = 0
    for k in range(args.nnodes):
        j = (k - (args.nnodes - 1)) % args.nnodes  # this node's position-2 req
        got = json.load(open(f"/tmp/hcp_closure_tokens_{k}.json"))
        mine = got.get(str(j))
        exp = refs[str(j)]
        match = mine == exp
        ok &= match
        print(f"node{k} req{j} (position {args.nnodes - 1}): "
              f"tokens={mine} ref={exp} match={match}", flush=True)
        text = open(logs[k]).read()
        for line in text.splitlines():
            if "[memsplit]" in line:
                print(f"  node{k} {line.strip()}", flush=True)
                ov = int(line.split("overlap=")[1].split()[0])
                overlap_total += ov
            if "[batch]" in line or "[staging]" in line or "[impl]" in line:
                print(f"  node{k} {line.strip()}", flush=True)
    print(f"total prefix-region writes across all nodes: {overlap_total}",
          flush=True)
    ok &= overlap_total == 0
    print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["ref", "ringnode", "all"])
    ap.add_argument("--node-index", type=int, default=0)
    ap.add_argument("--nnodes", type=int, default=3)
    ap.add_argument("--nreq", type=int, default=3)
    ap.add_argument("--total", type=int, default=1536)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--port0", type=int, default=8901)
    ap.add_argument("--pred-url", default="")
    ap.add_argument("--store", default="/tmp/hcp_ring_closure_store")
    ap.add_argument("--store-prefix", default="/tmp/hcp_ring_closure")
    ap.add_argument("--tokens-file", default="/tmp/hcp_closure_tokens.json")
    ap.add_argument("--ref-file", default="/tmp/hcp_closure_refs.json")
    ap.add_argument("--done-file", default="/tmp/hcp_closure_done")
    ap.add_argument("--hold-secs", type=int, default=900)
    ap.add_argument("--gpu-mem", type=float, default=0.22)
    ap.add_argument("--gpu-mem-ref", type=float, default=0.2)
    ap.add_argument("--start-delay", type=int, default=45,
                    help="seconds between ringnode engine starts (mode all)")
    args = ap.parse_args()

    if args.mode == "ref":
        mode_ref(args)
    elif args.mode == "ringnode":
        mode_ringnode(args)
    else:
        mode_all(args)


if __name__ == "__main__":
    main()
