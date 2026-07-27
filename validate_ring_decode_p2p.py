#!/usr/bin/env python
"""P2P ring decode validation — 3 instances, one GPU.

Topology (line prefill + P2P ring decode):
  * A = peer0 (chunk c0): prefills ids[:s0], serves KV store on :portA,
    runs RingDecodeNode relay on TCP :decode_portA (ring index 1).
  * B = peer1 (chunk c1): stages c0 from A, prefills c1, serves store on
    :portB, runs RingDecodeNode relay on TCP :decode_portB (ring index 2).
  * C = owner (chunk c2): full prompt; marks c0+c1 external, stages both
    for PREFILL, samples first token — then HCP_RING_DECODE_RING=1 +
    HCP_RING_DECODE_TRANSPORT=ring switches to P2P ring decode: Q +
    accumulator travel C -> A -> B -> C per layer per step.

Ring transport order: C(0) -> A(1) -> B(2) -> C(0)

Checks:
  1. C's greedy tokens match a single-node reference.
  2. MEMORY-SPLIT: staging empty right after generate.
  3. RingDecodeNode call counts on A/B (relay served partial merges).
  4. Growth rotation correct (p mod N).
  5. No HTTP partial_attn involved in decode (ring transport only).
  6. Pool-skip evidence (decode doesn't write paged pool).

Modes: producer | relay | consumer | all (orchestrates all three).

Run:
  HCP_RING_DECODE_RING=1 HCP_RING_DECODE_TRANSPORT=ring \
    python validate_ring_decode_p2p.py --mode all
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

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


def _base_cfg(ring_cfg: dict, kv_role: str) -> dict:
    return {
        "kv_connector": "HcpRingKvConnector",
        "kv_role": kv_role,
        "kv_connector_module_path": "hcp_vllm_plugin.ring_connector",
        "kv_connector_extra_config": ring_cfg,
    }


def _ring_decode_order(args) -> str:
    """Transport ring order: C(owner,0) -> A(1) -> B(2) -> C.
    Single-node default uses loopback; --ring-order overrides with real
    "host:port,..." entries for cross-node runs."""
    if getattr(args, "ring_order", ""):
        return args.ring_order
    return (
        f"127.0.0.1:{args.decode_port_c},"
        f"127.0.0.1:{args.decode_port_a},"
        f"127.0.0.1:{args.decode_port_b}"
    )


def mode_producer(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = build_prompt_ids(args.total)[: args.split0]
    print(f"[A/peer0] c0 = {len(ids)} tokens; store {args.store_a} "
          f"HTTP :{args.port_a}, ring TCP :{args.decode_port_a} (idx 1)",
          flush=True)
    cfg = _base_cfg({
        "ring_role": "producer",
        "ring_chunk_id": "c0",
        "ring_shared_path": args.store_a,
        "ring_run_id": args.run_id,
        "ring_serve_port": args.port_a,
        "ring_partial_chunks": "c0",
        "ring_decode_order": _ring_decode_order(args),
        "ring_decode_port": args.decode_port_a,
        "ring_decode_index": "1",
    }, "kv_producer")
    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_ab, max_model_len=args.max_model_len,
              max_num_seqs=1, disable_hybrid_kv_cache_manager=True,
              attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=1))
    print(f"[A/peer0] c0 prefilled, sample {out[0].outputs[0].token_ids[0]}; "
          f"ring relay active on :{args.decode_port_a}", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    _print_ring_node_stats()
    print("[A/peer0] exiting", flush=True)


def _print_ring_node_stats() -> None:
    try:
        from hcp_vllm_plugin.ring_decode import RING_NODES

        for node in RING_NODES:
            print(
                f"RingDecodeNode[{node._ring_index}] final: calls={node.calls} "
                f"growth_appends={node.growth_appends} "
                f"growth_tokens_per_layer={node.growth_tokens()}",
                flush=True,
            )
    except Exception as e:
        print(f"RingDecodeNode stats unavailable: {e}", flush=True)


def mode_relay(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ids = build_prompt_ids(args.total)[: args.split0 + args.split1]
    print(f"[B/peer1] prompt = {len(ids)} (prefix c0={args.split0} external, "
          f"own c1={args.split1}); peer A={args.url_a}; store {args.store_b} "
          f"HTTP :{args.port_b}, ring TCP :{args.decode_port_b} (idx 2)",
          flush=True)
    cfg = _base_cfg({
        "ring_role": "relay",
        "ring_chunk_id": "c1",
        "ring_prefix_chunk_ids": "c0",
        "ring_prefix_len": args.split0,
        "ring_peer_url": args.url_a,
        "ring_shared_path": args.store_b,
        "ring_run_id": args.run_id,
        "ring_serve_port": args.port_b,
        "ring_partial_chunks": "c1",
        "ring_decode_order": _ring_decode_order(args),
        "ring_decode_port": args.decode_port_b,
        "ring_decode_index": "2",
    }, "kv_both")
    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_ab, max_model_len=args.max_model_len,
              max_num_seqs=1, disable_hybrid_kv_cache_manager=True,
              enable_prefix_caching=False,
              attention_backend="CUSTOM", kv_transfer_config=cfg, seed=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=1))
    print(f"[B/peer1] c1 prefilled, sample {out[0].outputs[0].token_ids[0]}; "
          f"ring relay active on :{args.decode_port_b}", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.hold_secs and not os.path.exists(args.done_file):
        time.sleep(2)
    _print_ring_node_stats()
    print("[B/peer1] exiting", flush=True)


def mode_consumer(args) -> None:
    import torch
    from vllm import LLM

    s2 = args.total - args.split0 - args.split1
    ids = build_prompt_ids(args.total)
    print(f"[C/owner] full prompt = {len(ids)} (prefix c0+c1 = "
          f"{args.split0 + args.split1} external, own c2 = {s2}); "
          f"peers A={args.url_a} B={args.url_b}; "
          f"ring TCP :{args.decode_port_c} (idx 0)", flush=True)

    # ---- single-node reference ----
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

    # ---- owner instance (CUSTOM backend + ring connector + ring transport) ----
    os.environ["HCP_RING_SPLIT_TOKENS"] = "0"
    cfg = _base_cfg({
        "ring_role": "consumer",
        "ring_prefix_chunk_ids": "c0,c1",
        "ring_prefix_len": args.split0 + args.split1,
        "ring_shared_path": args.store_c,
        "ring_run_id": args.run_id,
        "ring_decode_order": _ring_decode_order(args),
        "ring_decode_port": args.decode_port_c,
        "ring_decode_index": "0",
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

    # ---- MEMORY-SPLIT evidence ----
    staging_left = len(rb.PEER_KV_STAGING)
    map_left = len(rb.PEER_REQ_MAP)
    ring_map_left = len(rb.DECODE_RING_MAP)
    print(f"[decode-p2p] staging right after generate: "
          f"PEER_KV_STAGING={staging_left} PEER_REQ_MAP={map_left} "
          f"DECODE_RING_MAP={ring_map_left}", flush=True)

    # Flush to trigger cleanup
    cons.generate([TokensPrompt(prompt_token_ids=ids[:32])],
                  SamplingParams(temperature=0.0, max_tokens=1,
                                 extra_args={"kv_transfer_params":
                                             {"hcp_ring": {"prefix_len": 0}}}))
    ring_map_after = len(rb.DECODE_RING_MAP)

    # ---- correctness ----
    token_match = cons_tokens == ref_tokens
    max_diff = (ref_logits - cons_logits).abs().max().item()
    argmax_diff = (ref_logits - cons_logits).abs()[ref_logits.argmax()].item()

    n_written = len(rb.WRITE_TRACK["slots"])
    expected_slots = s2 + 40
    slots_ok = n_written <= expected_slots
    decode_split_ok = staging_left == 0 and map_left == 0
    ring_clean_ok = ring_map_after == 0
    impl_ok = (
        rb.IMPL_STATS["attn_triton"] > 0 and rb.IMPL_STATS["attn_torch"] == 0
    )
    expected_skips = (args.decode - 1) * NUM_LAYERS
    skip_count = rb.WRITE_TRACK.get("skipped", 0)
    skip_ok = skip_count == expected_skips
    # Streaming prefill: transient GPU staging bounded at all times —
    # (window+1) layers × num prefix chunks, never the full prefix.
    max_staged = rb.STAGING_STATS["max_staged_layers"]
    stream_bound = 6
    window_ok = max_staged <= stream_bound
    print(f"[decode-p2p] peak staged layers: {max_staged} "
          f"(streaming bound <= {stream_bound}): "
          f"{'OK' if window_ok else 'VIOLATED'}", flush=True)

    # Ring transport evidence
    from hcp_vllm_plugin.ring_transport import get_ring_transport
    transport = get_ring_transport()
    transport_ok = transport is not None and transport.packets_sent > 0
    print(f"[decode-p2p] ring transport: sent={transport.packets_sent if transport else 0} "
          f"recv={transport.packets_recv if transport else 0}", flush=True)

    print(f"[impl] attn triton/torch: {rb.IMPL_STATS['attn_triton']}/"
          f"{rb.IMPL_STATS['attn_torch']}, merge triton/torch: "
          f"{rb.IMPL_STATS['merge_triton']}/{rb.IMPL_STATS['merge_torch']}")
    print(f"[decode-p2p] consumer wrote {n_written} pool slots "
          f"(expected <= {expected_slots}): {'OK' if slots_ok else 'VIOLATED'}")
    print(f"[decode-p2p] decode pool writes skipped: {skip_count} "
          f"(expected {expected_skips}): {'OK' if skip_ok else 'VIOLATED'}")
    print(f"[decode-p2p] decode-ring map cleared: {ring_map_after == 0}")

    ok = (token_match and max_diff < 0.1 and decode_split_ok
          and ring_clean_ok and slots_ok and skip_ok and impl_ok
          and transport_ok and window_ok)
    print("--- result ---")
    print(f"  tokens match        : {token_match} (ref={ref_tokens} "
          f"cons={cons_tokens})")
    print(f"  max|logit diff|     : {max_diff:.6f} (at ref argmax: "
          f"{argmax_diff:.6f})")
    print(f"  memsplit decode     : {'OK' if decode_split_ok else 'VIOLATED'}")
    print(f"  slots accounting    : {'OK' if slots_ok else 'VIOLATED'}")
    print(f"  pool-skip evidence  : {'OK' if skip_ok else 'VIOLATED'}")
    print(f"  streaming window    : {'OK' if window_ok else 'VIOLATED'}")
    print(f"  triton kernel path  : {'OK' if impl_ok else 'NOT USED'}")
    print(f"  ring transport used : {'OK' if transport_ok else 'NOT USED'}")
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
    if os.environ.get("HCP_RING_DECODE_RING", "0") != "1":
        print("HCP_RING_DECODE_RING=1 is required; set it and rerun", flush=True)
        sys.exit(2)
    if os.environ.get("HCP_RING_DECODE_TRANSPORT", "ring") != "ring":
        print("HCP_RING_DECODE_TRANSPORT=ring is required for this test",
              flush=True)
        sys.exit(2)
    # Check ports
    for port in (args.port_a, args.port_b,
                 args.decode_port_a, args.decode_port_b, args.decode_port_c):
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
              "--decode-port-a", str(args.decode_port_a),
              "--decode-port-b", str(args.decode_port_b),
              "--decode-port-c", str(args.decode_port_c),
              "--gpu-mem-ab", str(args.gpu_mem_ab),
              "--gpu-mem-c", str(args.gpu_mem_c)]

    # Start A (peer0, ring index 1)
    pa = subprocess.Popen([sys.executable, script, "--mode", "producer", *common],
                          stdout=open("/tmp/ring_p2p_a.log", "w"),
                          stderr=subprocess.STDOUT)
    print(f"A/peer0 pid={pa.pid}, log=/tmp/ring_p2p_a.log", flush=True)
    _wait_ready(os.path.join(args.store_a, args.run_id, "c0", "_READY"), pa, "A")

    # Start B (peer1, ring index 2)
    pb = subprocess.Popen([sys.executable, script, "--mode", "relay", *common,
                           "--url-a", f"http://127.0.0.1:{args.port_a}"],
                          stdout=open("/tmp/ring_p2p_b.log", "w"),
                          stderr=subprocess.STDOUT)
    print(f"B/peer1 pid={pb.pid}, log=/tmp/ring_p2p_b.log", flush=True)
    _wait_ready(os.path.join(args.store_b, args.run_id, "c1", "_READY"), pb, "B")

    with open(os.path.join(args.store_b, args.run_id, "c1", "_READY")) as f:
        ready_b = json.load(f)
    assert ready_b["tokens"] == args.split1 and ready_b["layers"] == NUM_LAYERS, (
        f"relay c1 store wrong: {ready_b}")
    print(f"B/peer1 store OK: {ready_b}", flush=True)

    # Run C (owner, ring index 0) — foreground
    cons = subprocess.run(
        [sys.executable, script, "--mode", "consumer", *common,
         "--url-a", f"http://127.0.0.1:{args.port_a}",
         "--url-b", f"http://127.0.0.1:{args.port_b}",
         "--decode", str(args.decode)],
        stdout=open("/tmp/ring_p2p_c.log", "w"), stderr=subprocess.STDOUT,
    )
    with open(args.done_file, "w") as f:
        f.write("done")
    for p in (pa, pb):
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.terminate()
    # Ring node stats from A/B logs
    for name, log in (("A", "/tmp/ring_p2p_a.log"), ("B", "/tmp/ring_p2p_b.log")):
        try:
            text = open(log).read()
            for line in text.splitlines():
                if "RingDecodeNode" in line:
                    print(f"  [{name}] {line.strip()}", flush=True)
        except FileNotFoundError:
            pass
    print(f"consumer exit={cons.returncode}; log=/tmp/ring_p2p_c.log", flush=True)
    sys.exit(cons.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["producer", "relay", "consumer", "all"])
    ap.add_argument("--total", type=int, default=1536)
    ap.add_argument("--split0", type=int, default=512)
    ap.add_argument("--split1", type=int, default=512)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--port-a", type=int, default=8901)
    ap.add_argument("--port-b", type=int, default=8902)
    ap.add_argument("--decode-port-a", type=int, default=8951)
    ap.add_argument("--decode-port-b", type=int, default=8952)
    ap.add_argument("--decode-port-c", type=int, default=8950)
    ap.add_argument("--ring-order", default="",
                    help='explicit "host:port,..." decode ring order '
                         "(index 0 = owner); overrides the loopback default "
                         "for cross-node runs")
    ap.add_argument("--url-a", default="")
    ap.add_argument("--url-b", default="")
    ap.add_argument("--store-a", default="/tmp/hcp_ring_p2p_store_a")
    ap.add_argument("--store-b", default="/tmp/hcp_ring_p2p_store_b")
    ap.add_argument("--store-c", default="/tmp/hcp_ring_p2p_store_c")
    ap.add_argument("--done-file", default="/tmp/hcp_ring_p2p_done")
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
