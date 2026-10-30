#!/usr/bin/env python3
"""HCP PD 闭环 consumer：vLLM 载入 HCP 导出的 KV，跳过 prefill 直接 decode。

用法（white）：
    cd <hcp-vllm-plugin> && ~/venv-vllm-pd/bin/python pd_consume.py \
        --model-dir <7B> --kv-dir /tmp/hcp_pd_kv --max-new-tokens 16

（connector 以包内模块 hcp_vllm_plugin.file_connector 注册，无需 PYTHONPATH）
"""

import argparse
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--kv-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="限制 KV cache 预留（7B bf16 在 4090 上余量有限）")
    args = ap.parse_args()

    with open(os.path.join(args.kv_dir, "meta.json")) as f:
        meta = json.load(f)

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    llm = LLM(
        model=args.model_dir,
        dtype=meta["dtype"],
        enforce_eager=True,
        max_model_len=args.max_model_len,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="HcpFileConnector",
            kv_role="kv_consumer",
            kv_connector_module_path="hcp_vllm_plugin.file_connector",
            kv_connector_extra_config={"hcp_kv_dir": args.kv_dir},
        ),
    )
    prompt = TokensPrompt(prompt_token_ids=meta["prompt_token_ids"])
    outs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
    )
    text = outs[0].outputs[0].text
    ids = list(outs[0].outputs[0].token_ids)
    print(f"[pd-consume] generated: {text!r}", flush=True)
    print(f"[pd-consume] token_ids: {ids}", flush=True)
    print(f"[pd-consume] golden   : {meta['golden_text']!r}", flush=True)
    print(f"[pd-consume] golden_ids: {meta['golden_token_ids']}", flush=True)
    match = ids == meta["golden_token_ids"]
    print(f"[pd-consume] MATCH={match}", flush=True)
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
