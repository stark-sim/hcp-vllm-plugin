# hcp-vllm-plugin

HCP(Heterogeneous Context Parallelism)的 vLLM 生态插件。两条路线：

1. **跨引擎 PD（当前主路线）**：HCP 主仓（transformers 线 worker）作为 dedicated
   prefill tier 完成长上下文 prefill 并导出 KV（`hcp-kv-v1` 文件格式）；本仓的
   `HcpFileConnector` 让 vLLM 作为 decode tier 直接载入该 KV、跳过 prefill。
   卖点：硬件成本低（混合存量卡组队）、通信要求低（ring P2P，无需 RDMA/collective 栈）、
   不强制同生态。
2. **vLLM 内 ring CP（保留路线）**：让多台**异构**节点上的 vLLM worker
   组成 ring，以**显存切分**的 context parallelism 合作处理长上下文——每个 worker 只持有自己
   chunk 的 KV，前序 chunk 的 KV 以瞬时 staging 参与计算，不写进本地分页缓存。

与 vLLM 官方 disaggregated prefill(P/D 分离,全量 KV 搬移)不同:ring 路线的每个节点**不需要**
容纳全量 KV,这是异构小显存节点组队跑长上下文的关键差异。

## 组成

| 组件 | 类型 | 说明 |
|------|------|------|
| `HcpFileConnector` | KV connector(V1, consumer only) | **PD 主路线**：从 HCP 导出的文件载入 prefill KV（逐层 safetensors，post-RoPE），注入 paged cache 后直接 decode；prompt 逐 token 匹配才命中，block 对齐尾部由 vLLM 补算。目标 vLLM 0.31 |
| `HcpRingKvConnector` | KV connector(V1) | ring 路线：调度侧把前序 chunk 标 external(全局 RoPE 位置、不重算);worker 侧经 HTTP 把 peer chunk 的逐层 KV 拉进瞬时 staging(不写 block table)。支持每请求参数与连续批 |
| `HcpRingAttentionBackend` | attention backend(注册为 `CUSTOM`) | local(causal) + peer(transient, non-causal) 的 online-softmax 合并 |
| `ring_triton_attn` | Triton kernel | flash 风格 attention + LSE 输出 + causal 偏移;CUDA 与 ROCm(RDNA/CDNA)同一实现 |
| `HcpCpConnector` | KV connector(旧) | 全量 KV context-passing,保留作对照 |

## 安装

```bash
pip install -e ./hcp_vllm_plugin
```

## 兼容性

- **PD 路线（`HcpFileConnector`）**：面向 **vLLM 0.31**（`KVConnectorBase_V1` + 统一
  kv cache layout `[num_blocks, Hkv, block_size, 2*head_dim]`）；已在 white（RTX 4090）+
  vLLM 0.31 + Qwen2.5-7B bf16 的 PD 闭环验证（主仓 `scripts/run_pd_loop_local_white.sh`，
  gate = vLLM 输出与 HCP golden 逐 token 一致）。
- **ring 路线**：已在 **vLLM 0.23.1rc1**(源码构建)+ torch 2.13 上验证:CUDA(RTX 4090)与 ROCm(gfx1200, RDNA4)。
- 依赖的 vLLM 接口面:`KVConnectorBase_V1`(**experimental**,可能随版本变动)、attention
  backend 注册表、`vllm.v1.attention.ops.merge_attn_states`、`vllm.triton_utils`。
- vLLM 升级后请先跑 `python compat_check.py` 冒烟,再跑
  `validate_ring_connector.py --mode all` 全量验证。
- 数值容差:kernel 级 vs fp32 参考 ~1e-3(fp16 舍入);端到端 logits 差 ~0.02–0.04,
  greedy token 与单节点参考逐 token 一致(16k/8k 与跨节点验证实测)。

## 快速开始：PD 路线（HcpFileConnector）

P 侧（主仓 transformers worker）导出 KV：

```bash
python python/hcp_pd_export.py --model-dir <7B> --prompt-file prompt.txt \
    --out-dir /tmp/hcp_pd_kv --max-new-tokens 16 --device cuda --dtype bfloat16
```

D 侧（本仓，vLLM 载入 KV 直接 decode）：

```bash
cd <hcp-vllm-plugin> && python pd_consume.py \
    --model-dir <7B> --kv-dir /tmp/hcp_pd_kv --max-new-tokens 16
# 输出 MATCH=True 即 PD 闭环通过（vLLM decode == HCP golden，逐 token）
```

注意：本 connector 不支持 HMA，LLM 需 `disable_hybrid_kv_cache_manager=True`；
consumer 是 TP=1 契约。

## 快速开始：ring 路线（两进程 loopback）

producer(算 chunk A 并供取 KV):

```python
LLM(model=..., attention_backend="CUSTOM",
    disable_hybrid_kv_cache_manager=True,   # 全部验证均携带此项
    kv_transfer_config={
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "ring_role": "producer",
            "ring_chunk_id": "c0",          # 本实例保存 KV 用的 chunk key
            "ring_shared_path": "/tmp/hcp_ring_prod",
            "ring_run_id": "demo",
            "ring_serve_port": 8901,      # 跨节点时绑 0.0.0.0
        },
    })
# 对 chunk A prompt 调用 generate;KV 写入 store 并经 HTTP 供取
```

consumer(全 prompt,chunk A 标 external):

```python
LLM(model=..., attention_backend="CUSTOM",
    enable_prefix_caching=False,           # CP 路径必须关
    kv_transfer_config={
        "kv_connector": "HcpRingKvConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "ring_role": "consumer",
            "ring_shared_path": "/tmp/hcp_ring_cons",
            "ring_run_id": "demo",
        },
    },
    disable_hybrid_kv_cache_manager=True)
# 每个请求用 SamplingParams 声明自己的 peer chunk:
SamplingParams(
    temperature=0, max_tokens=...,
    extra_args={"kv_transfer_params": {"hcp_ring": {
        "chunk_id": "c0",          # producer 保存的 chunk key
        "prefix_len": 1024,        # peer chunk 的 token 数(block_size 对齐)
        "peer_url": "http://<producer-ip>:8901",
    }}})
```

全局回退(单请求场景可省略显式每请求参数):
`ring_chunk_id` / `ring_prefix_chunk_ids` / `ring_prefix_len` / `ring_peer_url`。

## 环境变量（ring 路线）

| 变量 | 默认 | 说明 |
|------|------|------|
| `HCP_RING_IMPL` | `triton` | attention 实现;`torch` = fp32 debug 兜底 |
| `HCP_RING_MERGE` | `triton` | LSE merge 实现(vLLM `merge_attn_states`);`torch` 兜底 |
| `HCP_RING_SPLIT_TOKENS` | `0` | 无 connector 的单进程验证回退,connector 场景必须保持 `0` |
| `HCP_RING_ENABLED` | `1` | 总开关 |

## 当前限制（ring 路线，v0.1）

- 每个请求的前缀 chunk 数为 1(N>2 真 ring 未支持;配置面将来向后兼容追加 `chunk_ids`)。
- consumer 必须 `enable_prefix_caching=False`。
- KV cache dtype:fp16/bf16(fp8 未支持);模型需禁用 sliding window(走 torch 兜底)。
- 验证均以 `enforce_eager=True` 进行(CUDA graph 未验证)。
- 性能项(kernel-hardening backlog,正确性不受影响):local 段尚未 paged 直读
  (每层有 gather 拷贝)、长上下文 decode 未做 split-KV、kernel 未按 batch 合并启动。

## 验证脚本

- `pd_consume.py` — PD 闭环 consumer（vLLM 载入 HCP 导出的 KV，跳过 prefill 直接 decode；
  gate = 与 HCP golden 逐 token 一致；端到端驱动见主仓 `scripts/run_pd_loop_local_white.sh`）
- `validate_ring_triton_kernel.py` — kernel 数值探针(无需 engine)
- `validate_ring_backend.py` — 单进程 ring backend(env-split 路径)
- `validate_ring_connector.py` — 两进程切分 CP(单请求)
- `validate_ring_concurrent.py` — 两进程切分 CP(多请求连续批)
- `compat_check.py` — 安装/版本/注册冒烟

跨节点 ring CP 驱动见主仓 `scripts/run_cross_node_*.sh`（white CUDA + pearl ROCm，
通过 `*_PLUGIN_REPO` 环境变量指向本仓 clone，默认 `/home/stark/hcp-vllm-plugin`）。
