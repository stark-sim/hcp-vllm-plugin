"""HCP 文件式 KV consumer connector（vLLM v1, KVConnectorBase_V1）。

契约（hcp-kv-v1，与主仓 hetero-cp-ringattn 的 python/hcp_pd_export.py 配对）：
    <kv_dir>/meta.json:  format/prompt_token_ids/seq_len/num_layers/
                         num_kv_heads/head_dim/dtype
    <kv_dir>/layer_<i>.safetensors: {"kv": [S, 2, Hkv, D]}（[:,0]=K, [:,1]=V,
                         post-RoPE；K/V 已是模型 dtype）

行为：
- scheduler 侧：请求的 prompt_token_ids 与 meta.json 完全一致才算命中；
  命中数对齐到 block_size 向下取整（v1 scheduler 要求 block 对齐），
  尾部（含最后一个 prompt token）由 vLLM 自己补算，保证有 logits 可采样。
- worker 侧：start_load_kv 同步把文件 KV 注入 paged cache
  （vllm 0.31 统一 layout `[num_blocks, Hkv, block_size, 2*head_dim]`，
  最后一维前 head_dim 为 K、后半为 V——证据 flash_attn.py
  `kv_cache.transpose(1, 2).split(head_size, dim=-1)`）。
- 握手检查：meta 的 num_layers/num_kv_heads/head_dim/dtype 与 vLLM 模型
  config 不一致则启动即硬报错。

注册：KVTransferConfig(kv_connector="HcpFileConnector",
    kv_connector_module_path="hcp_vllm_plugin.file_connector", kv_role="kv_consumer",
    kv_connector_extra_config={"hcp_kv_dir": <dir>})
注意：本 connector 不支持 HMA，LLM 需 disable_hybrid_kv_cache_manager=True。

参考：vllm v0.31.0 v1/example_connector.py（disk-based debug connector）。
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _align_down(n: int, block_size: int) -> int:
    return n // block_size * block_size


@dataclass
class ReqMeta:
    slot_mapping: torch.Tensor  # [n_tokens_to_load]
    n_tokens: int


@dataclass
class HcpConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)


class HcpFileConnector(KVConnectorBase_V1):
    """从 HCP 导出的文件加载 prefill KV（consumer only，TP=1 契约）。"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size
        kv_dir = self._kv_transfer_config.get_from_extra_config("hcp_kv_dir", None)
        assert kv_dir, "hcp_kv_dir missing in kv_connector_extra_config"
        self._kv_dir = kv_dir
        with open(os.path.join(kv_dir, "meta.json")) as f:
            self._meta = json.load(f)
        assert self._meta["format"] == "hcp-kv-v1"

        # 握手检查：模型结构/dtype 必须与导出方一致
        hf = vllm_config.model_config.hf_config
        head_dim = hf.hidden_size // hf.num_attention_heads
        checks = {
            "num_layers": (self._meta["num_layers"], hf.num_hidden_layers),
            "num_kv_heads": (self._meta["num_kv_heads"], hf.num_key_value_heads),
            "head_dim": (self._meta["head_dim"], head_dim),
        }
        for name, (a, b) in checks.items():
            if a != b:
                raise ValueError(f"HCP KV meta mismatch: {name} {a} != vllm {b}")
        vllm_dtype = str(vllm_config.model_config.dtype).replace("torch.", "")
        if vllm_dtype != self._meta["dtype"]:
            raise ValueError(
                f"HCP KV dtype mismatch: {self._meta['dtype']} != vllm {vllm_dtype}"
            )
        self._requests_need_load: dict[str, "Request"] = {}
        logger.info(
            "HcpFileConnector ready: kv_dir=%s seq_len=%d block_size=%d role=%s",
            kv_dir, self._meta["seq_len"], self._block_size, role,
        )

    # ---- scheduler side ----

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        if token_ids != self._meta["prompt_token_ids"]:
            return 0, False
        n = _align_down(len(token_ids) - 1, self._block_size)
        if n <= num_computed_tokens:
            return 0, False
        logger.info("HCP KV hit: loading %d external tokens", n)
        return n - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> HcpConnectorMetadata:
        meta = HcpConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id not in self._requests_need_load:
                continue
            token_ids = list(new_req.prompt_token_ids or [])
            n_load = _align_down(len(token_ids) - 1, self._block_size)
            block_ids = list(new_req.block_ids[0])
            block_ids_t = torch.tensor(block_ids)
            num_blocks = block_ids_t.shape[0]
            slot_mapping = (
                torch.arange(0, self._block_size).reshape((1, self._block_size))
                + block_ids_t.reshape((num_blocks, 1)) * self._block_size
            ).flatten()[:n_load]
            meta.requests.append(ReqMeta(slot_mapping=slot_mapping, n_tokens=n_load))
        self._requests_need_load.clear()
        return meta

    # ---- worker side ----

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HcpConnectorMetadata)
        if not metadata.requests:
            return
        for req in metadata.requests:
            logger.info("HCP: injecting %d tokens of KV into paged cache",
                        req.n_tokens)
            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                m = re.search(r"layers\.(\d+)\.", layer_name)
                assert m, f"cannot parse layer index from {layer_name}"
                kv = load_file(
                    os.path.join(self._kv_dir, f"layer_{int(m.group(1))}.safetensors")
                )["kv"]  # [S, 2, Hkv, D]
                kv = kv[: req.n_tokens].to(kv_cache_layer.device)
                sm = req.slot_mapping.to(kv_cache_layer.device)
                block_idxs = sm // self._block_size
                offsets = sm % self._block_size
                # vllm 0.31 统一 layout：[num_blocks, Hkv, block_size, 2*head_dim]
                # 最后一维前 head_dim 为 K、后半为 V
                d = kv.shape[-1]
                kv_cache_layer[block_idxs, :, offsets, :d] = kv[:, 0]
                kv_cache_layer[block_idxs, :, offsets, d:] = kv[:, 1]

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        return
