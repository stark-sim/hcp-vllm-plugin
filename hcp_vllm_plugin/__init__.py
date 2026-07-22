"""HCP vLLM plugin — heterogeneous context-parallel ring attention for vLLM.

This package lets multiple vLLM instances on heterogeneous nodes (e.g. CUDA +
ROCm) cooperate on long sequences via memory-splitting context parallelism:

  * ``HcpRingKvConnector`` (KVConnectorBase_V1): the scheduler marks earlier
    chunks as externally computed (global RoPE positions, no recompute); the
    worker fetches their per-layer KV over HTTP and stages it TRANSIENTLY —
    never into the local paged pool.  Per-request chunk assignment rides on
    ``SamplingParams(extra_args={"kv_transfer_params": {"hcp_ring": ...}})``,
    so concurrent requests may reference different peer chunks.
  * ``HcpRingAttentionBackend`` (registered as CUSTOM): online-softmax merge
    of local (causal) + peer (transient, non-causal) attention via the
    plugin's Triton kernel (ring_triton_attn, LSE output, CUDA and ROCm).
  * ``HcpCpConnector``: earlier full-KV context-passing connector, kept for
    comparison/reference.

Registered as a ``vllm.general_plugins`` entry point; the connectors can also
be loaded directly via ``kv_connector_module_path``.
"""

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)


def register() -> None:
    """Register the HCP CP connector and the HCP ring attention backend."""
    KVConnectorFactory.register_connector(
        "HcpCpConnector",
        "hcp_vllm_plugin.connector",
        "HcpCpConnector",
    )
    # Memory-splitting ring-KV connector: scheduler marks the peer chunk as
    # externally computed; worker stages peer KV transiently into
    # ring_backend.PEER_KV_STAGING (never into the paged pool).
    KVConnectorFactory.register_connector(
        "HcpRingKvConnector",
        "hcp_vllm_plugin.ring_connector",
        "HcpRingKvConnector",
    )
    # Memory-splitting ring-attention (online-softmax) attention backend.
    # Select with `--attention-backend CUSTOM` / LLM(attention_backend="CUSTOM").
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "hcp_vllm_plugin.ring_backend.HcpRingAttentionBackend",
    )
