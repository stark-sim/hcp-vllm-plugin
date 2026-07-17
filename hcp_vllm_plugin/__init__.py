"""HCP vLLM plugin — context-parallel block-KV ring connector.

This package provides a vLLM KV-connector (KVConnectorBase_V1) that lets
multiple vLLM instances on heterogeneous nodes cooperate on one long sequence
via context-passing prefill: each instance computes only its own chunk and
loads the earlier chunks' KV from the previous instance.

It is registered as a ``vllm.general_plugins`` entry point and the connector
can also be loaded directly via ``kv_connector_module_path``.
"""

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)


def register() -> None:
    """Register the HCP CP connector with vLLM's connector factory."""
    KVConnectorFactory.register_connector(
        "HcpCpConnector",
        "hcp_vllm_plugin.connector",
        "HcpCpConnector",
    )
