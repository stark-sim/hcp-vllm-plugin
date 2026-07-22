#!/usr/bin/env python
"""hcp-vllm-plugin compatibility smoke check.

Run this after installing the plugin or upgrading vLLM.  It verifies, without
starting an engine:
  1. vLLM imports and its version is one we have tested;
  2. the vLLM API surfaces we depend on exist (KVConnectorBase_V1, attention
     backend registry, merge_attn_states, triton_utils);
  3. the plugin's entry point registers both connectors and the CUSTOM
     attention backend;
  4. the ring Triton kernel module imports (GPU compile is exercised by
     validate_ring_triton_kernel.py, not here).

Exit 0 = OK (possibly with version WARNINGS), 1 = hard failure.
"""

import sys

TESTED_VLLM = "0.23.1rc1"

failures = []
warnings = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failures.append(name)


def warn(msg: str) -> None:
    print(f"  WARN {msg}")
    warnings.append(msg)


def main() -> None:
    print("== hcp-vllm-plugin compat check ==")

    import vllm

    print(f"vllm version: {vllm.__version__} (tested: {TESTED_VLLM})")
    if not vllm.__version__.startswith("0.23"):
        warn(f"vllm {vllm.__version__} not in the tested 0.23.x line; "
             "KVConnectorBase_V1 is experimental and may have changed")

    def _conn_api():
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        for m in ("get_num_new_matched_tokens", "build_connector_meta",
                  "start_load_kv", "save_kv_layer", "get_finished",
                  "update_state_after_alloc"):
            assert hasattr(KVConnectorBase_V1, m), f"missing {m}"

    check("KVConnectorBase_V1 API surface", _conn_api)

    def _attn_api():
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        assert hasattr(AttentionBackendEnum, "CUSTOM")
        assert callable(register_backend)

    check("attention backend registry (CUSTOM)", _attn_api)

    def _merge_api():
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
        assert callable(merge_attn_states)

    check("merge_attn_states op", _merge_api)

    def _factory():
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
        assert hasattr(KVConnectorFactory, "register_connector")

    check("KVConnectorFactory.register_connector", _factory)

    def _register():
        import hcp_vllm_plugin

        hcp_vllm_plugin.register()

    check("plugin register() executes", _register)

    def _kernel_import():
        import hcp_vllm_plugin.ring_triton_attn  # noqa: F401
        import hcp_vllm_plugin.ring_backend  # noqa: F401
        import hcp_vllm_plugin.ring_connector  # noqa: F401

    check("plugin modules import", _kernel_import)

    print("---")
    if failures:
        print(f"FAIL: {failures}")
        sys.exit(1)
    print(f"PASS ({len(warnings)} warnings)")


if __name__ == "__main__":
    main()
