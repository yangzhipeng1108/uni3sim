#!/usr/bin/env python3
"""Identify which compile backend a GT-configured 910B engine actually takes.

Every synthetic microbenchmark in this repository has priced the eager step
through ``torch.compile(backend="eager")``.  The canonical manifest records
``configured_cudagraph_mode = FULL``, and ``scripts/evaluation/serving/
diagnostics/selected_step_runner.py`` builds the 910B engine with a
``compilation_config`` that sets *only* ``cudagraph_mode`` and
``max_cudagraph_capture_size`` -- no ``mode``/``level`` key, unlike the A800
and C500 branches beside it.  So ``CompilationMode`` is whatever the vLLM
default resolves to, and that decides what an above-capture-limit step runs:

  VLLM_COMPILE -> AscendCompiler -> npugraph_ex_compile -> ``import torchair``
  NONE         -> no compiled artifact at all; a plain eager module forward

``torchair`` is not installed in this container, so the first branch cannot
silently succeed.  This runner constructs the engine exactly as the capture
path does and reports the resolved configuration, which settles whether the
charged object (``compiled_fx_*_node_dispatch``, one dispatch per FX node)
corresponds to anything production executes.

No production duration is read and nothing is priced here; the output is a
configuration identity, not a parameter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

HARDWARE_ID = "ascend_910b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture-limit", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES must select an idle NPU")

    from vllm import LLM

    llm = LLM(
        model=arguments.model,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_prefix_caching=True,
        dtype="bfloat16",
        compilation_config={
            "cudagraph_mode": "FULL",
            "max_cudagraph_capture_size": arguments.capture_limit,
        },
        cudagraph_metrics=True,
    )

    engine_config = llm.llm_engine.vllm_config
    compilation = engine_config.compilation_config
    from vllm_ascend.ascend_config import get_ascend_config

    ascend = get_ascend_config().ascend_compilation_config
    resolved = {
        "compilation_mode": str(compilation.mode),
        "cudagraph_mode": str(compilation.cudagraph_mode),
        "splitting_ops": list(compilation.splitting_ops or []),
        "use_inductor": bool(compilation.use_inductor),
        "max_capture_size": getattr(compilation, "max_cudagraph_capture_size", None),
        "enable_npugraph_ex": bool(ascend.enable_npugraph_ex),
        "oot_compiler": str(getattr(compilation, "oot_compiler", None)),
    }
    for name, value in resolved.items():
        print(f"{name:24s} {value}")

    try:
        import torchair  # noqa: F401
        torchair_importable = True
    except ImportError:
        torchair_importable = False
    print(f"{'torchair_importable':24s} {torchair_importable}")

    arguments.output.write_text(json.dumps({
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "resolved compile-backend identity of a 910B engine constructed "
            "with the canonical capture-path compilation_config"),
        "forward_eligibility": "identity_only_not_a_parameter",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "container": arguments.container_name,
        "host": platform.node(),
        "model": arguments.model,
        "resolved": resolved,
        "torchair_importable": torchair_importable,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
