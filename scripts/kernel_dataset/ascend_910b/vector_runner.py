#!/usr/bin/env python3
"""Run one Ascend 910B Vector workload or a persistent Event batch."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel import device_event
from dataset_pipeline.kernel_dataset.ascend_910b_vector_runtime import (
    DTYPE_MAP,
    _build_graph_call_set,
    _load_event_batch_manifest,
    _rotating_asymmetric_paired_input_operation,
    _rotating_input_operation,
    _rotating_kvcache_operation,
    _rotating_paired_input_operation,
    _run,
    _run_event_batch,
    configure_event_backend,
    main,
    run_add_rmsnorm,
    run_elementwise,
    run_kvcache,
    run_rmsnorm,
    run_rope,
    run_silu,
    torch,
)


configure_event_backend(device_event)


if __name__ == "__main__":
    raise SystemExit(main())
