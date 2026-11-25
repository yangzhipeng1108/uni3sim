#!/usr/bin/env python3
"""Run a deterministic pure-decode selected-step probe on vLLM hardware."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.vllm_scheduler_trace import (  # noqa: E402
    finalize_scheduler_trace_from_environment,
    scheduler_trace_pending_step_count_from_environment,
)
from evaluation.serving.metrics.diagnostics.selected_step_probe import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main(
        scheduler_trace_pending_step_count_from_environment=(
            scheduler_trace_pending_step_count_from_environment
        ),
        finalize_scheduler_trace_from_environment=(
            finalize_scheduler_trace_from_environment
        ),
    )
