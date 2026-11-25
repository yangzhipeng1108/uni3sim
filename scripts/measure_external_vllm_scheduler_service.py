#!/usr/bin/env python3
"""Measure a container's installed vLLM Scheduler without model execution.

This runner is deliberately version-scoped evidence: it imports the vLLM
package installed in the target container instead of Midlayer's pinned 0.11
adapter.  Its output must therefore never be consumed by a Forward contract
for a different vLLM version.
"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.profiler.external_vllm_scheduler_service import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()
