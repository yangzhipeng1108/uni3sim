#!/usr/bin/env python3
'Extract real rank-0 attention-family kernel calls from H200 nsys traces.\n\nEvaluation only.  For each ``window_<n>_cuda_gpu_trace.csv`` this keeps\ndevice 0 (rank 0), selects kernels whose demangled name matches the same\n``attention`` family regex as ``summarize_nvidia_h200_nsys_decode_windows.py``, and\ngroups calls by the physical launch (name, grid, block, registers, static and\ndynamic shared memory) and by the token count of the step the call belongs to\n(grid X of the most recent ``concat_and_cache_mla`` launch).  Per group it reports the call count, calls per step\n(steps counted by the once-per-MoE-projection grouped GEMM, as in the\nsummariser) and the median / mean / p10 / p90 duration.  Runs with the plain\nsystem ``python3`` (standard library only) so it can execute next to the\ntraces on the capture host.  No duration feeds any model parameter.\n\nUsage:\n    python3 extract_nvidia_h200_attention_kernel_calls.py --output calls.json         C=1:<dir>/window_1_cuda_gpu_trace.csv ...'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_attention_kernel_calls import main

if __name__ == "__main__":
    raise SystemExit(main())
