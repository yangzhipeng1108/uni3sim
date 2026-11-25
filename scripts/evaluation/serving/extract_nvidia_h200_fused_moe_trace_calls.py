#!/usr/bin/env python3
'Extract H200 Triton fused-MoE launch geometry and per-call durations.\n\nEvaluation only.  Reads one or more nsys ``cuda_gpu_trace`` CSV exports of\nthe vLLM 0.24 DSV3-L32 TP4 server, keeps the ``fused_moe_kernel`` rows of one\ndevice, classifies each launch as W1 or W2 from its source-order pair (the\nsource launches W1 then W2 per MoE layer with the same M tiling, so the W2\ngrid is the W1 grid times ceil(hidden/BN)/ceil(2*intermediate/BN)), and\nreports, per distinct (variant, grid, block, registers, shared memory)\ngeometry, the launch count and the duration median/p10/p90.  Durations are\nthe evaluation truth for WP_REPORT tables; nothing here feeds a model\nparameter.  Standard library only (runs with the remote system python3).\n\nUsage:\n    python3 extract_nvidia_h200_fused_moe_trace_calls.py --device 0         --w2-to-w1-grid-ratio 7 --window C=1::<csv> [--window ...]         --output <json>'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_fused_moe_calls import main

if __name__ == "__main__":
    raise SystemExit(main())
