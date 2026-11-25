#!/usr/bin/env python3
'Read-only aggregation of one nsys window (sqlite export), device 0, for the family tables.\nRun beside the operator-selected trace inputs.\n\nSame output contract as the TP4 ``aggregate_remote_trace_v2.py`` but read\nstraight from CUPTI_ACTIVITY_KIND_KERNEL: per (name, grid, block) calls, raw\nns, overlap-shared ns (each busy instant split equally among the kernels\nrunning then; sums to the busy union), streams; idle gaps between busy\nsegments; and the window span.  Evaluation only.\nUsage: python aggregate_nvidia_h200_nsys_window_kernels.py <window.sqlite> > window_dev0_share.json'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_nsys_window_kernels import main

if __name__ == "__main__":
    raise SystemExit(main())
