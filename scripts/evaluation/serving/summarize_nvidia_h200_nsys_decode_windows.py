#!/usr/bin/env python3
'Summarise nsys decode windows of the real vLLM DSV3-L32 TP4 server.\n\nFor each ``window_<n>.nsys-rep`` this exports the CUDA GPU trace (graph nodes\ntraced individually), keeps device 0, splits it into engine steps at gaps\nlonger than ``--step-gap-us`` or at the sampler kernel, and reports per-step\nwall time, GPU-busy time, idle time and busy time by kernel family.  Output\nis one JSON per window plus a printed table; no timing is fed back to any\nmodel parameter (evaluation only).'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.nvidia_h200_nsys_decode_windows import main

if __name__ == "__main__":
    raise SystemExit(main())
