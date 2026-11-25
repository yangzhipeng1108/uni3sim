#!/usr/bin/env python3
"""Compare a captured Ascend startup with analytical activation and KV capacity."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from evaluation.serving.metrics.native_startup_memory import main

if __name__=='__main__':
    raise SystemExit(main())
