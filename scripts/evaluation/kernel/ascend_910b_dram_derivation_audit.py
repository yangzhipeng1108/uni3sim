#!/usr/bin/env python3
"""CLI for the Ascend 910B DRAM-derivation audit."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.ascend_910b_dram_derivation_audit import (
    main)


if __name__ == "__main__":
    raise SystemExit(main())
