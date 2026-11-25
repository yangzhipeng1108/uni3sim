#!/usr/bin/env python3
"""Export source-owned memory programs and their untimed diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.memory_interface import main

if __name__ == "__main__":
    main()
