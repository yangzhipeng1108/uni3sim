#!/usr/bin/env python3
"""Explicit entry point for the independent LLMServingSim CPU baseline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.serving.experiments.llmservingsim import main


if __name__ == "__main__":
    if any(flag in sys.argv[1:] for flag in ("--standard-results", "--prepare-targets", "--midlayer-targets", "--midlayer-command", "--target-case")):
        from dataset_pipeline.serving_dataset import load_compact_case
        raise SystemExit(main(load_case=load_compact_case))
    raise SystemExit(main())
