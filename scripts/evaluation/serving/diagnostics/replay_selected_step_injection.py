#!/usr/bin/env python3
'Derive a full prediction fingerprint for one injected Scheduler step.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.ascend_910b_selected_step_msprof import (
    summarize_ascend_910b_model_runner_outer_timing,
)
from evaluation.serving.metrics.diagnostics.selected_step_replay import main

if __name__ == "__main__":
    raise SystemExit(main(summarize_model_runner_outer_timing=summarize_ascend_910b_model_runner_outer_timing))
