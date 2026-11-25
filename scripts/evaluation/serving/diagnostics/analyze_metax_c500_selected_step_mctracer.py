#!/usr/bin/env python3
"""Correlate a validated C500 selected-step capture with mcTracer tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.metax_c500_selected_step_mctracer import (  # noqa: E402
    analyze_c500_independent_eager_capture,
    analyze_c500_selected_step_capture,
    analyze_c500_standalone_prefill_capture,
    analyze_c500_unanchored_graph_device_union,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--independent-eager", action="store_true")
    mode.add_argument("--unanchored-device-union", action="store_true")
    mode.add_argument("--standalone-prefill-eager", action="store_true")
    arguments = parser.parse_args()
    result = (
        analyze_c500_independent_eager_capture(arguments.capture_dir)
        if arguments.independent_eager
        else (
            analyze_c500_unanchored_graph_device_union(arguments.capture_dir)
            if arguments.unanchored_device_union
            else (
                analyze_c500_standalone_prefill_capture(arguments.capture_dir)
                if arguments.standalone_prefill_eager
                else analyze_c500_selected_step_capture(arguments.capture_dir)
            )
        )
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
