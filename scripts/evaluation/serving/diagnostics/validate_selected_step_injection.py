#!/usr/bin/env python3
"""Validate a selected-step hardware injection against its canonical state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.diagnostics.selected_step_injection import (  # noqa: E402
    validate_selected_step_injection_trace,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-step-manifest", type=Path, required=True)
    parser.add_argument("--token-prefixes", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument("--injection-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-eager", action="store_true")
    arguments = parser.parse_args()
    selected = json.loads(
        arguments.selected_step_manifest.resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    token_prefixes = json.loads(
        arguments.token_prefixes.resolve(strict=True).read_text(encoding="utf-8")
    )
    injection_replay = json.loads(
        arguments.injection_replay.resolve(strict=True).read_text(encoding="utf-8")
    )
    result = validate_selected_step_injection_trace(
        selected,
        token_prefixes,
        arguments.scheduler_trace.resolve(strict=True),
        injection_replay,
        independent_eager=arguments.independent_eager,
    )
    write_json(arguments.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
