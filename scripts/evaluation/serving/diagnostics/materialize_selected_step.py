#!/usr/bin/env python3
"""Write complete C_step rankings and one selected-step injection input."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.diagnostics.selected_step_injection import (  # noqa: E402
    build_step_rankings,
    materialize_selected_step,
    write_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    replay_path = arguments.replay.resolve(strict=True)
    scheduler_trace_path = arguments.scheduler_trace.resolve(strict=True)
    replay_payload = json.loads(replay_path.read_text())
    rankings = build_step_rankings(replay_payload)
    manifest = materialize_selected_step(
        replay_payload, scheduler_trace_path, arguments.step_id
    )
    write_json(arguments.ranking_output.resolve(), rankings)
    write_json(arguments.manifest_output.resolve(), manifest)
    print(json.dumps({
        "ranking_output": str(arguments.ranking_output.resolve()),
        "manifest_output": str(arguments.manifest_output.resolve()),
        "selected_step_id": arguments.step_id,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
