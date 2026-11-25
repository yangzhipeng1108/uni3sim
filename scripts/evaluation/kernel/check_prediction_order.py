#!/usr/bin/env python3
"""Thin CLI for deterministic three-hardware kernel order checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.prediction_order import (
    DEFAULT_EPSILON_NS,
    FAMILIES,
    HARDWARE_IDS,
    REPOSITORY_ROOT,
    run_order_check,
)


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-id", default="all", choices=("all",) + HARDWARE_IDS)
    parser.add_argument("--family", default="all", choices=("all",) + FAMILIES)
    parser.add_argument("--models", type=_csv, default=(), help="comma-separated configs/model stems")
    parser.add_argument("--jobs", type=int, default=48, help="independent prediction workers (1..48)")
    parser.add_argument("--scan-size", choices=("smoke", "full"), default="full")
    parser.add_argument("--epsilon-ns", type=float, default=DEFAULT_EPSILON_NS)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "experiment_logs/evaluation/kernel/order_check/latest",
        help="new or empty report directory",
    )
    arguments = parser.parse_args()
    hardware_ids = HARDWARE_IDS if arguments.hardware_id == "all" else (arguments.hardware_id,)
    families = FAMILIES if arguments.family == "all" else (arguments.family,)
    manifest = run_order_check(
        output_dir=arguments.output_dir, hardware_ids=hardware_ids,
        families=families, model_names=arguments.models, jobs=arguments.jobs,
        scan_size=arguments.scan_size, epsilon_ns=arguments.epsilon_ns)
    counts = manifest["counts"]
    print(
        f"order-check complete: success={counts['success']} "
        f"unsupported={counts['unsupported']} failed={counts['failed']} "
        f"violations={counts['violations']} output={arguments.output_dir}")
    return 2 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
