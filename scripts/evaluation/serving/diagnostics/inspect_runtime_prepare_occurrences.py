#!/usr/bin/env python3
"""Inspect source-authored runtime-prepare occurrence signatures for one cell.

This diagnostic wraps the in-process compact evaluator so the production
simulation remains unchanged while each lowered KernelPlan's mechanism counts
and token shape are summarized.  It consumes no observed duration.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dataset_pipeline.serving_dataset import load_compact_case
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.simulation import simulation_runner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    root = Path(__file__).resolve().parents[4]
    config = compact.load_phase1_config(
        root / "configs/evaluation/experiments/sharegpt_phase1.json")
    cell = compact.select_phase1_cells(
        config, [arguments.case], all_cases=False)[0]
    original_builder = simulation_runner.runtime_prepare_events
    signatures: dict[tuple[tuple[tuple[str, object], ...], tuple[tuple[str, int], ...]], int] = {}

    def wrapped_builder(plan, *, enabled_mechanisms):
        events = original_builder(
            plan, enabled_mechanisms=enabled_mechanisms)
        counts = Counter()
        for entries in events.values():
            for event in entries:
                counts[event.mechanism] += event.count
        token_calls = tuple(
            call for call in plan.calls
            if ":token-embedding" in call.call_id
            and ":tp-all-reduce" not in call.call_id
        )
        token_shape = (
            tuple(sorted(token_calls[0].kernel_spec.shape.items()))
            if token_calls else ()
        )
        signature = (token_shape, tuple(sorted(counts.items())))
        previous = signatures[signature] if signature in signatures else 0
        signatures[signature] = previous + 1
        return events

    simulation_runner.runtime_prepare_events = wrapped_builder
    compact.run_isolated_case(
        str(root),
        config,
        cell,
        compact.repository_source_commit(root),
        True,
        "accelerated",
        load_validated_case=load_compact_case,
        run_simulation=simulation_runner.run_cli_args,
        host_launch_profile_paths={
            "ascend_910b": str(
                root / "configs/host_timing/host_launch_profiles/ascend_910b.json")},
        scheduler_host_profile_paths={
            "ascend_910b": str(
                root / "configs/host_timing/scheduler_host_profiles/ascend_910b.json")},
    )
    print("occurrences token_shape mechanism_counts")
    for (token_shape, counts), occurrences in sorted(
            signatures.items(), key=lambda item: repr(item[0])):
        print(occurrences, token_shape, counts)


if __name__ == "__main__":
    main()
