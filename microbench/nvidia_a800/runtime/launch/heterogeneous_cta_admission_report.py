#!/usr/bin/env python3
"""Validate the two-slot SM80 CTA admission capture and publish facts."""

import argparse
import csv
import json
from pathlib import Path


GRID_X = 32
GRID_Y = 4
GRID_Z = 32
SM_COUNT = 108
RESIDENT_CTAS = 2
INITIAL_CTAS = SM_COUNT * RESIDENT_CTAS


def _expected_work_class(block_x, block_y):
    limit = (8, 8, 16, 32)[block_y]
    return 0 if block_x >= limit else (block_x + 4) // 4


def _load(path):
    rows = []
    metadata = {}
    for line in path.read_text().splitlines():
        fields = tuple(next(csv.reader([line])))
        if fields[0] == "meta":
            metadata[fields[1]] = fields[2:]
        elif fields[0] == "raw" and fields[1] != "trial":
            values = tuple(int(value) for value in fields[1:])
            names = (
                "trial", "linear", "x", "y", "z", "work", "sm",
                "start_ticket", "end_ticket", "start_cycle", "end_cycle")
            rows.append(dict(zip(names, values)))
    return metadata, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    capture = Path(args.capture).resolve()
    metadata, rows = _load(capture / "raw.csv")
    if metadata["sm_count"] != (str(SM_COUNT),):
        raise ValueError("capture SM count changed")
    if metadata["resident_ctas_per_sm"] != (str(RESIDENT_CTAS),):
        raise ValueError("capture residency changed")
    if metadata["grid"] != (str(GRID_X), str(GRID_Y), str(GRID_Z)):
        raise ValueError("capture grid changed")
    trials = sorted({row["trial"] for row in rows})
    expected_blocks = GRID_X * GRID_Y * GRID_Z
    trial_reports = []
    for trial in trials:
        records = [row for row in rows if row["trial"] == trial]
        if len(records) != expected_blocks:
            raise ValueError(f"trial {trial} has incomplete records")
        if {row["linear"] for row in records} != set(range(expected_blocks)):
            raise ValueError(f"trial {trial} has invalid block IDs")
        for row in records:
            if row["work"] != _expected_work_class(row["x"], row["y"]):
                raise ValueError(f"trial {trial} work class mismatch")
            if row["end_cycle"] <= row["start_cycle"]:
                raise ValueError(f"trial {trial} has nonpositive CTA span")
        starts = sorted(records, key=lambda row: row["start_ticket"])
        initial = starts[:INITIAL_CTAS]
        if {row["linear"] for row in initial} != set(range(INITIAL_CTAS)):
            raise ValueError(f"trial {trial} initial admission set changed")
        initial_by_sm = {}
        for row in initial:
            if row["sm"] not in initial_by_sm:
                initial_by_sm[row["sm"]] = []
            initial_by_sm[row["sm"]].append(row["linear"])
        pairs = sorted(tuple(sorted(values)) for values in initial_by_sm.values())
        expected_pairs = [(2 * index, 2 * index + 1)
                          for index in range(SM_COUNT)]
        if pairs != expected_pairs:
            raise ValueError(f"trial {trial} initial CTA pairing changed")
        queued_order = [row["linear"] for row in starts[INITIAL_CTAS:]]
        displacement = tuple(
            abs((INITIAL_CTAS + index) - block)
            for index, block in enumerate(queued_order))
        events = sorted(
            [(row["start_ticket"], 1) for row in records]
            + [(row["end_ticket"], -1) for row in records])
        active = 0
        maximum_active = 0
        for _, delta in events:
            active += delta
            if active < 0:
                raise ValueError(f"trial {trial} event order is invalid")
            maximum_active = max(maximum_active, active)
        if active != 0 or maximum_active != INITIAL_CTAS:
            raise ValueError(f"trial {trial} resident population changed")
        trial_reports.append({
            "trial": trial,
            "observed_sm_count": len(initial_by_sm),
            "initial_linear_pairing_exact": True,
            "maximum_active_ctas": maximum_active,
            "queued_start_order_max_displacement": max(displacement),
            "queued_start_order_p99_displacement": sorted(displacement)[
                int(0.99 * (len(displacement) - 1))],
        })
    candidate = {
        "schema": "nvidia_a800_heterogeneous_cta_admission_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "two_resident_cta_source_ordered_work_conserving_queue",
        "trial_count": len(trials),
        "trial_reports": trial_reports,
        "all_initial_linear_pairing_exact": all(
            report["initial_linear_pairing_exact"]
            for report in trial_reports),
        "maximum_queued_start_order_displacement": max(
            report["queued_start_order_max_displacement"]
            for report in trial_reports),
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "For this independent 128-thread, two-resident-CTA SM80 probe, "
            "the first 216 linear CTAs fill 108 paired SM slots and queued "
            "heterogeneous CTAs are admitted continuously in near-linear "
            "launch order as slots complete."),
        "excluded_conclusions": [
            "production FlashAttention CTA duration",
            "a fixed CTA-to-SM identity",
            "a round-robin assignment of heterogeneous CTAs",
        ],
    }
    (capture / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
