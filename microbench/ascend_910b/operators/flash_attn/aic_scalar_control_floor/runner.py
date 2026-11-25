#!/usr/bin/env python3
"""Run the registered FIA AIC scalar control-floor campaign.

Cells, gates and aggregation are fixed by REGISTRATION.md, written before any
data was collected.  The runner never selects between estimators.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

PROCESS_COUNT = 3
GROUP_COUNTS = (1, 4, 8, 16, 22, 24)
ITER_COUNTS = (0, 16, 64, 256)
PROCESS_MEDIAN_SPAN_LIMIT_PCT = 5.0
SEPARATION_MINIMUM_PCT = 10.0
PUBLISHED_ENVELOPE_STARTUP_NS = 5213.005105048369
PUBLISHED_ENVELOPE_PER_AIV_NS = 55.68101764860289
ENVELOPE_AGREEMENT_LIMIT_PCT = 15.0
EXPECTED_HEADER = (
    "active_aic_groups,iters,aic_max_cycles,aiv_max_cycles,tick_ns,"
    "internal_ns,host_ns,envelope_ns,validation")


def run_once(binary: Path, groups: int, iters: int) -> dict:
    completed = subprocess.run(
        (str(binary), str(groups), str(iters)),
        check=True, text=True, capture_output=True)
    if completed.stderr != "":
        raise RuntimeError(f"probe emitted stderr: {completed.stderr!r}")
    lines = completed.stdout.splitlines()
    if len(lines) != 2 or lines[0] != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected probe output: {completed.stdout!r}")
    f = lines[1].split(",")
    if len(f) != 9 or f[8] != "pass":
        raise RuntimeError(f"failed validation: {lines[1]!r}")
    result = {
        "active_aic_groups": int(f[0]), "iters": int(f[1]),
        "aic_max_cycles": int(f[2]), "aiv_max_cycles": int(f[3]),
        "tick_ns": float(f[4]), "internal_ns": float(f[5]),
        "host_ns": float(f[6]), "envelope_ns": float(f[7]),
    }
    if result["active_aic_groups"] != groups or result["iters"] != iters:
        raise RuntimeError(f"probe reported the wrong cell: {result!r}")
    return result


def span_pct(values):
    median = statistics.median(values)
    if median == 0.0:
        return 0.0
    return (max(values) - min(values)) / median * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if args.output.exists():
        raise FileExistsError(args.output)
    device = os.environ["MIDLAYER_ASCEND_DEVICE"]

    cells = []
    for groups in GROUP_COUNTS:
        for iters in ITER_COUNTS:
            runs = [run_once(args.binary, groups, iters)
                    for _ in range(PROCESS_COUNT)]
            internal = [r["internal_ns"] for r in runs]
            host = [r["host_ns"] for r in runs]
            envelope = [r["envelope_ns"] for r in runs]
            cell = {
                "active_aic_groups": groups, "iters": iters, "runs": runs,
                "internal_ns": statistics.median(internal),
                "host_ns": statistics.median(host),
                "envelope_ns": statistics.median(envelope),
                "internal_span_pct": span_pct(internal),
                "host_span_pct": span_pct(host),
                "internal_stable": (
                    span_pct(internal) <= PROCESS_MEDIAN_SPAN_LIMIT_PCT),
                "host_stable": span_pct(host) <= PROCESS_MEDIAN_SPAN_LIMIT_PCT,
            }
            cells.append(cell)
            print(json.dumps({
                "groups": groups, "iters": iters,
                "internal_ns": round(cell["internal_ns"], 1),
                "host_ns": round(cell["host_ns"], 1),
                "internal_span_pct": round(cell["internal_span_pct"], 2),
                "host_span_pct": round(cell["host_span_pct"], 2)}), flush=True)

    def cell_of(groups, iters):
        for cell in cells:
            if cell["active_aic_groups"] == groups and cell["iters"] == iters:
                return cell
        raise KeyError((groups, iters))

    base24 = cell_of(24, 0)["internal_ns"]
    top24 = cell_of(24, 256)["internal_ns"]
    separation_pct = (
        float("inf") if base24 == 0.0 else (top24 - base24) / base24 * 100.0)

    per_iteration = []
    for groups in GROUP_COUNTS:
        zero = cell_of(groups, 0)["internal_ns"]
        for iters in ITER_COUNTS:
            if iters == 0:
                continue
            per_iteration.append({
                "active_aic_groups": groups, "iters": iters,
                "ns_per_iteration": (
                    cell_of(groups, iters)["internal_ns"] - zero) / iters})

    envelope_check = []
    for groups in GROUP_COUNTS:
        measured = cell_of(groups, 0)["envelope_ns"]
        published = (PUBLISHED_ENVELOPE_STARTUP_NS
                     + PUBLISHED_ENVELOPE_PER_AIV_NS * 2 * groups)
        envelope_check.append({
            "active_aic_groups": groups,
            "measured_envelope_ns": measured,
            "published_envelope_ns": published,
            "excess_pct": (measured - published) / published * 100.0,
            "within_15pct": (
                abs(measured - published) / published * 100.0
                <= ENVELOPE_AGREEMENT_LIMIT_PCT),
        })

    record = {
        "schema": "aic_scalar_control_floor_v1",
        "baseline": args.baseline,
        "hardware_id": "ascend_910b",
        "physical_device_id": device,
        "process_count": PROCESS_COUNT,
        "timing_boundary": (
            "internal: GetSystemCycle before the first iteration's "
            "WaitFlag<MTE1_MTE2> to GetSystemCycle after PipeBarrier<PIPE_ALL> "
            "following the last slot release, max over AIC cores per launch; "
            "host: aclrtEventElapsedTime across launch and synchronize; "
            "envelope = host minus internal; process median per cell"),
        "cells": cells,
        "separation_pct_at_24_groups": separation_pct,
        "per_iteration_ns": per_iteration,
        "envelope_vs_published": envelope_check,
        "gates": {
            "all_cells_internal_stable": all(c["internal_stable"] for c in cells),
            "all_cells_host_stable": all(c["host_stable"] for c in cells),
            "stderr_empty": True,
            "separation_at_least_10pct": separation_pct >= SEPARATION_MINIMUM_PCT,
        },
    }
    record["passed"] = all(record["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "passed": record["passed"],
                      "gates": record["gates"]}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
