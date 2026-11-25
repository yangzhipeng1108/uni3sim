"""Reduce the AIV instruction-fetch sweep to a rate with a unit.

Reads one msprof capture per (text variant, repeat), pairs each process's first
AIV task against the warm tasks that follow it in the same process, and
regresses the cold-minus-warm penalty on the kernel-text bytes the variant
actually carries.  The slope is an instruction-fetch rate in bytes per second;
the intercept is the fixed cold-start cost that does not depend on footprint.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics


def _task_rows(capture_dir: pathlib.Path) -> list[dict[str, str]]:
    summaries = sorted(capture_dir.rglob("op_summary*.csv"))
    if not summaries:
        raise FileNotFoundError(f"no op_summary under {capture_dir}")
    rows: list[dict[str, str]] = []
    for summary in summaries:
        with summary.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _kernel_tasks(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["Op Name"].startswith(name)]
    if not selected:
        raise ValueError(f"capture holds no task named {name!r}")
    selected.sort(key=lambda row: float(row["Task Start Time(us)"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", required=True)
    parser.add_argument("--text-bytes", required=True,
                        help="JSON object mapping variant -> text byte count")
    parser.add_argument("--kernel-name", default="aiv_text_footprint")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    footprints = json.loads(pathlib.Path(arguments.text_bytes).read_text())
    captures = pathlib.Path(arguments.captures)
    cells = []
    for variant in sorted(footprints, key=int):
        colds: list[float] = []
        warms: list[float] = []
        cold_miss: list[float] = []
        warm_miss: list[float] = []
        for capture in sorted(path for path in captures.glob(f"n{variant}_r*")
                              if path.is_dir()):
            tasks = _kernel_tasks(_task_rows(capture), arguments.kernel_name)
            durations = [float(task["Task Duration(us)"]) for task in tasks]
            misses = [float(task["aiv_icache_miss_rate"]) for task in tasks]
            colds.append(durations[0])
            warms.append(statistics.median(durations[1:]))
            cold_miss.append(misses[0])
            warm_miss.append(statistics.median(misses[1:]))
        cells.append({
            "text_body": int(variant),
            "text_bytes": int(footprints[variant]),
            "processes": len(colds),
            "cold_us_median": statistics.median(colds),
            "warm_us_median": statistics.median(warms),
            "penalty_us_median": statistics.median(colds) - statistics.median(warms),
            "cold_icache_miss_rate_median": statistics.median(cold_miss),
            "warm_icache_miss_rate_median": statistics.median(warm_miss),
        })

    count = len(cells)
    mean_x = sum(cell["text_bytes"] for cell in cells) / count
    mean_y = sum(cell["penalty_us_median"] for cell in cells) / count
    covariance = sum(
        (cell["text_bytes"] - mean_x) * (cell["penalty_us_median"] - mean_y)
        for cell in cells)
    variance = sum((cell["text_bytes"] - mean_x) ** 2 for cell in cells)
    if variance <= 0.0:
        raise ValueError("text byte counts do not vary across the sweep")
    slope_us_per_byte = covariance / variance
    intercept_us = mean_y - slope_us_per_byte * mean_x
    residual = sum(
        (cell["penalty_us_median"]
         - (intercept_us + slope_us_per_byte * cell["text_bytes"])) ** 2
        for cell in cells)
    total = sum((cell["penalty_us_median"] - mean_y) ** 2 for cell in cells)

    pathlib.Path(arguments.output).write_text(json.dumps({
        "cells": cells,
        "fit": {
            "slope_us_per_byte": slope_us_per_byte,
            "intercept_us": intercept_us,
            "instruction_fetch_bytes_per_second": (
                1.0e6 / slope_us_per_byte if slope_us_per_byte > 0 else None),
            "r_squared": 1.0 - residual / total if total > 0 else None,
        },
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
