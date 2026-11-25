#!/usr/bin/env python3
"""Run position-balanced elementwise Mul compute-segment trials."""

import argparse
import csv
import io
import os
from pathlib import Path
import subprocess


BASE_GEOMETRIES = (
    (46, 1792), (46, 1280), (47, 1408), (47, 768),
    (47, 2816), (47, 1536),
)
ACTIVE48_MULTITILE_GEOMETRIES = (
    (48, 8832), (48, 8448), (48, 4352),
)
CELLS = tuple(
    (workers, elements, segment)
    for workers, elements in (
        BASE_GEOMETRIES + ACTIVE48_MULTITILE_GEOMETRIES)
    for segment in (0, 1))
EXPECTED_COLUMNS = (
    "workers", "elements", "segment", "cycles", "service_ns",
    "validation")


def _parse_child(stdout, expected):
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if tuple(rows[0]) != EXPECTED_COLUMNS or len(rows) != len(expected):
        raise RuntimeError("child emitted unexpected columns or row count")
    parsed = []
    for row, cell in zip(rows, expected):
        integers = tuple(int(row[name]) for name in EXPECTED_COLUMNS[:4])
        if integers[:3] != cell or row["validation"] != "pass":
            raise RuntimeError("child row disagrees with requested cell")
        service_ns = float(row["service_ns"])
        if service_ns <= 0.0:
            raise RuntimeError("child emitted nonpositive service")
        parsed.append(row)
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if args.device_id < 0 or args.device_id > 7 or args.cycles <= 0:
        parser.error("device-id must be in [0,7] and cycles must be positive")
    trials = args.cycles * len(CELLS)
    print(
        "run_id,trial,order_position,process_pid," + ",".join(
            EXPECTED_COLUMNS), flush=True)
    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = str(args.device_id)
    for trial in range(trials):
        offset = trial % len(CELLS)
        ordered = CELLS[offset:] + CELLS[:offset]
        command = [str(args.binary), "--batch"] + [
            f"{workers}:{elements}:{segment}"
            for workers, elements, segment in ordered]
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment)
        stdout, stderr = child.communicate()
        if child.returncode != 0:
            raise subprocess.CalledProcessError(
                child.returncode, command, output=stdout, stderr=stderr)
        rows = _parse_child(stdout, ordered)
        for order_position, row in enumerate(rows):
            print(
                f"{args.run_id},{trial + 1},{order_position},{child.pid},"
                + ",".join(row[name] for name in EXPECTED_COLUMNS),
                flush=True)


if __name__ == "__main__":
    main()
