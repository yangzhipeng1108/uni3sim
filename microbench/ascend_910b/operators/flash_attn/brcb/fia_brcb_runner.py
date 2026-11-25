#!/usr/bin/env python3
"""Run the independent FIA Brcb service binary for the frozen row domain."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for row_chunk in (8, 16, 32, 64):
        completed = subprocess.run(
            (str(args.binary), str(row_chunk), str(args.iterations)),
            check=True, text=True, capture_output=True)
        lines = completed.stdout.strip().splitlines()
        if len(lines) != 2 or not lines[1].endswith(",pass"):
            raise RuntimeError(f"invalid Brcb output for row {row_chunk}")
        rows.append(lines[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "row_chunk_round,iterations,system_cycles,duration_ns,validation\n"
        + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
