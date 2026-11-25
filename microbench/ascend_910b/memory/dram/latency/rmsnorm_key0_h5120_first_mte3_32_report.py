#!/usr/bin/env python3
"""Audit the exact 32-AIV RMSNorm key-0 H5120 first-MTE3 campaign."""

import argparse
from pathlib import Path

import first_mte_completion_report as base
from rmsnorm_key0_h5120_first_mte3_32_runner import (
    CELLS,
    REQUIRED_INDEPENDENT_RUNS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = base._audit_independent_runs(
        args.raw_csv, CELLS, REQUIRED_INDEPENDENT_RUNS)
    base._print_independent_report(audit)
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
