#!/usr/bin/env python3
"""Validate the A800 two-resident-CTA finite LDGSTS capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


SOURCE_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite.cu")
RUNNER_PATH = Path(__file__).with_name(
    "fa2_resident_ldgsts_finite_runner.py")
REPEATS = (1, 2, 4, 8, 16, 32, 64)
SAMPLES = 7
GRID_CTAS = 216
THREADS_PER_CTA = 128
DYNAMIC_SHARED_BYTES = 77824
PANEL_BYTES_PER_CTA = 65536
TRANSACTION_BYTES = 32
EXPECTED_REGISTERS = 79
EXPECTED_STATIC_LDGSTS = 32
MAX_HOLDOUT_APE_PCT = 15.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(text, name):
    normalized = text.strip().replace(",", "")
    try:
        value = float(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _dimension(text, name):
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        text.strip())
    if match is None:
        raise ValueError(f"{name} has invalid CUDA dimension syntax")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _read_ncu_rows(path, repeats):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(
        index for index, line in enumerate(lines)
        if line.startswith('"ID","Process ID"'))
    rows = tuple(
        row for row in csv.DictReader(lines[header:])
        if row["ID"].strip().isdigit())
    if len(rows) != SAMPLES:
        raise ValueError(f"{path} must contain exactly {SAMPLES} launches")
    cycles = []
    expected_instructions = GRID_CTAS * THREADS_PER_CTA * repeats
    expected_xbar_bytes = GRID_CTAS * PANEL_BYTES_PER_CTA * repeats
    for row in rows:
        if _dimension(row["Grid Size"], "NCU grid") != (GRID_CTAS, 1, 1):
            raise ValueError("NCU grid geometry changed")
        if _dimension(row["Block Size"], "NCU block") != (
                THREADS_PER_CTA, 1, 1):
            raise ValueError("NCU block geometry changed")
        if int(_number(
                row["launch__shared_mem_per_block_dynamic"],
                "dynamic shared bytes")) != DYNAMIC_SHARED_BYTES:
            raise ValueError("NCU dynamic shared-memory reservation changed")
        if int(_number(
                row["launch__registers_per_thread"],
                "registers per thread")) != EXPECTED_REGISTERS:
            raise ValueError("NCU register count changed")
        executed = int(_number(
            row["sm__sass_inst_executed_op_ldgsts.sum"],
            "executed LDGSTS instructions"))
        if executed != expected_instructions:
            raise ValueError(
                "dynamic LDGSTS instruction count does not close exactly")
        xbar_bytes = int(_number(
            row[
                "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_"
                "ldgsts_cache_bypass.sum"],
            "LDGSTS Xbar bytes"))
        if xbar_bytes != expected_xbar_bytes:
            raise ValueError("dynamic LDGSTS byte count does not close exactly")
        duration_ns = _number(row["gpu__time_duration.sum"], "duration")
        clock_hz = _number(
            row["sm__cycles_elapsed.avg.per_second"], "SM clock")
        cycles.append(duration_ns * clock_hz / 1e9)
    return statistics.median(cycles)


def _affine(points):
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = math.fsum(
        (x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise ValueError("affine fit requires distinct panel depths")
    slope = math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    if intercept <= 0 or slope <= 0:
        raise ValueError("finite LDGSTS affine service must be positive")
    return intercept, slope


def _holdout(train, validation):
    intercept, slope = _affine(train)
    rows = []
    for panels, observed in validation:
        predicted = intercept + slope * panels
        rows.append({
            "panels": panels,
            "observed_cycles": observed,
            "predicted_cycles": predicted,
            "absolute_error_pct": abs(predicted - observed) / observed * 100,
        })
    return {
        "training_panels": [point[0] for point in train],
        "validation": rows,
        "maximum_absolute_error_pct": max(
            row["absolute_error_pct"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    input_dir = arguments.input_dir.resolve()
    ptxas = (input_dir / "ptxas.txt").read_text(encoding="utf-8")
    sass = (input_dir / "probe.sass").read_text(encoding="utf-8")
    if "Used 79 registers" not in ptxas:
        raise ValueError("ptxas register count changed")
    if ("0 bytes spill stores" not in ptxas
            or "0 bytes spill loads" not in ptxas
            or "0 bytes stack frame" not in ptxas):
        raise ValueError("probe acquired stack or spill traffic")
    static_ldgsts = len(re.findall(r"\bLDGSTS\.E\.BYPASS\.128\b", sass))
    if static_ldgsts != EXPECTED_STATIC_LDGSTS:
        raise ValueError("static LDGSTS instruction count changed")

    cells = []
    for repeats in REPEATS:
        cycles = _read_ncu_rows(input_dir / f"final_r{repeats}.csv", repeats)
        transactions = (
            GRID_CTAS * PANEL_BYTES_PER_CTA * repeats / TRANSACTION_BYTES)
        cells.append({
            "panels": repeats,
            "median_cycles": cycles,
            "initial_32b_transactions_per_cycle": transactions / cycles,
        })
    recurrent = tuple(
        (cell["panels"], cell["median_cycles"])
        for cell in cells if cell["panels"] >= 2)
    affine_startup, affine_panel = _affine(recurrent)
    holdouts = (
        _holdout(recurrent[0::2], recurrent[1::2]),
        _holdout(recurrent[1::2], recurrent[0::2]),
    )
    maximum_holdout_ape = max(
        holdout["maximum_absolute_error_pct"] for holdout in holdouts)
    gates = {
        "register_count_is_79": True,
        "no_stack_or_spill": True,
        "static_ldgsts_bypass_128_count_is_32": True,
        "dynamic_instruction_and_byte_counters_close_exactly": True,
        "odd_even_holdout_max_ape_at_most_15_pct": (
            maximum_holdout_ape <= MAX_HOLDOUT_APE_PCT),
    }
    result = {
        "schema": "a800_fa2_resident_ldgsts_finite_report_v1",
        "status": "pass" if all(gates.values()) else "fail",
        "hardware_id": "nvidia_a800",
        "mechanism": "two_resident_cta_finite_ldgsts_service",
        "production_or_evaluation_duration_consumed": False,
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "capture_source_sha256": _sha256(input_dir / "source_snapshot.cu"),
        "ptxas_sha256": _sha256(input_dir / "ptxas.txt"),
        "sass_sha256": _sha256(input_dir / "probe.sass"),
        "resource_contract": {
            "threads_per_cta": THREADS_PER_CTA,
            "grid_ctas": GRID_CTAS,
            "physical_sm_count": 108,
            "resident_ctas_per_sm": 2,
            "dynamic_shared_bytes_per_cta": DYNAMIC_SHARED_BYTES,
            "panel_bytes_per_cta": PANEL_BYTES_PER_CTA,
            "initial_transaction_bytes": TRANSACTION_BYTES,
        },
        "cells": cells,
        "single_panel_cycles": cells[0]["median_cycles"],
        "recurrent_affine_startup_cycles": affine_startup,
        "recurrent_affine_panel_cycles": affine_panel,
        "recurrent_domain_panels": [2, 64],
        "holdouts": holdouts,
        "maximum_holdout_absolute_error_pct": maximum_holdout_ape,
        "gates": gates,
        "allowed_conclusion": (
            "For a full 216-CTA SM80 cohort with exactly two resident "
            "128-thread CTAs per SM and one 64-KiB LDGSTS panel per CTA "
            "recurrence, the one-panel endpoint and the recurrent 2..64 "
            "panel affine service bound finite fill, steady, and drain."),
        "excluded_conclusions": [
            "a complete production FlashAttention duration",
            "resident counts other than two",
            "partial cohorts or panel depths above 64",
            "MMA, softmax, output, cache-miss, or page-table service",
        ],
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "affine_panel_cycles": affine_panel,
        "affine_startup_cycles": affine_startup,
        "maximum_holdout_absolute_error_pct": maximum_holdout_ape,
        "output": str(arguments.output.resolve()),
        "single_panel_cycles": cells[0]["median_cycles"],
        "status": result["status"],
    }, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
