#!/usr/bin/env python3
"""Validate the H200 two-resident-CTA finite LDGSTS captures and publish.

Owner copy of microbench/nvidia_a800/memory/l1/fa2_resident_ldgsts_finite_report.py.
The one-panel endpoint, the recurrent 2..64-panel affine least-squares fit,
the odd/even held-out validation and its 15% gate, the static SASS gate and
the no-spill gate are unchanged.  Hardware coupling:

* per-launch cycles are the Nsight Systems kernel duration x the measured
  1.98-GHz SM clock (the A800 report used NCU duration x NCU clock; NCU is
  unavailable on the H200 host);
* the NCU dynamic instruction/byte counter-closure gate cannot be evaluated
  and is recorded as ``null`` (not verifiable) rather than true;
* the register-count gate records the sm_90 ptxas count of this build
  (the A800 sm_80 build used 79) instead of requiring 79;
* the A800 single capture becomes two independent capture processes on
  GPU 2; each cell's median pools the seven samples of both captures;
* grid 264 = two resident CTAs on each of 132 H200 SMs.
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
    SM_CLOCK_HZ,
    SM_COUNT,
    _clock_summary,
    _relative,
    _sha256,
    _write_reference,
)


SOURCE_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite.cu")
RUNNER_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite_runner.py")
REPEATS = (1, 2, 4, 8, 16, 32, 64)
SAMPLES = 7
GRID_CTAS = 2 * SM_COUNT
THREADS_PER_CTA = 128
DYNAMIC_SHARED_BYTES = 77824
PANEL_BYTES_PER_CTA = 65536
TRANSACTION_BYTES = 32
EXPECTED_STATIC_LDGSTS = 32
MAX_HOLDOUT_APE_PCT = 15.0


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


def _static_contract(capture_dir):
    ptxas = (capture_dir / "ptxas.txt").read_text(encoding="utf-8")
    sass = (capture_dir / "probe.sass").read_text(encoding="utf-8")
    registers = re.findall(r"Used ([0-9]+) registers", ptxas)
    if len(registers) != 1:
        raise ValueError("ptxas must report one kernel register count")
    if ("0 bytes spill stores" not in ptxas
            or "0 bytes spill loads" not in ptxas
            or "0 bytes stack frame" not in ptxas):
        raise ValueError("probe acquired stack or spill traffic")
    static_ldgsts = len(re.findall(r"\bLDGSTS\.E\.BYPASS\.128\b", sass))
    if static_ldgsts != EXPECTED_STATIC_LDGSTS:
        raise ValueError("static LDGSTS instruction count changed")
    return int(registers[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path,
                        action="append")
    arguments = parser.parse_args()
    if len(arguments.input_dir) < 2:
        raise ValueError("publication needs two independent captures")
    durations = {repeats: [] for repeats in REPEATS}
    manifests = []
    register_counts = set()
    for input_dir in arguments.input_dir:
        capture_dir = input_dir.resolve()
        manifest_path = capture_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["grid_ctas"] != GRID_CTAS
                or manifest["dynamic_shared_bytes"] != DYNAMIC_SHARED_BYTES):
            raise ValueError("capture resource contract changed")
        register_counts.add(_static_contract(capture_dir))
        for cell in manifest["cells"]:
            if len(cell["duration_ns"]) != SAMPLES:
                raise ValueError("each cell must hold seven launches")
            durations[cell["repeats"]].extend(cell["duration_ns"])
        manifests.append((capture_dir, manifest_path, manifest))
    if len(register_counts) != 1:
        raise ValueError("captures used different builds")
    registers = register_counts.pop()

    cells = []
    for repeats in REPEATS:
        cycles = statistics.median(durations[repeats]) * SM_CLOCK_HZ / 1e9
        transactions = (
            GRID_CTAS * PANEL_BYTES_PER_CTA * repeats / TRANSACTION_BYTES)
        cells.append({
            "panels": repeats,
            "median_cycles": cycles,
            "median_duration_ns": statistics.median(durations[repeats]),
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
        "register_count_is_79": None,
        "register_count_of_this_sm90_build": registers,
        "no_stack_or_spill": True,
        "static_ldgsts_bypass_128_count_is_32": True,
        "dynamic_instruction_and_byte_counters_close_exactly": None,
        "odd_even_holdout_max_ape_at_most_15_pct": (
            maximum_holdout_ape <= MAX_HOLDOUT_APE_PCT),
    }
    if not gates["odd_even_holdout_max_ape_at_most_15_pct"]:
        raise SystemExit(
            f"held-out gate failed: {maximum_holdout_ape}% > 15%")
    first = manifests[0][2]
    reference = {
        "schema": "nvidia_h200_fa2_resident_ldgsts_finite_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "evidence_kind": "independent_two_resident_cta_finite_ldgsts_microbenchmark",
        "source_capture": [
            _relative(capture_dir) for capture_dir, _, _ in manifests],
        "source_baseline_commit": sorted({
            manifest["source_commit"] for _, _, manifest in manifests}),
        "physical_device_index": first["device"]["index"],
        "device_uuid": first["device"]["uuid"],
        "resource_contract": {
            "threads_per_cta": THREADS_PER_CTA,
            "dynamic_shared_bytes_per_cta": DYNAMIC_SHARED_BYTES,
            "registers_per_thread": registers,
            "resident_ctas_per_sm": 2,
            "physical_sm_count": SM_COUNT,
            "full_chip_ctas": GRID_CTAS,
            "panel_bytes_per_cta": PANEL_BYTES_PER_CTA,
            "initial_transaction_bytes": TRANSACTION_BYTES,
            "static_ldgsts_bypass_128_instructions": EXPECTED_STATIC_LDGSTS,
            "stack_frame_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
        },
        "single_panel_cycles": cells[0]["median_cycles"],
        "recurrent_affine_startup_cycles": affine_startup,
        "recurrent_affine_panel_cycles": affine_panel,
        "recurrent_domain_panels": [2, 64],
        "maximum_odd_even_holdout_absolute_error_pct": maximum_holdout_ape,
        "publication_gates": gates,
        "artifact_sha256": {
            "source": _sha256(SOURCE_PATH),
            "runner": _sha256(RUNNER_PATH),
            "reporter": _sha256(Path(__file__)),
            "binary": sorted({
                manifest["binary_sha256"] for _, _, manifest in manifests}),
            "ptxas": sorted({
                manifest["ptxas_sha256"] for _, _, manifest in manifests}),
            "sass": sorted({
                manifest["sass_sha256"] for _, _, manifest in manifests}),
            "manifests": [
                _sha256(manifest_path) for _, manifest_path, _ in manifests],
            # This reference is itself the reporter output.
            "report": None,
        },
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            f"For a full {GRID_CTAS}-CTA H200 cohort with exactly two resident "
            "128-thread CTAs per SM and one 64-KiB LDGSTS panel per CTA "
            "recurrence, the one-panel endpoint and recurrent 2..64-panel "
            "affine service bound finite fill, steady, and drain."),
        "excluded_conclusions": [
            "a complete production FlashAttention duration",
            "resident counts other than two",
            "partial cohorts or panel depths above 64",
            "MMA, softmax, output, cache-miss, or page-table service",
        ],
        "provenance": {
            "source": _relative(SOURCE_PATH),
            "runner": _relative(RUNNER_PATH),
            "reporter": _relative(Path(__file__)),
            "timer": "nsys 2025.6.1 cuda_gpu_trace kernel duration",
            "sm_clock_hz": SM_CLOCK_HZ,
            "cells": cells,
            "holdouts": holdouts,
            "samples_per_cell_pooled": SAMPLES * len(manifests),
            "clock_gate": _clock_summary([
                cell["attempts"] for _, _, manifest in manifests
                for cell in manifest["cells"]]),
            "ncu_note": (
                "Nsight Compute is unavailable on the H200 host: NCU duration "
                "is replaced by the nsys kernel duration at the gated 1.98-GHz "
                "clock, and the dynamic counter-closure gate is not evaluated."),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("fa2_resident_ldgsts_finite_sm90.json", reference)


if __name__ == "__main__":
    main()
