#!/usr/bin/env python3
"""Validate and publish the H200 LDG128 load-to-operands-ready mechanism.

H200 owner copy of microbench/nvidia_a800/memory/lsu/
fa2_oaccum_ldg128_load_use_report.py.  Derivation, train/validation split,
invariant-then-segmented model choice and thresholds are unchanged.
Hardware-coupling changes only: grids 1/32/108 -> 1/32/132, valid %smid
bound 108 -> 132, capture/report schema, device and hardware_id names; the
CUDA-event plausibility check converts ns to cycles with the clock-gated
H200 SM clock (1980 MHz) instead of the instantaneous nvidia-smi before/after
reading (which reads the 345 MHz idle clock outside a live context); the
NCU cache-tier audit is blocked on H200 (ncu not installed) and is replaced
by the recorded blocked audit plus a clock64 tier-separation gate
(dram_cg first-load readiness > l2_warm_cg first-load readiness).
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE_NAME = "fa2_oaccum_ldg128_load_use.cu"
RUNNER_NAME = "fa2_oaccum_ldg128_load_use_runner.py"
REPORT_NAME = "fa2_oaccum_ldg128_load_use_report.py"
EXPECTED_SM_COUNT = 132
GATED_SM_CLOCK_MHZ = 1980.0
THREADS_PER_CTA = 128
GRIDS = (1, 32, 132)
REPEATS = (1, 2, 3, 4, 6, 8, 11, 16, 22)
MODES = ("empty", "fma_only", "ldg128_fma")
TIERS = ("l2_warm_cg", "dram_cg")
TRAIN_TRIALS = (0, 1)
VALIDATION_TRIALS = (2,)
SAMPLES = 9
MAX_VALIDATION_MAPE_PCT = 15.0
MAX_VALIDATION_APE_PCT = 25.0
MAX_TRAIN_GRID_REPEAT_SPREAD_PCT = 20.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(text, name):
    if not isinstance(text, str) or not text.isdigit():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(text)


def _require_positive_float(text, name):
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _read_manifest(path, raw_path, ncu_audit_path):
    capture_root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("capture manifest must be a mapping")
    expected_fields = {
        "schema", "baseline_commit", "source_sha256", "runner_sha256",
        "report_sha256", "binary_sha256", "sass_sha256", "raw_sha256",
        "ncu_audit_sha256", "device", "grids", "repeats", "modes",
        "tiers", "trials", "samples_per_cell", "timer_domain",
        "cuda_event_role", "production_or_evaluation_duration_consumed",
        "production_parameter_published", "clock_gate",
    }
    if set(manifest) != expected_fields:
        raise ValueError("capture manifest fields changed")
    if manifest["schema"] != "h200_fa2_oaccum_ldg128_load_use_capture_v1":
        raise ValueError("capture manifest schema changed")
    if manifest["production_or_evaluation_duration_consumed"]:
        raise ValueError("capture consumed a production/evaluation duration")
    if manifest["production_parameter_published"]:
        raise ValueError("capture already published a production parameter")
    if tuple(manifest["grids"]) != GRIDS:
        raise ValueError("capture grid matrix changed")
    if tuple(manifest["repeats"]) != REPEATS:
        raise ValueError("capture repeat matrix changed")
    if tuple(manifest["modes"]) != MODES or tuple(manifest["tiers"]) != TIERS:
        raise ValueError("capture mechanism matrix changed")
    if manifest["trials"] != len(TRAIN_TRIALS + VALIDATION_TRIALS):
        raise ValueError("capture trial count changed")
    if manifest["samples_per_cell"] != SAMPLES:
        raise ValueError("capture sample count changed")
    if manifest["timer_domain"] != "clock64_same_cta_envelope_cycles":
        raise ValueError("capture timing domain changed")
    if manifest["cuda_event_role"] != "validation_only_not_forward_parameter":
        raise ValueError("CUDA event role changed")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("raw capture SHA256 mismatch")
    if manifest["ncu_audit_sha256"] != _sha256(ncu_audit_path):
        raise ValueError("NCU audit SHA256 mismatch")
    snapshots = {
        "source_sha256": capture_root / "source_snapshot" / SOURCE_NAME,
        "runner_sha256": capture_root / "source_snapshot" / RUNNER_NAME,
        "report_sha256": capture_root / "source_snapshot" / REPORT_NAME,
    }
    for field, snapshot in snapshots.items():
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        if manifest[field] != _sha256(snapshot):
            raise ValueError(f"capture {field} does not match source snapshot")
    device = manifest["device"]
    if not isinstance(device, dict):
        raise TypeError("capture device must be a mapping")
    if device["name"] != "NVIDIA H200":
        raise ValueError("capture device changed")
    if manifest["clock_gate"]["passes"][-1]["rejected"] != 0:
        raise ValueError("capture has cells rejected by the clock gate")
    return manifest


def _expected_cell_keys():
    return {
        (trial, tier, mode, grid, repeats)
        for trial in TRAIN_TRIALS + VALIDATION_TRIALS
        for tier in TIERS
        for mode in MODES
        for grid in GRIDS
        for repeats in REPEATS
    }


def _read_rows(path):
    expected_fields = (
        "trial", "tier", "mode", "kernel", "grid", "repeats", "sample",
        "block", "sm_id", "cycles", "checksum", "event_ns",
        "sm_clock_before_mhz", "sm_clock_after_mhz")
    rows = []
    identities = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError("raw CSV fields changed")
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": _require_int(raw["trial"], "trial"),
                "tier": raw["tier"],
                "mode": raw["mode"],
                "kernel": raw["kernel"],
                "grid": _require_int(raw["grid"], "grid"),
                "repeats": _require_int(raw["repeats"], "repeats"),
                "sample": _require_int(raw["sample"], "sample"),
                "block": _require_int(raw["block"], "block"),
                "sm_id": _require_int(raw["sm_id"], "sm_id"),
                "cycles": _require_int(raw["cycles"], "cycles"),
                "checksum": _require_int(raw["checksum"], "checksum"),
                "event_ns": _require_positive_float(raw["event_ns"], "event_ns"),
                "sm_clock_before_mhz": _require_positive_float(
                    raw["sm_clock_before_mhz"], "sm_clock_before_mhz"),
                "sm_clock_after_mhz": _require_positive_float(
                    raw["sm_clock_after_mhz"], "sm_clock_after_mhz"),
            }
            if row["tier"] not in TIERS or row["mode"] not in MODES:
                raise ValueError(f"raw row {ordinal} has an unknown mechanism")
            if row["grid"] not in GRIDS or row["repeats"] not in REPEATS:
                raise ValueError(f"raw row {ordinal} is outside frozen matrix")
            if row["trial"] not in TRAIN_TRIALS + VALIDATION_TRIALS:
                raise ValueError(f"raw row {ordinal} has an unknown trial")
            if not (0 <= row["sample"] < SAMPLES and 0 <= row["block"] < row["grid"]):
                raise ValueError(f"raw row {ordinal} has an invalid index")
            if row["sm_id"] >= EXPECTED_SM_COUNT or row["cycles"] <= 0:
                raise ValueError(f"raw row {ordinal} has invalid CTA timing")
            if row["mode"] == "ldg128_fma":
                expected_kernel = f"fa2_oaccum_ldg128_fma_{row['tier']}"
            elif row["mode"] == "fma_only":
                expected_kernel = "fa2_oaccum_fma_only"
            elif row["mode"] == "empty":
                expected_kernel = "fa2_oaccum_empty"
            else:
                raise AssertionError("unhandled validated mode")
            if row["kernel"] != expected_kernel:
                raise ValueError(f"raw row {ordinal} kernel identity changed")
            identity = (
                row["trial"], row["tier"], row["mode"], row["grid"],
                row["repeats"], row["sample"], row["block"])
            if identity in identities:
                raise ValueError(f"raw row {ordinal} repeats an identity")
            identities.add(identity)
            rows.append(row)
    expected_rows = sum(
        SAMPLES * grid for _, _, _, grid, _ in _expected_cell_keys())
    if len(rows) != expected_rows:
        raise ValueError("raw capture row matrix is incomplete")
    return tuple(rows)


def _cell_summaries(rows):
    grouped = {}
    smids = {}
    events = {}
    for row in rows:
        key = (row["trial"], row["tier"], row["mode"], row["grid"],
               row["repeats"])
        sample_key = key + (row["sample"],)
        if key not in grouped:
            grouped[key] = []
            smids[key] = {}
            events[key] = {}
        grouped[key].append(row["cycles"])
        if row["sample"] not in smids[key]:
            smids[key][row["sample"]] = set()
            events[key][row["sample"]] = row["event_ns"]
        elif events[key][row["sample"]] != row["event_ns"]:
            raise ValueError("CTA rows disagree on their CUDA event interval")
        if row["sm_id"] in smids[key][row["sample"]]:
            raise ValueError("timed CTA sample has duplicate SM ID")
        smids[key][row["sample"]].add(row["sm_id"])
    if set(grouped) != _expected_cell_keys():
        raise ValueError("raw capture cell matrix is incomplete")
    summaries = {}
    for key in sorted(grouped):
        trial, tier, mode, grid, repeats = key
        values = grouped[key]
        if len(values) != SAMPLES * grid:
            raise ValueError("mechanism cell has an incomplete CTA matrix")
        for sample in range(SAMPLES):
            if len(smids[key][sample]) != grid:
                raise ValueError("timed CTA sample did not occupy unique SMs")
        # CUDA-event data is a validation channel only.  Compare it with the
        # longest CTA envelope only at the largest LDG+FMA workload, where
        # fixed kernel boundary cost cannot dwarf the same-CTA interval.
        # Empty and short kernels intentionally measure only the in-kernel
        # clock64 baseline and are therefore not comparable to CUDA events.
        longest_cycles = max(values)
        if mode == "ldg128_fma" and repeats == max(REPEATS):
            for event_ns in events[key].values():
                event_cycles = event_ns * GATED_SM_CLOCK_MHZ / 1000.0
                ratio = event_cycles / longest_cycles
                if not 0.25 <= ratio <= 4.0:
                    raise ValueError(
                        "CUDA event disagrees with the long LDG128 clock64 "
                        "envelope")
        summaries[key] = {
            "trial": trial,
            "tier": tier,
            "mode": mode,
            "grid": grid,
            "repeats": repeats,
            "median_cycles": statistics.median(values),
            "p10_cycles": statistics.quantiles(
                values, n=10, method="inclusive")[0],
            "p90_cycles": statistics.quantiles(
                values, n=10, method="inclusive")[8],
        }
    return summaries


def _ready_cells(summaries):
    values = []
    for trial in TRAIN_TRIALS + VALIDATION_TRIALS:
        for tier in TIERS:
            for grid in GRIDS:
                for repeats in REPEATS:
                    base = (trial, tier, "empty", grid, repeats)
                    fma = (trial, tier, "fma_only", grid, repeats)
                    ldg = (trial, tier, "ldg128_fma", grid, repeats)
                    empty_cycles = summaries[base]["median_cycles"]
                    fma_cycles = summaries[fma]["median_cycles"]
                    ldg_cycles = summaries[ldg]["median_cycles"]
                    # Empty is retained only to expose the timer/loop baseline.
                    # It is not a subtractand: address arithmetic in the empty
                    # path can serialize differently from the register-only
                    # FMA path for short loops.  The physical differential is
                    # the matched LDG128+FMA path minus FMA-only below.
                    total_ready_cycles = ldg_cycles - fma_cycles
                    ready_cycles = total_ready_cycles / repeats
                    if (not math.isfinite(total_ready_cycles)
                            or total_ready_cycles <= 0
                            or not math.isfinite(ready_cycles)
                            or ready_cycles <= 0):
                        raise ValueError("LDG128 readiness differential is non-positive")
                    values.append({
                        "trial": trial,
                        "tier": tier,
                        "grid": grid,
                        "repeats": repeats,
                        "total_load_to_operands_ready_cycles": total_ready_cycles,
                        "load_to_operands_ready_cycles": ready_cycles,
                    })
    return tuple(values)


def _entries(ready_cells):
    entries = []
    for tier in TIERS:
        train_rows = [
            row for row in ready_cells
            if row["tier"] == tier and row["trial"] in TRAIN_TRIALS]
        validation_rows = [
            row for row in ready_cells
            if row["tier"] == tier and row["trial"] in VALIDATION_TRIALS]
        train = [row["load_to_operands_ready_cycles"] for row in train_rows]
        validation = [
            row["load_to_operands_ready_cycles"]
            for row in validation_rows]
        expected_train_count = len(TRAIN_TRIALS) * len(GRIDS) * len(REPEATS)
        expected_validation_count = (
            len(VALIDATION_TRIALS) * len(GRIDS) * len(REPEATS))
        if len(train) != expected_train_count or len(validation) != expected_validation_count:
            raise ValueError("mechanism train/validation partition is incomplete")
        value = statistics.median(train)
        spread_pct = 100.0 * (max(train) - min(train)) / value
        validation_errors = [100.0 * abs(value - observed) / observed
                             for observed in validation]
        validation_mape = statistics.fmean(validation_errors)
        validation_max = max(validation_errors)
        if (spread_pct <= MAX_TRAIN_GRID_REPEAT_SPREAD_PCT
                and validation_mape <= MAX_VALIDATION_MAPE_PCT
                and validation_max <= MAX_VALIDATION_APE_PCT):
            entries.append({
                "cache_tier": tier,
                "model": "grid_repeat_invariant_per_load_v1",
                "load_to_operands_ready_cycles": value,
                "training_cells": len(train),
                "validation_cells": len(validation),
                "training_grid_repeat_spread_pct": spread_pct,
                "validation_mape_pct": validation_mape,
                "validation_max_ape_pct": validation_max,
            })
            continue

        # The source loop may have one first load and a different steady
        # interval without admitting a per-split table.  This is an isolated
        # mechanism fit over fixed generic repeats only; trial 2 remains a
        # frozen validation set and no production duration is consulted.
        x_values = [row["repeats"] - 1.0 for row in train_rows]
        y_values = [row["total_load_to_operands_ready_cycles"]
                    for row in train_rows]
        x_mean = statistics.fmean(x_values)
        y_mean = statistics.fmean(y_values)
        denominator = math.fsum((value - x_mean) ** 2 for value in x_values)
        if denominator <= 0.0:
            raise ValueError("segmented readiness fit has no repeat variation")
        additional = math.fsum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values)) / denominator
        first = y_mean - additional * x_mean
        if (not math.isfinite(first) or not math.isfinite(additional)
                or first <= 0.0 or additional <= 0.0):
            raise ValueError("segmented readiness coefficients must be positive")

        def predict_total(row):
            return first + (row["repeats"] - 1.0) * additional

        train_errors = [
            100.0 * abs(predict_total(row)
                        - row["total_load_to_operands_ready_cycles"])
            / row["total_load_to_operands_ready_cycles"]
            for row in train_rows]
        segmented_validation_errors = [
            100.0 * abs(predict_total(row)
                        - row["total_load_to_operands_ready_cycles"])
            / row["total_load_to_operands_ready_cycles"]
            for row in validation_rows]
        segmented_train_mape = statistics.fmean(train_errors)
        segmented_validation_mape = statistics.fmean(
            segmented_validation_errors)
        segmented_validation_max = max(segmented_validation_errors)
        if segmented_validation_mape > MAX_VALIDATION_MAPE_PCT:
            raise ValueError(
                f"{tier} segmented validation MAPE is too high: "
                f"{segmented_validation_mape:.2f}%")
        if segmented_validation_max > MAX_VALIDATION_APE_PCT:
            raise ValueError(
                f"{tier} segmented validation max APE is too high: "
                f"{segmented_validation_max:.2f}%")
        entries.append({
            "cache_tier": tier,
            "model": "first_plus_additional_load_interval_v1",
            "first_load_ready_cycles": first,
            "additional_load_interval_cycles": additional,
            "training_cells": len(train_rows),
            "validation_cells": len(validation_rows),
            "invariant_training_grid_repeat_spread_pct": spread_pct,
            "invariant_validation_mape_pct": validation_mape,
            "invariant_validation_max_ape_pct": validation_max,
            "segmented_training_mape_pct": segmented_train_mape,
            "segmented_validation_mape_pct": segmented_validation_mape,
            "segmented_validation_max_ape_pct": segmented_validation_max,
        })
    return tuple(entries)


def _read_ncu_audit(path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit["status"] != "blocked":
        raise ValueError("H200 NCU audit record must be the blocked record")
    for field in ("reason", "requested_metrics", "alternative"):
        if not audit[field]:
            raise ValueError(f"blocked NCU audit lacks {field}")
    return audit


def _tier_separation(entries):
    first = {}
    for entry in entries:
        if entry["model"] == "first_plus_additional_load_interval_v1":
            first[entry["cache_tier"]] = entry["first_load_ready_cycles"]
        elif entry["model"] == "grid_repeat_invariant_per_load_v1":
            first[entry["cache_tier"]] = entry[
                "load_to_operands_ready_cycles"]
        else:
            raise ValueError(f"unknown readiness model {entry['model']!r}")
    if not first["dram_cg"] > first["l2_warm_cg"]:
        raise ValueError(
            "clock64 data do not separate the DRAM and L2-warm tiers")
    return {
        "dram_cg_first_load_ready_cycles": first["dram_cg"],
        "l2_warm_cg_first_load_ready_cycles": first["l2_warm_cg"],
        "dram_over_l2_ratio": first["dram_cg"] / first["l2_warm_cg"],
        "passed": True,
    }


def _candidate(manifest, summaries, ready_cells, entries, ncu_audit):
    return {
        "schema": "nvidia_h200_fa2_oaccum_ldg128_load_use_candidate_v1",
        "status": "candidate",
        "hardware_id": "nvidia_h200",
        "model": "isolated_ldg128_fp32x4_load_to_operands_ready_v1",
        "cycle_unit": "sm_cycles_per_128_thread_cta_iteration",
        "timer_domain": "clock64_same_cta_envelope_differential",
        "cuda_event_role": "validation_only_not_forward_parameter",
        "threads_per_cta": THREADS_PER_CTA,
        "load_bytes_per_lane": 16,
        "values_per_lane": 4,
        "dependent_fp32_fma_per_lane": 4,
        "grids_validated": list(GRIDS),
        "repeats_validated": list(REPEATS),
        "train_trials": list(TRAIN_TRIALS),
        "validation_trials": list(VALIDATION_TRIALS),
        "cache_entries": list(entries),
        "cells": list(summaries.values()),
        "readiness_cells": list(ready_cells),
        "ncu_cache_traffic_audit": {
            **ncu_audit, "tier_separation": _tier_separation(entries)},
        "source_sha256": manifest["source_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "capture_report_sha256": manifest["report_sha256"],
        "raw_sha256": manifest["raw_sha256"],
        "production_or_evaluation_duration_consumed": False,
        "derivation": (
            "For each fixed cache tier, grid, repeat count, and trial, use "
            "median same-CTA clock64 envelope cycles and compute "
            "(LDG128->four-FMA minus matched four-FMA-only)/repeats.  "
            "Trials 0-1 define one cache-tier mechanism constant; trial 2 "
            "is frozen validation.  CUDA events are validation evidence "
            "only; the NCU counter audit is blocked on H200."),
        "allowed_forward_composition": (
            "Apply this only as max(existing source-specific global-read "
            "completion time, readiness_cycles/sm_clock_hz) on each Oaccum "
            "load stage.  Keep the existing FMA stage, because FMA-only was "
            "subtracted before publication."),
        "forbidden_forward_composition": [
            "add readiness cycles to a memory stage",
            "add a split-count latency correction",
            "consume CUDA-event or production/evaluation duration",
            "select an Oaccum cache tier from aggregate unrelated traffic",
        ],
    }


def _self_test():
    ready = []
    for tier in TIERS:
        if tier == "l2_warm_cg":
            base = 230.0
        elif tier == "dram_cg":
            base = 440.0
        else:
            raise AssertionError("unhandled frozen tier")
        for trial in TRAIN_TRIALS + VALIDATION_TRIALS:
            for grid in GRIDS:
                for repeats in REPEATS:
                    ready.append({
                        "trial": trial,
                        "tier": tier,
                        "grid": grid,
                        "repeats": repeats,
                        "load_to_operands_ready_cycles": (
                            base + 0.4 * trial + 0.1 * (grid // 32)
                            + 0.02 * repeats),
                    })
    entries = _entries(tuple(ready))
    if len(entries) != len(TIERS):
        raise AssertionError("synthetic report did not publish both cache tiers")
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ncu-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        if any(value is not None for value in (
                arguments.input, arguments.manifest, arguments.ncu_audit,
                arguments.output)):
            raise ValueError("self-test accepts no capture paths")
        print(json.dumps({"entries": _self_test()}, indent=2, sort_keys=True))
        return
    if any(value is None for value in (
            arguments.input, arguments.manifest, arguments.ncu_audit,
            arguments.output)):
        raise ValueError("input, manifest, ncu-audit, and output are required")
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    manifest = _read_manifest(
        arguments.manifest, arguments.input, arguments.ncu_audit)
    rows = _read_rows(arguments.input)
    summaries = _cell_summaries(rows)
    ready_cells = _ready_cells(summaries)
    entries = _entries(ready_cells)
    ncu_audit = _read_ncu_audit(arguments.ncu_audit)
    candidate = _candidate(manifest, summaries, ready_cells, entries, ncu_audit)
    arguments.output.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(candidate["cache_entries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
