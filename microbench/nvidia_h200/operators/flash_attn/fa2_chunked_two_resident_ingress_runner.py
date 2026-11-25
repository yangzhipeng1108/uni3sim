#!/usr/bin/env python3
"""Capture and publish the H200 two-resident chunked-prefill K/V ingress.

The A800 probes fa2_chunked_exact_two_resident_ingress.cu (wrapper: 82,944-B
reservation, two resident CTAs, one shared buffer) and
fa2_chunked_one_resident_ingress.cu (the included ingress fragment) had no
runner.  The A800 reference fa2_chunked_q8192_two_resident_ingress_sm80.json
records the procedure: two static source profiles (1 = skew Q1024/Q4096,
2 = uniform Q256..Q1019) at grids 2048 and 4096, 10 warm-up launches, 31
samples per cell, %globaltimer makespan per sample; payload =
total_rounds x 64 KiB; the selected service is the lower 4096-CTA direct
median rate across the two profiles; gates are <= 5% span in every cell and
the two 4096-CTA profiles within 5%.

This runner scripts that procedure.  Hardware coupling: the H200 owner copy
of the probe (132 SMs, sm_90; the 82,944-B class still admits exactly two
resident CTAs, enforced at runtime), the measured 1.98-GHz SM clock for
per-cycle values instead of the 1.14-GHz model clock, H200 GPU 2 only, 50-ms
SM-clock gate per probe process (helpers from
microbench/nvidia_h200/memory/l2/staged_ldgsts_l2_runner.py), and two
independent capture processes whose samples are pooled per cell.  The NCU
counter-closure gate is recorded as null (Nsight Compute is unavailable on
the H200 host); the probe's own total_rounds is checked instead.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "memory/l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
    SM_CLOCK_HZ,
    SM_COUNT,
    _clock_summary,
    _compile,
    _device,
    _dump_sass,
    _gated_run,
    _load_capture,
    _new_output,
    _relative,
    _sha256,
    _wait_for_idle,
    _write_reference,
)


WRAPPER = Path(__file__).with_name("fa2_chunked_exact_two_resident_ingress.cu")
INGRESS = Path(__file__).with_name("fa2_chunked_one_resident_ingress.cu")
RUNNER = Path(__file__).resolve()
PANEL_BYTES = 65536
DYNAMIC_SHARED_BYTES = 82944
SAMPLES = 31
WARMUP_LAUNCHES = 10
GRIDS = (2048, 4096)
PROFILES = {1: "skew_q1024_q4096", 2: "uniform_q256_to_q1019"}
LINE = re.compile(
    r"^sample=(\d+) profile=(\S+) grid=(\d+) resident=(\d+) "
    r"source_shared_bytes=(\d+) panel_bytes=65536 total_rounds=(\d+) "
    r"checksum=(\d+) makespan_ns=(\d+)$")


def _parse(stdout, profile, grid):
    makespans = []
    rounds = set()
    for line in stdout.splitlines():
        match = LINE.match(line)
        if not match:
            raise ValueError(f"unexpected probe line: {line}")
        if (int(match.group(1)) != len(makespans)
                or match.group(2) != PROFILES[profile]
                or int(match.group(3)) != grid or int(match.group(4)) != 2
                or int(match.group(5)) != DYNAMIC_SHARED_BYTES):
            raise ValueError(f"probe line violates contract: {line}")
        rounds.add(int(match.group(6)))
        makespans.append(int(match.group(8)))
    if len(makespans) != SAMPLES or len(rounds) != 1:
        raise ValueError("probe sample set changed")
    return makespans, rounds.pop()


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "fa2_chunked_exact_two_resident_ingress"
    compile_command, ptxas = _compile(WRAPPER, binary)
    (output / "ptxas.txt").write_text(ptxas, encoding="utf-8")
    sass = _dump_sass(binary, output / "probe.sass")
    static_ldgsts = len(re.findall(r"\bLDGSTS\.E\.BYPASS\.128\b", sass))
    if (static_ldgsts != 32 or "0 bytes spill stores" not in ptxas
            or "0 bytes stack frame" not in ptxas):
        raise ValueError("compiled probe violates the static contract")
    registers = int(re.search(r"Used ([0-9]+) registers", ptxas).group(1))
    cells = {}
    for profile, profile_name in PROFILES.items():
        for grid in GRIDS:
            command = [str(binary), str(SAMPLES), str(grid), str(profile)]
            result, attempts = _gated_run(
                command, device=args.device,
                log_stem=output / f"{profile_name}_grid{grid}")
            makespans, rounds = _parse(result.stdout, profile, grid)
            cells[f"{profile_name}/{grid}"] = {
                "profile": profile_name, "grid": grid,
                "makespan_ns": makespans, "total_rounds": rounds,
                "attempts": attempts,
            }
    device_after = _wait_for_idle(args.device)
    candidate = {
        "schema": "nvidia_h200_fa2_chunked_two_resident_ingress_capture_v1",
        "hardware_id": "nvidia_h200",
        "cells": cells,
        "registers_per_thread": registers,
        "static_ldgsts_bypass_128_instructions": static_ldgsts,
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "owner": "h200-port-wp3_ldgsts",
            "source_commit": args.source_commit,
            "device_before": device_before, "device_after": device_after,
            "compile_command": compile_command,
            "clock_gate": _clock_summary(
                [cell["attempts"] for cell in cells.values()]),
            "pid": os.getpid(),
            "wrapper_sha256": _sha256(WRAPPER),
            "ingress_sha256": _sha256(INGRESS),
            "runner_sha256": _sha256(RUNNER),
            "binary_sha256": _sha256(binary),
            "ptxas_sha256": _sha256(output / "ptxas.txt"),
            "sass_sha256": _sha256(output / "probe.sass"),
        },
    }
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output / "candidate.json")


def publish(args):
    candidates = [_load_capture(path, "candidate.json") for path in args.capture]
    if len(candidates) < 2:
        raise ValueError("publication needs two independent captures")
    registers = {candidate["registers_per_thread"] for _, _, candidate in candidates}
    if len(registers) != 1:
        raise ValueError("captures used different builds")
    profiles = {}
    spans = []
    rates_4096 = {}
    for profile_name in PROFILES.values():
        profiles[profile_name] = {}
        for grid in GRIDS:
            key = f"{profile_name}/{grid}"
            rounds = {candidate["cells"][key]["total_rounds"]
                      for _, _, candidate in candidates}
            if len(rounds) != 1:
                raise ValueError(f"{key}: captures disagree on total rounds")
            rounds = rounds.pop()
            makespans = [
                value for _, _, candidate in candidates
                for value in candidate["cells"][key]["makespan_ns"]]
            median = statistics.median(makespans)
            span_pct = (max(makespans) - min(makespans)) / median * 100
            spans.append(span_pct)
            cell = {
                "total_rounds": rounds,
                "median_makespan_ns": median,
                "relative_span_pct": span_pct,
            }
            if grid == 4096:
                rate = rounds * PANEL_BYTES / 32 / (median * 1e-9) / SM_CLOCK_HZ
                cell["payload_32b_transactions_per_model_sm_cycle"] = rate
                rates_4096[profile_name] = rate
            profiles[profile_name][str(grid)] = cell
    selected = min(rates_4096.values())
    profile_difference = (
        (max(rates_4096.values()) - selected) / max(rates_4096.values()))
    reference = {
        "schema": "nvidia_h200_fa2_chunked_q8192_two_resident_ingress_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "evidence_kind": "independent_chunked_prefill_kv_ingress_microbenchmark",
        "source_capture": [_relative(path) for path, _, _ in candidates],
        "source_baseline_commit": sorted({
            c["provenance"]["source_commit"] for _, _, c in candidates}),
        "physical_device_index": candidates[0][2]["provenance"]["device_before"]["index"],
        "device_uuid": candidates[0][2]["provenance"]["device_before"]["uuid"],
        "resource_contract": {
            "threads_per_cta": 128,
            "dynamic_shared_bytes_per_cta": DYNAMIC_SHARED_BYTES,
            "registers_per_thread": registers.pop(),
            "resident_ctas_per_sm": 2,
            "physical_sm_count": SM_COUNT,
            "panel_bytes_per_cta": PANEL_BYTES,
            "initial_transaction_bytes": 32,
            "static_ldgsts_bypass_128_instructions": 32,
            "stack_frame_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "source_period_ctas": 128,
            "source_profiles": {
                "skew_q1024_q4096": {
                    "q_lens": [1024, 4096],
                    "kv_lens": [1024, 4096],
                },
                "uniform_q256_to_q1019": {
                    "q_lens": [256, 365, 474, 583, 692, 801, 910, 1019],
                    "kv_lens": [1024, 1462, 1900, 2338, 2776, 3214, 3652, 4090],
                },
            },
        },
        "campaign": {
            "warmup_launches": WARMUP_LAUNCHES,
            "samples_per_cell": SAMPLES * len(candidates),
            "grids": list(GRIDS),
            "profiles": profiles,
            "ncu_dynamic_counter_closure": None,
        },
        "selected_service": {
            "selection_rule": "lower_4096_cta_direct_median_across_the_two_source_profiles",
            "exact_grid_ctas": 4096,
            "model_sm_clock_hz": SM_CLOCK_HZ,
            "full_chip_initial_32b_transactions_per_model_sm_cycle": selected,
            "logical_byte_capacity_per_s": selected * 32 * SM_CLOCK_HZ,
        },
        "publication_gates": {
            "two_resident_ctas_per_sm": True,
            "static_ldgsts_bypass_128_count_is_32": True,
            "no_stack_or_spill": True,
            "dynamic_instruction_and_byte_counters_close_exactly": None,
            "all_native_cells_relative_span_at_most_5_pct": max(spans) <= 5.0,
            "selected_grid_source_profiles_within_5_pct": profile_difference <= 0.05,
        },
        "artifact_sha256": {
            "wrapper_source": _sha256(WRAPPER),
            "ingress_source": _sha256(INGRESS),
            "runner": _sha256(RUNNER),
            "binary": sorted({c["provenance"]["binary_sha256"] for _, _, c in candidates}),
            "ptxas": sorted({c["provenance"]["ptxas_sha256"] for _, _, c in candidates}),
            "sass": sorted({c["provenance"]["sass_sha256"] for _, _, c in candidates}),
            "candidates": [_sha256(path) for _, path, _ in candidates],
            # Per-cell makespans live in the capture candidates; NCU captures
            # do not exist on the H200 host.
            **{f"profile_{short}_grid{grid}": [
                _sha256(path) for _, path, _ in candidates]
               for short in ("skew", "uniform") for grid in GRIDS},
            **{f"ncu_{short}_grid{grid}": None
               for short in ("skew", "uniform") for grid in GRIDS},
        },
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "For the two exact 4096-CTA chunked-prefill source profiles, each "
            "with 128 threads, 82944 dynamic shared bytes, two resident CTAs "
            "per SM, and one 64-KiB K/V LDGSTS round, the shared K/V ingress "
            f"service on H200 has at least {selected} initial 32-byte "
            "transactions per 1.98-GHz SM cycle."),
        "forbidden_conclusion": (
            "This reference does not price Q ingress, MMA, softmax, output, "
            "different source profiles, different grids, different resident "
            "topology, generic L2 or DRAM traffic, or a production "
            "FlashAttention duration."),
        "provenance": {
            "wrapper": _relative(WRAPPER),
            "ingress": _relative(INGRESS),
            "runner": _relative(RUNNER),
            "sm_clock_hz": SM_CLOCK_HZ,
            "profile_difference_at_4096": profile_difference,
            "clock_gate": _clock_summary([
                cell["attempts"] for _, _, candidate in candidates
                for cell in candidate["cells"].values()]),
            "ncu_note": (
                "Nsight Compute is unavailable on the H200 host; the probe's "
                "own total_rounds (static source profile) replaces the NCU "
                "dynamic counter closure."),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("fa2_chunked_q8192_two_resident_ingress_sm90.json", reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--device", type=int, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--capture", type=Path, action="append",
                                required=True)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
