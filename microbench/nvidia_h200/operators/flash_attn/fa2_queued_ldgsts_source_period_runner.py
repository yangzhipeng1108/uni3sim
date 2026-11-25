#!/usr/bin/env python3
"""Capture and publish the H200 queued FA2 LDGSTS source-period frontend.

The A800 probe
microbench/nvidia_a800/operators/flash_attn/fa2_queued_ldgsts_source_period.cu
had no runner.  Its two A800 references record the procedure:

* period 8 (maximum repeats 4, panels 1,1,2,2,3,3,4,4): grids 512/1024/2048/
  4096, 10 warm-up launches, 7 samples, direct %globaltimer makespan per
  sample (max block end - min block begin); payload bytes / median makespan;
  selected = lower of the 2048- and 4096-CTA medians
  (fa2_queued_ldgsts_steady_frontend_sm80.json);
* period 32 (maximum repeats 32): grids 2048/4096, same timing; selected =
  lower of the two medians (fa2_queued_ldgsts_period32_steady_frontend_sm80.json);
* gates: two resident CTAs, 32 static LDGSTS.E.BYPASS.128, no spill, <= 5%
  sample spread at the two steady grids, steady medians within 5%.

This runner scripts that procedure.  Hardware coupling: the probe is the H200
owner copy (132 SMs, sm_90, two resident CTAs still enforced), transactions
per cycle use the measured 1.98-GHz SM clock instead of the 1.14-GHz model
clock, H200 GPU 2 only, 50-ms SM-clock gate per probe process (helpers from
microbench/nvidia_h200/memory/l2/staged_ldgsts_l2_runner.py), and the A800
single capture becomes two independent capture processes whose samples are
pooled per grid.  The NCU dynamic counter-closure gate cannot be evaluated
(Nsight Compute is unavailable on the H200 host) and is recorded as null;
the probe prints each block's analytic repeat count, and the runner checks
their sum against the source period instead.
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


SOURCE = Path(__file__).with_name("fa2_queued_ldgsts_source_period.cu")
RUNNER = Path(__file__).resolve()
PANEL_BYTES = 65536
SAMPLES = 7
WARMUP_LAUNCHES = 10
GRIDS = {4: (512, 1024, 2048, 4096), 32: (2048, 4096)}
STEADY_GRIDS = (2048, 4096)
SUMMARY = re.compile(
    r"^sample=(\d+) grid=(\d+) resident=(\d+) min_repeats=1 max_repeats=(\d+) "
    r"period_ctas=(\d+) period_offset=0 panel_bytes=65536 checksum=(\d+)$")
BLOCK = re.compile(
    r"^block sample=(\d+) id=(\d+) sm=(\d+) repeats=(\d+) begin_ns=(\d+) "
    r"end_ns=(\d+) elapsed_ns=(\d+)$")


def _period(maximum_repeats):
    return [panel for panel in range(1, maximum_repeats + 1)
            for _ in range(2)]


def _parse(stdout, grid, maximum_repeats):
    samples = {}
    for line in stdout.splitlines():
        summary = SUMMARY.match(line)
        if summary:
            sample = int(summary.group(1))
            if (int(summary.group(2)) != grid or int(summary.group(3)) != 2
                    or int(summary.group(4)) != maximum_repeats
                    or int(summary.group(5)) != 2 * maximum_repeats):
                raise ValueError(f"summary violates contract: {line}")
            samples[sample] = {"blocks": 0, "rounds": 0, "makespan_ns": 0,
                               "sms": set()}
            continue
        block = BLOCK.match(line)
        if block:
            record = samples[int(block.group(1))]
            record["blocks"] += 1
            record["rounds"] += int(block.group(4))
            record["makespan_ns"] = max(
                record["makespan_ns"], int(block.group(6)))
            record["sms"].add(int(block.group(3)))
            continue
        raise ValueError(f"unexpected probe line: {line}")
    period = _period(maximum_repeats)
    expected_rounds = sum(period[block % len(period)] for block in range(grid))
    if sorted(samples) != list(range(SAMPLES)):
        raise ValueError("probe sample set changed")
    makespans = []
    for record in samples.values():
        if record["blocks"] != grid or record["rounds"] != expected_rounds:
            raise ValueError("block rows do not close the source period")
        if len(record["sms"]) != SM_COUNT:
            raise ValueError("launch did not cover every SM")
        makespans.append(record["makespan_ns"])
    return makespans, expected_rounds


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "fa2_queued_ldgsts_source_period"
    compile_command, ptxas = _compile(SOURCE, binary)
    (output / "ptxas.txt").write_text(ptxas, encoding="utf-8")
    sass = _dump_sass(binary, output / "probe.sass")
    static_ldgsts = len(re.findall(r"\bLDGSTS\.E\.BYPASS\.128\b", sass))
    if (static_ldgsts != 32 or "0 bytes spill stores" not in ptxas
            or "0 bytes stack frame" not in ptxas):
        raise ValueError("compiled probe violates the static contract")
    registers = int(re.search(r"Used ([0-9]+) registers", ptxas).group(1))
    cells = {}
    for grid in GRIDS[args.maximum_repeats]:
        command = [str(binary), str(SAMPLES), str(grid),
                   str(args.maximum_repeats), "0"]
        result, attempts = _gated_run(
            command, device=args.device,
            log_stem=output / f"grid{grid}")
        makespans, rounds = _parse(result.stdout, grid, args.maximum_repeats)
        cells[str(grid)] = {
            "makespan_ns": makespans,
            "total_rounds": rounds,
            "payload_bytes": rounds * PANEL_BYTES,
            "attempts": attempts,
            "stdout_lines": len(result.stdout.splitlines()),
        }
    device_after = _wait_for_idle(args.device)
    candidate = {
        "schema": "nvidia_h200_fa2_queued_ldgsts_source_period_capture_v1",
        "hardware_id": "nvidia_h200",
        "maximum_repeats": args.maximum_repeats,
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
            "source_sha256": _sha256(SOURCE),
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
    maximum_repeats = {candidate["maximum_repeats"] for _, _, candidate in candidates}
    registers = {candidate["registers_per_thread"] for _, _, candidate in candidates}
    if len(maximum_repeats) != 1 or len(registers) != 1:
        raise ValueError("captures disagree on period or build")
    maximum_repeats = maximum_repeats.pop()
    registers = registers.pop()
    grids = GRIDS[maximum_repeats]
    period = _period(maximum_repeats)
    pooled = {}
    for grid in grids:
        makespans = [
            value for _, _, candidate in candidates
            for value in candidate["cells"][str(grid)]["makespan_ns"]]
        payload = candidates[0][2]["cells"][str(grid)]["payload_bytes"]
        median_makespan = statistics.median(makespans)
        bandwidths = [payload / (value * 1e-9) for value in makespans]
        pooled[grid] = {
            "makespan_ns": makespans,
            "payload_bytes": payload,
            "median_makespan_ns": median_makespan,
            "median_payload_bytes_per_s": payload / (median_makespan * 1e-9),
            "median_payload_32b_transactions_per_sm_cycle": (
                payload / 32 / (median_makespan * 1e-9) / SM_CLOCK_HZ),
            "relative_bandwidth_spread": (
                (max(bandwidths) - min(bandwidths))
                / statistics.median(bandwidths)),
            "relative_span_pct": (
                (max(makespans) - min(makespans)) / median_makespan * 100),
        }
    steady = [pooled[grid]["median_payload_bytes_per_s"] for grid in STEADY_GRIDS]
    selected_bytes = min(steady)
    steady_difference = (max(steady) - min(steady)) / max(steady)
    selected_transactions = selected_bytes / 32 / SM_CLOCK_HZ
    artifact = {
        "source": _sha256(SOURCE),
        "runner": _sha256(RUNNER),
        "binary": sorted({c["provenance"]["binary_sha256"] for _, _, c in candidates}),
        "ptxas": sorted({c["provenance"]["ptxas_sha256"] for _, _, c in candidates}),
        "sass": sorted({c["provenance"]["sass_sha256"] for _, _, c in candidates}),
        "candidates": [_sha256(path) for _, path, _ in candidates],
    }
    for grid in grids:
        # The per-grid %globaltimer makespans live in the capture candidates;
        # NCU captures do not exist on the H200 host.
        artifact[f"grid_{grid}_globaltimer"] = artifact["candidates"]
        artifact[f"grid_{grid}_ncu"] = None
    common = {
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "evidence_kind": "independent_queued_heterogeneous_ldgsts_frontend_microbenchmark",
        "source_capture": [_relative(path) for path, _, _ in candidates],
        "source_baseline_commit": sorted({
            c["provenance"]["source_commit"] for _, _, c in candidates}),
        "physical_device_index": candidates[0][2]["provenance"]["device_before"]["index"],
        "device_uuid": candidates[0][2]["provenance"]["device_before"]["uuid"],
    }
    contract = {
        "threads_per_cta": 128,
        "dynamic_shared_bytes_per_cta": 77824,
        "registers_per_thread": registers,
        "resident_ctas_per_sm": 2,
        "physical_sm_count": SM_COUNT,
        "panel_bytes_per_cta": PANEL_BYTES,
        "initial_transaction_bytes": 32,
        "static_ldgsts_bypass_128_instructions": 32,
    }
    provenance = {
        "source": _relative(SOURCE),
        "runner": _relative(RUNNER),
        "sm_clock_hz": SM_CLOCK_HZ,
        "samples_pooled_per_grid": SAMPLES * len(candidates),
        "clock_gate": _clock_summary([
            cell["attempts"] for _, _, candidate in candidates
            for cell in candidate["cells"].values()]),
        "ncu_note": (
            "Nsight Compute is unavailable on the H200 host; the dynamic "
            "counter closure is replaced by the probe's per-block analytic "
            "repeat counts, which close the source period exactly in every "
            "sample."),
        "remote_raw_root": (
            'profiler_output/microbench/nvidia_h200'),
    }
    if maximum_repeats == 4:
        contract.update({
            "dynamic_ldgsts_warp_instructions_per_panel": 128,
            "dynamic_ldgsts_counter_bytes_per_panel": 65536,
            "minimum_repeats": 1,
            "maximum_repeats": 4,
            "source_period_ctas": 8,
            "source_period_panels": period,
            "average_panels_per_cta": statistics.fmean(period),
            "period_offset": 0,
        })
        reference = {
            "schema": "nvidia_h200_fa2_queued_ldgsts_steady_frontend_reference_v1",
            **common,
            "resource_contract": contract,
            "campaign": {
                "warmup_launches": WARMUP_LAUNCHES,
                "samples_per_grid": SAMPLES * len(candidates),
                "grids": list(grids),
                "direct_globaltimer": {
                    str(grid): {
                        "makespan_ns": pooled[grid]["makespan_ns"],
                        "median_payload_bytes_per_s": pooled[grid][
                            "median_payload_bytes_per_s"],
                        "relative_bandwidth_spread": pooled[grid][
                            "relative_bandwidth_spread"],
                    } for grid in grids},
                "steady_median_relative_difference": steady_difference,
                "ncu_dynamic_counter_closure": None,
            },
            "selected_service": {
                "selection_rule": "lower_direct_median_of_the_two_converged_steady_grids",
                "minimum_queued_grid_ctas": 2048,
                "full_chip_payload_bytes_per_s": selected_bytes,
                "model_sm_clock_hz": SM_CLOCK_HZ,
                "full_chip_initial_32b_transactions_per_model_sm_cycle": selected_transactions,
            },
            "publication_gates": {
                "two_resident_ctas_per_sm": True,
                "static_ldgsts_bypass_128_count_is_32": True,
                "dynamic_instruction_and_byte_counters_close_exactly": None,
                "grid_2048_relative_bandwidth_spread_at_most_5_pct": (
                    pooled[2048]["relative_bandwidth_spread"] <= 0.05),
                "grid_4096_relative_bandwidth_spread_at_most_5_pct": (
                    pooled[4096]["relative_bandwidth_spread"] <= 0.05),
                "steady_grid_medians_differ_at_most_5_pct": steady_difference <= 0.05,
            },
            "artifact_sha256": artifact,
            "production_or_evaluation_duration_consumed": False,
            "allowed_conclusion": (
                "For the source-exact two-resident Q64/KV128 causal period "
                "(1,1,2,2,3,3,4,4), queued grids at or above 2048 CTAs expose a "
                "stable full-chip LDGSTS frontend payload capacity of at least "
                f"{selected_bytes} bytes/s on H200 ({selected_transactions} "
                "initial 32-byte transactions per 1.98-GHz SM cycle)."),
            "forbidden_conclusion": (
                "This reference does not authorize a generic LSU, L2, DRAM, "
                "different source-period, different residency, or "
                "production-duration correction."),
            "provenance": provenance,
        }
        name = "fa2_queued_ldgsts_steady_frontend_sm90.json"
    else:
        contract.update({
            "stack_frame_bytes": 0,
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
            "minimum_repeats": 1,
            "maximum_repeats": 32,
            "source_period_ctas": 64,
            "source_period_panels": period,
            "average_panels_per_cta": statistics.fmean(period),
            "period_offset": 0,
        })
        reference = {
            "schema": "nvidia_h200_fa2_queued_ldgsts_period32_steady_frontend_reference_v1",
            **common,
            "resource_contract": contract,
            "campaign": {
                "warmup_launches": WARMUP_LAUNCHES,
                "samples_per_grid": SAMPLES * len(candidates),
                "grids": list(grids),
                "direct_globaltimer": {
                    str(grid): {
                        "makespan_ns": pooled[grid]["makespan_ns"],
                        "median_payload_32b_transactions_per_model_sm_cycle": pooled[grid][
                            "median_payload_32b_transactions_per_sm_cycle"],
                        "relative_span_pct": pooled[grid]["relative_span_pct"],
                    } for grid in grids},
                "steady_median_relative_difference_pct": steady_difference * 100,
                "ncu_dynamic_counter_closure": None,
            },
            "selected_service": {
                "selection_rule": "lower_direct_median_of_the_two_pre_registered_stable_queued_grids",
                "minimum_queued_grid_ctas": 2048,
                "model_sm_clock_hz": SM_CLOCK_HZ,
                "full_chip_initial_32b_transactions_per_model_sm_cycle": selected_transactions,
                "logical_byte_capacity_per_s": selected_bytes,
            },
            "publication_gates": {
                "two_resident_ctas_per_sm": True,
                "static_ldgsts_bypass_128_count_is_32": True,
                "no_stack_or_spill": True,
                "dynamic_instruction_and_byte_counters_close_exactly": None,
                "grid_2048_relative_span_at_most_5_pct": (
                    pooled[2048]["relative_span_pct"] <= 5.0),
                "grid_4096_relative_span_at_most_5_pct": (
                    pooled[4096]["relative_span_pct"] <= 5.0),
                "steady_grid_medians_differ_at_most_5_pct": steady_difference <= 0.05,
            },
            "artifact_sha256": artifact,
            "production_or_evaluation_duration_consumed": False,
            "allowed_conclusion": (
                "For the source-exact two-resident Q64/KV128 causal period "
                "(1,1,...,32,32), queued grids at or above 2048 CTAs expose a "
                "stable full-chip LDGSTS frontend payload capacity of at least "
                f"{selected_transactions} initial 32-byte transactions per "
                "1.98-GHz H200 SM cycle."),
            "forbidden_conclusion": (
                "This reference does not authorize a generic LSU, L2, DRAM, "
                "different source-period, different residency, or "
                "production-duration correction."),
            "provenance": provenance,
        }
        name = "fa2_queued_ldgsts_period32_steady_frontend_sm90.json"
    _write_reference(name, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--device", type=int, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    capture_parser.add_argument("--maximum-repeats", type=int, choices=(4, 32),
                                required=True)
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
