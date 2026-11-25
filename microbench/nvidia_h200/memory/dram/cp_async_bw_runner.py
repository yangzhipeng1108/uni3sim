#!/usr/bin/env python3
"""Capture and publish the H200 cp.async (LDGSTS) initial-transaction rates.

The A800 probe microbench/nvidia_a800/memory/dram/cp_async_bw.cu had no
runner; its A800 evidence (profiler_output/microbench/nvidia_a800/
a800_cp_async_service_20260825_392a34bc/) is one 256-MiB invocation whose
event-timed table (results.txt) gives the payload bandwidth per grid, plus
NCU counters that fixed the counter geometry: the retained Xbar LDGSTS
counter is 2x payload bytes (1024 counter bytes = 32 initial 32-B
transactions per 512-B warp LDGSTS).  The A800 parameters are
payload_B/s x 2 / 32 B / 1.14 GHz with the selection rule

* single_sm_rate: grid 1;
* per_active_sm_rate: median of the scaling-region per-SM rates, grids
  2/4/8/16/32;
* full_chip_rate: median of the saturated grids 108/216/432.

This runner scripts that derivation for H200.  Hardware coupling: the
saturated grids become 132/264/528 (1, 2, 4 CTAs per H200 SM), the SM clock
is the measured 1.98 GHz, H200 GPU 2 only, CUDA 12.9 sm_90 build, 50-ms
SM-clock gate per probe process (helpers from ../l2/staged_ldgsts_l2_runner.py),
and the A800 single invocation becomes two independent processes whose
per-parameter minimum is published.

Nsight Compute is not installed on the H200 host, so the 2x counter geometry
cannot be re-measured; it is carried from the A800 NCU capture as a property
of the same cp.async.cg 16-B instruction stream (identical PTX, the H200
binary issues the same LDGSTS.E.BYPASS.128 SASS), and the reference records
it as carried, not re-verified.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
    SM_CLOCK_HZ,
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


SOURCE = Path(__file__).with_name("cp_async_bw.cu")
RUNNER = Path(__file__).resolve()
SIZE_MB = 256
MAX_GRID = 528
GRIDS = (1, 2, 4, 8, 16, 32, 64, 132, 264, 528)
SCALING_GRIDS = (2, 4, 8, 16, 32)
SATURATED_GRIDS = (132, 264, 528)
COUNTER_BYTES_PER_PAYLOAD_BYTE = 2
TRANSACTION_BYTES = 32


def _parse(stdout):
    table = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0].isdigit():
            grid = int(fields[0])
            if grid in table:
                raise ValueError(f"duplicate grid row {grid}")
            table[grid] = {
                "time_ms": float(fields[1]),
                "bandwidth_gb_s": float(fields[2]),
                "per_sm_gb_s": float(fields[3]),
            }
    if tuple(sorted(table)) != GRIDS:
        raise ValueError(f"probe grid table changed: {sorted(table)}")
    return table


def _rates(table):
    def transactions_per_cycle(gb_s):
        return (gb_s * 1.0e9 * COUNTER_BYTES_PER_PAYLOAD_BYTE
                / TRANSACTION_BYTES / SM_CLOCK_HZ)
    return {
        "ldgsts_initial_32b_transactions_single_sm_cycle":
            transactions_per_cycle(table[1]["bandwidth_gb_s"]),
        "ldgsts_initial_32b_transactions_per_active_sm_cycle":
            transactions_per_cycle(statistics.median(
                table[grid]["per_sm_gb_s"] for grid in SCALING_GRIDS)),
        "ldgsts_initial_32b_transactions_full_chip_per_cycle":
            transactions_per_cycle(statistics.median(
                table[grid]["bandwidth_gb_s"] for grid in SATURATED_GRIDS)),
    }


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "cp_async_bw"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    if "LDGSTS.E.BYPASS.128" not in sass or "LDL" in sass or "STL" in sass:
        raise ValueError("compiled SASS violates the LDGSTS contract")
    command = [str(binary), str(SIZE_MB), str(MAX_GRID)]
    result, attempts = _gated_run(
        command, device=args.device, log_stem=output / "probe")
    (output / "results.txt").write_text(result.stdout, encoding="utf-8")
    table = _parse(result.stdout)
    device_after = _wait_for_idle(args.device)
    candidate = {
        "schema": "nvidia_h200_cp_async_service_capture_v1",
        "hardware_id": "nvidia_h200",
        "table": {str(grid): row for grid, row in sorted(table.items())},
        "parameters": _rates(table),
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "owner": "h200-port-wp3_ldgsts",
            "source_commit": args.source_commit,
            "device_before": device_before, "device_after": device_after,
            "compile_command": compile_command,
            "compiler_stderr": compiler_stderr,
            "attempts": attempts,
            "clock_gate": _clock_summary([attempts]),
            "pid": os.getpid(),
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(output / "probe.sass.txt"),
            "results_sha256": _sha256(output / "results.txt"),
        },
    }
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output / "candidate.json")


def publish(args):
    candidates = []
    for path in args.capture:
        candidates.append(_load_capture(path, "candidate.json"))
    if len(candidates) < 2:
        raise ValueError("publication needs two independent captures")
    names = tuple(candidates[0][2]["parameters"])
    parameters = {
        name: min(candidate["parameters"][name] for _, _, candidate in candidates)
        for name in names}
    reference = {
        "schema_version": 1,
        "status": "accepted_for_source_bound_sm80_fa2_global_read",
        "source_commit": sorted({
            candidate["provenance"]["source_commit"]
            for _, _, candidate in candidates}),
        "rate_unit": "initial_32b_transactions_per_sm_domain_cycle",
        "logical_to_counter_geometry": {
            "payload_bytes_per_warp_ldgsts": 512,
            "observed_xbar_bytes_per_warp_ldgsts": 1024,
            "initial_32b_transactions_per_warp_ldgsts": 32,
            "h200_status": (
                "carried from the A800 NCU capture a800_cp_async_service_"
                "20260825_392a34bc (ncu_size256_grid108/432.csv); not "
                "re-verified because Nsight Compute is unavailable on the "
                "H200 host.  The H200 binary issues the same "
                "LDGSTS.E.BYPASS.128 instruction for the same 16-B "
                "cp.async.cg PTX."),
        },
        "selection_rule": {
            "single_sm_rate": "grid 1",
            "per_active_sm_rate": "median of scaling-region grids 2,4,8,16,32",
            "full_chip_rate": "median of saturated grids 132,264,528",
            "across_captures": (
                "minimum over two independent clock-accepted probe processes "
                "on H200 GPU 2"),
            "conversion": (
                "event-timed payload GB/s x 2 counter bytes per payload byte "
                "/ 32 B / measured 1.98-GHz SM clock"),
            "uses_production_duration": False,
        },
        "parameters": parameters,
        "scope": {
            "hardware": "nvidia_h200",
            "source": "SM80 FA2 cp.async/LDGSTS global-to-shared read ingress",
            "block_threads": 128,
            "pipeline_groups": 4,
            "buffer_mib": SIZE_MB,
            "excluded": ["generic LSU", "ordinary global load", "global write",
                         "FA2 combine", "grouped MLA"],
        },
        "evidence": [
            {
                "path": _relative(capture_dir),
                "candidate_sha256": _sha256(candidate_path),
                "results_sha256": _sha256(capture_dir / "results.txt"),
                "parameters": candidate["parameters"],
                "table": candidate["table"],
            }
            for capture_dir, candidate_path, candidate in candidates],
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "source": _relative(SOURCE),
            "source_sha256": _sha256(SOURCE),
            "runner": _relative(RUNNER),
            "runner_sha256": _sha256(RUNNER),
            "sm_clock_hz": SM_CLOCK_HZ,
            "clock_gate": _clock_summary([
                candidate["provenance"]["attempts"]
                for _, _, candidate in candidates]),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("ldgsts_initial_32b_transactions_sm90.json", reference)


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
