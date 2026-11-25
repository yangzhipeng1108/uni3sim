#!/usr/bin/env python3
"""Derive the H200 BM16 grouped-GEMM pipeline and long-K references.

Scripted form of the A800 hand derivations recorded in
``bm16_grouped_gemm_pipeline_sm80.json`` and
``bm16_grouped_gemm_long_k_pipeline_sm80.json``; rules unchanged:
  pipeline: per capture and fixed grid,
    interval = (median K2048 - median K512) / (32 - 8),
    fixed    = median K512 - 8 * interval;
    Forward values are the component-wise maxima across captures and grids.
  long-K:   anchor = maximum K2048 median across captures;
    interval = maximum finite difference per 64-element K tile across the
    three adjacent K ranges and captures.
The A800 "two idle devices" become the first two clock-accepted processes
(``pN.csv`` in process order) on the owned H200 GPU; a process is accepted
when every 50-ms SM-clock sample of its timed section is >= 1980 MHz.

Usage (local, after copying the remote run directories back):
  python sm80_bf16_bm16_grouped_gemm_pipeline_report.py \
      --sparse-row-dir <dir> --long-k-dir <dir> --launch-facts <json> \
      --remote-root <remote runs dir>
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SPARSE_ROW_RUNNER = HERE / "sm80_bf16_bm16_sparse_row_pipeline_runner.py"
LONG_K_RUNNER = HERE / "sm80_bf16_bm16_long_k_pipeline_runner.py"
REPORT = Path(__file__).resolve()
BOOST_MHZ = 1980
REQUIRED_ACCEPTED = 2
TRIALS = 9
PIPELINE_CELLS = {
    "k512_n1024": "K512_N1024_grid256",
    "k2048_n1024": "K2048_N1024_grid256",
    "k512_n2048": "K512_N2048_grid512",
    "k2048_n2048": "K2048_N2048_grid512",
}
LONG_K_VALUES = (2048, 4096, 7168, 14336)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clock(path):
    clocks = [int(line.split(",")[0]) for line in
              Path(path).read_text().splitlines() if line.strip()]
    busy = [int(line.split(",")[0]) for line in
            Path(path).read_text().splitlines()
            if line.strip() and int(line.split(",")[1]) > 0]
    return {
        "samples": len(clocks), "minimum_mhz": min(clocks),
        "median_mhz": statistics.median(clocks),
        "maximum_mhz": max(clocks), "busy_samples": len(busy),
        "accepted": bool(clocks) and min(clocks) >= BOOST_MHZ,
    }


def _captures(directory, parse):
    directory = Path(directory)
    captures = []
    for index in (1, 2, 3):
        raw = directory / f"p{index}.csv"
        clock = _clock(directory / f"p{index}.clock.csv")
        clock_rows = [line for line in raw.read_text().splitlines()
                      if line.startswith("CLOCK,") and
                      not line.startswith("CLOCK,device")]
        if len(clock_rows) != 1 or (clock_rows[0].endswith(",1")
                                    != clock["accepted"]):
            raise ValueError(f"{raw}: runner clock verdict disagrees")
        captures.append({
            "process": index, "raw": raw, "clock": clock,
            "status": "accepted" if clock["accepted"] else "rejected_clock",
            "data": parse(raw),
        })
    accepted = [c for c in captures if c["status"] == "accepted"]
    if len(accepted) < REQUIRED_ACCEPTED:
        raise ValueError(f"{directory}: fewer than two accepted captures")
    return captures, accepted[:REQUIRED_ACCEPTED]


def _meta(raw):
    rows = [line.split(",") for line in Path(raw).read_text().splitlines()
            if line.startswith("META,")]
    if len(rows) != 1:
        raise ValueError(f"{raw}: missing META row")
    return {"triton": rows[0][2], "torch": rows[0][4]}


def _parse_sparse_row(raw):
    samples = {name: [] for name in PIPELINE_CELLS}
    for line in Path(raw).read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "DATA" and fields[1] != "trial":
            samples[fields[2]].append(float(fields[5]) * 1.0e6)
    if any(len(values) != TRIALS for values in samples.values()):
        raise ValueError(f"{raw}: incomplete pipeline matrix")
    return {PIPELINE_CELLS[name]: statistics.median(values)
            for name, values in samples.items()}


def _parse_long_k(raw):
    samples = {k: [] for k in LONG_K_VALUES}
    for line in Path(raw).read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "DATA" and fields[1] != "trial":
            if int(fields[3]) != 256:
                raise ValueError(f"{raw}: long-K grid changed")
            samples[int(fields[2])].append(float(fields[5]) * 1.0e6)
    if any(len(values) != TRIALS for values in samples.values()):
        raise ValueError(f"{raw}: incomplete long-K matrix")
    return {k: statistics.median(values) for k, values in samples.items()}


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-row-dir", type=Path, required=True)
    parser.add_argument("--long-k-dir", type=Path, required=True)
    parser.add_argument("--launch-facts", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--command-prefix", required=True)
    arguments = parser.parse_args()
    facts = json.loads(arguments.launch_facts.read_text())
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True).stdout.strip()
    production_source = facts["vllm_fused_moe_source"]
    production_sha = facts["vllm_fused_moe_source_sha256"]
    w1 = facts["facts"]["bm16_w1"]
    w2 = facts["facts"]["bm16_w2"]

    all_pipeline, pipeline = _captures(
        arguments.sparse_row_dir, _parse_sparse_row)
    meta = _meta(pipeline[0]["raw"])
    environment = {
        "conda_env": 'vLLM 0.24 virtual environment',
        "python": "3.12", "torch": meta["torch"], "triton": meta["triton"],
        "cuda": "12.9 driver API (driver 575.57.08)",
    }
    medians = {f"device_1_process_{c['process']}": c["data"]
               for c in pipeline}
    fixed_values = []
    interval_values = []
    for capture in pipeline:
        for grid, low, high in (
                ("256", "K512_N1024_grid256", "K2048_N1024_grid256"),
                ("512", "K512_N2048_grid512", "K2048_N2048_grid512")):
            interval = (capture["data"][high] - capture["data"][low]) / 24
            interval_values.append(interval)
            fixed_values.append(capture["data"][low] - 8 * interval)

    def capture_entries(captures, name):
        return [{
            "path": _relative(c["raw"]),
            "remote_path": f"{arguments.remote_root}/{name}/p{c['process']}.csv",
            "device_index": 1,
            "process": c["process"],
            "status": c["status"],
            "clock": c["clock"],
            "command": (f"{arguments.command_prefix} "
                        f"{_relative(SPARSE_ROW_RUNNER if name == 'sparse_row' else LONG_K_RUNNER)}"
                        f" --trials 9 --clock-log p{c['process']}.clock.csv"),
            "raw_sha256": _sha256(c["raw"]),
        } for c in captures]

    provenance = {
        "git_commit": commit,
        "report": _relative(REPORT),
        "report_sha256": _sha256(REPORT),
        "clock_gate": (
            "every 50-ms SM-clock sample of the timed section >= 1980 MHz; "
            "first two accepted processes are used"),
        "selection_substitution": (
            "A800 two idle devices -> first two clock-accepted independent "
            "processes on H200 GPU 1"),
        "launch_facts": _relative(arguments.launch_facts),
        "launch_facts_sha256": _sha256(arguments.launch_facts),
    }
    pipeline_reference = {
        "schema": "nvidia_h200_bm16_grouped_gemm_pipeline_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "software_revision": f"vllm_0.24.0_triton_{meta['triton']}_sm90",
        "environment": environment,
        "source_baseline_commit": commit,
        "runner": _relative(SPARSE_ROW_RUNNER),
        "runner_sha256": _sha256(SPARSE_ROW_RUNNER),
        "production_source": production_source,
        "production_source_sha256": production_sha,
        "canonical_launch_observation": {
            "path": provenance["launch_facts"],
            "sha256": provenance["launch_facts_sha256"],
            "w1_registers_per_thread": w1["registers_per_thread"],
            "w2_registers_per_thread": w2["registers_per_thread"],
            "dynamic_shared_memory_bytes_per_cta":
                w2["dynamic_shared_memory_bytes_per_cta"],
            "threads_per_cta": w2["threads_per_cta"],
            "residency_uses_conservative_w2_register_count": True,
        },
        "captures": capture_entries(pipeline, "sparse_row"),
        "source_and_launch_contract": {
            "dtype": "bf16", "tile_m": 16, "tile_n": 32, "tile_k": 64,
            "group_size_m": 1,
            "threads_per_cta": w2["threads_per_cta"],
            "registers_per_thread": w2["registers_per_thread"],
            "dynamic_shared_memory_bytes_per_cta":
                w2["dynamic_shared_memory_bytes_per_cta"],
            "pipeline_depth": 2,
            "physical_sm_count": 132,
            "resident_ctas_per_sm": w2["resident_ctas_per_sm"],
        },
        "probe_contract": {
            "active_rows_per_tile": 1,
            "independent_streams": 8,
            "projection_cells": list(PIPELINE_CELLS.values()),
            "trial_count_per_cell": TRIALS,
            "timer_domain": "cuda_event_gpu_timeline",
            "cache_polluter_bytes": 134217728,
            "production_routing_sort_reduce_absent": True,
            "production_or_evaluation_duration_consumed": False,
        },
        "median_duration_ns": medians,
        "service_parameters": {
            "fixed_span_ns": max(fixed_values),
            "local_k_round_interval_ns": max(interval_values),
            "derivation": (
                "For each accepted process and fixed grid, the K-round "
                "interval is (median(K=2048)-median(K=512))/(32-8), and the "
                "fixed span is median(K=512)-8*interval. Forward uses the "
                "component-wise maximum across both processes and both "
                "grids; no production accuracy or duration selects either "
                "value."),
        },
        "decision": {
            "status": "accepted_independent_pipeline_mechanism",
            "forward_eligible": True,
            "reason": (
                "The synthetic sparse-row program isolates the source-shaped "
                "two-slot load-to-HMMA K loop without routing, sorting, "
                "reduction, or production timing. The two-process upper "
                "envelope supplies launch-wide fixed and K-round services; "
                "HWSpec and source launch facts supply residency."),
        },
        "excluded_conclusions": [
            "a complete FusedMoE operator duration",
            "a production-shape latency lookup",
            "a residual correction",
            "tile signatures other than BF16 BM16 BN32 BK64 GSM1",
            "resources or devices outside the recorded H200 contract",
        ],
        "h200_capture_provenance": {
            **provenance,
            "all_processes": capture_entries(all_pipeline, "sparse_row"),
            "accepted_captures": sum(
                c["status"] == "accepted" for c in all_pipeline),
            "rejected_captures": sum(
                c["status"] != "accepted" for c in all_pipeline),
            "per_capture_grid_values_ns": {
                "fixed_span": fixed_values,
                "local_k_round_interval": interval_values},
            "pipeline_depth_note": (
                "the probe launches num_stages=2 as on A800; the H200 vLLM "
                "0.24 production launch compiles with Triton num_stages="
                f"{w2['num_stages']} and the same "
                f"{w2['dynamic_shared_memory_bytes_per_cta']}-B shared memory"),
        },
    }

    all_long_k, long_k = _captures(arguments.long_k_dir, _parse_long_k)
    anchors = [c["data"][2048] for c in long_k]
    tail_intervals = []
    for capture in long_k:
        for low, high in zip(LONG_K_VALUES, LONG_K_VALUES[1:]):
            tail_intervals.append(
                (capture["data"][high] - capture["data"][low])
                / ((high - low) // 64))
    long_k_reference = {
        "schema": "nvidia_h200_bm16_grouped_gemm_long_k_pipeline_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "software_revision": f"vllm_0.24.0_triton_{meta['triton']}_sm90",
        "environment": environment,
        "source_baseline_commit": commit,
        "runner": _relative(LONG_K_RUNNER),
        "runner_sha256": _sha256(LONG_K_RUNNER),
        "isolated_pipeline_source": _relative(SPARSE_ROW_RUNNER),
        "isolated_pipeline_source_sha256": _sha256(SPARSE_ROW_RUNNER),
        "production_source": production_source,
        "production_source_sha256": production_sha,
        "captures": capture_entries(long_k, "long_k"),
        "probe_contract": {
            "dtype": "bf16", "tile_m": 16, "tile_n": 32, "tile_k": 64,
            "group_size_m": 1, "threads_per_cta": 128, "pipeline_depth": 2,
            "active_rows_per_tile": 1, "independent_streams": 2,
            "projection_n": 4096, "grid_ctas": 256,
            "k_values": list(LONG_K_VALUES),
            "trial_count_per_cell": TRIALS,
            "timer_domain": "cuda_event_gpu_timeline",
            "cache_polluter_bytes": 134217728,
            "production_routing_sort_reduce_absent": True,
            "production_or_evaluation_duration_consumed": False,
        },
        "cross_device_upper_envelope": {
            "anchor_k": 2048,
            "anchor_k_tiles": 32,
            "anchor_completion_ns": max(anchors),
            "tail_round_interval_ns": max(tail_intervals),
            "derivation": (
                "The K=2048 anchor is the maximum median across both accepted "
                "processes. The tail interval is the maximum finite "
                "difference per 64-element K tile across the three adjacent "
                "K ranges and both processes. This fixed-resource upper "
                "envelope is selected without production timing or "
                "accuracy."),
        },
        "decision": {
            "status": "accepted_independent_long_k_pipeline_mechanism",
            "forward_eligible": True,
            "allowed_use": (
                "Price only one-wave, exactly 256-CTA BF16 "
                "BM16/BN32/BK64/GSM1 grouped-GEMM launches at K greater than "
                "or equal to 2048."),
            "excluded_conclusions": [
                "a complete FusedMoE operator duration",
                "a production-shape latency lookup",
                "a residual correction",
                "launches with any grid other than 256 CTAs",
                "tile signatures other than BF16 BM16/BN32/BK64/GSM1",
                "resources or devices outside the recorded H200 contract",
            ],
        },
        "h200_capture_provenance": {
            **provenance,
            "all_processes": capture_entries(all_long_k, "long_k"),
            "accepted_captures": sum(
                c["status"] == "accepted" for c in all_long_k),
            "rejected_captures": sum(
                c["status"] != "accepted" for c in all_long_k),
            "median_duration_ns_by_process": {
                f"process_{c['process']}": {
                    str(k): v for k, v in c["data"].items()}
                for c in long_k},
            "tail_intervals_ns": tail_intervals,
        },
    }
    references = ROOT / "microbench/nvidia_h200/references"
    for name, payload in (
            ("bm16_grouped_gemm_pipeline_sm90.json", pipeline_reference),
            ("bm16_grouped_gemm_long_k_pipeline_sm90.json",
             long_k_reference)):
        (references / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixed_span_ns": max(fixed_values),
        "local_k_round_interval_ns": max(interval_values),
        "long_k_anchor_completion_ns": max(anchors),
        "long_k_round_interval_ns": max(tail_intervals),
    }, indent=2))


if __name__ == "__main__":
    main()
