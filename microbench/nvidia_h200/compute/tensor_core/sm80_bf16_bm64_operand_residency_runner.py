#!/usr/bin/env python3
"""Capture H200 BM64 two-slot hit and grouped-weight miss services.

H200 owner copy of
microbench/nvidia_a800/compute/tensor_core/sm80_bf16_bm64_operand_residency_runner.py.
Parsing, per-case service arithmetic, the 10 % stability gate and every
selection rule are unchanged.  Hardware-coupling changes only:
  - build: nvcc/cuobjdump 12.2 local toolchain, ``sm_90`` target, static
    cudart (target host has no cuobjdump); the SASS contract (LDGSTS +
    HMMA.16816.F32.BF16, no local memory) is unchanged.
  - resource contract: A800 checked (86/93 regs, 0 static smem, binary 80,
    5 CTAs/SM).  On sm_90 the compiler reports its own register count and a
    1-KiB static shared reservation; the contract enforced is 128 threads,
    no local memory and binary version 90.  The unchanged
    __launch_bounds__(128, 5) source gives the operand-residency kernel
    94 registers -> 5 resident CTAs/SM (as on A800), but the DRAM-miss
    pipeline kernel 78 registers -> 6 resident CTAs/SM on sm_90 (A800: 93
    registers -> 5).  Both observed H200 counts are enforced and recorded;
    with 6 x 132 = 792 resident CTAs the K2048 (896-CTA) source-group case
    stays two-wave while the K2880 (720-CTA) case becomes one wave.
  - grid 540 -> 660 (5 x 132 SMs), logical bytes follow; hit set 48 MiB.
  - device: one H200 (``--device`` physical index, CUDA_VISIBLE_DEVICES set
    to it) instead of A800 devices 6/7; the A800 "minimum across two idle
    devices" becomes "minimum across two independent clock-accepted
    processes on the owned GPU".
  - clock gate: the SM clock cannot be locked, so nvidia-smi samples
    clocks.sm/utilization.gpu every 50 ms during each capture; a capture with
    any busy (utilization > 0) sample below 1980 MHz is rejected.

Subcommands:
  build   (local)  compile + SASS check into --build-dir
  run     (remote) clock-gated captures into --output-dir
  publish (local)  derive the three H200 references from copied summaries
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


SOURCE = Path(__file__).with_name("sm80_bf16_bm64_operand_residency.cu")
RUNNER = Path(__file__)
ROOT = Path(__file__).resolve().parents[4]
DEVICE_NAME = "NVIDIA H200"
NVCC = Path("/usr/local/cuda/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")
BOOST_MHZ = 1980
TRIAL_LIMIT = 0.10
REQUIRED_ACCEPTED = 2
MAXIMUM_ATTEMPTS = 6
GRID = 660
LOGICAL_BYTES = GRID * 256 * 8192
EXPECTED_CASES = (
    "l2_resident",
    "dram_stream",
    "grouped_fanout_1_n4096",
    "grouped_fanout_1_n7168",
    "grouped_fanout_8_n6144",
    "source_group_multiwave_k2048_n7168_fanout1",
    "source_group_multiwave_k2048_n7168_fanout2",
    "source_group_multiwave_k2048_n7168_fanout4",
    "source_group_multiwave_k2048_n7168",
    "source_group_multiwave_k2880_n5760_fanout1",
    "source_group_multiwave_k2880_n5760_fanout2",
    "source_group_multiwave_k2880_n5760_fanout4",
    "source_group_multiwave_k2880_n5760",
)
EXPECTED_LOGICAL_BYTES = {
    "l2_resident": LOGICAL_BYTES,
    "dram_stream": LOGICAL_BYTES,
    "grouped_fanout_1_n4096": LOGICAL_BYTES,
    "grouped_fanout_1_n7168": LOGICAL_BYTES,
    "grouped_fanout_8_n6144": LOGICAL_BYTES,
    "source_group_multiwave_k2048_n7168_fanout1": 469_762_048,
    "source_group_multiwave_k2048_n7168_fanout2": 469_762_048,
    "source_group_multiwave_k2048_n7168_fanout4": 469_762_048,
    "source_group_multiwave_k2048_n7168": 469_762_048,
    "source_group_multiwave_k2880_n5760_fanout1": 530_841_600,
    "source_group_multiwave_k2880_n5760_fanout2": 530_841_600,
    "source_group_multiwave_k2880_n5760_fanout4": 530_841_600,
    "source_group_multiwave_k2880_n5760": 530_841_600,
}
SOURCE_GROUP_CASES_BY_WEIGHT_STREAMS = {
    1: (
        "source_group_multiwave_k2048_n7168",
        "source_group_multiwave_k2880_n5760",
    ),
    2: (
        "source_group_multiwave_k2048_n7168_fanout4",
        "source_group_multiwave_k2880_n5760_fanout4",
    ),
    4: (
        "source_group_multiwave_k2048_n7168_fanout2",
        "source_group_multiwave_k2880_n5760_fanout2",
    ),
    8: (
        "source_group_multiwave_k2048_n7168_fanout1",
        "source_group_multiwave_k2880_n5760_fanout1",
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _device(device):
    fields = tuple(value.strip() for value in _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,"
        "utilization.gpu,clocks.current.sm,driver_version",
        "--format=csv,noheader,nounits",
    ]).stdout.split(","))
    if len(fields) != 8:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "pci_bus_id": fields[3], "memory_used_mib": int(fields[4]),
        "gpu_utilization_pct": int(fields[5]),
        "sm_clock_mhz": int(fields[6]), "driver_version": fields[7],
    }
    if row["index"] != device or not row["name"].startswith(DEVICE_NAME):
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _capture([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if any(line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
           for line in applications.splitlines()):
        raise RuntimeError("target GPU has a compute process")
    return row


def _clocked(command, environment, device, clock_path):
    """Run one probe process under a 50-ms clock logger (stopped by PID)."""
    with clock_path.open("w", encoding="utf-8") as clock_stream:
        sampler = subprocess.Popen(
            ["nvidia-smi", "-i", str(device),
             "--query-gpu=clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=clock_stream, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.3)
            result = subprocess.run(
                command, text=True, capture_output=True, env=environment)
        finally:
            time.sleep(0.2)
            sampler.terminate()
            sampler.wait()
    samples = []
    for line in clock_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            clock, utilization = (int(item) for item in line.split(","))
            samples.append((clock, utilization))
    busy = [clock for clock, utilization in samples if utilization > 0]
    clock = {
        "samples": len(samples), "busy_samples": len(busy),
        "busy_min_mhz": min(busy) if busy else None,
        "busy_max_mhz": max(busy) if busy else None,
        "busy_median_mhz": statistics.median(busy) if busy else None,
        "accepted": bool(busy) and min(busy) >= BOOST_MHZ,
    }
    return result, clock


def _parse(stdout, trials):
    attributes = None
    pipeline_attributes = None
    rows = []
    for line in stdout.splitlines():
        fields = next(csv.reader([line]))
        if fields[0] == "ATTRIBUTE":
            if len(fields) != 7 or attributes is not None:
                raise ValueError("invalid operand attribute row")
            attributes = tuple(int(value) for value in fields[1:])
        elif fields[0] == "PIPELINE_ATTRIBUTE":
            if len(fields) != 7 or pipeline_attributes is not None:
                raise ValueError("invalid pipeline attribute row")
            pipeline_attributes = tuple(int(value) for value in fields[1:])
        elif fields[:2] in (
                ["DATA", "trial"], ["PIPELINE_DATA", "trial"]):
            continue
        elif fields[0] in ("DATA", "PIPELINE_DATA"):
            if len(fields) != 7:
                raise ValueError("invalid measurement row")
            trial = int(fields[1])
            residency = fields[2]
            backing_bytes = int(fields[3])
            logical_bytes = int(fields[4])
            duration_ms = float(fields[5])
            checksum = float(fields[6])
            if (trial not in range(trials)
                    or residency not in EXPECTED_CASES
                    or backing_bytes <= 0
                    or logical_bytes != EXPECTED_LOGICAL_BYTES[residency]
                    or duration_ms <= 0 or not math.isfinite(checksum)):
                raise ValueError(f"measurement violates contract: {fields}")
            rows.append({
                "trial": trial, "residency": residency,
                "backing_bytes": backing_bytes,
                "logical_bytes": logical_bytes,
                "duration_ms": duration_ms,
                "checksum": checksum,
            })
        elif line:
            raise ValueError(f"unexpected probe output: {line!r}")
    for name, value, resident in (("operand", attributes, 5),
                                  ("pipeline", pipeline_attributes, 6)):
        if (value is None or value[2] != 0 or value[3] != 128
                or value[4] != 90 or value[5] != resident):
            raise ValueError(f"{name} resource envelope changed: {value}")
    expected = len(EXPECTED_CASES) * trials
    if len(rows) != expected or len({
            (row["trial"], row["residency"]) for row in rows}) != expected:
        raise ValueError("probe matrix is incomplete or duplicated")
    return attributes, pipeline_attributes, rows


def _relative_range(values):
    median = statistics.median(values)
    return (max(values) - min(values)) / median


def _report(rows):
    cases = {}
    for name in EXPECTED_CASES:
        selected = [row for row in rows if row["residency"] == name]
        durations = [row["duration_ms"] for row in selected]
        median_ms = statistics.median(durations)
        numerator = selected[0]["logical_bytes"]
        byte_meaning = "combined_a_plus_b_logical_bytes"
        if name.startswith("grouped_"):
            numerator //= 2
            byte_meaning = "strided_b_weight_bytes"
        capacity = numerator / (median_ms * 1.0e-3)
        spread = _relative_range(durations)
        cases[name] = {
            "median_duration_ms": median_ms,
            "trial_duration_ms": durations,
            "relative_range": spread,
            "stable": spread <= TRIAL_LIMIT,
            "service_bytes": numerator,
            "service_byte_meaning": byte_meaning,
            "logical_byte_capacity_per_s": capacity,
        }
    miss_names = ("grouped_fanout_1_n4096", "grouped_fanout_1_n7168")
    source_group = {
        str(streams): min(
            cases[name]["logical_byte_capacity_per_s"] for name in names)
        for streams, names in SOURCE_GROUP_CASES_BY_WEIGHT_STREAMS.items()
    }
    return {
        "stable": all(case["stable"] for case in cases.values()),
        "cases": cases,
        "candidates": {
            "grouped_operand_l2_hit_bytes_per_s":
                cases["l2_resident"]["logical_byte_capacity_per_s"],
            "grouped_weight_dram_miss_bytes_per_s": min(
                cases[name]["logical_byte_capacity_per_s"]
                for name in miss_names),
            "complete_source_group_pipeline_bytes_per_s_by_weight_streams":
                source_group,
        },
    }


def build(arguments):
    output = arguments.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "sm80_bf16_bm64_operand_residency"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3", "--use_fast_math",
        "--generate-code=arch=compute_90,code=sm_90", "--disable-warnings",
        "-cudart", "static", "-o", str(binary), str(SOURCE),
    ]
    _capture(compile_command)
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    if ("LDGSTS" not in sass or "HMMA.16816.F32.BF16" not in sass
            or "LDL" in sass or "STL" in sass):
        raise ValueError("compiled SASS violates the tile pipeline contract")
    sass_path = output / "probe.sass.txt"
    sass_path.write_text(sass, encoding="utf-8")
    (output / "build.json").write_text(json.dumps({
        "compile_command": compile_command,
        "git_commit": _capture(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip(),
        "source": str(SOURCE.relative_to(ROOT)),
        "sha256": {"source": _sha256(SOURCE), "runner": _sha256(RUNNER),
                   "binary": _sha256(binary), "sass": _sha256(sass_path)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(arguments):
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = arguments.binary.resolve()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device)
    command = [str(binary), "0", str(arguments.trials)]
    captures = []
    accepted = 0
    for attempt in range(MAXIMUM_ATTEMPTS):
        if accepted == REQUIRED_ACCEPTED:
            break
        device_before = _device(arguments.device)
        result, clock = _clocked(
            command, environment, arguments.device,
            output / f"capture{attempt}.clock.csv")
        if result.returncode != 0 or result.stderr.strip():
            raise RuntimeError(
                f"probe failed: rc={result.returncode} {result.stderr}")
        (output / f"capture{attempt}.raw.csv").write_text(
            result.stdout, encoding="utf-8")
        attributes, pipeline_attributes, rows = _parse(
            result.stdout, arguments.trials)
        report = _report(rows)
        status = "accepted"
        if not clock["accepted"]:
            status = "rejected_clock"
        elif not report["stable"]:
            status = "rejected_stability"
        accepted += status == "accepted"
        captures.append({
            "capture": attempt, "status": status, "clock": clock,
            "device_before": device_before,
            "resource_contract": {
                "operand": attributes, "pipeline": pipeline_attributes},
            "raw_sha256": _sha256(output / f"capture{attempt}.raw.csv"),
            "report": report,
        })
    summary = {
        "schema": "nvidia_h200_bm64_operand_residency_capture_v1",
        "device": arguments.device, "command": command,
        "binary_sha256": _sha256(binary), "trials": arguments.trials,
        "accepted_captures": accepted,
        "rejected_captures": len(captures) - accepted,
        "captures": captures,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if accepted < REQUIRED_ACCEPTED:
        raise RuntimeError("fewer than two clock-accepted stable captures")


def _contract(resident_ctas_per_sm):
    return {
        "dtype": "bf16", "tile_m": 64, "tile_n": 64, "tile_k": 32,
        "group_size_m": 8, "threads_per_cta": 128,
        "resident_ctas_per_sm": resident_ctas_per_sm,
        "physical_sm_count": 132,
        "pipeline_slots": 2, "dynamic_shared_bytes": 16384,
    }


def _source_group_shapes():
    return [
        {"projection_k": 2048, "projection_n": 7168, "grid_ctas": 896,
         "rounds": 64, "combined_a_plus_b_logical_bytes": 469762048},
        {"projection_k": 2880, "projection_n": 5760, "grid_ctas": 720,
         "rounds": 90, "combined_a_plus_b_logical_bytes": 530841600},
    ]


def publish(arguments):
    run_dir = arguments.run_dir.resolve()
    build_dir = arguments.build_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    build_record = json.loads((build_dir / "build.json").read_text())
    accepted = [c for c in summary["captures"] if c["status"] == "accepted"]
    if len(accepted) != REQUIRED_ACCEPTED:
        raise ValueError("publish requires exactly two accepted captures")
    commit = _capture(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    relative_run = str(run_dir.relative_to(ROOT))
    base = {
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "source_baseline_commit": commit,
        "source": str(SOURCE.relative_to(ROOT)),
        "runner": str(RUNNER.relative_to(ROOT)),
    }
    provenance = {
        "git_commit": commit,
        "build_git_commit": build_record["git_commit"],
        "compile_command": build_record["compile_command"],
        "source_sha256": build_record["sha256"]["source"],
        "runner_sha256": _sha256(RUNNER),
        "binary_sha256": build_record["sha256"]["binary"],
        "sass_sha256": build_record["sha256"]["sass"],
        "raw_dir": relative_run,
        "remote_raw_dir": arguments.remote_run_dir,
        "device_index": summary["device"],
        "command": summary["command"],
        "clock_gate": (
            f"nvidia-smi clocks.sm/utilization.gpu every 50 ms; capture "
            f"rejected when any busy sample < {BOOST_MHZ} MHz"),
        "captures": [
            {"capture": c["capture"], "status": c["status"],
             "clock": c["clock"], "raw_sha256": c["raw_sha256"],
             "resource_contract": c["resource_contract"]}
            for c in summary["captures"]],
        "accepted_captures": summary["accepted_captures"],
        "rejected_captures": summary["rejected_captures"],
        "selection_substitution": (
            "A800 minimum across idle devices 6 and 7 -> minimum across two "
            "independent clock-accepted processes on H200 GPU 1"),
    }
    references = ROOT / "microbench/nvidia_h200/references"

    residency_captures = []
    for c in accepted:
        candidates = c["report"]["candidates"]
        residency_captures.append({
            "path": relative_run, "capture": c["capture"],
            "device_index": summary["device"],
            "device_uuid": c["device_before"]["uuid"],
            "report_sha256": None,
            "raw_sha256": c["raw_sha256"],
            "source_sha256": build_record["sha256"]["source"],
            "sass_sha256": build_record["sha256"]["sass"],
            "grouped_operand_l2_hit_bytes_per_s":
                candidates["grouped_operand_l2_hit_bytes_per_s"],
            "grouped_weight_dram_miss_bytes_per_s":
                candidates["grouped_weight_dram_miss_bytes_per_s"],
            "all_cells_stable_within_10_percent": c["report"]["stable"],
        })
    summary_sha = _sha256(run_dir / "summary.json")
    for entry in residency_captures:
        entry["manifest_sha256"] = summary_sha
        entry["report_sha256"] = summary_sha
    residency = {
        "schema": "nvidia_h200_bm64_operand_residency_reference_v1",
        **base,
        "captures": residency_captures,
        "probe_contract": {
            **_contract(5), "grid_ctas": GRID, "rounds": 256,
            "weight_source_row_strides_elements": [4096, 7168]},
        "selected_services": {
            "selection_rule": (
                "minimum complete-capture capacity across two independent "
                "clock-accepted processes on H200 GPU 1"),
            "grouped_operand_l2_hit_bytes_per_s": min(
                e["grouped_operand_l2_hit_bytes_per_s"]
                for e in residency_captures),
            "grouped_weight_dram_miss_bytes_per_s": min(
                e["grouped_weight_dram_miss_bytes_per_s"]
                for e in residency_captures),
            "weight_miss_stride_rule": (
                "minimum of independently measured N=4096 and N=7168 "
                "source-stride services within each capture"),
        },
        "mechanism": (
            "bm64_double_slot_operand_l2_hit_and_grouped_strided_weight_"
            "dram_miss_service"),
        "allowed_conclusion": (
            "For the source-shaped BM64/BN64/BK32 five-CTA-per-SM pipeline, "
            "L2-resident operand bytes and strided DRAM-miss weight bytes "
            "have distinct full-chip logical-byte service ceilings."),
        "excluded_conclusions": [
            "production operator duration",
            "per-shape residual correction",
            "kernel-name or workload-duration lookup",
            "application outside the exact BM64/BN64/BK32/GSM8 BF16 launch",
        ],
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": True,
        "h200_capture_provenance": provenance,
    }

    def shape_median(c, name):
        return c["report"]["cases"][name]["median_duration_ms"]

    def shape_capacity(c, name):
        return c["report"]["cases"][name]["logical_byte_capacity_per_s"]

    full_captures = []
    for c in accepted:
        full_captures.append({
            "device_index": summary["device"],
            "device_uuid": c["device_before"]["uuid"],
            "path": relative_run, "capture": c["capture"],
            "manifest_sha256": summary_sha, "report_sha256": summary_sha,
            "raw_sha256": c["raw_sha256"],
            "binary_sha256": build_record["sha256"]["binary"],
            "sass_sha256": build_record["sha256"]["sass"],
            "k2048_n7168_median_duration_ms": shape_median(
                c, "source_group_multiwave_k2048_n7168"),
            "k2048_n7168_combined_pipeline_bytes_per_s": shape_capacity(
                c, "source_group_multiwave_k2048_n7168"),
            "k2880_n5760_median_duration_ms": shape_median(
                c, "source_group_multiwave_k2880_n5760"),
            "k2880_n5760_combined_pipeline_bytes_per_s": shape_capacity(
                c, "source_group_multiwave_k2880_n5760"),
        })
    full_values = [
        (e[f"{shape}_combined_pipeline_bytes_per_s"], shape)
        for e in full_captures for shape in ("k2048_n7168", "k2880_n5760")]
    full_selected, full_shape = min(full_values)
    shape_values = [
        e[f"{full_shape}_combined_pipeline_bytes_per_s"]
        for e in full_captures]
    full = {
        "schema": "nvidia_h200_bm64_full_group_pipeline_reference_v2",
        **base,
        "source_sha256": build_record["sha256"]["source"],
        "runner_sha256": _sha256(RUNNER),
        "probe_contract": {
            **_contract(6), "fanout": 8,
            "source_group_cases": _source_group_shapes()},
        "captures": full_captures,
        "selected_service": {
            "selection_rule": (
                "minimum complete source-group capacity across both static "
                "shapes and two independent clock-accepted processes on "
                "H200 GPU 1"),
            "logical_byte_capacity_per_s": full_selected,
            "cross_device_selected_shape_relative_spread": (
                (max(shape_values) - min(shape_values))
                / statistics.median(shape_values)),
        },
        "mechanism": (
            "complete_single_expert_gsm8_bm64_two_slot_copy_and_hmma_"
            "pipeline"),
        "allowed_conclusion": (
            "Complete single-expert GSM8 groups use the measured full-chip "
            "combined operand pipeline service for their L2-hit bytes."),
        "excluded_conclusions": [
            "production operator duration",
            "per-shape residual correction",
            "mixed-expert or partial source groups",
            "application outside the exact BM64/BN64/BK32/GSM8 BF16 launch",
        ],
        "production_or_evaluation_duration_consumed": False,
        "h200_capture_provenance": provenance,
    }

    mixed_captures = []
    for c in accepted:
        by_streams = c["report"]["candidates"][
            "complete_source_group_pipeline_bytes_per_s_by_weight_streams"]
        mixed_captures.append({
            "device_index": summary["device"],
            "device_uuid": c["device_before"]["uuid"],
            "path": relative_run, "capture": c["capture"],
            "manifest_sha256": summary_sha, "report_sha256": summary_sha,
            "raw_sha256": c["raw_sha256"],
            "binary_sha256": build_record["sha256"]["binary"],
            "sass_sha256": build_record["sha256"]["sass"],
            "logical_byte_capacity_per_s_by_weight_streams": {
                key: by_streams[key] for key in ("2", "4", "8")},
        })
    mixed = {
        "schema": "nvidia_h200_bm64_mixed_group_pipeline_reference_v1",
        **base,
        "source_sha256": build_record["sha256"]["source"],
        "runner_sha256_at_capture": _sha256(RUNNER),
        "runner_sha256_current": _sha256(RUNNER),
        "probe_contract": {
            **_contract(6), "weight_streams_per_group": [2, 4, 8],
            "fanout_to_weight_streams": {"4": 2, "2": 4, "1": 8},
            "source_group_shapes": _source_group_shapes()},
        "captures": mixed_captures,
        "selected_services_by_weight_streams": {
            key: min(
                e["logical_byte_capacity_per_s_by_weight_streams"][key]
                for e in mixed_captures)
            for key in ("2", "4", "8")},
        "selection_rule": (
            "for each exact weight-stream topology, minimum complete "
            "source-group capacity across both static shapes and two "
            "independent clock-accepted processes on H200 GPU 1"),
        "mechanism": (
            "complete_gsm8_bm64_two_slot_copy_and_hmma_pipeline_across_"
            "mixed_expert_weight_streams"),
        "allowed_conclusion": (
            "A complete GSM8 group with exactly 2, 4, or 8 expert weight "
            "streams uses its topology-matched full-chip A/B+HMMA pipeline "
            "service when each expert remains within the group or carries "
            "at most one M tile into its adjacent group."),
        "excluded_conclusions": [
            "production operator duration",
            "per-shape residual correction",
            "partial source groups",
            "boundary groups containing an expert M span with more than one "
            "tile carried into another source group",
            "unmeasured 3, 5, 6, or 7 weight-stream groups",
            "application outside the exact BM64/BN64/BK32/GSM8 BF16 launch",
        ],
        "production_or_evaluation_duration_consumed": False,
        "h200_capture_provenance": provenance,
    }
    for name, payload in (
            ("bm64_operand_residency_sm90.json", residency),
            ("bm64_full_group_pipeline_sm90.json", full),
            ("bm64_mixed_group_pipeline_sm90.json", mixed)):
        (references / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--build-dir", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", type=int, required=True)
    run_parser.add_argument("--trials", type=int, default=5)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument("--build-dir", type=Path, required=True)
    publish_parser.add_argument("--remote-run-dir", required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments)
    elif arguments.command == "run":
        if not 3 <= arguments.trials <= 11:
            raise ValueError("trials must be in [3, 11]")
        run(arguments)
    else:
        publish(arguments)


if __name__ == "__main__":
    main()
