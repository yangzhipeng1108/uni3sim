#!/usr/bin/env python3
"""Capture A800 L1 residency for a source-equivalent shared RMSNorm gamma."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("rmsnorm_gamma_residency.cu")
RUNNER = Path(__file__)
UPSTREAM_REFERENCE = (
    ROOT / "microbench/nvidia_a800/references/vllm_rmsnorm_v0_11_0_sm80.json")
OUTPUT_ROOT = (
    ROOT / "profiler_output/microbench/nvidia_a800/rmsnorm_gamma_residency")
NVCC = Path("/usr/local/cuda/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")
NCU = Path("/usr/local/cuda-12.2/bin/ncu")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
KERNEL_NAME = "rmsnorm_gamma_residency_probe"
EXPECTED_UPSTREAM_REFERENCE_SHA256 = (
    "5cfc88825fe890d0881a9d870e04d19dbd59c9c9af5a9d0feddacb8add72a53e")
EXPECTED_UPSTREAM_SOURCE_SHA256 = (
    "fb3b6102d6c0efdf66ed7a2030d0bebf6a25aa46e07fe7274ec545f39e200409")
GRID = 1024
BLOCK = 1024
REGISTERS_PER_THREAD = 24
STATIC_SHARED_BYTES = 168
RESIDENT_CTAS_PER_SM = 2
TRIALS = 3
MAX_RELATIVE_SPREAD = 0.05
MAXIMUM_IDLE_MEMORY_MIB = 8
CASES = (
    {"case_id": 0, "case_name": "shared_gamma_h2048", "hidden": 2048,
     "gamma_scope": "grid_shared"},
    {"case_id": 1, "case_name": "shared_gamma_h8192", "hidden": 8192,
     "gamma_scope": "grid_shared"},
    {"case_id": 2, "case_name": "shared_gamma_h16384", "hidden": 16384,
     "gamma_scope": "grid_shared"},
    {"case_id": 3, "case_name": "private_gamma_h16384_control",
     "hidden": 16384, "gamma_scope": "cta_private"},
    {"case_id": 4, "case_name": "private_gamma_h2048_control",
     "hidden": 2048, "gamma_scope": "cta_private"},
    {"case_id": 5, "case_name": "private_gamma_h8192_control",
     "hidden": 8192, "gamma_scope": "cta_private"},
)
NCU_METRICS = (
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
    "lts__t_requests_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
)
NCU_METRIC_UNITS = {
    NCU_METRICS[0]: "request",
    NCU_METRICS[1]: "byte",
    NCU_METRICS[2]: "byte",
    NCU_METRICS[3]: "request",
    NCU_METRICS[4]: "sector",
}
NATIVE_FIELDS = (
    "device_name", "case_name", "hidden", "grid", "block",
    "input_a_bytes", "input_b_bytes", "unique_gamma_bytes", "output_bytes",
    "private_input_read_bytes", "logical_gamma_read_bytes",
    "logical_read_bytes", "resident_ctas_per_sm", "checksum",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _parse_uint(text, label):
    normalized = text.strip().replace(",", "")
    if re.fullmatch(r"[0-9]+", normalized) is None:
        raise ValueError(f"{label} must be an unsigned integer")
    return int(normalized)


def _parse_dimension(text, label):
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        text.strip())
    if match is None:
        raise ValueError(f"{label} has invalid CUDA dimension syntax")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _expected_geometry(case):
    row_bytes = case["hidden"] * 2
    tensor_bytes = GRID * row_bytes
    gamma_bytes_by_scope = {
        "grid_shared": row_bytes,
        "cta_private": tensor_bytes,
    }
    unique_gamma_bytes = gamma_bytes_by_scope[case["gamma_scope"]]
    private_input_read_bytes = 3 * tensor_bytes
    logical_gamma_read_bytes = tensor_bytes
    return {
        "hidden": case["hidden"],
        "grid": GRID,
        "block": BLOCK,
        "input_a_bytes": tensor_bytes,
        "input_b_bytes": tensor_bytes,
        "unique_gamma_bytes": unique_gamma_bytes,
        "output_bytes": tensor_bytes,
        "private_input_read_bytes": private_input_read_bytes,
        "logical_gamma_read_bytes": logical_gamma_read_bytes,
        "logical_read_bytes": private_input_read_bytes + logical_gamma_read_bytes,
        "resident_ctas_per_sm": RESIDENT_CTAS_PER_SM,
    }


def _validate_upstream_reference():
    if _sha256(UPSTREAM_REFERENCE) != EXPECTED_UPSTREAM_REFERENCE_SHA256:
        raise ValueError("the frozen vLLM RMSNorm reference SHA changed")
    with UPSTREAM_REFERENCE.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    if reference["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("upstream reference consumed a forbidden duration")
    source = reference["upstream_sources"]
    if len(source) != 1:
        raise ValueError("upstream RMSNorm source identity is ambiguous")
    if source[0]["path"] != "csrc/layernorm_kernels.cu":
        raise ValueError("upstream RMSNorm source path changed")
    if source[0]["sha256"] != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("upstream RMSNorm source SHA changed")
    kernel = reference["kernel"]
    expected = {
        "registers_per_thread_sm80": REGISTERS_PER_THREAD,
        "static_shared_memory_bytes_per_cta_sm80": STATIC_SHARED_BYTES,
        "maximum_threads_per_cta": BLOCK,
        "resident_ctas_per_sm": RESIDENT_CTAS_PER_SM,
    }
    for name, value in expected.items():
        if kernel[name] != value:
            raise ValueError(f"upstream RMSNorm {name} changed")
    facts = reference["source_facts"]
    if facts["input_passes"] != 2 or facts["block_rule"] != "min_hidden_1024":
        raise ValueError("upstream RMSNorm input or block source fact changed")
    return reference


def _parse_native_stdout(text, case):
    rows = tuple(csv.reader(text.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(NATIVE_FIELDS):
        raise ValueError("native probe must emit one complete row")
    row = dict(zip(NATIVE_FIELDS, rows[0]))
    if row["device_name"] != DEVICE_NAME or row["case_name"] != case["case_name"]:
        raise ValueError("native device or case identity changed")
    integer_fields = NATIVE_FIELDS[2:]
    parsed = {name: _parse_uint(row[name], name) for name in integer_fields}
    expected = _expected_geometry(case)
    for name, value in expected.items():
        if parsed[name] != value:
            raise ValueError(f"native {name} disagrees with frozen geometry")
    return {"device_name": row["device_name"], "case_name": row["case_name"],
            **parsed}


def _parse_resource_usage(text):
    pattern = re.compile(
        rf"Function {KERNEL_NAME}:\s*\n\s*REG:([0-9]+) STACK:([0-9]+) "
        r"SHARED:([0-9]+) LOCAL:([0-9]+)")
    matches = pattern.findall(text)
    if matches != [(str(REGISTERS_PER_THREAD), "0", str(STATIC_SHARED_BYTES), "0")]:
        raise ValueError("compiled kernel resource contract changed")
    return {
        "registers_per_thread": REGISTERS_PER_THREAD,
        "stack_bytes": 0,
        "static_shared_memory_bytes": STATIC_SHARED_BYTES,
        "local_bytes": 0,
        "resident_ctas_per_sm": RESIDENT_CTAS_PER_SM,
    }


def _validate_sass(text):
    marker = f"Function : {KERNEL_NAME}"
    if text.count(marker) != 1:
        raise ValueError("SASS does not contain exactly one probe function")
    body = text.split(marker, 1)[1]
    loads = tuple(re.finditer(r"LDG\.E\.U16\.STRONG\.SM", body))
    barriers = tuple(re.finditer(r"BAR\.SYNC", body))
    stores = tuple(re.finditer(r"STG\.E\.U16", body))
    if len(loads) != 4 or len(barriers) != 1 or len(stores) != 1:
        raise ValueError("SASS BF16 load/barrier/store instruction count changed")
    barrier = barriers[0].start()
    if not (loads[1].start() < barrier < loads[2].start() < loads[3].start()
            < stores[0].start()):
        raise ValueError("SASS source-equivalent stream order changed")
    return {
        "cached_bf16_load_opcode": "LDG.E.U16.STRONG.SM",
        "cached_bf16_load_static_sites": len(loads),
        "barrier_static_sites": len(barriers),
        "bf16_store_static_sites": len(stores),
        "stream_order": (
            "private_input_a->private_input_b->barrier->"
            "private_input_a_plus_gamma->store"),
    }


def _parse_ncu_csv(text):
    lines = text.splitlines()
    header_index = None
    required = {"Kernel Name", "Grid Size", "Block Size", *NCU_METRICS}
    for index, line in enumerate(lines):
        fields = tuple(csv.reader([line]))[0]
        if required <= set(fields):
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV lacks the required wide metric schema")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or duplicated")
    rows = tuple(reader)
    unit_rows = tuple(row for row in rows if not row["Kernel Name"])
    data_rows = tuple(row for row in rows if row["Kernel Name"])
    if len(unit_rows) != 1 or len(data_rows) != 1:
        raise ValueError("NCU CSV requires one unit row and one data row")
    unit_row = unit_rows[0]
    data_row = data_rows[0]
    if data_row["Kernel Name"] != KERNEL_NAME:
        raise ValueError("NCU selected the wrong kernel")
    if _parse_dimension(data_row["Grid Size"], "NCU grid") != (GRID, 1, 1):
        raise ValueError("NCU grid changed")
    if _parse_dimension(data_row["Block Size"], "NCU block") != (BLOCK, 1, 1):
        raise ValueError("NCU block changed")
    metrics = {}
    for metric in NCU_METRICS:
        if unit_row[metric] != NCU_METRIC_UNITS[metric]:
            raise ValueError(f"NCU unit changed for {metric}")
        metrics[metric] = _parse_uint(data_row[metric], metric)
    if metrics[NCU_METRICS[1]] % 32 or metrics[NCU_METRICS[2]] % 32:
        raise ValueError("NCU byte counters violate sector granularity")
    if metrics[NCU_METRICS[2]] != metrics[NCU_METRICS[4]] * 32:
        raise ValueError("L1 xbar bytes and L2 sectors do not conserve")
    return metrics


def _relative_spread(values):
    median = sorted(values)[len(values) // 2]
    if median == 0:
        return 0.0 if max(values) == 0 else math.inf
    return (max(values) - min(values)) / median


def _summarize(case, native, trials):
    summary = {"case_id": case["case_id"], "case_name": case["case_name"],
               "hidden": case["hidden"], "gamma_scope": case["gamma_scope"],
               **_expected_geometry(case)}
    medians = {}
    spreads = {}
    for metric in NCU_METRICS:
        values = tuple(trial[metric] for trial in trials)
        spread = _relative_spread(values)
        if spread > MAX_RELATIVE_SPREAD:
            raise ValueError(f"unstable NCU metric {metric}: spread={spread}")
        medians[metric] = sorted(values)[len(values) // 2]
        spreads[metric] = spread
    if medians[NCU_METRICS[1]] != native["logical_read_bytes"]:
        raise ValueError("L1 requested bytes disagree with source byte accounting")
    summary["median_metrics"] = medians
    summary["relative_spread"] = spreads
    summary["l1_hit_bytes"] = medians[NCU_METRICS[1]] - medians[NCU_METRICS[2]]
    summary["l1_hit_fraction"] = (
        summary["l1_hit_bytes"] / medians[NCU_METRICS[1]])
    summary["l2_read_bytes"] = medians[NCU_METRICS[4]] * 32
    return summary


def _device(device_id):
    query = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    fields = tuple(field.strip() for field in query.stdout.strip().split(","))
    if len(fields) != 5 or fields[1] != DEVICE_NAME:
        raise ValueError("selected device is not the expected A800")
    memory_used_mib = _parse_uint(fields[3], "device memory.used")
    if memory_used_mib > MAXIMUM_IDLE_MEMORY_MIB:
        raise RuntimeError("selected device has allocated memory")
    if _parse_uint(fields[4], "device utilization") != 0:
        raise RuntimeError("selected device is active")
    processes = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if processes.stdout.strip():
        raise RuntimeError("selected device has compute processes")
    return {"physical_index": _parse_uint(fields[0], "device index"),
            "name": fields[1], "uuid": fields[2],
            "memory_used_mib_before_capture": memory_used_mib,
            "utilization_percent_before_capture": 0}


def _compile(output_dir):
    binary = output_dir / "rmsnorm_gamma_residency"
    command = [str(NVCC), "-std=c++17", "-O3", "-lineinfo", "-arch=sm_80",
               "-Xptxas=-v", "-o", str(binary), str(SOURCE)]
    completed = _run(command)
    (output_dir / "nvcc.stdout").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "nvcc.stderr").write_text(completed.stderr, encoding="utf-8")
    resource = _run([str(CUOBJDUMP), "--dump-resource-usage", str(binary)])
    sass = _run([str(CUOBJDUMP), "--dump-sass", str(binary)])
    (output_dir / "resource.txt").write_text(resource.stdout, encoding="utf-8")
    (output_dir / "sass.txt").write_text(sass.stdout, encoding="utf-8")
    return binary, command, _parse_resource_usage(resource.stdout), _validate_sass(sass.stdout)


def _capture(args):
    _validate_upstream_reference()
    device = _device(args.device)
    run_id = args.run_id
    if re.fullmatch(r"[A-Za-z0-9._-]{1,96}", run_id) is None:
        raise ValueError("run-id has invalid syntax")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    binary, compile_command, resource, sass = _compile(output_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    commands = [compile_command]
    summaries = []
    for case in CASES:
        native_command = [str(binary), str(case["case_id"])]
        native_result = _run(native_command, environment=environment)
        native = _parse_native_stdout(native_result.stdout, case)
        trials = []
        for trial in range(TRIALS):
            command = [
                str(NCU), "--csv", "--page", "raw", "--kernel-name",
                f"regex:^{KERNEL_NAME}$", "--launch-count", "1", "--metrics",
                ",".join(NCU_METRICS), str(binary), str(case["case_id"]),
            ]
            completed = _run(command, environment=environment)
            stem = f"case_{case['case_id']}_trial_{trial}"
            (output_dir / f"{stem}.stdout.csv").write_text(
                completed.stdout, encoding="utf-8")
            (output_dir / f"{stem}.stderr").write_text(
                completed.stderr, encoding="utf-8")
            trials.append(_parse_ncu_csv(completed.stdout))
            commands.append(command)
        summaries.append(_summarize(case, native, tuple(trials)))
    result = {
        "schema": "nvidia_a800_rmsnorm_gamma_l1_residency_v1",
        "hardware_id": "nvidia_a800",
        "evidence_kind": "independent_l1_residency_mechanism_microbenchmark",
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
        "capture": {
            "owner": args.owner, "device": device,
            "baseline_commit": args.baseline, "runner_pid": os.getpid(),
            "started_unix_seconds": args.started_unix_seconds,
            "completed_unix_seconds": int(time.time()),
            "output_directory": str(output_dir.relative_to(ROOT)),
            "commands": commands,
        },
        "source_identity": {
            "probe_source": str(SOURCE.relative_to(ROOT)),
            "probe_source_sha256": _sha256(SOURCE),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "upstream_reference": str(UPSTREAM_REFERENCE.relative_to(ROOT)),
            "upstream_reference_sha256": _sha256(UPSTREAM_REFERENCE),
            "upstream_source_sha256": EXPECTED_UPSTREAM_SOURCE_SHA256,
        },
        "static_contract": {"resource": resource, "sass": sass},
        "trials_per_cell": TRIALS,
        "maximum_relative_spread": MAX_RELATIVE_SPREAD,
        "cases": summaries,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.started_unix_seconds = int(time.time())
    _capture(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
