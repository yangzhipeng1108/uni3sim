#!/usr/bin/env python3
"""Fail closed report for the full-A narrow MatMulV2 Scalar control."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


K_VALUES = (2048, 4096)
MODES = ("dynamic_m1", "static_m16")
PIPE_COLUMNS = {
    "block_id",
    "sub_block_id",
    "aic_time(us)",
    "aic_total_cycles",
    "aic_cube_time(us)",
    "aic_cube_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aic_mte3_time(us)",
    "aic_mte3_ratio",
    "aic_mte3_active_bw(GB/s)",
    "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio",
    "aic_fixpipe_active_bw(GB/s)",
    "aic_icache_miss_rate",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte2_active_bw(GB/s)",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_mte3_active_bw(GB/s)",
    "aiv_icache_miss_rate",
    "aic_mte1_active_bw(GB/s)",
    "aic_mte2_active_bw(GB/s)",
}
OP_COLUMNS = {
    "Op Name",
    "Op Type",
    "Task Duration(us)",
    "Block Dim",
    "Mix Block Dim",
    "Device Id",
    "Pid",
    "Current Freq",
    "Rated Freq",
}
OP_OPTIONAL_EMPTY_COLUMN = ""
SHA256 = re.compile(r"[0-9a-f]{64}")
TOPOLOGIES = {
    "one_n": {
        "kernel_name": "matmul_v2_full_a_scalar_control_0",
        "active_aic_workers": 24,
        "base_n": 176,
        "worker_local_n_tiles": 1,
        "source_files": (
            "CMakeLists.txt",
            "matmul_v2_full_a_scalar_control_kernel.cpp",
            "matmul_v2_full_a_scalar_control_main.cpp",
            "matmul_v2_full_a_scalar_control_runner.py",
        ),
    },
    "five_n": {
        "kernel_name": "matmul_v2_full_a_five_n_scalar_control_0",
        "active_aic_workers": 23,
        "base_n": 256,
        "worker_local_n_tiles": 5,
        "capture_runner_file": (
            "matmul_v2_full_a_five_n_scalar_pipe_capture_runner.py"),
        "source_files": (
            "CMakeLists.txt",
            "matmul_v2_full_a_five_n_scalar_control_kernel.cpp",
            "matmul_v2_full_a_five_n_scalar_control_main.cpp",
            "matmul_v2_full_a_five_n_scalar_control_runner.py",
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(root: Path, source_files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in source_files:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _number(row: dict[str, str], name: str, path: Path) -> float:
    try:
        value = float(row[name])
    except ValueError as error:
        raise ValueError(f"{path}: {name} must be numeric") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{path}: {name} must be finite and non-negative")
    return value


def _integer(row: dict[str, str], name: str, path: Path) -> int:
    value = row[name]
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"{path}: {name} must be a non-negative integer")
    return int(value)


def _one_op_row(directory: Path, expected_workers: int,
                expected_kernel_name: str, expected_device_id: int) -> None:
    files = sorted(directory.glob("OpBasicInfo_*.csv"))
    if len(files) != 1:
        raise ValueError(f"{directory}: expected exactly one OpBasicInfo CSV")
    with files[0].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        fieldname_set = set(fieldnames)
        if len(fieldnames) != len(fieldname_set):
            raise ValueError(f"{files[0]}: duplicate OpBasicInfo column")
        if fieldname_set not in (
                OP_COLUMNS, OP_COLUMNS | {OP_OPTIONAL_EMPTY_COLUMN}):
            raise ValueError(f"{files[0]}: OpBasicInfo schema changed")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"{files[0]}: expected exactly one operation")
    row = rows[0]
    if OP_OPTIONAL_EMPTY_COLUMN in fieldname_set:
        if row[OP_OPTIONAL_EMPTY_COLUMN] != "":
            raise ValueError(f"{files[0]}: optional OpBasicInfo column is not empty")
        del row[OP_OPTIONAL_EMPTY_COLUMN]
    if set(row) != OP_COLUMNS:
        raise ValueError(f"{files[0]}: OpBasicInfo row schema changed")
    if (row["Op Name"] != expected_kernel_name
            or row["Op Type"] != "cube"
            or _integer(row, "Block Dim", files[0]) != expected_workers
            or _integer(row, "Device Id", files[0]) != expected_device_id):
        raise ValueError(f"{files[0]}: unexpected full-A Scalar operation")


def _launch_scalar_median(path: Path, expected_workers: int,
                          expected_kernel_name: str,
                          expected_device_id: int) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != PIPE_COLUMNS:
            raise ValueError(f"{path}: PipeUtilization schema changed")
        rows = list(reader)
    if len(rows) != expected_workers:
        raise ValueError(f"{path}: unexpected AIC worker count")
    block_ids = {_integer(row, "block_id", path) for row in rows}
    if block_ids != set(range(expected_workers)):
        raise ValueError(f"{path}: AIC block identities changed")
    if {row["sub_block_id"] for row in rows} != {"cube0"}:
        raise ValueError(f"{path}: unexpected AIC sub-block identity")
    scalar_values = [_number(row, "aic_scalar_time(us)", path) for row in rows]
    for row in rows:
        if _number(row, "aic_scalar_time(us)", path) > _number(
                row, "aic_time(us)", path):
            raise ValueError(f"{path}: Scalar pipe exceeds AIC elapsed time")
    _one_op_row(path.parent, expected_workers, expected_kernel_name,
                expected_device_id)
    return statistics.median(scalar_values)


def _capture_summary(directory: Path, profiler_log: Path,
                     expected_workers: int, expected_launches: int,
                     expected_kernel_name: str,
                     expected_device_id: int,
                     capture_manifest: dict | None = None) -> dict:
    if not directory.is_dir():
        raise ValueError(f"{directory}: PipeUtilization directory is absent")
    if not profiler_log.is_file() or profiler_log.stat().st_size == 0:
        raise ValueError(f"{profiler_log}: msprof log is absent or empty")
    files = sorted(directory.rglob("PipeUtilization_*.csv"))
    if len(files) != expected_launches:
        raise ValueError(f"{directory}: unexpected PipeUtilization launch count")
    medians_us = [
        _launch_scalar_median(path, expected_workers, expected_kernel_name,
                              expected_device_id)
        for path in files]
    summary = {
        "pipe_utilization_paths": [str(path) for path in files],
        "pipe_utilization_sha256": [_sha256(path) for path in files],
        "profiler_log_path": str(profiler_log),
        "profiler_log_sha256": _sha256(profiler_log),
        "aic_scalar_worker_median_us_by_launch": medians_us,
        "aic_scalar_median_of_launch_medians_ns": (
            statistics.median(medians_us) * 1000.0),
    }
    if capture_manifest is not None:
        summary["capture_manifest_path"] = capture_manifest["path"]
        summary["capture_manifest_sha256"] = capture_manifest["sha256"]
    return summary


def _five_n_capture_manifest(
        directory: Path, profiler_log: Path, root: Path, mode: str,
        projection_k: int, expected_launches: int, baseline: str,
        binary_sha256: str, source_snapshot_sha256: str,
        physical_device_id: int, capture_runner_file: str) -> dict:
    path = directory / "capture_manifest.json"
    if not path.is_file():
        raise ValueError(f"{path}: five-N capture manifest is absent")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: capture manifest is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: capture manifest must be an object")
    expected_fields = {
        "schema_version", "hardware_id", "topology", "owner",
        "measurement_baseline_commit", "physical_device_id", "mode",
        "logical_m", "projection_k", "static_no_tail", "run_id",
        "expected_launches", "binary_sha256", "source_snapshot_sha256",
        "capture_runner_sha256", "msprof_command", "profiler_log",
        "production_or_evaluation_duration_consumed",
    }
    if set(value) != expected_fields:
        raise ValueError(f"{path}: capture manifest schema changed")
    mode_values = {
        "dynamic_m1": (1, 0),
        "static_m16": (16, 1),
    }
    logical_m, static_no_tail = mode_values[mode]
    expected = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_full_a_scalar_pipe_capture.v1"),
        "hardware_id": "ascend_910b",
        "topology": "five_n",
        "owner": "capture-owner-3",
        "measurement_baseline_commit": baseline,
        "physical_device_id": physical_device_id,
        "mode": mode,
        "logical_m": logical_m,
        "projection_k": projection_k,
        "static_no_tail": static_no_tail,
        "expected_launches": expected_launches,
        "binary_sha256": binary_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "profiler_log": "msprof.log",
        "production_or_evaluation_duration_consumed": False,
    }
    for name, expected_value in expected.items():
        if value[name] != expected_value:
            raise ValueError(f"{path}: capture manifest {name} changed")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,96}", value["run_id"]) is None:
        raise ValueError(f"{path}: capture manifest run_id is invalid")
    capture_runner = root / capture_runner_file
    if value["capture_runner_sha256"] != _sha256(capture_runner):
        raise ValueError(f"{path}: capture runner source changed")
    command = value["msprof_command"]
    if (not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)
            or command[:2] != ["msprof", "op"]
            or "--aic-metrics=PipeUtilization" not in command
            or f"--launch-count={expected_launches}" not in command
            or "--launch-skip-before-match=3" not in command
            or not any(item.startswith("--output=") for item in command)
            or not any(item.startswith("--application=") for item in command)):
        raise ValueError(f"{path}: capture manifest msprof command changed")
    if profiler_log.resolve() != (directory / value["profiler_log"]).resolve():
        raise ValueError(f"{path}: reporter log does not match capture manifest")
    return {
        "path": str(path),
        "sha256": _sha256(path),
    }


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError as error:
        raise ValueError(f"{path}: artifact must be inside this worktree") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--dynamic-k2048-pipe-dir", required=True, type=Path)
    parser.add_argument("--static-k2048-pipe-dir", required=True, type=Path)
    parser.add_argument("--dynamic-k4096-pipe-dir", required=True, type=Path)
    parser.add_argument("--static-k4096-pipe-dir", required=True, type=Path)
    parser.add_argument("--dynamic-k2048-profiler-log", required=True, type=Path)
    parser.add_argument("--static-k2048-profiler-log", required=True, type=Path)
    parser.add_argument("--dynamic-k4096-profiler-log", required=True, type=Path)
    parser.add_argument("--static-k4096-profiler-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--cann-source-sha256", required=True)
    parser.add_argument("--expected-source-snapshot", required=True)
    parser.add_argument("--physical-device-id", required=True, type=int)
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES),
                        default="one_n")
    parser.add_argument("--expected-launches", type=int, default=3)
    args = parser.parse_args()
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline) is None:
        parser.error("--baseline must be a lowercase commit")
    for name, value in (
            ("--binary-sha256", args.binary_sha256),
            ("--cann-source-sha256", args.cann_source_sha256),
            ("--expected-source-snapshot", args.expected_source_snapshot)):
        if SHA256.fullmatch(value) is None:
            parser.error(f"{name} must be a lowercase SHA256")
    if args.physical_device_id < 0:
        parser.error("--physical-device-id must be non-negative")
    if args.expected_launches <= 0:
        parser.error("--expected-launches must be positive")
    if args.output.exists():
        raise FileExistsError("report output already exists")

    root = Path(__file__).resolve().parent
    topology = TOPOLOGIES[args.topology]
    source_files = topology["source_files"]
    if _source_snapshot(root, source_files) != args.expected_source_snapshot:
        raise ValueError("full-A Scalar source snapshot differs from expectation")
    repository_root = root.parents[4]
    locations = {
        ("dynamic_m1", 2048): (
            args.dynamic_k2048_pipe_dir, args.dynamic_k2048_profiler_log),
        ("static_m16", 2048): (
            args.static_k2048_pipe_dir, args.static_k2048_profiler_log),
        ("dynamic_m1", 4096): (
            args.dynamic_k4096_pipe_dir, args.dynamic_k4096_profiler_log),
        ("static_m16", 4096): (
            args.static_k4096_pipe_dir, args.static_k4096_profiler_log),
    }
    if len({path.resolve() for path, _ in locations.values()}) != len(locations):
        raise ValueError("each full-A Scalar mode/K capture needs its own directory")
    if len({path.resolve() for _, path in locations.values()}) != len(locations):
        raise ValueError("each full-A Scalar mode/K capture needs its own profiler log")
    captures = {}
    for mode in MODES:
        for projection_k in K_VALUES:
            directory, profiler_log = locations[(mode, projection_k)]
            capture_manifest = None
            if args.topology == "five_n":
                capture_manifest = _five_n_capture_manifest(
                    directory, profiler_log, root, mode, projection_k,
                    args.expected_launches, args.baseline, args.binary_sha256,
                    args.expected_source_snapshot, args.physical_device_id,
                    topology["capture_runner_file"])
            summary = _capture_summary(
                directory, profiler_log,
                expected_workers=topology["active_aic_workers"],
                expected_launches=args.expected_launches,
                expected_kernel_name=topology["kernel_name"],
                expected_device_id=args.physical_device_id,
                capture_manifest=capture_manifest)
            path_names = ["pipe_utilization_paths", "profiler_log_path"]
            if args.topology == "five_n":
                path_names.append("capture_manifest_path")
            for name in path_names:
                value = summary[name]
                if isinstance(value, list):
                    summary[name] = [
                        _repository_relative(Path(item), repository_root)
                        for item in value]
                else:
                    summary[name] = _repository_relative(
                        Path(value), repository_root)
            captures[f"{mode}_k{projection_k}"] = summary

    controls = {}
    for projection_k in K_VALUES:
        dynamic = captures[f"dynamic_m1_k{projection_k}"]
        static = captures[f"static_m16_k{projection_k}"]
        dynamic_runs = dynamic["aic_scalar_worker_median_us_by_launch"]
        static_runs = static["aic_scalar_worker_median_us_by_launch"]
        deltas_ns = [
            (dynamic_value - static_value) * 1000.0
            for dynamic_value, static_value in zip(dynamic_runs, static_runs)
        ]
        controls[f"k{projection_k}"] = {
            "dynamic_minus_static_scalar_ns_by_launch": deltas_ns,
            "dynamic_minus_static_scalar_median_ns": statistics.median(
                deltas_ns),
            "dynamic_scalar_median_ns": dynamic[
                "aic_scalar_median_of_launch_medians_ns"],
            "static_scalar_median_ns": static[
                "aic_scalar_median_of_launch_medians_ns"],
        }

    payload = {
        "schema_version": (
            "midlayer.ascend_910b.aic_matmul_v2_full_a_scalar_control.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_matmul_v2_full_a_dynamic_m_scalar_control",
        "duration_unit": (
            f"ns_per_{topology['active_aic_workers']}aic_scalar_pipe_program"),
        "source_program": {
            "dtype": "bf16",
            "active_aic_workers": topology["active_aic_workers"],
            "base_tile_m": 16,
            "base_tile_n": topology["base_n"],
            "base_tile_k": 64,
            "projection_k_values": list(K_VALUES),
            "a1_buffer_slots": 1,
            "b1_buffer_slots": 2,
            "a1_descriptor_rule": "one full-A descriptor per worker",
            "b1_descriptor_rule": "one 256-column descriptor per B1 slot",
            "worker_local_n_tiles": topology["worker_local_n_tiles"],
            "dynamic_logical_m": 1,
            "static_logical_m": 16,
        },
        "measurement": {
            "profiler": "msprof op PipeUtilization",
            "statistic": (
                "median_of_"
                f"{topology['active_aic_workers']}_AIC_Scalar_pipe_values_"
                "per_launch_then_median_of_launch_medians"),
            "independent_launches_per_mode": args.expected_launches,
            "captures": captures,
            "dynamic_static_controls": controls,
        },
        "forward_eligibility": {
            "status": "candidate_not_published",
            "allowed_signal": (
                "only the AIC Scalar-pipe dynamic-minus-static control; no "
                "whole-program, MTE2, MTE1, Cube, FixPipe, or task duration "
                "is parsed or may enter Forward"),
            "next_required_check": (
                "bind any stable Scalar excess to the CANN source's serial "
                "dynamic-M control placement and compare it with the existing "
                "source-defined Scalar service before changing the predictor"),
        },
        "provenance": {
            "measurement_baseline_commit": args.baseline,
            "physical_device_id": args.physical_device_id,
            "container": args.container_name,
            "source_snapshot_sha256": args.expected_source_snapshot,
            "source_files_sha256": {
                name: _sha256(root / name) for name in source_files},
            "reporter_sha256": _sha256(Path(__file__).resolve()),
            "binary_sha256": args.binary_sha256,
            "cann_source_sha256": args.cann_source_sha256,
        },
        "production_or_evaluation_duration_consumed": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
