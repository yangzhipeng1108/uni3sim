#!/usr/bin/env python3
"""Capture frozen A800 FA cases as NSYS windows plus NCU raw-page evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_audit import (
    build_a800_counter_candidate,
    build_a800_timing_audit,
    write_json_new,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_capture_inputs(path):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {
        "schema_version", "capture_kind", "manifest_path", "manifest_sha256",
        "hardware_id", "case_count", "cases",
    }
    if set(manifest) != required:
        raise ValueError("A800 capture-input manifest fields are not exact")
    if manifest["hardware_id"] != "nvidia_a800":
        raise ValueError("A800 full runner received a non-A800 input manifest")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != manifest["case_count"]:
        raise ValueError("A800 capture-input manifest case count is invalid")
    ids = []
    for item in cases:
        if not isinstance(item, dict):
            raise TypeError("A800 capture-input case must be a mapping")
        for field in ("case_id", "workload_digest", "capture_metadata", "runner",
                      "runner_arguments"):
            if field not in item:
                raise ValueError(f"A800 capture-input case lacks {field!r}")
        if not isinstance(item["case_id"], str) or not item["case_id"]:
            raise ValueError("A800 capture-input case_id must be non-empty")
        if not isinstance(item["runner_arguments"], list) or any(
                not isinstance(value, str) for value in item["runner_arguments"]):
            raise TypeError("A800 runner arguments must be a string list")
        if not Path(item["capture_metadata"]).is_file():
            raise FileNotFoundError(
                f"A800 capture metadata does not exist: {item['capture_metadata']}")
        ids.append(item["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("A800 capture-input case IDs are not unique")
    return manifest


def _run(command, *, log_path, environment):
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, text=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"A800 capture command failed: pid={process.pid}, log={log_path}")
    return process.pid


def _run_stdout_to_file(command, *, stdout_path, log_path, environment):
    """Run an export whose stdout is itself a raw evidence artifact."""
    with log_path.open("x", encoding="utf-8") as log_stream:
        log_stream.write(json.dumps(command) + "\n")
        log_stream.flush()
        with stdout_path.open("x", encoding="utf-8") as stdout_stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stdout_stream,
                stderr=log_stream, text=True)
            return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"A800 capture command failed: pid={process.pid}, log={log_path}")
    return process.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--nsys", type=Path, default=shutil.which("nsys"))
    parser.add_argument("--ncu", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument(
        "--runner-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--baseline-commit", default="0c530a00a9f7c1135210e99b843cce3f8250e3ac")
    parser.add_argument("--skip-case", action="append", default=[])
    args = parser.parse_args()

    if args.nsys is None:
        parser.error("--nsys is required when nsys is not available on PATH")
    input_manifest = _load_capture_inputs(args.capture_input_manifest)
    if args.output_root.exists() or args.audit_root.exists():
        raise FileExistsError("A800 full-capture output and audit roots must be new")
    if not args.source_snapshot.is_file():
        raise FileNotFoundError(f"A800 source snapshot does not exist: {args.source_snapshot}")
    if not args.nsys.is_file() or not os.access(args.nsys, os.X_OK):
        raise FileNotFoundError(f"A800 NSYS executable is unavailable: {args.nsys}")
    if not args.ncu.is_file() or not os.access(args.ncu, os.X_OK):
        raise FileNotFoundError(f"A800 NCU executable is unavailable: {args.ncu}")
    if not args.runner_python.is_file() or not os.access(args.runner_python, os.X_OK):
        raise FileNotFoundError(
            f"A800 runner Python executable is unavailable: {args.runner_python}")
    if args.device < 0:
        raise ValueError("A800 device must be non-negative")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit) is None:
        raise ValueError("A800 baseline-commit must be a 40-character lowercase SHA")

    source_sha256 = _sha256(args.source_snapshot)
    skip_cases = set(args.skip_case)
    known_cases = {item["case_id"] for item in input_manifest["cases"]}
    unknown_skips = skip_cases - known_cases
    if unknown_skips:
        raise ValueError(f"A800 skip-case is absent from the manifest: {sorted(unknown_skips)}")
    args.output_root.mkdir(parents=True)
    args.audit_root.mkdir(parents=True)
    shutil.copyfile(args.source_snapshot, args.output_root / "source_snapshot.tar")
    write_json_new(args.output_root / "source_snapshot.json", {
        "sha256": source_sha256,
        "source_snapshot": str(args.source_snapshot),
    })
    write_json_new(args.output_root / "capture_execution.json", {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "owner": "fa_serving_coverage_20260915",
        "device": args.device,
        "baseline_commit": args.baseline_commit,
        "runner_python": str(args.runner_python),
        "nsys": str(args.nsys),
        "ncu": str(args.ncu),
        "source_snapshot_sha256": source_sha256,
    })
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    records = []
    for item in input_manifest["cases"]:
        case_id = item["case_id"]
        if case_id in skip_cases:
            records.append({"case_id": case_id, "status": "skipped"})
            continue
        case_root = args.output_root / case_id
        golden_root = case_root / "golden"
        counter_root = case_root / "counter"
        log_root = case_root / "logs"
        candidate_root = case_root / "candidate"
        golden_root.mkdir(parents=True)
        counter_root.mkdir()
        log_root.mkdir()
        candidate_root.mkdir()
        runner = ROOT / item["runner"]
        if not runner.is_file():
            raise FileNotFoundError(f"A800 runner does not exist: {runner}")
        runner_command = [
            str(args.runner_python), str(runner), *item["runner_arguments"]]
        sqlite_paths = []
        timing_pids = []
        for window in range(1, 4):
            prefix = golden_root / f"nsys_window_{window}"
            pid = _run([
                str(args.nsys), "profile", "--force-overwrite=true",
                "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                "--export=sqlite", "-o", str(prefix), *runner_command,
            ], log_path=log_root / f"nsys_window_{window}.log",
                environment=environment)
            timing_pids.append(pid)
            sqlite_paths.append(prefix.with_suffix(".sqlite"))
        timing = build_a800_timing_audit(
            case_id=case_id, workload_digest=item["workload_digest"],
            nsys_sqlite_paths=sqlite_paths,
            source_snapshot_sha256=source_sha256)
        timing_path = args.audit_root / f"{case_id}.timing.json"
        write_json_new(timing_path, timing)
        if timing["status"] != "pass":
            raise RuntimeError(f"A800 timing stability failed: {timing_path}")
        report_prefix = counter_root / case_id
        ncu_pid = _run([
            str(args.ncu), "--target-processes", "all", "--set", "full",
            "--nvtx", "--nvtx-include", f"{case_id}/", "--force-overwrite",
            "--export", str(report_prefix), *runner_command,
        ], log_path=log_root / "ncu.log", environment=environment)
        report_path = report_prefix.with_suffix(".ncu-rep")
        raw_path = counter_root / f"{case_id}.raw.csv"
        _run_stdout_to_file([
            str(args.ncu), "-i", str(report_path), "--csv", "--page", "raw",
        ], stdout_path=raw_path, log_path=log_root / "ncu_export.log",
           environment=environment)
        candidate = build_a800_counter_candidate(
            case_id=case_id, workload_digest=item["workload_digest"],
            raw_csv_path=raw_path, nsys_sqlite_paths=sqlite_paths,
            source_snapshot_sha256=source_sha256)
        candidate_path = candidate_root / f"{case_id}.counter.json"
        write_json_new(candidate_path, candidate)
        record = {
            "case_id": case_id,
            "workload_digest": item["workload_digest"],
            "status": "pass",
            "timing_pids": timing_pids,
            "ncu_pid": ncu_pid,
            "timing_audit": str(timing_path),
            "counter_candidate": str(candidate_path),
        }
        write_json_new(case_root / "completion.json", record)
        records.append(record)
    write_json_new(args.audit_root / "completion_manifest.json", {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_a800_completion",
        "hardware_id": "nvidia_a800",
        "baseline_commit": args.baseline_commit,
        "device": args.device,
        "source_snapshot_sha256": source_sha256,
        "input_manifest": str(args.capture_input_manifest),
        "records": records,
    })


if __name__ == "__main__":
    main()
