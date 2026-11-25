#!/usr/bin/env python3
"""Capture one frozen 48-AIV elementwise source-control campaign."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


VARIANTS = ("control_baseline", "dag_tbuf", "full_lifecycle")
TILE_COUNTS = (18, 29, 30, 36, 58, 59)
CALIBRATION_TILE_COUNTS = (18, 36, 59)
HOLDOUT_TILE_COUNTS = (29, 30, 58)
CELLS = tuple(
    (variant, tile_count)
    for tile_count in TILE_COUNTS
    for variant in VARIANTS)
TRIALS = 6
AIV_WORKERS = 48
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "elementwise_control_contract.h",
    "elementwise_control_kernel.cpp",
    "elementwise_control_main.cpp",
    "elementwise_control_runner.py",
    "elementwise_control_report.py",
)
STATIC_REFERENCE_FILENAME = (
    "elementwise_add_mul_dag_accounting_cann_8_5_1.json")
STATIC_REFERENCE_SHA256 = (
    "ac2f90c76a4cc6b2966d6dc97dc34f45a6ec51333d53f9fb7b739b63c5f46c5b")
EXPECTED_CANN = {
    "package": "Ascend-cann-toolkit",
    "version": "8.5.1",
    "innerversion": "V100R001C25SPC002B220",
}
EXPECTED_CANN_SOURCE_SHA256 = {
    "broadcast_scheduler": (
        "4eebc1471b440575e3017a4be3d2e4f8b1c70c6764c7aa5f25a9573c687f3e67"),
    "dag": "d8e3151f89c8fa7a04a5898145bd0495ab76070cf4540eab7a3420589db36dd0",
    "sync": "7cb3198812792f775db46f47d175d773cced914cb34f9371c2b9800fdf69b618",
}
SOURCE_MARKERS = {
    "broadcast_scheduler": (
        "ElemDag::template GetBufferIds<true, true>()",
        "GetTensor<TPosition::VECCALC>(bufId);",
        "ReleaseTensor<TPosition::VECCALC>(bufId);",
    ),
    "dag": ("struct DAGSch", "GetBufferIds"),
    "sync": (
        "Mutex::Lock<PIPE_MTE2>(bufID);",
        "Mutex::Lock<PIPE_V>(bufID);",
        "Mutex::Lock<PIPE_MTE3>(bufID);",
        "Mutex::Unlock<PIPE_MTE2>(bufID);",
        "Mutex::Unlock<PIPE_V>(bufID);",
        "Mutex::Unlock<PIPE_MTE3>(bufID);",
    ),
}
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position", "process_pid",
    "variant", "tile_count", "completion_cycles", "completion_ns",
    "worker_cycles", "worker_checksums", "validation",
)
CHILD_FIELDS = (
    "variant", "tile_count", "worker_cycles", "worker_checksums",
    "validation",
)
MANAGED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_PROCESS = None
_REGISTRATION_IN_PROGRESS = False
_DEFERRED_SIGNAL = None
_SIGNAL_CLEANUP_IN_PROGRESS = False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _ordered_cells(trial_index):
    if trial_index < 0 or trial_index >= TRIALS:
        raise ValueError("trial index is outside the frozen campaign")
    if trial_index % 2 == 0:
        return "forward", CELLS
    return "reverse", tuple(reversed(CELLS))


def _terminate_process_group(process):
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
            return
        process.wait(timeout=10)


def _handle_signal(signum, _frame):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _SIGNAL_CLEANUP_IN_PROGRESS

    if _REGISTRATION_IN_PROGRESS or _SIGNAL_CLEANUP_IN_PROGRESS:
        if _DEFERRED_SIGNAL is None:
            _DEFERRED_SIGNAL = signum
        return
    process = _ACTIVE_PROCESS
    _ACTIVE_PROCESS = None
    _SIGNAL_CLEANUP_IN_PROGRESS = True
    try:
        if process is not None:
            _terminate_process_group(process)
    finally:
        _SIGNAL_CLEANUP_IN_PROGRESS = False
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for managed_signal in MANAGED_SIGNALS:
        signal.signal(managed_signal, _handle_signal)


def _spawn_managed_process(command, *, environment=None):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _REGISTRATION_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    _REGISTRATION_IN_PROGRESS = True
    try:
        def restore_child_signal_mask():
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True,
            preexec_fn=restore_child_signal_mask)
        _ACTIVE_PROCESS = process
    finally:
        _REGISTRATION_IN_PROGRESS = False
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    deferred_signal = _DEFERRED_SIGNAL
    _DEFERRED_SIGNAL = None
    if deferred_signal is not None:
        _handle_signal(deferred_signal, None)
    return process


def _unregister_process(process):
    global _ACTIVE_PROCESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _communicate_managed(command, *, environment=None, timeout):
    process = _spawn_managed_process(command, environment=environment)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after {timeout} seconds"
            ) from error
        except BaseException:
            _terminate_process_group(process)
            raise
    finally:
        _unregister_process(process)
    return process.returncode, process.pid, stdout, stderr


def _run_checked(command, *, timeout=30):
    returncode, _, stdout, stderr = _communicate_managed(
        command, timeout=timeout)
    if returncode != 0:
        raise RuntimeError(
            f"command returned {returncode}: {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}")
    return stdout


def _parse_idle_usage(text):
    labels = {
        "aic_percent": "Aicore Usage Rate(%)",
        "aiv_percent": "Aivector Usage Rate(%)",
        "hbm_bandwidth_percent": "HBM Bandwidth Usage Rate(%)",
    }
    result = {}
    for field, label in labels.items():
        matches = re.findall(
            rf"^\s*{re.escape(label)}\s*:\s*([0-9]+)\s*$",
            text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"usage telemetry has ambiguous {field}")
        value = int(matches[0])
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        result[field] = value
    return result


def _parse_global_identity(text, device_id):
    version = re.findall(
        r"npu-smi\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    header_version = re.findall(
        r"Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", text,
        flags=re.IGNORECASE)
    if len(version) != 1 or header_version != version:
        raise ValueError("npu-smi version identity is ambiguous")
    if len(re.findall(
            rf"^\|\s*{device_id}\s+910B2\s+\|", text,
            flags=re.MULTILINE)) != 1:
        raise ValueError("npu-smi does not identify the selected 910B2")
    return {
        "physical_device_id": device_id,
        "model": "910B2",
        "npu_smi_version": version[0],
    }


def _parse_cann_version(text):
    source_fields = {
        "package": "package_name",
        "version": "version",
        "innerversion": "innerversion",
    }
    result = {}
    for field, expected in EXPECTED_CANN.items():
        matches = re.findall(
            rf"^\s*{source_fields[field]}\s*=\s*([^\s]+)\s*$", text,
            flags=re.IGNORECASE | re.MULTILINE)
        if matches != [expected]:
            raise ValueError(f"CANN {field} identity changed")
        result[field] = expected
    return result


def _validate_cann_sources(paths):
    identities = {}
    for label, expected_hash in EXPECTED_CANN_SOURCE_SHA256.items():
        path = paths[label]
        if not path.is_file():
            raise FileNotFoundError(f"required CANN source is missing: {path}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"{label} SHA256 mismatch: {observed_hash}")
        source = path.read_text(encoding="utf-8")
        missing = tuple(
            marker for marker in SOURCE_MARKERS[label]
            if marker not in source)
        if missing:
            raise ValueError(f"{label} lost source markers: {missing}")
        identities[label] = {
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "required_markers": list(SOURCE_MARKERS[label]),
        }
    return identities


def _parse_worker_vector(text, *, allow_zero):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != AIV_WORKERS:
        raise ValueError("child output does not contain exactly 48 AIV values")
    if allow_zero:
        valid = all(value != 0 for value in values)
    else:
        valid = all(value > 0 for value in values)
    if not valid:
        raise ValueError("child output contains an invalid AIV value")
    return values


def _write_json_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_campaign(args, benchmark_root):
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256) is None:
        raise ValueError("expected snapshot must be a lowercase SHA256")
    observed_snapshot = _snapshot_sha256(benchmark_root)
    if observed_snapshot != args.expected_snapshot_sha256:
        raise ValueError("deployed source snapshot does not match audit")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise FileNotFoundError("benchmark binary is missing or not executable")
    reference = benchmark_root.parents[2] / "references" / STATIC_REFERENCE_FILENAME
    if _sha256(reference) != STATIC_REFERENCE_SHA256:
        raise ValueError("static DAG accounting reference SHA256 changed")
    cann_identity = _parse_cann_version(
        args.cann_version_file.read_text(encoding="utf-8"))
    cann_sources = _validate_cann_sources({
        "broadcast_scheduler": args.broadcast_scheduler_source,
        "dag": args.dag_source,
        "sync": args.sync_source,
    })
    global_text = _run_checked([str(args.npu_smi), "info"])
    usage_command = [
        str(args.npu_smi), "info", "-t", "usages", "-i",
        str(args.device_id),
    ]
    pre_usage_text = _run_checked(usage_command)
    device_identity = _parse_global_identity(global_text, args.device_id)
    pre_usage = _parse_idle_usage(pre_usage_text)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir()
    (args.output_dir / "npu_smi_global.txt").write_text(
        global_text, encoding="utf-8")
    (args.output_dir / "npu_smi_pre.txt").write_text(
        pre_usage_text, encoding="utf-8")
    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = "0"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    raw_path = args.output_dir / "raw.csv"
    trial_artifacts = []
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial_index in range(TRIALS):
            direction, cells = _ordered_cells(trial_index)
            command = [
                str(args.binary), "--batch",
                *(f"{variant}:{tile_count}"
                  for variant, tile_count in cells),
            ]
            returncode, process_pid, stdout, stderr = _communicate_managed(
                command, environment=environment,
                timeout=args.timeout_seconds)
            if returncode != 0:
                raise RuntimeError(
                    f"trial {trial_index + 1} returned {returncode}: "
                    f"{stderr}")
            stdout_path = args.output_dir / f"trial_{trial_index + 1}.stdout"
            stderr_path = args.output_dir / f"trial_{trial_index + 1}.stderr"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            rows = list(csv.DictReader(stdout.splitlines()))
            if not rows or tuple(rows[0]) != CHILD_FIELDS:
                raise ValueError("child output header changed")
            if len(rows) != len(cells):
                raise ValueError("child output lost campaign cells")
            for position, (row, expected) in enumerate(zip(rows, cells)):
                variant, tile_count = expected
                if (row["variant"] != variant
                        or int(row["tile_count"]) != tile_count
                        or row["validation"] != "ok"):
                    raise ValueError("child output changed the frozen cell")
                cycles = _parse_worker_vector(
                    row["worker_cycles"], allow_zero=False)
                _parse_worker_vector(
                    row["worker_checksums"], allow_zero=True)
                completion_cycles = max(cycles)
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": position,
                    "process_pid": process_pid,
                    "variant": variant,
                    "tile_count": tile_count,
                    "completion_cycles": completion_cycles,
                    "completion_ns": completion_cycles * SYSTEM_COUNTER_TICK_NS,
                    "worker_cycles": row["worker_cycles"],
                    "worker_checksums": row["worker_checksums"],
                    "validation": "ok",
                })
            stream.flush()
            os.fsync(stream.fileno())
            trial_artifacts.append({
                "trial": trial_index + 1,
                "direction": direction,
                "pid": process_pid,
                "command": command,
                "stdout": stdout_path.name,
                "stdout_sha256": _sha256(stdout_path),
                "stderr": stderr_path.name,
                "stderr_sha256": _sha256(stderr_path),
            })
    post_usage_text = _run_checked(usage_command)
    post_usage = _parse_idle_usage(post_usage_text)
    (args.output_dir / "npu_smi_post.txt").write_text(
        post_usage_text, encoding="utf-8")
    provenance = {
        "schema": "ascend_910b_elementwise_control_lifecycle_provenance_v1",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            filename: _sha256(benchmark_root / filename)
            for filename in SOURCE_FILENAMES
        },
        "binary": {
            "path": str(args.binary.resolve()),
            "sha256": _sha256(args.binary),
            "build_command": args.build_command,
        },
        "static_reference": {
            "path": str(reference.resolve()),
            "sha256": STATIC_REFERENCE_SHA256,
        },
        "cann": {
            "identity": cann_identity,
            "version_file": str(args.cann_version_file.resolve()),
            "version_file_sha256": _sha256(args.cann_version_file),
            "source_identities": cann_sources,
        },
        "device": {
            **device_identity,
            "logical_device_id": 0,
            "selection_environment": {
                "ASCEND_DEVICE_ID": "0",
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device_id),
            },
        },
        "idle_telemetry": {"pre": pre_usage, "post": post_usage},
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "measurement_scope": (
            "maximum completion over exactly 48 AIV workers for a source-"
            "control-only seven-node/twelve-buffer-lookup program; excludes "
            "Add/Mul tensor payloads and production/evaluation duration"),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "variants": list(VARIANTS),
            "tile_counts": list(TILE_COUNTS),
            "calibration_tile_counts": list(CALIBRATION_TILE_COUNTS),
            "holdout_tile_counts": list(HOLDOUT_TILE_COUNTS),
            "aiv_workers": AIV_WORKERS,
            "trials": TRIALS,
            "order": "balanced_forward_reverse",
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": TRIALS * len(CELLS),
            "runner_pid": os.getpid(),
            "argv": sys.argv,
            "working_directory": str(Path.cwd()),
            "trials": trial_artifacts,
        },
    }
    _write_json_exclusive(args.output_dir / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("snapshot")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--binary", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--source-commit", required=True)
    run_parser.add_argument("--expected-snapshot-sha256", required=True)
    run_parser.add_argument("--device-id", required=True, type=int)
    run_parser.add_argument(
        "--npu-smi", type=Path, default=Path("/usr/local/bin/npu-smi"))
    run_parser.add_argument("--cann-version-file", required=True, type=Path)
    run_parser.add_argument(
        "--broadcast-scheduler-source", required=True, type=Path)
    run_parser.add_argument("--dag-source", required=True, type=Path)
    run_parser.add_argument("--sync-source", required=True, type=Path)
    run_parser.add_argument("--build-command", required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    benchmark_root = Path(__file__).resolve().parent
    if args.action == "snapshot":
        print(_snapshot_sha256(benchmark_root))
        return
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    _install_signal_handlers()
    _run_campaign(args, benchmark_root)


if __name__ == "__main__":
    main()
