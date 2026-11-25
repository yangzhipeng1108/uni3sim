"""Execute one auditable MetaX C500 Device Event campaign attempt."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset import metax_c500_process_isolation
from dataset_pipeline.kernel_dataset.config import file_sha256


FAMILIES = ("vector", "flash_attn", "gemm", "fused_moe")
FORBIDDEN_PROFILER_MARKERS = ("mctracer", "mcprofiler", "perf_exec")
DEVICE_LOCK_ROOT = Path("/tmp")
POSTFLIGHT_DRAIN_ATTEMPTS = 20
POSTFLIGHT_DRAIN_INTERVAL_SECONDS = 0.5
CAPTURE_ERRORS = (
    OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError,
)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except CAPTURE_ERRORS:
        if path.exists():
            path.unlink()
        raise


def _docker_command_with_device(
        command_argv: Sequence[str], physical_device: int,
        container: str, device_gid: int) -> list[str]:
    if (not isinstance(container, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container) is None):
        raise ValueError("capture container must be an explicit Docker name or ID")
    if (isinstance(device_gid, bool) or not isinstance(device_gid, int)
            or device_gid < 0):
        raise ValueError("device GID must be an explicit non-negative integer")
    argv = list(command_argv)
    if (len(argv) >= 3 and Path(argv[0]).name == "docker"
            and argv[1:3] == ["exec", container]):
        return [
            argv[0], "exec",
            "--env", f"CUDA_VISIBLE_DEVICES={physical_device}",
            "--env", f"MACA_VISIBLE_DEVICES={physical_device}",
            "--user", f"{os.getuid()}:{device_gid}",
            container, *argv[3:],
        ]
    return argv


def _validate_command(
        command_argv: Sequence[str], attempt_root: Path,
        physical_device: int, container: str, device_gid: int) -> None:
    if (isinstance(device_gid, bool) or not isinstance(device_gid, int)
            or device_gid < 0):
        raise ValueError("device GID must be an explicit non-negative integer")
    if (not isinstance(command_argv, list) or not command_argv
            or any(not isinstance(value, str) or not value
                   for value in command_argv)):
        raise ValueError("attempt command must be a non-empty argv list")
    lowered = "\0".join(command_argv).lower()
    if any(marker in lowered for marker in FORBIDDEN_PROFILER_MARKERS):
        raise ValueError("attempt command must not invoke a profiler")
    expected_prefix = [
        "exec", "--env", f"CUDA_VISIBLE_DEVICES={physical_device}",
        "--env", f"MACA_VISIBLE_DEVICES={physical_device}",
        "--user", f"{os.getuid()}:{device_gid}",
        container,
    ]
    if (len(command_argv) < 10 or Path(command_argv[0]).name != "docker"
            or list(command_argv[1:9]) != expected_prefix):
        raise ValueError(
            "host-side attempt command must inject both device variables "
            "through docker exec --env")
    runner = "scripts/kernel_dataset/metax_c500/kernel_golden_graph_runner.py"
    if not any(value.endswith(runner) for value in command_argv):
        raise ValueError("attempt command must use kernel_golden_graph_runner.py")
    if "--output-dir" not in command_argv:
        raise ValueError("attempt command must declare --output-dir")
    output_index = command_argv.index("--output-dir") + 1
    if output_index == len(command_argv):
        raise ValueError("attempt command has no --output-dir value")
    if Path(command_argv[output_index]).resolve() != attempt_root:
        raise ValueError("attempt command output does not match attempt root")
    if "--resume" in command_argv or "--dry-run" in command_argv:
        raise ValueError("formal attempt cannot use --resume or --dry-run")


def _runner_argv_tail(command_argv: Sequence[str]) -> list[str]:
    runner = "scripts/kernel_dataset/metax_c500/kernel_golden_graph_runner.py"
    indexes = [index for index, value in enumerate(command_argv)
               if value.endswith(runner)]
    if len(indexes) != 1 or indexes[0] == 0:
        raise ValueError("command must contain one interpreted graph runner")
    return list(command_argv[indexes[0] - 1:])


def _matching_host_runners(
        snapshot: Mapping[str, Any], expected_tail: Sequence[str],
        attempt_root: Path) -> list[Mapping[str, Any]]:
    matches = []
    for process in snapshot["processes"]:
        argv = process["argv"]
        if (len(argv) != len(expected_tail)
                or argv[1:] != list(expected_tail[1:])):
            continue
        output_indexes = [index for index, value in enumerate(argv)
                          if value == "--output-dir"]
        if (len(output_indexes) != 1
                or output_indexes[0] + 1 == len(argv)
                or Path(argv[output_indexes[0] + 1]).resolve() != attempt_root):
            continue
        matches.append(process)
    return matches


def _failure_text(stage: str, error: BaseException | str) -> str:
    detail = error if isinstance(error, str) else (
        f"{type(error).__name__}: {error}")
    return f"{stage}: {detail}"


def _terminate_process_groups(process_groups: Sequence[int]) -> list[str]:
    """Terminate only process groups proven to belong to this attempt."""
    errors = []
    groups = list(dict.fromkeys(process_groups))
    for process_group in groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            process_group_missing = True
        except OSError as error:
            errors.append(_failure_text(
                f"SIGTERM process group {process_group}", error))
    time.sleep(0.1)
    for process_group in groups:
        process_group_exists = True
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            process_group_exists = False
        except OSError as error:
            errors.append(_failure_text(
                f"probe process group {process_group}", error))
            process_group_exists = False
        if process_group_exists:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                process_group_exists = False
            except OSError as error:
                errors.append(_failure_text(
                    f"SIGKILL process group {process_group}", error))
    return errors


def _run_event_attempt_locked(
        *, attempt_root: Path, attempt_id: str, ordinal: int, cohort: str,
        source_id: str, family: str, physical_device: int, container: str,
        device_gid: int,
        planned_case_ids: Sequence[str], command_argv: Sequence[str],
        snapshot_interval_seconds: float = 0.5,
        mx_smi_path: Path = metax_c500_process_isolation.MX_SMI_PATH,
        wrapper_path: Path | None = None,
        ) -> Mapping[str, Any]:
    """Run one attempt and freeze its raw/process evidence without overwrite."""
    root = attempt_root.resolve()
    if (not isinstance(attempt_id, str) or not attempt_id
            or isinstance(ordinal, bool) or not isinstance(ordinal, int)
            or ordinal <= 0 or cohort not in {"new", "existing_strict"}
            or not isinstance(source_id, str) or not source_id
            or family not in FAMILIES
            or isinstance(physical_device, bool)
            or not isinstance(physical_device, int)
            or not 0 <= physical_device <= 3):
        raise ValueError("attempt identity is invalid")
    case_ids = list(planned_case_ids)
    if (not case_ids or len(case_ids) != len(set(case_ids))
            or any(not isinstance(case_id, str) or not case_id
                   for case_id in case_ids)):
        raise ValueError("planned case IDs must be unique non-empty strings")
    argv = _docker_command_with_device(
        command_argv, physical_device, container, device_gid)
    _validate_command(argv, root, physical_device, container, device_gid)
    expected_runner_tail = _runner_argv_tail(argv)
    if snapshot_interval_seconds <= 0:
        raise ValueError("snapshot interval must be positive")
    root.mkdir(parents=True, exist_ok=False)
    isolation_root = root / "process_isolation"
    isolation_root.mkdir()
    snapshot_records = []

    def collect(snapshot_id: str) -> Mapping[str, Any]:
        snapshot = (
            metax_c500_process_isolation.collect_full_system_process_snapshot(
                mx_smi_path=mx_smi_path))
        path = isolation_root / f"{snapshot_id}.json"
        metax_c500_process_isolation.write_process_snapshot(path, snapshot)
        snapshot_records.append({
            "snapshot_id": snapshot_id, "path": str(path),
            "sha256": file_sha256(path),
        })
        return snapshot

    pre_id = None
    post_id = None
    stdout_path = root / "command.stdout"
    stderr_path = root / "command.stderr"
    process = None
    launcher_process_group_id = None
    during_ids = []
    snapshot_number = 0
    host_runner = None
    return_code = None
    failures = []
    start_wall_time_ns = None

    try:
        current_pre_id = f"{attempt_id}_pre"
        pre_snapshot = collect(current_pre_id)
        pre_id = current_pre_id
        start_wall_time_ns = time.time_ns()
        if _matching_host_runners(pre_snapshot, expected_runner_tail, root):
            raise RuntimeError("graph runner already exists before attempt launch")
        if any(record["physical_device"] == physical_device
               for record in pre_snapshot["device_processes"]):
            raise RuntimeError("target device has an owner before attempt launch")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, start_new_session=True)
            launcher_process_group_id = process.pid
            while process.poll() is None:
                snapshot_id = f"{attempt_id}_during_{snapshot_number:04d}"
                snapshot_number += 1
                snapshot = collect(snapshot_id)
                runner_matches = _matching_host_runners(
                    snapshot, expected_runner_tail, root)
                if len(runner_matches) > 1:
                    raise RuntimeError("multiple host graph runners match attempt")
                if runner_matches:
                    current_runner = runner_matches[0]
                    if host_runner is None:
                        host_runner = current_runner
                    elif (current_runner["pid"] != host_runner["pid"]
                          or current_runner["process_group_id"] !=
                          host_runner["process_group_id"]):
                        raise RuntimeError("host graph runner identity changed")
                process_index = {
                    record["pid"]: record for record in snapshot["processes"]}
                matching = [] if host_runner is None else [
                    record for record in snapshot["device_processes"]
                    if record["physical_device"] == physical_device
                    and process_index[record["pid"]]["process_group_id"] ==
                    host_runner["process_group_id"]]
                foreign = [] if host_runner is None else [
                    record for record in snapshot["device_processes"]
                    if record["physical_device"] == physical_device
                    and process_index[record["pid"]]["process_group_id"] !=
                    host_runner["process_group_id"]]
                if foreign:
                    raise RuntimeError("target device has a foreign owner")
                if runner_matches and matching:
                    during_ids.append(snapshot_id)
                if process.poll() is None:
                    time.sleep(snapshot_interval_seconds)
            return_code = process.wait()
    except CAPTURE_ERRORS as error:
        if start_wall_time_ns is None:
            start_wall_time_ns = time.time_ns()
        failures.append(_failure_text("execution", error))

    if process is not None and (failures or process.poll() is None):
        if host_runner is None:
            try:
                cleanup_snapshot = collect(f"{attempt_id}_cleanup_discovery")
                cleanup_matches = _matching_host_runners(
                    cleanup_snapshot, expected_runner_tail, root)
                if len(cleanup_matches) > 1:
                    failures.append(
                        "cleanup discovery: multiple host graph runners match attempt")
                elif cleanup_matches:
                    host_runner = cleanup_matches[0]
            except CAPTURE_ERRORS as error:
                failures.append(_failure_text("cleanup discovery", error))
        owned_groups = []
        if host_runner is not None:
            owned_groups.append(host_runner["process_group_id"])
        if launcher_process_group_id is not None:
            owned_groups.append(launcher_process_group_id)
        failures.extend(_terminate_process_groups(owned_groups))
        try:
            return_code = process.wait(timeout=5)
        except CAPTURE_ERRORS as error:
            failures.append(_failure_text("launcher wait after cleanup", error))

    end_wall_time_ns = time.time_ns()
    if start_wall_time_ns >= end_wall_time_ns:
        end_wall_time_ns = start_wall_time_ns + 1

    def post_state(
            snapshot: Mapping[str, Any],
            ) -> tuple[list[int], list[Mapping[str, Any]], bool]:
        residual_runners = _matching_host_runners(
            snapshot, expected_runner_tail, root)
        residual_groups = [
            runner["process_group_id"] for runner in residual_runners]
        processes = {record["pid"]: record for record in snapshot["processes"]}
        launcher_mismatch = False
        if process is not None and process.pid in processes:
            launcher = processes[process.pid]
            if (launcher["process_group_id"] ==
                launcher_process_group_id
                    and launcher["argv"] == argv):
                residual_groups.append(launcher_process_group_id)
            else:
                launcher_mismatch = True
        target_owners = [
            record for record in snapshot["device_processes"]
            if record["physical_device"] == physical_device]
        foreign_owners = []
        for owner in target_owners:
            if (host_runner is not None
                    and processes[owner["pid"]]["process_group_id"] ==
                    host_runner["process_group_id"]):
                residual_groups.append(host_runner["process_group_id"])
            else:
                foreign_owners.append(owner)
        return list(dict.fromkeys(residual_groups)), foreign_owners, launcher_mismatch

    post_snapshot = None
    residual_groups = []
    post_collection_errors = 0
    for observation in range(POSTFLIGHT_DRAIN_ATTEMPTS):
        if observation == 0:
            candidate_post_id = f"{attempt_id}_post"
        else:
            candidate_post_id = f"{attempt_id}_post_drain_{observation:02d}"
        try:
            candidate = collect(candidate_post_id)
        except CAPTURE_ERRORS as error:
            post_collection_errors += 1
            failures.append(_failure_text(
                f"postflight snapshot {observation + 1}", error))
            if post_collection_errors >= 3:
                break
            if observation + 1 < POSTFLIGHT_DRAIN_ATTEMPTS:
                time.sleep(POSTFLIGHT_DRAIN_INTERVAL_SECONDS)
        else:
            current_groups, foreign_owners, launcher_mismatch = post_state(
                candidate)
            if foreign_owners or launcher_mismatch:
                if foreign_owners:
                    failures.append(
                        "postflight: target device has a foreign owner")
                if launcher_mismatch:
                    failures.append(
                        "postflight: launcher PID has foreign identity")
                post_snapshot = candidate
                post_id = candidate_post_id
                break
            if not current_groups:
                post_snapshot = candidate
                post_id = candidate_post_id
                break
            residual_groups = current_groups
            end_wall_time_ns = time.time_ns()
            if observation + 1 < POSTFLIGHT_DRAIN_ATTEMPTS:
                time.sleep(POSTFLIGHT_DRAIN_INTERVAL_SECONDS)
    if post_snapshot is None and residual_groups:
        failures.append(
            "postflight: owned attempt process exceeded drain window")
        failures.extend(_terminate_process_groups(residual_groups))
        end_wall_time_ns = time.time_ns()
        post_id = None
        for retry in range(3):
            if retry == 0:
                candidate_post_id = f"{attempt_id}_post_cleanup"
            else:
                candidate_post_id = f"{attempt_id}_post_cleanup_retry_{retry}"
            try:
                post_snapshot = collect(candidate_post_id)
                post_id = candidate_post_id
                break
            except CAPTURE_ERRORS as error:
                failures.append(_failure_text(
                    f"post-cleanup snapshot {retry + 1}", error))
        if post_snapshot is not None:
            cleanup_groups, foreign_owners, launcher_mismatch = post_state(
                post_snapshot)
            if cleanup_groups or foreign_owners or launcher_mismatch:
                failures.append(
                    "postflight: process remains after exact cleanup")
    elif post_snapshot is None:
        failures.append("postflight: no complete snapshot was collected")
    if process is not None and host_runner is None:
        failures.append("execution: host graph runner was never observed")
    if process is not None and return_code != 0:
        failures.append(f"execution: launcher return code is {return_code}")

    isolation_manifest_path = root / "process_isolation_manifest.json"
    collector_path = Path(
        metax_c500_process_isolation.__file__).resolve()
    isolation_manifest = {
        "schema_version": 1,
        "artifact_kind": "metax_c500_full_system_process_isolation",
        "hardware_id": "metax_c500",
        "host": metax_c500_process_isolation.socket.gethostname(),
        "collector_path": str(collector_path),
        "collector_sha256": file_sha256(collector_path),
        "snapshots": snapshot_records,
    }
    _write_json_exclusive(isolation_manifest_path, isolation_manifest)
    isolation_reference = {
        "path": str(isolation_manifest_path),
        "sha256": file_sha256(isolation_manifest_path),
    }
    complete = (
        not failures and return_code == 0 and host_runner is not None
        and bool(during_ids)
    )
    if not complete and not failures:
        failures.append("execution: attempt did not produce active Device evidence")
    failure = "; ".join(failures) if failures else None
    error_path = stderr_path if stderr_path.is_file() else None
    if failure is not None:
        error_path = root / "wrapper_error.log"
        error_path.write_text(failure + "\n", encoding="utf-8")
    error_sha = file_sha256(error_path) if error_path is not None else None
    case_attempts = []
    for case_id in case_ids:
        artifact = root / "cases" / f"{case_id}.json"
        if artifact.is_file():
            case_attempts.append({
                "case_id": case_id, "status": "measured",
                "artifact_path": str(artifact),
                "artifact_sha256": file_sha256(artifact),
                "error_path": None, "error_sha256": None,
                "superseded_by_attempt_id": None,
            })
        else:
            case_attempts.append({
                "case_id": case_id, "status": "failed",
                "artifact_path": None, "artifact_sha256": None,
                "error_path": (
                    str(error_path) if error_path is not None else None),
                "error_sha256": error_sha,
                "superseded_by_attempt_id": None,
            })
    complete = (
        complete
        and all(record["status"] == "measured" for record in case_attempts)
    )
    if not complete and failure is None:
        failure = "execution: one or more planned case artifacts are absent"
        error_path = root / "wrapper_error.log"
        error_path.write_text(failure + "\n", encoding="utf-8")
        error_sha = file_sha256(error_path)
        for record in case_attempts:
            if record["status"] == "failed":
                record["error_path"] = str(error_path)
                record["error_sha256"] = error_sha
    module_path = Path(__file__).resolve()
    actual_wrapper_path = (
        module_path if wrapper_path is None else wrapper_path.resolve())
    attempt = {
        "attempt_id": attempt_id, "ordinal": ordinal, "cohort": cohort,
        "source_id": source_id, "family": family,
        "physical_device": physical_device, "root": str(root),
        "command_argv": host_runner["argv"] if host_runner is not None else None,
        "pid": host_runner["pid"] if host_runner is not None else None,
        "process_group_id": (
            host_runner["process_group_id"] if host_runner is not None else None),
        "start_wall_time_ns": start_wall_time_ns,
        "end_wall_time_ns": end_wall_time_ns,
        "planned_case_ids": case_ids, "case_attempts": case_attempts,
        "terminal_status": "completed" if complete else "failed",
        "error": failure,
        "superseded": False, "preflight_snapshot_id": pre_id,
        "during_snapshot_ids": during_ids, "postflight_snapshot_id": post_id,
        "process_isolation": isolation_reference,
        "root_inventory": None,
        "wrapper": {
            "path": str(actual_wrapper_path),
            "sha256": file_sha256(actual_wrapper_path),
        },
        "execution_module": {
            "path": str(module_path), "sha256": file_sha256(module_path),
        },
        "launcher_pid": process.pid if process is not None else None,
        "launcher_process_group_id": launcher_process_group_id,
        "launcher_command_argv": argv,
        "host_uid": os.getuid(), "device_gid": device_gid,
    }
    attempt_path = root / "attempt.json"
    _write_json_exclusive(attempt_path, attempt)
    return {
        "attempt": attempt, "attempt_path": str(attempt_path),
        "attempt_sha256": file_sha256(attempt_path),
        "process_isolation": isolation_reference,
        "return_code": return_code,
    }


def run_event_attempt(**kwargs: Any) -> Mapping[str, Any]:
    """Hold the host-global per-device lock through final evidence write."""
    physical_device = kwargs["physical_device"]
    if (isinstance(physical_device, bool)
            or not isinstance(physical_device, int)
            or not 0 <= physical_device <= 3):
        raise ValueError("attempt physical device is invalid")
    lock_path = DEVICE_LOCK_ROOT / (
        f"midlayer_metax_c500_device_{physical_device}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"MetaX C500 device {physical_device} is locked") from error
        return _run_event_attempt_locked(**kwargs)
