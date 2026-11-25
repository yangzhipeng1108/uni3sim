#!/usr/bin/env python3
"""Capture frozen Ascend 910B FA cases as Timeline and op-mode evidence."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_audit import (
    build_ascend_counter_candidate,
    write_json_new,
)


_INPUT_FIELDS = frozenset((
    "schema_version", "capture_kind", "manifest_path", "manifest_sha256",
    "hardware_id", "case_count", "cases",
))
_CASE_FIELDS = frozenset((
    "case_id", "workload_digest", "capture_metadata", "runner",
    "runner_arguments", "runner_workload",
))
_TIMELINE_FIELDS = frozenset((
    "Device_id", "Task ID", "Stream ID", "Op Name", "OP Type",
    "Task Type", "Task Duration(us)", "Block Dim", "Mix Block Dim",
))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(manifest):
    if not isinstance(manifest, dict) or set(manifest) != _INPUT_FIELDS:
        raise ValueError("Ascend capture-input manifest fields are not exact")
    if manifest["hardware_id"] != "ascend_910b":
        raise ValueError("Ascend full runner received a non-Ascend input manifest")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != manifest["case_count"]:
        raise ValueError("Ascend capture-input case count is invalid")
    case_ids = []
    for item in cases:
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError("Ascend capture-input case fields are not exact")
        if not isinstance(item["case_id"], str) or not item["case_id"]:
            raise ValueError("Ascend case_id must be non-empty")
        if (not isinstance(item["workload_digest"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["workload_digest"]) is None):
            raise ValueError("Ascend workload_digest must be a SHA256")
        if (not isinstance(item["runner"], str) or not item["runner"]
                or not isinstance(item["runner_arguments"], list)
                or any(not isinstance(value, str) for value in item["runner_arguments"])):
            raise TypeError("Ascend runner contract is invalid")
        case_ids.append(item["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Ascend capture-input case IDs are not unique")
    return manifest


def _load_inputs(path):
    with path.open(encoding="utf-8") as stream:
        return _validate_inputs(json.load(stream))


def _read_remote_inputs(*, host, container, remote_manifest_path):
    """Read the materialized manifest from the execution container itself."""
    if not remote_manifest_path.startswith("/"):
        raise ValueError("Ascend remote capture-input manifest must be an absolute container path")
    remote_command = shlex.join((
        "docker", "exec", container, "cat", remote_manifest_path,
    ))
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         host, remote_command],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Ascend could not read the remote capture-input manifest: "
            f"{completed.stderr.strip()}")
    try:
        manifest = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Ascend remote capture-input manifest is not JSON") from error
    return _validate_inputs(manifest)


def _require_remote_runner_inputs(*, local_manifest, remote_manifest):
    """Bind the remote materialization to the frozen local case semantics."""
    local_cases = local_manifest["cases"]
    remote_cases = remote_manifest["cases"]
    if len(local_cases) != len(remote_cases):
        raise ValueError("Ascend local and remote capture-input manifests disagree in case count")
    for local, remote in zip(local_cases, remote_cases):
        if (local["case_id"], local["workload_digest"], local["runner"]) != (
                remote["case_id"], remote["workload_digest"], remote["runner"]):
            raise ValueError(
                "Ascend remote capture-input manifest disagrees with the frozen local "
                f"semantics for {local['case_id']!r}")
        for field in ("capture_metadata", "runner_workload"):
            value = remote[field]
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"Ascend remote {field} must be an absolute normalized container path")
        arguments = remote["runner_arguments"]
        expected_pairs = {
            "--capture-metadata-json": remote["capture_metadata"],
            "--workload-json": remote["runner_workload"],
        }
        for flag, expected_value in expected_pairs.items():
            try:
                index = arguments.index(flag)
            except ValueError as error:
                raise ValueError(f"Ascend remote runner arguments lack {flag}") from error
            if index + 1 >= len(arguments) or arguments[index + 1] != expected_value:
                raise ValueError(
                    f"Ascend remote runner arguments have an invalid {flag} value")


def _run(command, *, log_path):
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Ascend host command failed: log={log_path}")


def _run_remote(*, host, container, container_repo, device, command,
                remote_log_path, log_path):
    command_text = shlex.join(command)
    remote_log_parent = str(PurePosixPath(remote_log_path).parent)
    wrapped = (
        "mkdir -p " + shlex.quote(remote_log_parent) + " && { printf '%s\\n' "
        + shlex.quote(command_text) + "; "
        + command_text + "; } > " + shlex.quote(remote_log_path) + " 2>&1")
    remote_command = shlex.join((
        "docker", "exec", "-w", container_repo,
        "-e", f"ASCEND_RT_VISIBLE_DEVICES={device}",
        "-e", f"ASCEND_CACHE_PATH=/tmp/fa_serving_capture_cache_{device}",
        "-e", f"ASCEND_WORK_PATH=/tmp/fa_serving_capture_work_{device}",
        container, "bash", "-lc", wrapped,
    ))
    _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
          host, remote_command], log_path=log_path)


def _group_remote_timeline_summaries(*, remote_host_case_root, paths):
    root = PurePosixPath(remote_host_case_root)
    expected_windows = {
        f"window_{window}": []
        for window in range(1, 4)
    }
    for value in paths:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Ascend remote Timeline summary path is invalid")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Ascend remote Timeline summary escaped its case root") from error
        if (len(relative.parts) < 3 or relative.parts[0] != "golden"
                or relative.parts[1] not in expected_windows
                or not path.name.startswith("op_summary_")
                or path.suffix != ".csv"):
            raise ValueError("Ascend remote Timeline summary has an invalid layout")
        expected_windows[relative.parts[1]].append(str(path))
    for window, values in expected_windows.items():
        if len(values) != 2 or len(set(values)) != 2:
            raise ValueError(
                f"Ascend remote {window} requires exactly two Timeline summaries")
        values.sort()
    return expected_windows


def _copy_timeline_summaries(*, host, remote_host_case_root,
                             local_stage_root, local_log_root):
    remote_command = shlex.join((
        "find", f"{remote_host_case_root}/golden", "-type", "f",
        "-name", "op_summary_*.csv", "-print",
    ))
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               host, remote_command]
    listing_log = local_log_root / "list_timeline_summaries.host.log"
    with listing_log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        completed = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        stream.write(completed.stdout)
        stream.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "Ascend could not list remote Timeline summaries: "
            f"log={listing_log}")
    groups = _group_remote_timeline_summaries(
        remote_host_case_root=remote_host_case_root,
        paths=[line for line in completed.stdout.splitlines() if line])
    if local_stage_root.exists():
        raise FileExistsError(
            f"Ascend Timeline summary staging root already exists: {local_stage_root}")
    for window, remote_paths in groups.items():
        local_window_root = local_stage_root / "golden" / window
        local_window_root.mkdir(parents=True)
        _run(
            ["scp", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
             *(f"{host}:{path}" for path in remote_paths),
             str(local_window_root)],
            log_path=local_log_root / f"copy_{window}_summaries.log")
    return [
        _timeline_summary_path(local_stage_root / "golden" / f"window_{window}")
        for window in range(1, 4)
    ]


def _timing_signature(timing):
    return {
        "status": timing["status"],
        "aggregation": timing["aggregation"],
        "windows": [{
            field: window[field]
            for field in (
                "op_summary_sha256", "task_id", "stream_id", "device_id",
                "block_dim", "mix_block_dim", "duration_ns",
            )
        } for window in timing["windows"]],
    }


def _single_path(root, pattern, context):
    paths = sorted(Path(root).rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"{context} expected exactly one path, found {len(paths)}")
    return paths[0]


def _timeline_summary_path(root):
    """Select the explicit export only after proving it duplicates the raw export."""
    paths = sorted(Path(root).rglob("op_summary_*.csv"))
    if len(paths) != 2:
        raise ValueError(
            "Ascend Timeline requires exactly the automatic and explicit text exports, "
            f"found {len(paths)} op_summary CSVs")
    if _sha256(paths[0]) != _sha256(paths[1]):
        raise ValueError(
            "Ascend automatic and explicit Timeline exports disagree; refusing to "
            "choose a timing record")
    return paths[1]


def _timeline_record(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not _TIMELINE_FIELDS <= set(reader.fieldnames):
            raise ValueError(f"Ascend Timeline CSV lacks required columns: {path}")
        matches = [
            row for row in reader
            if row["Op Name"].strip() == "FusedInferAttentionScore"
        ]
    if not matches:
        raise ValueError(f"Ascend Timeline has no FusedInferAttentionScore: {path}")
    selected = max(matches, key=lambda row: int(row["Task ID"]))
    if selected["OP Type"].strip() != "FusedInferAttentionScore":
        raise ValueError("Ascend selected Timeline task has a mismatched OP Type")
    if selected["Task Type"].strip() != "MIX_AIC":
        raise ValueError("Ascend selected Timeline task is not MIX_AIC")
    duration_us = float(selected["Task Duration(us)"])
    if not math.isfinite(duration_us) or duration_us <= 0:
        raise ValueError("Ascend selected Timeline duration must be positive")
    return {
        "task_id": int(selected["Task ID"]),
        "stream_id": int(selected["Stream ID"]),
        "device_id": int(selected["Device_id"]),
        "block_dim": int(selected["Block Dim"]),
        "mix_block_dim": int(selected["Mix Block Dim"]),
        "duration_ns": duration_us * 1_000,
    }


def _build_timing_audit(*, case_id, workload_digest, source_snapshot_sha256,
                        op_summary_paths, profile_warmup_iterations):
    if len(op_summary_paths) != 3:
        raise ValueError("Ascend timing audit requires three Timeline windows")
    if profile_warmup_iterations < 0:
        raise ValueError("Ascend profile warmup iterations must be non-negative")
    windows = []
    durations = []
    expected_launch = None
    for ordinal, path in enumerate(op_summary_paths, start=1):
        record = _timeline_record(path)
        launch = tuple(record[name] for name in (
            "stream_id", "device_id", "block_dim", "mix_block_dim"))
        if expected_launch is None:
            expected_launch = launch
        elif launch != expected_launch:
            raise ValueError("Ascend Timeline windows have different final launches")
        durations.append(record["duration_ns"])
        windows.append({
            "window": ordinal,
            "op_summary_path": str(path),
            "op_summary_sha256": _sha256(path),
            **record,
        })
    spread = (max(durations) - min(durations)) / max(durations)
    stable = spread <= 0.05
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_timeline_timing",
        "status": "pass" if stable else "fail",
        "hardware": {"hardware_id": "ascend_910b"},
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot": {"sha256": source_snapshot_sha256},
        "measurement_conditions": {
            "profile_warmup_iterations": profile_warmup_iterations,
            "final_timed_task": (
                "the final public-op FusedInferAttentionScore task after "
                "unmeasured warmup calls"),
        },
        "windows": windows,
        "aggregation": {
            "selection": "median final public-op FusedInferAttentionScore task from three independent Timeline processes",
            "median_duration_ns": sorted(durations)[1],
            "max_relative_spread": spread,
            "max_relative_spread_gate": 0.05,
            "pass": stable,
        },
    }


def _timing_failure_record(*, case_id, workload_digest, timing_audit_path):
    """Describe a rejected timing capture without presenting it as complete."""
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_timing_failure",
        "case_id": case_id,
        "workload_digest": workload_digest,
        "status": "timing_stability_failed",
        "timing_audit": str(timing_audit_path),
    }


def _counter_paths(counter_root):
    basic_paths = sorted(Path(counter_root).rglob("OpBasicInfo_*.csv"))
    if len(basic_paths) != 3:
        raise ValueError(f"Ascend msprof op expected three OpBasicInfo samples, found {len(basic_paths)}")
    pipe_paths = []
    memory_paths = []
    for basic in basic_paths:
        pipe_paths.append(_single_path(basic.parent, "PipeUtilization_*.csv", "Ascend pipe sample"))
        memory_paths.append(_single_path(basic.parent, "Memory_*.csv", "Ascend memory sample"))
    return basic_paths, pipe_paths, memory_paths


def _program_hash_path(timeline_root, program):
    stem = program.removesuffix("_mix_aic")
    candidates = []
    for path in sorted(Path(timeline_root).rglob("unaging.additional.hash_dic.slice_*")):
        if path.name.endswith(".done"):
            continue
        if stem in path.read_text(encoding="utf-8"):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Ascend Timeline program hash must identify {program!r}, found {len(candidates)}")
    return candidates[0]


def _counter_container_case_root(*, args, case_id):
    root = args.counter_output_container_root
    if root is None:
        root = args.remote_container_root
    return f"{root.rstrip('/')}/{case_id}"


def _counter_host_case_root(*, args, case_id):
    """Return the host path owning the immutable op-mode counter raw.

    The short path used by the Timeline application can be a symlink.  The
    profiler must instead write counters through a physical snapshot path, so
    callers must supply the matching host root whenever those paths differ.
    Falling back to the Timeline root could silently consume an old counter
    directory left by an earlier attempt.
    """
    container_root = args.counter_output_container_root
    host_root = args.counter_output_host_root
    if container_root is None:
        if host_root is not None:
            raise ValueError(
                "Ascend counter host root requires a counter container root")
        host_root = args.remote_output_root
    elif host_root is None:
        if container_root.rstrip("/") != args.remote_container_root.rstrip("/"):
            raise ValueError(
                "Ascend physical counter root requires "
                "--counter-output-host-root")
        host_root = args.remote_output_root
    return f"{host_root.rstrip('/')}/{case_id}"


def _existing_counter_host_case_root(*, args, case_id):
    """Resolve an immutable op-mode capture produced before Timeline resume.

    A hardware counter job may be deliberately scheduled before the large
    Timeline CSVs are repatriated.  Its physical raw root is an explicit input
    to the later resume step; it is never inferred from an old Timeline path.
    """

    root = args.resume_existing_counter_host_root
    if not isinstance(root, str) or not root.startswith("/"):
        raise ValueError("Ascend existing counter host root must be absolute")
    if ".." in PurePosixPath(root).parts:
        raise ValueError("Ascend existing counter host root must be normalized")
    return f"{root.rstrip('/')}/{case_id}"


def _group_remote_program_hashes(*, remote_host_case_root, paths):
    """Require program-hash files to stay in one of the three Timeline windows."""
    root = PurePosixPath(remote_host_case_root)
    expected_windows = {
        f"window_{window}": []
        for window in range(1, 4)
    }
    for value in paths:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Ascend remote program-hash path is invalid")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Ascend remote program-hash escaped its case root") from error
        if (len(relative.parts) < 3 or relative.parts[0] != "golden"
                or relative.parts[1] not in expected_windows
                or not path.name.startswith("unaging.additional.hash_dic.slice_")):
            raise ValueError("Ascend remote program-hash has an invalid layout")
        expected_windows[relative.parts[1]].append(str(path))
    for window, values in expected_windows.items():
        if not values:
            raise ValueError(f"Ascend remote {window} has no program-hash file")
        values.sort()
    return expected_windows


def _copy_timeline_program_hashes(*, args, remote_host_case_root,
                                  local_stage_root, local_log_root):
    """Copy only the hash dictionaries needed to bind counters to Timeline ops.

    Full msprof Timeline directories contain hundreds of binary slices per
    window.  They are immutable under the remote capture root; candidates need
    only the selected CSV exports and program hash dictionaries, so never make
    a completed device phase wait for a recursive raw transfer.
    """
    remote_command = shlex.join((
        "find", f"{remote_host_case_root}/golden", "-type", "f",
        "-name", "unaging.additional.hash_dic.slice_*", "-print",
    ))
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               args.host, remote_command]
    listing_log = local_log_root / "list_timeline_program_hashes.host.log"
    with listing_log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        completed = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        stream.write(completed.stdout)
        stream.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "Ascend could not list remote Timeline program hashes: "
            f"log={listing_log}")
    groups = _group_remote_program_hashes(
        remote_host_case_root=remote_host_case_root,
        paths=[line for line in completed.stdout.splitlines() if line])
    for window, remote_paths in groups.items():
        local_window_root = local_stage_root / "golden" / window / "program_hash"
        local_window_root.mkdir(parents=True)
        _run(
            ["scp", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
             *(f"{args.host}:{path}" for path in remote_paths),
             str(local_window_root)],
            log_path=local_log_root / f"copy_{window}_program_hashes.log")


def _copy_counter_raw(*, args, counter_host_case_root, local_case_root,
                      local_log_root, log_tag):
    """Copy the op-mode raw needed by the candidate counter parser."""
    local_case_root.mkdir(parents=True)
    _run([
        "scp", "-r", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        f"{args.host}:{counter_host_case_root}/counter",
        str(local_case_root),
    ], log_path=local_log_root / f"copy_{log_tag}_counter.log")


def _resume_existing_timeline(*, args, remote_manifest, skip_cases,
                              source_sha256):
    """Finish counter/candidate work for immutable, already-timed cases.

    A Timeline export can fail after earlier cases have completed their three
    device windows.  Re-running those successful windows would both waste a
    scarce device reservation and violate the capture rule that retries are
    scoped to the failed case.  This mode consumes only the previously
    materialized Timeline evidence, validates it with the ordinary strict
    path, then appends the missing op-mode counter evidence for those cases.
    """
    source_output_root = args.resume_source_output_root
    if source_output_root is None or not source_output_root.is_dir():
        raise FileNotFoundError("Ascend resume source output root does not exist")
    if args.output_root.exists():
        raise FileExistsError("Ascend resume output root must be new")
    existing_snapshot = source_output_root / "source_snapshot.tar"
    if not existing_snapshot.is_file():
        raise FileNotFoundError("Ascend resume output lacks source snapshot")
    if _sha256(existing_snapshot) != source_sha256:
        raise ValueError("Ascend resume source snapshot SHA256 disagrees")
    if args.audit_root.exists():
        raise FileExistsError("Ascend resume audit root must be new")
    staging_root = args.output_root / "_resume_timeline_summary_staging"
    resume_log_root = args.output_root / "resume_host_logs"
    args.audit_root.mkdir(parents=True)
    args.output_root.mkdir(parents=True)
    shutil.copyfile(existing_snapshot, args.output_root / "source_snapshot.tar")
    write_json_new(args.output_root / "capture_execution.json", {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "owner": "fa_serving_coverage_20260915",
        "kind": "resume_existing_timeline",
        "resume_source_output_root": str(source_output_root),
        "source_snapshot_sha256": source_sha256,
        "counter_output_container_root": args.counter_output_container_root,
        "counter_output_host_root": args.counter_output_host_root,
        "resume_existing_counter_host_root": (
            args.resume_existing_counter_host_root),
        "resume_log_tag": args.resume_log_tag,
    })
    records = []
    for item in remote_manifest["cases"]:
        case_id = item["case_id"]
        if case_id in skip_cases:
            continue
        remote_host_case_root = f"{args.remote_output_root}/{case_id}"
        remote_container_case_root = f"{args.remote_container_root}/{case_id}"
        local_log_root = resume_log_root / case_id
        local_log_root.mkdir(parents=True)
        summary_paths = _copy_timeline_summaries(
            host=args.host, remote_host_case_root=remote_host_case_root,
            local_stage_root=staging_root / case_id,
            local_log_root=local_log_root)
        provisional_timing = _build_timing_audit(
            case_id=case_id, workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256,
            op_summary_paths=summary_paths,
            profile_warmup_iterations=args.profile_warmup_iterations)
        timing_path = args.audit_root / f"{case_id}.timing.json"
        write_json_new(timing_path, provisional_timing)
        if provisional_timing["status"] != "pass":
            failure = _timing_failure_record(
                case_id=case_id, workload_digest=item["workload_digest"],
                timing_audit_path=timing_path)
            records.append(failure)
            continue
        runner_command = [
            args.runner_python, item["runner"], *item["runner_arguments"],
            "--profile-warmup-iterations", str(args.profile_warmup_iterations),
        ]
        if args.resume_existing_counter_host_root is None:
            counter_case_root = _counter_container_case_root(
                args=args, case_id=case_id)
            counter_host_case_root = _counter_host_case_root(
                args=args, case_id=case_id)
            counter_root = f"{counter_case_root}/counter/msprof_op"
            counter_log = (
                f"{counter_case_root}/logs/msprof_op_"
                f"{args.resume_log_tag}.log")
            _run_remote(
                host=args.host, container=args.container,
                container_repo=args.container_repo, device=args.device,
                command=[
                    args.msprof, "op", f"--output={counter_root}",
                    "--aic-metrics=PipeUtilization,Memory",
                    "--kernel-name=FusedInferAttentionScore", "--launch-count=3",
                    f"--application={shlex.join(runner_command)}",
                ], remote_log_path=counter_log,
                log_path=local_log_root / "msprof_op.host.log")
        else:
            counter_host_case_root = _existing_counter_host_case_root(
                args=args, case_id=case_id)
        local_case_root = args.output_root / case_id
        if local_case_root.exists():
            raise FileExistsError(
                f"Ascend resume local case root already exists: {local_case_root}")
        _copy_counter_raw(
            args=args, counter_host_case_root=counter_host_case_root,
            local_case_root=local_case_root, local_log_root=local_log_root,
            log_tag="counter_after_resume")
        summaries = summary_paths
        timing = _build_timing_audit(
            case_id=case_id, workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256,
            op_summary_paths=summaries,
            profile_warmup_iterations=args.profile_warmup_iterations)
        if _timing_signature(timing) != _timing_signature(provisional_timing):
            raise ValueError(
                "Ascend resumed Timeline summaries disagree with the staged "
                f"stability decision for {case_id}")
        basic_paths, pipe_paths, memory_paths = _counter_paths(
            local_case_root / "counter" / "msprof_op")
        program = None
        for basic_path in basic_paths:
            with basic_path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            candidate_program = row["Op Name"]
            if program is None:
                program = candidate_program
            elif candidate_program != program:
                raise ValueError("Ascend counter samples have different programs")
        if program is None:
            raise AssertionError("Ascend counter samples are unexpectedly empty")
        _copy_timeline_program_hashes(
            args=args, remote_host_case_root=remote_host_case_root,
            local_stage_root=staging_root / case_id,
            local_log_root=local_log_root)
        hash_paths = [_program_hash_path(
            staging_root / case_id / "golden" / f"window_{window}", program)
            for window in range(1, 4)]
        candidate = build_ascend_counter_candidate(
            case_id=case_id, workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256,
            counter_op_basic_paths=basic_paths, counter_pipe_paths=pipe_paths,
            counter_memory_paths=memory_paths,
            timeline_op_summary_paths=summaries,
            timeline_program_hash_paths=hash_paths)
        candidate_path = local_case_root / "candidate" / f"{case_id}.counter.json"
        candidate_path.parent.mkdir()
        write_json_new(candidate_path, candidate)
        record = {
            "case_id": case_id,
            "workload_digest": item["workload_digest"],
            "status": "pass",
            "timing_audit": str(timing_path),
            "counter_candidate": str(candidate_path),
        }
        write_json_new(local_case_root / "completion.json", record)
        records.append(record)
    write_json_new(args.audit_root / "completion_manifest.json", {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_completion",
        "hardware_id": "ascend_910b",
        "source_snapshot_sha256": source_sha256,
        "input_manifest": str(args.capture_input_manifest),
        "records": records,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-input-manifest", required=True, type=Path)
    parser.add_argument("--remote-capture-input-manifest", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument(
        "--remote-output-root", required=True,
        help="absolute host path that corresponds to --remote-container-root")
    parser.add_argument(
        "--remote-container-root", required=True,
        help="absolute container path for msprof output and runner logs")
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--container-repo", required=True)
    parser.add_argument("--runner-python", required=True)
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--profile-warmup-iterations", type=int, default=20,
        help=("unmeasured runner public-op calls before each Timeline final "
              "task; record this in the timing audit"))
    parser.add_argument("--skip-case", action="append", default=[])
    parser.add_argument("--continue-on-timing-failure", action="store_true")
    parser.add_argument("--resume-existing-timeline", action="store_true")
    parser.add_argument("--resume-source-output-root", type=Path)
    parser.add_argument("--resume-log-tag")
    parser.add_argument(
        "--resume-existing-counter-host-root",
        help=("absolute physical host root of an immutable, already-completed "
              "op-mode counter capture; only valid with Timeline resume"))
    parser.add_argument(
        "--counter-output-container-root",
        help=("absolute non-symlink container root for msprof op output; "
              "application paths may remain under a separate short root"))
    parser.add_argument(
        "--counter-output-host-root",
        help=("matching absolute host root for --counter-output-container-root; "
              "required when the container counter root differs from the "
              "Timeline container root"))
    args = parser.parse_args()
    if args.profile_warmup_iterations < 0:
        raise ValueError("--profile-warmup-iterations must be non-negative")

    manifest = _load_inputs(args.capture_input_manifest)
    if args.resume_existing_timeline:
        if args.resume_source_output_root is None:
            raise ValueError(
                "Ascend Timeline resume requires --resume-source-output-root")
        if (args.resume_log_tag is None
                or re.fullmatch(r"[a-z0-9_]+", args.resume_log_tag) is None):
            raise ValueError(
                "Ascend Timeline resume requires a normalized --resume-log-tag")
        if args.output_root.exists() or args.audit_root.exists():
            raise FileExistsError(
                "Ascend Timeline resume requires new output and audit roots")
    elif args.resume_source_output_root is not None:
        raise ValueError(
            "--resume-source-output-root requires --resume-existing-timeline")
    elif args.resume_log_tag is not None:
        raise ValueError("--resume-log-tag requires --resume-existing-timeline")
    elif args.resume_existing_counter_host_root is not None:
        raise ValueError(
            "--resume-existing-counter-host-root requires "
            "--resume-existing-timeline")
    elif args.output_root.exists() or args.audit_root.exists():
        raise FileExistsError("Ascend full-capture output and audit roots must be new")
    if not args.source_snapshot.is_file():
        raise FileNotFoundError(f"Ascend source snapshot does not exist: {args.source_snapshot}")
    if args.device < 0:
        raise ValueError("Ascend device must be non-negative")
    if not args.remote_output_root.startswith("/"):
        raise ValueError("Ascend remote host output root must be absolute")
    if not args.remote_container_root.startswith("/"):
        raise ValueError("Ascend remote container output root must be absolute")
    if (args.counter_output_container_root is not None
            and not args.counter_output_container_root.startswith("/")):
        raise ValueError("Ascend counter output container root must be absolute")
    if (args.counter_output_host_root is not None
            and not args.counter_output_host_root.startswith("/")):
        raise ValueError("Ascend counter output host root must be absolute")
    remote_manifest = _read_remote_inputs(
        host=args.host, container=args.container,
        remote_manifest_path=args.remote_capture_input_manifest)
    _require_remote_runner_inputs(
        local_manifest=manifest, remote_manifest=remote_manifest)
    known_cases = {item["case_id"] for item in manifest["cases"]}
    skip_cases = set(args.skip_case)
    if skip_cases - known_cases:
        raise ValueError("Ascend skip-case is absent from the manifest")
    source_sha256 = _sha256(args.source_snapshot)

    if args.resume_existing_timeline:
        _resume_existing_timeline(
            args=args, remote_manifest=remote_manifest, skip_cases=skip_cases,
            source_sha256=source_sha256)
        return

    args.output_root.mkdir(parents=True)
    args.audit_root.mkdir(parents=True)
    shutil.copyfile(args.source_snapshot, args.output_root / "source_snapshot.tar")
    write_json_new(args.output_root / "capture_execution.json", {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "owner": "fa_serving_coverage_20260915",
        "host": args.host,
        "container": args.container,
        "device": args.device,
        "profile_warmup_iterations": args.profile_warmup_iterations,
        "container_repo": args.container_repo,
        "remote_output_root": args.remote_output_root,
        "remote_container_root": args.remote_container_root,
        "remote_capture_input_manifest": args.remote_capture_input_manifest,
        "counter_output_container_root": args.counter_output_container_root,
        "counter_output_host_root": args.counter_output_host_root,
        "source_snapshot_sha256": source_sha256,
    })
    write_json_new(args.output_root / "remote_capture_input_manifest.json", remote_manifest)
    records_by_case = {}
    timing_stages = []
    stop_after_timing_failure = False
    # Each Timeline export is complete before this task is submitted.  Keep
    # the transfer off the device-control path so the next case's three
    # windows can start while its predecessor's six small CSVs are copied.
    with ThreadPoolExecutor(max_workers=1) as summary_transfers:
        for item in remote_manifest["cases"]:
            case_id = item["case_id"]
            if case_id in skip_cases:
                records_by_case[case_id] = {"case_id": case_id, "status": "skipped"}
                continue
            remote_host_case_root = f"{args.remote_output_root}/{case_id}"
            remote_container_case_root = f"{args.remote_container_root}/{case_id}"
            local_log_root = args.output_root / "host_logs" / case_id
            local_log_root.mkdir(parents=True)
            runner_command = [
                args.runner_python, item["runner"], *item["runner_arguments"],
                "--profile-warmup-iterations", str(args.profile_warmup_iterations),
            ]
            for window in range(1, 4):
                remote_window_root = (
                    f"{remote_container_case_root}/golden/window_{window}/msprof")
                remote_log_path = (
                    f"{remote_container_case_root}/logs/msprof_window_{window}.log")
                command = [
                    args.msprof, f"--output={remote_window_root}", "--ai-core=on",
                    "--aic-metrics=PipeUtilization", "--duration=8",
                    f"--application={shlex.join(runner_command)}",
                ]
                _run_remote(
                    host=args.host, container=args.container,
                    container_repo=args.container_repo, device=args.device,
                    command=command, remote_log_path=remote_log_path,
                    log_path=local_log_root / f"msprof_window_{window}.host.log")
                _run_remote(
                    host=args.host, container=args.container,
                    container_repo=args.container_repo, device=args.device,
                    command=[args.msprof, "--export=on", "--type=text",
                             f"--output={remote_window_root}"],
                    remote_log_path=(
                        f"{remote_container_case_root}/logs/"
                        f"msprof_window_{window}_export.log"),
                    log_path=local_log_root / f"msprof_window_{window}_export.host.log")
            summary_stage_root = (
                args.output_root / "_timeline_summary_staging" / case_id)
            stage = {
                "item": item,
                "case_id": case_id,
                "local_log_root": local_log_root,
                "remote_host_case_root": remote_host_case_root,
                "remote_container_case_root": remote_container_case_root,
                "runner_command": runner_command,
                "summary_stage_root": summary_stage_root,
                "summary_future": summary_transfers.submit(
                    _copy_timeline_summaries,
                    host=args.host, remote_host_case_root=remote_host_case_root,
                    local_stage_root=summary_stage_root,
                    local_log_root=local_log_root),
            }
            timing_stages.append(stage)

    finalized_stages = []
    for stage in timing_stages:
        item = stage["item"]
        summary_paths = stage["summary_future"].result()
        stage["summary_paths"] = summary_paths
        provisional_timing = _build_timing_audit(
            case_id=stage["case_id"], workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256, op_summary_paths=summary_paths,
            profile_warmup_iterations=args.profile_warmup_iterations)
        stage["provisional_timing"] = provisional_timing
        finalized_stages.append(stage)
        if provisional_timing["status"] == "pass":
            counter_case_root = _counter_container_case_root(
                args=args, case_id=stage["case_id"])
            stage["counter_host_case_root"] = _counter_host_case_root(
                args=args, case_id=stage["case_id"])
            counter_root = f"{counter_case_root}/counter/msprof_op"
            counter_log = (
                f"{stage['remote_container_case_root']}/logs/msprof_op.log")
            _run_remote(
                host=args.host, container=args.container,
                container_repo=args.container_repo, device=args.device,
                command=[
                    args.msprof, "op", f"--output={counter_root}",
                    "--aic-metrics=PipeUtilization,Memory",
                    "--kernel-name=FusedInferAttentionScore", "--launch-count=3",
                    f"--application={shlex.join(stage['runner_command'])}",
                ], remote_log_path=counter_log,
                log_path=stage["local_log_root"] / "msprof_op.host.log")
        elif not args.continue_on_timing_failure:
            stop_after_timing_failure = True
            break
    timing_stages = finalized_stages
    for stage in timing_stages:
        item = stage["item"]
        case_id = stage["case_id"]
        local_case_root = args.output_root / case_id
        if stage["provisional_timing"]["status"] == "pass":
            _copy_counter_raw(
                args=args,
                counter_host_case_root=stage["counter_host_case_root"],
                local_case_root=local_case_root,
                local_log_root=stage["local_log_root"],
                log_tag="counter_after_device_phase")
        summaries = stage["summary_paths"]
        timing = _build_timing_audit(
            case_id=case_id, workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256, op_summary_paths=summaries,
            profile_warmup_iterations=args.profile_warmup_iterations)
        if _timing_signature(timing) != _timing_signature(stage["provisional_timing"]):
            raise ValueError(
                "Ascend copied Timeline summaries disagree with the staged "
                f"stability decision for {case_id}")
        timing_path = args.audit_root / f"{case_id}.timing.json"
        write_json_new(timing_path, timing)
        if timing["status"] != "pass":
            local_case_root.mkdir(parents=True)
            failure = _timing_failure_record(
                case_id=case_id, workload_digest=item["workload_digest"],
                timing_audit_path=timing_path)
            write_json_new(local_case_root / "timing_failure.json", failure)
            records_by_case[case_id] = failure
            continue
        basic_paths, pipe_paths, memory_paths = _counter_paths(
            local_case_root / "counter" / "msprof_op")
        program = None
        for basic_path in basic_paths:
            with basic_path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            candidate_program = row["Op Name"]
            if program is None:
                program = candidate_program
            elif candidate_program != program:
                raise ValueError("Ascend counter samples have different programs")
        if program is None:
            raise AssertionError("Ascend counter samples are unexpectedly empty")
        _copy_timeline_program_hashes(
            args=args, remote_host_case_root=stage["remote_host_case_root"],
            local_stage_root=stage["summary_stage_root"],
            local_log_root=stage["local_log_root"])
        hash_paths = [_program_hash_path(
            stage["summary_stage_root"] / "golden" / f"window_{window}", program)
            for window in range(1, 4)]
        candidate = build_ascend_counter_candidate(
            case_id=case_id, workload_digest=item["workload_digest"],
            source_snapshot_sha256=source_sha256,
            counter_op_basic_paths=basic_paths, counter_pipe_paths=pipe_paths,
            counter_memory_paths=memory_paths,
            timeline_op_summary_paths=summaries,
            timeline_program_hash_paths=hash_paths)
        candidate_path = local_case_root / "candidate" / f"{case_id}.counter.json"
        candidate_path.parent.mkdir()
        write_json_new(candidate_path, candidate)
        record = {
            "case_id": case_id,
            "workload_digest": item["workload_digest"],
            "status": "pass",
            "timing_audit": str(timing_path),
            "counter_candidate": str(candidate_path),
        }
        write_json_new(local_case_root / "completion.json", record)
        records_by_case[case_id] = record
    records = [
        records_by_case[item["case_id"]]
        for item in remote_manifest["cases"]
        if item["case_id"] in records_by_case
    ]
    write_json_new(args.audit_root / "completion_manifest.json", {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_completion",
        "hardware_id": "ascend_910b",
        "source_snapshot_sha256": source_sha256,
        "input_manifest": str(args.capture_input_manifest),
        "records": records,
    })
    if stop_after_timing_failure:
        raise RuntimeError("Ascend timing stability failed; device phase stopped")


if __name__ == "__main__":
    main()
