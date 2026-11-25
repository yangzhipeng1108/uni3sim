#!/usr/bin/env python3
"""Capture frozen FA serving candidates on MetaX C500 one case at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_audit import write_json_new
from dataset_pipeline.kernel_dataset.fa_serving_capture_metax_c500 import (
    build_c500_canary_closure,
    build_c500_case_evidence,
)


_INPUT_FIELDS = frozenset((
    "schema_version", "capture_kind", "manifest_path", "manifest_sha256",
    "hardware_id", "case_count", "cases",
))
_CASE_FIELDS = frozenset((
    "case_id", "workload_digest", "capture_metadata", "runner",
    "runner_arguments", "runner_workload",
))

# A selected mcProfiler header set may omit optional chart-only sections, but
# never evidence that the candidate parser consumes.  In particular, Internal
# keeps the required RoofLine record; a whole-shape report is not an alternate
# way to satisfy this contract.
_REQUIRED_MCPROFILER_HEADNAMES = frozenset((
    "Summary",
    "CE Statistics",
    "Memory Statistics",
    "Workgroup Memory",
    "Occupancy",
    "GPU Throughput Statistics",
    "Compute workload",
    "Internal",
))
_OPTIONAL_MCPROFILER_HEADNAMES = frozenset((
    "ISU Statistics",
    "Instruction Statistics",
))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mcprofiler_lock_path() -> str:
    if "MIDLAYER_C500_MCPROFILER_LOCK" not in os.environ:
        raise ValueError(
            "MIDLAYER_C500_MCPROFILER_LOCK is required for mcProfiler execution"
        )
    return os.environ["MIDLAYER_C500_MCPROFILER_LOCK"]


def _validate_inputs(manifest: object, *, require_local_metadata: bool) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != _INPUT_FIELDS:
        raise ValueError("C500 capture-input manifest fields are not exact")
    if manifest["hardware_id"] != "metax_c500":
        raise ValueError("C500 runner received a non-C500 input manifest")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != manifest["case_count"]:
        raise ValueError("C500 capture-input manifest case count is invalid")
    case_ids = []
    for item in cases:
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError("C500 capture-input case fields are not exact")
        if not isinstance(item["case_id"], str) or not item["case_id"]:
            raise ValueError("C500 case_id must be non-empty")
        if (not isinstance(item["workload_digest"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["workload_digest"]) is None):
            raise ValueError("C500 workload_digest must be a SHA256")
        if (not isinstance(item["runner"], str) or not item["runner"]
                or not isinstance(item["runner_arguments"], list)
                or any(not isinstance(value, str)
                       for value in item["runner_arguments"])):
            raise TypeError("C500 runner contract is invalid")
        if require_local_metadata and not Path(item["capture_metadata"]).is_file():
            raise FileNotFoundError("C500 local capture metadata is absent")
        case_ids.append(item["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("C500 case IDs are not unique")
    return manifest


def _load_inputs(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return _validate_inputs(json.load(stream), require_local_metadata=True)


def _read_remote_inputs(*, host: str, container: str,
                        remote_manifest: str) -> dict:
    command = shlex.join(("docker", "exec", container, "cat", remote_manifest))
    result = subprocess.run(
        ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command),
        cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "C500 could not read its remote capture-input manifest: "
            f"{result.stderr.strip()}")
    try:
        return _validate_inputs(json.loads(result.stdout), require_local_metadata=False)
    except json.JSONDecodeError as error:
        raise ValueError("C500 remote capture-input manifest is not JSON") from error


def _bind_remote_inputs(local: dict, remote: dict) -> None:
    if len(local["cases"]) != len(remote["cases"]):
        raise ValueError("C500 local and remote input counts differ")
    for local_case, remote_case in zip(local["cases"], remote["cases"], strict=True):
        for field in ("case_id", "workload_digest", "runner"):
            if local_case[field] != remote_case[field]:
                raise ValueError(f"C500 remote input disagrees on {field}")
        for field in ("capture_metadata", "runner_workload"):
            remote_path = PurePosixPath(remote_case[field])
            if not remote_path.is_absolute() or ".." in remote_path.parts:
                raise ValueError(f"C500 remote {field} path is not normalized")
        for flag, expected in (
                ("--capture-metadata-json", remote_case["capture_metadata"]),):
            arguments = remote_case["runner_arguments"]
            if flag not in arguments:
                raise ValueError(f"C500 remote runner lacks {flag}")
            position = arguments.index(flag)
            if position + 1 >= len(arguments) or arguments[position + 1] != expected:
                raise ValueError(f"C500 remote runner has invalid {flag}")


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"C500 host command failed: log={log_path}")


def _run_remote(*, host: str, container: str, container_repo: str,
                command: list[str], remote_log: str, local_log: Path,
                environment: tuple[str, ...] = ()) -> None:
    remote_log_parent = str(PurePosixPath(remote_log).parent)
    command_text = shlex.join(command)
    wrapped = (
        "mkdir -p " + shlex.quote(remote_log_parent) + " && { printf '%s\\n' "
        + shlex.quote(command_text) + "; " + command_text + "; } > "
        + shlex.quote(remote_log) + " 2>&1")
    docker_command = ["docker", "exec", "-w", container_repo]
    for value in environment:
        docker_command.extend(("-e", value))
    docker_command.extend((container, "bash", "-lc", wrapped))
    _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         host, shlex.join(tuple(docker_command))],
        log_path=local_log)


def _perkernel_output_check_command(remote_allk_dir: str) -> str:
    """Return a shell check with the same admission rule as the parser.

    ``report.txt.json`` and numbered ``period`` rollups are whole-shape
    artifacts.  Neither can stand in for a per-invocation layer-1 record.
    """
    files = (
        "find " + shlex.quote(remote_allk_dir)
        + " -maxdepth 1 -type f -printf '%f\\n'")
    return (
        files + " | awk '"
        "$0 ~ /^[0-9]+_.+\\.txt\\.json$/ && "
        "$0 !~ /^[0-9]+_period.*\\.txt\\.json$/ { found=1 } "
        "END { exit(found ? 0 : 1) }'")


def _load_metadata(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError("C500 capture metadata must be a mapping")
    return metadata


def _positive_layer1_attempts(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("C500 layer-1 attempts must be positive")
    return value


def _positive_mcprofiler_timeout_seconds(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("C500 mcProfiler timeout must be positive")
    return value


def _positive_profiler_wait_seconds(value: int) -> int:
    """Validate the runner-side RPC attach window for one layer-1 pass."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("C500 profiler attach wait must be positive")
    return value


def _validated_mcprofiler_headnames(values: list[str] | None) -> tuple[str, ...]:
    """Validate an explicit profiler schema without weakening its evidence."""

    if values is None:
        return ()
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("C500 mcProfiler headnames must be non-empty strings")
    selected = tuple(values)
    if len(selected) != len(set(selected)):
        raise ValueError("C500 mcProfiler headnames must be unique")
    allowed = _REQUIRED_MCPROFILER_HEADNAMES | _OPTIONAL_MCPROFILER_HEADNAMES
    unknown = set(selected) - allowed
    if unknown:
        raise ValueError(f"C500 mcProfiler headnames are unknown: {sorted(unknown)}")
    missing = _REQUIRED_MCPROFILER_HEADNAMES - set(selected)
    if missing:
        raise ValueError(
            "C500 mcProfiler headnames omit required candidate evidence: "
            f"{sorted(missing)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-input-manifest", required=True, type=Path)
    parser.add_argument("--remote-capture-input-manifest", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--event-timer-sanity-path", required=True, type=Path)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--remote-event-output-root", required=True)
    parser.add_argument("--container-repo", required=True)
    parser.add_argument("--canary-closure-output", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--runner-python", required=True)
    parser.add_argument("--mctracer", default="/opt/maca/bin/mcTracer")
    parser.add_argument("--mcprofiler", default="/opt/mcProfiler-linux/mcProfiler")
    parser.add_argument(
        "--layer1-attempts", type=int, default=5,
        help=("maximum fresh mcProfiler layer-1 attempts for one case; only a "
              "numbered per-invocation artifact is admissible"))
    parser.add_argument(
        "--mcprofiler-timeout-seconds", type=int, default=600,
        help=("wall-clock bound for one mcProfiler invocation; timeout is a "
              "failed fresh attempt and never yields candidate evidence"))
    parser.add_argument(
        "--profiler-wait-seconds", type=int, default=15,
        help=("runner-side post-warmup delay that keeps the RPC attach window "
              "open before the one real profiled FlashAttention call"))
    parser.add_argument(
        "--mcprofiler-headname", action="append",
        help=("explicit validated mcProfiler report header; repeat for every "
              "required header when diagnosing a tool report-rendering failure"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    mcprofiler_lock_path = _mcprofiler_lock_path()

    local_inputs = _load_inputs(args.capture_input_manifest)
    if args.output_root.exists() or args.audit_root.exists():
        raise FileExistsError("C500 full-capture output and audit roots must be new")
    if not args.source_snapshot.is_file():
        raise FileNotFoundError("C500 source snapshot does not exist")
    if not args.event_timer_sanity_path.is_file():
        raise FileNotFoundError("C500 Event timer sanity artifact does not exist")
    if args.device != 0:
        raise ValueError("C500 serving capture is fixed to device 0")
    args.layer1_attempts = _positive_layer1_attempts(args.layer1_attempts)
    args.mcprofiler_timeout_seconds = _positive_mcprofiler_timeout_seconds(
        args.mcprofiler_timeout_seconds)
    args.profiler_wait_seconds = _positive_profiler_wait_seconds(
        args.profiler_wait_seconds)
    args.mcprofiler_headnames = _validated_mcprofiler_headnames(
        args.mcprofiler_headname)
    for path in (
            args.remote_output_root, args.remote_event_output_root,
            args.container_repo):
        if not path.startswith("/"):
            raise ValueError("C500 remote paths must be absolute")
    event_owner = args.container_repo.rstrip("/") + "/experiment_logs/kernel_golden/"
    if not args.remote_event_output_root.startswith(event_owner):
        raise ValueError(
            "C500 remote Event outputs must stay under the snapshot "
            "experiment_logs/kernel_golden owner")
    if (args.canary_closure_output is not None
            and len(local_inputs["cases"]) != 1):
        raise ValueError("C500 canary closure output requires exactly one case")
    remote_inputs = _read_remote_inputs(
        host=args.host, container=args.container,
        remote_manifest=args.remote_capture_input_manifest)
    _bind_remote_inputs(local_inputs, remote_inputs)
    source_snapshot_sha256 = _sha256(args.source_snapshot)

    args.output_root.mkdir(parents=True)
    args.audit_root.mkdir(parents=True)
    shutil.copyfile(args.source_snapshot, args.output_root / "source_snapshot.tar")
    write_json_new(args.output_root / "capture_execution.json", {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "owner": "fa_serving_coverage_20260915",
        "host": args.host,
        "container": args.container,
        "container_repo": args.container_repo,
        "device": args.device,
        "remote_output_root": args.remote_output_root,
        "remote_event_output_root": args.remote_event_output_root,
        "remote_capture_input_manifest": args.remote_capture_input_manifest,
        "mcprofiler_timeout_seconds": args.mcprofiler_timeout_seconds,
        "profiler_wait_seconds": args.profiler_wait_seconds,
        "source_snapshot_sha256": source_snapshot_sha256,
    })
    stages = []
    for local_case, remote_case in zip(
            local_inputs["cases"], remote_inputs["cases"], strict=True):
        case_id = remote_case["case_id"]
        metadata = _load_metadata(Path(local_case["capture_metadata"]))
        case_root = args.output_root / case_id
        host_case_root = f"{args.remote_output_root}/{case_id}"
        container_case_root = f"{args.remote_output_root}/{case_id}"
        remote_event_path = f"{args.remote_event_output_root}/{case_id}.event.json"
        local_logs = args.output_root / "host_logs" / case_id
        local_logs.mkdir(parents=True)
        runner_path = PurePosixPath(remote_case["runner"])
        if runner_path.is_absolute() or ".." in runner_path.parts:
            raise ValueError(
                "C500 runner path must be normalized and repository-relative")
        runner = [
            args.runner_python,
            str(PurePosixPath(args.container_repo) / runner_path),
            *remote_case["runner_arguments"],
        ]
        event_runner = [
            *runner,
            "--event-output", remote_event_path,
            "--event-case-id", case_id,
            "--event-baseline-commit", "0c530a00a9f7c1135210e99b843cce3f8250e3ac",
            "--event-source-snapshot-sha256", source_snapshot_sha256,
            "--event-timer-sanity-sha256", _sha256(args.event_timer_sanity_path),
            "--event-canonical-attributes-sha256", metadata["logical_identity_sha256"],
            "--event-warmup", "5", "--event-samples", "20",
            "--event-graph-repetitions", "50", "200",
        ]
        event_command = [
            "bash", "-lc", "mkdir -p "
            + shlex.quote(str(PurePosixPath(remote_event_path).parent))
            + " && " + shlex.join(event_runner)
            + " && chmod 0644 " + shlex.quote(remote_event_path),
        ]
        _run_remote(
            host=args.host, container=args.container, container_repo=args.container_repo,
            command=event_command,
            remote_log=f"{container_case_root}/logs/event.log",
            local_log=local_logs / "event.host.log",
            environment=(f"MACA_VISIBLE_DEVICES={args.device}",))
        trace_parent = f"{container_case_root}/traces"
        trace_command = [
            "bash", "-lc",
            "mkdir -p " + shlex.quote(trace_parent) + " && cd "
            + shlex.quote(trace_parent) + " && "
            + shlex.join([args.mctracer, "--odname", case_id, *runner,
                          "--profiler_wait", "0", "--use_marker"]),
        ]
        _run_remote(
            host=args.host, container=args.container, container_repo=args.container_repo,
            command=trace_command,
            remote_log=f"{container_case_root}/logs/mctracer.log",
            local_log=local_logs / "mctracer.host.log",
            environment=(f"MACA_VISIBLE_DEVICES={args.device}",))
        profiler_runner = [
            *runner, "--profiler_wait", str(args.profiler_wait_seconds),
            "--use_marker",
        ]
        child_command = (
            "export MACA_PATH=/opt/maca; "
            "export LD_LIBRARY_PATH=/opt/maca/lib/:$LD_LIBRARY_PATH; "
            + shlex.join(profiler_runner))
        selected_allk_dir = None
        for attempt in range(1, args.layer1_attempts + 1):
            remote_allk_dir = (
                f"{container_case_root}/mcprofiler_out/_allk_attempt_{attempt}")
            profiler_command = [
                "flock", "-n", mcprofiler_lock_path,
                "timeout", "--kill-after=30s",
                f"{args.mcprofiler_timeout_seconds}s",
                args.mcprofiler, "perf_exec", "--cmdline", child_command,
                "--casename", case_id, "--output", remote_allk_dir,
                "--host", "127.0.0.1", "--port", "50123", "--custom", "--per-kernel",
            ]
            if args.mcprofiler_headnames:
                profiler_command.extend(("--headnames", *args.mcprofiler_headnames))
            checked_command = [
                "bash", "-lc", shlex.join(profiler_command) + " && "
                + _perkernel_output_check_command(remote_allk_dir),
            ]
            try:
                _run_remote(
                    host=args.host, container=args.container,
                    container_repo=args.container_repo, command=checked_command,
                    remote_log=(f"{container_case_root}/logs/"
                                f"mcprofiler_layer1_attempt_{attempt}.log"),
                    local_log=(local_logs /
                               f"mcprofiler_layer1_attempt_{attempt}.host.log"),
                    environment=(f"MACA_VISIBLE_DEVICES={args.device}", "MACA_PATH=/opt/maca",
                                 "MACA_ETC_PATH=/opt/maca/etc"))
            except RuntimeError:
                continue
            selected_allk_dir = f"mcprofiler_out/_allk_attempt_{attempt}"
            break
        if selected_allk_dir is None:
            raise RuntimeError("C500 layer-1 produced no numbered per-invocation files")
        # The remote artifacts are immutable once mcProfiler has returned.  Do
        # not make the single globally serialized profiler wait for a network
        # copy: retrieve and validate every completed case after the device
        # phase instead.
        stages.append({
            "remote_case": remote_case,
            "metadata": metadata,
            "case_id": case_id,
            "case_root": case_root,
            "host_case_root": host_case_root,
            "remote_event_path": remote_event_path,
            "local_logs": local_logs,
            "selected_allk_dir": selected_allk_dir,
        })

    records = []
    for stage in stages:
        remote_case = stage["remote_case"]
        metadata = stage["metadata"]
        case_id = stage["case_id"]
        case_root = stage["case_root"]
        host_case_root = stage["host_case_root"]
        remote_event_path = stage["remote_event_path"]
        local_logs = stage["local_logs"]
        selected_allk_dir = stage["selected_allk_dir"]
        _run(
            ["scp", "-r", f"{args.host}:{host_case_root}", str(case_root)],
            log_path=local_logs / "copy_raw.log")
        event_path = case_root / "golden" / "event.json"
        event_path.parent.mkdir()
        _run(
            ["scp", f"{args.host}:{remote_event_path}", str(event_path)],
            log_path=local_logs / "copy_event.log")
        trace_paths = sorted((case_root / "traces" / case_id).glob("tracer_out*.json"))
        timing, candidate = build_c500_case_evidence(
            case_id=case_id, workload_digest=remote_case["workload_digest"],
            logical_identity_sha256=metadata["logical_identity_sha256"],
            source_snapshot_sha256=source_snapshot_sha256,
            case_root=case_root, event_path=event_path, trace_paths=trace_paths,
            allk_dir=case_root / selected_allk_dir)
        timing_path = args.audit_root / f"{case_id}.timing.json"
        candidate_path = case_root / "candidate" / f"{case_id}.counter.json"
        candidate_path.parent.mkdir()
        write_json_new(timing_path, timing)
        write_json_new(candidate_path, candidate)
        record = {
            "case_id": case_id,
            "workload_digest": remote_case["workload_digest"],
            "status": "pass",
            "timing_audit": str(timing_path),
            "counter_candidate": str(candidate_path),
        }
        write_json_new(case_root / "completion.json", record)
        records.append(record)
    if args.canary_closure_output is not None:
        record = records[0]
        closure = build_c500_canary_closure(
            case_id=record["case_id"], workload_digest=record["workload_digest"],
            timing_audit_path=Path(record["timing_audit"]),
            counter_candidate_path=Path(record["counter_candidate"]))
        write_json_new(args.canary_closure_output, closure)
    write_json_new(args.audit_root / "completion_manifest.json", {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_metax_c500_completion",
        "hardware_id": "metax_c500",
        "source_snapshot_sha256": source_snapshot_sha256,
        "input_manifest": str(args.capture_input_manifest),
        "records": records,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
