"""Execute the process and server stages of one prepared Serving experiment."""

from __future__ import annotations

from contextlib import ExitStack

from dataclasses import asdict, fields

import hashlib

import json

import math

import os

from pathlib import Path

import re

import subprocess

import tempfile

from typing import Any

from urllib import error as urllib_error

from urllib import request as urllib_request

from evaluation.serving.experiments.experiment_manifest import ManifestValidationError

from evaluation.serving.metrics.production_trace_report import (
    render_production_trace_sweep_report,
)

from evaluation.serving.metrics.latency_distribution import (
    latency_distribution_manifest,
)

from evaluation.serving.metrics.interactivity_tgs import (
    arrival_rate_sweep_analysis_manifest,
    load_interactivity_tgs_contract,
    sweep_metric_point_from_manifest,
)

from evaluation.serving.metrics.slo_goodput import (
    load_slo_contract,
    slo_goodput_manifest,
)

from evaluation.serving.metrics.system_resource_analysis import (
    critical_path_manifest,
    system_resource_manifest,
)

from evaluation.serving.simulation.execution_artifact import (
    execution_artifact_manifest,
    execution_step_components_manifest,
    performance_timing_status,
    prefix_alignment_manifest,
)

from src.kernel.hardware_spec import HWSpec

from src.kernel.kernel_base import predict

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)

from src.serving.backend.vllm_scheduler_process import (
    VLLMSchedulerRuntimeInfo,
    vllm_scheduler_runtime_sha256,
    vllm_scheduler_source_identity,
)

from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig

from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
)

from src.serving.backend.mtp_acceptance_replay import (
    mtp_acceptance_step_from_backend_step,
    mtp_acceptance_steps_manifest,
)

from src.serving.deployment.af_disagg import build_af_disagg_plan

from src.serving.deployment.colocated import build_colocated_plan

from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouteConfig,
    RouterConfig,
    WorkerPoolConfig,
)

from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    deployment_plan_manifest,
)

from src.serving.deployment.pd_disagg import build_pd_disagg_plan

from src.serving.kv_cache.kv_cache_events import KVCacheEvent

from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager

from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)

from src.serving.parallelism.parallel_config import ParallelConfig

from src.serving.request_gen.production_trace import (
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
    build_production_trace_replay,
    load_production_trace,
    production_trace_manifest,
    production_trace_replay_manifest,
)

from src.serving.request_gen.request_stream import (
    RequestStream,
    request_stream_sha256,
)

from src.serving.schedule.backend_protocol import (
    BackendConfig,
    scheduler_effective_prefix_hit_tokens,
)

from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)

from src.serving.schedule.schedule_trace import (
    RouteDecision,
    ScheduleTrace,
    ScheduleTraceEntry,
)

from src.serving.schedule.single_pool_engine import run_single_pool_schedule

from src.serving.metrics.serving_metrics import compute_serving_metrics

from src.serving.execution import ServingExecutionContext

from src.serving.simulation import (
    MultiSimulation,
    Simulation,
    SimulationConfig,
    SimulationResult,
    SimulationStep,
)

from src.serving.workload.model_config import (
    tp_collective_payload_bytes,
)

from src.serving.workload.graph_padding import GraphExecutionConfig

from src.serving.workload.model_config import ModelConfig, load_model_config

from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)

from src.serving.workload.shape_inference import infer_gemm_dimensions

from src.serving.workload.workload_graph import WorkloadKind

from evaluation.json_shape import write_json_file as _write_json

from evaluation.serving.experiments.production_trace_colocated_validation import (
    tp2_collective_bytes_from_schedule_and_model as _tp2_collective_bytes_from_schedule_and_model,
    validate_colocated_ep2_production_trace_point as _validate_colocated_ep2_production_trace_point,
    validate_colocated_kv_lifecycle_replay as _validate_colocated_kv_lifecycle_replay,
    validate_colocated_pp2_production_trace_point as _validate_colocated_pp2_production_trace_point,
    validate_colocated_production_trace_point as _validate_colocated_production_trace_point,
    validate_colocated_tp1_production_trace_point as _validate_colocated_tp1_production_trace_point,
    validate_colocated_tp2_pp2_production_trace_point as _validate_colocated_tp2_pp2_production_trace_point,
    validate_colocated_tp2_production_trace_point as _validate_colocated_tp2_production_trace_point,
)

from evaluation.serving.experiments.production_trace_contracts import (
    ROOT,
    backend_step_from_output_manifest as _backend_step_from_output_manifest,
    canonical_production_trace_metrics_manifest as _canonical_production_trace_metrics_manifest,
    expected_performance_timing_status as _expected_performance_timing_status,
    explicit_expert_routing_input as _explicit_expert_routing_input,
    interval_union_duration_from_output as _interval_union_duration_from_output,
    json_round_trip as _json_round_trip,
    kv_cache_event_from_output_manifest as _kv_cache_event_from_output_manifest,
    load_expected_hardware as _load_expected_hardware,
    maximum_entropy_mtp_assumption as _maximum_entropy_mtp_assumption,
    production_trace_replays_from_command_input as _production_trace_replays_from_command_input,
    reject_json_constant as _reject_json_constant,
    replay_pinned_parallel_scheduler as _replay_pinned_parallel_scheduler,
    require_exact_output_object as _require_exact_output_object,
    require_output_list as _require_output_list,
    route_decision_from_output_manifest as _route_decision_from_output_manifest,
    schedule_step_from_output_manifest as _schedule_step_from_output_manifest,
    unique_json_object as _unique_json_object,
    validate_canonical_production_trace_step_duration as _validate_canonical_production_trace_step_duration,
    validate_execution_components_output_schema as _validate_execution_components_output_schema,
    validate_host_kv_system_resources as _validate_host_kv_system_resources,
    validate_production_trace_capacity as _validate_production_trace_capacity,
    validate_production_trace_replay_binding as _validate_production_trace_replay_binding,
    validate_production_trace_scheduler_binding as _validate_production_trace_scheduler_binding,
    validate_production_trace_scheduler_runtime as _validate_production_trace_scheduler_runtime,
    validate_production_trace_v2_output_schema as _validate_production_trace_v2_output_schema,
    validate_tp1_kv_offload_invariants as _validate_tp1_kv_offload_invariants,
    validate_tp2_kv_offload_invariants as _validate_tp2_kv_offload_invariants,
)

from evaluation.serving.experiments.production_trace_disaggregated_validation import (
    artifact_af_payload_names_by_transfer_id as _artifact_af_payload_names_by_transfer_id,
    validate_af_canonical_rematerialization as _validate_af_canonical_rematerialization,
    validate_af_production_trace_point as _validate_af_production_trace_point,
    validate_pd_af_canonical_rematerialization as _validate_pd_af_canonical_rematerialization,
    validate_pd_af_production_trace_point as _validate_pd_af_production_trace_point,
    validate_pd_canonical_rematerialization as _validate_pd_canonical_rematerialization,
    validate_pd_capacity_and_routes as _validate_pd_capacity_and_routes,
    validate_pd_deployment_plan as _validate_pd_deployment_plan,
    validate_pd_production_trace_point as _validate_pd_production_trace_point,
)


def execute_experiment_stages(
    command_manifest: dict[str, Any],
    *,
    mode: str,
    subprocess_run,
    subprocess_popen,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    """Execute the explicit stages of one already materialized plan."""
    if mode == "dry_run":
        command_manifest["status"] = "dry_run"
        _write_cell_manifest(command_manifest)
        return command_manifest
    elif mode == "mock_run":
        return run_mock_stages(command_manifest, subprocess_run)
    elif mode == "real_run":
        return run_real_stages(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    else:
        raise ManifestValidationError(f"unsupported experiment mode: {mode}")


def run_mock_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
) -> dict[str, Any]:
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type == "production_trace":
            return _run_production_trace_stages(
                command_manifest, subprocess_run, mode="mock_run")
        raise ManifestValidationError(
            f"unknown experiment type: {experiment_type!r}")
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    _write_cell_manifest(command_manifest)
    for stage in _mock_stages(command_manifest):
        command = command_manifest["commands"][stage]
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
        _write_text(
            logs_dir / f"{stage}.stdout.log",
            _required_process_stream(result, "stdout"))
        _write_text(
            logs_dir / f"{stage}.stderr.log",
            _required_process_stream(result, "stderr"))
        if result.returncode != 0:
            status = _failure_status(stage, result, logs_dir, command)
            command_manifest["status"] = status["status"]
            command_manifest["failed_stage"] = stage
            _write_json(output_root / "status.json", status)
            _write_cell_manifest(command_manifest)
            return command_manifest

    _write_placeholder_outputs(command_manifest)
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "ground_truth": command_manifest["outputs"]["ground_truth"],
        "prediction": command_manifest["outputs"]["prediction"],
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def run_real_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type == "production_trace":
            return _run_production_trace_stages(
                command_manifest, subprocess_run, mode="real_run")
        raise ManifestValidationError(
            f"unknown experiment type: {experiment_type!r}")
    if readiness_probe is None:
        readiness_probe = _http_readiness_probe
    if command_manifest["deployment"] == "disaggregated":
        return _run_real_disaggregated_cell(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    elif command_manifest["deployment"] == "colocated":
        return _run_real_colocated_cell(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    else:
        raise ManifestValidationError(
            "unknown experiment deployment: "
            f"{command_manifest['deployment']!r}")


def _run_production_trace_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"mock_run", "real_run"}:
        raise ManifestValidationError(
            f"unsupported Production Trace execution mode: {mode!r}")
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": mode,
        "process_stages": [],
        "device_access": "forbidden",
    }
    _write_cell_manifest(command_manifest)
    for stage in (
        "production_trace_simulation",
        "production_trace_report",
    ):
        status = _run_production_trace_stage(
            command_manifest, stage, logs_dir, subprocess_run)
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        command_manifest["runtime"]["process_stages"].append(stage)
        _write_cell_manifest(command_manifest)
    try:
        validate_production_trace_outputs(command_manifest)
    except (OSError, TypeError, ValueError) as error:
        result = _SyntheticResult(
            returncode=1,
            stdout="",
            stderr=f"invalid Production Trace outputs: {error}",
        )
        status = _failure_status(
            "production_trace_report",
            result,
            logs_dir,
            command_manifest["commands"]["production_trace_report"],
        )
        return _finish_failed_cell(command_manifest, status)
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "sweep": command_manifest["outputs"]["sweep"],
        "report_markdown": command_manifest["outputs"]["report_markdown"],
        "reproducer_commands": _production_trace_reproducer_commands(
            command_manifest),
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def _run_production_trace_stage(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_run,
) -> dict[str, Any] | None:
    allowed = {
        "production_trace_simulation",
        "production_trace_report",
    }
    if stage not in allowed:
        raise ManifestValidationError(
            f"unsupported Production Trace stage: {stage!r}")
    command = command_manifest["commands"][stage]
    if stage == "production_trace_simulation":
        required_path = Path(command_manifest["outputs"]["sweep"])
    elif stage == "production_trace_report":
        required_path = Path(command_manifest["outputs"]["report_markdown"])
    else:
        raise AssertionError("Production Trace stage allowlist is exhaustive")
    previous_identity = _artifact_write_identity(required_path)
    try:
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        result = _missing_executable_result(error)
    stdout = _required_process_stream(result, "stdout")
    stderr = _required_process_stream(result, "stderr")
    _write_text(logs_dir / f"{stage}.stdout.log", stdout)
    _write_text(logs_dir / f"{stage}.stderr.log", stderr)
    if result.returncode != 0:
        return _failure_status(stage, result, logs_dir, command)
    if not required_path.is_file():
        missing = _SyntheticResult(
            returncode=1,
            stdout=stdout,
            stderr=(
                f"Production Trace stage {stage!r} did not write required "
                f"artifact: {required_path}"),
        )
        return _failure_status(stage, missing, logs_dir, command)
    current_identity = _artifact_write_identity(required_path)
    if current_identity == previous_identity:
        stale = _SyntheticResult(
            returncode=1,
            stdout=stdout,
            stderr=(
                f"Production Trace stage {stage!r} did not replace its "
                f"pre-existing artifact: {required_path}"),
        )
        return _failure_status(stage, stale, logs_dir, command)
    return None


def _artifact_write_identity(path: Path) -> tuple[int, int, int, str] | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    stat = path.stat()
    return (
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _run_real_colocated_cell(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": "real_run",
        "servers": {},
        "readiness": {},
    }
    _write_cell_manifest(command_manifest)

    server_record = None
    try:
        server_record, status = _start_server_process(
            command_manifest,
            "vllm_serve",
            logs_dir,
            subprocess_popen,
        )
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        _write_cell_manifest(command_manifest)
        status = _wait_for_server_readiness(
            command_manifest,
            server_record,
            "vllm_serve",
            logs_dir,
            timeout_seconds=server_wait_seconds,
            poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
        if status is not None:
            return _finish_failed_cell(command_manifest, status)

        for stage in (
            "vllm_benchmark",
            "colocated_midlayer_simulation",
            "comparison",
        ):
            status = _run_real_stage(command_manifest, stage, logs_dir, subprocess_run)
            if status is not None:
                return _finish_failed_cell(command_manifest, status)
        return _finish_successful_real_cell(command_manifest, logs_dir)
    finally:
        if server_record is not None:
            _terminate_server_process(server_record)


def _run_real_disaggregated_cell(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": "real_run",
        "servers": {},
        "readiness": {},
    }
    _write_cell_manifest(command_manifest)
    server_records = []
    try:
        for stage in ("pd_decode_serve", "pd_prefill_serve", "pd_proxy_serve"):
            server_record, status = _start_server_process(
                command_manifest,
                stage,
                logs_dir,
                subprocess_popen,
            )
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)
            server_records.append(server_record)
            _write_cell_manifest(command_manifest)

        for server_record in server_records:
            status = _wait_for_server_readiness(
                command_manifest,
                server_record,
                server_record["stage"],
                logs_dir,
                timeout_seconds=server_wait_seconds,
                poll_interval_seconds=readiness_poll_interval_seconds,
                sleep=sleep,
                readiness_probe=readiness_probe,
            )
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)
        for server_record in server_records:
            status = _server_exit_status(server_record, logs_dir)
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)

        status = _run_real_stage(command_manifest, "vllm_benchmark", logs_dir, subprocess_run)
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        _terminate_server_records(server_records)

        for stage in (
            "disaggregated_midlayer_simulation",
            "pd_comparison",
        ):
            status = _run_real_stage(command_manifest, stage, logs_dir, subprocess_run)
            if status is not None:
                return _finish_failed_cell(command_manifest, status)
        return _finish_successful_real_cell(command_manifest, logs_dir)
    finally:
        _terminate_server_records(server_records)


def _start_server_process(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_popen,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    command = command_manifest["commands"][stage]
    stdout_path = logs_dir / f"{stage}.stdout.log"
    stderr_path = logs_dir / f"{stage}.stderr.log"
    stdout_handle = stdout_path.open("w")
    stderr_handle = stderr_path.open("w")
    try:
        try:
            process = subprocess_popen(
                command["argv"],
                cwd=str(ROOT),
                env=_stage_environment(command_manifest, command),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        except FileNotFoundError as exc:
            stdout_handle.close()
            stderr_handle.close()
            status = _failure_status(
                stage,
                _missing_executable_result(exc),
                logs_dir,
                command,
            )
            return None, status
        runtime_servers = command_manifest["runtime"]["servers"]
        runtime_servers[stage] = {
            "pid": process.pid,
            "endpoint": command["endpoint"],
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        return {
            "stage": stage,
            "process": process,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "command": command,
        }, None
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise


def _wait_for_server_readiness(
    command_manifest: dict[str, Any],
    server_record: dict[str, Any],
    stage: str,
    logs_dir: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any] | None:
    if timeout_seconds < 0:
        raise ValueError("server readiness timeout cannot be negative")
    if poll_interval_seconds <= 0:
        raise ValueError("server readiness poll interval must be positive")
    command = server_record["command"]
    endpoint = command["endpoint"]
    health_url = endpoint["health_url"]
    readiness_stage = f"{stage}_readiness"
    elapsed = 0.0
    attempts = 0
    readiness = command_manifest["runtime"]["readiness"]

    while True:
        attempts += 1
        returncode = server_record["process"].poll()
        if returncode is not None:
            _flush_server_logs(server_record)
            status = _failure_status(
                stage,
                _process_result(
                    returncode,
                    server_record["stdout_path"],
                    server_record["stderr_path"],
                ),
                logs_dir,
                command,
            )
            readiness[stage] = {
                "status": "process_exited",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            return status

        if readiness_probe(health_url):
            readiness[stage] = {
                "status": "ready",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            _write_cell_manifest(command_manifest)
            return None

        if elapsed >= timeout_seconds:
            stderr = f"readiness timeout for {health_url}"
            _write_text(logs_dir / f"{readiness_stage}.stdout.log", "")
            _write_text(logs_dir / f"{readiness_stage}.stderr.log", stderr)
            readiness[stage] = {
                "status": "timeout",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            status = _failure_status(
                readiness_stage,
                _SyntheticResult(returncode=1, stdout="", stderr=stderr),
                logs_dir,
                command,
            )
            return status

        sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def _flush_server_logs(server_record: dict[str, Any]) -> None:
    server_record["stdout_handle"].flush()
    server_record["stderr_handle"].flush()


def _server_exit_status(
    server_record: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any] | None:
    returncode = server_record["process"].poll()
    if returncode is None:
        return None
    _flush_server_logs(server_record)
    return _failure_status(
        server_record["stage"],
        _process_result(
            returncode,
            server_record["stdout_path"],
            server_record["stderr_path"],
        ),
        logs_dir,
        server_record["command"],
    )


def _terminate_server_process(server_record: dict[str, Any]) -> None:
    process = server_record["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    server_record["stdout_handle"].close()
    server_record["stderr_handle"].close()


def _terminate_server_records(server_records: list[dict[str, Any]]) -> None:
    for server_record in reversed(server_records):
        _terminate_server_process(server_record)
    server_records.clear()


def _http_readiness_probe(health_url: str) -> bool:
    try:
        with urllib_request.urlopen(health_url, timeout=2.0) as response:
            return 200 <= response.status < 300
    except (OSError, urllib_error.URLError):
        return False


def _run_real_stage(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_run,
) -> dict[str, Any] | None:
    allowed_stages = {
        "vllm_benchmark",
        "colocated_midlayer_simulation",
        "disaggregated_midlayer_simulation",
        "comparison",
        "pd_comparison",
    }
    if stage not in allowed_stages:
        raise ManifestValidationError(
            f"unsupported real execution stage: {stage!r}")
    command = command_manifest["commands"][stage]
    try:
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        result = _missing_executable_result(exc)
    stdout = _required_process_stream(result, "stdout")
    stderr = _required_process_stream(result, "stderr")
    _write_text(logs_dir / f"{stage}.stdout.log", stdout)
    _write_text(logs_dir / f"{stage}.stderr.log", stderr)
    if result.returncode != 0:
        return _failure_status(stage, result, logs_dir, command)
    if stage == "vllm_benchmark":
        ground_truth_status = _ensure_ground_truth_output(
            command_manifest,
            stdout,
            logs_dir,
            command,
        )
        if ground_truth_status is not None:
            return ground_truth_status
        return None
    elif stage in {"comparison", "pd_comparison"}:
        return _ensure_real_success_outputs(command_manifest, stage, logs_dir, command)
    elif stage in {
        "colocated_midlayer_simulation",
        "disaggregated_midlayer_simulation",
    }:
        return None
    else:
        raise ManifestValidationError(
            f"unhandled real execution stage: {stage!r}")


def _ensure_ground_truth_output(
    command_manifest: dict[str, Any],
    stdout: str,
    logs_dir: Path,
    command: dict[str, Any],
) -> dict[str, Any] | None:
    output_path = Path(command_manifest["outputs"]["ground_truth"])
    if output_path.exists():
        return None
    text = stdout.strip()
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            _write_json(output_path, payload)
            return None
    result = _SyntheticResult(
        returncode=1,
        stdout=stdout,
        stderr=(
            "vLLM benchmark completed but did not write ground_truth.json "
            "or emit JSON stdout"
        ),
    )
    return _failure_status("vllm_benchmark", result, logs_dir, command)


def _ensure_real_success_outputs(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    command: dict[str, Any],
) -> dict[str, Any] | None:
    required = [
        command_manifest["outputs"]["ground_truth"],
        command_manifest["outputs"]["prediction"],
        command_manifest["outputs"]["comparison_report"],
        command_manifest["outputs"]["comparison_markdown"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if not missing:
        return None
    result = _SyntheticResult(
        returncode=1,
        stdout="",
        stderr="missing real-run output artifacts: " + ", ".join(missing),
    )
    return _failure_status(stage, result, logs_dir, command)


def _finish_successful_real_cell(
    command_manifest: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "ground_truth": command_manifest["outputs"]["ground_truth"],
        "prediction": command_manifest["outputs"]["prediction"],
        "comparison_report": command_manifest["outputs"]["comparison_report"],
        "comparison_markdown": command_manifest["outputs"]["comparison_markdown"],
        "reproducer_commands": _reproducer_commands(command_manifest),
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def _finish_failed_cell(
    command_manifest: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    status["cell_id"] = command_manifest["cell_id"]
    status["logs"] = command_manifest["outputs"]["logs"]
    command_manifest["status"] = status["status"]
    command_manifest["failed_stage"] = status["failed_stage"]
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def validate_production_trace_outputs(
    command_manifest: dict[str, Any],
) -> None:
    if command_manifest["experiment_type"] != "production_trace":
        raise ManifestValidationError(
            "Production Trace output validation requires "
            "experiment_type='production_trace'")
    sweep_path = Path(command_manifest["outputs"]["sweep"])
    report_path = Path(command_manifest["outputs"]["report_markdown"])
    if not sweep_path.is_file():
        raise ValueError(f"Production Trace sweep output is missing: {sweep_path}")
    if not report_path.is_file():
        raise ValueError(
            f"Production Trace report output is missing: {report_path}")
    try:
        payload = json.loads(
            sweep_path.read_bytes(),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Production Trace sweep JSON: {sweep_path}") from error
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError("Production Trace sweep must be a versioned object")
    if payload["schema_version"] == 2:
        _validate_production_trace_v2_output_schema(payload)
        identity_validator = _validate_production_trace_output_identity
    elif payload["schema_version"] == 3:
        identity_validator = (
            _validate_production_trace_aggregate_output_identity)
    else:
        raise ValueError("unsupported Production Trace sweep schema_version")
    expected_report = render_production_trace_sweep_report(payload)
    plot_dir = report_path.with_name(f"{report_path.stem}_plots")
    plot_paths = {
        "throughput": plot_dir / "production_trace_throughput.png",
        "ttft_p0_p100": plot_dir / "production_trace_ttft_p0_p100.png",
        "tpot_p0_p100": plot_dir / "production_trace_tpot_p0_p100.png",
        "simulation_speed": plot_dir / "production_trace_simulation_speed.png",
    }
    expected_plot_report = expected_report + "## Plots\n\n"
    for name, path in sorted(plot_paths.items()):
        relative_path = os.path.relpath(path, report_path.parent)
        expected_plot_report += f"- {name}: ![{name}]({relative_path})\n"
    expected_plot_report += "\n"
    actual_report = report_path.read_text(encoding="utf-8")
    if actual_report not in (expected_report, expected_plot_report):
        raise ValueError(
            "Production Trace report bytes disagree with canonical renderer")
    identity_validator(command_manifest, payload)


def _validate_production_trace_output_identity(
    command_manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    expected_input = command_manifest["input"]
    if payload["source_trace"] != expected_input["trace"]:
        raise ValueError(
            "Production Trace sweep source identity disagrees with command manifest")
    if tuple(payload["arrival_rate_scales"]) != tuple(
            expected_input["arrival_rate_scales"]):
        raise ValueError(
            "Production Trace sweep scales disagree with command manifest")
    expected_model_sha256 = command_manifest["model"]["config_sha256"]
    try:
        expected_model = load_model_config(command_manifest["model"]["path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "Production Trace command manifest model cannot be loaded") \
            from error
    if expected_model.config_sha256 != expected_model_sha256:
        raise ValueError(
            "Production Trace command manifest model content changed")
    expected_simulation = command_manifest["simulation"]
    expected_hardware = expected_simulation["hardware"]
    expected_timing_status = _expected_performance_timing_status(
        expected_simulation)
    if command_manifest["validation_status"]["performance_timing"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace command manifest performance timing status "
            "disagrees with launch policy and hardware")
    provenance_hardware = payload["execution_provenance"]["hardware_configs"]
    repository = payload["execution_provenance"]["repository"]
    if repository["git_commit"] != command_manifest["provenance"]["git_commit"]:
        raise ValueError(
            "Production Trace sweep code commit disagrees with command manifest")
    if repository["git_diff_hash"] != command_manifest["provenance"][
            "git_diff_hash"]:
        raise ValueError(
            "Production Trace sweep code diff identity disagrees with "
            "command manifest")
    hardware_path = Path(expected_hardware["config_path"]).resolve()
    hardware_payload = hardware_path.read_bytes()
    expected_hardware_provenance = {
        "path": str(hardware_path),
        "sha256": expected_hardware["config_sha256"],
        "size_bytes": len(hardware_payload),
    }
    if hashlib.sha256(hardware_payload).hexdigest() != (
            expected_hardware["config_sha256"]):
        raise ValueError(
            "Production Trace command manifest hardware content changed")
    if provenance_hardware["prefill"] != expected_hardware_provenance:
        raise ValueError(
            "Production Trace sweep hardware identity disagrees with command manifest")
    if expected_simulation["deployment"] in ("colocated", "af_disagg"):
        if provenance_hardware["decode"] is not None:
            raise ValueError(
                "non-PD Production Trace sweep unexpectedly used decode "
                "hardware")
    elif expected_simulation["deployment"] in (
            "pd_disagg", "pd_af_disagg"):
        if provenance_hardware["decode"] != expected_hardware_provenance:
            raise ValueError(
                "PD Production Trace sweep decode hardware identity "
                "disagrees with command manifest")
    else:
        raise ValueError(
            "Production Trace command manifest has an unsupported deployment")
    if tuple(provenance_hardware["additional_colocated"]):
        raise ValueError(
            "single-colocated Production Trace sweep unexpectedly used "
            "additional hardware")
    if tuple(payload["execution_provenance"]["loaded_hardware_ids"]) != (
            expected_hardware["hardware_id"],):
        raise ValueError(
            "Production Trace sweep loaded hardware IDs disagree with "
            "command manifest")
    if payload["execution_provenance"]["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace sweep performance timing provenance disagrees "
            "with command manifest")
    expected_replays = _production_trace_replays_from_command_input(
        expected_input)
    points = payload["points"]
    if (not isinstance(points, list)
            or len(points) != len(expected_input["arrival_rate_scales"])
            or payload["point_count"] != len(points)):
        raise ValueError(
            "Production Trace sweep point count disagrees with command manifest")
    expected_slo = command_manifest["analysis"]["slo_contract"]
    if expected_slo is None:
        expected_slo_contract = None
    else:
        try:
            expected_slo_contract = load_slo_contract(expected_slo["path"])
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "Production Trace command manifest SLO contract cannot be "
                "loaded") from error
        if expected_slo_contract.source_sha256 != expected_slo[
                "source_sha256"]:
            raise ValueError(
                "Production Trace command manifest SLO contract changed")
    expected_tgs = command_manifest["analysis"][
        "interactivity_tgs_contract"]
    if expected_tgs is None:
        expected_tgs_contract = None
    else:
        try:
            expected_tgs_contract = load_interactivity_tgs_contract(
                expected_tgs["path"])
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "Production Trace command manifest Interactivity--TGS "
                "contract cannot be loaded") from error
        if expected_tgs_contract.source_sha256 != expected_tgs[
                "source_sha256"]:
            raise ValueError(
                "Production Trace command manifest Interactivity--TGS "
                "contract changed")
    for point, expected_scale, expected_replay in zip(
            points,
            expected_input["arrival_rate_scales"],
            expected_replays,
            strict=True):
        if point["arrival_rate_scale"] != expected_scale:
            raise ValueError(
                "Production Trace sweep point scale disagrees with "
                "command manifest")
        expected_point_status = {
            "trace_replay": "validated_conservation",
            "scheduler_control_plane": "native_pinned_vllm",
            "performance_timing": expected_timing_status,
        }
        if point["validation_status"] != expected_point_status:
            raise ValueError(
                "Production Trace sweep point validation status disagrees "
                "with command manifest")
        _validate_production_trace_replay_binding(
            expected_input["trace"],
            expected_replay,
            point,
        )
        result = point["result"]
        _validate_production_trace_point_simulation_identity(
            expected_simulation,
            expected_model,
            command_manifest["model"]["path"],
            payload["execution_provenance"],
            result,
            expected_replay.requests,
            expected_slo_contract,
        )
        if result["model_config_provenance"]["config_sha256"] != (
                expected_model_sha256):
            raise ValueError(
                "Production Trace sweep model identity disagrees with "
                "command manifest")
        actual_slo = result["metrics"]["slo_goodput"]["contract"]
        if expected_slo is None:
            if actual_slo is not None:
                raise ValueError(
                    "Production Trace sweep unexpectedly used an SLO contract")
        elif actual_slo != _json_round_trip(asdict(expected_slo_contract)):
            raise ValueError(
                "Production Trace sweep SLO identity disagrees with "
                "command manifest")
    if expected_tgs_contract is None:
        expected_analysis = arrival_rate_sweep_analysis_manifest(None, None)
    else:
        expected_analysis_points = tuple(
            sweep_metric_point_from_manifest(
                point["arrival_rate_scale"],
                point["result"]["metrics"],
                expected_tgs_contract,
            )
            for point in points
        )
        expected_analysis = arrival_rate_sweep_analysis_manifest(
            expected_analysis_points,
            expected_tgs_contract,
        )
    expected_analysis_json = _json_round_trip(expected_analysis)
    for analysis_field in (
            "interactivity_tgs", "peak_selection", "saturation_analysis"):
        if payload[analysis_field] != expected_analysis_json[analysis_field]:
            raise ValueError(
                "Production Trace sweep Interactivity--TGS analysis "
                "disagrees with the command manifest contract and canonical "
                "point metrics")


def _validate_production_trace_aggregate_output_identity(
    command_manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """Bind aggregate-only persisted output to command inputs."""

    expected_input = command_manifest["input"]
    if payload["source_trace"] != expected_input["trace"]:
        raise ValueError(
            "aggregate Production Trace source identity disagrees with the "
            "command manifest")
    expected_scales = tuple(expected_input["arrival_rate_scales"])
    if tuple(payload["arrival_rate_scales"]) != expected_scales:
        raise ValueError(
            "aggregate Production Trace scales disagree with the command "
            "manifest")
    expected_replays = _production_trace_replays_from_command_input(
        expected_input)
    points = payload["points"]
    if (not isinstance(points, list)
            or len(points) != len(expected_replays)
            or payload["point_count"] != len(points)):
        raise ValueError(
            "aggregate Production Trace point count disagrees with the "
            "command manifest")
    expected_simulation = command_manifest["simulation"]
    expected_mtp = expected_simulation["mtp"]
    expected_model_sha256 = command_manifest["model"]["config_sha256"]
    try:
        expected_model = load_model_config(command_manifest["model"]["path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "aggregate Production Trace command model cannot be loaded") \
            from error
    if expected_model.config_sha256 != expected_model_sha256:
        raise ValueError(
            "aggregate Production Trace command model content changed")
    expected_timing = _expected_performance_timing_status(
        expected_simulation)
    repository = payload["execution_provenance"]["repository"]
    if repository["git_commit"] != command_manifest["provenance"][
            "git_commit"]:
        raise ValueError(
            "aggregate Production Trace code commit identity disagrees with "
            "the command manifest")
    if repository["git_diff_hash"] != command_manifest["provenance"][
            "git_diff_hash"]:
        raise ValueError(
            "aggregate Production Trace code diff identity disagrees with the "
            "command manifest")
    for point, expected_scale, replay in zip(
            points, expected_scales, expected_replays, strict=True):
        expected_replay = production_trace_replay_manifest(replay)
        result = point["result"]
        request_source = result["request_source"]
        if (point["arrival_rate_scale"] != expected_scale
                or point["request_stream_sha256"]
                != replay.request_stream_sha256
                or request_source["trace"] != expected_input["trace"]
                or request_source["replay"] != expected_replay):
            raise ValueError(
                "aggregate Production Trace replay identity disagrees with "
                "the command manifest")
        if (result["model_config_provenance"]["config_sha256"]
                != expected_model_sha256
                or result["deployment"] != expected_simulation["deployment"]
                or result["mtp_scenario"] != expected_mtp
                or result["performance_timing_status"] != expected_timing):
            raise ValueError(
                "aggregate Production Trace simulation identity disagrees "
                "with the command manifest")
        binding = result["workload_assumption_binding"]
        if binding["request_stream_sha256"] != replay.request_stream_sha256:
            raise ValueError(
                "aggregate Production Trace workload binding disagrees with "
                "the replay stream")
        _validate_production_trace_aggregate_scheduler_runtime(
            result["scheduler_runtime"],
            expected_simulation,
            expected_model,
        )
        prefix = request_source["prefix_alignment"]["aggregate"]
        statistics = expected_input["trace"]["statistics"]
        if (prefix["request_count"] != statistics["request_count"]
                or prefix["trace_prefix_hit_tokens"]
                != statistics["prefix_hit_tokens_total"]):
            raise ValueError(
                "aggregate Production Trace prefix totals disagree with the "
                "source Trace")
        artifact = result["execution_artifact"]
        if (artifact["counts"]["simulation_steps"] != result["num_steps"]
                or artifact["counts"]["kernel_calls"]
                != result["num_kernel_calls"]
                or artifact["counts"]["trace_transfers"]
                != result["num_transfers"]):
            raise ValueError(
                "aggregate Production Trace execution counts are inconsistent")
    expected_payload = _json_round_trip(
        _rerun_production_trace_aggregate(command_manifest))
    recorded_runtime = _production_trace_runtime_measurements(payload)
    replay_runtime = _production_trace_runtime_measurements(expected_payload)
    if (_production_trace_deterministic_projection(payload)
            != _production_trace_deterministic_projection(expected_payload)
            or (recorded_runtime == replay_runtime
                and payload != expected_payload)):
        raise ValueError(
            "aggregate Production Trace output disagrees with deterministic "
            "replay")


def _production_trace_runtime_measurements(
    payload: dict[str, Any],
) -> tuple[object, ...]:
    """Return run-local measurements excluded from deterministic replay."""

    measurements = [tuple(payload[field] if field in payload else None
        for field in ("interactive_report", "report_output_costs"))]
    for point in payload["points"]:
        result = point["result"]
        parallelism = result["execution_parallelism"]
        measurements.append((
            result["simulation_wall_time_s"],
            result["evaluation_wall_time_breakdown_s"],
            result["process_peak_rss_bytes"],
            parallelism["parallel_execution_wall_time_s"],
            parallelism["worker_aggregate_phase_s"],
            point["state_audit_output"] if "state_audit_output" in point else None,
            tuple(result[field] if field in result else None for field in (
                "interactive_report", "report_output_costs", "simulation_started_monotonic_s", "simulation_finished_monotonic_s")),
        ))
    return tuple(measurements)


def _production_trace_deterministic_projection(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove wall-clock and RSS observations before replay comparison."""

    projected = _json_round_trip(payload)
    for field in ("interactive_report", "report_output_costs"):
        if field in projected:
            projected[field] = None
    for point in projected["points"]:
        if "state_audit_output" in point:
            point["state_audit_output"] = None
        result = point["result"]
        for field in ("interactive_report", "report_output_costs", "simulation_started_monotonic_s", "simulation_finished_monotonic_s"):
            if field in result:
                result[field] = None
        result["persistence"]["full_in_memory_result_sha256"] = None
        result["simulation_wall_time_s"] = None
        result["evaluation_wall_time_breakdown_s"] = None
        result["process_peak_rss_bytes"] = None
        parallelism = result["execution_parallelism"]
        parallelism["parallel_execution_wall_time_s"] = None
        parallelism["worker_aggregate_phase_s"] = None
    return projected


def _rerun_production_trace_aggregate(
    command_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a request-safe aggregate under the recorded environment."""

    simulation_command = command_manifest["commands"][
        "production_trace_simulation"]
    simulation_argv = simulation_command["argv"]
    if (not isinstance(simulation_argv, list)
            or len(simulation_argv) < 3
            or any(not isinstance(value, str) or not value
                   for value in simulation_argv)
            or Path(simulation_argv[1]).resolve()
            != (ROOT / "scripts" / "run_serving_simulation.py").resolve()):
        raise ValueError(
            "aggregate Production Trace simulation command is not canonical")
    output_indices = tuple(
        index for index, value in enumerate(simulation_argv)
        if value == "--output" or value.startswith("--output=")
    )
    if len(output_indices) != 1:
        raise ValueError(
            "aggregate Production Trace simulation command must contain one "
            "--output")
    output_index = output_indices[0]
    output_uses_equals_form = simulation_argv[output_index].startswith("--output=")
    if not output_uses_equals_form and output_index + 1 >= len(simulation_argv):
        raise ValueError(
            "aggregate Production Trace simulation --output has no path")
    with tempfile.TemporaryDirectory(
            prefix="midlayer-production-trace-validation-") as directory:
        validation_output = Path(directory) / "aggregate.json"
        validation_argv = list(simulation_argv)
        if output_uses_equals_form:
            validation_argv[output_index] = "--output=" + str(validation_output)
        else:
            validation_argv[output_index + 1] = str(validation_output)
        for option, filename in (("--interactive-output", "report.html"), ("--state-audit-output", "state-audit.json")):
            indices = [index for index, value in enumerate(validation_argv) if value == option or value.startswith(option + "=")]
            if len(indices) > 1:
                raise ValueError("duplicate replay output option: " + option)
            if indices:
                index = indices[0]
                target = str(Path(directory) / filename)
                if validation_argv[index] == option:
                    if index + 1 >= len(validation_argv) or validation_argv[index + 1].startswith("--"):
                        raise ValueError("missing replay output path: " + option)
                    validation_argv[index + 1] = target
                else:
                    validation_argv[index] = option + "=" + target
        if "--interactive-report" in validation_argv and "--interactive-output" not in validation_argv:
            # A boolean flag with the default report path also writes beside
            # the redirected result; keep the replay hermetic by pinning the
            # output path explicitly.
            validation_argv.extend(("--interactive-output", str(Path(directory) / "report.html")))
        completed = subprocess.run(
            validation_argv,
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, simulation_command),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "aggregate Production Trace deterministic replay failed")
        if not validation_output.is_file():
            raise ValueError(
                "aggregate Production Trace deterministic replay did not "
                "write its aggregate")
        try:
            payload = json.loads(
                validation_output.read_bytes(),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "aggregate Production Trace deterministic replay is invalid "
                "JSON") from error
    if (not isinstance(payload, dict)
            or payload["schema_version"] != 3
            or payload["kind"] != "production_trace_arrival_rate_sweep"):
        raise ValueError(
            "aggregate Production Trace deterministic replay is not schema v3")
    return payload


def _validate_production_trace_aggregate_scheduler_runtime(
    scheduler_runtime: dict[str, Any],
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
) -> None:
    """Validate retained Scheduler identities without request-level replay."""

    deployment = expected_simulation["deployment"]
    scheduler = expected_simulation["scheduler"]
    mtp = expected_simulation["mtp"]
    if deployment == "colocated":
        expected_pools = (("main", "colocated", "prefill_tp_size"),)
    elif deployment == "af_disagg":
        expected_pools = (("attention", "attention", "prefill_tp_size"),)
    elif deployment == "pd_disagg":
        pd_disagg = expected_simulation["pd_disagg"]
        expected_pools = (
            *tuple(
                (f"prefill-{index}", "prefill", "prefill_tp_size")
                for index in range(pd_disagg["num_prefill_instances"])
            ),
            *tuple(
                (f"decode-{index}", "decode", "decode_tp_size")
                for index in range(pd_disagg["num_decode_instances"])
            ),
        )
    elif deployment == "pd_af_disagg":
        pd_disagg = expected_simulation["pd_disagg"]
        if pd_disagg["num_prefill_instances"] == 1:
            prefill_names = ("prefill-attention",)
        elif pd_disagg["num_prefill_instances"] == 2:
            prefill_names = ("prefill-attention-0", "prefill-attention-1")
        else:
            raise ValueError(
                "aggregate Production Trace has an unsupported prefill "
                "instance count")
        if pd_disagg["num_decode_instances"] == 1:
            decode_names = ("decode-attention",)
        elif pd_disagg["num_decode_instances"] == 2:
            decode_names = ("decode-attention-0", "decode-attention-1")
        else:
            raise ValueError(
                "aggregate Production Trace has an unsupported decode "
                "instance count")
        expected_pools = (
            *tuple(
                (name, "prefill-attention", "prefill_tp_size")
                for name in prefill_names
            ),
            *tuple(
                (name, "decode-attention", "decode_tp_size")
                for name in decode_names
            ),
        )
    else:
        raise ValueError(
            "aggregate Production Trace has an unsupported deployment")
    if set(scheduler_runtime) != {
            pool_name for pool_name, _role, _tp_field in expected_pools}:
        raise ValueError(
            "aggregate Production Trace Scheduler pools disagree with the "
            "command manifest")
    for pool_name, role, tensor_parallel_field in expected_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            scheduler,
            expected_model,
            mtp,
            role=role,
            tensor_parallel_field=tensor_parallel_field,
        )


def _validate_production_trace_point_simulation_identity(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    shared_execution_provenance: dict[str, Any],
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    for field in ("backend", "deployment", "prediction_method", "launch_policy"):
        if result[field] != expected_simulation[field]:
            raise ValueError(
                f"Production Trace sweep {field} disagrees with command manifest")
    expected_timing_status = _expected_performance_timing_status(
        expected_simulation)
    if result["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            "Production Trace sweep performance timing status disagrees with "
            "command manifest")
    if result["execution_artifact"]["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace execution artifact performance timing status "
            "disagrees with command manifest")
    if result["mtp_scenario"] != expected_simulation["mtp"]:
        raise ValueError(
            "Production Trace sweep MTP scenario disagrees with command "
            "manifest")
    expert_routing = expected_simulation["expert_routing"]
    expected_execution_mode = (
        "reference"
        if (expert_routing is not None
            and expert_routing["source_kind"] == "replay_observation")
        else "accelerated"
    )
    if result["execution_mode"] != expected_execution_mode:
        raise ValueError(
            "Production Trace execution mode disagrees with command manifest")
    cache_stats = _require_exact_output_object(
        result["execution_cache_stats"],
        {
            "execution_mode", "kernel_lowering", "kernel_prediction",
        },
        context="Production Trace execution cache stats",
    )
    if cache_stats["execution_mode"] != expected_execution_mode:
        raise ValueError(
            "Production Trace execution cache mode disagrees with result")
    for cache_name in ("kernel_lowering", "kernel_prediction"):
        cache = cache_stats[cache_name]
        if expected_execution_mode == "reference" and cache is not None:
            raise ValueError(
                "reference Production Trace execution cannot carry cache "
                "statistics")
        if expected_execution_mode == "accelerated":
            cache = _require_exact_output_object(
                cache,
                {"entries", "hits", "misses"},
                context=f"Production Trace {cache_name} cache stats",
            )
            if any(
                    isinstance(cache[field], bool)
                    or not isinstance(cache[field], int)
                    or cache[field] < 0
                    for field in ("entries", "hits", "misses")):
                raise ValueError(
                    f"Production Trace {cache_name} cache stats are invalid")
    execution_provenance = result["execution_provenance"]
    if execution_provenance != shared_execution_provenance:
        raise ValueError(
            "Production Trace sweep point execution provenance disagrees with "
            "the shared sweep provenance")
    runtime_inputs = execution_provenance["runtime_inputs"]
    if (expected_simulation["deployment"] in ("af_disagg", "pd_af_disagg")
            and set(runtime_inputs) != {
                "expert_routing", "attention_kv_layout",
                "attention_kv_distribution", "graph_execution",
            }):
        raise ValueError(
            "AF-bearing Production Trace runtime input fields disagree with the "
            "canonical contract")
    for field in ("attention_kv_layout", "attention_kv_distribution"):
        if runtime_inputs[field] != expected_simulation[field]:
            raise ValueError(
                f"Production Trace sweep {field} disagrees with command manifest")
    if (runtime_inputs["graph_execution"]
            != GraphExecutionConfig.disabled().manifest()):
        raise ValueError(
            "Production Trace sweep unexpectedly enabled graph execution")
    if (expected_simulation["deployment"] in ("af_disagg", "pd_af_disagg")
            and runtime_inputs["expert_routing"]
            != _json_round_trip(ExpertRoutingInput.not_applicable().manifest())):
        raise ValueError(
            "AF-bearing Production Trace expert routing runtime input must be "
            "not_applicable")
    if execution_provenance["launch_inputs"]["launch_policy"] != (
            expected_simulation["launch_policy"]):
        raise ValueError(
            "Production Trace sweep launch provenance disagrees with "
            "command manifest")
    if execution_provenance["prediction_method"] != expected_simulation[
            "prediction_method"]:
        raise ValueError(
            "Production Trace sweep prediction provenance disagrees with "
            "command manifest")
    expected_hardware_id = expected_simulation["hardware"]["hardware_id"]
    if tuple(result["hardware_ids"]) != (expected_hardware_id,):
        raise ValueError(
            "Production Trace sweep point hardware IDs disagree with "
            "command manifest")
    if expected_simulation["deployment"] == "colocated":
        _validate_colocated_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "pd_disagg":
        _validate_pd_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "af_disagg":
        _validate_af_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "pd_af_disagg":
        _validate_pd_af_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    else:
        raise ValueError(
            "Production Trace sweep point has an unsupported deployment")


def _failure_status(
    stage: str,
    result,
    logs_dir: Path,
    command: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stderr = _required_process_stream(result, "stderr").strip()
    dependency_category = _dependency_category(stderr)
    status = "blocked" if dependency_category else "failed"
    failure = {
        "status": status,
        "failed_stage": stage,
        "returncode": result.returncode,
        "stderr_summary": stderr[:500],
        "stdout_log": str(logs_dir / f"{stage}.stdout.log"),
        "stderr_log": str(logs_dir / f"{stage}.stderr.log"),
    }
    if command is not None:
        failure["reproducer_commands"] = {
            stage: list(command["argv"]),
        }
    if dependency_category:
        failure["dependency_category"] = dependency_category
        failure["unblock_checklist"] = _unblock_checklist(dependency_category)
    return failure


def _stage_environment(
    command_manifest: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(command_manifest["environment"])
    if "environment" in command:
        env.update(command["environment"])
    return env


def _missing_executable_result(exc: FileNotFoundError) -> "_SyntheticResult":
    if exc.filename is None:
        executable = "executable"
    elif isinstance(exc.filename, str) and exc.filename:
        executable = exc.filename
    else:
        raise TypeError("FileNotFoundError.filename must be a non-empty string or None")
    return _SyntheticResult(
        returncode=127,
        stdout="",
        stderr=f"dependency executable not found: {executable}",
    )


def _process_result(
    returncode: int,
    stdout_log: Path,
    stderr_log: Path,
) -> "_SyntheticResult":
    return _SyntheticResult(
        returncode=returncode,
        stdout=stdout_log.read_text(),
        stderr=stderr_log.read_text(),
    )


def _reproducer_commands(command_manifest: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: list(command["argv"])
        for name, command in command_manifest["commands"].items()
    }


def _production_trace_reproducer_commands(
    command_manifest: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        stage: list(command_manifest["commands"][stage]["argv"])
        for stage in (
            "production_trace_simulation",
            "production_trace_report",
        )
    }


def _pd_reproducer_commands(command_manifest: dict[str, Any]) -> dict[str, list[str]]:
    reproducers = _reproducer_commands(command_manifest)
    return {
        name: reproducers[name]
        for name in ("pd_decode_serve", "pd_prefill_serve", "pd_proxy_serve")
        if name in reproducers
    }


class _SyntheticResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mock_stages(command_manifest: dict[str, Any]) -> tuple[str, ...]:
    if command_manifest["deployment"] == "disaggregated":
        return (
            "vllm_benchmark",
            "disaggregated_midlayer_simulation",
        )
    if command_manifest["deployment"] == "colocated":
        return (
            "vllm_serve",
            "vllm_benchmark",
            "colocated_midlayer_simulation",
        )
    raise ManifestValidationError(
        f"unknown experiment deployment: {command_manifest['deployment']!r}")


def _dependency_category(stderr: str) -> str | None:
    text = stderr.lower()
    tokens = set(re.findall(r"[a-z0-9_+-]+", text))
    if not text:
        return None
    module_failure = (
        "modulenotfounderror" in tokens
        or "importerror" in tokens
        or "no module named" in text
        or "cannot import" in text
        or "missing package" in text
    )
    backend_failure = (
        "deepep" in tokens
        or "all-to-all" in tokens
        or "all2all" in tokens
        or ("backend" in tokens and ("unsupported" in tokens or "version" in tokens))
        or "not supported" in text
    )
    if module_failure:
        return "dependency"
    if backend_failure:
        return "backend"
    if (
        "model path" in text
        or "failed to load model" in text
        or "model not found" in text
        or "does not exist" in text
        or "no such file" in text
    ):
        return "model"
    if (
        "connection refused" in text
        or "failed to establish" in text
        or "readiness timeout" in text
        or "timed out" in text
        or "timeout" in tokens
    ):
        return "runtime"
    for category in ("nccl", "gdrcopy", "connector", "cuda", "dependency"):
        if category in tokens:
            return category
    if "dependencies" in tokens:
        return "dependency"
    if "port" in tokens or "eaddrinuse" in tokens or "address already in use" in text:
        return "port"
    return None


def _unblock_checklist(category: str) -> list[str]:
    category_specific = {
        "backend": "Verify the requested MoE/all-to-all backend is installed and supported by this vLLM build.",
        "model": "Verify the model path exists on this host and can be loaded by vLLM.",
        "runtime": "Verify the target HTTP endpoint becomes reachable before retrying the experiment.",
        "port": "Verify the configured HTTP/KV ports are free before retrying the experiment.",
        "dependency": "Verify the required runtime package is installed and importable.",
        "nccl": "Verify NCCL is installed and visible to the runtime.",
        "gdrcopy": "Verify GDRCopy is installed and visible to the runtime.",
        "connector": "Verify the requested KV connector is installed and configured.",
        "cuda": "Verify the CUDA driver and runtime are installed and compatible.",
    }
    if category not in category_specific:
        raise ValueError(f"unknown dependency category: {category!r}")
    return [
        category_specific[category],
        "Re-run the dry-run command manifest to confirm command arguments.",
        "Inspect the recorded stderr/stdout logs before retrying the experiment.",
    ]


def _write_placeholder_outputs(command_manifest: dict[str, Any]) -> None:
    if command_manifest["deployment"] == "disaggregated":
        _write_disaggregated_placeholder_outputs(command_manifest)
        return
    output_root = Path(command_manifest["output_root"])
    _write_json(output_root / "ground_truth.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
    })
    _write_json(output_root / "prediction.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
    })


def _write_disaggregated_placeholder_outputs(command_manifest: dict[str, Any]) -> None:
    output_root = Path(command_manifest["output_root"])
    _write_json(output_root / "ground_truth.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
        "comparison_role": "ground_truth",
    })
    _write_json(output_root / "prediction.json", {
        "status": "placeholder",
        "source": "mock-run",
        "kind": "disaggregated_prediction",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
        "inputs": {
            "prefill_trace": command_manifest["traces"]["prefill_trace"],
            "decode_trace": command_manifest["traces"]["decode_trace"],
            "prefill_sidecar": command_manifest["sidecars"]["prefill_sidecar"],
            "decode_sidecar": command_manifest["sidecars"]["decode_sidecar"],
        },
        "kv_connector": command_manifest["kv_connector"],
        "kv_transfer": {
            "status": "mock-placeholder",
            "metadata_source": "runner",
        },
        "metrics": [
            {
                "request_id": "mock-placeholder",
                "ttft_disagg_ms": None,
                "tpot_disagg_ms": None,
                "e2e_disagg_ms": None,
                "prefill_time_ms": None,
                "kv_transfer_time_ms": None,
                "decode_time_ms": None,
            }
        ],
    })




def _write_cell_manifest(command_manifest: dict[str, Any]) -> None:
    _write_json(
        Path(command_manifest["command_manifest_path"]), command_manifest)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _required_process_stream(result, stream: str) -> str:
    if stream == "stdout":
        value = result.stdout
    elif stream == "stderr":
        value = result.stderr
    else:
        raise ValueError(f"unknown process stream: {stream!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"captured subprocess {stream} must be text, got "
            f"{type(value).__name__}")
    return value
