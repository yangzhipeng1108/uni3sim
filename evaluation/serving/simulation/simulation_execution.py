"""Simulation execution; extracted without changing the established contracts."""

from __future__ import annotations

from evaluation.serving.simulation.interactive_recording import run_recorded_simulation

from contextlib import ExitStack
from copy import copy
from dataclasses import asdict
from functools import partial
import json
import math
from pathlib import Path
from src import _platform_compat as resource
import sys
import threading
import time
import warnings
from evaluation.serving.metrics.latency_distribution import latency_distribution_manifest
from evaluation.serving.metrics.interactivity_tgs import arrival_rate_sweep_analysis_manifest, load_interactivity_tgs_contract, sweep_metric_point_from_manifest
from evaluation.serving.metrics.slo_goodput import load_slo_contract, slo_goodput_manifest
from evaluation.serving.metrics.scheduler_progress import scheduler_progress_manifest
from evaluation.serving.metrics.system_resource_analysis import critical_path_manifest, system_resource_manifest
from evaluation.serving.simulation.execution_artifact import DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS, execution_artifact_manifest, performance_timing_status, prefix_alignment_manifest, StreamingExecutionArtifactAccumulator, UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS
from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig, mtp_scenario_manifest
from src.serving.backend.vllm_scheduler_process import vllm_scheduler_runtime_sha256
from src.serving.backend.mtp_acceptance_replay import PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT, closed_loop_admission_contract_sha256, pd_decode_admission_contract_sha256
from src.serving.backend.mtp_acceptance_assumption import MTPAcceptanceAssumption
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import DeploymentConfig, RouteConfig, RouterConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import deployment_plan_manifest
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.kv_cache.kv_capacity import kv_capacity_manifest, load_runtime_memory_policy, resolve_analytical_kv_capacity_plan, resolve_debug_kv_capacity_plan, resolve_debug_vllm030_byte_budget_capacity_plan, resolve_glm_block_fp8_kv_capacity_plan
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.kv_cache.model_residency import kimi_k3_source_workspace_bytes
from src.serving.host_timing import load_host_launch_profile, load_host_runtime_prepare_profile, load_host_request_path_profile, load_scheduler_host_profile, load_engine_core_host_round_profile
from src.serving.metrics.serving_metrics import engine_clock_first_schedule_start_s
from src.serving.runtime_prepare import runtime_prepare_events
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.request_gen.production_trace import build_production_trace_arrival_rate_sweep, load_production_trace, production_trace_manifest, production_trace_replay_manifest
from src.serving.request_gen.request_stream import RequestSpec, request_stream_sha256
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import BackendStep, ScheduledRequest, ScheduleStep
from src.serving.simulation import MultiSimulation, Simulation, SimulationConfig, StreamingSimulationResult, validate_pd_af_mtp_scope, validate_standalone_af_ep2_scope, validate_standalone_af_mtp_scope, validate_workload_emission_parallelism
from src.serving.workload.model_config import DEEPSEEK_V3_L32_CONFIG_SHA256, DEEPSEEK_V4_FLASH_CONFIG_SHA256, DEEPSEEK_V4_PRO_CONFIG_SHA256, is_validated_qwen3_tp2_host_kv_model_scope, load_model_config
from src.serving.workload.expert_frequency import load_expert_frequency_profile
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

from evaluation.serving.simulation.simulation_artifacts import (
    build_execution_provenance,
    file_provenance,
    host_timing_manifest,
    launch_data_provenance,
    load_request_source,
    mtp_acceptance_outcomes_manifest,
    mtp_pd_handoff_contract,
    persistence_projection,
    production_trace_aggregate_result,
    workload_assumption_binding_manifest,
)

from evaluation.serving.simulation.simulation_configuration import (
    ROOT,
    ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE,
    VERSIONED_HOST_PROFILE_HARDWARE,
    bind_tensor_collective_profiles,
    host_request_path_profile_paths,
    host_runtime_prepare_profile_path,
    is_dsv3_h200_pd_mtp_target_scope,
    is_glm_h200_pd_mtp_target_scope,
    p2p_resource_id,
    pipeline_p2p_configs,
    production_trace_reader_config,
    protobridge_p2p_profile,
    require_expected_sha256,
    tensor_collective_configs,
    moe_data_collective_configs,
    uses_development_protobridge,
    gqa_pd_cli_scope,
    mla_pcp_pd_cli_scope,
    glm_bf16_cli_source_scope,
)
from src.serving.kv_cache.kv_cache_config import KVCacheConfig
from src.serving.kv_cache.kv_cache_config import KVCacheTierConfig
from src.serving.kv_cache.vllm_allocation import VLLMKVAllocation, allocate_vllm030_kv_cache
from src.serving.kv_cache.vllm_allocation import allocate_vllm_kv_cache
from dataclasses import replace
from evaluation.serving.simulation.simulation_artifacts import repository_provenance
from src.serving.kv_cache.kv_capacity import resolve_symbolic_kv_capacity_plan
from src.serving.kv_cache.kv_capacity import runtime_memory_policy_payload
from evaluation.serving.simulation.simulation_artifacts import source_activation_capacity_manifest
from evaluation.serving.simulation.simulation_configuration import vllm030_cli_scheduler_settings


class ConformanceBackend:
    """Deterministic protocol exerciser; not a serving policy model."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._requests = {}
        self._pending = None
        self._step = 0

    def add_request(self, request: RequestSpec) -> None:
        if request.output_tokens == 0:
            raise NotImplementedError(
                "conformance zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "conformance backend prefix injection is not implemented")
        has_transferred_kv = self.config.is_decode_only
        self._requests[request.request_id] = {
            "request": request,
            "prompt_computed": request.input_tokens if has_transferred_kv else 0,
            "output_produced": 1 if has_transferred_kv else 0,
        }

    def has_work(self) -> bool:
        return bool(self._requests)

    def close(self) -> None:
        return None

    def plan_step(self, virtual_time_s: float):
        if self._pending is not None:
            raise RuntimeError("complete the conformance step before planning another")
        budget = self.config.max_batched_tokens
        scheduled = []
        completion = {}
        finished = []
        for request_id, state in list(self._requests.items())[:self.config.max_requests]:
            if budget <= 0:
                break
            completion[request_id] = 0
            request = state["request"]
            prompt_computed = state["prompt_computed"]
            if prompt_computed < request.input_tokens:
                count = min(request.input_tokens - prompt_computed, budget)
                phase = "prefill"
                input_tokens = count
                output_tokens = 0
                if prompt_computed + count == request.input_tokens:
                    completion[request_id] = 1
            else:
                count = 1
                phase = "decode"
                input_tokens = 0
                output_tokens = 1
                completion[request_id] = 1
            budget -= count
            scheduled.append(ScheduledRequest(
                request_id=request_id,
                phase=phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                computed_tokens=(
                    prompt_computed
                    + max(state["output_produced"] - 1, 0)
                ),
                raw_model_output_tokens=completion[request_id],
            ))
            output_limit = (
                1 if self.config.is_prefill_only
                else request.output_tokens)
            if state["output_produced"] + completion[request_id] >= output_limit:
                finished.append(request_id)
        if not scheduled:
            raise RuntimeError("conformance backend made no progress")
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:conformance:{self._step}",
                self.config.pool_name,
                tuple(scheduled),
                tuple(finished),
            ),
            generated_tokens=completion,
        )
        self._pending = step
        self._step += 1
        return step

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        if step is not self._pending:
            raise ValueError("completion does not match conformance pending step")
        for scheduled in step.schedule.requests:
            state = self._requests[scheduled.request_id]
            state["prompt_computed"] += scheduled.input_tokens
            state["output_produced"] += step.generated_tokens[
                scheduled.request_id]
        for request_id in step.finished_request_ids:
            self._requests.pop(request_id)
        self._pending = None
        return 0.0


def run(args) -> dict:
    if args.trace_replay_mode == "arrival_rate_sweep":
        payload = run_arrival_rate_sweep(args)
    elif args.trace_replay_mode in (None, "original", "arrival_rate_scale"):
        payload = run_single(args)
    else:
        raise ValueError(
            f"unknown production Trace replay mode: "
            f"{args.trace_replay_mode!r}")
    timing_status = payload["execution_provenance"][
        "performance_timing_status"]
    if timing_status in (
            UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
            DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS):
        warnings.warn(
            "\n"
            "============================================================\n"
            "UNSUPPORTED / DEVELOPMENT ONLY: Serving timing inputs\n"
            f"status={timing_status}\n"
            "Do not use these timings for publication performance claims, "
            "accuracy validation, hardware comparison, or DSE conclusions.\n"
            "============================================================",
            RuntimeWarning,
            stacklevel=2,
        )
    return payload


def run_arrival_rate_sweep(args) -> dict:
    sweep_started = time.perf_counter()
    if args.production_trace is None:
        raise ValueError("arrival-rate sweep requires --production-trace")
    if args.trace_arrival_rate_scales is None:
        raise ValueError(
            "arrival-rate sweep requires --trace-arrival-rate-scales")
    if args.interactivity_tgs_config is None:
        interactivity_tgs_contract = None
        analysis_points = None
    elif (isinstance(args.interactivity_tgs_config, str)
          and args.interactivity_tgs_config):
        interactivity_tgs_contract = load_interactivity_tgs_contract(
            args.interactivity_tgs_config)
        require_expected_sha256(
            "Interactivity--TGS contract",
            args.interactivity_tgs_config_sha256,
            interactivity_tgs_contract.source_sha256,
        )
        analysis_points = []
    else:
        raise ValueError(
            "--interactivity-tgs-config must be a non-empty path")
    trace = load_production_trace(
        args.production_trace,
        production_trace_reader_config(args),
    )
    require_expected_sha256(
        "production Trace",
        args.trace_source_sha256,
        trace.manifest.source_sha256,
    )
    replays = build_production_trace_arrival_rate_sweep(
        trace,
        args.trace_arrival_rate_scales,
    )
    trace_manifest = production_trace_manifest(trace)
    points = []
    shared_execution_provenance = None
    for point_index, replay in enumerate(replays):
        point_args = copy(args)
        point_args.trace_replay_mode = "arrival_rate_scale"
        point_args.trace_arrival_rate_scale = replay.arrival_rate_scale
        point_args.trace_arrival_rate_scales = None
        point_args.interactivity_tgs_config = None
        point_args.output = None
        if args.state_audit_output is not None:
            audit_root = Path(args.state_audit_output)
            point_args.state_audit_output = str(audit_root.parent / (audit_root.stem + ".points") /
                (f"point-{point_index:04d}.json"))
        if args.interactive_report:
            report_root = (Path(args.interactive_output) if args.interactive_output is not None else
                Path(args.output).with_suffix(".html") if args.output else
                Path("experiment_logs") / "serving_reports" / str(time.time_ns()) / "sweep.html")
            point_args.interactive_report = True
            point_args.interactive_output = str(report_root.parent / (report_root.stem + ".points") /
                (f"point-{point_index:04d}-scale-" + str(replay.arrival_rate_scale) + ".html"))
        result = run_single(point_args)
        request_source = result["request_source"]
        if request_source["kind"] != "production_trace":
            raise RuntimeError(
                "arrival-rate sweep point did not use a production Trace")
        if request_source["trace"] != trace_manifest:
            raise RuntimeError(
                "arrival-rate sweep point changed source Trace provenance")
        replay_manifest = production_trace_replay_manifest(replay)
        if request_source["replay"] != replay_manifest:
            raise RuntimeError(
                "arrival-rate sweep point disagrees with canonical replay")
        execution_provenance = result["execution_provenance"]
        if shared_execution_provenance is None:
            shared_execution_provenance = execution_provenance
        elif execution_provenance != shared_execution_provenance:
            raise RuntimeError(
                "arrival-rate sweep execution provenance changed between "
                "points")
        aggregate_result = production_trace_aggregate_result(result)
        points.append({
            "arrival_rate_scale": replay.arrival_rate_scale,
            "state_audit_output": point_args.state_audit_output,
            "request_stream_sha256": replay.request_stream_sha256,
            "validation_status": {
                "trace_replay": "validated_conservation",
                "scheduler_control_plane": "native_pinned_vllm",
                "performance_timing": result[
                    "performance_timing_status"],
            },
            "metrics": {
                "request_throughput_per_s": result["metrics"][
                    "request_throughput_per_s"],
                "output_token_throughput_per_s": result["metrics"][
                    "output_token_throughput_per_s"],
                "token_throughput_per_s": result["metrics"][
                    "token_throughput_per_s"],
                "latency_distributions": result["metrics"][
                    "latency_distributions"],
                "slo_goodput": result["metrics"]["slo_goodput"],
            },
            "result": aggregate_result,
        })
        if analysis_points is not None:
            analysis_points.append(sweep_metric_point_from_manifest(
                replay.arrival_rate_scale,
                result["metrics"],
                interactivity_tgs_contract,
            ))
    if shared_execution_provenance is None:
        raise RuntimeError("arrival-rate sweep produced no execution points")
    analysis = arrival_rate_sweep_analysis_manifest(
        analysis_points,
        interactivity_tgs_contract,
    )
    payload = {
        "schema_version": 3,
        "kind": "production_trace_arrival_rate_sweep",
        "interpretation": "successful_path_performance_only",
        "interactivity_tgs": analysis["interactivity_tgs"],
        "peak_selection": analysis["peak_selection"],
        "saturation_analysis": analysis["saturation_analysis"],
        "scale_order": "caller_supplied",
        "arrival_transform": (
            "t_anchor + (t - t_anchor) / arrival_rate_scale"),
        "point_count": len(points),
        "arrival_rate_scales": tuple(
            replay.arrival_rate_scale for replay in replays),
        "source_trace": trace_manifest,
        "execution_provenance": shared_execution_provenance,
        "persistence": {
            "schema_version": 1,
            "projection": "aggregate_only_production_trace_v1",
            "request_level_content": "omitted",
        },
        "points": points,
    }
    if not args.interactive_report:
        payload["interactive_report"] = {"status": "disabled", "reason": "offline HTML report not requested"}
    else:
        from evaluation.reporting.serving_interactive import default_report_path, write_status_report
        links = [{"label": "Point " + str(index) + " / arrival scale " + str(point["arrival_rate_scale"]),
            "path": point["result"]["interactive_report"]["path"]}
            for index, point in enumerate(points)]
        sweep_report_path = default_report_path(args.output, args.interactive_output)
        links.append({"label": "Sweep cost receipt", "path": str(sweep_report_path.with_suffix(".report.json"))})
        payload["interactive_report"] = write_status_report({
            "main_chain_status": {"status": "complete", "reason": "Sweep complete; open the per-point reports for timelines and effective configurations"},
            "report_links": links, "request_source": {"source_trace": trace_manifest,
                "arrival_rate_scales": payload["arrival_rate_scales"]}},
            sweep_report_path)
        payload["interactive_report"]["cost_receipt_path"] = str(sweep_report_path.with_suffix(".report.json"))
    payload["report_output_costs"] = {"sweep_wall_s": time.perf_counter() - sweep_started,
        "scope": "includes all point simulations, recording and outputs plus sweep overview; point wall times are nested"}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["report_output_costs"]["sweep_total_wall_s"] = time.perf_counter() - sweep_started
    payload["report_output_costs"]["persisted_boundary"] = "sweep total through initial aggregate write; refresh and sidecar write excluded from persisted total"
    if args.output:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.interactive_report:
        receipt_started = time.perf_counter()
        Path(payload["interactive_report"]["cost_receipt_path"]).write_text(json.dumps({
            "costs": payload["report_output_costs"], "report": payload["interactive_report"]}, indent=2) + "\n")
        payload["report_output_costs"]["cost_receipt_write_s"] = time.perf_counter() - receipt_started
        payload["report_output_costs"]["sweep_total_wall_s"] = time.perf_counter() - sweep_started
    return payload


def serving_progress_reporter(
        total_requests: int,
        every_steps: int = 100,
        heartbeat_seconds: float = 30.0,
):
    """Return a stderr reporter and heartbeat for one CLI simulation run."""

    if (isinstance(total_requests, bool)
            or not isinstance(total_requests, int)
            or total_requests <= 0):
        raise ValueError("Serving progress total_requests must be positive")
    if (isinstance(every_steps, bool)
            or not isinstance(every_steps, int)
            or every_steps <= 0):
        raise ValueError("Serving progress interval must be positive")
    if (isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or not math.isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0):
        raise ValueError("Serving progress heartbeat must be positive")
    completed_steps = 0
    scheduled_request_rows = 0
    scheduled_tokens = 0
    finished_request_ids = set()
    started_at = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(float(heartbeat_seconds)):
            peak_rss_bytes = max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            ) * 1024
            print(
                "[serving-progress-heartbeat] "
                f"wall_elapsed_s={time.monotonic() - started_at:.1f} "
                f"completed_steps={completed_steps} "
                f"finished_requests={len(finished_request_ids)}/"
                f"{total_requests} "
                f"scheduled_request_rows={scheduled_request_rows} "
                f"scheduled_tokens={scheduled_tokens} "
                f"peak_rss_bytes={peak_rss_bytes}",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(
        target=heartbeat,
        name="serving-progress-heartbeat",
        daemon=True,
    ).start()

    def report(backend_step: BackendStep, virtual_time_s: float) -> None:
        nonlocal completed_steps, scheduled_request_rows, scheduled_tokens
        if not isinstance(backend_step, BackendStep):
            raise TypeError("Serving progress requires BackendStep")
        if (isinstance(virtual_time_s, bool)
                or not isinstance(virtual_time_s, (int, float))
                or not math.isfinite(virtual_time_s)
                or virtual_time_s < 0):
            raise ValueError(
                "Serving progress virtual time must be finite and non-negative")
        schedule = backend_step.schedule
        completed_steps += 1
        scheduled_request_rows += len(schedule.requests)
        scheduled_tokens += schedule.token_count
        finished_request_ids.update(schedule.finished_request_ids)
        if completed_steps % every_steps:
            return
        print(
            "[serving-progress] "
            f"completed_steps={completed_steps} "
            f"finished_requests={len(finished_request_ids)}/{total_requests} "
            f"scheduled_request_rows={scheduled_request_rows} "
            f"scheduled_tokens={scheduled_tokens} "
            f"virtual_time_s={float(virtual_time_s):.9g} "
            f"pool={schedule.pool_name} step_id={schedule.step_id}",
            file=sys.stderr,
            flush=True,
        )

    def stop() -> None:
        stop_event.set()

    report.stop = stop
    return report


def resolve_executing_model_id(model_id: str | None, model_path: str) -> str:
    """Return the canonical model identity used by workload observations."""

    if model_id is not None:
        return model_id
    return Path(model_path).resolve().stem


def use_streaming_step_retention(args, request_source) -> bool:
    streaming_scope = (
        request_source is not None
        and args.deployment in ("colocated", "pd_disagg")
        and not args.additional_colocated_hardware_config
        and args.kv_lifecycle_replay is None
    )
    if args.step_retention == "auto":
        return streaming_scope
    elif args.step_retention == "full":
        return False
    elif args.step_retention == "streaming_aggregate":
        if not streaming_scope:
            raise NotImplementedError(
                "streaming aggregate retention requires an external request "
                "source, a colocated or non-MTP PD deployment, no additional "
                "colocated hardware, and no KV lifecycle replay")
        return True
    else:
        raise ValueError(
            f"unknown Serving step retention: {args.step_retention!r}")


def run_single(args) -> dict:
    report_total_started = time.perf_counter()
    repository_at_start = repository_provenance()
    adaptive_tail_workers = None
    adaptive_tail_worker_grants = None
    if args.slo_config is None:
        slo_contract = None
    elif isinstance(args.slo_config, str) and args.slo_config:
        slo_contract = load_slo_contract(args.slo_config)
        require_expected_sha256(
            "SLO contract",
            args.slo_config_sha256,
            slo_contract.source_sha256,
        )
    else:
        raise ValueError("--slo-config must be a non-empty path")
    if args.runtime_memory_policy is None:
        runtime_memory_policy = None
    elif (isinstance(args.runtime_memory_policy, str)
          and args.runtime_memory_policy):
        runtime_memory_policy = load_runtime_memory_policy(
            args.runtime_memory_policy)
    else:
        raise ValueError("--runtime-memory-policy must be a non-empty path")
    router_config = RouterConfig(
        args.router_resource_id,
        args.router_dispatch_service_time_s,
    )
    model = load_model_config(args.model)
    require_expected_sha256(
        "model config",
        args.model_config_sha256,
        model.config_sha256,
    )
    if args.kv_lifecycle_replay is not None:
        if (model.model_type == "qwen3"
                and model.num_experts == 0
                and model.attention_layout == "standard_gqa"
                and model.compute_dtype == "bf16"
                and model.weight_dtype == "bf16"):
            if args.mtp_acceptance_policy != "disabled":
                raise NotImplementedError(
                    "dense Qwen3 KV lifecycle replay requires MTP disabled")
        elif (model.model_type == "glm4_moe"
              and model.num_experts > 0
              and model.attention_layout == "standard_gqa"
              and model.compute_dtype == "bf16"
              and model.weight_dtype == "bf16"
              and model.mtp_architecture == "glm4_moe_mtp"
              and model.num_mtp_layers == 1):
            if args.mtp_acceptance_policy != "fixed_count":
                raise NotImplementedError(
                    "GLM4-MoE MTP KV lifecycle replay currently requires "
                    "fixed_count acceptance")
            if model.mtp_layer_uses_moe(0):
                raise NotImplementedError(
                    "KV lifecycle replay does not support a MoE MTP layer")
        else:
            raise NotImplementedError(
                "KV lifecycle replay currently requires dense BF16 Qwen3 "
                "or BF16 GLM4-MoE with one dense MTP layer and standard GQA")
    k3_model = (model.model_type == "kimi_linear"
        and args.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync")
    target_model = k3_model or model.model_type in (
        "kimi_k2", "glm_moe_dsa", "deepseek_v3", "deepseek_v4")
    if target_model:
        if args.attention_kv_layout != "mla":
            raise ValueError(
                "Kimi-K3, Kimi-K2.6, GLM-5.2, DeepSeek-V3, and DeepSeek-V4 require "
                "--attention-kv-layout mla")
        if model.model_type in ("deepseek_v3", "deepseek_v4") or k3_model:
            # DeepSeek-V3 has no source-backed PP production layout in
            # this repository; the H200 TP8/PP1/EP1 predicates below are
            # its only validated target topologies.  DeepSeek-V4 likewise
            # admits only its pinned H200 colocated rows below. K3 uses its
            # explicit TP1/PP1 or TP1/PP16 source scope instead.
            target_parallelism = None
        else:
            target_parallelism = {
                "kimi_k2": (8, 2),
                "glm_moe_dsa": (8, 4),
            }[model.model_type]
        target_pd_parallelism = (
            args.deployment == "pd_disagg"
            and args.prefill_tp_size == 8
            and args.decode_tp_size == 8
            and args.prefill_pp_size == 1
        )
        target_colocated_parallelism = (
            args.deployment == "colocated"
            and ((k3_model and (args.prefill_tp_size, args.prefill_pp_size) in ((1, 1), (1, 16)))
                or (not k3_model and target_parallelism is not None
                    and (args.prefill_tp_size, args.prefill_pp_size) == target_parallelism))
        )
        # InferenceX-style aggregate GLM-5.2 serving uses one TP8 pool with
        # EP=1 and PP=1.  It is distinct from the source-backed PP4/EP8
        # production-trace scope below, so retain an explicit predicate.
        target_h200_aggregate_parallelism = (
            model.model_type == "glm_moe_dsa"
            and args.deployment == "colocated"
            and (args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
            and not args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        # PD TP8/PP1 with EP=1 shares the EP8 KV transfer contract: the
        # compressed MLA KV and FP8 index cache are TP-replicated
        # single-head caches, and EP only shards MoE expert weights.
        target_h200_pd_ep1_parallelism = (
            model.model_type == "glm_moe_dsa"
            and target_pd_parallelism
            and not args.enable_expert_parallel
        )
        # Kimi-K2.6 H200 InferenceX-style colocated serving uses the
        # standard vLLM DeepSeek-V3-family single-node topology: one TP8
        # pool with EP=TP and PP1.  It is distinct from the source-backed
        # PP2/EP8 production scope, so retain an explicit predicate.
        target_kimi_h200_colocated_parallelism = (
            model.model_type == "kimi_k2"
            and args.deployment == "colocated"
            and (args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
            and args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        # DeepSeek-V3 H200 InferenceX scopes mirror the GLM-5.2 H200
        # aggregate and PD TP8/PP1/EP1 layouts; its PD transfer payload is
        # the compressed MLA cache only (no DSA index cache).
        target_dsv3_h200_aggregate_parallelism = (
            model.model_type == "deepseek_v3"
            and args.deployment == "colocated"
            and ((args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
                 or ((args.prefill_tp_size, args.prefill_pp_size) == (4, 1)
                     and model.config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256))
            and not args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        target_dsv3_h200_pd_ep1_parallelism = (
            model.model_type == "deepseek_v3"
            and target_pd_parallelism
            and not args.enable_expert_parallel
        )
        # DeepSeek-V4 H200 colocated scopes: Flash at TP4/PP1/EP1, Pro at
        # TP8/PP1/EP1, each pinned to its checkpoint identity.
        target_dsv4_h200_colocated_parallelism = (
            model.model_type == "deepseek_v4"
            and args.deployment == "colocated"
            and (((args.prefill_tp_size, args.prefill_pp_size) == (4, 1)
                  and model.config_sha256 == DEEPSEEK_V4_FLASH_CONFIG_SHA256)
                 or ((args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
                     and model.config_sha256
                     == DEEPSEEK_V4_PRO_CONFIG_SHA256))
            and not args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        if args.glm_mtp_step_inputs is None and not args.glm_mtp_native_source_inputs and not glm_bf16_cli_source_scope(args) and ((not target_pd_parallelism and not target_colocated_parallelism
             and not target_h200_aggregate_parallelism
             and not target_kimi_h200_colocated_parallelism
             and not target_dsv3_h200_aggregate_parallelism
             and not target_dsv3_h200_pd_ep1_parallelism
             and not target_dsv4_h200_colocated_parallelism)
                or (not k3_model and not target_h200_aggregate_parallelism
                    and not target_h200_pd_ep1_parallelism
                    and not target_dsv3_h200_aggregate_parallelism
                    and not target_dsv3_h200_pd_ep1_parallelism
                    and not target_dsv4_h200_colocated_parallelism
                    and not args.enable_expert_parallel)
                or args.additional_colocated_hardware_config):
            raise NotImplementedError(
                "target source scopes require K3 H200 TP1/PP1 or TP1/PP16, "
                "one colocated TP=EP=8 Scheduler pool (Kimi-K2.6 PP2 or "
                "GLM-5.2 PP4), the PD TP8/PP1 deployment, GLM-5.2 "
                "H200 aggregate TP8/PP1/EP1, GLM-5.2 H200 PD "
                "TP8/PP1/EP1, Kimi-K2.6 H200 colocated TP8/PP1/EP8, "
                "DeepSeek-V3 H200 aggregate TP8/PP1/EP1, DeepSeek-V3 "
                "H200 PD TP8/PP1/EP1, or DeepSeek-V4 H200 colocated "
                "TP4/PP1/EP1 (Flash) or TP8/PP1/EP1 (Pro)")
        if (args.deployment == "colocated"
                # PP1 has no pipeline links or inter-stage placement.
                and args.prefill_pp_size > 1
                and not glm_bf16_cli_source_scope(args)
                and args.pipeline_ranks_per_node != 8):
            raise NotImplementedError(
                "target source pipelines require explicit eight-GPU-node "
                "placement --pipeline-ranks-per-node 8")
        if args.kv_lifecycle_replay is not None:
            raise NotImplementedError(
                "Kimi-K2.6 and GLM-5.2 production traces do not yet support "
                "KV offload replay")
        if (model.model_type == "kimi_k2"
                and args.mtp_acceptance_policy != "disabled"):
            raise NotImplementedError(
                "Kimi-K2.6 fixed text config declares zero MTP layers")
    else:
        if (args.backend == "vllm"
                and args.attention_kv_layout != "paged_varlen"):
            raise ValueError(
                "the standard-GQA vLLM path requires "
                "--attention-kv-layout paged_varlen")
        if model.attention_layout != "standard_gqa":
            raise NotImplementedError(
                "the production runner supports non-standard attention only "
                "for the fixed Kimi-K2.6 and GLM-5.2 contracts")
        if model.weight_dtype != model.compute_dtype:
            raise NotImplementedError(
                "the production runner supports quantized weights only for "
                "the fixed Kimi-K2.6 contract")
    prefill_hardware = HWSpec(args.hardware_config)
    # Resolve each hardware's own launch-data provenance before scheduler
    # work, including H200's observed LUT despite shared analytical code.
    provenance_hardware_ids = {prefill_hardware.hardware_id}
    if args.decode_hardware_config is not None:
        provenance_hardware_ids.add(
            HWSpec(args.decode_hardware_config).hardware_id)
    for config_path in args.additional_colocated_hardware_config:
        provenance_hardware_ids.add(HWSpec(config_path).hardware_id)
    launch_data_provenance(tuple(sorted(provenance_hardware_ids)), repository_root=ROOT)
    require_expected_sha256(
        "hardware config",
        args.hardware_config_sha256,
        file_provenance(args.hardware_config)["sha256"],
    )
    bind_tensor_collective_profiles(
        args,
        prefill_hardware.hardware_id,
    )
    host_launch_profiles = None
    host_runtime_prepare_profiles = None
    host_request_path_profiles = None
    scheduler_host_profiles = None
    runtime_prepare_event_builder = None
    if args.serving_timing_mode != "device_only":
        profiles = tuple(
            load_host_launch_profile(path)
            for path in args.host_launch_profile
        )
        host_launch_profiles = {
            profile.hardware_id: profile for profile in profiles
        }
        if len(host_launch_profiles) != len(profiles):
            raise ValueError("duplicate Host launch profile hardware_id")
        if args.serving_timing_mode == "device_plus_launch_plus_scheduler":
            scheduler_profiles = tuple(
                load_scheduler_host_profile(path)
                for path in args.scheduler_host_profile
            )
            scheduler_host_profiles = {
                profile.hardware_id: profile for profile in scheduler_profiles
            }
            if len(scheduler_host_profiles) != len(scheduler_profiles):
                raise ValueError("duplicate Scheduler Host profile hardware_id")
            # The per-request Host request path is Host scheduling-path time,
            # so it joins the Scheduler Host CPU mode rather than the launch
            # mode, which deliberately excludes Host scheduling cost.
            request_path_profile_by_hardware = host_request_path_profile_paths(
                args.host_request_path_profile)
            if (prefill_hardware.hardware_id in VERSIONED_HOST_PROFILE_HARDWARE
                    and prefill_hardware.hardware_id
                    not in request_path_profile_by_hardware):
                raise ValueError(
                    f"{prefill_hardware.hardware_id} Host scheduling timing "
                    "requires an explicit --host-request-path-profile")
            if prefill_hardware.hardware_id in request_path_profile_by_hardware:
                request_path_profile = load_host_request_path_profile(
                    request_path_profile_by_hardware[
                        prefill_hardware.hardware_id])
                if (request_path_profile.hardware_id
                        != prefill_hardware.hardware_id):
                    raise ValueError(
                        "Host request path profile hardware disagrees with model")
                host_request_path_profiles = {
                    request_path_profile.hardware_id: request_path_profile,
                }
        runtime_prepare_profile_path = host_runtime_prepare_profile_path(
            prefill_hardware.hardware_id, args.host_runtime_prepare_profile)
        if runtime_prepare_profile_path is not None:
            runtime_prepare_profile = load_host_runtime_prepare_profile(
                runtime_prepare_profile_path)
            if runtime_prepare_profile.hardware_id != prefill_hardware.hardware_id:
                raise ValueError("runtime prepare profile hardware disagrees with model")
            host_runtime_prepare_profiles = {
                runtime_prepare_profile.hardware_id: runtime_prepare_profile,
            }
            enabled_mechanisms = tuple(
                mechanism for mechanism, _ in
                runtime_prepare_profile.service_time_s_by_mechanism)
            runtime_prepare_event_builder = partial(
                runtime_prepare_events,
                enabled_mechanisms=enabled_mechanisms,
                **({"ascend_compilation_mode": args.graph_mode}
                   if prefill_hardware.hardware_id == "ascend_910b" else {}),
            )
    target_h200_scope = (
        (k3_model or model.model_type in ("kimi_k2", "glm_moe_dsa"))
        and prefill_hardware.hardware_id == "nvidia_h200"
        and args.deployment in ("colocated", "pd_disagg"))
    if (args.prefill_pp_size > 1
            and prefill_hardware.hardware_id != "nvidia_a800"
            and not target_h200_scope and not gqa_pd_cli_scope(args)):
        raise NotImplementedError(
            "pipeline parallel Serving path requires nvidia_a800 or the "
            "explicit H200 target deployment scope")
    if ((args.prefill_tp_size, args.prefill_pp_size) == (2, 2)
            and not gqa_pd_cli_scope(args) and not glm_bf16_cli_source_scope(args) and (
            model.model_type != "qwen3"
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16")):
        raise NotImplementedError(
            "the validated TP=2, PP=2 Serving slice requires dense BF16 "
            "Qwen3 with standard GQA")
    if (not target_model
            and args.prefill_tp_size == 1
            and args.prefill_pp_size > 1 and not gqa_pd_cli_scope(args)) and (
            args.prefill_tp_size != 1
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.tie_word_embeddings
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or model.num_hidden_layers % args.prefill_pp_size):
        raise NotImplementedError(
            "the validated uniform TP=1 PP Serving path requires untied "
            "dense BF16 Qwen3 with standard GQA and decoder layers "
            "divisible by PP")
    (
        requests,
        arrival_process,
        workload_hash,
        request_source,
        result_version,
    ) = load_request_source(args)
    if (args.prefix_cache_mode == "trace_no_hit"
            and any(request.prefix_hit_tokens != 0 for request in requests)):
        raise ValueError("trace_no_hit requires a recorded zero prefix hit for every request")
    use_streaming_aggregate = use_streaming_step_retention(
        args, request_source)
    from evaluation.reporting.serving_interactive import default_report_path
    interactive_report_path = (default_report_path(args.output, args.interactive_output)
        if args.interactive_report else None)
    interactive_recording = None
    configuration_files, configuration_missing = [], []
    if interactive_report_path is not None:
        from evaluation.reporting.serving_viewer_ui import configuration_file_snapshots
        configuration_files, configuration_missing = configuration_file_snapshots(args.model,
            [args.hardware_config] + ([args.decode_hardware_config] if args.decode_hardware_config is not None else []))
        from evaluation.serving.simulation.interactive_recording import InteractiveRecording, CompactInteractiveRecording
        recording_hardware = {prefill_hardware.hardware_id: prefill_hardware}
        if args.decode_hardware_config is not None:
            recording_decode = HWSpec(args.decode_hardware_config)
            recording_hardware[recording_decode.hardware_id] = recording_decode
        if args.interactive_detail_mode == "all" and args.interactive_view != "swimlanes":
            raise ValueError("all iteration details require --interactive-view swimlanes")
        archive_dir = Path(interactive_report_path).with_suffix(".iterations") if args.interactive_detail_mode == "all" else None
        interactive_recording = (CompactInteractiveRecording(recording_hardware)
            if args.interactive_detail_mode == "summary" else InteractiveRecording(recording_hardware,
                max_detail_events=args.interactive_max_events, detail_archive_dir=archive_dir))
    streaming_artifact = (
        (StreamingExecutionArtifactAccumulator() if interactive_recording is None else interactive_recording)
        if use_streaming_aggregate else None
    )
    progress_callback = serving_progress_reporter(len(requests.requests))
    actual_request_stream_sha256 = request_stream_sha256(requests)
    if args.mtp_acceptance_policy == "maximum_entropy":
        args.mtp_acceptance_assumption = MTPAcceptanceAssumption(
            args.mtp_acceptance_source_id,
            args.mtp_num_speculative_tokens,
            args.mtp_mean_acceptance_rate,
            args.mtp_acceptance_seed,
            actual_request_stream_sha256,
        )
    elif args.mtp_acceptance_assumption is not None:
        raise ValueError(
            "non-maximum-entropy MTP cannot carry an acceptance assumption")
    if (args.kv_lifecycle_replay is not None
            and args.prefill_tp_size == 2):
        if not is_validated_qwen3_tp2_host_kv_model_scope(model):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires the exact "
                "source-backed dense BF16 Qwen3 fixture")
        if (len(requests.requests) != 1
                or requests.requests[0].arrival_time_s != 0.0
                or requests.requests[0].input_tokens != 4
                or requests.requests[0].output_tokens != 2
                or requests.requests[0].prefix_hit_tokens != 0):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires one 4->2/H0 "
                "request arriving at time zero")
        expected_bytes_per_token = (
            model.kv_bytes_per_token_per_parallel_rank(2, 1))
        replay = args.kv_lifecycle_replay
        hbm_tier = next(
            tier for tier in replay.cache_config.tiers
            if tier.name == "hbm")
        host_tier = next(
            tier for tier in replay.cache_config.tiers
            if tier.name == "host")
        if (expected_bytes_per_token != 512
                or replay.cache_config.bytes_per_token != 512
                or hbm_tier.block_size_tokens != 16
                or hbm_tier.capacity_bytes != 2 * 8192
                or host_tier.block_size_tokens != 16
                or host_tier.capacity_bytes != 8192
                or host_tier.link != "pcie"):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires 512 B/token "
                "per rank, two 8192-B HBM blocks, and one 8192-B Host block")
        replay_actions = tuple(
            tuple(event.action for event in step.events)
            for step in replay.steps
        )
        if replay_actions != (
                (KVCacheAction.EVICT,), (KVCacheAction.LOAD,)):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires exactly two "
                "Scheduler steps with EVICT then LOAD")
    mtp_decode_admission_sha256 = None
    if (args.mtp_acceptance_assumption is not None
            and args.deployment == "pd_disagg"):
        mtp_decode_admission_sha256 = (
            pd_decode_admission_contract_sha256(requests))
    # A closed-loop client population submits successors at client-driven
    # completion instants, so neither admission order nor arrival stamps
    # match the canonical stream; bind the order- and arrival-insensitive
    # admission contract instead.
    mtp_closed_loop_admission_sha256 = None
    if (args.mtp_acceptance_assumption is not None
            and args.closed_loop_client_count is not None
            and not args.glm_mtp_native_source_inputs):
        mtp_closed_loop_admission_sha256 = (
            closed_loop_admission_contract_sha256(requests))
        mtp_decode_admission_sha256 = None
    if args.mtp_acceptance_replay is not None:
        replay_context = args.mtp_acceptance_replay.scheduler_context
        if (actual_request_stream_sha256
                != replay_context.ingress_identity_sha256):
            raise ValueError(
                "MTP acceptance replay ingress request stream SHA256 "
                "disagrees with the selected request source")
        if args.mtp_acceptance_replay.schema_version == 2:
            if (args.deployment != "pd_disagg"
                    or replay_context.adapter_contract
                    != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT):
                raise ValueError(
                    "MTP acceptance replay schema v2 requires PD decode")
            mtp_decode_admission_sha256 = (
                pd_decode_admission_contract_sha256(requests))
            if (mtp_decode_admission_sha256
                    != replay_context.decode_admission_contract_sha256):
                raise ValueError(
                    "MTP acceptance replay decode admission SHA256 "
                    "disagrees with the selected request source")
        elif args.mtp_acceptance_replay.schema_version != 1:
            raise AssertionError("unknown MTP replay schema version")
    if args.kv_lifecycle_replay is not None:
        replay_context = args.kv_lifecycle_replay.context
        if actual_request_stream_sha256 != (
                replay_context.request_stream_sha256):
            raise ValueError(
                "KV lifecycle replay request stream SHA256 disagrees with "
                "the selected request source")
        if model.config_sha256 != replay_context.model_config_sha256:
            raise ValueError(
                "KV lifecycle replay model config SHA256 disagrees with "
                "the selected model")
    scheduler_runtime = {}
    kv_capacity_plans = {}
    source_capacity_reports = {}
    if model.num_experts == 0:
        if args.expert_routing_policy is not None:
            raise ValueError(
                "dense model simulation cannot accept expert routing input")
        expert_routing = ExpertRoutingInput.not_applicable()
    elif args.expert_routing_policy == "seeded_assumption":
        if args.expert_routing_hot_expert_count is None:
            expert_routing = ExpertRoutingInput.seeded_assumption(
                args.expert_routing_source_id,
                args.expert_routing_seed,
            )
        else:
            expert_routing = ExpertRoutingInput.skewed_assumption(
                args.expert_routing_source_id,
                args.expert_routing_seed,
                hot_expert_count=args.expert_routing_hot_expert_count,
                hot_expert_weight=args.expert_routing_hot_expert_weight,
            )
    elif args.expert_routing_policy == "replay_observation":
        expert_routing = ExpertRoutingInput.replay_observation(
            args.expert_routing_replay,
            actual_request_stream_sha256,
        )
    elif args.expert_routing_policy == "profile_observation":
        executing_model_id = resolve_executing_model_id(model.model_id, args.model)
        profile = load_expert_frequency_profile(
            args.expert_routing_profile_manifest,
            args.expert_routing_source_id,
            expected_model_id=executing_model_id,
            expected_moe_layers=sum(
                model.layer_uses_moe(layer_index)
                for layer_index in range(model.num_hidden_layers)
            ),
            expected_num_experts=model.num_experts,
        )
        expert_routing = ExpertRoutingInput.profile_observation(
            profile,
            per_moe_layer=(
                args.expert_routing_profile_resolution == "per_moe_layer"),
        )
    elif args.expert_routing_policy is None:
        raise ValueError(
            "MoE model simulation requires --expert-routing-policy")
    else:
        raise ValueError(
            f"unknown expert routing policy: "
            f"{args.expert_routing_policy!r}")
    runtime_inputs = RuntimeInputs(
        expert_routing=expert_routing,
        attention_kv_layout=args.attention_kv_layout,
        attention_kv_distribution=args.attention_kv_distribution,
        graph_execution=args.graph_execution_config,
        serving_framework="vllm",
        model_execution_version=("sglang_0_5_19" if args.symbolic_forward_source in (
            "sglang_0_5_19_deepseek_v3_target_fp8", "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm") else
            "vllm_0_27_1" if args.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8" else
            "vllm_0_18_0" if args.target_sampling_source == "vllm_0_18_plain_greedy_bf16" else
            "vllm_0_24_0" if args.target_sampling_source == "vllm_0_24_plain_greedy_bf16" else
            "vllm_0_30_0" if args.vllm_scheduler_version == "0.30.0" else
            "vllm_0_11_0" if args.symbolic_forward_source in ("cuda_eager_inplace", "vllm_0_11_glm4_sync") else "legacy"),
        v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid" if args.v4_eager_metadata else None,
        v4_source_max_model_len=args.max_model_len if args.v4_eager_metadata else None,
        v4_source_max_batched_tokens=args.max_batched_tokens if args.v4_eager_metadata else None,
        v4_indexer_selection_source=args.v4_indexer_selection_source,
        v4_source_use_deep_gemm=(None if args.v4_source_use_deep_gemm is None
            else args.v4_source_use_deep_gemm == "enabled"),
        v4_routing_source=args.v4_routing_source,
        target_sampling_source=args.target_sampling_source,
        maca_copy_runtime_source=args.maca_copy_runtime_source,
        glm_target_num_speculative_tokens=(args.mtp_num_speculative_tokens if args.glm_mtp_native_source_inputs else 0),
        mtp_draft_sampling_source=("vllm_0_30_legacy_full_logits_greedy" if args.glm_mtp_step_inputs is not None or args.glm_mtp_native_source_inputs else "vllm_0_11_eager_full_logits_greedy"
            if args.symbolic_forward_source == "vllm_0_11_glm4_sync" and args.mtp_num_speculative_tokens else None),
        symbolic_forward_source=args.symbolic_forward_source,
        norm_source_batch_invariant=(None if args.norm_source_batch_invariant is None else False),
        bf16_gemm_model=args.bf16_gemm_model,
        bf16_expert_model=args.bf16_expert_model,
        dense_source_input_policy=args.dense_source_input_policy,
        dense_source_capture_limits=((args.max_requests, args.max_model_len)
            if args.dense_source_input_policy is not None else None),
        symbolic_source_max_model_len=(args.max_model_len if args.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_30_k3_mxfp4_sync", "vllm_0_27_1_deepseek_v3_sync_fp8") or (
            args.symbolic_forward_source == "vllm_0_11_glm4_sync" and args.mtp_num_speculative_tokens) else None),
        symbolic_source_max_batched_tokens=(args.max_batched_tokens if args.symbolic_forward_source in (
            "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
            "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_30_k3_mxfp4_sync", "vllm_0_27_1_deepseek_v3_sync_fp8") or (
            args.symbolic_forward_source == "vllm_0_11_glm4_sync" and args.mtp_num_speculative_tokens) else None),
        symbolic_source_max_requests=(args.max_requests
            if (args.symbolic_forward_source == "vllm_0_11_glm4_sync" and args.mtp_num_speculative_tokens)
                or args.glm_mtp_step_inputs is not None or args.glm_mtp_native_source_inputs else None),
        glm_indexer_selection_source=args.glm_indexer_selection_source,
        glm_prefill_attention_source=args.glm_prefill_attention_source,
        symbolic_workspace_initial_capacity_bytes=(None if args.source_workspace_initial_capacity_bytes is None
            else tuple(args.source_workspace_initial_capacity_bytes)),
        model_source_inputs=(None if args.glm_mtp_step_inputs is None else args.glm_mtp_step_inputs["main:vllm:0"]),
        symbolic_source_prompt_lengths_by_request=({request.request_id: request.input_tokens for request in requests}
            if args.glm_mtp_native_source_inputs else None),
        embedding_token_ids=(None if args.glm_mtp_initial_target_token_ids is None else tuple(args.glm_mtp_initial_target_token_ids)),
    )
    if (args.vllm_async_terminal_guard_source is not None
            and runtime_inputs.model_execution_version != "vllm_0_18_0"):
        raise ValueError(
            "vLLM async terminal guard diagnostic requires vLLM 0.18 "
            "model execution source")

    if model.compute_dtype == "bf16":
        scheduler_dtype = "bfloat16"
    elif model.compute_dtype == "fp16":
        scheduler_dtype = "float16"
    else:
        raise NotImplementedError(
            "the pinned vLLM Scheduler path supports only BF16 and FP16 "
            f"model compute dtype, got {model.compute_dtype!r}")

    def make_backend(
            pool_name, role, hardware, hardware_config_path, tp_size,
            pp_size=1):
        nonlocal runtime_memory_policy
        if pool_name in kv_capacity_plans:
            raise RuntimeError(
                f"scheduler backend {pool_name!r} was constructed twice")
        parallelism = deployment.worker_pools[pool_name].parallel_plan.config
        if (parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size) != (tp_size, pp_size):
            raise ValueError("native backend constructor disagrees with its deployment TP/PP geometry")
        enabled_ep = parallelism.enable_expert_parallel and (tp_size > 1 or parallelism.context_parallel_size > 1
            or model.model_type == "glm_moe_dsa" and parallelism.data_parallel_size > 1)
        layer_partition = (args.decode_pp_layer_partition if role == "decode" else args.prefill_pp_layer_partition)
        if args.serving_timing_mode == "device_plus_launch_plus_scheduler":
            if scheduler_host_profiles is None:
                raise AssertionError("Scheduler Host profiles were not loaded")
            if hardware.hardware_id not in scheduler_host_profiles:
                raise ValueError(
                    "Scheduler Host profiles do not cover hardware: "
                    f"{hardware.hardware_id}")
            scheduler_host_profile = scheduler_host_profiles[
                hardware.hardware_id]
            if hardware.hardware_id in ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE:
                engine_core_host_round_profile = load_engine_core_host_round_profile(
                    ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE[
                        hardware.hardware_id])
                if (engine_core_host_round_profile.hardware_id
                        != hardware.hardware_id):
                    raise ValueError(
                        "EngineCore host-round profile hardware disagrees "
                        "with model")
            else:
                engine_core_host_round_profile = None
        else:
            scheduler_host_profile = None
            engine_core_host_round_profile = None
        if args.vllm_scheduler_version == "0.30.0":
            if runtime_memory_policy is None:
                if not k3_model or (tp_size, pp_size) != (1, 1):
                    raise ValueError("0.30 Serving requires a normal runtime memory policy outside mathematical K3 TP1")
                settings = vllm030_cli_scheduler_settings(args, tp_size=tp_size, pp_size=pp_size,
                    scheduler_dtype=scheduler_dtype, gpu_memory_utilization=0.9,
                    request_stream_identity=actual_request_stream_sha256,
                    scheduler_host_profile=scheduler_host_profile,
                    engine_core_host_round_profile=engine_core_host_round_profile)
                backend = VLLMBackend.from_vllm030_byte_budget(settings,
                    python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
                    pool_name=pool_name, max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
                    kv_block_size_tokens=args.kv_block_size_tokens, available_memory_by_rank=(2 * 1024**3,),
                    supported_layouts_by_rank=((args.vllm_cache_layout,),))
                kv_capacity_plans[pool_name] = resolve_debug_vllm030_byte_budget_capacity_plan(pool_name,
                    native_allocation=VLLMKVAllocation(**backend.runtime_info["native_allocation"]))
                return backend
            policy = runtime_memory_policy.policy_for_pool(pool_name)
            settings = vllm030_cli_scheduler_settings(args, tp_size=tp_size, pp_size=pp_size,
                scheduler_dtype=scheduler_dtype,
                gpu_memory_utilization=policy.gpu_memory_utilization_numerator / policy.gpu_memory_utilization_denominator,
                request_stream_identity=actual_request_stream_sha256, enable_expert_parallel=enabled_ep,
                cp_size=parallelism.context_parallel_size,
                dp_size=parallelism.data_parallel_size if model.model_type == "glm_moe_dsa" else 1,
                pipeline_layer_partition=layer_partition,
                scheduler_host_profile=scheduler_host_profile,
                engine_core_host_round_profile=engine_core_host_round_profile,
                role=role, decode_admission_identity=mtp_decode_admission_sha256)
            layouts = ((args.vllm_cache_layout,),) * parallelism.worker_count
            prepared_processes = []
            def retain_allocator_process(control):
                if prepared_processes:
                    raise RuntimeError("one native allocator Scheduler was already retained for this pool")
                construction_lifecycle.callback(control.close)
                prepared_processes.append(control)
            retain_startup = (args.activation_source_profile is None and role == "colocated"
                and args.deployment == "colocated" and model.attention_layout == "standard_gqa"
                and tp_size in (1, 2) and pp_size == parallelism.context_parallel_size == parallelism.data_parallel_size == 1
                and args.mtp_num_speculative_tokens == 0)
            allocator = partial(allocate_vllm030_kv_cache, scheduler_config=settings,
                python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
                max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
                supported_layouts_by_rank=layouts[:tp_size * pp_size * parallelism.context_parallel_size],
                **({"pool_name": pool_name, "process_consumer": retain_allocator_process} if retain_startup else {}))
            glm_fp8_normal = (model.model_type == "glm_moe_dsa"
                and model.weight_dtype == "fp8" and bool(model.quantized_weight_targets)
                and runtime_inputs.symbolic_forward_source == "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            if glm_fp8_normal:
                if settings.mtp_num_speculative_tokens:
                    raise NotImplementedError("GLM FP8 MTP5 normal capacity awaits complete D-only startup/state budget; target-only budget cannot certify MTP")
                if args.activation_source_profile is not None:
                    raise ValueError("GLM FP8 owns its small/large-M source budget; do not replace it with a dense activation profile")
                projected, plan, source_report = resolve_glm_block_fp8_kv_capacity_plan(
                    policy, model=model, parallelism=parallelism, hardware=hardware,
                    hardware_config_sha256=file_provenance(hardware_config_path)["sha256"],
                    allocator=allocator, expert_routing=runtime_inputs.expert_routing)
                runtime_memory_policy = replace(runtime_memory_policy,
                    policies={**runtime_memory_policy.policies, pool_name: projected})
                source_capacity_reports[pool_name] = {
                    "projected_policy": runtime_memory_policy_payload((projected,)), "report": source_report}
            elif args.activation_source_profile is not None:
                validate_activation_source_policy_binding(args, policy, hardware, model, pool_name, role, parallelism)
                projected, plan, source_report = resolve_symbolic_kv_capacity_plan(policy,
                    hardware=hardware, hardware_config_sha256=file_provenance(hardware_config_path)["sha256"],
                    model=model, parallelism=parallelism, source_profile=args.activation_source_profile,
                    allocator=allocator, scheduler_version="0.30.0",
                    model_execution_version=runtime_inputs.model_execution_version,
                    source_lifetime_policy=args.activation_source_lifetime_policy,
                    source_compilation_options=args.activation_source_compilation_options,
                    collective_configs=tensor_collective_configs(pool_name, max(tp_size, parallelism.context_parallel_size), hardware.hardware_id,
                        args.collective_backend, args.tensor_all_reduce_profile_sha256, args.tensor_all_gather_profile_sha256))
                runtime_memory_policy = replace(runtime_memory_policy,
                    policies={**runtime_memory_policy.policies, pool_name: projected})
                source_report["request_configured_graph_mode"] = args.graph_mode
                source_capacity_reports[pool_name] = {
                    "projected_policy": runtime_memory_policy_payload((projected,)), "report": source_report}
            else:
                plan = resolve_analytical_kv_capacity_plan(pool_name, hardware=hardware,
                    hardware_config_sha256=file_provenance(hardware_config_path)["sha256"],
                    model=model, parallelism=parallelism, deployment=args.deployment, worker_role=role,
                    kv_block_size_tokens=args.kv_block_size_tokens, max_model_len=args.max_model_len,
                    max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests, policy=policy,
                    allocator=allocator, scheduler_version="0.30.0", v4_weight_storage_profile=args.v4_weight_storage_profile,
                    pipeline_layer_partition=layer_partition,
                    include_mtp=args.mtp_num_speculative_tokens > 0)
            kv_capacity_plans[pool_name] = plan
            return VLLMBackend.from_vllm030_capacity_plan(plan, settings,
                python_executable=args.vllm_sdk_python, source_root=args.vllm_source_root,
                max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
                supported_layouts_by_rank=layouts,
                **({"prepared_process": prepared_processes[0]} if prepared_processes else {}))
        if runtime_memory_policy is None:
            kv_bytes_per_token = (
                model.kv_bytes_per_token_per_parallel_rank(tp_size, pp_size))
            if model.model_type == "deepseek_v4":
                # DeepSeek-V4 per-token KV bytes are fractional (C128 layers
                # store one 584-byte state per 128 tokens); the debug block
                # budget rounds the per-token byte total up so the explicit
                # --num-kv-blocks budget never under-provisions.
                kv_bytes_per_token = -(-kv_bytes_per_token.numerator
                                       // kv_bytes_per_token.denominator)
            scheduler_gpu_memory_utilization = 0.9
            capacity_plan = resolve_debug_kv_capacity_plan(
                pool_name,
                kv_block_size_tokens=args.kv_block_size_tokens,
                kv_bytes_per_token=kv_bytes_per_token,
                num_kv_blocks=args.num_kv_blocks,
            )
        else:
            policy = runtime_memory_policy.policy_for_pool(pool_name)
            scheduler_gpu_memory_utilization = (
                policy.gpu_memory_utilization_numerator
                / policy.gpu_memory_utilization_denominator)
            if args.activation_source_profile is not None:
                validate_activation_source_policy_binding(args, policy, hardware, model, pool_name, role, parallelism)
                projected, capacity_plan, source_report = resolve_symbolic_kv_capacity_plan(policy,
                    hardware=hardware, hardware_config_sha256=file_provenance(hardware_config_path)["sha256"],
                    model=model, parallelism=parallelism, source_profile=args.activation_source_profile,
                    allocator=allocate_vllm_kv_cache, scheduler_version="0.11.0",
                    model_execution_version=runtime_inputs.model_execution_version,
                    source_lifetime_policy=args.activation_source_lifetime_policy,
                    source_compilation_options=args.activation_source_compilation_options,
                    collective_configs=tensor_collective_configs(pool_name, tp_size, hardware.hardware_id,
                        args.collective_backend, args.tensor_all_reduce_profile_sha256, args.tensor_all_gather_profile_sha256))
                runtime_memory_policy = replace(runtime_memory_policy,
                    policies={**runtime_memory_policy.policies, pool_name: projected})
                source_report["request_configured_graph_mode"] = args.graph_mode
                source_capacity_reports[pool_name] = {
                    "projected_policy": runtime_memory_policy_payload((projected,)), "report": source_report}
            else:
                capacity_plan = resolve_analytical_kv_capacity_plan(
                    pool_name,
                    hardware=hardware,
                    hardware_config_sha256=(
                        file_provenance(hardware_config_path)["sha256"]),
                    model=model,
                    parallelism=parallelism,
                    deployment=args.deployment,
                    worker_role=role,
                    kv_block_size_tokens=args.kv_block_size_tokens,
                    max_model_len=args.max_model_len,
                    max_batched_tokens=args.max_batched_tokens,
                    max_requests=args.max_requests,
                    policy=policy,
                    include_mtp=(
                        model.model_type == "glm_moe_dsa"
                        and args.mtp_num_speculative_tokens > 0
                    ),
                )
            kv_bytes_per_token = capacity_plan.kv_bytes_per_token
        kv_capacity_plans[pool_name] = capacity_plan
        config = BackendConfig(
            pool_name,
            role,
            args.max_batched_tokens,
            args.max_requests,
            capacity_plan.num_kv_blocks,
            kv_block_size_tokens=args.kv_block_size_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
            scheduler_pipeline_depth=args.scheduler_pipeline_depth,
        )
        if (args.deployment in ("pd_disagg", "pd_af_disagg")
                and config.is_prefill_only):
            mtp_num_speculative_tokens = 0
            mtp_acceptance_policy = "disabled"
            mtp_fixed_accepted_tokens = 0
            mtp_acceptance_source_id = "disabled"
            mtp_acceptance_replay = None
            mtp_acceptance_assumption = None
            mtp_request_stream_sha256 = None
            mtp_decode_admission_contract_sha256 = None
            mtp_closed_loop_admission_contract_sha256 = None
        else:
            mtp_num_speculative_tokens = args.mtp_num_speculative_tokens
            mtp_acceptance_policy = args.mtp_acceptance_policy
            mtp_fixed_accepted_tokens = args.mtp_fixed_accepted_tokens
            mtp_acceptance_source_id = args.mtp_acceptance_source_id
            mtp_acceptance_replay = args.mtp_acceptance_replay
            mtp_acceptance_assumption = args.mtp_acceptance_assumption
            mtp_request_stream_sha256 = (
                actual_request_stream_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None
                    or (args.deployment == "af_disagg"
                        and args.max_requests == 2
                        and args.mtp_num_speculative_tokens != 0))
                else None
            )
            mtp_decode_admission_contract_sha256 = (
                mtp_decode_admission_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            )
            mtp_closed_loop_admission_contract_sha256 = (
                mtp_closed_loop_admission_sha256
                if args.mtp_acceptance_assumption is not None
                else None
            )
        if args.backend == "vllm":
            return VLLMBackend.from_model(
                config,
                VLLMSchedulerConfig(
                    args.model,
                    tensor_parallel_size=tp_size,
                    pipeline_parallel_size=pp_size,
                    enable_expert_parallel=args.enable_expert_parallel,
                    dtype=scheduler_dtype,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=(
                        scheduler_gpu_memory_utilization),
                    enable_chunked_prefill=args.enable_chunked_prefill,
                    enable_prefix_caching=args.enable_prefix_caching,
                    prefix_cache_mode=args.prefix_cache_mode,
                    mtp_draft_token_ids_by_step=(args.mtp_draft_token_ids_by_step
                        if mtp_num_speculative_tokens else None),
                    async_scheduling=args.vllm_async_scheduling,
                    async_terminal_guard_source=args.vllm_async_terminal_guard_source,
                    mtp_num_speculative_tokens=(
                        mtp_num_speculative_tokens),
                    mtp_acceptance_policy=mtp_acceptance_policy,
                    mtp_fixed_accepted_tokens=(
                        mtp_fixed_accepted_tokens),
                    mtp_acceptance_source_id=(
                        mtp_acceptance_source_id),
                    mtp_acceptance_replay=mtp_acceptance_replay,
                    mtp_acceptance_assumption=(
                        mtp_acceptance_assumption),
                    request_stream_sha256=mtp_request_stream_sha256,
                    decode_admission_contract_sha256=(
                        mtp_decode_admission_contract_sha256),
                    closed_loop_admission_contract_sha256=(
                        mtp_closed_loop_admission_contract_sha256),
                    authorized_h200_pd_mtp_sweep=(
                        args.glm_mtp_native_source_inputs or is_glm_h200_pd_mtp_target_scope(args, True)
                        or is_dsv3_h200_pd_mtp_target_scope(args)),
                    scheduler_host_profile=scheduler_host_profile,
                    engine_core_host_round_profile=(
                        engine_core_host_round_profile),
                ),
                in_process=args.execution_mode == "fast",
            )
        elif args.backend == "conformance":
            return ConformanceBackend(config)
        else:
            raise ValueError(f"unknown serving backend: {args.backend!r}")

    def validate_capacity_pools(pool_names):
        if runtime_memory_policy is not None:
            runtime_memory_policy.validate_scheduler_pools(pool_names)

    simulation_wall_started = time.perf_counter()
    if args.deployment == "colocated":
        if args.decode_hardware_config is not None:
            raise ValueError(
                "--decode-hardware-config is only valid for pd_disagg")
        if args.additional_colocated_hardware_config:
            hardware_config_paths = (
                args.hardware_config,
                *tuple(args.additional_colocated_hardware_config),
            )
            instance_hardware = (
                prefill_hardware,
                *tuple(HWSpec(path) for path in hardware_config_paths[1:]),
            )
            config_path_by_hardware_id = {}
            for config_path, hardware in zip(
                    hardware_config_paths, instance_hardware):
                resolved_path = Path(config_path).resolve()
                hardware_id = hardware.hardware_id
                if hardware_id in config_path_by_hardware_id:
                    previous_path = config_path_by_hardware_id[hardware_id]
                    if resolved_path != previous_path:
                        raise ValueError(
                            "colocated instances cannot bind different "
                            "hardware config files with the same hardware_id: "
                            f"{hardware_id!r} maps to {previous_path} and "
                            f"{resolved_path}; reuse one config path for "
                            "same-profile instances")
                else:
                    config_path_by_hardware_id[hardware_id] = resolved_path
            pool_names = tuple(
                f"colocated-{index}"
                for index in range(len(instance_hardware))
            )
            pool_configs = []
            next_rank = 0
            for pool_name, hardware in zip(
                    pool_names, instance_hardware, strict=True):
                ranks = tuple(range(
                    next_rank,
                    next_rank + args.prefill_tp_size,
                ))
                next_rank += args.prefill_tp_size
                pool_configs.append(WorkerPoolConfig(
                    pool_name,
                    "colocated",
                    hardware.hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=args.prefill_tp_size,
                        enable_expert_parallel=args.enable_expert_parallel,
                    ),
                    ranks,
                    collectives=tensor_collective_configs(
                        pool_name,
                        args.prefill_tp_size,
                        hardware.hardware_id,
                        args.collective_backend,
                        args.tensor_all_reduce_profile_sha256,
                        args.tensor_all_gather_profile_sha256,
                    ),
                ))
            deployment = build_colocated_plan(
                DeploymentConfig(
                    "colocated",
                    tuple(pool_configs),
                    router=router_config,
                ),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
            )
            hardware_by_id = {
                hardware.hardware_id: hardware
                for hardware in instance_hardware
            }
            validate_capacity_pools(pool_names)
            backends = {}
            with ExitStack() as construction_lifecycle:
                for pool_name, hardware, hardware_config_path in zip(
                        pool_names, instance_hardware, hardware_config_paths,
                        strict=True):
                    backend = make_backend(
                        pool_name, "colocated", hardware,
                        hardware_config_path, args.prefill_tp_size)
                    construction_lifecycle.callback(backend.close)
                    backends[pool_name] = backend
                if args.backend == "vllm":
                    scheduler_runtime.update({
                        pool_name: asdict(backend.runtime_info)
                        for pool_name, backend in backends.items()
                    })
                elif args.backend == "conformance":
                    pass
                else:
                    raise ValueError(
                        f"unknown serving backend: {args.backend!r}")
                result = run_recorded_simulation(MultiSimulation(
                    SimulationConfig(
                        model, deployment, requests, args.launch_policy,
                        args.prediction_method, runtime_inputs,
                        execution_mode=args.execution_mode,
                        collect_source_state=args.state_audit_output is not None,
                        record_execution_timeline=interactive_recording is not None and not interactive_recording.compact_mode,
                        timing_mode=args.serving_timing_mode,
                        host_launch_profiles=host_launch_profiles,
                        host_runtime_prepare_profiles=host_runtime_prepare_profiles,
                        host_request_path_profiles=host_request_path_profiles,
                        runtime_prepare_event_builder=runtime_prepare_event_builder,
                        closed_loop_client_count=args.closed_loop_client_count),
                    backends,
                    hardware_by_id,
                    progress_callback=progress_callback,
                ), interactive_recording)
                construction_lifecycle.pop_all()
        else:
            deployment = build_colocated_plan(
                DeploymentConfig(
                    "colocated",
                    (WorkerPoolConfig(
                        "main",
                        "colocated",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.prefill_tp_size,
                            pipeline_parallel_size=args.prefill_pp_size,
                            data_parallel_size=args.prefill_dp_size,
                            context_parallel_size=args.prefill_cp_size,
                            sequence_parallel_size=(args.prefill_tp_size if model.num_experts
                                and args.enable_expert_parallel and args.prefill_dp_size > 1 else 1),
                            enable_expert_parallel=(
                                args.enable_expert_parallel),
                        ),
                        collectives=tensor_collective_configs(
                            "main", max(args.prefill_tp_size, args.prefill_cp_size),
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256)
                            + moe_data_collective_configs("main", prefill_hardware.hardware_id,
                                args.prefill_dp_size * args.prefill_cp_size, args.collective_backend, model=model),
                        pipeline_layer_partition=args.prefill_pp_layer_partition,
                        pipeline_p2p=pipeline_p2p_configs(
                            "main", args.prefill_pp_size,
                            hardware_id=prefill_hardware.hardware_id,
                            backend=args.collective_backend,
                            tensor_parallel_size=args.prefill_tp_size,
                            ranks_per_node=args.pipeline_ranks_per_node),
                        kv_cache=(
                            None
                            if args.kv_lifecycle_replay is None
                            else args.kv_lifecycle_replay.cache_config
                        ),
                    ),),
                    router=router_config,
                ),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
                pipeline_layer_partition=(args.prefill_pp_layer_partition if args.prefill_pp_layer_partition is not None else tuple(
                    layer_end - layer_start
                    for layer_start, layer_end in (
                        pipeline_layer_ranges_for_model(
                            model, args.prefill_pp_size)
                    )
                )),
            )
            hardware_by_id = {prefill_hardware.hardware_id: prefill_hardware}
            validate_capacity_pools(("main",))
            validate_workload_emission_parallelism(
                deployment,
                model,
                runtime_inputs=runtime_inputs,
                allow_single_pool_tp2=True,
                allow_single_pool_ep2=True,
                allow_single_pool_tp1_pp=True,
                allow_single_pool_tp2_pp2=True,
            )
            with ExitStack() as construction_lifecycle:
                backend = make_backend(
                    "main",
                    "colocated",
                    prefill_hardware,
                    args.hardware_config,
                    args.prefill_tp_size,
                    args.prefill_pp_size,
                )
                construction_lifecycle.callback(backend.close)
                if args.backend == "vllm":
                    runtime_info = backend.runtime_info
                    scheduler_runtime["main"] = (runtime_info if args.vllm_scheduler_version == "0.30.0" else asdict(runtime_info))
                    if expert_routing.source_kind == "replay_observation":
                        expert_routing = expert_routing.bind_vllm_runtime(
                            version=runtime_info.version,
                            source_commit=runtime_info.source_commit,
                            package_python_manifest_sha256=(
                                runtime_info.package_python_manifest_sha256),
                            model_config_sha256=(
                                runtime_info.model_config_sha256),
                        )
                        runtime_inputs = replace(runtime_inputs, expert_routing=expert_routing)
                elif args.backend == "conformance":
                    pass
                else:
                    raise ValueError(
                        f"unknown serving backend: {args.backend!r}")
                deployment = bind_symbolic_mtp_cache(deployment, runtime_inputs, kv_capacity_plans)
                if args.execution_workers == 0 and args.output is not None:
                    from evaluation.serving.metrics.symbolic_model import (
                        _adaptive_tail_workers_for)
                    _adaptive_case_dir = Path(args.output).parent

                    def _adaptive_tail_workers(_dir=_adaptive_case_dir):
                        return _adaptive_tail_workers_for(_dir)

                    adaptive_tail_workers = _adaptive_tail_workers
                simulation = Simulation(
                    SimulationConfig(
                        model, deployment, requests, args.launch_policy,
                        args.prediction_method, runtime_inputs,
                        model_source_inputs_by_step=(args.vllm_step_inputs if args.vllm_step_inputs is not None
                            else args.glm_mtp_step_inputs),
                        execution_mode=args.execution_mode,
                        collect_source_state=args.state_audit_output is not None,
                        record_execution_timeline=interactive_recording is not None and not interactive_recording.compact_mode,
                        step_retention=(
                            "streaming_aggregate"
                            if use_streaming_aggregate else "full"),
                        execution_workers=args.execution_workers,
                        timing_mode=args.serving_timing_mode,
                        host_launch_profiles=host_launch_profiles,
                        host_runtime_prepare_profiles=host_runtime_prepare_profiles,
                        host_request_path_profiles=host_request_path_profiles,
                        runtime_prepare_event_builder=runtime_prepare_event_builder,
                        closed_loop_client_count=args.closed_loop_client_count,
                        vllm_executor_backend=args.vllm_executor_backend),
                    backend,
                    hardware_by_id,
                    kv_cache_tier_manager=(
                        None
                        if args.kv_lifecycle_replay is None
                        else KVCacheTierManager(
                            args.kv_lifecycle_replay.cache_config)
                    ),
                    kv_lifecycle_replay=args.kv_lifecycle_replay,
                    scheduler_runtime_sha256=(
                        None
                        if args.kv_lifecycle_replay is None
                        else vllm_scheduler_runtime_sha256(runtime_info)
                    ),
                    progress_callback=progress_callback,
                    streaming_step_consumer=(
                        None
                        if streaming_artifact is None
                        else streaming_artifact.consume
                    ),
                    streaming_step_projector=(
                        None
                        if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                        else (interactive_recording.project_step if interactive_recording is not None and interactive_recording.compact_mode else
                            streaming_artifact.execution_contract_digest)
                    ),
                    streaming_prehashed_step_consumer=(
                        None
                        if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                        else streaming_artifact.consume_prehashed
                    ),
                    adaptive_tail_workers=adaptive_tail_workers,
                )
                construction_lifecycle.pop_all()
            result = run_recorded_simulation(simulation, interactive_recording)
            if adaptive_tail_workers is not None:
                adaptive_tail_worker_grants = sorted(set(simulation._adaptive_tail_grants))
    elif args.deployment == "af_disagg":
        if args.collective_backend == "protobridge":
            _activation_profile, activation_profile_sha256 = protobridge_p2p_profile(
                prefill_hardware.hardware_id, args.activation_transfer_link)
        elif args.collective_backend == "analytical_roofline_v1":
            activation_profile_sha256 = None
        else:
            raise NotImplementedError("AF requires an analytical or ProtoBridge point-to-point service")
        attention_ranks = tuple(range(args.prefill_tp_size))
        ffn_ranks = tuple(range(
            args.prefill_tp_size,
            args.prefill_tp_size + args.ffn_tp_size,
        ))
        deployment = build_af_disagg_plan(
            DeploymentConfig(
                "af_disagg",
                (
                    WorkerPoolConfig(
                        "attention",
                        "attention",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.prefill_tp_size),
                        attention_ranks,
                        kv_cache=(
                            None
                            if args.kv_lifecycle_replay is None
                            else args.kv_lifecycle_replay.cache_config
                        ),
                        collectives=tensor_collective_configs(
                            "attention", args.prefill_tp_size,
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256),
                    ),
                    WorkerPoolConfig(
                        "ffn",
                        "ffn",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.ffn_tp_size,
                            enable_expert_parallel=(
                                args.enable_expert_parallel),
                        ),
                        ffn_ranks,
                        collectives=tensor_collective_configs(
                            "ffn", args.ffn_tp_size,
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256),
                    ),
                ),
                (
                    RouteConfig(
                        "attention", "ffn", "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=p2p_resource_id(
                            args.activation_transfer_link,
                            "attention",
                            "ffn",
                        ),
                        backend=args.collective_backend,
                        backend_config_sha256=activation_profile_sha256,
                    ),
                    RouteConfig(
                        "ffn", "attention", "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=p2p_resource_id(
                            args.activation_transfer_link,
                            "ffn",
                            "attention",
                        ),
                        backend=args.collective_backend,
                        backend_config_sha256=activation_profile_sha256,
                    ),
                ),
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
        }
        validate_capacity_pools(("attention",))
        validate_workload_emission_parallelism(
            deployment,
            model,
            runtime_inputs=runtime_inputs,
            allow_single_pool_tp2=False,
            allow_single_pool_ep2=False,
            allow_single_pool_tp1_pp=False,
            allow_single_pool_tp2_pp2=False,
        )
        simulation_config = SimulationConfig(
            model, deployment, requests, args.launch_policy,
            args.prediction_method, runtime_inputs,
            execution_mode=args.execution_mode,
                        collect_source_state=args.state_audit_output is not None,
                        record_execution_timeline=interactive_recording is not None and not interactive_recording.compact_mode,
            step_retention=(
                "streaming_aggregate" if use_streaming_aggregate else "full"),
            timing_mode=args.serving_timing_mode,
            host_launch_profiles=host_launch_profiles,
            host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
            runtime_prepare_event_builder=runtime_prepare_event_builder,
            closed_loop_client_count=args.closed_loop_client_count)
        if args.enable_expert_parallel:
            validate_standalone_af_ep2_scope(simulation_config)
        af_mtp_preemption = (
            args.mtp_num_speculative_tokens != 0
            and args.max_requests == 2)
        if args.mtp_num_speculative_tokens != 0:
            validate_standalone_af_mtp_scope(
                simulation_config,
                allow_preemption=af_mtp_preemption,
            )
        with ExitStack() as construction_lifecycle:
            backend = make_backend(
                "attention", "attention", prefill_hardware,
                args.hardware_config,
                args.prefill_tp_size)
            construction_lifecycle.callback(backend.close)
            if args.backend == "vllm":
                runtime_info = backend.runtime_info
                scheduler_runtime["attention"] = (runtime_info if args.vllm_scheduler_version == "0.30.0" else asdict(runtime_info))
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            deployment = bind_symbolic_mtp_cache(deployment, runtime_inputs, kv_capacity_plans)
            simulation_config = replace(simulation_config, deployment=deployment)
            result = run_recorded_simulation(MultiSimulation(
                simulation_config,
                {"attention": backend},
                hardware_by_id,
                kv_cache_tier_manager=(
                    None
                    if args.kv_lifecycle_replay is None
                    else KVCacheTierManager(
                        args.kv_lifecycle_replay.cache_config)
                ),
                kv_lifecycle_replay=args.kv_lifecycle_replay,
                scheduler_runtime_sha256=(
                    None
                    if args.kv_lifecycle_replay is None
                    else vllm_scheduler_runtime_sha256(backend.runtime_info)
                ),
                enable_standalone_af_mtp=(
                    args.mtp_num_speculative_tokens != 0),
                enable_standalone_af_mtp_preemption=(
                    af_mtp_preemption),
                progress_callback=progress_callback,
            ), interactive_recording)
            construction_lifecycle.pop_all()
    elif args.deployment == "pd_af_disagg":
        if args.collective_backend == "protobridge":
            _activation_profile, activation_profile_sha256 = protobridge_p2p_profile(
                prefill_hardware.hardware_id, args.activation_transfer_link)
            _kv_profile, kv_profile_sha256 = protobridge_p2p_profile(
                prefill_hardware.hardware_id, args.transfer_link)
        elif args.collective_backend == "analytical_roofline_v1":
            activation_profile_sha256, kv_profile_sha256 = None, None
        else:
            raise NotImplementedError("PD+AF requires a point-to-point analytical or ProtoBridge service")
        if (not isinstance(args.decode_hardware_config, str)
                or not args.decode_hardware_config):
            raise ValueError(
                "--decode-hardware-config is required for pd_af_disagg")
        if (Path(args.decode_hardware_config).resolve()
                != Path(args.hardware_config).resolve()):
            raise NotImplementedError(
                "the current PD+AF composition requires every worker pool "
                "to use the same hardware config file")
        def pd_af_pool_names(phase, operator_group, count):
            if count == 1:
                return (f"{phase}-{operator_group}",)
            return tuple(
                f"{phase}-{operator_group}-{index}"
                for index in range(count)
            )

        prefill_attention_names = pd_af_pool_names(
            "prefill", "attention", args.num_prefill_instances)
        prefill_ffn_names = pd_af_pool_names(
            "prefill", "ffn", args.num_prefill_instances)
        decode_attention_names = pd_af_pool_names(
            "decode", "attention", args.num_decode_instances)
        decode_ffn_names = pd_af_pool_names(
            "decode", "ffn", args.num_decode_instances)
        pool_configs = []
        next_rank = 0
        for phase, attention_names, ffn_names in (
                ("prefill", prefill_attention_names, prefill_ffn_names),
                ("decode", decode_attention_names, decode_ffn_names)):
            for attention_name, ffn_name in zip(
                    attention_names, ffn_names, strict=True):
                pool_configs.extend((
                    WorkerPoolConfig(
                        attention_name,
                        f"{phase}-attention",
                        prefill_hardware.hardware_id,
                        ParallelConfig(tensor_parallel_size=1),
                        (next_rank,),
                    ),
                    WorkerPoolConfig(
                        ffn_name,
                        f"{phase}-ffn",
                        prefill_hardware.hardware_id,
                        ParallelConfig(tensor_parallel_size=1),
                        (next_rank + 1,),
                    ),
                ))
                next_rank += 2
        routes = []
        for attention_names, ffn_names in (
                (prefill_attention_names, prefill_ffn_names),
                (decode_attention_names, decode_ffn_names)):
            for attention_name, ffn_name in zip(
                    attention_names, ffn_names, strict=True):
                routes.extend((
                    RouteConfig(
                        attention_name, ffn_name, "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=p2p_resource_id(
                            args.activation_transfer_link,
                            attention_name,
                            ffn_name,
                        ),
                        backend=args.collective_backend,
                        backend_config_sha256=activation_profile_sha256,
                    ),
                    RouteConfig(
                        ffn_name, attention_name, "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=p2p_resource_id(
                            args.activation_transfer_link,
                            ffn_name,
                            attention_name,
                        ),
                        backend=args.collective_backend,
                        backend_config_sha256=activation_profile_sha256,
                    ),
                ))
        for prefill_name in prefill_attention_names:
            for decode_name in decode_attention_names:
                routes.append(RouteConfig(
                    prefill_name, decode_name, "kv",
                    args.transfer_algorithm,
                    args.transfer_link,
                    "point_to_point",
                    resource_id=p2p_resource_id(
                        args.transfer_link,
                        prefill_name,
                        decode_name,
                    ),
                    backend=args.collective_backend,
                    backend_config_sha256=kv_profile_sha256,
                ))
        deployment = build_pd_disagg_plan(
            DeploymentConfig(
                "pd_af_disagg",
                pool_configs,
                routes,
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
        }
        scheduler_pool_names = (
            prefill_attention_names + decode_attention_names)
        validate_capacity_pools(scheduler_pool_names)
        scheduler_roles = tuple(
            (name, "prefill-attention")
            for name in prefill_attention_names
        ) + tuple(
            (name, "decode-attention")
            for name in decode_attention_names
        )
        simulation_config = SimulationConfig(
            model, deployment, requests, args.launch_policy,
            args.prediction_method, runtime_inputs,
            execution_mode=args.execution_mode,
                        collect_source_state=args.state_audit_output is not None,
                        record_execution_timeline=interactive_recording is not None and not interactive_recording.compact_mode,
            timing_mode=args.serving_timing_mode,
            host_launch_profiles=host_launch_profiles,
            host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
            runtime_prepare_event_builder=runtime_prepare_event_builder,
            closed_loop_client_count=args.closed_loop_client_count)
        if args.mtp_num_speculative_tokens != 0:
            validate_pd_af_mtp_scope(
                simulation_config,
                fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            )
        with ExitStack() as construction_lifecycle:
            backends = {}
            for pool_name, role in scheduler_roles:
                backend = make_backend(
                    pool_name, role, prefill_hardware,
                    args.hardware_config, 1)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            if args.backend == "vllm":
                scheduler_runtime.update({
                    pool_name: (backend.runtime_info if args.vllm_scheduler_version == "0.30.0" else asdict(backend.runtime_info))
                    for pool_name, backend in backends.items()
                })
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            deployment = bind_symbolic_mtp_cache(deployment, runtime_inputs, kv_capacity_plans)
            simulation_config = replace(simulation_config, deployment=deployment)
            result = run_recorded_simulation(MultiSimulation(
                simulation_config,
                backends,
                hardware_by_id,
                enable_pd_af_mtp=(
                    args.mtp_num_speculative_tokens != 0),
                pd_af_mtp_fixed_accepted_tokens=(
                    args.mtp_fixed_accepted_tokens
                    if args.mtp_num_speculative_tokens != 0 else None
                ),
                progress_callback=progress_callback,
                streaming_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume
                ),
                streaming_step_projector=(
                    None
                    if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                    else (interactive_recording.project_step if interactive_recording is not None and interactive_recording.compact_mode else
                        streaming_artifact.execution_contract_digest)
                ),
                streaming_prehashed_step_consumer=(
                    None
                    if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                    else streaming_artifact.consume_prehashed
                ),
            ), interactive_recording)
            construction_lifecycle.pop_all()
    elif args.deployment == "pd_disagg":
        if args.kv_transfer_backend == "protobridge":
            _kv_profile, kv_profile_sha256 = protobridge_p2p_profile(
                prefill_hardware.hardware_id, args.transfer_link)
        elif args.kv_transfer_backend == "analytical_roofline_v1":
            kv_profile_sha256 = None
        else:
            raise ValueError("unknown KV transfer backend")
        if (not isinstance(args.decode_hardware_config, str)
                or not args.decode_hardware_config):
            raise ValueError(
                "--decode-hardware-config is required for pd_disagg")
        decode_hardware = HWSpec(args.decode_hardware_config)
        prefill_hardware_path = Path(args.hardware_config).resolve()
        decode_hardware_path = Path(args.decode_hardware_config).resolve()
        if (prefill_hardware.hardware_id == decode_hardware.hardware_id
                and prefill_hardware_path != decode_hardware_path):
            raise ValueError(
                "pd_disagg cannot bind different hardware config files with "
                "the same hardware_id: "
                f"{prefill_hardware.hardware_id!r} maps to "
                f"{prefill_hardware_path} and {decode_hardware_path}; reuse "
                "one config path or assign distinct canonical hardware IDs")
        prefill_names = tuple(
            f"prefill-{index}"
            for index in range(args.num_prefill_instances)
        )
        decode_names = tuple(
            f"decode-{index}"
            for index in range(args.num_decode_instances)
        )
        pool_configs = []
        next_rank = 0
        for pool_name in prefill_names:
            ranks = tuple(range(
                next_rank,
                next_rank + args.prefill_tp_size * args.prefill_pp_size * args.prefill_dp_size * args.prefill_cp_size,
            ))
            next_rank += args.prefill_tp_size * args.prefill_pp_size * args.prefill_dp_size * args.prefill_cp_size
            pool_configs.append(WorkerPoolConfig(
                pool_name,
                "prefill",
                prefill_hardware.hardware_id,
                ParallelConfig(
                    tensor_parallel_size=args.prefill_tp_size,
                    pipeline_parallel_size=args.prefill_pp_size,
                    data_parallel_size=args.prefill_dp_size,
                    context_parallel_size=args.prefill_cp_size,
                    sequence_parallel_size=(args.prefill_tp_size if model.num_experts and args.prefill_expert_parallel and args.prefill_dp_size > 1 else 1),
                    enable_expert_parallel=args.prefill_expert_parallel,
                ),
                ranks,
                collectives=tensor_collective_configs(
                    pool_name,
                    max(args.prefill_tp_size, args.prefill_cp_size),
                    prefill_hardware.hardware_id,
                    args.collective_backend,
                    args.tensor_all_reduce_profile_sha256,
                    args.tensor_all_gather_profile_sha256,
                ) + moe_data_collective_configs(pool_name, prefill_hardware.hardware_id, args.prefill_dp_size * args.prefill_cp_size,
                    args.collective_backend, model=model),
                kv_transfer_layout=args.vllm_cache_layout,
                pipeline_layer_partition=args.prefill_pp_layer_partition,
                pipeline_p2p=pipeline_p2p_configs(pool_name, args.prefill_pp_size,
                    hardware_id=prefill_hardware.hardware_id, backend=args.collective_backend,
                    tensor_parallel_size=args.prefill_tp_size, ranks_per_node=args.pipeline_ranks_per_node),
            ))
        for pool_name in decode_names:
            ranks = tuple(range(
                next_rank,
                next_rank + args.decode_tp_size * args.decode_pp_size * args.decode_dp_size * args.decode_cp_size,
            ))
            next_rank += args.decode_tp_size * args.decode_pp_size * args.decode_dp_size * args.decode_cp_size
            pool_configs.append(WorkerPoolConfig(
                pool_name,
                "decode",
                decode_hardware.hardware_id,
                ParallelConfig(
                    tensor_parallel_size=args.decode_tp_size,
                    pipeline_parallel_size=args.decode_pp_size,
                    data_parallel_size=args.decode_dp_size,
                    context_parallel_size=args.decode_cp_size,
                    sequence_parallel_size=(args.decode_tp_size if model.num_experts and args.decode_expert_parallel and args.decode_dp_size > 1 else 1),
                    enable_expert_parallel=args.decode_expert_parallel,
                ),
                ranks,
                collectives=tensor_collective_configs(
                    pool_name,
                    max(args.decode_tp_size, args.decode_cp_size),
                    decode_hardware.hardware_id,
                    args.collective_backend,
                    args.tensor_all_reduce_profile_sha256,
                    args.tensor_all_gather_profile_sha256,
                ) + moe_data_collective_configs(pool_name, decode_hardware.hardware_id, args.decode_dp_size * args.decode_cp_size,
                    args.collective_backend, model=model),
                kv_transfer_layout=args.vllm_cache_layout,
                pipeline_layer_partition=args.decode_pp_layer_partition,
                pipeline_p2p=pipeline_p2p_configs(pool_name, args.decode_pp_size,
                    hardware_id=decode_hardware.hardware_id, backend=args.collective_backend,
                    tensor_parallel_size=args.decode_tp_size, ranks_per_node=args.pipeline_ranks_per_node),
            ))
        routes = tuple(
            RouteConfig(
                source,
                destination,
                "kv",
                args.transfer_algorithm,
                args.transfer_link,
                "point_to_point",
                resource_id=p2p_resource_id(
                    args.transfer_link,
                    source,
                    destination,
                ),
                backend=args.kv_transfer_backend,
                backend_config_sha256=kv_profile_sha256,
            )
            for source in prefill_names
            for destination in decode_names
        )
        deployment = build_pd_disagg_plan(
            DeploymentConfig(
                "pd_disagg",
                tuple(pool_configs),
                routes,
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
            decode_hardware.hardware_id: decode_hardware,
        }
        prefill_names = tuple(pool.name for pool in deployment.worker_pools.values() if pool.role == "prefill")
        decode_names = tuple(pool.name for pool in deployment.worker_pools.values() if pool.role == "decode")
        validate_capacity_pools(prefill_names + decode_names)
        backends = {}
        with ExitStack() as construction_lifecycle:
            for pool_name in prefill_names:
                backend = make_backend(
                    pool_name, "prefill", prefill_hardware,
                    args.hardware_config,
                    args.prefill_tp_size, pp_size=args.prefill_pp_size)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            for pool_name in decode_names:
                backend = make_backend(
                    pool_name, "decode", decode_hardware,
                    args.decode_hardware_config,
                    args.decode_tp_size, pp_size=args.decode_pp_size)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            if args.backend == "vllm":
                scheduler_runtime.update({
                    pool_name: (backend.runtime_info if args.vllm_scheduler_version == "0.30.0"
                        else asdict(backend.runtime_info))
                    for pool_name, backend in backends.items()
                })
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            deployment = bind_symbolic_mtp_cache(deployment, runtime_inputs, kv_capacity_plans)
            result = run_recorded_simulation(MultiSimulation(
                SimulationConfig(
                    model, deployment, requests, args.launch_policy,
                    args.prediction_method, runtime_inputs,
                    authorized_h200_pd_mtp_sweep=(
                        args.glm_mtp_native_source_inputs or is_glm_h200_pd_mtp_target_scope(args, True)
                        or is_dsv3_h200_pd_mtp_target_scope(args)),
                    execution_mode=args.execution_mode,
                        collect_source_state=args.state_audit_output is not None,
                        record_execution_timeline=interactive_recording is not None and not interactive_recording.compact_mode,
                    step_retention=(
                        "streaming_aggregate"
                        if use_streaming_aggregate else "full"),
                    timing_mode=args.serving_timing_mode,
                    host_launch_profiles=host_launch_profiles,
                    host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
                    runtime_prepare_event_builder=runtime_prepare_event_builder,
                    closed_loop_client_count=args.closed_loop_client_count),
                backends,
                hardware_by_id,
                progress_callback=progress_callback,
                streaming_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume
                ),
                streaming_step_projector=(
                    None
                    if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                    else (interactive_recording.project_step if interactive_recording is not None and interactive_recording.compact_mode else
                        streaming_artifact.execution_contract_digest)
                ),
                streaming_prehashed_step_consumer=(
                    None
                    if streaming_artifact is None or (interactive_recording is not None and not interactive_recording.compact_mode)
                    else streaming_artifact.consume_prehashed
                ),
            ), interactive_recording)
            construction_lifecycle.pop_all()
    else:
        raise ValueError(f"unknown deployment mode: {args.deployment!r}")
    simulation_wall_finished = time.perf_counter()
    simulation_wall_time_s = simulation_wall_finished - simulation_wall_started
    if (request_source is not None
            and request_source["kind"] == "production_trace"):
        request_source["prefix_alignment"] = prefix_alignment_manifest(
            requests,
            result,
            args.kv_block_size_tokens,
            prefix_cache_mode=args.prefix_cache_mode,
        )
    if (args.backend == "vllm"
            and args.deployment in ("pd_disagg", "pd_af_disagg")):
        kv_handoff_mode = "metadata_only_private_destination_kv"
    elif args.deployment in ("pd_disagg", "pd_af_disagg"):
        kv_handoff_mode = (
            "modeled_transfer_without_framework_block_ownership")
    elif args.deployment in ("colocated", "af_disagg"):
        kv_handoff_mode = "not_applicable"
    else:
        raise ValueError(
            f"unsupported serving deployment: {args.deployment!r}")
    slo_goodput = slo_goodput_manifest(
        requests,
        result.metrics,
        result.schedule_trace.end_time_s,
        slo_contract,
    )
    # Reported on the engine clock, the same normalization the observed
    # Scheduler trace carries, so that a predicted and a measured wait cover
    # the same boundary.
    first_schedule_start_s = engine_clock_first_schedule_start_s(
        result.schedule_trace, requests)
    expected_request_ids = {
        request.request_id for request in requests
    }
    if set(first_schedule_start_s) != expected_request_ids:
        raise ValueError(
            "schedule trace does not provide a first schedule boundary for "
            "every request"
        )
    timing_status = performance_timing_status(
        args.launch_policy,
        tuple(sorted(hardware_by_id)),
        uses_development_protobridge=uses_development_protobridge(args),
    )
    execution_provenance = build_execution_provenance(
        args,
        runtime_inputs,
        hardware_by_id,
        timing_status,
        repository_at_start=repository_at_start,
    )
    if isinstance(result, StreamingSimulationResult):
        if streaming_artifact is None:
            raise RuntimeError(
                "streaming simulation lost its artifact accumulator")
        execution_artifact = streaming_artifact.manifest(
            result,
            performance_timing_status=timing_status,
        )
        streaming_chain_field = (
            "ordered_step_execution_contract_chain_sha256"
            if execution_artifact["schema_version"] in (10, 11, 12, 13, 14, 16)
            else "ordered_step_manifest_chain_sha256"
        )
        analysis_status = {
            "schema_version": 1,
            "availability": "not_materialized_streaming_aggregate",
            "reason": (
                "full per-step IR is consumed once and released to keep "
                "external-request simulation memory bounded"),
            "ordered_step_manifest_chain_sha256": (
                execution_artifact["streaming_aggregate"][
                    streaming_chain_field]),
        }
        system_resources = dict(analysis_status)
        critical_path = dict(analysis_status)
    else:
        if streaming_artifact is not None:
            raise RuntimeError(
                "full simulation unexpectedly retained a streaming "
                "artifact accumulator")
        # Closed-loop runs stamp each successor at its client's completion
        # instant; downstream per-request analysis must see those injected
        # arrival times, not the canonical stream.
        analysis_requests = (
            requests if result.metrics_requests is None
            else result.metrics_requests)
        system_resources = system_resource_manifest(
            deployment,
            analysis_requests,
            result,
        )
        critical_path = critical_path_manifest(
            deployment,
            analysis_requests,
            result,
        )
        execution_artifact = execution_artifact_manifest(
            result,
            performance_timing_status=timing_status,
        )
    if args.state_audit_output is not None:
        if streaming_artifact is None:
            raise ValueError("state audit output requires streaming aggregate retention")
        state_audit_path = Path(args.state_audit_output)
        if args.output is not None and state_audit_path.resolve() == Path(args.output).resolve():
            raise ValueError("state audit output must differ from the simulation result path")
        state_audit_path.parent.mkdir(parents=True, exist_ok=True)
        state_audit_path.write_text(json.dumps(
            streaming_artifact.state_audit_manifest(result),
            indent=2, sort_keys=True) + "\n")
    mtp_pd_handoff = mtp_pd_handoff_contract(
        args,
        requests,
        result,
        deployment,
        scheduler_runtime,
    )
    payload = {
        "version": result_version,
        "backend": args.backend,
        "deployment_id": deployment.deployment_id,
        "hardware_ids": sorted(hardware_by_id),
        "model_compute_dtype": model.compute_dtype,
        "model_config_provenance": {
            "config_sha256": model.config_sha256,
            "source_config_sha256": model.source_config_sha256,
            "canonical_config_sha256": model.canonical_config_sha256,
            "canonicalization": model.config_canonicalization,
            "extraction_path": model.config_extraction_path,
            "extracted_config_sha256": model.extracted_config_sha256,
            "model_id": model.model_id,
            "revision": model.model_revision,
            "source_url": model.model_source_url,
            "scope": model.model_scope,
            "architecture": model.architecture,
            "model_type": model.model_type,
            "attention_layout": model.attention_layout,
            "compute_dtype": model.compute_dtype,
            "weight_dtype": model.weight_dtype,
            "weight_group_size": model.weight_group_size,
            "quantized_weight_targets": model.quantized_weight_targets,
            "quantized_weight_ignore": model.quantized_weight_ignore,
        },
        "deployment": args.deployment,
        "deployment_plan": deployment_plan_manifest(deployment),
        "routing_policy": "least_queued_tokens",
        "route_decision_version": 2,
        "transfer_trace_version": 2,
        "kv_handoff_mode": kv_handoff_mode,
        "pd_kv_lifecycle": result.pd_kv_lifecycle,
        "kv_capacity": {**kv_capacity_manifest(kv_capacity_plans.values(), runtime_memory_policy,
            models_by_pool=None if runtime_memory_policy is None else {name: model for name in kv_capacity_plans}),
            "source_activation_reconciliation": source_activation_capacity_manifest(
                result.source_activation_memory, kv_capacity_plans, runtime_memory_policy, deployment, model, hardware_by_id,
                source_workspace_budget=(None if not k3_model or runtime_memory_policy is None
                    else kimi_k3_source_workspace_bytes(model,
                        deployment.worker_pools['main'].parallel_plan.config,
                        max_batched_tokens=args.max_batched_tokens, max_requests=args.max_requests,
                        source_multiprocessors=prefill_hardware.num_compute_units,
                        pipeline_layer_partition=tuple(deployment.worker_pools['main'].parallel_plan.layer_to_pipeline_stage.count(stage)
                            for stage in range(deployment.worker_pools['main'].parallel_plan.config.pipeline_parallel_size))))),
            **({"source_lifetime_projection": source_capacity_reports} if source_capacity_reports else {})},
        "kv_lifecycle_replay": (
            None
            if args.kv_lifecycle_replay is None
            else args.kv_lifecycle_replay.manifest()
        ),
        "launch_policy": args.launch_policy,
        "prediction_method": args.prediction_method,
        "execution_mode": args.execution_mode,
        "serving_timing_mode": args.serving_timing_mode,
        "host_timing": host_timing_manifest(
            result, host_launch_profiles, host_runtime_prepare_profiles),
        "execution_cache_stats": result.execution_cache_stats,
        "execution_parallelism": {
            "worker_count": result.execution_performance_stats[
                "parallel_worker_count"],
            "parallel_step_count": result.execution_performance_stats[
                "parallel_step_count"],
            "parallel_execution_wall_time_s": (
                result.execution_performance_stats[
                    "parallel_execution_wall_time_s"]),
            "worker_aggregate_phase_s": (
                result.execution_performance_stats[
                    "parallel_worker_aggregate_phase_s"]),
        },
        "performance_timing_status": timing_status,
        "execution_provenance": execution_provenance,
        "workload_assumption_binding": (
            workload_assumption_binding_manifest(
                actual_request_stream_sha256,
                runtime_inputs,
                args,
            )
        ),
        "main_chain_status": {
            "status": "executed",
            "chain": (
                "RequestStream -> SchedulerBackend/BackendStep/ScheduleStep "
                "-> build_symbolic_workload -> WorkloadGraph -> lower_workload -> KernelSpec -> predict"
            ),
            "execution_artifact_sha256": execution_artifact["artifact_sha256"],
        },
        "execution_artifact": execution_artifact,
        "scheduler_runtime": scheduler_runtime,
        "mtp_scenario": mtp_scenario_manifest(VLLMSchedulerConfig(
            args.model,
            scheduler_version=args.vllm_scheduler_version,
            tensor_parallel_size=args.prefill_tp_size,
            pipeline_parallel_size=args.prefill_pp_size,
            context_parallel_size=args.prefill_cp_size,
            data_parallel_size=(args.prefill_dp_size if model.model_type == "glm_moe_dsa" else 1),
            pipeline_layer_partition=args.prefill_pp_layer_partition,
            enable_chunked_prefill=args.enable_chunked_prefill,
            enable_prefix_caching=args.enable_prefix_caching,
            prefix_cache_mode=args.prefix_cache_mode,
            async_scheduling=args.vllm_async_scheduling,
            mtp_draft_token_ids_by_step=args.mtp_draft_token_ids_by_step,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
            mtp_acceptance_policy=args.mtp_acceptance_policy,
            mtp_fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            mtp_acceptance_source_id=args.mtp_acceptance_source_id,
            mtp_acceptance_replay=args.mtp_acceptance_replay,
            mtp_acceptance_assumption=args.mtp_acceptance_assumption,
            request_stream_sha256=(
                actual_request_stream_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            ),
            decode_admission_contract_sha256=(
                mtp_decode_admission_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            ),
        )),
        "mtp_acceptance_outcomes": mtp_acceptance_outcomes_manifest(
            args, result, deployment),
        "mtp_pd_handoff_contract": mtp_pd_handoff,
        "arrival_process": arrival_process,
        "request_arrivals_s": [
            {
                "request_id": request.request_id,
                "arrival_time_s": request.arrival_time_s,
            }
            for request in requests
        ],
        "workload_hash": workload_hash,
        "num_steps": (
            result.num_steps
            if isinstance(result, StreamingSimulationResult)
            else len(result.steps)
        ),
        "num_transfers": len(result.schedule_trace.transfers),
        **({"source_activation_memory": result.source_activation_memory}
           if result.source_activation_memory is not None else {}),
        "num_kernel_calls": (
            result.num_kernel_calls
            if isinstance(result, StreamingSimulationResult)
            else (
                sum(
                    step.kernel_plan.logical_call_count
                    for step in result.steps
                )
                + sum(
                    transfer.kernel_plan.logical_call_count
                    for transfer in result.transfers
                )
            )
        ),
        "schedule_end_time_s": result.schedule_trace.end_time_s,
        "simulation_wall_time_s": simulation_wall_time_s,
        "simulation_started_monotonic_s": simulation_wall_started,
        "simulation_finished_monotonic_s": simulation_wall_finished,
        "evaluation_wall_time_breakdown_s": {
            **result.execution_performance_stats["phase_wall_time_s"],
            "artifact_generation": (
                0.0
                if streaming_artifact is None
                else streaming_artifact.consume_wall_time_s
            ),
            "scheduler_and_orchestration": max(
                0.0,
                simulation_wall_time_s
                - sum(result.execution_performance_stats[
                    "phase_wall_time_s"].values())
                - (
                    0.0
                    if streaming_artifact is None
                    else streaming_artifact.consume_wall_time_s
                ),
            ),
        },
        "process_peak_rss_bytes": max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        ) * 1024,
        "scheduler_progress": scheduler_progress_manifest(
            requests, result.schedule_trace),
        "route_decisions": [
            asdict(decision)
            for decision in result.schedule_trace.route_decisions
        ],
        "transfers": [
            asdict(transfer)
            for transfer in result.schedule_trace.transfers
        ],
        "metrics": {
            "request_throughput_per_s": result.metrics.request_throughput_per_s,
            "output_token_throughput_per_s": (
                result.metrics.output_token_throughput_per_s),
            "token_throughput_per_s": result.metrics.token_throughput_per_s,
            "latency_distributions": latency_distribution_manifest(
                result.metrics.requests),
            "slo_goodput": slo_goodput,
            "system_resources": system_resources,
            "critical_path": critical_path,
            "requests": {
                request_id: {
                    "ttft_s": metric.ttft_s,
                    "tpot_s": metric.tpot_s,
                    "latency_s": metric.latency_s,
                    "first_schedule_start_s": first_schedule_start_s[
                        request_id
                    ],
                }
                for request_id, metric in result.metrics.requests.items()
            },
        },
    }
    if request_source is not None:
        payload["request_source"] = request_source
    progress_callback.stop()
    report_finalize_started = time.perf_counter()
    if interactive_recording is not None:
        from evaluation.reporting.serving_interactive import write_interactive_report
        interactive_recording.hardware.update(hardware_by_id)
        if not isinstance(result, StreamingSimulationResult):
            entries = {entry.backend_step.schedule.step_id: entry for entry in result.schedule_trace.entries}
            for step in result.steps:
                entry = entries[step.backend_step.schedule.step_id]
                if interactive_recording.compact_mode:
                    # execution_artifact_manifest above already validated the
                    # full step contract; avoid a discarded streaming digest.
                    interactive_recording._record_after_artifact_validation(step, entry)
                else:
                    interactive_recording.record(step, entry)
        visual_payload = interactive_recording.visualization_payload(payload, result)
        visual_payload["visualization_configuration_files"] = configuration_files
        visual_payload["visualization_configuration_snapshot_missing"] = configuration_missing
        visual_payload["visualization_cost_receipt_path"] = Path(interactive_report_path).with_suffix(".report.json").name
        if args.interactive_view == "swimlanes":
            from evaluation.reporting.serving_swimlanes import write_swimlane_report
            report = write_swimlane_report(visual_payload, interactive_report_path)
        elif args.interactive_view == "breakdown":
            report = write_interactive_report(visual_payload, interactive_report_path)
        else:
            raise ValueError("unknown interactive report view")
        payload["interactive_report"] = {**report, **visual_payload["visualization_coverage"]}
    if not args.interactive_report:
        payload["interactive_report"] = {"status": "disabled", "reason": "offline HTML report not requested"}
    if adaptive_tail_worker_grants is not None:
        payload["adaptive_tail_worker_grants"] = adaptive_tail_worker_grants
    payload["report_output_costs"] = {"recording_projection_s": (interactive_recording.recording_wall_s
        if interactive_recording is not None else 0.0),
        "output_pipeline_wall_s": time.perf_counter() - report_total_started,
        "scope": "wall clock includes simulation and report finalization; recording is nested and may be parallel; do not add phase times"}
    if interactive_recording is not None:
        payload["interactive_report"]["cost_receipt_path"] = str(Path(interactive_report_path).with_suffix(".report.json"))
    result_write_started = time.perf_counter()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        persisted_payload = persistence_projection(payload)
        path.write_text(
            json.dumps(persisted_payload, indent=2, sort_keys=True) + "\n")
    payload["report_output_costs"].update({
        "result_json_projection_write_s": time.perf_counter() - result_write_started,
        "finalization_wall_s": time.perf_counter() - report_finalize_started,
        "total_wall_s": time.perf_counter() - report_total_started,
        "simulation_wall_s": simulation_wall_time_s,
        "scope": "total wall includes setup, simulation, recording, finalization and persisted result write; nested/parallel phases are not additive"})
    if interactive_recording is not None:
        receipt_path = Path(interactive_report_path).with_suffix(".report.json")
        if args.interactive_view == "swimlanes":
            from evaluation.reporting.serving_viewer_ui import attach_report_costs
            costs = payload["report_output_costs"]
            costs["cost_metadata_write_s"] = attach_report_costs(payload["interactive_report"], {
                "html_generation_s": report["page_generation_s"],
                "total_wall_through_result_write_s": costs["total_wall_s"],
                "simulation_wall_s": simulation_wall_time_s,
                "recording_projection_s": costs["recording_projection_s"],
                "finalization_wall_s": costs["finalization_wall_s"],
                "result_json_projection_write_s": costs["result_json_projection_write_s"],
                "scope": "producer wall through result JSON write; recording nested/parallel; cost metadata and final receipt writes measured separately"})
            costs["total_wall_s"] = time.perf_counter() - report_total_started
    receipt_path = (Path(interactive_report_path).with_suffix(".report.json") if interactive_recording is not None
        else Path(args.output).with_suffix(".costs.json") if args.output else None)
    payload["report_output_costs"]["cost_receipt_path"] = str(receipt_path) if receipt_path is not None else None
    payload["report_output_costs"]["persisted_boundary"] = "result includes costs through initial result write and optional metadata attachment; sidecar includes final refresh; returned total includes sidecar write"
    final_result_write_started = time.perf_counter()
    if args.output:
        path.write_text(json.dumps(persistence_projection(payload), indent=2, sort_keys=True) + "\n")
    payload["report_output_costs"]["final_result_refresh_s"] = time.perf_counter() - final_result_write_started
    payload["report_output_costs"]["total_wall_s"] = time.perf_counter() - report_total_started
    receipt_started = time.perf_counter()
    if receipt_path is not None:
        receipt_path.write_text(json.dumps({"report": payload["interactive_report"],
        "costs": payload["report_output_costs"]}, indent=2) + "\n")
    payload["report_output_costs"]["cost_receipt_write_s"] = time.perf_counter() - receipt_started
    payload["report_output_costs"]["total_wall_s"] = time.perf_counter() - report_total_started
    if interactive_recording is not None:
        payload["interactive_report"]["cost_receipt_path"] = str(receipt_path)
    return payload



_ConformanceBackend = ConformanceBackend


def bind_symbolic_mtp_cache(deployment, runtime_inputs, capacity_plans):
    """Pass resolved Scheduler block geometry to the virtual proposer only.

    This does not start a lifecycle manager or establish normal capacity.
    """
    if runtime_inputs.mtp_draft_sampling_source not in (
            "vllm_0_11_eager_full_logits_greedy", "vllm_0_30_legacy_full_logits_greedy"):
        return deployment
    pools = dict(deployment.worker_pools)
    for name, capacity in capacity_plans.items():
        existing = pools[name].kv_cache
        if existing is not None:
            hbm = tuple(tier for tier in existing.tiers if tier.name.lower() == "hbm")
            if (len(hbm) != 1 or existing.bytes_per_token != capacity.kv_bytes_per_token
                    or hbm[0].capacity_bytes != capacity.num_kv_blocks * capacity.kv_block_bytes
                    or hbm[0].block_size_tokens != capacity.kv_block_size_tokens):
                raise ValueError("virtual proposer cache must match the resolved Scheduler HBM geometry")
            # Host tiers and their lifecycle owner must survive source binding.
            continue
        cache = KVCacheConfig((KVCacheTierConfig("hbm",
            capacity.num_kv_blocks * capacity.kv_block_bytes,
            capacity.kv_block_size_tokens),), capacity.kv_bytes_per_token)
        pools[name] = replace(pools[name], kv_cache=cache)
    return replace(deployment, worker_pools=pools)


def validate_activation_source_policy_binding(args, policy, hardware, model, pool_name, role, parallelism):
    """Source projection takes limits from policy only after exact CLI agreement."""
    binding = policy.binding
    actual = (binding.hardware_id, binding.model_config_sha256, policy.pool_name, binding.deployment,
        binding.worker_role, binding.tensor_parallel_size, binding.pipeline_parallel_size,
        binding.expert_parallel_size, binding.data_parallel_size, binding.context_parallel_size, binding.pipeline_layer_partition,
        binding.max_model_len, binding.max_batched_tokens,
        binding.max_requests, binding.kv_block_size_tokens)
    expected = (hardware.hardware_id, model.config_sha256, pool_name, args.deployment, role,
        parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size, parallelism.expert_parallel_size,
        parallelism.data_parallel_size, parallelism.context_parallel_size,
        (args.decode_pp_layer_partition if role == "decode" else args.prefill_pp_layer_partition),
        args.max_model_len, args.max_batched_tokens, args.max_requests, args.kv_block_size_tokens)
    if actual != expected:
        raise ValueError("activation source policy binding must match the CLI identities, parallelism and constructor limits")
