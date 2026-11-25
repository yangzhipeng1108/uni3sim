"""Bounded CPU 0.30 replay of Production Trace or an explicit capped-output scenario."""

from __future__ import annotations

from evaluation.serving.simulation.interactive_recording import run_recorded_simulation

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from contextlib import contextmanager

from src import _platform_compat as resource
from evaluation.serving.simulation.execution_artifact import (
    StreamingExecutionArtifactAccumulator, performance_timing_status, _json_value,
)
from evaluation.serving.simulation.interactive_recording import CompactInteractiveRecording
from evaluation.reporting.serving_viewer_ui import _configuration_metadata, configuration_file_snapshots
from evaluation.serving.simulation.simulation_artifacts import repository_provenance
from evaluation.serving.simulation.state_audit import ServingStateAuditAccumulator
from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan, deployment_plan_manifest
from src.serving.kv_cache.kv_cache_config import KVCacheConfig, KVCacheTierConfig
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.production_trace import (
    SuccessfulRequestFilter, load_production_trace,
    production_trace_debug_schema_config, select_production_trace_window,
)
from src.serving.request_gen.request_stream import RequestStream
from src.serving.simulation import Simulation, SimulationConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


_PROFILES = {
    "glm-5.2": "glm_5_2_debug_csv_v1",
    "kimi-k2.6": "kimi_k2_6_debug_csv_v1",
}



class _ProductionStateAuditAccumulator(ServingStateAuditAccumulator):
    """Validate actual per-step source/rank coverage before ordinary accumulation."""

    def __init__(self, pool_name, ranks):
        super().__init__()
        self._expected_pool = pool_name
        self._expected_ranks = frozenset(str(rank) for rank in ranks)

    def consume_snapshot(self, snapshot):
        summary = snapshot["source_state_summary"]
        if (snapshot["pool_name"] != self._expected_pool or summary is None
                or set(summary["ranks"]) != self._expected_ranks):
            raise ValueError("Production state audit requires source ranks for every step")
        call_ranks = {(call[1], call[2]) for call in snapshot["calls"]}
        expected = {(self._expected_pool, rank) for rank in self._expected_ranks}
        # Complete source storage does not require a call on every rank.
        if not call_ranks <= expected:
            raise ValueError("Production state audit has an illegal call pool or rank")
        super().consume_snapshot(snapshot)


def _production_state_audit(enabled, result, aggregate, *, pool_name, ranks,
        step_count, kernel_call_count):
    if not enabled:
        return None
    if aggregate is None:
        audit = _ProductionStateAuditAccumulator(pool_name, ranks)
        for step in result.steps:
            audit.consume(step)
        payload = audit.manifest(result)
    else:
        if type(aggregate._state_audit) is not _ProductionStateAuditAccumulator:
            raise RuntimeError("Production streaming state audit requires its source-rank validator")
        payload = aggregate.state_audit_manifest(result)
    observed = {(rank["pool_name"], rank["rank"]) for rank in payload["ranks"]}
    expected = {(pool_name, rank) for rank in ranks}
    calls = sum(count for rank in payload["ranks"]
        for count in rank["logical_calls_by_kind"].values())
    if (not payload["all_steps_have_source_state"] or observed != expected
            or len(payload["ranks"]) != len(expected)
            or payload["step_count"] != step_count or calls != kernel_call_count):
        raise RuntimeError("Production state audit lost source state, steps, ranks or logical calls")
    return payload


def _record_retained_report_steps(recorder, result):
    """Production full retention has no prior artifact; use the checked path."""
    entries = {entry.backend_step.schedule.step_id: entry for entry in result.schedule_trace.entries}
    for step in result.steps:
        recorder.record(step, entries[step.backend_step.schedule.step_id])


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _worker_peak_rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text()
    rows = [row for row in status.splitlines() if row.startswith("VmHWM:")]
    if len(rows) != 1:
        raise ValueError("Scheduler worker status lacks one VmHWM field")
    parts = rows[0].split()
    if len(parts) != 3 or parts[2] != "kB":
        raise ValueError("Scheduler worker VmHWM has unexpected units")
    return int(parts[1]) * 1024


@contextmanager
def _wall_deadline(seconds: float):
    def expired(_signum, _frame):
        raise TimeoutError("Production request exceeded explicit wall limit")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _consume_production_prehashed_step(aggregate, progress, backend_step, entry, projected,
                                      count, compact, state_audit_snapshot=None):
    """Bind audit evidence to this callback before any accumulation or progress."""
    if aggregate is None:
        raise RuntimeError("prehashed step received without an accumulator")
    value, materialized, repeats = projected
    if state_audit_snapshot is not None:
        from collections.abc import Mapping
        if not isinstance(state_audit_snapshot, Mapping):
            raise TypeError("Production state audit snapshot must be a mapping")
        schedule = backend_step.schedule
        if (state_audit_snapshot["step_id"] != schedule.step_id
                or state_audit_snapshot["pool_name"] != schedule.pool_name):
            raise ValueError("Production state audit snapshot step/pool differs from callback")
        calls = state_audit_snapshot["calls"]
        if not isinstance(calls, tuple):
            raise TypeError("Production state audit calls must be a tuple")
        total = 0
        for call in calls:
            if not isinstance(call, tuple) or len(call) != 7:
                raise TypeError("Production state audit call has an invalid contract")
            multiplicity = call[4]
            if type(multiplicity) is not int or multiplicity <= 0:
                raise ValueError("Production state audit call multiplicity must be positive")
            total += multiplicity
        if total != count:
            raise ValueError("Production state audit per-step call count differs from callback")
    aggregate.consume_prehashed(backend_step, entry, value, count, compact, state_audit_snapshot)
    progress["materialized_kernel_plan_calls"] += materialized
    progress["logical_kernel_calls"] += count
    progress["block_repeat_groups"] += repeats


def _scheduler_report_configuration(backend, scheduler_config=None):
    """Expose the byte-budget bridge's already validated native constraints."""
    runtime = dict(backend.runtime_info)
    if runtime["version"] != "0.30.0":
        raise ValueError("fixed-tier report requires the 0.30 byte-budget bridge")
    if scheduler_config is None:
        # Existing callers may supply the complete GLM source receipt explicitly via runtime.
        source = runtime["source_scheduler_config"]
    else:
        if not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError("report requested configuration must be VLLMSchedulerConfig")
        source = asdict(scheduler_config)
        if "source_scheduler_config" in runtime and runtime["source_scheduler_config"] != source:
            raise ValueError("recorded source scheduler configuration differs from requested object")
        for actual, expected, name in (
                (runtime["effective_max_num_batched_tokens"], backend.config.max_batched_tokens, "batch tokens"),
                (runtime["effective_cache_block_size"], backend.config.kv_block_size_tokens, "cache block"),
                (runtime["native_allocation"]["max_model_len"], scheduler_config.max_model_len, "model length"),
                (runtime["effective_chunked_prefill"], scheduler_config.enable_chunked_prefill, "chunked prefill"),
                (runtime["effective_tensor_parallel_size"], scheduler_config.tensor_parallel_size, "tensor parallelism"),
                (runtime["effective_pipeline_parallel_size"], scheduler_config.pipeline_parallel_size, "pipeline parallelism"),
                (scheduler_config.async_scheduling, False, "synchronous bridge"),
                (scheduler_config.scheduler_version, "0.30.0", "requested version")):
            if actual != expected:
                raise ValueError("report effective " + name + " differs from requested/native configuration")
    return {**runtime,
        "max_num_batched_tokens": runtime["effective_max_num_batched_tokens"],
        "max_num_seqs": backend.config.max_requests,
        "max_model_len": source["max_model_len"],
        "kv_block_size_tokens": runtime["effective_cache_block_size"],
        **{"requested_" + key: source[field] for key, field in (
            ("prefix_caching", "enable_prefix_caching"), ("chunked_prefill", "enable_chunked_prefill"),
            ("async_scheduling", "async_scheduling"))},
        "effective_prefix_caching": source["enable_prefix_caching"],
        "effective_async_scheduling": False,
        "report_configuration_evidence": "0.30 byte-budget bridge validates native max_num_seqs/model length/cache block/prefix caching/chunked prefill/async against these declared constraints before simulation",
        "report_requested_scheduler_config": _configuration_metadata(source),
        "report_backend_config": _configuration_metadata(backend.config)}


def run_request(args: argparse.Namespace, *, report=False) -> dict[str, object]:
    total_started = time.perf_counter()
    preparation_started = total_started
    if type(report) is not bool:
        raise TypeError("report must be boolean")
    request_start = args.request_index
    request_end = args.request_end
    if request_start is None:
        if request_end is not None:
            raise ValueError("request end requires request index")
    else:
        if type(request_start) is not int or request_start < 0:
            raise ValueError("request index must be non-negative")
        if request_end is None:
            request_end = request_start + 1
        if type(request_end) is not int or request_end <= request_start:
            raise ValueError("request end must exceed request index")
    if request_start is None and args.arrival_start_s is None:
        raise ValueError("Production window requires a request or arrival range")
    if type(args.max_requests) is not int or args.max_requests < 1:
        raise ValueError("max concurrent requests must be positive")
    if type(args.hbm_blocks) is not int or args.hbm_blocks < 1:
        raise ValueError("HBM blocks must be positive")
    if type(args.host_blocks) is not int or args.host_blocks < 1:
        raise ValueError("Host blocks must be positive")
    if (type(args.max_wall_s) not in (int, float)
            or not math.isfinite(args.max_wall_s) or args.max_wall_s <= 0):
        raise ValueError("max wall seconds must be finite and positive")
    if (args.synthetic_output_cap is not None
            and (type(args.synthetic_output_cap) is not int
                 or args.synthetic_output_cap < 1)):
        raise ValueError("synthetic output cap must be a positive integer")
    source = args.data_root / args.model / args.case / "requests.csv"
    canonical = json.loads((source.parent / "manifest.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != canonical["requests"]["requests_sha256"]:
        raise ValueError("Production CSV disagrees with canonical manifest hash")
    trace = load_production_trace(source,
        production_trace_debug_schema_config(_PROFILES[args.model],
            successful_request_filter=SuccessfulRequestFilter(
                "upstream_attested", None, None,
                "canonical successful-request export: filter_production_trace.py v2.3.0")))
    if len(trace.requests) != canonical["requests"]["row_count"]:
        raise ValueError("Production CSV row count disagrees with canonical manifest")
    source_requests = select_production_trace_window(trace,
        request_start=request_start, request_end=request_end,
        arrival_start_s=args.arrival_start_s,
        arrival_end_s=args.arrival_end_s)
    canonical_selected = source_requests.requests
    selected = (canonical_selected if args.synthetic_output_cap is None else
        tuple(replace(request,
            output_tokens=min(request.output_tokens, args.synthetic_output_cap))
            for request in canonical_selected))
    arrival_origin_s = selected[0].arrival_time_s
    requests = RequestStream(replace(request,
        arrival_time_s=request.arrival_time_s - arrival_origin_s)
        for request in selected)
    model_path = f"configs/model/{args.model}.json"
    model = load_model_config(model_path)
    configuration_files, configuration_missing = configuration_file_snapshots(model_path, ["configs/hardware/nvidia_h200.yaml"])
    glm = model.model_type == "glm_moe_dsa"
    block_tokens = 16
    block_bytes = block_tokens * model.kv_bytes_per_token_per_parallel_rank(8, 1)
    cache = KVCacheConfig((
        KVCacheTierConfig("hbm", args.hbm_blocks * block_bytes, block_tokens),
        KVCacheTierConfig("host", args.host_blocks * block_bytes,
            block_tokens, "pcie")), block_bytes // block_tokens, "lru")
    collectives = tuple(CollectiveConfig(op, "ring", "nvlink", "ring",
        "source-fabric", "analytical_roofline_v1")
        for op in ("all_reduce", "all_gather"))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (
        WorkerPoolConfig("main", "colocated", "nvidia_h200",
            ParallelConfig(tensor_parallel_size=8), kv_cache=cache,
            collectives=collectives),)), num_layers=model.num_hidden_layers,
        num_experts=model.num_experts)
    model_len = max(request.input_tokens + request.output_tokens
        for request in selected) + block_tokens
    if model_len > (args.hbm_blocks - 1) * block_tokens:
        raise ValueError("declared HBM KV capacity cannot cover the longest complete request after vLLM reserves its null block")
    runtime = RuntimeInputs(ExpertRoutingInput.seeded_assumption(
        "declared-production-routing-unavailable", 0), "mla", "uniform",
        model_execution_version="vllm_0_30_0",
        target_sampling_source="vllm_0_30_plain_greedy_bf16",
        symbolic_forward_source=(
            f"vllm_0_30_v32_{model.weight_dtype}_sync_explicit_allreduce"
            if glm else "vllm_0_30_k2_6_int4_sync_explicit_allreduce"),
        symbolic_source_max_model_len=model_len,
        symbolic_source_max_batched_tokens=model_len,
        symbolic_source_prompt_lengths_by_request={request.request_id:
            request.input_tokens for request in selected},
        symbolic_workspace_initial_capacity_bytes=(0,) * 8 if glm else None,
        glm_indexer_selection_source="declared_causal_topk" if glm else None,
        glm_prefill_attention_source="force_sparse_mqa" if glm else None)
    config = SimulationConfig(model, deployment, requests,
        "analytical_launch", "roofline", runtime,
        step_retention=args.step_retention, execution_mode=args.execution_mode,
        collect_source_state=args.state_audit)
    manager = KVCacheTierManager(cache,
        fixed_prefix_hbm_percent=args.hbm_percent)
    recorder = CompactInteractiveRecording({"nvidia_h200": HWSpec("configs/hardware/nvidia_h200.yaml")}) if report else None
    aggregate = ((recorder if recorder is not None else StreamingExecutionArtifactAccumulator())
        if args.step_retention == "streaming_aggregate" else None)
    if args.state_audit and aggregate is not None:
        aggregate._state_audit = _ProductionStateAuditAccumulator(
            "main", deployment.worker_pools["main"].ranks)
    started = time.perf_counter()
    progress = {"steps": 0, "emitted_output_tokens": 0,
        "completed_requests": 0, "virtual_time_s": 0.0,
        "parent_peak_rss_bytes": 0, "scheduler_worker_peak_rss_bytes": 0,
        "h2d_placed_bytes_per_rank": 0}
    if aggregate is not None:
        progress.update({"materialized_kernel_plan_calls": 0,
            "logical_kernel_calls": 0, "block_repeat_groups": 0})
    completed_ids: set[str] = set()
    base = {
        "source_kind": ("canonical_production_trace_read_only"
            if args.synthetic_output_cap is None
            else "synthetic_output_capped_from_canonical_production_trace"),
        "source_path": str(source), "source_sha256": digest,
        "source_row_start": request_start,
        "source_row_end": request_end,
        "arrival_start_s": args.arrival_start_s,
        "arrival_end_s": args.arrival_end_s,
        "simulation_arrival_origin_s": arrival_origin_s,
        "requests": [asdict(request) for request in selected],
        "model_path": model_path,
        "model_config_sha256": model.config_sha256,
        "hbm_percent": args.hbm_percent,
        "host_percent": 100 - args.hbm_percent,
        "block_tokens": block_tokens,
        "block_bytes_per_rank": block_bytes,
        "hbm_capacity_bytes_per_rank": args.hbm_blocks * block_bytes,
        "host_capacity_bytes_per_rank": args.host_blocks * block_bytes,
        "max_wall_s": args.max_wall_s,
        "max_concurrent_requests": args.max_requests,
        "step_retention": args.step_retention,
        "execution_mode": args.execution_mode,
        "state_audit": None,
        "scenario_assumptions": [
            "hit blocks prewarmed at declared layers; request-private identities",
            "seeded uniform expert routing; production routing unavailable",
            "H200 PCIe roofline bandwidth lower bound; startup unavailable",
            "CPU Scheduler and symbolic Forward, no GPU model execution",
        ],
    }
    if args.synthetic_output_cap is not None:
        base["synthetic_output_cap"] = args.synthetic_output_cap
        base["canonical_requests"] = [asdict(request)
            for request in canonical_selected]
        base["scenario_assumptions"].append(
            "output lengths capped for a synthetic functional check; canonical requests retained separately")

    def save_failure(status, reason):
        failed = {**base, "status": status, "progress": progress.copy(),
            "wall_s": time.perf_counter() - started, "error": reason}
        if report:
            from evaluation.reporting.serving_interactive import write_status_report
            failed["interactive_report"] = write_status_report({
                "main_chain_status": {"status": status, "reason": reason},
                "request_source": {"run_configuration": base},
                "model_config_provenance": {"model_id": model.model_id, "architecture": model.architecture},
                "visualization_configuration_files": configuration_files,
                "visualization_configuration_snapshot_missing": configuration_missing}, args.output.with_suffix(".html"))
        else:
            failed["interactive_report"] = {"status": "disabled", "reason": "offline HTML report not requested"}
        _write(args.output, failed)

    def consume_streaming_step(step, entry):
        if aggregate is None:
            raise RuntimeError("streaming step received without an accumulator")
        progress["materialized_kernel_plan_calls"] += len(step.kernel_plan.calls)
        progress["logical_kernel_calls"] += step.kernel_plan.logical_call_count
        progress["block_repeat_groups"] += len(step.kernel_plan.block_repeats)
        aggregate.consume(step, entry)

    def project_streaming_step(step):
        projected = (recorder.project_step(step) if recorder is not None else aggregate.execution_contract_digest(step))
        return (projected, len(step.kernel_plan.calls), len(step.kernel_plan.block_repeats))

    def consume_prehashed_step(backend_step, entry, projected, count, compact, state_audit_snapshot=None):
        _consume_production_prehashed_step(aggregate, progress, backend_step, entry,
            projected, count, compact, state_audit_snapshot)

    def on_step_complete(step, virtual_time_s):
        progress["steps"] += 1
        progress["emitted_output_tokens"] += sum(step.generated_tokens.values())
        progress["h2d_placed_bytes_per_rank"] += sum(
            manager.fixed_prefix_block_counts(
                scheduled.scheduler_prefix_hit_tokens)[1] * block_bytes
            for scheduled in step.schedule.requests
            if scheduled.trace_prefix_hit_tokens)
        completed_ids.update(step.schedule.finished_request_ids)
        progress["completed_requests"] = len(completed_ids)
        progress["virtual_time_s"] = virtual_time_s
        progress["parent_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024
        progress["scheduler_worker_peak_rss_bytes"] = _worker_peak_rss_bytes(
            backend.runtime_info["worker_pid"])
        elapsed = time.perf_counter() - started
        if progress["steps"] % 16 == 0:
            _write(args.output, {**base, "status": "running",
                "progress": progress.copy(), "wall_s": elapsed})
        if elapsed >= args.max_wall_s:
            _write(args.output, {**base, "status": "checkpoint_wall_limit",
                "progress": progress.copy(), "wall_s": elapsed})
            raise TimeoutError("Production request exceeded explicit wall limit")

    try:
        with _wall_deadline(args.max_wall_s):
            scheduler_config = VLLMSchedulerConfig(model_path, max_model_len=model_len,
                tensor_parallel_size=8, enable_chunked_prefill=False, scheduler_version="0.30.0")
            backend = VLLMBackend.from_vllm030_byte_budget(
                scheduler_config,
                python_executable=args.sdk_python, source_root=args.sdk_source,
                pool_name="main", max_batched_tokens=model_len,
                max_requests=args.max_requests, kv_block_size_tokens=block_tokens,
                available_memory_by_rank=(args.hbm_blocks * block_bytes,) * 8,
                supported_layouts_by_rank=(("LBHNC",),) * 8)
            try:
                scheduler_report_configuration = _scheduler_report_configuration(backend, scheduler_config)
            except BaseException as primary_error:
                try:
                    backend.close()
                except BaseException as cleanup_error:
                    raise primary_error from cleanup_error
                raise
            backend_ready = time.perf_counter()
            result = run_recorded_simulation(Simulation(config, backend,
                {"nvidia_h200": HWSpec("configs/hardware/nvidia_h200.yaml")},
                kv_cache_tier_manager=manager,
                streaming_step_consumer=(consume_streaming_step
                    if aggregate is not None else None),
                streaming_step_projector=project_streaming_step if aggregate is not None else None,
                streaming_prehashed_step_consumer=consume_prehashed_step if aggregate is not None else None,
                progress_callback=on_step_complete), recorder)
            simulation_completed = time.perf_counter()
    except TimeoutError:
        save_failure("checkpoint_wall_limit", "Production simulation exceeded its explicit CPU wall deadline")
        raise
    except Exception as error:
        save_failure("unsupported" if isinstance(error, NotImplementedError) else "failed", type(error).__name__ + ": Production simulation did not complete; see the process log")
        raise
    if set(result.metrics.requests) != {request.request_id for request in selected}:
        raise RuntimeError("Production window did not complete every request exactly once")
    hits = [scheduled for entry in result.schedule_trace.entries
        for scheduled in entry.backend_step.schedule.requests
        if scheduled.trace_prefix_hit_tokens]
    expected_hit_ids = {request.request_id for request in selected
        if request.prefix_hit_tokens}
    if ({scheduled.request_id for scheduled in hits} != expected_hit_ids
            or len(hits) != len(expected_hit_ids)):
        raise RuntimeError("Production window lost first-schedule hit accounting")
    prefix_blocks = {scheduled.request_id: manager.fixed_prefix_block_counts(
        scheduled.scheduler_prefix_hit_tokens) for scheduled in hits}
    transfers = result.kv_offload.transfers
    h2d_bytes = sum(record.event.bytes for record in transfers
        if record.event.action is KVCacheAction.LOAD)
    if (h2d_bytes != sum(blocks[1] for blocks in prefix_blocks.values()) * block_bytes
            or h2d_bytes != progress["h2d_placed_bytes_per_rank"]):
        raise RuntimeError("fixed prefix Host placement disagrees with completed loads")
    h2d_service_s_by_rank = {rank: 0.0 for rank in range(8)}
    h2d_intervals_by_step = {}
    for record in transfers:
        if record.event.action is not KVCacheAction.LOAD:
            continue
        if record.step_id not in h2d_intervals_by_step:
            h2d_intervals_by_step[record.step_id] = []
        for physical in record.completion.physical_transfers:
            h2d_service_s_by_rank[physical.rank] += (
                physical.end_time_s - physical.start_time_s)
            h2d_intervals_by_step[record.step_id].append((
                physical.start_time_s, physical.end_time_s))
    h2d_gate_span_s = sum(
        max(end for _, end in intervals)
            - min(start for start, _ in intervals)
        for intervals in h2d_intervals_by_step.values())
    if aggregate is None:
        step_count = len(result.steps)
        kernel_call_count = sum(step.kernel_plan.logical_call_count
            for step in result.steps)
        materialized_call_count = sum(len(step.kernel_plan.calls)
            for step in result.steps)
        block_repeat_group_count = sum(len(step.kernel_plan.block_repeats)
            for step in result.steps)
        execution_artifact = None
    else:
        step_count = result.num_steps
        kernel_call_count = result.num_kernel_calls
        if progress["logical_kernel_calls"] != kernel_call_count:
            raise RuntimeError("streamed logical kernel call count disagrees with progress")
        materialized_call_count = progress["materialized_kernel_plan_calls"]
        block_repeat_group_count = progress["block_repeat_groups"]
        execution_artifact = aggregate.manifest(result,
            performance_timing_status=performance_timing_status(
                "analytical_launch", {"nvidia_h200"}))
    audit_payload = _production_state_audit(args.state_audit, result, aggregate,
        pool_name="main", ranks=deployment.worker_pools["main"].ranks,
        step_count=step_count, kernel_call_count=kernel_call_count)
    payload = {**base, "status": "complete",
        "progress": progress.copy(), "wall_s": time.perf_counter() - started,
        "completed_requests": len(result.metrics.requests),
        "num_steps": step_count,
        "num_kernel_calls": kernel_call_count,
        "materialized_kernel_plan_calls": materialized_call_count,
        "block_repeat_groups": block_repeat_group_count,
        "prefix_hit_accounting": [
            {"request_id": request.request_id,
             "observed_hit_tokens": request.prefix_hit_tokens,
             "effective_hit_tokens": next((scheduled.scheduler_prefix_hit_tokens
                for scheduled in hits if scheduled.request_id == request.request_id), 0),
             "initial_hbm_blocks": prefix_blocks[request.request_id][0]
                if request.request_id in prefix_blocks else 0,
             "initial_host_blocks": prefix_blocks[request.request_id][1]
                if request.request_id in prefix_blocks else 0}
            for request in selected],
        "h2d_bytes_per_rank": h2d_bytes,
        "h2d_service_s_by_rank": h2d_service_s_by_rank,
        "h2d_gate_span_s": h2d_gate_span_s,
        "d2h_bytes_per_rank": sum(record.event.bytes for record in transfers
            if record.event.action is KVCacheAction.EVICT),
        "transfer_count": len(transfers),
        "execution_cache_stats": result.execution_cache_stats,
        "execution_performance_stats": result.execution_performance_stats,
        "metrics": asdict(result.metrics),
        "final_tier_state": result.kv_offload.final_tier_state,
        "execution_artifact": execution_artifact,
        "state_audit": audit_payload,
    }
    payload["phase_wall_time_s"] = {
        "input_model_configuration": started - preparation_started,
        "backend_initialization": backend_ready - started,
        "simulation_including_construction_consumers_and_cleanup": simulation_completed - backend_ready,
        "postprocess_artifact": time.perf_counter() - simulation_completed,
        "streaming_artifact_consume_nested_in_simulation": (
            0.0 if aggregate is None else aggregate.consume_wall_time_s),
    }
    report_started = time.perf_counter()
    if recorder is not None:
        from evaluation.reporting.serving_swimlanes import write_swimlane_report
        if aggregate is None:
            _record_retained_report_steps(recorder, result)
        report_path = args.output.with_suffix(".html")
        cost_path = args.output.with_suffix(".report.json")
        visual = {"metrics": asdict(result.metrics), "simulation_wall_time_s": payload["wall_s"],
            "prediction_method": config.prediction_method, "launch_policy": config.launch_policy,
            "performance_timing_status": performance_timing_status(config.launch_policy, {"nvidia_h200"}),
            "schedule_end_time_s": result.schedule_trace.end_time_s,
            "model_compute_dtype": model.compute_dtype,
            "model_config_provenance": {"architecture": model.architecture, "model_id": model.model_id,
                "config_sha256": model.config_sha256, "weight_dtype": model.weight_dtype},
            "deployment": "colocated", "deployment_plan": deployment_plan_manifest(deployment),
            "scheduler_runtime": {"main": scheduler_report_configuration},
            "request_source": {"input_path": str(source), "source_sha256": digest, "row_count": len(selected)},
            "execution_provenance": {"repository": repository_provenance(),
                "runtime_inputs": _configuration_metadata(runtime)},
            "kv_capacity": {"cache": _configuration_metadata(cache), "fixed_tier": base},
            "main_chain_status": {"status": "complete", "chain": "Production fixed-tier CPU Scheduler and virtual Forward"},
            "visualization_configuration_files": configuration_files,
            "visualization_configuration_snapshot_missing": configuration_missing,
            "visualization_cost_receipt_path": cost_path.name}
        visual = recorder.visualization_payload(visual, result)
        payload["interactive_report"] = write_swimlane_report(visual, report_path)
        payload["interactive_report"]["cost_receipt_path"] = str(cost_path)
    else:
        payload["interactive_report"] = {"status": "disabled", "reason": "offline HTML report not requested"}
    report_generation_s = time.perf_counter() - report_started
    # This initial write and the later cost-metadata refresh are distinct measured phases.
    initial_write_started = time.perf_counter()
    _write(args.output, payload)
    initial_result_write_s = time.perf_counter() - initial_write_started
    payload["report_output_costs"] = {"total_wall_s": time.perf_counter() - total_started,
        "simulation_wall_s": payload["phase_wall_time_s"]["simulation_including_construction_consumers_and_cleanup"],
        "report_generation_s": report_generation_s,
        "initial_result_write_s": initial_result_write_s,
        "recording_projection_s": recorder.recording_wall_s if recorder is not None else 0.0,
        "scope": "total includes setup, simulation, recording, report compression/HTML/JSON and persisted result write; nested phases are not additive"}
    if recorder is not None:
        from evaluation.reporting.serving_viewer_ui import attach_report_costs
        costs = payload["report_output_costs"]
        costs["cost_metadata_write_s"] = attach_report_costs(payload["interactive_report"], {
            "html_generation_s": payload["interactive_report"]["page_generation_s"],
            "total_wall_through_result_write_s": costs["total_wall_s"],
            "simulation_wall_s": costs["simulation_wall_s"],
            "recording_projection_s": costs["recording_projection_s"],
            "scope": "producer wall through result JSON write; recording nested; metadata and final receipt writes are separate"})
        costs["total_wall_s"] = time.perf_counter() - total_started
    cost_path = args.output.with_suffix(".report.json" if recorder is not None else ".costs.json")
    payload["report_output_costs"]["cost_receipt_path"] = str(cost_path)
    payload["report_output_costs"]["persisted_boundary"] = "result includes costs through initial result write and optional metadata attachment; sidecar includes final refresh; returned total includes sidecar write"
    final_result_write_started = time.perf_counter()
    _write(args.output, payload)
    costs = payload["report_output_costs"]
    costs["final_result_refresh_s"] = time.perf_counter() - final_result_write_started
    costs["total_wall_s"] = time.perf_counter() - total_started
    receipt_started = time.perf_counter()
    _write(cost_path, {"report": payload["interactive_report"], "costs": costs})
    costs["cost_receipt_write_s"] = time.perf_counter() - receipt_started
    costs["total_wall_s"] = time.perf_counter() - total_started
    print(json.dumps({"production_cli_timing": {
        **payload["phase_wall_time_s"],
        "report_projection_and_generation": report_generation_s,
        "final_output_write": initial_result_write_s + costs["final_result_refresh_s"] + costs["cost_receipt_write_s"],
        "cost_metadata_attachment": costs["cost_metadata_write_s"] if "cost_metadata_write_s" in costs else 0.0,
        "run_request_total": time.perf_counter() - preparation_started,
        "scope": "main phases and writes are sequential; streaming consumption/recording is nested inside simulation; external CLI also includes startup and this final timing print",
    }}), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--request-index", type=int)
    parser.add_argument("--request-end", type=int)
    parser.add_argument("--arrival-start-s", type=float)
    parser.add_argument("--arrival-end-s", type=float)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--hbm-percent", type=int, required=True)
    parser.add_argument("--hbm-blocks", type=int, required=True)
    parser.add_argument("--host-blocks", type=int, required=True)
    parser.add_argument("--max-wall-s", type=float, required=True)
    parser.add_argument("--synthetic-output-cap", type=int,
        help="Cap each output length for an explicitly synthetic functional check")
    parser.add_argument("--step-retention", choices=("full", "streaming_aggregate"),
        default="streaming_aggregate")
    parser.add_argument("--execution-mode", choices=("accelerated", "fast"), default="accelerated",
        help="Run-local equivalent source/lowering/prediction reuse; never a saved prediction LUT")
    parser.add_argument("--state-audit", action="store_true",
        help="Collect independent per-rank work and source-state evidence; count its cost separately")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sdk-python", type=Path, required=True)
    parser.add_argument("--sdk-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactive-report", action="store_true",
        help="write the offline HTML report plus its visualization projection")
    args = parser.parse_args()
    run_request(args, report=args.interactive_report)


if __name__ == "__main__":
    main()
