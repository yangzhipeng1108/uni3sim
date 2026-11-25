"""Scheduler trace summarization for ShareGPT ground-truth captures:
runtime depth and clock metadata, A800 rank observation, and
post-benchmark compilation counters."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import time
from typing import Any, Sequence

from evaluation.serving.experiments.ground_truth_case_contract import (
    sha256_file,
)
from evaluation.serving.experiments.ground_truth_server_lifecycle import (
    _signal_process_group,
)


_A800_COMPILATION_COUNTER_FIELDS = (
    "num_models_seen",
    "num_graphs_seen",
    "num_piecewise_graphs_seen",
    "num_piecewise_capturable_graphs_seen",
    "num_backend_compilations",
    "num_gpu_runner_capture_triggers",
    "num_cudagraph_captured",
    "num_inductor_compiles",
    "num_eager_compiles",
    "num_cache_entries_updated",
    "num_compiled_artifacts_saved",
    "dynamo_as_is_count",
)


def _scheduler_runtime_depth_metadata(trace_path: Path) -> dict[str, Any]:
    """Load the trace sidecar written from the live Executor object."""

    path = Path(f"{trace_path}.runtime.json")
    if not path.is_file():
        raise RuntimeError(
            "Scheduler trace is missing live Executor depth metadata: "
            f"{path}")
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != {
            "schema_version", "scheduler_pipeline_depth",
            "scheduler_pipeline_depth_evidence"}:
        raise ValueError("Scheduler runtime depth metadata fields are invalid")
    if metadata["schema_version"] != 1:
        raise ValueError("Scheduler runtime depth metadata schema is invalid")
    depth = metadata["scheduler_pipeline_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("Scheduler runtime depth must be a positive integer")
    evidence = metadata["scheduler_pipeline_depth_evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("Scheduler runtime depth evidence fields are invalid")
    source_pair = (evidence["source_kind"], evidence["source_field"])
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        expected_fields = {
            "source_kind", "source_field", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        expected_fields = {
            "source_kind", "source_field", "configured_depth", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
        if evidence["configured_depth"] != depth:
            raise ValueError(
                "Scheduler runtime depth evidence configured depth is invalid")
    else:
        raise ValueError("Scheduler runtime depth evidence source is invalid")
    if set(evidence) != expected_fields:
        raise ValueError("Scheduler runtime depth evidence fields are invalid")
    if any(not isinstance(evidence[field], str) or not evidence[field]
           for field in ("executor_class", "runtime_version")):
        raise ValueError("Scheduler runtime depth evidence text is invalid")
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        if (not isinstance(evidence["source_identity"], str)
                or not evidence["source_identity"]):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        source_identity = evidence["source_identity"]
        if (not isinstance(source_identity, dict)
                or set(source_identity) != {"module_file", "module_name"}):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
        if any(not isinstance(source_identity[field], str)
               or not source_identity[field]
               for field in ("module_file", "module_name")):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
    pipeline_parallel_size = evidence["pipeline_parallel_size"]
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1):
        raise ValueError("Scheduler runtime depth evidence PP is invalid")
    return metadata


def _scheduler_clock_metadata(trace_path: Path) -> dict[str, Any]:
    """Load the explicit host-monotonic origin used by Scheduler rows."""

    path = Path(f"{trace_path}.clock.json")
    if not path.is_file():
        raise RuntimeError(
            f"Scheduler trace is missing clock metadata: {path}"
        )
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "clock_domain",
        "host_boot_id",
        "trace_origin_monotonic_ns",
    }:
        raise ValueError("Scheduler clock metadata fields are invalid")
    if metadata["schema_version"] != 1:
        raise ValueError("Scheduler clock metadata schema is invalid")
    if metadata["clock_domain"] != "shared_host_time.perf_counter_ns":
        raise ValueError("Scheduler clock domain is invalid")
    if (
        not isinstance(metadata["host_boot_id"], str)
        or not metadata["host_boot_id"]
    ):
        raise ValueError("Scheduler host boot ID is invalid")
    origin = metadata["trace_origin_monotonic_ns"]
    if isinstance(origin, bool) or not isinstance(origin, int) or origin < 0:
        raise ValueError("Scheduler trace origin must be a non-negative integer")
    return metadata


def _scheduler_trace_summary(
    path: str | Path,
    expected_request_count: int,
    *,
    max_result_consumption_gap_ns: int | None = None,
) -> dict[str, Any]:
    if max_result_consumption_gap_ns is not None and (
        isinstance(max_result_consumption_gap_ns, bool)
        or not isinstance(max_result_consumption_gap_ns, int)
        or max_result_consumption_gap_ns <= 0
    ):
        raise ValueError(
            "Scheduler result-consumption gap limit must be a positive integer"
        )
    trace_path = Path(path)
    if not trace_path.is_file():
        raise RuntimeError(
            f"Scheduler trace was not produced by the accepted run: {trace_path}"
        )
    depth_metadata = _scheduler_runtime_depth_metadata(trace_path)
    clock_metadata = _scheduler_clock_metadata(trace_path)
    configured_depth = depth_metadata["scheduler_pipeline_depth"]
    records = []
    for line_index, line in enumerate(trace_path.read_text().splitlines()):
        if not line:
            raise ValueError("Scheduler trace contains an empty JSONL row")
        value = json.loads(line)
        if value["step_index"] != line_index:
            raise ValueError("Scheduler trace step indexes are not contiguous")
        records.append(value)
    if not records:
        raise ValueError("Scheduler trace is empty")
    if records[-1]["cumulative_completed_requests"] != expected_request_count:
        raise ValueError("Scheduler trace did not complete every request")
    summary = {
        "path": str(trace_path.resolve()),
        "sha256": sha256_file(trace_path),
        "row_count": len(records),
        "completed_requests": records[-1]["cumulative_completed_requests"],
        "scheduled_tokens": records[-1]["cumulative_scheduled_tokens"],
        "runtime_metadata_path": str(
            Path(f"{trace_path}.runtime.json").resolve()),
        "runtime_metadata_sha256": sha256_file(
            Path(f"{trace_path}.runtime.json")),
        "scheduler_pipeline_depth": configured_depth,
        "scheduler_pipeline_depth_evidence": depth_metadata[
            "scheduler_pipeline_depth_evidence"],
        "clock_metadata_path": str(
            Path(f"{trace_path}.clock.json").resolve()),
        "clock_metadata_sha256": sha256_file(
            Path(f"{trace_path}.clock.json")),
        "clock": clock_metadata,
    }
    if "rank_observations" in records[0]:
        return _a800_scheduler_trace_summary(
            records, summary, configured_depth)
    if "configured_cudagraph_mode" not in records[0]:
        return summary
    graph_mode_counts = {"FULL": 0, "PIECEWISE": 0, "NONE": 0}
    fallback_counts: dict[str, int] = {}
    max_in_flight = 0
    for record in records:
        scheduled_tokens = record["scheduled_tokens"]
        request_tokens = sum(
            request["scheduled_tokens"]
            for request in record["scheduled_requests"]
        )
        if scheduled_tokens != request_tokens:
            raise ValueError("Scheduler row request tokens are inconsistent")
        if scheduled_tokens != (
            record["prefill_tokens"] + record["decode_tokens"]
        ):
            raise ValueError("Scheduler row prefill/decode tokens are inconsistent")
        if scheduled_tokens != record["num_unpadded_tokens"]:
            raise ValueError("Scheduler row unpadded tokens are inconsistent")
        if record["padded_total_tokens"] != (
            scheduled_tokens + record["padding_tokens"]
        ):
            raise ValueError("Scheduler row padding tokens are inconsistent")
        runtime_mode = record["runtime_cudagraph_mode"]
        if runtime_mode not in graph_mode_counts:
            raise ValueError("Scheduler row has an unsupported runtime Graph mode")
        graph_mode_counts[runtime_mode] += 1
        fallback_reason = record["graph_fallback_reason"]
        if runtime_mode == "NONE":
            if record["capture_bucket_tokens"] is not None:
                raise ValueError("NONE Scheduler row must not have a capture bucket")
            if not isinstance(fallback_reason, str) or not fallback_reason:
                raise ValueError("NONE Scheduler row must have a fallback reason")
            fallback_counts[fallback_reason] = (
                fallback_counts[fallback_reason] + 1
                if fallback_reason in fallback_counts
                else 1
            )
        elif runtime_mode in ("FULL", "PIECEWISE"):
            if record["capture_bucket_tokens"] != record["padded_total_tokens"]:
                raise ValueError("Graph Scheduler row capture bucket is inconsistent")
            if fallback_reason is not None:
                raise ValueError("Graph Scheduler row must not have a fallback reason")
        else:
            raise ValueError(f"unsupported Scheduler runtime Graph mode: {runtime_mode!r}")
        if not (
            record["schedule_start_ns"] <= record["schedule_end_ns"]
            <= record["device_execution_start_ns"]
            <= record["device_execution_end_ns"]
            <= record["scheduler_update_start_ns"]
            <= record["scheduler_update_end_ns"]
        ):
            raise ValueError("Scheduler row stage timestamps are not causal")
        if (
            max_result_consumption_gap_ns is not None
            and record["scheduler_update_start_ns"]
            - record["device_execution_end_ns"]
            > max_result_consumption_gap_ns
        ):
            raise RuntimeError(
                "Scheduler device result was not consumed within the accepted "
                "host-side gap"
            )
        in_flight = record["in_flight_step_count"]
        if isinstance(in_flight, bool) or not isinstance(in_flight, int):
            raise TypeError("Scheduler row in-flight count must be an integer")
        if not 1 <= in_flight <= configured_depth:
            raise ValueError(
                "Scheduler row in-flight count exceeds configured depth")
        max_in_flight = max(max_in_flight, in_flight)
    warmup_path = Path(f"{trace_path}.warmup.json")
    if not warmup_path.is_file():
        raise ValueError("Scheduler runtime warmup evidence is missing")
    warmup = json.loads(warmup_path.read_text())
    if warmup["max_scheduled_request_count"] < 8:
        raise ValueError("warmup did not schedule at least eight requests together")
    if warmup["mixed_prefill_decode_step_count"] < 1:
        raise ValueError("warmup did not exercise mixed prefill/decode")
    summary.update({
        "runtime_cudagraph_mode_counts": graph_mode_counts,
        "graph_fallback_counts": fallback_counts,
        "max_in_flight_step_count": max_in_flight,
        "warmup_path": str(warmup_path.resolve()),
        "warmup_sha256": sha256_file(warmup_path),
        "warmup": warmup,
    })
    return summary


_VLLM_011_MULTIPROC_EXECUTOR_CLASS = (
    "vllm.v1.executor.multiproc_executor.MultiprocExecutor"
)


def _worker_dispatch_reply_rank(
    evidence: dict[str, Any], world_size: int
) -> int:
    """Return the only worker rank whose reply the executor RPC envelope awaits.

    vLLM 0.11.0 ``v1/executor/multiproc_executor.py`` (sha256 ``ad938d34...
    e0cd16``, the file recorded by A800 and H200 0.11.0 captures):
    ``execute_model`` without a KV connector calls ``collective_rpc(...,
    unique_reply_rank=self.output_rank)`` (L179-186); ``collective_rpc``
    dequeues only ``self.workers[unique_reply_rank]`` (L237-238); the worker
    busy loop replies only when ``self.rank == output_rank`` (L674-678); and
    ``_get_output_rank`` returns ``world_size - tensor_parallel_size``
    (L335-345) with ``world_size == TP * PP`` (L62-67).  The ground-truth
    runner never passes ``--kv-transfer-config``, so the all-rank branch
    (``has_connector``) is unreachable here.  A non-reply rank's
    ``Worker.execute_model`` has no happens-before edge to the executor
    completion, so only the reply rank's worker envelope is contained.
    """

    if world_size == 1:
        # One worker: whichever executor runs it must wait for its reply.
        return 0
    if evidence["executor_class"] != _VLLM_011_MULTIPROC_EXECUTOR_CLASS:
        raise ValueError(
            "multi-rank worker-dispatch trace uses an unaudited executor reply "
            f"contract: {evidence['executor_class']!r}"
        )
    if evidence["runtime_version"] != "0.11.0":
        raise ValueError(
            "multi-rank worker-dispatch trace uses an unaudited vLLM release: "
            f"{evidence['runtime_version']!r}"
        )
    pipeline_parallel_size = evidence["pipeline_parallel_size"]
    if world_size % pipeline_parallel_size:
        raise ValueError(
            "worker rank count is not a multiple of pipeline parallel size"
        )
    return world_size - world_size // pipeline_parallel_size


def _a800_scheduler_trace_summary(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    configured_depth: int,
) -> dict[str, Any]:
    graph_mode_counts = {"FULL": 0, "PIECEWISE": 0, "NONE": 0}
    fallback_counts: dict[str, int] = {}
    max_in_flight = 0
    expected_ranks: set[int] | None = None
    first_observations = records[0]["rank_observations"]
    if not isinstance(first_observations, list) or not first_observations:
        raise TypeError("A800 Scheduler row rank observations are invalid")
    # Every row is checked below to cover exactly ranks 0..world_size-1.
    reply_rank = _worker_dispatch_reply_rank(
        summary["scheduler_pipeline_depth_evidence"], len(first_observations)
    )
    late_non_reply_ranks: list[dict[str, Any]] = []
    for record in records:
        scheduled_tokens = record["scheduled_tokens"]
        request_tokens = sum(
            request["scheduled_tokens"]
            for request in record["scheduled_requests"]
        )
        if scheduled_tokens != request_tokens:
            raise ValueError("Scheduler row request tokens are inconsistent")
        if scheduled_tokens != (
            record["prefill_tokens"] + record["decode_tokens"]
        ):
            raise ValueError("Scheduler row prefill/decode tokens are inconsistent")
        if not (
            record["schedule_start_ns"] <= record["schedule_end_ns"]
            <= record["device_execution_start_ns"]
            <= record["device_execution_end_ns"]
            <= record["scheduler_update_start_ns"]
            <= record["scheduler_update_end_ns"]
        ):
            raise ValueError("Scheduler row stage timestamps are not causal")
        if record["device_execution_clock"] != "time.perf_counter_ns":
            raise ValueError("Scheduler row executor clock is invalid")
        if record["device_execution_source"] != (
            "host_monotonic_executor_rpc_envelope"
        ):
            raise ValueError("Scheduler row executor source is invalid")
        in_flight = record["in_flight_step_count"]
        if isinstance(in_flight, bool) or not isinstance(in_flight, int):
            raise TypeError("Scheduler row in-flight count must be an integer")
        if not 1 <= in_flight <= configured_depth:
            raise ValueError(
                "Scheduler row in-flight count exceeds configured depth")
        max_in_flight = max(max_in_flight, in_flight)
        observations = record["rank_observations"]
        if not isinstance(observations, list) or not observations:
            raise TypeError("A800 Scheduler row rank observations are invalid")
        ranks = set()
        for observation in observations:
            rank = observation["rank"]
            _validate_a800_rank_observation_for_summary(
                observation,
                expected_rank=rank,
                expected_step_id=record["step_id"],
                expected_scheduled_tokens=scheduled_tokens,
            )
            if rank in ranks:
                raise ValueError("A800 Scheduler row repeats a worker rank")
            ranks.add(rank)
            if not (
                record["device_execution_start_ns"]
                <= observation["worker_execution_start_ns"]
                <= observation["worker_execution_end_ns"]
            ):
                raise ValueError(
                    "A800 worker execution envelope is outside executor envelope"
                )
            end_after_executor_ns = (
                observation["worker_execution_end_ns"]
                - record["device_execution_end_ns"]
            )
            if end_after_executor_ns > 0:
                if rank == reply_rank:
                    raise ValueError(
                        "A800 worker execution envelope is outside executor "
                        "envelope"
                    )
                late_non_reply_ranks.append({
                    "step_id": record["step_id"],
                    "rank": rank,
                    "worker_end_after_executor_end_ns": end_after_executor_ns,
                })
            runtime_mode = observation["runtime_cudagraph_mode"]
            graph_mode_counts[runtime_mode] += 1
            fallback_reason = observation["graph_fallback_reason"]
            if runtime_mode == "NONE":
                fallback_counts[fallback_reason] = (
                    fallback_counts[fallback_reason] + 1
                    if fallback_reason in fallback_counts
                    else 1
                )
        if expected_ranks is None:
            expected_ranks = ranks
        elif ranks != expected_ranks:
            raise ValueError("A800 Scheduler row worker-rank coverage changed")
    if expected_ranks is None or expected_ranks != set(range(len(expected_ranks))):
        raise ValueError("A800 Scheduler trace ranks must be contiguous from zero")
    trace_path = Path(summary["path"])
    warmup_path = Path(f"{trace_path}.warmup.json")
    if not warmup_path.is_file():
        raise ValueError("Scheduler runtime warmup evidence is missing")
    warmup = json.loads(warmup_path.read_text())
    if warmup["max_scheduled_request_count"] < 8:
        raise ValueError("warmup did not schedule at least eight requests together")
    if warmup["mixed_prefill_decode_step_count"] < 1:
        raise ValueError("warmup did not exercise mixed prefill/decode")
    summary.update({
        "runtime_cudagraph_mode_counts": graph_mode_counts,
        "graph_fallback_counts": fallback_counts,
        "max_in_flight_step_count": max_in_flight,
        "worker_ranks": sorted(expected_ranks),
        "worker_execution_envelope": {
            "executor_reply_rank": reply_rank,
            "non_reply_rank_end_after_executor_end_count": len(
                late_non_reply_ranks),
            "non_reply_rank_end_after_executor_end_max_ns": (
                max(
                    late["worker_end_after_executor_end_ns"]
                    for late in late_non_reply_ranks
                )
                if late_non_reply_ranks
                else None
            ),
            "non_reply_rank_end_after_executor_end": late_non_reply_ranks,
        },
        "warmup_path": str(warmup_path.resolve()),
        "warmup_sha256": sha256_file(warmup_path),
        "warmup": warmup,
    })
    return summary


def _validate_a800_rank_observation_for_summary(
    observation: Any,
    *,
    expected_rank: int,
    expected_step_id: str,
    expected_scheduled_tokens: int,
) -> None:
    if not isinstance(observation, dict):
        raise TypeError("A800 Scheduler rank observation must be an object")
    required = {
        "schema_version",
        "source",
        "rank",
        "trace_step_id",
        "scheduled_tokens",
        "configured_cudagraph_mode",
        "runtime_cudagraph_mode",
        "dispatch_input_key",
        "selected_graph_key",
        "graph_fallback_reason",
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
        "worker_execution_clock",
        "worker_execution_source",
    }
    if set(observation) != required:
        raise ValueError("A800 Scheduler rank observation fields are invalid")
    if observation["schema_version"] != 1:
        raise ValueError("A800 Scheduler rank observation schema is invalid")
    if observation["source"] != "vllm_v1_worker_cudagraph_dispatch":
        raise ValueError("A800 Scheduler rank observation source is invalid")
    if observation["rank"] != expected_rank:
        raise ValueError("A800 Scheduler rank observation rank is invalid")
    if observation["trace_step_id"] != expected_step_id:
        raise ValueError("A800 Scheduler rank observation step is invalid")
    if observation["scheduled_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 Scheduler rank observation tokens are invalid")
    if observation["configured_cudagraph_mode"] != "FULL_AND_PIECEWISE":
        raise ValueError("A800 Scheduler configured CUDAGraph mode is invalid")
    runtime_mode = observation["runtime_cudagraph_mode"]
    if runtime_mode not in {"FULL", "PIECEWISE", "NONE"}:
        raise ValueError("A800 Scheduler runtime CUDAGraph mode is invalid")
    _validate_a800_graph_key(
        observation["dispatch_input_key"], "A800 dispatch input key"
    )
    selected_key = observation["selected_graph_key"]
    if runtime_mode == "NONE":
        if selected_key is not None:
            raise ValueError("NONE A800 Scheduler observation selected graph key")
        fallback_reason = observation["graph_fallback_reason"]
        if not isinstance(fallback_reason, str) or not fallback_reason:
            raise ValueError("NONE A800 Scheduler observation lacks fallback")
    elif runtime_mode in ("FULL", "PIECEWISE"):
        _validate_a800_graph_key(selected_key, "A800 selected graph key")
        if observation["graph_fallback_reason"] is not None:
            raise ValueError("Graph A800 Scheduler observation has fallback")
    else:
        raise ValueError(f"unsupported A800 runtime Graph mode: {runtime_mode!r}")
    for field in (
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
    ):
        value = observation[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"A800 Scheduler rank observation {field} is invalid")
    if observation["num_unpadded_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 Scheduler unpadded tokens disagree")
    if observation["padded_total_tokens"] != (
        expected_scheduled_tokens + observation["padding_tokens"]
    ):
        raise ValueError("A800 Scheduler padding tokens disagree")
    if observation["dispatch_input_key"]["num_tokens"] != observation[
        "padded_total_tokens"
    ]:
        raise ValueError("A800 Scheduler dispatch input key is inconsistent")
    if observation["worker_execution_clock"] != "time.perf_counter_ns":
        raise ValueError("A800 Scheduler worker clock is invalid")
    if observation["worker_execution_source"] != (
        "vllm_v1_worker_execute_model_envelope"
    ):
        raise ValueError("A800 Scheduler worker source is invalid")


def _validate_a800_graph_key(value: Any, description: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "num_tokens", "uniform_decode"
    }:
        raise TypeError(f"{description} fields are invalid")
    num_tokens = value["num_tokens"]
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise TypeError(f"{description} num_tokens is invalid")
    if not isinstance(value["uniform_decode"], bool):
        raise TypeError(f"{description} uniform_decode is invalid")


def _validate_benchmark_log_window(
    paths_and_offsets: Sequence[tuple[Path, int]],
) -> dict[str, Any]:
    warning = "JIT compilation during inference"
    files = []
    matches = []
    for path, start_offset in paths_and_offsets:
        data = path.read_bytes()
        if not 0 <= start_offset <= len(data):
            raise ValueError(f"invalid benchmark log offset for {path}")
        window = data[start_offset:].decode("utf-8", errors="replace")
        file_matches = [line for line in window.splitlines() if warning in line]
        files.append({
            "path": str(path.resolve()),
            "start_offset_bytes": start_offset,
            "end_offset_bytes": len(data),
            "sha256": sha256_file(path),
        })
        matches.extend(file_matches)
    if matches:
        raise RuntimeError(
            "formal benchmark window contains runtime JIT warnings: "
            + json.dumps(matches, ensure_ascii=False)
        )
    return {
        "status": "validated",
        "jit_warning_pattern": warning,
        "jit_warning_count": 0,
        "files": files,
    }


def _capture_a800_post_benchmark_counters(
    process_group_id: int,
    instrumentation: dict[str, Any],
) -> dict[str, Any]:
    worker_rank_count = instrumentation["worker_rank_count"]
    if (
        isinstance(worker_rank_count, bool)
        or not isinstance(worker_rank_count, int)
        or worker_rank_count < 1
    ):
        raise TypeError("A800 Scheduler trace worker rank count is invalid")
    trace_path = Path(instrumentation["output_path"])
    _signal_process_group(process_group_id, signal.SIGUSR1)
    deadline = time.monotonic() + 10.0
    while True:
        paths = [
            Path(f"{trace_path}.compilation-counter.rank-{rank}.jsonl")
            for rank in range(worker_rank_count)
        ]
        if all(path.is_file() and len(path.read_text().splitlines()) == 2 for path in paths):
            return _validate_a800_compilation_counters(paths)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "A800 worker compilation counters did not arrive after formal benchmark"
            )
        time.sleep(0.05)


def _validate_a800_compilation_counters(
    paths: Sequence[Path],
) -> dict[str, Any]:
    rank_summaries = []
    for expected_rank, path in enumerate(paths):
        lines = path.read_text().splitlines()
        if len(lines) != 2:
            raise ValueError("A800 compilation counter file must have two snapshots")
        snapshots = [json.loads(line) for line in lines]
        before, after = snapshots
        for value, phase in (
            (before, "after_warmup_before_formal"),
            (after, "after_formal_benchmark"),
        ):
            if value["schema_version"] != 1:
                raise ValueError("A800 compilation counter schema version is invalid")
            if value["source"] != "vllm.compilation.counter.compilation_counter":
                raise ValueError("A800 compilation counter source is invalid")
            if value["rank"] != expected_rank:
                raise ValueError("A800 compilation counter rank disagrees with path")
            if value["phase"] != phase:
                raise ValueError("A800 compilation counter phase order is invalid")
            if not isinstance(value["counter"], dict):
                raise TypeError("A800 compilation counter snapshot must be an object")
        if not isinstance(before["trace_step_id"], str) or not before[
            "trace_step_id"
        ]:
            raise ValueError("A800 warmup counter snapshot lacks formal step identity")
        if after["trace_step_id"] is not None:
            raise ValueError("A800 post-benchmark counter snapshot has step identity")
        if before["timestamp_ns"] > after["timestamp_ns"]:
            raise ValueError("A800 compilation counter timestamps are not causal")
        before_counter = before["counter"]
        after_counter = after["counter"]
        if set(before_counter) != set(after_counter):
            raise ValueError("A800 compilation counter field sets differ")
        delta = {}
        for field in _A800_COMPILATION_COUNTER_FIELDS:
            before_value = before_counter[field]
            after_value = after_counter[field]
            if (
                isinstance(before_value, bool)
                or not isinstance(before_value, int)
                or before_value < 0
                or isinstance(after_value, bool)
                or not isinstance(after_value, int)
                or after_value < 0
            ):
                raise TypeError(
                    "A800 compilation counter value is not a non-negative int: "
                    f"{field}"
                )
            delta[field] = after_value - before_value
        nonzero = {
            field: value for field, value in delta.items() if value != 0
        }
        if nonzero:
            raise RuntimeError(
                "formal benchmark performed runtime compilation/capture: "
                + json.dumps(nonzero, sort_keys=True)
            )
        rank_summaries.append({
            "rank": expected_rank,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "after_warmup": before_counter,
            "after_formal_benchmark": after_counter,
            "formal_delta": delta,
        })
    return {
        "status": "validated",
        "counter_fields": list(_A800_COMPILATION_COUNTER_FIELDS),
        "rank_snapshots": rank_summaries,
    }
