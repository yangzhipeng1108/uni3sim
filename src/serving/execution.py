"""Run-local Serving workload construction, lowering, and prediction."""

from dataclasses import dataclass, replace
from collections import OrderedDict
import time
import gc
from contextlib import contextmanager
from threading import RLock
from typing import Callable, Mapping
from types import MappingProxyType

from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionResourceCalendar,
    _common_id_prefix,
    _normalize_id,
    _cached_plan_structure_key,
    _translate_segment_ids,
    build_execution_timeline,
    build_plan_structure_runtime_graph_segments,
    build_runtime_graph_segments,
    with_earliest_ready_resource_order,
    call_host_lanes,
    iter_call_occurrences,
)
from src.serving.kernel_lowering.kernel_predict import (
    KernelPredictionMemo,
    predict_kernel_plan,
)
from src.serving.kernel_lowering.kernel_plan import KernelCall, KernelPlan
from src.serving.kernel_lowering.lower_workload import (
    KernelLoweringMemo,
    lower_workload,
)
from src.serving.kv_cache.kv_cache_events import (
    KVCacheAction, KVCacheEvent, host_kv_transfer_attributes,
)
from src.serving.schedule.schedule_step import ScheduleStep
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.graph_padding import VLLMDenseGraphInputs
from src.serving.workload.runtime_inputs import RuntimeInputs, V32PaddedMTPInputs, DSparkSourceInputs
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode
from src.kernel.predictors.common import graph_replayed_launch
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    RuntimeGraphSegmentDescriptor,
)
from src.serving.runtime_prepare import (
    runtime_prepare_source_signature,
    WORKER_REQUEST_STATE_MECHANISMS,
)


_SOURCE_CAPTURE_NORMALIZATION_SOURCE = tuple((function, function.__code__)
    for function in (_normalize_id, _translate_segment_ids))


_METADATA_COLLECTION_LOCK = RLock()
_METADATA_COLLECTION_USERS = 0
_METADATA_COLLECTION_WAS_ENABLED = False


@contextmanager
def _metadata_collection_scope():
    """Defer cyclic scans while a transient metadata step is being built.

    CPython reference counting still releases tensor handles at their exact
    source lifetime boundaries. Automatic cycle collection resumes on scope
    exit, including exceptions; nested/threaded steps share the outer scope.
    """
    global _METADATA_COLLECTION_USERS, _METADATA_COLLECTION_WAS_ENABLED
    with _METADATA_COLLECTION_LOCK:
        if _METADATA_COLLECTION_USERS == 0:
            _METADATA_COLLECTION_WAS_ENABLED = gc.isenabled()
            if _METADATA_COLLECTION_WAS_ENABLED:
                gc.disable()
        _METADATA_COLLECTION_USERS += 1
    try:
        yield
    finally:
        with _METADATA_COLLECTION_LOCK:
            _METADATA_COLLECTION_USERS -= 1
            if _METADATA_COLLECTION_USERS == 0 and _METADATA_COLLECTION_WAS_ENABLED:
                gc.enable()


@dataclass(frozen=True)
class ServingExecutionComponents:
    workload: object
    kernel_plan: object
    predictions: Mapping[str, object]
    execution_timeline: object
    source_activation_memory: Mapping[str, object] | None = None
    source_state_summary: Mapping[str, object] | None = None


def bind_model_source_input_frames(runtime_inputs, frames):
    """Freeze explicit step scenarios at the common Serving boundary."""
    if frames is None:
        return None
    if (not isinstance(frames, Mapping) or not frames
            or any(not isinstance(step_id, str) or not step_id for step_id in frames)):
        raise ValueError("per-step source binding requires nonempty named explicit frames")
    if runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_dspark":
        valid = all(isinstance(state, DSparkSourceInputs) for state in frames.values())
    elif runtime_inputs.mtp_draft_sampling_source in ("vllm_0_30_legacy_full_logits_greedy",
            "vllm_0_27_1_eager_full_logits_greedy"):
        valid = all(isinstance(state, V32PaddedMTPInputs) for state in frames.values())
    elif (runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.model_execution_version == "vllm_0_30_0"):
        # None explicitly declares an eager step; it does not reuse a tail
        # from an earlier captured step or skip a missing frame key.
        valid = (any(isinstance(state, VLLMDenseGraphInputs) for state in frames.values())
            and all(isinstance(state, VLLMDenseGraphInputs)
                and (runtime_inputs.graph_execution.enabled or state.scheduled_token_ids is not None)
                or state is None and runtime_inputs.graph_execution.enabled for state in frames.values()))
    else:
        valid = False
    if not valid:
        raise ValueError("per-step source binding requires matching padded MTP or captured dense source and typed explicit frames")
    return MappingProxyType(dict(frames))


def execute_heterogeneous_flash_request(model, backend, *, input_ids, output_tokens, fpga_cards=8, root_seed=0):
    """Drive a live virtual PD request with the existing real CPU Scheduler.

    The supplied colocated 0.30 Scheduler is a length/control-plane driver,
    not the physical TP8/DP8 PD allocator. Its declared dummy feedback is
    retained. Event progression uses a fixed clock because no durations are
    supplied; callers must not interpret it as zero latency or measured QPS.
    This function owns and closes the supplied backend, including on failure.
    """
    from contextlib import closing
    from src.serving.request_gen.request_stream import RequestSpec
    from src.serving.workload.traced_heterogeneous_request import FlashHeterogeneousRequest

    with closing(backend):
        runtime = backend.runtime_info
        if (runtime["version"] != "0.30.0" or runtime["model_executed"] is not False
                or runtime["weights_loaded"] is not False or runtime["runtime_acceptance"] is not False
                or backend.config.role != "colocated" or backend.config.max_requests != 1
                or backend.config.scheduler_pipeline_depth != 1):
            raise ValueError("heterogeneous functional request requires the explicit CPU 0.30 single-request logical driver")
        program = FlashHeterogeneousRequest(model, input_ids=input_ids, output_tokens=output_tokens,
            fpga_cards=fpga_cards, root_seed=root_seed)
        backend.add_request(RequestSpec(0.0, "one", len(input_ids), output_tokens))
        steps, emitted, finished = [], 0, []
        while backend.has_work():
            if len(steps) >= output_tokens:
                raise RuntimeError("Scheduler exceeded the non-MTP single-request step bound")
            step = backend.plan_step(0.0)
            if step is None or len(step.schedule.requests) != 1:
                raise RuntimeError("real Scheduler did not produce the single admitted request")
            request, = step.schedule.requests
            if (request.request_id != "one" or step.embedding_token_source != "vllm_length_only_simulation"
                    or request.mtp_proposer_mode != "disabled" or step.generated_tokens["one"] != 1):
                raise ValueError("functional heterogeneous scope requires explicit length-only non-MTP feedback")
            if not steps:
                if (request.phase != "prefill" or request.computed_tokens != 0
                        or request.scheduled_tokens != len(input_ids) or step.embedding_token_ids != input_ids):
                    raise ValueError("Scheduler prompt IDs or query extent disagree with the virtual request")
                program.prefill()
            else:
                if (request.phase != "decode" or request.scheduled_tokens != 1
                        or request.computed_tokens != len(input_ids) + emitted - 1):
                    raise ValueError("Scheduler decode is not the consecutive live cache/token step")
                if program.decode_state is None:
                    program.handoff()
                program.decode()
            if program.emitted_tokens != emitted + 1:
                raise RuntimeError("virtual model output and Scheduler emission counts disagree")
            backend.complete_step(step, 0.0)
            emitted += step.generated_tokens["one"]
            finished.extend(step.finished_request_ids)
            steps.append(step)
        if emitted != output_tokens or finished != ["one"]:
            raise RuntimeError("real Scheduler did not finish the complete requested output length")
        return program, tuple(steps), runtime


def _runtime_graph_mode(
    schedule: ScheduleStep,
    graph_execution,
) -> str:
    uniform_decode = all(
        request.scheduled_tokens
        == graph_execution.uniform_decode_query_len
        for request in schedule.requests
    )
    return graph_execution.resolve(
        schedule.token_count,
        uniform_decode=uniform_decode,
    ).runtime_mode


_RUNTIME_PREPARE_EVENTS_CACHE_LIMIT = 8192
_SOURCE_CAPTURE_SEGMENTS_CACHE_LIMIT = 256


def _source_capture_host_segments(plan, regions, *, sync_before_replay):
    """Map executed source boundaries to existing Host replay descriptors."""
    # Repeats retain actual source node IDs for every layer. Map these to
    # template-call occurrence keys, which are the timeline's execution ABI;
    # never infer later capture boundaries from only the first layer.
    calls = {call.call_id: call for call in plan.calls}
    repeats = {name: repeat for repeat in plan.block_repeats for name in repeat.call_ids}
    source_maps = {}
    for repeat in plan.block_repeats:
        if not repeat.trace_node_ids:
            raise NotImplementedError("source capture Host timing requires every repeated source occurrence")
        source_maps[repeat.block_id] = dict(repeat.trace_node_ids)

    def occurrences():
        for call_id, index in iter_call_occurrences(plan):
            call = calls[call_id]
            if call_id not in repeats:
                yield call, index, call.source_node_ids
            else:
                sources = source_maps[repeats[call_id].block_id]
                yield call, index, tuple(sources[node][index] for node in call.source_node_ids)

    owners, members, lanes = {}, {}, {}
    for pool, rank, identifier, nodes in regions:
        key = (pool, rank, identifier)
        if key in members or not nodes:
            raise ValueError("source capture Host regions require unique nonempty source segments")
        members[key] = []
        lanes[key] = (pool, rank)
        for node in nodes:
            lane_node = (pool, rank, node)
            if lane_node in owners:
                raise ValueError("source capture Host regions overlap on a source node")
            owners[lane_node] = key
    covered, total = 0, 0
    for call, occurrence, source_nodes in occurrences():
        for pool, rank in call_host_lanes(call):
            total += 1
            labels = {owners[(pool, rank, node)] if (pool, rank, node) in owners else None
                for node in source_nodes}
            if len(labels) != 1:
                raise NotImplementedError("lowered fusion crosses a source capture/eager boundary")
            label, = labels
            if label is not None:
                members[label].append((call.call_id, occurrence))
                covered += 1
    if not covered or any(not calls for calls in members.values()):
        raise ValueError("source capture Host segment lost every lowered call occurrence")
    descriptors = tuple(RuntimeGraphSegmentDescriptor(key[2], *lanes[key], tuple(calls), sync_before_replay)
        for key, calls in members.items())
    # FULL/PIECEWISE here describes whole-step Host coverage; the source
    # capture's FULL mode and user GraphExecutionConfig remain unchanged.
    return descriptors, "FULL" if covered == total else "PIECEWISE"


def _cached_source_capture_host_segments(cache, plan, regions, *, sync_before_replay,
        normalize_prefix=None):
    """Reuse exact source-to-Host boundaries within one execution context.

    The schedule structure alone omits the source IDs of fused calls and
    repeated layers. Both those identities and the actual capture regions
    participate, so changing a later layer or rank still executes validation.
    No prediction, allocation, or per-step timing is retained here.
    Summary callers can request the timeline's normalized identifier space;
    its prefix and current normalization implementation also enter the key.
    """
    if not isinstance(sync_before_replay, bool):
        raise TypeError("runtime Graph segment sync_before_replay must be a bool")
    if normalize_prefix is not None and type(normalize_prefix) is not str:
        raise TypeError("source capture normalization prefix must be a string")
    normalization_source = None
    if normalize_prefix is not None:
        from src.serving.kernel_lowering import execution_timeline as timeline
        normalize, translate = timeline._normalize_id, timeline._translate_segment_ids
        if (any(function is not original or function.__code__ is not code
                or function.__defaults__ is not None or function.__kwdefaults__ is not None
                for function, (original, code) in zip((normalize, translate),
                    _SOURCE_CAPTURE_NORMALIZATION_SOURCE, strict=True))
                or type(plan) is not KernelPlan
                or any(type(call) is not KernelCall or type(call.call_id) is not str for call in plan.calls)):
            segments, mode = _source_capture_host_segments(plan, regions,
                sync_before_replay=sync_before_replay)
            return translate(segments, lambda call_id: normalize(call_id, normalize_prefix)), mode
        normalization_source = _SOURCE_CAPTURE_NORMALIZATION_SOURCE
    key = (
        _source_capture_host_segments, iter_call_occurrences, call_host_lanes,
        normalization_source, normalize_prefix,
        _cached_plan_structure_key(plan, plan.topological_order(), ""),
        tuple((call.call_id, call.source_node_ids) for call in plan.calls),
        tuple((repeat.block_id, repeat.trace_node_ids) for repeat in plan.block_repeats),
        regions, sync_before_replay,
    )
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    result = _source_capture_host_segments(plan, regions, sync_before_replay=sync_before_replay)
    if normalize_prefix is not None:
        segments, mode = result
        result = (translate(segments,
            lambda call_id: normalize(call_id, normalize_prefix)), mode)
    cache[key] = result
    if len(cache) > _SOURCE_CAPTURE_SEGMENTS_CACHE_LIMIT:
        cache.popitem(last=False)
    return result


def _cached_runtime_prepare_events(
    cache: OrderedDict,
    builder,
    kernel_plan,
    materialize_timeline_details: bool,
    worker_request_state=None,
    worker_state_mechanisms=(),
):
    """Return prepare events for one plan, cached per normalized structure.

    The builder output depends on the prefix-normalized scheduling topology
    and the explicitly consumed source fields (stage, dtype and TP/source
    program identity), plus enabled worker-state source counts. All enter
    this owner-specific cache key. The
    summary timeline consumes the normalized form directly; the detailed
    timeline gets keys remapped to the current step's identifier prefix.
    Returns ``(events, normalized_events, events_token)`` with exactly one
    of the two event forms populated."""

    prefix = _common_id_prefix(kernel_plan)
    structure_key = (_cached_plan_structure_key(
        kernel_plan,
        kernel_plan.topological_order(),
        prefix,
    ), runtime_prepare_source_signature(kernel_plan, prefix),
        worker_request_state if worker_state_mechanisms else None, worker_state_mechanisms)
    if structure_key in cache:
        cached = cache[structure_key]
        cache.move_to_end(structure_key)
    else:
        raw_events = (builder(kernel_plan, worker_request_state=worker_request_state)
            if worker_state_mechanisms else builder(kernel_plan))
        cached = (
            prefix,
            raw_events,
            {
                (
                    pool_name,
                    rank,
                    _normalize_id(call_id, prefix),
                    occurrence_index,
                ): mechanisms
                for (pool_name, rank, call_id, occurrence_index),
                mechanisms in raw_events.items()
            },
        )
        cache[structure_key] = cached
        while len(cache) > _RUNTIME_PREPARE_EVENTS_CACHE_LIMIT:
            cache.popitem(last=False)
    building_prefix, cached_events, normalized_events = cached
    if not materialize_timeline_details:
        return None, normalized_events, cached
    if building_prefix == prefix:
        return cached_events, None, None
    return (
        {
            (
                pool_name,
                rank,
                (
                    prefix + call_id[len(building_prefix):]
                    if call_id.startswith(building_prefix)
                    else call_id
                ),
                occurrence_index,
            ): mechanisms
            for (pool_name, rank, call_id, occurrence_index),
            mechanisms in cached_events.items()
        },
        None,
        None,
    )


class ServingExecutionContext:
    """Own all exact run-local reuse state for one Serving execution."""

    def __init__(
        self,
        model: ModelConfig,
        deployment: DeploymentPlan,
        hardware_by_id: Mapping[str, object],
        launch_policy: str,
        prediction_method: str,
        runtime_inputs: RuntimeInputs,
        *,
        execution_mode: str = "accelerated",
        host_launch_profiles: Mapping[str, HostLaunchProfile] | None = None,
        host_runtime_prepare_profiles: Mapping[
            str, HostRuntimePrepareProfile] | None = None,
        runtime_prepare_event_builder: Callable[
            [object], Mapping[
                tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
            ]
        ] | None = None,
        materialize_timeline_details: bool = True,
        model_source_inputs_by_step: Mapping[str, V32PaddedMTPInputs | VLLMDenseGraphInputs | DSparkSourceInputs | None] | None = None,
        collect_source_state: bool = False,
        fresh_kernel_predictions: bool = False,
    ) -> None:
        if not isinstance(model, ModelConfig):
            raise TypeError("Serving execution model must be ModelConfig")
        if not isinstance(deployment, DeploymentPlan):
            raise TypeError(
                "Serving execution deployment must be DeploymentPlan")
        if not isinstance(hardware_by_id, Mapping) or not hardware_by_id:
            raise TypeError(
                "Serving execution hardware_by_id must be a non-empty mapping")
        if not isinstance(runtime_inputs, RuntimeInputs):
            raise TypeError(
                "Serving execution runtime_inputs must be RuntimeInputs")
        from src.serving.workload.traced_models import bind_ordinary_forward_source
        runtime_inputs = bind_ordinary_forward_source(model, deployment, runtime_inputs)
        if runtime_inputs.symbolic_forward_source is None:
            raise ValueError("Serving requires a bound virtual model source and its declared runtime inputs")
        if execution_mode not in ("accelerated", "reference", "fast"):
            raise ValueError(
                f"unknown Serving execution mode: {execution_mode!r}")
        if (execution_mode in ("accelerated", "fast")
                and runtime_inputs.expert_routing.source_kind
                == "replay_observation"):
            raise NotImplementedError(
                "per-layer expert routing replay requires reference execution")
        self._initialize_workload_state(model, deployment, hardware_by_id, launch_policy,
            prediction_method, execution_mode, materialize_timeline_details)
        self._set_fresh_kernel_predictions(fresh_kernel_predictions)
        self._traced_device_source = False
        self.runtime_inputs = runtime_inputs
        self.model_source_inputs_by_step = bind_model_source_input_frames(runtime_inputs, model_source_inputs_by_step)
        self._source_workspace_capacity_by_pool = {}
        self._dspark_request_progress = {}
        if type(collect_source_state) is not bool:
            raise TypeError("source state collection must be boolean")
        self.collect_source_state = collect_source_state
        if host_launch_profiles is None:
            self.host_launch_profiles = None
        else:
            self.host_launch_profiles = dict(host_launch_profiles)
        if host_runtime_prepare_profiles is None:
            self.host_runtime_prepare_profiles = None
        else:
            self.host_runtime_prepare_profiles = dict(
                host_runtime_prepare_profiles)
        if ((self.host_runtime_prepare_profiles is None)
                != (runtime_prepare_event_builder is None)):
            raise ValueError(
                "runtime prepare profiles and event builder must be supplied together")
        self.runtime_prepare_event_builder = runtime_prepare_event_builder
        self._runtime_prepare_events_cache: OrderedDict = OrderedDict()
        self._source_capture_segments_cache: OrderedDict = OrderedDict()
        self._source_capture_ports = {}
        from src.serving.workload.traced_forward import SourceProgramCache
        self._source_program_cache = SourceProgramCache(
            glm_mtp_structural_reuse=self.execution_mode == "fast",
            glm_ordinary_structural_reuse=self.execution_mode == "fast")
        self._recorded_source_program_bound = False

    def _initialize_workload_state(self, model, deployment, hardware_by_id, launch_policy,
            prediction_method, execution_mode, materialize_timeline_details):
        if (not isinstance(model, ModelConfig) or not isinstance(deployment, DeploymentPlan)
                or not isinstance(hardware_by_id, Mapping) or not hardware_by_id):
            raise TypeError("workload execution requires ModelConfig, DeploymentPlan and nonempty hardware mapping")
        if execution_mode not in ("accelerated", "reference", "fast"):
            raise ValueError(f"unknown Serving execution mode: {execution_mode!r}")
        if type(materialize_timeline_details) is not bool:
            raise TypeError("timeline detail materialization flag must be boolean")
        self.model, self.deployment = model, deployment
        self.hardware_by_id = dict(hardware_by_id)
        self.launch_policy, self.prediction_method = launch_policy, prediction_method
        self.execution_mode = execution_mode
        self.materialize_timeline_details = materialize_timeline_details
        self._device_resource_order = "source"
        self._resource_order_wall_time_s = 0.0
        self._executed_workloads = 0
        self._phase_wall_time_s = dict.fromkeys(("workload_construction", "kernel_lowering",
            "kernel_prediction", "timeline_construction"), 0.0)
        accelerated = execution_mode in ("accelerated", "fast")
        self._lowering_memo = KernelLoweringMemo() if accelerated else None
        self._prediction_memo = KernelPredictionMemo() if accelerated else None

    def _set_fresh_kernel_predictions(self, enabled):
        """Keep topology reuse while recomputing every current physical class.

        Fresh mode retains no prediction values across workloads. Equal
        rank/layer specs within one workload may share its current formula
        evaluation; the temporary grouping is discarded at the return.
        """
        if type(enabled) is not bool:
            raise TypeError("fresh kernel predictions flag must be boolean")
        self.fresh_kernel_predictions = enabled
        self._fresh_prediction_hits = 0
        self._fresh_prediction_misses = 0
        if enabled:
            self._prediction_memo = None

    @classmethod
    def for_traced_device_sources(cls, model, deployment, hardware_by_id, *,
            launch_policy="analytical_launch", prediction_method="roofline",
            execution_mode="accelerated", materialize_timeline_details=False,
            device_resource_order="source", fresh_kernel_predictions=False):
        """Explicit actual-source factory; no ordinary runtime source is inferred.

        It shares source-body programs, workload execution, exact lowering
        recipes and the timeline with ordinary Serving. The builder owns its
        live source bindings; this entry does not bind an ordinary prefix.
        """
        if device_resource_order not in ("source", "earliest_ready"):
            raise ValueError("device resource order must be source or earliest_ready")
        context = object.__new__(cls)
        context._initialize_workload_state(model, deployment, hardware_by_id, launch_policy,
            prediction_method, execution_mode, materialize_timeline_details)
        context._set_fresh_kernel_predictions(fresh_kernel_predictions)
        context._traced_device_source = True
        context._device_resource_order = device_resource_order
        context.runtime_inputs = None
        context.model_source_inputs_by_step = None
        context.collect_source_state = False
        context.host_launch_profiles = None
        context.host_runtime_prepare_profiles = None
        context.runtime_prepare_event_builder = None
        return context

    def _prepare_dspark_progress(self, schedule, state, token_ids):
        """Validate logical request progress before source construction/memos.

        This ledger binds declared token scenarios, not native cache pointers.
        It stores scalar/ID facts only and advances after a successful step.
        """
        from src.serving.workload.runtime_inputs import DSparkFirstProposalInputs
        from src.serving.schedule.schedule_step import BackendStep
        if not isinstance(state, DSparkFirstProposalInputs):
            if any((schedule.pool_name, request.request_id) in self._dspark_request_progress
                    for request in schedule.requests):
                raise ValueError("DSpark logical request cannot lose its explicit source state")
            return None
        BackendStep(schedule, generated_tokens={request.request_id: (request.raw_model_output_tokens if state.delivered_token_count is None else state.delivered_token_count) for request in schedule.requests}, embedding_token_ids=token_ids,
            embedding_token_source="vllm_explicit_request_token_ids", model_source_inputs=state)
        if state.request_ids != tuple(request.request_id for request in schedule.requests):
            raise ValueError("DSpark logical progress request identity disagrees with schedule")
        key = (schedule.pool_name, state.request_ids[0])
        identity = (state.source_id, state.draft_model.config_sha256)
        if state.is_verification:
            if key not in self._dspark_request_progress:
                raise ValueError("DSpark verification has no successfully completed predecessor")
            previous = self._dspark_request_progress[key]
            if (previous["finished"] or previous["identity"] != identity
                    or previous["step_id"] != state.predecessor_step_id or schedule.step_id == state.predecessor_step_id
                    or previous["retained_kv_tokens"] != state.previous_retained_kv_tokens
                    or previous["anchor_token_ids"] != state.previous_anchor_token_ids
                    or previous["draft_token_ids"] != state.previous_draft_token_ids):
                raise ValueError("DSpark predecessor/cache identity, live length or proposal transcript changed")
            completion = previous["completion_time_s"]
            retained_tokens = previous["retained_token_ids"] + token_ids[:state.accepted_drafts + 1]
            emitted = (*previous["emitted_token_ids"],
                *(ids[0] for ids in state.previous_draft_token_ids[:state.accepted_drafts]), state.anchor_token_ids[0])
        else:
            if key in self._dspark_request_progress:
                raise ValueError("DSpark first proposal cannot reset an existing logical request")
            completion = 0.0
            retained_tokens = token_ids
            emitted = state.anchor_token_ids
        if state.delivered_token_count is not None:
            raw_count=1 if state.accepted_drafts is None else state.accepted_drafts+1
            emitted=emitted[:len(emitted)-raw_count+state.delivered_token_count]
        if len(retained_tokens) != state.retained_kv_tokens:
            raise ValueError("DSpark retained token history does not conserve live KV length")
        return key, state, identity, completion, retained_tokens, emitted

    def _commit_dspark_progress(self, prepared, step_id, result, origin_time_s):
        if prepared is None:
            return result
        key, state, identity, previous_completion, retained_tokens, emitted = prepared
        completion = origin_time_s + result.execution_timeline.duration_s
        if completion < previous_completion:
            raise RuntimeError("DSpark request completion precedes its required prior step")
        receipt = {
            "request_id": key[1], "pool_name": key[0], "step_id": step_id,
            "predecessor_step_id": state.predecessor_step_id,
            "source_id": state.source_id, "scope": "declared_logical_request_progress",
            "retained_kv_tokens": state.retained_kv_tokens,
            "target_input_rows": len(state.target_positions),
            "accepted_drafts": state.accepted_drafts,
            "rejected_drafts": None if state.accepted_drafts is None else 5 - state.accepted_drafts,
            "sampled_valid_count": 1 if state.accepted_drafts is None else state.accepted_drafts + 1,
            "emitted_token_ids": emitted, "retained_token_ids": retained_tokens,
            "delivered_token_count": (1 if state.accepted_drafts is None else state.accepted_drafts+1)
                if state.delivered_token_count is None else state.delivered_token_count,
            "next_anchor_token_ids": state.anchor_token_ids,
            "next_draft_token_ids": state.draft_token_ids,
            "request_finished": state.finish_request,
            "completion_time_s": completion,
            "native_cross_step_cache_alias_bound": False,
            "checkpoint_logits_numerically_computed": False,
            "normal_capacity_accepted": False,
        }
        summary = {} if result.source_state_summary is None else dict(result.source_state_summary)
        if "dspark_progress" in summary:
            raise ValueError("DSpark progress receipt cannot overwrite an existing source receipt")
        summary["dspark_progress"] = receipt
        result = replace(result, source_state_summary=summary)
        self._dspark_request_progress[key] = {
            "identity": identity, "step_id": step_id,
            "retained_kv_tokens": state.retained_kv_tokens,
            "anchor_token_ids": state.anchor_token_ids, "draft_token_ids": state.draft_token_ids,
            "retained_token_ids": retained_tokens, "emitted_token_ids": emitted,
            "completion_time_s": completion, "finished": state.finish_request,
        }
        return result


    def _independent_decode_workspace_capacities(self, schedule):
        """Certify unchanged *represented* V32 workspace for one decode.

        The complete current source requests only its fixed q_concat arena
        during ordinary q1 decode. Indexer FP8 gather is prefill-only; native
        selector/library scratch remains explicitly unbound. This reads real
        completed-Forward capacities and never initializes or advances state.
        """
        from src.serving.workload import traced_models, traced_forward
        runtime, model = self.runtime_inputs, self.model
        if (type(self) is not ServingExecutionContext or type(schedule) is not ScheduleStep
                or type(model) is not ModelConfig or type(runtime) is not RuntimeInputs
                or model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16"
                or model.weight_dtype not in ("bf16", "fp8")
                or runtime.serving_framework != "vllm" or runtime.model_execution_version != "vllm_0_30_0"
                or runtime.symbolic_forward_source != {
                    "bf16": "vllm_0_30_v32_bf16_sync_explicit_allreduce",
                    "fp8": "vllm_0_30_v32_fp8_sync_explicit_allreduce"}[model.weight_dtype]
                or runtime.graph_execution.enabled
                or runtime.mtp_draft_sampling_source is not None or runtime.mtp_proposer_execution_source is not None
                or runtime.model_source_inputs is not None or self.model_source_inputs_by_step is not None
                or runtime.symbolic_workspace_initial_capacity_bytes is None
                or runtime.glm_indexer_selection_source != "declared_causal_topk"
                or runtime.glm_prefill_attention_source not in (None, "force_sparse_mqa")
                or runtime.target_sampling_source not in (None, "vllm_0_30_plain_greedy_bf16",
                    "vllm_0_30_plain_sampler_bf16")
                or self.deployment.mode != "colocated" or len(self.deployment.worker_pools) != 1
                or schedule.pool_name not in self.deployment.worker_pools
                or not schedule.requests
                or any(request.phase != "decode" or request.scheduled_tokens != 1
                    or request.mtp_proposer_mode != "disabled"
                    or any((request.mtp_proposed_tokens, request.mtp_verified_tokens, request.mtp_accepted_tokens,
                        request.mtp_proposer_tokens, request.mtp_next_proposed_tokens)) for request in schedule.requests)):
            return None
        pool = self.deployment.worker_pools[schedule.pool_name]
        parallel = pool.parallel_plan.config
        if (pool.hardware_id != "nvidia_h200" or len(pool.ranks) != 8 or len(set(pool.ranks)) != 8
                or parallel.tensor_parallel_size != 8 or parallel.expert_parallel_size != 1
                or any(value != 1 for value in (parallel.pipeline_parallel_size, parallel.sequence_parallel_size,
                    parallel.context_parallel_size, parallel.data_parallel_size))):
            return None
        batch_limit, model_limit = runtime.symbolic_source_max_batched_tokens, runtime.symbolic_source_max_model_len
        if (type(batch_limit) is not int or type(model_limit) is not int
                or batch_limit <= 0 or batch_limit != model_limit
                or not len(schedule.requests) <= batch_limit <= model.max_position_embeddings
                or any(request.computed_tokens + 1 > model_limit for request in schedule.requests)
                or schedule.pool_name not in self._source_workspace_capacity_by_pool
                or "bf16" not in traced_forward.DTYPE_BYTES
                or type(traced_forward.DTYPE_BYTES["bf16"]) is not int or traced_forward.DTYPE_BYTES["bf16"] != 2
                or not traced_models._DENSE_METADATA_SOURCE_CONTRACT.unchanged()):
            return None
        capacities = self._source_workspace_capacity_by_pool[schedule.pool_name]
        # DeepseekV32Attention.__init__ uses BF16[B,64,kv_lora+rope].
        # ForwardWorkspace rounds each requested port to its 256-byte lane
        # alignment, and grows only when that total exceeds current capacity.
        needed = 2 * batch_limit * 64 * (model.kv_lora_rank + model.qk_rope_head_dim)
        needed = -(-needed // 256) * 256
        if (type(capacities) is not tuple or len(capacities) != 8
                or any(type(capacity) is not int or capacity < needed for capacity in capacities)):
            return None
        return capacities

    def _add_side_stream_fork_node_boundary(self, kernel_plan, predictions):
        """Charge SGLang's per-layer side-stream fork on every graph node.

        Under graph capture SGLang runs ``kv_a_layernorm`` on ``alt_stream``
        forked from and joined back to the main stream once per MLA layer
        (v0.5.19 ``forward_mla.py:316-335``; ``alt_stream`` exists on CUDA,
        ``deepseek_v2.py:2666-2674``).  A replayed graph with that topology
        spaces every dependent node by the hardware's measured side-stream
        fork boundary on top of the pure-chain node the kernel prices assume.
        """

        if self.runtime_inputs is None or self.runtime_inputs.serving_framework != "sglang":
            return predictions
        replayed = [
            call for call in kernel_plan.calls
            if graph_replayed_launch(call.kernel_spec)]
        if not replayed:
            return predictions
        if self.model.model_type != "deepseek_v3" or not self.model.q_lora_rank:
            raise NotImplementedError(
                "SGLang graph node boundary is source-backed only for the "
                "DeepSeek-V3 MLA alt-stream fork")
        adjusted = dict(predictions)
        for call in replayed:
            increment_s = self.hardware_by_id[
                call.kernel_spec.hardware_id
            ].side_stream_fork_graph_node_increment_s()
            prediction = adjusted[call.call_id]
            adjusted[call.call_id] = replace(
                prediction,
                t_total_s=prediction.t_total_s + increment_s,
                t_launch_s=prediction.t_launch_s + increment_s,
            )
        return adjusted

    @_metadata_collection_scope()
    def execute_step(
        self,
        schedule: ScheduleStep,
        *,
        kv_cache_events: tuple[KVCacheEvent, ...] = (),
        origin_time_s: float = 0.0,
        device_floor_s: float = 0.0,
        shared_resource_calendar: ExecutionResourceCalendar | None = None,
        runtime_prepare_events: Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
        embedding_token_ids: tuple[int, ...] | None = None,
        model_source_inputs: object | None = None,
        source_launch_inputs: Mapping[str, object] | None = None,
        require_timeline_details: bool = False,
        timeline_origin_provider: Callable[[], tuple[float, float]] | None = None,
        worker_request_state=None,
    ) -> ServingExecutionComponents:
        if self.runtime_inputs is None:
            raise ValueError("schedule execution requires explicit RuntimeInputs; use the actual traced-source builder")
        if worker_request_state is not None:
            from src.serving.host_timing import WorkerRequestState
            if not isinstance(worker_request_state, WorkerRequestState):
                raise TypeError("step worker state requires typed source counts")
            if worker_request_state.cached_requests + worker_request_state.admitted_requests != len(schedule.requests):
                raise ValueError("step worker state must cover its exact scheduled requests")
        if timeline_origin_provider is not None:
            if not callable(timeline_origin_provider):
                raise TypeError("timeline origin provider must be callable")
            if (isinstance(origin_time_s, bool) or isinstance(device_floor_s, bool)
                    or origin_time_s != 0.0 or device_floor_s != 0.0
                    or shared_resource_calendar is not None or schedule.is_transfer_only):
                raise ValueError("deferred timeline origins require an independent compute step")
        # A receipt belongs only to a successfully completed current step.
        self._recorded_source_program_bound = False
        if self.runtime_inputs.dense_source_input_policy is not None:
            pool = self.deployment.worker_pools[schedule.pool_name]
            parallel = pool.parallel_plan.config
            if (self.runtime_inputs.dense_source_input_policy != "declared_zero_tp2"
                    or self.deployment.mode != "colocated" or len(self.deployment.worker_pools) != 1
                    or pool.hardware_id != "nvidia_h200" or parallel.tensor_parallel_size != 2
                    or not parallel.is_rank_local or parallel.pipeline_parallel_size != 1
                    or self.model.model_type not in ("qwen3", "llama", "qwen3_moe")
                    or self.model.compute_dtype != self.model.weight_dtype or self.model.compute_dtype != "bf16"
                    or schedule.is_transfer_only or self.model_source_inputs_by_step is not None
                    or self.runtime_inputs.model_source_inputs is not None):
                raise ValueError("declared zero inputs require the ordinary H200 BF16 TP2 source without competing frames")
            # Preserve explicit backend frames and known IDs. Only absent
            # content receives the selected synthetic approximation.
            if model_source_inputs is None:
                if embedding_token_ids is None:
                    embedding_token_ids = (self.runtime_inputs.embedding_token_ids
                        if self.runtime_inputs.embedding_token_ids is not None else (0,) * schedule.token_count)
                shape = self.runtime_inputs.graph_execution.resolve(schedule.token_count,
                    uniform_decode=all(request.phase == "decode" and request.scheduled_tokens ==
                        self.runtime_inputs.graph_execution.uniform_decode_query_len for request in schedule.requests))
                if shape.runtime_mode != "NONE":
                    full = shape.runtime_mode == "FULL"
                    max_requests, max_model_len = self.runtime_inputs.dense_source_capture_limits
                    source_id = "declared-synthetic-canonical-lengths-zero-ids-and-tail-v2"
                    model_source_inputs = VLLMDenseGraphInputs(tuple(request.request_id for request in schedule.requests),
                        "declared_synthetic", source_id, (0,) * shape.padding_tokens,
                        max_requests if full else None, max_model_len if full else None,
                        source_id if full else None, scheduled_token_ids=embedding_token_ids)
        if schedule.is_transfer_only:
            if source_launch_inputs is not None:
                raise ValueError("transfer-only steps cannot bind compute source launches")
            if (self.runtime_inputs.model_execution_version != "vllm_0_30_0"
                    or self.launch_policy != "analytical_launch"
                    or self.deployment.mode != "colocated"
                    or len(self.deployment.worker_pools) != 1
                    or self.model.model_type != "qwen3"
                    or self.model.attention_layout != "standard_gqa"
                    or embedding_token_ids is not None
                    or model_source_inputs is not None
                    or not kv_cache_events):
                raise NotImplementedError(
                    "transfer-only Host KV step requires the native Qwen3 "
                    "SimpleCPU eager scope")
            pool = self.deployment.worker_pools[schedule.pool_name]
            if pool.kv_cache is None:
                raise ValueError("transfer-only Host KV step needs a tier plan")
            parallel = pool.parallel_plan.config
            if (pool.hardware_id != "nvidia_a800"
                    or parallel.tensor_parallel_size not in (1, 2)
                    or len(pool.ranks) != parallel.tensor_parallel_size
                    or parallel.pipeline_parallel_size != 1
                    or parallel.enable_expert_parallel
                    or not parallel.is_rank_local):
                raise NotImplementedError(
                    "transfer-only SimpleCPU step requires A800 TP1/TP2 PP1")
            intents = tuple(event for event in kv_cache_events
                            if event.action is KVCacheAction.STORE)
            if (not intents or any(event.action not in (
                    KVCacheAction.STORE, KVCacheAction.RELEASE)
                    for event in kv_cache_events)):
                raise ValueError("transfer-only step requires stores and releases")
            attributes = tuple(host_kv_transfer_attributes(
                event, cache_config=pool.kv_cache,
                model_bytes_per_token=self.model.kv_bytes_per_token_per_parallel_rank(
                    parallel.tensor_parallel_size, parallel.pipeline_parallel_size),
                scheduled_request_ids=set()) for event in intents)
            nodes = tuple(WorkloadNode(
                (f"{schedule.step_id}:kv-offload:{entry['ready_event']}"
                 + (f":tp-rank-{rank}" if len(pool.ranks) > 1 else "")),
                WorkloadKind.KV_OFFLOAD, pool.name, {}, rank=rank,
                attributes=entry)
                for rank in pool.ranks for entry in attributes)
            return self.execute_workload(
                WorkloadGraph(nodes, ()), origin_time_s=origin_time_s,
                shared_resource_calendar=shared_resource_calendar,
                runtime_graph_mode="NONE",
                require_timeline_details=True)
        # Reset before validation so failures never expose a preceding bind.
        # Token ownership is per scheduled step, unlike the run's immutable
        # execution/source settings. Never retain a preceding step's IDs.
        if self.model_source_inputs_by_step is not None:
            if model_source_inputs is not None:
                raise ValueError("model-source frames require one owner, not both backend transport and execution mapping")
            if schedule.step_id not in self.model_source_inputs_by_step:
                raise ValueError("scheduled step is absent from explicitly bound model-source inputs")
            model_source_inputs = self.model_source_inputs_by_step[schedule.step_id]
        if isinstance(model_source_inputs, VLLMDenseGraphInputs) and model_source_inputs.scheduled_token_ids is not None:
            declared_ids = model_source_inputs.scheduled_token_ids
            if (self.runtime_inputs.symbolic_forward_source != "cuda_eager_inplace"
                    or self.runtime_inputs.model_execution_version != "vllm_0_30_0"
                    or model_source_inputs.request_ids != tuple(request.request_id for request in schedule.requests)
                    or len(declared_ids) != schedule.token_count
                    or any(token >= self.model.vocab_size for token in declared_ids)):
                raise ValueError("scheduled source IDs must match the current versioned target request order, rows and vocabulary")
            if embedding_token_ids is not None and embedding_token_ids != declared_ids:
                raise ValueError("scheduled source IDs conflict with known backend input IDs")
            embedding_token_ids = declared_ids
            if not self.runtime_inputs.graph_execution.enabled:
                if (model_source_inputs.retained_padding_token_ids
                        or model_source_inputs.fa3_full_capture_id is not None):
                    raise ValueError("eager source IDs cannot silently consume captured padding or capture identity")
                model_source_inputs = None
        framed_mtp = (self.runtime_inputs.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
            or (self.runtime_inputs.mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy"
                and isinstance(self.runtime_inputs.model_source_inputs, V32PaddedMTPInputs)))
        pd_prefill_target_only = (framed_mtp and self.deployment.mode == "pd_disagg"
            and self.deployment.worker_pools[schedule.pool_name].role == "prefill"
            and self.runtime_inputs.symbolic_forward_source in (
                "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            and all(r.mtp_proposer_mode == "disabled" and r.mtp_proposed_tokens == 0
                and r.mtp_verified_tokens == 0 and r.mtp_proposer_tokens == 0 for r in schedule.requests))
        pd_decode_target_only = (framed_mtp and self.deployment.mode == "pd_disagg"
            and self.deployment.worker_pools[schedule.pool_name].role == "decode"
            and self.runtime_inputs.symbolic_forward_source in (
                "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            and self.runtime_inputs.glm_target_num_speculative_tokens > 0
            and model_source_inputs is None and embedding_token_ids is not None
            and all(r.phase == "decode" and r.input_tokens == 0 and r.output_tokens == 1
                and r.raw_model_output_tokens == 1 and r.computed_tokens > 0
                and not r.resumed_from_preemption
                and r.computed_tokens + 1 + self.runtime_inputs.glm_target_num_speculative_tokens
                    > self.runtime_inputs.symbolic_source_max_model_len
                and r.mtp_proposer_mode == r.mtp_acceptance_source_kind == r.mtp_acceptance_source_id == "disabled"
                and not any((r.mtp_proposed_tokens, r.mtp_verified_tokens, r.mtp_padding_tokens,
                    r.mtp_accepted_tokens, r.mtp_proposer_tokens, r.mtp_next_proposed_tokens,
                    r.prefix_replay_tokens)) for r in schedule.requests))
        pd_target_only = pd_prefill_target_only or pd_decode_target_only
        if framed_mtp and not pd_target_only:
            if not isinstance(model_source_inputs, V32PaddedMTPInputs) or embedding_token_ids is None:
                raise ValueError("padded MTP execution requires explicit per-step source state and current target token IDs")
            proposer_depths = {request.mtp_proposer_tokens for request in schedule.requests
                if request.mtp_proposer_mode == "padded"}
            verify_only = (proposer_depths == {0}
                and any(request.mtp_verified_tokens for request in schedule.requests)
                and all(request.mtp_next_proposed_tokens == 0 for request in schedule.requests))
            if (model_source_inputs.request_ids != tuple(request.request_id for request in schedule.requests)
                    or len(model_source_inputs.draft_input_token_ids[0]) != schedule.token_count
                    or any(len(ids) != len(schedule.requests) for ids in model_source_inputs.draft_input_token_ids[1:])
                    or (proposer_depths != {len(model_source_inputs.draft_input_token_ids)} and not verify_only)):
                raise ValueError("padded MTP per-step source state disagrees with scheduled requests, padded rows or draft depth")
        step_inputs = {}
        if framed_mtp and (not pd_target_only or pd_decode_target_only):
            step_inputs.update(symbolic_source_prompt_lengths=None,
                symbolic_source_prompt_lengths_by_request=None)
        if pd_target_only:
            if model_source_inputs is not None:
                raise ValueError("target-only PD prefill cannot consume a drafter frame")
            step_inputs.update(mtp_draft_sampling_source=None, model_source_inputs=None,
                target_sampling_source="vllm_0_30_plain_greedy_bf16", symbolic_source_max_requests=None,
                glm_target_num_speculative_tokens=(0 if pd_prefill_target_only
                    else self.runtime_inputs.glm_target_num_speculative_tokens))
        if embedding_token_ids is not None:
            step_inputs["embedding_token_ids"] = embedding_token_ids
        if model_source_inputs is not None or self.runtime_inputs.model_source_inputs is not None:
            step_inputs["model_source_inputs"] = model_source_inputs
        runtime = (replace(self.runtime_inputs, **step_inputs) if step_inputs else self.runtime_inputs)
        if runtime.symbolic_forward_source == "cuda_eager_inplace" and runtime.target_sampling_source is None:
            raise NotImplementedError("static model-only source cannot execute complete Serving step; an explicit sampler is required")
        if schedule.pool_name in self._source_workspace_capacity_by_pool:
            runtime = replace(runtime, symbolic_workspace_initial_capacity_bytes=
                self._source_workspace_capacity_by_pool[schedule.pool_name])
        elif (runtime.symbolic_forward_source == "vllm_0_30_v32_bf16_sync_explicit_allreduce"
                and runtime.symbolic_workspace_initial_capacity_bytes is not None
                and len(self.deployment.worker_pools) > 1):
            # PCP PD can have different physical rank counts at its endpoints.
            # An explicitly global constructor tuple follows deployment pool
            # order, then each pool's ordered physical ranks; no zero/history
            # is synthesized for an endpoint absent from that declaration.
            capacities = runtime.symbolic_workspace_initial_capacity_bytes
            total = sum(len(pool.ranks) for pool in self.deployment.worker_pools.values())
            if len(capacities) == total:
                offset = 0
                for name, pool in self.deployment.worker_pools.items():
                    if name == schedule.pool_name:
                        runtime = replace(runtime, symbolic_workspace_initial_capacity_bytes=
                            capacities[offset:offset + len(pool.ranks)])
                        break
                    offset += len(pool.ranks)
        dspark_progress = self._prepare_dspark_progress(schedule, runtime.model_source_inputs, embedding_token_ids)
        if dspark_progress is not None and timeline_origin_provider is not None:
            raise ValueError("source progress cannot defer its predecessor clock")
        if dspark_progress is not None:
            device_floor_s = max(device_floor_s, dspark_progress[3])
        source_storage, source_graph, source_capture_active = None, None, False
        source_memory_context = None
        source_state_summary = None
        source_capture_mode = "NONE"
        source_capture_regions = []
        recorded_source_program_bound = False

        def observe_source(source, graph):
            nonlocal source_storage, source_graph, source_capture_active, source_capture_mode
            nonlocal recorded_source_program_bound, source_memory_context, source_state_summary
            from src.serving.kv_cache.activation_memory import symbolic_forward_storage_report
            from src.serving.workload.traced_forward import ForwardTrace, ForwardCollectiveGroup
            if source_storage is not None:
                raise RuntimeError("symbolic step executed its forward more than once")
            # Only bind() installs a recorded graph on these concrete sources.
            # A cache lookup can still return a program whose bind is rejected;
            # wrappers and freshly executed unbound sources are not receipts.
            recorded_source_program_bound = (
                isinstance(source, (ForwardTrace, ForwardCollectiveGroup))
                and source._recorded_graph is not None and source.graph is graph)
            source_storage = symbolic_forward_storage_report(source, graph,
                source_profile=runtime.symbolic_forward_source)
            if runtime.symbolic_forward_source == 'vllm_0_30_v4_dspark':
                source_storage['scope']='retained_source_history_known_storage_envelope_not_current_step_or_normal_worker_peak'
                source_storage['limitations']=(*source_storage['limitations'],'retained_history_not_a_per_step_activation_measurement')
            source_graph = graph
            if self.collect_source_state:
                from src.serving.kv_cache.activation_memory import source_forward_state_summary
                source_state_summary = source_forward_state_summary(source, graph)
            if self.execution_mode != "fast":
                from src.serving.kernel_lowering.memory_bindings import _snapshot_execution_memory
                if runtime.symbolic_forward_source == "vllm_0_30_v4_dspark":
                    # The proposer retains its complete source history. Snapshot
                    # its real writer chronology, then select only this round.
                    # Old producers remain readiness facts, never priced calls.
                    from src.serving.kernel_lowering.memory_bindings import TraceMemoryContext
                    history = _snapshot_execution_memory(source, source.graph)
                    source_memory_context = TraceMemoryContext(tuple(history._by_node[node.node_id]
                        for node in graph.nodes))
                else:
                    source_memory_context = _snapshot_execution_memory(source, graph)
            if runtime.graph_execution.enabled:
                from src.serving.kv_cache.activation_memory import source_capture_activation_peak
                traces = (source,) if isinstance(source, ForwardTrace) else source.traces
                if self.host_launch_profiles is not None:
                    source_orders = source.operator_order_by_rank
                for trace in traces:
                    captured_regions = []
                    for capture in trace.source_captures:
                        source_capture_active = True
                        if source_capture_mode != "NONE" and source_capture_mode != capture["runtime_mode"]:
                            raise RuntimeError("source ranks disagree on capture runtime mode")
                        source_capture_mode = capture["runtime_mode"]
                        if capture["end_event"] is None:
                            raise RuntimeError("source capture lost its actual forward boundary")
                        key = (schedule.pool_name, trace.rank, capture["key"])
                        if capture["source_stage"] is None:
                            named_ports = {**capture["inputs"], "hidden_states": capture["output"]}
                        else:
                            named_ports = {**{"input:" + role: port for role, port in capture["inputs"].items()},
                                **{"output:" + role: port for role, port in capture["outputs"].items()}}
                        ports = {role: {name: value for name, value in port.items() if name != "allocation_id"}
                            for role, port in named_ports.items()}
                        replay = key in self._source_capture_ports
                        if replay and self._source_capture_ports[key] != ports:
                            raise RuntimeError("source capture changed static input/output identity or typed extent")
                        self._source_capture_ports[key] = ports
                        backing_bytes = {}
                        for port in named_ports.values():
                            identity = port["allocation_id"]
                            if identity in backing_bytes and backing_bytes[identity] != port["bytes"]:
                                raise RuntimeError("captured aliases disagree on their backing capacity")
                            backing_bytes[identity] = port["bytes"]
                        if self.host_launch_profiles is not None:
                            start = capture["node_start"]
                            local_nodes = capture["model_node_ids"]
                            canonical_nodes = source_orders[trace.rank][start:start + len(local_nodes)]
                            if len(canonical_nodes) != len(local_nodes):
                                raise RuntimeError("source capture lost its canonical operator-order boundary")
                            canonical = dict(zip(local_nodes, canonical_nodes, strict=True))
                            for index, nodes in enumerate(capture["source_segments"]):
                                segment_nodes = tuple(canonical[node] for node in nodes)
                                from src.serving.workload.traced_forward import _source_bound_pool_names
                                segment_pools = _source_bound_pool_names(graph, segment_nodes)
                                if len(segment_pools) != 1:
                                    raise ValueError("a local source capture segment must belong to exactly one worker pool")
                                source_capture_regions.append((next(iter(segment_pools)), trace.rank,
                                    f"{capture['key']}:source-segment-{index}", segment_nodes))
                        peak, live = source_capture_activation_peak(trace, capture)
                        captured_regions.append({
                            **capture, "logical_replay": replay, "static_ports": ports,
                            "configured_capture_sizes": runtime.graph_execution.capture_sizes,
                            "configured_capture_sizes_unit": "model_input_token_rows",
                            "static_port_bytes": sum(backing_bytes.values()),
                            "known_capture_activation_peak_bytes": peak,
                            "capture_peak_live_allocation_ids": live,
                            "capture_peak_scope": "source_region_including_eager_regions_not_graph_allocator_pool",
                            "scope": "same_source_forward_not_compiler_pool_reservation_or_GPU_replay"})
                    if len(captured_regions) == 1:
                        source_storage["ranks"][str(trace.rank)]["source_capture"] = captured_regions[0]
                    elif captured_regions:
                        source_storage["ranks"][str(trace.rank)]["source_captures"] = tuple(captured_regions)

        started = time.perf_counter()
        from src.serving.workload.traced_models import build_symbolic_workload
        workload = build_symbolic_workload(schedule, self.model, self.deployment,
            launch_policy=self.launch_policy, kv_cache_events=kv_cache_events,
            runtime_inputs=runtime, hardware_by_id=self.hardware_by_id, trace_observer=observe_source,
            # GLM ordinary input-buffer and MTP proposer histories belong to
            # the run in every mode. Their optional structural templates are
            # enabled separately, only in fast mode, so detailed execution
            # preserves its complete source/call layout.
            source_program_cache=(self._source_program_cache
                if self.model.model_type != "kimi_k2" or self.execution_mode == "fast"
                or runtime.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy" else None))
        if source_graph is not workload or source_storage is None:
            raise RuntimeError("symbolic step lost the storage facts of its actual forward graph")
        self._phase_wall_time_s["workload_construction"] += (
            time.perf_counter() - started)
        components = self.execute_workload(
            workload,
            origin_time_s=origin_time_s,
            device_floor_s=device_floor_s,
            shared_resource_calendar=shared_resource_calendar,
            runtime_graph_mode=(source_capture_mode
                if runtime.graph_execution.enabled
                else _runtime_graph_mode(schedule, self.runtime_inputs.graph_execution)),
            runtime_prepare_events=runtime_prepare_events,
            recorded_source_bound=recorded_source_program_bound,
            source_memory_context=source_memory_context,
            source_launch_inputs=source_launch_inputs,
            source_capture_regions=(tuple(source_capture_regions)
                if source_capture_active and self.host_launch_profiles is not None else None),
            require_timeline_details=require_timeline_details,
            timeline_origin_provider=timeline_origin_provider,
            worker_request_state=worker_request_state,
        )
        if runtime.symbolic_workspace_initial_capacity_bytes is not None:
            # The synchronous source has no surviving external arena aliases
            # at a forward return. Rebind its manager's final capacity on the
            # next trace; keep only scalar state, not an old graph or tensor.
            pool = self.deployment.worker_pools[schedule.pool_name]
            capacities = tuple(source_storage["ranks"][str(rank)]["source_workspace_capacity_bytes"]["worker.workspace"]
                for rank in pool.ranks)
            if any(final < initial for initial, final in zip(
                    runtime.symbolic_workspace_initial_capacity_bytes, capacities, strict=True)):
                raise RuntimeError("source workspace capacity cannot shrink across forward calls")
            self._source_workspace_capacity_by_pool[schedule.pool_name] = capacities
        result = (components if source_storage is None else
            replace(components, source_activation_memory={**source_storage,
                **({"host_capture_timing": {"source_runtime_mode": source_capture_mode,
                    "step_host_coverage_mode": components.execution_timeline.runtime_graph_mode,
                    "scope": "source capture regions replay; every uncovered call remains eager; not compiler/provider acceptance"}}
                    if source_capture_active and self.host_launch_profiles is not None else {})}))
        if source_state_summary is not None:
            result = replace(result, source_state_summary=source_state_summary)
        result = self._commit_dspark_progress(dspark_progress, schedule.step_id, result, origin_time_s)
        self._recorded_source_program_bound = recorded_source_program_bound
        return result

    @_metadata_collection_scope()
    def execute_source_workload(self, source_builder, *, retire_source=None,
            shared_resource_calendar: ExecutionResourceCalendar | None = None,
            require_timeline_details: bool = False) -> ServingExecutionComponents:
        """Construct a real source graph under the common run-local executor.

        The factory returns its actual graph and placement; dynamic source
        inputs are consumed afresh. No runtime/provider identity is inferred.
        """
        if not callable(source_builder):
            raise TypeError("traced-source execution requires a callable source builder")
        if not self._traced_device_source:
            raise ValueError("source builder execution requires the explicit actual-source factory")
        if retire_source is not None and not callable(retire_source):
            raise TypeError("source retirement must be an explicit callable")
        started = time.perf_counter()
        built = source_builder()
        primary_error = None
        try:
            if (type(built) is not tuple or len(built) != 2
                    or not isinstance(built[0], WorkloadGraph) or not isinstance(built[1], DeploymentPlan)):
                raise TypeError("source builder must return its WorkloadGraph and DeploymentPlan")
            workload, deployment = built
            self._phase_wall_time_s["workload_construction"] += time.perf_counter() - started
            return self.execute_workload(workload, shared_resource_calendar=shared_resource_calendar,
                require_timeline_details=require_timeline_details, traced_deployment=deployment)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if retire_source is not None:
                try:
                    retire_source()
                except BaseException as cleanup_error:
                    if primary_error is not None:
                        raise primary_error from cleanup_error
                    raise

    @_metadata_collection_scope()
    def execute_workload(
        self,
        workload: WorkloadGraph,
        *,
        origin_time_s: float = 0.0,
        device_floor_s: float = 0.0,
        shared_resource_calendar: ExecutionResourceCalendar | None = None,
        runtime_graph_mode: str | None = None,
        runtime_graph_segments: tuple[
            RuntimeGraphSegmentDescriptor, ...
        ] | None = None,
        runtime_prepare_events: Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
        source_capture_regions: tuple | None = None,
        require_timeline_details: bool = False,
        recorded_source_bound: bool = False,
        source_memory_context=None,
        traced_deployment: DeploymentPlan | None = None,
        source_launch_inputs: Mapping[str, object] | None = None,
        timeline_origin_provider: Callable[[], tuple[float, float]] | None = None,
        worker_request_state=None,
    ) -> ServingExecutionComponents:
        if type(recorded_source_bound) is not bool:
            raise TypeError("recorded source binding flag must be boolean")
        if timeline_origin_provider is not None:
            if not callable(timeline_origin_provider):
                raise TypeError("timeline origin provider must be callable")
            if (isinstance(origin_time_s, bool) or isinstance(device_floor_s, bool)
                    or origin_time_s != 0.0 or device_floor_s != 0.0 or shared_resource_calendar is not None):
                raise ValueError("deferred timeline origins cannot override a supplied clock")
        if not isinstance(require_timeline_details, bool):
            raise TypeError("require_timeline_details must be boolean")
        materialize_details = self.materialize_timeline_details or require_timeline_details
        if traced_deployment is not None and (not self._traced_device_source
                or not isinstance(traced_deployment, DeploymentPlan)):
            raise ValueError("placement override requires the explicit actual-source factory")
        deployment = self.deployment if traced_deployment is None else traced_deployment
        if shared_resource_calendar is not None and not isinstance(shared_resource_calendar, ExecutionResourceCalendar):
            raise TypeError("shared resource calendar must be ExecutionResourceCalendar")
        if self._device_resource_order == "earliest_ready":
            if (origin_time_s != 0.0 or device_floor_s != 0.0
                    or self.host_launch_profiles is not None or self.host_runtime_prepare_profiles is not None
                    or self.runtime_prepare_event_builder is not None or runtime_prepare_events is not None
                    or runtime_graph_mode is not None or runtime_graph_segments is not None
                    or source_capture_regions is not None
                    or shared_resource_calendar is not None and shared_resource_calendar.reservations):
                raise ValueError("earliest_ready requires time-zero device-only execution with an empty calendar")
        if not isinstance(workload, WorkloadGraph):
            raise TypeError(
                "Serving execution workload must be WorkloadGraph")
        from src.serving.workload.traced_models import _retained_full_execution_context
        source_context = _retained_full_execution_context(workload)
        bound_annotations = (source_context is not None and self.runtime_inputs is not None
            and source_context == (self.hardware_by_id[source_context[0].hardware_id], self.runtime_inputs.graph_execution,
                self.runtime_inputs.norm_source_batch_invariant, self.runtime_inputs.bf16_gemm_model, self.runtime_inputs.bf16_expert_model))
        if self.runtime_inputs is not None and self.runtime_inputs.bf16_expert_model is not None:
            from src.serving.workload.model_config import is_validated_qwen3_moe_ep2_model_scope
            if (self.launch_policy != "analytical_launch" or not is_validated_qwen3_moe_ep2_model_scope(self.model)
                    or self.model.compute_dtype != "bf16" or self.model.weight_dtype != "bf16"
                    or any(hardware.hardware_id != "nvidia_h200" for hardware in self.hardware_by_id.values())
                    or any(pool.parallel_plan.config.expert_parallel_size != 2 for pool in deployment.worker_pools.values())):
                raise ValueError("BF16 expert reference requires the declared H200 Qwen3-MoE BF16 EP2 analytical source")
            # Captured experts keep the existing aggregate boundary: there
            # are no intermediate external consumers or internal Host
            # submissions. Expand only the uncaptured source recipe.
            if not bound_annotations and runtime_graph_mode not in ("FULL", "PIECEWISE"):
                workload = workload.with_nodes(node.with_attributes({**node.attributes,
                    "bf16_expert_model": self.runtime_inputs.bf16_expert_model})
                    if node.kind is WorkloadKind.FUSED_MOE else node for node in workload.nodes)
        if self.runtime_inputs is not None and self.runtime_inputs.bf16_gemm_model is not None:
            if (self.launch_policy != "analytical_launch"
                    or self.model.compute_dtype != "bf16" or self.model.weight_dtype != "bf16"
                    or any(not hardware.supports_source_contract("h200_partial_source_programs")
                        for hardware in self.hardware_by_id.values())):
                raise ValueError("tiled BF16 reference requires homogeneous BF16 analytical H200 execution")
        if self.runtime_inputs is not None and self.runtime_inputs.bf16_gemm_model is not None and not bound_annotations:
            # Keep source identity, repeats, tensors, dependencies and physical
            # Graph rows. Mixed-output routers and quantized programs retain
            # their own existing model; the option declares ordinary BF16 only.
            workload = workload.with_nodes(
                node.with_attributes({**node.attributes, "bf16_gemm_model": self.runtime_inputs.bf16_gemm_model})
                if node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD)
                and node.dtype == "bf16" and node.attributes["weight_dtype"] == "bf16"
                and node.attributes["weight_group_size"] == 0 and node.attributes["has_bias"] is False
                and "weight_quantization" not in node.attributes
                and ("output_dtype" not in node.attributes or node.attributes["output_dtype"] in (None, "bf16"))
                else node for node in workload.nodes)
        if (self.runtime_inputs is not None
                and self.runtime_inputs.norm_source_batch_invariant is False and not bound_annotations):
            norm_inputs = {
                node.node_id: {
                    "shape": dict(node.shape),
                    "attributes": {
                        "model_execution_version": node.attributes["model_execution_version"],
                        "source_input_layout": node.attributes["source_input_layout"],
                        "source_batch_invariant": False,
                    },
                }
                for node in workload.nodes
                if node.kind is WorkloadKind.NORM
                and node.variant in ("rmsnorm", "add_rmsnorm")
            }
            if source_launch_inputs is not None:
                if norm_inputs.keys() & source_launch_inputs.keys():
                    raise ValueError("runtime Norm declaration conflicts with per-step source launch inputs")
                norm_inputs.update(source_launch_inputs)
            source_launch_inputs = norm_inputs
        if source_capture_regions is not None and (self.host_launch_profiles is None or runtime_graph_segments is not None
                or runtime_graph_mode not in ("FULL", "PIECEWISE")):
            raise ValueError("source capture Host regions require their own explicit captured-mode Host segment binding")
        started = time.perf_counter()
        # This complete raw-source MoE domain need not have a recorded-prefix
        # cache hit. lower_workload still validates every EP port and routing
        # record, then proves equal physical contracts and the exact recurrence
        # DAG before folding; nonuniform layers retain their actual calls.
        kernel_plan = lower_workload(
            workload,
            deployment,
            self.hardware_by_id,
            self._lowering_memo,
            source_launch_inputs=source_launch_inputs,
            source_memory_context=source_memory_context,
            compact_source_layers=(source_launch_inputs is None and self._lowering_memo is not None and not materialize_details
                and self.runtime_inputs is not None
                and (self.model.num_experts == 0 or recorded_source_bound
                    or (self.model.model_type == "qwen3_moe" and self.model.num_experts == 128
                        and self.model.compute_dtype == self.model.weight_dtype == "bf16"
                        and self.model.num_hidden_layers == 48 and self.model.hidden_size == 2048
                        and self.model.moe_intermediate_size == 768
                        and self.runtime_inputs.model_execution_version == "vllm_0_11_0"
                        and self.runtime_inputs.expert_routing.source_kind == "profile_observation"
                        and self.runtime_inputs.expert_routing.distribution == "model_profile_layer_mean_v1"
                        and len(self.deployment.worker_pools) == 1
                        and all(pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 2
                            and pool.parallel_plan.config.tensor_parallel_size == 2
                            and pool.parallel_plan.config.enable_expert_parallel
                            and pool.parallel_plan.config.expert_parallel_size == 2
                            and pool.parallel_plan.config.is_rank_local
                            and all(value == 1 for value in (pool.parallel_plan.config.pipeline_parallel_size,
                                pool.parallel_plan.config.sequence_parallel_size, pool.parallel_plan.config.context_parallel_size,
                                pool.parallel_plan.config.data_parallel_size))
                            for pool in self.deployment.worker_pools.values())))
                and self.deployment.mode == "colocated"),
        )
        if source_memory_context is not None:
            from src.serving.kernel_lowering.memory_bindings import bind_trace_memory
            kernel_plan = bind_trace_memory(kernel_plan, source_memory_context)
        self._phase_wall_time_s["kernel_lowering"] += (
            time.perf_counter() - started)
        started = time.perf_counter()
        prediction_memo = KernelPredictionMemo() if self.fresh_kernel_predictions else self._prediction_memo
        predictions = predict_kernel_plan(
            kernel_plan,
            self.hardware_by_id,
            self.prediction_method,
            prediction_memo,
        )
        if self.fresh_kernel_predictions:
            self._fresh_prediction_hits += prediction_memo.hit_count
            self._fresh_prediction_misses += prediction_memo.miss_count
        predictions = self._add_side_stream_fork_node_boundary(
            kernel_plan, predictions)
        self._phase_wall_time_s["kernel_prediction"] += (
            time.perf_counter() - started)
        started_order = time.perf_counter()
        if self._device_resource_order == "earliest_ready":
            kernel_plan = with_earliest_ready_resource_order(kernel_plan, predictions)
        order_wall_s = time.perf_counter() - started_order
        self._resource_order_wall_time_s += order_wall_s
        self._phase_wall_time_s["timeline_construction"] += order_wall_s
        normalized_runtime_prepare_events = None
        runtime_prepare_events_token = None
        if self.runtime_prepare_event_builder is not None:
            if runtime_prepare_events is not None:
                raise ValueError(
                    "explicit runtime prepare events conflict with context builder")
            if runtime_graph_mode is None:
                raise ValueError(
                    "runtime prepare builder requires an explicit runtime Graph mode")
            # Source occurrence rules are authored against the eager program;
            # the execution timeline discards events on occurrences captured
            # by a runtime Graph segment and charges the eager remainder.
            state_mechanisms = ()
            if self.host_runtime_prepare_profiles is not None:
                hardware_ids = {call.kernel_spec.hardware_id for call in kernel_plan.calls}
                if len(hardware_ids) != 1:
                    raise ValueError("runtime prepare builder requires one plan hardware")
                profile = self.host_runtime_prepare_profiles[next(iter(hardware_ids))]
                state_mechanisms = tuple(sorted(name for name, _ in profile.service_time_s_by_mechanism
                    if name in WORKER_REQUEST_STATE_MECHANISMS))
            (
                runtime_prepare_events,
                normalized_runtime_prepare_events,
                runtime_prepare_events_token,
            ) = _cached_runtime_prepare_events(
                self._runtime_prepare_events_cache,
                self.runtime_prepare_event_builder,
                kernel_plan,
                materialize_details,
                worker_request_state,
                state_mechanisms,
            )
        started = time.perf_counter()
        plan_structure_runtime_graph_segments = None
        if self.host_launch_profiles is not None:
            if runtime_graph_mode is None:
                raise ValueError(
                    "Host launch timing requires an explicit runtime Graph mode")
            if runtime_graph_segments is None:
                piecewise_eager_kinds = (
                    self.runtime_inputs.graph_execution
                    .piecewise_eager_workload_kinds
                )
                plan_hardware_ids = {
                    call.kernel_spec.hardware_id
                    for call in kernel_plan.calls
                }
                if len(plan_hardware_ids) != 1:
                    raise ValueError(
                        "runtime Graph segment production requires one hardware")
                host_profile = self.host_launch_profiles[
                    next(iter(plan_hardware_ids))]
                sync_before_replay = (
                    host_profile.runtime_name == "CANN"
                    and host_profile.framework_name == "torch_npu")
                if source_capture_regions is not None:
                    if self.execution_mode in ("accelerated", "fast"):
                        source_prefix = (_common_id_prefix(kernel_plan)
                            if not materialize_details else None)
                        runtime_graph_segments, runtime_graph_mode = _cached_source_capture_host_segments(
                            self._source_capture_segments_cache, kernel_plan, source_capture_regions,
                            sync_before_replay=sync_before_replay, normalize_prefix=source_prefix)
                        if source_prefix is not None:
                            # The source cache validated every rank, source
                            # boundary and repeated occurrence in its exact
                            # key. Consume its normalized result directly.
                            plan_structure_runtime_graph_segments = runtime_graph_segments, source_prefix
                            runtime_graph_segments = None
                    else:
                        runtime_graph_segments, runtime_graph_mode = _source_capture_host_segments(
                            kernel_plan, source_capture_regions, sync_before_replay=sync_before_replay)
                elif materialize_details:
                    runtime_graph_segments = build_runtime_graph_segments(
                        workload,
                        kernel_plan,
                        runtime_graph_mode,
                        piecewise_eager_kinds,
                        sync_before_replay=sync_before_replay,
                    )
                else:
                    # The summary builder works in the plan-structure
                    # identifier space, so take the segments there directly
                    # instead of translating them into plan space and back.
                    segments, prefix = (
                        build_plan_structure_runtime_graph_segments(
                            workload,
                            kernel_plan,
                            runtime_graph_mode,
                            piecewise_eager_kinds,
                            sync_before_replay=sync_before_replay,
                        ))
                    if prefix is None:
                        runtime_graph_segments = segments
                    else:
                        plan_structure_runtime_graph_segments = (
                            segments, prefix)
        if timeline_origin_provider is not None:
            # Workload, launch and prices are independent of the EngineCore
            # clock. A CPU worker may prepare them early, then wait here for
            # the FIFO owner to supply the actual Host start and Device floor.
            wait_started = time.perf_counter()
            origins = timeline_origin_provider()
            started += time.perf_counter() - wait_started
            if type(origins) is not tuple or len(origins) != 2:
                raise TypeError("timeline origin provider must return a Host/Device pair")
            origin_time_s, device_floor_s = origins
        execution_timeline = build_execution_timeline(
            kernel_plan,
            predictions,
            origin_time_s=origin_time_s,
            device_floor_s=device_floor_s,
            shared_resource_calendar=shared_resource_calendar,
            host_launch_profiles=self.host_launch_profiles,
            host_runtime_prepare_profiles=(
                self.host_runtime_prepare_profiles),
            runtime_prepare_events=runtime_prepare_events,
            runtime_graph_mode=(
                runtime_graph_mode
                if self.host_launch_profiles is not None else None),
            runtime_graph_segments=runtime_graph_segments,
            materialize_timeline_details=materialize_details,
            normalized_runtime_prepare_events=(
                normalized_runtime_prepare_events),
            runtime_prepare_events_token=runtime_prepare_events_token,
            plan_structure_runtime_graph_segments=(
                plan_structure_runtime_graph_segments),
        )
        self._phase_wall_time_s["timeline_construction"] += (
            time.perf_counter() - started)
        self._executed_workloads += 1
        return ServingExecutionComponents(
            workload,
            kernel_plan,
            predictions,
            execution_timeline,
        )

    def cache_stats(self) -> dict[str, object]:
        if self.execution_mode == "reference":
            return {
                "execution_mode": "reference",
                "kernel_lowering": None,
                "kernel_prediction": None,
            }
        return {
            "execution_mode": self.execution_mode,
            "kernel_lowering": self._memo_stats(self._lowering_memo),
            "kernel_prediction": ({"entries": 0, "hits": self._fresh_prediction_hits,
                "misses": self._fresh_prediction_misses} if self.fresh_kernel_predictions
                else self._memo_stats(self._prediction_memo)),
        }

    def performance_stats(self) -> dict[str, object]:
        result = {
            "executed_workloads": self._executed_workloads,
            "phase_wall_time_s": dict(self._phase_wall_time_s),
            "parallel_worker_count": 0,
            "parallel_step_count": 0,
            "parallel_execution_wall_time_s": 0.0,
            "parallel_worker_aggregate_phase_s": {
                name: 0.0 for name in self._phase_wall_time_s
            },
        }
        if self._traced_device_source:
            result["resource_ordering_wall_time_s"] = self._resource_order_wall_time_s
            if self._lowering_memo is not None:
                result["compiled_source_lowering"] = {
                    "physical_class_hits": self._lowering_memo.module_contract_hits,
                    "physical_class_misses": self._lowering_memo.module_contract_misses,
                    "call_recipe_hits": self._lowering_memo.call_template_hits,
                    "call_recipe_misses": self._lowering_memo.call_template_misses}
        return result

    @staticmethod
    def _memo_stats(memo: object) -> dict[str, int]:
        if memo is None:
            raise RuntimeError("accelerated Serving execution lost a memo")
        return {
            "entries": memo.entry_count,
            "hits": memo.hit_count,
            "misses": memo.miss_count,
        }
