"""Composition root for no-GPU serving simulation."""

from contextlib import ExitStack, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
import math
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import sys
import time
from typing import Callable, Mapping, Optional, Tuple

from src import _platform_compat
from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    pd_kv_transfer_rank_pairs,
)
from src.serving.deployment.validated_scope import (
    is_validated_colocated_tp2_host_kv_scope,
    is_validated_standalone_af_tp2_scope, is_validated_standalone_af_ep2_scope,
    validated_target_emission_scope,
)
from src.serving.kernel_lowering.execution_timeline import (
    call_reserved_resource_ids,
    experiment_cache_scope,
    ExecutionResourceCalendar,
)
from src.serving.kernel_lowering.kv_offload_completion import (
    build_kv_cache_transfer_completion,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.kv_cache.kv_lifecycle_replay import KVLifecycleReplay
from src.serving.kv_cache.kv_cache_tier_manager import (
    KVCacheTierManager,
    KVCacheTransferCompletion,
)
from src.serving.metrics.serving_metrics import ServingMetrics, compute_serving_metrics
from src.serving.execution import (
    ServingExecutionContext, bind_model_source_input_frames, _metadata_collection_scope,
)
from src.serving.request_gen.request_stream import (
    RequestStream,
    request_stream_sha256,
)
from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.schedule.backend_protocol import (
    BackendConfig,
    SchedulerBackend,
    TailBatchSchedulerBackend,
    backend_config_sha256,
    backend_phase,
    scheduler_effective_prefix_hit_tokens,
)
from src.serving.schedule.multi_pool_engine import (
    HandoffDispatch,
    HandoffTransfer,
    LeastQueuedTokensRouter,
    run_multi_pool_schedule,
)
from src.serving.schedule.schedule_step import BackendStep, ScheduledRequest
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.schedule.schedule_trace import (
    ScheduleTrace,
    ScheduleTraceEntry,
    TransferTraceEntry,
)
from src.serving.deployment.validated_scope import is_validated_tp1_tp2_multi_pool_scope, is_gqa_pd_source_scope, is_qwen3_moe_dp_sp_source_scope
from src.serving.workload.kv_transfer_workload import build_pd_kv_transfer_workload, validate_pd_kv_layout
from src.serving.schedule.schedule_step import is_non_mtp_async_terminal_placeholder
from src.serving.workload.model_config import (
    is_deepseek_v4_parallel_model_scope,
    ModelConfig,
    is_validated_llama_tp2_model_scope,
    is_validated_qwen3_moe_ep2_model_scope,
)
from src.serving.workload.runtime_inputs import RuntimeInputs, V32PaddedMTPInputs
from src.serving.workload.graph_padding import VLLMDenseGraphInputs
from src.serving.workload.workload_graph import WorkloadGraph
from src.serving.kv_cache.activation_memory import SourceActivationMemoryAccumulator
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRequestPathProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    validate_serving_timing_mode,
)


_METADATA_COLLECTION_WINDOW_STEPS = 128


@contextmanager
def _managed_backend(backend, *, censored_observation=False):
    """Close one backend without replacing a primary simulation failure."""

    try:
        yield backend
    except BaseException as primary_error:
        primary_traceback = primary_error.__traceback__
        try:
            if censored_observation:
                backend.abort_observation()
            else:
                backend.close()
        except BaseException:
            # Raising the primary error while handling the close failure keeps
            # both tracebacks, with the simulation failure as the final error.
            raise primary_error.with_traceback(primary_traceback)
        raise
    else:
        if censored_observation:
            backend.abort_observation()
        else:
            backend.close()
from src.serving.workload.workload_graph import WorkloadKind


def _assert_expert_routing_replay_coverage(
    runtime_inputs: RuntimeInputs,
    simulation_steps: Tuple["SimulationStep", ...],
) -> None:
    invocation_ids = set()
    for simulation_step in simulation_steps:
        for node in simulation_step.workload.nodes:
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            if "routing_provenance" not in node.attributes:
                raise ValueError(
                    f"FUSED_MOE node {node.node_id!r} lacks routing provenance")
            provenance = node.attributes["routing_provenance"]
            if not isinstance(provenance, Mapping):
                raise TypeError(
                    "FUSED_MOE routing provenance must be a mapping")
            if "moe_invocation_id" not in provenance:
                raise ValueError(
                    "FUSED_MOE routing provenance lacks moe_invocation_id")
            invocation_id = provenance["moe_invocation_id"]
            if not isinstance(invocation_id, str) or not invocation_id:
                raise ValueError(
                    "FUSED_MOE routing invocation ID must be non-empty")
            invocation_ids.add(invocation_id)
    runtime_inputs.expert_routing.assert_replay_consumed(
        tuple(sorted(invocation_ids)))


def host_request_path_profile(
    config: "SimulationConfig",
) -> HostRequestPathProfile | None:
    """Resolve the single Host front end that admits this deployment's requests.

    A client request enters exactly one Host process tree, so the per-request
    admission and delivery costs belong to one profile.  A deployment that
    spans distinct Hosts has no single front end and is therefore out of this
    mechanism's domain.
    """

    profiles = config.host_request_path_profiles
    if profiles is None:
        return None
    distinct = {profile.hardware_id: profile for profile in profiles.values()}
    if len(distinct) != 1:
        raise NotImplementedError(
            "the per-request Host request path needs one ingress Host; this "
            f"deployment spans {sorted(distinct)}")
    return next(iter(distinct.values()))


def _first_token_delivery_s(config: "SimulationConfig"):
    """Host cost of putting one produced token on the client connection.

    Returned as a callable of the number of outputs the producing step
    carries, because the runtime walks the delivery path once per output of
    that step rather than once per step.

    ``host_request_path_profiles`` is a schema-declared optional field whose
    two states are both validated in :meth:`SimulationConfig.__post_init__`:
    when present it must cover the deployment hardware exactly, and
    ``device_only`` rejects it outright.  Absent therefore means one specific,
    checked thing -- this run models no Host request path, which is the
    engine-visible boundary the device-only and launch-only timing modes are
    defined at, and the domain a hardware without a published profile falls
    back to.  Zero is that boundary's exact value, not a stand-in for a
    missing measurement: a profile that exists always charges a positive,
    measured service time.
    """

    profile = host_request_path_profile(config)
    if profile is None:
        return 0.0
    return profile.first_token_delivery_service_time_s


def scheduler_visible_requests(
    requests: RequestStream,
    profile: HostRequestPathProfile | None,
) -> RequestStream:
    """Delay Scheduler visibility by the measured per-request Host admission cost.

    The canonical arrival stream keeps the client submission times that the
    latency metrics are measured against; only the copy handed to the Scheduler
    carries the Host admission delay, which is what the real runtime does by
    reaching ``Scheduler.add_request`` after the entrypoint, tokenizer, input
    processor and IPC hop have run.
    """

    if not isinstance(requests, RequestStream):
        raise TypeError("Scheduler visibility requires a RequestStream")
    if profile is None:
        return requests
    return RequestStream(
        replace(
            request,
            arrival_time_s=(
                request.arrival_time_s
                + profile.admission_service_time_s(request.input_tokens)),
        )
        for request in requests
    )


def _closed_loop_clients(
        config: "SimulationConfig") -> ClosedLoopClients | None:
    """Build the closed-loop arrival process for one configured run."""

    if config.closed_loop_client_count is None:
        return None
    profile = host_request_path_profile(config)
    if profile is None:
        arrival_transform = None
    else:
        def arrival_transform(request):
            return replace(
                request,
                arrival_time_s=(
                    request.arrival_time_s
                    + profile.admission_service_time_s(request.input_tokens)))
    return ClosedLoopClients(
        config.requests,
        config.closed_loop_client_count,
        arrival_transform,
        successor_policy=config.closed_loop_successor_policy,
    )


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    deployment: DeploymentPlan
    requests: RequestStream
    launch_policy: str
    prediction_method: str
    runtime_inputs: RuntimeInputs
    authorized_h200_pd_mtp_sweep: bool = False
    execution_mode: str = "accelerated"
    step_retention: str = "full"
    execution_workers: int = 0
    timing_mode: str = "device_only"
    host_launch_profiles: Mapping[str, HostLaunchProfile] | None = None
    host_runtime_prepare_profiles: Mapping[
        str, HostRuntimePrepareProfile] | None = None
    host_request_path_profiles: Mapping[
        str, HostRequestPathProfile] | None = None
    runtime_prepare_event_builder: Callable[
        [object], Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
        ]
    ] | None = None
    closed_loop_client_count: int | None = None
    model_source_inputs_by_step: Mapping[str, V32PaddedMTPInputs | VLLMDenseGraphInputs | None] | None = None
    collect_source_state: bool = False
    record_execution_timeline: bool = False
    vllm_executor_backend: str | None = None
    closed_loop_successor_policy: str = "per_client"
    observation_end_time_s: float | None = None
    fresh_kernel_predictions: bool = False

    def __post_init__(self) -> None:
        if type(self.fresh_kernel_predictions) is not bool:
            raise TypeError("fresh kernel predictions flag must be boolean")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("SimulationConfig model must be ModelConfig")
        if not isinstance(self.deployment, DeploymentPlan):
            raise TypeError("SimulationConfig deployment must be DeploymentPlan")
        if not isinstance(self.requests, RequestStream):
            raise TypeError("SimulationConfig requests must be RequestStream")
        if self.observation_end_time_s is not None:
            value = self.observation_end_time_s
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("observation end must be finite and positive")
            if self.step_retention != "streaming_aggregate" or self.timing_mode != "device_only":
                raise NotImplementedError("finite observations require device-only streaming retention")
        if self.closed_loop_successor_policy not in ("per_client", "next_request"):
            raise ValueError("closed-loop successor policy must be per_client or next_request")
        if self.closed_loop_client_count is None and self.closed_loop_successor_policy != "per_client":
            raise ValueError("closed-loop successor policy requires a client count")
        if self.closed_loop_client_count is not None:
            if (isinstance(self.closed_loop_client_count, bool)
                    or not isinstance(self.closed_loop_client_count, int)
                    or self.closed_loop_client_count < 1):
                raise ValueError(
                    "closed-loop client count must be a positive integer")
        if not isinstance(self.runtime_inputs, RuntimeInputs):
            raise TypeError(
                "SimulationConfig runtime_inputs must be RuntimeInputs")
        from src.serving.workload.traced_models import bind_ordinary_forward_source
        object.__setattr__(self, "runtime_inputs", bind_ordinary_forward_source(
            self.model, self.deployment, self.runtime_inputs))
        prompt_lengths = self.runtime_inputs.symbolic_source_prompt_lengths_by_request
        if prompt_lengths is not None:
            expected = {request.request_id: request.input_tokens
                for request in self.requests}
            if dict(prompt_lengths) != expected:
                raise ValueError("source prompt lengths by request must exactly match the request stream")
        object.__setattr__(self, "model_source_inputs_by_step",
            bind_model_source_input_frames(self.runtime_inputs, self.model_source_inputs_by_step))
        if not isinstance(self.authorized_h200_pd_mtp_sweep, bool):
            raise TypeError("H200 PD MTP sweep authorization must be boolean")
        self.runtime_inputs.expert_routing.assert_request_stream_sha256(
            request_stream_sha256(self.requests))
        if self.launch_policy not in ("injected_launch", "analytical_launch"):
            raise ValueError(f"unknown launch_policy: {self.launch_policy!r}")
        if self.prediction_method not in (
                "midlayer", "roofline", "roofline_span_floor", "pipeweave"):
            raise ValueError(
                f"unsupported Serving prediction method: "
                f"{self.prediction_method!r}")
        if self.execution_mode not in ("accelerated", "reference", "fast"):
            raise ValueError(
                f"unknown Serving execution mode: {self.execution_mode!r}")
        if self.step_retention not in ("full", "streaming_aggregate"):
            raise ValueError(
                f"unknown Serving step retention: {self.step_retention!r}")
        if (isinstance(self.execution_workers, bool)
                or not isinstance(self.execution_workers, int)
                or self.execution_workers < 0):
            raise ValueError(
                "Serving execution_workers must be a non-negative integer")
        if type(self.collect_source_state) is not bool:
            raise TypeError("collect_source_state must be boolean")
        if not isinstance(self.record_execution_timeline, bool):
            raise TypeError("record_execution_timeline must be boolean")
        if self.vllm_executor_backend not in (None, "uni", "mp"):
            raise ValueError("vLLM executor backend must be 'uni', 'mp' or absent")
        validate_serving_timing_mode(self.timing_mode)
        hardware_ids = {
            pool.hardware_id for pool in self.deployment.worker_pools.values()
        }
        if self.vllm_executor_backend is not None:
            h200_v030 = (hardware_ids == {"nvidia_h200"}
                and self.runtime_inputs.model_execution_version == "vllm_0_30_0")
            if hardware_ids != {"ascend_910b"} and not h200_v030:
                raise ValueError(
                    "explicit vLLM executor backend supports 910B or H200 vLLM0.30")
            if (self.vllm_executor_backend == "uni"
                    and any(len(pool.ranks) != 1
                        for pool in self.deployment.worker_pools.values())):
                raise ValueError("UniProcExecutor requires one rank per worker pool")
        requires_host_profiles = self.timing_mode != "device_only"
        if not requires_host_profiles:
            if (self.host_launch_profiles is not None
                    or self.host_runtime_prepare_profiles is not None
                    or self.host_request_path_profiles is not None
                    or self.runtime_prepare_event_builder is not None):
                raise ValueError(
                    "device_only cannot consume Host timing profiles")
        if requires_host_profiles:
            if (not isinstance(self.host_launch_profiles, Mapping)
                    or set(self.host_launch_profiles) != hardware_ids):
                raise ValueError(
                    "Host timing modes require exactly one profile per "
                    f"deployment hardware: {sorted(hardware_ids)}")
            for hardware_id, profile in self.host_launch_profiles.items():
                if (not isinstance(profile, HostLaunchProfile)
                        or profile.hardware_id != hardware_id):
                    raise ValueError(
                        "Host launch profile mapping disagrees with hardware ID")
        if ((self.host_runtime_prepare_profiles is None)
                != (self.runtime_prepare_event_builder is None)):
            raise ValueError(
                "runtime prepare profiles and event builder must be supplied together")
        if self.host_runtime_prepare_profiles is not None:
            if not requires_host_profiles:
                raise AssertionError("device_only runtime prepare profiles escaped")
            if (not isinstance(self.host_runtime_prepare_profiles, Mapping)
                    or set(self.host_runtime_prepare_profiles) != hardware_ids):
                raise ValueError(
                    "runtime prepare profiles must cover deployment hardware exactly")
            for hardware_id, profile in self.host_runtime_prepare_profiles.items():
                if (not isinstance(profile, HostRuntimePrepareProfile)
                        or profile.hardware_id != hardware_id):
                    raise ValueError(
                        "runtime prepare profile mapping disagrees with hardware ID")
        if self.host_request_path_profiles is not None:
            if not requires_host_profiles:
                raise AssertionError("device_only request path profiles escaped")
            if (not isinstance(self.host_request_path_profiles, Mapping)
                    or set(self.host_request_path_profiles) != hardware_ids):
                raise ValueError(
                    "Host request path profiles must cover deployment "
                    "hardware exactly")
            for hardware_id, profile in self.host_request_path_profiles.items():
                if (not isinstance(profile, HostRequestPathProfile)
                        or profile.hardware_id != hardware_id):
                    raise ValueError(
                        "Host request path profile mapping disagrees with "
                        "hardware ID")
        if (self.execution_mode in ("accelerated", "fast")
                and self.runtime_inputs.expert_routing.source_kind
                == "replay_observation"):
            raise NotImplementedError(
                "per-layer expert routing replay requires reference execution")


def validate_standalone_af_ep2_scope(config: SimulationConfig) -> None:
    """Pin the independently exercised standalone AF + EP2 run contract."""

    if not isinstance(config, SimulationConfig):
        raise TypeError(
            "standalone AF+EP2 validation requires SimulationConfig")
    if not is_validated_standalone_af_ep2_scope(
            config.deployment, config.model):
        raise NotImplementedError(
            "standalone AF+EP2 requires the exact two-layer BF16 Qwen3MoE "
            "fixture on one nvidia_a800, Attention ranks 0/1 with TP=2, "
            "FFN ranks 2/3 with TP=EP=2, no KV tier, and bidirectional "
            "NVLink analytical Roofline activation routes")
    virtual = (config.runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
        and config.runtime_inputs.model_execution_version == "vllm_0_30_0")
    if (config.launch_policy != "analytical_launch"
            or config.prediction_method != ("roofline" if virtual else "midlayer")):
        raise NotImplementedError(
            "standalone AF+EP2 requires analytical launch and the pinned midlayer "
            "or explicitly declared virtual Roofline prediction method")
    if config.runtime_inputs.expert_routing.source_kind != (
            "explicit_assumption"):
        raise NotImplementedError(
            "standalone AF+EP2 requires seeded-assumption expert routing")
    if (config.runtime_inputs.attention_kv_layout != "paged_varlen"
            or config.runtime_inputs.attention_kv_distribution != "uniform"):
        raise NotImplementedError(
            "standalone AF+EP2 requires paged-varlen uniformly distributed "
            "Attention KV")
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in config.requests
    )
    if request_shapes != ((0.0, 4, 2, 0),):
        raise NotImplementedError(
            "standalone AF+EP2 currently requires exactly one request "
            "shaped 4->2 with zero Prefix hit at time zero")


def _validate_standalone_af_ep2_backend_scope(
    config: SimulationConfig,
    backends: Mapping[str, SchedulerBackend],
) -> None:
    """Bind the AF+EP2 slice to its pinned Scheduler and KV geometry."""

    from src.serving.backend.vllm import VLLMBackend
    from src.serving.backend.vllm_scheduler_process import (
        VLLMSchedulerRuntimeInfo,
    )

    attention_pools = tuple(
        pool for pool in config.deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    if len(attention_pools) != 1:
        raise AssertionError(
            "standalone AF+EP2 config validation lost its Attention pool")
    pool = attention_pools[0]
    backend = backends[pool.name]
    virtual = (config.runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
        and config.runtime_inputs.model_execution_version == "vllm_0_30_0")
    if not isinstance(backend, VLLMBackend):
        raise NotImplementedError(
            "standalone AF+EP2 requires the pinned vLLM Scheduler backend")
    backend_scope = (
        backend.config.max_batched_tokens,
        backend.config.max_requests,
        backend.config.num_kv_blocks,
        backend.config.kv_block_size_tokens,
        backend.config.kv_bytes_per_token,
    )
    # 0.30 get_kv_cache_configs reserves BlockPool's null block before
    # checking max_model_len. Keep eight usable pages (128 tokens), with
    # nine physical blocks; the pinned 0.11 contract remains unchanged.
    expected_blocks = 9 if virtual else 8
    if backend_scope != (8, 1, expected_blocks, 16, 512):
        raise NotImplementedError(
            "standalone AF+EP2 requires Scheduler/KV capacity "
            f"batch=8, max_requests=1, blocks={expected_blocks}, block_tokens=16, and "
            "512 bytes/token/rank")
    runtime = backend.runtime_info
    if virtual:
        from src.serving.backend.vllm_scheduler_process import VLLM030SchedulerProcess
        if (not isinstance(backend._control_plane, VLLM030SchedulerProcess)
                or runtime["version"] != "0.30.0" or runtime["model_executed"] or runtime["weights_loaded"]
                or runtime["gpu_worker_processes_started"] != 0
                or runtime["native_allocation"]["num_kv_blocks"] != backend.config.num_kv_blocks):
            raise ValueError("virtual AF EP2 requires its guarded real CPU0.30 Scheduler and native allocation")
        return
    if not isinstance(runtime, VLLMSchedulerRuntimeInfo):
        raise TypeError(
            "standalone AF+EP2 requires pinned vLLM runtime provenance")
    runtime_scope = (
        runtime.version,
        runtime.source_commit,
        runtime.model_config_sha256,
        runtime.requested_dtype,
        runtime.effective_dtype,
        runtime.requested_tensor_parallel_size,
        runtime.effective_tensor_parallel_size,
        runtime.requested_pipeline_parallel_size,
        runtime.effective_pipeline_parallel_size,
        runtime.requested_expert_parallel,
        runtime.effective_expert_parallel,
        runtime.scheduler_pipeline_stage_rank,
        runtime.scheduler_pipeline_stage_layer_start,
        runtime.scheduler_pipeline_stage_layer_end,
        runtime.scheduler_kv_cache_num_heads_per_rank,
        runtime.scheduler_kv_cache_page_size_bytes,
        runtime.max_num_batched_tokens,
        runtime.max_num_seqs,
        runtime.max_model_len,
        runtime.kv_block_size_tokens,
        runtime.num_kv_blocks,
        runtime.attention_kv_layout,
        runtime.backend_role,
        runtime.mtp_num_speculative_tokens,
    )
    expected_runtime_scope = (
        "0.11.0",
        "b8b302cde434df8c9289a2b465406b47ebab1c2d",
        config.model.config_sha256,
        "bfloat16",
        "bfloat16",
        2,
        2,
        1,
        1,
        True,
        True,
        0,
        0,
        2,
        1,
        4096,
        8,
        1,
        128,
        16,
        8,
        "paged_varlen",
        "attention",
        0,
    )
    if runtime_scope != expected_runtime_scope:
        raise RuntimeError(
            "standalone AF+EP2 pinned vLLM runtime disagrees with the "
            "validated TP/EP, model, KV, or Scheduler scope")


def _validated_glm4_moe_mtp_model_scope(model: ModelConfig) -> bool:
    return (
        model.architecture == "Glm4MoeForCausalLM"
        and model.model_type == "glm4_moe"
        and model.mtp_architecture == "glm4_moe_mtp"
        and model.num_mtp_layers == 1
        and not model.mtp_layer_uses_moe(0)
        and model.num_hidden_layers == 2
        and model.num_experts == 2
        and model.num_experts_per_tok == 1
        and model.num_dense_layers == 1
        and model.num_shared_experts == 1
        and model.routed_scaling_factor == 1.0
        and model.hidden_size == 64
        and model.intermediate_size == 128
        and model.moe_intermediate_size == 64
        and model.num_attention_heads == 1
        and model.num_key_value_heads == 1
        and model.head_dim == 64
        and model.vocab_size == 64
        and model.max_position_embeddings == 128
        and model.attention_layout == "standard_gqa"
        and not model.attention_bias
        and not model.mlp_bias
        and not model.qk_norm
        and not model.tie_word_embeddings
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and model.weight_group_size == 0
        and model.quantized_weight_targets == ()
        and model.quantized_weight_ignore == ()
        and model.kv_bytes_per_token_per_parallel_rank(1, 1) == 512
    )


def validate_standalone_af_mtp_scope(
    config: SimulationConfig,
    *,
    allow_preemption: bool = False,
) -> None:
    """Validate the exact standalone AF+MTP slice with independent evidence."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("standalone AF MTP validation requires SimulationConfig")
    if not isinstance(allow_preemption, bool):
        raise TypeError("standalone AF MTP preemption flag must be boolean")
    deployment = config.deployment
    model = config.model
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
    )
    ffn_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn"
    )
    model_scope = _validated_glm4_moe_mtp_model_scope(model)
    pool_scope = (
        deployment.mode == "af_disagg"
        and len(deployment.worker_pools) == 2
        and len(attention_pools) == 1
        and len(ffn_pools) == 1
        and attention_pools[0].hardware_id == "nvidia_a800"
        and ffn_pools[0].hardware_id == "nvidia_a800"
        and attention_pools[0].ranks == (0,)
        and ffn_pools[0].ranks == (1,)
    )
    parallel_scope = all(
        pool.parallel_plan.config.tensor_parallel_size == 1
        and pool.parallel_plan.config.pipeline_parallel_size == 1
        and not pool.parallel_plan.config.enable_expert_parallel
        and pool.parallel_plan.config.is_rank_local
        for pool in deployment.worker_pools.values()
    )
    route_scope = (
        len(deployment.routes) == 2
        and all(
            route.payload == "activation"
            and route.algorithm == "send_recv"
            and route.link == "nvlink"
            and route.topology == "point_to_point"
            for route in deployment.routes
        )
    )
    if not all((model_scope, pool_scope, parallel_scope, route_scope)):
        raise NotImplementedError(
            "standalone AF MTP currently requires one nvidia_a800 "
            "Attention/FFN pair on ranks 0/1, BF16 GLM4-MoE with two "
            "target layers, two top-1 experts, one shared expert, unit routed "
            "scaling, one dense MTP layer, TP=PP=EP=SP=CP=DP=1, and "
            "bidirectional NVLink activation routes")
    if (config.launch_policy != "analytical_launch"
            or config.prediction_method not in ("midlayer", "roofline")):
        raise NotImplementedError(
            "standalone AF MTP currently requires analytical launch and "
            "the midlayer or roofline prediction method")
    if config.runtime_inputs.expert_routing.source_kind != "explicit_assumption":
        raise NotImplementedError(
            "standalone AF MTP currently requires seeded-assumption expert "
            "routing")
    if allow_preemption:
        request_shapes = tuple(
            (
                request.arrival_time_s,
                request.input_tokens,
                request.output_tokens,
                request.prefix_hit_tokens,
            )
            for request in config.requests
        )
        if request_shapes != (
                (0.0, 32, 2, 0),
                (0.0, 32, 4, 32)):
            raise NotImplementedError(
                "standalone AF MTP preemption currently requires exactly two "
                "same-time requests shaped (32->2, prefix=0) and "
                "(32->4, prefix=32) in that order")


def _validate_standalone_af_mtp_host_kv_scope(
    config: SimulationConfig,
    backend_config: BackendConfig,
    replay: KVLifecycleReplay,
) -> None:
    """Pin the only independently exercised AF+MTP+Host-KV combination."""

    if not isinstance(config, SimulationConfig):
        raise TypeError(
            "standalone AF MTP Host KV validation requires SimulationConfig")
    if not isinstance(backend_config, BackendConfig):
        raise TypeError(
            "standalone AF MTP Host KV validation requires BackendConfig")
    if not isinstance(replay, KVLifecycleReplay):
        raise NotImplementedError(
            "standalone AF Host KV lifecycle with MTP requires a hash-bound "
            "lifecycle replay")
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in config.requests
    )
    backend_scope = (
        backend_config.max_batched_tokens,
        backend_config.max_requests,
        backend_config.num_kv_blocks,
        backend_config.kv_block_size_tokens,
        backend_config.kv_bytes_per_token,
    )
    cache = replay.cache_config
    tier_scope = tuple(sorted(
        (
            tier.name,
            tier.capacity_bytes,
            tier.block_size_tokens,
            tier.link,
        )
        for tier in cache.tiers
    ))
    event_scope = tuple(
        tuple(event.action for event in step.events)
        for step in replay.steps
    )
    supported_request_shapes = (
        ((0.0, 16, 4, 0),),
        ((0.0, 17, 4, 16),),
    )
    if (
        request_shapes not in supported_request_shapes
        or backend_scope != (128, 1, 3, 16, 512)
        or cache.bytes_per_token != 512
        or cache.offload_policy != "lru"
        or tier_scope != (
            ("hbm", 24_576, 16, None),
            ("host", 8_192, 16, "pcie"),
        )
        or event_scope != (
            (KVCacheAction.EVICT,),
            (KVCacheAction.LOAD,),
            (),
        )
    ):
        raise NotImplementedError(
            "standalone AF Host KV lifecycle with MTP currently requires "
            "one 16->4 request without Prefix or one 17->4 request with a "
            "16-token Prefix, batch=128, max_requests=1, "
            "three 16-token HBM blocks, one Host block, fixed acceptance "
            "one, and an EVICT/LOAD/empty three-step replay")


def validate_pd_af_mtp_scope(
    config: SimulationConfig,
    *,
    fixed_accepted_tokens: int = 1,
) -> None:
    """Validate the exact 1P x 1D PD+AF fixed-count MTP slice."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("PD+AF MTP validation requires SimulationConfig")
    if (isinstance(fixed_accepted_tokens, bool)
            or not isinstance(fixed_accepted_tokens, int)):
        raise TypeError("PD+AF MTP fixed acceptance must be an integer")
    if fixed_accepted_tokens not in (0, 1, 2):
        raise NotImplementedError(
            "PD+AF MTP currently supports fixed acceptance zero, one, or two")
    deployment = config.deployment
    expected_pools = {
        "prefill-attention": ("prefill", "attention", (0,)),
        "prefill-ffn": ("prefill", "ffn", (1,)),
        "decode-attention": ("decode", "attention", (2,)),
        "decode-ffn": ("decode", "ffn", (3,)),
    }
    pool_scope = (
        deployment.mode == "pd_af_disagg"
        and set(deployment.worker_pools) == set(expected_pools)
        and all(
            worker_role_phase(deployment.worker_pools[name].role) == phase
            and worker_role_operator_group(
                deployment.worker_pools[name].role) == operator_group
            and deployment.worker_pools[name].hardware_id == "nvidia_a800"
            and deployment.worker_pools[name].ranks == ranks
            for name, (phase, operator_group, ranks)
            in expected_pools.items()
        )
    )
    parallel_scope = all(
        pool.parallel_plan.config.tensor_parallel_size == 1
        and pool.parallel_plan.config.pipeline_parallel_size == 1
        and not pool.parallel_plan.config.enable_expert_parallel
        and pool.parallel_plan.config.is_rank_local
        for pool in deployment.worker_pools.values()
    )
    expected_routes = {
        ("prefill-attention", "prefill-ffn", "activation"):
            "nvlink:prefill-attention--prefill-ffn",
        ("prefill-ffn", "prefill-attention", "activation"):
            "nvlink:prefill-attention--prefill-ffn",
        ("decode-attention", "decode-ffn", "activation"):
            "nvlink:decode-attention--decode-ffn",
        ("decode-ffn", "decode-attention", "activation"):
            "nvlink:decode-attention--decode-ffn",
        ("prefill-attention", "decode-attention", "kv"):
            "nvlink:decode-attention--prefill-attention",
    }
    route_scope = (
        {
            (route.source_pool, route.destination_pool, route.payload)
            for route in deployment.routes
        } == set(expected_routes)
        and len(deployment.routes) == len(expected_routes)
        and all(
            route.algorithm == "send_recv"
            and route.link == "nvlink"
            and route.topology == "point_to_point"
            and route.resource_id == expected_routes[
                (route.source_pool, route.destination_pool, route.payload)]
            for route in deployment.routes
        )
    )
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in config.requests
    )
    request_scope = (
        request_shapes == ((0.0, 16, 5, 0),)
        or (
            request_shapes == ((0.0, 17, 5, 16),)
            and fixed_accepted_tokens == 1
        )
    )
    if not all((
            _validated_glm4_moe_mtp_model_scope(config.model),
            pool_scope,
            parallel_scope,
            route_scope,
            request_scope,
    )):
        raise NotImplementedError(
            "PD+AF MTP currently requires one 16->5 request without Prefix "
            "or one block-aligned 17->5 request with Prefix=16 and fixed "
            "acceptance one, "
            "one nvidia_a800 P Attention/FFN pair on ranks 0/1, one D pair "
            "on ranks 2/3, BF16 GLM4-MoE with two target layers and one "
            "dense MTP layer, TP=PP=EP=SP=CP=DP=1, and five NVLink "
            "analytical Roofline activation/KV routes")
    if (config.launch_policy != "analytical_launch"
            or config.prediction_method != "midlayer"):
        raise NotImplementedError(
            "PD+AF MTP currently requires analytical launch and the "
            "midlayer prediction method")
    if config.runtime_inputs.expert_routing.source_kind != "explicit_assumption":
        raise NotImplementedError(
            "PD+AF MTP currently requires seeded-assumption expert routing")
    if (config.runtime_inputs.attention_kv_layout != "paged_varlen"
            or config.runtime_inputs.attention_kv_distribution != "uniform"):
        raise NotImplementedError(
            "PD+AF MTP currently requires paged_varlen attention with "
            "uniform KV distribution")


def _validate_standalone_af_mtp_step(
    step: BackendStep,
    *,
    allow_preemption: bool,
    expected_fixed_accepted_tokens: int | None = None,
) -> None:
    if (expected_fixed_accepted_tokens is not None
            and (
                isinstance(expected_fixed_accepted_tokens, bool)
                or not isinstance(expected_fixed_accepted_tokens, int)
                or expected_fixed_accepted_tokens not in (0, 1, 2)
            )):
        raise ValueError(
            "standalone AF MTP expected fixed acceptance must be 0, 1, 2, "
            "or null")
    if not allow_preemption and (
            step.schedule.preempted_request_ids or any(
                request.resumed_from_preemption
                for request in step.schedule.requests)):
        raise NotImplementedError(
            "standalone AF MTP does not support preemption or resume")
    for request in step.schedule.requests:
        if request.mtp_acceptance_source_kind != "explicit_assumption":
            raise NotImplementedError(
                "standalone AF MTP requires fixed-count explicit-assumption "
                "acceptance")
        if (request.mtp_proposer_tokens != 2
                or request.mtp_next_proposed_tokens not in (0, 2)
                or request.mtp_proposed_tokens not in (0, 2)
                or request.mtp_verified_tokens not in (0, 2)
                or request.mtp_accepted_tokens not in (0, 1, 2)
                or (
                    expected_fixed_accepted_tokens is not None
                    and request.mtp_accepted_tokens
                    != (
                        expected_fixed_accepted_tokens
                        if request.mtp_verified_tokens else 0
                    )
                )):
            raise NotImplementedError(
                "standalone AF MTP requires exactly two speculative tokens "
                "and the validated fixed acceptance")


def _validate_pd_af_mtp_step(
    step: BackendStep,
    *,
    fixed_accepted_tokens: int,
) -> None:
    if (step.schedule.preempted_request_ids
            or any(
                request.resumed_from_preemption
                for request in step.schedule.requests)):
        raise NotImplementedError(
            "PD+AF MTP does not support preemption or resume")
    for request in step.schedule.requests:
        if (request.phase != "decode"
                or request.input_tokens != 0
                or request.trace_prefix_hit_tokens != 0
                or request.scheduler_prefix_hit_tokens != 0
                or request.mtp_acceptance_source_kind
                != "explicit_assumption"
                or request.mtp_proposer_tokens != 2
                or request.mtp_next_proposed_tokens not in (0, 2)
                or request.mtp_proposed_tokens not in (0, 2)
                or request.mtp_verified_tokens not in (0, 2)
                or request.mtp_accepted_tokens
                != (
                    fixed_accepted_tokens
                    if request.mtp_verified_tokens else 0
                )):
            raise NotImplementedError(
                "PD+AF MTP decode steps require fresh seeded decode state, "
                "two speculative tokens, and the configured fixed "
                "acceptance")


def _validate_pd_af_mtp_run(
    config: SimulationConfig,
    simulation_steps: Tuple["SimulationStep", ...],
    *,
    fixed_accepted_tokens: int,
) -> None:
    validate_pd_af_mtp_scope(
        config,
        fixed_accepted_tokens=fixed_accepted_tokens,
    )
    prefill_steps = tuple(
        step for step in simulation_steps
        if step.backend_step.schedule.pool_name == "prefill-attention"
    )
    decode_steps = tuple(
        step for step in simulation_steps
        if step.backend_step.schedule.pool_name == "decode-attention"
    )
    if fixed_accepted_tokens == 0:
        expected_emitted_tokens = (1, 1, 1, 1)
        expected_mtp_rounds = (
            (0, 0, 0, 2),
            (2, 2, 0, 2),
            (2, 2, 0, 2),
            (2, 2, 0, 0),
        )
    elif fixed_accepted_tokens == 1:
        expected_emitted_tokens = (1, 2, 1)
        expected_mtp_rounds = (
            (0, 0, 0, 2),
            (2, 2, 1, 2),
            (2, 2, 1, 0),
        )
    elif fixed_accepted_tokens == 2:
        expected_emitted_tokens = (1, 3)
        expected_mtp_rounds = (
            (0, 0, 0, 2),
            (2, 2, 2, 0),
        )
    else:
        raise AssertionError("validated unknown PD+AF MTP fixed acceptance")
    if (len(prefill_steps) != 1
            or len(decode_steps) != len(expected_emitted_tokens)):
        raise RuntimeError(
            "PD+AF MTP step count disagrees with the pinned fixed-count "
            "reference")
    request_id = config.requests.requests[0].request_id
    prompt_tokens = config.requests.requests[0].input_tokens
    prefix_tokens = config.requests.requests[0].prefix_hit_tokens
    prefill_request, = prefill_steps[0].backend_step.schedule.requests
    if (prefill_request.request_id != request_id
            or prefill_request.computed_tokens != prefix_tokens
            or prefill_request.input_tokens != prompt_tokens - prefix_tokens
            or prefill_request.trace_prefix_hit_tokens != prefix_tokens
            or prefill_request.scheduler_prefix_hit_tokens != prefix_tokens
            or prefill_request.prefix_replay_tokens != 0
            or prefill_request.mtp_acceptance_source_kind != "disabled"
            or prefill_steps[0].backend_step.generated_tokens[request_id] != 1):
        raise RuntimeError(
            "PD+AF MTP prefill must preserve the validated Prefix workload "
            "and emit one terminal seed without draft state")
    decode_requests = tuple(
        request
        for step in decode_steps
        for request in step.backend_step.schedule.requests
    )
    if len(decode_requests) != len(expected_emitted_tokens) or any(
            request.request_id != request_id for request in decode_requests):
        raise RuntimeError(
            "PD+AF MTP decode trace must contain one request in every "
            "reference step")
    if tuple(
            step.backend_step.generated_tokens[request_id]
            for step in decode_steps) != expected_emitted_tokens:
        raise RuntimeError(
            "PD+AF MTP decode emitted-token sequence disagrees with the "
            "pinned fixed-count reference")
    if tuple(
            (
                request.mtp_proposed_tokens,
                request.mtp_verified_tokens,
                request.mtp_accepted_tokens,
                request.mtp_next_proposed_tokens,
            )
            for request in decode_requests) != expected_mtp_rounds:
        raise RuntimeError(
            "PD+AF MTP decode proposal/verification sequence disagrees with "
            "the pinned fixed-count reference")
    first_decode = decode_requests[0]
    if (first_decode.computed_tokens != prompt_tokens
            or first_decode.raw_model_output_tokens != 1):
        raise RuntimeError(
            "PD+AF MTP decode admission must start from full prompt KV, one "
            "terminal seed, and empty draft state")


def _validate_standalone_af_mtp_preemption_run(
    config: SimulationConfig,
    simulation_steps: Tuple["SimulationStep", ...],
) -> None:
    validate_standalone_af_mtp_scope(config, allow_preemption=True)
    resumed_request_id = config.requests.requests[1].request_id
    preemptions = tuple(
        (index, step.backend_step.schedule.preempted_request_ids)
        for index, step in enumerate(simulation_steps)
        if step.backend_step.schedule.preempted_request_ids
    )
    resumed = tuple(
        (index, request)
        for index, step in enumerate(simulation_steps)
        for request in step.backend_step.schedule.requests
        if request.resumed_from_preemption
    )
    if (len(preemptions) != 1
            or preemptions[0][1] != (resumed_request_id,)):
        raise RuntimeError(
            "standalone AF MTP preemption scope requires exactly the second "
            "request to be preempted once")
    preemption_index = preemptions[0][0]
    stale_proposals = tuple(
        (index, request)
        for index, step in enumerate(simulation_steps[:preemption_index])
        for request in step.backend_step.schedule.requests
        if request.request_id == resumed_request_id
        and request.mtp_proposer_tokens == 2
        and request.mtp_next_proposed_tokens == 2
    )
    if len(stale_proposals) != 1:
        raise RuntimeError(
            "standalone AF MTP preemption scope requires exactly one "
            "two-token proposal for the second request before preemption")
    if (len(resumed) != 1
            or resumed[0][1].request_id != resumed_request_id
            or resumed[0][0] <= preemption_index):
        raise RuntimeError(
            "standalone AF MTP preemption scope requires the second request "
            "to resume exactly once after preemption")
    resumed_index, resumed_request = resumed[0]
    if (
        resumed_request.computed_tokens != 0
        or resumed_request.recomputed_tokens != 32
        or resumed_request.input_tokens != 32
        or resumed_request.output_tokens != 1
        or resumed_request.mtp_proposed_tokens != 2
        or resumed_request.mtp_verified_tokens != 0
        or resumed_request.mtp_accepted_tokens != 0
        or resumed_request.mtp_proposer_tokens != 2
        or resumed_request.mtp_next_proposed_tokens != 2
    ):
        raise RuntimeError(
            "standalone AF MTP resumed request must recompute 32 prompt "
            "tokens, discard stale draft verification, emit one terminal "
            "token, and install two fresh draft tokens")
    later_verified = tuple(
        request
        for step in simulation_steps[resumed_index + 1:]
        for request in step.backend_step.schedule.requests
        if request.request_id == resumed_request_id
        and request.mtp_proposed_tokens == 2
        and request.mtp_verified_tokens == 2
        and request.mtp_accepted_tokens == 1
    )
    if not later_verified:
        raise RuntimeError(
            "standalone AF MTP resumed request never verified the fresh "
            "two-token proposal with fixed acceptance one")


@dataclass(frozen=True)
class SimulationResult:
    schedule_trace: ScheduleTrace
    metrics: ServingMetrics
    steps: Tuple["SimulationStep", ...] = ()
    transfers: Tuple["SimulationTransfer", ...] = ()
    kv_offload: Optional["SimulationKVOffload"] = None
    execution_cache_stats: Mapping[str, object] | None = None
    execution_performance_stats: Mapping[str, object] | None = None
    # Requests at the arrival times the metrics were computed against.
    # Differs from the canonical stream only for closed-loop arrivals, where
    # each successor is stamped at its client's completion instant; ``None``
    # means the canonical stream was used.
    metrics_requests: RequestStream | None = None
    source_activation_memory: Mapping[str, object] | None = None
    pd_kv_lifecycle: tuple[dict, ...] = ()


@dataclass(frozen=True)
class StreamingStepTiming:
    """Lightweight per-step timing retained without the execution IR."""

    step_id: str
    predicted_device_time_s: float
    launch_cpu_time_s: float
    launch_critical_path_s: float
    host_device_overlap_s: float
    eager_dispatch_count: int
    graph_replay_count: int
    launch_kinds: Tuple[str, ...]


@dataclass(frozen=True)
class StreamingSimulationResult:
    """Aggregate-only result whose heavy per-step IR was consumed in flight."""

    schedule_trace: ScheduleTrace
    metrics: ServingMetrics | None
    num_steps: int
    num_kernel_calls: int
    step_timings: Tuple[StreamingStepTiming, ...]
    execution_cache_stats: Mapping[str, object]
    execution_performance_stats: Mapping[str, object]
    num_simulation_transfers: int = 0
    source_activation_memory: Mapping[str, object] | None = None
    pd_kv_lifecycle: tuple[dict, ...] = ()
    kv_offload: Optional["SimulationKVOffload"] = None
    recorded_transfers: Tuple["SimulationTransfer", ...] = ()

    def __post_init__(self) -> None:
        if (self.metrics is None) != (self.schedule_trace.observation_end_time_s is not None):
            raise ValueError("finite observations must omit complete-cohort metrics")
        if not isinstance(self.recorded_transfers, tuple) or any(
                not isinstance(item, SimulationTransfer) for item in self.recorded_transfers):
            raise TypeError("recorded_transfers must be a tuple of SimulationTransfer")
        if self.recorded_transfers and len(self.recorded_transfers) != self.num_simulation_transfers:
            raise ValueError("recorded transfer count disagrees with simulation")
        for name in ("num_steps", "num_kernel_calls"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(
                    f"streaming simulation {name} must be an integer")
            if value < 0 or (value == 0 and self.schedule_trace.observation_end_time_s is None):
                raise ValueError(
                    f"streaming simulation {name} must be positive")
        transfer_count = self.num_simulation_transfers
        if (isinstance(transfer_count, bool)
                or not isinstance(transfer_count, int)):
            raise TypeError(
                "streaming simulation num_simulation_transfers must be an integer")
        if transfer_count < 0:
            raise ValueError(
                "streaming simulation transfer count must be non-negative")
        if (self.kv_offload is not None
                and not isinstance(self.kv_offload, SimulationKVOffload)):
            raise TypeError("streaming KV offload must be SimulationKVOffload or null")
        if self.num_steps != len(self.schedule_trace.entries):
            raise ValueError(
                "streaming simulation step count disagrees with trace")
        if transfer_count != len(self.schedule_trace.transfers):
            raise ValueError(
                "streaming simulation transfer count disagrees with trace")
        if any(
                not isinstance(timing, StreamingStepTiming)
                for timing in self.step_timings):
            raise TypeError(
                "streaming simulation step timings must be canonical")
        timing_ids = tuple(timing.step_id for timing in self.step_timings)
        if len(timing_ids) != len(set(timing_ids)):
            raise ValueError(
                "streaming simulation step timing IDs must be unique")
        timing_mode = self.schedule_trace.entries[0].timing_mode if self.schedule_trace.entries else "device_only"
        if timing_mode == "device_only":
            if timing_ids:
                raise ValueError(
                    "device-only streaming simulation cannot retain Host "
                    "step timings")
        elif set(timing_ids) != {
                entry.backend_step.schedule.step_id
                for entry in self.schedule_trace.entries}:
            raise ValueError(
                "Host streaming step timings must cover the schedule trace")
        if not isinstance(self.execution_cache_stats, Mapping):
            raise TypeError(
                "streaming simulation cache stats must be a mapping")


@dataclass(frozen=True)
class SimulationStep:
    backend_step: BackendStep
    workload: object
    kernel_plan: object
    predictions: Mapping[str, object]
    execution_timeline: object
    source_activation_memory: Mapping[str, object] | None = None
    source_state_summary: Mapping[str, object] | None = None


def _apply_tail_worker_limit(requested_workers, cpu_count, memory_worker_limit):
    """Worker cap shared by the parallel tail and adaptive grant paths.

    Automatic execution uses the available CPU count. Explicit requests and
    adaptive grants retain the CPU and memory headroom bounds, and the caller
    bounds the result by the pending step count.
    """
    limit = cpu_count if requested_workers == 0 else requested_workers
    return min(limit, cpu_count, memory_worker_limit)


def _validate_tail_worker_grant(grant):
    if isinstance(grant, bool) or not isinstance(grant, int) or grant < 1:
        raise ValueError("adaptive tail worker grant must be a positive integer")
    return grant


def _state_audit_snapshot(step):
    if not isinstance(step, SimulationStep):
        raise TypeError("state audit requires SimulationStep")
    if step.source_state_summary is None:
        raise ValueError("state audit requires source state for every step")
    summary = step.source_state_summary
    if not isinstance(summary, Mapping) or not isinstance(summary["ranks"], Mapping):
        raise TypeError("state audit source summary has an invalid rank mapping")
    occurrences = {}
    for repeat in step.kernel_plan.block_repeats:
        for call_id in repeat.call_ids:
            occurrences[call_id] = repeat.repeat_count
    calls = []
    for call in step.kernel_plan.calls:
        count = occurrences[call.call_id] if call.call_id in occurrences else 1
        transfer = call.kernel_spec.transfer
        operation = (None if transfer is None else
            transfer.operation if transfer.operation is not None else "not_declared")
        calls.append((call.call_id, call.pool_name, str(call.rank),
            call.kernel_spec.kind.value, count, operation,
            0 if transfer is None else transfer.bytes))
    return {
        "schema_version": "serving_state_audit_step_v1",
        "step_id": step.backend_step.schedule.step_id,
        "pool_name": step.backend_step.schedule.pool_name,
        "source_state_summary": summary,
        "calls": tuple(calls),
    }


@dataclass(frozen=True)
class _ParallelStepExecution:
    duration_s: float
    execution_contract_digest: object
    logical_kernel_call_count: int
    compact: bool
    cache_deltas: Mapping[str, tuple[int, int, int]]
    phase_wall_time_s: Mapping[str, float]
    source_activation_memory: Mapping[str, object] | None = None
    step_timing: StreamingStepTiming | None = None
    state_audit_snapshot: Mapping[str, object] | None = None
    worker_host_wall_s: float = 0.0


_PARALLEL_EXECUTION_CONTEXT = None
_PARALLEL_STEP_PROJECTOR = None
_PARALLEL_TAIL_BATCH_SIZE = 1024
_PARALLEL_WORKER_MAX_TASKS = 4


def _transfer_free_decode_step(step):
    schedule = step.schedule
    return (bool(schedule.requests) and not schedule.preempted_request_ids
        and all(request.phase == "decode" and request.scheduled_tokens == 1
            and not request.trace_prefix_hit_tokens and not request.resumed_from_preemption
            for request in schedule.requests)
        and all(event.action not in (KVCacheAction.LOAD, KVCacheAction.EVICT, KVCacheAction.STORE)
            for event in step.kv_cache_events))


def _fixed_source_segment_fits(cache, steps):
    """Keep every actual q1 source class resident before any preparation."""
    from src.serving.workload.traced_forward import SourceProgramCache
    return (type(cache) is SourceProgramCache and type(cache.max_entries) is int
        and cache.max_entries > 0 and len({tuple(request.scheduled_tokens
            for request in step.schedule.requests) for step in steps}) <= cache.max_entries)


def _fixed_source_program_key(cache, graph):
    """Find the actual source owner's unique existing key, without deriving it."""
    from src.serving.workload.traced_models import _GLMDecodeSourceBinding
    binding = graph.__dict__["_glm_decode_binding"] if "_glm_decode_binding" in graph.__dict__ else None
    program = binding.owner(graph) if type(binding) is _GLMDecodeSourceBinding else None
    if program is None:
        raise RuntimeError("fixed-source preparation lost its owned decode program")
    keys = tuple(key for key, value in cache.programs.items() if value is program)
    if len(keys) != 1:
        raise RuntimeError("fixed-source preparation requires one actual source cache key")
    return keys[0], program


def _restore_fixed_source_program_order(cache, steps, programs_by_shape):
    """Replay every real lookup, preserving the next segment's LRU victims."""
    for step in steps:
        shape = tuple(request.scheduled_tokens for request in step.schedule.requests)
        if shape not in programs_by_shape:
            raise RuntimeError("fixed-source segment lost a prepared shape")
        key, program = programs_by_shape[shape]
        if cache.lookup(key) is not program:
            raise RuntimeError("fixed-source segment lost its actual prepared program")


@_metadata_collection_scope()
def _parallel_execute_step(backend_step: BackendStep, *, execution=None, projector=None,
        allow_capture_history=False, allow_invariant_decode_workspace=False,
        timeline_origin_provider=None) -> _ParallelStepExecution:
    execution = _PARALLEL_EXECUTION_CONTEXT if execution is None else execution
    projector = _PARALLEL_STEP_PROJECTOR if projector is None else projector
    if execution is None or projector is None:
        raise RuntimeError("parallel Serving execution worker is not initialized")
    if type(allow_capture_history) is not bool or type(allow_invariant_decode_workspace) is not bool:
        raise TypeError("parallel source-history permissions must be boolean")
    workspace_capacities = None
    if execution.runtime_inputs.symbolic_workspace_initial_capacity_bytes is not None:
        if allow_invariant_decode_workspace:
            workspace_capacities = execution._independent_decode_workspace_capacities(backend_step.schedule)
        if workspace_capacities is None:
            raise RuntimeError("source workspace history requires sequential cross-step execution")
    if (not allow_capture_history
            and execution.runtime_inputs.symbolic_forward_source is not None
            and execution.runtime_inputs.graph_execution.enabled):
        raise RuntimeError("source capture history requires sequential cross-step execution")
    cache_before = execution.cache_stats()
    performance_before = execution.performance_stats()
    components = execution.execute_step(
        backend_step.schedule,
        kv_cache_events=backend_step.kv_cache_events,
        embedding_token_ids=backend_step.embedding_token_ids,
        model_source_inputs=backend_step.model_source_inputs,
        worker_request_state=backend_step.worker_request_state,
        timeline_origin_provider=timeline_origin_provider,
    )
    if workspace_capacities is not None:
        pool = execution.deployment.worker_pools[backend_step.schedule.pool_name]
        actual = tuple(components.source_activation_memory["ranks"][str(rank)][
            "source_workspace_capacity_bytes"]["worker.workspace"] for rank in pool.ranks)
        if (actual != workspace_capacities or execution._independent_decode_workspace_capacities(
                backend_step.schedule) != workspace_capacities):
            raise RuntimeError("independent decode changed its actual source workspace history")
    simulation_step = SimulationStep(
        backend_step,
        components.workload,
        components.kernel_plan,
        components.predictions,
        components.execution_timeline,
        components.source_activation_memory,
        components.source_state_summary,
    )
    execution_contract_digest = projector(simulation_step)
    cache_after = execution.cache_stats()
    performance_after = execution.performance_stats()
    cache_deltas = {}
    for name in ("kernel_lowering", "kernel_prediction"):
        before = cache_before[name]
        after = cache_after[name]
        cache_deltas[name] = (
            after["entries"] - before["entries"],
            after["hits"] - before["hits"],
            after["misses"] - before["misses"],
        )
    phase_before = performance_before["phase_wall_time_s"]
    phase_after = performance_after["phase_wall_time_s"]
    return _ParallelStepExecution(
        components.execution_timeline.duration_s,
        execution_contract_digest,
        components.kernel_plan.logical_call_count,
        bool(components.kernel_plan.block_repeats),
        cache_deltas,
        {
            name: phase_after[name] - phase_before[name]
            for name in phase_after
        },
        components.source_activation_memory,
        (None if execution.host_launch_profiles is None else StreamingStepTiming(
            backend_step.schedule.step_id,
            components.execution_timeline.predicted_device_time_s,
            components.execution_timeline.launch_cpu_time_s,
            components.execution_timeline.launch_critical_path_s,
            components.execution_timeline.host_device_overlap_s,
            components.execution_timeline.eager_dispatch_count,
            components.execution_timeline.graph_replay_count,
            components.execution_timeline.launch_kinds,
        )),
        _state_audit_snapshot(simulation_step) if execution.collect_source_state else None,
        worker_host_wall_s=components.execution_timeline.worker_host_wall_s,
    )


def _parallel_execute_fixed_decode_step(backend_step):
    return _parallel_execute_step(backend_step, allow_invariant_decode_workspace=True)


def _single_worker_tail_executor(initial_execution, projector, *, allow_capture_history=False):
    """Bind a thread-local executor with bounded caches, without forking.

    Recycle after four steps, a stricter bound than the process pool's four
    chunks. Cache resets affect CPU implementation cost, never model work.
    """
    execution = initial_execution
    completed = 0

    def execute(backend_step):
        nonlocal execution, completed
        if completed == _PARALLEL_WORKER_MAX_TASKS:
            execution = ServingExecutionContext(
                initial_execution.model, initial_execution.deployment,
                initial_execution.hardware_by_id, initial_execution.launch_policy,
                initial_execution.prediction_method, initial_execution.runtime_inputs,
                execution_mode=initial_execution.execution_mode,
                fresh_kernel_predictions=initial_execution.fresh_kernel_predictions,
                host_launch_profiles=initial_execution.host_launch_profiles,
                host_runtime_prepare_profiles=initial_execution.host_runtime_prepare_profiles,
                runtime_prepare_event_builder=initial_execution.runtime_prepare_event_builder,
                model_source_inputs_by_step=initial_execution.model_source_inputs_by_step,
                collect_source_state=initial_execution.collect_source_state,
                materialize_timeline_details=initial_execution.materialize_timeline_details)
            completed = 0
        result = _parallel_execute_step(backend_step, execution=execution, projector=projector,
            allow_capture_history=allow_capture_history)
        completed += 1
        return result

    scope = copy_context()
    return lambda backend_step: scope.run(execute, backend_step)


def _tail_execution_pool(worker_count):
    if worker_count == 1:
        return ThreadPool(1)
    return multiprocessing.get_context("fork").Pool(
        worker_count, maxtasksperchild=_PARALLEL_WORKER_MAX_TASKS)


def _initialize_causal_execution_worker(execution, projector):
    """Initialize only the child's run-owned executor, never parent globals."""
    global _PARALLEL_EXECUTION_CONTEXT, _PARALLEL_STEP_PROJECTOR
    _PARALLEL_EXECUTION_CONTEXT = execution
    _PARALLEL_STEP_PROJECTOR = projector


def _causal_execute_step(backend_step):
    started = time.perf_counter()
    result = _parallel_execute_step(backend_step, allow_capture_history=True)
    return started, time.perf_counter(), result


def _source_ordered_causal_execute_step(backend_step, origin_receiver):
    """Prepare independently, then consume the FIFO owner's actual clock."""
    started = time.perf_counter()
    try:
        result = _parallel_execute_step(backend_step, allow_capture_history=True,
            timeline_origin_provider=origin_receiver.recv)
        return started, time.perf_counter(), result
    finally:
        origin_receiver.close()


def _resolve_causal_cpu_result(pending, workers):
    """Fail closed if a Pool replaces a worker with an unreturned task."""
    while True:
        try:
            return pending.get(timeout=0.1)
        except multiprocessing.TimeoutError:
            if pending.ready():
                return pending.get()
            if any(worker.exitcode is not None for worker in workers):
                raise RuntimeError("causal CPU execution worker exited before returning its result")


def _independent_runtime_prepare_builder(builder):
    """Recognize the source-owned pure mapper and its immutable configuration."""
    from src.serving.runtime_prepare import runtime_prepare_events
    if builder is None:
        return True
    if (type(builder) is not partial or builder.func is not runtime_prepare_events
            or builder.args or set(builder.keywords) not in (
                {"enabled_mechanisms"},
                {"enabled_mechanisms", "ascend_compilation_mode"})):
        return False
    if ("ascend_compilation_mode" in builder.keywords
            and builder.keywords["ascend_compilation_mode"] not in (
                "NONE", "FULL", "FULL_DECODE_ONLY", "PIECEWISE")):
        return False
    mechanisms = builder.keywords["enabled_mechanisms"]
    return (type(mechanisms) is tuple and bool(mechanisms)
        and all(type(value) is str for value in mechanisms))


@dataclass(frozen=True)
class SimulationTransfer:
    request_id: str
    source_engine_id: str
    destination_engine_id: str
    workload: object
    kernel_plan: object
    predictions: Mapping[str, object]
    execution_timeline: object


@dataclass(frozen=True)
class SimulationKVCacheTransfer:
    step_id: str
    event: KVCacheEvent
    completion: KVCacheTransferCompletion

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id:
            raise ValueError("KV cache transfer step_id must be non-empty")
        if not isinstance(self.event, KVCacheEvent):
            raise TypeError("KV cache transfer event must be a KVCacheEvent")
        if not isinstance(self.completion, KVCacheTransferCompletion):
            raise TypeError(
                "KV cache transfer completion must be canonical")
        if self.event.action not in (
                KVCacheAction.LOAD, KVCacheAction.EVICT,
                KVCacheAction.STORE):
            raise ValueError("KV cache transfer requires load, evict or store")
        if (self.event.ready_event != self.completion.ready_event
                or self.event.action is not self.completion.action
                or self.event.bytes != self.completion.bytes
                or self.event.source_tier != self.completion.source_tier
                or self.event.destination_tier
                != self.completion.destination_tier):
            raise ValueError(
                "KV cache transfer completion disagrees with its intent")


@dataclass(frozen=True)
class SimulationKVOffload:
    transfers: Tuple[SimulationKVCacheTransfer, ...]
    final_tier_state: Mapping[str, object]
    lifecycle_replay: Optional[Mapping[str, object]] = None
    mode: str = "modeled_host_kv_serial"
    validation_status: str = "modeled_request_private_full_blocks"

    def __post_init__(self) -> None:
        if not isinstance(self.transfers, (list, tuple)):
            raise TypeError("KV offload transfers must be a list or tuple")
        object.__setattr__(self, "transfers", tuple(self.transfers))
        if (any(not isinstance(item, SimulationKVCacheTransfer)
                for item in self.transfers)
                or (not self.transfers
                    and self.mode != "fixed_prefix_tier_replay")):
            raise ValueError(
                "KV offload requires canonical transfer completions")
        if not isinstance(self.final_tier_state, Mapping):
            raise TypeError("KV offload final tier state must be a mapping")
        if (self.lifecycle_replay is not None
                and not isinstance(self.lifecycle_replay, Mapping)):
            raise TypeError(
                "KV offload lifecycle replay must be a mapping or null")
        if self.mode not in ("modeled_host_kv_serial", "fixed_prefix_tier_replay"):
            raise ValueError(f"unknown KV offload mode: {self.mode!r}")
        if self.validation_status not in (
                "modeled_request_private_full_blocks",
                "modeled_shared_prefix_full_blocks"):
            raise ValueError(
                "unknown KV offload validation status: "
                f"{self.validation_status!r}")
        if self.final_tier_state["validation_status"] != self.validation_status:
            raise ValueError("KV offload validation status disagrees with tier state")


def _is_validated_host_kv_independent_preemption_scope(
    config: SimulationConfig,
    backend_config: BackendConfig,
    replay: Optional[KVLifecycleReplay],
) -> bool:
    """Recognize the one source-aligned Host-KV/preemption coexistence gate."""

    if replay is None:
        return False
    model = config.model
    deployment = config.deployment
    if (deployment.mode != "colocated"
            or len(deployment.worker_pools) != 1):
        return False
    pool = next(iter(deployment.worker_pools.values()))
    parallelism = pool.parallel_plan.config
    requests = config.requests.requests
    if (pool.hardware_id != "nvidia_a800"
            or len(pool.ranks) != 1
            or parallelism.tensor_parallel_size != 1
            or parallelism.expert_parallel_size != 1
            or parallelism.pipeline_parallel_size != 1
            or not parallelism.is_rank_local
            or parallelism.enable_expert_parallel
            or backend_config.pool_name != pool.name
            or backend_config.role != "colocated"
            or backend_config.max_batched_tokens != 32
            or backend_config.max_requests != 2
            or backend_config.num_kv_blocks != 3
            or backend_config.kv_block_size_tokens != 16
            or backend_config.kv_bytes_per_token != 256):
        return False
    if (model.model_type != "qwen3"
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or model.hidden_size != 64
            or model.intermediate_size != 128
            or model.num_attention_heads != 1
            or model.num_key_value_heads != 1
            or model.num_hidden_layers != 1
            or model.head_dim != 64
            or model.vocab_size != 64
            or model.num_experts != 0
            or model.num_mtp_layers != 0
            or model.mtp_architecture != "none"):
        return False
    if (len(requests) != 2
            or any(
                request.arrival_time_s != 0.0
                or request.input_tokens != 16
                or request.output_tokens != 2
                or request.prefix_hit_tokens != 0
                for request in requests)):
        return False
    hbm_tier = next(
        tier for tier in replay.cache_config.tiers if tier.name == "hbm")
    host_tier = next(
        tier for tier in replay.cache_config.tiers if tier.name == "host")
    block_bytes = 16 * 256
    if (replay.cache_config.bytes_per_token != 256
            or replay.cache_config.offload_policy != "lru"
            or hbm_tier.block_size_tokens != 16
            or hbm_tier.capacity_bytes != 3 * block_bytes
            or host_tier.block_size_tokens != 16
            or host_tier.capacity_bytes != block_bytes
            or host_tier.link != "pcie"):
        return False
    if len(replay.steps) != 3:
        return False
    event_actions = tuple(
        tuple(event.action for event in step.events)
        for step in replay.steps
    )
    if event_actions != (
            (KVCacheAction.EVICT,),
            (KVCacheAction.LOAD,),
            ()):
        return False
    transfer_request_ids = {
        event.request_id
        for step in replay.steps
        for event in step.events
    }
    return (
        len(transfer_request_ids) == 1
        and transfer_request_ids < {
            request.request_id for request in requests
        }
    )


class _BoundKVLifecycle:
    """Apply one Scheduler pool's bound KV lifecycle exactly once."""

    def __init__(
        self,
        manager: KVCacheTierManager,
        replay: Optional[KVLifecycleReplay],
        cursor: object,
        *,
        owner_pool_name: str,
        owner_ranks: tuple[int, ...],
        allow_independent_preemption: bool,
    ) -> None:
        if not isinstance(manager, KVCacheTierManager):
            raise TypeError(
                "bound KV lifecycle requires a KVCacheTierManager")
        if replay is not None and not isinstance(replay, KVLifecycleReplay):
            raise TypeError(
                "bound KV lifecycle replay must be KVLifecycleReplay or null")
        if (replay is None) != (cursor is None):
            raise ValueError(
                "bound KV lifecycle replay and cursor must be supplied together")
        if not isinstance(allow_independent_preemption, bool):
            raise TypeError(
                "allow_independent_preemption must be boolean")
        if not isinstance(owner_pool_name, str) or not owner_pool_name:
            raise ValueError(
                "bound KV lifecycle owner_pool_name must be non-empty")
        if (not isinstance(owner_ranks, (list, tuple))
                or not owner_ranks
                or any(
                    isinstance(rank, bool) or not isinstance(rank, int)
                    or rank < 0
                    for rank in owner_ranks)):
            raise ValueError(
                "bound KV lifecycle owner_ranks must be non-negative "
                "integers")
        owner_ranks = tuple(owner_ranks)
        if (owner_ranks != tuple(sorted(owner_ranks))
                or len(owner_ranks) != len(set(owner_ranks))):
            raise ValueError(
                "bound KV lifecycle owner_ranks must be unique and ascending")
        self._manager = manager
        self._replay = replay
        self._cursor = cursor
        self._owner_pool_name = owner_pool_name
        self._owner_ranks = owner_ranks
        self._allow_independent_preemption = allow_independent_preemption
        self._replay_transfer_request_ids = frozenset(
            event.request_id
            for step in replay.steps
            for event in step.events
        ) if replay is not None else frozenset()
        self._transfers = []
        self._saw_transfer_intent = False
        if (manager.fixed_prefix_hbm_percent is not None
                and replay is not None):
            raise ValueError("fixed prefix tiers cannot combine with lifecycle replay")

    def _fixed_prefix_events(
        self, backend_step: BackendStep,
    ) -> BackendStep:
        percent = self._manager.fixed_prefix_hbm_percent
        if percent is None:
            return backend_step
        block_tokens = self._manager.block_size_tokens
        placements = {}
        for request in backend_step.schedule.requests:
            if not request.trace_prefix_hit_tokens:
                continue
            hbm_blocks, host_blocks = self._manager.fixed_prefix_block_counts(
                request.scheduler_prefix_hit_tokens)
            placements[request.request_id] = (
                hbm_blocks * block_tokens,
                (hbm_blocks + host_blocks) * block_tokens)
        if not placements:
            return backend_step
        if any(event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                                KVCacheAction.STORE)
               for event in backend_step.kv_cache_events):
            raise ValueError("fixed prefix tiers cannot combine with native KV transfers")
        transformed = []
        for event in backend_step.kv_cache_events:
            if (event.request_id not in placements
                    or event.action not in (KVCacheAction.ALLOCATE, KVCacheAction.HIT)):
                transformed.append(event)
                continue
            host_start, host_end = placements[event.request_id]
            boundaries = sorted({event.token_start, event.token_end,
                max(event.token_start, min(host_start, event.token_end)),
                max(event.token_start, min(host_end, event.token_end))})
            for start, end in zip(boundaries, boundaries[1:]):
                tier = "host" if host_start <= start < host_end else "hbm"
                transformed.append(replace(event, tier=tier, token_start=start,
                    token_end=end, bytes=(end - start) * self._manager.config.bytes_per_token))
        loads = []
        for request_id, (host_start, host_end) in placements.items():
            if host_start < host_end:
                loads.append(KVCacheEvent(
                    request_id, KVCacheAction.LOAD, "host", host_start,
                    host_end, (host_end - host_start)
                    * self._manager.config.bytes_per_token,
                    source_tier="host", destination_tier="hbm",
                    ready_event=(f"fixed-prefix:{backend_step.schedule.step_id}:"
                        f"{request_id}:{host_start}-{host_end}")))
        if not loads:
            return replace(backend_step, kv_cache_events=tuple(transformed))
        first_release = next((index for index, event in enumerate(transformed)
            if event.action is KVCacheAction.RELEASE), len(transformed))
        return replace(backend_step, kv_cache_events=tuple(
            transformed[:first_release] + loads + transformed[first_release:]))

    def prepare_step(
        self,
        backend_step: BackendStep,
    ) -> tuple[BackendStep, Tuple[KVCacheEvent, ...]]:
        if not isinstance(backend_step, BackendStep):
            raise TypeError("KV lifecycle step must be a BackendStep")
        if set(backend_step.schedule.preempted_request_ids).intersection(
                request.request_id for request in backend_step.schedule.requests):
            raise NotImplementedError(
                "modeled Host KV lifecycle does not support folded native preemption/resume")
        backend_step = self._fixed_prefix_events(backend_step)
        if self._cursor is None:
            replay_events = ()
        else:
            replay_events = self._cursor.events_for_step(backend_step)
        if (self._cursor is not None
                and any(
                    event.action in (
                        KVCacheAction.LOAD, KVCacheAction.EVICT,
                        KVCacheAction.STORE)
                    for event in backend_step.kv_cache_events)):
            raise ValueError(
                "KV lifecycle replay cannot be combined with native "
                "backend LOAD/EVICT events")
        if replay_events:
            preempted_request_ids = frozenset(
                backend_step.schedule.preempted_request_ids)
            preemption_release_events = tuple(
                event for event in backend_step.kv_cache_events
                if (event.action is KVCacheAction.RELEASE
                    and event.request_id in preempted_request_ids))
            completion_release_events = tuple(
                event for event in backend_step.kv_cache_events
                if (event.action is KVCacheAction.RELEASE
                    and event.request_id not in preempted_request_ids))
            prefix_events = tuple(
                event for event in backend_step.kv_cache_events
                if event.action is not KVCacheAction.RELEASE)
            backend_step = replace(backend_step, kv_cache_events=preemption_release_events
                + prefix_events
                + replay_events
                + completion_release_events)
        transfer_events = tuple(
            event for event in backend_step.kv_cache_events
            if event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                                KVCacheAction.STORE)
        )
        preempted_or_resumed_request_ids = frozenset(
            backend_step.schedule.preempted_request_ids
        ) | frozenset(
            request.request_id
            for request in backend_step.schedule.requests
            if request.resumed_from_preemption
        )
        if (self._allow_independent_preemption
                and any(
                    request.mtp_acceptance_source_kind != "disabled"
                    for request in backend_step.schedule.requests)):
            raise NotImplementedError(
                "the validated Host KV/preemption slice requires MTP "
                "to remain disabled")
        if preempted_or_resumed_request_ids:
            if not self._allow_independent_preemption:
                raise NotImplementedError(
                    "modeled Host KV lifecycle preemption or resume is "
                    "outside the validated independent-request scope")
            overlap = (
                preempted_or_resumed_request_ids
                & self._replay_transfer_request_ids
            )
            if overlap:
                raise NotImplementedError(
                    "modeled Host KV lifecycle cannot LOAD/EVICT a request "
                    "that is preempted or resumed under recompute semantics: "
                    f"{sorted(overlap)!r}")
        preempted_request_ids = frozenset(
            backend_step.schedule.preempted_request_ids)
        host_replacement_releases = tuple(
            event for event in backend_step.kv_cache_events
            if (event.action is KVCacheAction.RELEASE
                and event.tier == "host")
        )
        for event in host_replacement_releases:
            self._manager.apply_capacity_event(event)
        preemption_release_events = tuple(
            event for event in backend_step.kv_cache_events
            if (event.action is KVCacheAction.RELEASE
                and event.tier == "hbm"
                and event.request_id in preempted_request_ids))
        released_preempted_request_ids = frozenset(
            event.request_id for event in preemption_release_events)
        if (preempted_request_ids
                and released_preempted_request_ids != preempted_request_ids):
            raise ValueError(
                "Host KV/preemption requires one pre-step RELEASE range for "
                "every preempted request")
        for event in preemption_release_events:
            self._manager.apply_capacity_event(event)
        lifecycle_phase = 0
        for event in backend_step.kv_cache_events:
            if event.action in (KVCacheAction.HIT, KVCacheAction.MISS):
                continue
            if event in preemption_release_events:
                continue
            if event in host_replacement_releases:
                continue
            if event.action in (
                    KVCacheAction.ALLOCATE, KVCacheAction.LOAD):
                event_phase = 0
            elif event.action in (KVCacheAction.EVICT, KVCacheAction.STORE):
                event_phase = 1
            elif event.action is KVCacheAction.RELEASE:
                event_phase = 2
            else:
                raise RuntimeError(
                    f"unknown KV lifecycle action: {event.action.value!r}")
            if event_phase < lifecycle_phase:
                raise ValueError(
                    "KV lifecycle events must preserve pre-step allocate/load, "
                    "post-compute evict, then release phase order")
            lifecycle_phase = event_phase
            if event.action is KVCacheAction.ALLOCATE:
                self._manager.apply_capacity_event(event)
            elif event.action in (
                    KVCacheAction.LOAD, KVCacheAction.EVICT,
                    KVCacheAction.STORE):
                self._manager.register_transfer_intent(
                    event,
                    pool_name=self._owner_pool_name,
                    ranks=self._owner_ranks,
                )
            elif event.action is KVCacheAction.RELEASE:
                pass
            else:
                raise RuntimeError(
                    f"unknown KV lifecycle action: {event.action.value!r}")
        return backend_step, transfer_events

    def complete_step(
        self,
        backend_step: BackendStep,
        transfer_events: Tuple[KVCacheEvent, ...],
        workload: object,
        kernel_plan: object,
        predictions: Mapping[str, object],
        execution_timeline: object,
    ) -> None:
        completions = []
        for event in transfer_events:
            completion = build_kv_cache_transfer_completion(
                event,
                workload,
                kernel_plan,
                predictions,
                execution_timeline,
                expected_pool_name=self._owner_pool_name,
                expected_ranks=self._owner_ranks,
            )
            completions.append((event, completion))
        for event, completion in sorted(
                completions,
                key=lambda item: (
                    item[1].end_time_s,
                    item[1].start_time_s,
                    item[1].ready_event,
                )):
            completed_event = self._manager.complete_transfer(completion)
            if completed_event != event:
                raise RuntimeError(
                    "KV lifecycle manager completed a different transfer intent")
            self._transfers.append(SimulationKVCacheTransfer(
                backend_step.schedule.step_id,
                event,
                completion,
            ))
            self._saw_transfer_intent = True
        for event in backend_step.kv_cache_events:
            if (event.action is KVCacheAction.RELEASE
                    and event.tier != "host"
                    and event.request_id not in
                    backend_step.schedule.preempted_request_ids):
                self._manager.apply_capacity_event(event)

    def prepare_independent_decode_step(self, backend_step):
        """Apply the actual ordered capacity events of a transfer-free step.

        The actual request metadata must show that prefix admission has
        already completed. Transfers, preemption and replay keep their
        ordinary sequential completion boundary.
        """
        if (self._cursor is not None or self._manager.fixed_prefix_hbm_percent is None
                or not _transfer_free_decode_step(backend_step)):
            raise ValueError("independent fixed-tier step requires actual transfer-free decode")
        prepared, transfers = self.prepare_step(backend_step)
        if transfers:
            raise RuntimeError("independent decode introduced a transfer intent")
        for event in prepared.kv_cache_events:
            if event.action is KVCacheAction.RELEASE and event.tier != "host":
                self._manager.apply_capacity_event(event)
        return prepared

    def apply_completed_simple_cpu_releases(
        self, events: tuple[KVCacheEvent, ...],
    ) -> None:
        for event in events:
            if event.action is not KVCacheAction.RELEASE or event.tier != "hbm":
                raise ValueError("completed SimpleCPU release must retire HBM")
            self._manager.apply_capacity_event(event)

    def finish(self) -> SimulationKVOffload:
        if self._cursor is not None:
            self._cursor.assert_exhausted()
        if (not self._saw_transfer_intent
                and self._manager.fixed_prefix_hbm_percent is None):
            raise NotImplementedError(
                "modeled Host KV offload requires explicit framework "
                "load/evict lifecycle intents")
        self._manager.assert_quiescent()
        final_tier_state = self._manager.manifest()
        return SimulationKVOffload(
            tuple(self._transfers),
            final_tier_state,
            None if self._replay is None else self._replay.manifest(),
            mode=("fixed_prefix_tier_replay"
                if self._manager.fixed_prefix_hbm_percent is not None
                else "modeled_host_kv_serial"),
            validation_status=final_tier_state["validation_status"],
        )



def _validate_dp_backend_placement(backend, pool, model):
    import re

    from src.serving.backend.vllm import VLLMBackend
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_plan

    config = pool.parallel_plan.config
    size = config.data_parallel_size
    if isinstance(backend, VLLMBackend):
        if backend.data_parallel_size != size:
            raise ValueError("native DP Scheduler replica count disagrees with physical deployment")
        # Legacy/injected single-replica control planes do not expose the
        # native0.30 receipt. They retain their existing validation path.
        if size == 1 and not hasattr(backend._control_plane, "runtime_info"):
            return
        info = backend.runtime_info
        if size == 1 and (not isinstance(info, dict) or "version" not in info or info["version"] != "0.30.0"):
            return
        replicas = info["replicas"] if size > 1 else (info,)
        expected_ranges = pipeline_layer_ranges_for_plan(pool.parallel_plan, num_layers=model.num_hidden_layers)
        expected_partition = tuple(end - first for first, end in expected_ranges)
        if (len(replicas) != size or info["model_config_sha256"] != model.config_sha256
                or any(replica["model_config_sha256"] != model.config_sha256
                    or replica["effective_tensor_parallel_size"] != config.tensor_parallel_size
                    or replica["effective_pipeline_parallel_size"] != config.pipeline_parallel_size
                    for replica in replicas)):
            raise ValueError("native Scheduler model or TP/PP geometry disagrees with physical deployment")
        if config.pipeline_parallel_size == 1:
            return
        for replica in replicas:
            fields = ("effective_pipeline_layer_ranges", "effective_pipeline_layer_partition",
                "native_worker_cache_owners_by_rank", "mtp_num_speculative_tokens")
            if any(field not in replica for field in fields):
                raise ValueError("native Scheduler is missing actual PP layer ownership")
            if (tuple(tuple(bounds) for bounds in replica["effective_pipeline_layer_ranges"]) != expected_ranges
                    or tuple(replica["effective_pipeline_layer_partition"]) != expected_partition):
                raise ValueError("native Scheduler actual PP layer ownership disagrees with physical deployment")
            owners = replica["native_worker_cache_owners_by_rank"]
            if len(owners) != config.tensor_parallel_size * config.pipeline_parallel_size:
                raise ValueError("native Scheduler cache owners must cover every local TP/PP rank")
            for rank, names in enumerate(owners):
                stage = rank // config.tensor_parallel_size
                first, end = expected_ranges[stage]
                expected_layers = set(range(first, end))
                if stage == config.pipeline_parallel_size - 1 and replica["mtp_num_speculative_tokens"]:
                    if model.model_type == 'deepseek_v4':
                        if (replica['speculative_method'] != 'dspark' or replica['mtp_num_speculative_tokens'] != 5
                                or tuple(replica['dspark_draft_layer_indices']) != tuple(range(model.num_hidden_layers,model.num_hidden_layers+3))):
                            raise ValueError('native DSpark Scheduler must retain all three final-stage cache owners')
                        expected_layers.update(replica['dspark_draft_layer_indices'])
                    else:
                        expected_layers.update(range(model.num_hidden_layers, model.num_hidden_layers + model.num_mtp_layers))
                layers = set()
                for name in names:
                    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                    if match is None:
                        raise ValueError("native Scheduler cache owner has no source decoder layer")
                    layers.add(int(match[1]))
                if layers != expected_layers:
                    raise ValueError("native Scheduler actual cache layer owners disagree with physical PP deployment")
    elif size != 1:
        raise NotImplementedError("coordinated DP requires actual per-replica native Scheduler/cache owners")

def validate_workload_emission_parallelism(
    deployment: DeploymentPlan,
    model: ModelConfig,
    *,
    allow_single_pool_tp2: bool,
    allow_single_pool_ep2: bool,
    allow_single_pool_tp1_pp: bool,
    allow_single_pool_tp2_pp2: bool,
    runtime_inputs: RuntimeInputs | None = None,
) -> None:
    """Reject parallel plans whose rank-local DAG is not implemented."""

    if not isinstance(allow_single_pool_tp2, bool):
        raise TypeError("allow_single_pool_tp2 must be boolean")
    if not isinstance(allow_single_pool_ep2, bool):
        raise TypeError("allow_single_pool_ep2 must be boolean")
    if not isinstance(allow_single_pool_tp1_pp, bool):
        raise TypeError("allow_single_pool_tp1_pp must be boolean")
    if not isinstance(allow_single_pool_tp2_pp2, bool):
        raise TypeError("allow_single_pool_tp2_pp2 must be boolean")
    if runtime_inputs is not None and not isinstance(runtime_inputs, RuntimeInputs):
        raise TypeError("emission source requires RuntimeInputs or None")
    validated_af_rank_local = (
        is_validated_standalone_af_tp2_scope(deployment, model)
        or is_validated_standalone_af_ep2_scope(deployment, model)
    )
    validated_tp1_tp2_multi_pool = (
        is_validated_tp1_tp2_multi_pool_scope(deployment, model))
    validated_pd_rank_local = (
        deployment.mode == "pd_disagg"
        and (validated_tp1_tp2_multi_pool or (
            runtime_inputs is not None
            and runtime_inputs.model_execution_version == "vllm_0_30_0"
            and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and is_gqa_pd_source_scope(deployment, model)))
    )
    validated_replicated_colocated_rank_local = (
        deployment.mode == "colocated"
        and validated_tp1_tp2_multi_pool
    )
    if (validated_pd_rank_local
            or validated_replicated_colocated_rank_local):
        for pool in deployment.worker_pools.values():
            operations = (("all_reduce", "all_gather") if pool.parallel_plan.config.tensor_parallel_size > 1 else ())
            if is_qwen3_moe_dp_sp_source_scope(pool, model):
                operations += ("all_gatherv", "reduce_scatterv")
            for operation in operations:
                pool.collective_for(operation)
    unsupported = []
    for pool_name in sorted(deployment.worker_pools):
        pool = deployment.worker_pools[pool_name]
        parallelism = pool.parallel_plan.config
        from src.serving.workload.runtime_inputs import DSparkFirstProposalInputs
        if (runtime_inputs is not None
                and isinstance(runtime_inputs.model_source_inputs, DSparkFirstProposalInputs)):
            from src.serving.workload.model_config import validate_named_target_identity
            validate_named_target_identity(model)
            draft = runtime_inputs.model_source_inputs.draft_model
            if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                    or pool.role != "colocated" or pool.hardware_id != "nvidia_h200"
                    or model.model_type != "deepseek_v4" or model.compute_dtype != "bf16"
                    or draft.model_id != model.model_id + "-DSpark"
                    or draft.dspark_block_size != 5
                    or draft.num_hidden_layers != model.num_hidden_layers
                    or any(getattr(model, key) != getattr(draft, key) for key in (
                        "hidden_size", "num_attention_heads", "head_dim", "vocab_size",
                        "hc_mult", "window_size", "compute_dtype", "weight_dtype"))
                    or len(pool.ranks) != 8 or parallelism.worker_count != 8
                    or parallelism.tensor_parallel_size != 8 or not parallelism.is_rank_local
                    or parallelism.enable_expert_parallel
                    or any(value != 1 for value in (
                        parallelism.pipeline_parallel_size, parallelism.sequence_parallel_size,
                        parallelism.context_parallel_size, parallelism.data_parallel_size))):
                raise NotImplementedError(
                    "DSpark emission requires paired official V4 target/draft on one H200 TP8 eager pool")
            for operation in ("all_reduce", "all_gather"):
                pool.collective_for(operation)
            continue
        if runtime_inputs is not None and runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_dspark":
            from src.serving.deployment.validated_scope import is_dspark_runtime_source_scope
            if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                    or not is_dspark_runtime_source_scope(pool, model, runtime_inputs)):
                raise NotImplementedError("DSpark runtime requires exact eager colocated V4 TP/DP/EP/PP source geometry")
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"): pool.collective_for(operation)
            if parallelism.data_parallel_size > 1:
                for operation in ("all_gatherv", "reduce_scatterv"): pool.collective_for(operation)
            if parallelism.pipeline_parallel_size > 1: pool.collective_for('broadcast')
            for stage in range(parallelism.pipeline_parallel_size - 1): pool.pipeline_p2p_for(stage, stage + 1)
            continue
        from src.serving.deployment.validated_scope import is_kimi_k3_runtime_source_scope
        if runtime_inputs is not None and runtime_inputs.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync":
            if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                    or not is_kimi_k3_runtime_source_scope(pool, model, runtime_inputs)):
                raise NotImplementedError("K3 initial source requires its exact eager geometry and explicit latent tier")
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
            if parallelism.data_parallel_size > 1:
                for operation in ("all_gatherv", "reduce_scatterv"):
                    pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        if len(pool.ranks) == 1 and parallelism.worker_count == 1:
            continue
        from src.serving.deployment.validated_scope import (is_dense_parallel_source_scope,
            is_qwen3_moe_rank_local_source_scope)
        if (runtime_inputs is not None and deployment.mode == "colocated"
                and any(predicate(pool, model,
                    model_execution_version=runtime_inputs.model_execution_version,
                    symbolic_forward_source=runtime_inputs.symbolic_forward_source,
                    graph_enabled=runtime_inputs.graph_execution.enabled)
                    for predicate in (is_dense_parallel_source_scope, is_qwen3_moe_rank_local_source_scope))):
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        from src.serving.deployment.validated_scope import (is_glm_distributed_source_scope,
            is_glm_bf16_tensor_source_scope, is_glm_bf16_pipeline_source_scope, is_glm_fp8_rank_local_source_scope,
            is_glm_fp8_mtp_rank_local_source_scope, is_glm_fp8_data_parallel_source_scope)
        from src.serving.workload.model_config import is_glm_mtp_data_parallel_model_scope
        glm_fp8_mtp_dp = (runtime_inputs is not None and model.weight_dtype == "fp8"
            and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
            and pool.hardware_id == "nvidia_h200" and len(pool.ranks) == parallelism.worker_count
            and isinstance(runtime_inputs.model_source_inputs, V32PaddedMTPInputs)
            and runtime_inputs.target_sampling_source == "vllm_0_30_rejection_raw_logits_bf16"
            and runtime_inputs.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
            and (is_glm_mtp_data_parallel_model_scope(model, parallelism) or is_glm_fp8_mtp_rank_local_source_scope(pool, model)))
        if (runtime_inputs is not None and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and runtime_inputs.symbolic_forward_source == f"vllm_0_30_v32_{model.weight_dtype}_sync_explicit_allreduce"
                and not runtime_inputs.graph_execution.enabled
                and (is_glm_distributed_source_scope(pool, model) or is_glm_bf16_tensor_source_scope(pool, model)
                    or is_glm_bf16_pipeline_source_scope(pool, model) or is_glm_fp8_rank_local_source_scope(pool, model)
                    or glm_fp8_mtp_dp
                    or deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                        and runtime_inputs.mtp_draft_sampling_source is None
                        and is_glm_fp8_data_parallel_source_scope(pool, model))
                and deployment.mode in ("colocated", "pd_disagg")):
            operations = ("all_reduce",) if parallelism.tensor_parallel_size > 1 else ()
            operations += (("all_gather", "all_gatherv", "reduce_scatterv")
                if parallelism.context_parallel_size > 1 or parallelism.data_parallel_size > 1 else ("all_gather",))
            for operation in operations:
                pool.collective_for(operation)
            continue
        if (runtime_inputs is not None and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
                and not runtime_inputs.graph_execution.enabled and is_qwen3_moe_dp_sp_source_scope(pool, model)
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1):
            operations = (("all_reduce", "all_gather") if parallelism.tensor_parallel_size > 1 else ())
            for operation in (*operations, "all_gatherv", "reduce_scatterv"):
                pool.collective_for(operation)
            continue
        from src.serving.deployment.validated_scope import is_kimi_k2_rank_local_source_scope
        from src.serving.deployment.validated_scope import is_deepseek_parallel_source_scope
        from src.serving.deployment.validated_scope import is_kimi_k2_data_parallel_source_scope
        if (runtime_inputs is not None and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and runtime_inputs.mtp_draft_sampling_source is None and runtime_inputs.mtp_proposer_execution_source is None
                and runtime_inputs.model_source_inputs is None
                and (is_deepseek_parallel_source_scope if model.model_type != "kimi_k2" else is_kimi_k2_data_parallel_source_scope)(pool, model,
                    model_execution_version=runtime_inputs.model_execution_version,
                    symbolic_forward_source=runtime_inputs.symbolic_forward_source,
                    graph_enabled=runtime_inputs.graph_execution.enabled)
                and (model.model_type != "deepseek_v4" or parallelism.pipeline_parallel_size == 1
                    or runtime_inputs.v4_source_use_deep_gemm is False)):
            operations = (("all_reduce", "all_gather") if parallelism.tensor_parallel_size > 1 else ())
            if parallelism.data_parallel_size > 1:
                operations += ("all_gatherv", "reduce_scatterv")
            for operation in operations:
                pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        from src.serving.workload.model_config import is_deepseek_v3_parallel_model_scope
        if (runtime_inputs is not None and runtime_inputs.model_execution_version == "vllm_0_27_1"
                and runtime_inputs.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
                and not runtime_inputs.graph_execution.enabled and deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1 and pool.hardware_id == "nvidia_h200"
                and parallelism.is_rank_local and not parallelism.enable_expert_parallel
                and len(pool.ranks) == parallelism.worker_count
                and is_deepseek_v3_parallel_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)):
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        if (runtime_inputs is not None and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla"
                and runtime_inputs.v4_source_use_deep_gemm is False and not runtime_inputs.graph_execution.enabled
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and pool.hardware_id == "nvidia_h200" and model.model_type == "deepseek_v4"
                and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"
                and is_deepseek_v4_parallel_model_scope(model, parallelism.tensor_parallel_size, parallelism.pipeline_parallel_size)
                and len(pool.ranks) == parallelism.worker_count
                and parallelism.is_rank_local and not parallelism.enable_expert_parallel):
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        if (runtime_inputs is not None and runtime_inputs.model_execution_version == "vllm_0_30_0"
                and runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
                and not runtime_inputs.graph_execution.enabled
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and is_kimi_k2_rank_local_source_scope(pool, model)):
            if parallelism.tensor_parallel_size > 1:
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
            for stage in range(parallelism.pipeline_parallel_size - 1):
                pool.pipeline_p2p_for(stage, stage + 1)
            continue
        if (runtime_inputs is not None
                and runtime_inputs.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
                and deployment.mode == "colocated" and len(deployment.worker_pools) == 1
                and pool.hardware_id == "nvidia_h200"
                and (model.model_id == "moonshotai/Kimi-K2.6" or (
                    model.model_id == "moonshotai/Kimi-K2.7-Code"
                    and runtime_inputs.model_execution_version == "vllm_0_30_0"
                    and not runtime_inputs.graph_execution.enabled))
                and model.model_type == "kimi_k2" and model.attention_layout == "mla"
                and model.compute_dtype == "bf16" and model.weight_dtype == "mixed_int4_bf16"
                and len(pool.ranks) == parallelism.worker_count == parallelism.tensor_parallel_size == 8
                and not parallelism.enable_expert_parallel
                and all(value == 1 for value in (parallelism.pipeline_parallel_size,
                    parallelism.sequence_parallel_size, parallelism.context_parallel_size, parallelism.data_parallel_size))):
            # The shared virtual Runner emits TP-sharded experts and both
            # vocabulary/hidden collectives. The legacy EP8 exception is
            # unchanged; this admits only the explicit EP-off source body.
            if model.model_id == "moonshotai/Kimi-K2.7-Code":
                from src.serving.workload.model_config import validate_named_target_identity
                validate_named_target_identity(model)
            for operation in ("all_reduce", "all_gather"):
                pool.collective_for(operation)
            continue
        if (runtime_inputs is not None and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
                and allow_single_pool_tp2 and deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1 and pool.hardware_id == "nvidia_h200"
                and model.model_type == "qwen3_moe" and model.attention_layout == "standard_gqa"
                and model.compute_dtype == model.weight_dtype == "bf16"
                and len(pool.ranks) == parallelism.worker_count == parallelism.tensor_parallel_size == 2
                and not parallelism.enable_expert_parallel
                and all(value == 1 for value in (parallelism.pipeline_parallel_size,
                    parallelism.sequence_parallel_size, parallelism.context_parallel_size, parallelism.data_parallel_size))):
            # The virtual EP-off Runner emits both vocabulary shards and
            # routed TP partials, with every collective producer validated by
            # lowering. Keep legacy/manual emission under its existing scope.
            for operation in ("all_reduce", "all_gather"):
                pool.collective_for(operation)
            continue
        if (validated_af_rank_local
                or validated_pd_rank_local
                or validated_replicated_colocated_rank_local):
            continue
        if model.model_type not in (
                "kimi_k2", "glm_moe_dsa", "qwen3", "qwen3_moe", "llama",
                "glm4_moe", "deepseek_v3", "deepseek_v4"):
            raise AssertionError(
                f"unhandled Serving model type: {model.model_type!r}")
        # The validated target-model scopes are one shared table
        # (src.serving.deployment.validated_scope): the A800 colocated
        # TP=EP=8 production pool (Kimi-K2.6 PP2 / GLM-5.2 PP4), the PD
        # TP8/PP1/EP8 pools on A800 or H200, the GLM-5.2 H200 aggregate and
        # PD TP8/PP1/EP1 layouts (rank-local emission is the same TP8 DAG as
        # EP8; EP only re-partitions MoE expert weights), the Kimi-K2.6 H200
        # colocated TP8/PP1/EP8 pool, the DeepSeek-V3 H200 aggregate
        # (TP8, or TP4 for the 32-layer local variant measured on four
        # H200s) and PD TP8/PP1/EP1 layouts, and the DeepSeek-V4 H200
        # colocated layouts (Flash TP4, Pro TP8).
        target_scope = validated_target_emission_scope(
            model,
            deployment_mode=deployment.mode,
            hardware_id=pool.hardware_id,
            single_pool=len(deployment.worker_pools) == 1,
            pool_rank_count=len(pool.ranks),
            worker_count=parallelism.worker_count,
            is_rank_local=parallelism.is_rank_local,
            allow_production_pipeline=allow_single_pool_tp1_pp,
            tensor_parallel_size=parallelism.tensor_parallel_size,
            pipeline_parallel_size=parallelism.pipeline_parallel_size,
            expert_parallel=parallelism.enable_expert_parallel,
            expert_parallel_size=parallelism.expert_parallel_size,
        )
        if target_scope is not None:
            if target_scope.pipeline_parallel_size > 1:
                tuple(
                    pool.pipeline_p2p_for(source_stage, source_stage + 1)
                    for source_stage in range(
                        target_scope.pipeline_parallel_size - 1)
                )
            else:
                tuple(
                    pool.collective_for(operation)
                    for operation in ("all_reduce", "all_gather")
                )
            continue
        supported_ep2 = (
            allow_single_pool_ep2
            and deployment.mode == "colocated"
            and len(deployment.worker_pools) == 1
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.enable_expert_parallel
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == 2
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and is_validated_qwen3_moe_ep2_model_scope(model)
        )
        if supported_ep2:
            tuple(
                pool.collective_for(operation)
                for operation in ("all_reduce", "all_gather")
            )
            continue
        if parallelism.enable_expert_parallel:
            raise NotImplementedError(
                "rank-local EP workload/kernel emission currently supports "
                "only the validated BF16 Qwen3MoE TP=EP=2 all-sparse slice, "
                "the target-model A800 colocated TP=EP=8 pool (Kimi-K2.6 "
                "PP2 / GLM-5.2 PP4), target-model PD TP8/PP1/EP8 pools on "
                "A800 or H200, or the H200 Kimi-K2.6 colocated "
                "TP8/PP1/EP8 pool, all with DP=SP=CP=1")
        supported_tp2_pp2 = (
            allow_single_pool_tp2_pp2
            and deployment.mode == "colocated"
            and len(deployment.worker_pools) == 1
            and len(pool.ranks) == 4
            and parallelism.worker_count == 4
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == 1
            and parallelism.pipeline_parallel_size == 2
            and parallelism.is_rank_local
            and pool.hardware_id == "nvidia_a800"
            and model.model_type == "qwen3"
            and model.attention_layout == "standard_gqa"
            and model.num_experts == 0
            and model.compute_dtype == "bf16"
            and model.weight_dtype == "bf16"
        )
        if supported_tp2_pp2:
            for operation in ("all_reduce", "all_gather"):
                pool.collective_for(operation)
            pool.pipeline_p2p_for(0, 1)
            continue
        supported_tp2 = (
            allow_single_pool_tp2
            and deployment.mode == "colocated"
            and len(deployment.worker_pools) == 1
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == 1
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and (
                model.model_type == "qwen3"
                or (
                    pool.hardware_id == "nvidia_a800"
                    and is_validated_llama_tp2_model_scope(model)
                )
                or (pool.hardware_id == "nvidia_h200" and runtime_inputs is not None
                    and runtime_inputs.model_execution_version == "vllm_0_30_0"
                    and runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
                    and runtime_inputs.graph_execution.configured_mode in (
                        "NONE", "PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
                    and model.model_type == "llama" and model.architecture == "LlamaForCausalLM"
                    and model.compute_dtype == model.weight_dtype == "bf16" and model.head_dim == 128
                    and model.hidden_act == "silu" and not model.attention_bias
                    and model.rope_scaling_type in ("default", "llama3") and model.rms_norm_eps is not None)
            )
            and model.attention_layout == "standard_gqa"
            and model.num_experts == 0
            and model.compute_dtype in ("bf16", "fp16")
            and model.weight_dtype == model.compute_dtype
        )
        if supported_tp2:
            tuple(
                pool.collective_for(operation)
                for operation in ("all_reduce", "all_gather")
            )
            continue
        pipeline_size = parallelism.pipeline_parallel_size
        supported_tp1_pp = (
            allow_single_pool_tp1_pp
            and deployment.mode == "colocated"
            and len(deployment.worker_pools) == 1
            and pipeline_size >= 2
            and len(pool.ranks) == pipeline_size
            and parallelism.worker_count == pipeline_size
            and parallelism.tensor_parallel_size == 1
            and parallelism.expert_parallel_size == 1
            and parallelism.is_rank_local
            and pool.hardware_id == "nvidia_a800"
            and model.model_type == "qwen3"
            and model.attention_layout == "standard_gqa"
            and model.num_experts == 0
            and not model.tie_word_embeddings
            and model.compute_dtype == "bf16"
            and model.weight_dtype == "bf16"
            and model.num_hidden_layers % pipeline_size == 0
        )
        if supported_tp1_pp:
            layers_per_stage = model.num_hidden_layers // pipeline_size
            expected_placement = tuple(
                layer // layers_per_stage
                for layer in range(model.num_hidden_layers)
            )
            if (pool.parallel_plan.layer_to_pipeline_stage
                    != expected_placement):
                raise NotImplementedError(
                    "the validated uniform TP=1 PP path requires contiguous "
                    "equal-sized layer placement")
            tuple(
                pool.pipeline_p2p_for(source_stage, source_stage + 1)
                for source_stage in range(pipeline_size - 1)
            )
            continue
        unsupported.append((
            pool_name,
            pool.ranks,
            parallelism.tensor_parallel_size,
            parallelism.expert_parallel_size,
            parallelism.pipeline_parallel_size,
            parallelism.sequence_parallel_size,
            parallelism.context_parallel_size,
            parallelism.data_parallel_size,
        ))
    if unsupported:
        raise NotImplementedError(
            "Serving workload/kernel emission currently requires one "
            "physical rank and TP=EP=PP=SP=CP=DP=1 per worker pool; "
            "the independently validated exceptions are single-colocated "
            "Qwen3 TP=2, uniform TP=1 PP, TP=2 with PP=2, "
            "Qwen3MoE TP=EP=2, "
            "standalone AF TP=2, the target-model A800 colocated "
            "TP=EP=8 pool (Kimi-K2.6 PP2 / GLM-5.2 PP4), target-model "
            "PD TP8/PP1/EP8 or GLM-5.2 H200 PD TP8/PP1/EP1 pools, the "
            "GLM-5.2 H200 colocated TP8/PP1/EP1 aggregate pool, and the "
            "Kimi-K2.6 H200 colocated TP8/PP1/EP8 pool; unsupported "
            "rank-local DAG or communication "
            f"emission: {unsupported!r}")


def _attribute_parallel_phase_wall_time(
        performance_stats, parallel_phase_wall_time_s,
        parallel_execution_wall_time_s) -> None:
    """Share the parallel execution's wall time across its phases.

    Each phase takes the execution wall time in proportion to its own
    measured phase time.  A run that reports no phase time at all cannot be
    attributed, so it fails loud rather than dividing by zero.
    """

    aggregate_phase_s = sum(parallel_phase_wall_time_s.values())
    if aggregate_phase_s <= 0:
        raise RuntimeError(
            "parallel execution reported no phase time")
    for name, value in parallel_phase_wall_time_s.items():
        performance_stats["phase_wall_time_s"][name] += (
            parallel_execution_wall_time_s * value / aggregate_phase_s)


def _new_parallel_accumulators():
    """Create the per-run accumulators the parallel executor fills in.

    The cache-delta accumulator keeps entries, hits and misses for each
    active kernel cache and the phase accumulator one wall-time slot per
    pipeline phase; the step-id map collects each step's parallel results.
    """

    results_by_step_id = {}
    cache_deltas = {
        name: [0, 0, 0]
        for name in ("kernel_lowering", "kernel_prediction")
    }
    phase_wall_time_s = {
        name: 0.0
        for name in (
            "workload_construction", "kernel_lowering",
            "kernel_prediction", "timeline_construction")
    }
    return results_by_step_id, cache_deltas, phase_wall_time_s


def _with_in_step_transfer_trace(
    schedule_trace: ScheduleTrace,
    simulation_steps: Tuple[SimulationStep, ...],
) -> ScheduleTrace:
    """Project transfer KernelCalls onto the global scheduler time axis."""

    entries_by_step_id = {
        entry.backend_step.schedule.step_id: entry
        for entry in schedule_trace.entries
    }
    if len(entries_by_step_id) != len(schedule_trace.entries):
        raise RuntimeError("schedule trace contains duplicate step IDs")
    transfers = list(schedule_trace.transfers)
    for simulation_step in simulation_steps:
        step_id = simulation_step.backend_step.schedule.step_id
        if step_id not in entries_by_step_id:
            raise RuntimeError(
                f"simulation step {step_id!r} has no schedule trace entry")
        schedule_entry = entries_by_step_id[step_id]
        calls_by_id = {
            call.call_id: call for call in simulation_step.kernel_plan.calls
        }
        intervals_by_id = {
            interval.call_id: interval
            for interval in simulation_step.execution_timeline.intervals
        }
        repeated_trace_node_ids = {
            template_node_id
            for repeat in simulation_step.workload.block_repeats
            for template_node_id, _ in repeat.trace_node_ids
        }

        def append_activation_transfer(node, transfer_id, interval_times):
            calls = tuple(
                call for call in simulation_step.kernel_plan.calls
                if node.node_id in call.source_node_ids
            )
            if len(calls) != 1:
                raise RuntimeError(
                    f"activation transfer {node.node_id!r} lacks one kernel call")
            call = calls[0]
            resource_id = node.attributes["resource_id"]
            if call.resource != resource_id:
                raise RuntimeError(
                    f"activation transfer {node.node_id!r} resource disagrees "
                    "with its kernel resource")
            start_time_s, end_time_s = interval_times
            transfers.append(TransferTraceEntry(
                transfer_id=transfer_id,
                request_ids=tuple(
                    request.request_id
                    for request in simulation_step.backend_step.schedule.requests
                ),
                payload="activation",
                source_pool_id=node.pool_name,
                destination_pool_id=node.attributes["destination_pool"],
                resource_id=resource_id,
                link=node.attributes["link"],
                reserved_resource_ids=call_reserved_resource_ids(call),
                byte_count=node.attributes["bytes"],
                start_time_s=(
                    schedule_entry.start_time_s
                    + schedule_entry.scheduler_plan_cpu_s
                    + schedule_entry.device_queue_wait_s
                    + start_time_s),
                end_time_s=(
                    schedule_entry.start_time_s
                    + schedule_entry.scheduler_plan_cpu_s
                    + schedule_entry.device_queue_wait_s
                    + end_time_s),
            ))

        for node in simulation_step.workload.nodes:
            if node.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER:
                continue
            if node.node_id in repeated_trace_node_ids:
                continue
            if node.node_id not in calls_by_id or node.node_id not in intervals_by_id:
                raise RuntimeError(
                    f"activation transfer {node.node_id!r} lacks a kernel interval")
            call = calls_by_id[node.node_id]
            interval = intervals_by_id[node.node_id]
            resource_id = node.attributes["resource_id"]
            if (call.resource != resource_id
                    or interval.resource != resource_id):
                raise RuntimeError(
                    f"activation transfer {node.node_id!r} resource disagrees "
                    "with its kernel resource")
            append_activation_transfer(
                node,
                node.node_id,
                (interval.start_time_s, interval.end_time_s),
            )
        timeline_repeats_by_id = {
            repeat.block_id: repeat
            for repeat in simulation_step.execution_timeline.block_repeats
        }
        for workload_repeat in simulation_step.workload.block_repeats:
            if not workload_repeat.trace_node_ids:
                continue
            if workload_repeat.block_id not in timeline_repeats_by_id:
                raise RuntimeError(
                    "activation transfer repeat lacks timeline metadata")
            timeline_repeat = timeline_repeats_by_id[
                workload_repeat.block_id]
            intervals_by_call_id = dict(timeline_repeat.call_intervals)
            for template_node_id, logical_node_ids in (
                    workload_repeat.trace_node_ids):
                node = simulation_step.workload.node(template_node_id)
                calls = tuple(
                    call for call in simulation_step.kernel_plan.calls
                    if template_node_id in call.source_node_ids
                )
                if len(calls) != 1:
                    raise RuntimeError(
                        "repeated activation transfer must lower to one call")
                call_id = calls[0].call_id
                if call_id not in intervals_by_call_id:
                    raise RuntimeError(
                        "repeated activation transfer lacks occurrence times")
                occurrence_intervals = intervals_by_call_id[call_id]
                if len(logical_node_ids) != len(occurrence_intervals):
                    raise RuntimeError(
                        "activation transfer identities and intervals disagree")
                for transfer_id, interval_times in zip(
                        logical_node_ids, occurrence_intervals, strict=True):
                    append_activation_transfer(
                        node, transfer_id, interval_times)
    simulation_steps_by_id = {
        step.backend_step.schedule.step_id: step
        for step in simulation_steps
    }
    if set(simulation_steps_by_id) != set(entries_by_step_id):
        raise RuntimeError(
            "simulation steps and schedule trace entries disagree")
    effective_entries = tuple(
        ScheduleTraceEntry(
            simulation_steps_by_id[
                entry.backend_step.schedule.step_id].backend_step,
            entry.start_time_s,
            entry.end_time_s,
            entry.timing_mode,
            entry.scheduler_plan_cpu_s,
            entry.scheduler_update_cpu_s,
            entry.execution_wall_s,
            entry.device_queue_wait_s,
            entry.completion_queue_wait_s,
        )
        for entry in schedule_trace.entries
    )
    return ScheduleTrace(
        effective_entries,
        schedule_trace.route_decisions,
        sorted(
            transfers,
            key=lambda transfer: (
                transfer.start_time_s,
                transfer.transfer_id,
            ),
        ),
    )


def _emit_weight_placements(config, consumer, v4_weight_storage_profile):
    """Emit the same optional startup inventory for either simulation root."""
    if v4_weight_storage_profile is not None and consumer is None:
        raise ValueError("weight storage profile requires a placement consumer")
    if consumer is not None:
        if not callable(consumer):
            raise TypeError("weight placement consumer must be callable")
        from src.serving.kv_cache.model_residency import deployment_weight_placements
        consumer(deployment_weight_placements(config.model, config.deployment,
            v4_weight_storage_profile=v4_weight_storage_profile,
            model_execution_version=config.runtime_inputs.model_execution_version))


class Simulation:
    """Single-use composition root that owns and closes its backend."""

    def __init__(
        self,
        config: SimulationConfig,
        backend: SchedulerBackend,
        hardware_by_id: Mapping[str, object],
        kv_cache_tier_manager: Optional[KVCacheTierManager] = None,
        kv_lifecycle_replay: Optional[KVLifecycleReplay] = None,
        scheduler_runtime_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[BackendStep, float], None]] = None,
        streaming_step_consumer: Optional[
            Callable[[SimulationStep, ScheduleTraceEntry], None]
        ] = None,
        streaming_step_projector: Optional[Callable[[SimulationStep], object]] = None,
        streaming_prehashed_step_consumer: Optional[Callable[..., None]] = None,
        weight_placement_consumer: Optional[Callable[[tuple[dict, ...]], None]] = None,
        v4_weight_storage_profile: Optional[str] = None,
        adaptive_tail_workers: Optional[Callable[[], int]] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.hardware_by_id = dict(hardware_by_id)
        self._adaptive_tail_workers = adaptive_tail_workers
        from src.serving.backend.vllm import VLLMBackend
        simple_cpu_runtime = None
        if (isinstance(backend, VLLMBackend)
                and hasattr(backend._control_plane, "runtime_info")):
            runtime = backend.runtime_info
            if (isinstance(runtime, dict)
                    and "kv_connector" in runtime
                    and runtime["kv_connector"] == "SimpleCPUOffloadConnector"):
                simple_cpu_runtime = runtime
        self._modeled_simple_cpu = simple_cpu_runtime is not None
        if (kv_cache_tier_manager is not None or kv_lifecycle_replay is not None
                or config.closed_loop_client_count is not None):
            if (any(request.prompt_token_ids is not None for request in config.requests)
                    and not (simple_cpu_runtime is not None
                             and kv_cache_tier_manager is not None
                             and kv_lifecycle_replay is None
                             and config.closed_loop_client_count is None)):
                raise NotImplementedError("token identity replay does not support external KV tiers/lifecycle or closed-loop clients")
        self.kv_cache_tier_manager = kv_cache_tier_manager
        self.kv_lifecycle_replay = kv_lifecycle_replay
        self.scheduler_runtime_sha256 = scheduler_runtime_sha256
        self.progress_callback = progress_callback
        self.streaming_step_consumer = streaming_step_consumer
        self.streaming_step_projector = streaming_step_projector
        self.streaming_prehashed_step_consumer = (
            streaming_prehashed_step_consumer)
        self._has_run = False
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("Simulation progress callback must be callable")
        if (streaming_step_consumer is not None
                and not callable(streaming_step_consumer)):
            raise TypeError(
                "Simulation streaming step consumer must be callable")
        if (config.step_retention == "full"
                and streaming_step_consumer is not None):
            raise ValueError(
                "full step retention cannot use a streaming step consumer")
        if (config.step_retention == "streaming_aggregate"
                and streaming_step_consumer is None):
            raise ValueError(
                "streaming aggregate retention requires a step consumer")
        if ((streaming_step_projector is None)
                != (streaming_prehashed_step_consumer is None)):
            raise ValueError(
                "parallel streaming projection and consumption must be "
                "configured together")
        if (config.deployment.mode != "colocated"
                or len(config.deployment.worker_pools) != 1):
            raise ValueError(
                "Simulation requires exactly one colocated scheduler pool; "
                "use MultiSimulation for routed or multi-pool deployments")
        pool_name = next(iter(config.deployment.worker_pools))
        _validate_dp_backend_placement(backend, config.deployment.worker_pools[pool_name], config.model)
        if backend.config.pool_name != pool_name:
            raise ValueError(
                "backend pool does not match the single deployment pool")
        config.runtime_inputs.expert_routing.assert_scheduler_config(
            pool_name=pool_name,
            scheduler_config_sha256=backend_config_sha256(backend.config),
        )
        pool = config.deployment.worker_pools[pool_name]
        if (backend_phase(backend.config.role) != worker_role_phase(pool.role)
                or worker_role_operator_group(backend.config.role)
                != worker_role_operator_group(pool.role)):
            raise ValueError(
                f"backend {pool_name!r} role {backend.config.role!r} "
                f"does not match deployment role {pool.role!r}")
        if (kv_cache_tier_manager is not None
                and not isinstance(
                    kv_cache_tier_manager, KVCacheTierManager)):
            raise TypeError(
                "Simulation KV cache tier manager must be "
                "KVCacheTierManager or null")
        if (kv_lifecycle_replay is not None
                and not isinstance(kv_lifecycle_replay, KVLifecycleReplay)):
            raise TypeError(
                "Simulation KV lifecycle replay must be "
                "KVLifecycleReplay or null")
        if (kv_lifecycle_replay is not None
                and kv_cache_tier_manager is None):
            raise ValueError(
                "KV lifecycle replay requires a tier lifecycle manager")
        if (kv_lifecycle_replay is None
                and scheduler_runtime_sha256 is not None):
            raise ValueError(
                "Scheduler runtime SHA256 is only valid with a KV "
                "lifecycle replay")
        if (kv_lifecycle_replay is not None
                and scheduler_runtime_sha256 is None):
            raise ValueError(
                "KV lifecycle replay requires a Scheduler runtime SHA256")
        if kv_cache_tier_manager is not None:
            if pool.kv_cache is None:
                raise ValueError(
                    "KV cache tier manager requires a deployment cache plan")
            if kv_cache_tier_manager.config != pool.kv_cache:
                raise ValueError(
                    "KV cache tier manager config disagrees with deployment")
        if kv_lifecycle_replay is None:
            kv_lifecycle_cursor = None
        else:
            kv_lifecycle_cursor = kv_lifecycle_replay.bind(
                config.model,
                config.deployment,
                config.requests,
                backend.config,
                scheduler_runtime_sha256,
            )
        if kv_cache_tier_manager is None:
            self._kv_lifecycle = None
        else:
            self._kv_lifecycle = _BoundKVLifecycle(
                kv_cache_tier_manager,
                kv_lifecycle_replay,
                kv_lifecycle_cursor,
                owner_pool_name=pool.name,
                owner_ranks=pool.ranks,
                allow_independent_preemption=(
                    _is_validated_host_kv_independent_preemption_scope(
                        config,
                        backend.config,
                        kv_lifecycle_replay,
                    )
                ),
            )
        validate_workload_emission_parallelism(
            config.deployment,
            config.model,
            runtime_inputs=config.runtime_inputs,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )
        if ((pool.parallel_plan.config.tensor_parallel_size == 2
             or pool.parallel_plan.config.pipeline_parallel_size > 1)
                and kv_cache_tier_manager is not None
                and not is_validated_colocated_tp2_host_kv_scope(
                    config.deployment, config.model)):
            raise NotImplementedError(
                    "rank-local TP=2/uniform-PP emission does not yet support KV "
                    "offload outside the validated colocated Qwen3 TP=2 scope")
        if simple_cpu_runtime is not None:
            if (kv_cache_tier_manager is None
                    or kv_lifecycle_replay is not None
                    or kv_cache_tier_manager.fixed_prefix_hbm_percent is not None
                    or any(request.prompt_token_ids is None
                           or request.output_token_ids is None
                           or request.prefix_hit_tokens != 0
                           for request in config.requests)):
                raise NotImplementedError(
                    "native SimpleCPU offload requires a tier manager and "
                    "explicit token IDs without trace hits or lifecycle replay")
            block_bytes = kv_cache_tier_manager.block_bytes
            cpu_bytes = simple_cpu_runtime["simple_cpu_offload_bytes_per_rank"]
            geometry = simple_cpu_runtime["kv_group_geometry"]
            tiers = {
                tier.name: tier for tier in kv_cache_tier_manager.config.tiers
            }
            if (len(geometry) != 1
                    or tuple(geometry[0]) != (
                        backend.config.kv_block_size_tokens, block_bytes)
                    or block_bytes != (backend.config.kv_bytes_per_token
                                       * backend.config.kv_block_size_tokens)
                    or len(pool.ranks) != simple_cpu_runtime["effective_tensor_parallel_size"]
                    or tiers["hbm"].capacity_bytes
                        != backend.config.num_kv_blocks * block_bytes
                    or tiers["host"].capacity_bytes
                        != cpu_bytes // block_bytes * block_bytes):
                raise ValueError(
                    "native SimpleCPU tier capacity or rank geometry disagrees "
                    "with the real Scheduler")
            backend.enable_modeled_simple_cpu_offload()
        _emit_weight_placements(config, weight_placement_consumer, v4_weight_storage_profile)

    def _async_output_release(self) -> str:
        """HWSpec-declared release semantics for this pool's async outputs."""

        pool_name = self.backend.config.pool_name
        worker_pools = self.config.deployment.worker_pools
        if pool_name not in worker_pools:
            raise KeyError(
                f"deployment has no worker pool named {pool_name!r}")
        hardware_id = worker_pools[pool_name].hardware_id
        if hardware_id not in self.hardware_by_id:
            raise KeyError(f"no HWSpec for hardware {hardware_id!r}")
        release = self.hardware_by_id[hardware_id].async_output_release
        if release is None:
            raise ValueError(
                "Serving requires features.async_output_release in the HWSpec "
                f"of {hardware_id!r}")
        return release

    @experiment_cache_scope()
    def run(self) -> SimulationResult | StreamingSimulationResult:
        if self._has_run:
            raise RuntimeError("Simulation instances are single-use")
        self._has_run = True
        if self.config.observation_end_time_s is not None:
            raise NotImplementedError("finite observations currently require MultiSimulation")
        source_memory = SourceActivationMemoryAccumulator(rank_pools={str(rank): pool.name
            for pool in self.config.deployment.worker_pools.values() for rank in pool.ranks})
        retain_full_steps = self.config.step_retention == "full"
        simulation_steps = []
        pending_streaming_steps = {}
        streamed_step_count = 0
        streamed_kernel_call_count = 0
        streamed_step_timings = []
        (parallel_results_by_step_id, parallel_cache_deltas,
         parallel_phase_wall_time_s) = _new_parallel_accumulators()
        parallel_worker_count = 0
        self._adaptive_tail_grants = []
        self._adaptive_tail_pool = None
        parallel_step_count = 0
        parallel_execution_wall_time_s = 0.0
        execution = ServingExecutionContext(
            self.config.model,
            self.config.deployment,
            self.hardware_by_id,
            self.config.launch_policy,
            self.config.prediction_method,
            self.config.runtime_inputs,
            execution_mode=self.config.execution_mode,
            fresh_kernel_predictions=self.config.fresh_kernel_predictions,
            host_launch_profiles=self.config.host_launch_profiles,
            host_runtime_prepare_profiles=self.config.host_runtime_prepare_profiles,
            runtime_prepare_event_builder=self.config.runtime_prepare_event_builder,
            materialize_timeline_details=(retain_full_steps or self.config.record_execution_timeline),
            model_source_inputs_by_step=self.config.model_source_inputs_by_step,
            collect_source_state=self.config.collect_source_state,
        )
        collection_window = ExitStack()
        collection_window_steps = 0

        @_metadata_collection_scope()
        def step_duration(backend_step, *, origin_time_s=0.0,
                          device_floor_s=0.0, return_worker_service=False):
            nonlocal collection_window_steps
            if self._kv_lifecycle is None:
                if any(
                        event.action in (
                            KVCacheAction.LOAD, KVCacheAction.EVICT,
                            KVCacheAction.STORE)
                        for event in backend_step.kv_cache_events):
                    raise ValueError(
                        "KV load/evict events require a tier lifecycle manager")
                transfer_events = ()
            else:
                backend_step, transfer_events = (
                    self._kv_lifecycle.prepare_step(backend_step))
            components = execution.execute_step(
                backend_step.schedule,
                origin_time_s=origin_time_s,
                device_floor_s=device_floor_s,
                kv_cache_events=backend_step.kv_cache_events,
                embedding_token_ids=backend_step.embedding_token_ids,
                model_source_inputs=backend_step.model_source_inputs,
                worker_request_state=backend_step.worker_request_state,
                require_timeline_details=bool(transfer_events),
            )
            # The step already defers cyclic scans while constructing metadata.
            # Keep that scope over a bounded run of bound recorded sources,
            # avoiding scans of the same immutable cached program after each
            # decode. This is a Python-object budget, not a hardware parameter.
            # Unbound sources, other phases and the finite boundary resume the
            # caller's collection policy; reference counting never changes.
            # Host-inclusive scheduling consumes the same bounded metadata;
            # timing profiles do not change Python object ownership.
            cached_decode = (not retain_full_steps and self.config.execution_mode in ("accelerated", "fast")
                and self._kv_lifecycle is None
                and execution._recorded_source_program_bound
                and backend_step.schedule.token_count > 0
                and all(request.phase == "decode" and request.scheduled_tokens == 1
                    and request.mtp_proposer_mode == "disabled" for request in backend_step.schedule.requests)
                and self.config.runtime_inputs.graph_execution.resolve(
                    backend_step.schedule.token_count, uniform_decode=True).runtime_mode == "FULL")
            if cached_decode:
                if collection_window_steps == 0:
                    collection_window.enter_context(_metadata_collection_scope())
                collection_window_steps += 1
            if not cached_decode or collection_window_steps >= _METADATA_COLLECTION_WINDOW_STEPS:
                collection_window.close()
                collection_window_steps = 0
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            execution_timeline = components.execution_timeline
            simulation_step = SimulationStep(
                backend_step=backend_step,
                workload=workload,
                kernel_plan=kernel_plan,
                predictions=predictions,
                execution_timeline=execution_timeline,
                source_activation_memory=components.source_activation_memory,
                source_state_summary=components.source_state_summary,
            )
            if components.source_activation_memory is not None:
                source_memory.consume(backend_step.schedule.pool_name, backend_step.schedule.step_id,
                    components.source_activation_memory)
            project_immediately = (not retain_full_steps and self._kv_lifecycle is None
                and not self.config.collect_source_state
                and self.config.timing_mode == "device_only"
                and self.streaming_step_projector is not None)
            if retain_full_steps:
                simulation_steps.append(simulation_step)
            elif not project_immediately:
                step_id = backend_step.schedule.step_id
                if step_id in pending_streaming_steps:
                    raise RuntimeError(
                        "streaming simulation received a duplicate pending step")
                pending_streaming_steps[step_id] = simulation_step
            if self._kv_lifecycle is not None:
                self._kv_lifecycle.complete_step(
                    backend_step,
                    transfer_events,
                    workload,
                    kernel_plan,
                    predictions,
                    execution_timeline,
                )
            if project_immediately:
                step_id = backend_step.schedule.step_id
                if step_id in parallel_results_by_step_id:
                    raise RuntimeError("streaming projection received a duplicate pending step")
                # Project the complete immutable execution contract now. The
                # consumer still binds its actual virtual completion interval
                # later, exactly like the existing parallel-tail protocol.
                # Keeping the full IR alive until that callback is unnecessary.
                parallel_results_by_step_id[step_id] = _ParallelStepExecution(
                    execution_timeline.duration_s,
                    self.streaming_step_projector(simulation_step),
                    kernel_plan.logical_call_count, bool(kernel_plan.block_repeats),
                    {}, {}, None)  # Source memory was consumed above, once.
            if return_worker_service:
                return (execution_timeline.duration_s,
                        execution_timeline.worker_host_wall_s)
            return execution_timeline.duration_s

        def on_step_complete(backend_step, virtual_time_s):
            nonlocal streamed_step_count, streamed_kernel_call_count
            if self._modeled_simple_cpu:
                if self._kv_lifecycle is None:
                    raise RuntimeError("modeled SimpleCPU lost its tier lifecycle")
                self._kv_lifecycle.apply_completed_simple_cpu_releases(
                    self.backend.take_modeled_simple_cpu_releases())
            step_id = backend_step.schedule.step_id
            if step_id in parallel_results_by_step_id:
                parallel_result = parallel_results_by_step_id.pop(step_id)
                if parallel_result.source_activation_memory is not None:
                    source_memory.consume(backend_step.schedule.pool_name, step_id, parallel_result.source_activation_memory)
                entry = ScheduleTraceEntry(
                    backend_step,
                    virtual_time_s - parallel_result.duration_s,
                    virtual_time_s,
                )
                completion_time_s = virtual_time_s
                arguments = (
                    backend_step,
                    entry,
                    parallel_result.execution_contract_digest,
                    parallel_result.logical_kernel_call_count,
                    parallel_result.compact,
                )
                if parallel_result.state_audit_snapshot is None:
                    self.streaming_prehashed_step_consumer(*arguments)
                else:
                    self.streaming_prehashed_step_consumer(
                        *arguments, parallel_result.state_audit_snapshot)
                if parallel_result.step_timing is not None:
                    streamed_step_timings.append(parallel_result.step_timing)
                streamed_step_count += 1
                streamed_kernel_call_count += (
                    parallel_result.logical_kernel_call_count)
                if self.progress_callback is not None:
                    self.progress_callback(backend_step, completion_time_s)
                return
            if not retain_full_steps:
                if step_id not in pending_streaming_steps:
                    raise RuntimeError(
                        "streaming simulation completed without a pending step")
                pending_streaming_step = pending_streaming_steps.pop(step_id)
                if pending_streaming_step.backend_step.schedule is not backend_step.schedule:
                    raise RuntimeError(
                        "streaming simulation completed a different step")
                duration_s = (
                    pending_streaming_step.execution_timeline.duration_s)
                entry = ScheduleTraceEntry(
                    pending_streaming_step.backend_step,
                    virtual_time_s - duration_s,
                    virtual_time_s,
                )
                completion_time_s = virtual_time_s
                self.streaming_step_consumer(
                    pending_streaming_step, entry)
                if self.config.timing_mode != "device_only":
                    timeline = pending_streaming_step.execution_timeline
                    streamed_step_timings.append(StreamingStepTiming(
                        step_id,
                        timeline.predicted_device_time_s,
                        timeline.launch_cpu_time_s,
                        timeline.launch_critical_path_s,
                        timeline.host_device_overlap_s,
                        timeline.eager_dispatch_count,
                        timeline.graph_replay_count,
                        timeline.launch_kinds,
                    ))
                streamed_step_count += 1
                streamed_kernel_call_count += (
                    pending_streaming_step.kernel_plan.logical_call_count)
            completion_time_s = virtual_time_s
            if self.progress_callback is not None:
                self.progress_callback(backend_step, completion_time_s)

        def normalize_capture_history(backend_step, result):
            if not self.config.runtime_inputs.graph_execution.enabled:
                return result
            report = result.source_activation_memory
            if not isinstance(report, Mapping) or not isinstance(report["ranks"], Mapping):
                raise TypeError("parallel capture history requires per-rank source memory")
            ranks = {}
            changed = False
            for rank, facts in report["ranks"].items():
                if "source_captures" in facts:
                    raise NotImplementedError("parallel capture history requires one capture per rank")
                if "source_capture" not in facts:
                    ranks[rank] = facts
                    continue
                capture = facts["source_capture"]
                key = (backend_step.schedule.pool_name, int(rank), capture["key"])
                static_ports = capture["static_ports"]
                replay = key in execution._source_capture_ports
                if replay and execution._source_capture_ports[key] != static_ports:
                    raise RuntimeError("parallel source capture changed static input/output identity or typed extent")
                if not replay:
                    execution._source_capture_ports[key] = static_ports
                updated = {**capture, "logical_replay": replay}
                changed = changed or updated != capture
                ranks[rank] = {**facts, "source_capture": updated}
            if not changed:
                return result
            return replace(result, source_activation_memory={**report, "ranks": ranks})

        def execute_fixed_source_segment(remaining):
            """Execute one actual transfer-free decode segment completely."""
            nonlocal parallel_worker_count, parallel_step_count, parallel_execution_wall_time_s
            if not remaining:
                return
            independent = all(_transfer_free_decode_step(step) for step in remaining)
            available_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            memory_worker_limit = max(
                1, (available_bytes - 64 * 1024 ** 3) // (2 * 1024 ** 3))
            workers = min(
                _apply_tail_worker_limit(
                    self.config.execution_workers,
                    len(_platform_compat.sched_getaffinity(0)),
                    memory_worker_limit,
                ),
                len(remaining),
            )
            if (not independent or workers < 2
                    or not _fixed_source_segment_fits(execution._source_program_cache, remaining)
                    or (self.config.runtime_inputs.symbolic_workspace_initial_capacity_bytes is not None
                        and execution._independent_decode_workspace_capacities(remaining[0].schedule) is None)):
                for step in remaining:
                    yield step, step_duration(step)
                return
            if self._kv_lifecycle is None:
                raise RuntimeError("fixed-source tail lost its actual KV lifecycle")
            self._kv_lifecycle._manager.assert_quiescent()
            # Actual ALLOCATE/RELEASE events retain native order and capacity
            # validation. There are no outstanding transfers at this boundary.
            remaining = tuple(self._kv_lifecycle.prepare_independent_decode_step(step) for step in remaining)
            # Source-program IDs name the first actual occurrence of a shape.
            # Warm precisely those real occurrences before forking so every
            # worker inherits the same complete program and source namespace
            # that sequential execution would have used. No price files or
            # summaries are read, and all this preparation remains timed.
            programs_by_shape = {}
            for step in remaining:
                shape = tuple(request.scheduled_tokens for request in step.schedule.requests)
                if shape in programs_by_shape:
                    continue
                capacities = (execution._independent_decode_workspace_capacities(step.schedule)
                    if execution.runtime_inputs.symbolic_workspace_initial_capacity_bytes is not None else None)
                components = execution.execute_step(step.schedule, kv_cache_events=step.kv_cache_events,
                    embedding_token_ids=step.embedding_token_ids, model_source_inputs=step.model_source_inputs,
                    worker_request_state=step.worker_request_state)
                programs_by_shape[shape] = _fixed_source_program_key(
                    execution._source_program_cache, components.workload)
                if (capacities is not None
                        and execution._independent_decode_workspace_capacities(step.schedule) != capacities):
                    raise RuntimeError("source preparation changed certified decode workspace")
            # First-occurrence warming establishes the real namespace. Every
            # actual lookup must also reach the parent cache in step order:
            # otherwise a later segment can evict a different namespace.
            _restore_fixed_source_program_order(execution._source_program_cache, remaining, programs_by_shape)
            parallel_worker_count = workers
            with multiprocessing.get_context("fork").Pool(workers,
                    initializer=_initialize_causal_execution_worker,
                    initargs=(execution, self.streaming_step_projector)) as cpu_workers:
                original_workers = tuple(cpu_workers._pool)
                batch_size = (min(_PARALLEL_TAIL_BATCH_SIZE, 2 * workers)
                    if self.config.collect_source_state else _PARALLEL_TAIL_BATCH_SIZE)
                for offset in range(0, len(remaining), batch_size):
                    batch = remaining[offset:offset + batch_size]
                    started = time.perf_counter()
                    pending = cpu_workers.map_async(_parallel_execute_fixed_decode_step, batch,
                        chunksize=max(1, len(batch) // (workers * 4)))
                    results = _resolve_causal_cpu_result(pending, original_workers)
                    parallel_execution_wall_time_s += time.perf_counter() - started
                    for step, result in zip(batch, results, strict=True):
                        step_id = step.schedule.step_id
                        if step_id in parallel_results_by_step_id:
                            raise RuntimeError("fixed-source tail produced a duplicate step")
                        parallel_results_by_step_id[step_id] = result
                        parallel_step_count += 1
                        for name, delta in result.cache_deltas.items():
                            totals = parallel_cache_deltas[name]
                            for index, value in enumerate(delta):
                                totals[index] += value
                        for name, value in result.phase_wall_time_s.items():
                            parallel_phase_wall_time_s[name] += value
                        yield step, result.duration_s

        def drain_fixed_source_tail(backend):
            """Replay native steps in order, retaining real prefix/transfer barriers.

            Once all arrivals are injected, device-only native decisions do
            not consume predicted durations. Transfers and prefills still run
            in the parent; independent decode on either side uses its own
            actually established KV state and source workspace capacity.
            """
            segment = []
            has_more = backend.has_work()
            while has_more:
                steps, has_more = backend.drain_steps(_PARALLEL_TAIL_BATCH_SIZE)
                if not steps and has_more:
                    raise RuntimeError("fixed-source tail control plane made no progress")
                for step in steps:
                    if _transfer_free_decode_step(step):
                        segment.append(step)
                    else:
                        yield from execute_fixed_source_segment(segment)
                        segment = []
                        yield step, step_duration(step)
            yield from execute_fixed_source_segment(segment)

        def drain_tail(backend):
            nonlocal parallel_worker_count, parallel_step_count
            nonlocal parallel_execution_wall_time_s
            if isinstance(backend, TailBatchSchedulerBackend):
                backend_steps, has_more = backend.drain_steps(
                    _PARALLEL_TAIL_BATCH_SIZE)
            else:
                backend_steps = []
                while backend.has_work():
                    backend_step = backend.plan_step(0.0)
                    if backend_step is None:
                        raise RuntimeError(
                            "backend has work but plan_step returned no step")
                    backend.complete_step(backend_step, 0.0)
                    backend_steps.append(backend_step)
                backend_steps = tuple(backend_steps)
                has_more = False
            if not backend_steps:
                if has_more:
                    raise RuntimeError(
                        "backend tail batch made no progress while work "
                        "remains")
                return
            if self.config.timing_mode != "device_only" and not hasattr(backend, "scheduler_update_cpu_s"):
                raise NotImplementedError("Host parallel tail requires profiled update CPU")
            available_bytes = (
                os.sysconf("SC_AVPHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE"))
            memory_worker_limit = max(
                1,
                (available_bytes - 64 * 1024 ** 3) // (2 * 1024 ** 3),
            )
            requested_workers = self.config.execution_workers
            parallel_worker_limit = _apply_tail_worker_limit(
                requested_workers, os.cpu_count() or 1, memory_worker_limit)
            self._adaptive_tail_pool = None

            def apply_adaptive_tail_grant():
                """Consume one grant per batch boundary, preserving all caps.

                A malformed grant fails closed instead of being silently
                ignored; the step count bound is applied by each caller on
                the returned value.
                """
                nonlocal parallel_worker_limit
                if self._adaptive_tail_workers is None:
                    return parallel_worker_limit
                grant = self._adaptive_tail_workers()
                if grant is None:
                    return parallel_worker_limit
                parallel_worker_limit = _apply_tail_worker_limit(
                    _validate_tail_worker_grant(grant),
                    os.cpu_count() or 1, memory_worker_limit)
                return parallel_worker_limit

            parallel_worker_count = min(
                apply_adaptive_tail_grant(),
                len(backend_steps),
            )
            parallel_execution = ServingExecutionContext(
                self.config.model,
                self.config.deployment,
                self.hardware_by_id,
                self.config.launch_policy,
                self.config.prediction_method,
                self.config.runtime_inputs,
                execution_mode=self.config.execution_mode,
                fresh_kernel_predictions=self.config.fresh_kernel_predictions,
                host_launch_profiles=self.config.host_launch_profiles,
                host_runtime_prepare_profiles=self.config.host_runtime_prepare_profiles,
                runtime_prepare_event_builder=self.config.runtime_prepare_event_builder,
                collect_source_state=self.config.collect_source_state,
                materialize_timeline_details=self.config.record_execution_timeline,
                model_source_inputs_by_step=self.config.model_source_inputs_by_step,
            )
            global _PARALLEL_EXECUTION_CONTEXT, _PARALLEL_STEP_PROJECTOR
            _PARALLEL_EXECUTION_CONTEXT = parallel_execution
            _PARALLEL_STEP_PROJECTOR = self.streaming_step_projector
            allow_capture_history = (
                self.config.execution_mode == "fast"
                and self.config.runtime_inputs.graph_execution.enabled)
            adaptive_grants = []

            def make_execute_tail_step(worker_count):
                if worker_count == 1:
                    return _single_worker_tail_executor(
                        parallel_execution, self.streaming_step_projector,
                        allow_capture_history=allow_capture_history)
                from functools import partial
                return partial(
                    _parallel_execute_step,
                    allow_capture_history=allow_capture_history,
                )

            try:
                pool = None
                pool_workers = 0
                current_execute_tail_step = None

                def ensure_pool(worker_count):
                    nonlocal pool, pool_workers, current_execute_tail_step
                    if pool is not None and pool_workers == worker_count:
                        return pool
                    if pool is not None:
                        pool.close()
                        pool.join()
                        pool = None
                        self._adaptive_tail_pool = None
                    pool_workers = worker_count
                    current_execute_tail_step = make_execute_tail_step(worker_count)
                    pool = _tail_execution_pool(worker_count)
                    self._adaptive_tail_pool = pool
                    return pool

                def submit_batch(steps):
                    applied_worker_count = min(
                        apply_adaptive_tail_grant(), len(steps))
                    if applied_worker_count not in adaptive_grants:
                        adaptive_grants.append(applied_worker_count)
                    if applied_worker_count not in self._adaptive_tail_grants:
                        self._adaptive_tail_grants.append(applied_worker_count)
                    pool = ensure_pool(applied_worker_count)
                    chunk_size = max(
                        1,
                        len(steps)
                        // (applied_worker_count * 4),
                    )
                    completed_at = []
                    completed_results = []
                    failed_with = []
                    started_at = time.perf_counter()

                    def record_completion(results):
                        completed_at.append(time.perf_counter())
                        completed_results.append(results)

                    def record_failure(error):
                        failed_with.append(error)

                    pending = pool.map_async(
                        current_execute_tail_step,
                        steps,
                        chunksize=chunk_size,
                        callback=record_completion,
                        error_callback=record_failure,
                    )
                    return (
                        applied_worker_count,
                        pending,
                        started_at,
                        completed_at,
                        completed_results,
                        failed_with,
                    )

                try:
                    pending_batch = submit_batch(backend_steps)
                    while backend_steps:
                        next_backend_steps = ()
                        next_has_more = False
                        if has_more:
                            next_backend_steps, next_has_more = (
                                backend.drain_steps(
                                    _PARALLEL_TAIL_BATCH_SIZE))
                            if not next_backend_steps and next_has_more:
                                raise RuntimeError(
                                    "backend tail batch made no progress "
                                    "while work remains")
                        (
                            batch_worker_count,
                            pending,
                            batch_started,
                            completed_at,
                            completed_results,
                            failed_with,
                        ) = pending_batch
                        if batch_worker_count != pool_workers:
                            raise RuntimeError(
                                "tail batch applied worker count disagrees "
                                "with the live pool size")
                        pending.wait()
                        if failed_with:
                            if len(failed_with) != 1:
                                raise RuntimeError(
                                    "parallel batch failure callback did not "
                                    "run exactly once")
                            raise failed_with[0]
                        if len(completed_at) != 1:
                            raise RuntimeError(
                                "parallel batch completion callback did not "
                                "run exactly once")
                        if len(completed_results) != 1:
                            raise RuntimeError(
                                "parallel batch result callback did not run "
                                "exactly once")
                        parallel_results = completed_results[0]
                        parallel_execution_wall_time_s += (
                            completed_at[0] - batch_started)
                        parallel_step_count += len(backend_steps)
                        if next_backend_steps:
                            next_pending_batch = submit_batch(next_backend_steps)
                        else:
                            next_pending_batch = None
                        for backend_step, result in zip(
                                backend_steps,
                                parallel_results,
                                strict=True):
                            result = normalize_capture_history(backend_step, result)
                            step_id = backend_step.schedule.step_id
                            if step_id in parallel_results_by_step_id:
                                raise RuntimeError(
                                    "parallel execution produced duplicate "
                                    f"step {step_id!r}")
                            parallel_results_by_step_id[step_id] = result
                            for name, delta in result.cache_deltas.items():
                                totals = parallel_cache_deltas[name]
                                for index, value in enumerate(delta):
                                    totals[index] += value
                            for name, value in result.phase_wall_time_s.items():
                                parallel_phase_wall_time_s[name] += value
                            if self.config.timing_mode == "device_plus_launch_plus_scheduler":
                                yield (backend_step, result.duration_s,
                                    backend.scheduler_update_cpu_s(backend_step, self.config.timing_mode))
                            else:
                                yield backend_step, result.duration_s
                        if next_pending_batch is None:
                            break
                        backend_steps = next_backend_steps
                        has_more = next_has_more
                        pending_batch = next_pending_batch
                finally:
                    # Exception cleanup cancels the speculative in-flight
                    # batch; the normal path drains it with close/join.
                    if pool is not None:
                        if sys.exc_info()[0] is None:
                            pool.close()
                            pool.join()
                        else:
                            pool.terminate()
                            pool.join()
                        pool = None
                        self._adaptive_tail_pool = None
            finally:
                if self._adaptive_tail_pool is None:
                    _PARALLEL_EXECUTION_CONTEXT = None
                    _PARALLEL_STEP_PROJECTOR = None

        legacy_parallel_tail = (
            not retain_full_steps
            and not self.config.collect_source_state
            and self._kv_lifecycle is None
            and self.config.execution_mode == "accelerated"
            and self.config.runtime_inputs.symbolic_workspace_initial_capacity_bytes is None
            and not (self.config.runtime_inputs.symbolic_forward_source is not None
                and self.config.runtime_inputs.graph_execution.enabled)
            and self.config.timing_mode == "device_only"
            and self.backend.config.scheduler_pipeline_depth == 1
            and self.config.closed_loop_client_count is None
            and self.streaming_step_projector is not None
            and sum(
                request.output_tokens for request in self.config.requests)
            >= 512
        )
        pool = next(iter(self.config.deployment.worker_pools.values()))
        fast_parallel_execution = (
            not retain_full_steps
            and self._kv_lifecycle is None
            and self.config.execution_mode == "fast"
            and self.config.deployment.mode == "colocated"
            and len(self.config.deployment.worker_pools) == 1
            and pool.parallel_plan.config.pipeline_parallel_size == 1
            and ((self.config.model.model_type == "qwen3_moe"
                and self.config.model.num_experts > 0
                and len(pool.ranks) == 2
                and pool.parallel_plan.config.tensor_parallel_size == 2
                and pool.parallel_plan.config.enable_expert_parallel)
                or (self.config.model.model_type in ("qwen3", "llama")
                and len(pool.ranks) == pool.parallel_plan.config.tensor_parallel_size
                and len(pool.ranks) in (1, 2)
                and not pool.parallel_plan.config.enable_expert_parallel))
            and self.config.runtime_inputs.symbolic_forward_source in (
                "cuda_eager_inplace", "vllm_ascend_eager")
            and self.config.runtime_inputs.model_execution_version in (
                "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0")
            and self.config.runtime_inputs.graph_execution.enabled
            and self.config.runtime_inputs.symbolic_workspace_initial_capacity_bytes is None
            and self.config.closed_loop_client_count is None
            and self.config.execution_workers != 1
            and self.backend.config.scheduler_pipeline_depth == 2
            and self.config.timing_mode == "device_plus_launch_plus_scheduler"
            and self.streaming_step_projector is not None
            and sum(
                request.output_tokens for request in self.config.requests)
            >= 512
        )
        #0.30 keeps its genuine online Scheduler. The older tail-drain
        # protocol cannot consume its state; only the existing causal,
        # two-pending-plan producer below may run its pure CPU work early.
        fast_parallel_tail = (fast_parallel_execution
            and self.config.runtime_inputs.model_execution_version != "vllm_0_30_0")
        use_parallel_tail = legacy_parallel_tail or fast_parallel_tail
        from src.serving.backend.vllm import VLLMBackend, _NativeVLLMControlPlane
        from src.serving.backend.vllm_scheduler_process import VLLMSchedulerProcess, VLLM030SchedulerProcess
        fixed_source_parallel_tail = (
            not retain_full_steps and self.config.execution_mode == "fast"
            and self.config.deployment.mode == "colocated"
            and len(self.config.deployment.worker_pools) == 1
            and pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 8
            and pool.parallel_plan.config.tensor_parallel_size == 8
            and pool.parallel_plan.config.expert_parallel_size == 1
            and all(value == 1 for value in (pool.parallel_plan.config.pipeline_parallel_size,
                pool.parallel_plan.config.context_parallel_size, pool.parallel_plan.config.sequence_parallel_size,
                pool.parallel_plan.config.data_parallel_size))
            and self.config.model.model_type in ("glm_moe_dsa", "kimi_k2")
            and self.config.runtime_inputs.model_execution_version == "vllm_0_30_0"
            and self.config.runtime_inputs.symbolic_forward_source in (
                "vllm_0_30_v32_bf16_sync_explicit_allreduce",
                "vllm_0_30_v32_fp8_sync_explicit_allreduce", "vllm_0_30_k2_6_int4_sync_explicit_allreduce")
            and not self.config.runtime_inputs.graph_execution.enabled
            and self.config.runtime_inputs.model_source_inputs is None
            and self.config.runtime_inputs.mtp_draft_sampling_source is None
            and self.config.runtime_inputs.mtp_proposer_execution_source is None
            and self.config.runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and self.config.model_source_inputs_by_step is None
            and self.config.closed_loop_client_count is None
            and self.config.timing_mode == "device_only"
            and self.config.execution_workers != 1
            and self.backend.config.scheduler_pipeline_depth == 1
            and type(self.backend) is VLLMBackend
            and type(self.backend._control_plane) is VLLM030SchedulerProcess
            and self._kv_lifecycle is not None
            and self._kv_lifecycle._cursor is None
            and self._kv_lifecycle._manager.fixed_prefix_hbm_percent is not None
            and self.streaming_step_projector is not None
            and self.streaming_prehashed_step_consumer is not None
            and len(_platform_compat.sched_getaffinity(0)) >= 2)
        causal_execution = (fast_parallel_execution
            and self._async_output_release() == "self"
            and _independent_runtime_prepare_builder(self.config.runtime_prepare_event_builder)
            and self.config.model_source_inputs_by_step is None
            and self.streaming_prehashed_step_consumer is not None
            and type(self.backend) is VLLMBackend
            and (type(self.backend._control_plane) is VLLMSchedulerProcess
                or (type(self.backend._control_plane) is VLLM030SchedulerProcess
                    and pool.hardware_id == "nvidia_h200"
                    and self.config.launch_policy == "analytical_launch"
                    and self.config.runtime_inputs.model_execution_version == "vllm_0_30_0"
                    and self.config.runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
                    and self.config.runtime_inputs.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
                    and self.config.runtime_inputs.model_source_inputs is None
                    and self.config.runtime_inputs.mtp_draft_sampling_source is None
                    and self.backend._control_plane.scheduler_config.mtp_num_speculative_tokens == 0)
                or (type(self.backend._control_plane) is _NativeVLLMControlPlane
                    and self.backend._in_process_baseline_libraries is not None))
            and self.backend._control_plane.scheduler_config.scheduler_host_profile is not None
            and len(_platform_compat.sched_getaffinity(0)) >= 2
            and os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") >= 72 * 1024 ** 3)
        cpu_pending = {}
        cpu_execution_intervals = []
        cpu_pool = None
        cpu_workers = ()
        cpu_clock_pipes = {}

        def close_cpu_clock_pipes():
            for sender, receiver in cpu_clock_pipes.values():
                sender.close()
                receiver.close()
            cpu_clock_pipes.clear()

        def submit_cpu_step(backend_step):
            step_id = backend_step.schedule.step_id
            if cpu_pool is None or step_id in cpu_pending or step_id in parallel_results_by_step_id:
                raise RuntimeError("causal CPU submission has no pool or duplicates a step")
            if len(cpu_pending) >= 2:
                raise RuntimeError("causal CPU execution exceeded its Scheduler slots")
            if source_ordered_parallel:
                receiver, sender = multiprocessing.get_context("fork").Pipe(duplex=False)
                cpu_clock_pipes[step_id] = sender, receiver
                cpu_pending[step_id] = cpu_pool.apply_async(
                    _source_ordered_causal_execute_step, (backend_step, receiver))
            else:
                cpu_pending[step_id] = cpu_pool.apply_async(_causal_execute_step, (backend_step,))

        def resolve_cpu_step(backend_step):
            nonlocal parallel_step_count
            step_id = backend_step.schedule.step_id
            if step_id not in cpu_pending:
                raise RuntimeError("causal CPU resolution has no submitted step")
            try:
                started, finished, result = _resolve_causal_cpu_result(
                    cpu_pending.pop(step_id), cpu_workers)
            finally:
                if step_id in cpu_clock_pipes:
                    sender, receiver = cpu_clock_pipes.pop(step_id)
                    sender.close()
                    receiver.close()
            cpu_execution_intervals.append((started, finished))
            result = normalize_capture_history(backend_step, result)
            parallel_results_by_step_id[step_id] = result
            parallel_step_count += 1
            for name, delta in result.cache_deltas.items():
                totals = parallel_cache_deltas[name]
                for index, value in enumerate(delta):
                    totals[index] += value
            for name, value in result.phase_wall_time_s.items():
                parallel_phase_wall_time_s[name] += value
            return result.duration_s

        closed_loop_clients = _closed_loop_clients(self.config)
        if closed_loop_clients is None:
            engine_requests = scheduler_visible_requests(
                self.config.requests,
                host_request_path_profile(self.config))
        else:
            engine_requests = RequestStream(
                closed_loop_clients.initial_requests())
        async_output_release = self._async_output_release()
        # WorkerProc.worker_busy_loop and vLLM0.30 UniProcExecutor both run
        # one Host call at a time. UniProc collective_rpc(non_block=True)
        # calls run_method on the EngineCore thread before returning a future;
        # only the output event wait is asynchronous. Retiming the per-call
        # DAG separates Host return from the step's own Device completion.
        h200_async_worker = (
            pool.hardware_id == "nvidia_h200"
            and self.config.runtime_inputs.model_execution_version == "vllm_0_30_0"
            and self.config.vllm_executor_backend is not None)
        source_async_worker = (
            self.backend.config.scheduler_pipeline_depth > 1
            and self.config.timing_mode != "device_only"
            and async_output_release == "self"
            and ({pool.hardware_id for pool in
                 self.config.deployment.worker_pools.values()}
                 == {"ascend_910b"} or h200_async_worker))

        source_ordered_parallel = (source_async_worker and causal_execution
            and pool.hardware_id == "nvidia_h200"
            and self.config.vllm_executor_backend == "mp")
        if source_async_worker:
            if self.config.vllm_executor_backend is None:
                raise ValueError("910B async Host timing requires explicit vLLM executor backend")
            # Only the FIFO owner determines absolute origins. Multiproc CPU
            # workers prepare graph/prices early and wait before the timeline;
            # UniProc must return its Host service before another plan exists.
            causal_execution = source_ordered_parallel
            use_parallel_tail = False

        from src.serving.workload.runtime_inputs import DSparkFirstProposalInputs
        source_request_progress = isinstance(
            self.config.runtime_inputs.model_source_inputs, DSparkFirstProposalInputs)
        if source_request_progress:
            if (self.backend.config.scheduler_pipeline_depth != 1
                    or self.config.timing_mode != "device_only"
                    or self.config.execution_workers not in (0, 1)):
                raise NotImplementedError(
                    "DSpark sequential source requires synchronous device-only execution and one state owner")
            # A predecessor ledger uses absolute completion times. The engine
            # already owns the actual start/floor; pass them through its
            # existing callback so a step returns elapsed time only once.
            causal_execution = False
            use_parallel_tail = False
            fixed_source_parallel_tail = False

        def execute_async_worker_step(backend_step, origin_time_s,
                                      device_floor_s):
            if source_ordered_parallel:
                step_id = backend_step.schedule.step_id
                if step_id not in cpu_clock_pipes:
                    raise RuntimeError("source-ordered CPU step has no origin pipe")
                sender, _receiver = cpu_clock_pipes[step_id]
                sender.send((origin_time_s, device_floor_s))
                duration_s = resolve_cpu_step(backend_step)
                return duration_s, parallel_results_by_step_id[step_id].worker_host_wall_s
            return step_duration(
                backend_step, origin_time_s=origin_time_s,
                device_floor_s=device_floor_s,
                return_worker_service=True)

        with ExitStack() as backend_lifecycle:
            backend_lifecycle.enter_context(
                _managed_backend(self.backend))
            backend_lifecycle.enter_context(collection_window)
            # The Kernel source recipe is immutable for this synchronous run.
            # Check it at entry and exit instead of rescanning it per step;
            # nested plan scopes still validate every operand boundary.
            from src.kernel.memory_access_builder import memory_prediction_source_scope
            backend_lifecycle.enter_context(memory_prediction_source_scope())
            if causal_execution:
                # The native Scheduler exposes at most two legal pending
                # plans. Keep these bounded run-owned caches warm; per-task
                # tail-worker recycling would rebuild them every four steps.
                cpu_pool = backend_lifecycle.enter_context(
                    multiprocessing.get_context("fork").Pool(2,
                        initializer=_initialize_causal_execution_worker,
                        initargs=(execution, self.streaming_step_projector)))
                cpu_workers = tuple(cpu_pool._pool)
                backend_lifecycle.callback(close_cpu_clock_pipes)
                parallel_worker_count = 2
            schedule_trace = run_single_pool_schedule(
                self.backend,
                engine_requests,
                resolve_cpu_step if causal_execution else step_duration,
                self.backend.config.scheduler_pipeline_depth,
                async_output_release=async_output_release,
                on_step_complete=on_step_complete,
                tail_drain=(drain_fixed_source_tail if fixed_source_parallel_tail else
                    drain_tail if use_parallel_tail and not causal_execution else None),
                timing_mode=self.config.timing_mode,
                closed_loop=closed_loop_clients,
                submit_step_duration=submit_cpu_step if causal_execution else None,
                step_execution_at=(
                    execute_async_worker_step if (source_async_worker or source_request_progress) else None),
                engine_core_worker_call_is_synchronous=(
                    source_async_worker
                    and self.config.vllm_executor_backend == "uni"),
            )
            if cpu_pending:
                raise RuntimeError("causal CPU execution ended with unresolved steps")
        # Actual worker occupancy includes execution, projection and metadata
        # scope exit. Queueing, IPC, pool lifecycle and parent Scheduler work
        # remain in the enclosing Simulation wall clock. Overlap is counted once.
        interval_end = -math.inf
        for started, finished in sorted(cpu_execution_intervals):
            parallel_execution_wall_time_s += max(0.0, finished - max(started, interval_end))
            interval_end = max(interval_end, finished)
        metrics_requests = (
            self.config.requests if closed_loop_clients is None
            else closed_loop_clients.effective_request_stream())
        if not retain_full_steps:
            if pending_streaming_steps:
                raise RuntimeError(
                    "streaming simulation ended with pending steps")
            if schedule_trace.transfers:
                raise NotImplementedError(
                    "streaming aggregate retention does not support trace "
                    "transfers")
            _assert_expert_routing_replay_coverage(
                self.config.runtime_inputs, ())
            cache_stats = execution.cache_stats()
            performance_stats = execution.performance_stats()
            if parallel_step_count:
                for name, totals in parallel_cache_deltas.items():
                    parent = cache_stats[name]
                    parent["entries"] += totals[0]
                    parent["hits"] += totals[1]
                    parent["misses"] += totals[2]
                _attribute_parallel_phase_wall_time(
                    performance_stats, parallel_phase_wall_time_s,
                    parallel_execution_wall_time_s)
                performance_stats["executed_workloads"] += parallel_step_count
            performance_stats["parallel_worker_count"] = parallel_worker_count
            performance_stats["parallel_step_count"] = parallel_step_count
            performance_stats["parallel_execution_wall_time_s"] = (
                parallel_execution_wall_time_s)
            performance_stats["parallel_worker_aggregate_phase_s"] = dict(
                parallel_phase_wall_time_s)
            if self._kv_lifecycle is not None:
                kv_offload = self._kv_lifecycle.finish()
            else:
                kv_offload = None
            return StreamingSimulationResult(
                schedule_trace=schedule_trace,
                metrics=compute_serving_metrics(
                    metrics_requests, schedule_trace,
                    _first_token_delivery_s(self.config)),
                num_steps=streamed_step_count,
                num_kernel_calls=streamed_kernel_call_count,
                step_timings=tuple(streamed_step_timings),
                execution_cache_stats=cache_stats,
                execution_performance_stats=performance_stats,
                source_activation_memory=source_memory.manifest(total_steps=streamed_step_count),
                kv_offload=kv_offload,
            )
        schedule_trace = _with_in_step_transfer_trace(
            schedule_trace, tuple(simulation_steps))
        if self._kv_lifecycle is not None:
            kv_offload = self._kv_lifecycle.finish()
        else:
            kv_offload = None
        _assert_expert_routing_replay_coverage(
            self.config.runtime_inputs,
            tuple(simulation_steps),
        )
        return SimulationResult(
            schedule_trace=schedule_trace,
            metrics=compute_serving_metrics(
                metrics_requests, schedule_trace,
                _first_token_delivery_s(self.config)),
            steps=tuple(simulation_steps),
            source_activation_memory=source_memory.manifest(total_steps=len(simulation_steps)),
            transfers=(),
            kv_offload=kv_offload,
            execution_cache_stats=execution.cache_stats(),
            execution_performance_stats=execution.performance_stats(),
            metrics_requests=(
                metrics_requests if closed_loop_clients is not None
                else None),
        )


def _is_discarded_async_terminal_step(
    backend_step: BackendStep,
    scheduled: ScheduledRequest,
    scheduler_pipeline_depth: int,
) -> bool:
    """Recognize vLLM's already-submitted terminal async tail.

    AsyncScheduler may submit one extra decode while the preceding terminal
    token is still an unresolved output placeholder.  vLLM executes that
    physical step but discards its sample after the earlier future removes
    the request.  The backend records this narrowly as one raw model output,
    zero emitted logical tokens, and no second finish event.
    """

    return (
        scheduler_pipeline_depth > 1
        and is_non_mtp_async_terminal_placeholder(scheduled)
        and backend_step.generated_tokens[scheduled.request_id] == 0
        and scheduled.request_id not in backend_step.finished_request_ids
    )


def _authorized_glm53_multinode_pd_mtp(config):
    pools = tuple(config.deployment.worker_pools.values())
    return (config.authorized_h200_pd_mtp_sweep and config.deployment.mode == "pd_disagg"
        and config.model.model_id == "zai-org/GLM-5.3" and config.model.weight_dtype == "fp8"
        and config.runtime_inputs.symbolic_forward_source == "vllm_0_30_v32_fp8_sync_explicit_allreduce"
        and {pool.role for pool in pools} == {"prefill", "decode"}
        and all(pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 8
            and pool.parallel_plan.config.tensor_parallel_size == 8
            and pool.parallel_plan.config.pipeline_parallel_size == 1
            and pool.parallel_plan.config.data_parallel_size == 1
            and pool.parallel_plan.config.context_parallel_size == 1
            and pool.parallel_plan.config.expert_parallel_size == 1
            and len(set(pool.node_ids)) == 1 for pool in pools)
        and len({pool.node_ids[0] for pool in pools}) == len(pools))


class MultiSimulation:
    """Single-use composition root owning all scheduler backends."""

    def __init__(
        self,
        config: SimulationConfig,
        backends: Mapping[str, SchedulerBackend],
        hardware_by_id: Mapping[str, object],
        *,
        kv_cache_tier_manager: Optional[KVCacheTierManager] = None,
        kv_lifecycle_replay: Optional[KVLifecycleReplay] = None,
        scheduler_runtime_sha256: Optional[str] = None,
        enable_standalone_af_mtp: bool = False,
        enable_standalone_af_mtp_preemption: bool = False,
        enable_pd_af_mtp: bool = False,
        pd_af_mtp_fixed_accepted_tokens: int | None = None,
        progress_callback: Optional[Callable[[BackendStep, float], None]] = None,
        streaming_step_consumer: Optional[
            Callable[[SimulationStep, ScheduleTraceEntry], None]
        ] = None,
        streaming_step_projector: Optional[
            Callable[[SimulationStep], object]
        ] = None,
        streaming_prehashed_step_consumer: Optional[Callable[..., None]] = None,
        weight_placement_consumer: Optional[Callable[[tuple[dict, ...]], None]] = None,
        v4_weight_storage_profile: Optional[str] = None,
        streaming_inflight_step_consumer: Optional[Callable[[SimulationStep, ScheduleTraceEntry], None]] = None,
        queue_observation_consumer: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self._glm53_multinode_pd_mtp = _authorized_glm53_multinode_pd_mtp(config)
        self.queue_observation_consumer = queue_observation_consumer
        if queue_observation_consumer is not None:
            if not callable(queue_observation_consumer):
                raise TypeError("queue observation consumer must be callable")
            from src.serving.backend.vllm import VLLMBackend
            if any(not isinstance(backend, VLLMBackend) for backend in backends.values()):
                raise NotImplementedError("native queue observation requires actual vLLM backends")
        self.streaming_inflight_step_consumer = streaming_inflight_step_consumer
        if config.observation_end_time_s is not None:
            if streaming_inflight_step_consumer is None or not callable(streaming_inflight_step_consumer):
                raise ValueError("finite observations require an explicit inflight resource consumer")
            if kv_cache_tier_manager is not None or streaming_step_projector is not None:
                raise NotImplementedError("finite observations do not yet support offload or projected parallel step results")
            from src.serving.backend.vllm import VLLMBackend
            if any(not isinstance(backend, VLLMBackend) for backend in backends.values()):
                raise NotImplementedError("finite observations require isolated native CPU backends")
        elif streaming_inflight_step_consumer is not None:
            raise ValueError("inflight consumer requires an observation end")
        self.backends = dict(backends)
        self.hardware_by_id = dict(hardware_by_id)
        self.kv_cache_tier_manager = kv_cache_tier_manager
        self.kv_lifecycle_replay = kv_lifecycle_replay
        self.scheduler_runtime_sha256 = scheduler_runtime_sha256
        self.progress_callback = progress_callback
        self.streaming_step_consumer = streaming_step_consumer
        self.streaming_step_projector = streaming_step_projector
        self.streaming_prehashed_step_consumer = (
            streaming_prehashed_step_consumer)
        self._has_run = False
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError(
                "MultiSimulation progress callback must be callable")
        if config.step_retention not in ("full", "streaming_aggregate"):
            raise ValueError(
                "MultiSimulation step retention must be full or "
                "streaming_aggregate")
        if (config.step_retention == "streaming_aggregate"
                and streaming_step_consumer is None):
            raise ValueError(
                "streaming aggregate retention requires a step consumer")
        if (config.step_retention == "full"
                and streaming_step_consumer is not None):
            raise ValueError(
                "full step retention cannot use a streaming step consumer")
        if ((streaming_step_projector is None)
                != (streaming_prehashed_step_consumer is None)):
            raise ValueError(
                "parallel streaming projection and consumption must be "
                "configured together")
        if (kv_cache_tier_manager is not None
                and not isinstance(
                    kv_cache_tier_manager, KVCacheTierManager)):
            raise TypeError(
                "MultiSimulation KV cache tier manager must be "
                "KVCacheTierManager or null")
        if (kv_lifecycle_replay is not None
                and not isinstance(kv_lifecycle_replay, KVLifecycleReplay)):
            raise TypeError(
                "MultiSimulation KV lifecycle replay must be "
                "KVLifecycleReplay or null")
        if (kv_lifecycle_replay is not None
                and kv_cache_tier_manager is None):
            raise ValueError(
                "KV lifecycle replay requires a tier lifecycle manager")
        if (kv_lifecycle_replay is None
                and scheduler_runtime_sha256 is not None):
            raise ValueError(
                "Scheduler runtime SHA256 is only valid with a KV "
                "lifecycle replay")
        if (kv_lifecycle_replay is not None
                and scheduler_runtime_sha256 is None):
            raise ValueError(
                "KV lifecycle replay requires a Scheduler runtime SHA256")
        if not isinstance(enable_standalone_af_mtp, bool):
            raise TypeError("standalone AF MTP enable flag must be boolean")
        if not isinstance(enable_standalone_af_mtp_preemption, bool):
            raise TypeError(
                "standalone AF MTP preemption enable flag must be boolean")
        if not isinstance(enable_pd_af_mtp, bool):
            raise TypeError("PD+AF MTP enable flag must be boolean")
        if (pd_af_mtp_fixed_accepted_tokens is not None
                and (
                    isinstance(pd_af_mtp_fixed_accepted_tokens, bool)
                    or not isinstance(
                        pd_af_mtp_fixed_accepted_tokens, int
                    )
                )):
            raise TypeError("PD+AF MTP fixed acceptance must be an integer")
        if (enable_pd_af_mtp
                and pd_af_mtp_fixed_accepted_tokens not in (0, 1, 2)):
            raise NotImplementedError(
                "PD+AF MTP requires fixed acceptance zero, one, or two")
        if (not enable_pd_af_mtp
                and pd_af_mtp_fixed_accepted_tokens is not None):
            raise ValueError(
                "PD+AF MTP fixed acceptance requires PD+AF MTP")
        if (enable_standalone_af_mtp_preemption
                and not enable_standalone_af_mtp):
            raise ValueError(
                "standalone AF MTP preemption requires standalone AF MTP")
        if enable_pd_af_mtp and enable_standalone_af_mtp:
            raise ValueError("PD+AF MTP cannot enable standalone AF MTP")
        if kv_cache_tier_manager is not None and enable_pd_af_mtp:
            raise NotImplementedError(
                "PD+AF Host KV lifecycle is not implemented")
        if (kv_cache_tier_manager is not None
                and enable_standalone_af_mtp_preemption):
            raise NotImplementedError(
                "standalone AF Host KV lifecycle with MTP currently "
                "requires the single-request non-preempting scope")
        self._enable_standalone_af_mtp = enable_standalone_af_mtp
        self._enable_standalone_af_mtp_preemption = (
            enable_standalone_af_mtp_preemption)
        self._enable_pd_af_mtp = enable_pd_af_mtp
        self._pd_af_mtp_fixed_accepted_tokens = (
            pd_af_mtp_fixed_accepted_tokens)
        standalone_af_ep_enabled = (
            config.deployment.mode == "af_disagg"
            and any(
                pool.parallel_plan.config.enable_expert_parallel
                for pool in config.deployment.worker_pools.values()
            )
        )
        if standalone_af_ep_enabled:
            validate_standalone_af_ep2_scope(config)
        if enable_standalone_af_mtp:
            validate_standalone_af_mtp_scope(
                config,
                allow_preemption=enable_standalone_af_mtp_preemption,
            )
        if enable_pd_af_mtp:
            validate_pd_af_mtp_scope(
                config,
                fixed_accepted_tokens=pd_af_mtp_fixed_accepted_tokens,
            )
        if not self.backends:
            raise ValueError("MultiSimulation requires at least one backend")
        if config.runtime_inputs.expert_routing.source_kind == (
                "replay_observation"):
            raise NotImplementedError(
                "expert routing replay v1 binds exactly one Scheduler pool")
        scheduler_engine_ids = {
            pool_name
            for pool_name, pool in config.deployment.worker_pools.items()
            if worker_role_operator_group(pool.role) != "ffn"
        }
        supplied_engine_ids = set(self.backends)
        if supplied_engine_ids != scheduler_engine_ids:
            missing = sorted(scheduler_engine_ids - supplied_engine_ids)
            unexpected = sorted(supplied_engine_ids - scheduler_engine_ids)
            raise ValueError(
                "backends must cover every deployment scheduler engine "
                f"exactly once: missing={missing}, unexpected={unexpected}")
        for pool_name, backend in self.backends.items():
            if backend.config.pool_name != pool_name:
                raise ValueError(
                    f"backend mapping key {pool_name!r} does not match its config")
            pool = config.deployment.worker_pools[pool_name]
            _validate_dp_backend_placement(backend, pool, config.model)
            if (backend_phase(backend.config.role)
                    != worker_role_phase(pool.role)
                    or worker_role_operator_group(backend.config.role)
                    != worker_role_operator_group(pool.role)):
                raise ValueError(
                    f"backend {pool_name!r} role {backend.config.role!r} "
                    f"does not match deployment role {pool.role!r}")
        if standalone_af_ep_enabled:
            _validate_standalone_af_ep2_backend_scope(config, self.backends)
        if kv_cache_tier_manager is None:
            self._kv_lifecycle = None
        else:
            if (config.deployment.mode != "af_disagg"
                    or len(config.deployment.worker_pools) != 2):
                raise NotImplementedError(
                    "MultiSimulation Host KV lifecycle currently supports "
                    "only standalone AF")
            attention_pools = tuple(
                pool for pool in config.deployment.worker_pools.values()
                if worker_role_operator_group(pool.role) == "attention"
                and worker_role_phase(pool.role) == "colocated"
            )
            ffn_pools = tuple(
                pool for pool in config.deployment.worker_pools.values()
                if worker_role_operator_group(pool.role) == "ffn"
                and worker_role_phase(pool.role) == "colocated"
            )
            if len(attention_pools) != 1 or len(ffn_pools) != 1:
                raise NotImplementedError(
                    "standalone AF Host KV lifecycle requires exactly one "
                    "Attention owner and one FFN worker pool")
            attention_pool = attention_pools[0]
            ffn_pool = ffn_pools[0]
            if set(self.backends) != {attention_pool.name}:
                raise NotImplementedError(
                    "standalone AF Host KV lifecycle requires exactly one "
                    "Attention Scheduler backend")
            if len(attention_pool.ranks) != 1 or len(ffn_pool.ranks) != 1:
                raise NotImplementedError(
                    "standalone AF Host KV lifecycle requires one physical "
                    "rank per worker pool")
            if enable_standalone_af_mtp:
                _validate_standalone_af_mtp_host_kv_scope(
                    config,
                    self.backends[attention_pool.name].config,
                    kv_lifecycle_replay,
                )
            if ffn_pool.kv_cache is not None:
                raise ValueError(
                    "standalone AF Host KV lifecycle requires the FFN pool "
                    "to have no KV cache")
            if attention_pool.kv_cache is None:
                raise ValueError(
                    "standalone AF Host KV lifecycle requires an Attention "
                    "cache plan")
            if kv_cache_tier_manager.config != attention_pool.kv_cache:
                raise ValueError(
                    "KV cache tier manager config disagrees with the AF "
                    "Attention pool")
            if kv_lifecycle_replay is None:
                cursor = None
            else:
                if kv_lifecycle_replay.context.pool_name != attention_pool.name:
                    raise ValueError(
                        "AF KV lifecycle replay must bind the Attention pool")
                cursor = kv_lifecycle_replay.bind(
                    config.model,
                    config.deployment,
                    config.requests,
                    self.backends[attention_pool.name].config,
                    scheduler_runtime_sha256,
                )
            self._kv_lifecycle = _BoundKVLifecycle(
                kv_cache_tier_manager,
                kv_lifecycle_replay,
                cursor,
                owner_pool_name=attention_pool.name,
                owner_ranks=attention_pool.ranks,
                allow_independent_preemption=False,
            )
        if enable_pd_af_mtp:
            scheduler_configs = tuple(
                backend.config for backend in self.backends.values())
            if (len(scheduler_configs) != 2
                    or any(
                        scheduler.max_batched_tokens != 128
                        or scheduler.max_requests != 1
                        or scheduler.num_kv_blocks != 8
                        or scheduler.kv_block_size_tokens != 16
                        for scheduler in scheduler_configs)):
                raise NotImplementedError(
                    "PD+AF MTP requires exactly two Attention Schedulers "
                    "with batch=128, max_requests=1, and eight 16-token KV "
                    "blocks each")
        if enable_standalone_af_mtp_preemption:
            attention_backends = tuple(
                backend for backend in self.backends.values()
                if worker_role_operator_group(backend.config.role)
                == "attention"
            )
            if (len(attention_backends) != 1
                    or attention_backends[0].config.max_batched_tokens != 64
                    or attention_backends[0].config.max_requests != 2
                    or attention_backends[0].config.num_kv_blocks != 6
                    or attention_backends[0].config.kv_block_size_tokens != 16):
                raise NotImplementedError(
                    "standalone AF MTP preemption requires exactly one "
                    "Attention Scheduler with batch=64, max_requests=2, and "
                    "six 16-token KV blocks")
        for route in config.deployment.routes:
            if route.payload != "activation":
                continue
            source_pool = config.deployment.worker_pools[route.source_pool]
            destination_pool = config.deployment.worker_pools[
                route.destination_pool]
            if source_pool.hardware_id != destination_pool.hardware_id:
                raise NotImplementedError(
                    "cross-hardware AF activation transfer requires a "
                    "bilateral endpoint/link predictor contract")
            validated_af_rank_local = (
                is_validated_standalone_af_tp2_scope(
                    config.deployment, config.model)
                or is_validated_standalone_af_ep2_scope(
                    config.deployment, config.model)
            )
            if ((len(source_pool.ranks) != 1
                 or len(destination_pool.ranks) != 1)
                    and not validated_af_rank_local):
                raise NotImplementedError(
                    "multi-rank AF activation transfer requires explicit "
                    "rank-pair or repartition semantics")
        phases = {
            pool_name: backend_phase(backend.config.role)
            for pool_name, backend in self.backends.items()
        }
        if config.deployment.mode in ("pd_disagg", "pd_af_disagg"):
            unexpected = sorted(
                pool_name for pool_name, phase in phases.items()
                if phase not in ("prefill", "decode")
            )
            if unexpected:
                raise ValueError(
                    "PD MultiSimulation contains non-P/D scheduler "
                    f"backends: {unexpected}")
            self._ingress_stage = "ingress_to_prefill"
            self._ingress_engine_ids = tuple(sorted(
                pool_name for pool_name, phase in phases.items()
                if phase == "prefill"
            ))
            self._decode_engine_ids = tuple(sorted(
                pool_name for pool_name, phase in phases.items()
                if phase == "decode"
            ))
            if not self._ingress_engine_ids or not self._decode_engine_ids:
                raise ValueError(
                    "PD MultiSimulation requires prefill and decode backends")
            for source in self._ingress_engine_ids:
                destinations = self._decode_destinations(source)
                if not destinations:
                    raise ValueError(
                        f"prefill engine {source!r} has no KV route to decode")
                for destination in destinations:
                    source_pool = config.deployment.worker_pools[source]
                    destination_pool = config.deployment.worker_pools[destination]
                    for source_dp in range(source_pool.parallel_plan.config.data_parallel_size):
                        for destination_dp in range(destination_pool.parallel_plan.config.data_parallel_size):
                            pd_kv_transfer_rank_pairs(source_pool, destination_pool, config.model.num_key_value_heads,
                                source_data_parallel_rank=source_dp, destination_data_parallel_rank=destination_dp,
                                attention_layout=config.model.attention_layout)
                    layout = validate_pd_kv_layout(source_pool, destination_pool, config.model.attention_layout)
                    if layout is not None:
                        if self.backends[source].config.kv_block_size_tokens != self.backends[destination].config.kv_block_size_tokens:
                            raise NotImplementedError("PD KV block-format conversion requires an explicit reblocking contract")
                        from src.serving.backend.vllm import VLLMBackend
                        for name in (source, destination):
                            backend = self.backends[name]
                            if isinstance(backend, VLLMBackend):
                                info = backend.runtime_info
                                if not isinstance(info, Mapping) or info["version"] != "0.30.0":
                                    raise NotImplementedError("declared PD layouts require the prepared0.30 cache-capability contract")
                                if any(layout not in capabilities for capabilities in info["supported_layouts_by_rank"]):
                                    raise ValueError("PD layout is outside an endpoint's declared per-rank cache capabilities")
        elif config.deployment.mode in ("colocated", "af_disagg"):
            unexpected = sorted(
                pool_name for pool_name, phase in phases.items()
                if phase != "colocated"
            )
            if unexpected:
                raise ValueError(
                    "colocated MultiSimulation contains non-colocated "
                    f"scheduler backends: {unexpected}")
            self._ingress_stage = "ingress_to_colocated"
            self._ingress_engine_ids = tuple(sorted(
                pool_name for pool_name, phase in phases.items()
                if phase == "colocated"
            ))
            self._decode_engine_ids = ()
            if not self._ingress_engine_ids:
                raise ValueError(
                    "colocated MultiSimulation requires colocated backends")
        else:
            raise ValueError(
                f"unsupported MultiSimulation mode: "
                f"{config.deployment.mode!r}")
        ingress_block_sizes = {
            engine_id: self.backends[
                engine_id].config.kv_block_size_tokens
            for engine_id in self._ingress_engine_ids
        }
        if len(set(ingress_block_sizes.values())) != 1:
            raise NotImplementedError(
                "prefix-aware routing requires every ingress Scheduler to "
                "use the same KV block size: "
                f"{ingress_block_sizes}")
        self._ingress_kv_block_size_tokens = next(iter(
            ingress_block_sizes.values()))
        validate_workload_emission_parallelism(
            config.deployment,
            config.model,
            runtime_inputs=config.runtime_inputs,
            allow_single_pool_tp2=False,
            allow_single_pool_ep2=False,
            allow_single_pool_tp1_pp=False,
            allow_single_pool_tp2_pp2=False,
        )
        _emit_weight_placements(config, weight_placement_consumer, v4_weight_storage_profile)

    def _decode_destinations(self, source_engine_id: str) -> Tuple[str, ...]:
        destinations = []
        for destination in self._decode_engine_ids:
            matches = [
                route for route in self.config.deployment.routes
                if route.payload == "kv"
                and route.source_pool == source_engine_id
                and route.destination_pool == destination
            ]
            if len(matches) > 1:
                raise ValueError(
                    "duplicate KV routes from "
                    f"{source_engine_id!r} to {destination!r}")
            if matches:
                destinations.append(destination)
        return tuple(destinations)

    @experiment_cache_scope()
    def run(self) -> SimulationResult | StreamingSimulationResult:
        if self._has_run:
            raise RuntimeError("MultiSimulation instances are single-use")
        self._has_run = True
        source_memory = SourceActivationMemoryAccumulator(rank_pools={str(rank): pool.name
            for pool in self.config.deployment.worker_pools.values() for rank in pool.ranks})
        retain_full_steps = self.config.step_retention == "full"
        simulation_steps = []
        pending_streaming_steps = {}
        streamed_step_count = 0
        streamed_kernel_call_count = 0
        streamed_step_timings = []
        simulation_transfers = []
        pd_kv_lifecycle = []
        (parallel_results_by_step_id, parallel_cache_deltas,
         parallel_phase_wall_time_s) = _new_parallel_accumulators()
        parallel_worker_count = 0
        parallel_step_count = 0
        parallel_execution_wall_time_s = 0.0
        requests_by_id = {
            request.request_id: request for request in self.config.requests
        }
        router = LeastQueuedTokensRouter(
            self.config.deployment.deployment_id,
            self.backends,
            self.config.deployment.router,
        )
        resource_calendar = ExecutionResourceCalendar()
        handed_off_request_ids = set()
        pending_handoffs = {}
        released_route_assignments = set()
        execution = ServingExecutionContext(
            self.config.model,
            self.config.deployment,
            self.hardware_by_id,
            self.config.launch_policy,
            self.config.prediction_method,
            self.config.runtime_inputs,
            execution_mode=self.config.execution_mode,
            fresh_kernel_predictions=self.config.fresh_kernel_predictions,
            host_launch_profiles=self.config.host_launch_profiles,
            host_runtime_prepare_profiles=self.config.host_runtime_prepare_profiles,
            runtime_prepare_event_builder=self.config.runtime_prepare_event_builder,
            materialize_timeline_details=(retain_full_steps or self.config.record_execution_timeline),
            model_source_inputs_by_step=self.config.model_source_inputs_by_step,
            collect_source_state=self.config.collect_source_state,
        )

        def request_placement(pool_name, request_id):
            from src.serving.backend.vllm import VLLMBackend
            pool = self.config.deployment.worker_pools[pool_name]
            size = pool.parallel_plan.config.data_parallel_size
            if size == 1:
                return 0, pool.ranks
            backend = self.backends[pool_name]
            if not isinstance(backend, VLLMBackend) or backend.data_parallel_size != size:
                raise ValueError("request placement requires this pool's actual DP Scheduler owners")
            replica = backend.request_data_parallel_rank(request_id)
            width = pool.parallel_plan.config.worker_count // size
            return replica, pool.ranks[replica * width:(replica + 1) * width]

        def step_duration(pool_name, backend_step, start_time_s):
            from src.serving.backend.vllm import VLLMBackend
            backend = self.backends[pool_name]
            if isinstance(backend, VLLMBackend) and backend.config.is_prefill_only:
                for request_id in backend_step.finished_request_ids:
                    if requests_by_id[request_id].output_tokens > 1:
                        blocks = backend.retain_pd_kv(request_id)
                        pd_kv_lifecycle.append({"action": "retain_source", "pool": pool_name,
                            "ranks": request_placement(pool_name, request_id)[1],
                            "data_parallel_rank": request_placement(pool_name, request_id)[0], "request_id": request_id, "time_s": start_time_s, "block_groups": blocks})
            if backend_step.schedule.pool_name != pool_name:
                raise ValueError("backend step pool does not match scheduler pool")
            if (self._enable_standalone_af_mtp
                    and not self._enable_standalone_af_mtp_preemption
                    and (
                        backend_step.schedule.preempted_request_ids
                        or any(
                            scheduled.resumed_from_preemption
                            for scheduled in backend_step.schedule.requests
                        ))):
                raise NotImplementedError(
                    "standalone AF MTP does not support preemption or resume")
            step_uses_mtp = any(
                scheduled.mtp_acceptance_source_kind != "disabled"
                for scheduled in backend_step.schedule.requests
            )
            if (step_uses_mtp
                    and self.config.deployment.mode != "colocated"):
                pool_phase = backend_phase(self.backends[pool_name].config.role)
                operator_group = worker_role_operator_group(
                    self.backends[pool_name].config.role)
                pd_mtp = (
                    self.config.deployment.mode == "pd_disagg"
                    and pool_phase == "decode"
                    and ((len(self._ingress_engine_ids) == 1 and len(self._decode_engine_ids) == 1)
                        or self._glm53_multinode_pd_mtp)
                    and (self.backends[pool_name].config.max_requests in (1, 64)
                         or (self.config.authorized_h200_pd_mtp_sweep
                             and self.backends[pool_name].config.max_requests
                             in (1, 2, 4, 8, 16, 32, 64, 128, 256)))
                )
                pd_af_mtp = (
                    self.config.deployment.mode == "pd_af_disagg"
                    and pool_phase == "decode"
                    and operator_group == "attention"
                    and len(self._ingress_engine_ids) == 1
                    and len(self._decode_engine_ids) == 1
                    and self.backends[pool_name].config.max_requests == 1
                    and self._enable_pd_af_mtp
                )
                af_mtp = (
                    self.config.deployment.mode == "af_disagg"
                    and pool_phase == "colocated"
                    and operator_group == "attention"
                    and len(self._ingress_engine_ids) == 1
                    and not self._decode_engine_ids
                    and len(self.config.deployment.worker_pools) == 2
                    and self.backends[pool_name].config.max_requests in (1, 2)
                    and self._enable_standalone_af_mtp
                )
                if af_mtp:
                    validate_standalone_af_mtp_scope(
                        self.config,
                        allow_preemption=(
                            self._enable_standalone_af_mtp_preemption),
                    )
                    if (
                        self.backends[pool_name].config.max_requests == 2
                        and not self._enable_standalone_af_mtp_preemption
                    ):
                        raise NotImplementedError(
                            "standalone AF MTP max_requests=2 requires the "
                            "validated preemption scope")
                    _validate_standalone_af_mtp_step(
                        backend_step,
                        allow_preemption=(
                            self._enable_standalone_af_mtp_preemption),
                        expected_fixed_accepted_tokens=(
                            1 if self._kv_lifecycle is not None else None),
                    )
                if pd_af_mtp:
                    _validate_pd_af_mtp_step(
                        backend_step,
                        fixed_accepted_tokens=(
                            self._pd_af_mtp_fixed_accepted_tokens),
                    )
                if not pd_mtp and not pd_af_mtp and not af_mtp:
                    raise NotImplementedError(
                        "non-colocated MTP is validated only for either one "
                        "decode Scheduler in a 1P x 1D PD/PD+AF deployment "
                        "or one "
                        "standalone AF Attention/FFN pair with fixed-count "
                        "acceptance; both require max_requests=1")
            if self._kv_lifecycle is None:
                if any(
                        event.action in (
                            KVCacheAction.LOAD, KVCacheAction.EVICT,
                            KVCacheAction.STORE)
                        for event in backend_step.kv_cache_events):
                    raise ValueError(
                        "KV load/evict events require a tier lifecycle manager")
                transfer_events = ()
            else:
                backend_step, transfer_events = (
                    self._kv_lifecycle.prepare_step(backend_step))
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
                embedding_token_ids=backend_step.embedding_token_ids,
                model_source_inputs=backend_step.model_source_inputs,
                worker_request_state=backend_step.worker_request_state,
                require_timeline_details=bool(transfer_events),
                origin_time_s=start_time_s,
                shared_resource_calendar=resource_calendar,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            execution_timeline = components.execution_timeline
            simulation_step = SimulationStep(
                backend_step=backend_step,
                workload=workload,
                kernel_plan=kernel_plan,
                predictions=predictions,
                execution_timeline=execution_timeline,
                source_activation_memory=components.source_activation_memory,
                source_state_summary=components.source_state_summary,
            )
            if components.source_activation_memory is not None:
                source_memory.consume(backend_step.schedule.pool_name, backend_step.schedule.step_id,
                    components.source_activation_memory)
            if retain_full_steps:
                simulation_steps.append(simulation_step)
            else:
                step_id = backend_step.schedule.step_id
                if step_id in pending_streaming_steps:
                    raise RuntimeError(
                        "streaming simulation received a duplicate pending "
                        "step")
                pending_streaming_steps[step_id] = simulation_step
            if self._kv_lifecycle is not None:
                self._kv_lifecycle.complete_step(
                    backend_step,
                    transfer_events,
                    workload,
                    kernel_plan,
                    predictions,
                    execution_timeline,
                )
            return execution_timeline.duration_s

        def on_step_complete(backend_step, virtual_time_s):
            nonlocal streamed_step_count, streamed_kernel_call_count
            step_id = backend_step.schedule.step_id
            if step_id in parallel_results_by_step_id:
                parallel_result = parallel_results_by_step_id.pop(step_id)
                if parallel_result.source_activation_memory is not None:
                    source_memory.consume(backend_step.schedule.pool_name, step_id, parallel_result.source_activation_memory)
                entry = ScheduleTraceEntry(
                    backend_step,
                    virtual_time_s - parallel_result.duration_s,
                    virtual_time_s,
                )
                completion_time_s = virtual_time_s
                arguments = (
                    backend_step,
                    entry,
                    parallel_result.execution_contract_digest,
                    parallel_result.logical_kernel_call_count,
                    parallel_result.compact,
                )
                if parallel_result.state_audit_snapshot is None:
                    self.streaming_prehashed_step_consumer(*arguments)
                else:
                    self.streaming_prehashed_step_consumer(
                        *arguments, parallel_result.state_audit_snapshot)
                if parallel_result.step_timing is not None:
                    streamed_step_timings.append(parallel_result.step_timing)
                streamed_step_count += 1
                streamed_kernel_call_count += (
                    parallel_result.logical_kernel_call_count)
                if self.progress_callback is not None:
                    self.progress_callback(backend_step, completion_time_s)
                return
            if not retain_full_steps:
                if step_id not in pending_streaming_steps:
                    raise RuntimeError(
                        "streaming simulation completed without a pending "
                        "step")
                pending_streaming_step = pending_streaming_steps.pop(step_id)
                if pending_streaming_step.backend_step.schedule is not backend_step.schedule:
                    raise RuntimeError(
                        "streaming simulation completed a different step")
                duration_s = pending_streaming_step.execution_timeline.duration_s
                entry = ScheduleTraceEntry(
                    pending_streaming_step.backend_step,
                    virtual_time_s - duration_s,
                    virtual_time_s,
                )
                self.streaming_step_consumer(
                    pending_streaming_step, entry)
                if self.config.timing_mode != "device_only":
                    timeline = pending_streaming_step.execution_timeline
                    streamed_step_timings.append(StreamingStepTiming(
                        step_id,
                        timeline.predicted_device_time_s,
                        timeline.launch_cpu_time_s,
                        timeline.launch_critical_path_s,
                        timeline.host_device_overlap_s,
                        timeline.eager_dispatch_count,
                        timeline.graph_replay_count,
                        timeline.launch_kinds,
                    ))
                streamed_step_count += 1
                streamed_kernel_call_count += (
                    pending_streaming_step.kernel_plan.logical_call_count)
            if self.progress_callback is not None:
                self.progress_callback(backend_step, virtual_time_s)

        def drain_decode_tail(pool_name, backend, _origin_time_s):
            nonlocal parallel_worker_count, parallel_step_count
            nonlocal parallel_execution_wall_time_s
            if (retain_full_steps
                    or self._kv_lifecycle is not None
                    or self.config.collect_source_state
                    or self.config.execution_mode != "accelerated"
                    or self.config.runtime_inputs.symbolic_workspace_initial_capacity_bytes is not None
                    or (self.config.runtime_inputs.symbolic_forward_source is not None
                        and self.config.runtime_inputs.graph_execution.enabled)
                    or self.config.timing_mode != "device_only"
                    or self.streaming_step_projector is None
                    or backend_phase(backend.config.role) != "decode"):
                return None
            if not hasattr(backend, "drain_steps"):
                raise TypeError(
                    f"decode tail backend {pool_name!r} lacks drain_steps")

            backend_steps, has_more = backend.drain_steps(
                _PARALLEL_TAIL_BATCH_SIZE)
            if not backend_steps:
                if has_more:
                    raise RuntimeError(
                        "multi-pool decode tail batch made no progress")
                return ()

            available_bytes = (
                os.sysconf("SC_AVPHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE"))
            requested_workers = self.config.execution_workers
            worker_count = min(
                64 if requested_workers == 0 else requested_workers,
                os.cpu_count() or 1,
                max(
                    1,
                    (available_bytes - 64 * 1024 ** 3)
                    // (2 * 1024 ** 3),
                ),
                len(backend_steps),
            )
            parallel_worker_count = max(parallel_worker_count, worker_count)
            parallel_execution = ServingExecutionContext(
                self.config.model,
                self.config.deployment,
                self.hardware_by_id,
                self.config.launch_policy,
                self.config.prediction_method,
                self.config.runtime_inputs,
                execution_mode=self.config.execution_mode,
                fresh_kernel_predictions=self.config.fresh_kernel_predictions,
                host_launch_profiles=self.config.host_launch_profiles,
                host_runtime_prepare_profiles=(
                    self.config.host_runtime_prepare_profiles),
                runtime_prepare_event_builder=(
                    self.config.runtime_prepare_event_builder),
                materialize_timeline_details=self.config.record_execution_timeline,
                model_source_inputs_by_step=self.config.model_source_inputs_by_step,
            )
            global _PARALLEL_EXECUTION_CONTEXT, _PARALLEL_STEP_PROJECTOR
            _PARALLEL_EXECUTION_CONTEXT = parallel_execution
            _PARALLEL_STEP_PROJECTOR = self.streaming_step_projector
            drained = []
            execute_tail_step = (_single_worker_tail_executor(parallel_execution, self.streaming_step_projector)
                if worker_count == 1 else _parallel_execute_step)
            try:
                with _tail_execution_pool(worker_count) as pool:
                    while backend_steps:
                        started = time.perf_counter()
                        results = pool.map(
                            execute_tail_step,
                            backend_steps,
                            chunksize=max(
                                1,
                                len(backend_steps) // (worker_count * 4),
                            ),
                        )
                        parallel_execution_wall_time_s += (
                            time.perf_counter() - started)
                        parallel_step_count += len(backend_steps)
                        for step, result in zip(
                                backend_steps, results, strict=True):
                            step_id = step.schedule.step_id
                            if step_id in parallel_results_by_step_id:
                                raise RuntimeError(
                                    "parallel execution produced duplicate "
                                    f"step {step_id!r}")
                            parallel_results_by_step_id[step_id] = result
                            for name, delta in result.cache_deltas.items():
                                totals = parallel_cache_deltas[name]
                                for index, value in enumerate(delta):
                                    totals[index] += value
                            for name, value in result.phase_wall_time_s.items():
                                parallel_phase_wall_time_s[name] += value
                            drained.append((step, result.duration_s))
                        if not has_more:
                            break
                        backend_steps, has_more = backend.drain_steps(
                            _PARALLEL_TAIL_BATCH_SIZE)
                        if not backend_steps and has_more:
                            raise RuntimeError(
                                "multi-pool decode tail batch made no progress")
            finally:
                _PARALLEL_EXECUTION_CONTEXT = None
                _PARALLEL_STEP_PROJECTOR = None
            return tuple(drained)

        def route_request(request, now):
            queued_tokens = (
                request.input_tokens
                - scheduler_effective_prefix_hit_tokens(
                    request, self._ingress_kv_block_size_tokens)
            )
            return router.route(
                request.request_id,
                self._ingress_stage,
                self._ingress_engine_ids,
                queued_tokens,
                now,
            )

        def dispatch_pending_handoffs(now):
            from src.serving.backend.vllm import VLLMBackend
            handoffs = []
            # FIFO by real P completion. Keep the source's native blocks held
            # while D is full; neither a copy nor a route assignment exists yet.
            for request_id, (pool_name, original) in tuple(pending_handoffs.items()):
                destinations = self._decode_destinations(pool_name)
                destination_request = replace(original, arrival_time_s=now, prefix_hit_tokens=0)
                capacity_snapshot, rejected = (), ()
                if all(isinstance(self.backends[name], VLLMBackend) for name in destinations):
                    from src.serving.schedule.multi_pool_engine import reserve_pd_handoff_destination
                    admission = reserve_pd_handoff_destination(router, self.backends,
                        destinations, destination_request, pool_name, now)
                    if admission is None:
                        if self.config.timing_mode != "device_only":
                            raise NotImplementedError(
                                "PD destination KV backpressure currently requires device_only; "
                                "Host completion wakeup is not bound")
                        break
                    decision, blocks, capacity_snapshot, rejected = admission
                    destination_backend = self.backends[decision.selected_engine_id]
                else:
                    # Non-native backends retain their existing strict path;
                    # only actual native adapters expose atomic KV attempts.
                    decision = router.route(request_id, "prefill_to_decode", destinations,
                        1, now, source_engine_id=pool_name)
                    destination_backend = self.backends[decision.selected_engine_id]
                    if isinstance(destination_backend, VLLMBackend):
                        blocks = destination_backend.reserve_pd_destination(destination_request)
                if isinstance(destination_backend, VLLMBackend):
                    pd_kv_lifecycle.append({"action": "reserve_destination", "pool": decision.selected_engine_id,
                        "ranks": request_placement(decision.selected_engine_id, request_id)[1],
                        "data_parallel_rank": request_placement(decision.selected_engine_id, request_id)[0],
                        "request_id": request_id, "time_s": now, "block_groups": blocks})
                    pd_kv_lifecycle[-1].update({"capacity_routing_snapshot": capacity_snapshot,
                        "capacity_rejected_destinations": rejected})
                transfer_workload = build_pd_kv_transfer_workload(
                    request_id,
                    original.input_tokens,
                    self.config.model,
                    self.config.deployment,
                    pool_name,
                    decision.selected_engine_id,
                    source_data_parallel_rank=request_placement(pool_name, request_id)[0],
                    destination_data_parallel_rank=request_placement(decision.selected_engine_id, request_id)[0],
                )
                physical_transfers = []
                for transfer_node in transfer_workload.nodes:
                    lane_workload = WorkloadGraph((transfer_node,))
                    # PD KV movement is issued outside model Graph capture.
                    # Keep its Host submission contract explicit so Host
                    # timing does not try to infer a Graph mode from a
                    # model ScheduleStep that does not exist here.
                    transfer_components = execution.execute_workload(
                        lane_workload,
                        origin_time_s=decision.dispatch_time_s,
                        shared_resource_calendar=resource_calendar,
                        runtime_graph_mode="NONE",
                        # The handoff consumes the physical interval even
                        # when model-step history is retained as summaries.
                        require_timeline_details=True,
                    )
                    transfer_plan = transfer_components.kernel_plan
                    transfer_predictions = transfer_components.predictions
                    transfer_timeline = (
                        transfer_components.execution_timeline)
                    if len(transfer_plan.calls) != 1:
                        raise RuntimeError(
                            "each PD KV lane must lower to exactly one call")
                    transfer_call = transfer_plan.calls[0]
                    if len(transfer_timeline.intervals) != 1:
                        raise RuntimeError(
                            "each PD KV lane requires one physical interval")
                    transfer_interval, = transfer_timeline.intervals
                    if (transfer_interval.call_id != transfer_call.call_id
                            or not math.isclose(transfer_interval.end_time_s,
                                transfer_timeline.duration_s,
                                rel_tol=1e-12, abs_tol=1e-15)):
                        raise RuntimeError(
                            "PD KV lane interval disagrees with its timeline")
                    transfer_reserved_resources = (
                        call_reserved_resource_ids(transfer_call))
                    simulation_transfers.append(SimulationTransfer(
                        request_id=request_id,
                        source_engine_id=pool_name,
                        destination_engine_id=(
                            decision.selected_engine_id),
                        workload=lane_workload,
                        kernel_plan=transfer_plan,
                        predictions=transfer_predictions,
                        execution_timeline=transfer_timeline,
                    ))
                    physical_transfers.append(HandoffTransfer(
                        transfer_id=transfer_node.node_id,
                        transfer_duration_s=(transfer_interval.end_time_s
                            - transfer_interval.start_time_s),
                        transfer_submission_s=max((
                            event.end_time_s for event in transfer_timeline.host_intervals), default=0.0),
                        reserved_interval=(
                            decision.dispatch_time_s + transfer_interval.start_time_s,
                            decision.dispatch_time_s + transfer_interval.end_time_s),
                        transfer_resource_id=(
                            transfer_node.attributes["resource_id"]),
                        transfer_link=transfer_node.attributes["link"],
                        transfer_reserved_resource_ids=(
                            transfer_reserved_resources),
                        transfer_byte_count=(
                            transfer_node.attributes["bytes"]),
                        bandwidth_segments=tuple(
                            (decision.dispatch_time_s + start, decision.dispatch_time_s + end, count)
                            for start, end, count in transfer_interval.bandwidth_segments),
                    ))
                handoffs.append(HandoffDispatch(
                    decision=decision,
                    request=destination_request,
                    transfers=tuple(physical_transfers),
                ))
                handed_off_request_ids.add(request_id)
                del pending_handoffs[request_id]
            return tuple(handoffs)

        def on_completion(pool_name, backend_step, now):
            pool_phase = backend_phase(self.backends[pool_name].config.role)
            finished = set(backend_step.finished_request_ids)
            if pool_phase == "prefill":
                assignment_stage = "ingress_to_prefill"
            elif pool_phase == "decode":
                assignment_stage = "prefill_to_decode"
            elif pool_phase == "colocated":
                assignment_stage = "ingress_to_colocated"
            else:
                raise ValueError(f"unknown backend phase: {pool_phase!r}")
            for scheduled in backend_step.schedule.requests:
                assignment_key = (assignment_stage, scheduled.request_id)
                if assignment_key in released_route_assignments:
                    if not _is_discarded_async_terminal_step(
                            backend_step,
                            scheduled,
                            self.backends[pool_name].config.scheduler_pipeline_depth,
                    ):
                        raise RuntimeError(
                            "a released route assignment received additional "
                            f"logical work: {assignment_key!r}")
                    continue
                original = requests_by_id[scheduled.request_id]
                if pool_phase == "decode":
                    # A resumed vLLM request replays evicted context as a
                    # prefill-shaped row without returning to the P instance.
                    decode_recompute = (
                        scheduled.phase == "prefill"
                        and (
                            scheduled.resumed_from_preemption
                            or scheduled.recomputed_tokens > 0
                        )
                    )
                    if scheduled.phase != "decode" and not decode_recompute:
                        raise RuntimeError(
                            "decode backend attempted work that is neither "
                            "decode nor preemption recompute")
                    remaining = (
                        0 if scheduled.request_id in finished else 1)
                elif scheduled.phase == "prefill":
                    computed_end = min(
                        original.input_tokens,
                        scheduled.computed_tokens + scheduled.input_tokens,
                    )
                    remaining_prompt = original.input_tokens - computed_end
                    if pool_phase == "prefill":
                        remaining = remaining_prompt
                    elif pool_phase == "colocated":
                        if remaining_prompt:
                            remaining = remaining_prompt
                        elif scheduled.request_id in finished:
                            remaining = 0
                        else:
                            remaining = 1
                    else:
                        raise RuntimeError(
                            "unexpected prefill work on a decode backend")
                elif scheduled.phase == "decode":
                    if pool_phase != "colocated":
                        raise RuntimeError(
                            "non-colocated backend attempted decode work")
                    remaining = (
                        0 if scheduled.request_id in finished else 1)
                else:
                    raise ValueError(
                        f"unknown scheduled phase: {scheduled.phase!r}")
                router.set_request_tokens(
                    scheduled.request_id,
                    assignment_stage,
                    remaining,
                    pool_name,
                )
                if remaining == 0:
                    released_route_assignments.add(assignment_key)
            if pool_phase == "prefill" and finished:
                scheduled_by_id = {scheduled.request_id: scheduled for scheduled in backend_step.schedule.requests}
                for request_id in backend_step.finished_request_ids:
                    if request_id in handed_off_request_ids or request_id in pending_handoffs:
                        raise RuntimeError(f"request {request_id!r} was handed off more than once")
                    original = requests_by_id[request_id]
                    produced = backend_step.generated_tokens[request_id]
                    if produced < original.output_tokens:
                        if produced != 1:
                            raise NotImplementedError(
                                "current vLLM-compatible PD handoff requires the "
                                "prefill engine to produce exactly one first token")
                        scheduled = scheduled_by_id[request_id]
                        if scheduled.raw_model_output_tokens != 1:
                            raise RuntimeError(
                                "PD prefill handoff requires exactly one raw "
                                f"terminal seed for {request_id!r}")
                        if (scheduled.mtp_proposer_mode != "disabled"
                                or scheduled.mtp_acceptance_source_kind != "disabled"
                                or scheduled.mtp_acceptance_source_id != "disabled"
                                or any((
                                    scheduled.mtp_proposed_tokens,
                                    scheduled.mtp_verified_tokens,
                                    scheduled.mtp_accepted_tokens,
                                    scheduled.mtp_rejected_tokens,
                                    scheduled.mtp_proposer_tokens,
                                    scheduled.mtp_next_proposed_tokens,
                                ))):
                            raise RuntimeError(
                                "PD prefill handoff cannot carry, verify, accept, "
                                f"or retain MTP draft state for {request_id!r}")
                        pending_handoffs[request_id] = (pool_name, original)
            return dispatch_pending_handoffs(now)

        closed_loop_clients = _closed_loop_clients(self.config)
        if closed_loop_clients is None:
            engine_requests = scheduler_visible_requests(
                self.config.requests,
                host_request_path_profile(self.config))
        else:
            roles = {
                backend.config.role for backend in self.backends.values()}
            if not roles <= {"colocated", "prefill", "decode"}:
                raise NotImplementedError(
                    "closed-loop arrival supports only colocated and "
                    f"pd_disagg pool roles: {sorted(roles)}")
            engine_requests = RequestStream(
                closed_loop_clients.initial_requests())
        def transfer_completed(source, request_id, now):
            from src.serving.backend.vllm import VLLMBackend
            backend = self.backends[source]
            if isinstance(backend, VLLMBackend):
                blocks = backend.release_pd_kv(request_id)
                pd_kv_lifecycle.append({"action": "release_source", "pool": source,
                    "ranks": request_placement(source, request_id)[1],
                    "data_parallel_rank": request_placement(source, request_id)[0], "request_id": request_id, "time_s": now, "blocks": blocks})

        observed_lifecycle_events = 0
        def observe_queues(now):
            nonlocal observed_lifecycle_events
            observed_started = time.perf_counter()
            snapshot = {"time_s": now,
                "boundary": "after causal event batch before the next Scheduler plan; native planning state, not SM activity",
                "pools": {name: backend.request_queue_snapshot() for name, backend in self.backends.items()},
                "capacity_wait_handoffs": tuple(sorted(pending_handoffs)),
                "pd_kv_lifecycle_delta": tuple(pd_kv_lifecycle[observed_lifecycle_events:])}
            snapshot["native_snapshot_wall_s"] = time.perf_counter() - observed_started
            self.queue_observation_consumer(snapshot)
            observed_lifecycle_events = len(pd_kv_lifecycle)

        with ExitStack() as backend_lifecycle:
            for backend in self.backends.values():
                backend_lifecycle.enter_context(_managed_backend(backend, censored_observation=self.config.observation_end_time_s is not None))
            schedule_trace = run_multi_pool_schedule(
                self.backends,
                engine_requests,
                route_request,
                step_duration,
                {
                    pool_name: backend.config.scheduler_pipeline_depth
                    for pool_name, backend in self.backends.items()
                },
                on_completion=on_completion,
                reserve_transfer=resource_calendar.reserve_many,
                tail_drain=(
                    drain_decode_tail
                    if not retain_full_steps and closed_loop_clients is None
                    and self.config.execution_mode == "accelerated"
                    and self.config.runtime_inputs.symbolic_workspace_initial_capacity_bytes is None
                    and not (self.config.runtime_inputs.symbolic_forward_source is not None
                        and self.config.runtime_inputs.graph_execution.enabled)
                    and self.config.timing_mode == "device_only"
                    and self.streaming_step_projector is not None
                    else None),
                on_step_complete=on_step_complete,
                timing_mode=self.config.timing_mode,
                closed_loop=closed_loop_clients,
                on_transfer_complete=transfer_completed,
                has_pending_handoffs=lambda: bool(pending_handoffs),
                observation_end_time_s=self.config.observation_end_time_s,
                on_queue_observation=observe_queues if self.queue_observation_consumer is not None else None,
            )
        bounded = self.config.observation_end_time_s is not None
        if bounded:
            for entry in schedule_trace.pending_entries:
                step_id = entry.backend_step.schedule.step_id
                self.streaming_inflight_step_consumer(pending_streaming_steps.pop(step_id), entry)
        if pending_handoffs and not bounded:
            raise RuntimeError("PD destination KV capacity never became available for pending handoffs: "
                + repr(tuple(pending_handoffs)))
        metrics_requests = (
            self.config.requests if closed_loop_clients is None or bounded
            else closed_loop_clients.effective_request_stream())
        if not bounded:
            router.assert_drained()
        if not retain_full_steps:
            if pending_streaming_steps:
                raise RuntimeError(
                    "streaming simulation ended with pending steps")
            _assert_expert_routing_replay_coverage(
                self.config.runtime_inputs, ())
            cache_stats = execution.cache_stats()
            performance_stats = execution.performance_stats()
            if parallel_step_count:
                for name, totals in parallel_cache_deltas.items():
                    parent = cache_stats[name]
                    parent["entries"] += totals[0]
                    parent["hits"] += totals[1]
                    parent["misses"] += totals[2]
                _attribute_parallel_phase_wall_time(
                    performance_stats, parallel_phase_wall_time_s,
                    parallel_execution_wall_time_s)
                performance_stats["executed_workloads"] += parallel_step_count
            performance_stats["parallel_worker_count"] = parallel_worker_count
            performance_stats["parallel_step_count"] = parallel_step_count
            performance_stats["parallel_execution_wall_time_s"] = (
                parallel_execution_wall_time_s)
            performance_stats["parallel_worker_aggregate_phase_s"] = dict(
                parallel_phase_wall_time_s)
            if self._kv_lifecycle is not None:
                kv_offload = self._kv_lifecycle.finish()
            else:
                kv_offload = None
            return StreamingSimulationResult(
                schedule_trace=schedule_trace,
                metrics=(None if bounded else compute_serving_metrics(
                    metrics_requests, schedule_trace,
                    _first_token_delivery_s(self.config))),
                num_steps=streamed_step_count,
                num_kernel_calls=streamed_kernel_call_count,
                num_simulation_transfers=len(simulation_transfers),
                pd_kv_lifecycle=tuple(pd_kv_lifecycle),
                recorded_transfers=(tuple(simulation_transfers) if self.config.record_execution_timeline else ()),
                step_timings=tuple(streamed_step_timings),
                execution_cache_stats=cache_stats,
                execution_performance_stats=performance_stats,
                source_activation_memory=source_memory.manifest(total_steps=streamed_step_count + len(schedule_trace.pending_entries)),
                kv_offload=kv_offload,
            )
        schedule_trace = _with_in_step_transfer_trace(
            schedule_trace, tuple(simulation_steps))
        if self._enable_standalone_af_mtp_preemption:
            _validate_standalone_af_mtp_preemption_run(
                self.config,
                tuple(simulation_steps),
            )
        if self._enable_pd_af_mtp:
            _validate_pd_af_mtp_run(
                self.config,
                tuple(simulation_steps),
                fixed_accepted_tokens=(
                    self._pd_af_mtp_fixed_accepted_tokens),
            )
        _assert_expert_routing_replay_coverage(
            self.config.runtime_inputs,
            tuple(simulation_steps),
        )
        if self._kv_lifecycle is None:
            kv_offload = None
        else:
            kv_offload = self._kv_lifecycle.finish()
        return SimulationResult(
            schedule_trace=schedule_trace,
            metrics=compute_serving_metrics(
                metrics_requests, schedule_trace,
                _first_token_delivery_s(self.config)),
            steps=tuple(simulation_steps),
            source_activation_memory=source_memory.manifest(total_steps=len(simulation_steps)),
            transfers=tuple(simulation_transfers),
            pd_kv_lifecycle=tuple(pd_kv_lifecycle),
            kv_offload=kv_offload,
            execution_cache_stats=execution.cache_stats(),
            execution_performance_stats=execution.performance_stats(),
            metrics_requests=(
                metrics_requests if closed_loop_clients is not None
                else None),
        )
