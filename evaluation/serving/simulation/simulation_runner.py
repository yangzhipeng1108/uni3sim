"""Run the production no-GPU Serving simulation and write canonical metrics.

``vllm`` selects the real vLLM v1 Scheduler adapter.  ``conformance`` is a
small deterministic backend used only by repository architecture/full-eval
gates when a framework package is unavailable; it is never selected by an
experiment manifest.
"""

from __future__ import annotations

import argparse

from contextlib import ExitStack

from copy import copy

from dataclasses import asdict

from functools import partial

import hashlib

import json

import math

import os

from collections.abc import Mapping, Sequence

from pathlib import Path

from src import _platform_compat as resource

import sys

import threading


import warnings

from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.serving.experiments.experiment_plan import (
    git_code_identity,
    load_workload_artifact,
    vllm_bench_arrival_process,
    vllm_bench_arrival_times,
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

from evaluation.serving.metrics.scheduler_progress import (
    scheduler_progress_manifest,
)

from evaluation.serving.metrics.system_resource_analysis import (
    critical_path_manifest,
    system_resource_manifest,
)

from evaluation.serving.simulation.execution_artifact import (
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    execution_artifact_manifest,
    performance_timing_status,
    prefix_alignment_manifest,
    StreamingExecutionArtifactAccumulator,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)

from src.kernel.hardware_spec import HWSpec

from src.kernel.predictors.protobridge.adapter import (
    development_profile_config_sha256,
    resolve_registered_profile,
)

from src.serving.backend.vllm import (
    VLLMBackend,
    VLLMSchedulerConfig,
    mtp_scenario_manifest,
)

from src.serving.backend.vllm_scheduler_process import (
    vllm_scheduler_runtime_sha256,
)

from src.serving.backend.mtp_acceptance_replay import (
    PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
    load_mtp_acceptance_replay,
    mtp_acceptance_replay_manifest,
    mtp_acceptance_step_from_backend_step,
    mtp_acceptance_steps_manifest,
    closed_loop_admission_contract_sha256,
    pd_decode_admission_contract_sha256,
)

from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
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
    worker_role_operator_group,
    worker_role_phase,
)

from src.serving.deployment.deployment_plan import deployment_plan_manifest

from src.serving.deployment.pd_disagg import build_pd_disagg_plan

from src.serving.kv_cache.kv_capacity import (
    kv_capacity_manifest,
    load_runtime_memory_policy,
    resolve_analytical_kv_capacity_plan,
    resolve_debug_kv_capacity_plan,
)

from src.serving.kv_cache.kv_cache_events import KVCacheAction

from src.serving.host_timing import (
    load_host_launch_profile,
    load_host_runtime_prepare_profile,
    load_host_request_path_profile,
    load_scheduler_host_profile,
    load_engine_core_host_round_profile,
    schedule_phase,
)

from src.serving.metrics.serving_metrics import (
    engine_clock_first_schedule_start_s,
)

from src.serving.runtime_prepare import runtime_prepare_events

from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager

from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)

from src.serving.parallelism.parallel_config import ParallelConfig

from src.serving.parallelism.pp import pipeline_layer_ranges_for_model

from src.serving.request_gen.production_trace import (
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
    build_production_trace_arrival_rate_sweep,
    build_production_trace_replay,
    load_production_trace,
    production_trace_manifest,
    production_trace_replay_manifest,
)

from src.serving.request_gen import (
    CANONICAL_EQUAL_ARRIVAL_ORDER,
    CANONICAL_REQUEST_COLUMNS,
    CANONICAL_REQUEST_ID_POLICY,
)

from src.serving.request_gen.canonical_requests import (
    parse_canonical_requests_bytes,
)

from src.serving.request_gen.request_stream import (
    RequestSpec,
    RequestStream,
    request_stream_sha256,
)

from src.serving.schedule.backend_protocol import BackendConfig

from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)

from src.serving.simulation import (
    MultiSimulation,
    Simulation,
    SimulationConfig,
    StreamingSimulationResult,
    validate_pd_af_mtp_scope,
    validate_standalone_af_ep2_scope,
    validate_standalone_af_mtp_scope,
    validate_workload_emission_parallelism,
)

from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    DEEPSEEK_V4_PRO_CONFIG_SHA256,
    is_validated_qwen3_tp2_host_kv_model_scope,
    load_model_config,
)

from src.serving.workload.expert_frequency import (
    load_expert_frequency_profile,
)

from src.serving.workload.graph_padding import GraphExecutionConfig

from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    load_expert_routing_replay,
    RuntimeInputs,
)

from evaluation.serving.simulation.simulation_artifacts import (
    aggregate_mtp_acceptance_outcomes as _aggregate_mtp_acceptance_outcomes,
    aggregate_mtp_pd_handoff as _aggregate_mtp_pd_handoff,
    canonical_json_sha256 as _canonical_json_sha256,
    canonical_requests_aggregate_result as _canonical_requests_aggregate_result,
    directory_provenance as _directory_provenance,
    build_execution_provenance as _execution_provenance,
    file_provenance as _file_provenance,
    host_timing_manifest as _host_timing_manifest,
    launch_data_provenance,
    load_request_source as _load_request_source,
    mtp_acceptance_outcomes_manifest as _mtp_acceptance_outcomes_manifest,
    mtp_pd_handoff_contract as _mtp_pd_handoff_contract,
    persistence_projection as _persistence_projection,
    production_trace_aggregate_result as _production_trace_aggregate_result,
    repository_provenance as _repository_provenance,
    request_safe_aggregate_result as _request_safe_aggregate_result,
    vllm_step_index as _vllm_step_index,
    workload_assumption_binding_manifest as _workload_assumption_binding_manifest,
)

from evaluation.serving.simulation.simulation_configuration import (
    ROOT,
    _cli_hardware_id,
    _cli_hardware_identity_scope,
    DSV3_PD_MTP_MAX_REQUESTS as _DSV3_PD_MTP_MAX_REQUESTS,
    ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE as _ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE,
    GLM_PD_MTP_MAX_REQUESTS as _GLM_PD_MTP_MAX_REQUESTS,
    REQUEST_PATH_PROFILE_BY_HARDWARE as _REQUEST_PATH_PROFILE_BY_HARDWARE,
    RUNTIME_PREPARE_PROFILE_BY_HARDWARE as _RUNTIME_PREPARE_PROFILE_BY_HARDWARE,
    TARGET_SCOPES as _TARGET_SCOPES,
    VERSIONED_HOST_PROFILE_HARDWARE as _VERSIONED_HOST_PROFILE_HARDWARE,
    arrival_schedule as _arrival_schedule,
    bind_tensor_collective_profiles as _bind_tensor_collective_profiles,
    host_request_path_profile_paths as _host_request_path_profile_paths,
    host_runtime_prepare_profile_path as _host_runtime_prepare_profile_path,
    is_dsv3_h200_aggregate_target_scope as _is_dsv3_h200_aggregate_target_scope,
    is_dsv3_h200_pd_mtp_target_scope as _is_dsv3_h200_pd_mtp_target_scope,
    is_dsv3_h200_pd_tp8_ep1_target_scope as _is_dsv3_h200_pd_tp8_ep1_target_scope,
    is_dsv4_flash_h200_colocated_target_scope as _is_dsv4_flash_h200_colocated_target_scope,
    is_dsv4_pro_h200_colocated_target_scope as _is_dsv4_pro_h200_colocated_target_scope,
    is_glm_h200_aggregate_target_scope as _is_glm_h200_aggregate_target_scope,
    is_glm_h200_pd_mtp_target_scope as _is_glm_h200_pd_mtp_target_scope,
    is_glm_h200_pd_tp8_ep1_target_scope as _is_glm_h200_pd_tp8_ep1_target_scope,
    is_kimi_h200_colocated_tp8_ep8_target_scope as _is_kimi_h200_colocated_tp8_ep8_target_scope,
    load_pd_scope_configs as _load_pd_scope_configs,
    load_target_scope_config as _load_target_scope_config,
    matches_target_scope as _matches_target_scope,
    p2p_resource_id as _p2p_resource_id,
    parse_success_value_json as _parse_success_value_json,
    pd_mtp_allowed_max_requests as _pd_mtp_allowed_max_requests,
    pipeline_p2p_configs as _pipeline_p2p_configs,
    production_trace_reader_config as _production_trace_reader_config,
    protobridge_p2p_profile as _protobridge_p2p_profile,
    require_expected_sha256 as _require_expected_sha256,
    tensor_collective_configs as _tensor_collective_configs,
    uses_development_protobridge as _uses_development_protobridge,
    validate_expected_sha256_args as _validate_expected_sha256_args,
    validate_request_source_args as _validate_request_source_args,
)

from evaluation.serving.simulation.simulation_execution import (
    ConformanceBackend as _ConformanceBackend,
    resolve_executing_model_id as _executing_model_id,
    run_arrival_rate_sweep as _run_arrival_rate_sweep,
    run_single as _run_single,
    serving_progress_reporter as _serving_progress_reporter,
    use_streaming_step_retention as _use_streaming_step_retention,
    run,
)
from src.serving.workload.runtime_inputs import MACA_COPY_RUNTIME_SOURCE
from evaluation.serving.simulation.simulation_configuration import SGLANG_VIRTUAL_TARGET_SOURCES as _SGLANG_VIRTUAL_TARGET_SOURCES
from evaluation.serving.simulation.simulation_configuration import ascend018_colocated_ep2_cli_scope as _ascend018_colocated_ep2_cli_scope
from evaluation.serving.simulation.simulation_configuration import ascend018_dense_cli_scope as _ascend018_dense_cli_scope
from evaluation.serving.simulation.simulation_configuration import metax024_colocated_ep2_cli_scope as _metax024_colocated_ep2_cli_scope
from evaluation.serving.simulation.simulation_configuration import normalize_default_model_source as _normalize_default_model_source
from evaluation.serving.simulation.simulation_configuration import normalize_glm_mtp_cli_source as _normalize_glm_mtp_cli_source
from evaluation.serving.simulation.simulation_configuration import normalize_vllm_step_cli_source as _normalize_vllm_step_cli_source
from evaluation.serving.simulation.simulation_configuration import normalize_mtp_draft_token_ids as _normalize_mtp_draft_token_ids
from evaluation.serving.simulation.simulation_configuration import shape_only_dense_conformance_cli_scope as _shape_only_dense_conformance_cli_scope
from evaluation.serving.simulation.simulation_configuration import shared_dense_graph_cli_scope as _shared_dense_graph_cli_scope
from evaluation.serving.simulation.simulation_configuration import vllm011_colocated_ep2_cli_scope as _vllm011_colocated_ep2_cli_scope
from evaluation.serving.simulation.simulation_configuration import vllm011_dense_disagg_cli_scope as _vllm011_dense_disagg_cli_scope
from evaluation.serving.simulation.simulation_configuration import vllm011_dense_pipeline_cli_scope as _vllm011_dense_pipeline_cli_scope
from evaluation.serving.simulation.simulation_configuration import vllm011_moe_cli_scope as _vllm011_moe_cli_scope
from evaluation.serving.simulation.simulation_configuration import vllm011_multi_colocated_cli_scope as _vllm011_multi_colocated_cli_scope
from src.serving.workload.model_config import is_validated_qwen3_moe_ep2_model_scope
from src.serving.workload.model_config import is_validated_qwen3_pipeline_model_scope
from evaluation.serving.simulation.simulation_configuration import gqa_pd_cli_scope, mla_pcp_pd_cli_scope, glm_bf16_cli_source_scope


def _launch_data_provenance(hardware_ids: tuple[str, ...]) -> dict:
    return launch_data_provenance(hardware_ids, repository_root=ROOT)


@_cli_hardware_identity_scope()
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload")
    source.add_argument("--production-trace")
    source.add_argument("--canonical-requests")
    parser.add_argument("--trace-source-sha256")
    parser.add_argument("--canonical-requests-sha256")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config-sha256")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--hardware-config-sha256")
    parser.add_argument("--decode-hardware-config")
    parser.add_argument(
        "--additional-colocated-hardware-config",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--deployment",
        choices=("colocated", "af_disagg", "pd_disagg", "pd_af_disagg"),
        required=True,
    )
    parser.add_argument(
        "--backend", choices=("vllm", "conformance"),
        required=True)
    parser.add_argument("--vllm-scheduler-version", choices=("0.11.0", "0.30.0"), default="0.11.0")
    parser.add_argument("--vllm-executor-backend", choices=("uni", "mp"))
    parser.add_argument("--vllm-sdk-python", type=Path)
    parser.add_argument("--vllm-source-root", type=Path)
    parser.add_argument("--vllm-cache-layout", choices=("LBHNC", "BLHNC", "BLNHC"))
    parser.add_argument("--activation-source-profile", choices=(
        "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache",
        "vllm_ascend_0_18_no_metadata_head_only",
        "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise",
        "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first",
        "cuda_eager_inplace_fa_no_metadata_native_triton_fp32_noise_cold_cache"),
        help="derive activation capacity from the declared metadata-only startup lifecycle; external reserves remain unverified")
    parser.add_argument("--activation-source-lifetime-policy", choices=("vllm_0_18_npugraph_ex_piecewise",))
    parser.add_argument("--activation-source-compilation-options-json",
        help="explicit upstream startup compilation options; does not identify the request graph mode")
    parser.add_argument("--v4-weight-storage-profile", choices=("sm90_deepgemm_marlin",),
                        help="explicit source parameter-storage scenario; does not establish actual provider selection")
    parser.add_argument("--v4-eager-metadata", action="store_true",
                        help="declared eager/all-valid source scenario, not observed GPU metadata")
    parser.add_argument("--v4-routing-source", choices=("vllm_0_30_triton_top6_pure_text", "vllm_0_30_modular_cuda_top6_pure_text"))
    parser.add_argument("--target-sampling-source", choices=("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16", "vllm_0_27_1_plain_greedy_bf16",
                        "vllm_0_18_plain_greedy_bf16",
                        "vllm_0_24_plain_greedy_bf16",
                        "vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_greedy_fp16", "vllm_0_11_greedy_rejection_bf16", "sglang_0_5_19_plain_greedy_fp32",
                        "vllm_0_30_rejection_raw_logits_bf16"))
    parser.add_argument("--maca-copy-runtime-source", choices=(MACA_COPY_RUNTIME_SOURCE,),
                        help="explicit C500 native allocator/unset target flags source scenario; never inferred from simulator environment")
    parser.add_argument("--symbolic-forward-source", choices=("cuda_eager_inplace", "vllm_ascend_eager", "vllm_0_30_v4_sync_fp8_ds_mla",
                        "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce",
                        "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_30_k3_mxfp4_sync", "vllm_0_27_1_deepseek_v3_sync_fp8", "vllm_0_11_glm4_sync",
                        *_SGLANG_VIRTUAL_TARGET_SOURCES),
                        help="metadata-only shared model forward; does not execute actual tensors or a GPU model")
    parser.add_argument("--source-workspace-initial-capacity-bytes", nargs="+", type=int,
                        help="declared workspace entry bytes per TP rank; BF16 GLM PD also accepts all deployment physical ranks in pool order, required with different endpoint geometry")
    parser.add_argument("--norm-source-batch-invariant", choices=("disabled",),
                        help="explicit vLLM0.30 CUDA Norm default-source scenario; capture flag and native provider remain unobserved")
    parser.add_argument("--bf16-gemm-model", choices=("tiled_reference",),
                        help="explicit ordinary BF16 finite-tile analytical reference; no cuBLAS provider or launch attestation")
    parser.add_argument("--bf16-expert-model", choices=("triton_reference",),
                        help="explicit BF16 prepared-expert Triton source reference with separate kernel submits")
    parser.add_argument("--dense-source-input-policy", choices=("declared_zero_tp2",),
                        help="explicit TP2 input approximation: retain known IDs; declare unknown IDs/padding zero, not observed content")
    parser.add_argument("--v4-source-use-deep-gemm", choices=("enabled", "disabled"),
                        help="explicit mHC source branch declaration; not installed provider evidence")
    parser.add_argument("--v4-indexer-selection-source", choices=("declared_causal_topk",),
                        help="declare exactly min(K, causal completed states) distinct valid indexer outputs per row; not observed IDs")
    parser.add_argument("--glm-indexer-selection-source", choices=("declared_causal_topk",),
                        help="explicit GLM cardinality domain, not observed selected indices")
    parser.add_argument("--glm-prefill-attention-source", choices=("force_sparse_mqa",),
                        help="force vLLM sparse MQA; BF16 CP1 may omit this flag for shape-checked fresh-dense prefill and ordinary sparse decode")
    parser.add_argument("--glm-mtp-native-source-inputs", action="store_true",
        help="explicit GLM53 FP8 PD D-only MTP5 length-only source frames from native Scheduler; A60 remains a caller assumption")
    parser.add_argument("--glm-mtp-step-source-input", action="append", nargs=2, metavar=("STEP_ID", "JSON"),
        help="repeat explicit V32PaddedMTPInputs JSON per Scheduler step; not numerically generated or observed drafts")
    parser.add_argument("--glm-mtp-initial-target-token-ids", nargs="+", type=int,
        help="explicit initial RuntimeInputs IDs; each executed step uses its BackendStep IDs, including the CPU Scheduler's declared length-only protocol")
    parser.add_argument("--vllm-step-source-input", action="append", nargs=2, metavar=("STEP_ID", "JSON"),
        help="explicit VLLMDenseGraphInputs with scheduled IDs and optional retained padding per target step; no IDs are inferred")
    parser.add_argument(
        "--prediction-method",
        choices=("midlayer", "roofline", "roofline_span_floor", "pipeweave"),
        required=True,
    )
    parser.add_argument(
        "--collective-backend",
        choices=("analytical_roofline_v1", "protobridge"),
        default=None,
    )
    parser.add_argument("--tensor-all-reduce-profile-sha256")
    parser.add_argument("--tensor-all-gather-profile-sha256")
    parser.add_argument(
        "--launch-policy", choices=("injected_launch", "analytical_launch"), required=True)
    parser.add_argument(
        "--attention-kv-layout",
        choices=("contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"),
        required=True,
    )
    parser.add_argument(
        "--attention-kv-distribution",
        choices=("uniform", "skew"),
        required=True,
    )
    parser.add_argument(
        "--graph-mode",
        choices=(
            "NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
            "FULL_AND_PIECEWISE",
        ),
        default="NONE",
    )
    parser.add_argument(
        "--graph-uniform-decode-query-len", type=int, default=1)
    parser.add_argument("--graph-capture-sizes", type=int, nargs="+")
    parser.add_argument(
        "--graph-piecewise-eager-workload-kinds", nargs="*", default=())
    parser.add_argument("--request-rate", type=float)
    parser.add_argument(
        "--arrival-process",
        choices=("vllm_bench_gamma", "fixed_interval", "simultaneous"),
    )
    parser.add_argument("--arrival-seed", type=int)
    parser.add_argument("--arrival-burstiness", type=float)
    parser.add_argument(
        "--trace-format", choices=("csv", "jsonl", "parquet"))
    parser.add_argument(
        "--trace-time-unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--trace-arrival-field")
    parser.add_argument("--trace-input-tokens-field")
    parser.add_argument("--trace-output-tokens-field")
    parser.add_argument("--trace-prefix-hit-tokens-field")
    parser.add_argument("--trace-prompt-token-ids-field")
    parser.add_argument("--trace-output-token-ids-field")
    parser.add_argument("--prefix-cache-mode", choices=("trace", "token_ids", "trace_no_hit"), default="trace")
    parser.add_argument("--trace-request-id-field")
    parser.add_argument(
        "--trace-success-filter-mode",
        choices=("upstream_attested", "field_equals"),
    )
    parser.add_argument("--trace-success-field")
    parser.add_argument(
        "--trace-success-value-json",
        dest="trace_success_value",
        type=_parse_success_value_json,
    )
    parser.add_argument("--trace-success-provenance")
    parser.add_argument(
        "--trace-replay-mode",
        choices=("original", "arrival_rate_scale", "arrival_rate_sweep"),
    )
    parser.add_argument("--trace-arrival-rate-scale", type=float)
    parser.add_argument(
        "--trace-arrival-rate-scales",
        nargs="+",
        type=float,
    )
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1,
        help="scheduler sequence capacity",
    )
    parser.add_argument(
        "--closed-loop-client-count",
        type=int,
        default=None,
        help=("zero-think-time closed-loop clients; the canonical request "
              "stream is chained interleaved across this many clients and "
              "each client's next request is injected at its previous "
              "request's completion (default: static canonical arrivals)"),
    )
    parser.add_argument("--scheduler-pipeline-depth", type=int, required=True)
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vllm-async-terminal-guard-source",
        choices=("vllm_0_18_0_running_max_tokens_guard",),
        default=None,
        help="explicit diagnostic bridge for the vLLM 0.18 async terminal guard",
    )
    kv_capacity = parser.add_mutually_exclusive_group()
    kv_capacity.add_argument(
        "--debug-num-kv-blocks",
        dest="num_kv_blocks",
        type=int,
        help=(
            "fixture-only explicit Scheduler capacity; forbidden for "
            "Production Trace and canonical Serving dataset requests"),
    )
    kv_capacity.add_argument("--runtime-memory-policy")
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--kv-lifecycle-replay")
    parser.add_argument("--kv-lifecycle-source-sha256")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=0)
    parser.add_argument(
        "--mtp-acceptance-policy",
        choices=("disabled", "fixed_count", "maximum_entropy", "replay"),
        default="disabled",
    )
    parser.add_argument("--mtp-fixed-accepted-tokens", type=int, default=0)
    parser.add_argument(
        "--mtp-acceptance-source-id",
        default="disabled",
    )
    parser.add_argument("--mtp-acceptance-replay")
    parser.add_argument("--mtp-draft-token-ids-by-step",
        help="JSON file mapping native step IDs to request IDs and real MTP draft token arrays")
    parser.add_argument("--mtp-mean-acceptance-rate")
    parser.add_argument("--mtp-acceptance-seed", type=int)
    parser.add_argument(
        "--expert-routing-policy",
        choices=(
            "seeded_assumption",
            "profile_observation",
            "replay_observation",
        ),
    )
    parser.add_argument("--expert-routing-source-id")
    parser.add_argument("--expert-routing-seed", type=int)
    parser.add_argument("--expert-routing-hot-expert-count", type=int)
    parser.add_argument("--expert-routing-hot-expert-weight", type=int)
    parser.add_argument("--expert-routing-replay")
    parser.add_argument("--expert-routing-profile-manifest")
    parser.add_argument(
        "--expert-routing-profile-resolution",
        choices=("layer_mean", "per_moe_layer"),
        default="layer_mean",
        help=(
            "profile routing realization: one histogram per step from the "
            "model-level layer-mean row, or one per MoE layer from its own "
            "row"),
    )
    parser.add_argument("--expert-routing-source-sha256")
    parser.add_argument("--expert-routing-canonical-payload-sha256")
    parser.add_argument(
        "--execution-mode",
        choices=("accelerated", "reference", "fast"),
        default="accelerated",
    )
    parser.add_argument(
        "--execution-workers",
        type=int,
        default=0,
        help=(
            "streaming tail worker processes; zero selects a bounded "
            "CPU/memory-aware default"),
    )
    parser.add_argument(
        "--serving-timing-mode",
        choices=(
            "device_only",
            "device_plus_launch",
            "device_plus_launch_plus_scheduler",
        ),
        default="device_only",
    )
    parser.add_argument(
        "--host-launch-profile",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--scheduler-host-profile",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--host-request-path-profile",
        action="append",
        default=[],
        help=(
            "replace the automatically bound per-request Host request path "
            "profile for one hardware (debug-only for published hardware; "
            "required for hardware with versioned Host profiles); repeat per "
            "hardware"),
    )
    parser.add_argument(
        "--host-runtime-prepare-profile",
        help=(
            "runtime-prepare profile of hardware with versioned Host profiles "
            "(required there in Host timing modes; rejected elsewhere)"),
    )
    parser.add_argument("--interactive-view", choices=("breakdown", "swimlanes"), default="swimlanes", help="offline report layout")
    parser.add_argument("--interactive-detail-mode", choices=("summary", "sampled", "all"), default="summary", help="summary preserves fast paths (default); sampled/all explicitly capture operator detail")
    parser.add_argument("--interactive-max-events", type=int, default=100000, help="maximum retained detailed Host/Device events; request-aware sampling across the run; all-step summaries remain complete")
    parser.add_argument("--no-interactive-report", dest="interactive_report", action="store_false",
        default=False, help=argparse.SUPPRESS)
    parser.add_argument("--interactive-report", dest="interactive_report", action="store_true",
        default=False, help="write the offline HTML report plus its visualization projection")
    parser.add_argument("--interactive-output", help="override the default offline HTML path; summaries preserve compact execution")
    parser.add_argument(
        "--step-retention",
        choices=("auto", "full", "streaming_aggregate"),
        default="auto",
        help=(
            "retain full per-step IR, consume it as a bounded-memory "
            "aggregate, or select streaming automatically for compatible "
            "external Serving requests"),
    )
    parser.add_argument("--prefill-tp-size", type=int, required=True)
    parser.add_argument("--prefill-pp-size", type=int, default=1)
    parser.add_argument("--prefill-pp-layer-partition", nargs="+", type=int)
    parser.add_argument("--pipeline-ranks-per-node", type=int)
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="enable the validated vLLM TP=EP=2 colocated slice",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ffn-tp-size", type=int)
    parser.add_argument("--decode-tp-size", type=int)
    parser.add_argument("--decode-pp-size", type=int, default=1)
    parser.add_argument("--decode-pp-layer-partition", nargs="+", type=int)
    parser.add_argument("--prefill-dp-size", type=int, default=1)
    parser.add_argument("--decode-dp-size", type=int, default=1)
    parser.add_argument("--prefill-cp-size", type=int, default=1)
    parser.add_argument("--decode-cp-size", type=int, default=1)
    parser.add_argument("--prefill-expert-parallel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decode-expert-parallel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-prefill-instances", type=int)
    parser.add_argument("--num-decode-instances", type=int)
    parser.add_argument("--transfer-algorithm", choices=("send_recv",))
    parser.add_argument("--kv-transfer-backend",
        choices=("analytical_roofline_v1", "protobridge"), default="protobridge")
    parser.add_argument(
        "--transfer-link", choices=("nvlink", "metaxlink", "hccs",
                                    "infiniband"))
    parser.add_argument("--router-resource-id", default="router_cpu")
    parser.add_argument(
        "--router-dispatch-service-time-s",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--activation-transfer-algorithm", choices=("send_recv",))
    parser.add_argument(
        "--activation-transfer-link",
        choices=("nvlink", "metaxlink", "hccs"),
    )
    parser.add_argument("--slo-config")
    parser.add_argument("--slo-config-sha256")
    parser.add_argument("--interactivity-tgs-config")
    parser.add_argument("--interactivity-tgs-config-sha256")
    parser.add_argument("--state-audit-output",
        help="optional per-rank logical-work and source-memory lifecycle sidecar")
    parser.add_argument("--output")
    parser.add_argument("--settings-summary", type=Path,
                        help="write a human-readable summary of the completed simulation")
    parser.add_argument("--quiet", action="store_true", help="do not print the saved result JSON")
    args = parser.parse_args(argv)
    if args.interactive_output is not None and args.output is not None and Path(args.interactive_output).resolve() == Path(args.output).resolve():
        parser.error("report and result output paths must differ")
    if args.interactive_detail_mode == "all" and (not args.interactive_report or args.interactive_view != "swimlanes"):
        parser.error("--interactive-detail-mode all requires enabled reports and --interactive-view swimlanes")
    for side in ("prefill", "decode"):
        field = side + "_pp_layer_partition"
        partition = getattr(args, field)
        if partition is not None:
            partition = tuple(partition)
            model = load_model_config(args.model)
            if (len(partition) != getattr(args, side + "_pp_size") or sum(partition) != model.num_hidden_layers
                    or any(count <= 0 for count in partition)):
                parser.error("--" + side + "-pp-layer-partition requires positive counts matching PP and all decoder layers")
            if model.model_type != "glm_moe_dsa" or model.weight_dtype != "bf16":
                parser.error("explicit per-pool pipeline source partitions require BF16 GLM")
            start = 0
            for count in partition:
                if (args.glm_prefill_attention_source == "force_sparse_mqa"
                        and model.indexer_types[start] != "full"):
                    parser.error("GLM source stage needs a local full indexer/top-k input; this does not mark the upstream partition illegal")
                start += count
            setattr(args, field, partition)
    if args.deployment != "pd_disagg" and args.decode_pp_layer_partition is not None:
        parser.error("--decode-pp-layer-partition requires pd_disagg")
    if args.deployment != "pd_disagg" and (
            args.prefill_expert_parallel is not None or args.decode_expert_parallel is not None):
        parser.error("independent P/D expert parallel options require pd_disagg")
    args.prefill_expert_parallel = (args.enable_expert_parallel
        if args.prefill_expert_parallel is None else args.prefill_expert_parallel)
    args.decode_expert_parallel = (args.enable_expert_parallel
        if args.decode_expert_parallel is None else args.decode_expert_parallel)
    if args.prefill_dp_size < 1 or args.decode_dp_size < 1:
        parser.error("P/D data parallel sizes must be positive")
    if args.prefill_cp_size < 1 or args.decode_cp_size < 1:
        parser.error("P/D context parallel sizes must be positive")
    if args.deployment != "pd_disagg" and (args.prefill_cp_size != 1 or args.decode_cp_size != 1):
        parser.error("P/D context parallel options require pd_disagg")
    if ((args.deployment not in ("pd_disagg", "colocated") and args.prefill_dp_size != 1)
            or (args.deployment != "pd_disagg" and args.decode_dp_size != 1)):
        parser.error("data parallel options require pd_disagg or the GLM colocated source")
    if args.deployment != "pd_disagg" and args.decode_pp_size != 1:
        parser.error("--decode-pp-size is only valid for pd_disagg")
    _normalize_default_model_source(args)
    if (args.symbolic_forward_source == "vllm_0_30_v32_bf16_sync_explicit_allreduce"
            and max(args.prefill_pp_size, args.decode_pp_size) > 1 and not glm_bf16_cli_source_scope(args)):
        parser.error("GLM pipeline source requires legal BF16 shards, CP1 and a full indexer for forced sparse attention; automatic dense/sparse selection validates top-k dependencies per step")
    if args.deployment == "colocated" and args.prefill_dp_size != 1 and not glm_bf16_cli_source_scope(args):
        parser.error("colocated data parallel requires the explicit BF16 GLM source")
    if max(args.prefill_cp_size, args.decode_cp_size) > 1 and not mla_pcp_pd_cli_scope(args):
        parser.error("PCP PD requires explicit analytical GLM BF16 sparse-MQA eager0.30, PP1 and legal TP*CP*DP expert shards")
    _normalize_glm_mtp_cli_source(args, parser)
    _normalize_vllm_step_cli_source(args, parser)
    _normalize_mtp_draft_token_ids(args, parser)
    args.activation_source_compilation_options = None
    if args.activation_source_compilation_options_json is not None:
        def unique_options(pairs):
            values = {}
            for name, value in pairs:
                if name in values:
                    raise ValueError("duplicate source compilation option: " + name)
                values[name] = value
            return values
        try:
            options = json.loads(args.activation_source_compilation_options_json, object_pairs_hook=unique_options)
        except ValueError as error:
            parser.error(str(error))
        if not isinstance(options, dict):
            parser.error("source compilation options must be a JSON object")
        args.activation_source_compilation_options = tuple((name, tuple(value)
            if name == "splitting_ops" and isinstance(value, list) else value) for name, value in options.items())
    if (args.activation_source_lifetime_policy is not None) != (args.activation_source_compilation_options is not None):
        parser.error("activation source lifetime policy and compilation options must be supplied together")
    if (args.activation_source_lifetime_policy is not None
            and args.activation_source_profile != "vllm_ascend_0_18_no_metadata_head_only"):
        parser.error("activation source lifetime policy requires explicit Ascend0.18 startup")
    if args.activation_source_profile is not None:
        source_model = load_model_config(args.model)
        metax024 = (args.activation_source_profile == "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache"
            and args.vllm_scheduler_version == "0.11.0" and _cli_hardware_id(args.hardware_config, HWSpec) == "metax_c500"
            and args.symbolic_forward_source == "cuda_eager_inplace"
            and args.target_sampling_source == "vllm_0_24_plain_greedy_bf16" and args.prefill_tp_size == 1)
        if (args.activation_source_profile == "vllm_metax_0_24_no_metadata_native_triton_fp32_noise_cold_cache"
                and not metax024):
            parser.error("MetaX0.24 startup projection requires C500 BF16 dense TP1 with its explicit0.24 model source")
        ascend018 = (args.activation_source_profile == "vllm_ascend_0_18_no_metadata_head_only"
            and args.vllm_scheduler_version == "0.11.0" and _cli_hardware_id(args.hardware_config, HWSpec) == "ascend_910b"
            and args.symbolic_forward_source == "vllm_ascend_eager"
            and args.target_sampling_source == "vllm_0_18_plain_greedy_bf16"
            and source_model.model_type == "qwen3" and args.prefill_tp_size == 1
            and args.graph_mode == ("PIECEWISE" if args.activation_source_lifetime_policy is not None else "NONE"))
        if args.activation_source_profile == "vllm_ascend_0_18_no_metadata_head_only" and not ascend018:
            parser.error("Ascend0.18 startup projection requires Qwen3 BF16 TP1 and matching eager or explicit PIECEWISE request configuration; FULL/NONE cannot borrow the PIECEWISE startup policy")
        flashinfer011 = (args.vllm_scheduler_version == "0.11.0"
            and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and args.activation_source_profile == "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first")
        source011 = (args.vllm_scheduler_version == "0.11.0"
            and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and (args.activation_source_profile == "cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise" or flashinfer011))
        if (args.activation_source_profile == "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first"
                and not flashinfer011):
            parser.error("FlashInfer0.4.1 startup policy requires explicit A8000.11 source")
        if ((args.vllm_scheduler_version != "0.30.0" and not source011 and not ascend018 and not metax024) or args.backend != "vllm"
                or args.runtime_memory_policy is None or args.num_kv_blocks is not None
                or args.deployment != "colocated" or args.additional_colocated_hardware_config
                or args.prefill_tp_size not in (1, 2) or args.prefill_pp_size != 1 or args.enable_expert_parallel
                or (args.graph_mode != "NONE" and not (flashinfer011 or ascend018 or metax024)) or args.mtp_num_speculative_tokens
                or (args.symbolic_forward_source != "cuda_eager_inplace" and not ascend018)
                or source_model.model_type not in ("qwen3", "llama") or source_model.num_experts
                or source_model.compute_dtype != "bf16" or source_model.weight_dtype != "bf16"
                or (_cli_hardware_id(args.hardware_config, HWSpec) not in ("nvidia_a800", "nvidia_h200") and not (ascend018 or metax024))):
            parser.error("activation source capacity requires versioned CPU dense BF16 Qwen3-or-Llama/TP1-or-TP2/PP1 colocated with a bound runtime policy;0.11 requires A800 PyTorch-sort or FlashInfer0.4.1; only FlashInfer0.4.1 permits separate graph configuration/reserves;0.30 A800-or-H200; other startup sources remain unbound")
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    dsv3_source = "vllm_0_27_1_deepseek_v3_sync_fp8"
    mla_sources = (*glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce", dsv3_source)
    if args.symbolic_forward_source not in glm_sources and (
            args.glm_indexer_selection_source is not None or args.glm_prefill_attention_source is not None):
        parser.error("GLM attention source options require the matching virtual forward")
    if args.source_workspace_initial_capacity_bytes is not None:
        workspace_ranks = args.prefill_tp_size
        if glm_bf16_cli_source_scope(args):
            workspace_ranks *= args.prefill_pp_size * args.prefill_dp_size * args.prefill_cp_size
            if args.deployment == "pd_disagg" and args.num_prefill_instances is not None and args.num_decode_instances is not None:
                workspace_ranks = (workspace_ranks * args.num_prefill_instances
                    + args.decode_tp_size * args.decode_pp_size * args.decode_dp_size * args.decode_cp_size * args.num_decode_instances)
        workspace_lengths = ((workspace_ranks, args.prefill_tp_size) if mla_pcp_pd_cli_scope(args)
            and args.prefill_cp_size == args.decode_cp_size == 1
            and args.prefill_pp_size == args.decode_pp_size == args.prefill_dp_size == args.decode_dp_size == 1
            and args.prefill_tp_size == args.decode_tp_size else (workspace_ranks,))
        if (args.symbolic_forward_source not in ("vllm_0_30_v4_sync_fp8_ds_mla", *glm_sources)
                or len(args.source_workspace_initial_capacity_bytes) not in workspace_lengths
                or any(size < 0 for size in args.source_workspace_initial_capacity_bytes)):
            parser.error("source workspace history requires virtual V4/GLM and one nonnegative entry capacity per TP rank; PCP PD requires every deployment physical rank")
    if args.vllm_scheduler_version == "0.30.0":
        if args.symbolic_forward_source == "vllm_0_11_glm4_sync":
            parser.error("virtual GLM4 source requires the explicit0.11 Scheduler")
        virtual_pp = (args.deployment == "colocated" and not args.enable_expert_parallel
            and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and args.symbolic_forward_source == "cuda_eager_inplace"
            and is_validated_qwen3_pipeline_model_scope(load_model_config(args.model), args.prefill_tp_size, args.prefill_pp_size))
        virtual_pd = args.glm_mtp_native_source_inputs or gqa_pd_cli_scope(args) or mla_pcp_pd_cli_scope(args) or (args.deployment == "pd_disagg"
            and args.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and load_model_config(args.model).model_type in ("qwen3", "qwen3_moe")
            and args.prefill_tp_size in (1, 2) and args.decode_tp_size in (1, 2)
            and (not args.enable_expert_parallel or args.decode_tp_size == args.prefill_tp_size == 2))
        virtual_af = (args.deployment == "af_disagg" and not args.enable_expert_parallel
            and args.symbolic_forward_source == "cuda_eager_inplace"
            and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and (load_model_config(args.model).model_type in ("qwen3", "llama")
                or (load_model_config(args.model).model_type == "qwen3_moe" and args.prefill_tp_size == 1))
            and args.prefill_pp_size == 1 and args.prefill_tp_size == args.ffn_tp_size
            and args.prefill_tp_size in (1, 2))
        virtual_ep2 = (args.enable_expert_parallel and args.prefill_tp_size == 2
            and args.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and is_validated_qwen3_moe_ep2_model_scope(load_model_config(args.model)))
        virtual_kimi_ep8 = (args.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            and _is_kimi_h200_colocated_tp8_ep8_target_scope(args))
        virtual_k3 = args.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync"
        virtual_pd_af = (args.deployment == "pd_af_disagg" and args.symbolic_forward_source == "cuda_eager_inplace"
            and not args.enable_expert_parallel and args.prefill_tp_size == args.decode_tp_size == args.ffn_tp_size == args.prefill_pp_size == 1
            and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and load_model_config(args.model).model_type in ("qwen3", "qwen3_moe", "llama"))
        virtual_host_kv = (args.symbolic_forward_source == "cuda_eager_inplace"
            and args.deployment in ("colocated", "af_disagg") and args.prefill_tp_size == args.prefill_pp_size == 1
            and not args.enable_expert_parallel and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
            and load_model_config(args.model).model_type in ("qwen3", "qwen3_moe", "llama"))
        ordinary_async = (args.vllm_async_scheduling and args.deployment == "colocated"
            and args.symbolic_forward_source == "cuda_eager_inplace"
            and HWSpec(args.hardware_config).hardware_id == "nvidia_h200"
            and args.prefill_tp_size in (1, 2) and args.prefill_pp_size == 1
            and load_model_config(args.model).compute_dtype == load_model_config(args.model).weight_dtype == "bf16"
            and ((load_model_config(args.model).model_type in ("qwen3", "llama") and not args.enable_expert_parallel)
                or (args.prefill_tp_size == 2 and args.enable_expert_parallel
                    and is_validated_qwen3_moe_ep2_model_scope(load_model_config(args.model))))
            and args.mtp_num_speculative_tokens == 0)
        ordinary_async_graph = (ordinary_async
            and args.graph_mode == "FULL_AND_PIECEWISE"
            and _shared_dense_graph_cli_scope(args))
        if (args.backend != "vllm" or (args.deployment != "colocated" and not virtual_pd and not virtual_af and not virtual_pd_af) or args.additional_colocated_hardware_config
                or (args.prefill_pp_size != 1 and not virtual_pp and not virtual_k3 and not gqa_pd_cli_scope(args) and not glm_bf16_cli_source_scope(args)) or (args.enable_expert_parallel and not virtual_ep2 and not virtual_kimi_ep8 and not gqa_pd_cli_scope(args) and not glm_bf16_cli_source_scope(args))
                or (args.vllm_async_scheduling and not ordinary_async)
                or (args.enable_prefix_caching and not (ordinary_async
                    and args.prefix_cache_mode == "trace_no_hit"))
                or (args.mtp_num_speculative_tokens and args.glm_mtp_step_inputs is None and not args.glm_mtp_native_source_inputs)
                or args.scheduler_pipeline_depth != (2 if ordinary_async else 1)
                or (args.serving_timing_mode != "device_only"
                    and not (ordinary_async and args.serving_timing_mode in (
                        "device_plus_launch", "device_plus_launch_plus_scheduler")))
                or (args.serving_timing_mode in ("device_plus_launch", "device_plus_launch_plus_scheduler")
                    and args.host_runtime_prepare_profile is None)
                or (args.graph_mode != "NONE" and not ordinary_async_graph)
                or (args.kv_lifecycle_replay is not None and not virtual_host_kv)
                or (args.symbolic_forward_source is None and args.expert_routing_policy != "seeded_assumption")):
            parser.error("0.30 CLI requires synchronous depth1 or ordinary H200 TP1/TP2 async depth2, device-only or explicit launch and runtime-prepare Host profiles for ordinary async, with only trace_no_hit prefix caching and no KV replay; async graph requires FULL_AND_PIECEWISE")
        if args.serving_timing_mode == "device_plus_launch_plus_scheduler":
            from src.serving.host_timing import load_scheduler_host_profile, load_host_request_path_profile
            from src.serving.backend.vllm import _validate_vllm030_host_profiles
            if not args.scheduler_host_profile:
                parser.error("0.30 full Host timing requires its exact source-bound Scheduler profile")
            try:
                profiles = tuple(load_scheduler_host_profile(path) for path in args.scheduler_host_profile)
                selected = tuple(profile for profile in profiles if profile.hardware_id == _cli_hardware_id(args.hardware_config, HWSpec))
                if len(selected) != 1:
                    raise ValueError("0.30 full Host timing requires one matching hardware Scheduler profile")
                _validate_vllm030_host_profiles(selected[0])
                if not args.host_request_path_profile:
                    raise ValueError("0.30 full Host timing requires its versioned request-path profile")
                request_profiles = tuple(load_host_request_path_profile(path) for path in args.host_request_path_profile)
                request_selected = tuple(profile for profile in request_profiles
                    if profile.hardware_id == _cli_hardware_id(args.hardware_config, HWSpec))
                if len(request_selected) != 1 or not request_selected[0].supports_runtime_version("0.30.0"):
                    raise ValueError("0.30 full Host request-path profile must match the native runtime version")
            except (OSError, ValueError, NotImplementedError) as error:
                parser.error(str(error))
        k3_mathematical_tp1 = (virtual_k3 and (args.prefill_tp_size, args.prefill_pp_size) == (1, 1)
            and args.runtime_memory_policy is None and args.num_kv_blocks is None)
        if ((args.runtime_memory_policy is None and not k3_mathematical_tp1) or args.num_kv_blocks is not None):
            parser.error("0.30 CLI requires --runtime-memory-policy; debug capacity cannot establish normal Serving; only explicit K3 TP1 may use its built-in mathematical byte budget")
        if any(value is None for value in (args.vllm_sdk_python, args.vllm_source_root, args.vllm_cache_layout)):
            parser.error("0.30 CLI requires explicit --vllm-sdk-python, --vllm-source-root and --vllm-cache-layout")
        if args.v4_indexer_selection_source is not None and args.symbolic_forward_source != "vllm_0_30_v4_sync_fp8_ds_mla":
            parser.error("--v4-indexer-selection-source requires the virtual V4 source profile")
        if args.symbolic_forward_source is not None:
            symbolic_model = load_model_config(args.model)
            if virtual_k3 and (
                    symbolic_model.model_id != "moonshotai/Kimi-K3" or symbolic_model.model_type != "kimi_linear"
                    or symbolic_model.compute_dtype != "bf16" or symbolic_model.weight_dtype != "mxfp4"
                    or symbolic_model.attention_layout != "hybrid_kda_mla"
                    or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_h200"
                    or args.deployment != "colocated" or (args.prefill_tp_size, args.prefill_pp_size) not in ((1, 1), (1, 16))
                    or (args.prefill_pp_size == 16 and args.pipeline_ranks_per_node != 8)
                    or args.enable_expert_parallel or args.expert_routing_policy != "seeded_assumption"
                    or args.attention_kv_layout != "mla" or args.attention_kv_distribution != "uniform"
                    or args.target_sampling_source != "vllm_0_30_plain_greedy_bf16"
                    or args.launch_policy != "analytical_launch" or args.prediction_method not in ("midlayer", "roofline")
                    or args.collective_backend != "analytical_roofline_v1"
                    or args.v4_eager_metadata or args.v4_routing_source is not None
                    or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None):
                parser.error("virtual K3 requires the bound BF16/MXFP4 checkpoint, H200 colocated TP1/PP16 or TP1/PP1, two-eight-GPU-node PP placement, seeded routing, uniform MLA, analytical launch and plain greedy")
            if args.symbolic_forward_source in mla_sources:
                expected_model_type = ("deepseek_v3" if args.symbolic_forward_source == dsv3_source else
                    "glm_moe_dsa" if args.symbolic_forward_source in glm_sources else "kimi_k2")
                expected_sampler = ("vllm_0_27_1_plain_greedy_bf16" if args.symbolic_forward_source == dsv3_source
                    else "vllm_0_30_rejection_raw_logits_bf16" if args.glm_mtp_step_inputs is not None or args.glm_mtp_native_source_inputs else "vllm_0_30_plain_greedy_bf16")
                if (symbolic_model.model_type != expected_model_type or symbolic_model.compute_dtype != "bf16"
                        or (_cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_h200" and not glm_bf16_cli_source_scope(args))
                        or (args.prefill_tp_size != 8 and args.glm_mtp_step_inputs is None and not glm_bf16_cli_source_scope(args))
                        or args.expert_routing_policy is None or args.attention_kv_layout != "mla"
                        or args.attention_kv_distribution != "uniform"
                        or args.target_sampling_source != expected_sampler
                        or args.v4_eager_metadata or args.v4_routing_source is not None
                        or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None):
                    parser.error("virtual GLM/K2.6/V3 requires its matching BF16-compute checkpoint, H200/TP8, explicit routing, uniform MLA and version-matched plain greedy")
                if args.symbolic_forward_source in glm_sources and (
                        args.glm_indexer_selection_source is None or args.source_workspace_initial_capacity_bytes is None):
                    parser.error("virtual GLM requires declared indexer cardinality and per-rank workspace entry history")
            if args.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager") and (symbolic_model.model_type not in ("qwen3", "qwen3_moe", "llama")
                    or symbolic_model.compute_dtype not in ("bf16", "fp16") or symbolic_model.weight_dtype != symbolic_model.compute_dtype
                    or (symbolic_model.compute_dtype == "fp16" and symbolic_model.model_type != "qwen3")
                    or _cli_hardware_id(args.hardware_config, HWSpec) not in (("ascend_910b",) if
                        args.symbolic_forward_source == "vllm_ascend_eager" else ("nvidia_a800", "nvidia_h200", "metax_c500"))
                    or (args.symbolic_forward_source == "vllm_ascend_eager" and symbolic_model.num_experts
                        and symbolic_model.model_type != "qwen3_moe")
                    or args.prefill_tp_size < 1 or (args.prefill_pp_size != 1 and not virtual_pp and not gqa_pd_cli_scope(args))
                    or (symbolic_model.num_experts > 0 and args.expert_routing_policy is None)
                    or (symbolic_model.num_experts == 0 and args.expert_routing_policy is not None)
                    or args.attention_kv_layout != "paged_varlen"
                    or args.attention_kv_distribution != "uniform"
                    or args.target_sampling_source != "vllm_0_30_plain_greedy_" + symbolic_model.compute_dtype
                    or args.v4_eager_metadata or args.v4_routing_source is not None
                    or args.v4_weight_storage_profile is not None):
                parser.error("symbolic CLI requires unquantized16-bit Qwen3 or BF16 Llama/MoE, CUDA-alike/PP1, matching plain greedy and uniform paged-varlen KV")
            if args.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla" and (
                    not (_is_dsv4_flash_h200_colocated_target_scope(args)
                        or _is_dsv4_pro_h200_colocated_target_scope(args))
                    or not args.v4_eager_metadata or args.v4_weight_storage_profile is None
                    or args.v4_routing_source != "vllm_0_30_modular_cuda_top6_pure_text"
                    or args.v4_source_use_deep_gemm is None or args.expert_routing_policy is None
                    or args.target_sampling_source != "vllm_0_30_plain_greedy_bf16"
                    or args.attention_kv_layout != "mla" or args.attention_kv_distribution != "uniform"):
                parser.error("symbolic V4 requires H200 Flash/TP4 or Pro/TP8, explicit eager/modular routing, source DeepGEMM branch, storage profile, dynamic routing and plain greedy MLA")
        elif (not _is_dsv4_flash_h200_colocated_target_scope(args) or not args.v4_eager_metadata
                or args.v4_weight_storage_profile is None
                or args.v4_routing_source is None or args.target_sampling_source != "vllm_0_30_plain_greedy_bf16"):
            parser.error("0.30 CLI currently requires Flash/H200/TP4 with explicit eager metadata, routing source and plain greedy sampling")
    elif any(value is not None for value in (args.vllm_sdk_python, args.vllm_source_root, args.vllm_cache_layout)):
        parser.error("explicit SDK/layout flags require --vllm-scheduler-version 0.30.0")
    if args.symbolic_forward_source == "vllm_ascend_eager" and args.vllm_scheduler_version == "0.11.0":
        symbolic_model = load_model_config(args.model)
        if (not (_ascend018_dense_cli_scope(args, symbolic_model)
                or _ascend018_colocated_ep2_cli_scope(args, symbolic_model))
                or args.target_sampling_source != "vllm_0_18_plain_greedy_bf16"
                or args.v4_eager_metadata or args.v4_routing_source is not None
                or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None
                or args.v4_indexer_selection_source is not None):
            parser.error("Ascend0.18 source requires CPU0.11 colocated BF16 dense Qwen3/Llama TP1/TP2 EP-off or validated Qwen3-MoE TP=EP2; PP1 MTP-off FULL/eager and matching greedy source")
    elif args.symbolic_forward_source == "cuda_eager_inplace" and args.vllm_scheduler_version == "0.11.0":
        symbolic_model = load_model_config(args.model)
        metax024 = (_cli_hardware_id(args.hardware_config, HWSpec) == "metax_c500"
            and ((symbolic_model.model_type in ("qwen3", "llama") and not symbolic_model.num_experts
            and symbolic_model.compute_dtype == "bf16" and args.deployment == "colocated"
            and args.prefill_pp_size == 1 and not args.additional_colocated_hardware_config)
            or _metax024_colocated_ep2_cli_scope(args, symbolic_model)))
        moe011 = _vllm011_moe_cli_scope(args, symbolic_model)
        dense_disagg011 = _vllm011_dense_disagg_cli_scope(args) and symbolic_model.model_type in ("qwen3", "llama") and not symbolic_model.num_experts
        dense_pipeline011 = _vllm011_dense_pipeline_cli_scope(args)
        if ((args.backend != "vllm" and not _shape_only_dense_conformance_cli_scope(args))
                or (args.deployment != "colocated" and not (dense_disagg011 or moe011))
                or (args.enable_prefix_caching and args.backend != "vllm")
                or args.prefill_tp_size not in (1, 2) or (args.prefill_pp_size != 1 and not dense_pipeline011)
                or (args.enable_expert_parallel and not moe011)
                or (args.additional_colocated_hardware_config and not _vllm011_multi_colocated_cli_scope(args))
                or (args.vllm_async_scheduling and not (args.backend == "vllm" and args.prefill_tp_size in (1, 2)
                and (args.graph_mode == "NONE" or _shared_dense_graph_cli_scope(args))))
                or not _shared_dense_graph_cli_scope(args)
                or args.kv_lifecycle_replay is not None
                or args.mtp_num_speculative_tokens or args.launch_policy not in ("analytical_launch", "injected_launch")
                or (not moe011 and (symbolic_model.model_type not in ("qwen3", "llama") or symbolic_model.num_experts))
                or symbolic_model.compute_dtype not in ("bf16", "fp16") or symbolic_model.weight_dtype != symbolic_model.compute_dtype
                or (symbolic_model.compute_dtype == "fp16" and symbolic_model.model_type != "qwen3")
                or (args.expert_routing_policy is not None and not moe011)
                or (_cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_a800" and not metax024
                    and not (_cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_h200"
                        and (not symbolic_model.num_experts or _vllm011_colocated_ep2_cli_scope(args, symbolic_model))
                        and symbolic_model.compute_dtype == "bf16"
                        and args.deployment == "colocated" and args.prefill_pp_size == 1
                        and not args.additional_colocated_hardware_config)
                    and not (_cli_hardware_id(args.hardware_config, HWSpec) in ("nvidia_h20", "nvidia_b200", "nvidia_b300", "ascend_950pr", "ascend_950dt")
                        and args.backend == "conformance" and not symbolic_model.num_experts
                        and symbolic_model.compute_dtype == "bf16"
                        and args.deployment == "colocated" and args.prefill_pp_size == 1
                        and not args.additional_colocated_hardware_config))
                or args.attention_kv_layout != "paged_varlen" or args.attention_kv_distribution != "uniform"
                or args.target_sampling_source != ("vllm_0_24_plain_greedy_" if metax024
                    else "vllm_0_11_plain_greedy_") + symbolic_model.compute_dtype
                or args.v4_eager_metadata or args.v4_routing_source is not None
                or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None
                or args.v4_indexer_selection_source is not None):
            parser.error("vLLM0.11 virtual source requires H200 BF16 dense colocated TP1/TP2 PP1, or existing A800 unquantized16-bit Qwen3 or BF16 Llama/MoE deployment; MTP-off and matching greedy source; async requires colocated dense PP1 eager or supported decode capture")
    elif args.symbolic_forward_source == "vllm_0_11_glm4_sync":
        symbolic_model = load_model_config(args.model)
        expected_sampler = ("vllm_0_11_greedy_rejection_bf16" if args.mtp_num_speculative_tokens
            else "vllm_0_11_plain_greedy_bf16")
        if (args.backend != "vllm" or args.deployment not in ("colocated", "pd_disagg", "af_disagg", "pd_af_disagg")
                or (args.deployment in ("af_disagg", "pd_af_disagg") and args.ffn_tp_size != 1)
                or (args.deployment == "pd_disagg" and args.decode_tp_size != 1)
                or args.prefill_tp_size != 1 or args.prefill_pp_size != 1 or args.enable_expert_parallel
                or args.additional_colocated_hardware_config or args.vllm_async_scheduling
                or args.scheduler_pipeline_depth != 1
                or args.graph_mode != "NONE"
                or args.serving_timing_mode not in ("device_only", "device_plus_launch")
                or (args.kv_lifecycle_replay is not None and args.deployment != "af_disagg")
                or args.launch_policy != "analytical_launch"
                or symbolic_model.model_type != "glm4_moe" or symbolic_model.compute_dtype != "bf16"
                or symbolic_model.weight_dtype != "bf16" or args.expert_routing_policy is None
                or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_a800"
                or args.attention_kv_layout != "paged_varlen" or args.attention_kv_distribution != "uniform"
                or args.target_sampling_source != expected_sampler
                or args.v4_eager_metadata or args.v4_routing_source is not None
                or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None
                or args.v4_indexer_selection_source is not None):
            parser.error("virtual GLM4 requires synchronous CPU0.11/A800/TP1/PP1 colocated, PD or matching AF/PD+AF BF16, eager analytical launch, uniform paged-varlen, explicit routing and its matching greedy sampler; PD+AF retains its existing Midlayer-only MTP scope")
    elif args.symbolic_forward_source not in _SGLANG_VIRTUAL_TARGET_SOURCES and (args.v4_eager_metadata or args.v4_routing_source is not None or args.target_sampling_source is not None
            or args.v4_weight_storage_profile is not None or args.symbolic_forward_source is not None) and args.vllm_scheduler_version != "0.30.0":
        parser.error("V4 source/sampling flags require the explicit0.30 Scheduler")
    capture_sizes = (() if args.graph_capture_sizes is None else
                     tuple(args.graph_capture_sizes))
    try:
        args.graph_execution_config = GraphExecutionConfig(
            args.graph_mode,
            capture_sizes,
            tuple(args.graph_piecewise_eager_workload_kinds),
            args.graph_uniform_decode_query_len,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.serving_timing_mode == "device_only":
        if args.host_launch_profile or args.scheduler_host_profile:
            parser.error(
                "Host timing profiles are forbidden for device_only")
    elif not args.host_launch_profile:
        parser.error(
            "Host timing modes require at least one --host-launch-profile")
    elif (args.serving_timing_mode == "device_plus_launch_plus_scheduler"
          and not args.scheduler_host_profile):
        parser.error(
            "device_plus_launch_plus_scheduler requires at least one "
            "--scheduler-host-profile")
    elif (args.serving_timing_mode != "device_plus_launch_plus_scheduler"
          and args.scheduler_host_profile):
        parser.error(
            "--scheduler-host-profile requires "
            "device_plus_launch_plus_scheduler")
    if (args.serving_timing_mode != "device_plus_launch_plus_scheduler"
            and args.host_request_path_profile):
        parser.error(
            "--host-request-path-profile requires "
            "device_plus_launch_plus_scheduler")
    if (args.serving_timing_mode == "device_only"
            and args.host_runtime_prepare_profile is not None):
        parser.error(
            "--host-runtime-prepare-profile requires a Host timing mode")
    _validate_expected_sha256_args(parser, args)
    if args.kv_lifecycle_replay is None:
        if args.kv_lifecycle_source_sha256 is not None:
            parser.error(
                "--kv-lifecycle-source-sha256 requires "
                "--kv-lifecycle-replay")
        kv_lifecycle_replay = None
    elif (not isinstance(args.kv_lifecycle_replay, str)
          or not args.kv_lifecycle_replay):
        parser.error("--kv-lifecycle-replay must be a non-empty path")
    else:
        if (not isinstance(args.kv_lifecycle_source_sha256, str)
                or len(args.kv_lifecycle_source_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in args.kv_lifecycle_source_sha256
                )):
            parser.error(
                "--kv-lifecycle-replay requires an expected lowercase "
                "--kv-lifecycle-source-sha256")
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                args.kv_lifecycle_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        if (kv_lifecycle_replay.source_sha256
                != args.kv_lifecycle_source_sha256):
            parser.error(
                "KV lifecycle replay source SHA256 disagrees with the "
                "expected immutable identity")
    args.kv_lifecycle_replay = kv_lifecycle_replay
    if args.expert_routing_replay is None:
        expert_routing_replay = None
    elif (not isinstance(args.expert_routing_replay, str)
          or not args.expert_routing_replay):
        parser.error("--expert-routing-replay must be a non-empty path")
    else:
        try:
            expert_routing_replay = load_expert_routing_replay(
                args.expert_routing_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    for field_name in (
            "expert_routing_source_sha256",
            "expert_routing_canonical_payload_sha256"):
        value = getattr(args, field_name)
        if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            parser.error(
                f"--{field_name.replace('_', '-')} must be a lowercase "
                "SHA256")
    if args.expert_routing_policy is None:
        if any((
                args.expert_routing_source_id is not None,
                args.expert_routing_seed is not None,
                args.expert_routing_hot_expert_count is not None,
                args.expert_routing_hot_expert_weight is not None,
                expert_routing_replay is not None,
                args.expert_routing_profile_manifest is not None,
                args.expert_routing_source_sha256 is not None,
                args.expert_routing_canonical_payload_sha256 is not None,
        )):
            parser.error(
                "expert routing source, seed, and replay require "
                "--expert-routing-policy")
    elif args.expert_routing_policy == "seeded_assumption":
        if (not isinstance(args.expert_routing_source_id, str)
                or not args.expert_routing_source_id):
            parser.error(
                "seeded expert routing requires "
                "--expert-routing-source-id")
        if args.expert_routing_seed is None:
            parser.error(
                "seeded expert routing requires --expert-routing-seed")
        if args.expert_routing_seed < 0:
            parser.error("--expert-routing-seed cannot be negative")
        if ((args.expert_routing_hot_expert_count is None)
                != (args.expert_routing_hot_expert_weight is None)):
            parser.error(
                "skewed expert routing requires both "
                "--expert-routing-hot-expert-count and "
                "--expert-routing-hot-expert-weight")
        if (args.expert_routing_hot_expert_count is not None
                and args.expert_routing_hot_expert_count < 1):
            parser.error(
                "--expert-routing-hot-expert-count must be positive")
        if (args.expert_routing_hot_expert_weight is not None
                and args.expert_routing_hot_expert_weight <= 1):
            parser.error(
                "--expert-routing-hot-expert-weight must be greater than one")
        if expert_routing_replay is not None:
            parser.error(
                "seeded expert routing cannot use "
                "--expert-routing-replay")
        if args.expert_routing_profile_manifest is not None:
            parser.error(
                "seeded expert routing cannot use "
                "--expert-routing-profile-manifest")
        if (args.expert_routing_source_sha256 is not None
                or args.expert_routing_canonical_payload_sha256 is not None):
            parser.error(
                "seeded expert routing cannot use replay SHA256 identities")
    elif args.expert_routing_policy == "profile_observation":
        if (not isinstance(args.expert_routing_source_id, str)
                or not args.expert_routing_source_id):
            parser.error(
                "profile expert routing requires --expert-routing-source-id")
        if (not isinstance(args.expert_routing_profile_manifest, str)
                or not args.expert_routing_profile_manifest):
            parser.error(
                "profile expert routing requires "
                "--expert-routing-profile-manifest")
        if (args.expert_routing_seed is not None
                or args.expert_routing_hot_expert_count is not None
                or args.expert_routing_hot_expert_weight is not None
                or expert_routing_replay is not None
                or args.expert_routing_source_sha256 is not None
                or args.expert_routing_canonical_payload_sha256 is not None):
            parser.error(
                "profile expert routing cannot use seed, hot-expert, replay, "
                "or replay SHA256 inputs")
    elif args.expert_routing_policy == "replay_observation":
        if args.execution_mode != "reference":
            parser.error(
                "replayed expert routing requires --execution-mode reference")
        if expert_routing_replay is None:
            parser.error(
                "replayed expert routing requires "
                "--expert-routing-replay")
        if args.expert_routing_seed is not None:
            parser.error("replayed expert routing cannot use a seed")
        if (args.expert_routing_hot_expert_count is not None
                or args.expert_routing_hot_expert_weight is not None):
            parser.error(
                "replayed expert routing cannot use hot-expert parameters")
        if (args.expert_routing_source_id is not None
                and args.expert_routing_source_id
                != expert_routing_replay.artifact_id):
            parser.error(
                "--expert-routing-source-id must match replay artifact_id")
        args.expert_routing_source_id = expert_routing_replay.artifact_id
        if (args.expert_routing_source_sha256 is None
                or args.expert_routing_canonical_payload_sha256 is None):
            parser.error(
                "replayed expert routing requires expected source and "
                "canonical payload SHA256 identities")
        if args.expert_routing_source_sha256 != (
                expert_routing_replay.source_sha256):
            parser.error(
                "expert routing replay source SHA256 disagrees with the "
                "expected immutable identity")
        if args.expert_routing_canonical_payload_sha256 != (
                expert_routing_replay.canonical_payload_sha256):
            parser.error(
                "expert routing replay canonical payload SHA256 disagrees "
                "with the expected immutable identity")
        if args.expert_routing_profile_manifest is not None:
            parser.error(
                "replayed expert routing cannot use "
                "--expert-routing-profile-manifest")
    else:
        parser.error(
            f"unknown expert routing policy: "
            f"{args.expert_routing_policy!r}")
    if (args.expert_routing_profile_resolution != "layer_mean"
            and args.expert_routing_policy != "profile_observation"):
        parser.error(
            "--expert-routing-profile-resolution requires profile routing")
    args.expert_routing_replay = expert_routing_replay
    if args.mtp_acceptance_replay is None:
        replay = None
    elif (not isinstance(args.mtp_acceptance_replay, str)
          or not args.mtp_acceptance_replay):
        parser.error("--mtp-acceptance-replay must be a non-empty path")
    else:
        try:
            replay = load_mtp_acceptance_replay(
                args.mtp_acceptance_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if args.mtp_acceptance_policy == "replay":
        if replay is None:
            parser.error(
                "--mtp-acceptance-policy replay requires "
                "--mtp-acceptance-replay")
        if args.mtp_acceptance_source_id == "disabled":
            args.mtp_acceptance_source_id = replay.artifact_id
    elif replay is not None:
        parser.error(
            "--mtp-acceptance-replay requires "
            "--mtp-acceptance-policy replay")
    args.mtp_acceptance_replay = replay
    if args.mtp_acceptance_policy == "maximum_entropy":
        if args.mtp_mean_acceptance_rate is None:
            parser.error(
                "maximum-entropy MTP requires "
                "--mtp-mean-acceptance-rate")
        if args.mtp_acceptance_seed is None:
            parser.error(
                "maximum-entropy MTP requires --mtp-acceptance-seed")
        if (not isinstance(args.mtp_acceptance_source_id, str)
                or not args.mtp_acceptance_source_id
                or args.mtp_acceptance_source_id == "disabled"):
            parser.error(
                "maximum-entropy MTP requires a non-disabled "
                "--mtp-acceptance-source-id")
        try:
            mtp_acceptance_assumption = MTPAcceptanceAssumption(
                args.mtp_acceptance_source_id,
                args.mtp_num_speculative_tokens,
                args.mtp_mean_acceptance_rate,
                args.mtp_acceptance_seed,
                "0" * 64,
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    elif args.mtp_acceptance_policy in (
            "disabled", "fixed_count", "replay"):
        if (args.mtp_mean_acceptance_rate is not None
                or args.mtp_acceptance_seed is not None):
            parser.error(
                "--mtp-mean-acceptance-rate and --mtp-acceptance-seed "
                "require --mtp-acceptance-policy maximum_entropy")
        mtp_acceptance_assumption = None
    else:
        raise ValueError(
            "unknown MTP acceptance policy: "
            f"{args.mtp_acceptance_policy!r}")
    args.mtp_acceptance_assumption = mtp_acceptance_assumption
    if (args.num_kv_blocks is None and args.runtime_memory_policy is None
            and not (args.vllm_scheduler_version == "0.30.0"
                and args.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync"
                and (args.prefill_tp_size, args.prefill_pp_size) == (1, 1))):
        parser.error(
            "production Forward requires --runtime-memory-policy; narrow "
            "fixtures may use --debug-num-kv-blocks; explicit K3 TP1 has a mathematical byte budget")
    if args.num_kv_blocks is not None and args.num_kv_blocks < 1:
        parser.error("--debug-num-kv-blocks must be positive")
    if (args.runtime_memory_policy is not None
            and not args.runtime_memory_policy):
        parser.error("--runtime-memory-policy must be a non-empty path")
    _validate_request_source_args(parser, args)
    if args.workload is None and args.num_kv_blocks is not None:
        parser.error(
            "Serving dataset execution forbids --debug-num-kv-blocks; "
            "use --runtime-memory-policy")
    if args.prefill_tp_size < 1:
        parser.error("--prefill-tp-size must be positive")
    if args.prefill_pp_size < 1:
        parser.error("--prefill-pp-size must be positive")
    if args.scheduler_pipeline_depth < 1:
        parser.error("--scheduler-pipeline-depth must be positive")
    if args.vllm_async_scheduling and args.scheduler_pipeline_depth < 2:
        parser.error(
            "--vllm-async-scheduling requires scheduler pipeline depth >= 2"
        )
    if (args.pipeline_ranks_per_node is not None
            and args.pipeline_ranks_per_node < 1):
        parser.error("--pipeline-ranks-per-node must be positive")
    supplied_collective_profiles = (
        args.tensor_all_reduce_profile_sha256,
        args.tensor_all_gather_profile_sha256,
    )
    if args.collective_backend is None:
        args.collective_backend = "protobridge"
    if args.collective_backend == "analytical_roofline_v1":
        if any(digest is not None for digest in supplied_collective_profiles):
            parser.error(
                "analytical Roofline collectives do not accept ProtoBridge "
                "profile SHA256 values")
    elif args.collective_backend == "protobridge":
        if ((supplied_collective_profiles[0] is None)
                != (supplied_collective_profiles[1] is None)):
            parser.error(
                "ProtoBridge collectives require both tensor profile SHA256 "
                "arguments when either one is supplied")
    else:
        raise ValueError(
            f"unknown collective backend: {args.collective_backend!r}")
    try:
        RouterConfig(
            args.router_resource_id,
            args.router_dispatch_service_time_s,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.backend != "vllm" and any((
            args.mtp_num_speculative_tokens,
            args.mtp_fixed_accepted_tokens,
            args.mtp_acceptance_policy != "disabled",
            args.mtp_acceptance_source_id != "disabled",
            args.mtp_acceptance_replay is not None,
    )):
        parser.error("MTP options require --backend vllm")
    if args.kv_lifecycle_replay is not None:
        if args.backend != "vllm":
            parser.error("KV lifecycle replay requires --backend vllm")
        if args.deployment not in ("colocated", "af_disagg"):
            parser.error(
                "KV lifecycle replay currently requires colocated or "
                "standalone AF deployment")
        if args.prefill_pp_size != 1:
            parser.error("KV lifecycle replay currently requires PP=1")
        if args.prefill_tp_size == 1:
            pass
        elif args.prefill_tp_size == 2:
            if not (
                args.deployment == "colocated"
                and args.max_batched_tokens == 8
                and args.max_requests == 1
                and args.num_kv_blocks == 2
                and args.runtime_memory_policy is None
                and args.kv_block_size_tokens == 16
                and args.max_model_len == 128
                and args.mtp_num_speculative_tokens == 0
                and args.mtp_acceptance_policy == "disabled"
                and not args.enable_expert_parallel
                and args.launch_policy == "analytical_launch"
                and args.prediction_method in ("midlayer", "roofline")
            ):
                parser.error(
                    "TP=2 KV lifecycle replay requires the exact colocated "
                    "Qwen3 scope with batch=8, max_requests=1, two 16-token "
                    "HBM blocks, max_model_len=128, MTP/EP disabled, "
                    "analytical launch, and the midlayer or roofline "
                    "predictor")
        else:
            parser.error(
                "KV lifecycle replay currently supports TP=1 or the exact "
                "colocated Qwen3 TP=2 scope")
        if args.additional_colocated_hardware_config:
            parser.error(
                "KV lifecycle replay currently requires one colocated "
                "instance")
        if args.deployment == "af_disagg":
            if (args.kv_lifecycle_replay.context.pool_name != "attention"
                    or args.kv_lifecycle_replay.context.backend_role
                    != "attention"):
                parser.error(
                    "standalone AF KV lifecycle replay must bind the "
                    "Attention Scheduler pool")
            if (args.mtp_acceptance_policy == "fixed_count"
                    and (
                        args.max_batched_tokens != 128
                        or args.max_requests != 1
                        or args.num_kv_blocks != 3
                        or args.kv_block_size_tokens != 16
                        or args.max_model_len != 128
                        or args.mtp_num_speculative_tokens != 2
                        or args.mtp_fixed_accepted_tokens != 1
                    )):
                parser.error(
                    "standalone AF Host KV lifecycle with MTP requires "
                    "batch=128, max_requests=1, three 16-token HBM blocks, "
                    "max_model_len=128, MTP depth two, and fixed acceptance "
                    "one")
        if args.mtp_acceptance_policy == "disabled":
            pass
        elif args.mtp_acceptance_policy == "fixed_count":
            if args.mtp_acceptance_replay is not None:
                parser.error(
                    "fixed-count MTP KV lifecycle replay cannot carry an "
                    "acceptance replay artifact")
            if args.max_requests != 1:
                parser.error(
                    "MTP KV lifecycle replay currently requires "
                    "--max-requests 1")
            if args.enable_expert_parallel:
                parser.error(
                    "MTP KV lifecycle replay currently requires EP disabled")
        else:
            parser.error(
                "KV lifecycle replay supports only disabled or fixed_count "
                "MTP acceptance")
            raise AssertionError("argparse parser.error unexpectedly returned")
        hbm_tier = next(
            tier for tier in args.kv_lifecycle_replay.cache_config.tiers
            if tier.name == "hbm")
        if args.kv_block_size_tokens != hbm_tier.block_size_tokens:
            parser.error(
                "--kv-block-size-tokens disagrees with KV lifecycle replay")
    if args.prefix_cache_mode == "token_ids":
        if (args.backend != "vllm" or args.production_trace is None or args.trace_prompt_token_ids_field is None):
            parser.error("token_ids prefix cache requires vllm and production trace token ID mapping")
        if args.closed_loop_client_count is not None or args.kv_lifecycle_replay is not None:
            parser.error("token prefix cache does not support closed-loop admission or KV lifecycle replay")
    if args.prefix_cache_mode == "trace_no_hit" and args.canonical_requests is None:
        parser.error("trace_no_hit requires canonical requests with recorded prefix-hit counts")
    if args.production_trace is None and any((args.trace_prompt_token_ids_field, args.trace_output_token_ids_field)):
        parser.error("token ID mappings require --production-trace")
    if args.backend != "vllm" and args.prefill_pp_size != 1:
        parser.error("pipeline parallelism currently requires --backend vllm")
    if not args.enable_chunked_prefill and args.vllm_scheduler_version == "0.11.0":
        parser.error(
            "the pinned current-vLLM Scheduler contract requires "
            "--enable-chunked-prefill")
    try:
        VLLMSchedulerConfig(
            args.model,
            scheduler_version=args.vllm_scheduler_version,
            tensor_parallel_size=args.prefill_tp_size,
            pipeline_parallel_size=args.prefill_pp_size,
            context_parallel_size=args.prefill_cp_size,
            data_parallel_size=args.prefill_dp_size if load_model_config(args.model).model_type == "glm_moe_dsa" else 1,
            pipeline_layer_partition=args.prefill_pp_layer_partition,
            enable_expert_parallel=(args.prefill_expert_parallel if args.deployment == "pd_disagg" else args.enable_expert_parallel),
            max_model_len=args.max_model_len,
            enable_chunked_prefill=args.enable_chunked_prefill,
            enable_prefix_caching=args.enable_prefix_caching,
            prefix_cache_mode=args.prefix_cache_mode,
            async_scheduling=args.vllm_async_scheduling,
            async_terminal_guard_source=args.vllm_async_terminal_guard_source,
            mtp_draft_token_ids_by_step=args.mtp_draft_token_ids_by_step,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
            mtp_acceptance_policy=args.mtp_acceptance_policy,
            mtp_fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            mtp_acceptance_source_id=args.mtp_acceptance_source_id,
            mtp_acceptance_replay=args.mtp_acceptance_replay,
            mtp_acceptance_assumption=args.mtp_acceptance_assumption,
            request_stream_sha256=(
                args.mtp_acceptance_assumption.request_stream_sha256
                if args.mtp_acceptance_assumption is not None
                else None
                if args.mtp_acceptance_replay is None
                else args.mtp_acceptance_replay.scheduler_context.
                ingress_identity_sha256
            ),
            decode_admission_contract_sha256=(
                None
                if args.mtp_acceptance_replay is None
                else args.mtp_acceptance_replay.scheduler_context.
                decode_admission_contract_sha256
            ),
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        parser.error(str(error))
    target_h200_aggregate_scope = _is_glm_h200_aggregate_target_scope(args)
    target_h200_pd_ep1_scope = _is_glm_h200_pd_tp8_ep1_target_scope(args)
    target_kimi_h200_colocated_scope = (
        _is_kimi_h200_colocated_tp8_ep8_target_scope(args))
    target_dsv3_h200_aggregate_scope = (
        _is_dsv3_h200_aggregate_target_scope(args))
    target_dsv3_h200_pd_ep1_scope = (
        _is_dsv3_h200_pd_tp8_ep1_target_scope(args))
    target_dsv4_h200_colocated_scope = (
        _is_dsv4_flash_h200_colocated_target_scope(args)
        or _is_dsv4_pro_h200_colocated_target_scope(args))
    target_tp8_scope = (
        args.prefill_tp_size == 8
        and ((args.deployment == "colocated"
              and args.prefill_pp_size in (2, 4))
             or (args.deployment == "pd_disagg"
                 and args.prefill_pp_size == 1
                 and args.decode_tp_size == 8))
        and args.enable_expert_parallel
        and not args.additional_colocated_hardware_config
    )
    pd_mtp_allowed_max_requests = _pd_mtp_allowed_max_requests(
        args, target_tp8_scope)
    glm_h200_pd_mtp_scope = _is_glm_h200_pd_mtp_target_scope(
        args, target_tp8_scope)
    dsv3_h200_pd_mtp_scope = _is_dsv3_h200_pd_mtp_target_scope(args)
    if args.backend == "vllm":
        if args.deployment not in (
                "colocated", "af_disagg", "pd_disagg", "pd_af_disagg"):
            parser.error(
                "the current vLLM baseline supports only colocated, "
                "af_disagg, pd_disagg, or pd_af_disagg deployment")
        if (args.prefill_tp_size not in (1, 2)
                and not gqa_pd_cli_scope(args)
                and not glm_bf16_cli_source_scope(args)
                and not target_tp8_scope
                and not target_h200_aggregate_scope
                and not target_h200_pd_ep1_scope
                and not target_kimi_h200_colocated_scope
                and not target_dsv3_h200_aggregate_scope
                and args.glm_mtp_step_inputs is None
                and not target_dsv3_h200_pd_ep1_scope
                and not target_dsv4_h200_colocated_scope):
            parser.error(
                "the current vLLM baseline supports TP=1 or the validated "
                "single-colocated TP=2 slice, plus the exact Kimi-K2.6 "
                "TP=EP=8/PP2, GLM-5.2 TP=EP=8/PP4, GLM-5.2 H200 "
                "aggregate TP8/PP1/EP1, GLM-5.2 H200 PD TP8/PP1/EP1, "
                "Kimi-K2.6 H200 colocated TP8/PP1/EP8, DeepSeek-V3 H200 "
                "aggregate TP8/PP1/EP1, DeepSeek-V3 H200 PD "
                "TP8/PP1/EP1, or DeepSeek-V4 H200 colocated TP4/TP8 "
                "target scope")
        if args.attention_kv_layout not in ("paged_varlen", "mla"):
            parser.error(
                "the pinned vLLM Scheduler path requires "
                "--attention-kv-layout paged_varlen or mla")
    if args.enable_expert_parallel:
        if args.backend != "vllm":
            parser.error("expert parallelism requires --backend vllm")
        colocated_ep2 = (
            args.deployment == "colocated"
            and args.prefill_tp_size == 2
            and args.prefill_pp_size == 1
        )
        pd_ep2 = (
            args.deployment == "pd_disagg"
            and args.prefill_tp_size == 2
            and args.decode_tp_size == 2
            and args.prefill_pp_size == 1
        )
        standalone_af_ep2 = (
            args.deployment == "af_disagg"
            and args.prefill_tp_size == 2
            and args.ffn_tp_size == 2
            and args.prefill_pp_size == 1
            and not args.additional_colocated_hardware_config
            and args.max_batched_tokens == 8
            and args.max_requests == 1
            and args.num_kv_blocks == 8
            and args.runtime_memory_policy is None
            and args.kv_block_size_tokens == 16
            and args.max_model_len == 128
            and args.mtp_num_speculative_tokens == 0
            and args.kv_lifecycle_replay is None
            and args.expert_routing_policy == "seeded_assumption"
            and args.launch_policy == "analytical_launch"
            and args.prediction_method == "midlayer"
        )
        if (not colocated_ep2 and not pd_ep2 and not standalone_af_ep2
                and not gqa_pd_cli_scope(args)
                and not glm_bf16_cli_source_scope(args)
                and not target_tp8_scope
                and not target_kimi_h200_colocated_scope):
            parser.error(
                "expert parallelism currently requires either one colocated "
                "or matching PD TP=EP=2 pools with PP=1, the exact "
                "standalone AF+EP2 "
                "seeded-routing scope, the exact Kimi-K2.6/GLM-5.2 "
                "TP=EP=8 target scope, or the exact Kimi-K2.6 H200 "
                "colocated TP8/PP1/EP8 scope")
    if (args.expert_routing_policy == "replay_observation"
            and not args.enable_expert_parallel):
        parser.error(
            "expert routing replay v1 requires the validated "
            "colocated TP=EP=2 slice")
    if args.deployment == "colocated":
        if (args.mtp_acceptance_replay is not None
                and args.mtp_acceptance_replay.schema_version != 1):
            parser.error(
                "colocated MTP replay requires the colocated replay schema v1")
        if (not args.additional_colocated_hardware_config
                and args.router_dispatch_service_time_s != 0.0):
            parser.error(
                "router dispatch service requires a routed multi-pool "
                "deployment")
        if (args.additional_colocated_hardware_config
                and (args.prefill_tp_size not in (1, 2)
                     or args.prefill_pp_size != 1)):
            parser.error(
                "additional colocated instances require TP=1 or TP=2 and "
                "PP=1")
        if (args.additional_colocated_hardware_config
                and args.mtp_acceptance_replay is not None):
            parser.error(
                "one MTP acceptance replay artifact binds exactly one "
                "colocated Scheduler pool; additional instances require "
                "separate per-pool artifacts")
        if ((args.prefill_tp_size != 1 or args.prefill_pp_size != 1)
                and not target_tp8_scope
                and not target_h200_aggregate_scope
                and not target_dsv3_h200_aggregate_scope
                and any((
                    args.mtp_num_speculative_tokens,
                    args.mtp_fixed_accepted_tokens,
                    args.mtp_acceptance_policy != "disabled",
                    args.mtp_acceptance_source_id != "disabled",
                ))):
            parser.error(
                "the validated single-colocated rank-local parallel paths do "
                "not support MTP outside the fixed GLM-5.2 or DeepSeek-V3 "
                "target scope")
        disaggregated_only = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        supplied = [
            option for option, value in disaggregated_only.items()
            if value is not None
        ]
        if supplied:
            parser.error(
                "colocated deployment does not accept disaggregated options: "
                + ", ".join(supplied))
    elif args.deployment == "af_disagg":
        if args.prefill_pp_size != 1:
            parser.error("af_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "af_disagg does not accept "
                "--additional-colocated-hardware-config")
        pd_only = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
        }
        supplied = [
            option for option, value in pd_only.items() if value is not None
        ]
        if supplied:
            parser.error(
                "af_disagg deployment does not accept PD options: "
                + ", ".join(supplied))
        required = {
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        missing = [
            option for option, value in required.items() if value is None
        ]
        if missing:
            parser.error(
                "af_disagg deployment requires: " + ", ".join(missing))
        af_tp_sizes = (args.prefill_tp_size, args.ffn_tp_size)
        if af_tp_sizes not in ((1, 1), (2, 2)):
            parser.error(
                "the current AF activation contract requires matching "
                "Attention/FFN TP sizes of either 1 or the validated TP=2 "
                "slice")
        af_mtp_requested = any((
                args.mtp_num_speculative_tokens,
                args.mtp_fixed_accepted_tokens,
                args.mtp_acceptance_policy != "disabled",
                args.mtp_acceptance_source_id != "disabled",
        ))
        if af_tp_sizes == (2, 2) and af_mtp_requested:
            parser.error("the validated standalone AF TP=2 slice disables MTP")
        if af_mtp_requested:
            if args.backend != "vllm":
                parser.error("standalone AF MTP requires --backend vllm")
            if (args.mtp_acceptance_policy != "fixed_count"
                    or args.mtp_acceptance_replay is not None):
                parser.error(
                    "standalone AF MTP currently requires fixed_count "
                    "acceptance without a replay artifact")
            if args.max_requests == 1:
                pass
            elif args.max_requests == 2:
                if (args.production_trace is None
                        or args.trace_replay_mode != "original"
                        or args.max_batched_tokens != 64
                        or not (
                            (args.num_kv_blocks == 6
                             and args.runtime_memory_policy is None)
                            or (args.num_kv_blocks is None
                                and args.runtime_memory_policy is not None)
                        )
                        or args.kv_block_size_tokens != 16
                        or args.max_model_len != 128
                        or args.mtp_num_speculative_tokens != 2
                        or args.mtp_fixed_accepted_tokens != 1):
                    parser.error(
                        "standalone AF MTP max_requests=2 requires the exact "
                        "original Production Trace preemption scope: batch=64, "
                        "six 16-token KV blocks, max_model_len=128, MTP depth "
                        "two, and fixed acceptance one")
            else:
                parser.error(
                    "standalone AF MTP currently requires --max-requests 1 "
                    "or the exact max_requests=2 preemption scope")
            if (args.mtp_num_speculative_tokens != 2
                    or args.mtp_fixed_accepted_tokens not in (0, 1, 2)):
                parser.error(
                    "standalone AF MTP currently requires exactly two "
                    "speculative tokens and fixed acceptance in {0, 1, 2}")
            if (args.launch_policy != "analytical_launch"
                    or args.prediction_method not in ("midlayer", "roofline")):
                parser.error(
                    "standalone AF MTP currently requires analytical launch "
                    "and --prediction-method midlayer or roofline")
            if args.enable_expert_parallel:
                parser.error("standalone AF MTP does not support EP")
            if args.expert_routing_policy != "seeded_assumption":
                parser.error(
                    "standalone AF MTP requires seeded-assumption expert "
                    "routing")
    elif args.deployment == "pd_disagg":
        if ((args.prefill_expert_parallel != args.enable_expert_parallel
                or args.decode_expert_parallel != args.enable_expert_parallel)
                and not gqa_pd_cli_scope(args) and not mla_pcp_pd_cli_scope(args)):
            parser.error("independent EP flags require source-supported eager0.30 GQA or GLM PCP PD")
        if ((args.prefill_dp_size != 1 or args.decode_dp_size != 1)
                and not gqa_pd_cli_scope(args) and not mla_pcp_pd_cli_scope(args)):
            parser.error("independent DP replica scheduling requires eager0.30 GQA or GLM PCP PD")
        if ((args.prefill_pp_size != 1 or args.decode_pp_size != 1)
                and not gqa_pd_cli_scope(args) and not mla_pcp_pd_cli_scope(args)):
            parser.error("pd_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "pd_disagg does not accept "
                "--additional-colocated-hardware-config")
        af_only = {
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        supplied = [
            option for option, value in af_only.items() if value is not None
        ]
        if supplied:
            parser.error(
                "pd_disagg deployment does not accept AF options: "
                + ", ".join(supplied))
        required = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
        }
        missing = [option for option, value in required.items() if value is None]
        if missing:
            parser.error(
                "pd_disagg deployment requires: " + ", ".join(missing))
        if args.decode_tp_size < 1:
            parser.error("--decode-tp-size must be positive")
        if args.num_prefill_instances < 1:
            parser.error("--num-prefill-instances must be positive")
        if args.num_decode_instances < 1:
            parser.error("--num-decode-instances must be positive")
        if args.backend == "vllm":
            if ((args.prefill_tp_size not in (1, 2)
                 or args.decode_tp_size not in (1, 2))
                    and not gqa_pd_cli_scope(args)
                    and not mla_pcp_pd_cli_scope(args)
                    and not target_tp8_scope
                    and not target_h200_pd_ep1_scope
                    and not target_dsv3_h200_pd_ep1_scope):
                parser.error(
                    "the current vLLM PD baseline requires independently "
                    "selected prefill/decode TP sizes in {1, 2}")
            if (Path(args.decode_hardware_config).resolve()
                    != Path(args.hardware_config).resolve()):
                if (args.transfer_link != "infiniband"
                        or args.kv_transfer_backend != "analytical_roofline_v1"):
                    parser.error("heterogeneous PD requires --transfer-link infiniband "
                        "--kv-transfer-backend analytical_roofline_v1")
                if args.mtp_num_speculative_tokens:
                    parser.error("heterogeneous PD draft MTP is outside this request contract")
            if args.mtp_num_speculative_tokens != 0 and not args.glm_mtp_native_source_inputs:
                if ((args.prefill_tp_size != 1
                     or args.decode_tp_size != 1)
                        and not target_tp8_scope
                        and not target_h200_pd_ep1_scope
                        and not target_dsv3_h200_pd_ep1_scope):
                    parser.error(
                        "the current vLLM PD MTP baseline requires P/D TP=1")
                if args.mtp_acceptance_policy in (
                        "fixed_count", "maximum_entropy"):
                    pass
                elif args.mtp_acceptance_policy == "replay":
                    replay = args.mtp_acceptance_replay
                    if (replay is None
                            or replay.schema_version != 2
                            or replay.scheduler_context.adapter_contract
                            != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT):
                        parser.error(
                            "the current vLLM PD MTP replay baseline requires "
                            "the PD decode replay schema v2 contract")
                else:
                    parser.error(
                        "the current vLLM PD MTP baseline supports only "
                        "fixed_count, maximum_entropy, or replay acceptance")
                    raise AssertionError("argparse.error unexpectedly returned")
                if (args.num_prefill_instances != 1
                        or args.num_decode_instances != 1):
                    parser.error(
                        "the current vLLM PD MTP baseline requires exactly "
                        "one prefill and one decode instance")
                if args.max_requests not in pd_mtp_allowed_max_requests:
                    if glm_h200_pd_mtp_scope:
                        parser.error(
                            "the GLM-5.2 PD MTP sweep scope "
                            "(model_type=glm_moe_dsa, deployment=pd_disagg, "
                            "P/D hardware=nvidia_h200, TP8/PP=1 with EP8 "
                            "or EP1) allows --max-requests in "
                            "{1,2,4,8,16,32,64,128}")
                    if dsv3_h200_pd_mtp_scope:
                        parser.error(
                            "the DeepSeek-V3 PD MTP sweep scope "
                            "(model_type=deepseek_v3, deployment=pd_disagg, "
                            "P/D hardware=nvidia_h200, TP8/PP=1 with EP1) "
                            "allows --max-requests in "
                            "{1,2,4,8,16,32,64,128}")
                    parser.error(
                        "the current vLLM PD MTP baseline requires "
                        "--max-requests 1 outside the named GLM-5.2 and "
                        "DeepSeek-V3 H200 PD sweep scopes")
    elif args.deployment == "pd_af_disagg":
        if args.prefill_pp_size != 1:
            parser.error(
                "pd_af_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "pd_af_disagg does not accept "
                "--additional-colocated-hardware-config")
        required = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        missing = [option for option, value in required.items() if value is None]
        if missing:
            parser.error(
                "pd_af_disagg deployment requires: " + ", ".join(missing))
        if args.num_prefill_instances < 1:
            parser.error("--num-prefill-instances must be positive")
        if args.num_decode_instances < 1:
            parser.error("--num-decode-instances must be positive")
        if (args.prefill_tp_size != 1
                or args.decode_tp_size != 1
                or args.ffn_tp_size != 1):
            parser.error(
                "the current PD+AF composition requires one rank per role")
        if (Path(args.decode_hardware_config).resolve()
                != Path(args.hardware_config).resolve()):
            parser.error(
                "the current PD+AF composition requires every worker pool "
                "to use the same hardware config file")
        pd_af_mtp_requested = any((
                args.mtp_num_speculative_tokens,
                args.mtp_fixed_accepted_tokens,
                args.mtp_acceptance_policy != "disabled",
                args.mtp_acceptance_source_id != "disabled",
        ))
        if pd_af_mtp_requested:
            if args.backend != "vllm":
                parser.error("PD+AF MTP requires --backend vllm")
            if (args.mtp_num_speculative_tokens != 2
                    or args.mtp_acceptance_policy != "fixed_count"
                    or args.mtp_fixed_accepted_tokens not in (0, 1, 2)
                    or args.mtp_acceptance_replay is not None):
                parser.error(
                    "PD+AF MTP currently requires depth two, fixed_count "
                    "acceptance zero, one, or two, and no replay artifact")
            if (args.num_prefill_instances != 1
                    or args.num_decode_instances != 1
                    or args.max_requests != 1
                    or args.max_batched_tokens != 128
                    or not (
                        (args.num_kv_blocks == 8
                         and args.runtime_memory_policy is None)
                        or (args.num_kv_blocks is None
                            and args.runtime_memory_policy is not None)
                    )
                    or args.kv_block_size_tokens != 16
                    or args.max_model_len != 128):
                parser.error(
                    "PD+AF MTP currently requires 1P x 1D, batch=128, "
                    "max_requests=1, eight 16-token KV blocks, and "
                    "max_model_len=128")
            if (args.launch_policy != "analytical_launch"
                    or args.prediction_method != "midlayer"):
                parser.error(
                    "PD+AF MTP currently requires analytical launch and "
                    "--prediction-method midlayer")
            if args.attention_kv_distribution != "uniform":
                parser.error(
                    "PD+AF MTP currently requires "
                    "--attention-kv-distribution uniform")
            if args.enable_expert_parallel:
                parser.error("PD+AF MTP does not support EP")
            if args.expert_routing_policy != "seeded_assumption":
                parser.error(
                    "PD+AF MTP requires seeded-assumption expert routing")
    else:
        raise AssertionError(
            f"argparse allowed unknown deployment: {args.deployment!r}")
    if args.norm_source_batch_invariant is not None and (
            args.vllm_scheduler_version != "0.30.0"
            or args.symbolic_forward_source != "cuda_eager_inplace"):
        parser.error("Norm source declaration requires non-batch-invariant vLLM0.30 CUDA source")
    if args.bf16_gemm_model is not None and (
            args.vllm_scheduler_version != "0.30.0"
            or args.symbolic_forward_source != "cuda_eager_inplace"
            or args.launch_policy != "analytical_launch"):
        parser.error("BF16 tiled reference requires vLLM0.30 CUDA analytical source")
    if args.bf16_expert_model is not None and (
            args.vllm_scheduler_version != "0.30.0"
            or args.symbolic_forward_source != "cuda_eager_inplace"
            or args.launch_policy != "analytical_launch"):
        parser.error("BF16 expert reference requires vLLM0.30 CUDA analytical source")
    return args


@_cli_hardware_identity_scope()
def _run_from_cli_args(argv=None) -> tuple[argparse.Namespace, dict]:
    args = _parse_args(argv)
    if args.quiet and args.output is None:
        raise ValueError("--quiet requires --output")
    if args.settings_summary is not None and args.output is None:
        raise ValueError("--settings-summary requires --output")
    if (args.settings_summary is not None
            and args.settings_summary.resolve() == Path(args.output).resolve()):
        raise ValueError("--settings-summary must differ from --output")
    if args.settings_summary is not None and args.settings_summary.exists():
        raise ValueError("--settings-summary already exists")
    try:
        payload = run(args)
    except Exception as error:
        if args.interactive_report:
            from evaluation.reporting.serving_interactive import write_status_report, default_report_path
            from evaluation.reporting.serving_viewer_ui import _configuration_metadata
            output = default_report_path(args.output, args.interactive_output)
            write_status_report({"main_chain_status": {"status": "unsupported" if isinstance(error, NotImplementedError) else "failed",
                "reason": type(error).__name__ + ": simulation failed; see the process log for details"},
                "request_source": {"requested_cli_options": _configuration_metadata(vars(args))}}, output)
        raise
    if args.settings_summary is not None:
        from evaluation.serving.experiments.experiment_runner import render_simulation_settings
        summary = render_simulation_settings(
            json.loads(Path(args.output).read_text()), Path(args.output))
        with args.settings_summary.open("x", encoding="utf-8") as stream:
            stream.write(summary)
    return args, payload


def run_cli_args(argv=None) -> dict:
    """Run one simulation from CLI arguments and return its full payload."""

    _, payload = _run_from_cli_args(argv)
    return payload


def _ensure_deterministic_hash_seed() -> None:
    if "PYTHONHASHSEED" in os.environ:
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def main(argv=None) -> int:
    if argv is None:
        _ensure_deterministic_hash_seed()
    args, payload = _run_from_cli_args(argv)
    if args.quiet:
        pass
    elif args.output:
        sys.stdout.write(Path(args.output).read_text())
    else:
        printed_payload = _persistence_projection(payload)
        print(json.dumps(printed_payload, indent=2, sort_keys=True))
    return 0
