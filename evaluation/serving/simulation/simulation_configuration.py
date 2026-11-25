"""Simulation configuration; extracted without changing the established contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from pathlib import Path
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS
from evaluation.serving.experiments.experiment_plan import vllm_bench_arrival_process, vllm_bench_arrival_times
from src.kernel.hardware_spec import HWSpec
from src.kernel.predictors.protobridge.adapter import development_profile_config_sha256, resolve_registered_profile
from src.serving.deployment.deployment_config import CollectiveConfig, PipelineP2PConfig
from src.serving.host_timing import SERVING_TIMING_MODES, load_host_request_path_profile
from src.serving.request_gen.production_trace import ProductionTraceReaderConfig, SuccessfulRequestFilter
from src.serving.workload.model_config import DEEPSEEK_V3_L32_CONFIG_SHA256, DEEPSEEK_V4_FLASH_CONFIG_SHA256, DEEPSEEK_V4_PRO_CONFIG_SHA256, load_model_config
from src.serving.workload.runtime_inputs import V32PaddedMTPInputs
from src.serving.backend.vllm import VLLMSchedulerConfig
from dataclasses import fields
from src.serving.workload.model_config import is_validated_qwen3_moe_ep2_model_scope
from src.serving.workload.model_config import is_validated_qwen3_pipeline_model_scope


ROOT = Path(__file__).resolve().parents[3]


_CLI_HARDWARE_IDENTITIES = ContextVar("midlayer_cli_hardware_identities")
_CLI_NATIVE_HARDWARE_TYPE = HWSpec


@contextmanager
def _cli_hardware_identity_scope():
    """Own validated immutable IDs for one CLI call, joining nested parsing."""
    if _CLI_HARDWARE_IDENTITIES in copy_context():
        yield
        return
    identities = {}
    token = _CLI_HARDWARE_IDENTITIES.set(identities)
    try:
        yield
    finally:
        _CLI_HARDWARE_IDENTITIES.reset(token)
        identities.clear()


def _cli_hardware_id(config_path, constructor):
    """Reuse only native fully validated file identities in the current CLI.

    Different paths and changed files require complete HWSpec validation.
    Keep standalone helpers and custom constructors on their original path.
    File stats are private invalidation tokens, never provenance identities.
    """
    context = copy_context()
    if (_CLI_HARDWARE_IDENTITIES not in context
            or constructor is not _CLI_NATIVE_HARDWARE_TYPE
            or not isinstance(config_path, (str, Path))):
        return constructor(config_path).hardware_id
    path = Path(config_path)
    resolved = path.resolve()
    before = path.stat()
    stamp = (before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns)
    identities = context[_CLI_HARDWARE_IDENTITIES]
    if resolved not in identities or identities[resolved][0] != stamp:
        identity = constructor(config_path).hardware_id
        after = path.stat()
        if (path.resolve() != resolved or stamp != (after.st_dev, after.st_ino,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("hardware configuration changed during CLI validation")
        identities[resolved] = (stamp, identity)
    return identities[resolved][1]


REQUEST_PATH_PROFILE_BY_HARDWARE = {
    hardware_id: (
        ROOT / "configs/host_timing/host_request_path_profiles/"
        f"{hardware_id}.json"
    )
    for hardware_id in EVALUATED_HARDWARE_IDS
}


RUNTIME_PREPARE_PROFILE_BY_HARDWARE = {
    hardware_id: (
        ROOT / "configs/host_timing/host_runtime_prepare_profiles/"
        f"{profile_name}"
    )
    for hardware_id, profile_name in (
        ("nvidia_a800", "nvidia_a800_eager.json"),
        ("metax_c500", "metax_c500_eager.json"),
        ("ascend_910b", "ascend_910b_eager.json"),
    )
}


VERSIONED_HOST_PROFILE_HARDWARE = frozenset({"nvidia_h200"})


ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE = {
    "ascend_910b": (
        ROOT / "configs/host_timing/"
        "engine_core_host_round_profiles/ascend_910b.json"
    ),
}


def p2p_resource_id(link: str, source_pool: str, destination_pool: str) -> str:
    endpoints = tuple(sorted((source_pool, destination_pool)))
    return f"{link}:{endpoints[0]}--{endpoints[1]}"


def protobridge_p2p_profile(hardware_id: str, requested_link: str):
    """Resolve the registered point-to-point profile for one hardware."""

    digest = development_profile_config_sha256(
        hardware_id, "send_recv", link=requested_link)
    profile = resolve_registered_profile(digest)
    if requested_link != profile.link:
        raise ValueError(
            "point-to-point link disagrees with the registered ProtoBridge "
            f"profile: requested={requested_link!r}, profile={profile.link!r}")
    return profile, digest


def tensor_collective_configs(
    pool_name: str,
    tensor_parallel_size: int,
    hardware_id: str,
    backend: str,
    all_reduce_profile_sha256: str | None,
    all_gather_profile_sha256: str | None,
) -> tuple[CollectiveConfig, ...]:
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("tensor collective pool name must be non-empty")
    if (isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)):
        raise TypeError("tensor parallel size must be an integer")
    if tensor_parallel_size == 1:
        return ()
    if tensor_parallel_size in (2, 4, 8) or (backend == "analytical_roofline_v1" and tensor_parallel_size > 1):
        if tensor_parallel_size == 4 and backend != "analytical_roofline_v1":
            raise NotImplementedError(
                "TP=4 tensor collectives exist only for the analytical "
                "DeepSeek-V3-L32 H200 path")
        if backend == "analytical_roofline_v1":
            link = {"nvidia_a800": "nvlink", "nvidia_h200": "nvlink",
                "metax_c500": "metaxlink", "ascend_910b": "hccs"}[hardware_id]
            if (all_reduce_profile_sha256 is not None
                    or all_gather_profile_sha256 is not None):
                raise ValueError(
                    "analytical Roofline collectives do not accept "
                    "ProtoBridge profile SHA256 values")
            digests = (None, None)
            algorithms = ("ring", "ring")
            topologies = ("ring", "ring")
        elif backend == "protobridge":
            if tensor_parallel_size != 2:
                raise NotImplementedError(
                    "ProtoBridge tensor collective profiles are validated "
                    "only for TP=2; TP=8 requires analytical_roofline_v1")
            supplied_digests = (
                all_reduce_profile_sha256,
                all_gather_profile_sha256,
            )
            if any(digest is None for digest in supplied_digests):
                raise ValueError(
                    "ProtoBridge collectives require explicit all-reduce and "
                    "all-gather profile SHA256 values")
            expected_digests = tuple(
                development_profile_config_sha256(
                    hardware_id, operation)
                for operation in ("all_reduce", "all_gather")
            )
            if supplied_digests != expected_digests:
                raise ValueError(
                    "ProtoBridge collective profile SHA256 values do not "
                    "match the registered hardware/operation profiles")
            digests = supplied_digests
            profiles = tuple(
                resolve_registered_profile(digest)
                for digest in digests
                if digest is not None
            )
            links = {profile.link for profile in profiles}
            if len(links) != 1:
                raise ValueError(
                    "ProtoBridge tensor profiles disagree on physical link")
            link = next(iter(links))
            algorithms = tuple(profile.algorithm for profile in profiles)
            topologies = tuple(profile.topology for profile in profiles)
        else:
            raise ValueError(
                f"unknown tensor collective backend: {backend!r}")
        resource_id = f"{link}:{pool_name}:tensor-fabric"
        return (
            CollectiveConfig(
                "all_reduce",
                algorithms[0],
                link,
                topologies[0],
                resource_id,
                backend,
                digests[0],
            ),
            CollectiveConfig(
                "all_gather",
                algorithms[1],
                link,
                topologies[1],
                resource_id,
                backend,
                digests[1],
            ),
        )
    raise NotImplementedError(
        "production tensor collective configuration supports only TP=1 or "
        "TP=2, plus the target-model analytical TP=4/TP=8 paths")


def bind_tensor_collective_profiles(args, hardware_id: str) -> None:
    supplied = (
        args.tensor_all_reduce_profile_sha256,
        args.tensor_all_gather_profile_sha256,
    )
    if args.collective_backend == "analytical_roofline_v1":
        if any(digest is not None for digest in supplied):
            raise ValueError(
                "analytical Roofline collectives do not accept ProtoBridge "
                "profile SHA256 values")
        return
    if args.collective_backend == "protobridge":
        expected = tuple(
            development_profile_config_sha256(hardware_id, operation)
            for operation in ("all_reduce", "all_gather")
        )
        if supplied == (None, None):
            (
                args.tensor_all_reduce_profile_sha256,
                args.tensor_all_gather_profile_sha256,
            ) = expected
        elif supplied != expected:
            raise ValueError(
                "ProtoBridge collective profile SHA256 values do not match "
                "the registered hardware/operation profiles")
        return
    raise ValueError(
        f"unknown collective backend: {args.collective_backend!r}")


def gqa_pd_cli_scope(args) -> bool:
    """Explicit eager0.30 GQA PD geometry, independently on both sides."""
    if (args.deployment != "pd_disagg" or args.backend != "vllm"
            or args.vllm_scheduler_version != "0.30.0"
            or args.symbolic_forward_source != "cuda_eager_inplace"
            or args.decode_hardware_config is None or args.decode_tp_size is None
            or args.prefill_cp_size != 1 or args.decode_cp_size != 1
            or args.graph_mode != "NONE"
            or args.mtp_num_speculative_tokens or args.vllm_async_scheduling):
        return False
    model = load_model_config(args.model)
    if (model.model_type not in ("qwen3", "qwen3_moe", "llama")
            or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
        return False
    for path, tp, pp, dp, ep in ((args.hardware_config, args.prefill_tp_size, args.prefill_pp_size,
            args.prefill_dp_size, args.prefill_expert_parallel),
            (args.decode_hardware_config, args.decode_tp_size, args.decode_pp_size,
                args.decode_dp_size, args.decode_expert_parallel)):
        if (_cli_hardware_id(path, HWSpec) not in ("nvidia_a800", "nvidia_h200")
                or tp < 1 or pp < 1 or pp > model.num_hidden_layers
                or model.num_attention_heads % tp or model.hidden_size % tp
                or model.intermediate_size % tp
                or (model.num_key_value_heads % tp and tp % model.num_key_value_heads)):
            return False
        if ep and model.model_type != "qwen3_moe":
            return False
        if model.model_type == "qwen3_moe" and (
                model.num_shared_experts or (not ep and model.moe_intermediate_size % (tp * dp))):
            return False
    return True


def glm_bf16_cli_source_scope(args) -> bool:
    """Explicit BF16-compute GLM source shapes and pipeline stages."""
    from src.serving.parallelism.pp import pipeline_layer_ranges, pipeline_layer_ranges_for_model
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce",
        "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if (args.deployment not in ("pd_disagg", "colocated") or args.backend != "vllm"
            or args.vllm_scheduler_version != "0.30.0"
            or args.symbolic_forward_source not in glm_sources
            or (args.deployment == "pd_disagg" and (args.decode_hardware_config is None or args.decode_tp_size is None))
            or args.graph_mode != "NONE" or args.mtp_num_speculative_tokens
            or args.vllm_async_scheduling
            or args.attention_kv_layout != "mla" or args.attention_kv_distribution != "uniform"
            or args.glm_indexer_selection_source != "declared_causal_topk"
            or args.glm_prefill_attention_source not in (None, "force_sparse_mqa")
            or args.collective_backend != "analytical_roofline_v1"
            or args.target_sampling_source != "vllm_0_30_plain_greedy_bf16"):
        return False
    model = load_model_config(args.model)
    fp8_source = args.symbolic_forward_source == "vllm_0_30_v32_fp8_sync_explicit_allreduce"
    if (model.model_type != "glm_moe_dsa" or model.compute_dtype != "bf16"
            or model.weight_dtype != ("fp8" if fp8_source else "bf16")
            or (fp8_source and not model.quantized_weight_targets)):
        return False
    endpoints = ((args.hardware_config, args.prefill_tp_size, args.prefill_cp_size,
        args.prefill_dp_size, args.prefill_expert_parallel, args.prefill_pp_size, args.prefill_pp_layer_partition),)
    if args.deployment == "pd_disagg":
        endpoints += ((args.decode_hardware_config, args.decode_tp_size, args.decode_cp_size,
            args.decode_dp_size, args.decode_expert_parallel, args.decode_pp_size, args.decode_pp_layer_partition),)
    for path, tp, cp, dp, ep, pp, partition in endpoints:
        hardware_id = _cli_hardware_id(path, HWSpec)
        if (tp < 1 or cp < 1 or dp < 1 or pp < 1 or pp > model.num_hidden_layers
                or (cp > 1 and pp > 1) or hardware_id not in ("nvidia_a800", "nvidia_h200")
                or model.num_attention_heads % tp or model.hidden_size % tp
                or model.intermediate_size % tp or model.moe_intermediate_size % tp
                or (not ep and model.moe_intermediate_size % (tp * cp * dp))
                or (fp8_source and (hardware_id != "nvidia_h200" or tp != 8
                    or cp != 1 or dp != 1 or ep or pp != 1 or partition is not None))):
            return False
        if cp > 1 and args.glm_prefill_attention_source != "force_sparse_mqa":
            return False
        if partition is not None and (len(partition) != pp or sum(partition) != model.num_hidden_layers
                or any(type(count) is not int or count <= 0 for count in partition)):
            return False
        ranges = (pipeline_layer_ranges_for_model(model, pp) if partition is None and args.deployment == "colocated"
            else pipeline_layer_ranges(model.num_hidden_layers, pp, partition))
        if (args.glm_prefill_attention_source == "force_sparse_mqa"
                and any(model.indexer_types[start] != "full" for start, _ in ranges)):
            return False
    return True


def mla_pcp_pd_cli_scope(args) -> bool:
    """PD spelling retained for existing callers of the BF16 GLM source."""
    return args.deployment == "pd_disagg" and glm_bf16_cli_source_scope(args)


def moe_data_collective_configs(pool_name, hardware_id, data_parallel_size, backend, *, model):
    """Explicit analytical shared-expert DP scenario, including TP1."""
    if data_parallel_size == 1 or model.num_experts == 0:
        return ()
    if backend != "analytical_roofline_v1" or hardware_id not in ("nvidia_a800", "nvidia_h200"):
        raise NotImplementedError("shared expert DP CLI requires its explicit CUDA analytical fabric scenario")
    return tuple(CollectiveConfig(operation, "pairwise", "nvlink", "fully_connected",
        f"nvlink:{pool_name}:expert-fabric", backend) for operation in ("all_gatherv", "reduce_scatterv"))


def uses_development_protobridge(args) -> bool:
    sizes = [args.prefill_tp_size]
    if args.decode_tp_size is not None:
        sizes.append(args.decode_tp_size)
    if args.ffn_tp_size is not None:
        sizes.append(args.ffn_tp_size)
    uses_collectives = (
        args.collective_backend == "protobridge"
        and any(size > 1 for size in sizes))
    # KV routes and intra-pool collectives have independent backends.
    uses_p2p = (
        (args.collective_backend == "protobridge"
         and (args.prefill_pp_size > 1 or args.decode_pp_size > 1))
        or (args.deployment == "pd_disagg" and args.kv_transfer_backend == "protobridge")
        or (args.deployment in ("af_disagg", "pd_af_disagg") and args.collective_backend == "protobridge"))
    return uses_collectives or uses_p2p


def pipeline_p2p_configs(
    pool_name: str,
    pipeline_parallel_size: int,
    *,
    hardware_id: str,
    backend: str = "protobridge",
    tensor_parallel_size: int = 1,
    ranks_per_node: int | None = None,
) -> tuple[PipelineP2PConfig, ...]:
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("pipeline P2P pool name must be non-empty")
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)):
        raise TypeError("pipeline parallel size must be an integer")
    if pipeline_parallel_size == 1:
        return ()
    if (isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1):
        raise ValueError("pipeline tensor parallel size must be positive")
    if ranks_per_node is not None:
        if (isinstance(ranks_per_node, bool)
                or not isinstance(ranks_per_node, int)
                or ranks_per_node < 1):
            raise ValueError("pipeline ranks per node must be positive")
    if pipeline_parallel_size >= 2:
        configs = []
        for source_stage in range(pipeline_parallel_size - 1):
            crosses_node = (
                ranks_per_node is not None
                and source_stage * tensor_parallel_size // ranks_per_node
                != ((source_stage + 1) * tensor_parallel_size
                    // ranks_per_node)
            )
            if backend == "analytical_roofline_v1":
                digest = None
                link = "infiniband" if crosses_node else "nvlink"
                if crosses_node:
                    source_node = (
                        source_stage * tensor_parallel_size // ranks_per_node)
                    destination_node = (
                        (source_stage + 1) * tensor_parallel_size
                        // ranks_per_node)
                    resource_id = (
                        f"infiniband:{pool_name}:node-{source_node}-to-"
                        f"node-{destination_node}")
                else:
                    node = (
                        0 if ranks_per_node is None
                        else source_stage * tensor_parallel_size
                        // ranks_per_node)
                    resource_id = f"nvlink:{pool_name}:node-{node}"
            elif backend == "protobridge":
                if crosses_node:
                    raise NotImplementedError(
                        "ProtoBridge pipeline P2P has no registered inter-node "
                        "profile")
                # Intra-node pipeline P2P shares the tensor-collective
                # scale-up fabric, never the scale-out record.
                scale_up_link = resolve_registered_profile(
                    development_profile_config_sha256(
                        hardware_id, "all_reduce")).link
                digest = development_profile_config_sha256(
                    hardware_id, "send_recv", link=scale_up_link)
                profile = resolve_registered_profile(digest)
                link = profile.link
                node = (
                    0 if ranks_per_node is None
                    else source_stage * tensor_parallel_size // ranks_per_node)
                resource_id = f"{link}:{pool_name}:node-{node}"
            else:
                raise ValueError(
                    f"unknown pipeline P2P backend: {backend!r}")
            configs.append(PipelineP2PConfig(
                source_stage,
                source_stage + 1,
                "send_recv",
                link,
                "point_to_point",
                resource_id,
                backend,
                digest,
            ))
        return tuple(configs)
    raise NotImplementedError(
        "production pipeline P2P configuration requires positive PP")


def production_trace_reader_config(args) -> ProductionTraceReaderConfig:
    success_filter = SuccessfulRequestFilter(
        mode=args.trace_success_filter_mode,
        field_name=args.trace_success_field,
        accepted_value=args.trace_success_value,
        provenance=args.trace_success_provenance,
    )
    return ProductionTraceReaderConfig(
        source_format=args.trace_format,
        arrival_time_field=args.trace_arrival_field,
        input_tokens_field=args.trace_input_tokens_field,
        output_tokens_field=args.trace_output_tokens_field,
        prefix_hit_tokens_field=args.trace_prefix_hit_tokens_field,
        request_id_field=args.trace_request_id_field,
        input_time_unit=args.trace_time_unit,
        successful_request_filter=success_filter,
        prompt_token_ids_field=args.trace_prompt_token_ids_field,
        output_token_ids_field=args.trace_output_token_ids_field,
    )


def arrival_schedule(args, artifact) -> tuple[tuple[float, ...], dict]:
    if args.arrival_process == "vllm_bench_gamma":
        arrival_times = vllm_bench_arrival_times(
            num_requests=artifact.num_requests,
            request_rate=args.request_rate,
            burstiness=args.arrival_burstiness,
            seed=args.arrival_seed,
        )
        process = vllm_bench_arrival_process(
            request_rate=args.request_rate,
            seed=args.arrival_seed,
        )
    elif args.arrival_process == "fixed_interval":
        interval = 1.0 / args.request_rate
        arrival_times = tuple(
            index * interval for index in range(artifact.num_requests))
        process = {
            "kind": "fixed_interval",
            "request_rate_per_s": float(args.request_rate),
        }
    elif args.arrival_process == "simultaneous":
        arrival_times = (0.0,) * artifact.num_requests
        process = {"kind": "simultaneous"}
    else:
        raise ValueError(
            f"unknown arrival process: {args.arrival_process!r}")
    return arrival_times, process


def parse_success_value_json(text: str) -> str | int | bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "--trace-success-value-json must be a JSON scalar") from error
    if (value is None
            or isinstance(value, float)
            or not isinstance(value, (str, int, bool))):
        raise argparse.ArgumentTypeError(
            "--trace-success-value-json must decode to string, integer, or bool")
    return value


def host_request_path_profile_paths(
    overrides: Sequence[str],
) -> dict[str, Path]:
    """Resolve which per-request Host request path profile each hardware uses.

    The published profile of every covered hardware is the default.  An
    override replaces exactly one hardware's entry, keyed by the hardware the
    overriding profile itself declares, so a profile can never be bound to a
    hardware it was not measured on.  Overrides exist for admission-sensitivity
    experiments that need a controlled service time; a run that passes none
    resolves exactly the published mapping.
    """

    paths = dict(REQUEST_PATH_PROFILE_BY_HARDWARE)
    for override in overrides:
        path = Path(override)
        hardware_id = load_host_request_path_profile(path).hardware_id
        if (hardware_id not in REQUEST_PATH_PROFILE_BY_HARDWARE
                and hardware_id not in VERSIONED_HOST_PROFILE_HARDWARE):
            raise ValueError(
                "Host request path profile override names uncovered hardware "
                f"{hardware_id!r}")
        paths[hardware_id] = path
    return paths


def host_runtime_prepare_profile_path(
    hardware_id: str,
    explicit_path: str | None,
) -> Path | None:
    """Resolve the runtime-prepare profile one Host timing run binds.

    Published hardware binds its profile by default; hardware with versioned
    profiles must name one explicitly.  An explicit profile is accepted only
    for versioned hardware, so the published bindings cannot be replaced.
    """

    if explicit_path is not None:
        if hardware_id not in VERSIONED_HOST_PROFILE_HARDWARE:
            raise ValueError(
                "--host-runtime-prepare-profile is only for hardware with "
                f"versioned Host profiles, not {hardware_id!r}")
        return Path(explicit_path)
    if hardware_id in VERSIONED_HOST_PROFILE_HARDWARE:
        raise ValueError(
            f"{hardware_id} Host timing requires an explicit "
            "--host-runtime-prepare-profile")
    if hardware_id in RUNTIME_PREPARE_PROFILE_BY_HARDWARE:
        return RUNTIME_PREPARE_PROFILE_BY_HARDWARE[hardware_id]
    return None


def require_expected_sha256(
    label: str,
    expected: str | None,
    actual: str,
) -> None:
    if expected is not None and expected != actual:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, actual {actual}")


def validate_expected_sha256_args(parser, args) -> None:
    bindings = (
        ("--trace-source-sha256", args.trace_source_sha256,
         "--production-trace", args.production_trace),
        ("--canonical-requests-sha256", args.canonical_requests_sha256,
         "--canonical-requests", args.canonical_requests),
        ("--model-config-sha256", args.model_config_sha256,
         "--model", args.model),
        ("--hardware-config-sha256", args.hardware_config_sha256,
         "--hardware-config", args.hardware_config),
        ("--slo-config-sha256", args.slo_config_sha256,
         "--slo-config", args.slo_config),
        ("--interactivity-tgs-config-sha256",
         args.interactivity_tgs_config_sha256,
         "--interactivity-tgs-config", args.interactivity_tgs_config),
    )
    for option, value, source_option, source_value in bindings:
        if value is None:
            continue
        if (not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            parser.error(f"{option} must be a lowercase SHA256")
        if source_value is None:
            parser.error(f"{option} requires {source_option}")


def validate_request_source_args(parser, args) -> None:
    if args.workload is not None:
        missing = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
            )
            if value is None
        ]
        if missing:
            parser.error("--workload requires: " + ", ".join(missing))
        trace_only = [
            option for option, value in (
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-request-id-field", args.trace_request_id_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-field", args.trace_success_field),
                ("--trace-success-value-json", args.trace_success_value),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
                ("--trace-arrival-rate-scale",
                 args.trace_arrival_rate_scale),
                ("--trace-arrival-rate-scales",
                 args.trace_arrival_rate_scales),
                ("--interactivity-tgs-config",
                 args.interactivity_tgs_config),
            )
            if value is not None
        ]
        if trace_only:
            parser.error(
                "--workload does not accept Production Trace options: "
                + ", ".join(trace_only))
        if not math.isfinite(args.request_rate) or args.request_rate <= 0:
            parser.error("--request-rate must be finite and positive")
        if args.arrival_process == "vllm_bench_gamma":
            if args.backend != "vllm":
                parser.error(
                    "--arrival-process vllm_bench_gamma requires --backend vllm")
            if args.arrival_seed is None or args.arrival_seed < 0:
                parser.error(
                    "vllm_bench_gamma requires a non-negative --arrival-seed")
            if args.arrival_burstiness != 1.0:
                parser.error(
                    "the pinned vLLM benchmark contract requires "
                    "--arrival-burstiness 1.0")
        elif args.arrival_process in ("fixed_interval", "simultaneous"):
            if (args.arrival_seed is not None
                    or args.arrival_burstiness is not None):
                parser.error(
                    f"--arrival-process {args.arrival_process} does not accept "
                    "arrival seed or burstiness")
            if (args.backend == "vllm"
                    and args.arrival_process == "fixed_interval"):
                parser.error(
                    "the vLLM backend cannot use a fixed-interval arrival process")
        else:
            raise AssertionError(
                "argparse allowed unknown arrival process: "
                f"{args.arrival_process!r}")
    elif args.production_trace is not None:
        generated_arrival_options = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
                ("--arrival-seed", args.arrival_seed),
                ("--arrival-burstiness", args.arrival_burstiness),
            )
            if value is not None
        ]
        if generated_arrival_options:
            parser.error(
                "--production-trace uses recorded arrivals and does not accept: "
                + ", ".join(generated_arrival_options))
        required = [
            option for option, value in (
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
            )
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if required:
            parser.error(
                "--production-trace requires: " + ", ".join(required))
        if args.backend != "vllm":
            parser.error(
                "--production-trace requires the pinned vllm Scheduler backend")
        if args.trace_success_filter_mode == "upstream_attested":
            if (args.trace_success_field is not None
                    or args.trace_success_value is not None):
                parser.error(
                    "upstream_attested success filtering does not accept "
                    "--trace-success-field or --trace-success-value-json")
        elif args.trace_success_filter_mode == "field_equals":
            filter_missing = []
            if (args.trace_success_field is None
                    or not args.trace_success_field.strip()):
                filter_missing.append("--trace-success-field")
            if args.trace_success_value is None:
                filter_missing.append("--trace-success-value-json")
            if filter_missing:
                parser.error(
                    "field_equals success filtering requires: "
                    + ", ".join(filter_missing))
        else:
            raise AssertionError(
                "argparse allowed unknown Trace success filter: "
                f"{args.trace_success_filter_mode!r}")
        if args.trace_replay_mode == "original":
            if (args.trace_arrival_rate_scale is not None
                    or args.trace_arrival_rate_scales is not None):
                parser.error(
                    "original Trace replay does not accept "
                    "arrival-rate scale options")
            if args.interactivity_tgs_config is not None:
                parser.error(
                    "original Trace replay does not accept "
                    "--interactivity-tgs-config")
        elif args.trace_replay_mode == "arrival_rate_scale":
            scale = args.trace_arrival_rate_scale
            if (scale is None or not math.isfinite(scale) or scale <= 0):
                parser.error(
                    "arrival_rate_scale replay requires a finite positive "
                    "--trace-arrival-rate-scale")
            if args.trace_arrival_rate_scales is not None:
                parser.error(
                    "arrival_rate_scale replay does not accept "
                    "--trace-arrival-rate-scales")
            if args.interactivity_tgs_config is not None:
                parser.error(
                    "arrival_rate_scale replay does not accept "
                    "--interactivity-tgs-config")
        elif args.trace_replay_mode == "arrival_rate_sweep":
            if args.trace_arrival_rate_scale is not None:
                parser.error(
                    "arrival_rate_sweep replay does not accept "
                    "--trace-arrival-rate-scale")
            scales = args.trace_arrival_rate_scales
            if scales is None:
                parser.error(
                    "arrival_rate_sweep replay requires "
                    "--trace-arrival-rate-scales")
            if any(not math.isfinite(scale) or scale <= 0
                   for scale in scales):
                parser.error(
                    "arrival_rate_sweep scales must be finite and positive")
            if len(scales) != len(set(scales)):
                parser.error("arrival_rate_sweep scales must be unique")
            if (args.interactivity_tgs_config is not None
                    and not args.interactivity_tgs_config):
                parser.error(
                    "--interactivity-tgs-config must be a non-empty path")
            if args.mtp_acceptance_replay is not None:
                parser.error(
                    "arrival_rate_sweep cannot reuse one MTP acceptance "
                    "replay across distinct request streams")
        else:
            raise AssertionError(
                "argparse allowed unknown Trace replay mode: "
                f"{args.trace_replay_mode!r}")
    elif args.canonical_requests is not None:
        incompatible_options = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
                ("--arrival-seed", args.arrival_seed),
                ("--arrival-burstiness", args.arrival_burstiness),
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-request-id-field", args.trace_request_id_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-field", args.trace_success_field),
                ("--trace-success-value-json", args.trace_success_value),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
                ("--trace-arrival-rate-scale",
                 args.trace_arrival_rate_scale),
                ("--trace-arrival-rate-scales",
                 args.trace_arrival_rate_scales),
                ("--interactivity-tgs-config",
                 args.interactivity_tgs_config),
            )
            if value is not None
        ]
        if incompatible_options:
            parser.error(
                "--canonical-requests uses recorded canonical arrivals and "
                "does not accept: " + ", ".join(incompatible_options))
    else:
        raise AssertionError("argparse did not select a request source")


GLM_PD_MTP_MAX_REQUESTS = frozenset((1, 2, 4, 8, 16, 32, 64, 128))


def load_pd_scope_configs(args):
    """Load the model and both hardware specs a two-hardware PD scope needs.

    The target-scope predicates all read the same three artifacts and treat
    an unreadable one as "this is not that scope" rather than an error, so an
    unreadable config returns None instead of raising.
    """

    try:
        model = load_model_config(args.model)
        prefill_hardware = HWSpec(args.hardware_config)
        decode_hardware = HWSpec(args.decode_hardware_config)
    except (OSError, TypeError, ValueError):
        return None
    return model, prefill_hardware, decode_hardware


def is_glm_h200_pd_mtp_target_scope(args, target_tp8_scope: bool) -> bool:
    """Return whether the named GLM/H200 PD MTP sweep scope is selected."""

    if (not target_tp8_scope
            and not is_glm_h200_pd_tp8_ep1_target_scope(args)):
        return False
    return matches_target_scope(args, TARGET_SCOPES["glm_h200_pd_mtp"])


def load_target_scope_config(args):
    """Load the model and single hardware spec a target scope needs.

    Same contract as the two-hardware variant: an unreadable config means the
    scope is not selected, so it returns None rather than raising.
    """

    try:
        model = load_model_config(args.model)
        hardware = HWSpec(args.hardware_config)
    except (OSError, TypeError, ValueError):
        return None
    return model, hardware


def is_glm_h200_aggregate_target_scope(args) -> bool:
    """Return whether the exact colocated GLM H200 aggregate scope is selected."""

    return matches_target_scope(args, TARGET_SCOPES["glm_h200_aggregate"])


def is_kimi_h200_colocated_tp8_ep8_target_scope(args) -> bool:
    """Return whether the exact colocated Kimi H200 TP8/PP1/EP8 scope is selected.

    This is the standard vLLM DeepSeek-V3-family single-node topology
    (TP8 with EP=TP, PP1) applied to Kimi-K2.6 W4A16 on H200; it is
    distinct from the source-backed PP2/EP8 A800 production scope.
    """

    return matches_target_scope(args, TARGET_SCOPES["kimi_h200_colocated"])


def is_glm_h200_pd_tp8_ep1_target_scope(args) -> bool:
    """Return whether the named GLM/H200 PD TP8/PP1/EP1 sweep scope is selected.

    The EP1 PD transfer contract is the EP8 one byte-for-byte: the
    compressed MLA KV and FP8 index cache are TP-replicated single-head
    caches, and expert parallelism only shards MoE expert weights, so the
    pinned vLLM REPLICATE rank-pair transfer is unchanged.
    """

    return matches_target_scope(args, TARGET_SCOPES["glm_h200_pd_tp8_ep1"])


DSV3_PD_MTP_MAX_REQUESTS = frozenset((1, 2, 4, 8, 16, 32, 64, 128))


def is_dsv3_h200_aggregate_target_scope(args) -> bool:
    """Return whether the exact colocated DeepSeek-V3 H200 aggregate scope is selected."""

    return matches_target_scope(args, TARGET_SCOPES["dsv3_h200_aggregate"])


def is_dsv3_h200_pd_tp8_ep1_target_scope(args) -> bool:
    """Return whether the named DeepSeek-V3/H200 PD TP8/PP1/EP1 sweep scope is selected.

    The DeepSeek-V3 PD transfer payload is the compressed MLA kv_c||k_pe
    cache only (no DSA index cache): a TP-replicated single-head cache, so
    the pinned vLLM REPLICATE rank-pair transfer is EP-independent and the
    EP1 contract matches the aggregate layout.
    """

    return matches_target_scope(args, TARGET_SCOPES["dsv3_h200_pd_tp8_ep1"])


def is_dsv3_h200_pd_mtp_target_scope(args) -> bool:
    """Return whether the named DeepSeek-V3/H200 PD MTP sweep scope is selected."""

    return matches_target_scope(args, TARGET_SCOPES["dsv3_h200_pd_mtp"])


def is_dsv4_flash_h200_colocated_target_scope(args) -> bool:
    """Return whether the DeepSeek-V4 Flash H200 colocated TP4/PP1/EP1 scope is selected."""

    return matches_target_scope(
        args, TARGET_SCOPES["dsv4_flash_h200_colocated"])


def is_dsv4_pro_h200_colocated_target_scope(args) -> bool:
    """Return whether the DeepSeek-V4 Pro H200 colocated TP8/PP1/EP1 scope is selected."""

    return matches_target_scope(
        args, TARGET_SCOPES["dsv4_pro_h200_colocated"])


TARGET_SCOPES = {
    "glm_h200_aggregate": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "kimi_h200_colocated": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "kimi_k2",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": True,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_aggregate": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (4, 8),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        # TP8 is the pinned local variant; the TP4 layout is only the named
        # 32-layer configuration.
        "tp8_bypasses_config_sha256": 8,
        "config_sha256": DEEPSEEK_V3_L32_CONFIG_SHA256,
    },
    "glm_h200_pd_tp8_ep1": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": (8,),
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_pd_tp8_ep1": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": (8,),
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    # The MTP sweep accepts whatever the PD scope accepts and only pins the
    # deployment plus the presence of the decode hardware config.
    "glm_h200_pd_mtp": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": None,
        "prefill_pp_size": None,
        "expert_parallel": None,
        "decode_tp_size": None,
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": False,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_pd_mtp": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": None,
        "prefill_pp_size": None,
        "expert_parallel": None,
        "decode_tp_size": None,
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": False,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    # DeepSeek-V4 H200 colocated rows pin one checkpoint identity each:
    # Flash at TP4/PP1/EP1, Pro at TP8/PP1/EP1.
    "dsv4_flash_h200_colocated": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "deepseek_v4",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (4,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    },
    "dsv4_pro_h200_colocated": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "deepseek_v4",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": DEEPSEEK_V4_PRO_CONFIG_SHA256,
    },
}


def matches_target_scope(args, scope: Mapping) -> bool:
    """Return whether ``args`` describes exactly the deployment ``scope`` pins.

    The argument guards are checked in the same order the hand-written
    predicates used, so an unreadable config is still not consulted when a
    cheap guard already decided the answer.
    """

    if args.deployment != scope["deployment"]:
        return False
    if scope["require_decode_hardware_config"] and (
            args.decode_hardware_config is None):
        return False
    prefill_tp_size = scope["prefill_tp_size"]
    if prefill_tp_size is not None and (
            args.prefill_tp_size not in prefill_tp_size):
        return False
    prefill_pp_size = scope["prefill_pp_size"]
    if prefill_pp_size is not None and (
            args.prefill_pp_size not in prefill_pp_size):
        return False
    decode_tp_size = scope["decode_tp_size"]
    if decode_tp_size is not None and (
            args.decode_tp_size not in decode_tp_size):
        return False
    expert_parallel = scope["expert_parallel"]
    if expert_parallel is not None and (
            args.enable_expert_parallel is not expert_parallel):
        return False
    if scope["reject_additional_colocated_config"] and (
            args.additional_colocated_hardware_config):
        return False
    if scope["hardware_roles"] == 2:
        loaded = load_pd_scope_configs(args)
        if loaded is None:
            return False
        model, prefill_hardware, decode_hardware = loaded
        if decode_hardware.hardware_id != scope["hardware_id"]:
            return False
    else:
        loaded = load_target_scope_config(args)
        if loaded is None:
            return False
        model, prefill_hardware = loaded
    if model.model_type != scope["model_type"]:
        return False
    if prefill_hardware.hardware_id != scope["hardware_id"]:
        return False
    tp8_bypasses = scope["tp8_bypasses_config_sha256"]
    if tp8_bypasses is not None and args.prefill_tp_size == tp8_bypasses:
        return True
    config_sha256 = scope["config_sha256"]
    if config_sha256 is None:
        return True
    return model.config_sha256 == config_sha256


def pd_mtp_allowed_max_requests(args, target_tp8_scope: bool) -> frozenset[int]:
    if is_glm_h200_pd_mtp_target_scope(args, target_tp8_scope):
        return GLM_PD_MTP_MAX_REQUESTS
    if is_dsv3_h200_pd_mtp_target_scope(args):
        return DSV3_PD_MTP_MAX_REQUESTS
    return frozenset((1,))


SGLANG_VIRTUAL_TARGET_SOURCES = ("sglang_0_5_19_deepseek_v3_target_fp8",
    "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm")




def ascend018_colocated_ep2_cli_scope(args, model):
    return (ordinary_colocated_ep2_cli_scope(args, model, ("ascend_910b",), ascend_full=True)
        and args.symbolic_forward_source in (None, "vllm_ascend_eager")
        and args.target_sampling_source in (None, "vllm_0_18_plain_greedy_bf16"))


def ascend018_dense_cli_scope(args, model):
    """Ascend0.18 model projection with the existing CPU0.11 Scheduler."""
    return (args.backend == "vllm" and args.vllm_scheduler_version == "0.11.0"
        and _cli_hardware_id(args.hardware_config, HWSpec) == "ascend_910b"
        and model.model_type in ("qwen3", "llama") and not model.num_experts
        and model.compute_dtype == model.weight_dtype == "bf16"
        and model.hidden_act == "silu" and model.rms_norm_eps is not None
        and ((model.model_type == "qwen3" and model.rope_scaling_type == "default")
            or (model.model_type == "llama" and not model.attention_bias
                and model.rope_scaling_type in ("default", "llama3")))
        and args.deployment == "colocated" and not args.additional_colocated_hardware_config
        and args.prefill_tp_size in (1, 2) and args.prefill_pp_size == 1
        and not args.enable_expert_parallel and not args.mtp_num_speculative_tokens
        and args.expert_routing_policy is None and args.kv_lifecycle_replay is None
        and ((args.graph_mode in ("NONE", "FULL") and not args.graph_piecewise_eager_workload_kinds)
            or (args.graph_mode == "PIECEWISE" and model.model_type == "qwen3" and args.prefill_tp_size == 1
                and set(args.graph_piecewise_eager_workload_kinds) == {"flash_attn", "kv_update"}))
        and args.graph_uniform_decode_query_len == 1
        and args.launch_policy in ("analytical_launch", "injected_launch")
        and args.prediction_method in ("roofline", "midlayer")
        and args.attention_kv_layout == "paged_varlen" and args.attention_kv_distribution == "uniform"
        and args.symbolic_forward_source in (None, "vllm_ascend_eager")
        and args.target_sampling_source in (None, "vllm_0_18_plain_greedy_bf16"))


def decode_typed_step_source_json(encoded, state_type):
    """CLI spelling of an existing dataclass, with no second payload schema."""
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("step source JSON must not contain duplicate fields")
            result[name] = value
        return result

    def immutable_arrays(value):
        if isinstance(value, list):
            return tuple(immutable_arrays(item) for item in value)
        return value

    value = json.loads(encoded, object_pairs_hook=unique_object)
    if state_type is None:
        if value is not None:
            raise ValueError("an explicit none step source requires JSON null")
        return None
    def construct(raw, constructor):
        schema = {field.name: field for field in fields(constructor)}
        # Only schema-defined optional None fields normalize here.
        if isinstance(raw, dict):
            raw = {**raw, **{name: None for name, field in schema.items()
                if field.default is None and name not in raw}}
        if not isinstance(raw, dict) or set(raw) != set(schema):
            raise ValueError(f"step source JSON must contain exactly the {constructor.__name__} fields")
        data = {name: immutable_arrays(item) for name, item in raw.items()}
        from src.serving.workload.traced_sglang import SGLangEagleChainState, SGLangGraphInputs
        if constructor is SGLangEagleChainState:
            for name in ("target_graph_inputs", "extend_graph_inputs"):
                if data[name] is not None:
                    data[name] = construct(data[name], SGLangGraphInputs)
        return constructor(**data)
    return construct(value, state_type)




def metax024_colocated_ep2_cli_scope(args, model):
    """Explicit0.24 mathematical model source; CPU Scheduler remains0.11."""
    return ordinary_colocated_ep2_cli_scope(args, model, ("metax_c500",))


def normalize_default_model_source(args):
    """Choose implemented source contracts; retain explicit branch/input requirements."""
    normalize_default_vllm011_source(args)
    if (args.symbolic_forward_source is not None or args.backend != "vllm"
            or args.launch_policy != "analytical_launch" or args.prediction_method not in ("roofline", "midlayer")
            or args.vllm_async_scheduling or args.scheduler_pipeline_depth != 1
            or (args.enable_prefix_caching and args.vllm_scheduler_version != "0.11.0") or args.graph_mode != "NONE"
            or args.serving_timing_mode not in ("device_only", "device_plus_launch")
            or args.additional_colocated_hardware_config
            or args.attention_kv_distribution != "uniform"):
        return
    model = load_model_config(args.model)
    hardware_id = _cli_hardware_id(args.hardware_config, HWSpec)
    source, sampler, version = None, None, None
    if args.vllm_scheduler_version == "0.11.0":
        if (model.model_type == "glm4_moe" and model.compute_dtype == model.weight_dtype == "bf16"
                and hardware_id == "nvidia_a800" and args.prefill_tp_size == args.prefill_pp_size == 1
                and not args.enable_expert_parallel
                and (args.kv_lifecycle_replay is None or args.deployment == "af_disagg")
                and args.deployment in ("colocated", "pd_disagg", "af_disagg", "pd_af_disagg")
                and (args.deployment not in ("af_disagg", "pd_af_disagg") or args.ffn_tp_size == 1)
                and (args.deployment != "pd_disagg" or args.decode_tp_size == 1)
                and args.expert_routing_policy is not None and args.attention_kv_layout == "paged_varlen"):
            source, version = "vllm_0_11_glm4_sync", "vllm_0_11_0"
            sampler = "vllm_0_11_greedy_rejection_bf16" if args.mtp_num_speculative_tokens else "vllm_0_11_plain_greedy_bf16"
    elif (args.vllm_scheduler_version == "0.30.0" and not args.mtp_num_speculative_tokens
            and args.serving_timing_mode == "device_only"):
        sampler, version = "vllm_0_30_plain_greedy_bf16", "vllm_0_30_0"
        if (model.model_type in ("qwen3", "qwen3_moe", "llama")
                and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
                and (model.compute_dtype == "bf16" or model.model_type == "qwen3")
                and hardware_id in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
                and args.attention_kv_layout == "paged_varlen"
                and ((model.num_experts > 0) == (args.expert_routing_policy is not None))):
            source = "vllm_ascend_eager" if hardware_id == "ascend_910b" else "cuda_eager_inplace"
            sampler = "vllm_0_30_plain_greedy_" + model.compute_dtype
        elif (hardware_id == "nvidia_h200" and args.deployment == "colocated"
                and args.prefill_pp_size == 1
                and args.attention_kv_layout == "mla" and args.expert_routing_policy is not None):
            if (model.model_type == "glm_moe_dsa" and model.compute_dtype == "bf16"
                    and model.weight_dtype in ("bf16", "fp8") and args.prefill_tp_size == 8
                    and not args.enable_expert_parallel
                    and args.glm_indexer_selection_source is not None
                    and args.source_workspace_initial_capacity_bytes is not None):
                source = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if model.weight_dtype == "bf16"
                    else "vllm_0_30_v32_fp8_sync_explicit_allreduce")
            elif (model.model_type == "kimi_k2" and model.compute_dtype == "bf16"
                    and model.weight_dtype == "mixed_int4_bf16"
                    and is_kimi_h200_colocated_tp8_ep8_target_scope(args)):
                source = "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
            elif model.model_type == "deepseek_v3" and args.prefill_tp_size == 8 and not args.enable_expert_parallel:
                source, sampler, version = ("vllm_0_27_1_deepseek_v3_sync_fp8",
                    "vllm_0_27_1_plain_greedy_bf16", "vllm_0_27_1")
            elif ((is_dsv4_flash_h200_colocated_target_scope(args)
                    or is_dsv4_pro_h200_colocated_target_scope(args))
                    and args.v4_eager_metadata and args.v4_weight_storage_profile is not None
                    and args.v4_routing_source == "vllm_0_30_modular_cuda_top6_pure_text"
                    and args.v4_source_use_deep_gemm is not None):
                source = "vllm_0_30_v4_sync_fp8_ds_mla"
    if source is None or args.target_sampling_source not in (None, sampler):
        return
    args.symbolic_forward_source, args.target_sampling_source = source, sampler
    args.model_source_normalization = {
        "selection": "default_existing_supported_source", "model_execution_version": version,
        "scope": "existing versioned source contract; all dynamic, provider-branch and capacity requirements retained",
        "token_input_scope": ("explicit request prompt/output token ID replay; not checkpoint sampling"
            if args.prefix_cache_mode == "token_ids" else "CPU Scheduler length-only simulation token IDs, not checkpoint token observations"),
        "sampling_scope": "declared source sampler; not checkpoint sampling validation"}
    if version == "vllm_0_11_0":
        args.model_source_normalization["prefix_cache_scope"] = (
            "CPU0.11 native content-prefix cache: explicit request token IDs, per-instance resident blocks"
            if args.prefix_cache_mode == "token_ids" else
            "CPU0.11 trace_length_private_nonshared_v1: requested flag recorded, content hashing disabled; "
            "only independently supplied prefix-hit lengths determine replay")


def normalize_default_vllm011_source(args):
    """Select the existing migrated ordinary source without changing other domains."""
    args.model_source_normalization = None
    dense_disagg = vllm011_dense_disagg_cli_scope(args)
    dense_pipeline = vllm011_dense_pipeline_cli_scope(args)
    multi_colocated = vllm011_multi_colocated_cli_scope(args)
    moe_disagg = (args.vllm_scheduler_version == "0.11.0" and args.backend == "vllm"
        and args.deployment in ("af_disagg", "pd_disagg")
        and vllm011_moe_cli_scope(args, load_model_config(args.model)))
    if (args.symbolic_forward_source is not None or args.vllm_scheduler_version != "0.11.0"
            or (args.backend != "vllm" and not shape_only_dense_conformance_cli_scope(args))
            or (args.deployment != "colocated" and not (dense_disagg or moe_disagg))
            or (args.enable_prefix_caching and args.backend != "vllm")
            or args.prefill_tp_size not in (1, 2) or (args.prefill_pp_size != 1 and not dense_pipeline)
            or (args.additional_colocated_hardware_config and not multi_colocated)
            or (args.vllm_async_scheduling and not (args.backend == "vllm" and args.prefill_tp_size in (1, 2)
                and (args.graph_mode == "NONE" or shared_dense_graph_cli_scope(args))))
            or not shared_dense_graph_cli_scope(args)
            or args.kv_lifecycle_replay is not None or args.mtp_num_speculative_tokens
            or args.launch_policy not in ("analytical_launch", "injected_launch")
            or args.prediction_method not in ("roofline", "midlayer")
            or args.attention_kv_layout != "paged_varlen" or args.attention_kv_distribution != "uniform"
            or args.target_sampling_source not in (None, "vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_greedy_fp16", "vllm_0_24_plain_greedy_bf16", "vllm_0_18_plain_greedy_bf16")
            or args.v4_eager_metadata or args.v4_routing_source is not None
            or args.v4_weight_storage_profile is not None or args.v4_source_use_deep_gemm is not None
            or args.v4_indexer_selection_source is not None):
        return
    model = load_model_config(args.model)
    if (dense_disagg or dense_pipeline) and model.num_experts and not moe_disagg:
        return
    moe = vllm011_moe_cli_scope(args, model)
    hardware_id = _cli_hardware_id(args.hardware_config, HWSpec)
    ascend018 = (ascend018_dense_cli_scope(args, model)
        or ascend018_colocated_ep2_cli_scope(args, model))
    metax024 = (hardware_id == "metax_c500"
        and ((not moe and model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == "bf16" and args.deployment == "colocated"
            and args.prefill_pp_size == 1 and not args.additional_colocated_hardware_config)
            or metax024_colocated_ep2_cli_scope(args, model)))
    if ((not moe and (model.model_type not in ("qwen3", "llama") or model.num_experts
                or args.expert_routing_policy is not None))
            or (args.enable_expert_parallel and not moe)
            or model.compute_dtype not in ("bf16", "fp16") or model.weight_dtype != model.compute_dtype
            or (model.compute_dtype == "fp16" and (moe or model.model_type != "qwen3"))
            or model.hidden_act != "silu" or model.rms_norm_eps is None
            or (model.model_type in ("qwen3", "qwen3_moe") and model.rope_scaling_type != "default")
            or (model.model_type == "llama" and (model.attention_bias or model.rope_scaling_type not in ("default", "llama3")))
            or (hardware_id != "nvidia_a800" and not metax024 and not ascend018
                and not (hardware_id == "nvidia_h200"
                    and (not moe or vllm011_colocated_ep2_cli_scope(args, model))
                    and args.deployment == "colocated" and args.prefill_pp_size == 1
                    and not args.additional_colocated_hardware_config
                    and model.compute_dtype == "bf16"))):
        return
    sampler = ("vllm_0_18_plain_greedy_" if ascend018 else "vllm_0_24_plain_greedy_" if metax024
        else "vllm_0_11_plain_greedy_") + model.compute_dtype
    if args.target_sampling_source not in (None, sampler):
        return
    args.symbolic_forward_source = "vllm_ascend_eager" if ascend018 else "cuda_eager_inplace"
    args.target_sampling_source = sampler
    args.model_source_normalization = {
        "selection": "default_existing_supported_source", "model_execution_version": (
            "vllm_0_18_0" if ascend018 else "vllm_0_24_0" if metax024 else "vllm_0_11_0"),
        "scope": ("existing same-profile A800 colocated instances: dense TP1/TP2 eager with the original Scheduler pipeline/async modes, or synchronous MoE TP1/validated TP=EP2; PP1 MTP-off, common per-pool model and unchanged router"
            if multi_colocated else "existing A800 Qwen3-MoE AF TP1 or validated AF+EP2, PD TP1; MTP-off eager, explicit routing and known IDs; original AF+EP2 method and PD parallel-Scheduler gates retained"
            if moe_disagg else "existing A800 untied BF16 Qwen TP1 uniform pipeline or TP2/PP2, EP-off MTP-off, sync/depth1 with eager or0.11 CUDA decode/piecewise capture; CPU Scheduler supplies per-step token IDs"
            if dense_pipeline else "A800 BF16 dense AF matching TP1/TP2 or PD+AF TP1; PP1 EP/MTP-off, sync/depth1 and vLLM0.11 decode/piecewise capture of Midlayer segmented AF services; full physical hidden/residual transport outside local captures, not upstream native/runtime binding"
            if dense_disagg and args.graph_mode != "NONE" else "existing A800 dense AF matching TP1/TP2, PD TP1/TP2 pools or PD+AF TP1; PP1 EP-off MTP-off eager with known IDs"
            if dense_disagg else "Ascend BF16 validated Qwen3-MoE colocated TP=EP2 PP1 MTP-off FULL/eager, explicit0.18 mathematical model source with unchanged CPU0.11 Scheduler; not native runtime binding"
            if ascend018 and moe else "C500 BF16 validated Qwen3-MoE colocated TP=EP2 PP1 MTP-off, explicit0.24 mathematical model source and CPU0.11 Scheduler; native router instruction/workspace and provider binding remain unknown"
            if metax024 and moe else "ordinary A800 BF16 Qwen3-MoE TP1/EP-off eager or NVIDIA validated colocated TP=EP2 with the0.11 model source, original async/depth and independent capture runtime identity; PP1 MTP-off, explicit seeded or profiled routing and known IDs"
            if moe else "C500 BF16 dense Qwen3/Llama TP1/TP2 PP1 EP-off, explicit vLLM0.24+MetaX model source with unchanged CPU0.11 Scheduler; not native runtime binding"
            if metax024 else "Ascend BF16 dense Qwen3/Llama TP1/TP2 PP1 EP/MTP-off FULL/eager, explicit vLLM0.18+Ascend model source with unchanged CPU0.11 Scheduler; not native runtime binding"
            if ascend018 else "ordinary A800 or H200 BF16 dense Qwen3/Llama eager or shared decode-capture TP1/TP2 PP1 EP-off MTP-off; H200 retains the declared0.11 CUDA model source, not capture/runtime identity; conformance is TP1/TP2 eager Roofline only"),
        "token_input_scope": ("unknown INT32 virtual IDs by shape for conformance TP1/TP2; not checkpoint-generated tokens"
            if args.backend == "conformance" else "explicit request prompt/output token ID replay; not checkpoint sampling"
            if args.prefix_cache_mode == "token_ids" else "CPU Scheduler length-only simulation token IDs, not checkpoint token observations"),
        "sampling_scope": "declared plain-greedy simulation source; not checkpoint sampling validation"}
    if args.backend == "vllm":
        args.model_source_normalization["prefix_cache_scope"] = (
            "CPU0.11 native content-prefix cache: explicit request token IDs, per-instance resident blocks"
            if args.prefix_cache_mode == "token_ids" else
            "CPU0.11 trace_length_private_nonshared_v1: requested flag recorded, content hashing disabled; "
            "only independently supplied prefix-hit lengths determine replay")


def normalize_mtp_draft_token_ids(args, parser):
    """Decode the CLI file spelling of VLLMSchedulerConfig draft frames."""
    path = args.mtp_draft_token_ids_by_step
    args.mtp_draft_token_ids_by_step = None
    if path is None:
        return

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("MTP draft JSON must not contain duplicate fields")
            result[name] = value
        return result

    try:
        raw = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
        if not isinstance(raw, dict) or any(not isinstance(rows, dict) for rows in raw.values()):
            raise ValueError("MTP draft JSON must map step IDs to request/token arrays")
        frames = {}
        for step_id, rows in raw.items():
            frame = {}
            for request_id, tokens in rows.items():
                if not isinstance(tokens, list):
                    raise ValueError("MTP draft JSON token IDs must be arrays")
                frame[request_id] = tuple(tokens)
            frames[step_id] = frame
        args.mtp_draft_token_ids_by_step = frames
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))


def normalize_vllm_step_cli_source(args, parser):
    """Bind target input ports using the existing source-frame dataclass."""
    args.vllm_step_inputs = None
    declarations = args.vllm_step_source_input
    if declarations is None:
        return
    from src.serving.workload.graph_padding import VLLMDenseGraphInputs
    model = load_model_config(args.model)
    if (args.backend != "vllm" or args.vllm_scheduler_version != "0.30.0"
            or args.deployment != "colocated" or args.additional_colocated_hardware_config
            or args.prefill_tp_size not in (1, 2) or args.prefill_pp_size != 1
            or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_h200"
            or args.symbolic_forward_source != "cuda_eager_inplace"
            or args.mtp_num_speculative_tokens or args.glm_mtp_step_inputs is not None
            or model.compute_dtype != model.weight_dtype or model.compute_dtype != "bf16"
            or model.model_type not in ("qwen3", "llama", "qwen3_moe")):
        parser.error("target step source ports require ordinary H2000.30 BF16 TP1/2 PP1 MTP-off target source")
    frames = {}
    for step_id, encoded in declarations:
        parts = step_id.split(":")
        if (len(parts) != 3 or parts[:2] != ["main", "vllm"] or not parts[2].isascii()
                or not parts[2].isdecimal() or str(int(parts[2])) != parts[2] or step_id in frames):
            parser.error("target step source ports require unique canonical main:vllm:<index> identities")
        try:
            state = decode_typed_step_source_json(encoded, VLLMDenseGraphInputs)
            if (state.scheduled_token_ids is None
                    or any(token >= model.vocab_size for token in (*state.scheduled_token_ids, *state.retained_padding_token_ids))):
                raise ValueError("target step source must declare scheduled IDs and in-vocabulary retained padding")
            if args.graph_mode == "NONE" and (state.retained_padding_token_ids or state.fa3_full_capture_id is not None):
                raise ValueError("eager target ports cannot consume captured padding or capture identity")
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        frames[step_id] = state
    if "main:vllm:0" not in frames:
        parser.error("target step source ports require the initial main:vllm:0 frame")
    args.vllm_step_inputs = frames


def normalize_glm_mtp_cli_source(args, parser):
    """Decode explicit instances of the existing V32 source-input contract."""
    args.glm_mtp_step_inputs = None
    declarations = args.glm_mtp_step_source_input
    if args.glm_mtp_native_source_inputs:
        model = load_model_config(args.model)
        if declarations is not None or args.glm_mtp_initial_target_token_ids is not None:
            parser.error("native GLM MTP frames cannot be combined with manual token/frame declarations")
        if (args.backend != "vllm" or args.vllm_scheduler_version != "0.30.0"
                or args.deployment != "pd_disagg" or args.prefill_tp_size != 8 or args.decode_tp_size != 8
                or args.prefill_pp_size != 1 or args.enable_expert_parallel
                or args.num_prefill_instances != 1 or args.num_decode_instances != 1
                or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_h200"
                or args.decode_hardware_config != args.hardware_config
                or model.model_id != "zai-org/GLM-5.3" or model.weight_dtype != "fp8"
                or args.mtp_num_speculative_tokens != 5 or args.mtp_acceptance_policy != "maximum_entropy"
                or args.mtp_acceptance_source_id == "disabled" or args.mtp_acceptance_replay is not None
                or args.graph_mode != "NONE" or args.vllm_async_scheduling
                or args.symbolic_forward_source != "vllm_0_30_v32_fp8_sync_explicit_allreduce"
                or args.target_sampling_source != "vllm_0_30_rejection_raw_logits_bf16"
                or args.glm_prefill_attention_source != "force_sparse_mqa"
                or args.kv_lifecycle_replay is not None or args.additional_colocated_hardware_config
                or args.launch_policy != "analytical_launch" or args.prediction_method not in ("roofline", "midlayer")):
            parser.error("native GLM MTP source requires GLM53 FP8 H2001P1D TP8/PP1/EP-off eager MTP5 with explicit maximum-entropy assumption")
        return
    if declarations is None and args.glm_mtp_initial_target_token_ids is None:
        return
    if declarations is None or args.glm_mtp_initial_target_token_ids is None:
        parser.error("GLM MTP requires explicit step source inputs and initial target token IDs together")
    model = load_model_config(args.model)
    if (args.backend != "vllm" or args.vllm_scheduler_version != "0.30.0"
            or args.deployment != "colocated" or args.additional_colocated_hardware_config
            or args.prefill_tp_size not in (1, 8) or args.prefill_pp_size != 1 or args.enable_expert_parallel
            or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_h200"
            or model.model_type != "glm_moe_dsa" or model.num_hidden_layers != 78
            or model.mtp_architecture != "glm_moe_dsa_mtp" or model.num_mtp_layers != 1
            or model.compute_dtype != "bf16" or model.weight_dtype not in ("bf16", "fp8")
            or args.mtp_num_speculative_tokens <= 0 or args.mtp_acceptance_policy not in ("fixed_count", "maximum_entropy")
            or args.mtp_acceptance_source_id == "disabled"
            or args.graph_mode != "NONE" or args.glm_prefill_attention_source != "force_sparse_mqa"
            or args.kv_lifecycle_replay is not None or args.closed_loop_client_count is not None
            or args.launch_policy != "analytical_launch" or args.prediction_method not in ("roofline", "midlayer")):
        parser.error("GLM MTP source inputs require H200 V32 TP1/TP8 PP1/EP-off eager positive depth and an explicit acceptance scenario")
    expected_source = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if model.weight_dtype == "bf16"
        else "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if args.symbolic_forward_source != expected_source or args.target_sampling_source != "vllm_0_30_rejection_raw_logits_bf16":
        parser.error("GLM MTP requires its matching explicit model source and raw-logits rejection sampler")
    frames = {}
    for step_id, encoded in declarations:
        parts = step_id.split(":")
        if (len(parts) != 3 or parts[:2] != ["main", "vllm"] or not parts[2].isascii()
                or not parts[2].isdecimal() or str(int(parts[2])) != parts[2] or step_id in frames):
            parser.error("GLM MTP step inputs need unique canonical main:vllm:<index> identities")
        try:
            state = decode_typed_step_source_json(encoded, V32PaddedMTPInputs)
            if (state.block_size_tokens != args.kv_block_size_tokens or len(state.draft_input_token_ids) != args.mtp_num_speculative_tokens
                    or any(len(ids) != len(state.request_ids) for ids in state.draft_input_token_ids[1:])
                    or any(token >= model.vocab_size for ids in state.draft_input_token_ids for token in ids)):
                raise ValueError("GLM MTP drafts must match declared depth, cache block size, request count and vocabulary")
        except (TypeError, ValueError, NotImplementedError) as error:
            parser.error(str(error))
        frames[step_id] = state
    initial = args.glm_mtp_initial_target_token_ids
    if ("main:vllm:0" not in frames or len(initial) != len(frames["main:vllm:0"].draft_input_token_ids[0])
            or any(not 0 <= token < model.vocab_size for token in initial)):
        parser.error("GLM MTP requires its explicit first frame and in-vocabulary initial target IDs covering its rows")
    args.glm_mtp_step_inputs = frames


def ordinary_colocated_ep2_cli_scope(args, model, hardware_ids, *, ascend_full=False):
    graph_supported = ((args.graph_mode in ("NONE", "FULL")
            and args.graph_uniform_decode_query_len == 1
            and not args.graph_piecewise_eager_workload_kinds) if ascend_full else
        (args.graph_mode in ("NONE", "FULL_DECODE_ONLY", "PIECEWISE", "FULL_AND_PIECEWISE")
            and shared_dense_graph_cli_scope(args)))
    return (model.model_type == "qwen3_moe" and model.compute_dtype == model.weight_dtype == "bf16"
            and args.backend == "vllm" and args.vllm_scheduler_version == "0.11.0"
            and _cli_hardware_id(args.hardware_config, HWSpec) in hardware_ids
            and args.deployment == "colocated" and not args.additional_colocated_hardware_config
            and args.prefill_tp_size == 2 and args.prefill_pp_size == 1 and args.enable_expert_parallel
            and is_validated_qwen3_moe_ep2_model_scope(model)
            and args.scheduler_pipeline_depth in (1, 2)
            and not args.mtp_num_speculative_tokens and args.kv_lifecycle_replay is None
            and args.serving_timing_mode in SERVING_TIMING_MODES
            and graph_supported
            and args.launch_policy in ("analytical_launch", "injected_launch")
            and args.prediction_method in ("roofline", "midlayer")
            and args.expert_routing_policy in ("seeded_assumption", "profile_observation"))


def shape_only_dense_conformance_cli_scope(args):
    return (args.backend == "conformance" and args.vllm_scheduler_version == "0.11.0"
        and args.prefill_tp_size in (1, 2) and args.prefill_pp_size == 1
        and not args.vllm_async_scheduling and args.graph_mode == "NONE"
        and args.prediction_method in ("roofline", "midlayer")
        and args.launch_policy in ("analytical_launch", "injected_launch"))


def shared_dense_graph_cli_scope(args):
    if args.graph_mode == "NONE":
        return True
    if args.graph_uniform_decode_query_len != 1:
        return False
    if args.graph_mode == "FULL":
        model = load_model_config(args.model)
        return (ascend018_dense_cli_scope(args, model)
            or ascend018_colocated_ep2_cli_scope(args, model))
    if args.graph_mode == "FULL_DECODE_ONLY":
        return not args.graph_piecewise_eager_workload_kinds
    return (args.graph_mode in ("PIECEWISE", "FULL_AND_PIECEWISE")
        and set(args.graph_piecewise_eager_workload_kinds) == {"flash_attn", "kv_update"})




def vllm011_colocated_ep2_cli_scope(args, model):
    """Same0.11 model program on NVIDIA, distinct from capture runtime identity."""
    return ordinary_colocated_ep2_cli_scope(args, model, ("nvidia_a800", "nvidia_h200"))


def vllm011_dense_disagg_cli_scope(args):
    if args.graph_mode != "NONE":
        if (args.deployment not in ("af_disagg", "pd_af_disagg")
                or args.vllm_scheduler_version != "0.11.0"
                or _cli_hardware_id(args.hardware_config, HWSpec) != "nvidia_a800"
                or not shared_dense_graph_cli_scope(args)):
            return False
        model = load_model_config(args.model)
        if (model.model_type not in ("qwen3", "llama") or model.num_experts
                or model.compute_dtype != "bf16" or model.weight_dtype != "bf16"):
            return False
    if not (args.backend == "vllm" and args.prefill_tp_size in (1, 2)
            and args.prefill_pp_size == 1 and not args.enable_expert_parallel
            and not args.vllm_async_scheduling and args.scheduler_pipeline_depth == 1
            and (not args.enable_prefix_caching or args.prefix_cache_mode == "token_ids")
            and args.serving_timing_mode in ("device_only", "device_plus_launch")
            and args.launch_policy in ("analytical_launch", "injected_launch")):
        return False
    if args.deployment == "af_disagg":
        return args.ffn_tp_size == args.prefill_tp_size
    if args.deployment not in ("pd_disagg", "pd_af_disagg"):
        return False
    if (args.decode_hardware_config is None or args.decode_tp_size not in (1, 2)
            or _cli_hardware_id(args.decode_hardware_config, HWSpec) != "nvidia_a800"):
        return False
    if args.deployment == "pd_disagg":
        return True
    return args.prefill_tp_size == args.decode_tp_size == args.ffn_tp_size == 1


def vllm011_dense_pipeline_cli_scope(args):
    return (args.backend == "vllm" and args.vllm_scheduler_version == "0.11.0"
        and _cli_hardware_id(args.hardware_config, HWSpec) == "nvidia_a800"
        and args.deployment == "colocated" and args.prefill_pp_size > 1
        and args.prefill_tp_size in (1, 2)
        and not load_model_config(args.model).tie_word_embeddings
        and not args.additional_colocated_hardware_config and not args.enable_expert_parallel
        and not args.vllm_async_scheduling and args.scheduler_pipeline_depth == 1
        and (not args.enable_prefix_caching or args.prefix_cache_mode == "token_ids")
        and shared_dense_graph_cli_scope(args)
        and args.serving_timing_mode in ("device_only", "device_plus_launch")
        and args.launch_policy in ("analytical_launch", "injected_launch")
        and is_validated_qwen3_pipeline_model_scope(load_model_config(args.model), args.prefill_tp_size, args.prefill_pp_size))


def vllm011_moe_cli_scope(args, model):
    # Keep all other eager/disaggregated domains on their original guards.
    if (vllm011_colocated_ep2_cli_scope(args, model)
            or metax024_colocated_ep2_cli_scope(args, model)
            or ascend018_colocated_ep2_cli_scope(args, model)):
        return True
    if not (model.model_type == "qwen3_moe" and model.num_experts in (2, 128)
        and args.backend == "vllm" and args.prefill_pp_size == 1
        and ((args.prefill_tp_size == 1 and not args.enable_expert_parallel)
            or (args.prefill_tp_size == 2 and args.enable_expert_parallel
                and is_validated_qwen3_moe_ep2_model_scope(model)))
        and not args.vllm_async_scheduling
        and args.graph_mode == "NONE" and args.launch_policy == "analytical_launch"
        and args.expert_routing_policy in ("seeded_assumption", "profile_observation")):
        return False
    if args.deployment == "colocated":
        return True
    if (args.scheduler_pipeline_depth != 1
            or (args.enable_prefix_caching and args.prefix_cache_mode != "token_ids")
            or args.serving_timing_mode not in ("device_only", "device_plus_launch")):
        return False
    if args.deployment == "af_disagg":
        return args.ffn_tp_size == args.prefill_tp_size
    if args.deployment == "pd_disagg":
        return (args.prefill_tp_size == args.decode_tp_size == 1 and not args.enable_expert_parallel
            and args.decode_hardware_config is not None
            and _cli_hardware_id(args.decode_hardware_config, HWSpec) == "nvidia_a800")
    return False


def vllm011_multi_colocated_cli_scope(args):
    """Same-profile replicas reuse the existing eager per-pool model source."""
    if not (args.backend == "vllm" and args.vllm_scheduler_version == "0.11.0"
        and args.deployment == "colocated" and bool(args.additional_colocated_hardware_config)
        and args.prefill_pp_size == 1 and args.prefill_tp_size in (1, 2)
        and not args.mtp_num_speculative_tokens
        and args.graph_mode == "NONE"
        and args.serving_timing_mode in ("device_only", "device_plus_launch")
        and all(Path(path).resolve() == Path(args.hardware_config).resolve()
            for path in args.additional_colocated_hardware_config)):
        return False
    model = load_model_config(args.model)
    if model.model_type == "qwen3_moe":
        return (not args.vllm_async_scheduling and args.scheduler_pipeline_depth == 1
            and vllm011_moe_cli_scope(args, model))
    return (not args.enable_expert_parallel and not model.num_experts and (model.model_type == "qwen3"
        or (model.model_type == "llama" and args.prefill_tp_size == 1)))


def vllm030_cli_scheduler_settings(args, *, tp_size, pp_size, scheduler_dtype,
                                  gpu_memory_utilization, request_stream_identity,
                                  enable_expert_parallel=None, cp_size=1, dp_size=1, pipeline_layer_partition=None,
                                  scheduler_host_profile=None, engine_core_host_round_profile=None,
                                  role="colocated", decode_admission_identity=None):
    prefill_only = args.glm_mtp_native_source_inputs and role == "prefill"
    enabled = args.enable_expert_parallel if enable_expert_parallel is None else enable_expert_parallel
    return VLLMSchedulerConfig(args.model, tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size, context_parallel_size=cp_size, data_parallel_size=dp_size,
        pipeline_layer_partition=pipeline_layer_partition,
        dtype=scheduler_dtype, max_model_len=args.max_model_len,
        enable_expert_parallel=enabled, gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=args.enable_chunked_prefill, enable_prefix_caching=args.enable_prefix_caching,
        prefix_cache_mode=args.prefix_cache_mode, async_scheduling=args.vllm_async_scheduling, scheduler_version="0.30.0",
        scheduler_host_profile=scheduler_host_profile, engine_core_host_round_profile=engine_core_host_round_profile,
        mtp_num_speculative_tokens=0 if prefill_only else args.mtp_num_speculative_tokens,
        mtp_acceptance_policy="disabled" if prefill_only else args.mtp_acceptance_policy,
        mtp_fixed_accepted_tokens=0 if prefill_only else args.mtp_fixed_accepted_tokens,
        mtp_acceptance_source_id="disabled" if prefill_only else args.mtp_acceptance_source_id,
        mtp_acceptance_assumption=None if prefill_only else args.mtp_acceptance_assumption,
        emit_glm_mtp_source_inputs=args.glm_mtp_native_source_inputs and role == "decode",
        authorized_h200_pd_mtp_sweep=args.glm_mtp_native_source_inputs,
        decode_admission_contract_sha256=decode_admission_identity if role == "decode" else None,
        request_stream_sha256=(request_stream_identity
            if args.mtp_num_speculative_tokens and not prefill_only else None))
