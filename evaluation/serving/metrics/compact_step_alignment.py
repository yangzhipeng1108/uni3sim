"""Structural alignment of compact observed steps against current lowering.

This module is evaluation-only.  It reconstructs a captured ``ScheduleStep``
and stops after logical workload construction.  Kernel lowering is also run as
a launch-drift diagnostic, but its result does not decide whether the captured
Scheduler state and logical workload are structurally aligned.  The module
never runs the Scheduler, a kernel predictor, or a simulation timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.serving.simulation.simulation_runner import (
    _tensor_collective_configs,
)
from evaluation.serving.traces.tiered_nsys_comparison import (
    ExclusionRegistry,
    _tiered_gate_category,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    DeploymentConfig,
    RouterConfig,
    WorkerPoolConfig,
)
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import (
    ScheduledRequest,
    ScheduleStep,
    schedule_step_sha256,
)
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    ExpertRoutingReplayRecord,
    ExpertRoutingResolution,
)
from src.serving.workload.workload_graph import WorkloadKind

from evaluation.json_shape import require_object_value as _mapping


_PHASES = ("prefill", "decode", "mixed")
_SCHEDULE_FIELDS = {
    "step_id", "pool_name", "requests", "finished_request_ids",
    "preempted_request_ids", "attention_kv_layout",
}
_REQUEST_FIELDS = {
    "request_id", "phase", "input_tokens", "output_tokens",
    "computed_tokens", "recomputed_tokens", "prefix_replay_tokens",
    "resumed_from_preemption", "trace_prefix_hit_tokens",
    "scheduler_prefix_hit_tokens", "prefix_scheduler_delta_tokens",
    "raw_model_output_tokens", "mtp_proposed_tokens",
    "mtp_verified_tokens", "mtp_accepted_tokens", "mtp_rejected_tokens",
    "mtp_proposer_tokens", "mtp_next_proposed_tokens", "mtp_proposer_mode",
    "mtp_acceptance_source_kind", "mtp_acceptance_source_id",
}
_REQUEST_INIT_FIELDS = _REQUEST_FIELDS - {
    "prefix_scheduler_delta_tokens", "mtp_rejected_tokens",
}
class _AlignmentUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class _ObservedAlignmentRouting(ExpertRoutingInput):
    """Evaluation-only exact histogram source for one representative step."""

    counts_by_layer: Mapping[int, tuple[int, ...]] = None
    evidence_sha256: str = ""

    def __init__(
        self,
        counts_by_layer: Mapping[int, tuple[int, ...]],
        evidence_sha256: str,
    ) -> None:
        object.__setattr__(self, "source_kind", "replay_observation")
        object.__setattr__(self, "source_id", "compact_alignment_step")
        object.__setattr__(self, "algorithm", "expert_histogram_replay_v1")
        object.__setattr__(self, "root_seed", None)
        object.__setattr__(self, "request_stream_sha256", evidence_sha256)
        object.__setattr__(self, "replay", None)
        object.__setattr__(self, "vllm_runtime_binding_sha256", evidence_sha256)
        object.__setattr__(self, "distribution", "captured_expert_histogram_v1")
        object.__setattr__(self, "hot_expert_count", None)
        object.__setattr__(self, "hot_expert_weight", None)
        object.__setattr__(self, "profile", None)
        object.__setattr__(self, "counts_by_layer", dict(counts_by_layer))
        object.__setattr__(self, "evidence_sha256", evidence_sha256)

    def resolve(
        self,
        *,
        step_id: str,
        layer_index: int | None,
        moe_invocation_id: str,
        schedule_step_sha256: str,
        tokens: int,
        num_experts: int,
        topk: int,
        model_config_sha256: str,
        pool_name: str,
        tensor_parallel_size: int,
        expert_parallel_size: int,
        expert_rank_group: tuple[int, ...],
        expert_placement_strategy: str,
        moe_layer_ordinal=None,
        assumption_model_semantics_sha256=None,
    ) -> ExpertRoutingResolution:
        del model_config_sha256, pool_name, tensor_parallel_size
        del expert_parallel_size, expert_rank_group, expert_placement_strategy
        if layer_index is None or layer_index not in self.counts_by_layer:
            raise _AlignmentUnsupported(
                f"representative step lacks observed routing for MoE layer {layer_index}"
            )
        if assumption_model_semantics_sha256 is not None:
            raise ValueError("observed routing cannot consume assumption semantics")
        if (moe_layer_ordinal is not None and (type(moe_layer_ordinal) is not int
                or moe_layer_ordinal != sorted(self.counts_by_layer).index(layer_index))):
            raise ValueError("observed routing MoE layer ordinal disagrees with captured coverage")
        counts = self.counts_by_layer[layer_index]
        if len(counts) != num_experts or sum(counts) != tokens * topk:
            raise _AlignmentUnsupported(
                "observed MoE histogram disagrees with current model topology or step tokens"
            )
        record = ExpertRoutingReplayRecord(
            step_id=step_id,
            layer_index=layer_index,
            moe_invocation_id=moe_invocation_id,
            schedule_step_sha256=schedule_step_sha256,
            tokens=tokens,
            tokens_per_expert=counts,
        )
        provenance = {
            "schema_version": 2,
            "source_kind": "replay_observation",
            "source_id": "compact_alignment_step",
            "algorithm": "expert_histogram_replay_v1",
            "distribution": "captured_expert_histogram_v1",
            "hot_expert_count": None,
            "hot_expert_weight": None,
            "root_seed": None,
            "policy_descriptor_sha256": None,
            "replay_source_sha256": self.evidence_sha256,
            "replay_canonical_payload_sha256": self.evidence_sha256,
            "record_sha256": record.record_sha256,
            "step_id": step_id,
            "layer_index": layer_index,
            "moe_invocation_id": moe_invocation_id,
            "schedule_step_sha256": schedule_step_sha256,
            "mechanism_source": "Captured",
            "validation_status": "Validated",
        }
        return ExpertRoutingResolution(counts, provenance)


def align_compact_case(
    repository_root: str | Path,
    config,
    cell,
    case,
    model,
) -> Mapping[str, object]:
    """Align every available representative phase and rank for one case."""

    root = Path(repository_root).resolve()
    prediction = config.prediction.values
    if model.config_sha256 != cell.model.model_config_sha256:
        raise ValueError("alignment model config SHA256 mismatch")
    hardware_path = (root / cell.hardware.hardware_config_path).resolve()
    if root not in hardware_path.parents:
        raise ValueError("alignment hardware config escapes repository root")
    hardware = HWSpec(hardware_path)
    if hardware.hardware_id != cell.hardware.hardware_id:
        raise ValueError("alignment hardware ID disagrees with config")
    deployment = build_deployment(prediction, model, hardware.hardware_id)

    phase_results = {}
    required_results = []
    for phase in _PHASES:
        sample = case.alignment_steps[phase]
        if sample is None:
            phase_results[phase] = {
                "status": "unsupported",
                "required": False,
                "reason": "phase was not observed in the compact capture",
                "rank_ids": [],
            }
            continue
        try:
            phase_result = _align_sample(
                phase,
                sample,
                case.operator_signatures,
                case.launch_signatures,
                model,
                deployment,
                hardware,
                prediction,
                cell.hardware.graph_execution,
                _mapping(case.manifest["runtime"], "manifest runtime")["runtime_version"],
            )
        except _AlignmentUnsupported as error:
            phase_result = {
                "status": "unsupported",
                "required": True,
                "reason": str(error),
                "rank_ids": list(sample.rank_ids),
            }
        except (KeyError, NotImplementedError) as error:
            phase_result = {
                "status": "unsupported",
                "required": True,
                "reason": f"{type(error).__name__}: {error}",
                "rank_ids": list(sample.rank_ids),
            }
        phase_results[phase] = phase_result
        required_results.append(phase_result)
    if not required_results:
        raise ValueError("compact case contains no observed alignment phase")
    matched = all(result["status"] == "matched" for result in required_results)
    if matched:
        status = "matched"
    elif any(result["status"] == "unsupported" for result in required_results):
        status = "unsupported"
    else:
        status = "mismatched"
    launch_drift_statuses = tuple(
        result["launch_drift_status"] for result in required_results
        if "launch_drift_status" in result
    )
    launch_drift_status = (
        _aggregate_status(*launch_drift_statuses)
        if launch_drift_statuses else "not_run_structural_unsupported"
    )
    return {
        "alignment_stage_ran": True,
        "alignment_checked": matched,
        "status": status,
        "formal_conclusion_allowed": matched,
        "launch_drift_status": launch_drift_status,
        "comparison_contract": {
            "association": "step_order_only",
            "scheduler_execution": "not_run",
            "predictor_execution": "not_run",
            "timeline_execution": "not_run",
            "observed_only_fields_not_compared": [
                "kernel_name", "shared_memory_bytes", "registers_per_thread",
                "stream", "device",
            ],
            "operator_dtype_comparison": "fp16_bf16_storage_width_equivalence",
            "operator_capture_projection": [
                "fused_moe_post_hook_expands_router_and_restores_internal_all_reduce_order",
                "global_lm_head_hook_expands_tp_local_linear_and_all_gather",
                "native_composite_rope_collapses_to_the_captured_rotary_module",
                "logits_advanced_indexing_internal_int32_to_int64_cast_is_not_a_module_hook",
                "virtual_forward_uses_recorded_rank_local_call_order_without_extra_dependencies",
            ],
            "launch_scope": "model_forward_only",
            "launch_comparison_role": "diagnostic_only",
            "launches_outside_scope": [
                "embedding_gather",
                "sampling_argmax_and_token_gather",
                "collective_library_decomposition",
                "split_k_and_flash_attention_combine_epilogues",
                "registered_framework_noise",
            ],
        },
        "phases": phase_results,
    }


def deserialize_schedule_step(value: Mapping[str, object]) -> ScheduleStep:
    """Strictly reconstruct the production ScheduleStep dataclasses."""

    schedule = _mapping(value, "alignment scheduler")
    _exact_fields(schedule, _SCHEDULE_FIELDS, "alignment scheduler")
    request_values = _sequence(schedule["requests"], "alignment scheduler requests")
    requests = []
    for index, request_value in enumerate(request_values):
        request = _mapping(request_value, f"alignment scheduler request {index}")
        _exact_fields(request, _REQUEST_FIELDS, f"alignment scheduler request {index}")
        init = {name: request[name] for name in _REQUEST_INIT_FIELDS}
        parsed = ScheduledRequest(**init)
        if parsed.prefix_scheduler_delta_tokens != request[
            "prefix_scheduler_delta_tokens"
        ]:
            raise ValueError(
                "alignment scheduler prefix_scheduler_delta_tokens is inconsistent"
            )
        if parsed.mtp_rejected_tokens != request["mtp_rejected_tokens"]:
            raise ValueError("alignment scheduler mtp_rejected_tokens is inconsistent")
        requests.append(parsed)
    return ScheduleStep(
        step_id=schedule["step_id"],
        pool_name=schedule["pool_name"],
        requests=tuple(requests),
        finished_request_ids=tuple(schedule["finished_request_ids"]),
        preempted_request_ids=tuple(schedule["preempted_request_ids"]),
        attention_kv_layout=schedule["attention_kv_layout"],
    )


def _align_sample(
    phase: str,
    sample,
    operator_signatures: Mapping[str, Mapping[str, object]],
    launch_signatures: Mapping[str, Mapping[str, object]],
    model,
    deployment,
    hardware,
    prediction: Mapping[str, object],
    graph_execution,
    runtime_version,
) -> Mapping[str, object]:
    step = deserialize_schedule_step(sample.scheduler)
    _validate_phase(step, phase)
    expected_ranks = tuple(deployment.worker_pools[step.pool_name].ranks)
    if tuple(sample.rank_ids) != expected_ranks:
        raise _AlignmentUnsupported(
            f"observed ranks {tuple(sample.rank_ids)} disagree with deployment ranks {expected_ranks}"
        )
    routing = _alignment_routing(sample, operator_signatures, model, step)
    pool = deployment.worker_pools[step.pool_name]
    c500_source_projection = (hardware.hardware_id == "metax_c500" and runtime_version == "0.24.0"
        and model.model_type in ("qwen3", "llama", "qwen3_moe") and len(pool.ranks) in (1, 2))
    ascend_source_projection = (hardware.hardware_id == "ascend_910b" and runtime_version == "0.18.0"
        and model.model_type in ("qwen3", "llama", "qwen3_moe") and len(pool.ranks) in (1, 2))
    virtual_scope = ((model.model_type in ("qwen3", "llama") and model.num_experts == 0
            or model.model_type == "qwen3_moe")
        and model.compute_dtype == model.weight_dtype == "bf16"
        and len(pool.ranks) == pool.parallel_plan.config.tensor_parallel_size
        and len(pool.ranks) in (1, 2)
        and (c500_source_projection or ascend_source_projection or (runtime_version in ("0.11.0", "0.30.0")
            and hardware.hardware_id in ("nvidia_a800", "nvidia_h200"))))
    captured_graph_shape = None
    if virtual_scope:
        input_all_reduce = len(pool.ranks) == 2
        hidden_shape, captured_dtype = (_observed_parallel_hidden_input(sample, operator_signatures, expected_ranks)
            if input_all_reduce else _observed_decoder_input(sample, operator_signatures, expected_ranks[0]))
        # Structural capture supplies the actual port extent. Configured
        # graph buckets are not evidence that this observed call was padded.
        virtual_scope = hidden_shape == (step.token_count, model.hidden_size)
        if (not virtual_scope and model.model_type in ("qwen3", "llama") and graph_execution.enabled
                and hidden_shape[1] == model.hidden_size):
            resolved = graph_execution.resolve(step.token_count,
                uniform_decode=all(request.phase == "decode" and request.scheduled_tokens
                    == graph_execution.uniform_decode_query_len for request in step.requests))
            if resolved.runtime_mode in ("FULL", "PIECEWISE") and resolved.model_input_tokens == hidden_shape[0]:
                captured_graph_shape = resolved
                virtual_scope = True
    if virtual_scope:
        from src.serving.workload.traced_models import trace_dense_structural_forward
        # Reuse the existing CUDA-alike mathematical source scenario for
        # C500/Ascend; their captured runtime remains a separate identity.
        source_version = ("vllm_0_30_0" if c500_source_projection or ascend_source_projection else
            {"0.11.0": "vllm_0_11_0", "0.30.0": "vllm_0_30_0"}[runtime_version])
        source = trace_dense_structural_forward(model, step, pool, hidden_shape=hidden_shape,
            hidden_dtype=model.compute_dtype, model_execution_version=source_version, input_all_reduce=input_all_reduce,
            graph_token_shape=captured_graph_shape,
            expert_routing=routing if model.model_type == "qwen3_moe" else None)
        graph = source.graph
        workload_scope = {**source.source_scope, "captured_hidden_dtype": captured_dtype,
            "captured_runtime_version": runtime_version,
            "captured_runtime_source_equivalence_claimed": False,
            "model_compute_dtype": model.compute_dtype, "input_source": "captured_hidden_shape_and_storage_width",
            "dtype_comparison": "fp16_bf16_storage_width_equivalence", "numerical_dtype_equivalence_claimed": False}
        workload_builder = "virtual_tensor_decoder_and_logits"
    else:
        raise _AlignmentUnsupported("captured model/version/hidden extent has no shared virtual forward contract")
    observed_operators = _observed_operator_sequences(
        sample, operator_signatures, expected_ranks, model
    )
    predicted_operators = _project_workload_graph(graph, expected_ranks,
        node_order_by_rank=source.operator_order_by_rank)
    observed_launches, observed_launch_error = _try_launch_projection(
        lambda: _observed_launch_sequences(
            sample, launch_signatures, expected_ranks, hardware.hardware_id
        ),
        "observed",
    )
    def project_predicted_launches():
        if input_all_reduce:
            raise _AlignmentUnsupported("captured hidden-input projection has no upstream producer launch binding")
        return _project_kernel_plan(lower_workload(graph, deployment, {hardware.hardware_id: hardware}),
            graph, expected_ranks)

    predicted_launches, predicted_launch_error = _try_launch_projection(project_predicted_launches, "predicted")
    launch_error = observed_launch_error or predicted_launch_error
    ranks = {}
    for rank in expected_ranks:
        operator_result = _compare_sequence(
            observed_operators[rank], predicted_operators[rank]
        )
        if launch_error is None:
            launch_result = _compare_sequence(
                observed_launches[rank], predicted_launches[rank]
            )
        else:
            launch_result = _unsupported_comparison(launch_error)
        ranks[str(rank)] = {
            "status": operator_result["status"],
            "operators": operator_result,
            "launches": launch_result,
        }
    status = _aggregate_status(*(value["status"] for value in ranks.values()))
    launch_drift_status = _aggregate_status(*(
        value["launches"]["status"] for value in ranks.values()
    ))
    if status == "matched":
        reason = None
    elif status == "unsupported":
        reason = "one or more required comparison components are unsupported"
    elif status == "mismatched":
        reason = "canonical sequences differ"
    else:
        raise ValueError(f"unknown alignment status: {status!r}")
    return {
        "status": status,
        "required": True,
        "reason": reason,
        "launch_drift_status": launch_drift_status,
        "rank_ids": list(expected_ranks),
        "schedule_step_sha256": schedule_step_sha256(step),
        "workload_builder": workload_builder,
        "workload_scope": workload_scope,
        "unobserved_logits_index_materializations": [
            {"node_id": node.node_id, "rank": node.rank,
                "operation": "cast_int32_int64", "elements": node.shape["elements"],
                "input_dtype": "int32", "output_dtype": "int64",
                "read_bytes": 4 * node.shape["elements"],
                "write_bytes": 8 * node.shape["elements"],
                "scope": "retained_in_workload_and_lowering; absent_from_module_hooks"}
            for node in _unobserved_logits_index_casts(graph)],
        "ranks": ranks,
    }


def _observed_decoder_input(sample, signatures, rank):
    groups = tuple(group for group in sample.operators if group["rank"] == rank)
    if len(groups) != 1:
        raise ValueError("structural forward requires exactly one observed operator group for its rank")
    events = tuple(groups[0]["events"])
    if not events:
        raise ValueError("structural forward requires a captured decoder input")
    retained = tuple(_mapping(signatures[event["signature_id"]], "decoder input signature")
        for event in events if signatures[event["signature_id"]]["operation"] != "all_reduce")
    if not retained:
        raise _AlignmentUnsupported("captured TP1 operators contain no decoder input")
    # TP1 collectives are identity operations already excluded by the shared
    # observed projection; do not skip any other captured operation here.
    signature = retained[0]
    if signature["operation"] != "rms_norm":
        raise _AlignmentUnsupported("virtual decoder projection requires the first captured operator to be input RMSNorm")
    attrs = _mapping(signature["attrs"], "decoder input attributes")
    shape = _mapping(attrs["shape"], "decoder input shape")
    dimensions = _shape_list(shape, "input_shape")
    raw_dtype = str(attrs["dtype"]).strip().lower()
    dtype_aliases = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16"}
    if raw_dtype not in dtype_aliases:
        raise _AlignmentUnsupported("captured decoder input requires an explicit floating dtype")
    return (_product(dimensions[:-1]), dimensions[-1]), dtype_aliases[raw_dtype]


def _observed_parallel_hidden_input(sample, signatures, ranks):
    groups = {group["rank"]: group for group in sample.operators}
    if set(groups) != set(ranks):
        raise ValueError("parallel hidden input must cover every captured rank")
    ports = []
    for rank in ranks:
        events = tuple(groups[rank]["events"])
        if len(events) < 2 or events[0]["layer"] is not None or events[1]["layer"] != 0:
            raise _AlignmentUnsupported("TP2 hidden boundary requires captured embedding reduction followed by layer zero")
        signature = _mapping(signatures[events[0]["signature_id"]], "parallel hidden signature")
        if signature["operation"] != "all_reduce":
            raise _AlignmentUnsupported("TP2 hidden boundary must retain its captured first all-reduce")
        attrs = _mapping(signature["attrs"], "parallel hidden attributes")
        shape = _mapping(attrs["shape"], "parallel hidden shape")
        dimensions = _shape_list(shape, "input_shape")
        dtype = str(attrs["dtype"]).strip().lower()
        aliases = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16"}
        if dtype not in aliases or shape["msg_bytes"] != _product(dimensions) * _floating_dtype_bytes(dtype):
            raise _AlignmentUnsupported("captured embedding reduction must preserve its full typed hidden payload")
        port = ((_product(dimensions[:-1]), dimensions[-1]), aliases[dtype])
        norm_port = _observed_decoder_input(sample, signatures, rank)
        if norm_port != port:
            raise _AlignmentUnsupported("captured embedding reduction and first decoder input disagree")
        ports.append(port)
    if any(port != ports[0] for port in ports[1:]):
        raise _AlignmentUnsupported("captured TP2 hidden input shape/dtype differs across ranks")
    return ports[0]


def load_case_setup(config, model, hardware_path):
    """Load one case's hardware and build its deployment from the config.

    The three alignment consumers (the kernel-coverage diagnostics, the paired
    Graph-padding steps and the C500 reduction) each did these three steps in
    the same order; the deployment builder stays private to this module.
    """

    hardware = HWSpec(hardware_path)
    prediction = config.prediction.values
    deployment = build_deployment(prediction, model, hardware.hardware_id)
    return hardware, prediction, deployment


def build_deployment(prediction, model, hardware_id):
    if prediction["deployment"] != "colocated":
        raise _AlignmentUnsupported(
            "compact representative-step alignment currently requires colocated deployment"
        )
    tp = prediction["tensor_parallel_size"]
    pp = prediction["pipeline_parallel_size"]
    ep = prediction["expert_parallel_size"]
    parallel = ParallelConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        enable_expert_parallel=ep > 1,
    )
    if parallel.expert_parallel_size != ep:
        raise ValueError("alignment parallel config disagrees with expert_parallel_size")
    collectives = _tensor_collective_configs(
        "main",
        tp,
        hardware_id,
        prediction["tensor_collective_backend"],
        prediction["tensor_all_reduce_profile_sha256"],
        prediction["tensor_all_gather_profile_sha256"],
    )
    config = DeploymentConfig(
        "colocated",
        (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                parallel,
                collectives=collectives,
            ),
        ),
        router=RouterConfig(
            prediction["router_resource_id"],
            prediction["router_dispatch_service_time_s"],
        ),
    )
    return build_colocated_plan(
        config,
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def _alignment_routing(sample, signatures, model, step):
    if model.num_experts == 0:
        return ExpertRoutingInput.not_applicable()
    counts_by_rank = {}
    for group in sample.operators:
        counts_by_layer = {}
        for event in group["events"]:
            signature = _mapping(
                signatures[event["signature_id"]], "operator signature"
            )
            if signature["operation"] != "moe":
                continue
            layer = event["layer"]
            if layer is None:
                raise _AlignmentUnsupported("observed MoE event has no layer index")
            attributes = _mapping(signature["attrs"], "operator signature attrs")
            shape = _mapping(attributes["shape"], "operator signature shape")
            if "tok_per_expert" in attributes:
                raw_counts = attributes["tok_per_expert"]
            elif "tok_per_expert" in shape:
                raw_counts = shape["tok_per_expert"]
            else:
                raise _AlignmentUnsupported(
                    "representative MoE operator lacks observed tok_per_expert"
                )
            counts = tuple(raw_counts)
            if layer in counts_by_layer and counts_by_layer[layer] != counts:
                raise _AlignmentUnsupported(
                    f"representative MoE layer {layer} has conflicting histograms"
                )
            counts_by_layer[layer] = counts
        counts_by_rank[group["rank"]] = counts_by_layer
    expected_layers = {
        layer for layer in range(model.num_hidden_layers)
        if model.layer_uses_moe(layer)
    }
    for rank in sample.rank_ids:
        if rank not in counts_by_rank:
            raise _AlignmentUnsupported(
                f"representative MoE routing lacks rank {rank}"
            )
        counts = counts_by_rank[rank]
        if set(counts) != expected_layers:
            missing = sorted(expected_layers - set(counts))
            extra = sorted(set(counts) - expected_layers)
            raise _AlignmentUnsupported(
                "representative MoE routing layer coverage mismatch: "
                f"rank={rank}, missing={missing}, extra={extra}"
            )
    counts_by_layer = counts_by_rank[sample.rank_ids[0]]
    for rank in sample.rank_ids[1:]:
        if counts_by_rank[rank] != counts_by_layer:
            raise _AlignmentUnsupported(
                "representative MoE routing histograms disagree across ranks"
            )
    evidence_sha = _canonical_sha256({
        "schedule_step_sha256": schedule_step_sha256(step),
        "counts_by_layer": counts_by_layer,
    })
    return _ObservedAlignmentRouting(counts_by_layer, evidence_sha)


def _validate_phase(step: ScheduleStep, phase: str) -> None:
    phases = {request.phase for request in step.requests}
    expected = (
        "mixed" if len(phases) > 1
        else next(iter(phases))
    )
    if expected != phase:
        raise ValueError(
            f"alignment scheduler phase mismatch: expected {phase}, reconstructed {expected}"
        )


def _observed_operator_sequences(sample, signatures, ranks, model):
    groups = {group["rank"]: group for group in sample.operators}
    if set(groups) != set(ranks):
        raise ValueError("observed operator groups do not cover exact ranks")
    output = {}
    for rank in ranks:
        sequence = []
        for event in groups[rank]["events"]:
            signature = _mapping(
                signatures[event["signature_id"]], "operator signature"
            )
            projected = _project_observed_operator_records(
                signature, rank_count=len(ranks), model=model
            )
            if signature["operation"] == "moe" and sequence:
                previous = sequence[-1]
                expected_bytes = _moe_activation_bytes(signature)
                if (
                    previous["operation"] == "all_reduce"
                    and previous["semantic"] == {"bytes": expected_bytes}
                ):
                    # The legacy FusedMoE post-hook fires after the internal
                    # naive-EP all-reduce, while its record represents the
                    # complete router+experts call. Restore logical order.
                    reduction = sequence.pop()
                    sequence.extend(projected)
                    sequence.append(reduction)
                    continue
            sequence.extend(projected)
        output[rank] = tuple(sequence)
    return output


def _project_observed_operator_records(signature, *, rank_count, model):
    """Expand composite legacy hooks into canonical logical operations."""

    operation = signature["operation"]
    attrs = _mapping(signature["attrs"], "operator signature attrs")
    shape = _mapping(attrs["shape"], "operator signature shape")
    projected = _project_observed_operator(signature, rank_count=rank_count)
    if projected is None:
        return ()
    if operation == "moe":
        values = _shape_list(shape, "input_shape")
        tokens = shape["batch"] * shape["seq_len"]
        if _product(values[:-1]) != tokens:
            raise _AlignmentUnsupported(
                "observed MoE input shape disagrees with its token count"
            )
        histogram = _sequence(shape["tok_per_expert"], "observed MoE histogram")
        if len(histogram) != model.num_experts:
            raise _AlignmentUnsupported(
                "observed MoE histogram width disagrees with model topology"
            )
        if values[-1] != model.hidden_size:
            raise _AlignmentUnsupported(
                "observed MoE hidden width disagrees with model topology"
            )
        router = _record(
            "linear",
            {
                "b": 1,
                "m": tokens,
                "n": len(histogram),
                "k": values[-1],
            },
            projected["dtype"],
        )
        return (router, projected)
    if operation == "linear" and attrs["variant"] == "lm_head" and rank_count > 1:
        semantic = dict(projected["semantic"])
        if model.vocab_size % rank_count != 0:
            raise _AlignmentUnsupported(
                "model vocabulary width is not divisible by TP rank count"
            )
        local_width = model.vocab_size // rank_count
        if semantic["n"] not in (local_width, model.vocab_size):
            raise _AlignmentUnsupported(
                "observed LM-head width is neither global nor TP rank-local"
            )
        semantic["n"] = local_width
        local_linear = {**projected, "semantic": semantic}
        all_gather = _record(
            "all_gather",
            {
                "bytes": (
                    semantic["b"]
                    * semantic["m"]
                    * local_width
                    * _floating_dtype_bytes(attrs["dtype"])
                )
            },
            None,
        )
        return (local_linear, all_gather)
    return (projected,)


def _moe_activation_bytes(signature):
    attrs = _mapping(signature["attrs"], "operator signature attrs")
    shape = _mapping(attrs["shape"], "operator signature shape")
    values = _shape_list(shape, "input_shape")
    return _product(values) * _floating_dtype_bytes(attrs["dtype"])


def _project_observed_operator(signature, *, rank_count):
    operation = signature["operation"]
    attrs = _mapping(signature["attrs"], "operator signature attrs")
    shape = _mapping(attrs["shape"], "operator signature shape")
    dtype = _dtype(attrs["dtype"])
    variant = attrs["variant"]
    if operation == "all_reduce" and rank_count == 1:
        return None
    if operation == "linear":
        semantic = {name: shape[name.upper()] for name in ("b", "m", "n", "k")}
        return _record("linear", semantic, dtype)
    if operation in ("rms_norm", "fused_add_rms_norm"):
        values = _shape_list(shape, "input_shape")
        return _record(
            operation,
            {"rows": _product(values[:-1]), "hidden": values[-1]},
            dtype,
        )
    if operation == "activation":
        values = _shape_list(shape, "input_shape")
        if values[-1] % 2 != 0:
            raise _AlignmentUnsupported(
                "observed gated activation input width must be even"
            )
        return _record(
            "activation",
            {
                "rows": _product(values[:-1]),
                "output_hidden": values[-1] // 2,
            },
            dtype,
        )
    if operation == "rotary":
        return _record("rotary", {"tokens": shape["seq_len"]}, dtype)
    if operation == "attention":
        return _record("attention", {"tokens": shape["seq_len"]}, dtype)
    if operation == "moe":
        return _record(
            "moe", {"tokens": shape["batch"] * shape["seq_len"]}, dtype
        )
    if operation in ("all_reduce", "all_gather", "all_to_all"):
        return _record(operation, {"bytes": shape["msg_bytes"]}, None)
    if operation == "elementwise":
        values = _shape_list(shape, "input_shape")
        return _record("elementwise", {"elements": _product(values)}, dtype)
    raise _AlignmentUnsupported(
        f"operator operation {operation!r} has no canonical comparison projection"
    )


def _is_unobserved_logits_layout(node):
    if (node.kind is not WorkloadKind.VECTOR
            or node.variant not in ("vllm_logits_row_select", "vllm_logits_gather_reorder")
            or "source_component" not in node.attributes or "model_execution_version" not in node.attributes):
        return False
    sources = {"vllm_0_11_symbolic_logits_boundary": "vllm_0_11_0",
        "vllm_0_24_symbolic_logits_boundary": "vllm_0_24_0",
        "vllm_0_30_symbolic_logits_boundary": "vllm_0_30_0",
        "vllm_0_27_1_symbolic_logits_boundary": "vllm_0_27_1"}
    source = node.attributes["source_component"]
    return source in sources and node.attributes["model_execution_version"] == sources[source]


def _unobserved_logits_index_casts(graph):
    """Identify only the typed index producer of a bound logits gather.

    The capture installs module hooks, while advanced indexing's internal
    INT32-to-INT64 materialization is an ATen operation outside those hooks.
    Keep it in the workload/lowering and report its exact logical traffic;
    only the module-interface sequence omits this internal call.
    """
    selected = {}
    for gather in graph.nodes:
        if (not _is_unobserved_logits_layout(gather)
                or gather.variant != "vllm_logits_row_select"
                or gather.attributes["model_execution_version"] not in ("vllm_0_11_0", "vllm_0_24_0")
                or gather.attributes["index_source"] != "query_start_loc_int32_to_int64"):
            continue
        index_view = gather.attributes["symbolic_tensor_inputs"][1]
        for predecessor in graph.predecessors(gather.node_id):
            cast = graph.node(predecessor)
            if (cast.kind is WorkloadKind.ELEMENTWISE and cast.variant == "elementwise"
                    and cast.dtype == "int32" and dict(cast.shape) == {"elements": gather.shape["rows"]}
                    and cast.attributes["operation"] == "cast_int32_int64"
                    and cast.attributes["reads"] == cast.attributes["writes"] == 1
                    and tuple(cast.attributes["symbolic_tensor_outputs"]) == (index_view,)):
                selected[cast.node_id] = cast
    return tuple(selected.values())


def _project_workload_graph(graph, ranks, *, node_order_by_rank):
    if set(node_order_by_rank) != set(ranks):
        raise ValueError("source call order must cover the exact observed ranks")
    output = {}
    internal_casts = {node.node_id for node in _unobserved_logits_index_casts(graph)}
    for rank in ranks:
        ordered = tuple(node_order_by_rank[rank])
        expected = {node.node_id for node in graph.nodes
            if (rank in node.attributes["participants"] if node.kind is WorkloadKind.COLLECTIVE else node.rank == rank)}
        if len(set(ordered)) != len(ordered) or set(ordered) != expected:
            raise ValueError("source call order must cover each rank's operators exactly once")
        positions = {name: index for index, name in enumerate(ordered)}
        if any(positions[edge.source] >= positions[edge.destination] for edge in graph.edges
                if edge.source in positions and edge.destination in positions):
            raise ValueError("source call order violates its rank-local dependency graph")
        output[rank] = tuple(record for name in ordered
            if name not in internal_casts and (record := _project_workload_node(graph.node(name))) is not None)
    return output


def _project_workload_node(node):
    if (node.kind is WorkloadKind.VECTOR and node.variant == "moe_topk"
            and "source_component" in node.attributes and node.attributes["source_component"] == "symbolic_softmax_topk"
            and node.attributes["source_program"] == "vllm_0_30_bf16_softmax_topk"
            and node.attributes["mathematical_program"] == "bf16_softmax_topk_flops_bytes_v1"):
        # The observed FusedMoE module expands into its gate plus expert call.
        # Its internal routing has no separate captured module event. Keep
        # the mathematical node and its actual source identity in the graph.
        return None
    if "rope_composite_position" in node.attributes:
        # vLLM's operator capture hooks the rotary module, so a rope that the
        # backend lowers to RotaryEmbedding.forward_native still appears as a
        # single observed `rotary` record.  Collapse the predicted composite
        # onto its first member so both sides compare at capture granularity.
        if node.attributes["rope_composite_position"] != 0:
            return None
        return _record(
            "rotary",
            {"tokens": node.attributes["rope_composite_tokens"]},
            _dtype(node.dtype),
        )
    if node.kind in (WorkloadKind.GATHER, WorkloadKind.KV_UPDATE):
        return None
    if _is_unobserved_logits_layout(node):
        # Captures observe the head/gather API, not these internal layouts.
        return None
    if node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER):
        return _record("linear", _shape(node.shape, ("b", "m", "n", "k")), _dtype(node.dtype))
    if node.kind is WorkloadKind.NORM:
        if node.variant == "add_rmsnorm":
            operation = "fused_add_rms_norm"
        elif node.variant in (None, "norm", "rmsnorm"):
            operation = "rms_norm"
        else:
            raise _AlignmentUnsupported(
                f"norm variant {node.variant!r} has no canonical projection"
            )
        return _record(operation, _shape(node.shape, ("rows", "hidden")), _dtype(node.dtype))
    if node.kind is WorkloadKind.ACTIVATION:
        return _record(
            "activation",
            {
                "rows": node.shape["tok"],
                "output_hidden": node.shape["intermediate"],
            },
            _dtype(node.dtype),
        )
    if node.kind is WorkloadKind.ROTARY:
        return _record("rotary", {"tokens": node.shape["seq_len"]}, _dtype(node.dtype))
    if node.kind in (WorkloadKind.FLASH_ATTN, WorkloadKind.EAGER_ATTN):
        lengths = (node.attributes["model_input_q_lens"]
            if "model_input_q_lens" in node.attributes else node.attributes["q_lens"])
        return _record("attention", {"tokens": sum(lengths)}, _dtype(node.dtype))
    if node.kind is WorkloadKind.FUSED_MOE:
        return _record("moe", {"tokens": node.shape["tokens"]}, _dtype(node.dtype))
    if node.kind is WorkloadKind.ELEMENTWISE:
        return _record("elementwise", {"elements": node.shape["elements"]}, _dtype(node.dtype))
    if node.kind is WorkloadKind.COLLECTIVE:
        return _record(node.attributes["operation"], {"bytes": node.attributes["bytes"]}, None)
    raise _AlignmentUnsupported(
        f"workload kind {node.kind.value!r} has no canonical operator projection"
    )


def _observed_launch_sequences(sample, signatures, ranks, hardware_id):
    groups = {group["rank"]: group for group in sample.launches}
    if set(groups) != set(ranks):
        raise ValueError("observed launch groups do not cover exact ranks")
    registry = ExclusionRegistry.load()
    output = {}
    for rank in ranks:
        sequence = []
        for event in groups[rank]["events"]:
            signature = _mapping(
                signatures[event["signature_id"]], "launch signature"
            )
            name = signature["kernel_name"]
            if registry.classify(name, scope="all") is not None:
                continue
            if _is_sampler_launch(name):
                continue
            category = _canonical_launch_category(
                name, signature["op_family"]
            )
            geometry = _mapping(signature["geometry"], "launch geometry")
            sequence.append(_observed_launch_record(
                category, geometry, hardware_id
            ))
        output[rank] = tuple(sequence)
    return output


def _canonical_launch_category(name, op_family):
    category = _tiered_gate_category(name)
    if category == "GEMV":
        return "GEMM"
    if category != "OTHER":
        return category
    family = str(op_family).lower()
    if "matmul" in family or "gemm" in family:
        return "GEMM"
    if "attention" in family or "flash" in family:
        return "FLASH_ATTN"
    if "silu" in family or "swiglu" in family:
        return "ACTIVATION"
    raise _AlignmentUnsupported(
        f"launch {name!r} / family {op_family!r} has no semantic projection"
    )


def _observed_launch_record(category, geometry, hardware_id):
    kind = geometry["kind"]
    if kind == "cuda_dim3":
        return _record(category, {
            "grid": tuple(geometry["grid"]),
            "block": tuple(geometry["block"]),
        }, None)
    if kind == "ascend_block_dim" and hardware_id == "ascend_910b":
        return _record(category, {"grid_x": geometry["block_dim"]}, None)
    raise _AlignmentUnsupported(
        f"launch geometry {kind!r} cannot be compared for {hardware_id}"
    )


def _project_kernel_plan(plan, graph, ranks):
    output = {rank: [] for rank in ranks}
    internal_casts = {node.node_id for node in _unobserved_logits_index_casts(graph)}
    for call in plan.topological_order():
        spec = call.kernel_spec
        if all(node_id in internal_casts or _is_unobserved_logits_layout(graph.node(node_id)) for node_id in call.source_node_ids):
            continue
        category = _kernel_category(spec)
        source_kinds = {
            graph.node(node_id).kind for node_id in call.source_node_ids
        }
        if source_kinds == {WorkloadKind.GATHER}:
            continue
        if category in ("SPLITK", "COLLECTIVE"):
            continue
        if spec.kind is KernelKind.FLASH_ATTN and spec.variant == "combine":
            continue
        if spec.kind is KernelKind.COMMUNICATION:
            call_ranks = tuple(spec.transfer.participants)
        elif call.rank is not None:
            call_ranks = (call.rank,)
        else:
            raise ValueError("non-communication kernel call has no rank")
        if spec.launch is None:
            if spec.execution_support == "unsupported":
                raise _AlignmentUnsupported(f"{spec.kind.value}/{spec.variant} has no bound launch geometry")
            geometry = {}
        elif spec.hardware_id == "ascend_910b":
            geometry = {"grid_x": spec.launch.grid[0]}
        else:
            geometry = {
                "grid": tuple(spec.launch.grid),
                "block": tuple(spec.launch.block),
            }
        projected = _record(category, geometry, None)
        for rank in call_ranks:
            if rank not in output:
                raise ValueError(f"kernel plan projection references unexpected rank {rank}")
            output[rank].append(projected)
    return {rank: tuple(values) for rank, values in output.items()}


def _is_sampler_launch(name):
    lowered = name.lower()
    return "argmax" in lowered or "vectorized_gather_kernel" in lowered


def _kernel_category(spec):
    if spec.kind is KernelKind.GEMM:
        return "GEMM"
    if spec.kind in (KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN):
        return "FLASH_ATTN"
    if spec.kind in (KernelKind.FUSED_MOE, KernelKind.UNFUSED_MOE):
        return "FUSEDMOE"
    if spec.kind is KernelKind.COMMUNICATION:
        return "COLLECTIVE"
    if spec.kind is KernelKind.KV_CACHE:
        if spec.variant == "paged_kv_cache":
            return "KV_CACHE"
        raise _AlignmentUnsupported(
            f"KernelSpec {spec.kind.value}/{spec.variant} has no launch "
            "projection")
    if spec.kind is KernelKind.VECTOR:
        variants = {
            "rmsnorm": "NORM", "add_rmsnorm": "NORM",
            "rotary": "ROTARY", "rope": "ROTARY",
            "activation": "ACTIVATION", "silu": "ACTIVATION",
            "reduce": "REDUCE", "gather": "GATHER",
            "elementwise": "ELEMENTWISE",
            "elementwise_pytorch_vec4": "ELEMENTWISE",
            "gemm_reduce": "SPLITK", "gemm_elementwise": "ELEMENTWISE",
        }
        if spec.variant in variants:
            return variants[spec.variant]
        if spec.variant.startswith("moe_"):
            return "FUSEDMOE"
    raise _AlignmentUnsupported(
        f"KernelSpec {spec.kind.value}/{spec.variant} has no launch projection"
    )


def _compare_sequence(observed, predicted):
    mismatch = None
    for index, (left, right) in enumerate(zip(observed, predicted)):
        if left != right:
            mismatch = {"index": index, "observed": left, "predicted": right}
            break
    if mismatch is None and len(observed) != len(predicted):
        index = min(len(observed), len(predicted))
        mismatch = {
            "index": index,
            "observed": observed[index] if index < len(observed) else None,
            "predicted": predicted[index] if index < len(predicted) else None,
        }
    return {
        "status": "matched" if mismatch is None else "mismatched",
        "observed_count": len(observed),
        "predicted_count": len(predicted),
        "observed_sha256": _canonical_sha256(observed),
        "predicted_sha256": _canonical_sha256(predicted),
        "first_difference": mismatch,
    }


def _try_launch_projection(project, side):
    try:
        return project(), None
    except _AlignmentUnsupported as error:
        return None, f"{side} launch projection is unsupported: {error}"
    except (KeyError, NotImplementedError) as error:
        return None, f"{side} launch lowering is unsupported: {type(error).__name__}: {error}"


def _unsupported_comparison(reason):
    return {
        "status": "unsupported",
        "reason": reason,
        "observed_count": None,
        "predicted_count": None,
        "observed_sha256": None,
        "predicted_sha256": None,
        "first_difference": None,
    }


def _aggregate_status(*statuses):
    if not statuses:
        raise ValueError("cannot aggregate an empty alignment status set")
    if any(status == "unsupported" for status in statuses):
        return "unsupported"
    if any(status == "mismatched" for status in statuses):
        return "mismatched"
    if all(status == "matched" for status in statuses):
        return "matched"
    raise ValueError(f"unknown alignment status set: {statuses}")


def _record(operation, semantic, dtype):
    return {
        "operation": operation,
        "semantic": dict(semantic),
        "dtype": dtype,
    }


def _shape(value, names):
    return {name: value[name] for name in names}


def _shape_list(shape, name):
    value = shape[name]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"observed operator {name} must be a non-empty sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"observed operator {name} must contain positive integers")
    return tuple(value)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _dtype(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    aliases = {
        "bfloat16": "float16_storage",
        "bf16": "float16_storage",
        "float16": "float16_storage",
        "fp16": "float16_storage",
    }
    return aliases[normalized] if normalized in aliases else normalized


def _floating_dtype_bytes(value):
    normalized = str(value).strip().lower()
    widths = {
        "bfloat16": 2,
        "bf16": 2,
        "float16": 2,
        "fp16": 2,
        "float32": 4,
        "fp32": 4,
    }
    if normalized not in widths:
        raise _AlignmentUnsupported(
            f"operator dtype {value!r} has no storage-width projection"
        )
    return widths[normalized]


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()




def _sequence(value, label):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_fields(value, expected, label):
    if set(value) != expected:
        raise ValueError(
            f"{label} fields must be exactly {sorted(expected)}, got {sorted(value)}"
        )
