"""Independent Serving-chain checkpoints and target-model support reports."""

from dataclasses import dataclass
import math
from typing import Mapping

from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.metrics.serving_metrics import ServingMetrics
from src.serving.request_gen.request_stream import (
    RequestStream,
    request_stream_sha256,
)
from src.serving.schedule.schedule_step import (
    BackendStep,
    schedule_step_sha256,
)
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.workload_graph import (
    WorkloadGraph,
    WorkloadKind,
)


class ServingValidationError(ValueError):
    """Raised when a modeled chain disagrees with its independent reference."""


@dataclass(frozen=True)
class OperatorCheckpoint:
    node_id: str
    kind: WorkloadKind
    shape: Mapping[str, int]
    predecessors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("operator checkpoint node_id must be non-empty")
        object.__setattr__(self, "kind", WorkloadKind(self.kind))
        if not isinstance(self.shape, Mapping):
            raise TypeError("operator checkpoint shape must be a mapping")
        shape = dict(self.shape)
        if any(
                not isinstance(name, str) or not name
                or isinstance(value, bool) or not isinstance(value, int)
                or value < 0
                for name, value in shape.items()):
            raise ValueError(
                "operator checkpoint shape must contain non-negative integer "
                "dimensions")
        object.__setattr__(self, "shape", shape)
        if not isinstance(self.predecessors, tuple) or any(
                not isinstance(value, str) or not value
                for value in self.predecessors):
            raise TypeError(
                "operator checkpoint predecessors must be non-empty strings")


@dataclass(frozen=True)
class KernelCheckpoint:
    call_id: str
    source_node_ids: tuple[str, ...]
    kind: KernelKind
    shape: Mapping[str, int]
    launch_algorithm: str
    launch_grid: tuple[int, int, int]
    launch_block: tuple[int, int, int]
    execution_variant: str

    def __post_init__(self) -> None:
        for name in ("call_id", "launch_algorithm", "execution_variant"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"kernel checkpoint {name} must be non-empty")
        if not isinstance(self.source_node_ids, tuple) or not (
                self.source_node_ids) or any(
                not isinstance(value, str) or not value
                for value in self.source_node_ids):
            raise TypeError(
                "kernel checkpoint source_node_ids must be non-empty strings")
        object.__setattr__(self, "kind", KernelKind.parse(self.kind))
        if not isinstance(self.shape, Mapping):
            raise TypeError("kernel checkpoint shape must be a mapping")
        shape = dict(self.shape)
        if any(
                not isinstance(name, str) or not name
                or isinstance(value, bool) or not isinstance(value, int)
                or value < 0
                for name, value in shape.items()):
            raise ValueError(
                "kernel checkpoint shape must contain non-negative integer "
                "dimensions")
        object.__setattr__(self, "shape", shape)
        for name in ("launch_grid", "launch_block"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or len(value) != 3
                    or any(isinstance(item, bool)
                           or not isinstance(item, int)
                           or item <= 0 for item in value)):
                raise ValueError(
                    f"kernel checkpoint {name} must contain three positive "
                    "integers")


@dataclass(frozen=True)
class RequestMetricCheckpoint:
    request_id: str
    ttft_s: float | None
    tpot_s: float | None
    latency_s: float
    absolute_tolerance_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError(
                "request metric checkpoint request_id must be non-empty")
        for name in (
                "ttft_s", "tpot_s", "latency_s", "absolute_tolerance_s"):
            value = getattr(self, name)
            if value is None and name in ("ttft_s", "tpot_s"):
                continue
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0):
                raise ValueError(
                    f"request metric checkpoint {name} must be finite and "
                    "non-negative")


@dataclass(frozen=True)
class ServingChainReference:
    reference_id: str
    request_stream_sha256: str
    schedule_step_sha256: tuple[str, ...]
    operator_sequence: tuple[str, ...]
    kernel_sequence: tuple[str, ...]
    operators: tuple[OperatorCheckpoint, ...]
    kernels: tuple[KernelCheckpoint, ...]
    request_metrics: tuple[RequestMetricCheckpoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or not self.reference_id:
            raise ValueError("Serving reference_id must be non-empty")
        if (not isinstance(self.request_stream_sha256, str)
                or len(self.request_stream_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.request_stream_sha256)):
            raise ValueError(
                "Serving reference request stream identity must be a SHA256")
        if not isinstance(self.schedule_step_sha256, tuple) or not (
                self.schedule_step_sha256):
            raise ValueError(
                "Serving reference requires Scheduler step identities")
        for digest in self.schedule_step_sha256:
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "Serving reference Scheduler identities must be SHA256")
        for name, values in (
                ("operator_sequence", self.operator_sequence),
                ("kernel_sequence", self.kernel_sequence)):
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(value, str) or not value
                           for value in values)):
                raise TypeError(
                    f"Serving reference {name} must contain non-empty IDs")
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Serving reference {name} IDs must be unique")
        typed_collections = (
            ("operators", self.operators, OperatorCheckpoint),
            ("kernels", self.kernels, KernelCheckpoint),
            ("request_metrics", self.request_metrics,
             RequestMetricCheckpoint),
        )
        for name, values, expected_type in typed_collections:
            if not isinstance(values, tuple) or not values or any(
                    not isinstance(value, expected_type) for value in values):
                raise TypeError(
                    f"Serving reference {name} must contain "
                    f"{expected_type.__name__} values")


def _unique_by_id(values, *, id_name: str, collection_name: str):
    indexed = {}
    for value in values:
        identifier = getattr(value, id_name)
        if identifier in indexed:
            raise ServingValidationError(
                f"duplicate {collection_name} ID: {identifier!r}")
        indexed[identifier] = value
    return indexed


def _metric_matches(
    expected: float | None,
    actual: float | None,
    tolerance: float,
) -> tuple[bool, float | None]:
    if expected is None or actual is None:
        return expected is actual, None
    delta = actual - expected
    return abs(delta) <= tolerance, delta


def compare_serving_chain(
    reference: ServingChainReference,
    requests: RequestStream,
    backend_steps: tuple[BackendStep, ...],
    workloads: tuple[WorkloadGraph, ...],
    kernel_plans: tuple[KernelPlan, ...],
    metrics: ServingMetrics,
) -> dict[str, object]:
    """Compare selected chain facts without evaluating predictor formulas."""

    if not isinstance(reference, ServingChainReference):
        raise TypeError("Serving comparison requires ServingChainReference")
    if not isinstance(requests, RequestStream):
        raise TypeError("Serving comparison requires RequestStream")
    for name, values, expected_type in (
            ("backend_steps", backend_steps, BackendStep),
            ("workloads", workloads, WorkloadGraph),
            ("kernel_plans", kernel_plans, KernelPlan)):
        if not isinstance(values, tuple) or not values or any(
                not isinstance(value, expected_type) for value in values):
            raise TypeError(
                f"Serving comparison {name} must contain "
                f"{expected_type.__name__} values")
    if len(backend_steps) != len(workloads) or len(workloads) != len(
            kernel_plans):
        raise ValueError(
            "Serving comparison Scheduler, workload, and KernelPlan counts "
            "must align")
    if not isinstance(metrics, ServingMetrics):
        raise TypeError("Serving comparison requires ServingMetrics")

    actual_request_sha256 = request_stream_sha256(requests)
    if actual_request_sha256 != reference.request_stream_sha256:
        raise ServingValidationError(
            "request stream disagrees with the Serving reference")
    actual_schedule_sha256 = tuple(
        schedule_step_sha256(step.schedule) for step in backend_steps)
    if actual_schedule_sha256 != reference.schedule_step_sha256:
        raise ServingValidationError(
            "Scheduler steps disagree with the Serving reference")

    nodes = tuple(
        node for workload in workloads for node in workload.nodes)
    actual_operator_sequence = tuple(node.node_id for node in nodes)
    if actual_operator_sequence != reference.operator_sequence:
        raise ServingValidationError(
            "logical operator sequence disagrees with the Serving reference")
    nodes_by_id = _unique_by_id(
        nodes, id_name="node_id", collection_name="workload node")
    graphs_by_node_id = {}
    for workload in workloads:
        for node in workload.nodes:
            graphs_by_node_id[node.node_id] = workload
    operator_rows = []
    for checkpoint in reference.operators:
        if checkpoint.node_id not in nodes_by_id:
            raise ServingValidationError(
                f"missing reference operator: {checkpoint.node_id!r}")
        node = nodes_by_id[checkpoint.node_id]
        workload = graphs_by_node_id[checkpoint.node_id]
        actual = (
            node.kind,
            dict(node.shape),
            workload.predecessors(node.node_id),
        )
        expected = (
            checkpoint.kind,
            dict(checkpoint.shape),
            checkpoint.predecessors,
        )
        if actual != expected:
            raise ServingValidationError(
                f"operator checkpoint mismatch for {checkpoint.node_id!r}: "
                f"actual={actual!r}, expected={expected!r}")
        operator_rows.append({
            "node_id": checkpoint.node_id,
            "kind": checkpoint.kind.value,
            "shape": dict(checkpoint.shape),
            "predecessors": checkpoint.predecessors,
        })

    calls = tuple(call for plan in kernel_plans for call in plan.calls)
    actual_kernel_sequence = tuple(call.call_id for call in calls)
    if actual_kernel_sequence != reference.kernel_sequence:
        raise ServingValidationError(
            "KernelCall sequence disagrees with the Serving reference")
    calls_by_id = _unique_by_id(
        calls, id_name="call_id", collection_name="KernelCall")
    kernel_rows = []
    for checkpoint in reference.kernels:
        if checkpoint.call_id not in calls_by_id:
            raise ServingValidationError(
                f"missing reference KernelCall: {checkpoint.call_id!r}")
        call = calls_by_id[checkpoint.call_id]
        launch = call.kernel_spec.launch
        if launch is None:
            raise ServingValidationError(
                f"reference KernelCall lacks a launch: {checkpoint.call_id!r}")
        actual = (
            call.source_node_ids,
            call.kernel_spec.kind,
            dict(call.kernel_spec.shape),
            launch.algorithm,
            launch.grid,
            launch.block,
            launch.execution_variant,
        )
        expected = (
            checkpoint.source_node_ids,
            checkpoint.kind,
            dict(checkpoint.shape),
            checkpoint.launch_algorithm,
            checkpoint.launch_grid,
            checkpoint.launch_block,
            checkpoint.execution_variant,
        )
        if actual != expected:
            raise ServingValidationError(
                f"kernel checkpoint mismatch for {checkpoint.call_id!r}: "
                f"actual={actual!r}, expected={expected!r}")
        kernel_rows.append({
            "call_id": checkpoint.call_id,
            "source_node_ids": checkpoint.source_node_ids,
            "kind": checkpoint.kind.value,
            "shape": dict(checkpoint.shape),
            "launch_algorithm": checkpoint.launch_algorithm,
            "launch_grid": checkpoint.launch_grid,
            "launch_block": checkpoint.launch_block,
            "execution_variant": checkpoint.execution_variant,
        })

    metric_rows = []
    reference_metric_ids = tuple(
        checkpoint.request_id for checkpoint in reference.request_metrics)
    if len(reference_metric_ids) != len(set(reference_metric_ids)):
        raise ServingValidationError(
            "Serving reference contains duplicate request metric IDs")
    if set(reference_metric_ids) != set(metrics.requests):
        raise ServingValidationError(
            "request metrics disagree with the Serving reference ID set")
    for checkpoint in reference.request_metrics:
        actual_metric = metrics.requests[checkpoint.request_id]
        comparisons = {}
        for name in ("ttft_s", "tpot_s", "latency_s"):
            matches, delta = _metric_matches(
                getattr(checkpoint, name),
                getattr(actual_metric, name),
                checkpoint.absolute_tolerance_s,
            )
            if not matches:
                raise ServingValidationError(
                    f"request metric {name} disagrees for "
                    f"{checkpoint.request_id!r}")
            comparisons[name] = {
                "reference": getattr(checkpoint, name),
                "actual": getattr(actual_metric, name),
                "delta_s": delta,
            }
        metric_rows.append({
            "request_id": checkpoint.request_id,
            "absolute_tolerance_s": checkpoint.absolute_tolerance_s,
            "metrics": comparisons,
        })

    return {
        "schema_version": 1,
        "reference_id": reference.reference_id,
        "request_stream_sha256": actual_request_sha256,
        "schedule_step_sha256": actual_schedule_sha256,
        "operator_sequence": actual_operator_sequence,
        "kernel_sequence": actual_kernel_sequence,
        "operator_checkpoints": tuple(operator_rows),
        "kernel_checkpoints": tuple(kernel_rows),
        "request_metric_checkpoints": tuple(metric_rows),
        "validation_status": "matched",
    }


def target_model_support_matrix(model: ModelConfig) -> dict[str, object]:
    """Describe distinct semantic and execution support stages."""

    if not isinstance(model, ModelConfig):
        raise TypeError("target support matrix requires ModelConfig")
    identity = (
        model.model_type,
        model.model_id,
        model.model_revision,
        model.config_sha256,
        model.attention_layout,
    )
    kimi_identity = (
        "kimi_k2",
        "moonshotai/Kimi-K2.6",
        "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        "03af24788bd7788c65db1fcb8326c5f5c5270aab717c442df67ba9db01041b3d",
        "mla",
    )
    glm_identity = (
        "glm_moe_dsa",
        "zai-org/GLM-5.2",
        "b4734de4facf877f85769a911abafc5283eab3d9",
        "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a",
        "dsa_mla",
    )
    if identity == kimi_identity:
        representative_layers = (
            ("dense", 0, "mla"),
            ("moe", 1, "mla_int4_routed_experts"),
        )
        stages = (
            ("configuration_parse", "supported", "fixed official text_config"),
            (
                "scheduler_step",
                "supported_scheduler_only",
                "pinned vLLM 0.11 CPU-only Scheduler drives the colocated "
                "TP=PP=1 production-trace path with MLA KV accounting",
            ),
            (
                "logical_workload",
                "supported_static",
                "MLA, compressed KV, Dense/MoE, shared experts, and RoPE",
            ),
            (
                "kernel_spec",
                "supported_structural",
                "typed MLA attention/KV, batched GEMM, RoPE, router, and "
                "INT4 routed-expert specs; unavailable launches are explicit",
            ),
            (
                "hardware_launch",
                "unsupported",
                "no canonical full-model launch package",
            ),
            (
                "hardware_validation",
                "supported_transformers_tiny",
                "A800 BF16 prefill/decode control-flow smoke; vLLM blocked by "
                "its PyTorch 2.13 requirement",
            ),
        )
    elif identity == glm_identity:
        representative_layers = (
            ("dense", 0, "dsa_full_indexer"),
            ("moe", 3, "dsa_shared_indexer"),
        )
        stages = (
            ("configuration_parse", "supported", "fixed official config"),
            (
                "scheduler_step",
                "supported_scheduler_only",
                "pinned vLLM 0.11 CPU-only Scheduler drives the colocated "
                "TP=PP=1 production-trace path with exact aggregate DSA KV "
                "accounting",
            ),
            (
                "logical_workload",
                "supported_logical",
                "prefill/decode/mixed MLA/DSA, indexer reuse, KV, and MoE",
            ),
            (
                "kernel_spec",
                "supported_structural",
                "typed DSA indexer/cache/sparse-attention, MLA, router, and "
                "routed-expert specs; MTP remains outside this layer path",
            ),
            (
                "hardware_launch",
                "unsupported",
                "no canonical DSA launch package",
            ),
            (
                "hardware_validation",
                "supported_transformers_tiny",
                "A800 BF16 prefill/decode control-flow smoke; vLLM blocked by "
                "its PyTorch 2.13 requirement",
            ),
        )
    else:
        raise ValueError(
            "support matrix is defined only for the fixed Kimi-K2.6 and "
            "GLM-5.2 fixtures")
    return {
        "schema_version": 3,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "config_sha256": model.config_sha256,
        "stages": tuple({
            "stage": stage,
            "status": status,
            "boundary": boundary,
        } for stage, status, boundary in stages),
        "cases": tuple({
            "phase": phase,
            "layer_type": layer_type,
            "representative_layer": layer_index,
            "attention_variant": attention_variant,
            "validation_levels": (
                {
                    "level": "fixed_upstream_source",
                    "status": "supported",
                },
                {
                    "level": "production_shape_logical_workload",
                    "status": "supported",
                },
                {
                    "level": "tiny_transformers_cuda_control_flow",
                    "status": "supported",
                },
                {
                    "level": "kernel_lowering",
                    "status": "supported_structural",
                },
                {
                    "level": "kernel_prediction",
                    "status": "unsupported",
                },
            ),
        } for phase in ("prefill", "decode", "mixed")
          for layer_type, layer_index, attention_variant
          in representative_layers),
    }
