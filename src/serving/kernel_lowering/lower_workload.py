"""Explicit WorkloadNode-to-KernelSpec lowering rules."""

import dataclasses
from math import prod
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import fields
import pickle
import weakref

_SOURCE_LAYOUT_UNSET = object()
_CURRENT_GEMM_SOURCE_BINDINGS = ContextVar("current_gemm_source_bindings", default=None)
_CURRENT_DSPARK_SOURCE_GRAPH = ContextVar("current_dspark_source_graph", default=None)
from types import MappingProxyType
from typing import Callable, Mapping

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import (
    DTYPE_BYTES,
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.fused_moe_physical_specs import (
    build_fused_moe_physical_specs,
)
from src.kernel.predictors.fused_moe import (
    ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS,
    canonical_expert_axis,
)
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_WEIGHT_QUANTIZATION,
    PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
)
from src.kernel.launch_config.select_launch_config import (
    normalize_flash_attn_execution_attributes,
    select_fused_moe_recipe,
    select_flash_attn_copy_launches,
    select_flash_attn_forward_partition,
    select_flash_attn_output_fill_launches,
    select_flash_attn_setup_launches,
    select_block_fp8_input_quant_launch,
    select_dsv4_decode_split_counts,
    select_mhc_stage_launches,
    select_fused_moe_input_quant_launch,
    select_gemm_reducer_launch,
    select_launch_config,
    select_workload_source_launch_recipe,
    select_source_flash_attn_partial_launch,
)
from src.serving.deployment.deployment_config import (
    collective_ring_order,
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    pd_kv_transfer_rank_pairs,
)
from src.serving.workload.helper_launch import (
    RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
)
from src.serving.workload.runtime_inputs import (
    validate_expert_routing_provenance,
)
from src.serving.workload.workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode

from .heterogeneous_moe import (is_heterogeneous_moe, heterogeneous_moe_specs, validate_heterogeneous_transfer,
    validate_heterogeneous_pd_transfer, validate_heterogeneous_graph)

from .kernel_plan import (
    KernelBlockRepeat, KernelCall, KernelPlan, require_hardware)


_SINGLE_COMPUTE_KIND = {
    WorkloadKind.LINEAR: KernelKind.GEMM,
    WorkloadKind.LM_HEAD: KernelKind.GEMM,
    WorkloadKind.ROUTER: KernelKind.GEMM,
    WorkloadKind.FLASH_ATTN: KernelKind.FLASH_ATTN,
    WorkloadKind.NORM: KernelKind.VECTOR,
    WorkloadKind.ACTIVATION: KernelKind.VECTOR,
    WorkloadKind.ROTARY: KernelKind.VECTOR,
    WorkloadKind.ELEMENTWISE: KernelKind.VECTOR,
    WorkloadKind.REDUCE: KernelKind.VECTOR,
    WorkloadKind.GATHER: KernelKind.VECTOR,
    WorkloadKind.VECTOR: KernelKind.VECTOR,
    WorkloadKind.KV_UPDATE: KernelKind.KV_CACHE,
}

_COMMUNICATION_KINDS = {
    WorkloadKind.COLLECTIVE,
    WorkloadKind.P2P,
    WorkloadKind.PD_KV_TRANSFER,
    WorkloadKind.AF_ACTIVATION_TRANSFER,
}

_GRAPH_TOKEN_DOMAIN_FIELDS = (
    "graph_logical_tokens",
    "graph_model_input_tokens",
    "graph_runtime_mode",
)


def _graph_token_domain(node: WorkloadNode) -> tuple[int, int, str] | None:
    """Validate an operator's effective-work/captured-tensor row contract."""

    if node.kind is WorkloadKind.FLASH_ATTN:
        # FlashAttention has its more precise per-request contract below.
        return None
    present = tuple(
        name for name in _GRAPH_TOKEN_DOMAIN_FIELDS
        if name in node.attributes)
    if not present:
        return None
    if present != _GRAPH_TOKEN_DOMAIN_FIELDS:
        raise ValueError(
            f"workload node {node.node_id!r} has incomplete graph token "
            f"domains: present={present!r}")
    logical_tokens = node.attributes["graph_logical_tokens"]
    model_input_tokens = node.attributes["graph_model_input_tokens"]
    runtime_mode = node.attributes["graph_runtime_mode"]
    for name, value in (
            ("graph_logical_tokens", logical_tokens),
            ("graph_model_input_tokens", model_input_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if model_input_tokens < logical_tokens:
        raise ValueError("graph model-input tokens cannot shrink work")
    if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError(f"unknown graph runtime mode: {runtime_mode!r}")
    if runtime_mode == "NONE" and logical_tokens != model_input_tokens:
        raise ValueError("eager node cannot declare graph padding")
    return logical_tokens, model_input_tokens, runtime_mode


def _logical_prediction_shape(
    node: WorkloadNode,
    physical_shape: Mapping[str, int],
) -> tuple[dict[str, int], tuple[int, int, str] | None]:
    """Keep semantic work logical while preserving physical launch input."""

    domain = _graph_token_domain(node)
    if domain is None:
        return dict(physical_shape), None
    logical_tokens, model_input_tokens, _ = domain
    result = dict(physical_shape)
    token_dimensions = {
        KernelKind.GEMM: ("m",),
        KernelKind.VECTOR: ("rows", "seq_len", "tok", "element_count"),
        KernelKind.KV_CACHE: ("seq_len", "element_count"),
    }
    kind = _SINGLE_COMPUTE_KIND[node.kind]
    if kind not in token_dimensions:
        raise ValueError(
            f"graph token domains are unsupported for {kind.value}")
    changed = False
    for name in token_dimensions[kind]:
        if name not in result:
            continue
        value = result[name]
        if value % model_input_tokens:
            raise ValueError(
                f"{node.node_id!r} physical {name}={value} is not an "
                "integral graph-token multiple")
        result[name] = value // model_input_tokens * logical_tokens
        changed = True
    if not changed:
        raise ValueError(
            f"{node.node_id!r} graph token domain has no token-shaped "
            f"dimension in {sorted(result)!r}")
    return result, domain


def _kernel_graph_token_attributes(
    domain: tuple[int, int, str] | None,
    physical_shape: Mapping[str, int],
) -> dict[str, object]:
    if domain is None:
        return {}
    logical_tokens, model_input_tokens, runtime_mode = domain
    return {
        "graph_logical_tokens": logical_tokens,
        "graph_model_input_tokens": model_input_tokens,
        "graph_runtime_mode": runtime_mode,
        "model_input_shape": dict(physical_shape),
    }


def _derived_graph_token_attributes(
    parent: KernelSpec,
    physical_shape: Mapping[str, int],
) -> dict[str, object]:
    """Carry a parent's graph domain into a generated physical helper.

    Lowering expands one graph-attributed operation into source-defined
    metadata, copy, and reduction kernels.  Those helpers execute on a
    physical captured buffer, so their own shape remains physical; retaining
    the parent domain prevents the helper from silently becoming a
    single-token-domain KernelSpec.
    """

    if parent.kind is KernelKind.FLASH_ATTN:
        names = (
            "model_input_q_lens", "model_input_kv_lens",
            "graph_runtime_mode",
        )
        present = tuple(name for name in names if name in parent.attributes)
        if not present:
            return {}
        if present != names:
            raise ValueError(
                "FLASH_ATTN helper graph token domains are incomplete: "
                f"present={present!r}")
        if "q_lens" not in parent.attributes:
            raise ValueError("FLASH_ATTN helper has no logical request lengths")
        logical_tokens = sum(parent.attributes["q_lens"])
        model_input_tokens = sum(parent.attributes["model_input_q_lens"])
        runtime_mode = parent.attributes["graph_runtime_mode"]
        if model_input_tokens < logical_tokens:
            raise ValueError(
                "FLASH_ATTN helper model-input tokens cannot shrink work")
        return {
            "graph_logical_tokens": logical_tokens,
            "graph_model_input_tokens": model_input_tokens,
            "graph_runtime_mode": runtime_mode,
            "model_input_shape": dict(physical_shape),
        }
    names = (
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode", "model_input_shape",
    )
    present = tuple(name for name in names if name in parent.attributes)
    if not present:
        return {}
    if present != names:
        raise ValueError(
            "generated helper graph token domains are incomplete: "
            f"present={present!r}")
    return {
        "graph_logical_tokens": parent.attributes["graph_logical_tokens"],
        "graph_model_input_tokens": parent.attributes[
            "graph_model_input_tokens"],
        "graph_runtime_mode": parent.attributes["graph_runtime_mode"],
        "model_input_shape": dict(physical_shape),
    }


def _validate_graph_domain_boundary(
    node: WorkloadNode,
    graph: WorkloadGraph,
) -> None:
    """Require an activation-transfer boundary to preserve its token domain."""

    domain = _graph_token_domain(node)
    predecessors = graph.predecessors(node.node_id)
    if not predecessors:
        return
    predecessor_domains = tuple(
        _graph_token_domain(graph.node(predecessor))
        for predecessor in predecessors)
    if domain is None:
        if any(predecessor_domain is not None
               for predecessor_domain in predecessor_domains):
            raise ValueError(
                "graph activation boundary omitted its producer token domain")
        return
    if any(predecessor_domain != domain
           for predecessor_domain in predecessor_domains):
        raise ValueError(
            "graph activation boundary token domain disagrees with its "
            "producer")


_SOURCE_OBJECT_ATTRIBUTES = frozenset((
    "symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls", "symbolic_parameter_inputs",
    "symbolic_slot_mapping", "symbolic_model_input_order", "symbolic_prefill_workspace_requests",
    "symbolic_prefill_gather_workspace",
    "symbolic_shared_topk_clear", "symbolic_topk_token_range",
    "symbolic_collective_input_producers",
    "fa3_gqa_metadata_prepare_node",
    "fa3_full_input_producers", "fa3_full_aot_input_producers",
    "source_topk_allocation_id", "source_topk_input_writer",
))
_FA3_AOT_COPY_CLEAR_OBJECT_ATTRIBUTES = _SOURCE_OBJECT_ATTRIBUTES | frozenset((
    "kv_lens", "model_input_kv_lens", "fa3_full_aot_used_key",
))


def _source_object_attributes(node):
    """Keep current AOT payload values outside copy/fill KernelSpec keys.

    The original source graph retains and validates these length arrays.
    A copy is priced by its explicit bytes/link/direction/algorithm, and the
    tail fill by its element count. Neither helper reads the copied values.
    The prepare operator still consumes the current lengths in full.
    """
    attrs = node.attributes
    if ("source_model_execution_version" in attrs
            and attrs["source_model_execution_version"] == "vllm_0_30_0"
            and "fa3_full_aot_component" in attrs and "source_component" in attrs
            and ((node.kind is WorkloadKind.MEMCPY
                and attrs["fa3_full_aot_component"] == "copy"
                and attrs["source_component"] == "fa3_full_aot_copy")
                or (node.kind is WorkloadKind.VECTOR
                    and attrs["fa3_full_aot_component"] == "clear"
                    and attrs["source_component"] == "fa3_full_aot_clear"))):
        return _FA3_AOT_COPY_CLEAR_OBJECT_ATTRIBUTES
    return _SOURCE_OBJECT_ATTRIBUTES

# These fields bind the actual heterogeneous source's payload rows/views.
# validate_heterogeneous_graph still validates scope/placement/transfers on
# the original complete graph (these row/view records remain source evidence);
# heterogeneous_moe_specs selects typed shape/traffic/format fields and does
# not consume these identities when constructing any KernelSpec. Keep all
# network bytes/endpoints/NICs, gather counts and other physical inputs.
_HETEROGENEOUS_SOURCE_BINDING_ATTRIBUTES = frozenset((
    "source_view", "destination_view", "assignment_rows_experts", "expert_gpu_row_order",
    "expert_output_ids", "return_assignment_rows_experts", "global_token_ids", "assumed_topk_ids",
))
_SOURCE_NODE_FIELDS = tuple(field.name for field in fields(WorkloadNode))
if _SOURCE_NODE_FIELDS != (
        "node_id", "kind", "pool_name", "shape", "dtype", "variant", "rank", "attributes"):
    raise RuntimeError("canonical WorkloadNode field layout changed; update the exact lowering contract")
_SOURCE_CONTRACT_FIELDS = tuple(name for name in _SOURCE_NODE_FIELDS
    if name not in ("node_id", "rank", "attributes"))


def _declared_dspark_source(node):
    """Closed all-valid standalone source domain; this is not native admission."""
    attrs = node.attributes
    if "source_v4_aux_capture" in attrs:
        required_aux = {"model_execution_version", "model_source_profile", "source_projection_only",
            "launch_policy", "symbolic_tensor_inputs", "symbolic_tensor_outputs"}
        return (required_aux.issubset(attrs) and attrs["source_v4_aux_capture"] is True
            and attrs["model_execution_version"] == "vllm_0_30_0"
            and attrs["model_source_profile"] == "vllm_0_30_v4_sync_fp8_ds_mla"
            and attrs["source_projection_only"] is True
            and attrs["launch_policy"] == "analytical_launch"
            and ("source_dummy_run" not in attrs or attrs["source_dummy_run"] is False)
            and "symbolic_tensor_inputs" in attrs and "symbolic_tensor_outputs" in attrs)
    required = {"source_dspark_standalone", "source_dspark_data_parallel", "source_dspark_bound_target",
        "model_execution_version", "model_source_profile", "source_projection_only",
        "symbolic_tensor_inputs", "symbolic_tensor_outputs"}
    if (not required.issubset(attrs) or any(type(attrs[name]) is not bool for name in (
                "source_dspark_standalone", "source_dspark_bound_target", "source_dspark_data_parallel"))
            or (attrs["source_dspark_standalone"], attrs["source_dspark_bound_target"], attrs["model_source_profile"]) not in (
                (True, False, "vllm_0_30_v4_dspark_standalone"),
                (False, True, "vllm_0_30_v4_dspark_same_tp_target_bound"))
            or attrs["source_dspark_data_parallel"] is not False
            or attrs["model_execution_version"] != "vllm_0_30_0"
            or attrs["source_projection_only"] is not True
            or ("source_dummy_run" in attrs and attrs["source_dummy_run"] is not False)
            or any(not isinstance(attrs[name], (tuple, list)) or any(not isinstance(port, str) or not port
                for port in attrs[name]) for name in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"))
            or not attrs["symbolic_tensor_outputs"]):
        return False
    components = {
        (WorkloadKind.GATHER, "gather"): (None,),
        (WorkloadKind.GATHER, "embedding"): ("vllm_0_30_symbolic_vocab_embedding",),
        (WorkloadKind.GATHER, "dspark_markov_embedding"): ("vllm_0_30_dspark_torch_embedding",),
        (WorkloadKind.LINEAR, "gemm"): (None,),
        (WorkloadKind.LINEAR, "batched_gemm"): ("vllm_0_30_symbolic_v4_grouped_output",),
        (WorkloadKind.LM_HEAD, "gemm"): (None,), (WorkloadKind.LM_HEAD, "gemv"): (None,),
        (WorkloadKind.ROUTER, "gemm"): (None,),
        (WorkloadKind.NORM, "rmsnorm"): (None,),
        (WorkloadKind.NORM, "dsv4_q_kv_rmsnorm"): ("vllm_0_30_symbolic_v4_qkv_norm",),
        (WorkloadKind.ELEMENTWISE, "elementwise"): (None,), (WorkloadKind.ELEMENTWISE, "add"): (None,),
        (WorkloadKind.ACTIVATION, "silu"): (None,),
        (WorkloadKind.KV_UPDATE, "dsv4_qnorm_rope_kv_insert"): ("vllm_0_30_symbolic_v4_swa_insert",),
        (WorkloadKind.EAGER_ATTN, "dspark_noncausal_swa"): ("vllm_0_30_dspark_noncausal_swa",),
        (WorkloadKind.VECTOR, "dspark_noncausal_swa_indices"): ("vllm_0_30_dspark_noncausal_swa",),
        (WorkloadKind.VECTOR, "dspark_direct_argmax"): ("vllm_0_30_dspark_greedy_full_vocab",),
        (WorkloadKind.VECTOR, "dsv4_inv_rope_fp8_quant"): ("vllm_0_30_symbolic_v4_inverse_rope",),
        (WorkloadKind.VECTOR, "dsv4_moe_route"): ("vllm_0_30_symbolic_v4_routing",),
        (WorkloadKind.VECTOR, "vllm_logits_gather_reorder"): ("vllm_0_30_symbolic_logits_boundary",),
        (WorkloadKind.FUSED_MOE, "full"): ("vllm_0_30_symbolic_v4_marlin",),
        (WorkloadKind.COLLECTIVE, None): (None,),
    }
    for variant in ("mhc_prenorm", "mhc_pre_norm"):
        components[(WorkloadKind.VECTOR, variant)] = ("vllm_0_30_symbolic_v4_dspark_prenorm",)
    for variant in ("mhc_post_pre", "mhc_post", "hc_head"):
        components[(WorkloadKind.VECTOR, variant)] = ("vllm_0_30_symbolic_mhc",)
    if attrs["source_dspark_bound_target"] is True:
        components[(WorkloadKind.VECTOR, "vllm_embedding_id_widen")] = ("vllm_0_30_symbolic_embedding",)
        components[(WorkloadKind.GATHER, "embedding")] = (None, "vllm_0_30_symbolic_vocab_embedding")
        components[(WorkloadKind.ELEMENTWISE, "elementwise")] = (None,
            "vllm_0_30_RequestState_last_sampled_constructor", "vllm_0_30_InputBuffers_constructor",
            "vllm_0_30_RequestState_SamplingStates_constructor", "vllm_0_30_RequestState_draft_constructor")
        for variant, component in (
                ("dspark_prepare_inputs", "vllm_0_30_prepare_dflash_inputs"),
                ("dspark_token_request_mapping", "vllm_0_30_dspark_attention_metadata"),
                ("dspark_slot_validity", "vllm_0_30_dspark_attention_metadata"),
                ("dspark_validity_copy", "vllm_0_30_dspark_attention_metadata"),
                ("vllm_gpu_v2_post_update", "vllm_0_30_gpu_post_update"),
                ("vllm_gpu_v2_store_drafts", "vllm_0_30_RequestState_drafts"),
                ("vllm_gpu_v2_stage_requests", "vllm_0_30_RequestState_add_request")):
            components[(WorkloadKind.VECTOR, variant)] = (component,)
    key = (node.kind, node.variant)
    if key not in components or (attrs["source_component"] if "source_component" in attrs else None) not in components[key]:
        return False
    if node.kind is WorkloadKind.COLLECTIVE:
        return (attrs["operation"] in ("all_reduce", "all_gather")
            and len(attrs["symbolic_tensor_inputs"]) == len(attrs["participants"])
            and len(attrs["symbolic_tensor_outputs"]) == len(attrs["participants"]))
    if ("target_capture_layers" not in attrs or "draft_layer_indices" not in attrs
            or (tuple(attrs["target_capture_layers"]), tuple(attrs["draft_layer_indices"])) not in (
                ((40, 41, 42), (43, 44, 45)), ((58, 59, 60), (61, 62, 63)))
            or attrs["launch_policy"] != "analytical_launch"):
        return False
    return True


def _dspark_typed_math_specs(node, hardware_id):
    """Adapt only four exact source operations to their existing mathematical owners."""
    attrs, shape = node.attributes, node.shape
    ports, outputs = attrs["symbolic_tensor_inputs"], attrs["symbolic_tensor_outputs"]
    common = dict(launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (node.variant == "dsv4_qnorm_rope_kv_insert" and "masked_dspark_context" in attrs
            and attrs["masked_dspark_context"] is True):
        valid = attrs["cache_write_valid_rows"]
        slots = tuple(attrs["symbolic_slot_mapping"])
        scoped = _CURRENT_DSPARK_SOURCE_GRAPH.get()
        if (not _declared_dspark_source(node) or attrs["source_dspark_bound_target"] is not True
                or attrs["masked_dspark_context"] is not True or node.dtype != "bf16"
                or node.kind is not WorkloadKind.KV_UPDATE or shape["seq_len"] != 6
                or type(valid) is not int or not 1 <= valid <= 6 or len(slots) != 6
                or any(type(slot) is not int or slot < 0 for slot in slots[:valid])
                or slots[valid:] != (-1,) * (6 - valid)
                or attrs["slot_mapping_all_valid"] is not (valid == 6)
                or attrs["cache_write_semantics"] != "masked_target_context_overwrite"
                or scoped is None or scoped[2] is None or node.node_id not in scoped[2]._by_node):
            raise ValueError("DSpark context overwrite requires the current six-row source and exact accepted slot prefix")
        source = scoped[2]._by_node[node.node_id]
        if (source.rank != node.rank or source.pool_name != node.pool_name or source.attributes != attrs
                or tuple(view.view_id for view in source.inputs) != tuple(ports)
                or tuple(view.view_id for view in source.outputs) != tuple(outputs)
                or len(source.inputs) != 5 or len(source.outputs) != 2
                or source.inputs[0].dtype != "bf16" or prod(source.inputs[0].shape) != 6 * shape["num_query_heads"] * 512
                or source.inputs[1].dtype != "bf16" or prod(source.inputs[1].shape) != 6 * 512
                or source.inputs[2].dtype != "int64" or prod(source.inputs[2].shape) != 6
                or source.inputs[4].dtype != "int64" or prod(source.inputs[4].shape) != 6):
            raise ValueError("DSpark context overwrite source ports differ from current typed receipt")
        return (KernelSpec(KernelKind.KV_CACHE, hardware_id, _vector_shape(node), dtype="bf16",
            variant=node.variant, attributes={**{name: attrs[name] for name in (
                "cache_layout", "cache_write_semantics", "quantization", "slot_mapping_all_valid")},
                "mathematical_program": "vllm_0_30_dspark_partial_context_insert_v1", "valid_token_count": valid}, **common),)
    metadata_operations = {
        "dspark_prepare_inputs": ("vllm_0_30_prepare_dflash_inputs", ("prepare_inputs",)),
        "dspark_token_request_mapping": ("vllm_0_30_dspark_attention_metadata", ("token_request_mapping",)),
        "dspark_slot_validity": ("vllm_0_30_dspark_attention_metadata", ("slot_mapping_ge_zero",)),
        "dspark_validity_copy": ("vllm_0_30_dspark_attention_metadata", ("validity_copy",)),
        "vllm_gpu_v2_greedy_sampler": ("vllm_0_30_gpu_V2_plain_sampler", (
            "gumbel_block_argmax", "gumbel_reduce_argmax", "gumbel_gather", "seed_num_sampled_ones", "sampled_rejected_counts")),
        "vllm_gpu_v2_greedy_rejection": ("vllm_0_30_gpu_V2_rejection_sampler", (
            "chunk_cu_subtract", "local_stats", "rejection", "resample", "insert", "sampled_rejected_counts")),
        "vllm_gpu_v2_post_update": ("vllm_0_30_gpu_post_update", ("post_update",)),
        "vllm_gpu_v2_store_drafts": ("vllm_0_30_RequestState_drafts", ("indexed_assignment",)),
        "vllm_gpu_v2_stage_requests": ("vllm_0_30_RequestState_add_request", ("project_staged_writes",)),
    }
    scoped = _CURRENT_DSPARK_SOURCE_GRAPH.get()
    constructor = (node.kind is WorkloadKind.ELEMENTWISE and node.variant == "elementwise"
        and attrs["operation"] == "fill" and scoped is not None and node.rank in scoped[1]
        and ("source_dspark_bound_target" in attrs and attrs["source_dspark_bound_target"] is True
            or "source_v4_aux_capture" in attrs
            or "source_component" in attrs and attrs["source_component"] == "vllm_0_30_RequestState_draft_constructor"))
    if node.variant in metadata_operations or constructor:
        if scoped is None or node.rank not in scoped[1] or scoped[0].node(node.node_id) is not node:
            raise ValueError("DSpark typed role projection requires current target/capture source proof")
        memory = scoped[2]
        if memory is None or node.node_id not in memory._by_node:
            raise ValueError("DSpark mixed typed roles require the current TraceMemoryContext")
        source = memory._by_node[node.node_id]
        if (source.node_id != node.node_id or source.rank != node.rank or source.pool_name != node.pool_name
                or source.kind is not node.kind or source.variant != node.variant or source.attributes != attrs
                or tuple(view.view_id for view in source.inputs) != tuple(ports)
                or tuple(view.view_id for view in source.outputs) != tuple(outputs)
                or tuple(key for key in source.predecessors if key in scoped[0]._nodes_by_id)
                    != tuple(scoped[0].predecessors(node.node_id))):
            raise ValueError("DSpark typed source context is stale or differs from this node/ports/pool/rank")
        if not constructor:
            component, operations = metadata_operations[node.variant]
            if (attrs["source_component"] != component or attrs["source_operation"] not in operations
                    or attrs["source_projection_only"] is not True):
                raise ValueError("DSpark typed role projection rejects foreign metadata programs")
        if node.variant in ("vllm_gpu_v2_greedy_sampler", "vllm_gpu_v2_greedy_rejection"):
            if attrs["source_profile"] != "vllm_0_30_V2_plain_allgreedy_no_sampling_params":
                raise ValueError("DSpark V2 payload projection requires its explicit all-greedy source")
            if attrs["source_operation"] == "gumbel_block_argmax":
                rows = shape["rows"]
                if (node.dtype != "bf16" or shape["vocab"] != 129280 or shape["elements"] != rows * 129280
                        or attrs["vocab_block_size"] != 1024 or attrs["all_greedy"] is not True
                        or attrs["apply_temperature"] is not False or source.inputs[0].shape != (rows, 129280)
                        or source.inputs[0].dtype != "bf16"
                        or tuple(view.shape for view in source.outputs) != ((rows, 127), (rows, 127))
                        or tuple(view.dtype for view in source.outputs) != ("int64", "fp32")):
                    raise ValueError("DSpark Gumbel projection must scan exact masked BF16 vocab once and write both real partial roles")
        specs = []
        for reading, views in ((True, source.inputs), (False, source.outputs)):
            for view in views:
                if view.dtype not in ("bool", "int32", "int64", "fp32", "bf16"):
                    raise ValueError("DSpark typed payload cannot change an unknown source dtype")
                elements = prod(view.shape)
                if elements == 0:
                    continue
                if view.element_bytes != DTYPE_BYTES[view.dtype]:
                    raise ValueError("DSpark source dtype and element width disagree")
                specs.append(KernelSpec(KernelKind.VECTOR, hardware_id, {"element_count": elements},
                    dtype=view.dtype, variant="elementwise", attributes={
                        "mathematical_program": "vllm_0_30_dspark_typed_role_bytes_v1",
                        "operation": "read" if reading else "fill", "reads": int(reading), "writes": int(not reading)}, **common))
        if not specs:
            raise ValueError("DSpark source metadata must expose at least one real typed payload role")
        return tuple(specs)
    if node.variant == "v4_aux_hidden_mean":
        if (not _declared_dspark_source(node) or node.kind is not WorkloadKind.VECTOR or node.dtype != "bf16"
                or shape["streams"] != 4 or shape["hidden"] not in (4096, 7168)
                or shape["elements"] != shape["rows"] * 4 * shape["hidden"]
                or attrs["source_component"] != "vllm_0_30_v4_target_aux_mhc_mean"
                or attrs["operation"] != "torch_mean" or attrs["reduction_axis"] != 1
                or attrs["output_elements"] != shape["rows"] * shape["hidden"]
                or len(ports) != 1 or len(outputs) != 1):
            raise ValueError("DSpark auxiliary mean requires exact target hc4 BF16 source geometry")
        return (KernelSpec(KernelKind.VECTOR, hardware_id,
            {"element_count": attrs["output_elements"], "rows": shape["rows"], "width": shape["hidden"], "source_rows": 4 * shape["rows"]},
            dtype="bf16", variant="vllm_mtp_eager", attributes={"model_execution_version": "vllm_0_30_0", "operation": "dspark_aux_mean"}, **common),)
    if node.variant in ("dspark_noncausal_swa", "dspark_noncausal_swa_indices"):
        names = {"causal", "q_lens", "prefix_lens", "visible_kv_lens", "visible_logical_slot_ids_by_request",
            "index_capacity", "future_query_keys_visible", "invalid_index_fill", "native_provider_bound",
            "physical_output_rows", "active_output_rows", "unwritten_output_tail_rows"}
        if not names.issubset(attrs):
            raise ValueError("DSpark noncausal source requires its complete shape and visibility facts")
        queries, prefixes, visible = (tuple(attrs[name]) for name in ("q_lens", "prefix_lens", "visible_kv_lens"))
        if (not queries or any(type(q) is not int or q != 5 for q in queries)
                or len(queries) != len(prefixes) or len(queries) != len(visible)
                or any(type(kv) is not int or kv < 1 for kv in prefixes)
                or visible != tuple(5 + min(prefix, 128) for prefix in prefixes)
                or tuple(len(slots) for slots in attrs["visible_logical_slot_ids_by_request"]) != visible
                or attrs["causal"] is not False or attrs["future_query_keys_visible"] is not True
                or attrs["invalid_index_fill"] != -1 or attrs["native_provider_bound"] is not False
                or attrs["index_capacity"] != 192 or attrs["active_output_rows"] != sum(queries)
                or attrs["physical_output_rows"] != sum(queries) or attrs["unwritten_output_tail_rows"] != 0):
            raise ValueError("DSpark math requires complete noncausal five-query blocks without dummy or unwritten tail")
        lengths = tuple(prefix + 5 for prefix in prefixes)
        if node.variant == "dspark_noncausal_swa_indices":
            if (node.kind is not WorkloadKind.VECTOR or node.dtype != "int32" or len(ports) not in (1, 3, 6)
                    or len(outputs) != 2 or shape != {"elements": sum(queries) * 192, "rows": sum(queries), "index_width": 192}):
                raise ValueError("DSpark index source requires its exact INT32 two-output geometry")
            return (KernelSpec(KernelKind.VECTOR, hardware_id,
                {"element_count": sum(queries) * 192, "rows": sum(queries), "width": 192, "source_rows": len(queries)},
                dtype="int32", variant="vllm_mtp_eager", attributes={"model_execution_version": "vllm_0_30_0",
                    "operation": "dspark_swa_indices", "num_speculative_tokens": 5, "kv_lens": lengths}, **common),)
        if (node.kind is not WorkloadKind.EAGER_ATTN or node.dtype != "bf16" or len(ports) != 4 or len(outputs) != 1
                or shape["batch"] != len(queries) or shape["q_len"] != 5 or shape["kv_len"] != max(lengths)
                or shape["qk_head_dim"] != 512 or shape["v_head_dim"] != 512 or shape["num_kv_heads"] != 1
                or shape["num_query_heads"] not in (64, 128) or shape["window_size"] != 128):
            raise ValueError("DSpark attention requires its exact BF16 source geometry and Q/KV/index/lens ports")
        return (KernelSpec(KernelKind.SPARSE_ATTN, hardware_id,
            {"batch": len(queries), "q_len": 5, "kv_len": max(lengths), "num_query_heads": shape["num_query_heads"],
             "qk_head_dim": 512, "v_head_dim": 512, "topk": 0, "window_size": 128, "compress_ratio": 1,
             "compressed_state_bytes": 584, "swa_token_bytes": 584}, dtype="bf16", variant="dsv4_sparse_decode",
            attributes={"causal": False, "indexer_type": "shared", "kv_layout": "mla", "kv_dist": "uniform",
                "q_lens": queries, "kv_lens": lengths, "selected_kv_lens": (0,) * len(queries),
                "topk_capacity": attrs["index_capacity"], "topk_index_dtype": "int32", "swa_kv_lens": visible,
                "compressed_selection": "candidate_upper_bound", "num_speculative_tokens": 5,
                "mathematical_program": "vllm_0_30_dspark_swa_noncausal_softmax_v1"}, **common),)
    if node.variant == "dspark_markov_embedding":
        parameters = attrs["symbolic_parameter_inputs"] if "symbolic_parameter_inputs" in attrs else ()
        if (node.kind is not WorkloadKind.GATHER or node.dtype != "bf16" or len(ports) != 2 or len(outputs) != 1
                or tuple(parameters) != (ports[1],) or shape["hidden"] not in (256, 512)
                or not 1 <= shape["rows"] <= 16 or shape["elements"] != shape["rows"] * shape["hidden"]
                or attrs["index_bytes"] not in (4, 8) or attrs["gather_factor"] != 1
                or attrs["index_count"] != shape["rows"] or attrs["scattered"] is not True
                or attrs["materialized_output"] is not True):
            raise ValueError("DSpark Markov embedding requires actual BF16 parameter and INT32/INT64 index ports")
        return (KernelSpec(KernelKind.VECTOR, hardware_id, _vector_shape(node), dtype="bf16", variant="gather",
            attributes={name: attrs[name] for name in ("gather_factor", "index_count", "index_bytes", "scattered", "materialized_output")}, **common),)
    if node.variant == "dspark_direct_argmax":
        if (node.kind is not WorkloadKind.VECTOR or node.dtype != "bf16" or len(ports) != 1 or len(outputs) != 1
                or attrs["argmax_output_dtype"] != "int64" or attrs["probabilistic"] is not False
                or attrs["topk_truncation"] is not False or attrs["values_computed"] is not False):
            raise ValueError("DSpark argmax requires BF16 full-vocab input and explicit INT64 greedy output")
        return (KernelSpec(KernelKind.VECTOR, hardware_id, _vector_shape(node), dtype="bf16", variant="dsv32_draft_argmax",
            attributes={"sampling_source": "vllm_0_30_dspark_full_vocab_greedy_bf16"}, **common),)
    return None


def _validate_dspark_source_ports(node, graph):
    """Keep actual producer ports before any cached lowering can be reused."""
    attrs = node.attributes
    bound = ("source_v4_aux_capture" in attrs or
        "source_dspark_bound_target" in attrs and attrs["source_dspark_bound_target"] is True)
    if bound:
        binding = _CURRENT_DSPARK_SOURCE_GRAPH.get()
        if binding is None or binding[0] is not graph or graph.node(node.node_id) is not node:
            raise ValueError("DSpark bound source requires this invocation's actual graph/node proof")
        proofs = binding[1]
        if node.rank not in proofs:
            # The first actual target embedding identifies this invocation.
            # Source lookup is direct; no graph scan or lazy materialization.
            marker = ":embedding"
            tp1_widen = (node.kind is WorkloadKind.VECTOR and node.variant == "vllm_embedding_id_widen"
                and node.dtype == "int32" and "source_component" in attrs
                and attrs["source_component"] == "vllm_0_30_symbolic_embedding"
                and "embedding_source" in attrs and attrs["embedding_source"] == "vllm_0_30_unquantized_tp1")
            if ((node.variant != "embedding" and not tp1_widen) or marker not in node.node_id
                    or "source_v4_aux_capture" not in attrs or not _declared_dspark_source(node)):
                raise ValueError("DSpark bound target requires its actual current target embedding before capture")
            if tp1_widen:
                memory = binding[2]
                if memory is None or node.node_id not in memory._by_node:
                    raise ValueError("DSpark TP1 embedding requires the current TraceMemoryContext")
                source = memory._by_node[node.node_id]
                rows = node.shape["rows"]
                if (node.shape != {"rows": rows, "elements": rows} or rows < 1
                        or source.rank != node.rank or source.pool_name != node.pool_name or source.attributes != attrs
                        or source.kind is not node.kind or source.variant != node.variant
                        or len(source.inputs) != 1 or len(source.outputs) != 1
                        or source.inputs[0].dtype != "int32" or source.outputs[0].dtype != "int64"
                        or source.inputs[0].shape != (rows,) or source.outputs[0].shape != (rows,)
                        or tuple(view.view_id for view in source.inputs) != tuple(attrs["symbolic_tensor_inputs"])
                        or tuple(view.view_id for view in source.outputs) != tuple(attrs["symbolic_tensor_outputs"])):
                    raise ValueError("DSpark TP1 embedding requires its actual INT32-to-INT64 producer ports")
            prefix, invocation = node.node_id.split(marker, 1)
            if invocation and (not invocation.startswith(":") or not invocation[1:].isdigit()):
                raise ValueError("DSpark target invocation must retain its source recorder identity")
            ordinal = int(invocation[1:]) if invocation else 0
            candidates = tuple(prefix + ":draft:capture-concat" + (f":{index}" if index else "")
                for index in range(ordinal + 1))
            present = tuple(name for name in candidates if name in graph._nodes_by_id)
            if len(present) != 1:
                raise ValueError("DSpark target auxiliary source requires one actual current draft capture consumer")
            concat_id, = present
            suffix = concat_id[len(prefix + ":draft:capture-concat"):]
            concat = graph.node(concat_id)
            if (not _declared_dspark_source(concat) or concat.attributes["source_dspark_bound_target"] is not True
                    or concat.rank != node.rank or concat.pool_name != node.pool_name
                    or concat.attributes["source_operation"] != "cat_last_dim"):
                raise ValueError("DSpark capture consumer must retain exact same-TP bound target identity")
            layers = tuple(concat.attributes["target_capture_layers"])
            hidden = { (40, 41, 42): 4096, (58, 59, 60): 7168 }[layers]
            rows = concat.shape["rows"]
            if (concat.kind is not WorkloadKind.GATHER or concat.variant != "gather" or concat.dtype != "bf16"
                    or concat.shape != {"rows": rows, "hidden": 3 * hidden, "elements": rows * 3 * hidden}
                    or rows <= 0 or len(concat.attributes["symbolic_tensor_outputs"]) != 1):
                raise ValueError("DSpark capture concat must materialize three exact BF16 target layers")
            means = []
            for layer in layers:
                mean_id = prefix + f":aux-layer-{layer}:mean" + suffix
                post_id = prefix + f":aux-layer-{layer}:mhc-post" + suffix
                if mean_id not in graph._nodes_by_id or post_id not in graph._nodes_by_id:
                    raise ValueError("DSpark target capture is missing its actual layer mean/post producer")
                mean, post = graph.node(mean_id), graph.node(post_id)
                if (mean.rank != node.rank or post.rank != node.rank or mean.pool_name != node.pool_name
                        or post.pool_name != node.pool_name or not _declared_dspark_source(mean)
                        or not _declared_dspark_source(post) or mean.variant != "v4_aux_hidden_mean"
                        or mean.shape != {"elements": rows * 4 * hidden, "rows": rows, "streams": 4, "hidden": hidden}
                        or post.kind is not WorkloadKind.VECTOR or post.variant != "mhc_post" or post.dtype != "bf16"
                        or post.shape["hidden"] != hidden or post.shape["hc_mult"] != 4
                        or post.shape["elements"] != rows * 4 * hidden
                        or tuple(mean.attributes["symbolic_tensor_inputs"]) != tuple(post.attributes["symbolic_tensor_outputs"])
                        or post_id not in graph.predecessors(mean_id)
                        or mean_id not in graph.predecessors(concat_id)):
                    raise ValueError("DSpark capture requires real typed target hc4 post->mean->concat producer ports")
                means.extend(mean.attributes["symbolic_tensor_outputs"])
            if tuple(concat.attributes["symbolic_tensor_inputs"]) != tuple(means):
                raise ValueError("DSpark concat must consume the three ordered actual target mean outputs")
            projection_id = prefix + ":draft:main-proj" + suffix
            if projection_id not in graph._nodes_by_id:
                raise ValueError("DSpark captured target states require their actual draft main projection")
            projection = graph.node(projection_id)
            if (projection.kind is not WorkloadKind.LINEAR or projection.dtype != "bf16"
                    or projection.shape != {"b": 1, "m": rows, "n": hidden, "k": 3 * hidden}
                    or tuple(projection.attributes["symbolic_tensor_inputs"]) != tuple(concat.attributes["symbolic_tensor_outputs"])
                    or concat_id not in graph.predecessors(projection_id)):
                raise ValueError("DSpark captured states must feed exact BF16/FP8 draft main projection")
            proofs[node.rank] = (prefix, suffix, layers, hidden)
        prefix, suffix, layers, hidden = proofs[node.rank]
        if ((node.kind is not WorkloadKind.COLLECTIVE and not node.node_id.startswith(prefix + ":")) or node.pool_name != graph.node(prefix + ":draft:capture-concat" + suffix).pool_name
                or "target_capture_layers" in attrs and tuple(attrs["target_capture_layers"]) != layers):
            raise ValueError("DSpark source node disagrees with its current target/draft capture identity")
    if node.variant not in ("dspark_noncausal_swa", "dspark_markov_embedding", "dspark_direct_argmax"):
        return
    attrs = node.attributes
    predecessors = tuple(graph.node(key) for key in graph.predecessors(node.node_id))
    producers = {port: prior for prior in predecessors for port in (
        prior.attributes["symbolic_tensor_outputs"] if "symbolic_tensor_outputs" in prior.attributes else ())}
    inputs = attrs["symbolic_tensor_inputs"]
    if node.variant == "dspark_noncausal_swa":
        if any(port not in producers for port in inputs):
            raise ValueError("DSpark attention requires its actual Q/KV/index/lens producers")
        indices, lengths = (producers[inputs[index]] for index in (2, 3))
        if (indices.node_id != lengths.node_id or indices.variant != "dspark_noncausal_swa_indices"
                or tuple(inputs[2:4]) != tuple(indices.attributes["symbolic_tensor_outputs"])
                or any(tuple(indices.attributes[name]) != tuple(attrs[name]) for name in ("q_lens", "prefix_lens", "visible_kv_lens"))):
            raise ValueError("DSpark attention indices and visible rows must come from the same actual source kernel")
    elif node.variant == "dspark_markov_embedding":
        if inputs[0] not in producers:
            raise ValueError("DSpark Markov embedding needs its actual preceding token producer")
        previous = producers[inputs[0]]
        if previous.variant == "dspark_direct_argmax" and previous.attributes["argmax_output_dtype"] == "int64":
            width = 8
        elif previous.dtype in ("int32", "int64"):
            width = DTYPE_BYTES[previous.dtype]
        else:
            raise ValueError("DSpark Markov embedding requires an integer token producer")
        if attrs["index_bytes"] != width:
            raise ValueError("DSpark Markov input index width differs from its actual producer")
    elif inputs[0] not in producers or producers[inputs[0]].kind is not WorkloadKind.ELEMENTWISE or producers[inputs[0]].dtype != "bf16":
        raise ValueError("DSpark argmax requires its actual BF16 biased-logit producer")


def _require_source_lowering_contract(node):
    """Validate each consumed node without materializing a lazy source graph."""
    late_dspark_storage = (node.variant == "vllm_gpu_v2_store_drafts" or node.kind is WorkloadKind.ELEMENTWISE
        and "source_component" in node.attributes
        and node.attributes["source_component"] == "vllm_0_30_RequestState_draft_constructor")
    if late_dspark_storage:
        binding = _CURRENT_DSPARK_SOURCE_GRAPH.get()
        if binding is None or node.rank not in binding[1] or binding[0].node(node.node_id) is not node:
            raise ValueError("DSpark resident storage requires current graph/capture proof")
    if "source_gemm_runtime" in node.attributes:
        checked = _CURRENT_GEMM_SOURCE_BINDINGS.get()
        if checked is None or id(node) not in checked:
            raise ValueError("pre-marked GEMM runtime requires current source_launch_inputs and full source validation")
    if ("source_v4_aux_capture" in node.attributes or node.variant == "v4_aux_hidden_mean"
            or "source_dspark_bound_target" in node.attributes and node.attributes["source_dspark_bound_target"] is True):
        binding = _CURRENT_DSPARK_SOURCE_GRAPH.get()
        if binding is None or not _declared_dspark_source(node):
            raise NotImplementedError("V4 target auxiliary capture source has no bound DSpark operator/lowering contract")
        _validate_dspark_source_ports(node, binding[0])
    if "source_dspark_standalone" in node.attributes or node.variant in (
            "dspark_noncausal_swa", "dspark_noncausal_swa_indices", "dspark_markov_embedding", "dspark_direct_argmax"):
        if not _declared_dspark_source(node):
            raise NotImplementedError("DSpark complete noncausal source has no bound KernelSpec/lowering program; Goal05 pricing remains unbound")


def _node_physical_attributes(node, routing_relabelling=None):
    """Keep pricing geometry while leaving source identities on their calls."""
    attributes = dict(node.attributes)
    # These are WorkloadGraph provenance/binding metadata, not inputs to
    # KernelSpec construction. Keep them on the original graph and calls;
    # per-step object names and logical slot coordinates must not retain a
    # new template for identical math. Shapes, dtypes, layouts, q/kv lengths
    # and all semantic memory/launch attributes remain in the exact key.
    for name in _source_object_attributes(node):
        if name in attributes:
            attributes.pop(name)
    if (node.kind is WorkloadKind.FLASH_ATTN
            and "source_program" in attributes
            and attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode"
            and "fa3_full_physical_parents" in attributes):
        # Allocation/producer/lifetime identities are checked by the source
        # and memory binders, per actual call. Retain their offsets, shapes
        # and strides in the physical key as well.
        parents = attributes["fa3_full_physical_parents"]
        if (type(parents) is tuple and len(parents) == 2
                and all(type(parent) is tuple and len(parent) == 6 for parent in parents)):
            attributes["fa3_full_physical_parents"] = tuple(parent[1:4] for parent in parents)
    if is_heterogeneous_moe(node):
        for name in _HETEROGENEOUS_SOURCE_BINDING_ATTRIBUTES:
            if name in attributes:
                attributes.pop(name)
    if node.kind is WorkloadKind.FUSED_MOE:
        attributes.pop("routing_provenance")
        attributes.pop("moe_invocation_id")
        if routing_relabelling is not None:
            global_counts, local_counts = routing_relabelling
            attributes["tokens_per_expert"] = global_counts
            attributes["local_tokens_per_expert"] = local_counts
    elif (node.kind is WorkloadKind.COLLECTIVE
          and "moe_invocation_id" in attributes):
        attributes.pop("moe_invocation_id")
    return attributes


def _logical_node_contract(node: WorkloadNode, routing_relabelling=None):
    """Freeze KernelSpec inputs, excluding placement and traced object IDs."""

    _require_source_lowering_contract(node)
    attributes = _node_physical_attributes(node, routing_relabelling)
    # TP embedding reads the actual rank's vocabulary shard.
    rank_sensitive_embedding = (node.kind is WorkloadKind.GATHER and node.variant == "embedding"
        and "embedding_source" in attributes)
    if type(node) is WorkloadNode:
        # Canonical fields have a fixed public order. Keep the exact existing
        # pickle contract without rebuilding dataclass metadata for each node.
        contract_fields = (
            ("kind", node.kind), ("pool_name", node.pool_name),
            ("shape", dict(node.shape)), ("dtype", node.dtype),
            ("variant", node.variant),
        )
        if rank_sensitive_embedding:
            contract_fields += (("rank", node.rank),)
        contract_fields += (("attributes", attributes),)
    else:
        contract_fields = []
        for field in fields(node):
            if field.name == "node_id" or (field.name == "rank" and not rank_sensitive_embedding):
                continue
            value = getattr(node, field.name)
            if field.name == "shape":
                value = dict(value)
            if field.name == "attributes":
                value = attributes
            contract_fields.append((field.name, value))
    return (
        "pickle_protocol_5",
        pickle.dumps(
            (
                type(node).__module__,
                type(node).__qualname__,
                tuple(contract_fields),
            ),
            protocol=5,
        ),
    )


def _memo_contracts(graph: WorkloadGraph, memo=None, *, source_layout=_SOURCE_LAYOUT_UNSET):
    """Freeze every node contract that can affect emitted KernelSpecs."""

    if memo is not None and not isinstance(memo, KernelLoweringMemo):
        raise TypeError("source contract reuse requires KernelLoweringMemo")
    # A fresh source graph has fresh nodes. Its ordinary C-backed serializer
    # costs less than proving/rechecking reuse for every physical input; bound
    # graphs retain the existing run-local immutable-node fast path.
    if memo is not None and graph._is_node_binding:
        bound = memo.source_bound_contracts(graph, source_layout=source_layout)
        if bound is not None:
            return bound
    if memo is not None and not graph._is_node_binding:
        from src.serving.workload.workload_graph import (
            _node_update_source_unchanged, _workload_node_field_updates_read)
        if _node_update_source_unchanged():
            return {node.node_id: (memo.module_contract(node, _workload_node_field_updates_read(node))
                    if "_module_source_frame" in node.__dict__ else _logical_node_contract(node))
                for node in graph.nodes if node.kind is not WorkloadKind.FUSED_MOE}
    freeze = memo.logical_contract if memo is not None and graph._is_node_binding else _logical_node_contract
    return {
        node.node_id: freeze(node)
        for node in graph.nodes
        if node.kind is not WorkloadKind.FUSED_MOE
    }


def _fused_moe_routing_relabelling(node, hardware_id, hardware, canonical_axes=None):
    """Return this MoE node's canonical routing, or ``None`` to keep its own.

    The grouped projections read the expert axis only through the physical
    class ``canonical_expert_axis`` names, so every node in one class lowers
    to the same KernelSpec tuple and the run-local memo may share it.  A
    recipe that also emits a kernel reading the axis as an ordered prefix
    stream keeps its exact histogram instead.
    """

    if ("source_component" in node.attributes
            and node.attributes["source_component"] == "sglang_0_5_19_symbolic_fused_shared_moe"
            and hardware.supports_source_contract("sglang_0_5_19_symbolic_fused_shared_moe")):
        # Its existing external router also supplies the fused shared column.
        # Keep the declared axis; a vLLM recipe cannot identify this caller.
        return None
    if _h200_bf16_expert_math_domain(node, hardware_id):
        if "owned_experts" not in node.attributes or not node.attributes["owned_experts"]:
            return None
        counts, owned = node.attributes["tokens_per_expert"], node.attributes["owned_experts"]
        native = (canonical_axes is not None and type(counts) is tuple and type(owned) is tuple
            and all(type(value) is int for value in counts) and all(type(value) is int for value in owned))
        if not native:
            return canonical_expert_axis(counts, owned)
        key = id(counts), owned
        if key in canonical_axes:
            retained, result = canonical_axes[key]
            if retained is not counts:
                raise RuntimeError("source expert-axis identity collision")
            return result
        result = canonical_expert_axis(counts, owned)
        canonical_axes[key] = counts, result
        return result
    if "owned_experts" not in node.attributes or not node.attributes["owned_experts"]:
        return None
    recipe = select_fused_moe_recipe(hardware_id, node.shape, hardware, node.attributes)
    if ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS.intersection(recipe):
        return None
    return canonical_expert_axis(
        node.attributes["tokens_per_expert"],
        node.attributes["owned_experts"],
    )


class KernelLoweringMemo:
    """Bounded run-local LRU of exact KernelSpec templates.

    Capacity limits retained Python metadata, never physical work. Eviction
    only causes the ordinary lowering builder to run again.
    """

    def __init__(self, max_entries: int = 4096):
        if type(max_entries) is not int:
            raise TypeError("kernel lowering memo max_entries must be a positive int")
        if max_entries <= 0:
            raise ValueError("kernel lowering memo max_entries must be a positive int")
        self._max_entries = max_entries
        self._specs = OrderedDict()
        self._source_vector_recipes = OrderedDict()
        self._hardware_objects = {}
        self._hit_count = 0
        self._miss_count = 0
        self._source_contracts = OrderedDict()
        self._source_node_contracts = OrderedDict()
        self._source_compactions = OrderedDict()
        self._source_binding_contracts = OrderedDict()
        self._source_moe_static_contracts = OrderedDict()
        self._module_contracts = OrderedDict()
        self._call_templates = OrderedDict()
        self.call_template_hits = 0
        self.call_template_misses = 0
        self.module_contract_hits = 0
        self.module_contract_misses = 0
        self._glm_decode_plans = weakref.WeakKeyDictionary()

    def source_vector_recipe(self, node, contract, hardware_id, hardware):
        """Reuse immutable geometry after the ordinary exact node contract.

        Source object identities are excluded by the same contract as specs;
        vector selectors read only the retained shape/layout/branch fields.
        Current source ports and graph validation still run on every step.
        """
        key = (contract, id(hardware))
        if key in self._source_vector_recipes:
            self._source_vector_recipes.move_to_end(key)
            return self._source_vector_recipes[key]
        recipe = _node_source_launches(node, hardware_id, hardware)
        self._retain_source_value(self._source_vector_recipes, key, recipe)
        return recipe

    def bind_call(self, call_id, spec, pool, rank, resource, source, dependencies):
        """Reuse validated per-node compiled call fields; always bind the DAG."""
        key = (id(spec), pool, rank, resource)
        if key in self._call_templates:
            template = self._call_templates[key]
            if template.kernel_spec is not spec:
                raise RuntimeError("compiled call recipe identity collision")
            self._call_templates.move_to_end(key)
            self.call_template_hits += 1
            return template.bind_source_call(call_id, source, dependencies)
        call = KernelCall(call_id, spec, pool, rank, resource, (source,), dependencies)
        self._retain_source_value(self._call_templates, key, call)
        self.call_template_misses += 1
        return call

    def module_contract(self, node, lineage):
        """Compile an exact physical class from recorded source write facts.

        Source ports and annotations remain current on the complete graph.
        Only this owner's existing KernelSpec projection is shared. Opaque
        physical fields take the ordinary serializer path on every call.
        """
        if lineage is None:
            return _logical_node_contract(node)
        parent, shapes, attributes = lineage
        from src.serving.workload.traced_forward import _ForwardModuleFrame
        if "_module_source_frame" not in parent.__dict__:
            return _logical_node_contract(node)
        reference = parent.__dict__["_module_source_frame"]
        frame = reference() if type(reference) is weakref.ReferenceType else None
        if (type(frame) is not _ForwardModuleFrame or parent.node_id not in frame.node_templates
                or frame.node_templates[parent.node_id] is not parent):
            return _logical_node_contract(node)
        ignored = _source_object_attributes(node)
        if is_heterogeneous_moe(node):
            ignored = ignored | _HETEROGENEOUS_SOURCE_BINDING_ATTRIBUTES
        # Fused/collective provenance has its own graph validation/recipes.
        if node.kind in (WorkloadKind.FUSED_MOE, WorkloadKind.COLLECTIVE):
            return _logical_node_contract(node)
        physical = tuple((name, node.attributes[name]) for name in attributes if name not in ignored)
        geometry = tuple((name, node.shape[name]) for name in shapes)
        if not self._immutable_source_value((geometry, physical)):
            return _logical_node_contract(node)
        # Python tuple equality merges bool/int/float and signed zero. Keep
        # the ordinary exact contract's type/bit distinctions in bound writes.
        key = (id(parent), pickle.dumps((geometry, physical), protocol=5), is_heterogeneous_moe(node))
        if key in self._module_contracts:
            reference, contract = self._module_contracts[key]
            if reference() is parent:
                self._module_contracts.move_to_end(key)
                self.module_contract_hits += 1
                return contract
            del self._module_contracts[key]
        if (not self._immutable_source_value(tuple(parent.shape.items()))
                or any(not self._immutable_source_value(value) for name, value in parent.attributes.items()
                    if name not in ignored)):
            return _logical_node_contract(node)
        contract = _logical_node_contract(node)
        self._retain_source_value(self._module_contracts, key, (weakref.ref(parent), contract))
        self.module_contract_misses += 1
        return contract

    def _retain_source_value(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._max_entries:
            cache.popitem(last=False)

    @staticmethod
    def _immutable_source_value(value):
        if type(value) in (type(None), bool, int, float, str, bytes):
            return True
        return type(value) is tuple and all(KernelLoweringMemo._immutable_source_value(item) for item in value)

    def logical_contract(self, node):
        """Reuse a source contract only for deeply immutable physical inputs.

        Canonical WorkloadNode owns its copied, read-only shape. Metadata
        annotations and alpha-renamed IDs may change while all physical input
        objects remain identical. Keep those referents in this bounded run-local
        cache; mutable nested attributes always use the ordinary serializer.
        """
        if type(node) is not WorkloadNode or node.kind is WorkloadKind.FUSED_MOE:
            return _logical_node_contract(node)
        identity = id(node)
        if identity in self._source_node_contracts:
            reference, contract = self._source_node_contracts[identity]
            retained = reference()
            if retained is node:
                self._source_node_contracts.move_to_end(identity)
                return contract
            if retained is not None:
                raise RuntimeError("source node identity collision")
            del self._source_node_contracts[identity]
        values = [getattr(node, name) for name in _SOURCE_CONTRACT_FIELDS]
        if node.kind is WorkloadKind.GATHER and node.variant == "embedding" and "embedding_source" in node.attributes:
            values.append(node.rank)
        referents = tuple(values)
        key = tuple(id(value) for value in referents)
        ignored = _source_object_attributes(node)
        if key in self._source_contracts:
            _, expected, contract = self._source_contracts[key]
            position = 0
            for name, value in node.attributes.items():
                if name in ignored or (node.kind is WorkloadKind.COLLECTIVE and name == "moe_invocation_id"):
                    continue
                if (position == len(expected) or name is not expected[position][0]
                        or value is not expected[position][1]):
                    break
                position += 1
            else:
                if position == len(expected):
                    self._source_contracts.move_to_end(key)
                    self._retain_source_value(self._source_node_contracts, identity, (weakref.ref(node), contract))
                    return contract
        physical = tuple((name, value) for name, value in node.attributes.items()
            if name not in ignored
            and not (node.kind is WorkloadKind.COLLECTIVE and name == "moe_invocation_id"))
        # Only a miss needs to prove immutability. Every hot-key referent
        # is already kept alive by the successful proof's cache entry.
        if (type(node.shape) is not MappingProxyType
                or any(type(name) is not str or type(size) is not int for name, size in node.shape.items())
                or type(node.kind) is not WorkloadKind
                or any(value is not node.shape and value is not node.kind
                    and not self._immutable_source_value(value) for value in referents)
                or any(type(name) is not str or not self._immutable_source_value(value)
                    for name, value in physical)):
            return _logical_node_contract(node)
        contract = _logical_node_contract(node)
        self._retain_source_value(self._source_contracts, key, (referents, physical, contract))
        # The weak fast path never pins excluded annotations or a trace owner.
        self._retain_source_value(self._source_node_contracts, identity, (weakref.ref(node), contract))
        return contract

    def _source_binding_contract_layout(self, nodes, bindings):
        """Prove native physical contracts once; never retain source nodes."""
        contracts, by_name = {}, {}
        for node in nodes:
            if node.kind is WorkloadKind.FUSED_MOE:
                continue
            if (type(node) is not WorkloadNode or type(node.kind) is not WorkloadKind
                    or type(node.shape) is not MappingProxyType
                    or any(type(key) is not str or type(value) is not int for key, value in node.shape.items())
                    or any(not self._immutable_source_value(value)
                        for value in (node.pool_name, node.dtype, node.variant, node.rank))):
                return None
            ignored = _source_object_attributes(node)
            physical = tuple((key, value) for key, value in node.attributes.items()
                if key not in ignored
                and not (node.kind is WorkloadKind.COLLECTIVE and key == "moe_invocation_id"))
            if any(type(key) is not str or not self._immutable_source_value(value)
                    for key, value in physical):
                return None
            contracts[node.node_id] = _logical_node_contract(node)
            by_name[node.node_id] = node
        grouped = {}
        for target, origin, shapes, attributes in bindings:
            if target not in by_name:
                continue  # MoE remains on its ordinary current-record path.
            node = by_name[target]
            ignored = _source_object_attributes(node)
            physical = tuple(name for name in attributes if name not in ignored
                and not (node.kind is WorkloadKind.COLLECTIVE and name == "moe_invocation_id"))
            if not shapes and not physical:
                continue
            # Equal complete base contracts prove every unchanged physical
            # field, including rank-sensitive embedding and arbitrary scalars.
            # The actual source recipe proves the same origin and same writes.
            key = (contracts[target], origin, shapes, physical)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(target)
        groups = tuple((tuple(targets), shapes, attributes)
            for (_, _, shapes, attributes), targets in grouped.items())
        return contracts, groups

    def source_bound_contracts(self, graph, *, source_layout=_SOURCE_LAYOUT_UNSET):
        """Use intrinsic current-node lineage plus a lower-owned physical proof."""
        from src.serving.workload.traced_forward import _source_bound_node_layout
        layout = _source_bound_node_layout(graph) if source_layout is _SOURCE_LAYOUT_UNSET else source_layout
        if layout is None:
            return None
        token, nodes, bindings = layout
        template = token()
        if template is None:
            return None
        key = id(template)
        entry = self._source_binding_contracts[key] if key in self._source_binding_contracts else None
        if entry is not None and entry[0]() is template and entry[1] == bindings:
            state = entry[2]
            self._source_binding_contracts.move_to_end(key)
        else:
            state = self._source_binding_contract_layout(nodes, bindings)
            # A live weak referent, not an ID alone, prevents template-ID reuse.
            # Native recipe content is part of validity: one program may change
            # its actual template or writes between legitimate invocations.
            self._source_binding_contracts[key] = (token, bindings, state)
            if len(self._source_binding_contracts) > min(self._max_entries, 128):
                self._source_binding_contracts.popitem(last=False)
        if state is None:
            return None
        contracts, groups = state
        result = dict(contracts)
        for targets, shapes, attributes in groups:
            node = graph.node(targets[0])
            if (type(node) is not WorkloadNode
                    or any(type(node.shape[name]) is not int for name in shapes)
                    or any(not self._immutable_source_value(node.attributes[name]) for name in attributes)):
                return None
            contract = _logical_node_contract(node)
            for target in targets:
                result[target] = contract
        return result

    def source_moe_physical_contracts(self, graph, source_layout):
        """Group complete physical projections inside a checked native EP call.

        The caller has checked actual source lineage and every current EP
        record. Only static native bytes survive this call; current routing
        identities and contracts are grouped in a fresh local dictionary.
        """
        token, nodes, bindings = source_layout
        template = token()
        if template is None:
            return None
        identity = id(template)
        entry = self._source_moe_static_contracts[identity] if identity in self._source_moe_static_contracts else None
        if entry is not None and entry[0]() is template and entry[1] == bindings:
            bases = entry[2]
            self._source_moe_static_contracts.move_to_end(identity)
        else:
            bases = {}
            for node in nodes:
                if node.kind is not WorkloadKind.FUSED_MOE:
                    continue
                physical = tuple((name, value) for name, value in node.attributes.items()
                    if name not in _SOURCE_OBJECT_ATTRIBUTES and name not in _EP_SOURCE_DYNAMIC_FIELDS)
                if (type(node) is not WorkloadNode or type(node.kind) is not WorkloadKind
                        or type(node.shape) is not MappingProxyType
                        or any(type(name) is not str or type(value) is not int for name, value in node.shape.items())
                        or not self._immutable_source_value(physical)
                        or not self._immutable_source_value((node.pool_name, node.dtype, node.variant))):
                    return None
                bases[node.node_id] = _logical_node_contract(node, ((), ()))
            self._source_moe_static_contracts[identity] = (token, bindings, bases)
            if len(self._source_moe_static_contracts) > min(self._max_entries, 128):
                self._source_moe_static_contracts.popitem(last=False)
        groups, contracts = {}, {}
        for node in graph.nodes:
            if node.kind is not WorkloadKind.FUSED_MOE:
                continue
            counts = node.attributes["tokens_per_expert"]
            local = node.attributes["local_tokens_per_expert"]
            key = (bases[node.node_id], id(counts), id(local))
            if key not in groups:
                groups[key] = (counts, local, _logical_node_contract(node))
            retained_counts, retained_local, contract = groups[key]
            if retained_counts is not counts or retained_local is not local:
                raise RuntimeError("synchronous MoE contract identity collision")
            contracts[node.node_id] = contract
        return contracts

    def compact_source_layers(self, graph, contracts):
        from src.serving.workload.workload_graph import (
            compact_source_layer_repeats, _SourceLayerCompaction,
            _SOURCE_REPEAT_TOPOLOGY_CACHE_SIZE)
        # Ordinary unbound graphs retain the original cold path. with_nodes
        # proves the binding reused an already validated complete topology.
        if type(graph) is not WorkloadGraph or not graph._is_node_binding:
            return compact_source_layer_repeats(graph, contracts)
        identity = id(graph._predecessors_by_id)
        if identity in self._source_compactions:
            layout = self._source_compactions[identity]
            bound = layout.bind(graph, contracts)
            if bound is not None:
                self._source_compactions.move_to_end(identity)
                return bound
            del self._source_compactions[identity]
        compact = compact_source_layer_repeats(graph, contracts)
        if (not graph.block_repeats and compact.block_repeats
                and len(graph.nodes) == len(contracts)
                and all(node.node_id in contracts for node in graph.nodes)):
            self._source_compactions[identity] = _SourceLayerCompaction(graph, contracts, compact)
            if len(self._source_compactions) > min(self._max_entries, _SOURCE_REPEAT_TOPOLOGY_CACHE_SIZE):
                self._source_compactions.popitem(last=False)
        return compact

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def entry_count(self) -> int:
        return len(self._specs)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def lower(
        self,
        node: WorkloadNode,
        logical_node_contract,
        hardware,
        build_specs: Callable[[], tuple[KernelSpec, ...]],
    ) -> tuple[KernelSpec, ...]:
        if not isinstance(node, WorkloadNode):
            raise TypeError("kernel lowering memo requires WorkloadNode")
        hardware_identity = id(hardware)
        if hardware_identity in self._hardware_objects:
            if self._hardware_objects[hardware_identity] is not hardware:
                raise RuntimeError(
                    "kernel lowering memo hardware identity collision")
        else:
            self._hardware_objects[hardware_identity] = hardware
        if not isinstance(logical_node_contract, tuple):
            raise TypeError(
                "kernel lowering memo requires a frozen logical node "
                "contract")
        try:
            hash(logical_node_contract)
        except TypeError as exc:
            raise TypeError(
                "kernel lowering memo requires a hashable logical node "
                "contract") from exc
        # KernelSpec-relevant shard identity is retained in the contract.
        # Other placement and graph dependencies remain on KernelCall.
        key = (logical_node_contract, hardware_identity)
        if key in self._specs:
            self._specs.move_to_end(key)
            self._hit_count += 1
            return self._specs[key]
        specs = build_specs()
        if not isinstance(specs, tuple):
            raise TypeError("kernel lowering memo requires a KernelSpec tuple")
        if any(not isinstance(spec, KernelSpec) for spec in specs):
            raise TypeError(
                "kernel lowering memo entries must contain only KernelSpec")
        self._specs[key] = specs
        if len(self._specs) > self._max_entries:
            self._specs.popitem(last=False)
        self._miss_count += 1
        return specs


def _target_model_semantics_contract(
    graph: WorkloadGraph,
) -> str | None:
    contracts = tuple(
        node.attributes["model_semantics_contract"]
        for node in graph.nodes
        if "model_semantics_contract" in node.attributes
    )
    if not contracts:
        return None
    if len(contracts) != 1:
        raise ValueError(
            "target-model workload must declare exactly one semantics "
            "contract")
    contract = contracts[0]
    if contract not in (
            "kimi_k2_6_vllm_0_27_1_logical_v1", "kimi_k2_7_code_vllm_0_30_0_logical_v1",
            "glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1",
            "deepseek_v3_vllm_0_27_1_logical_v1",
            "deepseek_v4_vllm_0_30_0_logical_v1"):
        raise ValueError(
            f"unknown target-model semantics contract: {contract!r}")
    return contract


def _native_graph_routing_certificates(nodes):
    """One closed batch of current canonical node ledgers; False uses ordinary checks."""
    from src.serving.workload.runtime_inputs import _routing_certificate_contexts_native
    records = []
    for node in nodes:
        if type(node) is not WorkloadNode:
            return False
        if node.kind is not WorkloadKind.FUSED_MOE:
            continue
        if type(node.shape) is not MappingProxyType or type(node.attributes) is not MappingProxyType:
            return False
        attrs = node.attributes
        if (any(name not in attrs for name in ("routing_provenance", "tokens_per_expert", "moe_invocation_id"))
                or any(name not in node.shape for name in ("tokens", "experts", "topk"))):
            return False
        present = tuple(name for name in _GRAPH_TOKEN_DOMAIN_FIELDS if name in attrs)
        if present and (present != _GRAPH_TOKEN_DOMAIN_FIELDS
                or type(attrs["graph_logical_tokens"]) is not int
                or type(attrs["graph_model_input_tokens"]) is not int
                or type(attrs["graph_runtime_mode"]) is not str):
            return False
        domain = _graph_token_domain(node)
        tokens = node.shape["tokens"]
        if domain is not None:
            if tokens != domain[1]:
                return False
            tokens = domain[0]
        records.append((attrs["routing_provenance"], attrs["tokens_per_expert"], tokens,
            node.shape["experts"], node.shape["topk"], attrs["moe_invocation_id"]))
    return _routing_certificate_contexts_native(tuple(records))


def _validate_routing_provenance(node: WorkloadNode) -> Mapping[str, object]:
    if "routing_provenance" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks routing_provenance")
    if "tokens_per_expert" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks tokens_per_expert")
    if "moe_invocation_id" not in node.attributes:
        raise ValueError(
            f"FUSED_MOE node {node.node_id!r} lacks moe_invocation_id")
    routing_tokens = node.shape["tokens"]
    domain = _graph_token_domain(node)
    if domain is not None:
        logical_tokens, model_input_tokens, _ = domain
        if routing_tokens != model_input_tokens:
            raise ValueError(
                "FUSED_MOE model-input tokens disagree with its graph "
                "token-domain contract")
        routing_tokens = logical_tokens
    return validate_expert_routing_provenance(
        node.attributes["routing_provenance"],
        tokens_per_expert=tuple(node.attributes["tokens_per_expert"]),
        tokens=routing_tokens,
        num_experts=node.shape["experts"],
        topk=node.shape["topk"],
        moe_invocation_id=node.attributes["moe_invocation_id"],
    )


def _source_linear_data_inputs(node):
    inputs = node.attributes["symbolic_tensor_inputs"]
    if "symbolic_parameter_inputs" not in node.attributes:
        return inputs
    parameters = node.attributes["symbolic_parameter_inputs"]
    if (type(parameters) is not tuple or len(parameters) != 1 or len(inputs) != 2
            or tuple(inputs[1:]) != parameters or len(set(inputs)) != 2):
        raise ValueError("virtual EP source linear gate requires one data port and one explicit readonly weight port")
    return (inputs[0],)




def _qwen030_dp_sp(config):
    return (config.data_parallel_size > 1 and config.tensor_parallel_size >= 1
        and config.expert_parallel_size == (config.tensor_parallel_size * config.data_parallel_size if config.enable_expert_parallel else 1)
        and config.sequence_parallel_size in ((1, config.tensor_parallel_size) if config.enable_expert_parallel else (1,))
        and config.context_parallel_size == 1)


def _validate_distributed_ep_gate(graph, node, logits_gather, gate_outputs, expert_inputs):
    """Each gathered router shard consumes its matching gathered hidden shard."""
    rows = node.attributes["source_dp_lane_rows"]
    logical = node.attributes["source_dp_logical_rows"]
    tp = node.attributes["source_tensor_parallel_size"]
    ep = node.attributes["expert_parallel_size"] > 1
    sp = ep and tp > 1
    if (not isinstance(rows, tuple) or not isinstance(logical, tuple) or len(logical) < 2
            or any(type(value) is not int or value < 0 for value in (*rows, *logical))
            or type(tp) is not int or tp < 1
            or (ep and len(rows) != node.attributes["expert_parallel_size"])):
        raise ValueError("distributed MoE requires complete logical/lane row geometry")
    expected = tuple(-(-value // tp) for value in logical for _ in range(tp)) if sp else logical
    order = "dp_then_tp_lane_then_row" if sp else "dp_then_row"
    if (rows != expected or sum(rows) != node.shape["tokens"]
            or node.attributes["model_execution_version"] != "vllm_0_30_0"
            or node.attributes["source_dp_row_order"] != order
            or node.attributes["source_padding_routing"] != "declared_routing_for_physical_rows_including_padding_not_observed_topk"):
        raise ValueError("distributed MoE padding and physical routing geometry disagree")
    candidates = tuple(graph.node(name) for name in graph.predecessors(node.node_id)
        if graph.node(name).kind is WorkloadKind.COLLECTIVE
        and expert_inputs[0] in graph.node(name).attributes["symbolic_tensor_outputs"])
    if len(candidates) != 1:
        raise ValueError("distributed EP experts require one actual gathered hidden-state input")
    hidden_gather, = candidates
    participants = logits_gather.attributes["participants"]
    for gather, width in ((hidden_gather, node.shape["hidden"]), (logits_gather, node.shape["experts"])):
        attributes = gather.attributes
        if (gather.kind is not WorkloadKind.COLLECTIVE or gather.pool_name != node.pool_name or gather.dtype != "bf16"
                or attributes["operation"] != "all_gatherv" or attributes["participants"] != participants
                or attributes["symbolic_variable_rows"] != rows or attributes["symbolic_variable_row_shape"] != (width,)
                or attributes["bytes"] != sum(rows) * width * 2
                or len(attributes["symbolic_tensor_outputs"]) != len(rows)
                or len(attributes["symbolic_tensor_inputs"]) != sum(bool(value) for value in rows)):
            raise ValueError("distributed EP gather must preserve the complete ordered participant row extents")
    if node.rank not in participants:
        raise ValueError("distributed MoE rank is outside its actual dispatch group")
    rank_index = participants.index(node.rank)
    if (len(participants) != len(rows) or participants[rank_index] != node.rank
            or gate_outputs != (logits_gather.attributes["symbolic_tensor_outputs"][rank_index],)
            or expert_inputs[0] != hidden_gather.attributes["symbolic_tensor_outputs"][rank_index]):
        raise ValueError("distributed EP routing must bind this rank's gathered public tensor views")
    gates = tuple(graph.node(edge.source) for edge in graph.edges
        if edge.destination == logits_gather.node_id and edge.kind == "data")
    active = tuple((rank, count) for rank, count in zip(participants, rows, strict=True) if count)
    if len(gates) != len(active):
        raise ValueError("distributed EP logits require one real gate per nonempty lane")
    for (rank, count), hidden_port, logits_port in zip(active,
            hidden_gather.attributes["symbolic_tensor_inputs"], logits_gather.attributes["symbolic_tensor_inputs"], strict=True):
        matches = tuple(gate for gate in gates if gate.rank == rank)
        if len(matches) != 1:
            raise ValueError("distributed EP router gates must preserve physical participant ownership")
        gate, = matches
        if (gate.kind is not WorkloadKind.ROUTER or gate.pool_name != node.pool_name or gate.dtype != "bf16"
                or gate.variant != "gemm" or dict(gate.shape) != {"b": 1, "m": count, "n": node.shape["experts"], "k": node.shape["hidden"]}
                or _source_linear_data_inputs(gate) != (hidden_port,)
                or gate.attributes["symbolic_tensor_outputs"] != (logits_port,)):
            raise ValueError("distributed EP router shards must consume the corresponding actual hidden segments")
    outputs = node.attributes["symbolic_tensor_outputs"]
    combines = tuple(graph.node(edge.destination) for edge in graph.edges
        if edge.source == node.node_id and edge.kind == "data" and graph.node(edge.destination).kind is WorkloadKind.COLLECTIVE
        and outputs[0] in graph.node(edge.destination).attributes["symbolic_tensor_inputs"])
    if len(combines) != 1:
        raise ValueError("distributed MoE expert output must enter one matching sum-scatter")
    combine, = combines
    attrs = combine.attributes
    if (attrs["operation"] != "reduce_scatterv" or attrs["participants"] != participants
            or attrs["reduction"] != "sum" or attrs["symbolic_variable_rows"] != rows
            or attrs["symbolic_variable_row_shape"] != (node.shape["hidden"],)
            or attrs["moe_invocation_id"] != node.attributes["moe_invocation_id"]
            or attrs["symbolic_tensor_inputs"][rank_index] != outputs[0]
            or len(attrs["symbolic_tensor_outputs"]) != sum(bool(value) for value in rows)):
        raise ValueError("distributed MoE combine must retain actual participant slices and expert output identities")

def _validate_virtual_ep_router_inputs(graph, node):
    """Bind the separate gate/route outputs to this rank's EP experts."""
    if "symbolic_tensor_inputs" not in node.attributes:
        raise ValueError("virtual EP experts omitted their source tensor inputs")
    inputs = node.attributes["symbolic_tensor_inputs"]
    if "memory_input_roles" in node.attributes:
        empty = "symbolic_zero_extent_parameter_roles" in node.attributes
        if empty and (node.attributes["symbolic_zero_extent_parameter_roles"] != ("w13_weight", "w2_weight")
                or node.attributes["expert_parallel_size"] <= node.shape["experts"]
                or node.attributes["owned_experts"] or node.attributes["model_execution_version"] != "vllm_0_30_0"
                or "symbolic_parameter_inputs" in node.attributes):
            raise ValueError("empty expert rank cannot invent parameters or nonempty ownership")
        roles = (("hidden_states", "routing_weights", "expert_ids") if empty else
            ("hidden_states", "routing_weights", "expert_ids", "w13_weight", "w2_weight"))
        if (node.attributes["memory_input_roles"] != roles or len(inputs) != len(roles)
                or len(set(inputs)) != len(inputs)
                or node.attributes["memory_output_roles"] != ("output",)
                or (not empty and node.attributes["symbolic_parameter_inputs"] != tuple(inputs[3:]))
                or node.attributes["expert_parameter_layout"] != "unquantized_checkpoint_expert_major_nk"
                or node.attributes["model_execution_version"] not in (
                    "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0")):
            raise ValueError("virtual EP experts require exact declared public tensor/parameter roles")
        # Only the three data ports are router outputs; the two independently
        # declared readonly parameter ports remain in the actual source node.
        inputs = tuple(inputs[index] for index in range(3))
    elif len(inputs) != 3 or len(set(inputs)) != 3:
        raise ValueError("virtual EP experts require distinct hidden/weight/ID inputs")
    predecessors = tuple(graph.node(name) for name in graph.predecessors(node.node_id))
    kimi = ("source_component" in node.attributes and "routing_source_stage" in node.attributes
        and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
        and node.attributes["routing_source_stage"] == "kimi_moe_route")
    glm = ("source_component" in node.attributes and "routing_source_stage" in node.attributes
        and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
        and node.attributes["routing_source_stage"] == "dsv32_moe_route")
    routes = tuple(producer for producer in predecessors
        if producer.kind is WorkloadKind.VECTOR and producer.variant == (
            "kimi_moe_route" if kimi else "dsv32_moe_route" if glm else "moe_topk"))
    masks = tuple(producer for producer in routes if "source_component" in producer.attributes
        and producer.attributes["source_component"] == "symbolic_expert_owner_mask")
    if masks:
        routes = masks
    if len(routes) != 1:
        raise ValueError("virtual EP experts require exactly one rank-local routing producer")
    route = routes[0]
    if ("source_component" in route.attributes
            and route.attributes["source_component"] == "symbolic_expert_owner_mask"):
        mask = route
        mask_inputs = mask.attributes["symbolic_tensor_inputs"]
        mask_outputs = mask.attributes["symbolic_tensor_outputs"]
        if (mask.rank != node.rank or mask.pool_name != node.pool_name or mask.dtype != "bf16"
                or mask.attributes["source_program"] != "vllm_0_18_bf16_softmax_topk"
                or mask.attributes["stage"] != "owner_mask" or len(mask_inputs) != 3 or len(mask_outputs) != 3
                or mask.attributes["expert_rank"] != node.attributes["expert_rank"]
                or mask_outputs[0] != inputs[1] or mask_inputs[1] != inputs[2]
                or node.attributes["routing_weight_dtype"] != "bf16" or mask.shape != node.shape):
            raise ValueError("Ascend EP expert owner mask disagrees with its hidden/weight/ID bindings")
        routes = tuple(graph.node(name) for name in graph.predecessors(mask.node_id)
            if graph.node(name).variant == "moe_topk"
            and graph.node(name).attributes["source_component"] == "symbolic_softmax_topk")
        if len(routes) != 1:
            raise ValueError("Ascend owner mask requires exactly one raw routing producer")
        route = routes[0]
        if tuple(route.attributes["symbolic_tensor_outputs"]) != tuple(mask_inputs[:2]):
            raise ValueError("Ascend owner mask must consume that rank's raw BF16 weights and IDs")
    if kimi or glm:
        if "symbolic_tensor_outputs" not in node.attributes or len(node.attributes["symbolic_tensor_outputs"]) != 1:
            raise ValueError("Kimi EP experts require one explicit routed output")
        expected = {"source_component": "vllm_0_30_symbolic_grouped_routing", "expert_id_selection": "score_topk",
            "routed_scaling_factor": 1.0 if kimi else 2.5, "score_function": "sigmoid", "expert_group_count": 1,
            "topk_group_count": 1, "normalize_topk_probability": True,
            "mathematical_program": "biased_sigmoid_topk_flops_bytes_v1"}
        if (set(expected) - set(route.attributes)
                or any(route.attributes[key] != value for key, value in expected.items())
                or {"symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(route.attributes)
                or route.rank != node.rank or route.pool_name != node.pool_name or route.dtype != ("bf16" if kimi else "fp32")
                or dict(route.shape) != {"elements": node.shape["tokens"] * node.shape["experts"],
                    "rows": node.shape["tokens"], "experts": node.shape["experts"], "topk": node.shape["topk"]}
                or tuple(route.attributes["symbolic_tensor_outputs"]) != tuple(inputs[1:])
                or len(route.attributes["symbolic_tensor_inputs"]) != 2):
            raise ValueError("Kimi EP experts disagree with their rank-local grouped routing")
        gate_ids = graph.predecessors(route.node_id)
        if len(gate_ids) != 1:
            raise ValueError("Kimi EP routing requires exactly one gate projection")
        gate = graph.node(gate_ids[0])
        if ({"symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(gate.attributes)
                or gate.kind is not WorkloadKind.ROUTER or gate.rank != node.rank or gate.pool_name != node.pool_name
                or gate.dtype != "bf16" or gate.variant != "gemm"
                or dict(gate.shape) != {"b": 1, "m": node.shape["tokens"], "n": node.shape["experts"], "k": node.shape["hidden"]}
                or tuple(_source_linear_data_inputs(gate)) != (inputs[0],)
                or tuple(gate.attributes["symbolic_tensor_outputs"]) != tuple(route.attributes["symbolic_tensor_inputs"][:1])
                or (glm and ("output_dtype" not in gate.attributes or gate.attributes["output_dtype"] != "fp32"))
                or route.attributes["symbolic_tensor_inputs"][1] in (*inputs, *gate.attributes["symbolic_tensor_outputs"])):
            raise ValueError("Kimi EP gate must bind the same hidden input and one separate correction bias")
        return
    if ({"source_component", "source_program", "symbolic_tensor_inputs", "symbolic_tensor_outputs"}
            - set(route.attributes) or {"tokens", "experts", "topk", "hidden"} - set(route.shape)):
        raise ValueError("virtual EP routing omitted its source program or tensor ports")
    source011 = route.attributes["source_program"] == "vllm_0_11_fp32_softmax_topk"
    source018 = route.attributes["source_program"] == "vllm_0_18_bf16_softmax_topk"
    if source018 and (len(masks) != 1 or route.attributes["stage"] != "route"):
        raise ValueError("Ascend EP routing must bind its raw route through the owner mask")
    if route.attributes["source_program"] not in ("vllm_0_11_fp32_softmax_topk", "vllm_0_18_bf16_softmax_topk", "vllm_0_24_bf16_softmax_topk", "vllm_0_30_bf16_softmax_topk"):
        raise ValueError("virtual EP routing requires its version-bound softmax program")
    if (route.rank != node.rank or route.pool_name != node.pool_name or route.dtype != ("fp32" if source011 else "bf16")
            or route.attributes["source_component"] != "symbolic_softmax_topk"
            or (not source018 and tuple(route.attributes["symbolic_tensor_outputs"][:2]) != tuple(inputs[1:]))
            or any(route.shape[key] != node.shape[key] for key in ("tokens", "experts", "topk", "hidden"))):
        raise ValueError("virtual EP experts disagree with rank-local routing tensor bindings")
    gate_ids = graph.predecessors(route.node_id)
    if len(gate_ids) != 1:
        raise ValueError("virtual EP routing requires exactly one gate projection")
    gate = graph.node(gate_ids[0])
    gate_outputs = route.attributes["symbolic_tensor_inputs"]
    if source011:
        cast = gate
        if ({"operation", "reads", "writes", "symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(cast.attributes)
                or "mathematical_program" not in route.attributes
                or route.attributes["mathematical_program"] != "fp32_softmax_topk_flops_bytes_v1"
                or cast.kind is not WorkloadKind.ELEMENTWISE or cast.variant != "elementwise" or cast.dtype != "bf16"
                or cast.rank != node.rank or cast.pool_name != node.pool_name
                or dict(cast.shape) != {"elements": node.shape["tokens"] * node.shape["experts"]}
                or cast.attributes["operation"] != "cast_bf16_float32"
                or type(cast.attributes["reads"]) is not int or type(cast.attributes["writes"]) is not int
                or cast.attributes["reads"] != 1 or cast.attributes["writes"] != 1
                or len(cast.attributes["symbolic_tensor_inputs"]) != 1 or len(cast.attributes["symbolic_tensor_outputs"]) != 1
                or tuple(cast.attributes["symbolic_tensor_outputs"]) != tuple(gate_outputs)
                or cast.attributes["symbolic_tensor_outputs"] == cast.attributes["symbolic_tensor_inputs"]):
            raise ValueError("virtual EP0.11 routing requires its explicit rank-local BF16-to-FP32 cast")
        gate_ids = graph.predecessors(cast.node_id)
        if len(gate_ids) != 1:
            raise ValueError("virtual EP0.11 routing cast requires exactly one gate projection")
        gate = graph.node(gate_ids[0])
        gate_outputs = cast.attributes["symbolic_tensor_inputs"]
    if {"symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(gate.attributes):
        raise ValueError("virtual EP gate omitted its source tensor ports")
    if "source_dp_lane_rows" in node.attributes:
        _validate_distributed_ep_gate(graph, node, gate, gate_outputs, inputs)
        return
    if (gate.kind is not WorkloadKind.ROUTER or gate.rank != node.rank or gate.pool_name != node.pool_name
            or gate.dtype != "bf16" or gate.variant != "gemm"
            or gate.shape != {"b": 1, "m": node.shape["tokens"], "n": node.shape["experts"], "k": node.shape["hidden"]}
            or _source_linear_data_inputs(gate) != (inputs[0],)
            or gate.attributes["symbolic_tensor_outputs"] != gate_outputs):
        raise ValueError("virtual EP router gate must consume the same hidden states as experts")


_EP_SOURCE_VALIDATION_LAYOUTS = weakref.WeakKeyDictionary()
_EP_SOURCE_DYNAMIC_FIELDS = frozenset(('tokens_per_expert', 'local_tokens_per_expert',
    'moe_invocation_id', 'routing_provenance'))


def _ep_native_value(value):
    return type(value) in (str,int,float,bool,bytes,type(None)) or (type(value) is tuple and all(_ep_native_value(item) for item in value))

def _ep_deployment_projection(deployment):
    from src.serving.deployment.deployment_plan import DeploymentPlan,WorkerPool
    from src.serving.deployment.deployment_config import CollectiveConfig
    from src.serving.parallelism.parallel_plan import ParallelPlan
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.parallelism.rank_groups import RankGroups
    if type(deployment) is not DeploymentPlan:return None
    if any(type(pool) is not WorkerPool or type(pool.parallel_plan) is not ParallelPlan
            or type(pool.parallel_plan.config) is not ParallelConfig
            or type(pool.parallel_plan.rank_groups) is not RankGroups
            or any(type(c) is not CollectiveConfig for c in pool.collectives)
            for pool in deployment.worker_pools.values()):return None
    # Only fields read by the actual TP2/EP2 validator; no KV/runtime objects.
    values=(deployment.mode,tuple((name,pool.name,pool.role,pool.hardware_id,pool.ranks,
        tuple(vars(pool.parallel_plan.config).items()),tuple(vars(pool.parallel_plan.rank_groups).items()),
        pool.parallel_plan.expert_to_shard,tuple(tuple(vars(c).items()) for c in pool.collectives))
        for name,pool in deployment.worker_pools.items()))
    return values if _ep_native_value(values) else None

def _ep_static_node(node):
    return (node.kind in (WorkloadKind.FUSED_MOE,WorkloadKind.ROUTER,WorkloadKind.COLLECTIVE)
        or (node.kind is WorkloadKind.VECTOR and node.variant=='moe_topk')
        or (node.kind is WorkloadKind.ELEMENTWISE and 'operation' in node.attributes and node.attributes['operation']=='cast_bf16_float32'))

def _ep_dynamic_fields(node):return _EP_SOURCE_DYNAMIC_FIELDS if node.kind is WorkloadKind.FUSED_MOE else frozenset(('moe_invocation_id',)) if node.kind is WorkloadKind.COLLECTIVE else frozenset()

def _ep_verifier_functions():
    """The actual scoped EP verifier's implementation dependencies."""
    from types import FunctionType
    from src.serving.deployment.deployment_plan import WorkerPool
    from src.serving.parallelism.rank_groups import RankGroups
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.kernel.hardware_ids import canonical_hardware_id
    from src.serving.deployment.deployment_config import _normalize_worker_role
    names = ("_validate_ep_fused_moe_graph", "_validate_virtual_ep_router_inputs",
        "_graph_token_domain", "_symbolic_glm_pipeline_projection", "_symbolic_kimi_pipeline_projection",
        "_symbolic_kimi_parallel_projection",
        "_reuse_ep_native_frame_recipe", "_native_plan_binding_layout", "_explicit_source_expert_specs",
        "_flash_attn_specs", "_source_flash_attn_specs", "_fused_moe_specs", "_logical_node_contract", "_fused_moe_routing_relabelling",
        "_node_physical_attributes", "_source_object_attributes", "_native_lowering_group_key")
    if any(name not in globals() for name in names):
        return None
    functions = tuple(globals()[name] for name in names)
    for owner, methods in ((WorkerPool, ("logical_rank_for_physical_rank", "physical_rank_for_logical_rank", "collective_for")),
            (RankGroups, ("for_rank",)), (WorkloadGraph, ("node", "predecessors"))):
        if any(name not in vars(owner) for name in methods):
            return None
        functions += tuple(getattr(owner, name) for name in methods)
    for name in ("expert_parallel_size", "worker_count"):
        if name not in vars(ParallelConfig) or type(vars(ParallelConfig)[name]) is not property:
            return None
        functions += (vars(ParallelConfig)[name].fget,)
    functions += (HWSpec.source_contract_supported, canonical_hardware_id,
        worker_role_phase, worker_role_operator_group, _normalize_worker_role,
        KernelLoweringMemo.lower, KernelPlan.bind_kernel_specs, KernelCall.with_kernel_spec)
    if any(type(function) is not FunctionType for function in functions):
        return None
    return functions


def _ep_verifier_snapshot():
    functions = _ep_verifier_functions()
    if functions is None or any(not _ep_native_value(function.__defaults__)
            or function.__kwdefaults__ is not None for function in functions):
        return None
    return tuple((function, function.__code__, function.__defaults__, function.__kwdefaults__)
        for function in functions)


def _ep_verifier_unchanged(snapshot):
    functions = _ep_verifier_functions()
    return (functions is not None and len(functions) == len(snapshot)
        and all(function is original and function.__code__ is code
            and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
            for function, (original, code, defaults, kwdefaults) in zip(functions, snapshot, strict=True)))


def _ep_unpadded_source_rows(graph):
    from src.serving.workload.traced_forward import _ep_source_token_domains
    domains = _ep_source_token_domains(graph)
    if type(domains) is not tuple or len(domains) != 2:
        return None
    rows = {}
    padded = False
    for domain in domains:
        if (type(domain) is not tuple or len(domain) != 3
                or any(type(value) is not int for value in domain)):
            return None
        rank, logical, physical = domain
        if rank in rows or rank < 0 or logical <= 0 or logical > physical:
            return None
        padded = padded or logical != physical
        rows[rank] = physical
    if padded:
        from src.serving.workload.traced_forward import _NativeSourceFrame, _native_source_scope
        frame = graph.__dict__["_native_source_frame"] if "_native_source_frame" in graph.__dict__ else None
        owner = frame.owner(check_source=False) if type(frame) is _NativeSourceFrame else None
        if owner is None or not _native_source_scope(owner,owner.prefix): return None
        actual = []
        for trace in owner.source.traces:
            if len(trace.source_captures) != 1: return None
            capture = trace.source_captures[0]
            if capture["runtime_mode"] != "FULL" or capture["source_stage"] is not None: return None
            actual.append((trace.rank,capture["logical_tokens"],capture["model_input_tokens"]))
        # token_domains were copied from actual bound traces by the owner
        # issuer. The bound traces may already be released by Serving.finish.
        if tuple(actual) != domains: return None
        # This exact source routes its physical model rows. A node-local
        # domain changes slow-validator routing to logical rows and stays out.
        for name in frame.routing_ids:
            _, attrs = frame.fields(name)
            if any(field in attrs for field in _GRAPH_TOKEN_DOMAIN_FIELDS): return None
    return rows


def _ep_validation_layout(graph,deployment):
    from src.serving.workload.traced_forward import _source_bound_topology
    dep=_ep_deployment_projection(deployment)
    if dep is None or graph.block_repeats:return None
    experts=tuple(node for node in graph.nodes if node.kind is WorkloadKind.FUSED_MOE)
    producers=tuple(node for node in graph.nodes if node.kind is WorkloadKind.VECTOR and node.variant=='moe_topk')
    routes=tuple(node for node in producers if 'source_component' in node.attributes and node.attributes['source_component']=='symbolic_softmax_topk')
    masks=tuple(node for node in producers if 'source_component' in node.attributes and node.attributes['source_component']=='symbolic_expert_owner_mask')
    if (len(routes)+len(masks)!=len(producers)
            or (masks and (len(masks)!=len(routes) or any(node.attributes['source_program']!='vllm_0_18_bf16_softmax_topk'
                or node.attributes['stage']!='owner_mask' or not HWSpec.source_contract_supported('vllm_0_18_bf16_softmax_topk', deployment.worker_pools[node.pool_name].hardware_id) for node in masks)))
            or not experts or len(routes)!=len(experts) or any(type(node.shape['tokens']) is not int or node.shape['tokens']<=0
            or node.attributes['expert_parallel_size']!=2
            or 'source_component' not in node.attributes
            or node.attributes['source_component']!='symbolic_unquantized_experts' for node in experts)
            or any('source_program' not in node.attributes
                or not (node.attributes['source_program']=='vllm_0_11_fp32_softmax_topk'
                    or (node.attributes['source_program']=='vllm_0_30_bf16_softmax_topk'
                        and node.dtype == 'bf16'
                        and HWSpec.source_contract_supported('unbound_bf16_expert_source_math', deployment.worker_pools[node.pool_name].hardware_id))
                    or (node.attributes['source_program']=='vllm_0_24_bf16_softmax_topk'
                        and HWSpec.source_contract_supported('vllm_0_24_bf16_softmax_topk', deployment.worker_pools[node.pool_name].hardware_id))
                    or (node.attributes['source_program']=='vllm_0_18_bf16_softmax_topk'
                        and len(masks)==len(routes) and node.attributes['stage']=='route'
                        and HWSpec.source_contract_supported('vllm_0_18_bf16_softmax_topk', deployment.worker_pools[node.pool_name].hardware_id))) for node in routes)):
        return None
    rows = _ep_unpadded_source_rows(graph)
    if rows is None or set(rows) != {node.rank for node in experts}:
        return None
    for node in experts:
        domain = _graph_token_domain(node)
        if (node.shape["tokens"] != rows[node.rank]
                or (domain is not None and domain[0] != domain[1])):
            return None
    proof=[]
    for node in graph.nodes:
        if type(node) is not WorkloadNode or type(node.kind) is not WorkloadKind:return None
        if not _ep_static_node(node):proof.append((node.kind,None));continue
        if type(node.shape) is not MappingProxyType or not all(type(k) is str and type(v) is int for k,v in node.shape.items()):return None
        static=tuple((k,v) for k,v in node.attributes.items() if k not in _ep_dynamic_fields(node))
        if not _ep_native_value(static):return None
        fixed=(node.node_id,node.kind,node.pool_name,node.dtype,node.variant,node.rank,node.shape)
        if not _ep_native_value(fixed[:1]+fixed[2:6]):return None
        proof.append((node.kind,(node,fixed,static)))
    if any(type(edge) is not WorkloadEdge or not all(type(value) is str for value in (edge.source,edge.destination,edge.kind)) for edge in graph.edges):return None
    return dep,tuple(proof),_source_bound_topology(graph)[0]

_EP_SOURCE_RECIPE_LAYOUTS = weakref.WeakKeyDictionary()


def _record_ep_source_recipe(graph, owner, layout, *, source_layout=_SOURCE_LAYOUT_UNSET):
    """Prove source template and immutable recipe only after full EP success."""
    from src.serving.workload.traced_forward import _source_bound_node_layout
    fact = _source_bound_node_layout(graph) if source_layout is _SOURCE_LAYOUT_UNSET else source_layout
    if fact is None:
        return
    identity, template_nodes, bindings = fact
    template = identity() if type(identity) is weakref.ReferenceType else None
    if (type(template) is not WorkloadGraph or template.nodes is not template_nodes
            or type(template_nodes) is not tuple or not _ep_native_value(bindings)
            or len(template_nodes) != len(graph.nodes)):
        return
    _, _, proof, edges = layout
    dynamic = []
    selected = {}
    for index, (node, template, (kind, entry)) in enumerate(zip(graph.nodes, template_nodes, proof, strict=True)):
        if type(template) is not WorkloadNode or template.kind is not kind or template.node_id != node.node_id:
            return
        if entry is None:
            continue
        old, fixed, static = entry
        if (type(template.shape) is not MappingProxyType
                or any(type(name) is not str or type(size) is not int for name, size in template.shape.items())
                or any(actual is not expected for actual, expected in zip(
                    (template.node_id, template.kind, template.pool_name, template.dtype,
                        template.variant, template.rank, template.shape), fixed, strict=True))):
            return
        template_static = tuple((name, value) for name, value in template.attributes.items()
            if name not in _ep_dynamic_fields(template))
        if (not _ep_native_value(template_static) or len(template_static) != len(static)
                or any(name is not old_name or value is not old_value
                    for (name, value), (old_name, old_value) in zip(template_static, static, strict=True))):
            return
        selected[template.node_id] = template.kind
        if kind is WorkloadKind.FUSED_MOE or (kind is WorkloadKind.COLLECTIVE and "moe_invocation_id" in node.attributes):
            oldinv = old.attributes["moe_invocation_id"]
            owned = old.attributes["owned_experts"] if kind is WorkloadKind.FUSED_MOE else ()
            if type(oldinv) is not str or not _ep_native_value(owned):
                return
            dynamic.append((index, kind, owned, oldinv))
    for target, origin, shape_fields, attribute_fields in bindings:
        if target not in selected:
            continue
        kind = selected[target]
        if shape_fields:
            return
        allowed = _EP_SOURCE_DYNAMIC_FIELDS if kind is WorkloadKind.FUSED_MOE else (
            frozenset(("moe_invocation_id",)) if kind is WorkloadKind.COLLECTIVE else frozenset())
        if any(name not in allowed for name in attribute_fields):
            return
    _EP_SOURCE_RECIPE_LAYOUTS[owner] = (identity, bindings, tuple(dynamic), edges)


def _reuse_ep_source_recipe(graph, deployment, owner, layout, *, source_layout=_SOURCE_LAYOUT_UNSET):
    """Reuse only source-issued immutable structure; all current routing stays live."""
    from src.serving.workload.traced_forward import _source_bound_node_layout
    from src.serving.workload.runtime_inputs import _routing_certificate_context, _routing_certificate_contexts_native
    from src.serving.workload.traced_forward import _source_bound_node_sequence, _source_bound_topology
    nodes = _source_bound_node_sequence(graph)
    if owner not in _EP_SOURCE_RECIPE_LAYOUTS:
        return None
    fact = _source_bound_node_layout(graph) if source_layout is _SOURCE_LAYOUT_UNSET else source_layout
    if fact is None:
        return None
    identity, template_nodes, bindings = fact
    expected_template, expected_bindings, dynamic, expected_edges = _EP_SOURCE_RECIPE_LAYOUTS[owner]
    template = identity() if type(identity) is weakref.ReferenceType else None
    if (type(template) is not WorkloadGraph or expected_template() is not template
            or template.nodes is not template_nodes or bindings != expected_bindings):
        return None
    behavior, dep, proof, edges = layout
    if not _ep_verifier_unchanged(behavior):
        return False
    current_dep = _ep_deployment_projection(deployment)
    if (current_dep is None or current_dep != dep or _source_bound_topology(graph)[0] is not expected_edges
            or len(nodes) != len(template_nodes)):
        return False
    rows = _ep_unpadded_source_rows(graph)
    if rows is None:
        return False
    # Source lineage proves canonical nodes and readonly mappings. Build a
    # closed native batch; unknown field containers decline to the ordinary
    # per-record verifier below. Each local ownership check remains live.
    certificate_records = []
    for index, kind, owned, oldinv in dynamic:
        if kind is WorkloadKind.FUSED_MOE:
            node = nodes[index]
            attrs = node.attributes
            if _EP_SOURCE_DYNAMIC_FIELDS - set(attrs):
                return False
            certificate_records.append((attrs["routing_provenance"], attrs["tokens_per_expert"],
                node.shape["tokens"], node.shape["experts"], node.shape["topk"], attrs["moe_invocation_id"]))
    certificates_valid = _routing_certificate_contexts_native(tuple(certificate_records))
    invocations = {}
    new_names = set()
    # Run-local arithmetic proofs only. Every record still checks its own
    # certificate, invocation, rank and actual immutable local tuple.
    local_arithmetic = {}
    for index, kind, owned, oldinv in dynamic:
        node = nodes[index]
        attrs = node.attributes
        if node.kind is WorkloadKind.FUSED_MOE:
            if (node.rank not in rows or node.shape["tokens"] != rows[node.rank]
                    or _EP_SOURCE_DYNAMIC_FIELDS - set(attrs)):
                return False
            counts = attrs["tokens_per_expert"]
            local = attrs["local_tokens_per_expert"]
            inv = attrs["moe_invocation_id"]
            if type(local) is not tuple:
                return False
            if not certificates_valid and not _routing_certificate_context(attrs["routing_provenance"], counts,
                    node.shape["tokens"], node.shape["experts"], node.shape["topk"], inv):
                return False
            arithmetic_key = (id(counts), id(local), owned, node.rank,
                tuple(node.shape.items()))
            if arithmetic_key in local_arithmetic:
                retained_counts, retained_local = local_arithmetic[arithmetic_key]
                if retained_counts is not counts or retained_local is not local:
                    raise RuntimeError("synchronous local arithmetic identity collision")
            else:
                if any(type(count) is not int for count in local):
                    return False
                expected_local = [0] * node.shape["experts"]
                for expert in owned:
                    expected_local[expert] = counts[expert]
                if local != tuple(expected_local):
                    return False
                local_arithmetic[arithmetic_key] = (counts, local)
        else:
            inv = attrs["moe_invocation_id"]
            if type(inv) is not str or not inv:
                return False
        if oldinv in invocations:
            if invocations[oldinv] != inv:
                return False
        else:
            if inv in new_names:
                return False
            invocations[oldinv] = inv
            new_names.add(inv)
    return True


def _reuse_ep_source_validation(graph, deployment, *, source_layout=_SOURCE_LAYOUT_UNSET):
    from src.serving.workload.traced_forward import _source_bound_topology
    from src.serving.workload.traced_forward import _ep_source_owner
    from src.serving.workload.runtime_inputs import _routing_certificate_context
    owner = _ep_source_owner(graph, check_source=False)
    if owner is None or owner not in _EP_SOURCE_VALIDATION_LAYOUTS:
        return False
    layout = _EP_SOURCE_VALIDATION_LAYOUTS[owner]
    if layout is None:
        return False
    recipe_result = _reuse_ep_source_recipe(graph, deployment, owner, layout, source_layout=source_layout)
    if recipe_result is not None:
        return recipe_result
    behavior,dep,proof,edges = layout
    if _ep_source_owner(graph, check_source=True) is not owner or not _ep_verifier_unchanged(behavior):
        return False
    current_dep=_ep_deployment_projection(deployment)
    if current_dep is None or current_dep!=dep or _source_bound_topology(graph)[0] is not edges or len(graph.nodes)!=len(proof):return False
    rows = _ep_unpadded_source_rows(graph)
    if rows is None:
        return False
    invocations={};new_names=set()
    for node,(kind,entry) in zip(graph.nodes,proof,strict=True):
        if type(node) is not WorkloadNode or node.kind is not kind:return False
        if entry is None:continue
        old,fixed,static=entry
        if node is not old:
            if any(actual is not expected for actual,expected in zip((node.node_id,node.kind,node.pool_name,node.dtype,node.variant,node.rank,node.shape),fixed,strict=True)):return False
            position=0
            for key,value in node.attributes.items():
                if key in _ep_dynamic_fields(node):continue
                if position==len(static) or key is not static[position][0] or value is not static[position][1]:return False
                position+=1
            if position!=len(static):return False
        if node.kind is WorkloadKind.FUSED_MOE:
            if node.rank not in rows or node.shape["tokens"] != rows[node.rank]:
                return False
            attrs=node.attributes;oldattrs=old.attributes
            if _EP_SOURCE_DYNAMIC_FIELDS-set(attrs):return False
            counts=attrs['tokens_per_expert'];local=attrs['local_tokens_per_expert'];inv=attrs['moe_invocation_id']
            if type(local) is not tuple or any(type(c) is not int for c in local):return False
            if not _routing_certificate_context(attrs['routing_provenance'],counts,node.shape['tokens'],node.shape['experts'],node.shape['topk'],inv):return False
            expected_local=[0]*node.shape['experts']
            for expert in oldattrs['owned_experts']:expected_local[expert]=counts[expert]
            if local!=tuple(expected_local):return False
        elif node.kind is WorkloadKind.COLLECTIVE and 'moe_invocation_id' in node.attributes:
            inv=node.attributes['moe_invocation_id'];oldattrs=old.attributes
            if type(inv) is not str or not inv:return False
        else:continue
        oldinv=oldattrs['moe_invocation_id']
        if oldinv in invocations:
            if invocations[oldinv]!=inv:return False
        else:
            if inv in new_names:return False
            invocations[oldinv]=inv;new_names.add(inv)
    return True

def _record_ep_source_validation(graph, deployment, *, source_layout=_SOURCE_LAYOUT_UNSET):
    from src.serving.workload.traced_forward import _ep_source_owner
    owner = _ep_source_owner(graph, check_source=False)
    if owner is None:
        return
    if owner in _EP_SOURCE_VALIDATION_LAYOUTS and _EP_SOURCE_VALIDATION_LAYOUTS[owner] is None:
        return
    # Called only after ordinary complete EP validation has succeeded. A
    # negative entry disables this optimization forever for this owner; it
    # never grants validation trust, including if later values become native.
    if _ep_source_owner(graph, check_source=True) is not owner:
        _EP_SOURCE_VALIDATION_LAYOUTS[owner] = None
        return
    behavior = _ep_verifier_snapshot()
    if behavior is None:
        _EP_SOURCE_VALIDATION_LAYOUTS[owner] = None
        return
    layout = _ep_validation_layout(graph, deployment)
    _EP_SOURCE_VALIDATION_LAYOUTS[owner] = None if layout is None else (behavior, *layout)
    if layout is not None:
        _record_ep_source_recipe(graph, owner, _EP_SOURCE_VALIDATION_LAYOUTS[owner], source_layout=source_layout)


def _validate_ep_fused_moe_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate the complete Serving EP contract before kernel lowering."""

    ep_nodes = []
    for node in graph.nodes:
        if node.kind is not WorkloadKind.FUSED_MOE:
            continue
        if "expert_parallel_size" not in node.attributes:
            raise ValueError(
                f"FUSED_MOE node {node.node_id!r} lacks "
                "expert_parallel_size")
        expert_parallel_size = node.attributes["expert_parallel_size"]
        if (isinstance(expert_parallel_size, bool)
                or not isinstance(expert_parallel_size, int)
                or expert_parallel_size < 1):
            raise TypeError(
                "FUSED_MOE expert_parallel_size must be an integer >= 1")
        if node.pool_name not in deployment.worker_pools:
            raise ValueError(
                f"FUSED_MOE node references unknown pool {node.pool_name!r}")
        source_pool = deployment.worker_pools[node.pool_name]
        if any(key.startswith("source_fp8_") for key in node.attributes) and not (
                _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, source_pool)
                or _symbolic_glm_mtp_data_projection(node, source_pool)
                or _symbolic_glm_fp8_target_dp_projection(node, source_pool)):
            raise ValueError("deferred FP8 expert metadata requires its exact source scope; legacy mapped EP8 is distinct")
        if _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, source_pool):
            _validate_glm_rank_local_fp8_expert_node(node, source_pool)
        expected_expert_parallel_size = (
            deployment.worker_pools[
                node.pool_name].parallel_plan.config.expert_parallel_size)
        if expert_parallel_size != expected_expert_parallel_size:
            raise ValueError(
                "FUSED_MOE expert_parallel_size disagrees with the "
                f"deployment plan: {expert_parallel_size} != "
                f"{expected_expert_parallel_size}")
        if expected_expert_parallel_size > 1:
            ep_nodes.append(node)
    ep_pools = tuple(
        pool
        for pool in deployment.worker_pools.values()
        if pool.parallel_plan.config.enable_expert_parallel
        and any(node.pool_name == pool.name for node in graph.nodes)
    )
    if not ep_pools:
        if ep_nodes:
            raise ValueError(
                "EP FUSED_MOE nodes require an expert-parallel worker pool")
        return
    if not ep_nodes:
        if all(
                node.kind is WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            return
        raise ValueError(
            "expert-parallel deployment does not contain EP FUSED_MOE nodes")
    ep_pool_names = {node.pool_name for node in ep_nodes}
    if len(ep_pool_names) != 1:
        raise NotImplementedError(
            "one workload graph may bind only one expert-parallel worker pool")
    ep_pool_name = next(iter(ep_pool_names))
    ep_pool = deployment.worker_pools[ep_pool_name]
    parallelism = ep_pool.parallel_plan.config
    if parallelism.data_parallel_size > 1 and all(_symbolic_kimi_dp_projection(node, ep_pool) for node in ep_nodes):
        return  # Dedicated Kimi DP graph validator already checked dispatch/combine/ownership.
    if ((parallelism.context_parallel_size > 1 or parallelism.data_parallel_size > 1)
            and all(_symbolic_glm_pcp_projection(node, ep_pool) for node in ep_nodes)):
        _validate_glm_pcp_expert_graph(graph, ep_pool, ep_nodes)
        return
    if (parallelism.context_parallel_size == parallelism.data_parallel_size == parallelism.sequence_parallel_size == 1
            and all(_symbolic_glm_pcp_projection(node, ep_pool) for node in ep_nodes)):
        _validate_glm_tensor_expert_graph(graph, ep_pool, ep_nodes)
        return
    common_ep2 = (
        len(ep_pool.ranks) == 2
        and parallelism.worker_count == 2
        and parallelism.tensor_parallel_size == 2
        and parallelism.expert_parallel_size == 2
        and parallelism.pipeline_parallel_size == 1
        and parallelism.sequence_parallel_size == 1
        and parallelism.context_parallel_size == 1
        and parallelism.data_parallel_size == 1
    )
    source_ep = (parallelism.tensor_parallel_size >= 1
        and len(ep_pool.ranks) == parallelism.worker_count == parallelism.tensor_parallel_size * parallelism.pipeline_parallel_size * parallelism.data_parallel_size
        and parallelism.expert_parallel_size == parallelism.tensor_parallel_size * parallelism.data_parallel_size
        and parallelism.context_parallel_size == 1
        and (parallelism.sequence_parallel_size == 1 if parallelism.data_parallel_size == 1
            else _qwen030_dp_sp(parallelism))
        and all("source_component" in node.attributes and node.attributes["source_component"] == "symbolic_unquantized_experts"
            and "model_execution_version" in node.attributes
            and node.attributes["model_execution_version"] == "vllm_0_30_0" for node in ep_nodes))
    glm_pipeline_ep8 = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
        and all(_symbolic_glm_pipeline_projection(node, ep_pool) for node in ep_nodes)
        and all(node.attributes["checkpoint_weight_dtype"] == node.attributes["routed_weight_dtype"] == "bf16"
            for node in ep_nodes)
        and HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", ep_pool.hardware_id)
        and parallelism.pipeline_parallel_size == 4)
    kimi_pipeline_ep8 = (deployment.mode == "colocated" and len(deployment.worker_pools) == 1
        and all(_symbolic_kimi_pipeline_projection(node, ep_pool)
            or _symbolic_kimi_parallel_projection(node, ep_pool) for node in ep_nodes))
    replicated_ep8 = (((deployment.mode == "colocated" and len(deployment.worker_pools) == 1)
            or (deployment.mode == "pd_disagg" and worker_role_phase(ep_pool.role) in ("prefill", "decode")))
        and HWSpec.source_contract_supported("mla_tp8_ep8_source_projection", ep_pool.hardware_id)
        and len(ep_pool.ranks) == parallelism.worker_count == 8
        and parallelism.tensor_parallel_size == parallelism.expert_parallel_size == 8
        and all(value == 1 for value in (parallelism.pipeline_parallel_size, parallelism.sequence_parallel_size,
            parallelism.context_parallel_size, parallelism.data_parallel_size)))
    kimi_ep8 = ((replicated_ep8 or kimi_pipeline_ep8) and all("source_component" in node.attributes and "routing_source_stage" in node.attributes
            and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
            and node.attributes["routing_source_stage"] == "kimi_moe_route"
            and {"routed_weight_dtype", "checkpoint_weight_dtype", "weight_group_size", "routed_scaling_factor", "output_is_reduced"} <= set(node.attributes)
            and node.attributes["routed_weight_dtype"] == node.attributes["checkpoint_weight_dtype"] == "int4"
            and node.attributes["weight_group_size"] == 32 and node.attributes["routed_scaling_factor"] == 1.0
            and node.attributes["output_is_reduced"] is False
            and {"experts", "topk", "hidden", "intermediate"} <= set(node.shape)
            and tuple(node.shape[key] for key in ("experts", "topk", "hidden", "intermediate")) == (384, 8, 7168, 2048)
            for node in ep_nodes))
    glm_ep8 = ((replicated_ep8 or glm_pipeline_ep8) and all("source_component" in node.attributes and "routing_source_stage" in node.attributes
        and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
        and node.attributes["routing_source_stage"] == "dsv32_moe_route"
        and {"routed_weight_dtype", "checkpoint_weight_dtype", "weight_group_size", "routed_scaling_factor", "output_is_reduced"} <= set(node.attributes)
        and node.attributes["routed_weight_dtype"] == node.attributes["checkpoint_weight_dtype"]
        and node.attributes["routed_weight_dtype"] in ("bf16", "fp8")
        and node.attributes["weight_group_size"] == 0 and node.attributes["routed_scaling_factor"] == 2.5
        and node.attributes["output_is_reduced"] is False
        and {"experts", "topk", "hidden", "intermediate"} <= set(node.shape)
        and tuple(node.shape[key] for key in ("experts", "topk", "hidden", "intermediate")) == (256, 8, 6144, 2048)
        for node in ep_nodes))
    independently_scheduled_ep2 = deployment.mode in (
        "colocated", "pd_disagg")
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    af_ep2 = (
        deployment.mode == "af_disagg"
        and len(deployment.worker_pools) == 2
        and worker_role_operator_group(ep_pool.role) == "ffn"
        and worker_role_phase(ep_pool.role) == "colocated"
        and len(attention_pools) == 1
        and attention_pools[0].hardware_id == ep_pool.hardware_id
        and len(attention_pools[0].ranks) == 2
        and attention_pools[0].parallel_plan.config.worker_count == 2
        and attention_pools[0].parallel_plan.config.tensor_parallel_size == 2
        and attention_pools[0].parallel_plan.config.expert_parallel_size == 1
        and attention_pools[0].parallel_plan.config.pipeline_parallel_size == 1
        and attention_pools[0].parallel_plan.config.sequence_parallel_size == 1
        and attention_pools[0].parallel_plan.config.context_parallel_size == 1
        and attention_pools[0].parallel_plan.config.data_parallel_size == 1
    )
    if not (kimi_ep8 or glm_ep8 or (common_ep2 and (independently_scheduled_ep2 or af_ep2))
            or (source_ep and independently_scheduled_ep2)):
        raise NotImplementedError(
            "rank-local EP lowering supports independently scheduled "
            "colocated/PD or standalone AF TP=EP=2 pools with "
            "DP=PP=SP=CP=1, or the H200 GLM/Kimi replicated-row TP=EP8 source")
    tuple(
        ep_pool.collective_for(operation)
        for operation in (("all_reduce", "all_gather") if parallelism.tensor_parallel_size > 1 else ())
    )
    required_attributes = {
        "tokens_per_expert",
        "local_tokens_per_expert",
        "expert_parallel_size",
        "expert_rank",
        "owned_experts",
        "moe_invocation_id",
        "expert_placement_strategy",
        "routing_provenance",
        "launch_policy",
    }
    groups = {}
    for node in ep_nodes:
        missing = required_attributes - set(node.attributes)
        if missing:
            raise ValueError(
                f"EP FUSED_MOE node {node.node_id!r} lacks attributes: "
                f"{sorted(missing)!r}")
        invocation_id = node.attributes["moe_invocation_id"]
        if not isinstance(invocation_id, str) or not invocation_id:
            raise TypeError(
                "EP FUSED_MOE moe_invocation_id must be a non-empty string")
        if invocation_id not in groups:
            groups[invocation_id] = []
        groups[invocation_id].append(node)

    # Build typed adjacency once for this actual graph. No graph or dynamic
    # routing validation is memoized; all current contracts below still run.
    successors_by_id = {node.node_id: set() for node in graph.nodes}
    data_predecessors_by_id = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        successors_by_id[edge.source].add(edge.destination)
        if edge.kind == "data":
            data_predecessors_by_id[edge.destination].append(edge.source)

    # Deployment ownership is static during this complete graph validation;
    # routing counts, records and rank-node coverage remain invocation-local.
    owned_by_pool_rank = {}
    for invocation_id, invocation_nodes in groups.items():
        first = invocation_nodes[0]
        pool_name = first.pool_name
        if pool_name not in deployment.worker_pools:
            raise ValueError(
                f"EP FUSED_MOE references unknown pool {pool_name!r}")
        pool = deployment.worker_pools[pool_name]
        plan = pool.parallel_plan
        ep_size = first.attributes["expert_parallel_size"]
        if len(invocation_nodes) != ep_size:
            raise ValueError(
                f"EP MoE invocation {invocation_id!r} requires one node per "
                f"rank: {len(invocation_nodes)} != {ep_size}")
        if plan.config.expert_parallel_size != ep_size:
            raise ValueError(
                "EP MoE invocation size disagrees with its deployment plan")
        if first.dtype != "bf16":
            raise NotImplementedError(
                "rank-local EP FUSED_MOE validation currently requires BF16")
        required_shape = {
            "tokens", "experts", "topk", "hidden", "intermediate"}
        if set(first.shape) != required_shape:
            raise ValueError(
                "EP FUSED_MOE shape must contain exactly tokens, experts, "
                "topk, hidden, and intermediate")
        if any(first.shape[name] <= 0 for name in required_shape):
            raise ValueError("EP FUSED_MOE shape dimensions must be positive")
        first_graph_domain = _graph_token_domain(first)
        if first_graph_domain is None:
            routing_tokens = first.shape["tokens"]
        else:
            routing_tokens, model_input_tokens, _ = first_graph_domain
            if first.shape["tokens"] != model_input_tokens:
                raise ValueError(
                    "EP FUSED_MOE physical tokens disagree with its graph "
                    "token-domain contract")
        if (
            (first.shape["experts"] % ep_size and not source_ep)
            or first.shape["topk"] > first.shape["experts"]
        ):
            raise NotImplementedError(
                "rank-local EP lowering requires experts divisible by EP and "
                "top-k no larger than the expert count")
        global_counts = tuple(first.attributes["tokens_per_expert"])
        expert_count = first.shape["experts"]
        if len(global_counts) != expert_count:
            raise ValueError(
                "EP global routing count must match the global expert count")
        if any(isinstance(count, bool) or not isinstance(count, int)
               or count < 0 for count in global_counts):
            raise TypeError(
                "EP global routing counts must be non-negative integers")
        if sum(global_counts) != routing_tokens * first.shape["topk"]:
            raise ValueError(
                "EP global routing counts must sum to logical input rows * "
                "top-k")
        if any(count > routing_tokens for count in global_counts):
            raise ValueError(
                "EP global routing count cannot exceed input tokens for any "
                "expert")

        expert_group = plan.rank_groups.for_rank("expert", pool.logical_rank_for_physical_rank(first.rank))
        if len(expert_group) != ep_size:
            raise ValueError(
                "EP deployment rank group disagrees with expert_parallel_size")
        expected_participants = tuple(
            pool.physical_rank_for_logical_rank(rank)
            for rank in expert_group)
        nodes_by_rank = {}
        all_owned_experts = []
        for node in invocation_nodes:
            if (node.pool_name != pool_name
                    or dict(node.shape) != dict(first.shape)
                    or node.dtype != first.dtype
                    or node.variant != "full"
                    or node.attributes["expert_parallel_size"] != ep_size
                    or tuple(node.attributes["tokens_per_expert"])
                    != global_counts
                    or _graph_token_domain(node) != first_graph_domain
                    or node.attributes["moe_invocation_id"] != invocation_id
                    or dict(node.attributes["routing_provenance"])
                    != dict(first.attributes["routing_provenance"])):
                raise ValueError(
                    "EP MoE rank nodes disagree on the full-input invocation")
            if node.attributes["expert_placement_strategy"] != "linear":
                raise NotImplementedError(
                    "rank-local EP FUSED_MOE currently requires linear "
                    "expert placement")
            expert_rank = node.attributes["expert_rank"]
            if (isinstance(expert_rank, bool)
                    or not isinstance(expert_rank, int)
                    or not 0 <= expert_rank < ep_size):
                raise ValueError("EP FUSED_MOE expert_rank is outside the group")
            if expert_rank in nodes_by_rank:
                raise ValueError(
                    "EP MoE invocation contains a duplicate expert rank")
            expected_logical_rank = expert_group[expert_rank]
            expected_physical_rank = pool.physical_rank_for_logical_rank(
                expected_logical_rank)
            if node.rank != expected_physical_rank:
                raise ValueError(
                    "EP FUSED_MOE physical rank disagrees with expert_rank")
            owned_experts = tuple(node.attributes["owned_experts"])
            ownership_key = (pool_name, expert_rank)
            if ownership_key not in owned_by_pool_rank:
                owned_by_pool_rank[ownership_key] = tuple(
                    expert_id
                    for expert_id, owner_rank in enumerate(plan.expert_to_shard)
                    if owner_rank == expert_rank
                )
            expected_owned_experts = owned_by_pool_rank[ownership_key]
            if owned_experts != expected_owned_experts:
                raise ValueError(
                    "EP FUSED_MOE owned_experts disagrees with the deployment "
                    "expert placement")
            local_counts = tuple(
                node.attributes["local_tokens_per_expert"])
            if len(local_counts) != expert_count:
                raise ValueError(
                    "EP local routing count must match the global expert count")
            if any(isinstance(count, bool) or not isinstance(count, int)
                   or count < 0 for count in local_counts):
                raise TypeError(
                    "EP local routing counts must be non-negative integers")
            expected_local_counts = [0] * expert_count
            for expert_id in expected_owned_experts:
                if expert_id < expert_count:
                    expected_local_counts[expert_id] = global_counts[expert_id]
            if local_counts != tuple(expected_local_counts):
                raise ValueError(
                    "EP local routing counts disagree with global routing and "
                    "expert ownership")
            if (kimi_ep8 or glm_ep8 or ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_unquantized_experts")):
                _validate_virtual_ep_router_inputs(graph, node)
            else:
                predecessors = graph.predecessors(node.node_id)
                if len(predecessors) != 1:
                    raise ValueError(
                        "EP FUSED_MOE must have exactly one rank-local router "
                        "predecessor")
                router = graph.node(predecessors[0])
                if router.kind is not WorkloadKind.ROUTER or router.rank != node.rank:
                    raise ValueError(
                        "EP FUSED_MOE predecessor must be the same-rank router")
            nodes_by_rank[expert_rank] = node
            all_owned_experts.extend(owned_experts)
        if tuple(sorted(all_owned_experts)) != tuple(range(expert_count)):
            raise ValueError(
                "EP expert ownership must cover every global expert exactly "
                "once")
        if set(nodes_by_rank) != set(range(ep_size)):
            raise ValueError("EP MoE invocation does not cover every expert rank")

        node_ids = {node.node_id for node in invocation_nodes}
        output_ports = {}
        if kimi_ep8 or glm_ep8:
            # Kimi combines full local routed experts with its TP-sharded
            # shared expert only after the in-place routed-output scaling.
            combined_ids = set()
            for node in invocation_nodes:
                successors = tuple(graph.node(edge.destination) for edge in graph.edges if edge.source == node.node_id)
                if len(successors) != 1:
                    raise ValueError("Kimi EP routed output requires one scale successor")
                scale = successors[0] if kimi_ep8 else node
                if kimi_ep8 and (scale.kind is not WorkloadKind.ELEMENTWISE or scale.variant != "elementwise"
                        or scale.rank != node.rank or scale.pool_name != node.pool_name or scale.dtype != "bf16"
                        or dict(scale.shape) != {"elements": node.shape["tokens"] * node.shape["hidden"]}
                        or {"operation", "reads", "writes", "scalar_value", "symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(scale.attributes)
                        or (scale.attributes["operation"], scale.attributes["reads"], scale.attributes["writes"], scale.attributes["scalar_value"]) != ("unary", 1, 1, 2.827)
                        or tuple(scale.attributes["symbolic_tensor_inputs"]) != tuple(node.attributes["symbolic_tensor_outputs"])
                        or tuple(scale.attributes["symbolic_tensor_outputs"]) != tuple(node.attributes["symbolic_tensor_outputs"])
                        or set(graph.predecessors(scale.node_id)) != {node.node_id}):
                    raise ValueError("Kimi EP output scaling must preserve its same-rank routed storage")
                additions = tuple(graph.node(edge.destination) for edge in graph.edges if edge.source == scale.node_id)
                if len(additions) != 1:
                    raise ValueError("Kimi EP scaled output requires one shared-add successor")
                addition = additions[0]
                parents = tuple(graph.node(name) for name in graph.predecessors(addition.node_id) if name != scale.node_id)
                if len(parents) != 1:
                    raise ValueError("Kimi EP shared-add requires one separate shared-expert output")
                shared = parents[0]
                if ("symbolic_tensor_outputs" not in shared.attributes or len(shared.attributes["symbolic_tensor_outputs"]) != 1
                        or shared.kind is not WorkloadKind.LINEAR or shared.variant != "gemm" or shared.dtype != "bf16"
                        or shared.rank != node.rank or shared.pool_name != node.pool_name
                        or dict(shared.shape) != {"b": 1, "m": node.shape["tokens"], "n": node.shape["hidden"],
                            "k": node.shape["intermediate"] // parallelism.tensor_parallel_size}
                        or addition.kind is not WorkloadKind.ELEMENTWISE or addition.variant != "add"
                        or addition.rank != node.rank or addition.pool_name != node.pool_name or addition.dtype != "bf16"
                        or dict(addition.shape) != {"elements": node.shape["tokens"] * node.shape["hidden"]}
                        or {"operation", "reads", "writes", "symbolic_tensor_inputs", "symbolic_tensor_outputs"} - set(addition.attributes)
                        or (addition.attributes["operation"], addition.attributes["reads"], addition.attributes["writes"]) != ("add", 2, 1)
                        or tuple(addition.attributes["symbolic_tensor_inputs"]) != (
                            shared.attributes["symbolic_tensor_outputs"][0], scale.attributes["symbolic_tensor_outputs"][0])
                        or len(addition.attributes["symbolic_tensor_outputs"]) != 1
                        or addition.attributes["symbolic_tensor_outputs"][0] in addition.attributes["symbolic_tensor_inputs"]):
                    raise ValueError("grouped EP shared-add must combine separate same-rank shared and routed outputs")
                combined_ids.add(addition.node_id)
                output_ports[node.rank] = addition.attributes["symbolic_tensor_outputs"][0]
            if len(combined_ids) != ep_size:
                raise ValueError("grouped EP shared-add must cover every rank independently")
            node_ids = combined_ids
        if glm_pipeline_ep8:
            stage = plan.layer_to_pipeline_stage[first.attributes["routing_provenance"]["layer_index"]]
            if stage < plan.config.pipeline_parallel_size - 1:
                hidden_messages = tuple(node for node in graph.nodes if node.kind is WorkloadKind.P2P
                    and node.pool_name == pool_name and node.attributes["source_stage"] == stage
                    and node.attributes["payload_tensor"]["name"] == "hidden_states")
                hidden_message_ids = {node.node_id for node in hidden_messages}
                hidden_producers = {edge.source for edge in graph.edges if edge.kind == "data"
                    and edge.destination in hidden_message_ids}
                if hidden_producers == node_ids:
                    # The preceding pipeline validation binds all eight
                    # partial outputs to exact lane slices, receives, gathers,
                    # runner copies and destination input reduction. This
                    # source invocation has no pre-send MoE all-reduce.
                    messages = {node.node_id for node in graph.nodes if node.kind is WorkloadKind.P2P
                        and node.pool_name == pool_name and node.attributes["source_stage"] == stage}
                    successors = {edge.destination for edge in graph.edges if edge.source in node_ids}
                    if successors != messages:
                        raise ValueError("GLM EP stage-final partial outputs must feed exactly their ordered PP messages")
                    continue
        successor_ids = set().union(*(successors_by_id[node_id] for node_id in node_ids))
        if len(successor_ids) != 1:
            raise ValueError(
                "EP MoE rank nodes must converge on exactly one output "
                "all-reduce")
        collective = graph.node(next(iter(successor_ids)))
        if (collective.kind is not WorkloadKind.COLLECTIVE
                or collective.pool_name != pool_name
                or collective.attributes["operation"] != ("reduce_scatterv" if parallelism.data_parallel_size > 1 and source_ep else "all_reduce")
                or collective.attributes["bytes_semantics"] != "per_rank_input"
                or collective.attributes["reduction"] != "sum"
                or tuple(collective.attributes["participants"])
                != expected_participants
                or collective.attributes["moe_invocation_id"] != invocation_id):
            raise ValueError(
                "EP MoE output must use the matching TP-group all-reduce")
        if _graph_token_domain(collective) != first_graph_domain:
            raise ValueError(
                "EP MoE output all-reduce must retain the producer graph "
                "token-domain contract")
        expected_bytes = (
            first.shape["tokens"] * first.shape["hidden"] * 2)
        if collective.attributes["bytes"] != expected_bytes:
            raise ValueError(
                "EP MoE output all-reduce bytes must equal B * H * BF16")
        payload_predecessors = (tuple(data_predecessors_by_id[collective.node_id])
            if "symbolic_tensor_inputs" in collective.attributes else graph.predecessors(collective.node_id))
        if (len(payload_predecessors) != ep_size or set(payload_predecessors) != node_ids):
            raise ValueError(
                "EP MoE output all-reduce must wait for every expert rank")
        if (kimi_ep8 or glm_ep8) and tuple(collective.attributes["symbolic_tensor_inputs"]) != tuple(output_ports[rank] for rank in expected_participants):
            raise ValueError("grouped EP late all-reduce must bind every rank's combined output port")

    if any(
            node.kind is WorkloadKind.COLLECTIVE
            and node.attributes["operation"] == "all_to_all"
            for node in graph.nodes):
        raise ValueError(
            "TP=2, DP=1 EP workload must not emit all-to-all communication")


def _vector_shape(node: WorkloadNode) -> Mapping[str, int]:
    shape = dict(node.shape)
    elements = shape["elements"]
    # The completed-state store finishes floor(new/r)-floor(old/r) states per
    # request, so no-boundary steps legitimately carry zero work; the launch
    # is preserved (KernelSpec allows element_count == 0 only for this
    # variant).
    zero_allowed = (node.kind is WorkloadKind.KV_UPDATE
                    and node.variant == "dsv4_compressed_state_cache")
    if elements <= 0 and not (zero_allowed and elements == 0):
        raise ValueError(f"{node.kind.value} requires positive shape['elements']")
    del shape["elements"]
    shape["element_count"] = elements
    return shape




def _selected_target_compute_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    kind: KernelKind,
    variant: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, object],
) -> KernelSpec:
    if variant in ("dsv32_eh_norm", "dsv32_embed_eh_norm") and "input_values_source" in node.attributes:
        raise NotImplementedError("GLM legacy dummy EH norm requires retained position-mask/input-ID values; Goal05 pricing is unbound")
    if (variant == "dsv32_fused_norm_rope" and attributes["cache_slot_scope"] in (
            "outputs_only_no_cache_slots", "no_metadata_no_cache_slots")):
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype, variant=variant,
            attributes=attributes, launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    launch = select_launch_config(
        hardware_id,
        kind,
        shape,
        node.dtype,
        variant,
        node.attributes,
        hardware=hardware,
    )
    prediction_shape = dict(shape)
    prediction_attributes = dict(attributes)
    if kind is not KernelKind.FLASH_ATTN:
        # Graph-replayed target nodes carry padded physical rows; the
        # launch uses them, the prediction keeps the logical work (the same
        # contract as the generic single-kernel lowering).
        prediction_shape, domain = _logical_prediction_shape(node, shape)
        prediction_attributes.update(
            _kernel_graph_token_attributes(domain, shape))
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=prediction_shape,
        dtype=node.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes=prediction_attributes,
    )


def _source_vocab_embedding_spec(node, pool, hardware_id):
    """Shared typed CUDA TP embedding lowering for source and traced callers."""
    if (node.kind is not WorkloadKind.GATHER or node.variant != "embedding"
            or node.attributes["embedding_source"] not in ("vllm_0_30_cuda_unquantized_tp", "sglang_0_5_19_triton_unquantized_tp",
                "vllm_0_27_1_useful_tp_embedding_aggregate")):
        raise ValueError("unsupported target embedding source")
    tp_size = pool.parallel_plan.config.tensor_parallel_size
    if node.attributes["tensor_parallel_size"] != tp_size:
        raise ValueError("embedding source TP size disagrees with deployment")
    if (node.attributes["index_count"] != node.shape["rows"]
            or node.attributes["gather_factor"] != 1
            or node.attributes["scattered"] is not True
            or node.attributes["materialized_output"] is not True):
        raise ValueError("embedding source requires one id and complete partial output per row")
    logical_rank = pool.logical_rank_for_physical_rank(node.rank)
    group = pool.parallel_plan.rank_groups.for_rank("tensor", logical_rank)
    owned_rows = None
    if "embedding_token_ids" in node.attributes and node.attributes["embedding_token_ids"] is not None:
        token_ids = node.attributes["embedding_token_ids"]
        vocab = node.attributes["vocab_size"]
        if (not isinstance(token_ids, tuple) or len(token_ids) != node.shape["rows"]
                or any(type(token) is not int or not 0 <= token < vocab for token in token_ids)):
            raise ValueError("embedding token IDs must cover valid scheduled rows")
        shard_rows = (-(-vocab // 64) * 64) // tp_size
        start = group.index(logical_rank) * shard_rows
        owned_rows = sum(start <= token < start + shard_rows for token in token_ids)
    return KernelSpec(
        kind=KernelKind.VECTOR, hardware_id=hardware_id,
        shape={"element_count": node.shape["elements"], "rows": node.shape["rows"],
               "hidden": node.shape["hidden"], "input_id_bytes": node.attributes["index_bytes"]},
        dtype=node.dtype, variant="vocab_parallel_embedding",
        attributes={"vocab_size": node.attributes["vocab_size"],
                    "tensor_parallel_size": tp_size, "tensor_parallel_rank": group.index(logical_rank),
                    "owned_token_rows": owned_rows},
        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")


def _target_mhc_specs(
        node: WorkloadNode, hardware_id: str, hardware) -> tuple[KernelSpec, ...]:
    """Expand vLLM's logical mHC call into its actual stage sequence.

    Preserve the source prenorm split contract, including a declared
    env-disabled optional TileLang branch. External prenorm launch facts
    remain unbound; ordinary GEMM geometry is not evidence for either call.
    """
    stages = select_mhc_stage_launches(
        hardware_id, _vector_shape(node), node.dtype, node.variant,
        node.attributes, hardware)
    specs = []
    for variant, shape, launch in stages:
        prediction_shape, domain = _logical_prediction_shape(node, shape)
        attributes = _kernel_graph_token_attributes(domain, shape)
        if variant in ("mhc_prenorm", "mhc_prenorm_broadcast"):
            attributes["source_prenorm_backend"] = (
                "deep_gemm" if variant == "mhc_prenorm_broadcast" else
                "tilelang" if "source_use_deep_gemm" in node.attributes
                    and node.attributes["source_use_deep_gemm"] is False else None)
        specs.append(KernelSpec(
            KernelKind.VECTOR, hardware_id, prediction_shape,
            dtype=node.dtype, variant=variant, launch=launch,
            attributes=attributes,
            execution_support="unsupported" if launch is None else "supported",
            unsupported_reason="launch_selector_unavailable" if launch is None else None))
    return tuple(specs)


def _target_rope_spec(
        node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.variant != "rotary":
        raise ValueError(
            f"target-model RoPE requires rotary variant, got {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in (
            "rope_layout", "rope_type", "rope_theta", "rope_factor",
            "rope_original_max_position_embeddings", "rope_beta_fast",
            "rope_beta_slow", "rope_mscale", "rope_mscale_all_dim",
        )
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.VECTOR, "mla_rope",
        _vector_shape(node), attributes)


def _target_layernorm_spec(
        node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.variant != "layernorm":
        raise ValueError(
            "target-model LayerNorm requires layernorm variant")
    attributes = {
        name: node.attributes[name]
        for name in ("affine", "epsilon")
    }
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.VECTOR, "layernorm",
        node.shape, attributes)


# DeepSeek-V3 dense projections that the official FP8 checkpoint stores as
# 128x128 block-scaled e4m3 weights (vLLM ``DeepseekV2`` builds each with the
# checkpoint ``quant_config``).  The absorbed MLA W_UK/W_UV batched GEMMs,
# the router and the LM head are not in this set.  DeepSeek-V4 stores the
# same module families in its FP8 128x128 ue8m0 block-scaled checkpoint
# (fused wqa_wkv, wq_b, the indexer wq_b, wo_b and the shared-expert MLP);
# the batched wo_a, the bf16 compressor/indexer projections, the router and
# the LM head stay out.
_DSV3_BLOCK_FP8_LINEAR_POSITIONS = frozenset({
    "fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj",
    "gate_up_proj", "down_proj", "indexer_wq_b",
})


def _dsv3_block_fp8_checkpoint(
    graph: WorkloadGraph, target_contract: str | None,
) -> bool:
    """Whether the graph is a DeepSeek-V3 step of a block-FP8 checkpoint.

    The DeepSeek-V3 model contract admits FP8 weights only as the official
    128x128 e4m3 block-scaled checkpoint with dynamic 1x128 activation groups
    (``model_config``).  The MLP linears carry the checkpoint dtype resolved
    from the model (``weight_dtype_for_module``); the MLA projections are
    emitted by the builder shared with Kimi/GLM with a fixed ``bf16`` label
    that does not describe this checkpoint, so the checkpoint format is read
    from the MLP linears and applies to every projection in
    ``_DSV3_BLOCK_FP8_LINEAR_POSITIONS``.

    The DeepSeek-V4 contract works the same way: its non-expert linears are
    the FP8 128x128 ue8m0 block-scaled checkpoint, and the shared-expert
    gate_up/down linears carry the resolved ``fp8`` dtype.
    """

    if target_contract == "glm_moe_dsa_vllm_0_30_0_logical_v1":
        # GLM-5.3 declares the block format and resolves exact BF16
        # exclusions per module. The call site preserves those exclusions.
        return True
    if target_contract not in (
            "deepseek_v3_vllm_0_27_1_logical_v1",
            "deepseek_v4_vllm_0_30_0_logical_v1"):
        return False
    dtypes = {
        node.attributes["weight_dtype"] for node in graph.nodes
        if node.kind is WorkloadKind.LINEAR
        and node.attributes["position"] in ("gate_up_proj", "down_proj")
    }
    if not dtypes:
        # Single-kernel lowering graphs carry no MLP linears; without the
        # checkpoint dtype evidence no projection is declared block-FP8.
        return False
    if len(dtypes) != 1:
        raise ValueError(
            "DeepSeek target step must carry one MLP checkpoint weight "
            f"dtype, got {sorted(dtypes)}")
    return dtypes == {"fp8"}


def _block_fp8_linear_node(
    node: WorkloadNode, block_fp8_checkpoint: bool,
) -> WorkloadNode:
    """Declare the static block-FP8 weight format of one dense projection.

    The declaration is a launch-selection attribute only: a hardware that
    owns a block-FP8 GEMM selector routes the linear there, every other
    hardware ignores it, and no prediction attribute changes.
    """

    if (not block_fp8_checkpoint
            or node.kind is not WorkloadKind.LINEAR
            or node.attributes["position"]
            not in _DSV3_BLOCK_FP8_LINEAR_POSITIONS):
        return node
    if "weight_quantization" in node.attributes:
        raise ValueError(
            f"linear {node.node_id!r} already declares weight_quantization")
    return dataclasses.replace(node, attributes={
        **node.attributes,
        "weight_quantization": BLOCK_FP8_WEIGHT_QUANTIZATION,
    })


def _vllm030_group_fp8_source_specs(specs, node):
    """Keep the declared FP32-scale quantizer independent of generic launch.

    Its CUDA/Triton/rounding program is conditional source input. A generic
    helper launch selected for a GEMM does not bind those vLLM0.30 choices.
    """
    return tuple(dataclasses.replace(spec,
        shape={**spec.shape, "columns": node.shape["k"],
            "rows": spec.shape["element_count"] // node.shape["k"]},
        attributes={**spec.attributes, "source_program": "vllm_0_30_group_fp8_fp32_scales"},
        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
        if spec.kind is KernelKind.VECTOR and spec.variant == "fp8_group_quant" else spec for spec in specs)


def _target_gemm_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    block_fp8_checkpoint: bool,
) -> tuple[KernelSpec, ...]:
    if node.variant == "batched_gemm":
        attributes = {
            name: node.attributes[name]
            for name in (
                "batch_layout", "a_batch_stride_elements",
                "b_batch_stride_elements", "c_batch_stride_elements",
                "weight_dtype", "weight_group_size",
            )
        }
        domain = _graph_token_domain(node)
        if domain is not None:
            # Strides describe the logical prediction operands; the padded
            # physical strides stay in the node's selection attributes.
            logical_tokens, model_input_tokens, _ = domain
            if node.shape["m"] != model_input_tokens:
                raise ValueError(
                    "padded batched GEMM rows disagree with the graph domain")
            attributes["a_batch_stride_elements"] = (
                logical_tokens * node.shape["k"])
            attributes["c_batch_stride_elements"] = (
                node.shape["n"] if attributes["batch_layout"] == "einsum_bhr_hdr_bhd"
                else logical_tokens * node.shape["n"])
        return (_selected_target_compute_spec(
            node, hardware_id, hardware, KernelKind.GEMM, "batched_gemm",
            node.shape, attributes),)
    if node.kind is WorkloadKind.ROUTER:
        if node.variant != "gemm":
            raise ValueError(
                f"target-model router requires gemm variant, got "
                f"{node.variant!r}")
        if "output_dtype" in node.attributes:
            # Explicit GateLinear emits only logits: V4/GLM produce FP32,
            # while Kimi v0.30's ReplicatedLinear branch produces BF16.
            # Hash lookup/top-k and weights are separate source operations;
            # moe_router's homogeneous fused score model does not cover it.
            if node.attributes["launch_policy"] != "analytical_launch":
                raise NotImplementedError("GateLinear actual launch binding is unavailable")
            gate_attributes = {"output_dtype": node.attributes["output_dtype"]}
            if gate_attributes["output_dtype"] != node.dtype:
                gate_attributes.update({name: node.attributes[name] for name in (
                    "weight_dtype", "weight_group_size")})
            elif (node.attributes["weight_dtype"] != node.dtype
                  or node.attributes["weight_group_size"] != 0):
                raise NotImplementedError("plain GateLinear requires homogeneous unquantized weights")
            return (_selected_target_compute_spec(
                node, hardware_id, hardware, KernelKind.GEMM, "gemm",
                node.shape, gate_attributes),)
        attributes = {
            name: node.attributes[name]
            for name in (
                "score_function", "topk_method", "expert_group_count",
                "topk_group_count", "normalize_topk_probability",
                "weight_dtype", "weight_group_size",
            )
        }
        main = _selected_target_compute_spec(
            node, hardware_id, hardware, KernelKind.GEMM, "moe_router",
            node.shape, attributes)
        if (main.launch.tiling is not None
                or main.launch.parameters[
                    "split_k_partial_bytes_per_element"] == 0):
            return (main,)
        # An untiled router launch that writes external partials was observed
        # with its separate split-K reducer; the reducer launch is part of the
        # same observation.
        reducer_launch = select_gemm_reducer_launch(
            hardware_id, node.shape, node.dtype, "moe_router",
            node.attributes, hardware=hardware)
        element_count = (
            node.shape["b"] * node.shape["m"] * node.shape["n"])
        reducer = KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": element_count},
            dtype=main.dtype,
            variant=reducer_launch.execution_variant,
            launch=reducer_launch,
            attributes={
                # The split count is not observed; two partials is its lower
                # bound (the reducer prices the declared limitation).
                "split_k": 2,
                **_derived_graph_token_attributes(
                    main, {"element_count": element_count}),
            },
        )
        return main, reducer
    return _gemm_node_specs(
        _block_fp8_linear_node(node, block_fp8_checkpoint),
        hardware_id, hardware)


def _target_mla_attention_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> tuple[KernelSpec, ...]:
    if node.variant not in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill"):
        raise ValueError(
            f"unknown target MLA attention variant: {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens",
            "full_cuda_graphs", "max_cudagraph_capture_size")
    }
    if "attention_backend" in node.attributes:
        attributes["attention_backend"] = node.attributes["attention_backend"]
    if "return_softmax_lse" in node.attributes:
        attributes["return_softmax_lse"] = node.attributes["return_softmax_lse"]
    graph_fields = (
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode")
    present_graph_fields = tuple(
        name for name in graph_fields if name in node.attributes)
    if present_graph_fields:
        if present_graph_fields != graph_fields:
            raise ValueError(
                "target MLA graph token domain is incomplete: "
                f"present={present_graph_fields!r}")
        if node.variant != "mla_decode":
            raise ValueError("only MLA decode carries a graph token domain")
        # Padded requests (query 0, seq_len 0) add no attention work; the
        # KernelSpec keeps the logical requests and the replay domain.
        attributes.update({name: node.attributes[name]
                           for name in graph_fields})
    main = _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.FLASH_ATTN, node.variant,
        node.shape, attributes)
    if node.variant in ("mla_prefill", "mla_absorbed_prefill"):
        return (main,)
    tiling = main.launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("A800 MLA decode launch must declare KV splits")
    if tiling.num_kv_splits == 1:
        return (main,)
    combine_shape = {
        "batch": node.shape["batch"],
        "q_len": node.shape["q_len"],
        "kv_len": node.shape["kv_len"],
        "head_dim": node.shape["v_head_dim"],
        "num_query_heads": node.shape["num_query_heads"],
        "num_kv_heads": 1,
    }
    combine_attributes = {
        **{name: attributes[name] for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")},
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
        "num_kv_splits": tiling.num_kv_splits,
    }
    combine_selection_attributes = dict(combine_attributes)
    if present_graph_fields:
        combine_selection_attributes.update(
            {name: node.attributes[name] for name in graph_fields})
    combine_launch = select_launch_config(
        hardware_id, KernelKind.FLASH_ATTN, combine_shape, node.dtype,
        "combine", combine_selection_attributes, hardware=hardware)
    combine = KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id=hardware_id,
        shape=combine_shape,
        dtype=node.dtype,
        variant="combine",
        launch=combine_launch,
        attributes={
            **{name: combine_attributes[name]
               for name in (
                   "causal", "kv_layout", "kv_dist", "fa_ksteps",
                   "fa_eff_mfrac", "q_lens", "kv_lens")},
            **({"graph_runtime_mode": node.attributes["graph_runtime_mode"]}
               if present_graph_fields else {}),
        },
    )
    return main, combine


def _target_sparse_indexer_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    context_parallel_size: int,
) -> KernelSpec:
    if node.variant == "dsa_index_reuse":
        attributes = {
            name: node.attributes[name]
            for name in (
                "reuse_semantics", "topk_capacity", "topk_index_dtype")
        }
        shape = _vector_shape(node)
    elif node.variant in (
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed", "dsv4_indexer_prefill",
            "dsv4_indexer_decode"):
        attributes = {
            name: node.attributes[name]
            for name in (
                "causal", "score_accumulator_dtype", "score_function",
                "cache_layout", "cache_quantization", "q_lens", "kv_lens",
                "request_phases", "topk_capacity", "topk_index_dtype",
                "topk_sentinel",
            )
        }
        shape = node.shape
        if node.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
            attributes["context_parallel_size"] = context_parallel_size
        else:
            for name in ("mathematical_program", "context_parallel_size", "skip_topk_buffer_clear",
                    "num_speculative_tokens", "decode_threshold",
                    "clean_logits", "indexer_topk_backend", "model_execution_version",
                    "source_block_size", "source_block_table_inner_stride", "source_candidate_blocks",
                    "source_context_lens_layout", "source_cost_model", "source_decode_layout",
                    "source_deepgemm_snapshot", "source_indexer_input_stage", "source_indices",
                    "source_input_layout", "source_kv_cache_layout", "source_max_model_len",
                    "source_num_sms", "source_q_scale", "source_schedule_metadata",
                    "source_schedule_shape", "source_tie_refinement_policy",
                    "source_parallel_mode", "source_dense_mha_metadata_layer", "source_topk_clear_state",
                    "source_prefill_chunks", "source_chunk_policy", "source_sm_budget"):
                if name in node.attributes:
                    attributes[name] = node.attributes[name]
    else:
        raise ValueError(
            f"unknown DSA indexer variant: {node.variant!r}")
    if node.variant != "dsa_index_reuse" and (node.variant.startswith("dsa_")
            or HWSpec.source_contract_supported("multistage_attention_indexer_unbound_source_math", hardware_id)):
        # An indexer comprises multiple dependent physical kernels. Its
        # logical arithmetic is valid without a single aggregate launch;
        # prepared stage recipes bind through the KernelCall sidecar.
        # Neither the TopK stage alone nor another GPU's recipe can stand
        # in for this whole operator.
        if "source_cost_model" in attributes and attributes["source_cost_model"] is not None:
            budget = attributes["source_sm_budget"]
            sms = hardware.num_compute_units
            if budget == "h200_132_sm_declared_v1":
                if sms != 132:
                    raise NotImplementedError("declared DSA SM budget applies only to H200 132-SM HWSpec")
            elif budget != "actual_hwspec_num_compute_units":
                raise ValueError("DSA source SM budget must be actual HWSpec or the declared H200 132-SM domain")
            attributes["source_num_sms"] = sms
            attributes["source_schedule_shape"] = (sms + 1, 2)
        if ("source_cost_model" in attributes
                and attributes["source_cost_model"] == "vllm_0_30_bounded_stage_roofline_v1"
                and node.variant in ("dsa_indexer_decode", "dsa_indexer_mixed")):
            decode_rows = sum(q for q, phase in zip(node.attributes["q_lens"],
                node.attributes["request_phases"], strict=True) if phase == "decode")
            source_shape = {"batch": decode_rows, "next_n": 1, "rows": sum(node.attributes["q_lens"]),
                "num_decode_tokens": decode_rows,
                "num_prefill_tokens": sum(node.attributes["q_lens"]) - decode_rows,
                "num_heads": node.shape["num_query_heads"], "head_dim": node.shape["head_dim"],
                "block_size": attributes["source_block_size"],
                "max_model_len": attributes["source_max_model_len"], "topk": attributes["topk_capacity"]}
            from src.kernel.launch_config.select_launch_config import dsa_decode_topk_source_route
            recipe = select_workload_source_launch_recipe(hardware_id, source_shape, "fp8", node.variant,
                {**attributes, "source_component": "vllm_0_30_sparse_attn_indexer"}, hardware)
            route = dsa_decode_topk_source_route(hardware_id, recipe)
            attributes["source_topk_route"] = route
            attributes["source_topk_cluster_size"] = route["cluster"]
        spec = KernelSpec(kind=KernelKind.SPARSE_ATTN_INDEXER,
            hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=node.variant, attributes=attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
        return spec
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.SPARSE_ATTN_INDEXER,
        node.variant, shape, attributes)


def _target_sparse_attention_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> KernelSpec:
    attribute_names = [
        "causal", "indexer_type", "kv_layout", "kv_dist", "q_lens",
        "kv_lens", "selected_kv_lens", "topk_capacity", "topk_index_dtype",
    ]
    if node.variant.startswith("dsv4_"):
        attribute_names.append("swa_kv_lens")
        if "compressed_selection" in node.attributes:
            attribute_names.append("compressed_selection")
    attributes = {
        name: node.attributes[name]
        for name in attribute_names
    }
    for name in ("mathematical_program", "num_speculative_tokens", "decode_threshold",
            "reused_selection_kv_lens"):
        if name in node.attributes:
            attributes[name] = node.attributes[name]
    source_enabled = ("source_cost_model" in node.attributes and node.attributes["source_cost_model"]
        == "vllm_0_30_bounded_stage_roofline_v1")
    if "source_cost_model" in node.attributes and node.attributes["source_cost_model"] is not None:
        if not source_enabled:
            # Normalizing an unknown non-None cost model to None would drop the
            # explicit program and silently fall back to aggregate math; fail here
            # at the real WorkloadNode entry instead.
            raise ValueError(f"unknown DSA sparse source_cost_model: {node.attributes['source_cost_model']!r}")
        if node.attributes["source_tie_refinement_policy"] != "capacity_bound":
            raise ValueError("DSA sparse source_cost_model requires its explicit tie refinement policy")
    if source_enabled:
        heads, dim = node.shape["num_query_heads"], node.shape["qk_head_dim"]
        strides = tuple(node.attributes["source_query_strides_bytes"])
        offset = node.attributes["source_query_offset_bytes"]
        layout = node.attributes["source_q_input_layout"]
        backing = node.attributes["source_q_backing_heads"]
        if layout == "caller_padded_head_backing":
            if (type(backing) is not int or backing != -(-heads // 64) * 64
                    or strides != (backing * dim * 2, dim * 2, 2)
                    or offset % 16):
                raise ValueError("DSA sparse source query backing disagrees with its actual view strides/alignment")
        elif layout == "contiguous_unpadded":
            if (backing != heads or strides != (heads * dim * 2, dim * 2, 2)
                    or (heads % 64 == 0 and offset % 16)):
                raise ValueError("DSA sparse source query disagrees with its actual view strides/alignment")
        else:
            raise ValueError(f"unknown DSA sparse Q input layout: {layout!r}")
        for name in ("model_execution_version", "source_attention_backend", "source_block_size",
                "source_block_stride_rows", "source_cost_model", "source_extra_cache",
                "source_flashmla_snapshot", "source_hisparse", "source_index_inputs_layout",
                "source_index_remap_state", "source_index_remap_stream", "source_kv_cache_layout",
                "source_metadata_layout", "source_parallel_mode", "source_q_backing_heads",
                "source_q_input_layout", "source_attn_sink", "tensor_iterator_source",
                "source_tie_refinement_policy"):
            attributes[name] = node.attributes[name]
        attributes["source_component"] = "vllm_0_30_flashmla_sparse_bf16"
    if node.variant.startswith("dsa_") or "mathematical_program" in attributes or source_enabled:
        return KernelSpec(KernelKind.SPARSE_ATTN, hardware_id, node.shape, dtype=node.dtype,
            variant=node.variant, attributes=attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if node.variant == "dsv4_sparse_decode":
        # The per-request split-KV plan is a hardware launch decision owned
        # by the launch package (H200 vector_launch); Serving lowering
        # consumes it and never derives it.
        attributes["split_counts"] = select_dsv4_decode_split_counts(
            hardware_id, node.shape, None, node.variant, attributes,
            hardware)
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.SPARSE_ATTN, node.variant,
        node.shape, attributes)


def _target_kv_cache_update_spec(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
) -> KernelSpec:
    shape = _vector_shape(node)
    if node.variant == "mla_kv_cache":
        attribute_names = ("cache_layout", "cache_write_semantics")
    elif node.variant == "dsa_fp8_index_cache":
        attribute_names = (
            "cache_layout", "cache_write_semantics", "quantization")
    elif node.variant == "dsv4_compressed_state_cache":
        attribute_names = (
            "cache_layout", "cache_write_semantics", "quantization",
            "state_old_lens", "state_new_lens")
    elif node.variant == "dsv4_compressor_state_cache":
        attribute_names = ("cache_layout", "cache_write_semantics")
    elif node.variant == "dsv4_qnorm_rope_kv_insert":
        attribute_names = (
            "cache_layout", "cache_write_semantics", "quantization",
            "slot_mapping_all_valid")
        if "mathematical_program" in node.attributes:
            if node.attributes["mathematical_program"] != "vllm_0_30_dspark_partial_context_insert_v1":
                raise ValueError("unknown DSpark partial context insert mathematical program")
            return KernelSpec(kind=KernelKind.KV_CACHE, hardware_id=hardware_id,
                shape=shape, dtype=node.dtype, variant=node.variant,
                attributes={name: node.attributes[name] for name in (
                    *attribute_names, "mathematical_program", "valid_token_count")},
                launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    elif node.variant == "paged_kv_cache":
        attribute_names = (
            "cache_layout", "cache_head_layout", "kv_cache_dtype",
            "cache_storage_matches_input", "slot_mapping_all_valid",
            "vector_copy_base_alignment_bytes")
    else:
        raise ValueError(
            f"unknown target KV cache update variant: {node.variant!r}")
    attributes = {
        name: node.attributes[name]
        for name in attribute_names
    }
    if node.variant == "dsv4_compressed_state_cache" and "finish_stage" in node.attributes:
        attributes["finish_stage"] = node.attributes["finish_stage"]
    return _selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.KV_CACHE,
        node.variant, shape, attributes)


def _target_kv_cache_update_specs(node, hardware_id, hardware):
    if (node.variant == "dsv4_compressed_state_cache"
            and node.shape["compress_ratio"] == 128):
        if "finish_stage" in node.attributes and node.attributes["finish_stage"] is not None:
            raise ValueError("logical DSV4 finish node cannot preselect a split stage")
        # Validate the logical packed output through the real store spec;
        # selecting a parent aggregate would invent one physical launch for
        # two distinct source kernels. KernelSpec owns request-boundary and
        # payload validation, including zero completed states.
        store = _target_kv_cache_update_spec(dataclasses.replace(node,
            attributes={**node.attributes, "finish_stage": "store"}),
            hardware_id, hardware)
        states = sum(new // 128 - old // 128 for old, new in zip(
            store.attributes["state_old_lens"], store.attributes["state_new_lens"], strict=True))
        # seq_len remains the source token extent in both launches. Only
        # initialized scratch payload scales with completed boundaries.
        compress = _target_kv_cache_update_spec(
            dataclasses.replace(node,
                dtype="fp32",
                shape={**node.shape, "dtype_bytes": 4, "elements": states * 512},
                attributes={**node.attributes, "finish_stage": "compress"}),
            hardware_id, hardware)
        return compress, store
    return (_target_kv_cache_update_spec(node, hardware_id, hardware),)


def _flash_attn_selection_attributes(
    node: WorkloadNode,
    hardware_id: str,
) -> dict[str, object]:
    """Return the captured ABI fields without replacing logical work fields."""

    selection_attributes = dict(node.attributes)
    model_input_fields = {"model_input_q_lens", "model_input_kv_lens"}
    present_model_input_fields = model_input_fields & set(
        selection_attributes)
    if present_model_input_fields:
        if (present_model_input_fields != model_input_fields
                or "graph_runtime_mode" not in selection_attributes):
            raise ValueError(
                "FLASH_ATTN graph domains must be complete before launch "
                f"selection: present={sorted(present_model_input_fields)}")
        # vLLM-Ascend re-derives the FIA launch at every FULL-graph replay
        # via ``update_graph_params`` with the runtime ``seq_lens_list``:
        # real requests keep their logical lengths and the padded slots are
        # zero-filled (``model_runner_v1.py:732-734``).  Publish the logical
        # request lengths alongside the captured slot arrays so hardware
        # launch selectors can reflect that replay ABI.  The keys are inert
        # for selectors that do not consume them.
        selection_attributes["graph_logical_q_lens"] = tuple(
            selection_attributes["q_lens"])
        selection_attributes["graph_logical_kv_lens"] = tuple(
            selection_attributes["kv_lens"])
        selection_attributes["q_lens"] = tuple(
            selection_attributes["model_input_q_lens"])
        selection_attributes["kv_lens"] = tuple(
            selection_attributes["model_input_kv_lens"])
    normalized = dict(normalize_flash_attn_execution_attributes(
        hardware_id, node.shape, selection_attributes))
    normalized["is_decode"] = node.variant.lower() == "decode"
    return normalized


def _h200_legacy_norm_copy_source(node, hardware_id):
    """The actual 0.11 wrapper materializes its grouped strided input.

    Copy.cu routes this noncontiguous no-cast iterator to CUDALoops.cuh;
    the byte-offset bound includes the gaps between the real input groups.
    No pointer alignment is needed by this strided launch branch.
    """
    if (not HWSpec.source_contract_supported("pytorch_2_8_grouped_strided_norm_copy_source", hardware_id) or node.kind is not WorkloadKind.ELEMENTWISE
            or node.variant != "elementwise"
            or "model_execution_version" not in node.attributes
            or node.attributes["model_execution_version"] != "vllm_0_11_0"
            or "input_materialization" not in node.attributes
            or node.attributes["input_materialization"] != "grouped_strided_to_contiguous"):
        return None
    attrs = node.attributes
    groups, rows, group_stride, row_stride = (
        attrs[name] for name in _NORM_INPUT_MATERIALIZATION_FIELDS[1:])
    if (any(type(value) is not int or value <= 0
            for value in (groups, rows, group_stride, row_stride))
            or node.shape["elements"] != groups * rows * row_stride
            or group_stride <= rows * row_stride
            or (attrs["operation"], attrs["reads"], attrs["writes"]) != ("copy", 1, 1)):
        raise ValueError("legacy Norm copy must preserve its actual grouped input extent and strides")
    return {**attrs, "tensor_iterator_source": "pytorch_2_8_sm90",
        "source_input_layout": "strided_nocast",
        "source_maximum_offset_bytes": DTYPE_BYTES[node.dtype] * (
            (groups - 1) * group_stride + rows * row_stride - 1)}


def _compute_spec(node: WorkloadNode, hardware_id: str, hardware) -> KernelSpec:
    if node.dtype is None or not node.dtype.strip():
        raise ValueError(f"compute workload {node.node_id!r} requires dtype")
    if node.variant is None or not node.variant.strip():
        raise ValueError(f"compute workload {node.node_id!r} requires variant")
    if (node.kind is WorkloadKind.VECTOR and "source_component" in node.attributes
            and node.attributes["source_component"] == "symbolic_vllm030_kda_chunk_stage"):
        raise NotImplementedError("KDA chunk source children require reviewed per-stage KernelSpec and typed-service admission")
    if (HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id) and "source_component" in node.attributes
            and node.attributes["source_component"] == "vllm_0_11_symbolic_router_logits_cast"
            and node.attributes["launch_policy"] == "injected_launch"):
        attrs = node.attributes
        if (hardware.hardware_id != hardware_id or node.kind is not WorkloadKind.ELEMENTWISE
                or node.dtype != "bf16" or node.variant != "elementwise"
                or attrs["model_execution_version"] != "vllm_0_11_0"):
            raise ValueError("0.11 router cast requires its exact H200 BF16 source")
        geometry = attrs["source_moe_router_shape"]
        if (not isinstance(geometry, tuple) or len(geometry) != 3
                or any(type(value) is not int or value <= 0 for value in geometry)):
            raise ValueError("0.11 router cast requires actual tokens/experts/topk")
        tokens, experts, topk = geometry
        if dict(node.shape) != {"elements": tokens * experts}:
            raise ValueError("0.11 router cast extent differs from its logits matrix")
        semantic = {name: attrs[name] for name in ("operation", "reads", "writes")}
        # The selector's FP32 argument names the output lambda. KernelSpec
        # retains the BF16 input; its existing cast ABI writes real FP32 bytes.
        launch = select_launch_config(hardware_id, KernelKind.VECTOR,
            {"tokens": tokens, "experts": experts, "topk": topk, "element_count": tokens * experts},
            "fp32", "elementwise", {**attrs, "source_moe_router_stage": "logits_cast",
                "routing_input_dtype": "bf16"}, hardware)
        return KernelSpec(KernelKind.VECTOR, hardware_id, {"element_count": tokens * experts},
            "bf16", "elementwise", attributes=semantic, launch=launch)
    kind = _SINGLE_COMPUTE_KIND[node.kind]
    if kind is KernelKind.VECTOR and node.kind is WorkloadKind.NORM:
        physical_shape = node.shape
    elif kind is KernelKind.KV_CACHE:
        physical_shape = _vector_shape(node)
    elif kind is KernelKind.VECTOR:
        physical_shape = _vector_shape(node)
    elif kind in (KernelKind.GEMM, KernelKind.FLASH_ATTN):
        physical_shape = node.shape
    else:
        raise ValueError(f"unhandled compute shape kind: {kind.value}")
    if kind is KernelKind.FLASH_ATTN:
        shape = dict(physical_shape)
        token_domain = None
    else:
        shape, token_domain = _logical_prediction_shape(node, physical_shape)
    selection_attributes = dict(node.attributes)
    prediction_attributes = {}
    if (kind is KernelKind.VECTOR
            and node.kind is WorkloadKind.ELEMENTWISE
            and node.variant not in (
                "elementwise", "add", "masked-fill-zero")):
        raise ValueError(
            "unsupported elementwise workload variant: "
            f"{node.variant!r}")
    if kind is KernelKind.FLASH_ATTN:
        selection_attributes = _flash_attn_selection_attributes(
            node, hardware_id)
        prediction_attributes = {
            name: node.attributes[name]
            for name in (
                "causal", "kv_layout", "kv_dist", "fa_ksteps",
                "fa_eff_mfrac", "q_lens", "kv_lens")
        }
        if "source_program" in node.attributes:
            prediction_attributes["source_program"] = node.attributes["source_program"]
            if node.attributes["source_program"] == "ascend_0_18_full_fia_split_fuse":
                prediction_attributes["ascend_fia_prefill_no_cache"] = node.attributes["ascend_fia_prefill_no_cache"]
        model_input_fields = (
            "model_input_q_lens", "model_input_kv_lens",
        )
        present_model_input_fields = tuple(
            name for name in model_input_fields if name in node.attributes)
        if present_model_input_fields:
            if (present_model_input_fields != model_input_fields
                    or "graph_runtime_mode" not in node.attributes):
                raise ValueError(
                    "FLASH_ATTN graph prediction domains must be complete")
            prediction_attributes.update({
                name: node.attributes[name]
                for name in (*model_input_fields, "graph_runtime_mode")
            })
    elif kind is KernelKind.VECTOR:
        if node.kind in (WorkloadKind.ELEMENTWISE, WorkloadKind.VECTOR):
            prediction_attribute_names = ("operation", "reads", "writes")
        elif node.kind is WorkloadKind.GATHER:
            prediction_attribute_names = (
                "gather_factor", "index_count", "index_bytes", "scattered")
        elif node.kind is WorkloadKind.REDUCE:
            prediction_attribute_names = ("reduction_factor",)
        elif node.kind in (
                WorkloadKind.NORM, WorkloadKind.ACTIVATION,
                WorkloadKind.ROTARY, WorkloadKind.VECTOR):
            prediction_attribute_names = ()
        else:
            raise AssertionError(
                f"unhandled VECTOR workload kind: {node.kind.value}")
        missing_attributes = set(prediction_attribute_names) - set(node.attributes)
        if missing_attributes:
            raise ValueError(
                f"unsupported VECTOR/{node.variant} workload: missing required attributes "
                f"{sorted(missing_attributes)}")
        prediction_attributes = {
            name: node.attributes[name]
            for name in prediction_attribute_names
        }
        if node.kind is WorkloadKind.ROTARY and node.variant == "rope":
            prediction_attributes.update({name: node.attributes[name] for name in ("key_present", "is_neox")})
        if node.kind is WorkloadKind.GATHER and "materialized_output" in node.attributes:
            prediction_attributes["materialized_output"] = node.attributes["materialized_output"]
        if node.kind is WorkloadKind.GATHER and "index_select_source" in node.attributes:
            prediction_attributes["index_select_source"] = node.attributes["index_select_source"]
        if (node.kind is WorkloadKind.GATHER
                and hardware.supports_source_contract("pytorch_2_4_maca_materialized_gather_launch")
                and "launch_policy" in selection_attributes
                and selection_attributes["launch_policy"] == "injected_launch"
                and (("rope_composite_full_width" in node.attributes
                      and node.attributes["rope_composite_full_width"] is True)
                     or (node.variant == "embedding"
                         and "embedding_source" in node.attributes
                         and node.attributes["embedding_source"] == "vllm_0_24_unquantized_index_select"
                         and "source_operation" in node.attributes
                         and node.attributes["source_operation"] == "index_select_dim0"))):
            if "source_operation" in node.attributes and node.attributes["source_operation"] == "cat_last_dim":
                materialization_source = "pytorch_2_4_maca_cat_last_dim"
            elif (node.attributes["scattered"] and node.attributes["index_count"] == node.shape["rows"]
                    and "source_rows" in node.shape):
                materialization_source = "pytorch_2_4_maca_index_select"
            else:
                raise NotImplementedError("C500 source materialization lacks its contiguous source layout")
            prediction_attributes["materialization_source"] = materialization_source
            selection_attributes["materialization_source"] = materialization_source
        if node.kind is WorkloadKind.ELEMENTWISE and "tensor_iterator_source" in node.attributes:
            prediction_attributes["tensor_iterator_source"] = node.attributes["tensor_iterator_source"]
        if node.kind is WorkloadKind.ELEMENTWISE and "input_materialization" in node.attributes:
            prediction_attributes.update({name: node.attributes[name] for name in _NORM_INPUT_MATERIALIZATION_FIELDS})
        if (node.kind is WorkloadKind.ELEMENTWISE
                and "helper_launch" in node.attributes):
            # Helper/glue emission contract attribute: structural counting in
            # the alignment harness reads it from spec.attributes, so the
            # prediction-attribute filter must forward it explicitly.
            prediction_attributes["helper_launch"] = (
                node.attributes["helper_launch"])
        if (node.kind is WorkloadKind.GATHER
                and token_domain is not None):
            logical_tokens, model_input_tokens, _ = token_domain
            index_count = prediction_attributes["index_count"]
            if index_count % model_input_tokens:
                raise ValueError(
                    "GATHER index_count is not an integral graph-token "
                    "multiple")
            prediction_attributes["index_count"] = (
                index_count // model_input_tokens * logical_tokens)
    elif kind is KernelKind.KV_CACHE:
        prediction_attributes = {
            name: node.attributes[name]
            for name in (
                "cache_layout", "cache_head_layout", "kv_cache_dtype",
                "cache_storage_matches_input", "slot_mapping_all_valid",
                "vector_copy_base_alignment_bytes",
            )
        }
    elif kind is KernelKind.GEMM:
        prediction_attributes = {}
        if "source_gemm_runtime" in node.attributes:
            prediction_attributes["source_gemm_runtime"] = node.attributes["source_gemm_runtime"]
        if "output_dtype" in node.attributes:
            prediction_attributes["output_dtype"] = node.attributes["output_dtype"]
        if "helper_launch" in node.attributes:
            # Helper/glue emission contract attribute (e.g. the router-run
            # stamp on the ROUTER GEMM): structural counting in the
            # alignment harness reads it from spec.attributes.
            prediction_attributes["helper_launch"] = (
                node.attributes["helper_launch"])
    else:
        raise ValueError(
            f"unhandled prediction attributes for {kind.value}")
    vector_variants = {
        WorkloadKind.ROTARY: "rotary",
        WorkloadKind.REDUCE: "reduce",
        WorkloadKind.GATHER: "gather",
        WorkloadKind.VECTOR: "elementwise",
    }
    if kind is KernelKind.VECTOR:
        if node.kind in (WorkloadKind.NORM, WorkloadKind.ACTIVATION):
            variant = node.variant
            prediction_attributes = {}
            if (node.kind is WorkloadKind.NORM and node.variant == "rmsnorm"
                    and "epsilon" in node.attributes):
                prediction_attributes["epsilon"] = node.attributes["epsilon"]
            if "swiglu_limit" in node.attributes:
                prediction_attributes["swiglu_limit"] = node.attributes["swiglu_limit"]
                if selection_attributes["launch_policy"] != "analytical_launch":
                    raise NotImplementedError("clamped SiLU actual launch binding is unavailable")
        elif node.kind is WorkloadKind.ELEMENTWISE:
            elementwise_variants = {
                "elementwise": "elementwise",
                "add": "elementwise",
                "masked-fill-zero": "elementwise",
            }
            if node.variant not in elementwise_variants:
                raise ValueError(
                    "unsupported elementwise workload variant: "
                    f"{node.variant!r}")
            variant = elementwise_variants[node.variant]
        elif node.kind in vector_variants:
            variant = vector_variants[node.kind]
            if node.kind is WorkloadKind.ROTARY and node.variant == "rope":
                variant = "rope"
        else:
            raise ValueError(f"no VECTOR variant for {node.kind.value}")
    elif kind is KernelKind.KV_CACHE:
        if node.variant != "paged_kv_cache":
            raise ValueError(
                "KV cache workload requires canonical paged_kv_cache "
                f"variant, got {node.variant!r}")
        variant = node.variant
    elif kind in (KernelKind.GEMM, KernelKind.FLASH_ATTN):
        variant = node.variant
    else:
        raise ValueError(f"unhandled compute variant kind: {kind.value}")
    if kind is not KernelKind.FLASH_ATTN:
        prediction_attributes.update(_kernel_graph_token_attributes(
            token_domain, physical_shape))
    if "bf16_gemm_model" in node.attributes:
        if (kind is not KernelKind.GEMM
                or node.kind not in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD)
                or not hardware.supports_source_contract("h200_partial_source_programs")):
            raise ValueError("BF16 tiled reference requires ordinary H200 Linear/head")
        launch = select_launch_config(hardware_id, kind, physical_shape, node.dtype,
            variant, selection_attributes, hardware=hardware)
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=launch.execution_variant, launch=launch, attributes=prediction_attributes)
    if (kind is KernelKind.GEMM and "output_dtype" in prediction_attributes
            and prediction_attributes["output_dtype"] is not None
            and prediction_attributes["output_dtype"] != node.dtype
            and selection_attributes["launch_policy"] != "analytical_launch"):
        raise NotImplementedError("mixed-output GEMM actual launch binding is unavailable")
    if (node.dtype == "fp16" and kind is KernelKind.VECTOR and variant == "gather"
            and "index_select_source" in prediction_attributes
            and prediction_attributes["index_select_source"] == "vllm_0_11_embedding"
            and "symbolic_tensor_inputs" in node.attributes
            and selection_attributes["launch_policy"] == "analytical_launch"):
        # The source fixes the FP16/INT64 materialization payload. The
        # BF16-only installed-library binding cannot supply an FP16 launch.
        # Keep that source identity on the graph and lower its typed logical
        # work through the existing unbound FP16 mathematical contract.
        attributes = {key: value for key, value in prediction_attributes.items()
            if key != "index_select_source"}
        attributes["mathematical_program"] = "dense_fp16_source_flops_bytes_v1"
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=variant, attributes=attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (HWSpec.source_contract_supported("pytorch_2_8_embedding_index_select_launch", hardware_id)
            and kind is KernelKind.VECTOR and variant == "gather"
            and "index_select_source" in prediction_attributes
            and prediction_attributes["index_select_source"] == "vllm_0_11_embedding"):
        if not hardware.source_embedding_index_select_launch_domain(
                shape["rows"], shape["hidden"], shape["source_rows"], shape["element_count"]):
            # The source still determines typed index-select payload outside
            # the compiled launch's domain. Expose unbound mathematical work;
            # Midlayer keeps refusing it rather than borrowing that launch.
            return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
                variant=variant, attributes=prediction_attributes, launch=None,
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (hardware.supports_source_contract("pytorch_2_4_maca_int32_int64_launch")
            and kind is KernelKind.VECTOR and variant == "elementwise"
            and prediction_attributes["operation"] == "cast_int32_int64"
            and prediction_attributes["tensor_iterator_source"] == "aten_advanced_indexing_long"
            and physical_shape["element_count"] < 2048):
        prediction_attributes["tensor_iterator_source"] = "pytorch_2_4_maca_3_7_2"
        selection_attributes["tensor_iterator_source"] = "pytorch_2_4_maca_3_7_2"
    if HWSpec.source_contract_supported("aten_advanced_indexing_long_unbound_lowering", hardware_id) and (
            (HWSpec.source_contract_supported("vllm_0_11_embedding_unbound_index_select", hardware_id)
                and not HWSpec.source_contract_supported("pytorch_2_8_embedding_index_select_launch", hardware_id)
                and kind is KernelKind.VECTOR and variant == "gather"
                and "index_select_source" in prediction_attributes
                and prediction_attributes["index_select_source"] == "vllm_0_11_embedding")
            or (kind is KernelKind.VECTOR and variant == "elementwise"
                and prediction_attributes["operation"] == "cast_int32_int64"
                and prediction_attributes["tensor_iterator_source"] == "aten_advanced_indexing_long")):
        # The source fixes typed payloads, but the installed sm80 library
        # launch evidence belongs to A800. H200 has no such device binding.
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=variant, attributes=prediction_attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (node.dtype in ("bf16", "fp16") and node.kind is WorkloadKind.NORM
            and HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id)
            and "model_execution_version" in node.attributes
            and node.attributes["model_execution_version"] == "vllm_0_30_0"
            and "source_batch_invariant" not in node.attributes
            and selection_attributes["launch_policy"] == "injected_launch"):
        raise NotImplementedError("FP16 normalization native selection requires explicit source_batch_invariant")
    native_fp16_vector_declared = (node.dtype == "fp16"
        and HWSpec.source_contract_supported("vllm_source_vector_launch_binding", hardware_id)
        and node.kind in (WorkloadKind.NORM, WorkloadKind.ACTIVATION, WorkloadKind.ROTARY, WorkloadKind.KV_UPDATE)
        and variant in ("rmsnorm", "add_rmsnorm", "silu", "rotary", "rope", "paged_kv_cache")
        and {"model_execution_version", "source_input_layout"}.issubset(node.attributes)
        and (node.kind is not WorkloadKind.NORM or node.attributes["model_execution_version"] == "vllm_0_11_0"
            or "source_batch_invariant" in node.attributes)
        and ("swiglu_limit" not in node.attributes or node.attributes["swiglu_limit"] is None))
    if ((HWSpec.source_contract_supported("symbolic_fp16_unbound_source_lowering", hardware_id)
            or HWSpec.source_contract_supported("dense_fp16_symbolic_source_math", hardware_id)
                and not native_fp16_vector_declared) and node.dtype == "fp16"
            and "symbolic_tensor_inputs" in node.attributes
            and selection_attributes["launch_policy"] == "analytical_launch"):
        # Keep the C500 mathematical domain and unbound H200 operands.
        # An explicitly declared H200 vector/cache source reaches its exact
        # selector below; invalid source facts raise there without fallback.
        if token_domain is not None:
            raise NotImplementedError("FP16 source math does not bind graph-padded physical execution")
        if kind is KernelKind.GEMM:
            if (variant not in ("gemm", "gemv") or shape["b"] != 1
                    or "has_bias" not in node.attributes or node.attributes["has_bias"] is not False
                    or ("weight_dtype" in node.attributes and node.attributes["weight_dtype"] != "fp16")
                    or ("weight_group_size" in node.attributes and node.attributes["weight_group_size"] != 0)
                    or "weight_quantization" in node.attributes
                    or ("output_dtype" in prediction_attributes and prediction_attributes["output_dtype"] not in (None, "fp16"))):
                raise NotImplementedError("FP16 source matmul math requires homogeneous unquantized unbiased Linear")
            return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="fp16", variant="gemm",
                attributes={"mathematical_program": "unquantized_fp16_flops_bytes_v1",
                    "weight_dtype": "fp16", "weight_group_size": 0, "has_bias": False},
                launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
        if ((kind is KernelKind.VECTOR and variant in ("gather", "rmsnorm", "add_rmsnorm", "silu", "rotary"))
                or (kind is KernelKind.KV_CACHE and variant == "paged_kv_cache")):
            return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="fp16", variant=variant,
                attributes={**prediction_attributes, "mathematical_program": "dense_fp16_source_flops_bytes_v1"},
                launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
        raise NotImplementedError("this FP16 source operator has no bound C500 launch or explicit mathematical program")
    if (HWSpec.source_contract_supported("bf16_fp32_symbolic_matmul_source_math", hardware_id) and node.dtype == "bf16" and kind is KernelKind.GEMM
            and {"has_bias", "weight_dtype", "weight_group_size"}.issubset(node.attributes)
            and selection_attributes["launch_policy"] == "analytical_launch"
            and variant == "gemm" and "output_dtype" in prediction_attributes and prediction_attributes["output_dtype"] == "fp32"
            and node.attributes["has_bias"] is False and node.attributes["weight_dtype"] == "bf16"
            and node.attributes["weight_group_size"] == 0 and "weight_quantization" not in node.attributes):
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="bf16", variant="gemm",
            attributes={"output_dtype": "fp32", "weight_dtype": "bf16", "weight_group_size": 0,
                **({"helper_launch": prediction_attributes["helper_launch"]} if "helper_launch" in prediction_attributes else {})},
            launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (HWSpec.source_contract_supported("bf16_symbolic_matmul_source_math", hardware_id) and node.dtype == "bf16" and kind is KernelKind.GEMM
            and {"has_bias", "weight_dtype", "weight_group_size"}.issubset(node.attributes)
            and selection_attributes["launch_policy"] == "analytical_launch"
            and variant in ("gemm", "gemv") and shape["b"] == 1
            and node.attributes["has_bias"] is False
            and "weight_dtype" in node.attributes and node.attributes["weight_dtype"] == "bf16"
            and node.attributes["weight_group_size"] == 0 and "weight_quantization" not in node.attributes
            and ("output_dtype" not in prediction_attributes or prediction_attributes["output_dtype"] in (None, "bf16"))):
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="bf16", variant="gemm",
            attributes={"mathematical_program": "unquantized_bf16_flops_bytes_v1",
                "weight_dtype": "bf16", "weight_group_size": 0, "has_bias": False},
            launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (HWSpec.source_contract_supported("bf16_symbolic_vector_source_math", hardware_id) and node.dtype == "bf16"
            and "symbolic_tensor_inputs" in node.attributes
            and selection_attributes["launch_policy"] == "analytical_launch"
            and ((kind is KernelKind.VECTOR and variant in ("rmsnorm", "add_rmsnorm", "silu", "rotary", "elementwise"))
                or (kind is KernelKind.KV_CACHE and variant == "paged_kv_cache"))
            and not ({"model_execution_version", "source_input_layout", "source_batch_invariant"}.issubset(selection_attributes)
                or ({"model_execution_version", "source_input_layout"}.issubset(selection_attributes)
                    and selection_attributes["model_execution_version"] in ("vllm_0_30_0", "vllm_0_27_1")
                    and node.kind in (WorkloadKind.NORM, WorkloadKind.ACTIVATION, WorkloadKind.ROTARY, WorkloadKind.KV_UPDATE)
                    and variant in ("rmsnorm", "add_rmsnorm", "silu", "rotary", "paged_kv_cache")
                    and (node.kind is not WorkloadKind.NORM or "source_batch_invariant" in selection_attributes)
                    and ("swiglu_limit" not in prediction_attributes or prediction_attributes["swiglu_limit"] is None)))
            and not {"tensor_iterator_source", "source_input_layout"}.issubset(selection_attributes)):
        # Only Norm dispatch reads the batch-invariant flag. SiLU, RoPE and
        # cache source wrappers need their version/layout, not that flag.
        # The virtual source fixes the arithmetic and typed storage even
        # when provider/layout/branch facts do not identify a native launch.
        # Reuse the existing source-work implementation with BF16 operands;
        # no SM80 launch, missing flag or native price is substituted.
        if token_domain is not None:
            raise NotImplementedError("BF16 source math does not bind graph-padded native execution")
        if (kind is KernelKind.VECTOR and variant == "silu" and "swiglu_limit" in prediction_attributes
                and prediction_attributes["swiglu_limit"] is not None):
            return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="bf16", variant=variant,
                attributes=prediction_attributes, launch=None,
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype="bf16", variant=variant,
            attributes={**prediction_attributes, "mathematical_program": "dense_bf16_source_flops_bytes_v1"},
            launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id)
            and HWSpec.source_contract_supported("unbound_materialized_gather_source_math", hardware_id)
            and kind is KernelKind.VECTOR and variant == "gather" and node.dtype in ("bf16", "fp16")
            and node.kind is WorkloadKind.GATHER and node.variant == "embedding"
            and selection_attributes["launch_policy"] == "injected_launch"
            and {"embedding_source", "model_execution_version", "source_operation"}.issubset(node.attributes)
            and node.attributes["embedding_source"] == "vllm_0_30_unquantized_tp1"
            and node.attributes["model_execution_version"] == "vllm_0_30_0"
            and node.attributes["source_operation"] == "embedding"
            and {"rows", "hidden", "source_rows"}.issubset(physical_shape)
            and {"index_bytes", "index_count", "gather_factor", "scattered", "materialized_output"}.issubset(prediction_attributes)
            and prediction_attributes["index_bytes"] == 8
            and prediction_attributes["index_count"] == physical_shape["rows"]
            and prediction_attributes["gather_factor"] == 1
            and prediction_attributes["scattered"] is True
            and prediction_attributes["materialized_output"] is True
            and "index_select_source" not in prediction_attributes):
        # The approved performance-only helper policy prices the existing
        # selected-row read + fresh output + Long-ID ledger. This does not
        # identify Torch/compiler resources or widen native launch support.
        if token_domain is not None:
            raise NotImplementedError("FP16 embedding helper does not bind graph-padded native execution")
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=variant, attributes=prediction_attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    if (HWSpec.source_contract_supported("unbound_materialized_gather_source_math", hardware_id) and kind is KernelKind.VECTOR and variant == "gather"
            and selection_attributes["launch_policy"] == "analytical_launch"
            and "index_select_source" not in prediction_attributes):
        # Existing source typed-byte math does not identify an ATen/native
        # embedding program. Select that declared analytical domain before
        # native lookup; do not invent missing runtime/source_rows facts.
        if token_domain is not None:
            raise NotImplementedError("generic gather source does not bind graph-padded native execution")
        return KernelSpec(kind=kind, hardware_id=hardware_id, shape=shape, dtype=node.dtype,
            variant=variant, attributes=prediction_attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
    norm_copy_source = _h200_legacy_norm_copy_source(node, hardware_id)
    if norm_copy_source is not None:
        selection_attributes = norm_copy_source
    launch = select_launch_config(
        hardware_id, kind, physical_shape, node.dtype, variant,
        selection_attributes, hardware=hardware)
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=shape,
        dtype=node.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes=prediction_attributes,
    )


def _symbolic_dsv3_l32_tp4_projection(node, pool):
    """Consume actual validated model/source facts, not a shape-only TP4 label."""
    from src.serving.workload.model_config import DEEPSEEK_V3_L32_CONFIG_SHA256
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("symbolic_dsv3_routed_experts", pool.hardware_id)
        and config.tensor_parallel_size == 4
        and len(pool.ranks) == 4 and len(pool.parallel_plan.layer_to_pipeline_stage) == 32
        and not config.enable_expert_parallel and all(value == 1 for value in (
            config.pipeline_parallel_size, config.sequence_parallel_size,
            config.context_parallel_size, config.data_parallel_size))
        and "source_component" in node.attributes and node.attributes["source_component"] in (
            "symbolic_mla_decode_mathematical", "symbolic_mla_cache_update",
            "vllm_0_27_1_symbolic_grouped_routing", "symbolic_dsv3_routed_experts",
            "vllm_0_27_1_symbolic_mtp_eager", "vllm_0_27_1_symbolic_mtp_position_where")
        and "source_model_config_sha256" in node.attributes
        and node.attributes["source_model_config_sha256"] == DEEPSEEK_V3_L32_CONFIG_SHA256
        and "source_model_profile" in node.attributes
        and node.attributes["source_model_profile"] == "vllm_0_27_1_deepseek_v3_sync_fp8"
        and "source_tensor_parallel_size" in node.attributes
        and type(node.attributes["source_tensor_parallel_size"]) is int
        and node.attributes["source_tensor_parallel_size"] == 4)


def _source_flash_attn_specs(node, hardware_id, hardware):
    if "source_component" in node.attributes and node.attributes["source_component"] == "fa3_expanded_mla_prefill_call":
        revision = hardware.fa3_mla_source_revision_for_model(node.attributes["source_model_execution_version"])
        if (node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_prefill" or node.dtype != "bf16"
                or (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) != (256, 256, 512)
                or node.attributes["fa3_expanded_source_revision"] != revision
                or node.attributes["fa3_backend_source"] != "vllm_0_30_default_hopper_flash_attn3"
                or node.attributes["scheduler_metadata_precomputed"] is not False
                or node.attributes["fa3_runtime_binary_bound"] is not False
                or node.attributes["fa3_length_array_transport_bound"] is not False
                or node.attributes["fa3_native_workspace_bound"] is not False
                or node.attributes["causal"] is not True or node.attributes["return_softmax_lse"] is not False
                or node.attributes["q_lens"] != node.attributes["kv_lens"]
                or node.attributes["full_cuda_graphs"] is not False
                or type(node.attributes["max_cudagraph_capture_size"]) is not int
                or node.attributes["max_cudagraph_capture_size"] != 0
                or "flash_attn_version" in node.attributes and (
                    type(node.attributes["flash_attn_version"]) is not int or node.attributes["flash_attn_version"] != 3)
                or any(field in node.attributes for field in ("attention_backend", "graph_runtime_mode",
                    "graph_logical_tokens", "graph_model_input_tokens", "model_input_shape",
                    "model_input_q_lens", "model_input_kv_lens"))):
            raise ValueError("expanded FA3 prefill requires its exact fresh eager source scenario")
        attrs = {name: node.attributes[name] for name in (
            "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "return_softmax_lse",
            "full_cuda_graphs", "max_cudagraph_capture_size")}
        return hardware.fa3_mla_forward_specs(node.shape, node.dtype, node.variant, attrs,
            scheduler_metadata_precomputed=False)
    if "fa3_full_aot_component" in node.attributes:
        from dataclasses import replace
        component=node.attributes["fa3_full_aot_component"]
        if not hardware.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill") or node.dtype!="int32":
            raise ValueError("FA3 AOT helper requires its exact H200 INT32 caller")
        if component=="prepare":
            attrs={name:node.attributes[name] for name in ("q_lens","kv_lens","model_input_q_lens","model_input_kv_lens",
                "graph_runtime_mode","graph_model_input_tokens")}
            attrs.update({"source_program":"vllm_0_30_fa3_gqa_full_decode","causal":True,"kv_layout":"paged_varlen","kv_dist":"uniform"})
            if (node.attributes["fa3_full_aot_head_shape"] != tuple(node.shape[field] for field in (
                    "num_query_heads", "num_kv_heads", "head_dim"))
                    or node.attributes["fa3_gqa_operand_dtype"] not in ("bf16", "fp16")):
                raise ValueError("FA3 AOT prepare must preserve the actual rank-local head geometry and operand dtype")
            selected=hardware.fa3_gqa_forward_specs(node.shape,node.attributes["fa3_gqa_operand_dtype"],attrs,
                page_size=16,num_splits=32,scheduler_metadata_precomputed=False)[0]
            return (replace(selected,attributes={"source_program":"vllm_0_30_fa3_full_aot_prepare",
                "use_dynamic_split":node.shape["batch"]<=992,"head_swizzle":True}),)
        if component=="clear":
            return (KernelSpec(kind=KernelKind.VECTOR,hardware_id=hardware_id,shape={"elements":node.shape["elements"]},
                dtype="int32",variant="elementwise",attributes={"reads":0,"writes":1,"operation":"fill",
                    "mathematical_program":"vllm_0_30_fa3_aot_tail_clear"},launch=None,execution_support="unsupported",
                    unsupported_reason="launch_selector_unavailable"),)
        raise ValueError("FA3 AOT helper component is unknown")
    if "fa3_gqa_caller_scenario" in node.attributes:
        from .attention_memory_bindings import fa3_gqa_caller_facts
        metadata = node.attributes["source_component"] == "fa3_gqa_shared_metadata_prepare"
        expected = fa3_gqa_caller_facts(node.attributes,hardware)
        if any(type(node.attributes[name]) is not type(value) or node.attributes[name] != value
                for name, value in expected.items()):
            raise ValueError("FA3 GQA caller facts differ from the declared source scenario")
        full = ("source_program" in node.attributes
            and node.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
        if not full and (node.attributes["graph_runtime_mode"] != "NONE"
                or any(name in node.attributes for name in (
                    "graph_logical_tokens", "graph_model_input_tokens", "model_input_q_lens", "model_input_kv_lens"))):
            raise ValueError("FA3 GQA source caller has unbound captured/padded inputs")
        attributes = {name: node.attributes[name] for name in (
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens", "source_model_execution_version")}
        if full:
            attributes.update({name:node.attributes[name] for name in ("source_program", "graph_runtime_mode",
                "graph_model_input_tokens", "model_input_q_lens", "model_input_kv_lens")})
        rows = attributes["graph_model_input_tokens"] if full else sum(attributes["q_lens"])
        if "fa3_gqa_query_tokens" in expected and rows!=expected["fa3_gqa_query_tokens"]:
            raise ValueError("FA3 source split policy must use the actual caller query extent")
        for name in ("flash_attn_version", "attention_backend", "full_cuda_graphs", "max_cudagraph_capture_size"):
            if name in node.attributes:
                attributes[name] = node.attributes[name]
        parts = hardware.fa3_gqa_forward_specs(node.shape, node.attributes["fa3_gqa_operand_dtype"], attributes,
            page_size=node.attributes["fa3_gqa_page_size"], num_splits=node.attributes["fa3_gqa_num_splits"],
            scheduler_metadata_precomputed=False if metadata else node.attributes["fa3_gqa_scheduler_metadata_precomputed"])
        return (parts[0],) if metadata else parts
    if ("source_component" in node.attributes
            and node.attributes["source_component"] in (
                "fa3_mla_split_prefill_call", "fa3_mla_unsplit_prefill_call")):
        if (node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_prefill"
                or node.dtype != "bf16"
                or node.attributes["source_component"] == "fa3_mla_split_prefill_call" and node.attributes["causal"] is not False
                or node.attributes["scheduler_metadata_precomputed"] is not False
                or node.attributes["fa3_backend_source"] != hardware.fa3_mla_model_backend_source(
                    node.attributes["source_model_execution_version"])
                or node.attributes["fa3_runtime_binary_bound"] is not False
                or node.attributes["fa3_length_array_transport_bound"] is not False
                or "attention_backend" in node.attributes
                or any(name in node.attributes for name in (
                    "graph_runtime_mode", "graph_logical_tokens", "graph_model_input_tokens",
                    "model_input_shape", "model_input_q_lens", "model_input_kv_lens"))):
            raise ValueError("FA3 model source requires its explicit eager default-backend call")
        if "flash_attn_version" in node.attributes and (
                type(node.attributes["flash_attn_version"]) is not int or node.attributes["flash_attn_version"] != 3):
            raise ValueError("FA3 model source cannot replace a FlashAttention version override")
        revision = hardware.fa3_mla_source_revision_for_model(node.attributes["source_model_execution_version"])
        if node.attributes["fa3_source_revision"] != revision:
            raise ValueError("FA3 model source and selected dependency revision disagree")
        hardware.require_fa3_mla_source_revision(revision)
        attributes = {name: node.attributes[name] for name in (
            "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "full_cuda_graphs",
            "max_cudagraph_capture_size", "return_softmax_lse")}
        return hardware.fa3_mla_forward_specs(node.shape, node.dtype, node.variant, attributes,
            scheduler_metadata_precomputed=node.attributes["scheduler_metadata_precomputed"])
    if ("source_component" not in node.attributes or node.attributes["source_component"] != "symbolic_gqa_mathematical"):
        return _flash_attn_specs(node,hardware_id,hardware)
    if node.kind is not WorkloadKind.FLASH_ATTN or node.variant not in ("prefill", "decode"):
        raise ValueError("symbolic GQA math requires an ordinary attention node")
    if ("source_model_execution_version" not in node.attributes
            or node.attributes["source_model_execution_version"] not in
                ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0")):
        raise ValueError("symbolic GQA requires its explicit model source version")
    ascend_fia = ("mathematical_program" in node.attributes
        and node.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1")
    if node.attributes["launch_policy"] == "injected_launch":
        # full_graph_fia uses TND/sparse_mode=3 and torch_npu's inner_precise=0
        # default. The matching CPU tiler selects SplitFuse, not the regular
        # FIA selected by the generic decode label. Cached calls retain their
        # independent zero-KV tail; fresh prefill reads contiguous physical KV
        # and uses cumulative boundaries, including an inert zero-row tail.
        bound_fia = ascend_fia and hardware.supports_source_contract("vllm_0_18_full_fia_split_fuse")
        if bound_fia:
            native_node = dataclasses.replace(node, attributes={**node.attributes,
                "source_program": "ascend_0_18_full_fia_split_fuse",
                "kv_layout": "contiguous" if node.attributes["ascend_fia_prefill_no_cache"] else node.attributes["kv_layout"]})
            return _flash_attn_specs(native_node, hardware_id, hardware)
        if not ascend_fia:
            if (hardware.supports_source_contract("vllm_0_18_eager_fia_direct_kv")
                    and node.attributes["source_model_execution_version"] == "vllm_0_18_0"
                    and node.attributes["graph_runtime_mode"] == "NONE"
                    and (node.variant == "prefill" or tuple(node.attributes["q_lens"]) == tuple(node.attributes["kv_lens"]))):
                # _get_fia_params uses direct TND KV for PrefillNoCache;
                # no block table or paged-address dependency belongs to it.
                fresh = tuple(node.attributes["q_lens"]) == tuple(node.attributes["kv_lens"])
                native_node = dataclasses.replace(node, variant="prefill" if fresh else node.variant,
                    attributes={**node.attributes, "kv_layout": "contiguous" if fresh else node.attributes["kv_layout"]})
                return _flash_attn_specs(native_node, hardware_id, hardware)
            return _flash_attn_specs(node, hardware_id, hardware)
    if node.attributes["launch_policy"] not in ("analytical_launch", "injected_launch"):
        raise ValueError("symbolic GQA requires an explicit analytical or injected launch policy")
    if (hardware.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill")
            and node.attributes["source_model_execution_version"] == "vllm_0_30_0"
            and node.attributes["launch_policy"] == "analytical_launch"
            and node.dtype == "bf16"
            and node.variant == "prefill"):
        # This mathematical source binds only the main. Complete caller
        # facts above retain their own setup, combine and semaphore Fill.
        source_launch = select_source_flash_attn_partial_launch(
            hardware_id, node.shape, node.dtype, node.variant, node.attributes, hardware)
        if source_launch is not None:
            return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id,
                shape=node.shape, dtype=node.dtype, variant=node.variant, launch=source_launch,
                attributes={key: node.attributes[key] for key in (
                    "causal", "kv_layout", "kv_dist", "fa_ksteps", "fa_eff_mfrac", "q_lens", "kv_lens")}),)
    return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id,
        shape=node.shape, dtype=node.dtype, variant=node.variant,
        attributes={"mathematical_program": "ascend_0_18_full_fia_softmax_v1" if ascend_fia else "gqa_softmax_v1",
            **({"ascend_fia_prefill_no_cache": node.attributes["ascend_fia_prefill_no_cache"]} if ascend_fia else {}), **{
            key: node.attributes[key] for key in (
                "causal", "kv_layout", "kv_dist", "fa_ksteps", "fa_eff_mfrac", "q_lens", "kv_lens")},
            **({key: node.attributes[key] for key in ("graph_runtime_mode", "model_input_q_lens", "model_input_kv_lens")}
                if "model_input_q_lens" in node.attributes else {})},
        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)


def _flash_attn_specs(node: WorkloadNode, hardware_id: str, hardware):
    # The backend partitions the request ABI it is called with: the padded
    # model-input requests inside a FULL Graph replay (vLLM pads attention
    # metadata only there), the logical requests otherwise.
    if ("graph_runtime_mode" in node.attributes
            and node.attributes["graph_runtime_mode"] == "FULL"
            and "model_input_q_lens" in node.attributes):
        q_lens = tuple(node.attributes["model_input_q_lens"])
        kv_lens = tuple(node.attributes["model_input_kv_lens"])
    else:
        q_lens = tuple(node.attributes["q_lens"])
        kv_lens = tuple(node.attributes["kv_lens"])
    parts = select_flash_attn_forward_partition(
        hardware_id, node.shape, node.variant,
        {**node.attributes, "q_lens": q_lens, "kv_lens": kv_lens})
    if not parts:
        return _whole_flash_attn_specs(
            node, hardware_id, hardware, output_fill_rows=None)
    # The backend runs the call as several pathways over disjoint request
    # subsets, each on its own requests, then (when the backend allocates a
    # fresh result) slice-assigns that result into the shared output buffer.
    # The output buffer belongs to the whole call: it is zero-filled once,
    # ahead of every pathway, over all of the call's query rows.  The first
    # pathway carries that fill; the others share the buffer and emit none.
    call_rows = sum(q_lens)
    specs = []
    for position, part in enumerate(parts):
        part_q_lens = tuple(q_lens[index] for index in part.request_indexes)
        part_kv_lens = tuple(kv_lens[index] for index in part.request_indexes)
        attributes = {
            name: value for name, value in node.attributes.items()
            if name not in ("model_input_q_lens", "model_input_kv_lens")
        }
        attributes.update({
            "q_lens": part_q_lens,
            "kv_lens": part_kv_lens,
            "flash_attn_forward_role": part.role,
        })
        part_node = dataclasses.replace(
            node,
            shape={
                **node.shape,
                "batch": len(part_q_lens),
                "q_len": max(part_q_lens),
                "kv_len": max(part_kv_lens),
            },
            variant=part.variant,
            attributes=attributes,
        )
        specs.extend(_whole_flash_attn_specs(
            part_node, hardware_id, hardware,
            output_fill_rows=call_rows if position == 0 else 0))
        if part.output_copy_launch is not None:
            if (part.output_copy_launch.algorithm
                    != "native_d2d_flash_attention_copy"):
                raise ValueError(
                    "FlashAttention pathway output copy must be the native "
                    "D2D FlashAttention copy")
            if node.dtype != "bf16":
                raise ValueError(
                    "FlashAttention pathway output copy supports bf16 only")
            specs.append(KernelSpec(
                kind=KernelKind.MEMCPY,
                hardware_id=hardware_id,
                transfer=PhysicalTransferConfig(
                    algorithm="native_d2d_flash_attention_copy",
                    bytes=part.output_copy_elements * 2,
                    direction="d2d",
                    link="dram",
                ),
            ))
    return tuple(specs)


def _whole_flash_attn_specs(
        node: WorkloadNode, hardware_id: str, hardware,
        output_fill_rows: int | None):
    """Lower one backend FlashAttention invocation.

    ``output_fill_rows`` is ``None`` for a call lowered whole, whose output
    fill covers its own execution shape; a pathway of a partitioned call
    passes the query rows of the whole call it zero-fills, or ``0`` when an
    earlier pathway of the same call already carries that fill.
    """

    main = _compute_spec(node, hardware_id, hardware)
    tiling = main.launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("FLASH_ATTN launch must declare num_kv_splits")
    splits = tiling.num_kv_splits
    execution_attributes = _flash_attn_selection_attributes(
        node, hardware_id)
    setup_specs = tuple(
        KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=selection.shape,
            dtype=selection.dtype,
            variant=selection.variant,
            launch=selection.launch,
            attributes={
                **selection.attributes,
                **_derived_graph_token_attributes(main, selection.shape),
            },
        )
        for selection in select_flash_attn_setup_launches(
            hardware_id, main.shape, main.dtype, main.variant,
            execution_attributes, hardware)
    )
    compute_specs = [main]
    if splits > 1:
        selection_attributes = execution_attributes
        selection_attributes["num_kv_splits"] = splits
        combine_launch = select_launch_config(
            hardware_id, KernelKind.FLASH_ATTN, main.shape, main.dtype,
            "combine", selection_attributes, hardware=hardware)
        compute_specs.append(KernelSpec(
            kind=KernelKind.FLASH_ATTN,
            hardware_id=hardware_id,
            shape=main.shape,
            dtype=main.dtype,
            variant="combine",
            launch=combine_launch,
            attributes=main.attributes,
        ))

    def copy_spec(dtype: str, elements: int, launch) -> KernelSpec:
        if launch.algorithm == "native_d2d_flash_attention_copy":
            dtype_bytes = {"bf16": 2, "float32": 4}
            if dtype not in dtype_bytes:
                raise ValueError(
                    f"unsupported native FlashAttention copy dtype: {dtype!r}")
            # A native copy launch is a physical D2D transfer rather than a
            # Vector compute kernel.  The selector's element count is the
            # source-defined output view: one read and one write per value.
            return KernelSpec(
                kind=KernelKind.MEMCPY,
                hardware_id=hardware_id,
                transfer=PhysicalTransferConfig(
                    algorithm="native_d2d_flash_attention_copy",
                    bytes=elements * dtype_bytes[dtype],
                    direction="d2d",
                    link="dram",
                ),
            )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": elements},
            dtype=dtype,
            variant="elementwise",
            launch=launch,
            attributes={
                "reads": 1,
                "writes": 1,
                "operation": "copy",
                **_derived_graph_token_attributes(
                    main, {"element_count": elements}),
            },
        )

    def fill_spec(dtype: str, elements: int, launch) -> KernelSpec:
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={"element_count": elements},
            dtype=dtype,
            variant="elementwise",
            launch=launch,
            attributes={
                "reads": 0,
                "writes": 1,
                "operation": "fill",
                **_derived_graph_token_attributes(
                    main, {"element_count": elements}),
            },
        )

    # The backend-owned output buffer zero-fill (vLLM Attention.forward
    # use_output branch) runs before the framework copies that surround the
    # captured FA invocation, on the same normalized padded execution shape.
    if output_fill_rows is None:
        fill_shape = main.shape
    else:
        # One row per query token of the whole call.
        fill_shape = {**main.shape, "batch": output_fill_rows, "q_len": 1}
    source_managed_buffers = ("source_component" in node.attributes
        and node.attributes["source_component"] == "symbolic_gqa_mathematical")
    legacy_cuda_buffers = (source_managed_buffers
        and "source_model_execution_version" in node.attributes
        and node.attributes["source_model_execution_version"] == "vllm_0_11_0")
    # Source versions own different buffer programs: vLLM0.11 allocates
    # zeros in Attention.forward, and its existing FA2 native model includes
    # the wrapper copies. Newer source profiles use their declared buffers.
    if legacy_cuda_buffers:
        source_rows = node.attributes["source_output_rows"]
        if type(source_rows) is not int or source_rows < sum(node.attributes["q_lens"]):
            raise ValueError("vLLM0.11 output fill requires its complete physical query rows")
        fill_shape = {**main.shape, "batch": source_rows, "q_len": 1}
    output_fills = () if output_fill_rows == 0 or (source_managed_buffers and not legacy_cuda_buffers) else tuple(
        fill_spec(dtype, elements, launch)
        for dtype, elements, launch in select_flash_attn_output_fill_launches(
            hardware_id, fill_shape, main.dtype, execution_attributes,
            hardware))
    # MetaX0.24 allocates an empty frontend output, but its decode backend
    # returns a fresh result and assigns it into that output. This copy is
    # independent of the removed frontend zero-fill and Q/K norm copies.
    source_backend_copies = legacy_cuda_buffers or (source_managed_buffers
        and node.attributes["source_model_execution_version"] == "vllm_0_24_0")
    copy_launches = () if (source_managed_buffers and not source_backend_copies) else select_flash_attn_copy_launches(
        # A framework copy surrounds the physical captured FA invocation.
        # Keep its selector on the same normalized execution ABI as main,
        # setup, and combine: q_lens/kv_lens here describe the captured
        # request slots, never the logical active-row workload.
        hardware_id, main.shape, execution_attributes, hardware)
    before = tuple(
        copy_spec(dtype, elements, launch)
        for position, dtype, elements, launch in copy_launches
        if position == "before")
    after = tuple(
        copy_spec(dtype, elements, launch)
        for position, dtype, elements, launch in copy_launches
        if position == "after")

    # vLLM-Ascend's ``Attention.forward`` performs source-visible TensorMove
    # assignments around the fused FIA call.  CANN's FIA selector does not
    # expose those framework moves, so the workload builder carries the
    # source-authored logical/physical row plan and 910B materializes it here.
    # This is intentionally a discrete source contract; no serving duration
    # or residual correction participates in the launch choice.
    source_copies = ()
    if (hardware.kernel_memory_model == "npu_tile"
            and hardware.aic_aiv_separated
            and "source_output_copy_plan" in node.attributes):
        source_copy_plan = node.attributes["source_output_copy_plan"]
        if (not isinstance(source_copy_plan, tuple)
                or not source_copy_plan
                or any(role not in ("logical", "physical")
                       for role in source_copy_plan)):
            raise ValueError(
                "910B FlashAttention source_output_copy_plan must contain "
                "logical/physical roles")
        q_lens = node.attributes["q_lens"]
        model_input_q_lens = (
            node.attributes["model_input_q_lens"]
            if "model_input_q_lens" in node.attributes else q_lens)
        if (not isinstance(q_lens, tuple)
                or not isinstance(model_input_q_lens, tuple)):
            raise ValueError(
                "910B FlashAttention output-copy row domains must be tuples")
        logical_rows = sum(q_lens)
        physical_rows = sum(model_input_q_lens)
        if logical_rows <= 0 or physical_rows < logical_rows:
            raise ValueError(
                "910B FlashAttention output-copy rows are inconsistent")
        hidden = node.shape["num_query_heads"] * node.shape["head_dim"]
        selection_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": node.attributes["launch_policy"],
        }
        copy_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
        }
        generated = []
        for index, role in enumerate(source_copy_plan):
            rows = logical_rows if role == "logical" else physical_rows
            shape = {"element_count": rows * hidden}
            launch = select_launch_config(
                hardware_id,
                KernelKind.VECTOR,
                shape,
                node.dtype,
                "elementwise",
                selection_attributes,
                hardware=hardware,
            )
            generated.append(KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=shape,
                dtype=node.dtype,
                variant=launch.execution_variant,
                launch=launch,
                attributes={
                    **copy_attributes,
                    **_derived_graph_token_attributes(main, shape),
                },
            ))
        source_copies = tuple(generated)
    return (setup_specs + output_fills + before + tuple(compute_specs)
            + after + source_copies)


_NORM_INPUT_MATERIALIZATION_FIELDS = (
    "input_materialization",
    "input_outer_groups",
    "input_rows_per_group",
    "input_group_stride_elements",
    "input_row_stride_elements",
)


def _norm_specs(node: WorkloadNode, hardware_id: str, hardware):
    """Lower a source-required input materialization before the Norm call."""

    main = _compute_spec(node, hardware_id, hardware)
    present = tuple(
        name for name in _NORM_INPUT_MATERIALIZATION_FIELDS
        if name in node.attributes)
    if not present:
        return (main,)
    if present != _NORM_INPUT_MATERIALIZATION_FIELDS:
        raise ValueError(
            "Norm input materialization contract is incomplete: "
            f"present={present!r}")
    if node.attributes["input_materialization"] != (
            "grouped_strided_to_contiguous"):
        raise ValueError(
            "unsupported Norm input materialization: "
            f"{node.attributes['input_materialization']!r}")
    dimensions = {
        name: node.attributes[name]
        for name in _NORM_INPUT_MATERIALIZATION_FIELDS[1:]
    }
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in dimensions.values()):
        raise TypeError(
            "Norm input materialization dimensions must be integers")
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError(
            "Norm input materialization dimensions must be positive")
    rows = node.shape["rows"]
    hidden = node.shape["hidden"]
    if dimensions["input_outer_groups"] * dimensions[
            "input_rows_per_group"] != rows:
        raise ValueError(
            "Norm grouped input rows disagree with the normalized shape")
    if dimensions["input_row_stride_elements"] != hidden:
        raise ValueError(
            "Norm grouped input row stride must equal hidden width")
    if dimensions["input_group_stride_elements"] <= (
            dimensions["input_rows_per_group"] * hidden):
        raise ValueError(
            "Norm grouped input must contain a physical inter-group gap")
    selection_attributes = {
        "operation": "copy",
        "reads": 1,
        "writes": 1,
        "launch_policy": node.attributes["launch_policy"],
        "input_materialization": node.attributes["input_materialization"],
        **dimensions,
    }
    copy_shape = {"element_count": rows * hidden}
    copy_launch = select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        copy_shape,
        node.dtype,
        "elementwise",
        selection_attributes,
        hardware=hardware,
    )
    copy = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=copy_shape,
        dtype=node.dtype,
        variant=copy_launch.execution_variant,
        launch=copy_launch,
        attributes={
            "reads": 1,
            "writes": 1,
            "operation": "copy",
            "input_materialization": node.attributes[
                "input_materialization"],
            **dimensions,
            **_derived_graph_token_attributes(main, copy_shape),
        },
    )
    return copy, main


def _ascend_qkv_rmsnorm_rope_fusions(
        graph: WorkloadGraph, deployment: DeploymentPlan,
        hardware_by_id: Mapping[str, object]):
    """Identify source-exact vLLM-Ascend Q/K Norm plus RoPE programs.

    The compiler pass matches a structural three-op chain and replaces it
    with one physical split-QKV/RMSNorm/RoPE task. This matcher mirrors that
    graph contract; it does not inspect model names, Serving case IDs, or
    measured durations.
    """

    # The canonical launch observation contradicts the fused-program
    # premise on every captured NPU serving case: the launch signatures of
    # the {qwen3-4b,qwen3-8b}/tp1_qps16 alignment captures under
    # serving_dataset/sharegpt/<hardware>/
    # contain only the separate AddRmsNormBias, RmsNorm and _triton_rope
    # kernels and no split_qkv_rmsnorm_rope signature, and the msprof
    # per-kernel counts agree (73 RmsNorm + 72 AddRmsNormBias + 36
    # _triton_rope per step for a 36-layer model).  Emitting one fused task
    # therefore under-counts both the launch occurrence and its device work.
    # Keep the matcher inert until a capture actually shows the fused
    # signature; the Q/K Norm and RoPE nodes then lower through their own
    # measured vector programs.
    return {}, {}
    successors = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        successors[edge.source].append(edge.destination)
    head_to_group = {}
    member_to_head = {}
    for rotary in graph.nodes:
        if (rotary.kind is not WorkloadKind.ROTARY
                or rotary.variant != "rotary"
                or rotary.dtype != "bf16"):
            continue
        if rotary.pool_name not in deployment.worker_pools:
            continue
        pool = deployment.worker_pools[rotary.pool_name]
        if pool.hardware_id not in hardware_by_id:
            continue
        hardware = hardware_by_id[pool.hardware_id]
        if (hardware.kernel_memory_model != "npu_tile"
                or not hardware.aic_aiv_separated):
            continue
        k_predecessors = graph.predecessors(rotary.node_id)
        if len(k_predecessors) != 1:
            continue
        k_norm = graph.node(k_predecessors[0])
        q_predecessors = graph.predecessors(k_norm.node_id)
        if len(q_predecessors) != 1:
            continue
        q_norm = graph.node(q_predecessors[0])
        group = (q_norm, k_norm, rotary)
        if any(
                node.kind is not WorkloadKind.NORM
                or node.variant != "rmsnorm"
                or node.dtype != "bf16"
                for node in group[:2]):
            continue
        if any(
                node.pool_name != rotary.pool_name
                or node.rank != rotary.rank
                for node in group[:2]):
            continue
        if (tuple(successors[q_norm.node_id]) != (k_norm.node_id,)
                or tuple(successors[k_norm.node_id]) != (rotary.node_id,)):
            continue
        materialization = "grouped_strided_to_contiguous"
        if any(
                "input_materialization" not in node.attributes
                or node.attributes["input_materialization"] != materialization
                for node in group[:2]):
            continue
        if (set(rotary.shape) != {
                "elements", "seq_len", "num_query_heads", "num_kv_heads",
                "head_dim"}
                or set(q_norm.shape) != {"rows", "hidden"}
                or set(k_norm.shape) != {"rows", "hidden"}):
            continue
        tokens = rotary.shape["seq_len"]
        query_heads = rotary.shape["num_query_heads"]
        kv_heads = rotary.shape["num_kv_heads"]
        head_dim = rotary.shape["head_dim"]
        if head_dim != 128:
            # QKNormRopeFusionPass registers only the source program whose
            # compile-time HEAD_DIM is 128. This is a versioned compiler
            # eligibility rule, not an empirical performance threshold.
            continue
        if (q_norm.shape != {
                "rows": tokens * query_heads, "hidden": head_dim}
                or k_norm.shape != {
                    "rows": tokens * kv_heads, "hidden": head_dim}
                or rotary.shape["elements"] != (
                    tokens * (query_heads + kv_heads) * head_dim)):
            continue
        stride = (query_heads + 2 * kv_heads) * head_dim
        expected_materialization = (
            (tokens, query_heads, stride, head_dim),
            (tokens, kv_heads, stride, head_dim),
        )
        actual_materialization = tuple(
            (
                node.attributes["input_outer_groups"],
                node.attributes["input_rows_per_group"],
                node.attributes["input_group_stride_elements"],
                node.attributes["input_row_stride_elements"],
            )
            for node in group[:2]
        )
        if actual_materialization != expected_materialization:
            continue
        graph_domains = tuple(_graph_token_domain(node) for node in group)
        if any(domain is not None for domain in graph_domains):
            if graph_domains[0] is None or any(
                    domain != graph_domains[0] for domain in graph_domains):
                raise ValueError(
                    "Ascend Q/K Norm RoPE fusion requires one shared graph "
                    "token-domain contract")
            _, model_input_tokens, _ = graph_domains[0]
            if tokens != model_input_tokens:
                raise ValueError(
                    "Ascend Q/K Norm RoPE fusion physical sequence length "
                    "disagrees with its graph model-input tokens")
        if any(node.node_id in member_to_head for node in group):
            raise ValueError("overlapping Ascend Q/K Norm RoPE fusion groups")
        head_to_group[q_norm.node_id] = group
        for member in group:
            member_to_head[member.node_id] = q_norm.node_id
    return head_to_group, member_to_head


def _ascend_qkv_rmsnorm_rope_spec(
        q_norm: WorkloadNode,
        k_norm: WorkloadNode,
        rotary: WorkloadNode,
        hardware_id: str,
        hardware,
) -> KernelSpec:
    physical_shape = {
        "seq_len": rotary.shape["seq_len"],
        "num_query_heads": rotary.shape["num_query_heads"],
        "num_kv_heads": rotary.shape["num_kv_heads"],
        "head_dim": rotary.shape["head_dim"],
        "rotary_dim": rotary.shape["head_dim"],
    }
    total_heads = (
        physical_shape["num_query_heads"]
        + 2 * physical_shape["num_kv_heads"])
    physical_shape["element_count"] = (
        physical_shape["seq_len"]
        * total_heads
        * physical_shape["head_dim"])
    domain = _graph_token_domain(rotary)
    if domain is not None:
        if (_graph_token_domain(q_norm) != domain
                or _graph_token_domain(k_norm) != domain):
            raise ValueError(
                "Ascend Q/K Norm RoPE fusion graph token domains disagree")
        logical_tokens, model_input_tokens, _ = domain
        if physical_shape["seq_len"] != model_input_tokens:
            raise ValueError(
                "Ascend Q/K Norm RoPE fusion physical sequence length "
                "disagrees with graph model-input tokens")
        shape = dict(physical_shape)
        shape["seq_len"] = logical_tokens
        shape["element_count"] = (
            logical_tokens * total_heads * shape["head_dim"])
    else:
        shape = dict(physical_shape)
    attributes = {"bias": False, "is_neox": True}
    selection_attributes = {
        **attributes,
        "launch_policy": rotary.attributes["launch_policy"],
    }
    launch = select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        physical_shape,
        rotary.dtype,
        "qkv_rmsnorm_rope",
        selection_attributes,
        hardware=hardware,
    )
    return KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=rotary.dtype,
        variant=launch.execution_variant,
        launch=launch,
        attributes={
            **attributes,
            **_kernel_graph_token_attributes(domain, physical_shape),
        },
    )


def _ascend_qkv_rmsnorm_rope_specs(
        q_norm: WorkloadNode,
        k_norm: WorkloadNode,
        rotary: WorkloadNode,
        hardware_id: str,
        hardware,
) -> tuple[KernelSpec, ...]:
    """Full spec tuple for the Ascend QKV-RMSNorm-RoPE fusion.

    The fused launch reads contiguous q/k rows, so the same-rank q/k Norm
    input materializations (``grouped_strided_to_contiguous``, one copy
    each, already validated by the fusion matcher) precede the fused
    kernel, followed by the V materializations whose count the workload
    builder resolved from the versioned emission observation.  Every helper
    copy is a plain VECTOR elementwise copy priced by the existing copy
    family; no serving duration or residual correction participates.
    """

    fused = _ascend_qkv_rmsnorm_rope_spec(
        q_norm, k_norm, rotary, hardware_id, hardware)

    def copy_spec(node: WorkloadNode, elements: int) -> KernelSpec:
        shape = {"element_count": elements}
        selection_attributes = {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": node.attributes["launch_policy"],
        }
        launch = select_launch_config(
            hardware_id,
            KernelKind.VECTOR,
            shape,
            node.dtype,
            "elementwise",
            selection_attributes,
            hardware=hardware,
        )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=shape,
            dtype=node.dtype,
            variant=launch.execution_variant,
            launch=launch,
            attributes={
                "operation": "copy",
                "reads": 1,
                "writes": 1,
                **_derived_graph_token_attributes(fused, shape),
            },
        )

    copies = [
        copy_spec(
            q_norm, q_norm.shape["rows"] * q_norm.shape["hidden"]),
        copy_spec(
            k_norm, k_norm.shape["rows"] * k_norm.shape["hidden"]),
    ]
    if "qkv_v_materialization_count" not in rotary.attributes:
        raise ValueError(
            "Ascend QKV-RMSNorm-RoPE fusion requires the "
            "qkv_v_materialization_count emission contract on the rotary "
            "node")
    v_count = rotary.attributes["qkv_v_materialization_count"]
    if (isinstance(v_count, bool) or not isinstance(v_count, int)
            or v_count < 0):
        raise ValueError(
            "qkv_v_materialization_count must be a non-negative integer; "
            f"found {v_count!r}")
    v_elements = (
        rotary.shape["seq_len"]
        * rotary.shape["num_kv_heads"]
        * rotary.shape["head_dim"])
    for _ in range(v_count):
        copies.append(copy_spec(rotary, v_elements))
    return tuple(copies) + (fused,)


def _block_fp8_input_quant_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        hardware_id: str, hardware) -> tuple[KernelSpec, ...]:
    """The separate activation quantiser launched before a block-FP8 GEMM."""

    element_count = (
        selection_shape["b"] * selection_shape["m"] * selection_shape["k"])
    quant_launch = select_block_fp8_input_quant_launch(
        hardware_id, main.launch, element_count, hardware)
    if quant_launch is None:
        return ()
    shape = {"element_count": element_count}
    return (KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=main.dtype,
        variant=quant_launch.execution_variant,
        launch=quant_launch,
        attributes={
            "group_size": PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
            **_derived_graph_token_attributes(main, shape),
        },
    ),)


def _gemm_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        selection_variant: str,
        selection_attributes: Mapping[str, object],
        hardware_id: str, hardware):
    if (main.launch is None and (("mathematical_program" in main.attributes
            and main.attributes["mathematical_program"] in (
                "unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1"))
            or (main.dtype == "bf16" and main.attributes["output_dtype"] == "fp32"))):
        # Useful source matmul has neither a physical quantizer nor a
        # selected split-K launch from which to derive reducer work.
        return (main,)
    return (
        _block_fp8_input_quant_specs(main, selection_shape, hardware_id, hardware)
        + _gemm_main_and_reducer_specs(
            main, selection_shape, selection_variant, selection_attributes,
            hardware_id, hardware))


def _gemm_main_and_reducer_specs(
        main: KernelSpec, selection_shape: Mapping[str, int],
        selection_variant: str,
        selection_attributes: Mapping[str, object],
        hardware_id: str, hardware):
    if main.launch.algorithm == "h200_cublaslt129_bf16_external_aggregate_gemm":
        reducer_launch = select_gemm_reducer_launch(hardware_id, selection_shape,
            main.dtype, selection_variant, selection_attributes, hardware=hardware)
        if reducer_launch is None:
            raise ValueError("legacy BF16 aggregate requires its observed separate reducer")
        reducer = KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
            shape={"element_count": selection_shape["b"] * selection_shape["m"] * selection_shape["n"]},
            dtype=main.dtype, variant="gemm_reduce", launch=reducer_launch,
            attributes={"split_k": int(main.launch.parameters["modeled_split_k"]),
                "source_gemm_shape": {name: selection_shape[name] for name in ("b", "m", "n", "k")}})
        return main, reducer
    if main.launch.algorithm == "h200_vllm030_torch213_fp16_external_aggregate_gemm":
        reducer_launch = select_gemm_reducer_launch(hardware_id, selection_shape,
            main.dtype, selection_variant, selection_attributes, hardware=hardware)
        if reducer_launch is None:
            raise ValueError("external aggregate main requires its separate source reducer launch")
        reducer = KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
            shape={"element_count": selection_shape["b"] * selection_shape["m"] * selection_shape["n"]},
            dtype=main.dtype, variant="gemm_reduce", launch=reducer_launch,
            attributes={"split_k": int(main.launch.parameters["modeled_split_k"]),
                "source_gemm_runtime": main.attributes["source_gemm_runtime"],
                "source_gemm_shape": {name: selection_shape[name] for name in ("b", "m", "n", "k")}})
        return main, reducer
    tiling = main.launch.tiling
    if tiling is None:
        if (main.hardware_id == hardware_id
                and HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id)
                and main.launch.algorithm in ("h200_cublaslt129_injected_gemm",
                    "h200_vllm030_torch213_fp16_observed_gemm")
                and main.variant == "gemm" and main.dtype in ("bf16", "fp16") and main.shape["b"] == 1
                and main.attributes["output_dtype"] in (None, main.dtype)):
            # The observed ordinary call already contains its whole result.
            # A partial-output main still needs unknown split/reducer facts;
            # do not turn that missing consumer into an omitted zero service.
            if main.launch.parameters["split_k_partial_bytes_per_element"] != 0.0:
                raise NotImplementedError("grid-only GEMM external reduction requires actual split/reducer facts")
            return (main,)
        if main.launch.execution_variant == "gemv":
            return (main,)
        raise ValueError("production GEMM launch must declare tiling")
    if tiling.split_k is None:
        raise ValueError("production GEMM launch must declare split_k")
    if tiling.split_k == 1:
        return (main,)
    reducer_launch = select_gemm_reducer_launch(
        hardware_id, selection_shape, main.dtype, selection_variant,
        selection_attributes, hardware=hardware)
    if reducer_launch is None:
        return (main,)
    reducer = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape={
            "element_count": (
                selection_shape["b"]
                * selection_shape["m"]
                * selection_shape["n"]),
        },
        dtype=main.dtype,
        variant=reducer_launch.execution_variant,
        launch=reducer_launch,
        attributes={
            "split_k": tiling.split_k,
            **_derived_graph_token_attributes(
                main,
                {"element_count": (
                    selection_shape["b"]
                    * selection_shape["m"]
                    * selection_shape["n"]),
                },
            ),
        },
    )
    return main, reducer


def _gemm_node_specs(node: WorkloadNode, hardware_id: str, hardware):
    return _gemm_specs(
        _compute_spec(node, hardware_id, hardware),
        node.shape,
        node.variant,
        node.attributes,
        hardware_id, hardware)


def _fused_moe_prediction_attributes(
    node: WorkloadNode,
    routing_relabelling=None,
) -> dict[str, object]:
    _validate_routing_provenance(node)
    shape = node.shape
    experts = shape["experts"]
    counts = tuple(node.attributes["tokens_per_expert"])
    if len(counts) != experts:
        raise ValueError("FUSED_MOE routing count must match expert count")
    expert_parallel_size = node.attributes["expert_parallel_size"]
    if expert_parallel_size == 1:
        local_counts = counts
        expert_rank = 0
        owned_experts = tuple(range(experts))
    elif expert_parallel_size > 1:
        local_counts = tuple(node.attributes["local_tokens_per_expert"])
        expert_rank = node.attributes["expert_rank"]
        owned_experts = tuple(node.attributes["owned_experts"])
    else:
        raise ValueError("expert_parallel_size must be positive")
    if routing_relabelling is not None:
        counts, local_counts = routing_relabelling
    attributes = {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": local_counts,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank": expert_rank,
        "owned_experts": owned_experts,
        "expert_weight_tile_rows": None,
    }
    target_names = {
        "routed_scaling_factor", "routed_weight_dtype", "weight_group_size",
    }
    present_target_names = target_names & set(node.attributes)
    if present_target_names:
        if present_target_names != target_names:
            raise ValueError(
                "target FUSED_MOE routing semantics must be complete: "
                f"present={sorted(present_target_names)}")
        for name in target_names:
            attributes[name] = node.attributes[name]
    if "fused_shared_experts" in node.attributes:
        attributes["fused_shared_experts"] = (
            node.attributes["fused_shared_experts"])
    domain = _graph_token_domain(node)
    attributes.update(_kernel_graph_token_attributes(domain, node.shape))
    return attributes


def _fused_moe_specs(node: WorkloadNode, hardware_id: str, hardware,
                     routing_relabelling=None):
    if node.variant != "full":
        raise ValueError("logical FUSED_MOE workload variant must be 'full'")
    selection_shape = node.shape
    domain = _graph_token_domain(node)
    if domain is None:
        shape = dict(selection_shape)
    else:
        logical_tokens, model_input_tokens, _ = domain
        if selection_shape["tokens"] != model_input_tokens:
            raise ValueError(
                "FUSED_MOE physical tokens disagree with graph model-input "
                "tokens")
        shape = dict(selection_shape)
        shape["tokens"] = logical_tokens
    prediction_attributes = _fused_moe_prediction_attributes(
        node, routing_relabelling)
    selection_attributes = dict(node.attributes)
    selection_attributes.pop("routing_provenance")
    if routing_relabelling is not None:
        global_counts, local_counts = routing_relabelling
        selection_attributes["tokens_per_expert"] = global_counts
        selection_attributes["local_tokens_per_expert"] = local_counts
    specs = build_fused_moe_physical_specs(
        hardware_id=hardware_id,
        shape=shape,
        selection_shape=selection_shape,
        dtype=node.dtype,
        prediction_attributes=prediction_attributes,
        selection_attributes=selection_attributes,
        hardware=hardware,
    )
    return _with_fused_moe_input_quant_specs(specs, hardware_id, hardware)


def _with_fused_moe_input_quant_specs(
        specs: tuple[KernelSpec, ...], hardware_id: str, hardware,
        ) -> tuple[KernelSpec, ...]:
    """Insert the activation quantiser launched before each fused-MoE GEMM."""

    result = []
    for spec in specs:
        if spec.kind is KernelKind.FUSED_MOE and spec.variant in ("w1", "w2"):
            tokens = spec.shape["tokens"]
            if spec.variant == "w1":
                element_count = tokens * spec.shape["hidden"]
            else:
                # A fused shared expert is one more routed column per token.
                kernel_topk = spec.shape["topk"] + (
                    spec.attributes["fused_shared_experts"]
                    if "fused_shared_experts" in spec.attributes else 0)
                element_count = (
                    tokens * kernel_topk * spec.shape["intermediate"])
            quant_launch = select_fused_moe_input_quant_launch(
                hardware_id, spec.launch, element_count, hardware)
            if quant_launch is not None:
                shape = {"element_count": element_count}
                result.append(KernelSpec(
                    kind=KernelKind.VECTOR,
                    hardware_id=hardware_id,
                    shape=shape,
                    dtype=spec.dtype,
                    variant=quant_launch.execution_variant,
                    launch=quant_launch,
                    attributes={
                        "group_size": PER_TOKEN_GROUP_QUANT_GROUP_SIZE,
                        **_derived_graph_token_attributes(spec, shape),
                    },
                ))
        result.append(spec)
    return tuple(result)


def _target_fused_moe_specs(
    node: WorkloadNode,
    hardware_id: str,
    hardware,
    *,
    model_contract: str,
) -> tuple[KernelSpec, ...]:
    routed_weight_dtype = node.attributes["routed_weight_dtype"]
    if (HWSpec.source_contract_supported("unbound_grouped_bf16_expert_source_math", hardware_id) and routed_weight_dtype == "bf16"
            and node.attributes["launch_policy"] == "analytical_launch"
            and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
            and node.attributes["routing_source_stage"] == "dsv32_moe_route"
            and node.attributes["checkpoint_weight_dtype"] == "bf16"
            and node.attributes["weight_group_size"] == 0
            and node.attributes["output_is_reduced"] is False):
        attributes = _fused_moe_prediction_attributes(node)
        attributes["mathematical_program"] = "unquantized_bf16_grouped_swiglu_v1"
        return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id, shape=node.shape,
            dtype="bf16", variant="routed_experts", attributes=attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
    if routed_weight_dtype in ("bf16", "fp8"):
        specs = _fused_moe_specs(node, hardware_id, hardware)
        if "routing_source_stage" in node.attributes:
            if node.attributes["routing_source_stage"] not in ("dsv32_moe_route", "kimi_moe_route"):
                raise ValueError("unknown explicit target routing source stage")
            if sum(spec.variant == "moe_topk" for spec in specs) != 1:
                raise ValueError("explicit source routing requires exactly one legacy top-k replacement")
            return tuple(spec for spec in specs if spec.variant != "moe_topk")
        return specs
    if routed_weight_dtype != "int4" and routed_weight_dtype != "mxfp4":
        raise ValueError(
            f"unsupported target routed weight dtype: {routed_weight_dtype!r}")
    if (HWSpec.source_contract_supported("kimi_grouped_marlin_source_math", hardware_id) and routed_weight_dtype == "int4"
            and node.attributes["launch_policy"] == "analytical_launch"
            and node.attributes["source_component"] == "symbolic_grouped_routed_experts"
            and node.attributes["routing_source_stage"] == "kimi_moe_route"):
        # The historical program name also covers EP-off's full expert set.
        # Preserve the producer's actual I shard and EP1 histogram; Core and
        # the mathematical helper jointly validate its supported source domain.
        attributes = _fused_moe_prediction_attributes(node)
        attributes["mathematical_program"] = "kimi_ep_marlin_swiglu_v1"
        return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id, shape=node.shape,
            dtype="bf16", variant="routed_experts", attributes=attributes, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
    if (model_contract == "deepseek_v4_vllm_0_30_0_logical_v1"
            and node.attributes["expert_parallel_size"] == 1):
        # EP-off TP shards apply the same expert assignment, each with its
        # own I/TP columns. Do not silently replace a contradictory local
        # histogram/ownership declaration with the global one below.
        local = tuple(node.attributes["local_tokens_per_expert"])
        owned = tuple(node.attributes["owned_experts"])
        if (any(type(value) is not int for value in (*local, *owned))
                or type(node.attributes["expert_rank"]) is not int):
            raise TypeError("V4 EP-off expert assignment requires integer counts, expert IDs and rank")
        if (local != tuple(node.attributes["tokens_per_expert"])
                or owned != tuple(range(node.shape["experts"]))
                or node.attributes["expert_rank"] != 0):
            raise ValueError("V4 EP-off expert assignment requires identical global/local counts and full expert ownership")
    attributes = _fused_moe_prediction_attributes(node)
    if "swiglu_limit" in node.attributes:
        attributes["swiglu_limit"] = node.attributes["swiglu_limit"]
    return (_selected_target_compute_spec(
        node, hardware_id, hardware, KernelKind.FUSED_MOE,
        "routed_experts", node.shape, attributes),)


def _transfer_spec(node: WorkloadNode, hardware_id: str, *,
                   destination_hardware=None) -> KernelSpec:
    attributes = dict(node.attributes)
    if "bytes" not in attributes:
        raise ValueError(f"{node.kind.value} requires explicit bytes")
    if node.kind in _COMMUNICATION_KINDS:
        _graph_token_domain(node)
        kind = KernelKind.COMMUNICATION
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"],
            bytes=attributes["bytes"],
            direction=attributes["direction"],
            link=attributes["link"],
            topology=attributes["topology"],
            participants=tuple(attributes["participants"]),
            operation=attributes["operation"],
            bytes_semantics=attributes["bytes_semantics"],
            source_rank=attributes["source_rank"],
            destination_rank=attributes["destination_rank"],
            ring_order=tuple(attributes["ring_order"]),
            reduction=attributes["reduction"],
            backend=attributes["backend"],
            resource_id=attributes["resource_id"],
            backend_config_sha256=attributes["backend_config_sha256"],
            partition_bytes=(tuple(attributes["partition_bytes"]) if "partition_bytes" in attributes else ()),
            participant_nic_ids=(tuple(attributes["participant_nic_ids"])
                if "participant_nic_ids" in attributes else ()),
            implementation=(
                attributes["implementation"]
                if "implementation" in attributes else None),
            destination_hardware_id=(destination_hardware.hardware_id
                if destination_hardware is not None else None),
            destination_bandwidth_bytes_s=(min(
                destination_hardware.analytical_link_bandwidth_bytes_s(attributes["link"]),
                destination_hardware.resource_by_role("dram").capacity,
                destination_hardware.interconnect["pcie_bw_bytes_s"])
                if destination_hardware is not None else None),
        )
    elif node.kind is WorkloadKind.MEMCPY:
        kind = KernelKind.MEMCPY
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"], bytes=attributes["bytes"],
            direction=attributes["direction"], link=attributes["link"],
        )
    elif node.kind is WorkloadKind.KV_OFFLOAD:
        kind = KernelKind.KV_OFFLOAD
        source = attributes["source_tier"]
        destination = attributes["destination_tier"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("KV_OFFLOAD source_tier must be a non-empty string")
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError(
                "KV_OFFLOAD destination_tier must be a non-empty string")
        source_is_hbm = source.lower() == "hbm"
        destination_is_hbm = destination.lower() == "hbm"
        if source_is_hbm and not destination_is_hbm:
            non_hbm_tier = destination
        elif destination_is_hbm and not source_is_hbm:
            non_hbm_tier = source
        else:
            raise ValueError(
                "KV_OFFLOAD requires exactly one HBM endpoint, got "
                f"{source!r} -> {destination!r}")
        # STORE has the same D2H byte service as EVICT. The workload event
        # keeps STORE identity so the KV ledger retains the GPU copy.
        transfer = PhysicalTransferConfig(
            algorithm=attributes["algorithm"], bytes=attributes["bytes"],
            direction=("evict" if attributes["direction"] == "store"
                       else attributes["direction"]), link=attributes["link"],
            tier=non_hbm_tier,
        )
    else:
        raise NotImplementedError(f"no transfer lowering for {node.kind.value}")
    return KernelSpec(
        kind=kind, hardware_id=hardware_id, transfer=transfer,
    )


def _dense_tp_producer_output_bytes(node: WorkloadNode) -> int:
    if node.dtype not in ("bf16", "fp16"):
        raise ValueError(
            "validated dense TP collective producer dtype must be bf16 or fp16")
    if (node.kind is WorkloadKind.VECTOR and node.variant == "vllm_masked_embedding"
            and "source_component" in node.attributes
            and node.attributes["source_component"] in ("vllm_0_11_symbolic_masked_embedding", "vllm_0_18_symbolic_masked_embedding", "vllm_0_24_symbolic_masked_embedding", "vllm_0_27_1_symbolic_masked_embedding", "vllm_0_30_symbolic_masked_embedding")):
        if (node.attributes["stage"] != "mask_output" or set(node.shape) != {"elements", "rows", "hidden"}
                or node.shape["elements"] != node.shape["rows"] * node.shape["hidden"]):
            raise ValueError("masked embedding collective producer must expose its complete BF16/FP16 output")
        elements = node.shape["elements"]
    elif node.kind is WorkloadKind.GATHER:
        if node.variant != "embedding":
            raise ValueError(
                "dense TP gather collective producer must use the embedding "
                "variant")
        required_shape = {"elements", "rows", "hidden"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP embedding producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["rows"] * node.shape["hidden"]
        if node.shape["elements"] != elements:
            raise ValueError(
                "dense TP embedding elements disagree with rows and hidden")
    elif node.kind is WorkloadKind.LINEAR:
        if node.variant != "gemm":
            raise ValueError(
                "dense TP linear collective producer must use the GEMM "
                "variant")
        required_shape = {"b", "m", "n", "k"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP GEMM producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["b"] * node.shape["m"] * node.shape["n"]
    elif (node.kind is WorkloadKind.FUSED_MOE and "source_component" in node.attributes
            and node.attributes["source_component"] == "symbolic_unquantized_experts"):
        if (node.variant != "full" or node.attributes["expert_parallel_size"] != 1
                or set(node.shape) != {"tokens", "experts", "topk", "hidden", "intermediate"}):
            raise ValueError("virtual expert collective producer requires the complete EP-off output contract")
        elements = node.shape["tokens"] * node.shape["hidden"]
    elif node.kind is WorkloadKind.LM_HEAD:
        if node.variant not in ("gemm", "gemv"):
            raise ValueError(
                "dense TP LM-head collective producer must use a GEMM/GEMV "
                "variant")
        required_shape = {"b", "m", "n", "k"}
        if set(node.shape) != required_shape:
            raise ValueError(
                "dense TP LM-head producer shape fields disagree with the "
                "collective payload contract")
        elements = node.shape["b"] * node.shape["m"] * node.shape["n"]
    else:
        raise ValueError(
            "dense TP collective predecessor must be an embedding, linear, "
            f"or LM-head node, got {node.kind.value!r}")
    if elements <= 0:
        raise ValueError(
            "dense TP collective producer output must contain elements")
    return elements * 2


def _validate_collective_node(
    node: WorkloadNode,
    pool,
    graph: WorkloadGraph,
    *, gqa_source: bool = False,
) -> None:
    attributes = node.attributes
    operation = attributes["operation"]
    collective = pool.collective_for(operation)
    logical_rank = pool.logical_rank_for_physical_rank(node.rank)
    variable = operation in ("all_gatherv", "reduce_scatterv")
    if variable:
        dimension = attributes["collective_group"]
        if dimension not in ("expert", "data", "data_context", "context", "tensor") or (dimension != "expert"
                and pool.parallel_plan.config.enable_expert_parallel and not (
                    _symbolic_kimi_dp_projection(node, pool) and (
                        dimension == "data" and pool.parallel_plan.config.tensor_parallel_size == 1
                        or dimension == "tensor" and operation == "reduce_scatterv"
                        and pool.parallel_plan.config.pipeline_parallel_size == 1)
                    or _symbolic_glm_pcp_projection(node, pool) and (
                        dimension == "data_context" and pool.parallel_plan.config.data_parallel_size == 1
                        or dimension == "tensor" and operation == "reduce_scatterv"
                        and pool.parallel_plan.config.pipeline_parallel_size == 1
                        and pool.parallel_plan.config.data_parallel_size > 1
                        and pool.parallel_plan.config.tensor_parallel_size > 1
                        and pool.parallel_plan.config.sequence_parallel_size in (1, pool.parallel_plan.config.tensor_parallel_size)))):
            raise ValueError("variable MoE collective requires the declared expert or EP-off data group")
        if dimension in ("context", "tensor") and not (_symbolic_glm_pcp_projection(node, pool)
                or dimension == "tensor" and operation == "reduce_scatterv" and _symbolic_kimi_dp_projection(node, pool)
                and pool.parallel_plan.config.pipeline_parallel_size == 1):
            raise ValueError("variable TP/PCP collective requires its explicit GLM PCP source scope")
    else:
        dimension = attributes["collective_group"] if "collective_group" in attributes else "tensor"
        if (dimension not in ("tensor", "context") or (dimension == "context" and (
                operation != "all_gather" or pool.parallel_plan.config.context_parallel_size < 2))):
            raise ValueError("ordinary collective group must be TP or a configured PCP all-gather")
    participants = pool.physical_rank_group(dimension, logical_rank)
    if variable:
        expected_reduction = "sum" if operation == "reduce_scatterv" else None
        expected_ring_order = ()
    elif operation == "all_reduce":
        expected_reduction = "sum"
        expected_ring_order = collective_ring_order(
            collective.algorithm, participants)
    elif operation == "all_gather":
        expected_reduction = None
        expected_ring_order = collective_ring_order(
            collective.algorithm, participants)
    elif operation == "all_to_all":
        expected_reduction = None
        expected_ring_order = ()
    else:
        raise ValueError(
            f"unknown collective operation: {operation!r}")
    if (is_heterogeneous_moe(node) and collective.link == "roce"
            and collective.backend == "analytical_roofline_v1"
            and collective.algorithm == "ring" and (pool.name == "prefill_gpu"
                or pool.name == "decode_gpu" and "heterogeneous_forward_scope" in node.attributes
                and node.attributes["heterogeneous_forward_scope"] == "gpu_decode_only")):
        # This scoped scenario configures a switch-only logical ring;
        # TP shard order stays unchanged. It is not a discovered library
        # ring or permission to change any other collective contract.
        expected_ring_order = participants[::2] + participants[1::2]
    if node.rank != participants[0]:
        raise ValueError(
            "collective workload leader must be logical tensor rank zero")
    if tuple(attributes["participants"]) != participants:
        raise ValueError(
            "collective workload participants disagree with the physical "
            "tensor rank group")
    expected = {
        "algorithm": collective.algorithm,
        "backend": collective.backend,
        "backend_config_sha256": collective.backend_config_sha256,
        "bytes_semantics": "aggregate_input" if operation == "all_gatherv" else "per_rank_input",
        "direction": "collective",
        "link": collective.link,
        "operation": collective.operation,
        "participants": participants,
        "source_rank": None,
        "destination_rank": None,
        "ring_order": expected_ring_order,
        "reduction": expected_reduction,
        "topology": collective.topology,
        "resource_id": collective.resource_id,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "collective workload disagrees with deployment config: "
            f"{mismatched!r}")
    if variable:
        rows, shape = attributes["symbolic_variable_rows"], attributes["symbolic_variable_row_shape"]
        if (not isinstance(rows, tuple) or len(rows) != len(participants)
                or any(type(value) is not int or value < 0 for value in rows) or not sum(rows)
                or not isinstance(shape, tuple) or not shape
                or any(type(value) is not int or value <= 0 for value in shape) or node.dtype not in DTYPE_BYTES):
            raise ValueError("variable collective requires complete row geometry")
        partitions = tuple(value * prod(shape) * DTYPE_BYTES[node.dtype] for value in rows)
        if partitions != attributes["partition_bytes"] or sum(partitions) != attributes["bytes"]:
            raise ValueError("variable collective byte extents disagree with participant rows")
        expected_inputs = len(tuple(value for value in rows if value)) if operation == "all_gatherv" else len(rows)
        expected_outputs = len(rows) if operation == "all_gatherv" else len(tuple(value for value in rows if value))
        if (len(attributes["symbolic_tensor_inputs"]) != expected_inputs
                or len(attributes["symbolic_tensor_outputs"]) != expected_outputs):
            raise ValueError("variable collective requires exactly its nonempty participant payload ports")
        if _symbolic_glm_pcp_projection(node, pool):
            producers = attributes["symbolic_collective_input_producers"]
            if len(producers) != expected_inputs or any(not isinstance(name, str) or name == node.node_id for name in producers):
                raise ValueError("GLM PCP collective requires each actual payload producer identity")
            incoming = {edge.source for edge in graph.edges if edge.destination == node.node_id and edge.kind == "data"}
            if not set(producers) <= incoming:
                raise ValueError("GLM PCP collective requires each actual payload producer completion dependency")
        return
    if _symbolic_glm_pcp_projection(node, pool):
        shape = attributes["symbolic_collective_input_shape"]
        if (not isinstance(shape, tuple) or not shape or any(type(size) is not int or size <= 0 for size in shape)
                or node.dtype not in DTYPE_BYTES or prod(shape) * DTYPE_BYTES[node.dtype] != attributes["bytes"]):
            raise ValueError("GLM source collective bytes must match its actual public payload shape")
        producers = attributes["symbolic_collective_input_producers"]
        incoming = {edge.source for edge in graph.edges if edge.destination == node.node_id and edge.kind == "data"}
        if "symbolic_collective_external_inputs" in attributes:
            from src.kernel.memory_access import TensorView, region_end
            origins = attributes["symbolic_collective_external_inputs"]
            first = next(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage)
                if owner == (pool.logical_rank_for_physical_rank(participants[0]) // pool.parallel_plan.config.tensor_parallel_size
                    % pool.parallel_plan.config.pipeline_parallel_size))
            if (operation != "all_reduce" or dimension != "tensor" or attributes["symbolic_collective_storage"] != "fresh"
                    or first == 0 or not node.node_id.endswith(f":layer-{first}:input-all-reduce")
                    or len(origins) != len(participants) or len(producers) != len(participants)
                    or len(attributes["symbolic_tensor_inputs"]) != len(participants)):
                raise ValueError("GLM external PP input requires its complete stage-entry TP all-reduce")
            for lane, (producer, origin) in enumerate(zip(producers, origins, strict=True)):
                if producer is not None:
                    if origin is not None or not isinstance(producer,str) or producer not in incoming:
                        raise ValueError("GLM collective input needs exactly its producer or its external origin")
                    continue
                if (not isinstance(origin,Mapping) or set(origin)!={"source","rank","role","values_bound","view","allocation_bytes"}
                        or origin["source"] != "vllm_0_30_legacy_dummy_PP_retained" or type(origin["rank"]) is not int
                        or origin["rank"] != participants[lane] or origin["role"] != "hidden_states" or origin["values_bound"] is not False
                        or type(origin["allocation_bytes"]) is not int or origin["allocation_bytes"] <= 0):
                    raise ValueError("GLM external PP input must declare its actual rank-local unknown hidden buffer")
                view = TensorView(**dict(origin["view"]))
                consumers = tuple(graph.node(edge.destination) for edge in graph.edges if edge.source==node.node_id and edge.kind=="data"
                    and graph.node(edge.destination).rank==participants[lane]
                    and graph.node(edge.destination).node_id.endswith(f":layer-{first}:input-norm"))
                if (view.view_id != attributes["symbolic_tensor_inputs"][lane] or view.shape != shape or view.dtype != "bf16"
                        or view.offset_bytes != 0 or len(shape)!=2 or view.strides_bytes != (shape[1]*2,2)
                        or not view.allocation_id.endswith(f":rank-{participants[lane]}:worker.intermediate:hidden_states")
                        or region_end(view.region()) > origin["allocation_bytes"] or len(consumers)!=1
                        or "source_dummy_run" not in consumers[0].attributes or consumers[0].attributes["source_dummy_run"] is not True
                        or consumers[0].attributes["source_actual_request_count"] != 0):
                    raise ValueError("GLM external PP input must match its complete active view and actual dummy norm consumer")
            return
        if (len(producers) != len(participants) or any(not isinstance(name, str) or name == node.node_id for name in producers)
                or not set(producers) <= incoming):
            raise ValueError("GLM source collective requires each actual payload producer completion dependency")
        return
    parallelism = pool.parallel_plan.config
    dense_tp_only = (
        (parallelism.tensor_parallel_size == 2 or
            (gqa_source and parallelism.tensor_parallel_size > 1 and "symbolic_collective_storage" in attributes))
        and parallelism.expert_parallel_size == 1
        and (parallelism.pipeline_parallel_size in (1, 2) or "symbolic_collective_storage" in attributes)
        and parallelism.sequence_parallel_size == 1
        and parallelism.context_parallel_size == 1
        and parallelism.data_parallel_size == 1
    )
    if not dense_tp_only:
        return
    incoming_edges = tuple(
        edge for edge in graph.edges if edge.destination == node.node_id)
    if (parallelism.pipeline_parallel_size > 1 and operation == "all_gather"
            and any(graph.node(edge.source).kind is WorkloadKind.P2P for edge in incoming_edges)):
        # The complete sliced boundary, ports, backing geometry and ordering
        # are checked by _validate_symbolic_sliced_pipeline_graph (or the legacy
        # TP2/PP2 validator). Its producers are ranked at the sending stage.
        return
    if "symbolic_collective_storage" in attributes:
        if attributes["symbolic_collective_storage"] not in ("fresh", "inplace"):
            raise ValueError("symbolic collective storage contract must be fresh or inplace")
        ordering = tuple(edge for edge in incoming_edges if edge.kind == "barrier")
        if any(graph.node(edge.source).kind is not WorkloadKind.COLLECTIVE for edge in ordering):
            raise ValueError("symbolic collective ordering barriers must come from collectives")
        if any(edge.kind not in ("data", "barrier") for edge in incoming_edges):
            raise ValueError("symbolic collective requires payload data and group-order barriers")
        incoming_edges = tuple(edge for edge in incoming_edges if edge.kind == "data")
        if len(incoming_edges) != len(participants):
            raise ValueError("symbolic collective must have one payload producer per participant")
        inputs = tuple(attributes["symbolic_tensor_inputs"])
        if len(inputs) != len(participants) or len(set(inputs)) != len(inputs):
            raise ValueError("symbolic collective requires distinct per-rank input views")
        producers = tuple(graph.node(edge.source) for edge in incoming_edges)
        bound_inputs = set()
        for producer in producers:
            if ("symbolic_tensor_outputs" not in producer.attributes
                    or len(set(producer.attributes["symbolic_tensor_outputs"]) & set(inputs)) != 1):
                raise ValueError("symbolic collective payload edge must bind its producer output view")
            bound_inputs.update(set(producer.attributes["symbolic_tensor_outputs"]) & set(inputs))
        if bound_inputs != set(inputs):
            raise ValueError("symbolic collective payload producers must cover every input view")
    elif any(edge.kind != "barrier" for edge in incoming_edges):
        raise ValueError("dense TP collective producer edges must be barriers")
    if len(incoming_edges) != len(participants):
        raise ValueError(
            "dense TP collective must have one producer per participant")
    predecessor_ids = tuple(edge.source for edge in incoming_edges)
    producers = tuple(graph.node(node_id) for node_id in predecessor_ids)
    if (any(producer.pool_name != pool.name for producer in producers)
            or {producer.rank for producer in producers}
            != set(participants)):
        raise ValueError(
            "dense TP collective producers disagree with its pool/rank group")
    producer_kinds = {producer.kind for producer in producers}
    if len(producer_kinds) != 1:
        raise ValueError(
            "dense TP collective producers must have one consistent kind")
    producer_contracts = {
        (
            tuple(sorted(producer.shape.items())),
            producer.dtype,
            producer.variant,
        )
        for producer in producers
    }
    if len(producer_contracts) != 1:
        raise ValueError(
            "dense TP collective producers must have the same shape, dtype, "
            "and variant")
    kimi_sum = all((_symbolic_kimi_parallel_projection(producer, pool) or _symbolic_v3_parallel_projection(producer, pool))
            and producer.kind is WorkloadKind.ELEMENTWISE and producer.variant == "add"
            and producer.dtype == "bf16" and producer.node_id.endswith(":mlp:shared-add")
            and set(producer.shape) == {"elements"} for producer in producers)
    v4_sum = all(_symbolic_v4_parallel_projection(producer, pool)
        and producer.kind is WorkloadKind.ELEMENTWISE and producer.variant == "add" and producer.dtype == "bf16"
        and producer.node_id.endswith(":ffn:shared-add") and set(producer.shape) == {"elements"} for producer in producers)
    producer_bytes = tuple(producer.shape["elements"] * 2 if kimi_sum or v4_sum else
        _dense_tp_producer_output_bytes(producer) for producer in producers)
    if len(set(producer_bytes)) != 1:
        raise ValueError(
            "dense TP collective producers have inconsistent output bytes")
    expected_bytes = producer_bytes[0]
    if attributes["bytes"] != expected_bytes:
        raise ValueError(
            "dense TP collective payload bytes disagree with producer shape "
            f"and dtype: expected={expected_bytes}, "
            f"actual={attributes['bytes']}")
    producer_kind = producers[0].kind
    if kimi_sum or v4_sum or producer_kind in (WorkloadKind.GATHER, WorkloadKind.LINEAR, WorkloadKind.FUSED_MOE, WorkloadKind.VECTOR):
        expected_operation = "all_reduce"
    elif producer_kind is WorkloadKind.LM_HEAD:
        expected_operation = "all_gather"
    else:
        raise RuntimeError(
            "validated dense TP producer kind became unreachable: "
            f"{producer_kind.value!r}")
    if operation != expected_operation:
        raise ValueError(
            "dense TP collective operation disagrees with producer kind: "
            f"expected={expected_operation!r}, actual={operation!r}")


def _validate_pipeline_p2p_node(node: WorkloadNode, pool) -> None:
    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "source_stage",
        "destination_stage",
        "payload_tensor",
    }
    graph_domain_fields = set(_GRAPH_TOKEN_DOMAIN_FIELDS)
    present_graph_domain_fields = graph_domain_fields & set(attributes)
    if present_graph_domain_fields:
        if present_graph_domain_fields != graph_domain_fields:
            raise ValueError(
                "pipeline P2P graph token domains must be complete")
        required |= graph_domain_fields
    symbolic_fields = {"symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_output_allocations"}
    if symbolic_fields & set(attributes):
        tp = pool.parallel_plan.config.tensor_parallel_size
        dense_pipeline = ((pool.parallel_plan.config.is_rank_local or _qwen030_dp_sp(pool.parallel_plan.config))
            and (HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", pool.hardware_id)
                or HWSpec.source_contract_supported("glm_tp8_pp2_ep1_source_projection", pool.hardware_id)))
        if (tp not in (1, 2) and not dense_pipeline and not _symbolic_mla_pipeline_projection(node, pool)
                and not _symbolic_v4_pipeline_topology_projection(node, pool) and not _symbolic_mla_dp_pipeline_projection(node, pool)
                and not _symbolic_k3_pipeline_topology_projection(node, pool)):
            raise NotImplementedError("symbolic pipeline ports require TP1, TP2/PP2 or the declared GLM TP8/PP2 projection")
        required |= symbolic_fields
        if tp > 1 or "symbolic_pipeline_views" in attributes:
            required.add("symbolic_pipeline_views")
        if _symbolic_mla_pipeline_projection(node, pool):
            required |= {"source_projection_only", "model_execution_version"}
        if _symbolic_kimi_pipeline_projection(node, pool) and "model_source_profile" in attributes:
            required.add("model_source_profile")
        if _symbolic_v3_parallel_projection(node, pool) or _symbolic_mla_dp_pipeline_projection(node, pool):
            required |= {"source_projection_only", "model_execution_version", "model_source_profile"}
        if _symbolic_v4_pipeline_topology_projection(node, pool):
            required |= {"source_projection_only", "model_execution_version", "model_source_profile"}
        if _symbolic_k3_pipeline_topology_projection(node, pool):
            required |= {"source_projection_only", "model_execution_version", "model_source_profile"}
        if _symbolic_glm_pcp_projection(node, pool):
            required |= {"glm_parallel_source", "source_projection_only", "model_execution_version",
                "model_source_profile", "source_pipeline_stage", "pipeline_layer_partition"}
            if _symbolic_glm_fp8_target_dp_projection(node, pool):
                required.add("source_glm_fp8_target_only")
            if _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, pool):
                required.add("source_glm_rank_local_fp8_mtp_ep")
            required |= {key for key in ("source_data_parallel_rank", "source_glm_model_sequence_parallel",
                "source_glm_moe_sequence_parallel") if key in attributes}
        for key in symbolic_fields:
            if (key not in attributes or not isinstance(attributes[key], tuple) or len(attributes[key]) != 1
                    or not isinstance(attributes[key][0], str) or not attributes[key][0]):
                raise ValueError("symbolic pipeline requires one typed source and destination port")
    if set(attributes) != required:
        raise ValueError(
            "pipeline P2P workload attributes must match the canonical "
            f"contract: missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    source_stage = attributes["source_stage"]
    destination_stage = attributes["destination_stage"]
    config = pool.pipeline_p2p_for(source_stage, destination_stage)
    source_ranks = pool.physical_ranks_for_pipeline_stage(source_stage)
    destination_ranks = pool.physical_ranks_for_pipeline_stage(
        destination_stage)
    distributed = (_qwen030_dp_sp(pool.parallel_plan.config)
        or pool.parallel_plan.config.data_parallel_size > 1 and _symbolic_v4_pipeline_topology_projection(node, pool)
        or _symbolic_mla_dp_pipeline_projection(node, pool)
        or pool.parallel_plan.config.data_parallel_size > 1 and _symbolic_k3_pipeline_topology_projection(node, pool))
    if distributed:
        source_rank = attributes["source_rank"]
        if source_rank not in source_ranks or symbolic_fields - set(attributes):
            raise ValueError("distributed pipeline requires a typed source in its declared stage")
        logical_rank = pool.logical_rank_for_physical_rank(source_rank)
        source_ranks = pool.physical_rank_group("tensor", logical_rank)
        destination_ranks = tuple(pool.physical_rank_group("pipeline",
            pool.logical_rank_for_physical_rank(rank))[destination_stage] for rank in source_ranks)
    if len(source_ranks) == 1 and len(destination_ranks) == 1:
        source_rank = source_ranks[0]
        destination_rank = destination_ranks[0]
        expected_payload_rank = 3 if _symbolic_v4_pipeline_topology_projection(node, pool) else 2
    elif (len(source_ranks) == len(destination_ranks)
          and len(source_ranks) > 1):
        parallelism = pool.parallel_plan.config
        symbolic_dense = (symbolic_fields <= set(attributes)
            and (parallelism.is_rank_local or distributed))
        tp2_scope = (
            len(source_ranks) == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.pipeline_parallel_size == 2
            and parallelism.expert_parallel_size == 1
        )
        target_scope = (
            len(source_ranks) == 8
            and parallelism.tensor_parallel_size == 8
            and parallelism.pipeline_parallel_size in (2, 4)
            and parallelism.enable_expert_parallel
            and parallelism.expert_parallel_size == 8
        )
        tp8_pp2_unpartitioned_scope = (
            parallelism.tensor_parallel_size == 8
            and parallelism.pipeline_parallel_size == 2
            and not parallelism.enable_expert_parallel
            and parallelism.expert_parallel_size == 1
        )
        if (not (tp2_scope or target_scope or tp8_pp2_unpartitioned_scope or symbolic_dense)
                or not (parallelism.is_rank_local or distributed)):
            raise NotImplementedError(
                "multi-lane pipeline P2P lowering requires either exact "
                "TP2/PP2, target TP=EP=8 with PP2/PP4, or TP8/PP2/EP1")
        source_rank = attributes["source_rank"]
        if source_rank not in source_ranks:
            raise ValueError(
                "pipeline P2P source rank is outside the source stage")
        lane_index = source_ranks.index(source_rank)
        destination_rank = destination_ranks[lane_index]
        expected_payload_rank = 1
    else:
        raise NotImplementedError(
            "pipeline P2P lowering requires either one rank per stage or "
            "a validated two/eight-lane TP pipeline slice")
    if node.rank != source_rank:
        raise ValueError(
            "pipeline P2P workload rank must be its physical source rank")
    expected = {
        "algorithm": config.algorithm,
        "bytes_semantics": "payload",
        "backend": config.backend,
        "backend_config_sha256": config.backend_config_sha256,
        "direction": "point_to_point",
        "link": config.link,
        "topology": config.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": config.resource_id,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "pipeline P2P workload disagrees with deployment config: "
            f"{mismatched!r}")
    payload = attributes["payload_tensor"]
    if not isinstance(payload, Mapping):
        raise TypeError("pipeline P2P payload_tensor must be a mapping")
    payload_required = {"name", "shape", "dtype", "bytes"}
    if set(payload) != payload_required:
        raise ValueError(
            "pipeline P2P payload_tensor fields must match the canonical "
            f"contract: missing={sorted(payload_required - set(payload))!r}, "
            f"extra={sorted(set(payload) - payload_required)!r}")
    if payload["name"] not in ("hidden_states", "residual"):
        raise ValueError(
            "pipeline P2P payload_tensor name must be hidden_states or "
            "residual")
    shape = payload["shape"]
    if (not isinstance(shape, (list, tuple))
            or len(shape) != expected_payload_rank
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in shape)):
        raise TypeError(
            "pipeline P2P payload_tensor shape rank disagrees with the "
            "pipeline/TP contract")
    if any(value <= 0 for value in shape):
        raise ValueError(
            "pipeline P2P payload_tensor dimensions must be positive")
    if payload["dtype"] not in ("bf16", "fp16"):
        raise ValueError(
            "validated pipeline P2P payload_tensor dtype must be bf16 or fp16")
    expected_elements = 1
    for dimension in shape:
        expected_elements *= dimension
    expected_bytes = expected_elements * 2
    if (payload["bytes"] != expected_bytes
            or attributes["bytes"] != expected_bytes):
        raise ValueError(
            "pipeline P2P payload bytes disagree with shape and dtype")


def _validate_pd_kv_transfer_node(
    node: WorkloadNode,
    deployment: DeploymentPlan,
) -> None:
    """Bind one request-scoped KV transfer to its configured P-to-D route."""

    if deployment.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("PD KV transfer lowering requires a PD deployment")
    if node.shape:
        raise ValueError("PD KV transfer workload shape must be empty")
    if node.dtype is not None or node.variant is not None:
        raise ValueError(
            "PD KV transfer workload dtype and variant must both be null")
    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "token_start", "token_end", "bytes_per_token", "attention_layout", "head_dim", "cache_dtype", "cache_layout",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "destination_pool",
        "rank_mapping",
        "source_logical_rank",
        "destination_logical_rank",
        "source_tensor_parallel_size",
        "destination_tensor_parallel_size",
        "kv_head_start",
        "kv_head_end",
        "layer_start",
        "layer_end",
        "num_key_value_heads",
    }
    compressed = attributes["attention_layout"] in ("mla", "dsa_mla") if "attention_layout" in attributes else False
    if "participant_nic_ids" in attributes:
        required.add("participant_nic_ids")
    if compressed:
        required |= {"model_num_key_value_heads", "compressed_dim", "index_head_dim", "full_indexer_layers",
            "source_context_parallel_size", "destination_context_parallel_size"}
    if set(attributes) != required:
        raise ValueError(
            "PD KV transfer attributes must match the canonical contract: "
            f"missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    destination_pool_name = attributes["destination_pool"]
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            "PD KV transfer references an unknown destination pool: "
            f"{destination_pool_name!r}")
    source_pool = deployment.worker_pools[node.pool_name]
    destination_pool = deployment.worker_pools[destination_pool_name]
    from src.serving.workload.kv_transfer_workload import validate_pd_kv_layout
    layout = validate_pd_kv_layout(source_pool, destination_pool, attributes["attention_layout"])
    if attributes["cache_layout"] != layout:
        raise ValueError("PD KV cache layout disagrees with its endpoint declarations")
    if worker_role_phase(source_pool.role) != "prefill":
        raise ValueError("PD KV transfer source must have a prefill role")
    if worker_role_phase(destination_pool.role) != "decode":
        raise ValueError("PD KV transfer destination must have a decode role")
    if (worker_role_operator_group(source_pool.role) == "ffn"
            or worker_role_operator_group(destination_pool.role) == "ffn"):
        raise ValueError(
            "PD KV transfer endpoints must be unsplit or Attention-owner "
            "pools, not FFN pools")
    if attributes["destination_rank"] not in destination_pool.ranks:
        raise ValueError("PD KV transfer destination rank disagrees with deployment config")
    routes = tuple(
        route for route in deployment.routes
        if route.payload == "kv"
        and route.source_pool == source_pool.name
        and route.destination_pool == destination_pool.name
    )
    if len(routes) != 1:
        raise ValueError(
            "PD KV transfer requires exactly one configured route: "
            f"{source_pool.name!r}->{destination_pool.name!r}")
    route = routes[0]
    from src.serving.workload.kv_transfer_workload import pd_transfer_node_nic_ids
    declared_nics = tuple(attributes["participant_nic_ids"]) if "participant_nic_ids" in attributes else ()
    expected_nics = pd_transfer_node_nic_ids(source_pool, destination_pool, route,
        node.rank, attributes["destination_rank"])
    if declared_nics != expected_nics:
        raise ValueError("PD transfer node-port identities disagree with placement")
    rank_pairs = pd_kv_transfer_rank_pairs(
        source_pool,
        destination_pool,
        attributes["num_key_value_heads"],
        source_data_parallel_rank=source_pool.logical_rank_for_physical_rank(node.rank) // (
            source_pool.parallel_plan.config.worker_count // source_pool.parallel_plan.config.data_parallel_size),
        destination_data_parallel_rank=destination_pool.logical_rank_for_physical_rank(attributes["destination_rank"]) // (
            destination_pool.parallel_plan.config.worker_count // destination_pool.parallel_plan.config.data_parallel_size),
        attention_layout=attributes["attention_layout"],
        kv_transport_scope=route.kv_transport_scope,
    )
    if source_pool.hardware_id != destination_pool.hardware_id and (
            route.link != "infiniband" or route.backend != "analytical_roofline_v1"):
        raise ValueError("cross-hardware KV requires a declared analytical InfiniBand route")
    matching_rank_pairs = tuple(
        rank_pair for rank_pair in rank_pairs
        if (rank_pair.source_logical_rank
            == attributes["source_logical_rank"]
            and rank_pair.destination_logical_rank
            == attributes["destination_logical_rank"]
            and rank_pair.source_rank == node.rank
            and rank_pair.source_rank == attributes["source_rank"]
            and rank_pair.destination_rank == attributes["destination_rank"]
            and rank_pair.kv_head_start == attributes["kv_head_start"]
            and rank_pair.kv_head_end == attributes["kv_head_end"]
            and rank_pair.layer_start == attributes["layer_start"]
            and rank_pair.layer_end == attributes["layer_end"])
    )
    if len(matching_rank_pairs) != 1:
        raise ValueError(
            "PD KV transfer workload disagrees with deployment config: "
            "not one canonical KV-head ownership edge")
    rank_pair = matching_rank_pairs[0]
    source_rank = rank_pair.source_rank
    destination_rank = rank_pair.destination_rank
    transfer_bytes = attributes["bytes"]
    if (isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes <= 0):
        raise TypeError("PD KV transfer bytes must be a positive integer")
    if (type(attributes["token_start"]) is not int or attributes["token_start"] != 0
            or type(attributes["token_end"]) is not int or attributes["token_end"] <= 0
            or type(attributes["bytes_per_token"]) is not int or attributes["bytes_per_token"] <= 0):
        raise ValueError("PD KV transfer requires a positive whole-prompt token interval and byte stride")
    if attributes["attention_layout"] == "standard_gqa":
        from src.serving.workload.kv_transfer_workload import gqa_kv_interval_bytes_per_token
        stride = gqa_kv_interval_bytes_per_token(
            rank_pair.layer_end - rank_pair.layer_start,
            rank_pair.kv_head_end - rank_pair.kv_head_start, attributes["head_dim"], attributes["cache_dtype"])
        if stride != attributes["bytes_per_token"]:
            raise ValueError("GQA KV payload stride disagrees with its layer/head/dtype interval")
    elif attributes["attention_layout"] in ("mla", "dsa_mla"):
        if (attributes["head_dim"] is not None or attributes["cache_dtype"] != "bf16"
                or type(attributes["compressed_dim"]) is not int or attributes["compressed_dim"] <= 0
                or type(attributes["model_num_key_value_heads"]) is not int or attributes["model_num_key_value_heads"] <= 0
                or attributes["num_key_value_heads"] != 1
                or (attributes["kv_head_start"], attributes["kv_head_end"]) != (0, 1)
                or attributes["source_context_parallel_size"] != source_pool.parallel_plan.config.context_parallel_size
                or attributes["destination_context_parallel_size"] != destination_pool.parallel_plan.config.context_parallel_size):
            raise ValueError("compressed MLA payload disagrees with deployment config: requires one replicated BF16 cache with explicit CP identities")
        full_layers = attributes["full_indexer_layers"]
        if (not isinstance(full_layers, tuple) or full_layers != tuple(sorted(set(full_layers)))
                or any(type(layer) is not int or not rank_pair.layer_start <= layer < rank_pair.layer_end for layer in full_layers)
                or (attributes["attention_layout"] == "mla" and (full_layers or attributes["index_head_dim"] != 0))
                or (attributes["attention_layout"] == "dsa_mla" and attributes["index_head_dim"] != 128)):
            raise ValueError("compressed MLA payload must bind its actual full-indexer layer intervals")
        stride = ((rank_pair.layer_end - rank_pair.layer_start) * attributes["compressed_dim"]
            * DTYPE_BYTES[attributes["cache_dtype"]] + len(full_layers) * 132)
        if stride != attributes["bytes_per_token"]:
            raise ValueError("compressed MLA byte stride disagrees with layer and full-indexer intervals")
    else:
        raise ValueError("PD KV payload has an unsupported attention layout")
    if transfer_bytes != attributes["token_end"] * attributes["bytes_per_token"]:
        raise ValueError("PD KV payload bytes disagree with the token interval")
    expected = {
        "algorithm": route.algorithm,
        "bytes_semantics": "payload",
        "backend": route.backend,
        "backend_config_sha256": route.backend_config_sha256,
        "direction": "point_to_point",
        "link": route.link,
        "topology": route.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": route.physical_resource_id(rank_pair.destination_logical_rank),
        "destination_pool": destination_pool.name,
        "rank_mapping": "replicated_mla_cache" if compressed else "kv_head_overlap",
        "source_logical_rank": rank_pair.source_logical_rank,
        "destination_logical_rank": rank_pair.destination_logical_rank,
        "source_tensor_parallel_size": (
            source_pool.parallel_plan.config.tensor_parallel_size),
        "destination_tensor_parallel_size": (
            destination_pool.parallel_plan.config.tensor_parallel_size),
        "kv_head_start": rank_pair.kv_head_start,
        "kv_head_end": rank_pair.kv_head_end,
        "layer_start": rank_pair.layer_start,
        "layer_end": rank_pair.layer_end,
        "num_key_value_heads": attributes["num_key_value_heads"],
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "PD KV transfer workload disagrees with deployment config: "
            f"{mismatched!r}")


def _validate_af_activation_transfer_node(
    node: WorkloadNode,
    deployment: DeploymentPlan,
    graph: WorkloadGraph,
) -> None:
    """Bind one AF transfer to its configured route and physical TP lane."""

    attributes = node.attributes
    required = {
        "algorithm",
        "bytes",
        "bytes_semantics",
        "backend",
        "backend_config_sha256",
        "direction",
        "link",
        "topology",
        "participants",
        "operation",
        "source_rank",
        "destination_rank",
        "ring_order",
        "reduction",
        "resource_id",
        "destination_pool",
        "payload_tensor",
    }
    graph_domain_fields = set(_GRAPH_TOKEN_DOMAIN_FIELDS)
    present_graph_domain_fields = graph_domain_fields & set(attributes)
    if present_graph_domain_fields:
        if present_graph_domain_fields != graph_domain_fields:
            raise ValueError(
                "AF activation graph token domains must be complete")
        required |= graph_domain_fields
    if "residual_anchor" in attributes:
        # Only the observation-suppression marker is legal, and only on a
        # residual payload transfer (the builder stamps it nowhere else).
        if (attributes["residual_anchor"]
                != RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED
                or attributes["payload_tensor"]["name"] != "residual"):
            raise ValueError(
                "AF activation transfer carries an unknown residual_anchor "
                f"contract: {attributes['residual_anchor']!r}")
        required.add("residual_anchor")
    symbolic_fields = {"symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_af_boundary"}
    typed = bool(symbolic_fields & set(attributes))
    if typed:
        required |= symbolic_fields
        if (symbolic_fields - set(attributes) or not isinstance(attributes["symbolic_af_boundary"], str)
                or not attributes["symbolic_af_boundary"]):
            raise ValueError("typed AF transport needs complete ports and a boundary identity")
        for key in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"):
            if (not isinstance(attributes[key], tuple) or len(attributes[key]) != 1
                    or not isinstance(attributes[key][0], str) or not attributes[key][0]):
                raise ValueError("typed AF transport requires one input and output view")
    if set(attributes) != required:
        raise ValueError(
            "AF activation transfer attributes must match the canonical "
            f"contract: missing={sorted(required - set(attributes))!r}, "
            f"extra={sorted(set(attributes) - required)!r}")
    destination_pool_name = attributes["destination_pool"]
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            "AF activation transfer references an unknown destination pool: "
            f"{destination_pool_name!r}")
    source_pool = deployment.worker_pools[node.pool_name]
    destination_pool = deployment.worker_pools[destination_pool_name]
    routes = tuple(
        route for route in deployment.routes
        if route.payload == "activation"
        and route.source_pool == source_pool.name
        and route.destination_pool == destination_pool.name
    )
    if len(routes) != 1:
        raise ValueError(
            "AF activation transfer requires exactly one configured route: "
            f"{source_pool.name!r}->{destination_pool.name!r}")
    route = routes[0]
    source_parallelism = source_pool.parallel_plan.config
    destination_parallelism = destination_pool.parallel_plan.config
    if len(source_pool.ranks) == 1 and len(destination_pool.ranks) == 1:
        source_rank = source_pool.ranks[0]
        destination_rank = destination_pool.ranks[0]
        tp2_lane = False
    elif len(source_pool.ranks) == 2 and len(destination_pool.ranks) == 2:
        common_tp2 = all(
            parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.pipeline_parallel_size == 1
            and parallelism.sequence_parallel_size == 1
            and parallelism.context_parallel_size == 1
            and parallelism.data_parallel_size == 1
            for parallelism in (
                source_parallelism, destination_parallelism)
        )
        operator_parallelism = {
            worker_role_operator_group(source_pool.role): source_parallelism,
            worker_role_operator_group(
                destination_pool.role): destination_parallelism,
        }
        dense_tp2 = all(
            parallelism.expert_parallel_size == 1
            for parallelism in (
                source_parallelism, destination_parallelism)
        )
        af_ep2 = (
            deployment.mode == "af_disagg"
            and set(operator_parallelism) == {"attention", "ffn"}
            and operator_parallelism["attention"].expert_parallel_size == 1
            and operator_parallelism["ffn"].expert_parallel_size == 2
        )
        exact_tp2 = common_tp2 and (dense_tp2 or af_ep2)
        if not exact_tp2:
            raise NotImplementedError(
                "two-rank AF activation lowering requires matching TP=2 "
                "lanes in the validated dense or FFN-EP2 scope")
        source_rank = attributes["source_rank"]
        if source_rank not in source_pool.ranks:
            raise ValueError(
                "AF activation source rank is outside the source pool")
        lane_index = source_pool.ranks.index(source_rank)
        destination_rank = destination_pool.ranks[lane_index]
        tp2_lane = True
    else:
        raise NotImplementedError(
            "AF activation lowering requires one rank per pool or matching "
            "two-lane TP=2 pools")
    if node.rank != source_rank:
        raise ValueError(
            "AF activation workload rank must be its physical source rank")
    expected = {
        "algorithm": route.algorithm,
        "bytes_semantics": "payload",
        "backend": route.backend,
        "backend_config_sha256": route.backend_config_sha256,
        "direction": "point_to_point",
        "link": route.link,
        "topology": route.topology,
        "participants": (source_rank, destination_rank),
        "operation": "send_recv",
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "ring_order": (),
        "reduction": None,
        "resource_id": route.resource_id,
        "destination_pool": destination_pool.name,
    }
    mismatched = {
        name: (attributes[name], value)
        for name, value in expected.items()
        if attributes[name] != value
    }
    if mismatched:
        raise ValueError(
            "AF activation workload disagrees with deployment route/lane: "
            f"{mismatched!r}")
    transfer_bytes = attributes["bytes"]
    if (isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes <= 0):
        raise TypeError("AF activation payload bytes must be a positive integer")
    payload = attributes["payload_tensor"]
    if not isinstance(payload, Mapping):
        raise TypeError("AF activation payload_tensor must be a mapping")
    payload_required = {"name", "shape", "dtype", "bytes"}
    if set(payload) != payload_required:
        raise ValueError(
            "AF activation payload_tensor fields must match the canonical "
            f"contract: missing={sorted(payload_required - set(payload))!r}, "
            f"extra={sorted(set(payload) - payload_required)!r}")
    tensor_name = payload["name"]
    if tensor_name not in ("hidden_states", "residual"):
        raise ValueError(
            "AF activation payload_tensor name must be hidden_states or "
            "residual")
    payload_shape = payload["shape"]
    if (not isinstance(payload_shape, (list, tuple))
            or len(payload_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in payload_shape)):
        raise TypeError(
            "AF activation payload_tensor shape must be a two-dimensional "
            "integer sequence")
    if any(value <= 0 for value in payload_shape):
        raise ValueError(
            "AF activation payload_tensor dimensions must be positive")
    payload_dtype = payload["dtype"]
    if payload_dtype not in ("bf16", "fp16"):
        raise ValueError(
            "AF activation payload_tensor dtype must be BF16 or FP16")
    expected_payload_bytes = payload_shape[0] * payload_shape[1] * 2
    payload_bytes = payload["bytes"]
    if (isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0):
        raise TypeError(
            "AF activation payload_tensor bytes must be a positive integer")
    if (payload_bytes != expected_payload_bytes
            or transfer_bytes != expected_payload_bytes):
        raise ValueError(
            "AF activation payload bytes disagree with shape and dtype")
    incoming_edges = tuple(
        edge for edge in graph.edges if edge.destination == node.node_id)
    if typed:
        if any(edge.kind not in ("ready", "barrier") for edge in incoming_edges):
            raise ValueError("typed AF transport only accepts producer-ready and send-order dependencies")
        incoming_edges = tuple(edge for edge in incoming_edges if edge.kind == "ready")
    if len(incoming_edges) != 1 or incoming_edges[0].kind != "ready":
        raise ValueError(
            "AF activation transfer requires exactly one ready producer edge")
    producer = graph.node(incoming_edges[0].source)
    if typed and ("symbolic_tensor_outputs" not in producer.attributes
            or attributes["symbolic_tensor_inputs"][0] not in producer.attributes["symbolic_tensor_outputs"]):
        raise ValueError("typed AF source port must bind its producer output")
    if producer.pool_name != source_pool.name:
        raise ValueError(
            "AF activation producer must belong to the source pool")
    if tensor_name == "residual":
        if producer.kind is WorkloadKind.COLLECTIVE:
            if (producer.attributes["operation"] != "all_reduce"
                    or tuple(producer.attributes["participants"])
                    != source_pool.ranks
                    or source_rank not in producer.attributes["participants"]):
                raise ValueError(
                    "AF initial residual collective must be the canonical "
                    "source-pool all-reduce")
            producer_bytes = producer.attributes["bytes"]
        elif producer.kind in (
                WorkloadKind.NORM,
                WorkloadKind.GATHER,
                WorkloadKind.LINEAR,
        ):
            if (producer.rank != source_rank
                    or producer.dtype != payload_dtype):
                raise ValueError(
                    "AF residual producer rank or dtype disagrees with the "
                    "source lane")
            if producer.kind is WorkloadKind.NORM:
                if (producer.variant != "add_rmsnorm"
                        or set(producer.shape) != {"rows", "hidden"}):
                    raise ValueError(
                        "AF carried residual requires a canonical "
                        "add_rmsnorm producer")
                producer_shape = (
                    producer.shape["rows"], producer.shape["hidden"])
            elif producer.kind is WorkloadKind.GATHER:
                embedding_shape = {"elements", "rows", "hidden"}
                if typed and "index_select_source" in producer.attributes:
                    if producer.attributes["index_select_source"] != "vllm_0_11_embedding":
                        raise ValueError("AF initial residual requires its canonical embedding source")
                    # The0.11 embedding program carries its padded table
                    # extent; that input extent does not alter this output.
                    embedding_shape.add("source_rows")
                elif typed and "embedding_source" in producer.attributes:
                    source_fields = {"model_execution_version", "source_operation"}
                    if (source_fields - set(producer.attributes)
                            or producer.attributes["embedding_source"] != "vllm_0_30_unquantized_tp1"
                            or producer.attributes["model_execution_version"] != "vllm_0_30_0"
                            or producer.attributes["source_operation"] != "embedding"):
                        raise ValueError("AF initial residual requires its canonical embedding source")
                    # The0.30 ordinary embedding now retains its known,
                    # globally padded table extent, separate from output rows.
                    if ("source_rows" not in producer.shape
                            or type(producer.shape["source_rows"]) is not int
                            or producer.shape["source_rows"] <= 0
                            or producer.shape["source_rows"] % 64):
                        raise ValueError("AF initial residual requires its padded embedding table extent")
                    embedding_shape.add("source_rows")
                if (producer.variant != "embedding"
                        or set(producer.shape) != embedding_shape):
                    raise ValueError(
                        "AF initial residual requires a canonical embedding "
                        "producer")
                producer_shape = (
                    producer.shape["rows"], producer.shape["hidden"])
            elif producer.kind is WorkloadKind.LINEAR:
                if (producer.variant != "gemm"
                        or set(producer.shape) != {"b", "m", "n", "k"}):
                    raise ValueError(
                        "AF initial residual requires a canonical linear "
                        "producer")
                producer_shape = (
                    producer.shape["b"] * producer.shape["m"],
                    producer.shape["n"],
                )
            else:
                raise ValueError(
                    "AF residual payload has an unsupported producer kind")
            if tuple(payload_shape) != producer_shape:
                raise ValueError(
                    "AF residual payload shape disagrees with its producer")
            producer_bytes = producer_shape[0] * producer_shape[1] * 2
        else:
            raise ValueError(
                "AF residual payload has an unsupported producer kind")
    elif tp2_lane:
        if (producer.kind is not WorkloadKind.COLLECTIVE
                or producer.attributes["operation"] != "all_reduce"
                or tuple(producer.attributes["participants"])
                != source_pool.ranks):
            raise ValueError(
                "AF TP=2 hidden_states lanes must follow the source-pool "
                "all-reduce frontier")
        producer_bytes = producer.attributes["bytes"]
    elif tensor_name == "hidden_states":
        if producer.rank != source_rank:
            raise ValueError(
                "AF activation producer rank disagrees with the source rank")
        if producer.dtype != payload_dtype:
            raise ValueError(
                "single-rank AF hidden_states producer dtype disagrees with "
                "the payload")
        if producer.kind is WorkloadKind.LINEAR:
            if (producer.variant != "gemm"
                    or set(producer.shape) != {"b", "m", "n", "k"}):
                raise ValueError(
                    "single-rank AF linear producer requires canonical GEMM "
                    "dimensions")
            producer_elements = (
                producer.shape["b"]
                * producer.shape["m"]
                * producer.shape["n"]
            )
        elif producer.kind is WorkloadKind.FUSED_MOE:
            if (producer.variant != "full"
                    or set(producer.shape) != {
                        "tokens", "experts", "topk", "hidden",
                        "intermediate",
                    }):
                raise ValueError(
                    "single-rank AF fused-MoE producer requires canonical "
                    "dimensions")
            producer_elements = (
                producer.shape["tokens"] * producer.shape["hidden"])
        elif producer.kind is WorkloadKind.ELEMENTWISE:
            if (producer.variant != "add"
                    or set(producer.shape) != {"elements"}):
                raise ValueError(
                    "single-rank AF elementwise producer requires a canonical "
                    "add shape")
            producer_elements = producer.shape["elements"]
        else:
            raise ValueError(
                "single-rank AF hidden_states must follow a canonical linear, "
                "fused-MoE, or elementwise-add producer")
        producer_bytes = producer_elements * 2
    else:
        raise AssertionError("validated AF payload tensor name vanished")
    if transfer_bytes != producer_bytes:
        raise ValueError(
            "AF activation payload must contain one full source tensor: "
            f"name={tensor_name!r}, expected={producer_bytes}, "
            f"actual={transfer_bytes}")
    outgoing_edges = tuple(
        edge for edge in graph.edges if edge.source == node.node_id)
    if typed:
        outgoing_edges = tuple(edge for edge in outgoing_edges if edge.kind == "data")
    if len(outgoing_edges) != 1 or outgoing_edges[0].kind != "data":
        raise ValueError(
            "AF activation transfer requires exactly one destination data edge")
    consumer = graph.node(outgoing_edges[0].destination)
    if typed and ("symbolic_tensor_inputs" not in consumer.attributes
            or attributes["symbolic_tensor_outputs"][0] not in consumer.attributes["symbolic_tensor_inputs"]):
        raise ValueError("typed AF receive port must bind its destination consumer")
    if (consumer.pool_name != destination_pool.name
            or consumer.rank != destination_rank):
        raise ValueError(
            "AF activation consumer disagrees with the destination pool/lane")
    if (consumer.kind is WorkloadKind.NORM
            and consumer.variant == "add_rmsnorm"
            and set(consumer.shape) == {"rows", "hidden"}):
        consumer_elements = (
            consumer.shape["rows"] * consumer.shape["hidden"])
        if tuple(payload_shape) != (
                consumer.shape["rows"], consumer.shape["hidden"]):
            raise ValueError(
                "AF activation payload shape disagrees with its "
                "add_rmsnorm consumer")
    elif (consumer.kind is WorkloadKind.ELEMENTWISE
          and consumer.variant == "add"
          and set(consumer.shape) == {"elements"}):
        consumer_elements = consumer.shape["elements"]
    else:
        raise ValueError(
            "AF hidden_states/residual payload pair must feed a canonical "
            "add_rmsnorm or elementwise-add consumer")
    if consumer_elements != payload_shape[0] * payload_shape[1]:
        raise ValueError(
            "AF activation payload shape disagrees with its destination "
            "consumer")


def _validate_symbolic_af_boundaries(graph, deployment, af_nodes):
    """Require both real tensor ports on every lane of each typed AF call."""
    incoming = {}
    for edge in graph.edges:
        if edge.destination not in incoming:
            incoming[edge.destination] = []
        incoming[edge.destination].append(edge)
    ancestor_cache = {}

    def ancestors(node_id):
        if node_id not in ancestor_cache:
            found, pending = set(), list(graph.predecessors(node_id))
            while pending:
                parent = pending.pop()
                if parent not in found:
                    found.add(parent)
                    pending.extend(graph.predecessors(parent))
            ancestor_cache[node_id] = found
        return ancestor_cache[node_id]

    boundaries = {}
    for node in af_nodes:
        if "symbolic_af_boundary" not in node.attributes:
            raise ValueError("AF cannot mix typed and legacy activation messages")
        key = (node.pool_name, node.attributes["destination_pool"], node.attributes["symbolic_af_boundary"])
        if key not in boundaries:
            boundaries[key] = {}
        lane = (node.rank, node.attributes["payload_tensor"]["name"])
        if lane in boundaries[key]:
            raise ValueError("typed AF boundary duplicates a tensor lane")
        boundaries[key][lane] = node
    for (source, destination, _), messages in boundaries.items():
        source_pool, destination_pool = deployment.worker_pools[source], deployment.worker_pools[destination]
        if set(messages) != {(rank, role) for rank in source_pool.ranks for role in ("hidden_states", "residual")}:
            raise ValueError("typed AF boundary requires a hidden_states/residual activation pair on every lane")
        for source_rank, destination_rank in zip(source_pool.ranks, destination_pool.ranks, strict=True):
            hidden, residual = (messages[source_rank, role] for role in ("hidden_states", "residual"))
            if (dict(hidden.attributes["payload_tensor"]) != {**residual.attributes["payload_tensor"], "name": "hidden_states"}
                    or WorkloadEdge(hidden.node_id, residual.node_id, "barrier") not in graph.edges):
                raise ValueError("typed AF pair must have identical layout and hidden-then-residual send order")
            if len(incoming[hidden.node_id]) != 1 or len(incoming[residual.node_id]) != 2:
                raise ValueError("typed AF messages require their exact producer and dictionary send order")
            hidden_producer = next(edge.source for edge in incoming[hidden.node_id] if edge.kind == "ready")
            residual_producer = next(edge.source for edge in incoming[residual.node_id] if edge.kind == "ready")
            rows, width = hidden.attributes["payload_tensor"]["shape"]
            norms = tuple(node for node in graph.nodes if node.node_id in ancestors(hidden_producer)
                and node.kind is WorkloadKind.NORM and node.variant in ("rmsnorm", "add_rmsnorm")
                and node.pool_name == source and node.rank == source_rank
                and dict(node.shape) == {"rows": rows, "hidden": width})
            latest = tuple(node for node in norms if all(other.node_id == node.node_id
                or other.node_id in ancestors(node.node_id) for other in norms))
            if len(latest) != 1:
                raise ValueError("typed AF hidden state must follow one current source residual norm")
            norm = latest[0]
            if norm.variant == "add_rmsnorm":
                expected_residual = norm.node_id
            else:
                norm_inputs = tuple(edge.source for edge in incoming[norm.node_id] if edge.kind == "data")
                if len(norm_inputs) != 1:
                    raise ValueError("typed AF initial residual norm requires its embedding producer")
                expected_residual = norm_inputs[0]
            if residual_producer != expected_residual:
                raise ValueError("typed AF residual must use the current boundary residual producer")
            consumers = tuple(tuple(graph.node(edge.destination) for edge in graph.edges
                if edge.source == message.node_id and edge.kind == "data") for message in (hidden, residual))
            if (any(len(nodes) != 1 for nodes in consumers) or consumers[0][0].node_id != consumers[1][0].node_id
                    or consumers[0][0].rank != destination_rank):
                raise ValueError("typed AF pair must feed the same destination residual norm")
    for node in graph.nodes:
        if node.kind is WorkloadKind.NORM and node.variant == "add_rmsnorm":
            inputs = tuple(graph.node(edge.source) for edge in graph.edges
                if edge.destination == node.node_id and edge.kind == "data")
            if len(inputs) != 2 or any(item.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER for item in inputs):
                raise ValueError("typed AF residual norm must read the complete transferred pair")
    for edge in graph.edges:
        source, destination = graph.node(edge.source), graph.node(edge.destination)
        if source.pool_name != destination.pool_name and (source.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER
                or source.attributes["destination_pool"] != destination.pool_name
                or source.attributes["destination_rank"] != destination.rank):
            raise ValueError("typed AF cross-pool dependency must use its configured transfer lane")


def _validate_af_activation_boundaries(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Require one ordered hidden_states/residual pair on every AF lane."""

    af_nodes = tuple(
        node for node in graph.nodes
        if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER
    )
    if deployment.mode not in ("af_disagg", "pd_af_disagg"):
        if af_nodes:
            raise ValueError(
                "AF activation transfers require an AF deployment mode")
        return
    if not af_nodes:
        if all(
                node.kind is WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            return
        raise ValueError(
            "AF deployment workload must contain activation transfers")
    if any("symbolic_af_boundary" in node.attributes for node in af_nodes):
        _validate_symbolic_af_boundaries(graph, deployment, af_nodes)
        return

    for edge in graph.edges:
        source = graph.node(edge.source)
        destination = graph.node(edge.destination)
        if (source.pool_name != destination.pool_name
                and source.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER):
            raise ValueError(
                "AF cross-pool dependencies must pass through an activation "
                "transfer node")

    for node in graph.nodes:
        requires_pair = False
        if (node.kind is WorkloadKind.NORM
                and node.variant == "add_rmsnorm"):
            requires_pair = True
        elif (node.kind is WorkloadKind.ELEMENTWISE
              and node.variant == "add"):
            pool_role = deployment.worker_pools[node.pool_name].role
            if pool_role == "attention" or pool_role.endswith("-attention"):
                requires_pair = True
        if not requires_pair:
            continue
        incoming_edges = tuple(
            edge for edge in graph.edges if edge.destination == node.node_id)
        if (len(incoming_edges) != 2
                or any(
                    graph.node(edge.source).kind
                    is not WorkloadKind.AF_ACTIVATION_TRANSFER
                    for edge in incoming_edges)):
            raise ValueError(
                "AF residual consumer requires exactly one hidden_states/"
                "residual activation pair")

    ancestor_cache = {}

    def ancestor_ids(node_id):
        if node_id in ancestor_cache:
            return ancestor_cache[node_id]
        ancestors = set()
        pending = list(graph.predecessors(node_id))
        while pending:
            predecessor_id = pending.pop()
            if predecessor_id in ancestors:
                continue
            ancestors.add(predecessor_id)
            pending.extend(graph.predecessors(predecessor_id))
        ancestor_cache[node_id] = frozenset(ancestors)
        return ancestor_cache[node_id]

    def establishes_residual(candidate):
        if candidate.kind is not WorkloadKind.NORM:
            return False
        if candidate.variant == "add_rmsnorm":
            return True
        if candidate.variant != "rmsnorm":
            return False
        successors = tuple(
            graph.node(edge.destination)
            for edge in graph.edges
            if edge.source == candidate.node_id
        )
        return (
            len(successors) == 1
            and successors[0].kind is WorkloadKind.LINEAR
        )

    pairs = {}
    for node in af_nodes:
        outgoing_edge, = tuple(
            edge for edge in graph.edges if edge.source == node.node_id)
        key = (
            node.pool_name,
            node.attributes["destination_pool"],
            node.attributes["source_rank"],
            node.attributes["destination_rank"],
            outgoing_edge.destination,
        )
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(node)

    boundaries = {}
    for pair_key, nodes in pairs.items():
        source_name, destination_name, source_rank, destination_rank, _ = (
            pair_key)
        names = tuple(
            node.attributes["payload_tensor"]["name"] for node in nodes)
        if len(nodes) != 2 or names != ("hidden_states", "residual"):
            raise ValueError(
                "AF boundary lane must contain one ordered hidden_states/"
                "residual payload pair: "
                f"lane={pair_key!r}, payloads={names!r}")
        hidden_node, residual_node = nodes
        hidden_payload = hidden_node.attributes["payload_tensor"]
        residual_payload = residual_node.attributes["payload_tensor"]
        if any(
                hidden_payload[field] != residual_payload[field]
                for field in ("shape", "dtype", "bytes")):
            raise ValueError(
                "AF hidden_states/residual payload pair must have identical "
                "shape, dtype, and byte count")
        hidden_incoming, = tuple(
            edge for edge in graph.edges
            if edge.destination == hidden_node.node_id)
        residual_incoming, = tuple(
            edge for edge in graph.edges
            if edge.destination == residual_node.node_id)
        payload_shape = tuple(hidden_payload["shape"])
        payload_dtype = hidden_payload["dtype"]
        current_residual_norms = tuple(
                candidate for candidate in graph.nodes
                if (candidate.node_id in ancestor_ids(hidden_incoming.source)
                    and candidate.pool_name == source_name
                    and candidate.rank == source_rank
                    and establishes_residual(candidate)
                    and candidate.dtype == payload_dtype
                and set(candidate.shape) == {"rows", "hidden"}
                and (candidate.shape["rows"], candidate.shape["hidden"])
                == payload_shape)
        )
        if not current_residual_norms:
            raise ValueError(
                "AF hidden_states producer has no current residual norm")
        latest_residual_norms = tuple(
            candidate for candidate in current_residual_norms
            if all(
                other.node_id == candidate.node_id
                or other.node_id in ancestor_ids(candidate.node_id)
                for other in current_residual_norms
            )
        )
        if len(latest_residual_norms) != 1:
            raise ValueError(
                "AF hidden_states producer must have one unique latest "
                "residual norm in the dependency DAG: "
                f"candidates={tuple(candidate.node_id for candidate in current_residual_norms)!r}, "
                f"latest={tuple(candidate.node_id for candidate in latest_residual_norms)!r}")
        residual_norm, = latest_residual_norms
        if residual_norm.variant == "add_rmsnorm":
            expected_residual_source = residual_norm.node_id
        elif residual_norm.variant == "rmsnorm":
            residual_predecessors = graph.predecessors(
                residual_norm.node_id)
            if len(residual_predecessors) == 1:
                expected_residual_source = residual_predecessors[0]
            elif not residual_predecessors:
                # The step-head input embedding was suppressed by the
                # versioned emission observation, so the initial residual
                # norm has no in-graph producer and the residual transfer
                # must carry the builder's observation marker and anchor on
                # the hidden-state producer of the same boundary.
                if ("residual_anchor" not in residual_node.attributes
                        or residual_node.attributes["residual_anchor"]
                        != RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED
                        or residual_incoming.source != hidden_incoming.source):
                    raise ValueError(
                        "AF initial residual norm lost its input producer "
                        "without the observation-suppression anchor")
                expected_residual_source = residual_incoming.source
            else:
                raise ValueError(
                    "AF initial residual norm requires exactly one input "
                    "producer")
        else:
            raise AssertionError(
                "validated residual RMSNorm variant vanished")
        if residual_incoming.source != expected_residual_source:
            raise ValueError(
                "AF residual transfer must use the current boundary residual "
                "producer: "
                f"expected={expected_residual_source!r}, "
                f"actual={residual_incoming.source!r}")
        boundary_key = (
            hidden_incoming.source,
            source_name,
            destination_name,
        )
        if boundary_key not in boundaries:
            boundaries[boundary_key] = []
        boundaries[boundary_key].append((
            source_rank,
            destination_rank,
            hidden_node.attributes["participants"],
        ))

    for (producer_id, source_name, destination_name), lanes in (
            boundaries.items()):
        source_pool = deployment.worker_pools[source_name]
        destination_pool = deployment.worker_pools[destination_name]
        expected_lanes = tuple(zip(
            source_pool.ranks,
            destination_pool.ranks,
            strict=True,
        ))
        actual_lanes = tuple(
            (source_rank, destination_rank)
            for source_rank, destination_rank, _ in lanes)
        actual_participants = tuple(
            tuple(participants) for _, _, participants in lanes)
        if (len(lanes) != len(expected_lanes)
                or len(set(actual_lanes)) != len(expected_lanes)
                or set(actual_lanes) != set(expected_lanes)
                or set(actual_participants) != set(expected_lanes)):
            raise ValueError(
                "AF boundary must cover each logical lane exactly once: "
                f"producer={producer_id!r}, source={source_name!r}, "
                f"destination={destination_name!r}, "
                f"expected_lanes={expected_lanes!r}, "
                f"actual_lanes={actual_lanes!r}")


def _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, pool):
    """MTP's declared un-reduced Cutlass FP8 EP source, separate from pricing."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and pool.role == "colocated" and config.is_rank_local
        and config.enable_expert_parallel and len(pool.ranks) == config.worker_count
        and pool.parallel_plan.num_layers == 78 and pool.parallel_plan.num_experts == 256
        and "source_glm_rank_local_fp8_mtp_ep" in node.attributes
        and node.attributes["source_glm_rank_local_fp8_mtp_ep"] is True
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and node.attributes["glm_parallel_source"] is True
        and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and node.attributes["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce")


def _symbolic_glm_fp8_target_dp_projection(node, pool):
    """Ordinary target source identity; never impersonates an MTP pass."""
    config, attrs = pool.parallel_plan.config, node.attributes
    tp, dp = config.tensor_parallel_size, config.data_parallel_size
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and pool.role == "colocated" and config.context_parallel_size == 1
        and dp > 1 and 2048 % (tp * 128) == 0
        and (256 % config.expert_parallel_size == 0 if config.enable_expert_parallel else 2048 % (tp * dp * 128) == 0)
        and config.sequence_parallel_size in ((1, tp) if config.enable_expert_parallel else (1,))
        and pool.parallel_plan.num_layers == 78 and pool.parallel_plan.num_experts == 256
        and "source_glm_fp8_target_only" in attrs and attrs["source_glm_fp8_target_only"] is True
        and "glm_parallel_source" in attrs and attrs["glm_parallel_source"] is True
        and "source_projection_only" in attrs and attrs["source_projection_only"] is True
        and attrs["model_execution_version"] == "vllm_0_30_0"
        and attrs["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce")


def _symbolic_glm_pcp_projection(node, pool):
    config = pool.parallel_plan.config
    pcp = ("pcp_source" in node.attributes and node.attributes["pcp_source"] is True
        and config.context_parallel_size > 1)
    ordinary = ("glm_parallel_source" in node.attributes and node.attributes["glm_parallel_source"] is True
        and config.context_parallel_size == 1)
    pipeline = (config.pipeline_parallel_size == 1 or
        "source_pipeline_stage" in node.attributes and type(node.attributes["source_pipeline_stage"]) is int
        and node.attributes["source_pipeline_stage"] == (
            pool.logical_rank_for_physical_rank(node.rank) // config.tensor_parallel_size) % config.pipeline_parallel_size)
    return ((pcp or ordinary)
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and (node.attributes["model_source_profile"] == "vllm_0_30_v32_bf16_sync_explicit_allreduce"
            or ordinary and node.attributes["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce"
            and config.context_parallel_size == 1 and HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id)
            and (not config.enable_expert_parallel and config.sequence_parallel_size == 1
                or _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, pool)
                or _symbolic_glm_fp8_target_dp_projection(node, pool)
                or config.enable_expert_parallel and config.data_parallel_size > 1
                and config.sequence_parallel_size in (1, config.tensor_parallel_size)
                and "source_data_parallel_rank" in node.attributes))
        and (HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", pool.hardware_id)
            or HWSpec.source_contract_supported("glm_tp8_pp2_ep1_source_projection", pool.hardware_id))
        and pipeline
        and (config.sequence_parallel_size == 1 and (not config.enable_expert_parallel or config.data_parallel_size == 1)
            or config.enable_expert_parallel and config.data_parallel_size > 1
                and config.sequence_parallel_size in (1, config.tensor_parallel_size) and config.tensor_parallel_size > 1
            or config.enable_expert_parallel and config.tensor_parallel_size == config.sequence_parallel_size == 1
                and config.data_parallel_size > 1
                and (config.context_parallel_size == 1 or "source_pcp_expert_communication_scope" in node.attributes
                and node.attributes["source_pcp_expert_communication_scope"] == "analytical_global_expert_projection"
                and "native_provider_equivalence" in node.attributes and node.attributes["native_provider_equivalence"] is False))
        and len(pool.ranks) == config.worker_count)


def _symbolic_glm_mtp_data_projection(node, pool):
    """Only the explicit whole-shard BF16-compute MTP DP source declares distributed drafts."""
    config = pool.parallel_plan.config
    return ("source_glm_fp8_target_only" not in node.attributes and HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id)
        and config.data_parallel_size > 1 and config.context_parallel_size == 1
        and config.sequence_parallel_size in ((1, config.tensor_parallel_size) if config.enable_expert_parallel else (1,))
        and pool.parallel_plan.num_layers == 78 and pool.parallel_plan.num_experts == 256
        and _symbolic_glm_pcp_projection(node, pool)
        and node.attributes["model_source_profile"] in ("vllm_0_30_v32_bf16_sync_explicit_allreduce",
            "vllm_0_30_v32_fp8_sync_explicit_allreduce")
        and "source_data_parallel_rank" in node.attributes
        and type(node.attributes["source_data_parallel_rank"]) is int
        and "source_glm_model_sequence_parallel" in node.attributes
        and node.attributes["source_glm_model_sequence_parallel"] is (config.enable_expert_parallel
            and config.tensor_parallel_size > 1 and config.pipeline_parallel_size == 1)
        and "source_glm_moe_sequence_parallel" in node.attributes
        and node.attributes["source_glm_moe_sequence_parallel"] is (config.enable_expert_parallel and config.tensor_parallel_size > 1)
        and node.attributes["source_data_parallel_rank"] == (
            pool.logical_rank_for_physical_rank(node.rank) // (config.tensor_parallel_size * config.pipeline_parallel_size)))


def _symbolic_glm_mtp_last_stage_projection(node, pool):
    """The complete GLM drafter is local to the target's last PP stage."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and (config.is_rank_local or _symbolic_glm_mtp_data_projection(node, pool))
        and pool.parallel_plan.num_layers == 78 and pool.parallel_plan.num_experts == 256
        and set(pool.parallel_plan.layer_to_pipeline_stage) == set(range(config.pipeline_parallel_size))
        and _symbolic_glm_pcp_projection(node, pool)
        and node.rank in pool.physical_ranks_for_pipeline_stage(config.pipeline_parallel_size - 1))


def _validate_glm_mtp_dp_expert_graph(graph, deployment):
    """Bind each MTP DP shard to actual dispatch and sum-scatter ports."""
    for pool in deployment.worker_pools.values():
        if not any(node.pool_name == pool.name and node.variant == "vllm_mtp_eager"
                and _symbolic_glm_mtp_data_projection(node, pool) for node in graph.nodes):
            continue
        nodes = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.kind is WorkloadKind.FUSED_MOE)
        config = pool.parallel_plan.config
        dimension = "expert" if config.enable_expert_parallel else "data"
        predecessors = {node.node_id: [] for node in graph.nodes}
        successors = {node.node_id: [] for node in graph.nodes}
        additions = {}
        for item in graph.nodes:
            if item.kind is WorkloadKind.ELEMENTWISE and "symbolic_tensor_inputs" in item.attributes:
                for port in item.attributes["symbolic_tensor_inputs"]:
                    key = (item.rank, port)
                    if key not in additions:
                        additions[key] = []
                    additions[key].append(item)
        for edge in graph.edges:
            if edge.kind == "data":
                predecessors[edge.destination].append(graph.node(edge.source))
                successors[edge.source].append(graph.node(edge.destination))
        combines, reductions = {}, {}
        for item in graph.nodes:
            if (item.pool_name == pool.name and item.kind is WorkloadKind.COLLECTIVE
                    and item.attributes["operation"] == "reduce_scatterv" and "moe_invocation_id" in item.attributes):
                name = (item.attributes["moe_invocation_id"], item.attributes["participants"])
                if name not in combines:
                    combines[name] = []
                combines[name].append(item)
            if (item.pool_name == pool.name and item.kind is WorkloadKind.COLLECTIVE
                    and item.attributes["operation"] == "all_reduce" and "moe_invocation_id" in item.attributes):
                name = (item.attributes["moe_invocation_id"], item.attributes["participants"])
                if name not in reductions:
                    reductions[name] = []
                reductions[name].append(item)
        invocations, programs = {}, {}
        for node in nodes:
            if not _symbolic_glm_mtp_data_projection(node, pool) or node.attributes["expert_parallel_size"] != config.expert_parallel_size:
                raise ValueError("MTP DP experts require their exact whole-shard BF16 source expert group")
            weight_dtype = ("fp8" if node.attributes["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce" else "bf16")
            if (node.attributes["checkpoint_weight_dtype"] != weight_dtype or node.attributes["routed_weight_dtype"] != weight_dtype
                    or node.attributes["weight_group_size"] != 0
                    or weight_dtype == "fp8" and node.shape["intermediate"] % 128):
                raise ValueError("MTP DP experts must preserve their source weight dtype and whole FP8 scale blocks")
            if weight_dtype == "fp8":
                from src.serving.workload.traced_glm_pcp import glm_fp8_deferred_expert_source_metadata
                expected = glm_fp8_deferred_expert_source_metadata()
                if any(key not in node.attributes or type(node.attributes[key]) is not type(value) or node.attributes[key] != value
                        for key, value in expected.items()):
                    raise ValueError("FP8 MTP DP requires its exact deferred Cutlass BF16 prepare source facts")
            _validate_routing_provenance(node)
            invocation = node.attributes["moe_invocation_id"]
            participants = pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(node.rank))
            key = (invocation, participants)
            if key not in invocations:
                invocations[key] = []
            invocations[key].append(node)
            if invocation not in programs:
                programs[invocation] = []
            programs[invocation].append(node)
            rank_index = participants.index(node.rank)
            rows = node.attributes["source_pcp_expert_rows" if config.enable_expert_parallel else "source_pcp_data_rows"]
            if config.enable_expert_parallel:
                order = ("data_then_context_then_tensor_lane_then_local_row_including_padding" if config.tensor_parallel_size > 1
                    else "data_then_context_then_local_row_including_padding")
                if node.attributes["source_pcp_row_order"] != order:
                    raise ValueError("MTP DP expert row order must retain its actual TP lanes, including internal MoE SP")
            if len(rows) != len(participants) or sum(rows) != node.shape["tokens"]:
                raise ValueError("MTP DP expert rows must conserve the complete target/draft dispatch")
            inputs = node.attributes["symbolic_tensor_inputs"]
            if len(inputs) != 3:
                raise ValueError("MTP DP experts require actual hidden, weights and IDs inputs")
            for port, width, dtype in zip(inputs, (node.shape["hidden"], node.shape["topk"], node.shape["topk"]),
                    ("bf16", "fp32", "int32"), strict=True):
                sources = tuple(item for item in predecessors[node.node_id]
                    if "symbolic_tensor_outputs" in item.attributes and port in item.attributes["symbolic_tensor_outputs"])
                if len(sources) != 1:
                    raise ValueError("MTP DP expert input requires one actual data dispatch producer")
                source, = sources
                if (source.kind is not WorkloadKind.COLLECTIVE or source.attributes["operation"] != "all_gatherv"
                        or source.attributes["participants"] != participants or source.attributes["collective_group"] != dimension
                        or source.attributes["symbolic_variable_rows"] != rows
                        or source.attributes["symbolic_variable_row_shape"] != (width,) or source.dtype != dtype
                        or source.attributes["symbolic_tensor_outputs"][rank_index] != port):
                    raise ValueError("MTP DP expert input must preserve its own gathered rank and physical rows")
            if key not in combines or len(combines[key]) != 1:
                raise ValueError("MTP DP experts require one matching routed-output sum-scatter")
            combine, = combines[key]
            attrs = combine.attributes
            if (attrs["participants"] != participants or attrs["collective_group"] != dimension
                    or attrs["reduction"] != "sum" or attrs["symbolic_variable_rows"] != rows
                    or attrs["symbolic_variable_row_shape"] != (node.shape["hidden"],)
                    or combine.dtype != "bf16" or node not in predecessors[combine.node_id]
                    or attrs["symbolic_tensor_inputs"][rank_index] != node.attributes["symbolic_tensor_outputs"][0]):
                raise ValueError("MTP DP expert output must reach its matching actual sum-scatter input")
        for members in invocations.values():
            expected = pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(members[0].rank))
            if len(members) != len(expected) or {node.rank for node in members} != set(expected):
                raise ValueError("MTP DP invocation must include every actual expert shard, including idle replicas")
            if any(node.shape != members[0].shape or node.attributes["tokens_per_expert"] != members[0].attributes["tokens_per_expert"]
                    for node in members):
                raise ValueError("MTP DP shards must preserve the same global routing and geometry")
            invocation = members[0].attributes["moe_invocation_id"]
            combine, = combines[(invocation, expected)]
            active = tuple(rank for rank, count in zip(expected, combine.attributes["symbolic_variable_rows"], strict=True) if count)
            outputs = combine.attributes["symbolic_tensor_outputs"]
            if len(outputs) != len(active):
                raise ValueError("MTP DP sum-scatter must return one output for each active request replica")
            for rank, port in zip(active, outputs, strict=True):
                key = (rank, port)
                if key not in additions or len(additions[key]) != 1 or combine not in predecessors[additions[key][0].node_id]:
                    raise ValueError("MTP DP shared/routed add must consume its actual sum-scatter data output")
                if pool.parallel_plan.config.tensor_parallel_size == 1:
                    continue
                add = additions[key][0]
                tensor = pool.physical_rank_group("tensor", pool.logical_rank_for_physical_rank(rank))
                if config.enable_expert_parallel and config.pipeline_parallel_size == 1:
                    # Model SP keeps each combined shard local through the
                    # next residual norm, then gathers it for attention or
                    # the final head. There is no source late TP sum here.
                    count = combine.attributes["symbolic_variable_rows"][expected.index(rank)]
                    norms = tuple(item for item in successors[add.node_id]
                        if item.kind is WorkloadKind.NORM and item.variant == "add_rmsnorm" and item.rank == rank
                        and item.shape == {"rows": count, "hidden": members[0].shape["hidden"]})
                    if len(norms) != 1:
                        raise ValueError("MTP model SP result requires its actual local residual norm data input")
                    norm, = norms
                    gathers = tuple(item for item in successors[norm.node_id] if item.kind is WorkloadKind.COLLECTIVE
                        and item.attributes["operation"] == "all_gather" and item.attributes["participants"] == tensor)
                    if len(gathers) != 1:
                        raise ValueError("MTP model SP norm requires its actual complete TP gather data input")
                    gather, = gathers
                    index = tensor.index(rank)
                    attrs = gather.attributes
                    if (gather.dtype != "bf16" or attrs["symbolic_collective_dim"] != 0
                            or attrs["symbolic_collective_storage"] != "fresh"
                            or attrs["symbolic_collective_input_shape"] != (count, members[0].shape["hidden"])
                            or len(attrs["symbolic_collective_input_producers"]) != len(tensor)
                            or attrs["symbolic_collective_input_producers"][index] != norm.node_id
                            or attrs["symbolic_tensor_inputs"][index] not in norm.attributes["symbolic_tensor_outputs"]):
                        raise ValueError("MTP model SP gather must preserve its own normalized TP shard and rows")
                    continue
                producer = add
                if config.enable_expert_parallel:
                    restores = tuple(item for item in graph.nodes if item.kind is WorkloadKind.COLLECTIVE
                        and item.attributes["operation"] == "all_gather" and item.attributes["participants"] == tensor
                        and "symbolic_glm_moe_sp_restore" in item.attributes and add in predecessors[item.node_id])
                    if len(restores) != 1:
                        raise ValueError("MTP DP MoE sequence shards require their actual complete TP restore")
                    producer, = restores
                reduction_key = (invocation, tensor)
                if reduction_key in reductions:
                    local = reductions[reduction_key]
                    if len(local) != 1 or producer not in predecessors[local[0].node_id]:
                        raise ValueError("MTP DP routed/shared result requires its actual late TP reduction data input")
                else:
                    stage = pool.logical_rank_for_physical_rank(rank) // len(tensor) % pool.parallel_plan.config.pipeline_parallel_size
                    layer = members[0].attributes["routing_provenance"]["layer_index"]
                    final = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
                    messages = tuple(item for item in graph.nodes if item.kind is WorkloadKind.P2P and item.rank == rank
                        and item.attributes["payload_tensor"]["name"] == "hidden_states" and producer in predecessors[item.node_id])
                    member = next(item for item in members if item.rank==rank)
                    replica = pool.logical_rank_for_physical_rank(rank)//(config.tensor_parallel_size*config.pipeline_parallel_size)
                    if (layer==final and stage+1<config.pipeline_parallel_size
                            and "source_dummy_run" in member.attributes and member.attributes["source_dummy_run"] is True
                            and member.attributes["source_actual_request_count"]==0
                            and member.attributes["source_pcp_request_rows"][replica]==0):
                        # Legacy eager dummy discards this non-last stage's
                        # actual partial result locally; it does not send PP.
                        if any(item.kind is WorkloadKind.P2P and item.rank==rank for item in graph.nodes):
                            raise ValueError("MTP dummy stage result cannot introduce a request PP message")
                        continue
                    if layer != final or stage + 1 >= pool.parallel_plan.config.pipeline_parallel_size or len(messages) != 1:
                        raise ValueError("MTP DP expert result requires its late TP reduction or actual target PP frontier")
        for members in programs.values():
            logical = pool.logical_rank_for_physical_rank(members[0].rank)
            stage = logical // pool.parallel_plan.config.tensor_parallel_size % pool.parallel_plan.config.pipeline_parallel_size
            expected = pool.physical_ranks_for_pipeline_stage(stage)
            if len(members) != len(expected) or {node.rank for node in members} != set(expected):
                raise ValueError("MTP TP/DP invocation requires every tensor/data expert shard")
            if any(node.shape != members[0].shape or node.attributes["tokens_per_expert"] != members[0].attributes["tokens_per_expert"] for node in members):
                raise ValueError("MTP TP/DP shards must preserve global routing and intermediate geometry")


def _validate_glm_rank_local_fp8_expert_node(node, pool):
    from src.kernel.kernel_spec import linear_expert_partition
    from src.serving.workload.traced_glm_mtp import glm_fp8_rank_local_expert_source_metadata
    expected = glm_fp8_rank_local_expert_source_metadata()
    if {key for key in node.attributes if key.startswith("source_fp8_")} != set(expected):
        raise ValueError("rank-local FP8 MTP EP requires its complete NoDPEP deferred Cutlass source fields")
    if any(key not in node.attributes or type(node.attributes[key]) is not type(value) or node.attributes[key] != value
            for key, value in expected.items()):
        raise ValueError("rank-local FP8 MTP EP requires exact NoDPEP deferred Cutlass source facts")
    config, attrs = pool.parallel_plan.config, node.attributes
    participants = pool.physical_rank_group("tensor", pool.logical_rank_for_physical_rank(node.rank))
    rank = participants.index(node.rank)
    owned = linear_expert_partition(256, config.expert_parallel_size, rank)
    counts = attrs["tokens_per_expert"]
    layer = attrs["routing_provenance"]["layer_index"]
    expected_stage = config.pipeline_parallel_size - 1 if layer is None else pool.parallel_plan.layer_to_pipeline_stage[layer]
    actual_stage = pool.logical_rank_for_physical_rank(node.rank) // config.tensor_parallel_size
    if actual_stage != expected_stage:
        raise ValueError("rank-local FP8 target/draft expert must remain in its actual source pipeline stage")
    if (node.dtype != "bf16" or tuple(node.shape[key] for key in ("experts", "topk", "hidden", "intermediate")) != (256, 8, 6144, 2048)
            or config.expert_parallel_size != config.tensor_parallel_size or 256 % config.expert_parallel_size
            or attrs["checkpoint_weight_dtype"] != "fp8" or attrs["routed_weight_dtype"] != "fp8"
            or any(type(value) is not int for value in (*attrs["tokens_per_expert"], *attrs["local_tokens_per_expert"], *attrs["owned_experts"]))
            or type(attrs["expert_rank"]) is not int or type(attrs["expert_parallel_size"]) is not int
            or attrs["weight_group_size"] != 0 or attrs["routed_scaling_factor"] != 2.5
            or attrs["output_is_reduced"] is not False or attrs["expert_rank"] != rank
            or attrs["expert_parallel_size"] != config.expert_parallel_size or tuple(attrs["owned_experts"]) != owned
            or tuple(attrs["local_tokens_per_expert"]) != tuple(count if expert in owned else 0 for expert, count in enumerate(counts))):
        raise ValueError("rank-local FP8 MTP EP must conserve exact dimensions and stage-local linear expert ownership")


def _validate_glm_tensor_expert_graph(graph, pool, nodes):
    """Ordinary TP expert contributions reach the source's late TP sum."""
    # These indices are local to this complete validation. Keep only data
    # edges, preserving duplicate producers and edge order, so control edges
    # cannot replace a missing expert contribution or PP frontier.
    data_successors = {item.node_id: [] for item in graph.nodes}
    data_predecessors = {item.node_id: [] for item in graph.nodes}
    for edge in graph.edges:
        if edge.kind == "data":
            data_successors[edge.source].append(edge.destination)
            data_predecessors[edge.destination].append(edge.source)
    all_reductions = tuple(item for item in graph.nodes if item.kind is WorkloadKind.COLLECTIVE
        and item.attributes["operation"] == "all_reduce" and "moe_invocation_id" in item.attributes)
    grouped = {}
    for node in nodes:
        _validate_routing_provenance(node)
        _validate_virtual_ep_router_inputs(graph, node)
        invocation = node.attributes["moe_invocation_id"]
        if invocation not in grouped:
            grouped[invocation] = []
        grouped[invocation].append(node)
        reductions = tuple(item for item in all_reductions if item.attributes["moe_invocation_id"] == invocation)
        participants = pool.physical_rank_group("tensor", pool.logical_rank_for_physical_rank(node.rank))
        local_reductions = tuple(item for item in reductions if item.attributes["participants"] == participants)
        adds = {name for name in data_successors[node.node_id] if graph.node(name).kind is WorkloadKind.ELEMENTWISE}
        stage = (pool.logical_rank_for_physical_rank(node.rank) // pool.parallel_plan.config.tensor_parallel_size
            % pool.parallel_plan.config.pipeline_parallel_size)
        final_layer = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
        if (not local_reductions and stage + 1 < pool.parallel_plan.config.pipeline_parallel_size
                and node.attributes["routing_provenance"]["layer_index"] == final_layer):
            messages = tuple(graph.node(name) for source in adds for name in data_successors[source]
                if graph.node(name).kind is WorkloadKind.P2P and graph.node(name).rank == node.rank
                and graph.node(name).attributes["payload_tensor"]["name"] == "hidden_states")
            if (len(messages) != 1 or messages[0].attributes["source_stage"] != stage
                    or messages[0].attributes["destination_stage"] != stage + 1):
                raise ValueError("GLM TP experts require their matching TP-group reduction or actual next-stage frontier")
            producers = tuple(graph.node(name) for name in data_predecessors[messages[0].node_id])
            if len(producers) != 1:
                raise ValueError("GLM TP pipeline requires one actual final-layer source producer")
            _validate_glm_pipeline_frontier(graph, pool, producers[0], participants, stage,
                node.shape["tokens"], node.shape["hidden"], source_rank=node.rank)
            continue
        if len(local_reductions) != 1 or len(reductions) != 1:
            raise ValueError("GLM TP experts require their matching TP-group late tensor reduction")
        if not any(name in adds for name in data_predecessors[local_reductions[0].node_id]):
            raise ValueError("GLM TP expert contributions must complete before the shared/routed tensor sum")
    for members in grouped.values():
        participants = pool.physical_rank_group("tensor", pool.logical_rank_for_physical_rank(members[0].rank))
        if len(members) != len(participants) or {node.rank for node in members} != set(participants):
            raise ValueError("GLM TP expert invocation requires every physical expert rank")
        if any(node.shape != members[0].shape or node.attributes["tokens_per_expert"] != members[0].attributes["tokens_per_expert"]
                for node in members):
            raise ValueError("GLM TP expert ranks must preserve global routing and geometry")


def _validate_glm_pcp_expert_graph(graph, pool, nodes):
    """Bind PCP's actual modular dispatch ports and expert participants."""
    config = pool.parallel_plan.config
    by_invocation = {}
    checked_dispatches = set()
    for node in nodes:
        fp8 = ((_symbolic_glm_mtp_data_projection(node, pool) or _symbolic_glm_fp8_target_dp_projection(node, pool))
            and node.attributes["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce")
        expected_dtype = "fp8" if fp8 else "bf16"
        if fp8:
            from src.serving.workload.traced_glm_pcp import glm_fp8_deferred_expert_source_metadata
            if (any(key not in node.attributes or type(node.attributes[key]) is not type(value) or node.attributes[key] != value
                    for key, value in glm_fp8_deferred_expert_source_metadata().items())
                    or tuple(node.shape[key] for key in ("experts", "topk", "hidden", "intermediate")) != (256, 8, 6144, 2048)):
                raise ValueError("GLM FP8 EP requires exact deferred Cutlass source dimensions and prepare facts")
        if (node.attributes["source_component"] != "symbolic_grouped_routed_experts"
                or node.attributes["routing_source_stage"] != "dsv32_moe_route"
                or node.attributes["checkpoint_weight_dtype"] != expected_dtype
                or node.attributes["routed_weight_dtype"] != expected_dtype):
            raise ValueError("GLM PCP EP requires its explicit source weight dtype and modular expert source")
        _validate_routing_provenance(node)
        invocation = node.attributes["moe_invocation_id"]
        if invocation not in by_invocation:
            by_invocation[invocation] = []
        by_invocation[invocation].append(node)
        inputs = node.attributes["symbolic_tensor_inputs"]
        if len(inputs) != 3:
            raise ValueError("GLM PCP EP requires hidden, routing weights and expert IDs")
        predecessors = tuple(graph.node(edge.source) for edge in graph.edges
            if edge.destination == node.node_id and edge.kind == "data")
        dimension = "expert" if config.data_parallel_size > 1 else "data_context"
        expected_group = pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(node.rank))
        rank_index = expected_group.index(node.rank)
        dispatch_rows = None
        dispatches = []
        for port, width, dtype in zip(inputs, (node.shape["hidden"], node.shape["topk"], node.shape["topk"]),
                ("bf16", "fp32", "int32"), strict=True):
            producers = tuple(item for item in predecessors if "symbolic_tensor_outputs" in item.attributes
                and port in item.attributes["symbolic_tensor_outputs"])
            if (len(producers) != 1 or producers[0].kind is not WorkloadKind.COLLECTIVE
                    or producers[0].attributes["operation"] != "all_gatherv"
                    or producers[0].attributes["collective_group"] != dimension
                    or producers[0].attributes["participants"] != expected_group
                    or producers[0].attributes["symbolic_variable_row_shape"] != (width,)
                    or producers[0].dtype != dtype):
                raise ValueError("GLM PCP EP must consume its actual hidden/weights/IDs context gathers")
            gather = producers[0].attributes
            rows = gather["symbolic_variable_rows"]
            if (not isinstance(rows, tuple) or len(rows) != len(expected_group)
                    or any(type(value) is not int or value < 0 for value in rows)
                    or sum(rows) != node.shape["tokens"]
                    or dispatch_rows is not None and rows != dispatch_rows
                    or len(gather["symbolic_tensor_outputs"]) != len(expected_group)
                    or gather["symbolic_tensor_outputs"][rank_index] != port):
                raise ValueError("GLM PCP EP dispatch partitions and rank output ports must match its actual expert rows")
            dispatch_rows = rows
            dispatches.append(producers[0])
        dispatch_key = tuple(item.node_id for item in dispatches)
        if dispatch_key not in checked_dispatches:
            hidden, weights, ids = dispatches
            active = tuple((rank, count) for rank, count in zip(expected_group, dispatch_rows, strict=True) if count)
            if any(len(item.attributes["symbolic_tensor_inputs"]) != len(active)
                    or len(item.attributes["symbolic_collective_input_producers"]) != len(active) for item in dispatches):
                raise ValueError("GLM PCP EP dispatch must retain every active source payload")
            for index, (rank, count) in enumerate(active):
                weight_producer = weights.attributes["symbolic_collective_input_producers"][index]
                id_producer = ids.attributes["symbolic_collective_input_producers"][index]
                if weight_producer != id_producer:
                    raise ValueError("GLM PCP EP weights and IDs must come from the same actual route")
                route = graph.node(weight_producer)
                if (route.kind is not WorkloadKind.VECTOR or route.variant != "dsv32_moe_route"
                        or route.rank != rank or route.pool_name != node.pool_name
                        or route.shape != {"elements": count * node.shape["experts"], "rows": count,
                            "experts": node.shape["experts"], "topk": node.shape["topk"]}
                        or route.attributes["symbolic_tensor_outputs"] != (
                            weights.attributes["symbolic_tensor_inputs"][index], ids.attributes["symbolic_tensor_inputs"][index])):
                    raise ValueError("GLM PCP EP routing payloads must preserve their actual source rank and row extent")
                logits = route.attributes["symbolic_tensor_inputs"][0]
                gates = tuple(graph.node(name) for name in graph.predecessors(route.node_id)
                    if graph.node(name).kind is WorkloadKind.ROUTER
                    and logits in graph.node(name).attributes["symbolic_tensor_outputs"])
                if (len(gates) != 1 or gates[0].rank != rank
                        or gates[0].shape != {"b": 1, "m": count, "n": node.shape["experts"], "k": node.shape["hidden"]}
                        or _source_linear_data_inputs(gates[0]) != (hidden.attributes["symbolic_tensor_inputs"][index],)
                        or not any(edge.source == hidden.attributes["symbolic_collective_input_producers"][index]
                            and edge.destination == gates[0].node_id and edge.kind == "data" for edge in graph.edges)):
                    raise ValueError("GLM PCP EP gate and hidden dispatch must share the actual rank-local payload producer")
            checked_dispatches.add(dispatch_key)
        combines = tuple(item for item in graph.nodes if item.kind is WorkloadKind.COLLECTIVE
            and item.attributes["operation"] == "reduce_scatterv"
            and "moe_invocation_id" in item.attributes and item.attributes["moe_invocation_id"] == invocation
            and item.attributes["participants"] == expected_group)
        if len(combines) != 1 or not any(edge.source == node.node_id and edge.destination == combines[0].node_id
                and edge.kind == "data" for edge in graph.edges):
            raise ValueError("GLM PCP EP requires its matching routed-output sum-scatter dependency")
        combine = combines[0]
        attrs = combine.attributes
        if (attrs["collective_group"] != dimension or attrs["reduction"] != "sum"
                or attrs["symbolic_variable_rows"] != dispatch_rows
                or attrs["symbolic_variable_row_shape"] != (node.shape["hidden"],)
                or combine.dtype != "bf16"
                or len(node.attributes["symbolic_tensor_outputs"]) != 1
                or len(attrs["symbolic_tensor_inputs"]) != len(expected_group)
                or attrs["symbolic_tensor_inputs"][rank_index] != node.attributes["symbolic_tensor_outputs"][0]):
            raise ValueError("GLM PCP EP combine partitions and rank input ports must match its actual expert output")
    for members in by_invocation.values():
        participants = pool.physical_rank_group("expert", pool.logical_rank_for_physical_rank(members[0].rank))
        if {node.rank for node in members} != set(participants) or len(members) != len(participants):
            raise ValueError("GLM PCP EP must execute each expert rank once per invocation")
        first = members[0]
        if any(node.attributes["tokens_per_expert"] != first.attributes["tokens_per_expert"]
                or node.shape != first.shape for node in members):
            raise ValueError("GLM PCP EP ranks must preserve the same global routing and expert geometry")


def _symbolic_glm_pipeline_projection(node, pool):
    """Only admit the declared original GLM TP-partial PP source program."""
    config = pool.parallel_plan.config
    h200_pp2 = (HWSpec.source_contract_supported("glm_tp8_pp2_ep1_source_projection", pool.hardware_id)
        and len(pool.ranks) == 16
        and config.pipeline_parallel_size == 2 and not config.enable_expert_parallel
        and pool.parallel_plan.layer_to_pipeline_stage == (0,) * 38 + (1,) * 40)
    a800_pp4 = (HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", pool.hardware_id)
        and len(pool.ranks) == 32
        and config.pipeline_parallel_size == 4 and config.enable_expert_parallel
        and config.expert_parallel_size == 8
        and pool.parallel_plan.layer_to_pipeline_stage == (0,) * 18 + (1,) * 20 + (2,) * 20 + (3,) * 20)
    return ("source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and config.tensor_parallel_size == 8 and config.is_rank_local and (h200_pp2 or a800_pp4)
        and all(value == 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))


def _symbolic_v3_parallel_projection(node, pool):
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.is_rank_local and not config.enable_expert_parallel
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_27_1"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_27_1_deepseek_v3_sync_fp8")


def _symbolic_v3_dp_pipeline_projection(node, pool):
    """Versioned distributed PP topology; operator pricing stays separate."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.context_parallel_size == config.sequence_parallel_size == 1
        and config.data_parallel_size > 1 and config.pipeline_parallel_size > 1
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_27_1"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_27_1_deepseek_v3_sync_fp8")


def _validate_kimi_dp_expert_graph(graph, deployment):
    """Bind every BF16 dispatch, linear INT4 owner and actual returned rows.

    Routing includes the declared physical padding rows. This source check
    does not select a GPU provider, assign a price or accept normal memory.
    """
    parents = {node.node_id: [] for node in graph.nodes}
    children = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.kind == "data":
            parents[edge.destination].append(graph.node(edge.source))
            children[edge.source].append(graph.node(edge.destination))
    for pool in deployment.worker_pools.values():
        nodes = tuple(node for node in graph.nodes if node.pool_name == pool.name)
        if not any(_symbolic_kimi_dp_projection(node, pool) for node in nodes):
            continue
        config = pool.parallel_plan.config
        tp, dp, pp = config.tensor_parallel_size, config.data_parallel_size, config.pipeline_parallel_size
        ep, sp = config.expert_parallel_size, config.enable_expert_parallel and tp > 1
        expert_tp = 1 if config.enable_expert_parallel else tp * dp
        if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                or 2048 % expert_tp or 2048 // expert_tp % 32 or 384 % ep
                or ep != (tp * dp if config.enable_expert_parallel else 1)):
            raise ValueError("Kimi DP requires whole INT4/group32 shards or uniform linear global EP")
        logical_rows = []
        for replica in range(dp):
            ranks = tuple(pool.physical_rank_for_logical_rank(replica * pp * tp + lane) for lane in range(tp))
            norms = tuple(node for node in nodes if node.rank in ranks and node.kind is WorkloadKind.NORM
                and node.node_id.endswith(":layer-0:input-norm"))
            if norms and (len(norms) != tp or {node.rank for node in norms} != set(ranks)
                    or len({node.shape["rows"] for node in norms}) != 1):
                raise ValueError("Kimi DP must preserve each original replica's actual input rows")
            logical_rows.append(norms[0].shape["rows"] if norms else 0)
        rows = tuple(-(-count // tp) for count in logical_rows for _ in range(tp)) if sp else tuple(logical_rows)
        physical_rows = sum(rows)
        experts = tuple(node for node in nodes if node.kind is WorkloadKind.FUSED_MOE)
        # Both named checkpoints have one dense prefix and sixty MoE layers.
        if len(pool.parallel_plan.layer_to_pipeline_stage) != 61 or len(experts) != 60 * tp * dp:
            raise ValueError("Kimi DP requires all sixty source MoE layers and every expert participant")
        for layer in range(1, 61):
            calls = tuple(node for node in experts if node.attributes["routing_provenance"]["layer_index"] == layer)
            stage = pool.parallel_plan.layer_to_pipeline_stage[layer]
            stage_ranks = tuple(pool.physical_rank_for_logical_rank((replica * pp + stage) * tp + lane)
                for replica in range(dp) for lane in range(tp))
            if len(calls) != tp * dp or {node.rank for node in calls} != set(stage_ranks):
                raise ValueError("Kimi DP expert layer must cover its actual source-stage ranks")
            counts = calls[0].attributes["tokens_per_expert"]
            invocation = calls[0].attributes["moe_invocation_id"]
            for expert in calls:
                attrs = expert.attributes
                logical = pool.logical_rank_for_physical_rank(expert.rank)
                replica, lane = logical // (pp * tp), logical % tp
                expert_rank = replica * tp + lane if ep > 1 else 0
                from src.kernel.kernel_spec import linear_expert_partition
                owned = linear_expert_partition(384, ep, expert_rank)
                if (not _symbolic_kimi_dp_projection(expert, pool)
                        or expert.shape != {"tokens": physical_rows, "experts": 384, "topk": 8, "hidden": 7168,
                            "intermediate": 2048 // expert_tp}
                        or expert.dtype != "bf16" or attrs["routed_weight_dtype"] != "int4"
                        or attrs["checkpoint_weight_dtype"] != "int4" or attrs["weight_group_size"] != 32
                        or attrs["routing_source_stage"] != "kimi_moe_route" or attrs["routed_scaling_factor"] != 1.0
                        or attrs["output_is_reduced"] is not False or attrs["expert_parallel_size"] != ep
                        or attrs["expert_rank"] != expert_rank or attrs["owned_experts"] != owned
                        or attrs["tokens_per_expert"] != counts or attrs["moe_invocation_id"] != invocation
                        or attrs["local_tokens_per_expert"] != tuple(count if index in owned else 0 for index, count in enumerate(counts))):
                    raise ValueError("Kimi DP must preserve exact dimensions, INT4/group32, routing and global expert ownership")
                _validate_routing_provenance(expert)
                dimension = "expert" if sp else "data"
                participants = pool.physical_rank_group(dimension, logical)
                dispatched = []
                for port, role, width, dtype, item_bytes in zip(attrs["symbolic_tensor_inputs"],
                        ("input", "weights", "ids"), (7168, 8, 8), ("bf16", "fp32", "int32"), (2, 4, 4), strict=True):
                    matches = tuple(node for node in parents[expert.node_id] if node.node_id.endswith(":dispatch-" + role))
                    if len(matches) != 1:
                        raise ValueError("Kimi DP expert requires each actual dispatch data edge")
                    dispatch, = matches
                    da = dispatch.attributes
                    if (dispatch.kind is not WorkloadKind.COLLECTIVE or dispatch.dtype != dtype
                            or da["operation"] != "all_gatherv" or da["participants"] != participants
                            or da["collective_group"] != dimension or da["symbolic_variable_rows"] != rows
                            or da["symbolic_variable_row_shape"] != (width,) or da["bytes"] != physical_rows * width * item_bytes
                            or port != da["symbolic_tensor_outputs"][participants.index(expert.rank)]):
                        raise ValueError("Kimi DP dispatch must bind exact BF16/weight/ID rows, bytes and participant ports")
                    dispatched.append(dispatch)
                hidden_gather, weights_gather, ids_gather = dispatched
                active = tuple((rank, count) for rank, count in zip(participants, rows, strict=True) if count)
                routes = tuple(node for node in parents[weights_gather.node_id] if node.variant == "kimi_moe_route")
                if len(routes) != len(active) or {node.node_id for node in parents[ids_gather.node_id]} != {node.node_id for node in routes}:
                    raise ValueError("Kimi DP must route each nonempty local shard before dispatch")
                for index, (rank, count) in enumerate(active):
                    route_matches = tuple(node for node in routes if node.rank == rank)
                    if len(route_matches) != 1:
                        raise ValueError("Kimi DP router must retain the actual nonempty source rank")
                    route, = route_matches
                    gates = tuple(node for node in parents[route.node_id] if node.kind is WorkloadKind.ROUTER)
                    if (len(gates) != 1 or gates[0].dtype != "bf16" or gates[0].rank != rank
                            or gates[0].shape != {"b": 1, "m": count, "n": 384, "k": 7168}
                            or _source_linear_data_inputs(gates[0]) != (hidden_gather.attributes["symbolic_tensor_inputs"][index],)
                            or route.attributes["symbolic_tensor_outputs"] != (weights_gather.attributes["symbolic_tensor_inputs"][index],
                                ids_gather.attributes["symbolic_tensor_inputs"][index])):
                        raise ValueError("Kimi DP routing must consume its matching local BF16 gate/input and dispatch actual outputs")
                combines = tuple(node for node in children[expert.node_id] if node.kind is WorkloadKind.COLLECTIVE)
                if len(combines) != 1:
                    raise ValueError("Kimi DP expert requires one actual sum-scatter data edge")
                combine, = combines
                ca = combine.attributes
                contributions = tuple(node for node in parents[combine.node_id] if node.kind is WorkloadKind.FUSED_MOE)
                if (ca["operation"] != "reduce_scatterv" or ca["reduction"] != "sum"
                        or ca["participants"] != participants or ca["collective_group"] != dimension
                        or ca["symbolic_variable_rows"] != rows or ca["symbolic_variable_row_shape"] != (7168,)
                        or ca["bytes"] != physical_rows * 7168 * 2 or ca["moe_invocation_id"] != invocation
                        or len(contributions) != len(participants) or {node.rank for node in contributions} != set(participants)
                        or ca["symbolic_tensor_inputs"] != tuple(next(node for node in contributions if node.rank == rank)
                            .attributes["symbolic_tensor_outputs"][0] for rank in participants)):
                    raise ValueError("Kimi DP combine must retain every physical expert contribution including idle owners")
                own_rows = -(-logical_rows[replica] // tp) if sp else logical_rows[replica]
                if not own_rows:
                    continue
                copies = tuple(node for node in children[combine.node_id] if node.rank == expert.rank
                    and node.node_id.endswith(f":layer-{layer}:mlp:finalize-copy"))
                if len(copies) != 1:
                    raise ValueError("Kimi DP returned rows require a same-rank finalize copy")
                copy, = copies
                if (copy.attributes["symbolic_tensor_inputs"] != (ca["symbolic_tensor_outputs"][[rank for rank, _ in active].index(expert.rank)],)
                        or copy.shape != {"elements": own_rows * 7168, "rows": own_rows, "hidden": 7168}):
                    raise ValueError("Kimi DP finalize must consume only its original local returned rows")
                scales = tuple(node for node in children[copy.node_id] if node.node_id.endswith(":output-scale"))
                if (len(scales) != 1 or scales[0].attributes["scalar_value"] != 2.827
                        or scales[0].kind is not WorkloadKind.ELEMENTWISE or scales[0].rank != expert.rank
                        or scales[0].dtype != "bf16" or scales[0].shape != {"elements": own_rows * 7168}
                        or scales[0].attributes["symbolic_tensor_inputs"] != copy.attributes["symbolic_tensor_outputs"]):
                    raise ValueError("Kimi DP must preserve routed output scaling after finalize")
                scale, = scales
                additions = tuple(node for node in children[scale.node_id] if node.node_id.endswith(":shared-add"))
                if len(additions) != 1:
                    raise ValueError("Kimi DP scale must feed its real shared-add")
                addition, = additions
                shared = tuple(node for node in parents[addition.node_id] if node.kind is WorkloadKind.LINEAR)
                if (addition.rank != expert.rank or addition.shape != {"elements": own_rows * 7168}
                        or len(shared) != 1 or shared[0].rank != expert.rank
                        or shared[0].shape != {"b": 1, "m": own_rows, "n": 7168, "k": 2048 // (1 if sp else tp)}
                        or addition.attributes["symbolic_tensor_inputs"] != (shared[0].attributes["symbolic_tensor_outputs"][0],
                            scale.attributes["symbolic_tensor_outputs"][0])):
                    raise ValueError("Kimi DP shared weights and local routed output must retain the actual TP/SP sum")
                if tp > 1 and not (sp and pp == 1):
                    restores = tuple(node for node in children[addition.node_id] if node.kind is WorkloadKind.COLLECTIVE)
                    if len(restores) != 1:
                        raise ValueError("Kimi DP shared-add requires its actual late TP sum or sequence restore")
                    restore, = restores
                    ra = restore.attributes
                    tensor_ranks = pool.physical_rank_group("tensor", logical)
                    contributors = tuple(node for node in parents[restore.node_id] if node.node_id.endswith(f":layer-{layer}:mlp:shared-add"))
                    if (ra["operation"] != ("all_gather" if sp else "all_reduce") or ra["participants"] != tensor_ranks
                            or ra["bytes"] != own_rows * 7168 * 2
                            or {node.rank for node in contributors} != set(tensor_ranks)
                            or ra["symbolic_tensor_inputs"] != tuple(next(node for node in contributors if node.rank == rank)
                                .attributes["symbolic_tensor_outputs"][0] for rank in tensor_ranks)):
                        raise ValueError("Kimi DP late TP operation must consume all actual local shared-add ports")



def _symbolic_kimi_dp_projection(node, pool):
    """Versioned K2 DP program marker; exact expert ports checked graph-wide."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.data_parallel_size > 1
        and config.context_parallel_size == config.sequence_parallel_size == 1
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_30_k2_6_int4_sync_explicit_allreduce")


def _symbolic_mla_dp_pipeline_projection(node, pool):
    return (_symbolic_v3_dp_pipeline_projection(node, pool)
        or pool.parallel_plan.config.pipeline_parallel_size > 1 and _symbolic_kimi_dp_projection(node, pool))


def _symbolic_kimi_parallel_projection(node, pool):
    """H200 K2 source marker, with stage-local TP/EP and nonempty layers."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.is_rank_local
        and len(pool.ranks) == config.worker_count
        and all(pool.parallel_plan.layer_to_pipeline_stage.count(stage) > 0
            for stage in range(config.pipeline_parallel_size))
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_30_k2_6_int4_sync_explicit_allreduce")


def _symbolic_kimi_pipeline_projection(node, pool):
    config = pool.parallel_plan.config
    h200 = config.pipeline_parallel_size > 1 and _symbolic_kimi_parallel_projection(node, pool)
    return ("source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and (h200 or HWSpec.source_contract_supported("kimi_tp8_pp2_ep8_source_projection", pool.hardware_id)
        and len(pool.ranks) == 16
        and config.tensor_parallel_size == config.expert_parallel_size == 8
        and config.pipeline_parallel_size == 2 and config.enable_expert_parallel and config.is_rank_local
        and pool.parallel_plan.layer_to_pipeline_stage == (0,) * 31 + (1,) * 30
        and all(value == 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))))


def _symbolic_kimi_single_rank_projection(node, pool):
    config = pool.parallel_plan.config
    return ("source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
        and HWSpec.source_contract_supported("kimi_tp1_source_projection", pool.hardware_id)
        and len(pool.ranks) == 1
        and config.is_rank_local and not config.enable_expert_parallel
        and all(value == 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))


def _symbolic_k3_pipeline_projection(node, pool):
    from src.serving.parallelism.pp import pipeline_layer_ranges
    config = pool.parallel_plan.config
    pp = config.pipeline_parallel_size
    if not 1 < pp <= pool.parallel_plan.num_layers or pool.parallel_plan.num_layers != 93:
        return False
    expected_layers = tuple(stage for stage, (start, end) in enumerate(
        pipeline_layer_ranges(pool.parallel_plan.num_layers, pp)) for _ in range(start, end))
    return ("source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and HWSpec.source_contract_supported("kimi_k3_pp16_source_projection", pool.hardware_id)
        and len(pool.ranks) == pp
        and config.tensor_parallel_size == 1
        and not config.enable_expert_parallel and config.is_rank_local
        and pool.parallel_plan.layer_to_pipeline_stage == expected_layers
        and all(value == 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))


def _symbolic_k3_pipeline_topology_projection(node, pool):
    """K3 PP transport identity, independent of operator pricing scope."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and pool.parallel_plan.num_layers == 93
        and 1 < config.pipeline_parallel_size <= pool.parallel_plan.num_layers
        and config.context_parallel_size == config.sequence_parallel_size == 1
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes and node.attributes["model_source_profile"] == "vllm_0_30_k3_mxfp4_sync")


def _is_k3_source_pipeline_graph(graph, pool):
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and pool.parallel_plan.num_layers == 93
        and pool.parallel_plan.config.pipeline_parallel_size > 1
        and any(node.pool_name == pool.name and node.variant == "kimi_k3_attn_res"
            and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
            and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
            for node in graph.nodes))


def _symbolic_v4_parallel_projection(node, pool):
    config = pool.parallel_plan.config
    # The two pinned V4 targets have8/16 output groups. The model boundary
    # checks its own exact group count and whole quantized intermediate.
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.is_rank_local and not config.enable_expert_parallel
        and 16 % config.tensor_parallel_size == 0 and 1 <= config.pipeline_parallel_size <= pool.parallel_plan.num_layers
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_30_v4_sync_fp8_ds_mla")


def _symbolic_v4_pipeline_projection(node, pool):
    return pool.parallel_plan.config.pipeline_parallel_size > 1 and _symbolic_v4_parallel_projection(node, pool)


def _symbolic_v4_pipeline_topology_projection(node, pool):
    """Versioned V4 PP geometry, independent of operator pricing support."""
    config = pool.parallel_plan.config
    return (HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id) and config.context_parallel_size == config.sequence_parallel_size == 1
        and 16 % config.tensor_parallel_size == 0 and 1 < config.pipeline_parallel_size <= pool.parallel_plan.num_layers
        and len(pool.ranks) == config.worker_count
        and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True
        and "model_execution_version" in node.attributes and node.attributes["model_execution_version"] == "vllm_0_30_0"
        and "model_source_profile" in node.attributes
        and node.attributes["model_source_profile"] == "vllm_0_30_v4_sync_fp8_ds_mla")


def _symbolic_mla_pipeline_projection(node, pool):
    return (_symbolic_glm_pipeline_projection(node, pool) or _symbolic_kimi_pipeline_projection(node, pool)
        or pool.parallel_plan.config.pipeline_parallel_size > 1 and _symbolic_glm_pcp_projection(node, pool)
        or _symbolic_k3_pipeline_projection(node, pool)
        or pool.parallel_plan.config.pipeline_parallel_size > 1 and _symbolic_v3_parallel_projection(node, pool))


def _symbolic_pipeline_endpoint(node, pool, stage):
    config = pool.parallel_plan.config
    return (config.pipeline_parallel_size > 1 and (config.is_rank_local or _qwen030_dp_sp(config) or _symbolic_kimi_dp_projection(node, pool))
        and node.rank in pool.physical_ranks_for_pipeline_stage(stage))


def _pipeline_worker_copy_consumer(graph, copy, input_port, rank, input_allocation, k3_projection, *, dtype="bf16"):
    """Follow one typed receive/gather through its source runner copy_."""
    if dtype not in ("bf16", "fp16"):
        raise ValueError("pipeline worker copy requires its declared 16-bit dtype")
    required = {"index_count", "gather_factor", "scattered", "materialized_output",
        "symbolic_tensor_inputs", "symbolic_tensor_outputs"}
    if (required - set(copy.attributes) or copy.rank != rank or copy.kind is not WorkloadKind.GATHER or copy.variant != "gather" or copy.dtype != dtype
            or copy.attributes["index_count"] != 0 or copy.attributes["gather_factor"] != 1
            or copy.attributes["scattered"] is not False or copy.attributes["materialized_output"] is not True
            or len(copy.attributes["symbolic_tensor_outputs"]) != 1
            or copy.attributes["symbolic_tensor_outputs"][0] == input_port):
        raise ValueError("pipeline receive port must enter its same-rank worker copy")
    input_mismatch = copy.attributes["symbolic_tensor_inputs"] != (input_port,)
    k3_reshape = (k3_projection and copy.node_id.endswith(":receive:runner-copy:residual")
        and "symbolic_tensor_input_allocations" in copy.attributes
        and copy.attributes["symbolic_tensor_input_allocations"] == (input_allocation,))
    if input_mismatch and not k3_reshape:
        raise ValueError("pipeline receive port must enter its same-rank worker copy")
    successors = tuple(graph.node(edge.destination) for edge in graph.edges
        if edge.source == copy.node_id and edge.kind == "data")
    k3_consumers = bool(successors) and all(
        node.rank == rank and node.pool_name == copy.pool_name
        and (node.kind is WorkloadKind.P2P or (node.variant == "kimi_k3_attn_res"
            and node.attributes["source_component"] == "vllm_0_30_kimi_k3_attention_residual"))
        and "symbolic_tensor_inputs" in node.attributes
        and (copy.node_id.endswith((":residual", ":attn-res:receive-copy"))
            or copy.attributes["symbolic_tensor_outputs"][0] in node.attributes["symbolic_tensor_inputs"])
        for node in successors)
    if not k3_consumers and (len(successors) != 1 or successors[0].rank != rank or successors[0].pool_name != copy.pool_name
            or "symbolic_tensor_inputs" not in successors[0].attributes
            or copy.attributes["symbolic_tensor_outputs"][0] not in successors[0].attributes["symbolic_tensor_inputs"]):
        raise ValueError("pipeline copied receive port must feed one destination input norm: "
            f"copy={copy.node_id}, successors={tuple(node.node_id for node in successors)}, "
            f"output={copy.attributes['symbolic_tensor_outputs'][0]}, "
            f"details={tuple((node.variant, node.attributes['source_component'] if 'source_component' in node.attributes else '', node.attributes['symbolic_tensor_inputs']) for node in successors)}")
    residual_copy = successors[0].node_id.endswith(":attn-res:receive-copy")
    if not k3_consumers and (set(copy.shape) != {"elements", "rows", "hidden"}
            or (residual_copy and copy.shape != successors[0].shape)
            or (not residual_copy and set(successors[0].shape) != {"rows", "hidden"})
            or any(copy.shape[name] != successors[0].shape[name] for name in ("rows", "hidden"))
            or copy.shape["elements"] != copy.shape["rows"] * copy.shape["hidden"]):
        raise ValueError("pipeline worker copy must match its destination decoder extent")
    if (copy.node_id.endswith(":receive:runner-copy:residual")
            and successors[0].node_id.endswith(":attn-res:receive-copy")):
        return _pipeline_worker_copy_consumer(graph, successors[0],
            copy.attributes["symbolic_tensor_outputs"][0], rank,
            copy.attributes["symbolic_tensor_output_allocations"][0], k3_projection, dtype=dtype)
    return successors[0]


def _validate_symbolic_tp1_pipeline_graph(graph, pool, p2p_nodes):
    """Bind PP messages to produced views, stage frontiers and received inputs.

    Typed dependency tracing keeps both destination inputs, unlike the old
    manually serialized pair. Existing endpoint/byte checks still apply.
    """
    pairs = {}
    for node in p2p_nodes:
        _validate_pipeline_p2p_node(node, pool)
        if "symbolic_tensor_inputs" not in node.attributes:
            raise ValueError("pipeline cannot mix typed and legacy tensor messages")
        key = (node.attributes["source_stage"], node.attributes["payload_tensor"]["name"])
        if key in pairs:
            raise ValueError("pipeline boundary duplicates an intermediate tensor")
        pairs[key] = node
    expected = {(stage, name) for stage in range(pool.parallel_plan.config.pipeline_parallel_size - 1)
        for name in ("hidden_states", "residual")}
    if set(pairs) != expected:
        raise ValueError("pipeline must carry both intermediate tensors at every boundary")
    for stage in range(pool.parallel_plan.config.pipeline_parallel_size - 1):
        hidden, residual = (pairs[stage, name] for name in ("hidden_states", "residual"))
        k3_scope = _symbolic_k3_pipeline_projection(hidden, pool)
        if k3_scope:
            if (hidden.attributes["payload_tensor"]["shape"][-1] != residual.attributes["payload_tensor"]["shape"][-1]
                    or hidden.attributes["payload_tensor"]["dtype"] != residual.attributes["payload_tensor"]["dtype"]):
                raise ValueError("K3 pipeline tensor width and dtype must agree")
        elif hidden.attributes["payload_tensor"]["shape"] != residual.attributes["payload_tensor"]["shape"]:
            raise ValueError("pipeline intermediate tensor shapes must agree")
        source_compute = {node.node_id for node in graph.nodes if node.pool_name == pool.name
            and node.rank == hidden.rank and node.kind is not WorkloadKind.P2P}
        nonterminals = {edge.source for edge in graph.edges
            if edge.source in source_compute and edge.destination in source_compute}
        terminals = source_compute - nonterminals
        if len(terminals) != 1:
            raise ValueError("pipeline source stage must have one completed compute frontier")
        consumers, source_producers = [], []
        for node in (hidden, residual):
            producers = [graph.node(edge.source) for edge in graph.edges
                if edge.destination == node.node_id and edge.kind == "data"]
            alias_bound = False
            if (len(producers) == 1 and (_symbolic_kimi_pipeline_projection(node, pool) or _symbolic_v3_parallel_projection(node, pool))
                    and "symbolic_pipeline_views" in node.attributes):
                from src.kernel.memory_access import TensorView
                views = node.attributes["symbolic_pipeline_views"]
                sent = TensorView(**dict(views["source"]))
                outputs = tuple(TensorView(**dict(view)) for view in views["source_producer_views"])
                alias_bound = (sent.view_id == node.attributes["symbolic_tensor_inputs"][0]
                    and sent.dtype == node.attributes["payload_tensor"]["dtype"]
                    and sent.shape == tuple(node.attributes["payload_tensor"]["shape"])
                    and sent.strides_bytes == (sent.shape[-1] * 2, 2)
                    and any(output.view_id in producers[0].attributes["symbolic_tensor_outputs"]
                        and output.allocation_id == sent.allocation_id and output.dtype == sent.dtype
                        and output.strides_bytes == sent.strides_bytes and len(output.shape) == 2
                        and output.offset_bytes <= sent.offset_bytes
                        and sent.offset_bytes + node.attributes["bytes"] <= output.offset_bytes + prod(output.shape) * 2
                        for output in outputs))
            if (len(producers) != 1 or producers[0].rank != node.rank or producers[0].dtype != node.attributes["payload_tensor"]["dtype"]
                    or "symbolic_tensor_outputs" not in producers[0].attributes
                    or ((node is hidden or not k3_scope)
                        and node.attributes["symbolic_tensor_inputs"][0] not in producers[0].attributes["symbolic_tensor_outputs"]
                        and not alias_bound)):
                raise ValueError("pipeline source port must bind its same-rank producer")
            source_producers.append(producers[0])
            if node is hidden and producers[0].node_id not in terminals:
                raise ValueError("pipeline hidden state must follow the final source-stage frontier")
            successors = [graph.node(edge.destination) for edge in graph.edges
                if edge.source == node.node_id and edge.kind == "data"]
            if len(successors) == 1:
                successors = [_pipeline_worker_copy_consumer(graph, successors[0],
                    node.attributes["symbolic_tensor_outputs"][0], node.attributes["destination_rank"],
                    node.attributes["symbolic_tensor_output_allocations"][0], k3_scope,
                    dtype=node.attributes["payload_tensor"]["dtype"])]
            if k3_scope:
                if (not successors or any(item.rank != node.attributes["destination_rank"]
                        or (item.kind is not WorkloadKind.P2P and (item.variant != "kimi_k3_attn_res"
                            or item.attributes["source_component"] != "vllm_0_30_kimi_k3_attention_residual"))
                        or "symbolic_tensor_inputs" not in item.attributes for item in successors)):
                    raise ValueError("K3 pipeline receive port must feed Attention Residual consumers")
            elif (len(successors) != 1 or successors[0].rank != node.attributes["destination_rank"]
                    or successors[0].kind is not WorkloadKind.NORM
                    or "symbolic_tensor_inputs" not in successors[0].attributes
                    ):
                raise ValueError("pipeline receive port must feed the destination input norm")
            consumers.append(successors[0])
        if consumers[0].node_id != consumers[1].node_id:
            raise ValueError("pipeline hidden and residual must feed the same destination decoder")
        if hidden.attributes["payload_tensor"]["dtype"] != residual.attributes["payload_tensor"]["dtype"]:
            raise ValueError("pipeline hidden and residual must preserve the same source dtype")
        shape = consumers[0].shape
        if k3_scope:
            if (consumers[0].dtype != "bf16" or set(shape) != {"rows", "hidden", "elements"}
                    or tuple(hidden.attributes["payload_tensor"]["shape"]) != (shape["rows"], shape["hidden"])
                    or shape["elements"] != shape["rows"] * shape["hidden"]):
                raise ValueError("K3 pipeline payload must match the Attention Residual shape and dtype")
        elif (set(shape) != {"rows", "hidden"} or consumers[0].dtype != hidden.attributes["payload_tensor"]["dtype"]
                or tuple(hidden.attributes["payload_tensor"]["shape"]) != (shape["rows"], shape["hidden"])):
            raise ValueError("pipeline payload must match the destination decoder shape and dtype")
        final, residual_writer = source_producers
        if k3_scope:
            if (final.kind is not WorkloadKind.ELEMENTWISE or final.variant != "add"
                    or final.shape["elements"] != shape["rows"] * shape["hidden"]
                    or (residual_writer.variant != "kimi_k3_attn_res"
                        and (residual_writer.variant != "gather"
                            or residual_writer.shape["elements"] != residual.attributes["payload_tensor"]["bytes"] // 2))):
                raise ValueError("K3 pipeline tensor shapes must match the shared add and Attention Residual: "
                    f"final={final.kind}/{final.variant}/{final.shape}, "
                    f"residual={residual_writer.kind}/{residual_writer.variant}/{residual_writer.shape}, expected={shape}")
        else:
            dense_final = (final.kind is WorkloadKind.LINEAR and final.variant == "gemm"
                and final.shape["b"] == 1 and final.shape["m"] == shape["rows"] and final.shape["n"] == shape["hidden"])
            moe_final = (final.kind is WorkloadKind.FUSED_MOE and final.variant == "full"
                and "source_component" in final.attributes
                and final.attributes["source_component"] == "symbolic_unquantized_experts"
                and final.attributes["model_execution_version"] == "vllm_0_30_0"
                and final.shape["tokens"] == shape["rows"] and final.shape["hidden"] == shape["hidden"])
            kimi_final = ((_symbolic_kimi_pipeline_projection(hidden, pool) or _symbolic_v3_parallel_projection(hidden, pool))
                and final.kind is WorkloadKind.ELEMENTWISE and final.variant == "add"
                and final.shape["elements"] == shape["rows"] * shape["hidden"]
                and final.node_id.endswith(":mlp:shared-add"))
            if (not (dense_final or moe_final or kimi_final) or residual_writer.kind is not WorkloadKind.NORM
                    or dict(residual_writer.shape) != dict(shape)):
                raise ValueError("pipeline tensor shapes must match the source MLP/MoE and residual norm")
        if WorkloadEdge(hidden.node_id, residual.node_id, "barrier") not in graph.edges:
            raise ValueError("pipeline messages must retain source dictionary send order")
    for edge in graph.edges:
        source, destination = graph.node(edge.source), graph.node(edge.destination)
        if source.rank != destination.rank and (source.kind is not WorkloadKind.P2P
                or source.attributes["destination_rank"] != destination.rank):
            raise ValueError("pipeline cross-rank dependencies must pass through the declared message")


def _validate_tp1_pipeline_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate every tensor and dependency at a TP=1 PP boundary."""

    target_contract = _target_model_semantics_contract(graph)
    for pool in deployment.worker_pools.values():
        if _is_k3_source_pipeline_graph(graph, pool):
            # The dedicated PP validator below checks both TP1 and sliced
            # K3 payloads, including complete removal of the PP messages.
            continue
        if any(node.pool_name == pool.name and _symbolic_v4_pipeline_topology_projection(node, pool) for node in graph.nodes):
            # V4 sends one expanded mHC tensor; the complete boundary is
            # validated by _validate_v4_pipeline_graph below.
            continue
        if not any(node.pool_name == pool.name and node.kind is not WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            continue
        parallelism = pool.parallel_plan.config
        pipeline_size = parallelism.pipeline_parallel_size
        if parallelism.tensor_parallel_size != 1 or pipeline_size == 1:
            continue
        glm = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.kind is WorkloadKind.P2P)
        if any(node.pool_name == pool.name and _symbolic_kimi_dp_projection(node, pool) for node in graph.nodes):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if glm and all(_symbolic_glm_pcp_projection(node, pool) for node in glm):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if _qwen030_dp_sp(parallelism):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if (pipeline_size < 2
                or deployment.mode not in ("colocated", "pd_disagg")
                or (deployment.mode == "colocated" and len(deployment.worker_pools) != 1
                    and not (glm and all("symbolic_tensor_inputs" in node.attributes for node in glm)))
                or parallelism.worker_count != pipeline_size
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "TP=1 pipeline lowering supports one colocated uniform PP>=2 "
                "pool with EP=SP=CP=DP=1")
        p2p_nodes = tuple(
            node for node in graph.nodes
            if node.pool_name == pool.name and node.kind is WorkloadKind.P2P
        )
        expected_count = 2 * (pipeline_size - 1)
        if len(p2p_nodes) != expected_count:
            raise ValueError(
                "TP=1 pipeline graph requires hidden_states and residual at "
                f"every boundary: expected={expected_count}, "
                f"actual={len(p2p_nodes)}")
        if any("symbolic_tensor_inputs" in node.attributes for node in p2p_nodes):
            _validate_symbolic_tp1_pipeline_graph(graph, pool, p2p_nodes)
            continue
        nodes_by_boundary = {
            (source_stage, source_stage + 1): {
                "hidden_states": [],
                "residual": [],
            }
            for source_stage in range(pipeline_size - 1)
        }
        allowed_cross_rank_edges = set()
        for node in p2p_nodes:
            _validate_pipeline_p2p_node(node, pool)
            attributes = node.attributes
            boundary = (
                attributes["source_stage"],
                attributes["destination_stage"],
            )
            if boundary not in nodes_by_boundary:
                raise ValueError(
                    "TP=1 pipeline P2P node crosses an unconfigured boundary: "
                    f"{boundary!r}")
            tensor_name = attributes["payload_tensor"]["name"]
            nodes_by_boundary[boundary][tensor_name].append(node)

        for boundary, nodes_by_tensor in nodes_by_boundary.items():
            source_stage, destination_stage = boundary
            hidden_nodes = tuple(nodes_by_tensor["hidden_states"])
            residual_nodes = tuple(nodes_by_tensor["residual"])
            if len(hidden_nodes) != 1 or len(residual_nodes) != 1:
                raise ValueError(
                    "each TP=1 pipeline boundary requires exactly one "
                    "hidden_states node and one residual node: "
                    f"boundary={boundary!r}, hidden={len(hidden_nodes)}, "
                    f"residual={len(residual_nodes)}")
            hidden = hidden_nodes[0]
            residual = residual_nodes[0]
            hidden_payload = hidden.attributes["payload_tensor"]
            residual_payload = residual.attributes["payload_tensor"]
            if (tuple(hidden_payload["shape"])
                    != tuple(residual_payload["shape"])
                    or hidden_payload["dtype"] != residual_payload["dtype"]
                    or hidden_payload["bytes"] != residual_payload["bytes"]
                    or hidden.attributes["bytes"]
                    != residual.attributes["bytes"]):
                raise ValueError(
                    "pipeline hidden_states and residual payload geometry "
                    f"must match at boundary {boundary!r}")

            hidden_incoming = tuple(
                edge for edge in graph.edges
                if edge.destination == hidden.node_id)
            residual_incoming = tuple(
                edge for edge in graph.edges
                if edge.destination == residual.node_id)
            hidden_outgoing = tuple(
                edge for edge in graph.edges if edge.source == hidden.node_id)
            residual_outgoing = tuple(
                edge for edge in graph.edges
                if edge.source == residual.node_id)
            if (len(hidden_incoming) != 1
                    or hidden_incoming[0].kind != "ready"
                    or len(residual_incoming) != 1
                    or residual_incoming[0].source != hidden.node_id
                    or residual_incoming[0].kind != "ready"
                    or len(hidden_outgoing) != 1
                    or hidden_outgoing[0].destination != residual.node_id
                    or hidden_outgoing[0].kind != "ready"
                    or len(residual_outgoing) != 1
                    or residual_outgoing[0].kind != "data"):
                raise ValueError(
                    "pipeline tensor pair must be an ordered source -> "
                    "hidden_states -> residual -> destination chain: "
                    f"boundary={boundary!r}")
            source = graph.node(hidden_incoming[0].source)
            destination = graph.node(residual_outgoing[0].destination)
            source_ranks = pool.physical_ranks_for_pipeline_stage(source_stage)
            destination_ranks = pool.physical_ranks_for_pipeline_stage(
                destination_stage)
            if (len(source_ranks) != 1 or len(destination_ranks) != 1):
                raise RuntimeError(
                    "TP=1 pipeline stage must resolve to one physical rank")
            valid_source_geometry = (
                source.kind is WorkloadKind.LINEAR
                and source.variant == "gemm"
                and set(source.shape) == {"b", "m", "n", "k"}
                and source.shape["b"] == 1
            )
            if target_contract is not None:
                valid_source_geometry = source.kind in (
                    WorkloadKind.LINEAR, WorkloadKind.ELEMENTWISE)
            if (source.rank != source_ranks[0]
                    or source.dtype != "bf16"
                    or not valid_source_geometry):
                raise ValueError(
                    "pipeline tensor pair must follow one supported BF16 "
                    f"source-stage frontier: boundary={boundary!r}")
            source_outgoing = tuple(
                edge for edge in graph.edges if edge.source == source.node_id)
            if (len(source_outgoing) != 1
                    or source_outgoing[0].destination != hidden.node_id
                    or source_outgoing[0].kind != "ready"):
                raise ValueError(
                    "pipeline transfer must start at the final source-stage "
                    f"frontier: boundary={boundary!r}")
            source_stage_compute = tuple(
                node for node in graph.nodes
                if (node.pool_name == pool.name
                    and node.rank == source_ranks[0]
                    and node.kind is not WorkloadKind.P2P)
            )
            source_stage_terminals = tuple(
                node for node in source_stage_compute
                if not any(
                    edge.source == node.node_id
                    and graph.node(edge.destination).rank == source_ranks[0]
                    and graph.node(edge.destination).kind
                    is not WorkloadKind.P2P
                    for edge in graph.edges
                )
            )
            if tuple(node.node_id for node in source_stage_terminals) != (
                    source.node_id,):
                raise ValueError(
                    "pipeline source stage must have exactly one final compute "
                    f"frontier: boundary={boundary!r}, terminals="
                    f"{tuple(node.node_id for node in source_stage_terminals)!r}")
            if (destination.rank != destination_ranks[0]
                    or destination.kind is not WorkloadKind.NORM
                    or destination.dtype != "bf16"
                    or set(destination.shape) != {"rows", "hidden"}):
                raise ValueError(
                    "pipeline tensor pair must release one BF16 destination "
                    f"input norm: boundary={boundary!r}")
            payload_shape = tuple(hidden_payload["shape"])
            if target_contract is None:
                expected_shape = (source.shape["m"], source.shape["n"])
            else:
                expected_shape = payload_shape
            destination_shape = (
                destination.shape["rows"], destination.shape["hidden"])
            if (payload_shape != expected_shape
                    or destination_shape != expected_shape):
                raise ValueError(
                    "pipeline source, payload, and destination tensor shapes "
                    f"disagree at boundary {boundary!r}")
            if tuple(graph.predecessors(destination.node_id)) != (
                    residual.node_id,):
                raise ValueError(
                    "destination stage must wait for the complete ordered "
                    f"tensor pair at boundary {boundary!r}")
            allowed_cross_rank_edges.add(
                (residual.node_id, destination.node_id, "data"))

        actual_cross_rank_edges = {
            (edge.source, edge.destination, edge.kind)
            for edge in graph.edges
            if graph.node(edge.source).rank != graph.node(edge.destination).rank
        }
        if actual_cross_rank_edges != allowed_cross_rank_edges:
            raise ValueError(
                "TP=1 pipeline graph contains a missing or unmodeled "
                "cross-rank dependency: "
                f"expected={sorted(allowed_cross_rank_edges)!r}, "
                f"actual={sorted(actual_cross_rank_edges)!r}")


def _validate_symbolic_sliced_pipeline_graph(graph, pool, p2p_nodes):
    """Validate every ordered boundary, retaining all lane/role obligations."""
    tp = pool.parallel_plan.config.tensor_parallel_size
    pp = pool.parallel_plan.config.pipeline_parallel_size
    if len(p2p_nodes) != (pp - 1) * tp * 2:
        raise ValueError("sliced pipeline requires both roles on every adjacent-stage lane")
    for stage in range(pp - 1):
        boundary = tuple(node for node in p2p_nodes if node.attributes["source_stage"] == stage
            and node.attributes["destination_stage"] == stage + 1)
        if len(boundary) != tp * 2:
            raise ValueError("sliced pipeline requires a complete adjacent-stage boundary")
        _validate_symbolic_sliced_pipeline_boundary(graph, pool, boundary, stage)


def _validate_mla_dp_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, hidden):
    """Bind0.27 FP8 /0.30 Kimi BF16 expert combine/copy/scale/shared-add to the PP output.

    This proves physical ports and stage ownership, not operator pricing or
    GPU provider support. MoE-only SP restores padded hidden rows before PP;
    the wire and destination retain exactly the original request rows.
    """
    kimi = _symbolic_kimi_dp_projection(producer, pool)
    config = pool.parallel_plan.config
    tp = config.tensor_parallel_size
    layer = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
    stage_experts = tuple(node for node in graph.nodes if node.pool_name == pool.name
        and node.kind is WorkloadKind.FUSED_MOE and node.attributes["routing_provenance"]["layer_index"] == layer)
    sequence = bool(stage_experts) and config.enable_expert_parallel and tp > 1
    local_rows = -(-rows // tp) if sequence else rows

    def parents(node):
        return tuple(graph.node(edge.source) for edge in graph.edges if edge.destination == node.node_id and edge.kind == "data")

    if tp > 1:
        attrs = producer.attributes
        if (producer.kind is not WorkloadKind.COLLECTIVE
                or attrs["operation"] != ("all_gather" if sequence else "all_reduce")
                or attrs["participants"] != source_ranks or attrs["symbolic_collective_storage"] != "fresh"
                or attrs["bytes"] != local_rows * hidden * 2
                or sequence and attrs["symbolic_collective_dim"] != 0):
            raise ValueError("V3 PP requires its actual source TP reduction or padded MoE-only SP restore")
        outputs = parents(producer)
        if (len(outputs) != tp or {node.rank for node in outputs} != set(source_ranks)
                or tuple(producer.attributes["symbolic_tensor_inputs"]) != tuple(
                    next(node for node in outputs if node.rank == rank).attributes["symbolic_tensor_outputs"][0] for rank in source_ranks)):
            raise ValueError("V3 PP restore must consume each local source output port exactly once")
    else:
        outputs = (producer,)
    if not stage_experts:
        if any(node.kind is not WorkloadKind.LINEAR or not node.node_id.endswith(f":layer-{layer}:mlp:down")
                or node.shape["m"] != rows or node.shape["n"] != hidden for node in outputs):
            raise ValueError("V3 dense-only PP stage must finish its actual final dense projection")
        return rows * hidden * 2
    for output in outputs:
        if (output.kind is not WorkloadKind.ELEMENTWISE or output.variant != "add"
                or output.shape != {"elements": local_rows * hidden}
                or not output.node_id.endswith(f":layer-{layer}:mlp:shared-add")):
            raise ValueError("V3 PP MoE frontier must finish its final shared/routed sum")
        inputs = parents(output)
        shared = tuple(node for node in inputs if node.kind is WorkloadKind.LINEAR)
        scaled = tuple(node for node in inputs if node.kind is WorkloadKind.ELEMENTWISE)
        if len(inputs) != 2 or len(shared) != 1 or len(scaled) != 1:
            raise ValueError("V3 PP final sum requires both shared projection and scaled routed output")
        shared, scale = shared[0], scaled[0]
        if (shared.rank != output.rank or not shared.node_id.endswith(f":layer-{layer}:mlp:shared:down")
                or shared.shape["m"] != local_rows or shared.shape["n"] != hidden
                or scale.rank != output.rank or not scale.node_id.endswith(f":layer-{layer}:mlp:output-scale")
                or scale.shape != {"elements": local_rows * hidden}
                or set(output.attributes["symbolic_tensor_inputs"]) != {
                    shared.attributes["symbolic_tensor_outputs"][0], scale.attributes["symbolic_tensor_outputs"][0]}):
            raise ValueError("V3 PP shared-add must retain its actual same-rank source ports")
        copies = parents(scale)
        if len(copies) != 1:
            raise ValueError("V3 PP routed scale requires its actual finalize copy")
        copy, = copies
        if (copy.kind is not WorkloadKind.GATHER or copy.rank != output.rank
                or not copy.node_id.endswith(f":layer-{layer}:mlp:finalize-copy")
                or copy.shape != {"elements": local_rows * hidden, "rows": local_rows, "hidden": hidden}
                or scale.attributes["symbolic_tensor_inputs"] != copy.attributes["symbolic_tensor_outputs"]):
            raise ValueError("V3 PP routed scale must consume its own real finalize-copy output")
        combines = parents(copy)
        if len(combines) != 1:
            raise ValueError("V3 PP finalize copy requires its matching DP sum-scatter")
        combine, = combines
        attrs = combine.attributes
        dimension = "expert" if sequence else "data"
        participants = pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(output.rank))
        if (combine.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != "reduce_scatterv"
                or attrs["collective_group"] != dimension or attrs["reduction"] != "sum"
                or attrs["participants"] != participants or attrs["symbolic_variable_row_shape"] != (hidden,)):
            raise ValueError("V3 PP combine must stay in its source-stage expert/data group")
        active = tuple(rank for rank, count in zip(participants, attrs["symbolic_variable_rows"], strict=True) if count)
        if (output.rank not in active or copy.attributes["symbolic_tensor_inputs"] != (
                attrs["symbolic_tensor_outputs"][active.index(output.rank)],)):
            raise ValueError("V3 PP finalize copy must bind its own sum-scatter output")
        experts = parents(combine)
        if (len(experts) != len(participants) or {node.rank for node in experts} != set(participants)
                or attrs["symbolic_tensor_inputs"] != tuple(
                    next(node for node in experts if node.rank == rank).attributes["symbolic_tensor_outputs"][0] for rank in participants)):
            raise ValueError("V3 PP combine must retain every physical expert contribution")
        for expert in experts:
            if (expert.kind is not WorkloadKind.FUSED_MOE or expert not in stage_experts
                    or expert.attributes["source_component"] != ("symbolic_grouped_routed_experts" if kimi else "symbolic_dsv3_routed_experts")
                    or not kimi and expert.attributes["source_dispatched_input_dtype"] != "fp8"
                    or expert.attributes["moe_invocation_id"] != attrs["moe_invocation_id"]
                    or expert.shape["tokens"] != sum(attrs["symbolic_variable_rows"])):
                raise ValueError("V3 PP expert contribution must bind the final FP8 source invocation")
            dispatched = parents(expert)
            suffixes = (":dispatch-input", ":dispatch-weights", ":dispatch-ids") + (() if kimi else (":dispatch-scales",))
            widths = (hidden, expert.shape["topk"], expert.shape["topk"]) + (() if kimi else (hidden // 128,))
            for port, suffix, width, item_bytes in zip(expert.attributes["symbolic_tensor_inputs"], suffixes, widths, (2, 4, 4) if kimi else (1, 4, 4, 4), strict=True):
                owners = tuple(node for node in dispatched if node.node_id.endswith(suffix))
                if len(owners) != 1:
                    raise ValueError("V3 PP expert requires each actual FP8/scale/weight/ID dispatch")
                dispatch, = owners
                metadata = dispatch.attributes
                if (dispatch.kind is not WorkloadKind.COLLECTIVE or metadata["operation"] != "all_gatherv"
                        or metadata["participants"] != participants or metadata["collective_group"] != dimension
                        or metadata["symbolic_variable_rows"] != attrs["symbolic_variable_rows"]
                        or metadata["symbolic_variable_row_shape"] != (width,)
                        or metadata["bytes"] != sum(attrs["symbolic_variable_rows"]) * width * item_bytes
                        or port != metadata["symbolic_tensor_outputs"][participants.index(expert.rank)]):
                    raise ValueError("V3 PP FP8 dispatch bytes/ports must match the actual expert group")
    return (local_rows * tp if sequence else rows) * hidden * 2


def _validate_distributed_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, hidden, *, output_port=None):
    """Prove the real-row prefix of a DP MoE combine/sequence restore."""
    config = pool.parallel_plan.config
    tp = config.tensor_parallel_size
    ep = config.enable_expert_parallel
    sequence = ep and tp > 1
    expected_operation = "all_gather" if sequence else "all_reduce"
    if tp > 1:
        attrs = producer.attributes
        if (producer.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != expected_operation
                or attrs["participants"] != source_ranks or attrs["symbolic_collective_storage"] != "fresh"
                or attrs["bytes"] != (-(-rows // tp) * hidden * 2 if sequence else rows * hidden * 2)
                or (sequence and attrs["symbolic_collective_dim"] != 0)):
            raise ValueError("distributed pipeline requires its source TP restore or output reduction")
        combines = tuple(graph.node(edge.source) for edge in graph.edges
            if edge.destination == producer.node_id and edge.kind == "data")
        if len(combines) != (1 if ep else tp):
            raise ValueError("distributed pipeline restore must consume exactly its DP combines")
    else:
        combines = (producer,)
    final_layer = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
    ports = {}
    for combine in combines:
        attrs = combine.attributes
        dimension = "expert" if ep else "data"
        if (combine.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != "reduce_scatterv"
                or attrs["collective_group"] != dimension or attrs["reduction"] != "sum"
                or attrs["symbolic_variable_row_shape"] != (hidden,)):
            raise ValueError("distributed pipeline must follow the matching sum-scatter")
        participants = attrs["participants"]
        expected = pool.physical_rank_group(dimension, pool.logical_rank_for_physical_rank(combine.rank))
        if participants != expected:
            raise ValueError("distributed pipeline combine cannot cross its stage or TP lane")
        experts = tuple(graph.node(edge.source) for edge in graph.edges
            if edge.destination == combine.node_id and edge.kind == "data")
        if len(experts) != len(participants) or {node.rank for node in experts} != set(participants):
            raise ValueError("distributed pipeline requires every actual expert contribution")
        for expert in experts:
            if (expert.kind is not WorkloadKind.FUSED_MOE or expert.pool_name != pool.name
                    or expert.attributes["source_component"] != "symbolic_unquantized_experts"
                    or expert.attributes["model_execution_version"] != "vllm_0_30_0"
                    or expert.attributes["routing_provenance"]["layer_index"] != final_layer
                    or expert.attributes["moe_invocation_id"] != attrs["moe_invocation_id"]
                    or expert.attributes["source_dp_lane_rows"] != attrs["symbolic_variable_rows"]
                    or expert.shape["tokens"] != sum(attrs["symbolic_variable_rows"])
                    or expert.shape["hidden"] != hidden):
                raise ValueError("distributed pipeline frontier must be the final source-stage MoE invocation")
            _validate_virtual_ep_router_inputs(graph, expert)
        active_ranks = tuple(rank for rank, count in zip(participants, attrs["symbolic_variable_rows"], strict=True) if count)
        for rank, port in zip(active_ranks, attrs["symbolic_tensor_outputs"], strict=True):
            if rank in source_ranks:
                ports[rank] = port
    if set(ports) != set(source_ranks):
        raise ValueError("distributed pipeline must retain exactly every active local TP lane")
    if tp > 1 and producer.attributes["symbolic_tensor_inputs"] != tuple(ports[rank] for rank in source_ranks):
        raise ValueError("distributed pipeline TP restore must read its own ordered combine output ports")
    if tp == 1 and output_port != ports[source_ranks[0]]:
        raise ValueError("distributed pipeline TP1 requires its own combine output")
    return (-(-rows // tp) * tp if sequence else rows) * hidden * 2


def _validate_glm_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, hidden, *, source_rank):
    """Prove GLM's last dense partial or shared/routed output and SP restore."""
    final_layer = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)

    def partial(node, count, rank):
        if (node.rank != rank or node.dtype != "bf16"
                or f":layer-{final_layer}:mlp:" not in node.node_id):
            raise ValueError("GLM pipeline must retain its actual source-rank final-layer FFN producer")
        if node.kind is WorkloadKind.LINEAR:
            if (node.variant != "gemm" or not node.node_id.endswith(":mlp:down")
                    or (node.shape["b"], node.shape["m"], node.shape["n"]) != (1, count, hidden)):
                raise ValueError("GLM dense pipeline frontier must preserve all active hidden rows")
            return
        if (node.kind is not WorkloadKind.ELEMENTWISE or node.variant != "add"
                or node.shape != {"elements": count * hidden} or not node.node_id.endswith(":shared-add")):
            raise ValueError("GLM pipeline frontier requires its real shared/routed sum")
        inputs = tuple(graph.node(edge.source) for edge in graph.edges if edge.destination == node.node_id and edge.kind == "data")
        shared = tuple(item for item in inputs if item.kind is WorkloadKind.LINEAR and item.rank == node.rank
            and item.node_id.endswith(f":layer-{final_layer}:mlp:shared:down")
            and item.shape["m"] == count and item.shape["n"] == hidden)
        routed = tuple(item for item in inputs if item not in shared)
        if len(inputs) != 2 or len(shared) != 1 or len(routed) != 1:
            raise ValueError("GLM pipeline sum must retain exactly its local shared and routed producers")
        item, = routed
        if item.kind is WorkloadKind.FUSED_MOE:
            if (item.rank != node.rank or item.shape["tokens"] != count or item.shape["hidden"] != hidden
                    or item.attributes["routing_provenance"]["layer_index"] != final_layer):
                raise ValueError("GLM pipeline direct routed producer must preserve its local rows")
        elif item.kind is WorkloadKind.COLLECTIVE:
            attrs = item.attributes
            group = "expert" if pool.parallel_plan.config.enable_expert_parallel else "data"
            expected = pool.physical_rank_group(group, pool.logical_rank_for_physical_rank(node.rank))
            if (attrs["operation"] != "reduce_scatterv" or attrs["collective_group"] != group
                    or attrs["participants"] != expected or attrs["symbolic_variable_row_shape"] != (hidden,)
                    or attrs["symbolic_variable_rows"][expected.index(node.rank)] != count):
                raise ValueError("GLM pipeline routed sum must use its actual stage-local DP/EP combine")
            active = tuple(rank for rank, extent in zip(expected, attrs["symbolic_variable_rows"], strict=True) if extent)
            port = attrs["symbolic_tensor_outputs"][active.index(node.rank)]
            if port not in node.attributes["symbolic_tensor_inputs"]:
                raise ValueError("GLM pipeline sum must consume its own rank's actual routed output")
            experts = tuple(graph.node(name) for name in attrs["symbolic_collective_input_producers"])
            if (len(experts) != len(expected) or any(expert.kind is not WorkloadKind.FUSED_MOE
                    or expert.rank != rank or expert.attributes["routing_provenance"]["layer_index"] != final_layer
                    or expert.attributes["moe_invocation_id"] != attrs["moe_invocation_id"]
                    or expert.attributes["symbolic_tensor_outputs"] != (input_port,)
                    for expert, rank, input_port in zip(experts, expected, attrs["symbolic_tensor_inputs"], strict=True))):
                raise ValueError("GLM pipeline routed combine must retain every actual final-layer expert output")
        else:
            raise ValueError("GLM pipeline routed result requires its actual experts or sum-scatter")

    if producer.kind is WorkloadKind.COLLECTIVE:
        attrs = producer.attributes
        if (attrs["operation"] != "all_gather" or attrs["participants"] != source_ranks
                or "symbolic_glm_moe_sp_restore" not in attrs):
            raise ValueError("GLM pipeline SP output requires its actual padded TP gather")
        proof = attrs["symbolic_glm_moe_sp_restore"]
        padded = -(-rows // len(source_ranks)) * len(source_ranks)
        if (dict(proof) != {"logical_rows": rows, "gathered_rows": padded, "tensor_parallel_size": len(source_ranks)}
                or attrs["symbolic_collective_input_shape"] != (padded // len(source_ranks), hidden)):
            raise ValueError("GLM pipeline SP restore must preserve its real-row prefix and padding")
        writers = tuple(graph.node(name) for name in attrs["symbolic_collective_input_producers"])
        if (len(writers) != len(source_ranks)
                or len(attrs["symbolic_tensor_inputs"]) != len(source_ranks)
                or any(node.rank != rank or port not in node.attributes["symbolic_tensor_outputs"]
                    for node, rank, port in zip(writers, source_ranks, attrs["symbolic_tensor_inputs"], strict=True))):
            raise ValueError("GLM pipeline SP restore requires every actual local TP shard")
        for node, rank in zip(writers, source_ranks, strict=True):
            partial(node, padded // len(source_ranks), rank)
        return padded * hidden * 2
    partial(producer, rows, source_rank)
    return rows * hidden * 2


def _validate_glm_pipeline_full_views(message, producer, rows, hidden, backing_bytes):
    from src.kernel.memory_access import TensorView
    views = message.attributes["symbolic_pipeline_views"]
    source, destination = (TensorView(**dict(views[name])) for name in ("source", "destination"))
    originals = tuple(TensorView(**dict(value)) for value in views["source_producer_views"])
    if (source.view_id != message.attributes["symbolic_tensor_inputs"][0]
            or destination.view_id != message.attributes["symbolic_tensor_outputs"][0]
            or source.allocation_id == destination.allocation_id or source.shape != (rows, hidden)
            or source.dtype != "bf16" or source.offset_bytes != 0 or source.strides_bytes != (2 * hidden, 2)
            or destination.shape != source.shape or destination.dtype != source.dtype or destination.offset_bytes != 0
            or destination.strides_bytes != source.strides_bytes or views["source_backing_bytes"] != backing_bytes
            or backing_bytes < rows * hidden * 2 or views["destination_backing_bytes"] != rows * hidden * 2
            or not any(view.view_id in producer.attributes["symbolic_tensor_outputs"]
                and view.allocation_id == source.allocation_id and view.shape == source.shape
                and view.offset_bytes == source.offset_bytes and view.strides_bytes == source.strides_bytes
                for view in originals)):
        raise ValueError("GLM pipeline full tensor must bind its actual producer and independent receive view")


def _validate_symbolic_dp_pipeline_graph(graph, pool):
    """Validate each active request replica's PP boundaries independently."""
    config = pool.parallel_plan.config
    experts = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.kind is WorkloadKind.FUSED_MOE)
    if any("source_component" in node.attributes
            and node.attributes["source_component"] == "vllm_0_30_symbolic_v4_marlin" for node in experts):
        _validate_v4_pipeline_graph(graph, pool, tuple(node for node in graph.nodes
            if node.pool_name == pool.name and node.kind is WorkloadKind.P2P))
        return
    v3 = bool(experts) and all(_symbolic_mla_dp_pipeline_projection(node, pool) for node in experts)
    if any("source_component" in node.attributes and node.attributes["source_component"] == "symbolic_dsv3_routed_experts"
            for node in experts) and not v3:
        raise NotImplementedError("V3 DP source requires its versioned FP8 dispatch, expert and pipeline pricing bindings")
    glm = bool(experts) and all(_symbolic_glm_pcp_projection(node, pool) for node in experts)
    # MTP recurrences have their own physical DP row extents. PP transports
    # the target batch; draft dispatch/combine is validated per invocation.
    pipeline_experts = tuple(node for node in experts if not (
        glm and _symbolic_glm_mtp_data_projection(node, pool)
        and node.attributes["routing_provenance"]["layer_index"] is None))
    if v3:
        rows = []
        for replica in range(config.data_parallel_size):
            first_stage = tuple(pool.physical_rank_for_logical_rank(replica * config.pipeline_parallel_size
                * config.tensor_parallel_size + lane) for lane in range(config.tensor_parallel_size))
            norms = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.rank in first_stage
                and node.kind is WorkloadKind.NORM and node.node_id.endswith(":layer-0:input-norm"))
            if norms and (len(norms) != config.tensor_parallel_size or {node.rank for node in norms} != set(first_stage)
                    or len({node.shape["rows"] for node in norms}) != 1):
                raise ValueError("MLA DP pipeline requires the original matching rows on every input lane")
            dummy = tuple("source_dummy_run" in node.attributes and node.attributes["source_dummy_run"] is True
                for node in norms)
            if any(dummy) and (not all(dummy) or any(node.shape["rows"] != 1
                    or node.attributes["source_actual_request_count"] != 0 for node in norms)):
                raise ValueError("MLA DP dummy pipeline requires matching one-row non-request inputs on every lane")
            if any(dummy):
                replica_ranks = set(pool.physical_rank_for_logical_rank(replica * config.pipeline_parallel_size
                    * config.tensor_parallel_size + offset) for offset in range(config.pipeline_parallel_size * config.tensor_parallel_size))
                if any(node.pool_name == pool.name and node.rank in replica_ranks
                        and node.kind in (WorkloadKind.KV_UPDATE, WorkloadKind.LM_HEAD) for node in graph.nodes):
                    raise ValueError("legacy MLA dummy pipeline cannot write request cache or sample target logits")
            rows.append(norms[0].shape["rows"] if norms and not any(dummy) else 0)
        geometries = {tuple(rows)}
    elif glm and config.data_parallel_size == 1:
        geometries = {(node.shape["rows"],) for node in graph.nodes if node.pool_name == pool.name
            and node.kind is WorkloadKind.NORM and node.node_id.endswith(":layer-0:input-norm")}
    else:
        geometries = {node.attributes["source_pcp_request_rows" if glm else "source_dp_request_rows"] for node in pipeline_experts}
    if len(geometries) != 1:
        raise ValueError("distributed pipeline requires one consistent DP row geometry")
    logical_rows, = geometries
    if len(logical_rows) != config.data_parallel_size:
        raise ValueError("distributed pipeline row geometry must cover all DP replicas")
    messages = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.kind is WorkloadKind.P2P)
    tp, pp = config.tensor_parallel_size, config.pipeline_parallel_size
    if len(messages) != sum(bool(rows) for rows in logical_rows) * (pp - 1) * tp * 2:
        raise ValueError("distributed pipeline requires both payload roles for exactly its active DP replicas")
    for replica, rows in enumerate(logical_rows):
        if not rows:
            continue
        for stage in range(pp - 1):
            source_ranks = tuple(pool.physical_rank_for_logical_rank(replica * pp * tp + stage * tp + lane) for lane in range(tp))
            destination_ranks = tuple(pool.physical_rank_for_logical_rank(replica * pp * tp + (stage + 1) * tp + lane) for lane in range(tp))
            boundary = tuple(node for node in messages if node.rank in source_ranks)
            if tp > 1:
                _validate_symbolic_sliced_pipeline_boundary(graph, pool, boundary, stage,
                    rank_pair=(source_ranks, destination_ranks))
                continue
            roles = {node.attributes["payload_tensor"]["name"]: node for node in boundary}
            if len(boundary) != 2 or set(roles) != {"hidden_states", "residual"}:
                raise ValueError("distributed TP1 pipeline requires both roles at every active boundary")
            consumers = []
            for role, message in roles.items():
                _validate_pipeline_p2p_node(message, pool)
                if (message.attributes["source_stage"] != stage or message.attributes["destination_stage"] != stage + 1
                        or message.attributes["destination_rank"] != destination_ranks[0]
                        or message.attributes["payload_tensor"]["shape"] != (rows, experts[0].shape["hidden"])):
                    raise ValueError("distributed TP1 payload must preserve its own DP stage and real rows")
                producers = tuple(graph.node(edge.source) for edge in graph.edges if edge.destination == message.node_id and edge.kind == "data")
                if len(producers) != 1:
                    raise ValueError("distributed TP1 pipeline requires one actual source producer")
                producer, = producers
                if (message.attributes["symbolic_tensor_inputs"][0] not in producer.attributes["symbolic_tensor_outputs"]
                        and not (glm or v3)):
                    raise ValueError("distributed pipeline message must bind an actual producer output")
                if role == "hidden_states":
                    if glm:
                        backing_bytes = _validate_glm_pipeline_frontier(graph, pool, producer, source_ranks, stage,
                            rows, experts[0].shape["hidden"], source_rank=message.rank)
                        _validate_glm_pipeline_full_views(message, producer, rows, experts[0].shape["hidden"], backing_bytes)
                    elif v3:
                        backing_bytes = _validate_mla_dp_pipeline_frontier(graph, pool, producer, source_ranks, stage,
                            rows, experts[0].shape["hidden"])
                        _validate_glm_pipeline_full_views(message, producer, rows, experts[0].shape["hidden"], backing_bytes)
                    else:
                        _validate_distributed_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, experts[0].shape["hidden"],
                            output_port=message.attributes["symbolic_tensor_inputs"][0])
                elif (producer.kind is not WorkloadKind.NORM or producer.rank != source_ranks[0]
                        or producer.shape != {"rows": rows, "hidden": experts[0].shape["hidden"]}):
                    raise ValueError("distributed pipeline residual must retain its source norm")
                elif glm or v3:
                    _validate_glm_pipeline_full_views(message, producer, rows, experts[0].shape["hidden"],
                        message.attributes["symbolic_pipeline_views"]["source_backing_bytes"])
                successors = tuple(graph.node(edge.destination) for edge in graph.edges if edge.source == message.node_id and edge.kind == "data")
                if len(successors) != 1:
                    raise ValueError("distributed TP1 receive must enter exactly one worker copy")
                consumers.append(_pipeline_worker_copy_consumer(graph, successors[0],
                    message.attributes["symbolic_tensor_outputs"][0], destination_ranks[0],
                    None, False))
            if (consumers[0].node_id != consumers[1].node_id or consumers[0].kind is not WorkloadKind.NORM
                    or WorkloadEdge(roles["hidden_states"].node_id, roles["residual"].node_id, "barrier") not in graph.edges):
                raise ValueError("distributed pipeline roles must retain dictionary order and the same destination input norm")
    if v3:
        for stage in range(pp - 1):
            stage_ranks = set(pool.physical_ranks_for_pipeline_stage(stage))
            active_stage_ranks = {pool.physical_rank_for_logical_rank((replica * pp + stage) * tp + lane)
                for replica, rows in enumerate(logical_rows) if rows for lane in range(tp)}
            local = {node.node_id for node in graph.nodes if node.pool_name == pool.name and node.kind is not WorkloadKind.P2P
                and (set(node.attributes["participants"]) <= stage_ranks
                    and bool(set(node.attributes["participants"]) & active_stage_ranks) if node.kind is WorkloadKind.COLLECTIVE
                    else node.rank in active_stage_ranks)}
            nonterminals = {edge.source for edge in graph.edges if edge.source in local and edge.destination in local}
            boundary = tuple(message for message in messages if message.attributes["source_stage"] == stage)
            parents = {edge.source for edge in graph.edges if edge.kind == "data"
                and any(edge.destination == message.node_id for message in boundary)}
            hidden_parents = {edge.source for edge in graph.edges if edge.kind == "data"
                and any(edge.destination == message.node_id and message.attributes["payload_tensor"]["name"] == "hidden_states"
                    for message in boundary)}
            terminals = local - nonterminals
            if terminals - parents or hidden_parents - terminals:
                raise ValueError("V3 PP must finish every source-stage dependency before sending its actual hidden frontier")
    for edge in graph.edges:
        source, destination = graph.node(edge.source), graph.node(edge.destination)
        if source.pool_name != pool.name or destination.pool_name != pool.name:
            continue
        source_group = source.attributes["participants"] if source.kind is WorkloadKind.COLLECTIVE else (source.rank,)
        destination_group = destination.attributes["participants"] if destination.kind is WorkloadKind.COLLECTIVE else (destination.rank,)
        source_stages = {pool.logical_rank_for_physical_rank(rank) // tp % pp for rank in source_group}
        destination_stages = {pool.logical_rank_for_physical_rank(rank) // tp % pp for rank in destination_group}
        if (len(source_stages) != 1 or len(destination_stages) != 1
                or (source_stages != destination_stages and (source.kind is not WorkloadKind.P2P
                    or source.attributes["destination_rank"] not in destination_group))):
            raise ValueError("distributed pipeline cross-stage dependencies must pass through their declared PP message")
        if v3:
            if (source.kind is not WorkloadKind.COLLECTIVE and destination.kind is not WorkloadKind.COLLECTIVE
                    and pool.logical_rank_for_physical_rank(source.rank) // (tp * pp)
                    != pool.logical_rank_for_physical_rank(destination.rank) // (tp * pp)):
                raise ValueError("V3 cross-replica dependency must use its actual collective")
            if (source.kind is WorkloadKind.P2P and destination.kind is WorkloadKind.COLLECTIVE
                    and any(pool.logical_rank_for_physical_rank(rank) // (tp * pp)
                        != pool.logical_rank_for_physical_rank(source.rank) // (tp * pp) for rank in destination_group)):
                raise ValueError("V3 PP receive cannot enter another replica's collective")


def _validate_symbolic_sliced_pipeline_boundary(graph, pool, p2p_nodes, stage, *, rank_pair=None):
    """Bind source lane views to independent receives and fresh TP gathers."""
    from src.kernel.memory_access import TensorView
    source_ranks, destination_ranks = ((pool.physical_ranks_for_pipeline_stage(stage),
        pool.physical_ranks_for_pipeline_stage(stage + 1)) if rank_pair is None else rank_pair)
    tp = pool.parallel_plan.config.tensor_parallel_size
    glm_projection = all(_symbolic_glm_pipeline_projection(node, pool) or _symbolic_glm_pcp_projection(node, pool)
        for node in p2p_nodes)
    v3_distributed = bool(p2p_nodes) and all(_symbolic_mla_dp_pipeline_projection(node, pool) for node in p2p_nodes)
    distributed = _qwen030_dp_sp(pool.parallel_plan.config) and not (glm_projection or v3_distributed)
    kimi_projection = all(_symbolic_kimi_pipeline_projection(node, pool) for node in p2p_nodes)
    v3_projection = v3_distributed or all(_symbolic_v3_parallel_projection(node, pool) for node in p2p_nodes)
    target_projection = glm_projection or kimi_projection or v3_projection
    if tp < 2:
        raise NotImplementedError("sliced pipeline requires at least two TP lanes")
    messages = {}
    for node in p2p_nodes:
        _validate_pipeline_p2p_node(node, pool)
        if "symbolic_pipeline_views" not in node.attributes:
            raise ValueError("pipeline cannot mix typed sliced and legacy messages")
        key = (node.rank, node.attributes["payload_tensor"]["name"])
        if key in messages:
            raise ValueError("sliced pipeline duplicates a source lane")
        messages[key] = node
    if set(messages) != {(rank, role) for rank in source_ranks for role in ("hidden_states", "residual")}:
        raise ValueError("sliced pipeline requires both roles on all source lanes")
    gathers = {}
    for role in ("hidden_states", "residual"):
        lanes = tuple(messages[rank, role] for rank in source_ranks)
        dtype = lanes[0].attributes["payload_tensor"]["dtype"]
        successors = tuple({edge.destination for edge in graph.edges
            if edge.source == node.node_id and edge.kind == "data"} for node in lanes)
        if any(len(ids) != 1 for ids in successors) or any(ids != successors[0] for ids in successors[1:]):
            raise ValueError("sliced pipeline receive ports must converge on one role all-gather")
        gather = graph.node(next(iter(successors[0])))
        attrs = gather.attributes
        if (gather.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != "all_gather"
                or tuple(attrs["participants"]) != destination_ranks
                or attrs["symbolic_collective_storage"] != "fresh" or attrs["symbolic_collective_dim"] != 0
                or tuple(attrs["symbolic_tensor_inputs"]) != tuple(node.attributes["symbolic_tensor_outputs"][0] for node in lanes)
                or len(attrs["symbolic_tensor_outputs"]) != tp
                or len(set(attrs["symbolic_tensor_outputs"])) != tp):
            raise ValueError("sliced pipeline requires fresh destination all-gather with matching receive ports")
        data_sources = {edge.source for edge in graph.edges if edge.destination == gather.node_id and edge.kind == "data"}
        if data_sources != {node.node_id for node in lanes}:
            raise ValueError("sliced pipeline all-gather must read exactly its lane messages")
        direct_consumers = tuple(graph.node(edge.destination) for edge in graph.edges
            if edge.source == gather.node_id and edge.kind == "data")
        if target_projection:
            if (len(direct_consumers) != tp or {node.rank for node in direct_consumers} != set(destination_ranks)
                    or any(node.kind is not WorkloadKind.GATHER or node.variant != "gather"
                        or node.dtype != "bf16" or node.attributes["index_count"] != 0
                        or node.attributes["gather_factor"] != 1 or node.attributes["scattered"] is not False
                        or node.attributes["materialized_output"] is not True for node in direct_consumers)):
                raise ValueError("GLM pipeline must copy both fresh gathers into the worker's intermediate buffers")
            consumers = []
            for copy in direct_consumers:
                lane = destination_ranks.index(copy.rank)
                if (copy.attributes["symbolic_tensor_inputs"] != (attrs["symbolic_tensor_outputs"][lane],)
                        or len(copy.attributes["symbolic_tensor_outputs"]) != 1):
                    raise ValueError("GLM pipeline worker copy must read its own gathered port")
                successors_copy = tuple(graph.node(edge.destination) for edge in graph.edges
                    if edge.source == copy.node_id and edge.kind == "data")
                if len(successors_copy) != 1:
                    raise ValueError("GLM pipeline copied port must have one source consumer")
                successor = successors_copy[0]
                if role == "hidden_states" and glm_projection:
                    reduce_attrs = successor.attributes
                    if (successor.kind is not WorkloadKind.COLLECTIVE or reduce_attrs["operation"] != "all_reduce"
                            or tuple(reduce_attrs["participants"]) != destination_ranks
                            or set(reduce_attrs["symbolic_tensor_inputs"]) != {
                                node.attributes["symbolic_tensor_outputs"][0] for node in direct_consumers}):
                        raise ValueError("GLM pipeline hidden copy must enter its source destination-stage input reduction")
                    reduced_consumers = tuple(graph.node(edge.destination) for edge in graph.edges
                        if edge.source == successor.node_id and edge.kind == "data")
                    own = tuple(node for node in reduced_consumers if node.rank == copy.rank)
                    if (len(reduced_consumers) != tp or len(own) != 1
                            or reduce_attrs["symbolic_tensor_outputs"][lane] not in own[0].attributes["symbolic_tensor_inputs"]):
                        raise ValueError("GLM pipeline post-receive reduction must bind every lane's input norm")
                    successor = own[0]
                elif (successor.rank != copy.rank
                        or copy.attributes["symbolic_tensor_outputs"][0] not in successor.attributes["symbolic_tensor_inputs"]):
                    raise ValueError("GLM pipeline copied residual must feed its own stage input norm")
                consumers.append(successor)
            consumers = tuple(consumers)
        else:
            consumers = tuple(_pipeline_worker_copy_consumer(graph, copy,
                attrs["symbolic_tensor_outputs"][destination_ranks.index(copy.rank)], copy.rank,
                None, False, dtype=dtype) for copy in direct_consumers)
        if (len(consumers) != tp or {node.rank for node in consumers} != set(destination_ranks)
                or any(node.kind is not WorkloadKind.NORM or node.dtype != dtype
                    or set(node.shape) != {"rows", "hidden"} for node in consumers)
                or any(dict(node.shape) != dict(consumers[0].shape) for node in consumers[1:])):
            raise ValueError("sliced pipeline gathered tensors must feed all destination input norms")
        rows, hidden = consumers[0].shape["rows"], consumers[0].shape["hidden"]
        full_bytes = rows * hidden * 2
        if full_bytes % tp or attrs["bytes"] != full_bytes // tp:
            raise ValueError("sliced pipeline all-gather bytes must match the full destination tensor")
        for lane, (message, destination_rank) in enumerate(zip(lanes, destination_ranks, strict=True)):
            consumer = next(node for node in consumers if node.rank == destination_rank)
            if not target_projection:
                copy = next(node for node in direct_consumers if node.rank == destination_rank)
                if copy.attributes["symbolic_tensor_outputs"][0] not in consumer.attributes["symbolic_tensor_inputs"]:
                    raise ValueError("sliced pipeline copied output port must bind the destination norm")
            payload = message.attributes["payload_tensor"]
            views = message.attributes["symbolic_pipeline_views"]
            if set(views) != {"source", "destination", "source_backing_bytes", "destination_backing_bytes", "source_producer_views"}:
                raise ValueError("sliced pipeline view contract fields must be complete")
            source_view, destination_view = (TensorView(**dict(views[key])) for key in ("source", "destination"))
            producers = tuple(graph.node(edge.source) for edge in graph.edges
                if edge.destination == message.node_id and edge.kind == "data")
            if len(producers) != 1:
                raise ValueError("sliced pipeline requires one source storage producer")
            producer = producers[0]
            producer_bytes = (_validate_glm_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, hidden,
                source_rank=message.rank)
                if glm_projection and _symbolic_glm_pcp_projection(message, pool) and role == "hidden_states" else
                _validate_mla_dp_pipeline_frontier(graph, pool, producer, source_ranks, stage, rows, hidden)
                    if v3_distributed and role == "hidden_states" else
                _validate_distributed_pipeline_frontier(graph, pool, producer,
                    source_ranks, stage, rows, hidden) if distributed and role == "hidden_states" else full_bytes)
            # A later stage's in-place residual aliases the active prefix of
            # its persistent max-batched-token intermediate buffer. Transmit
            # only the active lane; retain the larger physical backing in the
            # lifecycle report and validate its full active producer below.
            source_backing_valid = (views["source_backing_bytes"] == producer_bytes or (
                target_projection and stage > 0 and role == "residual"
                and views["source_backing_bytes"] >= full_bytes
                and views["source_backing_bytes"] % (hidden * 2) == 0))
            if (source_view.view_id != message.attributes["symbolic_tensor_inputs"][0]
                    or destination_view.view_id != message.attributes["symbolic_tensor_outputs"][0]
                    or source_view.allocation_id == destination_view.allocation_id
                    or not source_backing_valid or views["destination_backing_bytes"] != full_bytes
                    or tuple(payload["shape"]) != (rows * hidden // tp,)
                    or payload["bytes"] != full_bytes // tp):
                raise ValueError("sliced pipeline view ports and independent full backings must match its payload")
            for view in (source_view, destination_view):
                if (view.shape != (rows * hidden // tp,) or view.strides_bytes != (2,)
                        or view.dtype != dtype or view.offset_bytes != lane * (full_bytes // tp)):
                    raise ValueError("sliced pipeline view must select its canonical flattened TP lane")
            producer_views = tuple(TensorView(**dict(view)) for view in views["source_producer_views"])
            if (not producer_views or any(view.view_id not in producer.attributes["symbolic_tensor_outputs"]
                    for view in producer_views)
                    or not any(view.allocation_id == source_view.allocation_id
                        and view.shape == (producer_bytes // (hidden * 2), hidden) and view.offset_bytes == 0
                        and view.strides_bytes == (hidden * 2, 2) and view.dtype == dtype for view in producer_views)):
                raise ValueError("sliced pipeline source alias must bind its producer's full output")
            if role == "hidden_states":
                if glm_projection and _symbolic_glm_pcp_projection(message, pool):
                    continue
                if distributed or v3_distributed:
                    # The exact padded restore and its real-row prefix were
                    # proved above, including each stage-local expert input.
                    continue
                if glm_projection:
                    final_layer = max(index for index, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage)
                        if owner == stage)
                    if (producer.rank != message.rank or producer.kind is not WorkloadKind.ELEMENTWISE
                            or producer.variant != "add" or producer.shape["elements"] != rows * hidden
                            or not producer.node_id.endswith(f":layer-{final_layer}:mlp:shared-add")):
                        raise ValueError("GLM pipeline must send the final TP-partial shared/routed sum without an added reduction")
                    inputs = tuple(graph.node(edge.source) for edge in graph.edges
                        if edge.destination == producer.node_id and edge.kind == "data")
                    if (len(inputs) != 2 or {node.kind for node in inputs} != {WorkloadKind.LINEAR, WorkloadKind.FUSED_MOE}
                            or any(node.rank != message.rank for node in inputs)):
                        raise ValueError("GLM pipeline partial sum must retain both local shared and routed producers")
                elif (producer.kind is not WorkloadKind.COLLECTIVE or producer.attributes["operation"] != "all_reduce"
                        or tuple(producer.attributes["participants"]) != source_ranks or producer.attributes["bytes"] != full_bytes):
                    raise ValueError("sliced pipeline hidden state requires the source-stage MLP all-reduce")
                if not glm_projection:
                    inputs = tuple(graph.node(edge.source) for edge in graph.edges
                        if edge.destination == producer.node_id and edge.kind == "data")
                    valid_inputs = (all(node.kind is WorkloadKind.ELEMENTWISE and node.variant == "add"
                        and node.dtype == "bf16" and dict(node.shape) == {"elements": rows * hidden}
                        or node.kind is WorkloadKind.LINEAR and node.variant == "gemm" and node.dtype == "bf16"
                        and node.node_id.endswith(":mlp:down") and node.shape["m"] == rows and node.shape["n"] == hidden
                        for node in inputs) if kimi_projection or v3_projection else
                        all((node.kind is WorkloadKind.LINEAR and node.variant == "gemm"
                                and node.shape["m"] == rows and node.shape["n"] == hidden)
                            or (node.kind is WorkloadKind.FUSED_MOE
                                and node.attributes["source_component"] == "symbolic_unquantized_experts"
                                and node.attributes["model_execution_version"] == "vllm_0_30_0"
                                and node.shape["tokens"] == rows and node.shape["hidden"] == hidden)
                            for node in inputs))
                    if len(inputs) != tp or not valid_inputs or {node.rank for node in inputs} != set(source_ranks):
                        raise ValueError("sliced pipeline frontier must follow the source MLP outputs")
            else:
                hidden_message = messages[message.rank, "hidden_states"]
                if (producer.kind is not WorkloadKind.NORM or producer.rank != message.rank
                        or dict(producer.shape) != {"rows": rows, "hidden": hidden}
                        or WorkloadEdge(hidden_message.node_id, message.node_id, "barrier") not in graph.edges):
                    raise ValueError("sliced pipeline residual must retain its source norm and dictionary send order")
        gathers[role] = (gather, {node.node_id for node in consumers})
    if gathers["hidden_states"][0].node_id == gathers["residual"][0].node_id or gathers["hidden_states"][1] != gathers["residual"][1]:
        raise ValueError("sliced pipeline roles need distinct gathers and the same destination decoder")
    if WorkloadEdge(gathers["hidden_states"][0].node_id, gathers["residual"][0].node_id, "barrier") not in graph.edges:
        raise ValueError("sliced pipeline receive postprocess must retain dictionary gather order")
    # No direct stage-to-stage shortcut may bypass the declared P2P lane.
    for edge in graph.edges:
        source, destination = graph.node(edge.source), graph.node(edge.destination)
        source_group = (tuple(source.attributes["participants"]) if source.kind is WorkloadKind.COLLECTIVE else (source.rank,))
        destination_group = (tuple(destination.attributes["participants"]) if destination.kind is WorkloadKind.COLLECTIVE else (destination.rank,))
        if not set(source_group) & set(destination_group) and (source.kind is not WorkloadKind.P2P
                or source.attributes["destination_rank"] not in destination_group):
            raise ValueError("sliced pipeline cross-stage dependencies must use their declared receive lane")


def _validate_k3_pipeline_graph(graph, pool, messages):
    """Bind both K3 PP roles through real reshapes and block-buffer copies.

    Attention Residual capacity comes from the source operators. The
    transport proof never grants operator/provider or normal-memory support.
    """
    from src.kernel.memory_access import TensorView
    config = pool.parallel_plan.config
    tp, dp, pp = config.tensor_parallel_size, config.data_parallel_size, config.pipeline_parallel_size
    if pp <= 1 or any(not _symbolic_k3_pipeline_topology_projection(node, pool) for node in messages):
        raise ValueError("K3 pipeline requires its versioned typed source messages")
    parents, children = {}, {}
    for edge in graph.edges:
        if edge.kind == "data":
            if edge.destination not in parents:
                parents[edge.destination] = []
            if edge.source not in children:
                children[edge.source] = []
            parents[edge.destination].append(graph.node(edge.source))
            children[edge.source].append(graph.node(edge.destination))
    first_layers = tuple(next(layer for layer, owner in enumerate(pool.parallel_plan.layer_to_pipeline_stage) if owner == stage)
        for stage in range(pp))
    entries = {}
    for node in graph.nodes:
        if node.pool_name != pool.name or node.variant != "kimi_k3_attn_res":
            continue
        stage = pool.logical_rank_for_physical_rank(node.rank) // tp % pp
        if node.node_id.endswith(f":layer-{first_layers[stage]}:pre-attn"):
            if (node.rank in entries or node.kind is not WorkloadKind.VECTOR or node.dtype != "bf16"
                    or node.attributes["source_component"] != "vllm_0_30_kimi_k3_attention_residual"):
                raise ValueError("K3 pipeline requires one real Attention Residual entry per active worker")
            entries[node.rank] = node
    rows_by_replica, widths = [], set()
    for replica in range(dp):
        ranks = tuple(pool.physical_rank_for_logical_rank(replica * pp * tp + lane) for lane in range(tp))
        present = tuple(entries[rank] for rank in ranks if rank in entries)
        if not present:
            rows_by_replica.append(0)
            continue
        if len(present) != tp or len({node.shape["rows"] for node in present}) != 1:
            raise ValueError("K3 PP requires matching original rows on all active TP lanes")
        dummy=tuple('source_dummy_run' in node.attributes and node.attributes['source_dummy_run'] is True for node in present)
        if any(dummy) and (not all(dummy) or any(node.shape['rows']!=1 or node.attributes['source_actual_request_count']!=0
                or node.attributes['source_k3_request_rows'][replica]!=0 for node in present)):
            raise ValueError("K3 dummy PP requires one physical non-request input on every lane")
        rows_by_replica.append(0 if any(dummy) else present[0].shape["rows"])
        widths.update(node.shape["hidden"] for node in present)
    if len(widths) != 1 or len(messages) != sum(bool(rows) for rows in rows_by_replica) * (pp - 1) * tp * 2:
        raise ValueError("K3 PP requires both payload roles on every active replica boundary")
    hidden, = widths

    def only_relation(node_id, relations, description):
        if node_id not in relations or len(relations[node_id]) != 1:
            raise ValueError("K3 PP requires exactly one actual " + description)
        return relations[node_id][0]

    def copy_node(node, rank, rows, width):
        if (node.rank != rank or node.kind is not WorkloadKind.GATHER or node.variant != "gather" or node.dtype != "bf16"
                or node.shape != {"elements": rows * width, "rows": rows, "hidden": width}
                or node.attributes["index_count"] != 0 or node.attributes["gather_factor"] != 1
                or node.attributes["scattered"] is not False or node.attributes["materialized_output"] is not True
                or len(node.attributes["symbolic_tensor_inputs"]) != 1 or len(node.attributes["symbolic_tensor_outputs"]) != 1):
            raise ValueError("K3 PP requires its same-rank materialized worker copy and exact extent")

    def dense_view(view, shape, *, offset=0):
        stride, strides = 2, []
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        return (view.shape == shape and view.strides_bytes == tuple(strides)
            and view.dtype == "bf16" and view.offset_bytes == offset)

    for stage in range(pp - 1):
        hidden_frontiers = set()
        for replica, rows in enumerate(rows_by_replica):
            source = tuple(pool.physical_rank_for_logical_rank((replica * pp + stage) * tp + lane) for lane in range(tp))
            destination = tuple(pool.physical_rank_for_logical_rank((replica * pp + stage + 1) * tp + lane) for lane in range(tp))
            lanes = tuple(message for message in messages if message.rank in source)
            if not rows:
                if lanes or any(rank not in entries or 'source_dummy_run' not in entries[rank].attributes
                        or entries[rank].attributes['source_dummy_run'] is not True
                        or entries[rank].attributes['source_actual_request_count']!=0
                        or entries[rank].attributes['source_k3_request_rows'][replica]!=0
                        or entries[rank].shape!={'rows':1,'hidden':hidden,'elements':hidden}
                        for rank in (*source,*destination)):
                    raise ValueError("idle K3 replica requires its complete local dummy body without PP payload")
                continue
            if any(rank not in entries for rank in (*source, *destination)):
                raise ValueError("K3 PP must preserve its active owner's stage-entry workers")
            capacity = entries[source[0]].attributes["block_capacity"]
            if (any(entries[rank].shape != {"rows": rows, "hidden": hidden, "elements": rows * hidden}
                    or entries[rank].attributes["block_capacity"] != capacity for rank in source)
                    or any(entries[rank].shape != entries[source[0]].shape
                        or entries[rank].attributes["num_blocks"] != capacity
                        or entries[rank].attributes["block_capacity"] < capacity
                        or entries[rank].attributes["has_delta"] is not False for rank in destination)):
                raise ValueError("K3 PP residual prefix must match its original rows and next-stage block count")
            for rank in source:
                inherited = entries[rank].attributes["num_blocks"]
                writes = tuple(node.attributes["block_write_idx"] for node in graph.nodes
                    if node.pool_name == pool.name and node.rank == rank and node.variant == "kimi_k3_attn_res"
                    and node.attributes["block_write_idx"] >= 0)
                if not 0 <= inherited <= capacity or sorted(writes) != list(range(inherited, capacity)):
                    raise ValueError("K3 PP residual prefix must retain each inherited or actually written block")
            paired = {(message.rank, message.attributes["payload_tensor"]["name"]): message for message in lanes}
            if len(lanes) != tp * 2 or set(paired) != {(rank, role) for rank in source for role in ("hidden_states", "residual")}:
                raise ValueError("K3 PP requires both ordered roles on each active TP lane")
            gathers = {}
            for role in ("hidden_states", "residual"):
                role_lanes = tuple(paired[rank, role] for rank in source)
                total = rows * hidden * (capacity if role == "residual" else 1)
                if total % tp:
                    raise ValueError("K3 PP payload must divide its physical TP lanes")
                wire_shape = ((rows * (capacity if role == "residual" else 1), hidden) if tp == 1 else (total // tp,))
                for lane, message in enumerate(role_lanes):
                    _validate_pipeline_p2p_node(message, pool)
                    metadata = message.attributes["symbolic_pipeline_views"]
                    sent, received = (TensorView(**dict(metadata[name])) for name in ("source", "destination"))
                    writer = only_relation(message.node_id, parents, "source producer")
                    if (message.attributes["payload_tensor"]["dtype"] != "bf16"
                            or tuple(message.attributes["payload_tensor"]["shape"]) != wire_shape
                            or message.attributes["bytes"] != total * 2 // tp
                            or metadata["source_backing_bytes"] != total * 2 or metadata["destination_backing_bytes"] != total * 2
                            or not dense_view(sent, wire_shape, offset=lane * total * 2 // tp)
                            or not dense_view(received, wire_shape, offset=lane * total * 2 // tp)
                            or sent.view_id != message.attributes["symbolic_tensor_inputs"][0]
                            or received.view_id != message.attributes["symbolic_tensor_outputs"][0]
                            or received.allocation_id != message.attributes["symbolic_tensor_output_allocations"][0]
                            or sent.allocation_id == received.allocation_id or writer.rank != message.rank or writer.dtype != "bf16"):
                        raise ValueError("K3 PP wire must preserve its actual full backing and exact hidden/residual lane")
                    produced = tuple(TensorView(**dict(view)) for view in metadata["source_producer_views"])
                    bound = tuple(view for view in produced if view.allocation_id == sent.allocation_id
                        and view.view_id in writer.attributes["symbolic_tensor_outputs"])
                    if role == "hidden_states":
                        if (writer.kind is not WorkloadKind.ELEMENTWISE or writer.variant != "add"
                                or not writer.node_id.endswith(":pipeline:prefix-add") or writer.shape != {"elements": rows * hidden}
                                or not any(dense_view(view, (rows, hidden)) for view in bound)):
                            raise ValueError("K3 PP hidden must follow its actual final prefix-add")
                        hidden_frontiers.add(writer.node_id)
                    elif writer.variant == "kimi_k3_attn_res":
                        index = writer.attributes["block_write_idx"]
                        if (writer.attributes["block_capacity"] != capacity or index != capacity - 1
                                or not any(view.shape == (rows, 1, hidden) and view.strides_bytes == (capacity * hidden * 2, hidden * 2, 2)
                                    and view.dtype == "bf16" and view.offset_bytes == index * hidden * 2 for view in bound)):
                            raise ValueError("K3 PP residual must alias the actual last block writer")
                    elif (not writer.node_id.endswith(":attn-res:receive-copy")
                            or not any(dense_view(view, (rows, capacity, hidden)) for view in bound)):
                        raise ValueError("K3 PP unchanged residual must retain its actual inherited block copy")
                    post = only_relation(message.node_id, children, "destination receive postprocess")
                    if tp > 1:
                        attrs = post.attributes
                        if (post.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != "all_gather"
                                or attrs["participants"] != destination or attrs["symbolic_collective_storage"] != "fresh"
                                or attrs["symbolic_collective_dim"] != 0 or attrs["bytes"] != total * 2 // tp
                                or attrs["symbolic_tensor_inputs"] != tuple(node.attributes["symbolic_tensor_outputs"][0] for node in role_lanes)
                                or {node.node_id for node in parents[post.node_id]} != {node.node_id for node in role_lanes}):
                            raise ValueError("K3 PP receive lanes must enter their own fresh destination gather")
                        gathers[role] = post.node_id
                        if post.node_id not in children:
                            raise ValueError("K3 PP gather requires exactly one actual runner copy per TP rank")
                        copies = tuple(node for node in children[post.node_id] if node.rank == destination[lane])
                        if len(children[post.node_id]) != tp or len(copies) != 1:
                            raise ValueError("K3 PP gather requires exactly one actual runner copy per TP rank")
                        copy, = copies
                        port = attrs["symbolic_tensor_outputs"][lane]
                    else:
                        copy, port = post, received.view_id
                    width = hidden * (capacity if role == "residual" else 1)
                    copy_node(copy, destination[lane], rows, width)
                    if not copy.node_id.endswith(":receive:runner-copy:" + role):
                        raise ValueError("K3 PP must enter its actual role-specific runner copy")
                    views = metadata["destination_postprocess_views"]
                    original, reshaped = (TensorView(**dict(views[key])) for key in ("received_or_gathered", "runner_input"))
                    expected = (rows, capacity, hidden) if role == "residual" else (rows, hidden)
                    if (original.view_id != port or not dense_view(original, (rows * (capacity if role == "residual" else 1), hidden))
                            or not dense_view(reshaped, expected) or original.allocation_id != reshaped.allocation_id
                            or reshaped.view_id != copy.attributes["symbolic_tensor_inputs"][0]
                            or copy.attributes["symbolic_tensor_input_allocations"] != (reshaped.allocation_id,)
                            or tp > 1 and original.allocation_id == received.allocation_id):
                        raise ValueError("K3 PP worker copy receive reshape must alias its actual gathered view without inventing a copy")
                    target = entries[destination[lane]]
                    if role == "hidden_states":
                        if (metadata["destination_residual_views"] is not None
                                or copy.attributes["symbolic_tensor_outputs"][0] != target.attributes["symbolic_tensor_inputs"][0]
                                or WorkloadEdge(copy.node_id, target.node_id, "data") not in graph.edges):
                            raise ValueError("K3 PP hidden copy must enter its own first Attention Residual prefix")
                    else:
                        block_copy = only_relation(copy.node_id, children, "destination block copy")
                        copy_node(block_copy, destination[lane], rows, capacity * hidden)
                        if not block_copy.node_id.endswith(":attn-res:receive-copy"):
                            raise ValueError("K3 residual runner buffer must be copied into its actual block arena")
                        receipts = metadata["destination_residual_views"]
                        runner_out, block_in, block_out, attention_in = (TensorView(**dict(receipts[key]))
                            for key in ("runner_output", "block_copy_input", "block_copy_output", "attention_input"))
                        target_capacity = target.attributes["block_capacity"]
                        if (runner_out.view_id != copy.attributes["symbolic_tensor_outputs"][0]
                                or not dense_view(runner_out, expected) or block_in != runner_out
                                or block_in.view_id != block_copy.attributes["symbolic_tensor_inputs"][0]
                                or block_out.view_id != block_copy.attributes["symbolic_tensor_outputs"][0]
                                or attention_in.view_id != target.attributes["symbolic_tensor_inputs"][-1]
                                or block_out.allocation_id == runner_out.allocation_id
                                or block_out.allocation_id != attention_in.allocation_id
                                or block_out.shape != expected or attention_in.shape != expected
                                or block_out.offset_bytes != 0 or attention_in.offset_bytes != 0
                                or block_out.strides_bytes != (target_capacity * hidden * 2, hidden * 2, 2)
                                or attention_in.strides_bytes != block_out.strides_bytes
                                or block_out.dtype != "bf16" or attention_in.dtype != "bf16"
                                or receipts["block_residual_backing_bytes"] != rows * target_capacity * hidden * 2
                                or WorkloadEdge(block_copy.node_id, target.node_id, "data") not in graph.edges):
                            raise ValueError("K3 PP block copy must bind the same real residual arena read by Attention Residual")
                if tp > 1 and len(set(gathers.values())) != len(gathers):
                    raise ValueError("K3 hidden and block residual need distinct destination gathers")
            for rank in source:
                if WorkloadEdge(paired[rank, "hidden_states"].node_id, paired[rank, "residual"].node_id, "barrier") not in graph.edges:
                    raise ValueError("K3 PP must retain hidden-then-residual dictionary send order")
            if tp > 1 and WorkloadEdge(gathers["hidden_states"], gathers["residual"], "barrier") not in graph.edges:
                raise ValueError("K3 PP must retain hidden-then-residual destination gather order")
        stage_ranks = set(pool.physical_ranks_for_pipeline_stage(stage))
        active_stage_ranks={rank for rank in stage_ranks
            if rows_by_replica[pool.logical_rank_for_physical_rank(rank)//(tp*pp)]}
        local = {node.node_id for node in graph.nodes if node.pool_name == pool.name and node.kind is not WorkloadKind.P2P
            and (set(node.attributes["participants"]) <= stage_ranks and bool(set(node.attributes['participants'])&active_stage_ranks)
                if node.kind is WorkloadKind.COLLECTIVE else node.rank in active_stage_ranks
                or node.kind is WorkloadKind.FUSED_MOE and node.rank in stage_ranks)}
        nonterminals = {edge.source for edge in graph.edges if edge.source in local and edge.destination in local}
        if local - nonterminals != hidden_frontiers:
            raise ValueError("K3 PP must finish all stage compute and idle expert dependencies before sending its hidden frontier")
    for edge in graph.edges:
        source, destination = graph.node(edge.source), graph.node(edge.destination)
        if source.pool_name != pool.name or destination.pool_name != pool.name:
            continue
        left = source.attributes["participants"] if source.kind is WorkloadKind.COLLECTIVE else (source.rank,)
        right = destination.attributes["participants"] if destination.kind is WorkloadKind.COLLECTIVE else (destination.rank,)
        before = {pool.logical_rank_for_physical_rank(rank) // tp % pp for rank in left}
        after = {pool.logical_rank_for_physical_rank(rank) // tp % pp for rank in right}
        if (len(before) != 1 or len(after) != 1 or before != after and (source.kind is not WorkloadKind.P2P
                or source.attributes["destination_rank"] not in right)):
            raise ValueError("K3 cross-stage dependencies must pass through their actual PP message")
        if (source.kind is not WorkloadKind.COLLECTIVE and destination.kind is not WorkloadKind.COLLECTIVE
                and pool.logical_rank_for_physical_rank(source.rank) // (tp * pp)
                != pool.logical_rank_for_physical_rank(destination.rank) // (tp * pp)):
            raise ValueError("K3 cross-replica dependencies must pass through their actual collective")
        if source.kind is WorkloadKind.P2P and any(pool.logical_rank_for_physical_rank(rank) // (tp * pp)
                != pool.logical_rank_for_physical_rank(source.rank) // (tp * pp) for rank in right):
            raise ValueError("K3 PP receive cannot enter a foreign replica")


def _validate_v4_pipeline_boundary(graph, pool, lanes, source, destination, *, expected_rows, expected_hidden, expected_hc_mult):
    """One active DP owner's expanded hidden lanes and receive/gather ports."""
    from src.kernel.memory_access import TensorView
    tp = pool.parallel_plan.config.tensor_parallel_size
    if len(lanes) != tp or {node.rank for node in lanes} != set(source):
        raise ValueError("V4 pipeline duplicates or loses a source lane")
    gather_ids = set()
    posts = set()
    for message in lanes:
        _validate_pipeline_p2p_node(message, pool)
        if message.attributes["payload_tensor"]["name"] != "hidden_states":
            raise ValueError("V4 pipeline must not invent a residual message")
        parents = tuple(graph.node(edge.source) for edge in graph.edges
            if edge.destination == message.node_id and edge.kind == "data")
        if len(parents) != 1:
            raise ValueError("V4 hidden lane needs one completed mHC post producer")
        post = parents[0]
        if (post.rank != message.rank or post.kind is not WorkloadKind.VECTOR or post.variant != "mhc_post"
                or post.dtype != "bf16" or not post.node_id.endswith(":final-mhc-post")):
            raise ValueError("V4 PP hidden must follow its stage-final mHC post")
        posts.add(post.node_id)
        h, c = post.shape["hidden"], post.shape["hc_mult"]
        size = post.shape["elements"] * 2
        if post.shape["elements"] % (h * c) or size % tp:
            raise ValueError("V4 PP mHC rows must divide the typed TP lanes")
        rows, lane = post.shape["elements"] // (h * c), source.index(message.rank)
        if rows != expected_rows or h != expected_hidden or c != expected_hc_mult:
            raise ValueError("V4 pipeline must preserve its own replica rows and hidden width")
        metadata = message.attributes["symbolic_pipeline_views"]
        sent, received = (TensorView(**dict(metadata[name])) for name in ("source", "destination"))
        output_views = tuple(TensorView(**dict(view)) for view in metadata["source_producer_views"])
        wire_shape = (rows, c, h) if tp == 1 else (size // (2 * tp),)
        wire_strides = (c * h * 2, h * 2, 2) if tp == 1 else (2,)
        if (message.attributes["bytes"] != size // tp
                or tuple(message.attributes["payload_tensor"]["shape"]) != wire_shape
                or metadata["source_backing_bytes"] != size or metadata["destination_backing_bytes"] != size
                or sent.allocation_id == received.allocation_id
                or sent.view_id != message.attributes["symbolic_tensor_inputs"][0]
                or received.view_id != message.attributes["symbolic_tensor_outputs"][0]
                or any(view.shape != wire_shape or view.strides_bytes != wire_strides
                    or view.offset_bytes != lane * size // tp or view.dtype != "bf16" for view in (sent, received))
                or not any(view.view_id in post.attributes["symbolic_tensor_outputs"]
                    and view.allocation_id == sent.allocation_id and view.shape == (rows, c, h)
                    and view.strides_bytes == (c * h * 2, h * 2, 2) and view.offset_bytes == 0 for view in output_views)):
            raise ValueError("V4 PP lane must preserve its expanded mHC views and independent full backing")
        children = tuple(graph.node(edge.destination) for edge in graph.edges
            if edge.source == message.node_id and edge.kind == "data")
        if len(children) != 1:
            raise ValueError("V4 PP lane must converge on one destination gather")
        if tp == 1:
            copies = children
            input_ports = message.attributes["symbolic_tensor_outputs"]
            if copies[0].rank != destination[0]:
                raise ValueError("V4 TP1 hidden must enter its destination runner copy")
        else:
            gather = children[0]
            gather_ids.add(gather.node_id)
            attrs = gather.attributes
            if (gather.kind is not WorkloadKind.COLLECTIVE or attrs["operation"] != "all_gather"
                    or tuple(attrs["participants"]) != destination or attrs["bytes"] != size // tp
                    or attrs["symbolic_collective_storage"] != "fresh" or attrs["symbolic_collective_dim"] != 0
                    or len(attrs["symbolic_tensor_outputs"]) != tp):
                raise ValueError("V4 pipeline needs a fresh all-gather over its destination TP group")
            incoming = {edge.source for edge in graph.edges if edge.destination == gather.node_id and edge.kind == "data"}
            if incoming != {node.node_id for node in lanes} or set(attrs["symbolic_tensor_inputs"]) != {
                    node.attributes["symbolic_tensor_outputs"][0] for node in lanes}:
                raise ValueError("V4 gather must bind exactly the received lane ports")
            copies = tuple(graph.node(edge.destination) for edge in graph.edges
                if edge.source == gather.node_id and edge.kind == "data")
            if len(copies) != tp or {node.rank for node in copies} != set(destination):
                raise ValueError("V4 gathered hidden needs one runner copy per destination rank")
            input_ports = attrs["symbolic_tensor_outputs"]
        for copy in copies:
            index = destination.index(copy.rank)
            if (copy.kind is not WorkloadKind.GATHER or copy.variant != "gather" or copy.dtype != "bf16"
                    or dict(copy.shape) != {"elements": rows * c * h, "rows": rows, "hidden": c * h}
                    or copy.attributes["symbolic_tensor_inputs"] != (input_ports[index],)
                    or not copy.node_id.endswith(":receive:runner-copy:hidden_states")):
                raise ValueError("V4 runner copy must retain its own expanded hidden gather")
            consumers = tuple(graph.node(edge.destination) for edge in graph.edges
                if edge.source == copy.node_id and edge.kind == "data")
            prenorms = tuple(node for node in consumers if node.variant == "mhc_prenorm")
            if len(prenorms) != 1 or prenorms[0].rank != copy.rank:
                raise ValueError("V4 copied hidden must enter its same-rank nonbroadcast prenorm")
            prenorm = prenorms[0]
            if prenorm.attributes["symbolic_tensor_inputs"] != copy.attributes["symbolic_tensor_outputs"]:
                raise ValueError("V4 stage-entry prenorm must consume its actual copied port")
    if len(gather_ids) != (1 if tp > 1 else 0):
        raise ValueError("V4 boundary lanes must converge on the same all-gather")
    return posts


def _validate_v4_pipeline_graph(graph, pool, messages):
    """V4 mHC transport over each active replica, with global stage frontiers.

    DP expert collectives share a stage across replicas. Idle replicas keep
    expert dependencies but have no embedding, PP payload or model head.
    This validates topology only, without extending operator pricing scope.
    """
    config = pool.parallel_plan.config
    tp, dp, pp = config.tensor_parallel_size, config.data_parallel_size, config.pipeline_parallel_size
    if any(not _symbolic_v4_pipeline_topology_projection(node, pool) for node in messages):
        raise ValueError("V4 pipeline requires one complete hidden lane per boundary/rank")
    embeddings = tuple(node for node in graph.nodes if node.pool_name == pool.name
        and node.kind in (WorkloadKind.GATHER, WorkloadKind.VECTOR) and node.node_id.endswith(":embedding"))
    rows_by_replica, hidden_geometry = [], set()
    for replica in range(dp):
        first_stage = tuple(pool.physical_rank_for_logical_rank(replica * pp * tp + lane) for lane in range(tp))
        entries = tuple(node for node in embeddings if node.rank in first_stage)
        if not entries:
            rows_by_replica.append(0)
            continue
        if (len(entries) != tp or {node.rank for node in entries} != set(first_stage)
                or any(not _symbolic_v4_pipeline_topology_projection(node, pool) for node in entries)
                or len({node.shape["rows"] for node in entries}) != 1):
            raise ValueError("V4 pipeline requires matching original embedding rows on each active TP lane")
        dummy = tuple("source_dummy_run" in node.attributes and node.attributes["source_dummy_run"] is True for node in entries)
        if any(dummy) and (not all(dummy) or any(node.attributes["source_actual_request_count"] != 0 for node in entries)):
            raise ValueError("V4 dummy pipeline requires the same non-request declaration on every lane")
        if any(dummy):
            owned = {pool.physical_rank_for_logical_rank(replica * pp * tp + offset) for offset in range(pp * tp)}
            if any(node.pool_name == pool.name and node.rank in owned and (node.kind is WorkloadKind.LM_HEAD
                    or node.kind is WorkloadKind.KV_UPDATE and node.attributes["cache_write_valid_rows"] != 0) for node in graph.nodes):
                raise ValueError("V4 dummy pipeline cannot sample or write a real request cache")
        rows_by_replica.append(0 if any(dummy) else entries[0].shape["rows"])
        prelude = tuple(node for node in graph.nodes if node.pool_name == pool.name
            and node.rank in first_stage and node.node_id.endswith(":layer-0:pre"))
        if (len(prelude) != tp or {node.rank for node in prelude} != set(first_stage)
                or any(node.kind is not WorkloadKind.VECTOR or node.variant != "mhc_pre"
                    or node.shape["elements"] != entries[0].shape["rows"] * node.shape["hidden"] * node.shape["hc_mult"]
                    for node in prelude)):
            raise ValueError("V4 pipeline embedding rows must enter the original expanded mHC prelude")
        hidden_geometry.update((node.shape["hidden"], node.shape["hc_mult"]) for node in prelude)
    if len(hidden_geometry) != 1 or len(messages) != sum(bool(rows) for rows in rows_by_replica) * tp * (pp - 1):
        raise ValueError("V4 pipeline requires one complete hidden lane per boundary/rank of each active replica")
    (hidden, hc_mult), = hidden_geometry
    for stage in range(pp - 1):
        posts = set()
        for replica, rows in enumerate(rows_by_replica):
            source = tuple(pool.physical_rank_for_logical_rank((replica * pp + stage) * tp + lane) for lane in range(tp))
            destination = tuple(pool.physical_rank_for_logical_rank((replica * pp + stage + 1) * tp + lane) for lane in range(tp))
            lanes = tuple(node for node in messages if node.rank in source)
            if not rows:
                if lanes:
                    raise ValueError("idle V4 replica cannot send a pipeline payload")
                continue
            posts.update(_validate_v4_pipeline_boundary(graph, pool, lanes, source, destination,
                expected_rows=rows, expected_hidden=hidden, expected_hc_mult=hc_mult))
        stage_ranks = set(pool.physical_ranks_for_pipeline_stage(stage))
        active_stage_ranks = {pool.physical_rank_for_logical_rank((replica * pp + stage) * tp + lane)
            for replica, count in enumerate(rows_by_replica) if count for lane in range(tp)}
        stage_nodes = {node.node_id for node in graph.nodes if node.pool_name == pool.name and node.kind is not WorkloadKind.P2P
            and (set(node.attributes["participants"]) <= stage_ranks
                and bool(set(node.attributes["participants"]) & active_stage_ranks) if node.kind is WorkloadKind.COLLECTIVE
                else node.rank in active_stage_ranks)}
        nonterminals = {edge.source for edge in graph.edges if edge.source in stage_nodes and edge.destination in stage_nodes}
        if stage_nodes - nonterminals != posts:
            raise ValueError("V4 stage must finish all its compute before sending its completed mHC hidden: "
                f"extra={sorted(stage_nodes - nonterminals - posts)}, missing={sorted(posts - (stage_nodes - nonterminals))}")
    for edge in graph.edges:
        left, right = graph.node(edge.source), graph.node(edge.destination)
        if left.pool_name != pool.name or right.pool_name != pool.name:
            continue
        def stages(node):
            ranks = node.attributes["participants"] if node.kind is WorkloadKind.COLLECTIVE else (node.rank,)
            return {pool.logical_rank_for_physical_rank(rank) // tp % pp for rank in ranks}
        left_stages, right_stages = stages(left), stages(right)
        if (len(left_stages) != 1 or len(right_stages) != 1 or (left_stages != right_stages
                and (left.kind is not WorkloadKind.P2P or right_stages != {left.attributes["destination_stage"]}))):
            raise ValueError("V4 cross-stage dependency must pass through its declared PP message")
        if (left.kind is not WorkloadKind.COLLECTIVE and right.kind is not WorkloadKind.COLLECTIVE
                and pool.logical_rank_for_physical_rank(left.rank) // (tp * pp)
                != pool.logical_rank_for_physical_rank(right.rank) // (tp * pp)):
            raise ValueError("V4 cross-replica dependency must pass through its declared collective")
        if (left.kind is WorkloadKind.P2P and right.kind is WorkloadKind.COLLECTIVE
                and any(pool.logical_rank_for_physical_rank(rank) // (tp * pp)
                    != pool.logical_rank_for_physical_rank(left.rank) // (tp * pp) for rank in right.attributes["participants"])):
            raise ValueError("V4 pipeline receive cannot enter another replica's collective")


def _validate_tp2_pp2_pipeline_graph(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
) -> None:
    """Validate the complete pinned vLLM TP=2, PP=2 boundary."""

    for pool in deployment.worker_pools.values():
        if not any(node.pool_name == pool.name and node.kind is not WorkloadKind.PD_KV_TRANSFER
                for node in graph.nodes):
            continue
        parallelism = pool.parallel_plan.config
        glm = tuple(node for node in graph.nodes if node.pool_name == pool.name and node.kind is WorkloadKind.P2P)
        if _is_k3_source_pipeline_graph(graph, pool):
            _validate_k3_pipeline_graph(graph, pool, glm)
            continue
        if parallelism.pipeline_parallel_size > 1 and any(node.pool_name == pool.name and _symbolic_kimi_dp_projection(node, pool) for node in graph.nodes):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if glm and all(_symbolic_mla_dp_pipeline_projection(node, pool) for node in glm):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if (parallelism.tensor_parallel_size > 1 and parallelism.pipeline_parallel_size > 1
                and any(node.pool_name == pool.name and _symbolic_v3_parallel_projection(node, pool) for node in graph.nodes)):
            if len(glm) != 2 * parallelism.tensor_parallel_size * (parallelism.pipeline_parallel_size - 1):
                raise ValueError("V3 pipeline requires both hidden/residual lanes at every boundary")
            _validate_symbolic_sliced_pipeline_graph(graph, pool, glm)
            continue
        if parallelism.pipeline_parallel_size > 1 and any(node.pool_name == pool.name
                and "model_source_profile" in node.attributes
                and node.attributes["model_source_profile"] == "vllm_0_30_v4_sync_fp8_ds_mla"
                for node in graph.nodes):
            _validate_v4_pipeline_graph(graph, pool, glm)
            continue
        if (parallelism.tensor_parallel_size > 1 and parallelism.pipeline_parallel_size > 1
                and glm and all(_symbolic_glm_pcp_projection(node, pool) for node in glm)):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if parallelism.tensor_parallel_size > 1 and parallelism.pipeline_parallel_size > 1 and _qwen030_dp_sp(parallelism):
            _validate_symbolic_dp_pipeline_graph(graph, pool)
            continue
        if (parallelism.tensor_parallel_size > 1 and parallelism.pipeline_parallel_size > 1
                and parallelism.is_rank_local and deployment.mode in ("colocated", "pd_disagg")
                and glm and any(node.pool_name == pool.name and "source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_gqa_mathematical" for node in graph.nodes)):
            if any("symbolic_tensor_inputs" not in node.attributes for node in glm):
                raise ValueError("GQA source pipeline requires complete typed boundaries")
            # Each routed replica owns a complete local PP graph. Validate
            # its actual lanes/ports at every width and every PP boundary.
            _validate_symbolic_sliced_pipeline_graph(graph, pool, glm)
            continue
        if (deployment.mode == "pd_disagg" and parallelism.tensor_parallel_size > 1
                and parallelism.pipeline_parallel_size > 1 and parallelism.is_rank_local):
            p2p_nodes = tuple(node for node in graph.nodes if node.pool_name == pool.name
                and node.kind is WorkloadKind.P2P)
            if not p2p_nodes or any("symbolic_tensor_inputs" not in node.attributes for node in p2p_nodes):
                raise ValueError("PD pipeline requires complete typed source boundaries")
            _validate_symbolic_sliced_pipeline_graph(graph, pool, p2p_nodes)
            continue
        if parallelism.tensor_parallel_size == 8 and parallelism.pipeline_parallel_size > 1:
            projection_nodes = tuple(node for node in graph.nodes if node.pool_name == pool.name
                and node.kind is WorkloadKind.P2P and "symbolic_tensor_inputs" in node.attributes)
            if projection_nodes:
                if (deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                        or len(projection_nodes) != 16 * (parallelism.pipeline_parallel_size - 1)
                        or any(not _symbolic_mla_pipeline_projection(node, pool) for node in graph.nodes
                            if node.pool_name == pool.name and node.kind is not WorkloadKind.COLLECTIVE)):
                    raise ValueError("GLM TP8 PP source projection requires all complete declared lane boundaries")
                _validate_symbolic_sliced_pipeline_graph(graph, pool, projection_nodes)
                continue
        if (parallelism.tensor_parallel_size,
                parallelism.pipeline_parallel_size) != (2, 2):
            continue
        if (deployment.mode != "colocated"
                or len(deployment.worker_pools) != 1
                or parallelism.worker_count != 4
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "TP=2, PP=2 lowering supports exactly one colocated pool "
                "with EP=SP=CP=DP=1")
        source_ranks = pool.physical_ranks_for_pipeline_stage(0)
        destination_ranks = pool.physical_ranks_for_pipeline_stage(1)
        if len(source_ranks) != 2 or len(destination_ranks) != 2:
            raise RuntimeError(
                "TP=2, PP=2 lowering requires two ranks per pipeline stage")
        p2p_nodes = tuple(
            node for node in graph.nodes
            if node.pool_name == pool.name and node.kind is WorkloadKind.P2P
        )
        if len(p2p_nodes) != 4:
            raise ValueError(
                "TP=2, PP=2 boundary requires exactly four P2P lane nodes")
        if any("symbolic_tensor_inputs" in node.attributes for node in p2p_nodes):
            _validate_symbolic_sliced_pipeline_graph(graph, pool, p2p_nodes)
            continue
        expected_pairs = tuple(zip(source_ranks, destination_ranks))
        nodes_by_tensor = {
            "hidden_states": [],
            "residual": [],
        }
        for node in p2p_nodes:
            _validate_pipeline_p2p_node(node, pool)
            attributes = node.attributes
            if (attributes["source_stage"],
                    attributes["destination_stage"]) != (0, 1):
                raise ValueError(
                    "TP=2, PP=2 P2P nodes must cross stage 0 to stage 1")
            pair = (
                attributes["source_rank"],
                attributes["destination_rank"],
            )
            if pair not in expected_pairs:
                raise ValueError(
                    "TP=2, PP=2 P2P endpoint pair is not a canonical PP lane")
            tensor_name = attributes["payload_tensor"]["name"]
            if tensor_name not in nodes_by_tensor:
                raise ValueError(
                    "TP=2, PP=2 boundary payload must be hidden_states or "
                    "residual")
            nodes_by_tensor[tensor_name].append(node)

        all_gathers = {}
        for tensor_name in ("hidden_states", "residual"):
            tensor_nodes = tuple(nodes_by_tensor[tensor_name])
            if len(tensor_nodes) != 2:
                raise ValueError(
                    "TP=2, PP=2 boundary requires two P2P lanes per tensor")
            actual_pairs = tuple(
                (
                    node.attributes["source_rank"],
                    node.attributes["destination_rank"],
                )
                for node in tensor_nodes
            )
            if actual_pairs != expected_pairs:
                raise ValueError(
                    "TP=2, PP=2 tensor lanes must follow ordered TP rank "
                    "pairs")
            successor_sets = tuple(
                {
                    edge.destination
                    for edge in graph.edges
                    if edge.source == node.node_id
                }
                for node in tensor_nodes
            )
            if (any(len(successors) != 1 for successors in successor_sets)
                    or successor_sets[0] != successor_sets[1]):
                raise ValueError(
                    "both TP=2, PP=2 P2P lanes must converge on exactly one "
                    "tensor all-gather")
            collective_id = next(iter(successor_sets[0]))
            collective = graph.node(collective_id)
            if (collective.kind is not WorkloadKind.COLLECTIVE
                    or collective.attributes["operation"] != "all_gather"
                    or collective.attributes["bytes_semantics"]
                    != "per_rank_input"
                    or tuple(collective.attributes["participants"])
                    != destination_ranks):
                raise ValueError(
                    "TP=2, PP=2 P2P lanes must feed the destination TP "
                    "all-gather")
            if tuple(graph.predecessors(collective_id)) != tuple(
                    node.node_id for node in tensor_nodes):
                raise ValueError(
                    "TP=2, PP=2 all-gather must wait for both matching P2P "
                    "lanes")
            barrier_edges = tuple(
                edge for edge in graph.edges
                if edge.destination == collective_id
            )
            if any(edge.kind != "barrier" for edge in barrier_edges):
                raise ValueError(
                    "TP=2, PP=2 P2P-to-all-gather dependencies must be "
                    "barriers")
            if (collective.attributes["direction"] != "collective"
                    or collective.attributes["source_rank"] is not None
                    or collective.attributes["destination_rank"] is not None
                    or tuple(collective.attributes["ring_order"])
                    != destination_ranks
                    or collective.attributes["reduction"] is not None):
                raise ValueError(
                    "TP=2, PP=2 boundary all-gather attributes are not "
                    "canonical")
            all_gathers[tensor_name] = collective
        if (all_gathers["hidden_states"].node_id
                == all_gathers["residual"].node_id):
            raise ValueError(
                "hidden_states and residual require distinct all-gathers")
        destination_group_all_gathers = tuple(
            node for node in graph.nodes
            if (node.pool_name == pool.name
                and node.kind is WorkloadKind.COLLECTIVE
                and node.attributes["operation"] == "all_gather"
                and tuple(node.attributes["participants"])
                == destination_ranks)
        )
        if len(destination_group_all_gathers) != 3:
            raise ValueError(
                "TP=2, PP=2 graph requires exactly two boundary all-gathers "
                "and one destination-stage LM-head all-gather")
        boundary_all_gather_ids = {
            all_gathers["hidden_states"].node_id,
            all_gathers["residual"].node_id,
        }
        lm_head_all_gathers = tuple(
            node for node in destination_group_all_gathers
            if node.node_id not in boundary_all_gather_ids
        )
        if len(lm_head_all_gathers) != 1:
            raise ValueError(
                "TP=2, PP=2 graph requires one non-boundary destination "
                "all-gather")
        lm_head_all_gather = lm_head_all_gathers[0]
        lm_head_predecessors = tuple(
            graph.node(node_id)
            for node_id in graph.predecessors(lm_head_all_gather.node_id)
        )
        if (len(lm_head_predecessors) != 2
                or tuple(node.rank for node in lm_head_predecessors)
                != destination_ranks
                or any(node.kind is not WorkloadKind.LM_HEAD
                       for node in lm_head_predecessors)
                or any(
                    edge.kind != "barrier"
                    for edge in graph.edges
                    if edge.destination == lm_head_all_gather.node_id
                )
                or any(edge.source == lm_head_all_gather.node_id
                       for edge in graph.edges)):
            raise ValueError(
                "TP=2, PP=2 destination LM-head all-gather topology is not "
                "canonical")

        destination_successors = []
        for tensor_name in ("hidden_states", "residual"):
            collective = all_gathers[tensor_name]
            release_edges = tuple(
                edge for edge in graph.edges
                if edge.source == collective.node_id
            )
            if any(edge.kind != "data" for edge in release_edges):
                raise ValueError(
                    "TP=2, PP=2 all-gather must release destination compute "
                    "through data dependencies")
            destination_successors.append(tuple(
                edge.destination for edge in release_edges))
        if (destination_successors[0] != destination_successors[1]
                or len(destination_successors[0]) != 2):
            raise ValueError(
                "both TP=2, PP=2 boundary all-gathers must release the same "
                "two destination-rank compute nodes")
        destination_nodes = tuple(
            graph.node(node_id) for node_id in destination_successors[0])
        if tuple(node.rank for node in destination_nodes) != destination_ranks:
            raise ValueError(
                "TP=2, PP=2 boundary destination nodes disagree with the "
                "destination TP group")
        if any(
                node.kind is not WorkloadKind.NORM
                or node.dtype != "bf16"
                or set(node.shape) != {"rows", "hidden"}
                or node.shape["rows"] <= 0
                or node.shape["hidden"] <= 0
                for node in destination_nodes):
            raise ValueError(
                "TP=2, PP=2 boundary must release two BF16 decoder input "
                "normalization nodes")
        first_shape = dict(destination_nodes[0].shape)
        if any(dict(node.shape) != first_shape for node in destination_nodes[1:]):
            raise ValueError(
                "TP=2, PP=2 destination-rank input shapes must match")
        for node in destination_nodes:
            successor_edges = tuple(
                edge for edge in graph.edges if edge.source == node.node_id)
            if (len(successor_edges) != 1
                    or successor_edges[0].kind != "data"):
                raise ValueError(
                    "each TP=2, PP=2 destination input norm must feed one "
                    "rank-local QKV node")
            qkv = graph.node(successor_edges[0].destination)
            if (qkv.kind is not WorkloadKind.LINEAR
                    or qkv.rank != node.rank
                    or qkv.dtype != "bf16"
                    or qkv.variant != "gemm"
                    or set(qkv.shape) != {"b", "m", "n", "k"}
                    or qkv.shape["b"] != 1
                    or qkv.shape["m"] != first_shape["rows"]
                    or qkv.shape["k"] != first_shape["hidden"]):
                raise ValueError(
                    "TP=2, PP=2 destination input norm shape disagrees with "
                    "its rank-local QKV successor")
        all_gather_ids = tuple(
            all_gathers[name].node_id
            for name in ("hidden_states", "residual")
        )
        if any(tuple(graph.predecessors(node.node_id)) != all_gather_ids
               for node in destination_nodes):
            raise ValueError(
                "each TP=2, PP=2 destination node must wait for both tensor "
                "all-gathers")

        source_predecessors = tuple(
            graph.predecessors(node.node_id) for node in p2p_nodes)
        if (any(len(predecessors) != 1
                for predecessors in source_predecessors)
                or len({predecessors[0]
                        for predecessors in source_predecessors}) != 1):
            raise ValueError(
                "all TP=2, PP=2 P2P lanes must share one completed source "
                "frontier")
        ready_edges = tuple(
            edge for edge in graph.edges
            if edge.destination in {node.node_id for node in p2p_nodes}
        )
        if (len(ready_edges) != 4
                or any(edge.kind != "ready" for edge in ready_edges)):
            raise ValueError(
                "TP=2, PP=2 source-to-P2P dependencies must contain four "
                "ready edges")
        source_frontier = graph.node(source_predecessors[0][0])
        if (source_frontier.kind is not WorkloadKind.COLLECTIVE
                or source_frontier.attributes["operation"] != "all_reduce"
                or tuple(source_frontier.attributes["participants"])
                != source_ranks):
            raise ValueError(
                "TP=2, PP=2 P2P lanes must follow the source-stage TP "
                "all-reduce frontier")
        source_compute_nodes = tuple(
            graph.node(node_id)
            for node_id in graph.predecessors(source_frontier.node_id)
        )
        if (len(source_compute_nodes) != 2
                or tuple(node.rank for node in source_compute_nodes)
                != source_ranks
                or any(
                    node.kind is not WorkloadKind.LINEAR
                    or node.dtype != "bf16"
                    or node.variant != "gemm"
                    or set(node.shape) != {"b", "m", "n", "k"}
                    or node.shape["b"] != 1
                    for node in source_compute_nodes)
                or any(
                    edge.kind != "barrier"
                    for edge in graph.edges
                    if edge.destination == source_frontier.node_id
                )):
            raise ValueError(
                "TP=2, PP=2 source frontier must follow two rank-local BF16 "
                "linear nodes")
        source_shape = {
            "rows": source_compute_nodes[0].shape["m"],
            "hidden": source_compute_nodes[0].shape["n"],
        }
        if (any(node.shape["m"] != source_shape["rows"]
                or node.shape["n"] != source_shape["hidden"]
                for node in source_compute_nodes[1:])
                or first_shape != source_shape):
            raise ValueError(
                "TP=2, PP=2 source linear, destination norm, and QKV "
                "tensor shapes must agree")

        total_elements = source_shape["rows"] * source_shape["hidden"]
        if total_elements % 2:
            raise ValueError(
                "TP=2, PP=2 boundary tensor elements must divide TP size")
        expected_wire_elements = total_elements // 2
        expected_wire_bytes = expected_wire_elements * 2
        if source_frontier.attributes["bytes"] != total_elements * 2:
            raise ValueError(
                "TP=2, PP=2 source all-reduce bytes disagree with the "
                "source [T,H] tensor")
        for node in p2p_nodes:
            payload = node.attributes["payload_tensor"]
            if (tuple(payload["shape"]) != (expected_wire_elements,)
                    or payload["bytes"] != expected_wire_bytes
                    or node.attributes["bytes"] != expected_wire_bytes):
                raise ValueError(
                    "TP=2, PP=2 P2P wire shape/bytes disagree with the "
                    "destination [T,H] tensor and TP size")
        for collective in all_gathers.values():
            if collective.attributes["bytes"] != expected_wire_bytes:
                raise ValueError(
                    "TP=2, PP=2 all-gather bytes disagree with the canonical "
                    "per-rank wire slice")


def _eager_attn_specs(node: WorkloadNode, hardware_id: str, hardware):
    required = ("batch", "q_len", "kv_len", "head_dim", "num_query_heads")
    missing = [name for name in required if name not in node.shape]
    if missing:
        raise ValueError(f"eager attention missing shape fields: {missing}")
    b = node.shape["batch"]
    h = node.shape["num_query_heads"]
    q = node.shape["q_len"]
    kv = node.shape["kv_len"]
    d = node.shape["head_dim"]
    shapes = (
        {"b": b * h, "m": q, "n": kv, "k": d},
        {"element_count": b * h * q * kv},
        {"b": b * h, "m": q, "n": d, "k": kv},
    )
    result = []
    common_attributes = dict(node.attributes)
    if "launch_policy" not in common_attributes:
        raise ValueError(
            "eager attention lowering requires explicit launch_policy")
    for name in ("tensor_parallel_size", "expert_parallel_size"):
        if name not in common_attributes:
            raise ValueError(
                f"eager attention lowering requires explicit {name}")
    for index, (kind, shape, variant) in enumerate((
        (KernelKind.GEMM, shapes[0], "gemm"),
        (KernelKind.VECTOR, shapes[1], "softmax"),
        (KernelKind.GEMM, shapes[2], "gemm"),
    )):
        selection_attributes = dict(common_attributes)
        if index in (0, 2):
            selection_attributes.update({
                "a_layout": "row_major",
                "b_layout": "row_major",
                "c_layout": "row_major",
                "trans_a": False,
                "trans_b": index == 0,
                "epilogue": "none",
                "has_bias": False,
                "position": (
                    "attention_qk" if index == 0 else "attention_pv"),
            })
        launch = select_launch_config(
            hardware_id, kind, shape, node.dtype, variant,
            selection_attributes, hardware=hardware)
        spec = KernelSpec(
            kind, hardware_id, shape, dtype=node.dtype,
            variant=launch.execution_variant,
            launch=launch)
        if kind is KernelKind.GEMM:
            result.extend(_gemm_specs(
                spec, shape, variant, selection_attributes,
                hardware_id, hardware))
        elif kind is KernelKind.VECTOR:
            result.append(spec)
        else:
            raise ValueError(
                f"unhandled eager-attention kernel kind: {kind.value}")
    return tuple(result)


def _h200_bf16_expert_math_domain(node, hardware_id):
    return (HWSpec.source_contract_supported("unbound_bf16_expert_source_math", hardware_id) and node.kind is WorkloadKind.FUSED_MOE
        and node.variant == "full" and node.dtype == "bf16"
        and "source_component" in node.attributes
        and node.attributes["source_component"] == "symbolic_unquantized_experts"
        and {"model_execution_version", "launch_policy", "expert_parallel_size",
            "routing_weight_dtype", "expert_parameter_layout"}.issubset(node.attributes)
        and node.attributes["model_execution_version"] in ("vllm_0_11_0", "vllm_0_30_0")
        and node.attributes["launch_policy"] == "analytical_launch"
        # Useful expert math consumes the already validated per-rank routing
        # and exact linear ownership, including ordinary TP/EP groups >2.
        # DP/SP collectives retain their separate graph validation; they are
        # not prerequisites for an ordinary single-replica expert group.
        and (node.attributes["expert_parallel_size"] in (1, 2)
            or node.attributes["model_execution_version"] == "vllm_0_30_0"
            and type(node.attributes["expert_parallel_size"]) is int
            and node.attributes["expert_parallel_size"] > 2
            and not {"source_moe_config", "source_routing", "source_batch_invariant", "source_has_expert_map"}.intersection(node.attributes)
            and (not {"source_dp_logical_rows", "source_dp_request_rows", "source_dp_lane_rows",
                    "source_dp_row_order", "source_tensor_parallel_size", "source_padding_routing"}.intersection(node.attributes)
                or {"source_dp_logical_rows", "source_dp_request_rows", "source_dp_lane_rows",
                    "source_dp_row_order", "source_tensor_parallel_size", "source_padding_routing"}.issubset(node.attributes)
                and node.attributes["source_padding_routing"] == "declared_routing_for_physical_rows_including_padding_not_observed_topk"))
        and node.attributes["routing_weight_dtype"] == "fp32"
        and node.attributes["expert_parameter_layout"] == "unquantized_checkpoint_expert_major_nk"
        and "graph_runtime_mode" not in node.attributes)


def _explicit_source_expert_specs(node, hardware_id, hardware, routing_relabelling=None):
    """Emit the same expert-axis representative that names the memo key."""
    if "bf16_expert_model" in node.attributes:
        if (node.attributes["bf16_expert_model"] != "triton_reference" or hardware.prepared_moe_operand_resources is None
                or node.dtype != "bf16" or node.attributes["model_execution_version"] != "vllm_0_30_0"
                or node.attributes["source_component"] != "symbolic_unquantized_experts"
                or node.attributes["launch_policy"] != "analytical_launch"
                or node.attributes["expert_parallel_size"] != 2
                or node.attributes["routing_weight_dtype"] != "fp32"
                or not 0 < node.shape["tokens"] <= 32768):
            raise ValueError("declared BF16 expert reference requires its single-chunk0.30 EP2 source")
        # A declared reference uses the existing source factory. The router
        # has already produced weights/IDs and remains a separate call.
        # Keeping every expert stage also restores each Host enqueue removed
        # from the independent fused-experts wrapper prepare measurement.
        source = {**node.attributes, "source_moe_config": "bundled_or_default",
            "source_routing": "softmax", "source_batch_invariant": False,
            "source_has_expert_map": True, "source_input_layout": "contiguous_aligned_16",
            "source_topk_id_dtype": "int32", "source_router_state": "prepared_topk_weights_and_ids",
            "routed_weight_dtype": "bf16", "routed_scaling_factor": 1.0, "weight_group_size": 0,
            "routing_input_dtype": "bf16", "tensor_iterator_source": "pytorch_2_13_sm90"}
        return _fused_moe_specs(node.with_attributes(source), hardware_id, hardware, routing_relabelling)
    if "symbolic_zero_extent_parameter_roles" in node.attributes:
        if (node.attributes["owned_experts"] or sum(node.attributes["local_tokens_per_expert"])
                or node.attributes["model_execution_version"] != "vllm_0_30_0"):
            raise ValueError("empty expert rank requires an actual empty zero contribution")
        fill = WorkloadNode(node.node_id, WorkloadKind.ELEMENTWISE, node.pool_name,
            {"elements": node.shape["tokens"] * node.shape["hidden"]}, dtype=node.dtype, variant="elementwise",
            rank=node.rank, attributes={"operation": "fill", "reads": 0, "writes": 1,
                "launch_policy": node.attributes["launch_policy"]})
        if HWSpec.source_contract_supported("bf16_symbolic_vector_source_math", hardware_id) and node.dtype == "bf16":
            return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                shape={"element_count": fill.shape["elements"]}, dtype="bf16", variant="elementwise",
                attributes={"operation": "fill", "reads": 0, "writes": 1,
                    "mathematical_program": "dense_bf16_source_flops_bytes_v1"}, launch=None,
                execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
        return (_compute_spec(fill, hardware_id, hardware),)
    native_source_facts = {"model_execution_version", "source_moe_config", "source_routing",
        "source_batch_invariant", "source_has_expert_map", "source_input_layout",
        "source_topk_id_dtype", "routed_weight_dtype", "routing_input_dtype"}
    legacy_recipe_facts = {"source_router_renormalize", "source_moe_chunk_size"}
    if (HWSpec.source_contract_supported("vllm_unquantized_prepared_expert_source_recipe", hardware_id) and native_source_facts.issubset(node.attributes)
            and (node.attributes["model_execution_version"] == "vllm_0_30_0"
                or node.attributes["model_execution_version"] == "vllm_0_11_0"
                and legacy_recipe_facts.issubset(node.attributes))):
        source = dict(node.attributes)
        if "source_router_state" in source and source["source_router_state"] != "prepared_topk_weights_and_ids":
            raise ValueError("explicit source experts consume existing top-k weights and IDs")
        source["source_router_state"] = "prepared_topk_weights_and_ids"
        return _fused_moe_specs(node.with_attributes(source), hardware_id, hardware, routing_relabelling)
    if _h200_bf16_expert_math_domain(node, hardware_id):
        attributes = _fused_moe_prediction_attributes(node, routing_relabelling)
        return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id, shape=node.shape,
            dtype="bf16", variant="routed_experts", attributes={**attributes,
                "routed_weight_dtype": "bf16", "weight_group_size": 0, "routed_scaling_factor": 1.0,
                "mathematical_program": ("unquantized_bf16_swiglu_v1" if attributes["expert_parallel_size"] == 1
                    else "unquantized_bf16_legacy_grouped_swiglu_v1"
                    if node.attributes["model_execution_version"] == "vllm_0_11_0"
                    else "unquantized_bf16_grouped_swiglu_v1")}, launch=None,
            execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
    specs = _fused_moe_specs(node, hardware_id, hardware, routing_relabelling)
    if "routing_weight_dtype" in node.attributes and node.attributes["routing_weight_dtype"] == "bf16":
        if (not hardware.supports_source_contract("vllm_0_18_bf16_routed_expert_native_projection")
                or any(spec.variant in ("moe_gating_topk", "moe_cast") for spec in specs)):
            raise ValueError("Ascend0.18 explicit BF16 routing must not add a second route or count cast")
        return specs
    route_variant = hardware.source_routing_variant("symbolic_unquantized_experts")
    if sum(spec.kind is KernelKind.VECTOR and spec.variant == route_variant for spec in specs) != 1:
        raise ValueError("explicit routing must replace exactly one legacy top-k stage")
    return tuple(spec for spec in specs if not (spec.kind is KernelKind.VECTOR and spec.variant == route_variant))


def _node_source_launches(node, hardware_id, hardware):
    """Attach exact source stages without replacing logical cost semantics."""
    norm_copy_source = _h200_legacy_norm_copy_source(node, hardware_id)
    if norm_copy_source is not None:
        launch = select_launch_config(hardware_id, KernelKind.VECTOR,
            _vector_shape(node), node.dtype, "elementwise", norm_copy_source, hardware)
        return (("native_input_materialization", launch),)
    if (HWSpec.source_contract_supported("vllm_source_vector_launch_binding", hardware_id) and node.dtype in ("bf16", "fp16")
            and node.kind in (WorkloadKind.NORM, WorkloadKind.ACTIVATION, WorkloadKind.ROTARY, WorkloadKind.KV_UPDATE)
            and node.variant in ("rmsnorm", "add_rmsnorm", "silu", "rotary", "rope", "paged_kv_cache")
            and {"model_execution_version", "source_input_layout"}.issubset(node.attributes)
            and node.attributes["model_execution_version"] in ("vllm_0_11_0", "vllm_0_27_1", "vllm_0_30_0")):
        runtime, layout = node.attributes["model_execution_version"], node.attributes["source_input_layout"]
        norm = node.variant in ("rmsnorm", "add_rmsnorm")
        branch_bound = (not norm or runtime == "vllm_0_11_0" or (
            "source_batch_invariant" in node.attributes and node.attributes["source_batch_invariant"] is False))
        layout_bound = (layout == "contiguous_aligned_16"
            or layout == "contiguous" and (node.variant != "add_rmsnorm" or runtime != "vllm_0_11_0")
            or layout == "unit_final_stride" and (node.variant in ("rotary", "rope", "paged_kv_cache")
                or node.variant == "rmsnorm" and runtime != "vllm_0_11_0"))
        unclamped = "swiglu_limit" not in node.attributes or node.attributes["swiglu_limit"] is None
        if branch_bound and layout_bound and unclamped:
            kind = KernelKind.KV_CACHE if node.kind is WorkloadKind.KV_UPDATE else KernelKind.VECTOR
            shape = node.shape if node.kind is WorkloadKind.NORM else _vector_shape(node)
            launch = select_launch_config(hardware_id, kind, shape, node.dtype, node.variant,
                node.attributes, hardware=hardware)
            return (("native_main", launch),)
    if (HWSpec.source_contract_supported("vllm_0_30_symbolic_mtp_eager", hardware_id) and node.kind is WorkloadKind.VECTOR
            and node.variant == "vllm_mtp_eager"
            and node.attributes["source_component"] == "vllm_0_30_symbolic_mtp_eager"):
        operation = node.attributes["operation"]
        bound_triton_stage = operation in ("prepare_next_padded", "prepare_inputs_padded", "step_metadata", "rejection")
        bound_materialization = (operation in ("copy", "cast_ids", "cast_fp32", "fill", "argmax")
            and "source_geometry_program" in node.attributes
            and node.attributes["source_geometry_program"] == "vllm_0_30_torch_2_13_cuda_source"
            and tuple(node.attributes["source_geometry_input_contiguous"]) == (() if operation == "fill" else (True,))
            and (operation != "fill" or node.dtype in ("int32", "int64", "bf16", "fp32"))
            and node.attributes["source_geometry_output_contiguous"] is True)
        if bound_triton_stage or bound_materialization:
            return select_workload_source_launch_recipe(hardware_id, _vector_shape(node), node.dtype,
                node.variant, node.attributes, hardware)
    return None


def _count_sort_producer_source_inputs(graph, node, attributes):
    """Qualify only the actual0.30 unmasked, unmodified global TopK ID path.

    The ordinary GPUModelRunner source binds its None forward-context mask;
    GPU-v2 instead clears actual rows and marks a captured tail. Consume
    the actual producer's source argument when bound, otherwise require an
    explicit argument. An absent mask stays unknown.
    """
    from src.serving.workload.traced_operators import VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT
    routes = tuple(graph.node(name) for name in graph.predecessors(node.node_id)
        if graph.node(name).kind is WorkloadKind.VECTOR and graph.node(name).variant == "moe_topk")
    producer_mask = len(routes) == 1 and "source_topk_padding_mask" in routes[0].attributes
    if (not producer_mask and "source_topk_padding_mask" not in attributes
            and "source_topk_id_domain" not in attributes):
        return attributes
    if "source_topk_id_domain" in attributes:
        raise ValueError("count-sort ID domain must be derived from its actual producer")
    if (graph.source_model_execution_version != "vllm_0_30_0"
            or node.attributes["model_execution_version"] != "vllm_0_30_0"
            or node.attributes["source_component"] != "symbolic_unquantized_experts"
            or attributes["source_has_expert_map"] is not False
            or attributes["source_topk_id_dtype"] != "int32"
            or node.attributes["expert_parallel_size"] != 1):
        raise ValueError("count-sort producer proof requires the exact0.30 global INT32 no-map caller")
    domain = _graph_token_domain(node)
    if domain is not None and domain[0] != domain[1]:
        raise NotImplementedError("count-sort padded rows require their filtered-ID source")
    _validate_virtual_ep_router_inputs(graph, node)
    route, = routes
    if (route.attributes["source_component"] != "symbolic_softmax_topk"
            or route.attributes["source_program"] != "vllm_0_30_bf16_softmax_topk"
            or route.dtype != "bf16" or len(route.attributes["symbolic_tensor_outputs"]) != 3):
        raise ValueError("count-sort requires the actual unmodified0.30 softmax TopK producer")
    gate_id, = graph.predecessors(route.node_id)
    allowed = {route.node_id, *graph.predecessors(gate_id)}
    if set(graph.predecessors(node.node_id)) - allowed:
        raise ValueError("count-sort IDs have an intervening or unproved writer")
    if producer_mask:
        if (route.attributes["source_topk_padding_mask_program"] != VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT
                or route.attributes["source_topk_padding_mask"] is not None):
            raise ValueError("count-sort producer padding argument differs from its exact runner source")
        mask = route.attributes["source_topk_padding_mask"]
        if "source_topk_padding_mask" in attributes and attributes["source_topk_padding_mask"] != mask:
            raise ValueError("count-sort caller padding mask disagrees with its actual producer")
    else:
        mask = attributes["source_topk_padding_mask"]
    if mask is not None:
        if (type(mask) is not tuple or len(mask) != node.shape["tokens"]
                or any(type(value) is not bool for value in mask)):
            raise ValueError("count-sort padding mask must cover every actual producer row")
        if any(mask):
            raise NotImplementedError("count-sort masked TopK IDs require their filtered-ID source")
    return {**attributes, "source_topk_id_domain": "vllm_0_30_topk_softmax_global_int32_no_padding"}


def _bind_plan_source_launch_inputs(plan, inputs, hardware_by_id, graph):
    """Bind explicit physical MLA inputs to actual lowered call identities.

    Logical cache arenas do not prove native page/layout/build facts. Those
    are supplied explicitly here and checked against both the logical call
    shape and the versioned source selector; none are guessed from a model.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("source launch inputs require a nonempty mapping of actual call IDs")
    calls = {call.call_id: call for call in plan.calls}
    normalized = {}
    for source_id, binding in inputs.items():
        if source_id in calls:
            targets = (source_id,)
        else:
            targets = (source_id + ":0", source_id + ":1")
            if (any(target not in calls for target in targets)
                    or tuple(calls[target].kernel_spec.launch.algorithm for target in targets) != (
                        "h200_vllm030_torch213_fp16_external_aggregate_gemm",
                        "h200_vllm030_torch213_fp16_external_aggregate_reduce")
                    or any(calls[target].source_node_ids != (source_id,) for target in targets)
                    or calls[targets[1]].dependencies != (targets[0],)):
                raise ValueError("source launch inputs refer to unknown lowered call IDs")
        for target in targets:
            if target in normalized:
                raise ValueError("ambiguous duplicate source launch binding")
            normalized[target] = binding
    inputs = normalized
    if set(inputs) - set(calls):
        raise ValueError("source launch inputs refer to unknown lowered call IDs")
    updates = {}
    for call_id, binding in inputs.items():
        call = calls[call_id]
        kernel = call.kernel_spec
        if (kernel.kind in (KernelKind.GEMM, KernelKind.VECTOR)
                and "source_gemm_runtime" in kernel.attributes):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("observed FP16 GEMM binding requires complete shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if (not isinstance(shape, Mapping) or not isinstance(attrs, Mapping)
                    or set(attrs) != {"model_execution_version", "source_gemm_runtime"}
                    or len(call.source_node_ids) != 1):
                raise ValueError("observed FP16 GEMM binding requires its explicit runtime and source caller")
            node = graph.node(call.source_node_ids[0])
            if (dict(shape) != dict(node.shape)
                    or attrs["model_execution_version"] != graph.source_model_execution_version
                    or attrs["source_gemm_runtime"] != kernel.attributes["source_gemm_runtime"]
                    or node.attributes["source_gemm_runtime"] != attrs["source_gemm_runtime"]
                    or kernel.launch is None):
                raise ValueError("observed FP16 GEMM facts disagree with their actual source caller")
            updates[call_id] = dataclasses.replace(call,
                source_launches=((("modeled_aggregate_main" if kernel.kind is KernelKind.GEMM
                    else "modeled_aggregate_reducer") if kernel.launch.algorithm in (
                        "h200_vllm030_torch213_fp16_external_aggregate_gemm",
                        "h200_vllm030_torch213_fp16_external_aggregate_reduce") else "native_main", kernel.launch),))
            continue
        if (HWSpec.source_contract_supported("vllm_0_30_symbolic_dsa_sparse", kernel.hardware_id) and kernel.kind is KernelKind.SPARSE_ATTN
                and kernel.variant in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed")):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("BF16 sparse source binding requires complete shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping):
                raise TypeError("BF16 sparse source binding requires shape and attribute mappings")
            if len(call.source_node_ids) != 1:
                raise ValueError("BF16 sparse source binding requires one actual workload producer")
            node = graph.node(call.source_node_ids[0])
            if (graph.source_model_execution_version != "vllm_0_30_0"
                    or attrs["model_execution_version"] != graph.source_model_execution_version
                    or node.attributes["source_component"] != "vllm_0_30_symbolic_dsa_sparse"
                    or attrs["source_component"] != "vllm_0_30_flashmla_sparse_bf16"
                    or attrs["source_kv_cache_layout"] != "paged_bf16_rows_576"):
                raise ValueError("BF16 sparse source identity/cache disagrees with its actual model caller")
            physical = {"rows": sum(node.attributes["q_lens"]), "num_heads": node.shape["num_query_heads"],
                "qk_head_dim": node.shape["qk_head_dim"], "v_head_dim": node.shape["v_head_dim"],
                "topk": node.attributes["topk_capacity"]}
            if any(shape[name] != value for name, value in physical.items()):
                raise ValueError("BF16 sparse source shape disagrees with its actual logical call")
            strides = tuple(node.attributes["source_query_strides_bytes"])
            heads, dim = physical["num_heads"], physical["qk_head_dim"]
            if attrs["source_q_input_layout"] == "caller_padded_head_backing":
                backing = attrs["source_q_backing_heads"]
                if (type(backing) is not int or strides != (backing * dim * 2, dim * 2, 2)
                        or node.attributes["source_query_offset_bytes"] % 16):
                    raise ValueError("BF16 sparse padded-Q source disagrees with its actual view strides")
            elif attrs["source_q_input_layout"] == "contiguous_unpadded":
                if (strides != (heads * dim * 2, dim * 2, 2)
                        or (heads % 64 == 0 and node.attributes["source_query_offset_bytes"] % 16)):
                    raise ValueError("BF16 sparse tensor-Q source disagrees with its actual view strides")
            else:
                raise NotImplementedError("BF16 sparse Q layout requires its actual source view")
            dependencies = call.dependencies
            group_mode = attrs["source_index_group_mode"]
            if node.variant != "dsa_sparse_decode":
                if (group_mode != "direct_batch" or attrs["source_index_remap_state"] != "build_current_indices"
                        or attrs["source_index_remap_stream"] != "caller_stream"):
                    raise ValueError("BF16 sparse prefill/mixed requires its actual direct batch remap")
            elif group_mode == "fresh_first_layer":
                if (node.attributes["indexer_type"] != "full"
                        or attrs["source_index_remap_state"] != "build_current_indices"
                        or attrs["source_index_remap_stream"] != "index_group_side_stream"):
                    raise ValueError("BF16 sparse first index-group layer requires its actual side-stream remap")
            elif group_mode == "capacity_bypass":
                capacity = attrs["source_index_group_capacity_rows"]
                if (type(capacity) is not int or not 0 <= capacity < physical["rows"]
                        or attrs["source_index_remap_state"] != "build_current_indices"
                        or attrs["source_index_remap_stream"] != "caller_stream"):
                    raise ValueError("BF16 sparse remap bypass requires actual index-group capacity overflow")
            elif group_mode != "reuse_current_group" or attrs["source_index_remap_state"] != "provided_current_indices_and_counts":
                raise ValueError("BF16 sparse decode requires its actual index-group remap ownership")
            if attrs["source_index_remap_state"] == "provided_current_indices_and_counts":
                producer_id = attrs["source_index_producer_call_id"]
                order = tuple(calls)
                if (producer_id not in inputs or producer_id not in calls
                        or order.index(producer_id) >= order.index(call_id)):
                    raise ValueError("BF16 sparse index reuse requires an earlier bound physical-index producer")
                producer = calls[producer_id]
                prior = graph.node(producer.source_node_ids[0])
                if (producer.kernel_spec.kind is not KernelKind.SPARSE_ATTN
                        or producer.rank != call.rank
                        or prior.attributes["source_component"] != "vllm_0_30_symbolic_dsa_sparse"
                        or prior.attributes["symbolic_tensor_inputs"][2] != node.attributes["symbolic_tensor_inputs"][2]
                        or tuple(prior.attributes["q_lens"]) != tuple(node.attributes["q_lens"])
                        or tuple(prior.attributes["kv_lens"]) != tuple(node.attributes["kv_lens"])
                        or inputs[producer_id]["attributes"]["source_index_remap_state"] != "build_current_indices"
                        or inputs[producer_id]["attributes"]["source_kv_cache_layout"] != "paged_bf16_rows_576"
                        or any(inputs[producer_id]["shape"][name] != shape[name]
                            for name in ("topk", "block_size", "block_stride_rows"))):
                    raise ValueError("BF16 sparse index reuse disagrees with the actual same-index producer")
                # ForwardTrace tracks the last writer per allocation, so
                # every alias mutation advances this input generation.
                # Equal view IDs and lengths alone do not establish that
                # the producer remapped the indices this consumer reads.
                if (prior.attributes["source_topk_allocation_id"] != node.attributes["source_topk_allocation_id"]
                        or prior.attributes["source_topk_input_writer"] != node.attributes["source_topk_input_writer"]):
                    raise ValueError("BF16 sparse index reuse refers to a stale top-k writer generation")
                for owner in (prior, node):
                    writer = owner.attributes["source_topk_input_writer"]
                    if writer is not None and writer not in graph.predecessors(owner.node_id):
                        raise ValueError("BF16 sparse top-k writer generation is not an actual input dependency")
                if producer_id not in dependencies:
                    dependencies = (*dependencies, producer_id)
            hardware = require_hardware(kernel.hardware_id, hardware_by_id)
            stages = select_workload_source_launch_recipe(kernel.hardware_id, shape, kernel.dtype,
                kernel.variant, attrs, hardware)
            updates[call_id] = dataclasses.replace(call, source_launches=stages, dependencies=dependencies)
            continue
        if (HWSpec.source_contract_supported("vllm_0_30_indexer_source_launch_binding", kernel.hardware_id) and (
                (kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.variant in (
                    "dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed",
                    "dsv4_indexer_decode", "dsv4_indexer_prefill"))
                or (kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_indexer_q_rope_quant"))):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("indexer source binding requires complete shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping):
                raise TypeError("indexer source binding requires shape and attribute mappings")
            if len(call.source_node_ids) != 1:
                raise ValueError("indexer source binding requires one actual workload producer")
            node = graph.node(call.source_node_ids[0])
            if (graph.source_model_execution_version != "vllm_0_30_0"
                    or attrs["model_execution_version"] != graph.source_model_execution_version):
                raise ValueError("indexer source runtime disagrees with the actual model caller")
            query = kernel.variant == "dsv4_indexer_q_rope_quant"
            v4 = query or kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode")
            component = ("vllm_0_30_symbolic_v4_indexer_query" if query else
                "vllm_0_30_symbolic_v4_scored_indexer" if v4 else "vllm_0_30_symbolic_dsa_indexer")
            native_component = "vllm_0_30_deepseek_v4_indexer" if v4 else "vllm_0_30_sparse_attn_indexer"
            if ("source_component" not in node.attributes or node.attributes["source_component"] != component
                    or attrs["source_component"] != native_component):
                raise ValueError("indexer source identity disagrees with its actual workload producer")
            if query:
                physical = {"rows": node.shape["seq_len"], "num_heads": node.shape["num_query_heads"],
                    "head_dim": node.shape["head_dim"], "rope_dim": node.shape["rope_dim"]}
                if dict(shape) != physical:
                    raise ValueError("indexer query shape disagrees with its actual logical call")
                source_dtype = node.dtype
            else:
                if node.attributes["selection_cardinality_source"] != "declared_causal_topk":
                    raise ValueError("indexer source geometry requires the actual declared selection program")
                queries, phases = tuple(node.attributes["q_lens"]), tuple(node.attributes["request_phases"])
                decode_rows = sum(q for q, phase in zip(queries, phases, strict=True) if phase == "decode")
                physical = {"rows": sum(queries), "num_heads": node.shape["num_query_heads"],
                    "head_dim": node.shape["head_dim"], "topk": node.attributes["topk_capacity"],
                    "num_decode_tokens": decode_rows, "num_prefill_tokens": sum(queries) - decode_rows}
                if v4:
                    physical["compress_ratio"] = node.shape["compress_ratio"]
                    clear_fields = {"source_topk_clear_rows", "source_topk_clear_state", "source_topk_phase_token_start"}
                    if clear_fields & set(attrs):
                        raise ValueError("V4 shared clear is owned by the actual trace,not caller attributes")
                    clear = node.attributes["symbolic_shared_topk_clear"]
                    if clear is not None:
                        if not isinstance(clear, Mapping) or set(clear) != {"owner_node_id", "view_id", "rows"}:
                            raise ValueError("V4 shared clear requires its complete actual output binding")
                        owner = graph.node(clear["owner_node_id"])
                        peers = tuple(item for item in graph.nodes
                            if "symbolic_shared_topk_clear" in item.attributes
                            and item.attributes["symbolic_shared_topk_clear"] is not None
                            and item.attributes["symbolic_shared_topk_clear"]["owner_node_id"] == owner.node_id)
                        if (len(peers) != 2 or owner.variant != "dsv4_indexer_prefill"
                                or {item.variant for item in peers} != {"dsv4_indexer_prefill", "dsv4_indexer_decode"}
                                or any(item.rank != node.rank or item.pool_name != node.pool_name
                                    or item.attributes["source_component"] != component
                                    or item.attributes["symbolic_shared_topk_clear"] != clear for item in peers)
                                or clear["view_id"] not in owner.attributes["symbolic_tensor_outputs"]
                                or type(clear["rows"]) is not int
                                or clear["rows"] != sum(sum(item.attributes["q_lens"]) for item in peers)):
                            raise ValueError("V4 shared clear disagrees with the actual phase output owner")
                        decode_peer, = (item for item in peers if item.variant == "dsv4_indexer_decode")
                        shared_decode_rows = sum(decode_peer.attributes["q_lens"])
                        for peer in peers:
                            expected_range = ((0, shared_decode_rows) if peer is decode_peer
                                else (shared_decode_rows, clear["rows"]))
                            if tuple(peer.attributes["symbolic_topk_token_range"]) != expected_range:
                                raise ValueError("V4 shared clear phase slices must partition the actual live buffer")
                        owner_call, = (item for item in plan.calls if item.source_node_ids == (owner.node_id,))
                        if node is not owner and (owner_call.call_id not in inputs
                                or owner_call.call_id not in call.dependencies):
                            raise ValueError("V4 decode requires its bound shared clear owner dependency")
                        attrs = {**attrs, "source_topk_clear_rows": clear["rows"],
                            "source_topk_phase_token_start": node.attributes["symbolic_topk_token_range"][0],
                            "source_topk_clear_state": ("owned_current_invocation" if node is owner
                                else "provided_current_invocation")}
                for name, value in physical.items():
                    if shape[name] != value:
                        raise ValueError("indexer source shape disagrees with its actual logical call: " + name)
                if physical["num_prefill_tokens"]:
                    ratio = node.shape["compress_ratio"] if v4 else 1
                    ranges, cursor = [], (attrs["source_topk_phase_token_start"]
                        if v4 and "source_topk_phase_token_start" in attrs else 0)
                    for q, length, phase in zip(queries, node.attributes["kv_lens"], phases, strict=True):
                        if phase == "prefill":
                            ranges.append((cursor, cursor + q, length // ratio))
                        cursor += q
                    for chunk in attrs["source_prefill_chunks"]:
                        start, end = chunk["token_start"], chunk["token_end"]
                        # A chunk flattens the full KV extent of every
                        # overlapping request. Query slicing does not slice
                        # those contexts (indexer.py build_prefill_chunk_metadata).
                        required = sum(length for first, last, length in ranges
                            if start < last and first < end)
                        if (type(chunk["kv_rows"]) is not int or chunk["kv_rows"] < required
                                or (v4 and (type(chunk["local_total_seq_lens"]) is not int
                                    or chunk["local_total_seq_lens"] < required))):
                            raise ValueError("indexer chunk K extent disagrees with actual request contexts")
                if decode_rows:
                    decode_queries = tuple(q for q, phase in zip(queries, phases, strict=True) if phase == "decode")
                    if (shape["batch"] != len(decode_queries)
                            or decode_queries != (shape["next_n"],) * len(decode_queries)):
                        raise ValueError("indexer native decode requires the actual uniform query rows")
                    if v4:
                        logical = node.attributes["symbolic_score_logical_shape"]
                        if logical is None or tuple(logical) != (decode_rows, shape["max_model_len"]):
                            raise ValueError("V4 score capacity disagrees with its actual constructor input")
                if not v4 and attrs["skip_topk_buffer_clear"] != node.attributes["skip_topk_buffer_clear"]:
                    raise ValueError("DSA source clear ownership disagrees with its actual caller")
                source_dtype = "fp8"
            for name in set(attrs) & set(kernel.attributes) - {"mathematical_program"}:
                if attrs[name] != kernel.attributes[name]:
                    raise ValueError("indexer source attributes disagree with their logical call: " + name)
            hardware = require_hardware(kernel.hardware_id, hardware_by_id)
            stages = select_workload_source_launch_recipe(kernel.hardware_id, shape, source_dtype,
                kernel.variant, attrs, hardware)
            # Chunked prefill and mixed scorer recipes repeat semantic roles.
            # Keep physical order and assign each submission a unique name.
            roles = tuple(role for role, _ in stages)
            if len(roles) != len(set(roles)):
                stages = tuple((f"{index}_{role}", launch) for index, (role, launch) in enumerate(stages))
            updates[call_id] = dataclasses.replace(call, source_launches=stages)
            continue
        if (HWSpec.source_contract_supported("vllm_source_vector_launch_binding", kernel.hardware_id) and kernel.kind in (KernelKind.VECTOR, KernelKind.KV_CACHE)
                and kernel.variant in ("rmsnorm", "add_rmsnorm", "silu", "rotary", "rope", "paged_kv_cache", "elementwise")):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("source vector binding requires complete shape and attribute mappings")
            shape, attrs = binding["shape"], binding["attributes"]
            if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping) or not shape:
                raise TypeError("source vector binding requires nonempty shape and attribute mappings")
            if len(call.source_node_ids) != 1:
                raise ValueError("source vector binding requires one actual workload producer")
            node = graph.node(call.source_node_ids[0])
            if graph.source_model_execution_version is None:
                raise ValueError("source vector binding requires the actual model caller's source version")
            if attrs["model_execution_version"] != graph.source_model_execution_version:
                raise ValueError("source vector runtime disagrees with its actual model caller")
            physical = node.shape if node.kind is WorkloadKind.NORM else _vector_shape(node)
            if dict(shape) != dict(physical):
                raise ValueError("source vector launch shape disagrees with its actual physical workload node")
            for name in set(attrs) & set(kernel.attributes) - {"mathematical_program"}:
                if attrs[name] != kernel.attributes[name]:
                    raise ValueError("source vector attributes disagree with their actual logical call: " + name)
            hardware = require_hardware(kernel.hardware_id, hardware_by_id)
            launch = select_launch_config(kernel.hardware_id, kernel.kind, shape, kernel.dtype,
                kernel.variant, attrs, hardware=hardware)
            updates[call_id] = dataclasses.replace(call, source_launches=(("native_main", launch),))
            continue
        if (HWSpec.source_contract_supported("marlin_routed_expert_source_launch_binding", kernel.hardware_id) and kernel.kind is KernelKind.FUSED_MOE
                and kernel.variant == "routed_experts"
                and (kernel.attributes["routed_weight_dtype"] == "mxfp4"
                    or "mathematical_program" in kernel.attributes
                    and kernel.attributes["mathematical_program"] == "kimi_ep_marlin_swiglu_v1")):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("Marlin source binding requires complete shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping):
                raise TypeError("Marlin source binding requires shape and attribute mappings")
            if len(call.source_node_ids) != 1:
                raise ValueError("Marlin source binding requires one actual expert producer")
            node = graph.node(call.source_node_ids[0])
            if (graph.source_model_execution_version != "vllm_0_30_0"
                    or attrs["model_execution_version"] != graph.source_model_execution_version
                    or ("model_execution_version" in node.attributes
                        and node.attributes["model_execution_version"] != graph.source_model_execution_version)):
                raise ValueError("Marlin source runtime disagrees with its actual model caller")
            physical = {**node.shape, "local_experts": len(node.attributes["owned_experts"])}
            if (node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full"
                    or node.dtype != kernel.dtype or kernel.dtype != "bf16"
                    or dict(kernel.shape) != {**node.shape, "dtype_bytes": 2}
                    or "symbolic_tensor_inputs" not in node.attributes or "symbolic_tensor_outputs" not in node.attributes
                    or len(node.attributes["symbolic_tensor_inputs"]) < 3 or not node.attributes["symbolic_tensor_outputs"]
                    or dict(shape) != physical or node.attributes["weight_group_size"] != 32
                    or node.attributes["output_is_reduced"] is not False
                    or node.attributes["launch_policy"] != "analytical_launch"
                    or type(attrs["source_has_expert_map"]) is not bool
                    or attrs["source_has_expert_map"] != (node.attributes["expert_parallel_size"] > 1)
                    or tuple(node.attributes["owned_experts"]) != tuple(kernel.attributes["owned_experts"])):
                raise ValueError("Marlin source geometry/ownership disagrees with its actual expert caller")
            if kernel.attributes["routed_weight_dtype"] == "int4":
                valid = (node.attributes["source_component"] == "symbolic_grouped_routed_experts"
                    and node.attributes["routing_source_stage"] == "kimi_moe_route"
                    and node.attributes["routed_weight_dtype"] == node.attributes["checkpoint_weight_dtype"] == "int4"
                    and kernel.launch is None and kernel.execution_support == "unsupported"
                    and attrs["source_weight_type"] == "uint4b8" and attrs["source_scale_dtype"] == kernel.dtype
                    and attrs["source_activation"] == "silu" and attrs["source_clamp_limit"] is None
                    and kernel.attributes["swiglu_limit"] is None)
            else:
                valid = (node.attributes["source_component"] == "vllm_0_30_symbolic_v4_marlin"
                    and node.attributes["routed_weight_dtype"] == "mxfp4"
                    and kernel.launch is not None and kernel.launch.algorithm == "vllm_marlin_moe_mxfp4_aggregate"
                    and attrs["source_weight_type"] == "float4_e2m1f"
                    and attrs["source_scale_dtype"] == "float8_e8m0fnu"
                    and attrs["source_activation"] == "silu_clamped"
                    and attrs["source_clamp_limit"] == node.attributes["swiglu_limit"] == kernel.attributes["swiglu_limit"]
                    and node.attributes["expert_parallel_size"] == 1
                    and (node.attributes["symbolic_marlin_multiprocessors"] is None
                        or node.attributes["symbolic_marlin_multiprocessors"] == attrs["source_num_compute_units"]))
            if not valid or attrs["source_group_size"] != node.attributes["weight_group_size"]:
                raise ValueError("Marlin weight/scale/activation identity disagrees with its actual caller")
            if attrs["source_router_state"] != "prepared_topk_weights_and_ids":
                raise ValueError("Marlin expert sidecar must consume its actual external router outputs")
            for name in set(attrs) & set(kernel.attributes) - {"mathematical_program"}:
                if attrs[name] != kernel.attributes[name]:
                    raise ValueError("Marlin source attributes disagree with their logical call: " + name)
            hardware = require_hardware(kernel.hardware_id, hardware_by_id)
            stages = select_workload_source_launch_recipe(kernel.hardware_id, shape, kernel.dtype,
                "marlin_routed_experts", attrs, hardware)
            updates[call_id] = dataclasses.replace(call, source_launches=stages)
            continue
        if (HWSpec.source_contract_supported("bf16_fp8_expert_source_launch_binding", kernel.hardware_id) and kernel.kind is KernelKind.FUSED_MOE
                and kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in (
                    "unquantized_bf16_swiglu_v1", "unquantized_bf16_legacy_grouped_swiglu_v1",
                    "block_fp8_swiglu_v1", "block_fp8_replicated_ep_swiglu_v1")):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("expert source binding requires shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping):
                raise TypeError("expert source binding requires shape and attribute mappings")
            if len(call.source_node_ids) != 1:
                raise ValueError("expert source binding requires one actual expert producer")
            node = graph.node(call.source_node_ids[0])
            fp8 = kernel.attributes["mathematical_program"] in (
                "block_fp8_swiglu_v1", "block_fp8_replicated_ep_swiglu_v1")
            if fp8:
                source_domain = (node.kind is WorkloadKind.FUSED_MOE and node.variant == "full"
                    and node.dtype == kernel.dtype == "bf16"
                    and node.attributes["source_component"] in (
                        "symbolic_grouped_routed_experts", "symbolic_dsv3_routed_experts")
                    and node.attributes["routed_weight_dtype"] == "fp8"
                    and node.attributes["checkpoint_weight_dtype"] == "fp8"
                    and node.attributes["weight_group_size"] == 0
                    and node.attributes["launch_policy"] == "analytical_launch"
                    and node.attributes["output_is_reduced"] is False
                    and kernel.launch is None and kernel.execution_support == "unsupported"
                    and attrs["model_execution_version"] == graph.source_model_execution_version == "vllm_0_30_0"
                    and ("model_execution_version" not in node.attributes
                        or node.attributes["model_execution_version"] == "vllm_0_30_0"))
                if attrs["source_router_state"] != "prepared_topk_weights_and_ids":
                    raise ValueError("FP8 expert sidecar must consume its actual external router outputs")
            else:
                source_domain = (_h200_bf16_expert_math_domain(node, kernel.hardware_id)
                    and attrs["model_execution_version"] == node.attributes["model_execution_version"]
                    and graph.source_model_execution_version == node.attributes["model_execution_version"])
                if kernel.attributes["mathematical_program"] == "unquantized_bf16_legacy_grouped_swiglu_v1":
                    source_domain = (source_domain and node.attributes["model_execution_version"] == "vllm_0_11_0"
                        and node.attributes["expert_parallel_size"] == 2)
                if source_domain and attrs["model_execution_version"] == "vllm_0_11_0":
                    if attrs["source_router_state"] != "prepared_topk_weights_and_ids":
                        raise ValueError("legacy expert sidecar must consume its actual external router outputs")
                    if ("symbolic_tensor_inputs" not in node.attributes
                            or "symbolic_tensor_outputs" not in node.attributes
                            or len(node.attributes["symbolic_tensor_outputs"]) != 1
                            or len(node.attributes["symbolic_tensor_inputs"]) < 3):
                        raise ValueError("legacy expert binding requires its actual input/output tensor ports")
                    inplace = (node.attributes["symbolic_tensor_outputs"][0]
                        == node.attributes["symbolic_tensor_inputs"][0])
                    if type(attrs["source_inplace"]) is not bool or attrs["source_inplace"] != inplace:
                        raise ValueError("legacy expert output-alias flag disagrees with its actual caller")
            if not source_domain or dict(shape) != dict(node.shape):
                raise ValueError("expert source geometry/version disagrees with its actual model caller")
            for name in set(attrs) & set(kernel.attributes) - {"mathematical_program"}:
                if attrs[name] != kernel.attributes[name]:
                    raise ValueError("expert source attributes disagree with their logical call: " + name)
            hardware = require_hardware(kernel.hardware_id, hardware_by_id)
            if not fp8 and attrs["model_execution_version"] == "vllm_0_30_0":
                attrs = _count_sort_producer_source_inputs(graph, node, attrs)
            stages = select_workload_source_launch_recipe(kernel.hardware_id, shape, kernel.dtype,
                kernel.variant, attrs, hardware)
            updates[call_id] = dataclasses.replace(call, source_launches=stages)
            continue
        if (not HWSpec.source_contract_supported("mla_decode_source_launch_binding", kernel.hardware_id) or kernel.kind is not KernelKind.FLASH_ATTN
                or kernel.variant != "mla_decode" or not isinstance(binding, Mapping)
                or set(binding) != {"shape", "attributes"}):
            raise ValueError("explicit source launch binding currently requires a complete H200 MLA decode input")
        shape, attrs = binding["shape"], binding["attributes"]
        if not isinstance(shape, Mapping) or not isinstance(attrs, Mapping):
            raise TypeError("MLA source binding requires shape and attribute mappings")
        if len(call.source_node_ids) != 1:
            raise ValueError("MLA source binding requires one exact workload producer")
        node = graph.node(call.source_node_ids[0])
        identity = (attrs["model_execution_version"], attrs["source_flashmla_snapshot"])
        if (graph.source_model_execution_version is not None
                and graph.source_model_execution_version != attrs["model_execution_version"]):
            raise ValueError("MLA source launch runtime/snapshot disagrees with the actual model caller")
        if "source_flashmla_launch_identity" in node.attributes:
            if tuple(node.attributes["source_flashmla_launch_identity"]) != identity:
                raise ValueError("MLA source launch runtime/snapshot disagrees with the actual workload producer")
        else:
            # The target source may leave backend selection open while
            # already producing BF16 absorbed Q and latent-cache ports.
            # Explicit physical inputs can bind FlashMLA to that boundary;
            # the runtime comes from the actual model caller, never shape.
            if (graph.source_model_execution_version != "vllm_0_27_1"
                    or "source_model_execution_version" not in node.attributes
                    or node.attributes["source_model_execution_version"] != graph.source_model_execution_version
                    or node.attributes["source_component"] != "symbolic_mla_decode_mathematical"
                    or identity != ("vllm_0_27_1", "vllm_0_27_1_cmake_pin")
                    or attrs["source_attention_backend"] != "flashmla_dense"):
                raise ValueError("MLA source launch runtime/snapshot requires its actual model caller and explicit backend")
            rows = sum(node.attributes["q_lens"])
            if (node.dtype != "bf16" or node.attributes["source_query_dtype"] != "bf16"
                    or node.attributes["source_cache_dtype"] != "bf16"
                    or tuple(node.attributes["source_query_shape"]) != (rows, node.shape["num_query_heads"], 576)
                    or len(node.attributes["source_cache_shape"]) != 2
                    or node.attributes["source_cache_shape"][1] != 576
                    or node.attributes["source_cache_shape"][0] < sum(node.attributes["kv_lens"])):
                raise ValueError("MLA source binding disagrees with its actual BF16 query/cache ports")
        for physical, logical in (("batch", "batch"), ("q_len", "q_len"), ("num_heads", "num_query_heads"),
                ("qk_head_dim", "qk_head_dim"), ("v_head_dim", "v_head_dim")):
            if shape[physical] != kernel.shape[logical]:
                raise ValueError("MLA source launch shape disagrees with its actual logical call")
        if tuple(kernel.attributes["q_lens"]) != (shape["q_len"],) * shape["batch"]:
            raise ValueError("dense FlashMLA requires the actual uniform query batch")
        hardware = require_hardware(kernel.hardware_id, hardware_by_id)
        stages = select_workload_source_launch_recipe(kernel.hardware_id, shape, kernel.dtype,
            kernel.variant, attrs, hardware)
        updates[call_id] = dataclasses.replace(call, source_launches=stages)
    return KernelPlan((updates[call.call_id] if call.call_id in updates else call for call in plan.calls),
        plan.block_repeats, memory_context=plan.memory_context)


def _validate_dspark_pool_domain(pool, deployment, hardware):
    config = pool.parallel_plan.config
    if (not hardware.supports_source_contract("vllm_0_30_symbolic_mhc") or deployment.mode != "colocated"
            or config.tensor_parallel_size not in (1, 2, 4, 8) or config.enable_expert_parallel
            or any(value != 1 for value in (config.pipeline_parallel_size,
                config.data_parallel_size, config.context_parallel_size, config.sequence_parallel_size))):
        raise NotImplementedError("DSpark complete noncausal source outside H200 standalone eager TP1/2/4/8 without EP/DP/PP/SP/CP; Goal05 pricing remains unbound")


def _is_dspark_source_node(node):
    return ("source_dspark_standalone" in node.attributes or "source_v4_aux_capture" in node.attributes
        or node.variant == "vllm_gpu_v2_store_drafts" or node.kind is WorkloadKind.ELEMENTWISE
        and "source_component" in node.attributes
        and node.attributes["source_component"] == "vllm_0_30_RequestState_draft_constructor")


def lower_workload(
    graph: WorkloadGraph,
    deployment: DeploymentPlan,
    hardware_by_id: Mapping[str, object],
    memo: KernelLoweringMemo | None = None,
    *,
    source_projection_scope: str | None = None,
    compact_source_layers: bool = False,
) -> KernelPlan:
    if (graph.nodes and "source_component" in graph.nodes[0].attributes
            and graph.nodes[0].attributes["source_component"] == "flash_fp4_decode_switch_v1"):
        if memo is not None or source_projection_scope is not None or compact_source_layers:
            raise ValueError("FP4 decode scenario requires its explicit source lowering")
        from .heterogeneous_moe import lower_flash_fp4_decode_scenario
        return lower_flash_fp4_decode_scenario(graph, deployment, hardware_by_id)
    if type(compact_source_layers) is not bool:
        raise TypeError("source layer compaction flag must be boolean")
    if compact_source_layers and (memo is None or source_projection_scope is not None):
        raise ValueError("source layer compaction requires ordinary memoized lowering")
    structural_target = source_projection_scope is not None
    if structural_target:
        if source_projection_scope != "a800_tp1_target_representative_layers_v1" or memo is not None:
            raise ValueError("target structural lowering requires its exact uncached projection scope")
        if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
            raise NotImplementedError("target structural lowering requires one colocated A800 TP1 pool")
        structural_pool, = deployment.worker_pools.values()
        config = structural_pool.parallel_plan.config
        if (not HWSpec.source_contract_supported("target_tp1_representative_source_projection", structural_pool.hardware_id)
                or len(structural_pool.ranks) != 1
                or config.tensor_parallel_size != 1 or not config.is_rank_local
                or config.enable_expert_parallel or config.pipeline_parallel_size != 1):
            raise NotImplementedError("target structural lowering requires A800 TP/PP/SP/CP/DP1 EP-off")
        if (not graph.nodes or any("symbolic_tensor_inputs" not in node.attributes
                or node.kind in _COMMUNICATION_KINDS or node.kind is WorkloadKind.LM_HEAD
                for node in graph.nodes)):
            raise ValueError("target structural lowering requires typed hidden-only source operators")
        routes = {node.variant for node in graph.nodes if node.variant in ("dsv32_moe_route", "kimi_moe_route")}
        layers = {part for node in graph.nodes for part in node.node_id.split(":") if part.startswith("layer-")}
        if not ((routes == {"dsv32_moe_route"} and layers == {"layer-0", "layer-3"})
                or (routes == {"kimi_moe_route"} and layers == {"layer-0", "layer-1"})):
            raise ValueError("target structural scope requires its GLM or Kimi dense/MoE representative pair")
    if memo is not None and not isinstance(memo, KernelLoweringMemo):
        raise TypeError("kernel lowering memo must be KernelLoweringMemo")
    # One synchronous observation, shared by memo and EP consumers. Explicit
    # None means declined; it is never re-fetched or trusted across calls.
    from src.serving.workload.traced_forward import _source_bound_node_layout
    source_layout = _source_bound_node_layout(graph)
    if memo is None:
        logical_node_contracts = None
    else:
        logical_node_contracts = _memo_contracts(graph, memo, source_layout=source_layout)
    target_contract = _target_model_semantics_contract(graph)
    gqa_source = any("source_component" in node.attributes
        and node.attributes["source_component"] == "symbolic_gqa_mathematical" for node in graph.nodes)
    block_fp8_checkpoint = _dsv3_block_fp8_checkpoint(graph, target_contract)
    # Reject an unbound DSpark deployment at its own owner boundary before
    # ordinary V4 pipeline/EP validators interpret its different source DAG.
    dspark_pools = set()
    for source_node in graph.nodes:
        if _is_dspark_source_node(source_node) and source_node.pool_name not in dspark_pools:
            if source_node.pool_name not in deployment.worker_pools:
                raise ValueError(f"workload node references unknown pool {source_node.pool_name!r}")
            source_pool = deployment.worker_pools[source_node.pool_name]
            _validate_dspark_pool_domain(source_pool, deployment,
                require_hardware(source_pool.hardware_id, hardware_by_id))
            dspark_pools.add(source_node.pool_name)
    _validate_tp1_pipeline_graph(graph, deployment)
    _validate_tp2_pp2_pipeline_graph(graph, deployment)
    _validate_glm_mtp_dp_expert_graph(graph, deployment)
    _validate_kimi_dp_expert_graph(graph, deployment)
    ep_source_validation_reused = False
    if graph.nodes and all(is_heterogeneous_moe(n) for n in graph.nodes):
        validate_heterogeneous_graph(graph, deployment)
    elif target_contract is None:
        ep_source_validation_reused = _reuse_ep_source_validation(graph, deployment, source_layout=source_layout)
        if not ep_source_validation_reused:
            _validate_ep_fused_moe_graph(graph, deployment)
            _record_ep_source_validation(graph, deployment, source_layout=source_layout)
    if deployment.mode in ("af_disagg", "pd_af_disagg"):
        for node in graph.nodes:
            if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER:
                _validate_af_activation_transfer_node(
                    node, deployment, graph)
        _validate_af_activation_boundaries(graph, deployment)
    fusion_heads, fusion_member_heads = (
        _ascend_qkv_rmsnorm_rope_fusions(
            graph, deployment, hardware_by_id))
    if compact_source_layers:
        # The complete EP graph was checked above. Validate every real routing
        # record before comparing raw physical contracts across source layers.
        # The reused EP proof establishes deep-native static MoE physical
        # attributes; current dynamic fields were checked above. Thus the
        # following serializers cannot call an unknown value's __reduce__.
        # Unbound/opaque graphs retain interleaved ordinary validation.
        routing_certificates_valid = (ep_source_validation_reused and source_layout is not None
            and _native_graph_routing_certificates(graph.nodes))
        moe_physical_contracts = (memo.source_moe_physical_contracts(graph, source_layout)
            if routing_certificates_valid else None)
        # Collective payload producer identities belong to the complete source
        # DAG. Layer folding redirects edges to representatives, while keeping
        # those physical identities intact. Validate every actual collective
        # before folding, including occurrences absent from the compact graph.
        for current in graph.nodes:
            if current.kind is WorkloadKind.COLLECTIVE:
                if current.pool_name not in deployment.worker_pools:
                    raise ValueError(f"workload node references unknown pool {current.pool_name!r}")
                _validate_collective_node(current, deployment.worker_pools[current.pool_name],
                    graph, gqa_source=gqa_source)
            if current.kind is WorkloadKind.FUSED_MOE:
                if not routing_certificates_valid:
                    _validate_routing_provenance(current)
                logical_node_contracts[current.node_id] = (_logical_node_contract(current)
                    if moe_physical_contracts is None else moe_physical_contracts[current.node_id])
        graph = memo.compact_source_layers(graph, logical_node_contracts)
    calls = []
    final_call_by_node = {}
    call_ids_by_node = {}
    for node_id in graph.topological_order():
        node = graph.node(node_id)
        _require_source_lowering_contract(node)
        if "source_dspark_standalone" in node.attributes or "source_v4_aux_capture" in node.attributes:
            _validate_dspark_source_ports(node, graph)
        if node.variant == "symbolic_storage_only":
            raise NotImplementedError("source storage-only graph has no mathematical or native timing contract")
        if (node.kind is WorkloadKind.FUSED_MOE and "source_component" in node.attributes
                and node.attributes["source_component"] == "vllm_0_30_symbolic_v4_marlin"
                and "source_logical_intermediate" in node.attributes
                and node.attributes["source_logical_intermediate"] != node.shape["intermediate"]):
            raise NotImplementedError("V4 padded Marlin source requires separate logical-work and physical-storage pricing bindings")
        if (node_id in fusion_member_heads
                and fusion_member_heads[node_id] != node_id):
            continue
        if node.pool_name not in deployment.worker_pools:
            raise ValueError(f"workload node references unknown pool {node.pool_name!r}")
        pool = deployment.worker_pools[node.pool_name]
        if node.rank is None:
            raise ValueError(
                f"workload node {node.node_id!r} requires an explicit rank")
        if node.rank not in pool.ranks:
            raise ValueError(
                f"workload node {node.node_id!r} rank {node.rank} is not "
                f"assigned to pool {pool.name!r}: {pool.ranks!r}")
        hardware_id = pool.hardware_id
        hardware = require_hardware(hardware_id, hardware_by_id)
        current_dspark_specs = None
        dspark_typed_role_key = None
        if _is_dspark_source_node(node):
            current_dspark_specs = _dspark_typed_math_specs(node, hardware_id)
            if current_dspark_specs is not None and all("mathematical_program" in spec.attributes
                    and spec.attributes["mathematical_program"] == "vllm_0_30_dspark_typed_role_bytes_v1"
                    for spec in current_dspark_specs):
                dspark_typed_role_key = tuple((spec.dtype, spec.shape["element_count"], spec.attributes["operation"])
                    for spec in current_dspark_specs)
        if (node.variant=='kimi_k3_kda_decode' and (
                'source_null_state_skip' in node.attributes
                or 'source_component' in node.attributes and node.attributes['source_component']=='vllm_0_30_kimi_k3_native_decode')):
            raise NotImplementedError("K3 null-state/native fused KDA source requires its Goal05 pricing contract; no launch bound")
        if node_id in fusion_heads:
            group = fusion_heads[node_id]
            group_ids = tuple(member.node_id for member in group)
            group_id_set = set(group_ids)
            external_node_dependencies = tuple(dict.fromkeys(
                predecessor
                for member in group
                for predecessor in graph.predecessors(member.node_id)
                if predecessor not in group_id_set
            ))
            external_dependencies = tuple(
                final_call_by_node[predecessor]
                for predecessor in external_node_dependencies)

            def build_fused_specs():
                return _ascend_qkv_rmsnorm_rope_specs(
                    *group, hardware_id, hardware)

            if memo is None:
                specs = build_fused_specs()
            else:
                composite_contract = tuple(
                    logical_node_contracts[member_id]
                    for member_id in group_ids)
                specs = memo.lower(
                    node, composite_contract, hardware, build_fused_specs)
            if not specs:
                raise ValueError(
                    f"workload node {node.node_id!r} lowered to no "
                    "KernelSpecs")
            # The materialization helpers and the fused kernel form one
            # source-level chained string: each helper's only in-string
            # predecessor is the previous spec, so the fused head lowers to
            # a call per spec sharing the group's source identity.
            fused_call_ids = []
            previous = None
            for index, spec in enumerate(specs):
                call_id = (
                    node.node_id if len(specs) == 1
                    else f"{node.node_id}:{index}")
                dependencies = (
                    external_dependencies if previous is None
                    else (previous,))
                calls.append(KernelCall(
                    call_id,
                    spec,
                    node.pool_name,
                    node.rank,
                    "compute",
                    group_ids,
                    dependencies,
                ))
                fused_call_ids.append(call_id)
                previous = call_id
            for member_id in group_ids:
                final_call_by_node[member_id] = previous
                call_ids_by_node[member_id] = tuple(fused_call_ids)
            continue
        if node.kind in (
                WorkloadKind.COLLECTIVE,
                WorkloadKind.P2P,
                WorkloadKind.AF_ACTIVATION_TRANSFER) and not is_heterogeneous_moe(node):
            _validate_graph_domain_boundary(node, graph)
        if node.kind is WorkloadKind.COLLECTIVE and not compact_source_layers:
            _validate_collective_node(node, pool, graph, gqa_source=gqa_source)
        if node.kind is WorkloadKind.P2P:
            if is_heterogeneous_moe(node):
                validate_heterogeneous_transfer(node, deployment)
            else:
                _validate_pipeline_p2p_node(node, pool)
        if node.kind is WorkloadKind.PD_KV_TRANSFER:
            if is_heterogeneous_moe(node):
                validate_heterogeneous_pd_transfer(node, deployment)
            else:
                _validate_pd_kv_transfer_node(node, deployment)
        routing_relabelling = None
        if node.kind is WorkloadKind.FUSED_MOE and target_contract is None:
            # The relabelling only names the physical class, so the routing
            # this node actually carries still has to answer for itself: keep
            # its provenance check on every step rather than on the steps that
            # happen to open a new class.
            _validate_routing_provenance(node)
            if "source_dp_lane_rows" in node.attributes and node.attributes["expert_parallel_size"] == 1:
                _validate_virtual_ep_router_inputs(graph, node)
            if not is_heterogeneous_moe(node) and not ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "vllm_0_30_kimi_k3_marlin_experts", "vllm_0_30_symbolic_v4_marlin",
                        "symbolic_grouped_routed_experts", "symbolic_dsv3_routed_experts")):
                routing_relabelling = _fused_moe_routing_relabelling(
                    node, hardware_id, hardware)
        def build_specs():
            _require_source_lowering_contract(node)
            if "source_dspark_standalone" in node.attributes or "source_v4_aux_capture" in node.attributes:
                adapted = _dspark_typed_math_specs(node, hardware_id)
                if adapted is not None:
                    return adapted
                if node.kind is WorkloadKind.VECTOR and node.variant in ("mhc_prenorm", "mhc_pre_norm"):
                    if node.shape["n_splits"] != 1 or node.attributes["source_use_deep_gemm"] is not False:
                        raise ValueError("DSpark first expanded prenorm requires its declared TileLang source")
                    if node.variant == "mhc_pre_norm":
                        return (_selected_target_compute_spec(node, hardware_id, hardware, KernelKind.VECTOR,
                            node.variant, _vector_shape(node), {}),)
                    return (KernelSpec(KernelKind.VECTOR, hardware_id, _vector_shape(node), dtype=node.dtype,
                        variant=node.variant, attributes={"source_prenorm_backend": "tilelang"},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if (node.variant == "vllm_gpu_v2_store_drafts" or node.kind is WorkloadKind.ELEMENTWISE
                    and "source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_RequestState_draft_constructor"):
                if not hardware.supports_source_contract("vllm_0_30_symbolic_mhc"):
                    raise NotImplementedError("DSpark resident draft storage requires the same bound target source")
                adapted = _dspark_typed_math_specs(node, hardware_id)
                if adapted is not None:
                    return adapted
            a800_glm_pipeline = (HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", hardware_id)
                and _symbolic_glm_pipeline_projection(node, pool))
            a800_kimi_pipeline = _symbolic_kimi_pipeline_projection(node, pool)
            kimi_parallel = _symbolic_kimi_parallel_projection(node, pool) or _symbolic_kimi_dp_projection(node, pool)
            v3_parallel = _symbolic_v3_parallel_projection(node, pool)
            a800_kimi_single = (HWSpec.source_contract_supported("kimi_tp1_unbound_source_lowering", hardware_id)
                and _symbolic_kimi_single_rank_projection(node, pool))
            a800_glm_pcp = (HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", hardware_id)
                and _symbolic_glm_pcp_projection(node, pool))
            unbound_target_source = structural_target or a800_glm_pipeline or a800_kimi_pipeline or a800_kimi_single or a800_glm_pcp or kimi_parallel or v3_parallel
            projection_scope = ("a800_tp8_pp4_ep8_glm_source_v1" if a800_glm_pipeline else
                ("vllm_0_30_kimi_parallel_source_v1" if HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id)
                    else "a800_tp8_pp2_ep8_kimi_source_v1") if a800_kimi_pipeline else
                "a800_tp1_kimi_source_v1" if a800_kimi_single else
                "vllm_0_30_glm_pcp_source_v1" if a800_glm_pcp else source_projection_scope)
            if is_heterogeneous_moe(node):
                return heterogeneous_moe_specs(node, hardware_id)
            if (unbound_target_source and node.kind in (WorkloadKind.LINEAR, WorkloadKind.ROUTER)
                    and "output_dtype" in node.attributes and node.attributes["output_dtype"] == "fp32"):
                return (KernelSpec(kind=KernelKind.GEMM, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant="gemm", attributes={"output_dtype": "fp32",
                        "source_projection_scope": projection_scope},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if (node.kind is WorkloadKind.GATHER and "mathematical_program" in node.attributes):
                if node.variant != "gather":
                    raise ValueError("logical materialized copy requires its gather program")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant,
                    attributes={name: node.attributes[name] for name in ("mathematical_program", "gather_factor",
                        "index_count", "index_bytes", "scattered", "materialized_output")},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_mla_context_merge"):
                if node.kind is not WorkloadKind.VECTOR or node.variant != "vllm_attention_state_merge":
                    raise ValueError("MLA context merge requires its typed vector program")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant=node.variant,
                    attributes={key: node.attributes[key] for key in ("mathematical_program", "output_lse", "empty_prefix_rows")
                        if key in node.attributes},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "sglang_0_5_19_symbolic_absorbed_prefill"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract("sglang_0_5_19_symbolic_logits_boundary")
                        or not config.is_rank_local or config.tensor_parallel_size != 8
                        or config.enable_expert_parallel or config.pipeline_parallel_size != 1
                        or any(value != 1 for value in (config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("SGLang absorbed prefill source requires H200 TP8 PP/SP/CP/DP1 EP-off")
                if (node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_absorbed_prefill"
                        or node.attributes["mathematical_program"] != "bf16_mla_absorbed_prefill_softmax_v1"):
                    raise ValueError("SGLang absorbed prefill requires its declared no-LSE mathematical program")
                return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant=node.variant, attributes={key: node.attributes[key] for key in (
                        "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "full_cuda_graphs", "max_cudagraph_capture_size",
                        "return_softmax_lse", "mathematical_program")}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "symbolic_mla_fresh_prefill_mathematical", "symbolic_mla_context_prefill_mathematical",
                        "sglang_0_5_19_symbolic_one_shot_prefill")):
                config = pool.parallel_plan.config
                # The complete EP graph validator above admits only the
                # source-backed Kimi TP=EP8 pool; expert ownership does not
                # change that pool's TP8 attention/cache geometry.
                glm_tp1 = (config.tensor_parallel_size == 1 and not config.enable_expert_parallel
                    and (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (256, 256, 512))
                glm_parallel_fresh = (_symbolic_glm_pcp_projection(node, pool)
                    and config.context_parallel_size == 1 and node.dtype == "bf16"
                    and node.attributes["source_component"] == "symbolic_mla_fresh_prefill_mathematical"
                    and (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (256, 256, 512))
                if glm_parallel_fresh and (node.attributes["q_lens"] != node.attributes["kv_lens"]
                        or node.attributes["causal"] is not True or node.attributes["return_softmax_lse"] is not False
                        or node.attributes["mathematical_program"] != "bf16_mla_prefill_softmax_v1"):
                    raise ValueError("GLM dense fresh-prefill math requires matching causal query/KV lengths and its no-LSE program")
                kimi_tp1 = (_symbolic_kimi_single_rank_projection(node, pool)
                    and (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (192, 128, 512))
                v3_tp1 = (hardware.supports_source_contract(node.attributes["source_component"])
                    and not config.enable_expert_parallel
                    and all(value == 1 for value in (config.pipeline_parallel_size,
                        config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                    and "source_model_profile" in node.attributes
                    and node.attributes["source_model_profile"] in ("vllm_0_27_1_deepseek_v3_sync_fp8", "vllm_0_11_deepseek_v3_sync_fp8")
                    and type(node.attributes["source_tensor_parallel_size"]) is int
                    and node.attributes["source_tensor_parallel_size"] == config.tensor_parallel_size
                    and (config.tensor_parallel_size == 1 or (config.tensor_parallel_size == 4
                        and node.attributes["source_model_profile"] == "vllm_0_27_1_deepseek_v3_sync_fp8"
                        and node.attributes["source_target_layers"] == 32))
                    and (node.shape["num_query_heads"], node.shape["qk_head_dim"],
                        node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (128 // config.tensor_parallel_size, 192, 128, 512))
                if (not glm_parallel_fresh and not kimi_parallel and not v3_parallel and ((config.enable_expert_parallel and config.expert_parallel_size != 8)
                        or (not unbound_target_source and (not hardware.supports_source_contract(node.attributes["source_component"])
                            or (config.tensor_parallel_size != 8 and not (glm_tp1 or kimi_tp1 or v3_tp1))))
                        or not config.is_rank_local or (config.pipeline_parallel_size != 1
                            and not ((_symbolic_glm_pipeline_projection(node, pool)
                                and (node.shape["qk_head_dim"], node.shape["v_head_dim"], node.shape["kv_lora_rank"]) == (256, 256, 512))
                                or a800_kimi_pipeline)))):
                    raise NotImplementedError("ordinary MLA math requires H200 TP8 or its declared GLM/Kimi/V3 TP1 EP-off source")
                if node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_prefill" or node.dtype != "bf16":
                    raise ValueError("ordinary MLA mathematical source requires expanded BF16 attention")
                return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant=node.variant, attributes={key: node.attributes[key] for key in (
                        "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "full_cuda_graphs", "max_cudagraph_capture_size",
                        "return_softmax_lse", "mathematical_program", *(
                            ("prefix_cache_threshold", "max_kv_chunk_capacity")
                            if node.attributes["source_component"] == "sglang_0_5_19_symbolic_one_shot_prefill" else ()))}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_k3_prefill_attention"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 prefill source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if (node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_prefill" or node.dtype != "bf16"
                        or node.attributes["prefill_output_source"] != "supports_out"
                        or node.attributes["q_lens"] != node.attributes["kv_lens"]
                        or len(node.attributes["q_lens"]) != 1):
                    raise ValueError("K3 prefill attention requires one fresh BF16 supports_out source sequence")
                return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant=node.variant, attributes={key: node.attributes[key] for key in (
                        "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "full_cuda_graphs", "max_cudagraph_capture_size",
                        "return_softmax_lse", "mathematical_program")},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in ("symbolic_mla_decode_mathematical", "symbolic_mla_cache_update")):
                config = pool.parallel_plan.config
                supported_tp = (1, 4, 8) if _symbolic_dsv3_l32_tp4_projection(node, pool) else (1, 8)
                if (not kimi_parallel and not v3_parallel and not _symbolic_glm_pcp_projection(node, pool) and ((not unbound_target_source and (not hardware.supports_source_contract(node.attributes["source_component"])
                            or config.tensor_parallel_size not in supported_tp))
                        or (config.enable_expert_parallel and config.expert_parallel_size != 8)
                        or ((config.pipeline_parallel_size != 1 and not _symbolic_mla_pipeline_projection(node, pool))
                            or any(value != 1 for value in (config.sequence_parallel_size,
                            config.context_parallel_size, config.data_parallel_size))))):
                    raise NotImplementedError("symbolic BF16 MLA requires H200 and its explicit TP source scope, PP/SP/CP/DP1 EP-off")
                if node.attributes["source_component"] == "symbolic_mla_cache_update":
                    if node.kind is not WorkloadKind.KV_UPDATE or node.variant != "mla_kv_cache" or node.dtype != "bf16":
                        raise ValueError("symbolic MLA cache update has an unexpected operation")
                    return (_target_kv_cache_update_spec(node, hardware_id, hardware),)
                if node.kind is not WorkloadKind.FLASH_ATTN or node.variant != "mla_decode" or node.dtype != "bf16":
                    raise ValueError("symbolic MLA decode has an unexpected operation")
                kimi_multiquery = (kimi_parallel or _symbolic_kimi_single_rank_projection(node, pool) or a800_kimi_pipeline
                    or hardware.supports_source_contract("mla_tp8_ep8_source_projection") and len(pool.ranks) == 8
                    and config.tensor_parallel_size == config.expert_parallel_size == 8
                    and config.pipeline_parallel_size == 1 and config.enable_expert_parallel
                    and "model_source_profile" in node.attributes
                    and node.attributes["model_source_profile"] == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
                    and "model_execution_version" in node.attributes
                    and node.attributes["model_execution_version"] == "vllm_0_30_0"
                    and "source_projection_only" in node.attributes and node.attributes["source_projection_only"] is True)
                if (any(q > 1 for q in node.attributes["q_lens"])
                        and (config.tensor_parallel_size != 8 or config.enable_expert_parallel)
                        and not kimi_multiquery and not _symbolic_dsv3_l32_tp4_projection(node, pool)):
                    raise NotImplementedError("multiquery MLA math requires TP8 EP-off or its declared Kimi source scope")
                return (KernelSpec(kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in (
                        "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens", "full_cuda_graphs",
                        "max_cudagraph_capture_size", "mathematical_program")}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_block_fp8_linear"):
                if (node.kind is not WorkloadKind.LINEAR or node.variant != "gemm" or node.dtype != "bf16"
                        or node.shape["b"] != 1 or node.attributes["launch_policy"] != "analytical_launch"
                        or node.attributes["has_bias"] is not False
                        or ("output_dtype" in node.attributes and node.attributes["output_dtype"] not in (None, "bf16"))
                        or node.attributes["weight_dtype"] != "fp8" or node.attributes["weight_group_size"] != 128
                        or node.attributes["weight_quantization"] != BLOCK_FP8_WEIGHT_QUANTIZATION
                        or node.attributes["symbolic_scale_layout"] != "logical_group128_fp32"
                        or node.attributes["symbolic_workspace_complete"] is not False):
                    raise ValueError("symbolic block-FP8 linear requires its logical FP32-scale source")
                glm_fp8_source = (target_contract == "glm_moe_dsa_vllm_0_30_0_logical_v1"
                    or ("model_source_profile" in node.attributes
                        and node.attributes["model_source_profile"] == "vllm_0_30_v32_fp8_sync_explicit_allreduce"
                        and _symbolic_glm_pcp_projection(node, pool)))
                if (glm_fp8_source and HWSpec.source_contract_supported(
                        "h200_partial_source_programs", hardware_id)):
                    return _gemm_node_specs(node, hardware_id, hardware)
                rows, columns = node.shape["m"], node.shape["k"]
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                    shape={"element_count": rows * columns, "rows": rows, "columns": columns},
                    dtype="bf16", variant="fp8_group_quant", attributes={"group_size": 128,
                        "source_program": "vllm_0_30_group_fp8_fp32_scales"},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),
                    KernelSpec(kind=KernelKind.GEMM, hardware_id=hardware_id, shape=node.shape,
                        dtype="bf16", variant="gemm", attributes={"mathematical_program": "block_fp8_fp32_scales_v1"},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"))
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in ("symbolic_mtp_eager", "vllm_0_27_1_symbolic_mtp_eager", "vllm_0_30_symbolic_mtp_eager")):
                config = pool.parallel_plan.config
                source027 = node.attributes["source_component"] == "vllm_0_27_1_symbolic_mtp_eager"
                source030 = node.attributes["source_component"] == "vllm_0_30_symbolic_mtp_eager"
                glm_source = source030 and _symbolic_glm_pcp_projection(node, pool)
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.VECTOR
                        or node.variant != "vllm_mtp_eager" or config.enable_expert_parallel and not glm_source
                        or node.attributes["model_execution_version"] != ("vllm_0_30_0" if source030 else "vllm_0_27_1" if source027 else "vllm_0_11_0")
                        or (not glm_source
                            and (config.tensor_parallel_size not in ((1, 8) if source030 else (8,) if source027 else (1,))
                                and not (source027 and _symbolic_dsv3_l32_tp4_projection(node, pool))))
                        or config.pipeline_parallel_size != 1 and not _symbolic_glm_mtp_last_stage_projection(node, pool)
                        or config.context_parallel_size != 1
                        or config.sequence_parallel_size != 1 and not _symbolic_glm_mtp_data_projection(node, pool)
                        or config.data_parallel_size != 1 and not _symbolic_glm_mtp_data_projection(node, pool)):
                    raise NotImplementedError("MTP eager source requires its version-bound legacy TP or declared GLM0.30 TP domain")
                names = {"operation", "model_execution_version"}
                if node.attributes["operation"] in ("gather", "scatter"):
                    names |= {"index_bytes", "selected_indices"}
                elif node.attributes["operation"] == "masked_fill":
                    names.add("masked_rows")
                elif node.attributes["operation"] == "rejection":
                    names |= {"verified_tokens", "accepted_tokens"}
                elif node.attributes["operation"] == "prepare_next_padded":
                    names |= {"vocab_size", "valid_sampled_counts", "discarded_rows"}
                elif node.attributes["operation"] == "step_metadata":
                    names |= {"block_size", "max_model_len", "block_table_columns"}
                elif node.attributes["operation"] in ("dspark_greedy_verify","dspark_roundtrip_prepare"):
                    names |= {"num_speculative_tokens", "accepted_drafts"}
                elif node.attributes["operation"] == "dspark_swa_indices":
                    names |= {"num_speculative_tokens", "kv_lens"}
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in names},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_v32_eh_norm"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"])
                        or node.kind is not WorkloadKind.NORM or node.dtype != "bf16"
                        or node.variant not in ("dsv32_eh_norm", "dsv32_embed_eh_norm")
                        or node.attributes["model_execution_version"] != "vllm_0_30_0"
                        or (config.tensor_parallel_size not in (1, 8) and not _symbolic_glm_pcp_projection(node, pool))
                        or (node.variant == "dsv32_embed_eh_norm" and config.tensor_parallel_size != 1)
                        or config.enable_expert_parallel and not _symbolic_glm_pcp_projection(node, pool)
                        or config.pipeline_parallel_size != 1 and not _symbolic_glm_mtp_last_stage_projection(node, pool)
                        or config.context_parallel_size != 1
                        or config.sequence_parallel_size != 1 and not _symbolic_glm_mtp_data_projection(node, pool)
                        or config.data_parallel_size != 1 and not _symbolic_glm_mtp_data_projection(node, pool)):
                    raise NotImplementedError("V32 fused EH requires its declared H200 TP source and full-vocabulary TP1 embedding")
                return (_selected_target_compute_spec(node, hardware_id, hardware, KernelKind.VECTOR,
                    node.variant, node.shape, {}),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "symbolic_mtp_position_mask", "vllm_0_27_1_symbolic_mtp_position_where")):
                source027 = node.attributes["source_component"] == "vllm_0_27_1_symbolic_mtp_position_where"
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.VECTOR
                        or node.variant != "vllm_mtp_position_mask"
                        or node.attributes["model_execution_version"] != ("vllm_0_27_1" if source027 else "vllm_0_11_0")
                        or (pool.parallel_plan.config.tensor_parallel_size != (8 if source027 else 1)
                            and not (source027 and _symbolic_dsv3_l32_tp4_projection(node, pool)))):
                    raise NotImplementedError("MTP position mask requires its version-bound A800 TP1 or H200 TP8 source")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in (
                        "stage", "model_execution_version", "masked_rows")}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_fp32_linear"):
                if (node.kind not in (WorkloadKind.LINEAR, WorkloadKind.ROUTER, WorkloadKind.LM_HEAD)
                        or node.variant != "gemm" or node.dtype != "fp32"
                        or node.attributes["has_bias"] is not False
                        or ("output_dtype" in node.attributes and node.attributes["output_dtype"] not in (None, "fp32"))):
                    raise NotImplementedError("FP32 logical linear requires unquantized bias-free FP32 A/B/C")
                return (KernelSpec(kind=KernelKind.GEMM, hardware_id=hardware_id, shape=node.shape,
                    dtype="fp32", variant="gemm", attributes={"mathematical_program": "unquantized_fp32_flops_bytes_v1"},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_glm4_sigmoid_topk"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.VECTOR
                        or node.variant != "moe_topk" or node.dtype != "fp32"
                        or config.enable_expert_parallel
                        or any(value != 1 for value in (config.tensor_parallel_size, config.pipeline_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("GLM4 sigmoid routing currently requires its existing A800 TP1 source")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                    dtype="fp32", variant="moe_topk", attributes={name: node.attributes[name] for name in (
                        "source_program", "normalize_topk_probability", "mathematical_program")}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "symbolic_softmax_topk", "symbolic_expert_owner_mask", "symbolic_unquantized_experts")):
                config = pool.parallel_plan.config
                fp32_route011 = (node.attributes["source_component"] == "symbolic_softmax_topk"
                    and node.attributes["source_program"] == "vllm_0_11_fp32_softmax_topk")
                route011_parallel = ((config.tensor_parallel_size == 1 and not config.enable_expert_parallel)
                    or (config.tensor_parallel_size == config.expert_parallel_size == 2 and config.enable_expert_parallel))
                if fp32_route011 and (not hardware.supports_source_contract("vllm_0_11_fp32_softmax_topk") or not route011_parallel):
                    raise NotImplementedError("0.11 FP32 softmax routing requires CUDA TP1 EP-off or TP=EP2")
                contract = ("symbolic_softmax_topk" if node.attributes["source_component"] == "symbolic_expert_owner_mask"
                    else node.attributes["source_component"])
                source030 = ((node.attributes["source_component"] == "symbolic_unquantized_experts"
                        and node.attributes["model_execution_version"] == "vllm_0_30_0")
                    or (node.attributes["source_component"] == "symbolic_softmax_topk"
                        and node.attributes["source_program"] == "vllm_0_30_bf16_softmax_topk"))
                if (not hardware.supports_source_contract(contract)
                        or node.dtype != ("fp32" if fp32_route011 else "bf16")
                        or (config.enable_expert_parallel and (
                            (config.expert_parallel_size != config.tensor_parallel_size and not (source030 and _qwen030_dp_sp(config)))
                            or (config.expert_parallel_size != 2 and not source030)))
                        or (config.pipeline_parallel_size != 1 and not source030)
                        or (not (source030 and _qwen030_dp_sp(config)) and any(value != 1 for value in (config.sequence_parallel_size,
                            config.context_parallel_size, config.data_parallel_size)))):
                    raise NotImplementedError("virtual Qwen3 MoE requires BF16, PP/SP/CP/DP1 and EP1 or TP=EP=2")
                if node.attributes["source_component"] in ("symbolic_softmax_topk", "symbolic_expert_owner_mask"):
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "moe_topk":
                        raise ValueError("softmax routing source has an unexpected operation")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant="moe_topk", attributes={name: node.attributes[name] for name in (
                            "source_program", "normalize_topk_probability", "mathematical_program", "stage")
                            if name in node.attributes}, launch=None,
                        execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full":
                    raise ValueError("unquantized expert source has an unexpected operation")
                return _explicit_source_expert_specs(node, hardware_id, hardware, routing_relabelling)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_27_1_symbolic_grouped_routing", "vllm_0_11_symbolic_grouped_routing", "symbolic_dsv3_routed_experts")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract("symbolic_dsv3_routed_experts")
                        or (not v3_parallel and ((config.tensor_parallel_size not in (1, 8) and not _symbolic_dsv3_l32_tp4_projection(node, pool)) or config.enable_expert_parallel
                            or config.pipeline_parallel_size != 1 or not config.is_rank_local))):
                    raise NotImplementedError("DSV3 grouped source requires its declared ordinary FP8 TP/PP EP-off binding")
                if node.attributes["source_component"] in ("vllm_0_27_1_symbolic_grouped_routing", "vllm_0_11_symbolic_grouped_routing"):
                    if (node.kind is not WorkloadKind.VECTOR or node.variant != "moe_topk" or node.dtype != "bf16"
                            or (node.shape["experts"], node.shape["topk"]) != (256, 8)
                            or node.attributes["source_program"] != ({
                                "vllm_0_27_1_symbolic_grouped_routing": "vllm_0_27_1_bf16_sigmoid_bias_grouped_topk",
                                "vllm_0_11_symbolic_grouped_routing": "vllm_0_11_bf16_sigmoid_bias_grouped_topk",
                            }[node.attributes["source_component"]])
                            or node.attributes["mathematical_program"] != "bf16_sigmoid_bias_grouped_topk_flops_bytes_v1"
                            or (node.attributes["expert_group_count"], node.attributes["topk_group_count"]) != (8, 4)
                            or node.attributes["normalize_topk_probability"] is not True
                            or node.attributes["routed_scaling_factor"] != 1.0):
                        raise ValueError("DSV3 grouped routing requires its exact BF16 sigmoid/bias grouped-top8 source")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in (
                            "source_program", "mathematical_program", "normalize_topk_probability",
                            "expert_group_count", "topk_group_count", "routed_scaling_factor")},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if (node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full" or node.dtype != "bf16"
                        or (node.shape["experts"], node.shape["topk"], node.shape["hidden"], node.shape["intermediate"])
                            != (256, 8, 7168, 2048 // config.tensor_parallel_size)
                        or node.attributes["routed_weight_dtype"] != "fp8"
                        or node.attributes["checkpoint_weight_dtype"] != "fp8"
                        or type(node.attributes["weight_group_size"]) is not int or node.attributes["weight_group_size"] != 0
                        or node.attributes["routing_source_stage"] != "moe_topk"
                        or node.attributes["routed_scaling_factor"] != 1.0
                        or node.attributes["output_is_reduced"] is not False):
                    raise ValueError("DSV3 routed experts require block-FP8 checkpoint storage and partial outputs before source scaling")
                counts, local, owned = (tuple(node.attributes[name]) for name in (
                    "tokens_per_expert", "local_tokens_per_expert", "owned_experts"))
                if (any(type(value) is not int for value in (*counts, *local, *owned))
                        or counts != local or owned != tuple(range(node.shape["experts"]))
                        or type(node.attributes["expert_rank"]) is not int or node.attributes["expert_rank"] != 0
                        or type(node.attributes["expert_parallel_size"]) is not int
                        or node.attributes["expert_parallel_size"] != 1):
                    raise ValueError("DSV3 EP-off experts must preserve identical global/local counts and full expert ownership")
                if HWSpec.source_contract_supported("symbolic_dsv3_routed_experts", hardware_id) and node.attributes["launch_policy"] == "analytical_launch":
                    attributes = _fused_moe_prediction_attributes(node)
                    attributes["mathematical_program"] = "block_fp8_swiglu_v1"
                    return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id,
                        shape=node.shape, dtype="bf16", variant="routed_experts", attributes=attributes,
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                specs = _fused_moe_specs(node, hardware_id, hardware)
                if sum(spec.kind is KernelKind.VECTOR and spec.variant == "moe_topk" for spec in specs) != 1:
                    raise ValueError("explicit DSV3 routing must replace exactly one legacy top-k stage")
                return tuple(spec for spec in specs if not (spec.kind is KernelKind.VECTOR and spec.variant == "moe_topk"))
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_grouped_routing", "symbolic_grouped_routed_experts")):
                config = pool.parallel_plan.config
                kimi_ep8 = ((config.expert_parallel_size == 8 or kimi_parallel and config.enable_expert_parallel)
                    and ((node.kind is WorkloadKind.VECTOR and node.variant == "kimi_moe_route")
                         or (node.kind is WorkloadKind.FUSED_MOE and node.attributes["routing_source_stage"] == "kimi_moe_route")))
                glm_source = ((node.kind is WorkloadKind.VECTOR and node.variant == "dsv32_moe_route")
                    or (node.kind is WorkloadKind.FUSED_MOE and node.attributes["routing_source_stage"] == "dsv32_moe_route"))
                supported_tp = (1, 8)
                grouped_ep8 = (kimi_parallel and config.enable_expert_parallel or (config.tensor_parallel_size == config.expert_parallel_size == 8
                    and (kimi_ep8 or glm_source)) or glm_source and config.enable_expert_parallel
                    and _symbolic_glm_pcp_projection(node, pool))
                if (not kimi_parallel and not _symbolic_glm_pcp_projection(node, pool) and ((not unbound_target_source and (not hardware.supports_source_contract(node.attributes["source_component"])
                            or config.tensor_parallel_size not in supported_tp))
                        or (config.enable_expert_parallel and not grouped_ep8)
                        or ((config.pipeline_parallel_size != 1 and not (
                            _symbolic_glm_pipeline_projection(node, pool) and glm_source or a800_kimi_pipeline))
                            or any(value != 1 for value in (config.sequence_parallel_size,
                            config.context_parallel_size, config.data_parallel_size))))):
                    raise NotImplementedError("symbolic grouped routing/experts require the declared H200 GLM TP1/8 or Kimi TP8 source scope")
                if node.attributes["source_component"] == "vllm_0_30_symbolic_grouped_routing":
                    if (node.kind is not WorkloadKind.VECTOR
                            or (node.variant, node.dtype) not in (("dsv32_moe_route", "fp32"), ("kimi_moe_route", "bf16"))
                            or (node.variant == "kimi_moe_route" and node.attributes["routed_scaling_factor"] != 1.0)):
                        raise ValueError("symbolic grouped routing has an unexpected operation or scale placement")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                        dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in (
                            "expert_id_selection", "routed_scaling_factor", "score_function", "expert_group_count",
                            "topk_group_count", "normalize_topk_probability", "mathematical_program")
                            if name != "mathematical_program" or name in node.attributes}
                            | ({"source_projection_scope": projection_scope} if unbound_target_source and not HWSpec.source_contract_supported("h200_partial_source_programs", hardware_id) else {}), launch=None,
                        execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full":
                    raise ValueError("symbolic grouped experts have an unexpected operation")
                if (node.dtype != "bf16" or node.attributes["routed_weight_dtype"] not in ("bf16", "fp8", "int4")
                        or node.attributes["routed_weight_dtype"] != node.attributes["checkpoint_weight_dtype"]
                        or type(node.attributes["weight_group_size"]) is not int
                        or node.attributes["output_is_reduced"] is not False
                        or (node.attributes["routed_weight_dtype"] == "int4" and (
                            node.attributes["weight_group_size"] != 32 or node.attributes["routing_source_stage"] != "kimi_moe_route"
                            or node.attributes["routed_scaling_factor"] != 1.0))
                        or (node.attributes["routed_weight_dtype"] != "int4" and (
                            node.attributes["weight_group_size"] != 0 or node.attributes["routing_source_stage"] != "dsv32_moe_route"))):
                    raise ValueError("symbolic grouped experts require matching checkpoint storage, partial outputs and the declared routing stage")
                counts, local, owned = (tuple(node.attributes[name]) for name in (
                    "tokens_per_expert", "local_tokens_per_expert", "owned_experts"))
                expert_group = pool.parallel_plan.rank_groups.for_rank(
                    "expert", pool.logical_rank_for_physical_rank(node.rank))
                expected_rank = expert_group.index(pool.logical_rank_for_physical_rank(node.rank)) if grouped_ep8 else 0
                from src.kernel.kernel_spec import linear_expert_partition
                expected_owned = linear_expert_partition(node.shape["experts"], config.expert_parallel_size, expected_rank)
                expected_local = tuple(count if expert in expected_owned else 0 for expert, count in enumerate(counts))
                if (any(type(value) is not int for value in (*counts, *local, *owned))
                        or local != expected_local or owned != expected_owned
                        or type(node.attributes["expert_rank"]) is not int or node.attributes["expert_rank"] != expected_rank
                        or type(node.attributes["expert_parallel_size"]) is not int
                        or node.attributes["expert_parallel_size"] != config.expert_parallel_size):
                    raise ValueError("symbolic grouped experts must preserve expert_rank, owned_experts and the declared global/local expert axes")
                if "symbolic_zero_extent_parameter_roles" in node.attributes:
                    if (not _symbolic_glm_pcp_projection(node, pool) or node.attributes["routed_weight_dtype"] != "bf16"
                            or node.attributes["symbolic_zero_extent_parameter_roles"] != ("w13_weight", "w2_weight")
                            or node.attributes["memory_input_roles"] != ("hidden_states", "routing_weights", "expert_ids")
                            or node.attributes["memory_output_roles"] != ("output",)
                            or node.attributes["expert_parameter_layout"] != "unquantized_checkpoint_expert_major_nk"
                            or "symbolic_parameter_inputs" in node.attributes
                            or config.expert_parallel_size <= node.shape["experts"]):
                        raise ValueError("empty grouped experts require the explicit GLM BF16 source")
                    return _explicit_source_expert_specs(node, hardware_id, hardware)
                if node.attributes["routed_weight_dtype"] == "fp8":
                    if config.enable_expert_parallel and (any(key.startswith("source_fp8_") for key in node.attributes)
                            or _symbolic_glm_mtp_data_projection(node, pool)
                            or _symbolic_glm_fp8_target_dp_projection(node, pool)
                            or _symbolic_glm_fp8_rank_local_mtp_ep_projection(node, pool)):
                        raise NotImplementedError("GLM block-FP8 EP source uses deferred Cutlass input quantization; Goal05 pricing is unbound, mapped Triton EP8 is not this source")
                    # The existing H200 block-FP8 launch census is for
                    # DeepSeek-V3, not this GLM geometry/provider. Preserve
                    # the expert operation without borrowing BF16 pricing.
                    attributes = _fused_moe_prediction_attributes(node)
                    attributes["mathematical_program"] = ("block_fp8_replicated_ep_swiglu_v1"
                        if grouped_ep8 else "block_fp8_swiglu_v1")
                    return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id,
                        shape=node.shape, dtype="bf16", variant="routed_experts", attributes=attributes,
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if node.attributes["routed_weight_dtype"] == "int4" and kimi_ep8:
                    attributes = _fused_moe_prediction_attributes(node)
                    attributes["mathematical_program"] = "kimi_ep_marlin_swiglu_v1"
                    return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id,
                        shape=node.shape, dtype="bf16", variant="routed_experts", attributes=attributes,
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if node.attributes["routed_weight_dtype"] == "int4":
                    return _target_fused_moe_specs(node, hardware_id, hardware,
                        model_contract="kimi_k2_6_vllm_0_27_1_logical_v1")
                return _target_fused_moe_specs(node, hardware_id, hardware,
                    model_contract="glm_moe_dsa_vllm_0_30_0_logical_v1")
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_v4_swa_insert", "vllm_0_30_symbolic_v4_short_topk",
                    "vllm_0_30_symbolic_v4_sparse")):
                if not hardware.supports_source_contract(node.attributes["source_component"]):
                    raise NotImplementedError("symbolic V4 attention middle currently binds H200 source")
                component = node.attributes["source_component"]
                if component == "vllm_0_30_symbolic_v4_swa_insert":
                    if node.kind is not WorkloadKind.KV_UPDATE or node.variant != "dsv4_qnorm_rope_kv_insert":
                        raise ValueError("symbolic V4 insert has an unexpected operation")
                    return _target_kv_cache_update_specs(node, hardware_id, hardware)
                if component == "vllm_0_30_symbolic_v4_short_topk":
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "dsv4_short_context_topk":
                        raise ValueError("symbolic V4 short-context fill has an unexpected operation")
                    return (_selected_target_compute_spec(node, hardware_id, hardware,
                        KernelKind.VECTOR, node.variant, _vector_shape(node),
                        {"position_bytes": node.attributes["position_bytes"]}),)
                if node.kind is not WorkloadKind.EAGER_ATTN or node.variant not in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
                    raise ValueError("symbolic V4 sparse attention has an unexpected operation")
                return (_target_sparse_attention_spec(node, hardware_id, hardware),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_v4_compressor"):
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.KV_UPDATE
                        or node.variant not in ("dsv4_compressor_state_cache", "dsv4_compressed_state_cache")):
                    raise NotImplementedError("symbolic V4 compressor requires H200 partial/finish source")
                if "finish_stage" in node.attributes:
                    if (node.variant != "dsv4_compressed_state_cache"
                            or node.shape["compress_ratio"] != 128
                            or node.attributes["finish_stage"] not in ("compress", "store")):
                        raise ValueError("symbolic V4 compressor split stage disagrees with its source")
                    return (_target_kv_cache_update_spec(node, hardware_id, hardware),)
                return _target_kv_cache_update_specs(node, hardware_id, hardware)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_batched_linear"):
                if node.kind is not WorkloadKind.LINEAR or node.variant != "batched_gemm":
                    raise ValueError("symbolic batched linear requires a batched GEMM node")
                return (KernelSpec(kind=KernelKind.GEMM, hardware_id=hardware_id,
                    shape=node.shape, dtype=node.dtype, variant=node.variant,
                    attributes={key: node.attributes[key] for key in (
                        "batch_layout", "a_batch_stride_elements", "b_batch_stride_elements",
                        "c_batch_stride_elements", "weight_dtype", "weight_group_size")},
                    launch=None, execution_support="unsupported",
                    unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "symbolic_dsa_index_cache_update"):
                if (not _symbolic_glm_pcp_projection(node, pool) or node.kind is not WorkloadKind.KV_UPDATE
                        or node.variant != "dsa_fp8_index_cache" or node.dtype != "uint8"):
                    raise NotImplementedError("separate DSA index-cache insertion requires explicit GLM PCP source")
                return (_target_kv_cache_update_spec(node, hardware_id, hardware),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_dsa"):
                if ((not unbound_target_source and not _symbolic_glm_pcp_projection(node, pool)
                        and not hardware.supports_source_contract(node.attributes["source_component"])) or node.kind is not WorkloadKind.VECTOR
                        or node.variant not in ("dsv32_fused_norm_rope", "dsv32_fused_q")):
                    raise NotImplementedError("symbolic V32 fused attention prefix requires H200 source")
                names = (("has_indexer", "index_rope_interleave", "mla_cache_dtype", "cache_slot_scope")
                    if node.variant == "dsv32_fused_norm_rope" else
                    ("has_indexer", "index_rope_interleave", "quantize_mqa"))
                if node.variant == "dsv32_fused_norm_rope" and node.attributes["cache_slot_scope"] == "no_metadata_no_cache_slots":
                    names += ("source_no_attention_metadata", "index_k_output_present", "topk_active_fill")
                if unbound_target_source or (_symbolic_glm_pcp_projection(node, pool)
                        and HWSpec.source_contract_supported("glm_tp8_pp4_ep8_source_projection", hardware_id)):
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in names},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                return (_selected_target_compute_spec(node, hardware_id, hardware,
                    KernelKind.VECTOR, node.variant, node.shape,
                    {name: node.attributes[name] for name in names}),)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_dsa_indexer", "vllm_0_30_symbolic_dsa_sparse")):
                if "source_cost_model" in node.attributes and node.attributes["source_cost_model"] is not None:
                    if node.attributes["source_cost_model"] != "vllm_0_30_bounded_stage_roofline_v1":
                        # Normalizing an unknown non-None cost model to None
                        # would drop the explicit program and silently fall
                        # back to aggregate math; fail here at the real
                        # WorkloadNode entry instead.
                        raise ValueError(
                            f"unknown DSA source_cost_model: {node.attributes['source_cost_model']!r}")
                    if node.attributes["source_tie_refinement_policy"] != "capacity_bound":
                        raise ValueError("DSA source_cost_model requires its explicit tie refinement policy")
                if (not unbound_target_source and not _symbolic_glm_pcp_projection(node, pool)
                        and not hardware.supports_source_contract(node.attributes["source_component"])) or node.kind is not WorkloadKind.EAGER_ATTN:
                    raise NotImplementedError("symbolic DSA suffix currently requires H200 source")
                if node.attributes["selection_cardinality_source"] != "declared_causal_topk":
                    raise ValueError("symbolic DSA suffix requires explicit selected-cardinality applicability")
                if node.attributes["source_component"] == "vllm_0_30_symbolic_dsa_indexer":
                    if node.variant not in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed"):
                        raise ValueError("symbolic DSA indexer requires its single-token decode source")
                    kind = KernelKind.SPARSE_ATTN_INDEXER
                else:
                    if node.variant not in ("dsa_sparse_decode", "dsa_sparse_prefill", "dsa_sparse_mixed"):
                        raise ValueError("symbolic DSA sparse attention requires its BF16 decode source")
                    kind = KernelKind.SPARSE_ATTN
                if node.variant in ("dsa_indexer_prefill", "dsa_indexer_mixed", "dsa_sparse_prefill", "dsa_sparse_mixed"):
                    source = node.attributes["prefill_attention_source"]
                    if (source not in ("force_sparse_mqa", "dense_mha_fresh")
                            or (source == "dense_mha_fresh" and (kind is not KernelKind.SPARSE_ATTN_INDEXER
                                or node.attributes["mathematical_program"] not in ("dsa_weighted_relu_dense_mha_v1", "dsa_weighted_relu_specdecode_v1")))):
                        raise ValueError("DSA symbolic prefill/mixed requires its declared MQA or fresh dense-MHA source")
                    config = pool.parallel_plan.config
                    if (not _symbolic_glm_pcp_projection(node, pool) and ((not unbound_target_source and config.tensor_parallel_size not in (1, 8))
                            or (config.enable_expert_parallel and config.expert_parallel_size != 8)
                            or ((config.pipeline_parallel_size != 1 and not (_symbolic_glm_pipeline_projection(node, pool)))
                            or any(value != 1 for value in (config.sequence_parallel_size,
                                config.context_parallel_size, config.data_parallel_size))))):
                        raise NotImplementedError("DSA source requires TP1/8 PP/SP/CP/DP1 and EP-off or EP8")
                if kind is KernelKind.SPARSE_ATTN_INDEXER:
                    return (_target_sparse_indexer_spec(node, hardware_id, hardware,
                        pool.parallel_plan.config.context_parallel_size),)
                return (_target_sparse_attention_spec(node, hardware_id, hardware),)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_v4_inverse_rope", "vllm_0_30_symbolic_v4_grouped_output")):
                if not hardware.supports_source_contract(node.attributes["source_component"]):
                    raise NotImplementedError("symbolic V4 FP8 output projection currently binds SM90 H200 source")
                if node.attributes["source_component"] == "vllm_0_30_symbolic_v4_inverse_rope":
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "dsv4_inv_rope_fp8_quant":
                        raise ValueError("symbolic V4 inverse-RoPE has an unexpected operation")
                    return (_selected_target_compute_spec(node, hardware_id, hardware,
                        KernelKind.VECTOR, node.variant, _vector_shape(node),
                        {"quant_group_size": node.attributes["quant_group_size"]}),)
                if node.kind is not WorkloadKind.LINEAR or node.variant != "batched_gemm":
                    raise ValueError("symbolic V4 output einsum has an unexpected operation")
                return _target_gemm_specs(node, hardware_id, hardware, False)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_v4_qkv_norm"):
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.NORM
                        or node.variant != "dsv4_q_kv_rmsnorm"):
                    raise NotImplementedError("symbolic fused Q/KV norm requires H200 V4 source")
                return (_selected_target_compute_spec(node, hardware_id, hardware,
                    KernelKind.VECTOR, node.variant, node.shape, {"epsilon": node.attributes["epsilon"]}),)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_v4_indexer_query", "vllm_0_30_symbolic_v4_scored_indexer")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or 16 % config.tensor_parallel_size != 0
                        or config.enable_expert_parallel or not config.is_rank_local
                        or (config.pipeline_parallel_size != 1 and not _symbolic_v4_pipeline_projection(node, pool))):
                    raise NotImplementedError("symbolic V4 indexer requires H200 legal TP rank-local EP-off and a declared PP projection")
                if node.attributes["source_component"] == "vllm_0_30_symbolic_v4_indexer_query":
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "dsv4_indexer_q_rope_quant":
                        raise ValueError("symbolic V4 indexer query has an unexpected operation")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={}, launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if (node.kind is not WorkloadKind.EAGER_ATTN
                        or node.variant not in ("dsv4_indexer_prefill", "dsv4_indexer_decode")
                        or node.attributes["selection_cardinality_source"] != "declared_causal_topk"):
                    raise ValueError("symbolic V4 scored indexer disagrees with the declared source domain")
                return (_target_sparse_indexer_spec(node, hardware_id, hardware, config.context_parallel_size),)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "vllm_0_30_symbolic_v4_routing", "vllm_0_30_symbolic_v4_marlin")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or 16 % config.tensor_parallel_size != 0
                        or not config.is_rank_local
                        or (config.pipeline_parallel_size != 1 and not _symbolic_v4_pipeline_projection(node, pool))):
                    raise NotImplementedError("symbolic V4 modular routing/experts require H200 legal TP rank-local EP-off and a declared PP projection")
                if node.attributes["source_component"] == "vllm_0_30_symbolic_v4_routing":
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "dsv4_moe_route":
                        raise ValueError("symbolic V4 routing has an unexpected operation")
                    return (_selected_target_compute_spec(node, hardware_id, hardware,
                        KernelKind.VECTOR, node.variant, _vector_shape(node),
                        {key: node.attributes[key] for key in (
                            "expert_id_selection", "routed_scaling_factor", "routing_source")}),)
                if node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full":
                    raise ValueError("symbolic V4 Marlin has an unexpected operation")
                return _target_fused_moe_specs(node, hardware_id, hardware,
                    model_contract="deepseek_v4_vllm_0_30_0_logical_v1")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_27_1_symbolic_masked_embedding_gather"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract("vllm_0_27_1_symbolic_masked_embedding")
                        or config.tensor_parallel_size not in (4, 8) or config.enable_expert_parallel
                        or any(value != 1 for value in (config.pipeline_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))
                        or (graph.source_model_execution_version is not None
                            and graph.source_model_execution_version != "vllm_0_27_1")
                        or node.kind is not WorkloadKind.GATHER or node.variant != "embedding" or node.dtype != "bf16"
                        or node.attributes["model_execution_version"] != "vllm_0_27_1"
                        or node.attributes["embedding_source"] != "vllm_0_27_1_masked_index_select"
                        or node.attributes["source_operation"] != "index_select_dim0"
                        or node.attributes["index_bytes"] != 8 or node.attributes["gather_factor"] != 1
                        or node.attributes["index_count"] != node.shape["rows"]
                        or node.attributes["scattered"] is not True
                        or node.attributes["materialized_output"] is not True
                        or set(node.shape) != {"elements", "rows", "hidden", "source_rows"}
                        or node.shape["elements"] != node.shape["rows"] * node.shape["hidden"]):
                    raise ValueError("vLLM0.27.1 masked gather requires its complete BF16/INT64 TP4/8 source")
                masks = tuple(graph.node(name) for name in graph.predecessors(node.node_id)
                    if "source_component" in graph.node(name).attributes
                    and graph.node(name).attributes["source_component"] == "vllm_0_27_1_symbolic_masked_embedding"
                    and graph.node(name).attributes["stage"] == "mask_ids")
                if len(masks) != 1:
                    raise ValueError("vLLM0.27.1 masked gather requires its actual mask-ID producer")
                mask, = masks
                if (mask.rank != node.rank or mask.pool_name != node.pool_name
                        or mask.attributes["model_execution_version"] != "vllm_0_27_1"
                        or mask.attributes["mask_program"] != "vllm_0_27_1"
                        or mask.attributes["tensor_parallel_size"] != config.tensor_parallel_size
                        or mask.attributes["tensor_parallel_rank"] != pool.ranks.index(node.rank)
                        or tuple(node.attributes["symbolic_tensor_inputs"]) != tuple(mask.attributes["symbolic_tensor_outputs"][:1])
                        or (mask.shape["rows"], mask.shape["hidden"]) != (node.shape["rows"], node.shape["hidden"])
                        or node.shape["source_rows"] != -(-mask.attributes["vocab_size"] // 64) * 64 // config.tensor_parallel_size):
                    raise ValueError("vLLM0.27.1 masked gather disagrees with its actual INT64 mask port or padded shard")
                # Every rank gathers every masked row, including placeholder
                # row0 on non-owner ranks. Reuse generic typed gather bytes;
                # table extent stays on the actual source, with no0.11 launch.
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                    shape=_vector_shape(node), dtype=node.dtype, variant="gather",
                    attributes={name: node.attributes[name] for name in (
                        "gather_factor", "index_count", "index_bytes", "scattered", "materialized_output")},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in ("vllm_0_11_symbolic_masked_embedding", "vllm_0_18_symbolic_masked_embedding", "vllm_0_24_symbolic_masked_embedding", "vllm_0_27_1_symbolic_masked_embedding", "vllm_0_30_symbolic_masked_embedding")):
                config = pool.parallel_plan.config
                if node.attributes["source_component"] == "vllm_0_18_symbolic_masked_embedding":
                    if node.attributes["mask_program"] != "vllm_ascend_no_added_vocab":
                        raise ValueError("Ascend0.18 embedding requires its no-added-vocabulary source")
                elif node.attributes["source_component"] in ("vllm_0_11_symbolic_masked_embedding", "vllm_0_24_symbolic_masked_embedding", "vllm_0_27_1_symbolic_masked_embedding"):
                    expected = {"vllm_0_11_symbolic_masked_embedding": "vllm_0_11",
                        "vllm_0_24_symbolic_masked_embedding": "vllm_0_24",
                        "vllm_0_27_1_symbolic_masked_embedding": "vllm_0_27_1"}[node.attributes["source_component"]]
                    if node.attributes["mask_program"] != expected:
                        raise ValueError("masked embedding source identity disagrees with its expression program")
                elif "mask_program" in node.attributes and node.attributes["mask_program"] in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1"):
                    raise ValueError("masked embedding source identity disagrees with its expression program")
                if node.attributes["source_component"] == "vllm_0_27_1_symbolic_masked_embedding" and (
                        config.tensor_parallel_size not in (4, 8) or config.enable_expert_parallel
                        or node.attributes["model_execution_version"] != "vllm_0_27_1"
                        or (graph.source_model_execution_version is not None
                            and graph.source_model_execution_version != "vllm_0_27_1")
                        or (node.attributes["stage"] == "mask_output" and node.dtype != "bf16")):
                    raise ValueError("vLLM0.27.1 masked embedding requires its exact BF16 TP4/8 caller")
                if (not hardware.supports_source_contract(node.attributes["source_component"])
                        or config.tensor_parallel_size < 2
                        or (config.enable_expert_parallel and not ((config.tensor_parallel_size, config.expert_parallel_size) == (2, 2) or ((hardware.supports_source_contract("mla_tp8_ep8_source_projection") or _symbolic_mla_pipeline_projection(node, pool)) and (config.tensor_parallel_size, config.expert_parallel_size) == (8, 8))))
                        or (config.pipeline_parallel_size != 1 and not (
                            node.attributes["source_component"] == "vllm_0_11_symbolic_masked_embedding"
                            and hardware.supports_source_contract("vllm_0_11_masked_embedding_pipeline_endpoint")
                            and _symbolic_pipeline_endpoint(node, pool, 0)))
                        or any(value != 1 for value in (config.sequence_parallel_size,
                            config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("masked TP embedding requires its CUDA-alike EP-off source")
                if (node.kind is not WorkloadKind.VECTOR or node.variant != "vllm_masked_embedding"
                        or node.attributes["tensor_parallel_size"] != config.tensor_parallel_size
                        or node.attributes["tensor_parallel_rank"] != pool.ranks.index(node.rank)):
                    raise ValueError("masked embedding stage disagrees with deployment")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                    shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                    attributes={key: node.attributes[key] for key in (
                        "stage", "vocab_size", "tensor_parallel_size", "tensor_parallel_rank", "mask_program")
                        if key != "mask_program" or key in node.attributes},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_27_1_symbolic_vocab_embedding_aggregate"):
                if (not v3_parallel or node.attributes["native_source_equivalence_bound"] is not False
                        or node.attributes["aggregated_source_operations"] != ("mask_ids", "int64_gather", "masked_fill")
                        or node.attributes["omitted_auxiliary_details"] != (
                            "masked_id_and_mask_materializations", "foreign_row_zero_gather_reads", "separate_launches")):
                    raise ValueError("V3 TP embedding requires its declared unbound useful-payload aggregation")
                return (_source_vocab_embedding_spec(node, pool, hardware_id),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in ("vllm_0_30_symbolic_vocab_embedding", "sglang_0_5_19_symbolic_vocab_embedding")):
                if not hardware.supports_source_contract(node.attributes["source_component"]):
                    raise NotImplementedError("symbolic fused TP embedding requires a NVIDIA CUDA source")
                return (_source_vocab_embedding_spec(node, pool, hardware_id),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_dspark_regular_mhc_pre"):
                if not hardware.supports_source_contract("vllm_0_30_symbolic_mhc") or node.variant not in ("mhc_prenorm","mhc_pre_norm"):
                    raise ValueError("DSpark expanded first mHC requires its H200 source stages")
                shape=_vector_shape(node)
                attributes=({"source_prenorm_backend":"tilelang"} if node.variant=="mhc_prenorm" else {})
                if node.variant=="mhc_prenorm":
                    return (KernelSpec(KernelKind.VECTOR,hardware_id,shape,dtype=node.dtype,variant=node.variant,
                        attributes=attributes,launch=None,execution_support="unsupported",unsupported_reason="launch_selector_unavailable"),)
                return (_selected_target_compute_spec(node,hardware_id,hardware,KernelKind.VECTOR,node.variant,shape,attributes),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_v4_pipeline_prenorm"):
                config = pool.parallel_plan.config
                if (not _symbolic_v4_pipeline_projection(node, pool) or node.kind is not WorkloadKind.VECTOR
                        or node.variant not in ("mhc_prenorm", "mhc_pre_norm") or node.dtype != "bf16"
                        or node.shape["n_splits"] != 1 or node.attributes["source_use_deep_gemm"] is not False):
                    raise ValueError("V4 stage-entry prenorm requires its nonbroadcast optional TileLang source")
                if node.variant == "mhc_prenorm":
                    return (KernelSpec(KernelKind.VECTOR, hardware_id, _vector_shape(node), dtype="bf16",
                        variant=node.variant, attributes={"source_prenorm_backend": "tilelang"}, launch=None,
                        execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                return (_selected_target_compute_spec(node, hardware_id, hardware, KernelKind.VECTOR,
                    node.variant, _vector_shape(node), {}),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_symbolic_mhc"):
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.VECTOR
                        or node.variant not in ("mhc_pre", "mhc_post_pre", "mhc_post", "hc_head")):
                    raise NotImplementedError("symbolic mHC currently binds H200 V4 TileLang wrappers only")
                if type(node.attributes["source_use_deep_gemm"]) is not bool:
                    raise TypeError("symbolic mHC requires explicit source DeepGEMM branch")
                return _target_mhc_specs(node, hardware_id, hardware)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "symbolic_gqa_mathematical", "fa3_gqa_shared_metadata_prepare", "fa3_full_aot_prepare", "fa3_full_aot_clear", "fa3_expanded_mla_prefill_call",
                        "fa3_mla_split_prefill_call", "fa3_mla_unsplit_prefill_call")):
                return _source_flash_attn_specs(node, hardware_id, hardware)
            if ("source_component" in node.attributes and node.attributes["source_component"] in (
                    "sglang_0_5_19_symbolic_grouped_routing", "sglang_0_5_19_symbolic_fused_shared_moe",
                    "sglang_0_5_19_symbolic_logits_boundary")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"])
                        or config.tensor_parallel_size != 8 or config.pipeline_parallel_size != 1
                        or not config.is_rank_local or config.enable_expert_parallel):
                    raise NotImplementedError("SGLang source requires H200 TP8 PP/SP/CP/DP1 EP-off")
                component = node.attributes["source_component"]
                if component == "sglang_0_5_19_symbolic_grouped_routing":
                    if node.kind is not WorkloadKind.VECTOR or node.variant != "moe_topk" or node.dtype != "fp32":
                        raise ValueError("SGLang grouped routing requires FP32 logits")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in (
                            "source_program", "mathematical_program", "normalize_topk_probability", "expert_group_count",
                            "topk_group_count", "routed_scaling_factor", "fused_shared_experts")},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if component == "sglang_0_5_19_symbolic_logits_boundary":
                    if node.kind is not WorkloadKind.VECTOR:
                        raise ValueError("SGLang logits boundary requires a typed vector materialization")
                    if node.variant == "vllm_logits_fp32_cast":
                        if node.attributes["sampling_source"] != "sglang_0_5_19_logits_processor_bf16":
                            raise ValueError("SGLang logits cast requires its declared BF16-to-FP32 source")
                        names = ("sampling_source",)
                    elif node.variant == "vllm_mtp_eager":
                        if node.attributes["model_execution_version"] != "sglang_0_5_19":
                            raise ValueError("SGLang logits pruning requires its exact source version")
                        names = ("model_execution_version", "operation")
                        if "eagle_source" in node.attributes:
                            names += ("eagle_source",)
                        elif node.attributes["operation"] not in ("cumsum", "unary", "gather"):
                            raise ValueError("SGLang logits pruning has an unexpected materialization")
                        if node.attributes["operation"] == "gather":
                            names += ("index_bytes", "selected_indices")
                        elif node.attributes["operation"] == "eagle_tree_verify":
                            names += ("accepted_drafts",)
                        elif node.attributes["operation"] == "eagle_eh_norm":
                            names += ("epsilon",)
                        elif node.attributes["operation"] == "eagle_positions":
                            names += ("query_lens",)
                        elif node.attributes["operation"] == "eagle_topk1_finalize":
                            names += ("write_draft_token",)
                    elif node.variant == "vllm_plain_greedy_sample":
                        if (node.dtype != "fp32" or node.attributes["sampling_source"] != "sglang_0_5_19_plain_greedy_fp32"
                                or node.attributes["stage"] != "argmax"):
                            raise ValueError("SGLang ordinary greedy requires FP32 logits and INT64 argmax only")
                        names = ("sampling_source", "stage")
                    else:
                        raise ValueError("SGLang logits boundary has an unsupported source operation")
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                        dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in names},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if (node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full" or node.dtype != "bf16"
                        or node.attributes["fused_shared_experts"] != 1 or node.attributes["routed_scaling_factor"] != 1.0):
                    raise ValueError("SGLang fused shared experts require top-k-scaled routed weights and one shared expert")
                attributes = _fused_moe_prediction_attributes(node)
                attributes["mathematical_program"] = "sglang_0_5_19_fp8_fused_shared_swiglu_v1"
                return (KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id=hardware_id, shape=node.shape,
                    dtype=node.dtype, variant="routed_experts", attributes=attributes,
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "vllm_0_30_symbolic_logits_boundary", "vllm_0_11_symbolic_logits_boundary",
                        "vllm_0_18_symbolic_logits_boundary", "vllm_0_24_symbolic_logits_boundary", "vllm_0_27_1_symbolic_logits_boundary")):
                config = pool.parallel_plan.config
                source_versions = {
                    "vllm_0_30_symbolic_logits_boundary": "vllm_0_30_0",
                    "vllm_0_11_symbolic_logits_boundary": "vllm_0_11_0",
                    "vllm_0_18_symbolic_logits_boundary": "vllm_0_18_0",
                    "vllm_0_24_symbolic_logits_boundary": "vllm_0_24_0",
                    "vllm_0_27_1_symbolic_logits_boundary": "vllm_0_27_1",
                }
                if node.attributes["model_execution_version"] != source_versions[node.attributes["source_component"]]:
                    raise ValueError("symbolic logits source component disagrees with model_execution_version")
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or node.kind is not WorkloadKind.VECTOR
                        or (config.tensor_parallel_size not in (1, 2, 4, 8) and not v3_parallel
                            and node.attributes["model_execution_version"] != "vllm_0_30_0")
                        or (config.enable_expert_parallel and not (_symbolic_kimi_dp_projection(node, pool) or _symbolic_glm_pcp_projection(node, pool) or (config.tensor_parallel_size, config.expert_parallel_size) == (2, 2)
                            or (node.attributes["model_execution_version"] == "vllm_0_30_0" and (config.expert_parallel_size == config.tensor_parallel_size or _qwen030_dp_sp(config)))
                            or ((hardware.supports_source_contract("mla_tp8_ep8_source_projection") or _symbolic_mla_pipeline_projection(node, pool)) and (config.tensor_parallel_size, config.expert_parallel_size) == (8, 8))))
                        or (config.pipeline_parallel_size != 1 and not _symbolic_pipeline_endpoint(
                            node, pool, config.pipeline_parallel_size - 1))
                        or (not (_symbolic_kimi_dp_projection(node, pool) or _symbolic_glm_pcp_projection(node, pool)
                            or node.attributes["model_execution_version"] == "vllm_0_30_0" and _qwen030_dp_sp(config))
                            and any(value != 1 for value in (config.sequence_parallel_size,
                                config.context_parallel_size, config.data_parallel_size)))):
                    raise NotImplementedError("symbolic logits boundary requires CUDA-alike TP1/2/4/8 PP/SP/CP/DP1 EP-off")
                if node.variant == "vllm_logits_row_select":
                    names = ("model_execution_version", "index_source", "selected_indices")
                    if "disable_index_broadcast" in node.attributes:
                        names += ("disable_index_broadcast",)
                    shape = _vector_shape(node)
                    if node.attributes["index_source"] == "query_start_loc_int32_to_int64":
                        if "symbolic_tensor_inputs" not in node.attributes or len(node.attributes["symbolic_tensor_inputs"]) != 2:
                            raise ValueError("converted row indices require their explicit materialization producer")
                        index_view = node.attributes["symbolic_tensor_inputs"][1]
                        producers = tuple(graph.node(edge.source) for edge in graph.edges
                            if edge.destination == node.node_id and edge.kind == "data"
                            and "symbolic_tensor_outputs" in graph.node(edge.source).attributes
                            and index_view in graph.node(edge.source).attributes["symbolic_tensor_outputs"])
                        if (len(producers) != 1 or producers[0].kind is not WorkloadKind.ELEMENTWISE
                                or producers[0].dtype != "int32"
                                or "elements" not in producers[0].shape
                                or producers[0].shape["elements"] != shape["rows"]
                                or "operation" not in producers[0].attributes
                                or producers[0].attributes["operation"] != "cast_int32_int64"
                                or "tensor_iterator_source" not in producers[0].attributes
                                or producers[0].attributes["tensor_iterator_source"] != (
                                    "pytorch_2_8_sm80" if hardware.supports_source_contract("pytorch_2_8_advanced_indexing_int64_launch")
                                    else "pytorch_2_8_sm90" if hardware.supports_source_contract("pytorch_2_8_sm90_int32_int64_launch")
                                    else "aten_advanced_indexing_long")):
                            raise ValueError("converted row indices require their explicit materialization producer")
                    if ((hardware.supports_source_contract("pytorch_2_8_advanced_indexing_int64_launch")
                            or hardware.supports_source_contract("pytorch_2_8_sm90_row_gather_launch")) and node.dtype == "bf16"
                            and node.attributes["model_execution_version"] == "vllm_0_11_0"
                            and node.attributes["index_source"] == "query_start_loc_int32_to_int64"
                            and shape["hidden"] % 128 == 0
                            and max(2 * shape["source_rows"] * shape["hidden"],
                                    2 * shape["element_count"], 8 * shape["rows"]) < 2**31):
                        attributes = {name: node.attributes[name] for name in names}
                        return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=shape,
                            dtype=node.dtype, variant=node.variant, attributes=attributes,
                            launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                                node.dtype, node.variant, attributes, hardware=hardware)),)
                    if (hardware.supports_source_contract("pytorch_2_4_maca_scalar_index_launch")
                            and node.dtype == "bf16" and node.attributes["model_execution_version"] == "vllm_0_24_0"
                            and node.attributes["index_source"] == "query_start_loc_int32_to_int64"
                            and (shape["rows"] == 1 or shape["hidden"] == 1 or ("disable_index_broadcast" in node.attributes
                                and node.attributes["disable_index_broadcast"] is True))
                            and max(2 * shape["source_rows"] * shape["hidden"], 2 * shape["element_count"], 8 * shape["rows"]) < 2**31):
                        attributes = {name: node.attributes[name] for name in names}
                        return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=shape,
                            dtype=node.dtype, variant=node.variant, attributes=attributes,
                            launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                                node.dtype, node.variant, attributes, hardware=hardware)),)
                elif node.variant == "vllm_logits_gather_reorder":
                    if config.tensor_parallel_size == 1 or node.shape["tensor_parallel_size"] != config.tensor_parallel_size:
                        raise ValueError("symbolic logits reorder must match deployment TP2/4/8")
                    names = ("model_execution_version",)
                else:
                    raise ValueError("symbolic logits boundary has an unexpected operation")
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant, attributes={name: node.attributes[name] for name in names},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "vllm_0_30_symbolic_plain_greedy", "vllm_0_11_symbolic_plain_greedy",
                        "vllm_0_18_symbolic_plain_greedy", "vllm_0_24_symbolic_plain_greedy", "vllm_0_27_1_symbolic_plain_greedy")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"])
                        or (config.enable_expert_parallel and not (_symbolic_kimi_dp_projection(node, pool) or _symbolic_glm_pcp_projection(node, pool) or (config.tensor_parallel_size, config.expert_parallel_size) == (2, 2)
                            or (node.attributes["source_component"] == "vllm_0_30_symbolic_plain_greedy" and (config.expert_parallel_size == config.tensor_parallel_size or _qwen030_dp_sp(config)))
                            or ((hardware.supports_source_contract("mla_tp8_ep8_source_projection") or _symbolic_mla_pipeline_projection(node, pool)) and (config.tensor_parallel_size, config.expert_parallel_size) == (8, 8))))
                        or (config.tensor_parallel_size not in (1, 2, 4, 8) and not v3_parallel
                            and node.attributes["source_component"] != "vllm_0_30_symbolic_plain_greedy")
                        or (config.pipeline_parallel_size != 1 and not _symbolic_pipeline_endpoint(
                            node, pool, config.pipeline_parallel_size - 1))
                        or (not (_symbolic_kimi_dp_projection(node, pool) or _symbolic_glm_pcp_projection(node, pool)
                            or node.attributes["source_component"] == "vllm_0_30_symbolic_plain_greedy" and _qwen030_dp_sp(config))
                            and any(value != 1 for value in (config.sequence_parallel_size, config.context_parallel_size,
                                config.data_parallel_size)))):
                    raise NotImplementedError("symbolic CUDA-source greedy requires CUDA-alike TP1/2/4/8 PP/SP/CP/DP1 EP-off")
                if node.kind is not WorkloadKind.VECTOR or node.variant not in (
                        "vllm_logits_fp32_cast", "vllm_plain_greedy_sample"):
                    raise ValueError("symbolic plain greedy has an unexpected source operation")
                version = {
                    "vllm_0_30_symbolic_plain_greedy": "vllm_0_30",
                    "vllm_0_11_symbolic_plain_greedy": "vllm_0_11",
                    "vllm_0_18_symbolic_plain_greedy": "vllm_0_18",
                    "vllm_0_24_symbolic_plain_greedy": "vllm_0_24",
                    "vllm_0_27_1_symbolic_plain_greedy": "vllm_0_27_1",
                }[node.attributes["source_component"]]
                source_dtype = "fp16" if node.attributes["sampling_source"] in (
                    "vllm_0_30_plain_sampler_fp16", "vllm_0_11_plain_sampler_fp16",
                    "vllm_0_30_plain_greedy_fp16", "vllm_0_11_plain_greedy_fp16") else "bf16"
                suffix = ("_plain_sampler_" if node.variant == "vllm_logits_fp32_cast"
                    else "_plain_greedy_") + source_dtype
                if node.attributes["sampling_source"] != version + suffix:
                    raise ValueError("symbolic plain greedy source version must match its sampling program")
                if config.tensor_parallel_size > 1 and node.variant == "vllm_logits_fp32_cast":
                    incoming = tuple(graph.node(name) for name in graph.predecessors(node.node_id))
                    logits_rows = node.shape["rows"]
                    if (version in ("vllm_0_27_1", "vllm_0_30") and len(incoming) == 1
                            and incoming[0].kind is WorkloadKind.VECTOR and incoming[0].variant == "vllm_mtp_eager"
                            and incoming[0].attributes["operation"] == "gather"):
                        selection = incoming[0]
                        if (selection.attributes["source_component"] != version + "_symbolic_mtp_eager"
                                or selection.attributes["model_execution_version"] != (
                                    "vllm_0_27_1" if version == "vllm_0_27_1" else "vllm_0_30_0")
                                or selection.dtype != source_dtype or selection.rank != node.rank
                                or selection.shape["rows"] != node.shape["rows"]
                                or selection.shape["width"] != node.shape["vocab"]):
                            raise ValueError("MTP bonus sampling requires its rank-local complete-vocabulary selection")
                        logits_rows = selection.shape["source_rows"]
                        incoming = tuple(producer for name in graph.predecessors(selection.node_id)
                            for producer in (graph.node(name),) if
                            (producer.kind is WorkloadKind.VECTOR and producer.variant == "vllm_logits_gather_reorder")
                            or (producer.kind is WorkloadKind.COLLECTIVE and producer.attributes["operation"] == "all_gather"))
                    if (len(incoming) == 1 and incoming[0].kind is WorkloadKind.VECTOR
                            and incoming[0].variant == "vllm_logits_gather_reorder"):
                        reorder = incoming[0]
                        boundary = version + "_symbolic_logits_boundary"
                        if ("source_component" not in reorder.attributes
                                or reorder.attributes["source_component"] != boundary
                                or reorder.rank != node.rank or reorder.dtype != source_dtype
                                or reorder.shape["rows"] != logits_rows
                                or reorder.shape["tensor_parallel_size"] != config.tensor_parallel_size
                                or not 0 <= reorder.shape["vocab"] - node.shape["vocab"] < 64):
                            raise ValueError("TP symbolic greedy requires its rank-local complete logits reorder")
                        incoming = tuple(graph.node(name) for name in graph.predecessors(reorder.node_id))
                    if (len(incoming) != 1 or incoming[0].kind is not WorkloadKind.COLLECTIVE
                            or incoming[0].attributes["operation"] != "all_gather"):
                        raise ValueError("TP symbolic greedy requires the full-vocabulary all-gather producer")
                    producers = tuple(graph.node(edge.source) for edge in graph.edges
                        if edge.destination == incoming[0].node_id and edge.kind == "data")
                    if (len(producers) != config.tensor_parallel_size
                            or any(producer.kind is not WorkloadKind.LM_HEAD or producer.dtype != source_dtype for producer in producers)
                            or {producer.rank for producer in producers} != set(incoming[0].attributes["participants"])):
                        raise ValueError(f"TP symbolic greedy requires every {source_dtype.upper()} vocabulary-shard projection")
                    widths = {producer.shape["n"] for producer in producers}
                    if (len(widths) != 1 or any(producer.shape["m"] != logits_rows for producer in producers)
                            or not 0 <= next(iter(widths)) * config.tensor_parallel_size - node.shape["vocab"] < 64
                            or incoming[0].attributes["bytes"] != logits_rows * next(iter(widths)) * 2):
                        raise ValueError("TP symbolic greedy requires complete source vocabulary rows and collective bytes")
                attributes = {"sampling_source": node.attributes["sampling_source"]}
                if (node.variant == "vllm_logits_fp32_cast"
                        and attributes["sampling_source"] in ("vllm_0_24_plain_sampler_bf16", "vllm_0_11_plain_sampler_bf16")
                        and "copy_input_contiguous" in node.attributes):
                    attributes["copy_input_contiguous"] = node.attributes["copy_input_contiguous"]
                if (node.variant == "vllm_logits_fp32_cast"
                        and (hardware.supports_source_contract("pytorch_2_8_sm90_logits_cast_launch")
                            or hardware.supports_source_contract("pytorch_2_8_sm80_logits_cast_launch"))
                        and attributes["sampling_source"] == "vllm_0_11_plain_sampler_bf16"
                        and "copy_input_contiguous" in attributes and attributes["copy_input_contiguous"] is True
                        and 0 < node.shape["elements"] <= 2**29):
                    shape = _vector_shape(node)
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=shape, dtype=node.dtype, variant=node.variant, attributes=attributes,
                        launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                            node.dtype, node.variant, attributes, hardware=hardware)),)
                if node.variant == "vllm_logits_fp32_cast" and ("copy_dispatch" in node.attributes
                        or (hardware.supports_source_contract("pytorch_2_4_maca_logits_cast_launch")
                            and attributes["sampling_source"] == "vllm_0_24_plain_sampler_bf16"
                            and "copy_input_contiguous" in attributes
                            and attributes["copy_input_contiguous"] is True
                            and 0 < node.shape["elements"] < 2048)):
                    if "copy_dispatch" in node.attributes:
                        attributes["copy_dispatch"] = node.attributes["copy_dispatch"]
                    if not hardware.supports_source_contract("pytorch_2_4_maca_logits_cast_launch"):
                        raise ValueError("explicit copy dispatch requires the matching C500 Torch source")
                    shape = _vector_shape(node)
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=shape, dtype=node.dtype, variant=node.variant, attributes=attributes,
                        launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                            node.dtype, node.variant, attributes, hardware=hardware)),)
                if node.variant == "vllm_plain_greedy_sample":
                    attributes["stage"] = node.attributes["stage"]
                    if (hardware.supports_source_contract("pytorch_2_8_sm90_argmax_source_launch")
                            and attributes["sampling_source"] == "vllm_0_11_plain_greedy_bf16"
                            and attributes["stage"] == "argmax" and node.shape["vocab"] > 128
                            and max(4 * node.shape["elements"], 8 * node.shape["rows"]) < 2**31):
                        shape = _vector_shape(node)
                        attributes.update(source_num_compute_units=int(hardware.num_compute_units),
                            source_max_threads_per_unit=int(hardware.max_threads_per_unit))
                        return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                            shape=shape, dtype=node.dtype, variant=node.variant, attributes=attributes,
                            launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                                node.dtype, node.variant, attributes, hardware=hardware),
                            execution_support="unsupported", unsupported_reason="typed_service_unavailable"),)
                    torch_output_cast = (
                        (hardware.supports_source_contract("pytorch_2_8_sm90_int64_int32_launch")
                            or hardware.supports_source_contract("pytorch_2_8_sm80_int64_int32_launch"))
                        and node.attributes["sampling_source"] == "vllm_0_11_plain_greedy_bf16"
                        and node.attributes["stage"] == "output_int32"
                        and 0 < node.shape["rows"] <= 2**28)
                    if torch_output_cast or (hardware.supports_source_contract("pytorch_2_4_maca_small_int64_int32_launch")
                            and node.attributes["sampling_source"] == "vllm_0_24_plain_greedy_bf16"
                            and node.attributes["stage"] == "output_int32" and node.shape["rows"] < 2048):
                        shape = _vector_shape(node)
                        return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                            shape=shape, dtype=node.dtype, variant=node.variant, attributes=attributes,
                            launch=select_launch_config(hardware_id, KernelKind.VECTOR, shape,
                                node.dtype, node.variant, attributes, hardware=hardware)),)
                return (KernelSpec(
                    kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant, attributes=attributes,
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_plain_greedy"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 plain greedy source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is not WorkloadKind.VECTOR or node.variant not in (
                        "vllm_logits_fp32_cast", "vllm_plain_greedy_sample"):
                    raise ValueError("K3 plain greedy has an unexpected source operation")
                attributes = {"sampling_source": node.attributes["sampling_source"]}
                if node.variant == "vllm_plain_greedy_sample":
                    attributes["stage"] = node.attributes["stage"]
                return (KernelSpec(
                    kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                    dtype=node.dtype, variant=node.variant, attributes=attributes,
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_nope_context_prefill"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 NoPE context requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_nope_context":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                        dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "stage", "slot_mapping", "context_block_table", "cache_capacity_tokens", "rms_norm_eps")}
                        | {key: node.attributes[key] for key in ("context_start", "context_length") if key in node.attributes},
                        launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    if node.variant == "kimi_k3_nope_context_attention":
                        return (KernelSpec(
                            kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=node.shape,
                            dtype=node.dtype, variant=node.variant,
                            attributes={key: node.attributes[key] for key in (
                                "causal", "kv_dist", "kv_layout", "q_lens", "kv_lens", "return_softmax_lse",
                                "full_cuda_graphs", "max_cudagraph_capture_size")}
                            | {key: node.attributes[key] for key in ("output_target", "context_start") if key in node.attributes},
                            launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
                    return _target_mla_attention_specs(node, hardware_id, hardware)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 NoPE context contains an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_nope_decode"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 NoPE decode source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.LINEAR and node.variant == "kimi_k3_nope_bmm":
                    return (KernelSpec(
                        kind=KernelKind.GEMM, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant=node.variant,
                        attributes={"source_position": node.attributes["position"]},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_nope_decode":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "stage", "slot_mapping", "cache_capacity_tokens", "rms_norm_eps")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    return _target_mla_attention_specs(node, hardware_id, hardware)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 NoPE decode has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_nope_mixed"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 NoPE mixed source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.LINEAR and node.variant == "kimi_k3_nope_bmm":
                    return (KernelSpec(
                        kind=KernelKind.GEMM, hardware_id=hardware_id, shape=node.shape,
                        dtype=node.dtype, variant=node.variant,
                        attributes={"source_position": node.attributes["position"]},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_nope_mixed":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "stage", "slot_mapping", "cache_capacity_tokens", "rms_norm_eps")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    return _target_mla_attention_specs(node, hardware_id, hardware)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 NoPE mixed has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_27_1_symbolic_embedding"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract("vllm_0_27_1_symbolic_embedding") or config.enable_expert_parallel
                        or config.tensor_parallel_size != 1 or not config.is_rank_local
                        or (config.pipeline_parallel_size != 1 and not (v3_parallel
                            and _symbolic_pipeline_endpoint(node, pool, 0)))
                        or node.kind is not WorkloadKind.VECTOR or node.variant != "vllm_embedding_id_widen"
                        or node.dtype != "int32"
                        or node.attributes["embedding_source"] != "vllm_0_27_1_unquantized_tp1"):
                    raise NotImplementedError("vLLM0.27.1 TP1 embedding requires unbound H200 ordinary source")
                # VocabParallelEmbedding.forward unconditionally calls
                # masked_input.long(). Reuse the existing logical ATen cast;
                # source identity stays on the graph, with no sm80 launch.
                return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                    shape={"element_count": node.shape["elements"]}, dtype="int32", variant="elementwise",
                    attributes={"operation": "cast_int32_int64", "reads": 1, "writes": 1,
                        "tensor_iterator_source": "aten_advanced_indexing_long"}, launch=None,
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in (
                        "vllm_0_30_kimi_k3_tp1_embedding", "vllm_0_30_symbolic_embedding", "vllm_0_18_symbolic_embedding", "vllm_0_24_symbolic_embedding", "vllm_0_11_symbolic_embedding")):
                config = pool.parallel_plan.config
                distributed_embedding = (node.attributes["source_component"] == "vllm_0_30_symbolic_embedding"
                    and (_qwen030_dp_sp(config) or _symbolic_glm_pcp_projection(node, pool)))
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or (config.enable_expert_parallel and not distributed_embedding)
                        or (config.pipeline_parallel_size != 1 and not (
                            node.attributes["source_component"] in ("vllm_0_30_symbolic_embedding", "vllm_0_11_symbolic_embedding")
                            and _symbolic_pipeline_endpoint(node, pool, 0)))
                        or config.tensor_parallel_size != 1
                        or (not distributed_embedding and any(value != 1 for value in (
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size)))):
                    raise NotImplementedError("symbolic TP1 embedding requires its NVIDIA source and PP/SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_embedding_id_widen":
                    shape = _vector_shape(node)
                    attributes = {"embedding_source": node.attributes["embedding_source"]}
                    if "copy_dispatch" in node.attributes:
                        attributes["copy_dispatch"] = node.attributes["copy_dispatch"]
                    nvidia_cast = ((hardware.supports_source_contract("pytorch_2_8_advanced_indexing_int64_launch")
                            or hardware.supports_source_contract("pytorch_2_8_sm90_int32_int64_launch"))
                            and attributes["embedding_source"] == "vllm_0_11_unquantized_tp1"
                            and 8 * (shape["rows"] - 1) <= 2**31 - 1)
                    maca_cast = (hardware.supports_source_contract("pytorch_2_4_maca_int32_int64_launch")
                            and attributes["embedding_source"] == "vllm_0_24_unquantized_tp1"
                            and (shape["rows"] < 2048 or "copy_dispatch" in attributes))
                    if nvidia_cast or maca_cast:
                        return (KernelSpec(
                            kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=shape,
                            dtype=node.dtype, variant=node.variant, attributes=attributes,
                            launch=select_launch_config(
                                hardware_id, KernelKind.VECTOR, shape, node.dtype,
                                node.variant, attributes, hardware=hardware)),)
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id, shape=_vector_shape(node),
                        dtype=node.dtype, variant=node.variant,
                        attributes=attributes,
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.GATHER or node.variant != "gather":
                    raise ValueError("K3 embedding has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_fresh_prefill_logits"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 logits source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_logits_row_select":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "model_execution_version", "index_source", "selected_indices")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind not in (WorkloadKind.NORM, WorkloadKind.LM_HEAD):
                    raise ValueError("K3 logits boundary has an unexpected source operation")
                if node.kind is WorkloadKind.NORM:
                    # This static source component does not carry a runtime
                    # batch-invariant flag or actual tensor layout. Its
                    # ordinary RMSNorm arithmetic is explicit; no native
                    # 0.30 branch or compiler identity follows from that.
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=node.shape, dtype=node.dtype, variant=node.variant,
                        attributes={"epsilon": node.attributes["rms_norm_eps"],
                            "mathematical_program": "dense_bf16_source_flops_bytes_v1"},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] in ("vllm_0_30_kimi_k3_triton_fresh_prefill",
                                                               "vllm_0_30_kimi_k3_triton_mixed",
                                                               "vllm_0_30_kimi_k3_triton_context_prefill")):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 fresh Triton prefill source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "stage", "state_indices", "state_pool_slots", "query_start_loc", "has_initial_state", "gate_lower_bound")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 fresh Triton prefill has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_native_marlin_moe"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 native Marlin MoE source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_moe_route":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "expert_id_selection", "score_function", "expert_group_count", "topk_group_count",
                            "normalize_topk_probability", "routed_scaling_factor")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind not in (WorkloadKind.LINEAR, WorkloadKind.ROUTER):
                    raise ValueError("K3 native Marlin MoE frontend has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_marlin_experts"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 Marlin expert source requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is not WorkloadKind.FUSED_MOE or node.variant != "full":
                    raise ValueError("K3 Marlin expert source requires its explicit logical aggregate")
                return (KernelSpec(
                    kind=KernelKind.FUSED_MOE, hardware_id=hardware_id,
                    shape=node.shape, dtype=node.dtype, variant="kimi_k3_marlin_experts",
                    attributes={key: node.attributes[key] for key in (
                        "tokens_per_expert", "local_tokens_per_expert", "expert_parallel_size", "expert_rank",
                        "owned_experts", "expert_weight_tile_rows", "activation_beta",
                        "activation_linear_beta", "routed_scaling_factor")},
                    launch=None, execution_support="unsupported",
                    unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_tp1_native_latent_output"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 native latent output source chain requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind not in (WorkloadKind.NORM, WorkloadKind.LINEAR, WorkloadKind.ELEMENTWISE):
                    raise ValueError("K3 native latent output has an unexpected source operation")
                if node.kind in (WorkloadKind.NORM, WorkloadKind.ELEMENTWISE):
                    attributes = ({"epsilon": node.attributes["rms_norm_eps"]}
                        if node.kind is WorkloadKind.NORM else
                        {key: node.attributes[key] for key in ("operation", "reads", "writes")})
                    return (KernelSpec(kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=node.shape if node.kind is WorkloadKind.NORM else _vector_shape(node),
                        dtype=node.dtype, variant="rmsnorm" if node.kind is WorkloadKind.NORM else "elementwise",
                        attributes={**attributes, "mathematical_program": "dense_bf16_source_flops_bytes_v1"},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_nope_fresh_prefill"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 NoPE prefill source chain requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_nope_prefill":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in (
                            "stage", "slot_mapping", "cache_capacity_tokens", "rms_norm_eps")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    return _target_mla_attention_specs(node, hardware_id, hardware)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 NoPE prefill has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_triton_decode"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 Triton decode source chain requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_kda_decode":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in ("stage", "state_indices", "state_pool_slots", "gate_lower_bound")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 Triton decode has an unexpected source operation")
            if ("source_component" in node.attributes
                    and node.attributes["source_component"] == "vllm_0_30_kimi_k3_ordinary_mlp"):
                config = pool.parallel_plan.config
                if (not hardware.supports_source_contract(node.attributes["source_component"]) or config.enable_expert_parallel
                        or (config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or any(value != 1 for value in (
                            config.tensor_parallel_size,
                            config.sequence_parallel_size, config.context_parallel_size, config.data_parallel_size))):
                    raise NotImplementedError("K3 ordinary MLP component requires H200 TP1 nonempty PP SP/CP/DP1 EP-off")
                if node.kind is WorkloadKind.ACTIVATION and node.variant == "kimi_k3_situ":
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in ("activation_beta", "activation_linear_beta")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is not WorkloadKind.LINEAR:
                    raise ValueError("K3 ordinary MLP has an unexpected source operation")
            if node.kind is WorkloadKind.VECTOR and node.variant == "kimi_k3_attn_res":
                if node.attributes["source_component"] != "vllm_0_30_kimi_k3_attention_residual":
                    raise ValueError("K3 attention residual requires an explicit source component")
                if ((pool.parallel_plan.config.pipeline_parallel_size != 1 and not _symbolic_k3_pipeline_projection(node, pool))
                        or pool.parallel_plan.config.sequence_parallel_size != 1):
                    raise NotImplementedError("K3 attention residual source component requires nonempty PP and SP1")
                return (KernelSpec(
                    kind=KernelKind.VECTOR, hardware_id=hardware_id,
                    shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                    attributes={key: node.attributes[key] for key in (
                        "num_blocks", "block_write_idx", "block_capacity", "has_delta", "apply_output_norm")},
                    launch=None, execution_support="unsupported",
                    unsupported_reason="launch_selector_unavailable"),)
            if target_contract is not None:
                if node.kind is WorkloadKind.VECTOR and node.variant == "dsv4_indexer_q_rope_quant":
                    if target_contract != "deepseek_v4_vllm_0_30_0_logical_v1":
                        raise ValueError("V4 indexer Q source requires the V4 contract")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={}, launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_rejection_logits_materialize":
                    if target_contract not in ("glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1"):
                        raise ValueError("rejection sampler source currently requires GLM")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in
                                    ("sampling_source", "materialization_stage", "selected_indices")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_plain_greedy_sample":
                    if target_contract not in (
                            "glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1",
                            "kimi_k2_6_vllm_0_27_1_logical_v1", "kimi_k2_7_code_vllm_0_30_0_logical_v1",
                            "deepseek_v4_vllm_0_30_0_logical_v1"):
                        raise ValueError("plain greedy sampler requires an assigned GLM/Kimi/V4 contract")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in ("sampling_source", "stage")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_logits_gather_reorder":
                    if target_contract != "deepseek_v4_vllm_0_30_0_logical_v1":
                        raise ValueError("logits gather reorder requires the assigned V4 source contract")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={"model_execution_version": node.attributes["model_execution_version"]},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_logits_fp32_cast":
                    if target_contract not in (
                            "glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1", "kimi_k2_6_vllm_0_27_1_logical_v1", "kimi_k2_7_code_vllm_0_30_0_logical_v1",
                            "deepseek_v4_vllm_0_30_0_logical_v1"):
                        raise ValueError("target sampler cast requires an assigned GLM/Kimi/V4 contract")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={"sampling_source": node.attributes["sampling_source"]},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "vllm_logits_row_select":
                    if target_contract not in (
                            "glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1", "kimi_k2_6_vllm_0_27_1_logical_v1", "kimi_k2_7_code_vllm_0_30_0_logical_v1",
                            "deepseek_v4_vllm_0_30_0_logical_v1"):
                        raise ValueError("target logits source selection requires an assigned model contract")
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes={key: node.attributes[key] for key in
                                    ("model_execution_version", "index_source", "selected_indices")},
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant in (
                        "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
                    if target_contract not in ("glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1"):
                        raise ValueError("draft sampling source currently requires GLM")
                    attributes = {"sampling_source": node.attributes["sampling_source"]}
                    if node.variant == "dsv32_draft_row_select":
                        attributes.update({key: node.attributes[key] for key in ("selection_role", "selected_indices")})
                    elif node.variant == "dsv32_draft_buffer_copy":
                        attributes["copy_role"] = node.attributes["copy_role"]
                    elif node.variant == "dsv32_draft_id_scatter":
                        attributes["selected_indices"] = node.attributes["selected_indices"]
                    elif node.variant in ("dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback"):
                        attributes["index_preparation_mode"] = node.attributes["index_preparation_mode"]
                    elif node.variant == "dsv32_draft_step_metadata":
                        attributes["positions_alias_output"] = node.attributes["positions_alias_output"]
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes=attributes,
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.GATHER and "embedding_source" in node.attributes:
                    if target_contract not in (
                                "glm_5_2_vllm_0_27_1_logical_v1", "glm_moe_dsa_vllm_0_30_0_logical_v1",
                                "kimi_k2_6_vllm_0_27_1_logical_v1", "kimi_k2_7_code_vllm_0_30_0_logical_v1",
                                "deepseek_v4_vllm_0_30_0_logical_v1"):
                        raise ValueError("unsupported target embedding source")
                    return (_source_vocab_embedding_spec(node, pool, hardware_id),)
                if (node.kind is WorkloadKind.VECTOR and node.variant in (
                        "dsv32_fused_norm_rope", "dsv32_fused_q")):
                    return (_selected_target_compute_spec(
                        node, hardware_id, hardware, KernelKind.VECTOR,
                        node.variant, node.shape,
                        {name: node.attributes[name] for name in (
                            ("has_indexer", "index_rope_interleave", "mla_cache_dtype", "cache_slot_scope")
                            if node.variant == "dsv32_fused_norm_rope" else
                            ("has_indexer", "index_rope_interleave", "quantize_mqa"))}),)
                if node.kind is WorkloadKind.ROTARY:
                    return (_target_rope_spec(node, hardware_id, hardware),)
                if (node.kind is WorkloadKind.NORM
                        and node.variant in (
                            "dsv4_q_kv_rmsnorm", "dsv32_eh_norm", "dsv32_embed_eh_norm", "dsv32_embed_norm")):
                    return (_selected_target_compute_spec(
                        node, hardware_id, hardware, KernelKind.VECTOR,
                        node.variant, node.shape, {"epsilon": node.attributes["epsilon"]}
                        if node.variant == "dsv4_q_kv_rmsnorm" and "epsilon" in node.attributes else {}),)
                if (node.kind is WorkloadKind.NORM
                        and node.variant == "layernorm"):
                    return (_target_layernorm_spec(
                        node, hardware_id, hardware),)
                if node.kind is WorkloadKind.KV_UPDATE:
                    return _target_kv_cache_update_specs(node, hardware_id, hardware)
                if (node.kind is WorkloadKind.GATHER
                        and node.variant == "dsa_index_reuse"):
                    return (_target_sparse_indexer_spec(
                        node, hardware_id, hardware, pool.parallel_plan.config.context_parallel_size),)
                if node.kind is WorkloadKind.FLASH_ATTN:
                    return _target_mla_attention_specs(
                        node, hardware_id, hardware)
                if node.kind is WorkloadKind.EAGER_ATTN:
                    if node.variant.startswith(("dsa_indexer_", "dsv4_indexer_")):
                        return (_target_sparse_indexer_spec(
                            node, hardware_id, hardware, pool.parallel_plan.config.context_parallel_size),)
                    if node.variant.startswith(("dsa_sparse_", "dsv4_sparse_")):
                        return (_target_sparse_attention_spec(
                            node, hardware_id, hardware),)
                    raise ValueError(
                        "unknown target-model EAGER_ATTN logical variant: "
                        f"{node.variant!r}")
                if (node.kind is WorkloadKind.VECTOR
                        and node.variant in (
                            "mhc_pre", "mhc_post_pre", "mhc_post",
                            "hc_head")):
                    return _target_mhc_specs(node, hardware_id, hardware)
                if (node.kind is WorkloadKind.VECTOR
                        and node.variant == "dsv4_short_context_topk"):
                    return (_selected_target_compute_spec(
                        node, hardware_id, hardware, KernelKind.VECTOR,
                        node.variant, _vector_shape(node),
                        {"position_bytes": node.attributes["position_bytes"]}),)
                if node.kind is WorkloadKind.VECTOR and node.variant in ("dsv32_moe_route", "kimi_moe_route"):
                    route_attributes = {name: node.attributes[name] for name in (
                        "expert_id_selection", "routed_scaling_factor", "score_function",
                        "expert_group_count", "topk_group_count", "normalize_topk_probability")}
                    if "mathematical_program" in node.attributes:
                        route_attributes["mathematical_program"] = node.attributes["mathematical_program"]
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes=route_attributes,
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if node.kind is WorkloadKind.VECTOR and node.variant == "dsv4_moe_route":
                    route_attributes = {name: node.attributes[name] for name in (
                        "expert_id_selection", "routed_scaling_factor", "routing_source")}
                    if route_attributes["routing_source"] is not None:
                        return (_selected_target_compute_spec(
                            node, hardware_id, hardware, KernelKind.VECTOR,
                            node.variant, _vector_shape(node), route_attributes),)
                    return (KernelSpec(
                        kind=KernelKind.VECTOR, hardware_id=hardware_id,
                        shape=_vector_shape(node), dtype=node.dtype, variant=node.variant,
                        attributes=route_attributes,
                        launch=None, execution_support="unsupported",
                        unsupported_reason="launch_selector_unavailable"),)
                if (node.kind is WorkloadKind.VECTOR
                        and node.variant == "dsv4_inv_rope_fp8_quant"):
                    return (_selected_target_compute_spec(
                        node, hardware_id, hardware, KernelKind.VECTOR,
                        node.variant, _vector_shape(node),
                        {"quant_group_size": (
                            node.attributes["quant_group_size"])}),)
                if node.kind is WorkloadKind.FUSED_MOE:
                    return _target_fused_moe_specs(
                        node, hardware_id, hardware, model_contract=target_contract)
                if node.kind in (
                        WorkloadKind.LINEAR, WorkloadKind.LM_HEAD,
                        WorkloadKind.ROUTER):
                    specs = _target_gemm_specs(
                        node, hardware_id, hardware,
                        block_fp8_checkpoint and (
                            target_contract != "glm_moe_dsa_vllm_0_30_0_logical_v1"
                            or node.attributes["weight_dtype"] == "fp8"))
                    if (target_contract == "deepseek_v4_vllm_0_30_0_logical_v1"
                            and hardware.supports_source_contract("vllm_0_30_group_fp8_fp32_scales") and _graph_token_domain(node) is None):
                        specs = _vllm030_group_fp8_source_specs(specs, node)
                    return specs
            if node.kind is WorkloadKind.FLASH_ATTN:
                return _flash_attn_specs(node, hardware_id, hardware)
            if node.kind is WorkloadKind.NORM:
                return _norm_specs(node, hardware_id, hardware)
            if node.kind is WorkloadKind.FUSED_MOE:
                return _fused_moe_specs(
                    node, hardware_id, hardware, routing_relabelling)
            if node.kind in (
                    WorkloadKind.LINEAR, WorkloadKind.LM_HEAD,
                    WorkloadKind.ROUTER):
                specs = _gemm_node_specs(node, hardware_id, hardware)
                if (hardware.supports_source_contract("vllm_0_30_group_fp8_fp32_scales") and "symbolic_scale_layout" in node.attributes
                        and node.attributes["symbolic_scale_layout"] == "row_major_group128_fp32"):
                    specs = _vllm030_group_fp8_source_specs(specs, node)
                return specs
            if node.kind in _SINGLE_COMPUTE_KIND:
                return (_compute_spec(node, hardware_id, hardware),)
            if node.kind is WorkloadKind.EAGER_ATTN:
                return _eager_attn_specs(node, hardware_id, hardware)
            if node.kind in _COMMUNICATION_KINDS or node.kind in (
                    WorkloadKind.MEMCPY, WorkloadKind.KV_OFFLOAD):
                destination_hardware = None
                if node.kind is WorkloadKind.PD_KV_TRANSFER:
                    destination_pool = deployment.worker_pools[node.attributes["destination_pool"]]
                    if node.attributes["link"] == "infiniband":
                        destination_hardware = require_hardware(destination_pool.hardware_id, hardware_by_id)
                        if destination_hardware.interconnect["pcie_bw_bytes_s"] is None:
                            raise ValueError("bilateral transport requires receiver PCIe bandwidth")
                return (_transfer_spec(node, hardware_id,
                    destination_hardware=destination_hardware),)
            raise NotImplementedError(
                f"no lowering rule for {node.kind.value}")

        if memo is None:
            specs = build_specs()
        elif dspark_typed_role_key is not None:
            # Current mixed-port effective geometry belongs to this owner's
            # physical memo contract, alongside the immutable source node.
            specs = memo.lower(node, (logical_node_contracts[node.node_id], dspark_typed_role_key),
                hardware, build_specs)
        elif node.kind is WorkloadKind.FUSED_MOE:
            specs = memo.lower(
                node,
                _logical_node_contract(node, routing_relabelling),
                hardware,
                build_specs,
            )
        else:
            specs = memo.lower(
                node,
                logical_node_contracts[node.node_id],
                hardware,
                build_specs,
            )
        if not specs:
            raise ValueError(
                f"workload node {node.node_id!r} lowered to no KernelSpecs")
        if structural_target:
            specs = tuple(dataclasses.replace(spec, launch=None, execution_support="unsupported",
                unsupported_reason="launch_selector_unavailable") for spec in specs)
        external_dependencies = tuple(
            final_call_by_node[pred] for pred in graph.predecessors(node.node_id))
        previous = None
        node_call_ids = []
        for index, spec in enumerate(specs):
            # The source recorder uses :1 for repeated module invocations.
            # DSpark context/query FP8 expansion needs a distinct stage axis.
            call_id = (node.node_id if len(specs) == 1 else
                f"{node.node_id}:dspark-stage-{index}" if ("source_dspark_standalone" in node.attributes
                    or "source_v4_aux_capture" in node.attributes or node.variant == "vllm_gpu_v2_store_drafts") else
                f"{node.node_id}:{index}")
            dependencies = external_dependencies if previous is None else (previous,)
            if spec.transfer is None:
                resource = "compute"
            elif spec.kind is KernelKind.COMMUNICATION:
                resource = spec.transfer.resource_id
            else:
                resource = spec.transfer.link
            rank = node.rank
            call = (KernelCall(call_id, spec, node.pool_name, rank, resource, (node.node_id,), dependencies)
                if memo is None else memo.bind_call(call_id, spec, node.pool_name, rank, resource,
                    node.node_id, dependencies))
            source_launches = (memo.source_vector_recipe(node,
                logical_node_contracts[node.node_id], hardware_id, hardware)
                if memo is not None and node.kind in (
                    WorkloadKind.NORM, WorkloadKind.ACTIVATION,
                    WorkloadKind.ROTARY, WorkloadKind.KV_UPDATE)
                else _node_source_launches(node, hardware_id, hardware))
            calls.append(call if source_launches is None else dataclasses.replace(call, source_launches=source_launches))
            node_call_ids.append(call_id)
            previous = call_id
        final_call_by_node[node.node_id] = previous
        call_ids_by_node[node.node_id] = tuple(node_call_ids)
    if set(call_ids_by_node) != {node.node_id for node in graph.nodes}:
        raise AssertionError("kernel lowering did not consume every logical node")
    calls_by_id = {call.call_id: call for call in calls}
    if any(
            node_id not in calls_by_id[call_id].source_node_ids
            for node_id, call_ids in call_ids_by_node.items()
            for call_id in call_ids):
        raise AssertionError(
            "each logical source node must map to its lowered KernelCall")
    materialized_plan = KernelPlan(calls)
    topological_calls = materialized_plan.topological_order()
    block_repeats = []
    for repeat in graph.block_repeats:
        repeated_node_ids = set(repeat.node_ids)
        block_call_ids = tuple(
            call.call_id
            for call in topological_calls
            if any(
                node_id in repeated_node_ids
                for node_id in call.source_node_ids
            )
        )
        feedback_by_destination = {}
        for edge in repeat.feedback_edges:
            destination_call_id = call_ids_by_node[edge.destination][0]
            source_call_id = final_call_by_node[edge.source]
            if destination_call_id not in feedback_by_destination:
                feedback_by_destination[destination_call_id] = []
            feedback_by_destination[destination_call_id].append(source_call_id)
        feedback_dependencies = tuple(
            (
                call_id,
                tuple(feedback_by_destination[call_id]),
            )
            for call_id in block_call_ids
            if call_id in feedback_by_destination
        )
        block_repeats.append(KernelBlockRepeat(
            repeat.block_id,
            block_call_ids,
            repeat.repeat_count,
            feedback_dependencies,
            repeat.trace_node_ids,
        ))
    return KernelPlan(calls, block_repeats)


def _reuse_ep_native_frame_recipe(graph, deployment, owner, layout, *, source_layout):
    """Validate live EP fields against the ordinary compiler's cold recipe.

    Only canonical cold template nodes are inspected. Current shape/attribute
    values come directly from the source frame; no public node views are made.
    """
    from src.serving.workload.traced_forward import _NativeSourceFrame
    from src.serving.workload.runtime_inputs import (
        _routing_certificate_context, _routing_certificate_contexts_native)
    if owner not in _EP_SOURCE_RECIPE_LAYOUTS:
        return None
    if type(graph) is not WorkloadGraph or "_native_source_frame" not in graph.__dict__:
        return None
    frame = graph.__dict__["_native_source_frame"]
    if type(frame) is not _NativeSourceFrame or frame.owner(check_source=False) is not owner:
        return None
    if type(source_layout) is not tuple or len(source_layout) != 3:
        return None
    identity, template_nodes, bindings = source_layout
    expected_template, expected_bindings, dynamic, expected_edges = _EP_SOURCE_RECIPE_LAYOUTS[owner]
    template = identity() if type(identity) is weakref.ReferenceType else None
    if (type(template) is not WorkloadGraph or frame.template is not template
            or expected_template() is not template or template.nodes is not template_nodes
            or frame.layout[0]() is not template or frame.layout[1] is not template_nodes
            or bindings != expected_bindings or frame.layout[2] != bindings):
        return None
    behavior, dep, proof, edges = layout
    if not _ep_verifier_unchanged(behavior):
        return False
    current_dep = _ep_deployment_projection(deployment)
    if (current_dep is None or current_dep != dep or frame.workload_contract[1] is not expected_edges
            or len(template_nodes) != len(proof)):
        return False
    rows = _ep_unpadded_source_rows(graph)
    if rows is None:
        return False
    certificate_records = []
    current_records = []
    for index, kind, owned, oldinv in dynamic:
        base = template_nodes[index]
        if type(base) is not WorkloadNode or base.kind is not kind:
            return False
        shape, attrs = frame.fields(base.node_id)
        current_records.append((base.rank, kind, owned, oldinv, shape, attrs))
        if kind is WorkloadKind.FUSED_MOE:
            if _EP_SOURCE_DYNAMIC_FIELDS - set(attrs):
                return False
            certificate_records.append((attrs["routing_provenance"], attrs["tokens_per_expert"],
                shape["tokens"], shape["experts"], shape["topk"], attrs["moe_invocation_id"]))
    certificates_valid = _routing_certificate_contexts_native(tuple(certificate_records))
    invocations, new_names, local_arithmetic = {}, set(), {}
    for rank, kind, owned, oldinv, shape, attrs in current_records:
        if kind is WorkloadKind.FUSED_MOE:
            if (rank not in rows or type(shape["tokens"]) is not int
                    or shape["tokens"] != rows[rank]
                    or _EP_SOURCE_DYNAMIC_FIELDS - set(attrs)):
                return False
            counts = attrs["tokens_per_expert"]
            local = attrs["local_tokens_per_expert"]
            inv = attrs["moe_invocation_id"]
            if type(local) is not tuple:
                return False
            if not certificates_valid and not _routing_certificate_context(
                    attrs["routing_provenance"], counts, shape["tokens"],
                    shape["experts"], shape["topk"], inv):
                return False
            arithmetic_key = (id(counts), id(local), owned, rank, tuple(shape.items()))
            if arithmetic_key in local_arithmetic:
                retained_counts, retained_local = local_arithmetic[arithmetic_key]
                if retained_counts is not counts or retained_local is not local:
                    raise RuntimeError("synchronous local arithmetic identity collision")
            else:
                if any(type(count) is not int for count in local):
                    return False
                expected_local = [0] * shape["experts"]
                for expert in owned:
                    expected_local[expert] = counts[expert]
                if local != tuple(expected_local):
                    return False
                local_arithmetic[arithmetic_key] = (counts, local)
        else:
            inv = attrs["moe_invocation_id"]
            if type(inv) is not str or not inv:
                return False
        if oldinv in invocations:
            if invocations[oldinv] != inv:
                return False
        else:
            if inv in new_names:
                return False
            invocations[oldinv] = inv
            new_names.add(inv)
    return True


_ordinary_lower_workload = lower_workload
_COMPILED_NATIVE_PLANS = weakref.WeakKeyDictionary()


def _dense_compiler_source():
    from types import FunctionType
    return tuple((name, value, value.__code__, value.__defaults__,
        None if value.__kwdefaults__ is None else dict(value.__kwdefaults__))
        for name, value in globals().items() if type(value) is FunctionType)


def _dense_compiler_unchanged(snapshot):
    return all(name in globals() and globals()[name] is function
        and function.__code__ is code and function.__defaults__ == defaults
        and function.__kwdefaults__ == keywords
        for name, function, code, defaults, keywords in snapshot)



def _native_plan_binding_layout(frame, plan):
    representatives = {name:name for name in frame.ids}
    for repeat in plan.block_repeats:
        for representative, occurrences in repeat.trace_node_ids:
            for name in occurrences: representatives[name] = representative
    groups, guards = {}, []
    for target, origin, shapes, attrs in frame.layout[2]:
        base = frame.template.node(target)
        physical = tuple(name for name in attrs if name not in _SOURCE_OBJECT_ATTRIBUTES
            and name not in ("moe_invocation_id", "routing_provenance"))
        if not shapes and not physical: continue
        if base.kind is WorkloadKind.FLASH_ATTN:
            dynamic_shapes = tuple(name for name in shapes if name == "kv_len")
            dynamic_attrs = tuple(name for name in physical if name in ("kv_lens", "model_input_kv_lens"))
        elif (base.kind is WorkloadKind.FUSED_MOE and not shapes
                and base.attributes["source_component"] in ("symbolic_unquantized_experts", "symbolic_grouped_routed_experts")
                and all(name in ("tokens_per_expert", "local_tokens_per_expert") for name in physical)):
            dynamic_shapes, dynamic_attrs = shapes, physical
        elif (base.kind in (WorkloadKind.VECTOR, WorkloadKind.MEMCPY)
                and "fa3_full_aot_component" in base.attributes
                and all(name == "kv_len" for name in shapes)
                and all(name in ("kv_lens", "model_input_kv_lens", "fa3_full_aot_used_key") for name in physical)):
            dynamic_shapes, dynamic_attrs = shapes, physical
        else:
            dynamic_shapes, dynamic_attrs = (), ()
        fixed_shapes = tuple(name for name in shapes if name not in dynamic_shapes)
        fixed_attrs = tuple(name for name in physical if name not in dynamic_attrs)
        if fixed_shapes or fixed_attrs:
            # Only dense native frames may extend the original KV-only path.
            # These fields are never rebound into old specs: an exact mismatch
            # selects the complete ordinary compiler and replaces this plan.
            if frame.routing_ids: return None
            shape, attributes = frame.fields(target)
            expected = (tuple((key, shape[key]) for key in fixed_shapes),
                tuple((key, attributes[key]) for key in fixed_attrs))
            if not _ep_native_value(expected): return None
            guards.append((target, fixed_shapes, fixed_attrs, expected))
        if not dynamic_shapes and not dynamic_attrs: continue
        representative = representatives[target]
        if representative not in groups: groups[representative] = []
        groups[representative].append((target, dynamic_shapes, dynamic_attrs))
    calls = {}
    for call in plan.calls:
        if len(call.source_node_ids) != 1: return None
        name, = call.source_node_ids
        if name not in calls: calls[name] = []
        calls[name].append(call.call_id)
    if any(name not in calls for name in groups): return None
    by_call = {call.call_id: call for call in plan.calls}
    merged, keys = [], {}
    for name, rows in groups.items():
        node = frame.template.node(name)
        shapes, attrs = rows[0][1:]
        key = _native_lowering_group_key(node, shapes, attrs)
        ids = tuple(calls[name])
        specs = tuple(by_call[call_id].kernel_spec for call_id in ids)
        if key is not None and key in keys:
            index, original_specs = keys[key]
            if specs == original_specs:
                merged[index][1].extend(rows)
                merged[index][2].append(ids)
                continue
        if key is not None:
            keys[key] = len(merged), specs
        merged.append((name, list(rows), [ids]))
    return (tuple((name, tuple(rows), tuple(occurrences)) for name, rows, occurrences in merged), tuple(guards))


def _native_lowering_group_key(node, shapes, attrs):
    """Group equal physical source recipes, without folding calls or their DAG.

    Current dynamic fields are compared at every replay. The complete cold
    compiler also has to emit equal specs before two occurrences can share
    one invocation of their unchanged lowering factory.
    """
    if type(node) is not WorkloadNode:
        return None
    attributes = _node_physical_attributes(node)
    contract = (node.kind.value, node.pool_name, node.dtype, node.variant, shapes, attrs,
        tuple(node.shape.items()), tuple(attributes.items()))
    if not _ep_native_value(contract):
        return None
    return contract


def _native_plan_geometry_matches(frame, guards):
    for target, shapes, attrs, expected in guards:
        shape, attributes = frame.fields(target)
        current = (tuple((key, shape[key]) for key in shapes),
            tuple((key, attributes[key]) for key in attrs))
        if not _ep_native_value(current) or current != expected:
            return False
    return True


def _glm_decode_plan_layout(owner, plan):
    representatives = {node.node_id: node.node_id for node in owner.graph.nodes}
    for repeat in plan.block_repeats:
        for representative, occurrences in repeat.trace_node_ids:
            for name in occurrences:
                representatives[name] = representative
    calls = {}
    for call in plan.calls:
        if len(call.source_node_ids) != 1 or call.memory_source is not None:
            return None
        name, = call.source_node_ids
        if name not in calls:
            calls[name] = []
        calls[name].append(call)
    bases = {node.node_id: _logical_node_contract(node, ((), ()))
        for node in owner.graph.nodes if node.kind is WorkloadKind.FUSED_MOE}
    return representatives, calls, bases


def _glm_decode_bind_plan(binding, owner, graph, deployment, hardware_by_id, memo, entry):
    if not binding.dynamic_nodes:
        # Initial snapshots carry the original mutable provenance records;
        # they are cold-compiled and validated in full on every explicit use.
        # A later branch recompile must never price an older snapshot.
        return None
    old_deployment, old_hardware, source, template, layout = entry
    if (old_deployment is not deployment or len(old_hardware) != len(hardware_by_id)
            or any(name not in hardware_by_id or hardware_by_id[name] is not hardware for name, hardware in old_hardware)
            or not _dense_compiler_unchanged(source)):
        return None
    representatives, calls, bases = layout
    groups, physical, current_representatives, validated = {}, {}, {}, set()
    retained = []
    from src.serving.workload.traced_glm_mtp import _GLMMTP5SourceProgram
    for node in binding.dynamic_nodes:
        mtp_vector = (type(owner) is _GLMMTP5SourceProgram and node.kind is WorkloadKind.VECTOR
            and node.variant == 'vllm_mtp_eager'
            and node.attributes['source_component'] == 'vllm_0_30_symbolic_mtp_eager'
            and node.attributes['model_execution_version'] == 'vllm_0_30_0'
            and ('source_geometry_program' not in node.attributes
                or node.attributes['source_geometry_program'] == 'vllm_0_30_torch_2_13_cuda_source'))
        if not mtp_vector and node.kind not in (WorkloadKind.GATHER, WorkloadKind.EAGER_ATTN, WorkloadKind.FUSED_MOE,
                WorkloadKind.FLASH_ATTN, WorkloadKind.KV_UPDATE):
            return None
        hardware_id = deployment.worker_pools[node.pool_name].hardware_id
        hardware = require_hardware(hardware_id, hardware_by_id)
        if node.kind is WorkloadKind.FUSED_MOE:
            attrs = node.attributes
            # The issuer shares one immutable current routing packet across
            # equivalent EP1 ranks. Validate every distinct packet before
            # physical-class deduplication, including its invocation context.
            if (type(attrs["routing_provenance"]) is not MappingProxyType
                    or type(attrs["tokens_per_expert"]) is not tuple
                    or type(attrs["local_tokens_per_expert"]) is not tuple):
                return None
            context = (id(attrs["routing_provenance"]), id(attrs["tokens_per_expert"]),
                node.shape["tokens"], node.shape["experts"], node.shape["topk"], attrs["moe_invocation_id"])
            if context not in validated:
                _validate_routing_provenance(node)
                validated.add(context)
                retained.append((attrs["routing_provenance"], attrs["tokens_per_expert"]))
            key = (bases[node.node_id], id(attrs["tokens_per_expert"]), id(attrs["local_tokens_per_expert"]))
            if key not in physical:
                # This explicit source dispatcher preserves expert order in
                # its specs. Share exact ordered contracts, not merely an
                # expert-axis permutation class used by other MoE recipes.
                physical[key] = _logical_node_contract(node)
            contract = physical[key]
        else:
            contract = _logical_node_contract(node)
        representative = representatives[node.node_id]
        if representative in current_representatives and current_representatives[representative] != contract:
            # An old repeated block has split into distinct physical classes.
            return None
        current_representatives[representative] = contract
        if contract not in groups:
            groups[contract] = (node, [])
        groups[contract][1].append(representative)
    # This EP1/PP1 source's three dynamic operator domains have node-local
    # lowering. Reuse the actual dispatcher, never a second pricing formula.
    refreshed = _ordinary_lower_workload(WorkloadGraph(tuple(node for node, _ in groups.values()), ()),
        deployment, hardware_by_id, memo)
    specs = {}
    for call in refreshed.calls:
        name, = call.source_node_ids
        if name not in specs:
            specs[name] = []
        specs[name].append(call)
    updates, call_updates = {}, {}
    for node, targets in groups.values():
        current = specs[node.node_id]
        for target in set(targets):
            if target not in calls or len(calls[target]) != len(current):
                return None
            for call, refreshed_call in zip(calls[target], current, strict=True):
                spec = refreshed_call.kernel_spec
                if (spec.kind is not call.kernel_spec.kind or spec.variant != call.kernel_spec.variant
                        or spec.hardware_id != call.kernel_spec.hardware_id
                        or spec.transfer is not None or call.kernel_spec.transfer is not None):
                    return None
                updates[call.call_id] = spec
                if type(owner) is _GLMMTP5SourceProgram:
                    # Rejection/vector launches can depend on current values.
                    # The ordinary dispatcher above reselected them; retain
                    # the complete template's lanes, source IDs and DAG ports.
                    call_updates[call.call_id] = refreshed_call.source_launches
    if type(owner) is _GLMMTP5SourceProgram:
        return template.bind_kernel_specs(updates,source_launches=call_updates)
    return template.bind_kernel_specs(updates)


def _source_vector_runtime_graph(graph, inputs, deployment, hardware_by_id):
    """Bind explicit runtime branch facts before native vector selection.

    The existing source-launch input is keyed by actual call ID. A single
    normalization call has the same ID as its source node. Its typed layout,
    shape, version and payload remain owned by that node; only the explicit
    runtime flag is added. No simulator environment or default is consulted.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("source launch inputs require a nonempty mapping of actual call IDs")
    nodes = {node.node_id: node for node in graph.nodes}
    updates = {}
    for call_id, binding in inputs.items():
        if call_id not in nodes:
            continue  # Compound/unknown IDs are checked by the final binder.
        node = nodes[call_id]
        pool = deployment.worker_pools[node.pool_name]
        if (node.kind in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD) and node.dtype == "fp16"
                and HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id)):
            if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
                raise ValueError("observed FP16 GEMM binding requires complete shape and attributes")
            shape, attrs = binding["shape"], binding["attributes"]
            if (not isinstance(shape, Mapping) or not isinstance(attrs, Mapping)
                    or set(attrs) != {"model_execution_version", "source_gemm_runtime"}):
                raise ValueError("observed FP16 GEMM requires an explicit source runtime profile")
            required = {"launch_policy", "has_bias", "weight_dtype", "weight_group_size",
                "a_layout", "b_layout", "c_layout", "trans_a", "trans_b", "epilogue",
                "symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_parameter_inputs",
                "source_gemm_input_shape", "source_gemm_input_strides_bytes", "source_gemm_input_offset_bytes",
                "source_gemm_weight_shape", "source_gemm_weight_strides_bytes", "source_gemm_weight_offset_bytes"}
            parallel = pool.parallel_plan.config
            if (graph.source_model_execution_version != "vllm_0_30_0"
                    or attrs["model_execution_version"] != graph.source_model_execution_version
                    or attrs["source_gemm_runtime"] != "vllm_0_30_torch_2_13_cu130_cublas13_1_1_fp16_unquantized"
                    or dict(shape) != dict(node.shape) or not required.issubset(node.attributes)
                    or deployment.mode != "colocated" or len(deployment.worker_pools) != 1
                    or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                    or not parallel.is_rank_local or parallel.enable_expert_parallel
                    or len(pool.ranks) != 1 or node.rank != pool.ranks[0]
                    or node.attributes["launch_policy"] != "injected_launch"
                    or node.attributes["has_bias"] is not False
                    or node.attributes["weight_dtype"] != "fp16"
                    or type(node.attributes["weight_group_size"]) is not int or node.attributes["weight_group_size"] != 0
                    or "weight_quantization" in node.attributes
                    or "source_component" in node.attributes
                    or ("source_batch_invariant" in node.attributes and node.attributes["source_batch_invariant"] is not False)
                    or ("beta" in node.attributes and (type(node.attributes["beta"]) is bool or node.attributes["beta"] != 0))
                    or ("alpha" in node.attributes and (type(node.attributes["alpha"]) is bool or node.attributes["alpha"] != 1))
                    or any(node.attributes[name] != "row_major" for name in ("a_layout", "b_layout", "c_layout"))
                    or node.attributes["trans_a"] is not False or node.attributes["trans_b"] is not True
                    or node.attributes["epilogue"] != "none" or node.shape["b"] != 1
                    or _graph_token_domain(node) is not None
                    or ("model_execution_version" in node.attributes
                        and node.attributes["model_execution_version"] != graph.source_model_execution_version)
                    or ("source_gemm_runtime" in node.attributes
                        and node.attributes["source_gemm_runtime"] != attrs["source_gemm_runtime"])):
                raise ValueError("observed FP16 GEMM runtime/layout disagrees with the actual eager source caller")
            ports, outputs, parameters = (node.attributes[name] for name in (
                "symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_parameter_inputs"))
            m, n, k = (node.shape[name] for name in ("m", "n", "k"))
            if (len(ports) != 2 or len(outputs) != 1 or tuple(parameters) != (ports[1],)
                    or ports[0] == ports[1] or outputs[0] in ports
                    or tuple(node.attributes["source_gemm_input_shape"]) != (m, k)
                    or tuple(node.attributes["source_gemm_weight_shape"]) != (n, k)
                    or tuple(node.attributes["source_gemm_input_strides_bytes"]) != (2 * k, 2)
                    or tuple(node.attributes["source_gemm_weight_strides_bytes"]) != (2 * k, 2)
                    or any(type(node.attributes[name]) is not int or node.attributes[name] < 0
                        or node.attributes[name] % 16 for name in (
                            "source_gemm_input_offset_bytes", "source_gemm_weight_offset_bytes"))):
                raise ValueError("observed FP16 GEMM requires its real contiguous input and parameter view bindings")
            require_hardware(pool.hardware_id, hardware_by_id)
            updates[call_id] = node.with_attributes({**node.attributes,
                "source_gemm_runtime": attrs["source_gemm_runtime"],
                "model_execution_version": attrs["model_execution_version"]})
            checked = _CURRENT_GEMM_SOURCE_BINDINGS.get()
            if checked is not None:
                checked.add(id(updates[call_id]))
            continue
        if (node.kind is not WorkloadKind.NORM or node.dtype not in ("bf16", "fp16")
                or node.variant not in ("rmsnorm", "add_rmsnorm")
                or graph.source_model_execution_version != "vllm_0_30_0"
                or not HWSpec.source_contract_supported("h200_partial_source_programs", pool.hardware_id)):
            continue
        if not isinstance(binding, Mapping) or set(binding) != {"shape", "attributes"}:
            raise ValueError("source norm binding requires complete shape and attributes")
        shape, attrs = binding["shape"], binding["attributes"]
        required = {"model_execution_version", "source_input_layout", "source_batch_invariant"}
        if (not isinstance(shape, Mapping) or not isinstance(attrs, Mapping)
                or not required.issubset(attrs)):
            raise ValueError("source norm binding requires explicit runtime, layout and batch-invariant facts")
        if (not {"model_execution_version", "source_input_layout"}.issubset(node.attributes)
                or dict(shape) != dict(node.shape) or graph.source_model_execution_version != "vllm_0_30_0"
                or attrs["model_execution_version"] != graph.source_model_execution_version
                or node.attributes["model_execution_version"] != graph.source_model_execution_version
                or attrs["source_input_layout"] != node.attributes["source_input_layout"]
                or type(attrs["source_batch_invariant"]) is not bool
                or ("source_batch_invariant" in node.attributes
                    and node.attributes["source_batch_invariant"] != attrs["source_batch_invariant"])):
            raise ValueError("source norm runtime facts disagree with the actual source node")
        require_hardware(pool.hardware_id, hardware_by_id)
        updates[call_id] = node.with_attributes({**node.attributes,
            "source_batch_invariant": attrs["source_batch_invariant"]})
    if not updates:
        return graph
    return graph.with_nodes(updates[node.node_id] if node.node_id in updates else node for node in graph.nodes)


def _lower_workload_with_current_source_inputs(graph, deployment, hardware_by_id, memo=None, *, source_projection_scope=None,
        compact_source_layers=False, source_launch_inputs=None):
    if source_launch_inputs is not None:
        if compact_source_layers:
            raise ValueError("explicit native source inputs require an uncompressed logical call plan")
        nodes = {node.node_id: node for node in graph.nodes}
        norm_inputs_only = (type(source_launch_inputs) is dict and bool(source_launch_inputs)
            and graph.source_model_execution_version == "vllm_0_30_0"
            and all(name in nodes and nodes[name].kind is WorkloadKind.NORM
                and nodes[name].variant in ("rmsnorm", "add_rmsnorm")
                and nodes[name].dtype in ("bf16", "fp16")
                and HWSpec.source_contract_supported("h200_partial_source_programs",
                    deployment.worker_pools[nodes[name].pool_name].hardware_id)
                and type(binding) is dict and set(binding) == {"shape", "attributes"}
                and type(binding["shape"]) is dict and type(binding["attributes"]) is dict
                and set(binding["attributes"]) == {
                    "model_execution_version", "source_input_layout", "source_batch_invariant"}
                for name, binding in source_launch_inputs.items()))
        graph = _source_vector_runtime_graph(graph, source_launch_inputs, deployment, hardware_by_id)
        plan = _ordinary_lower_workload(graph, deployment, hardware_by_id, memo,
            source_projection_scope=source_projection_scope, compact_source_layers=False)
        if norm_inputs_only:
            # The current shape/layout/version/branch was checked above by
            # _source_vector_runtime_graph for every logical Norm node. The
            # ordinary compiler already selected its exact KernelSpec and
            # native_main recipe (including on memo hits). Re-running the
            # selector and replacing every call adds no distinct fact.
            return plan
        return _bind_plan_source_launch_inputs(plan, source_launch_inputs, hardware_by_id, graph)
    from src.serving.workload.traced_models import _GLMDecodeSourceBinding
    binding = graph.__dict__["_glm_decode_binding"] if "_glm_decode_binding" in graph.__dict__ else None
    glm_owner = binding.owner(graph) if type(binding) is _GLMDecodeSourceBinding else None
    # A PD decode forward is still one complete pool-local source program.
    # Its transfer is compiled separately, so retain every layer/rank/call
    # and bind only the issuer's current dynamic operator contracts. Unlike
    # the colocated path, this does not enable layer folding for PD.
    glm_eligible = (glm_owner is not None and isinstance(memo, KernelLoweringMemo)
        and source_projection_scope is None
        and glm_owner.pool.name in deployment.worker_pools
        and deployment.worker_pools[glm_owner.pool.name] is glm_owner.pool
        and ((compact_source_layers and deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1)
            or (not compact_source_layers and deployment.mode == "pd_disagg"
                and worker_role_phase(glm_owner.pool.role) == "decode"
                and all(node.pool_name == glm_owner.pool.name
                    and node.kind is not WorkloadKind.PD_KV_TRANSFER for node in graph.nodes))))
    if glm_eligible and glm_owner in memo._glm_decode_plans:
        bound = _glm_decode_bind_plan(binding, glm_owner, graph, deployment, hardware_by_id, memo,
            memo._glm_decode_plans[glm_owner])
        if bound is not None:
            return bound
    from src.serving.workload.traced_forward import _NativeSourceFrame
    frame = graph.__dict__["_native_source_frame"] if "_native_source_frame" in graph.__dict__ else None
    owner = frame.owner() if type(frame) is _NativeSourceFrame else None
    eligible = (owner is not None and memo is not None and compact_source_layers
        and source_projection_scope is None and deployment.mode == "colocated"
        and len(deployment.worker_pools) == 1)
    if eligible and owner in _COMPILED_NATIVE_PLANS:
        old_memo, old_hardware, old_dep, old_recipe, layout, template, dense, bound_plans = _COMPILED_NATIVE_PLANS[owner]
        from src.serving.workload.traced_forward import _native_recipe_matches
        if (old_memo is memo and old_dep is deployment and old_hardware is hardware_by_id
                and _native_recipe_matches(owner,old_recipe)
                and _native_plan_geometry_matches(frame, layout[1])):
            current_groups = []
            for name, rows, call_ids in layout[0]:
                signature = None
                for target, shapes, attrs in rows:
                    shape, attributes = frame.fields(target)
                    current = (tuple((key,shape[key]) for key in shapes),tuple((key,attributes[key]) for key in attrs))
                    if dense is not None and not _ep_native_value(current): break
                    if signature is None: signature = current
                    elif current != signature:
                        break
                else:
                    current_groups.append((name,call_ids))
                    continue
                break
            else:
                ep_layout = (_EP_SOURCE_VALIDATION_LAYOUTS[owner]
                    if owner in _EP_SOURCE_VALIDATION_LAYOUTS else None)
                ep_valid = (_reuse_ep_native_frame_recipe(graph,deployment,owner,ep_layout,source_layout=frame.layout)
                    if ep_layout is not None else None)
                binding_valid = ep_valid is True
                if dense is not None:
                    behavior, dep, compiler = dense
                    binding_valid = (not frame.routing_ids and _ep_verifier_unchanged(behavior)
                        and _ep_deployment_projection(deployment) == dep
                        and frame.workload_contract[1] is frame.template.edges
                        and frame.workload_contract[2] is frame.template.block_repeats
                        and _dense_compiler_unchanged(compiler))
                if binding_valid:
                    prepared_groups = []
                    canonical_axes = {}
                    for name, call_groups in current_groups:
                        node = frame.node(name)
                        pool = deployment.worker_pools[node.pool_name]
                        hardware_id, hardware = pool.hardware_id, require_hardware(pool.hardware_id, hardware_by_id)
                        relabel = (_fused_moe_routing_relabelling(node, hardware_id, hardware, canonical_axes=canonical_axes)
                            if node.kind is WorkloadKind.FUSED_MOE else None)
                        contract = _logical_node_contract(node, relabel)
                        prepared_groups.append((node, hardware_id, hardware, relabel, contract, call_groups))
                    # Reuse only after this step's complete EP source proof.
                    # Use the ordinary compiler's complete physical contracts,
                    # including its existing expert-axis canonicalization.
                    # Source IDs and routing certificates stay on this graph.
                    bound_key = None
                    if (ep_valid is True and type(memo) is KernelLoweringMemo
                            and "lower" not in memo.__dict__ and type(hardware_by_id) is dict
                            and all(type(name) is str and type(hardware) is HWSpec
                                for name, hardware in hardware_by_id.items())):
                        bound_key = tuple((contract, id(hardware))
                            for _node, _hardware_id, hardware, _relabel, contract, _groups in prepared_groups)
                        if bound_key in bound_plans:
                            retained_hardware, result = bound_plans[bound_key]
                            if any(name not in hardware_by_id or hardware_by_id[name] is not hardware
                                    for name, hardware in retained_hardware):
                                raise RuntimeError("native bound-plan hardware identity collision")
                            bound_plans.move_to_end(bound_key)
                            return result
                    updates = {}
                    for node, hardware_id, hardware, relabel, contract, call_groups in prepared_groups:
                        def build_specs():
                            _require_source_lowering_contract(node)
                            if node.kind is WorkloadKind.FLASH_ATTN:
                                return _source_flash_attn_specs(node, hardware_id, hardware)
                            if node.kind is WorkloadKind.FUSED_MOE:
                                return _explicit_source_expert_specs(node, hardware_id, hardware, relabel)
                            if node.kind is WorkloadKind.VECTOR:
                                return _source_flash_attn_specs(node, hardware_id, hardware)
                            if node.kind is WorkloadKind.MEMCPY:
                                return (_transfer_spec(node, hardware_id),)
                            raise AssertionError("uncompiled native source binding kind")
                        specs = memo.lower(node,contract,hardware,build_specs)
                        if any(len(specs) != len(call_ids) for call_ids in call_groups): break
                        for call_ids in call_groups:
                            for call_id,spec in zip(call_ids,specs,strict=True): updates[call_id] = spec
                    else:
                        # A physical branch change selects the complete ordinary
                        # compiler below. Never swallow invalid input exceptions.
                        call_map = {call.call_id:call for call in template.calls}
                        same_structure = all(call_map[call_id].kernel_spec.kind is spec.kind
                            and call_map[call_id].kernel_spec.hardware_id == spec.hardware_id
                            and call_map[call_id].resource == ("compute" if spec.transfer is None
                                else spec.transfer.resource_id if spec.kind is KernelKind.COMMUNICATION else spec.transfer.link)
                            for call_id,spec in updates.items())
                        if same_structure:
                            result = template.bind_kernel_specs(updates)
                            if bound_key is not None:
                                bound_plans[bound_key] = tuple(hardware_by_id.items()), result
                                bound_plans.move_to_end(bound_key)
                                while len(bound_plans) > memo._max_entries:
                                    bound_plans.popitem(last=False)
                            return result
    result = _ordinary_lower_workload(graph,deployment,hardware_by_id,memo,
        source_projection_scope=source_projection_scope,compact_source_layers=compact_source_layers)
    if glm_eligible:
        layout = _glm_decode_plan_layout(glm_owner, result)
        if layout is not None:
            memo._glm_decode_plans[glm_owner] = (deployment, tuple(hardware_by_id.items()), _dense_compiler_source(), result, layout)
    if eligible:
        layout = _native_plan_binding_layout(frame,result)
        if layout is not None:
            dense = None
            if (not frame.routing_ids and all(not pool.parallel_plan.config.enable_expert_parallel
                    for pool in deployment.worker_pools.values())
                    and all(type(node) is WorkloadNode and _ep_native_value(tuple(node.shape.items()))
                        and _ep_native_value(tuple(node.attributes.items())) for node in frame.template.nodes)):
                behavior, dep = _ep_verifier_snapshot(), _ep_deployment_projection(deployment)
                if behavior is not None and dep is not None:
                    dense = (behavior, dep, _dense_compiler_source())
            if not layout[1] or dense is not None:
                _COMPILED_NATIVE_PLANS[owner] = (memo,hardware_by_id,deployment,frame.recipe_signature,layout,result,dense,OrderedDict())
    return result


def lower_workload(graph, deployment, hardware_by_id, memo=None, *, source_projection_scope=None,
        compact_source_layers=False, source_launch_inputs=None, source_memory_context=None):
    # Validation belongs to this public invocation. Never trust a marker from
    # an earlier graph/call, including nested calls and lowering-memo hits.
    from .memory_bindings import TraceMemoryContext
    if source_memory_context is not None and type(source_memory_context) is not TraceMemoryContext:
        raise TypeError("source memory input requires the existing current TraceMemoryContext")
    token = _CURRENT_GEMM_SOURCE_BINDINGS.set(set())
    dspark_token = _CURRENT_DSPARK_SOURCE_GRAPH.set((graph, {}, source_memory_context))
    try:
        return _lower_workload_with_current_source_inputs(graph, deployment, hardware_by_id, memo,
            source_projection_scope=source_projection_scope, compact_source_layers=compact_source_layers,
            source_launch_inputs=source_launch_inputs)
    finally:
        _CURRENT_DSPARK_SOURCE_GRAPH.reset(dspark_token)
        _CURRENT_GEMM_SOURCE_BINDINGS.reset(token)


def lower_vllm030_kda_chunk_source_calls(context, hardware_by_id, *, source_attributes_by_parent):
    """Bind explicit prepared bodies to their actual source ports and ready edges.

    Returns existing KernelCalls, retaining external predecessors for the
    caller's complete plan. This neither substitutes the default aggregate
    chunk nor admits a native memory ABI; shared child Core/selector admission
    must already be present. Compiler configurations are mandatory caller facts.
    """
    from .memory_bindings import TraceMemoryContext
    if type(context) is not TraceMemoryContext or not isinstance(source_attributes_by_parent, Mapping):
        raise TypeError("KDA source lowering requires actual memory receipts and explicit parent compiler facts")
    ports = {
        "q_l2norm": (("q",), ("q_normalized",)),
        "k_l2norm": (("k",), ("k_normalized",)),
        "gate_cumsum_beta": (("raw_gate", "raw_beta", "query_start", "chunk_indices", "A_log", "dt_bias"), ("cumulative_gate_log2", "sigmoid_beta")),
        "zero_Akk": ((), ("Akk",)),
        "intra_sub_chunk": (("q_normalized", "k_normalized", "gate", "beta", "query_start", "chunk_indices"), ("Akkd", "Aqk")),
        "inter_solve": (("q_normalized", "k_normalized", "gate", "beta", "Akkd", "query_start", "chunk_indices"), ("Akk", "Aqk")),
        "recompute_w_u": (("k_normalized", "v", "beta", "Akk", "gate", "query_start", "chunk_indices"), ("w", "u", "kg")),
        "state": (("kg", "w", "u", "gate", "initial_state", "query_start", "chunk_offsets"), ("chunk_h", "v_new", "final_state")),
        "output": (("q_normalized", "v_new", "gate", "chunk_h", "Aqk", "query_start", "chunk_indices"), ("output_original_v",))}
    parent_fields = ("stage", "state_indices", "state_pool_slots", "query_start_loc", "has_initial_state", "gate_lower_bound")
    caller_fields = ("model_execution_version", "source_norm_backend", "source_metadata_state",
        "source_use_qk_l2norm_in_kernel", "source_output_final_state", "source_out",
        "source_chunk_indices", "source_chunk_offsets")
    compiler_fields = {"source_compiler", "source_norm_backend", "source_metadata_state", "tensor_iterator_source",
        "metadata_dtype", "chunk_offsets_dtype", "source_use_gather", "source_selected_configs"}
    required_source = set(parent_fields) | set(caller_fields) | {"source_parent_name", "source_parent_variant",
        "source_stage_role", "memory_input_roles", "memory_output_roles", "source_query_start_view_id"}
    groups = {}
    for source in context.nodes:
        attrs = source.attributes
        if "source_component" not in attrs or attrs["source_component"] != "symbolic_vllm030_kda_chunk_stage":
            continue
        if (required_source - set(attrs) or type(attrs["source_parent_name"]) is not str
                or not attrs["source_parent_name"] or type(attrs["source_stage_role"]) is not str
                or attrs["source_stage_role"] not in ports):
            raise ValueError("KDA source lowering requires complete source stage and parent identities")
        key = (source.pool_name, source.rank, attrs["source_parent_name"])
        if key not in groups:
            groups[key] = []
        groups[key].append(source)
    if not groups or set(groups) != set(source_attributes_by_parent):
        raise ValueError("KDA source lowering requires exact coverage of every actual chunk body")
    calls = []
    for key, sources in groups.items():
        first = sources[0]
        if tuple(source.attributes["source_stage_role"] for source in sources) != tuple(ports):
            raise ValueError("KDA source lowering requires all nine ordered submissions")
        facts = source_attributes_by_parent[key]
        if not isinstance(facts, Mapping) or set(facts) != compiler_fields:
            raise ValueError("KDA source lowering requires complete explicit compiler facts without defaults")
        parent = {name: first.attributes[name] for name in parent_fields}
        caller = {name: first.attributes[name] for name in caller_fields}
        if any(caller[name] != facts[name] for name in ("source_norm_backend", "source_metadata_state")):
            raise ValueError("KDA compiler facts disagree with the actual prepared source caller")
        views = _validate_kda_chunk_source_ports(sources, ports)
        source_hardware = tuple(item for item in hardware_by_id.values()
            if HWSpec.source_contract_supported("h200_partial_source_programs", item.hardware_id))
        if len(source_hardware) != 1:
            raise ValueError("KDA source calls require one declared source hardware profile")
        hardware, = source_hardware
        from src.kernel.launch_config.select_launch_config import source_launch_function
        select_kda_chunk_stage_launch = source_launch_function(hardware.hardware_id, "select_kda_chunk_stage_launch")
        _, rows, heads, dim = views["q"].shape
        shape = dict(rows=rows, heads=heads, head_dim=dim, element_count=rows*heads*dim)
        for index, source in enumerate(sources):
            attrs = source.attributes
            role = attrs["source_stage_role"]
            if (source.kind is not WorkloadKind.VECTOR or source.variant != "kda_chunk_" + role
                    or any(attrs[name] != value for name, value in (*parent.items(), *caller.items()))
                    or attrs["source_parent_variant"] != first.attributes["source_parent_variant"]
                    or attrs["source_query_start_view_id"] != views["query_start"].view_id
                    or index > 0 and sources[index-1].node_id not in source.predecessors):
                raise ValueError("KDA child changed its source operation, parent, metadata identity or ready order")
            spec_attrs = {**parent, **caller, **facts, "source_component": "vllm_0_30_kimi_k3_triton_chunk",
                "source_stage": role, "source_input_layout": "contiguous_qkv_gate_initial", "source_initial_state_dtype": "fp32"}
            hardware = require_hardware(hardware.hardware_id, hardware_by_id)
            variant = attrs["source_parent_variant"]
            launch = select_kda_chunk_stage_launch(shape, "bf16", variant, spec_attrs, hardware)
            spec = KernelSpec(KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16", variant=variant,
                attributes=spec_attrs, launch=launch, execution_support="unsupported", unsupported_reason="typed_service_unavailable")
            calls.append(context.bind_call(KernelCall(source.node_id, spec, source.pool_name, source.rank, "compute",
                (source.node_id,), source.predecessors)))
    return tuple(calls)



def _validate_kda_chunk_source_ports(sources, ports):
    """Exact mixed-precision roles/aliases of the existing nine-stage emitter."""
    aliases = {"cumulative_gate_log2": "gate", "sigmoid_beta": "beta", "output_original_v": "v"}
    views, writers, external = {}, {}, {}
    for source in sources:
        reads, writes = ports[source.attributes["source_stage_role"]]
        if (source.attributes["memory_input_roles"] != reads or source.attributes["memory_output_roles"] != writes
                or len(source.inputs) != len(reads) or len(source.outputs) != len(writes)):
            raise ValueError("KDA source ports must preserve every declared input/output role")
        for name, view in zip((*reads, *writes), (*source.inputs, *source.outputs), strict=True):
            role = aliases[name] if name in aliases else name
            if role in views and views[role] != view:
                raise ValueError("KDA source stage changed a shared intermediate or original-V alias")
            views[role] = view
        for name, producer in zip(reads, source.input_producers, strict=True):
            role = aliases[name] if name in aliases else name
            if producer is not None and producer not in source.predecessors:
                raise ValueError("KDA source input lost its actual external or intermediate ready edge")
            if role in writers:
                if producer != writers[role] or producer not in source.predecessors:
                    raise ValueError("KDA source intermediate lost its actual producer or ready edge")
            elif role in external and external[role] != producer:
                raise ValueError("KDA source input changed its external producer readiness")
            else:
                external[role] = producer
        for name in writes:
            writers[aliases[name] if name in aliases else name] = source.node_id
    if len(views["q"].shape) != 4 or views["q"].shape[0] != 1:
        raise ValueError("KDA source requires its actual packed batch-one Q geometry")
    dense = views["q"].shape
    _, rows, heads, dim = dense
    starts = sources[0].attributes["query_start_loc"]
    sequences = len(starts)-1
    chunks = sum((b-a+63)//64 for a, b in zip(starts, starts[1:]))
    layouts = {name: (dense, "bf16") for name in ("q", "k", "v", "raw_gate", "q_normalized", "k_normalized", "w", "u", "kg", "v_new")}
    layouts.update(raw_beta=((1, rows, heads), "bf16"), gate=(dense, "fp32"), beta=((1, rows, heads), "fp32"),
        initial_state=((sequences, heads, dim, dim), "fp32"), final_state=((sequences, heads, dim, dim), "fp32"),
        chunk_h=((1, chunks, heads, dim, dim), "bf16"), Aqk=((1, rows, heads, 64), "bf16"),
        Akk=((1, rows, heads, 64), "bf16"), Akkd=((1, rows, heads, 16), "fp32"),
        query_start=((sequences+1,), "int32"), chunk_indices=((chunks, 2), "int32"),
        chunk_offsets=((sequences+1,), "int64"), A_log=((heads,), "fp32"), dt_bias=((heads*dim,), "fp32"))
    for name, (shape, dtype) in layouts.items():
        view = views[name]
        if view.shape != shape or view.dtype != dtype:
            raise ValueError("KDA source role changed its actual shape or mixed-precision dtype: " + name)
        if name != "raw_beta":
            stride = view.element_bytes
            for size, actual in zip(reversed(shape), reversed(view.strides_bytes), strict=True):
                if size > 1 and actual != stride:
                    raise ValueError("KDA source requires contiguous prepared role: " + name)
                stride *= size
    if any(views["v"].allocation_id == view.allocation_id for name, view in views.items() if name != "v"):
        raise ValueError("KDA original V output must retain its independent source backing")
    if any(views["final_state"].allocation_id == view.allocation_id for name, view in views.items() if name != "final_state"):
        raise ValueError("KDA final state must retain its fresh separate FP32 target")
    return views
