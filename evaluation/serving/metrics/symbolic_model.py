"""Evaluate metadata-only model graphs through the production Roofline route."""

import ast
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from time import perf_counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor

from src.kernel.hardware_spec import HWSpec
from src.kernel.predictors.sparse_attn_indexer import dsa_indexer_logical_work
from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kernel_lowering.kernel_predict import KernelPredictionMemo
from src.serving.kernel_lowering.kernel_plan import hardware_for_call
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kv_cache.activation_memory import peak_live_activation_bytes
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.workload.model_config import load_model_config
from src.serving.workload import traced_models
from src.serving.workload.rope_dispatch import vllm_rope_program
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

from .qwen3_source import qwen3_source_reference, compare_qwen3_source_graph
from .v4_ffn_source import compare_v4_ffn_work
from .v4_source import compare_v4_source_dependencies
from .v4_fp8_source import compare_v4_fp8_dataflow, compare_v4_quantizer_work
from .v4_prepare_source import compare_v4_prepare_work
from .v4_activation_source import compare_v4_ffn_vector_work
from .v4_output_source import compare_v4_output_work
from .v4_indexer_source import compare_v4_indexer_work
from .v4_attention_source import compare_v4_attention_math
from .v4_mhc_source import compare_v4_mhc_work
from .v4_norm_source import compare_v4_qkv_norm_work, compare_v4_final_norm_work
from .v4_target_source import compare_v4_target_work, compare_v4_argmax_source
from .v4_routing_source import compare_v4_routing_objects
from .v4_routing_math_source import compare_v4_routing_math
from .v4_state_source import logical_state_reference, compare_v4_logical_state, compare_v4_reductions
from .v4_coverage import v4_source_scope_inventory
from .v4_storage_source import wrapper_storage_reference, compare_v4_wrapper_storage
from .v4_terminal_storage import compare_v4_terminal_storage
from .v4_decode_storage import compare_v4_decode_index_storage
from .v4_expert_storage import compare_v4_expert_storage
from .v4_indexer_storage import compare_v4_decode_selector_storage
from .v4_sparse_storage import compare_v4_sparse_prefill_storage, compare_v4_sparse_decode_views
from .v4_source import (v4_model_source_reference, observe_v4_source_calls,
    compare_v4_model_source_calls, compare_v4_input_projection_work, compare_v4_compressor_dependencies,
    compare_v4_partial_state_work, compare_v4_finish_work)


class _ForwardSyntax(ast.NodeTransformer):
    """Ignore annotations, method name and the symbolic view API spelling."""
    def visit_FunctionDef(self, node):
        node.name, node.returns = "forward", None
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_Attribute(self, node):
        if node.attr == "view_as_shape":
            node.attr = "view"
        return self.generic_visit(node)


def _normalized_forward_body(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return ast.dump(_ForwardSyntax().visit(function), include_attributes=False)


def compare_source_forward_bodies(upstream_root: Path, *, model_type="qwen3"):
    """Check copied control/dataflow bodies against independent upstream files.

    This proves the three forward bodies only, not library/operator internals,
    construction, simplified PP1 model loop or provider/runtime equivalence.
    No Midlayer builder/lowering/predictor creates the upstream expectation.
    """
    own = Path(traced_models.__file__)
    rows = []
    if model_type == "qwen3":
        programs = (
            ("qwen2.py", "Qwen2MLP", "QwenMLP"),
            ("qwen3.py", "Qwen3Attention", "Qwen3Attention"),
            ("qwen3.py", "Qwen3DecoderLayer", "Qwen3DecoderLayer"))
    elif model_type == "llama":
        programs = (
            ("llama.py", "LlamaMLP", "LlamaMLP"),
            ("llama.py", "LlamaAttention", "LlamaAttention"),
            ("llama.py", "LlamaDecoderLayer", "Qwen3DecoderLayer"))
    elif model_type == "qwen3_moe":
        programs = (
            ("qwen3_moe.py", "Qwen3MoeAttention", "Qwen3Attention"),
            ("qwen3_moe.py", "Qwen3MoeDecoderLayer", "Qwen3DecoderLayer"))
    else:
        raise ValueError("forward-body comparison requires an adapted dense model family")
    for file_name, upstream_class, adapted_class in programs:
        source = upstream_root / "vllm/model_executor/models" / file_name
        rows.append({"source": str(source), "source_symbol": upstream_class + ".forward",
            "adapted_symbol": adapted_class + ".__call__",
            "matched": _normalized_forward_body(source, upstream_class, "forward")
                == _normalized_forward_body(own, adapted_class, "__call__")})
    return {"scope": f"{len(rows)} forward bodies after annotation/method/view-spelling normalization",
        "checks": rows, "failed": sum(not row["matched"] for row in rows)}


def compare_v4_ffn_forward_bodies(upstream_root: Path):
    """Independent syntax check of two retained V4 FFN wrapper bodies only.

    Runner specialization and operator/provider internals are outside this
    denominator. The source expectation never uses the workload builder.
    """
    source = upstream_root / "vllm/models/deepseek_v4/nvidia/model.py"
    own = Path(traced_models.__file__)
    rows = []
    for cls, original, adapted in (("DeepseekV4MLP", "forward", "__call__"),
                                  ("DeepseekV4MoE", "_forward_fused_moe", "_forward_fused_moe")):
        rows.append({"source": str(source), "source_symbol": cls + "." + original,
            "adapted_symbol": cls + "." + adapted,
            "matched": _normalized_forward_body(source, cls, original) == _normalized_forward_body(own, cls, adapted)})
    return {"scope": "two V4 FFN wrapper bodies only; not Runner/operator/native equivalence",
        "checks": rows, "failed": sum(not row["matched"] for row in rows)}


def compare_indexer_gather_workspace(upstream_root, graph, traces, *, max_model_len, head_dim, model_type):
    """Independent source allocation reference for existing scored prefill calls.

    Expected sizes come from the upstream helper AST, never from a Midlayer
    builder or allocation formula. This checks the two known caller ports,
    not the completeness of the indexer inventory or its external callees.
    """
    metadata_path = upstream_root / "vllm/v1/attention/backends/mla/indexer.py"
    operator_path = upstream_root / "vllm/model_executor/layers/sparse_attn_indexer.py"
    if model_type == "deepseek_v4":
        source_family, source_class, ratio = "deepseek_v4", "DeepseekV4Indexer", 4
        binding = "get_max_prefill_buffer_size(vllm_config) // self.compress_ratio"
        variants = ("dsv4_indexer_prefill",)
    elif model_type == "glm_moe_dsa":
        source_family, source_class, ratio = "deepseek_v32", "DeepseekV32Indexer", 1
        binding = "get_max_prefill_buffer_size(vllm_config)"
        variants = ("dsa_indexer_prefill", "dsa_indexer_mixed")
    else:
        raise ValueError("source gather comparison requires a declared V4 or GLM indexer")
    model_path = upstream_root / f"vllm/models/{source_family}/attention.py"
    metadata = ast.parse(metadata_path.read_text())
    operator = ast.parse(operator_path.read_text())
    model = ast.parse(model_path.read_text())
    size_helper = next(n for n in metadata.body if isinstance(n, ast.FunctionDef)
        and n.name == "get_max_prefill_buffer_size")
    gather_helper = next(n for n in operator.body if isinstance(n, ast.FunctionDef)
        and n.name == "_gather_workspace_shapes")
    cls = next(n for n in model.body if isinstance(n, ast.ClassDef) and n.name == source_class)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
        and any(ast.unparse(target) == "self.max_total_seq_len" for target in n.targets))
    if ast.unparse(assignment.value) != binding:
        raise ValueError("source indexer constructor workspace binding changed")
    size_assignment = next(n for n in size_helper.body if isinstance(n, ast.Assign))
    if (len(size_assignment.targets) != 1 or ast.unparse(size_assignment.targets[0]) != "max_model_len"
            or ast.unparse(size_assignment.value) != "vllm_config.model_config.max_model_len"):
        raise ValueError("source prefill capacity no longer binds runtime maximum length")
    capacity_expression = next(n.value for n in size_helper.body if isinstance(n, ast.Return))
    ports_expression = next(n.value for n in gather_helper.body if isinstance(n, ast.Return))

    class DtypeLiteral(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if ast.unparse(node) != "torch.uint8":
                raise ValueError("unsupported source gather dtype expression")
            return ast.copy_location(ast.Constant("uint8"), node)

    def metadata_value(expression, names):
        # Only tuple construction and scalar metadata arithmetic are allowed;
        # no source module import, Tensor creation, call or numerical forward.
        expression = DtypeLiteral().visit(expression)
        allowed = (ast.Tuple, ast.Load, ast.Name, ast.Constant, ast.BinOp, ast.Mult, ast.FloorDiv)
        if any(not isinstance(node, allowed) for node in ast.walk(expression)):
            raise ValueError("unsupported source workspace metadata expression")
        return eval(compile(ast.Expression(expression), "<source-workspace-metadata>", "eval"),
            {"__builtins__": {}}, names)

    # Only the declared CP1 FP8 constructor ports, not chunk accesses.
    capacity = metadata_value(capacity_expression, {"max_model_len": max_model_len}) // ratio
    expected = metadata_value(ports_expression,
        {"total_seq_lens": capacity, "head_dim": head_dim, "fp8_dtype": "fp8"})
    views = {view.view_id: view for trace in traces for view in trace.views}
    rows = []
    for node in graph.nodes:
        if node.variant not in variants:
            continue
        request = node.attributes["symbolic_prefill_gather_workspace"]
        actual = None if request is None else tuple((views[name].shape, views[name].dtype) for name in request["view_ids"])
        alias_matched = None
        if request is not None:
            values, scale_bytes = (views[name] for name in request["view_ids"])
            scale = views[request["scale_fp32_view_id"]]
            alias_matched = (values.allocation_id == scale_bytes.allocation_id == scale.allocation_id
                and scale.offset_bytes == scale_bytes.offset_bytes and scale.dtype == "fp32"
                and scale.shape == (capacity,) and scale.strides_bytes == (4,))
        rows.append({"call": node.node_id, "expected_ports": expected, "actual_ports": actual,
            "scale_alias_matched": alias_matched,
            "status": "not_checked" if request is None else "matched" if actual == expected and alias_matched else "mismatched"})
    return {"scope": "two caller gather ports of existing CP1 FP8 scored prefill calls only; not full indexer alignment",
        "source": [{"path": str(path), "symbol": symbol, "line": line} for path, symbol, line in (
            (metadata_path, "get_max_prefill_buffer_size", size_helper.lineno),
            (model_path, source_class + ".__init__.max_total_seq_len", assignment.lineno),
            (operator_path, "_gather_workspace_shapes", gather_helper.lineno))],
        "checks": rows, "total": len(rows),
        "matched": sum(row["status"] == "matched" for row in rows),
        "mismatched": sum(row["status"] == "mismatched" for row in rows),
        "not_checked": sum(row["status"] == "not_checked" for row in rows),
        "whole_model_alignment_accepted": False, "normal_capacity_accepted": False}


def _diagnose_symbolic_pricing(plan, hardware):
    """Report every refused call without turning it into a zero-cost result.

    Production prediction remains fail-fast. Only this evaluation diagnostic
    collects explicit NotImplemented failures, and its caller suppresses the
    aggregate time whenever any call is unpriced.
    """
    memo, predictions, refused = KernelPredictionMemo(), {}, []
    for call in plan.topological_order():
        try:
            prediction = memo.predict(call.kernel_spec, hardware_for_call(call, hardware), "roofline")
        except NotImplementedError as error:
            refused.append({"call_id": call.call_id, "kind": call.kernel_spec.kind.value,
                "variant": call.kernel_spec.variant, "reason": str(error)})
        else:
            predictions[call.call_id] = prediction
    return predictions, refused


def _summarize_dsa_indexer_payload(plan):
    """Aggregate production logical roles; no duplicated byte formula."""
    totals = {name: 0 for name in ("calls", "prefill_gather_calls", "prefill_gather_rows",
        "prefill_gather_read_write_bytes", "prefill_shortcut_rows", "prefill_nontrivial_rows",
        "prefill_selector_useful_score_values", "prefill_sentinel_padding_values")}
    for call in plan.calls:
        spec = call.kernel_spec
        if spec.variant not in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed"):
            continue
        work = dsa_indexer_logical_work(spec)
        totals["calls"] += 1
        if work["prefill_gather"] is not None:
            totals["prefill_gather_calls"] += 1
            totals["prefill_gather_rows"] += work["prefill_gather"]["total_copied_rows"]
            totals["prefill_gather_read_write_bytes"] += work["prefill_gather_materialization_bytes"]
            selection = work["prefill_selection"]
            totals["prefill_shortcut_rows"] += sum(selection["shortcut_rows_per_request"])
            totals["prefill_nontrivial_rows"] += selection["nontrivial_rows"]
            totals["prefill_selector_useful_score_values"] += selection["nontrivial_useful_score_values"]
        totals["prefill_sentinel_padding_values"] += sum(op["elements"] for stage in work["stages"]
            for op in stage["operands"] if op["role"] == "prefill_sentinel_padding")
    return {**totals, "actual_provider_bound": False, "actual_chunk_partition_bound": False,
        "complete_indexer_source_accepted": False}


def _comparison_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _comparison_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _comparison_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_comparison_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported comparison metadata: {type(value).__name__}")


def compare_workload_paths(legacy, symbolic, deployment, hardware):
    """Exact metadata comparison; graph-local IDs alone are normalized.

    Both graphs are supplied by the caller; this utility never executes the
    deleted legacy builder. Saved migration comparisons remain historical
    evidence and are not rebound to the current forward.
    Kind/rank occurrence pairs are only alignment candidates. Shape, dtype,
    variants, attributes, complete lowered programs, prices and dependencies
    are still compared; pairing itself never proves equivalence.
    """
    observed = []
    for graph in (legacy, symbolic):
        if graph.block_repeats:
            raise NotImplementedError("path comparison requires materialized logical blocks")
        plan = lower_workload(graph, deployment, hardware)
        predictions, refused = _diagnose_symbolic_pricing(plan, hardware)
        occurrences, nodes, ids = Counter(), {}, {}
        for node in graph.nodes:
            ranks = deployment.worker_pools[node.pool_name].ranks
            rank = ranks[0] if node.rank is None and len(ranks) == 1 else node.rank
            family = (node.pool_name, rank, node.kind.value)
            occurrence = occurrences[family]
            occurrences[family] += 1
            key = json.dumps((*family, occurrence), separators=(",", ":"))
            ids[node.node_id] = key
            attrs = {key: value for key, value in node.attributes.items()
                if key not in ("symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls")}
            nodes[key] = {"node_id": node.node_id, "kind": node.kind.value,
                "shape": dict(node.shape), "dtype": node.dtype, "variant": node.variant,
                "attributes": _comparison_value(attrs), "calls": [], "serial_roofline_seconds": 0.0}
        for call in plan.calls:
            for source in call.source_node_ids:
                row = nodes[ids[source]]
                row["calls"].append(_comparison_value(call.kernel_spec))
                if call.call_id in predictions:
                    row["serial_roofline_seconds"] += predictions[call.call_id].t_total_s
        edges = Counter((ids[edge.source], ids[edge.destination], edge.kind) for edge in graph.edges)
        observed.append({"nodes": nodes, "edges": edges,
            "summary": {"nodes": len(graph.nodes), "edges": len(graph.edges), "calls": len(plan.calls),
                "priced_calls": len(predictions), "unpriced_calls": refused,
                "serial_roofline_seconds": None if refused else sum(p.t_total_s for p in predictions.values())}})
    left, right = observed
    comparisons, equal = [], 0
    keys = sorted(set(left["nodes"]) | set(right["nodes"]))
    for key in keys:
        old = left["nodes"][key] if key in left["nodes"] else None
        new = right["nodes"][key] if key in right["nodes"] else None
        if old is None or new is None:
            comparisons.append({"role": key, "status": "only_legacy" if new is None else "only_symbolic",
                "legacy": old, "symbolic": new})
            continue
        differences = {field: {"legacy": old[field], "symbolic": new[field]}
            for field in ("shape", "dtype", "variant", "attributes", "calls", "serial_roofline_seconds")
            if old[field] != new[field]}
        if not differences:
            equal += 1
        else:
            comparisons.append({"role": key, "status": "different",
                "legacy_node": old["node_id"], "symbolic_node": new["node_id"], "differences": differences})
    missing_edges = left["edges"] - right["edges"]
    extra_edges = right["edges"] - left["edges"]
    edge_diff = {"only_legacy": [{"edge": list(edge), "count": count} for edge, count in missing_edges.items()],
        "only_symbolic": [{"edge": list(edge), "count": count} for edge, count in extra_edges.items()]}
    unsupported = bool(left["summary"]["unpriced_calls"] or right["summary"]["unpriced_calls"])
    equivalent = not comparisons and not missing_edges and not extra_edges and not unsupported
    return {"status": "equivalent" if equivalent else "different_pending_review",
        "scope": "graph metadata, complete lowered programs, serial Roofline prices and dependencies; storage equivalence separately unproved",
        "equivalent": equivalent, "accepted_by_user": False, "method": "roofline",
        "legacy": left["summary"], "symbolic": right["summary"],
        "node_pairing": "kind/rank occurrence candidates; all fields and dependency edges checked afterwards",
        "nodes_total": len(keys), "nodes_equal": equal, "nodes_different": len(comparisons),
        "nodes_with_program_or_price_differences": sum(row["status"] != "different"
            or any(field in row["differences"] for field in ("shape", "dtype", "variant", "calls", "serial_roofline_seconds"))
            for row in comparisons),
        "node_differences": comparisons, "dependency_differences": edge_diff,
        "storage_lifetime_equivalence": "not_comparable_legacy_has_no_automatic_storage_trace",
        "limits": ["metadata equivalence does not prove numerical tensor equality or GPU/provider equivalence",
            "no unknown storage or skipped operation is counted as equivalent"]}


def evaluate_symbolic_model(model_path, query_lengths, context_lengths, *, upstream_root=None,
        source_profile="cuda_eager_inplace", tensor_parallel_size=1,
        source_max_model_len=None, max_batched_tokens=None, source_use_deep_gemm=None, expert_routing=None,
        indexer_selection_source=None, workspace_initial_capacity_bytes=None, target_sampling_source=None,
        prefill_attention_source=None, torch_include=None, flashinfer_source_root=None,
        k3_state_indices=None, k3_state_pool_slots=None, k3_prefill_output_source=None,
        mtp_draft_sampling_source=None, mtp_verified_tokens=None, mtp_accepted_tokens=None,
        mtp_proposer_tokens=None, mtp_acceptance_source_id=None, embedding_token_ids=None,
        v3_mtp_draft_input_token_ids=None, v3_mtp_attention_source=None, v3_mtp_batch_invariant=None,
        source_max_requests=None, source_block_size_tokens=None,
        hardware_config="configs/hardware/nvidia_h200.yaml", expert_parallel_size=1):
    if torch_include is not None and (source_profile != "vllm_0_30_v4_sync_fp8_ds_mla" or upstream_root is None):
        raise ValueError("Torch source-header comparison requires V4 and its explicit upstream root")
    if flashinfer_source_root is not None and (source_profile != "vllm_0_30_v4_sync_fp8_ds_mla" or upstream_root is None):
        raise ValueError("FlashInfer source comparison requires V4 and its explicit upstream root")
    if (not query_lengths or len(query_lengths) != len(context_lengths)
            or any(type(q) is not int or q <= 0 for q in query_lengths)
            or any(type(c) is not int or c < 0 for c in context_lengths)):
        raise ValueError("symbolic report requires aligned positive query and nonnegative context lengths")
    model = load_model_config(model_path)
    dense_source = source_profile in ("cuda_eager_inplace", "vllm_ascend_eager")
    glm4_source = source_profile == "vllm_0_11_glm4_sync"
    if type(expert_parallel_size) is not int or expert_parallel_size not in (1, 2):
        raise ValueError("virtual model report requires an explicit supported expert-parallel size")
    if expert_parallel_size > 1 and (model.model_type != "qwen3_moe" or tensor_parallel_size != 2 or not dense_source):
        raise NotImplementedError("virtual EP report currently requires ordinary Qwen3MoE TP=EP=2")
    glm_source = source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    kimi_source = source_profile == "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
    dsv3_source = source_profile == "vllm_0_27_1_deepseek_v3_sync_fp8"
    dsv3_mtp = mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy"
    mtp_fields = (mtp_verified_tokens, mtp_accepted_tokens, mtp_proposer_tokens, mtp_acceptance_source_id,
        embedding_token_ids, v3_mtp_draft_input_token_ids, v3_mtp_attention_source, source_max_requests, source_block_size_tokens)
    if dsv3_mtp:
        if (not dsv3_source or len(query_lengths) != 1 or any(value is None for value in mtp_fields)
                or any(type(count) is not int or count < 0 for count in
                    (mtp_verified_tokens, mtp_accepted_tokens, mtp_proposer_tokens))
                or not isinstance(mtp_acceptance_source_id, str) or not mtp_acceptance_source_id):
            raise ValueError("V3 MTP report requires one request, explicit acceptance and independent draft inputs")
        RuntimeInputs(expert_routing, "mla", "uniform", model_execution_version="vllm_0_27_1",
            symbolic_forward_source=source_profile, target_sampling_source=target_sampling_source,
            mtp_draft_sampling_source=mtp_draft_sampling_source, embedding_token_ids=embedding_token_ids,
            v3_mtp_draft_input_token_ids=v3_mtp_draft_input_token_ids, v3_mtp_attention_source=v3_mtp_attention_source,
            v3_mtp_batch_invariant=v3_mtp_batch_invariant,
            symbolic_source_max_model_len=source_max_model_len, symbolic_source_max_batched_tokens=max_batched_tokens,
            symbolic_source_max_requests=source_max_requests)
    elif (mtp_draft_sampling_source is not None or v3_mtp_batch_invariant is not None
            or any(value is not None for value in mtp_fields)):
        raise ValueError("MTP report inputs require the explicit0.27.1 V3 padded source")
    k3_source = source_profile == "vllm_0_30_k3_triton_marlin_bf16"
    if not k3_source and any(value is not None for value in (k3_state_indices, k3_state_pool_slots, k3_prefill_output_source)):
        raise ValueError("K3 state inputs require the explicit K3 source program")
    if prefill_attention_source is not None and not glm_source:
        raise ValueError("GLM prefill source does not apply to other model programs")
    if target_sampling_source is not None and not (glm_source or kimi_source or k3_source or glm4_source or dsv3_source):
        raise ValueError("this static target-sampling option requires the GLM/Kimi source")
    if dsv3_mtp:
        requests = (ScheduledRequest("0", "decode", 0, query_lengths[0], computed_tokens=context_lengths[0],
            raw_model_output_tokens=mtp_accepted_tokens + 1, mtp_proposed_tokens=mtp_verified_tokens,
            mtp_verified_tokens=mtp_verified_tokens, mtp_accepted_tokens=mtp_accepted_tokens,
            mtp_proposer_tokens=mtp_proposer_tokens, mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption", mtp_acceptance_source_id=mtp_acceptance_source_id),)
    else:
        requests = tuple(ScheduledRequest(str(i), "decode" if q == 1 and c > 0 else "prefill",
            0 if q == 1 and c > 0 else q, 1 if q == 1 and c > 0 else 0, computed_tokens=c)
            for i, (q, c) in enumerate(zip(query_lengths, context_lengths, strict=True)))
    step = ScheduleStep("symbolic-report", "main", requests)
    tokens = sum(query_lengths)
    positions = tuple(p for q, c in zip(query_lengths, context_lengths, strict=True) for p in range(c, c + q))
    slots, offset = [], 0
    for q, c in zip(query_lengths, context_lengths, strict=True):
        slots.extend(range(offset + c, offset + c + q))
        offset += c + q
    target_hardware = HWSpec(hardware_config)
    hardware_id = target_hardware.hardware_id
    if (hardware_id not in ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
            or (not dense_source and not glm4_source and hardware_id != "nvidia_h200")
            or (glm4_source and hardware_id != "nvidia_a800")
            or ((source_profile == "vllm_ascend_eager") != (hardware_id == "ascend_910b"))):
        raise NotImplementedError("symbolic source report requires its supported CUDA-alike hardware domain")
    hardware = {hardware_id: target_hardware}
    collectives = () if tensor_parallel_size == 1 else tuple(CollectiveConfig(operation,
        "ring", {"nvidia_a800": "nvlink", "nvidia_h200": "nvlink", "metax_c500": "metaxlink",
            "ascend_910b": "hccs"}[hardware_id], "ring", "declared-source-fabric", "analytical_roofline_v1")
        for operation in ("all_reduce", "all_gather"))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (
        WorkerPoolConfig("main", "colocated", hardware_id, ParallelConfig(tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=expert_parallel_size > 1),
            collectives=collectives),)), num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    start = perf_counter()
    v4_observation = None
    v4_observation_seconds = None
    execution_scope = "complete_model_with_plain_greedy"
    if dense_source:
        if any(value is not None for value in (
                source_max_model_len, max_batched_tokens, source_use_deep_gemm,
                indexer_selection_source, workspace_initial_capacity_bytes)):
            raise ValueError("dense source report does not consume V4-only source inputs")
        if model.model_type == "qwen3_moe":
            if expert_routing is None:
                raise ValueError("virtual Qwen3 MoE requires an explicit routing input")
            source = traced_models.trace_qwen3_moe_forward(model, step, deployment.worker_pools["main"],
                expert_routing, input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=source_profile)
            graph = source.graph
            traces = (source,) if tensor_parallel_size == 1 else source.traces
        elif model.model_type == "qwen3":
            forward = traced_models.trace_qwen3_forward
        elif model.model_type == "llama":
            forward = traced_models.trace_llama_forward
        else:
            raise NotImplementedError("dense symbolic report requires an adapted Qwen3 or Llama forward")
        if model.model_type != "qwen3_moe" and expert_routing is not None:
            raise ValueError("dense non-MoE source cannot consume expert routing")
        if model.model_type == "qwen3_moe":
            pass
        elif tensor_parallel_size == 1:
            trace = forward(model, step, input_ids=(1,) * tokens,
                positions=positions, slot_mapping=tuple(slots), cache_capacity_tokens=offset,
                source_profile=source_profile, hardware_id=hardware_id)
            graph, traces = trace.graph, (trace,)
        else:
            group = traced_models.trace_dense_parallel_forward(model, step, deployment.worker_pools["main"],
                input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=source_profile)
            graph, traces = group.graph, group.traces
        model_limitations = ["GQA uses ideal mathematical FLOPs/argument bytes, not FA2/FA3 source-issued online-softmax/split work"]
        if model.model_type == "qwen3_moe":
            model_limitations += [
                "routing uses explicit128-expert softmax/top-k arithmetic and public-role payload; comparisons, sanitation, index selection, padding and actual library issue are outside this conventional FLOPs/bytes baseline",
                "declared routing histogram is not an observed checkpoint routing result; native source selection and normal capacity remain unverified"]
        if hardware_id == "ascend_910b":
            model_limitations += [
                "vLLM0.30 model body composed with local Ascend plugin storage source; not installed NPU0.30 runtime acceptance",
                "ordinary TP/DP1/SP1 with embedding_tp and lmhead_tp disabled, no added vocabulary or norm bias; compiler fusion/provider and prefetch are not inferred",
                "HCCS ring bytes use the retained independent two-rank broadcast bandwidth as ideal link capacity, not observed collective service"]
    elif dsv3_source:
        from src.serving.workload.traced_deepseek_v3 import trace_deepseek_v3_decode, trace_deepseek_v3_mtp_forward
        l32_tp4 = model.num_hidden_layers == 32 and tensor_parallel_size == 4
        if ((tensor_parallel_size != 8 and not l32_tp4) or expert_routing is None
                or source_max_model_len is None or max_batched_tokens is None
                or any(value is not None for value in (
                    upstream_root, source_use_deep_gemm, indexer_selection_source,
                    workspace_initial_capacity_bytes))):
            raise ValueError("DeepSeek-V3 report requires explicit TP8/source limits/routing and excludes other-model audit inputs")
        if dsv3_mtp:
            next_position = offset - mtp_verified_tokens + mtp_accepted_tokens
            group = trace_deepseek_v3_mtp_forward(model, step, deployment.worker_pools["main"], expert_routing,
                input_ids=embedding_token_ids, positions=positions, slot_mapping=tuple(slots),
                cache_capacity_tokens=max(offset, next_position + 1), source_profile=source_profile,
                source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
                max_requests=source_max_requests, block_size_tokens=source_block_size_tokens,
                draft_input_token_ids=v3_mtp_draft_input_token_ids, attention_source=v3_mtp_attention_source,
                source_batch_invariant=v3_mtp_batch_invariant)
        else:
            group = trace_deepseek_v3_decode(model, step, deployment.worker_pools["main"], expert_routing,
                input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots),
                cache_capacity_tokens=offset, source_profile=source_profile,
                source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
                target_sampling_source=target_sampling_source)
        graph, traces = group.graph, group.traces
        execution_scope = "model_and_logits_without_sampler" if target_sampling_source is None else "complete_model_with_plain_greedy"
        if dsv3_mtp:
            execution_scope = "complete_target_verify_and_two_drafts"
        model_limitations = [
            ("registered local32-layer DeepSeek-V3 variant using the same0.27.1 make_layers(config.num_hidden_layers) source"
                if model.num_hidden_layers == 32 else "official DeepSeek-V3 vLLM0.27.1 synchronous TP8 EP-off source projection, not a0.30 runtime binding"),
            "scheduled metadata and expert routing are declared synthetic inputs, not observed checkpoint outputs",
            "MLA attention, block-FP8 projections, routing and greedy use useful mathematical FLOPs and public-role bytes",
            "library/provider launches, internal scratch and normal capacity remain unbound",
            (f"MTP is an explicit synchronous no-discard B1/K2 source scenario: {v3_mtp_attention_source}; selected provider/runtime unbound"
                if dsv3_mtp else "MTP remains unbound in this target-only source scope")]
    elif k3_source:
        if (tensor_parallel_size != 1 or target_sampling_source != "vllm_0_30_plain_greedy_bf16"
                or any(value is not None for value in (upstream_root, source_max_model_len, max_batched_tokens,
                    source_use_deep_gemm, indexer_selection_source, workspace_initial_capacity_bytes))):
            raise ValueError("K3 source requires TP1/plain greedy and does not consume other model-source options")
        trace = traced_models.trace_kimi_k3_forward(model, step, deployment.worker_pools["main"],
            expert_routing, input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots),
            cache_capacity_tokens=offset,
            state_indices_by_layer=tuple(
                k3_state_indices if kind == "kda" else ()
                for kind in model.kimi_k3_semantics.layer_attention_types),
            state_pool_slots=k3_state_pool_slots, source_max_model_len=offset,
            max_batched_tokens=tokens, prefill_output_source=k3_prefill_output_source)
        graph, traces = trace.graph, (trace,)
        model_limitations = ["single-token decode, one fresh T>=512 full/tail64-chunk prefill or decode-first/fresh mixed; context/MTP are not migrated",
            "fresh prefill requires an explicit backend supports_out declaration; no actual prefill launch/provider is selected",
            "KDA Triton, BF16 NoPE MLA and modular Marlin are source declarations, not runtime dispatch",
            "grouped routing, SiTU, expert internals and output-gate use analytical source prices; native instruction and provider services remain unbound",
            "automatic public-storage lifetimes do not cover unknown callee scratch, streams or normal capacity",
            "no independent whole-model source or native/runtime equivalence acceptance"]
    elif source_profile == "vllm_0_30_v4_sync_fp8_ds_mla":
        if model.model_type != "deepseek_v4":
            raise ValueError("V4 source profile requires its accurate model config")
        if any(value is None for value in (source_max_model_len, max_batched_tokens, source_use_deep_gemm, expert_routing)):
            raise ValueError("V4 source report requires explicit context/batch limits, DeepGEMM branch and routing input")
        alignment = max(model.compress_ratios)
        logical_capacity = -(-max(offset, alignment) // alignment) * alignment
        def trace_v4():
            return traced_models.trace_deepseek_v4_forward(model, step, deployment.worker_pools["main"],
                hardware["nvidia_h200"], expert_routing, input_ids=(1,) * tokens,
                positions=positions, slot_mapping=tuple(slots), cache_capacity_tokens=logical_capacity,
                source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
                source_use_deep_gemm=source_use_deep_gemm, linear_source_profile="vllm_0_30_row_major_group128_fp32",
                indexer_selection_source=indexer_selection_source,
                workspace_initial_capacity_bytes=workspace_initial_capacity_bytes)
        group = trace_v4()
        if upstream_root is not None:
            observed_started = perf_counter()
            with observe_v4_source_calls() as v4_observation:
                observed = trace_v4()
            if observed.graph != group.graph:
                raise ValueError("source observation replay changed V4 model dataflow")
            del observed
            v4_observation_seconds = perf_counter() - observed_started
        graph, traces = group.graph, group.traces
        model_limitations = ["scored long C4 needs explicit declared causal top-k cardinality; it is not inferred from finite logits or verified against checkpoint outputs",
            "indexer uses useful mathematical/byte baseline; selection comparisons/refinement, score backing and actual backend remain unbound",
            "Marlin modular arena requires explicit per-rank entry capacities; absent history is not replaced by a cold0 assumption",
            "prefill sparse and modular arenas share the declared manager; Marlin source alignment/locks/serial FP32 allocation requests are represented; unknown reduction accesses and callee storage remain separate",
            "C128 metadata construction and FlashMLA decode planner remain external forward services"]
    elif glm4_source:
        if (expert_routing is None or tensor_parallel_size != 1
                or target_sampling_source not in (None, "vllm_0_11_plain_greedy_bf16")
                or any(value is not None for value in (source_max_model_len, max_batched_tokens,
                    source_use_deep_gemm, indexer_selection_source, workspace_initial_capacity_bytes))):
            raise ValueError("GLM4 report requires explicit routing, TP1 and its0.11 plain greedy source")
        trace = traced_models.trace_glm4_forward(model, step, deployment.worker_pools["main"], expert_routing,
            input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_profile=source_profile)
        graph, traces = trace.graph, (trace,)
        model_limitations = ["existing synthetic two-layer GLM4 fixture; not an accurate checkpoint model",
            "explicit0.11 non-modular single-group FP32 sigmoid routing and separate shared FFN",
            "useful arithmetic/public-byte Roofline; selection, finite precision and actual provider remain unbound",
            "MTP predictor model body exists; proposer/rejection caller is not migrated by this target-off case",
            "normal capacity and runtime/provider are not accepted"]
    elif glm_source:
        if source_use_deep_gemm is not None:
            raise ValueError("GLM decoder does not consume the V4 DeepGEMM source switch")
        if any(value is None for value in (source_max_model_len, max_batched_tokens, expert_routing)):
            raise ValueError("GLM decoder report requires explicit limits and canonical routing input")
        entry = traced_models.trace_glm_dsa_decoder if target_sampling_source is None else traced_models.trace_glm_dsa_forward
        group = entry(model, step, deployment.worker_pools["main"], expert_routing,
            input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            selection_source=indexer_selection_source, source_profile=source_profile,
            prefill_attention_source=prefill_attention_source,
            **({} if target_sampling_source is None else {"target_sampling_source": target_sampling_source}))
        graph, traces = group.graph, group.traces
        execution_scope = "decoder_forward_without_logits_sampler" if target_sampling_source is None else "complete_model_with_plain_greedy"
        model_limitations = [("GLM model.forward scope only; logits/sampler are excluded from this static case"
            if target_sampling_source is None else
            "GLM target chain is connected to production virtual dispatch; request completion is evaluated separately"),
            "modular BF16-compute non-overlap source and explicit all-reduce/norm branch, not an installed provider",
            "routing uses an explicit dynamic assumption and useful mathematical FLOPs/public-role bytes; packed selection, rounded source branches and actual provider remain unbound",
            "DSA score/selector and latent attention have useful mathematical baselines, not full backend service",
            "logical-to-physical index conversion, score backing, backend ignored return and callee scratch remain unbound",
            "TP8/PP1 source graph does not establish normal H200 capacity"]
        if model.weight_dtype == "fp8":
            model_limitations.append("GLM FP8 grouped experts price an explicit useful mathematical program only; actual provider/padding/alignment/internal storage remain unbound and the DeepSeek-V3 FP8 launch census is a different geometry")
        if prefill_attention_source is not None:
            model_limitations.append("prefill/mixed declares attention_config.sparse_mla_force_mqa=True; default dense/masked-MHA selection is not inferred; shared gather capacity/FP32 scale alias and unique copy payload are represented; gather chunk partition, rectangular score backing and internal workspace remain unbound")
        elif any(q > 1 for q in query_lengths):
            model_limitations.append("fresh short prefill follows the source default dense-MHA condition with HiSparse/force-MQA off and every context within top-k; KV expansion, prepared Q/K/V and output copy are represented; context/long masked-MHA and internal backend workspace remain unbound")
        if target_sampling_source is not None:
            model_limitations.append("row-index metadata production, possible ATen index conversion and internal library/collective scratch remain unbound")
    elif kimi_source:
        if any(value is not None for value in (source_use_deep_gemm, indexer_selection_source,
                workspace_initial_capacity_bytes, upstream_root)):
            raise ValueError("K2.6 static source does not bind V4/indexer/arena options or a whole-source comparison")
        if any(value is None for value in (source_max_model_len, max_batched_tokens, expert_routing)):
            raise ValueError("K2.6 source requires explicit context/batch limits and canonical routing input")
        group = traced_models.trace_kimi_k2_forward(model, step, deployment.worker_pools["main"], expert_routing,
            input_ids=(1,) * tokens, positions=positions, slot_mapping=tuple(slots), cache_capacity_tokens=offset,
            source_max_model_len=source_max_model_len, max_batched_tokens=max_batched_tokens,
            source_profile=source_profile, target_sampling_source=target_sampling_source)
        graph, traces = group.graph, group.traces
        execution_scope = "decoder_forward_without_logits_sampler" if target_sampling_source is None else "complete_model_with_plain_greedy"
        model_limitations = [
            "K2.6 pure-text CP1 decode/fresh-prefill/mixed and one nonempty single-chunk context-prefill source are connected to production virtual dispatch; general context batches and request completion are evaluated separately",
            "modular/no-overlap shared experts, BF16 gate logits, INT4/group32 routed weights; not an installed provider",
            "router factor1 and a separate in-place routed-output scale precede shared-add and late TP all-reduce",
            "routing uses an explicit dynamic assumption and useful mathematical FLOPs/public-role bytes; CUDA FP32 and torch BF16 source programs and complete native selection remain unbound",
            "INT4 experts reuse existing Marlin pricing; that is not a vLLM0.30 native/compiled binding",
            "latent MLA/logsumexp is useful mathematical work; actual pages, scheduler metadata, scratch and stride remain unbound",
            "Marlin/shared workspace, complete worker memory and normal capacity remain unbound"]
    else:
        raise ValueError("unsupported symbolic source profile")
    trace_seconds = perf_counter() - start - (0 if v4_observation_seconds is None else v4_observation_seconds)
    start = perf_counter()
    per_rank = {str(trace.rank): peak_live_activation_bytes(trace.activation_allocations, trace.event_order)
        for trace in traces}
    peak = max(value[0] for value in per_rank.values())
    live = tuple(storage for value in per_rank.values() if value[0] == peak for storage in value[1])
    peak_seconds = perf_counter() - start
    plan = lower_workload(graph, deployment, hardware)
    predictions, refused = _diagnose_symbolic_pricing(plan, hardware)
    unpriced = tuple(row["call_id"] for row in refused)
    result = {"model": str(model_path), "model_config_sha256": model.config_sha256,
        "hardware_id": hardware_id, "hardware_config": str(hardware_config),
        "source_profile": source_profile, "execution": "pure_python_tensor_metadata_only",
        "execution_scope": execution_scope, "full_model_graph_complete": execution_scope in (
            "complete_model_with_plain_greedy", "complete_target_verify_and_two_drafts"),
        "tensor_parallel_size": tensor_parallel_size, "expert_parallel_size": expert_parallel_size,
        "source_inputs": {"max_model_len": source_max_model_len, "max_batched_tokens": max_batched_tokens,
            **({"model_execution_version": "vllm_0_27_1"} if dsv3_source else {}),
            "rotary_source_program": vllm_rope_program(hardware_id, model.rope_scaling_type) if dense_source else None,
            "use_deep_gemm": source_use_deep_gemm,
            "indexer_selection_source": indexer_selection_source,
            "target_sampling_source": target_sampling_source,
            "prefill_attention_source": prefill_attention_source,
            "workspace_initial_capacity_bytes": workspace_initial_capacity_bytes,
            "workspace_lock_state": None if workspace_initial_capacity_bytes is None else "declared_unlocked_source_projection",
            **({"k3_state_indices": k3_state_indices, "k3_state_pool_slots": k3_state_pool_slots,
                "k3_prefill_output_source": k3_prefill_output_source} if k3_source else {}),
            "expert_routing": None if expert_routing is None else {
                "source_kind": expert_routing.source_kind, "source_id": expert_routing.source_id, "root_seed": expert_routing.root_seed}},
        "input_kind": "explicit_synthetic_token_ids_and_logical_cache_slots",
        "query_lengths": query_lengths, "context_lengths": context_lengths,
        "layers": model.num_hidden_layers, "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges), "operator_counts": dict(Counter(n.kind.value for n in graph.nodes)),
        "kernel_calls": len(plan.calls), "unpriced_calls": unpriced,
        "priced_calls": len(predictions), "unpriced_details": refused,
        "unpriced_by_variant": dict(Counter(row["variant"] for row in refused)),
        "scored_indexer_baseline_calls": sum(call.kernel_spec.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode")
            or (call.kernel_spec.variant in ("dsa_indexer_decode", "dsa_indexer_prefill", "dsa_indexer_mixed")
                and call.kernel_spec.attributes["mathematical_program"] is not None)
            for call in plan.calls),
        "scored_selection_cardinality_observed": False,
        "serial_roofline_seconds": None if unpriced else sum(p.t_total_s for p in predictions.values()),
        "activation_storage_peak_bytes": peak, "peak_live_storage_ids": live,
        "activation_storage_peak_bytes_by_rank": {rank: value[0] for rank, value in per_rank.items()},
        "peak_live_storage_ids_by_rank": {rank: value[1] for rank, value in per_rank.items()},
        "peak_live_storage_by_rank": {str(trace.rank): [
            {"allocation_id": a.allocation_id, "size_bytes": a.size_bytes,
             "allocated_event": a.allocated_event, "freed_event": a.freed_event}
            for a in trace.activation_allocations if a.allocation_id in per_rank[str(trace.rank)][1]]
            for trace in traces},
        "activation_storage_complete": False,
        "allocation_metadata_count": sum(len(trace.allocations) for trace in traces),
        "shared_workspace_allocations_by_rank": {str(trace.rank): sum(a.allocation_id.startswith(trace.qualify("worker.workspace"))
            for a in trace.allocations) for trace in traces},
        "shared_workspace_storage_by_rank": {str(trace.rank): [
            {"allocation_id": a.allocation_id, "size_bytes": a.size_bytes,
             "allocated_event": a.allocated_event, "freed_event": a.freed_event}
            for a in trace.allocations if a.allocation_id.startswith(trace.qualify("worker.workspace"))]
            for trace in traces},
        "view_metadata_count": sum(len(trace.views) for trace in traces),
        "trace_cpu_seconds": trace_seconds, "peak_sweep_cpu_seconds": peak_seconds,
        "normal_capacity_accepted": False, "serving_accepted": False, "runtime_accepted": False,
        "limitations": ["source profile does not bind actual CUDA/IR provider or compiler lifetimes",
            *model_limitations,
            "library/internal workspace and allocator/non-torch persistent memory remain separate",
            "persistent cache objects describe logical pool, not physical allocator capacity",
            "greedy comparison/index instructions remain separate from FLOPs/bytes Roofline baseline",
            "static source graph scope is explicit; it is not Scheduler request completion or QPS16"]}
    if dsv3_mtp:
        result["mtp_source_scenario"] = {"source_kind": "explicit_assumption",
            "source_id": mtp_acceptance_source_id, "verified_tokens": mtp_verified_tokens,
            "accepted_tokens": mtp_accepted_tokens, "proposer_tokens": mtp_proposer_tokens,
            "attention_source": v3_mtp_attention_source, "target_token_ids": embedding_token_ids,
            "source_batch_invariant": v3_mtp_batch_invariant,
            "draft_input_token_ids_by_pass": v3_mtp_draft_input_token_ids,
            "source_max_requests": source_max_requests, "source_block_size_tokens": source_block_size_tokens,
            "actual_runtime_bound": False}
    if glm_source:
        result["dsa_indexer_source_payload"] = _summarize_dsa_indexer_payload(plan)
    if upstream_root is not None and not (glm_source or kimi_source or k3_source or glm4_source):
        comparison = (compare_source_forward_bodies if dense_source
            else compare_v4_ffn_forward_bodies)
        result["source_forward_comparison"] = (comparison(Path(upstream_root), model_type=model.model_type)
            if dense_source else comparison(Path(upstream_root)))
        if source_profile == "cuda_eager_inplace" and model.model_type == "qwen3" and tensor_parallel_size == 1:
            reference = qwen3_source_reference(upstream_root, model_path, query_lengths=query_lengths,
                kv_lengths=tuple(q + c for q, c in zip(query_lengths, context_lengths, strict=True)),
                slot_mapping=tuple(slots), cache_capacity_tokens=offset)
            result["source_caller_comparison"] = compare_qwen3_source_graph(reference, trace)
            result["source_caller_reference"] = {key: value for key, value in reference.items() if key != "calls"}
    if upstream_root is not None and (glm_source or source_profile == "vllm_0_30_v4_sync_fp8_ds_mla"):
        result["source_indexer_gather_workspace_comparison"] = compare_indexer_gather_workspace(
            Path(upstream_root), graph, traces, max_model_len=source_max_model_len,
            head_dim=model.index_head_dim, model_type=model.model_type)
    if v4_observation is not None:
        reference_started = perf_counter()
        reference = v4_model_source_reference(upstream_root, model_path, query_lengths=query_lengths,
            kv_lengths=tuple(q + c for q, c in zip(query_lengths, context_lengths, strict=True)),
            request_phases=tuple(request.phase for request in requests), max_batched_tokens=max_batched_tokens,
            cache_capacity_tokens=logical_capacity, tensor_parallel_size=tensor_parallel_size,
            input_token_ids=(1,) * tokens,
            indexer_selection_source=indexer_selection_source, source_use_deep_gemm=source_use_deep_gemm,
            multiprocessors=int(hardware["nvidia_h200"].num_compute_units), torch_include=torch_include,
            flashinfer_source_root=flashinfer_source_root, source_max_model_len=source_max_model_len)
        result["source_v4_caller_comparison"] = compare_v4_model_source_calls(reference, v4_observation,
            primitive_nodes=len(graph.nodes))
        result["source_v4_caller_reference"] = reference
        result["source_v4_mixed_projection_comparison"] = compare_v4_input_projection_work(reference, plan, hardware)
        result["source_v4_compressor_dependency_comparison"] = compare_v4_compressor_dependencies(reference,
            v4_observation, graph)
        result["source_v4_dependency_comparison"] = compare_v4_source_dependencies(reference, v4_observation, graph)
        result["source_v4_prepare_comparison"] = compare_v4_prepare_work(reference["prepare_work_reference"], plan, hardware)
        result["source_v4_partial_state_comparison"] = compare_v4_partial_state_work(reference, plan, hardware)
        result["source_v4_finish_comparison"] = compare_v4_finish_work(reference, plan, hardware)
        result["source_v4_ffn_comparison"] = compare_v4_ffn_work(reference, plan, hardware)
        result["source_v4_fp8_dataflow_comparison"] = compare_v4_fp8_dataflow(reference, plan)
        if "fp8_quantizer_reference" in reference:
            result["source_v4_quantizer_comparison"] = compare_v4_quantizer_work(reference["fp8_quantizer_reference"], plan)
        result["source_v4_ffn_vector_comparison"] = compare_v4_ffn_vector_work(reference, plan, hardware)
        result["source_v4_output_comparison"] = compare_v4_output_work(reference, plan, hardware)
        result["source_v4_indexer_comparison"] = compare_v4_indexer_work(reference, plan, hardware)
        result["source_v4_attention_math_comparison"] = compare_v4_attention_math(reference, plan)
        result["source_v4_mhc_comparison"] = compare_v4_mhc_work(reference, v4_observation, plan, hardware)
        result["source_v4_qkv_norm_comparison"] = compare_v4_qkv_norm_work(reference, plan, hardware)
        result["source_v4_final_norm_comparison"] = compare_v4_final_norm_work(reference, plan)
        result["source_v4_target_comparison"] = compare_v4_target_work(reference, plan, hardware)
        result["source_v4_routing_objects_comparison"] = compare_v4_routing_objects(reference, traces, graph)
        result["source_v4_routing_math_comparison"] = compare_v4_routing_math(reference, plan, hardware)
        state_reference = logical_state_reference(json.loads(Path(model_path).read_text()), upstream_root,
            query_lengths=tuple(query_lengths), kv_lengths=tuple(q + c for q, c in zip(query_lengths, context_lengths, strict=True)),
            request_phases=tuple(reference["declared_inputs"]["request_phases"]), capacity=logical_capacity, tp=tensor_parallel_size)
        result["source_v4_logical_state_reference"] = state_reference
        result["source_v4_logical_state_comparison"] = compare_v4_logical_state(state_reference, traces, graph)
        result["source_v4_reduction_comparison"] = compare_v4_reductions(state_reference, traces, graph)
        if reference["argmax_source_work"] is not None:
            result["source_v4_argmax_comparison"] = compare_v4_argmax_source(reference["argmax_source_work"], plan)
        result["source_v4_caller_comparison"]["unobserved_graph_equals_observed_replay"] = True
        storage_reference = wrapper_storage_reference(reference)
        result["source_v4_wrapper_storage_reference"] = storage_reference
        result["source_v4_wrapper_storage_comparison"] = compare_v4_wrapper_storage(storage_reference,
            v4_observation, traces, graph)
        result["source_v4_terminal_storage_comparison"] = compare_v4_terminal_storage(reference["terminal_storage_work"], traces, graph)
        result["source_v4_decode_index_storage_comparison"] = compare_v4_decode_index_storage(reference["decode_index_storage_work"], traces, graph)
        result["source_v4_expert_storage_comparison"] = compare_v4_expert_storage(reference["expert_storage_work"], traces, graph)
        result["source_v4_decode_selector_storage_comparison"] = compare_v4_decode_selector_storage(reference["indexer_decode_storage_work"], traces, graph)
        result["source_v4_sparse_prefill_storage_comparison"] = compare_v4_sparse_prefill_storage(reference["sparse_prefill_storage_work"], traces, graph)
        result["source_v4_sparse_decode_views_comparison"] = compare_v4_sparse_decode_views(reference["sparse_decode_view_work"], traces, graph)
        result["source_v4_scope_inventory"] = v4_source_scope_inventory(reference, result)
        result["source_v4_observation_cpu_seconds"] = v4_observation_seconds
        result["source_v4_reference_cpu_seconds"] = perf_counter() - reference_started
    return result


def write_symbolic_model_report(model_path, query_lengths, context_lengths, output, *, upstream_root=None,
        report=False, **source_inputs):
    from evaluation.reporting.serving_interactive import write_status_report, default_report_path
    from evaluation.reporting.serving_viewer_ui import configuration_file_snapshots, _configuration_metadata
    if type(report) is not bool:
        raise TypeError("report must be boolean")
    started = perf_counter()
    files, missing = (configuration_file_snapshots(model_path,
        [source_inputs["hardware_config"]] if "hardware_config" in source_inputs else []) if report else ([], []))
    try:
        result = evaluate_symbolic_model(model_path, query_lengths, context_lengths, upstream_root=upstream_root, **source_inputs)
    except Exception as error:
        if report:
            write_status_report({"main_chain_status": {
                "status": "unsupported" if isinstance(error, NotImplementedError) else "failed",
                "reason": type(error).__name__ + ": static evaluation did not complete; see process log"},
                "request_source": {"model_path": str(model_path), "query_lengths": query_lengths, "context_lengths": context_lengths},
                "visualization_configuration_files": files, "visualization_configuration_snapshot_missing": missing},
                default_report_path(output))
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    cost_path = output.with_suffix(".report.json")
    if report:
        options = dict(source_inputs)
        if "expert_routing" in options and options["expert_routing"] is not None:
            options["expert_routing"] = options["expert_routing"].manifest()
        status = "complete" if result["full_model_graph_complete"] and not result["unpriced_calls"] else "partial"
        payload = {"report_links": [{"label": "Cost receipt", "path": str(cost_path)}],
            "main_chain_status": {"status": status,
                "reason": "Static model-source projection only; Serving, normal capacity, and actual runtime remain unaccepted."},
            "report_facts": [{"label": "Model", "value": result["model"]},
                {"label": "Hardware", "value": result["hardware_id"]},
                {"label": "Full model graph", "value": result["full_model_graph_complete"]},
                {"label": "Operator calls", "value": result["kernel_calls"]},
                {"label": "Unpriced calls", "value": len(result["unpriced_calls"])},
                {"label": "Serial predicted time (s)", "value": result["serial_roofline_seconds"]}],
            "model_config_provenance": {"model_id": result["model"], "config_sha256": result["model_config_sha256"]},
            "deployment": "static_source_projection",
            "deployment_plan": {"tensor_parallel_size": result["tensor_parallel_size"], "expert_parallel_size": result["expert_parallel_size"]},
            "visualization_hardware": {result["hardware_id"]: {}},
            "request_source": {"scope": "static source inputs, not scheduled Serving requests",
                "query_lengths": query_lengths, "context_lengths": context_lengths,
                "requested_source_options": _configuration_metadata(options), "upstream_root": upstream_root},
            "visualization_configuration_files": files, "visualization_configuration_snapshot_missing": missing}
        result["interactive_report"] = write_status_report(payload, default_report_path(output))
        result["interactive_report"]["cost_receipt_path"] = str(cost_path)
    else:
        result["interactive_report"] = {"status": "disabled", "reason": "offline HTML report not requested"}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_symbolic_summary(result, output.with_suffix(".md"))
    result["report_output_costs"] = {"total_wall_s": perf_counter() - started,
        "scope": "static evaluation plus HTML/projection/result JSON/Markdown; final receipt writing is separate"}
    if report:
        from evaluation.reporting.serving_viewer_ui import attach_report_costs
        costs = result["report_output_costs"]
        costs["cost_metadata_write_s"] = attach_report_costs(result["interactive_report"], {
            "html_generation_s": result["interactive_report"]["page_generation_s"],
            "total_wall_through_result_write_s": costs["total_wall_s"],
            "scope": "static evaluation and initial HTML/projection/JSON/Markdown writes; final metadata/receipt writes are separately measured"})
    refresh_started = perf_counter()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    result["report_output_costs"]["final_result_refresh_s"] = perf_counter() - refresh_started
    result["report_output_costs"]["total_wall_s"] = perf_counter() - started
    receipt_started = perf_counter()
    cost_path.write_text(json.dumps({"report": result["interactive_report"], "costs": result["report_output_costs"]}, indent=2) + "\n")
    result["report_output_costs"]["cost_receipt_write_s"] = perf_counter() - receipt_started
    result["report_output_costs"]["total_wall_s"] = perf_counter() - started
    return result


def write_symbolic_summary(result, output):
    """Render saved evaluation facts without rerunning a model or prediction."""
    lines = ["# Virtual model static Roofline report", "",
        "Virtual tensors retain only shape/dtype/stride/storage metadata; no real tensors, weights, or numerical forward are executed. This report validates the static graph; request-loop results are recorded separately.", "",
        "| Item | Result |", "|---|---|",
        f"| Model | {result['model']} |",
        f"| Hardware | {result['hardware_id']} |",
        f"| Execution scope | {result['execution_scope']} |",
        f"| query / context | {result['query_lengths']} / {result['context_lengths']} |",
        f"| Full layer count | {result['layers']} |",
        f"| Source branch / TP | {result['source_profile']} / {result['tensor_parallel_size']} |",
        f"| Declared GLM prefill setting | {result['source_inputs']['prefill_attention_source']} |",
        f"| Declared indexer output count | {result['source_inputs']['indexer_selection_source']}; not evidence of real checkpoint/provider outputs |",
        f"| Graph nodes / kernel calls | {result['graph_nodes']} / {result['kernel_calls']} |",
        f"| Unpriced | {len(result['unpriced_calls'])} |",
        f"| Unpriced groups | {result['unpriced_by_variant']} |",
        f"| Total serial Roofline time | {'Unavailable: unpriced calls remain' if result['serial_roofline_seconds'] is None else str(result['serial_roofline_seconds']) + ' s'} |",
        f"| Scored indexer calls with useful math/byte baseline only | {result['scored_indexer_baseline_calls']}; selection comparisons/refinement remain excluded |",
        f"| Represented activation storage, maximum single-rank peak | {result['activation_storage_peak_bytes']} B ({result['activation_storage_peak_bytes'] / 2**30:.6f} GiB) |",
        f"| Peaks by rank | {result['activation_storage_peak_bytes_by_rank']} B |",
        f"| CPU graph construction / peak scan | {result['trace_cpu_seconds']:.6f} s / {result['peak_sweep_cpu_seconds']:.6f} s |",
        "| Normal capacity / Serving / runtime acceptance in this report | None passed |", "",
        "Unique live storage is scanned independently on each physical rank; multiple ranks are not summed into single-card capacity. Missing workspace and runtime state are listed below; this value is not a full worker peak.", ""]
    if "source_forward_comparison" in result:
        comparison = result["source_forward_comparison"]
        lines.extend([f"Independent source-function-body comparison: {len(comparison['checks'])} checks, {comparison['failed']} failures; scope: {comparison['scope']}. This does not accept full-model source alignment.", ""])
    if "source_caller_comparison" in result:
        comparison = result["source_caller_comparison"]
        lines.extend([f"Independent caller-interface comparison: {comparison['total']} calls ({comparison['upstream_forward_driven_calls']} driven by upstream model forward, {comparison['declared_suffix_calls']} target-suffix calls, suffix source-bound={comparison['target_suffix_source_bound']}); {comparison['matched']} matched, {comparison['mismatched']} mismatched, {comparison['not_checked']} unchecked. The target suffix executes original LogitsProcessor and Sampler methods. Row-select indices and materialization expressions independently read worker source. CUDA in-place norm/RoPE, logical cache, head GEMM, and argmax remain declared primitive-interface contracts. Coverage includes kinds, shapes, dtypes, inputs/outputs, aliases, and dependencies; it does not establish operator internals, numerical NaN/tie handling, physical KV, complete worker memory, or actual provider/runtime.", ""])
    if "source_v4_caller_comparison" in result:
        comparison = result["source_v4_caller_comparison"]
        lines.extend([f"V4 full-layer-stack and attention caller interfaces: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Executes original model/decoder, attention forward/input/norm/prepare, short or scored indexer branches, and plain-greedy methods. Checks order, shape, dtype, and storage identity of Q projection, KV insert, compressor, top-k, and sparse ports. Short top-k projects the source-written active-row view as symbolic output; the source itself returns None. Rotary/cache objects, mutation dependencies, and kernel internals remain unchecked by these ports. Observation runs in independent replay and requires an identical graph; pricing/memory use the unobserved original graph. Complete internal semantics of {comparison['primitive_internal_checks']['not_checked']} primitive nodes are not verified by caller checks; full-model source alignment is not accepted.", ""])
    if "legacy_path_comparison" in result:
        pair = result["legacy_path_comparison"]
        if pair["legacy"] is None:
            lines.extend([f"Old/new routes have no comparable result: {pair['status']}; baseline reason: {pair['baseline_error']['reason']}. The new graph is retained; checks were not removed and no alternative mode was selected to manufacture equivalence.", ""])
        else:
            lines.extend([f"Old/new route comparison: {pair['status']}; {pair['nodes_total']} per-operator candidate roles, {pair['nodes_equal']} identical, {pair['nodes_different']} different. Old/new call counts: {pair['legacy']['calls']}/{pair['symbolic']['calls']}; JSON contains all differences and dependency changes. Differences remain unaccepted by the user. The old route lacks automatic storage lifetimes; similar prices cannot prove equivalence.", ""])
    if "source_v4_scope_inventory" in result:
        inventory = result["source_v4_scope_inventory"]
        lines.extend([f"Full source-call inventory: {inventory['source_calls']} calls across {inventory['source_role_families']} role families. Math/typed-payload check entry points are registered individually for {inventory['primitive_role_families']} primitive role families; {len(inventory['unchecked_math_role_families'])} families remain uncovered. Unified comparison of internal temporary allocations, aliases, and lifetimes is incomplete.", ""])
    if "source_v4_logical_state_comparison" in result:
        state = result["source_v4_logical_state_comparison"]
        reduction = result["source_v4_reduction_comparison"]
        lines.extend([f"Logical-cache readers/writers: {state['matched']}/{state['total']}; TP reduction shape/dtype/per-card payload and output storage: {reduction['matched']}/{reduction['total']}. Physical pages and actual provider remain independently unaccepted.", ""])
    if "source_v4_wrapper_storage_comparison" in result:
        storage = result["source_v4_wrapper_storage_comparison"]
        lines.extend([f"FP8/mHC wrapper allocations and lifetimes: {storage['matched']}/{storage['total']} matched. Checks source shape/dtype/bytes, independent backing, returned objects and temporaries, quantization before matrix-output allocation, and liveness during calls. Uses graph records without profile hooks; actual-provider internal workspace remains independently unaccepted.", ""])
    if "source_v4_terminal_storage_comparison" in result:
        terminal = result["source_v4_terminal_storage_comparison"]
        decode = result["source_v4_decode_index_storage_comparison"]
        lines.extend([f"Embedding/final norm/TP vocabulary and greedy-chain storage: {terminal['matched']}/{terminal['total']}; two C4 decode index temporaries: {decode['matched']}/{decode['total']}. Final norm uses the explicitly declared vllm_c CUDA source-storage program; native IR reference and actual-provider identities remain separate.", ""])
    if "source_v4_expert_storage_comparison" in result:
        expert = result["source_v4_expert_storage_comparison"]
        lines.extend([f"Marlin expert allocation/alias/lifetime: {expert['matched']}/{expert['total']}. Covers independent sorting buffers, cumsum, locks, two sequential FP32 scratch allocations, and the shared BF16 arena with overlapping views. Allocation capacity does not establish FP32 reduction traffic or normal worker capacity.", ""])
    if "source_v4_decode_selector_storage_comparison" in result:
        selector = result["source_v4_decode_selector_storage_comparison"]
        lines.extend([f"CUDA/auto decode top-k shared workspace and FP32 score logical shape: {selector['matched']}/{selector['total']}. Cooperative/persistent branches both allocate a shared 1MiB arena; neither unknown score stride nor actual backend is selected. Physical score backing and private scratch remain unbound.", ""])
    if "source_v4_sparse_prefill_storage_comparison" in result:
        sparse = result["source_v4_sparse_prefill_storage_comparison"]
        lines.extend([f"Sparse prefill original chunk/typed-port/alias program: {sparse['matched']}/{sparse['total']}. Checks shared backing, 256B offsets, active rows/flattened views, expansion overlap, and last-chunk liveness. FlashMLA statistics outputs/planner storage remain separately unbound.", ""])
    if "source_v4_sparse_decode_views_comparison" in result:
        sparse = result["source_v4_sparse_decode_views_comparison"]
        lines.extend([f"Sparse decode Q/output and C4/C128 index callee views: {sparse['matched']}/{sparse['total']}. Checks singleton shapes, dtype, stride, and parent aliases; physical paged-cache axes and missing FlashMLA statistics/planner remain unaccepted.", ""])
    if "source_v4_mixed_projection_comparison" in result:
        comparison = result["source_v4_mixed_projection_comparison"]
        lines.extend([f"V4 mixed-output projection workload: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original torch.mm retains BF16 A/B and FP32 C. Checks constructor geometry, dtype, 2MNK mathematical FLOPs, A+B logical reads, and C writes. Input projection, Q/KV norm, prepare, and output ports are also in caller comparisons above. These checks do not accept sparse/cache, quantization/library internals, or physical traffic.", ""])
    if "source_v4_compressor_dependency_comparison" in result:
        comparison = result["source_v4_compressor_dependency_comparison"]
        lines.extend([f"V4 compressor state/intermediate dependencies: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Expected edges derive independently from original compressor forward, C4/C128 dispatch, and FP32 scratch-allocation read/write ports; checks actual graph data edges. Kernel mutation projects written logical storage as output. Physical page layout, finish math, and other consumers remain unaccepted by these dependency checks.", ""])
    if "source_v4_dependency_comparison" in result:
        comparison = result["source_v4_dependency_comparison"]
        lines.extend([f"V4 caller storage dependencies: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. RAW/WAR/WAW dependencies derive independently from serial tensor ports in original source and are checked against actual graph data paths. Shape/dtype/storage/stride/offset are compared at the same call boundary. This does not accept custom-op internals or cross-step cache behavior.", ""])
    if "source_v4_partial_state_comparison" in result:
        comparison = result["source_v4_partial_state_comparison"]
        lines.extend([f"V4 save-partial source workload/layout: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original kernels execute on payload-free lane/pointer metadata, independently recording FP32 additions, masked reads/writes, and state columns. Original wrappers supply stride; split preserves offset/alias. These are compared with actual graph operand layout and owner objects/resource inputs. Numerical locations/slots are unspecified; integer logical counts do not represent actual instructions/latency or physical DRAM observations.", ""])
    if "source_v4_finish_comparison" in result:
        comparison = result["source_v4_finish_comparison"]
        lines.extend([f"V4 finish useful-value workload: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. C4 online, C128 two-pass/merge, and indexer softmax/norm/RoPE/quant expressions independently read source. Shape metadata counts useful-value FLOPs/SFU and data payload, checking FP32 state and full scratch-allocation shapes. Uses minimal mathematical reduction, without verifying duplicate/masked instructions or floating-point numerical equivalence. Data payload comparison subtracts metadata bytes from the actual side only; metadata traffic remains unaccepted.", ""])
    if "source_v4_ffn_comparison" in result:
        comparison = result["source_v4_ffn_comparison"]
        lines.extend([f"V4 FFN workload and expert host: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Independently executes original MoE/Runner/serial SharedExperts, shared MLP, Marlin apply, and inner/outer wrappers. Checks hash-layer gate, FP32 logits, clamp, late TP reduction, cache exchange/reuse, and W13 → activation → weighted W2 → top-k sum. Expert checks cover useful math, typed reads/writes, BF16 weight multiply, FP32 zero-seeded summation, source views, and production serial-stage templates. Unpriced BF16 conversion/instruction service remains; actual routing, conditional FP32 scratch traffic, complete dequantization, and workspace/provider remain unaccepted.", ""])
    if "source_v4_fp8_dataflow_comparison" in result:
        comparison = result["source_v4_fp8_dataflow_comparison"]
        lines.extend([f"V4 shared FP8 logical read/write chain: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original apply_weights, QuantFP8.forward_cuda, per_token_group_quant_fp8, and weight-scale construction execute on metadata. Checks BF16 input, FP8 intermediates, FP32 scales, and logical payload consumed by matmul; normalizes in-call/separate wrappers and rejects missing or duplicate quantization. The conditional program neither binds actual provider nor accepts private layout/alias/packing, quantized values, or full workspace.", ""])
    if "source_v4_quantizer_comparison" in result:
        comparison = result["source_v4_quantizer_comparison"]
        lines.extend([f"V4 conditional quantization program: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Independently binds local vLLM and actual SDK source to check in-call scale_1x128 and external CUDA/Triton arithmetic/typed actions/scale layouts. Amax-zero execution count remains unknown; external rounded/unrounded branches both remain, without assuming numerical equivalence or selecting actual provider.", ""])
    if "source_v4_prepare_comparison" in result:
        comparison = result["source_v4_prepare_comparison"]
        lines.extend([f"V4 embedding/short index/KV insertion/indexer-Q: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Reuses independent source programs to check typed payload, comparisons/conversions, and conditional backend math. KV floating-point operations are not INT32 ALU; cache addresses remain parameterized.", ""])
    if "source_v4_ffn_vector_comparison" in result:
        comparison = result["source_v4_ffn_vector_comparison"]
        lines.extend([f"V4 shared/routed activation and shared-add: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Reuses independent C++ scalar/packed expression references to check clamp parameters, ordinary floating-point work, exp, typed negate/minmax, and logical reads/writes. Shared-add checks addition with two BF16 inputs and one BF16 output. Vector payload corresponds to owner L2 loads/stores; cache-adjusted DRAM traffic cannot substitute for it. Specific instructions, rounded values, and provider remain unaccepted.", ""])
    if "source_v4_output_comparison" in result:
        comparison = result["source_v4_output_comparison"]
        lines.extend([f"V4 output projection: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original inverse-RoPE host, grouped einsum, and RowParallelLinear control flow execute on metadata. Checks FP8/FP32 scale views, interleaved outputs, matrix geometry, and useful FP32 quantization divisions. Floating-point comparisons/abs/casts are not INT32 ALU. Source eager rotation sites are listed separately; compiler issue, actual provider, and numerical equivalence remain unaccepted.", ""])
    if "source_v4_indexer_comparison" in result:
        comparison = result["source_v4_indexer_comparison"]
        lines.extend([f"V4 scored indexer: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Independently binds upstream weighted-ReLU math and wrapper order, checking useful workload and typed payload for clear/gather/FP32 score/select. Score backing, refinement, and provider remain unaccepted.", ""])
    if "source_v4_attention_math_comparison" in result:
        comparison = result["source_v4_attention_math_comparison"]
        lines.extend([f"V4 attention math: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Checks scaled QK, mathematical softmax with one learned sink per row, and PV excluding the sink. Uses explicit stable normalized-exponential decomposition; floating-point max is not INT32 ALU. This does not accept FlashMLA online/split algorithms, actual instructions, or numerical equivalence.", ""])
    if "source_v4_mhc_comparison" in result:
        comparison = result["source_v4_mhc_comparison"]
        lines.extend([f"V4 mHC: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original broadcast/fused/post/head wrappers and split integer program execute on metadata, independently comparing FP32 partials, workload, typed payload, and before/after dependencies by stage. Actual provider and full scratch lifetimes remain unaccepted.", ""])
    if "source_v4_qkv_norm_comparison" in result:
        comparison = result["source_v4_qkv_norm_comparison"]
        lines.extend([f"V4 fused Q/KV norm: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original wrappers and FP32 expressions independently check epsilon, workload, conversions, and logical reads/writes. BF16/FP32 conversions are not priced as INT32 ALU.", ""])
    if "source_v4_target_comparison" in result:
        comparison = result["source_v4_target_comparison"]
        lines.extend([f"V4 final output chain: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original compute_logits, LogitsProcessor, communicator layout, and Sampler execute on metadata, checking sharded matrices, typed reads/writes, and dependencies. Argmax values, actual collective provider, and complete scratch remain unaccepted.", ""])
    if "source_v4_final_norm_comparison" in result:
        comparison = result["source_v4_final_norm_comparison"]
        lines.extend([f"V4 final normalization: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original IR RMS reference performs metadata operations only, checking ideal math and BF16 logical input/output. Reference rounding/temporary requests are retained, without selecting actual CUDA implementation or claiming numerical equivalence.", ""])
    if "source_v4_argmax_comparison" in result:
        comparison = result["source_v4_argmax_comparison"]
        lines.extend([f"V4 argmax source semantics: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Explicit Torch2.11 headers bind first-index rules for NaN/ties/signed zero, INT64 pair-branch ranges, and INT32 valid-index conversion. Actual provider is not selected; no comparison/index instruction pricing is added.", ""])
    if "source_v4_routing_objects_comparison" in result:
        comparison = result["source_v4_routing_objects_comparison"]
        lines.extend([f"V4 routing wrapper objects: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Original hash/Triton wrappers execute metadata allocations/returns, checking FP32/INT32 outputs and lifetimes of unused hash temporaries. Routing values, kernel math, and actual provider remain unaccepted.", ""])
    if "source_v4_routing_math_comparison" in result:
        comparison = result["source_v4_routing_math_comparison"]
        lines.extend([f"V4 routing math/reads/writes: {comparison['matched']}/{comparison['total']} matched, {comparison['mismatched']} mismatched. Complete ordinary-kernel source executes on metadata registers; active hash source branches are checked independently. Retains NaN conditional scope, typed comparisons, and source-role bytes, without selecting actual expert IDs/weights or provider.", ""])
    if "source_indexer_gather_workspace_comparison" in result:
        comparison = result["source_indexer_gather_workspace_comparison"]
        if comparison["total"]:
            lines.extend([f"Indexer gather storage, independent source comparison: {comparison['total']} calls, {comparison['matched']} matched, {comparison['mismatched']} mismatched, {comparison['not_checked']} unchecked. Expected dimensions directly read upstream helper expressions; checks FP8/UINT8 ports and FP32 scale aliases. External score backing and selector storage are excluded; this does not accept full-model alignment.", ""])
        else:
            lines.extend(["This scenario has no scored prefill calls; indexer gather storage comparison is inapplicable. A zero denominator cannot establish acceptance.", ""])
    lines.extend(["Remaining boundaries:", "", *("- " + item for item in result["limitations"]), "",
        "Request simulation, normal capacity, and comparable startup-log 10% checks are accepted separately; fully priced static graphs do not replace those checks.", ""])
    if "interactive_report" in result and "path" in result["interactive_report"]:
        lines.extend(["Offline overview in the same style: [HTML](" + result["interactive_report"]["path"] + "); static experiments have no Serving timeline.", ""])
    output.write_text("\n".join(lines))


def write_symbolic_serving_report(result_path: Path, output: Path, *, interactive_report=None):
    """Render a completed virtual-forward CLI result without rerunning it."""
    if output.suffix != ".md" or output.resolve() == result_path.resolve():
        raise ValueError("Serving summary needs a separate Markdown output")
    result = json.loads(result_path.read_text())
    if interactive_report is not None:
        result["interactive_report"] = interactive_report
    source = result["execution_provenance"]["runtime_inputs"]["symbolic_forward_source"]
    if (not isinstance(source, str) or not source or result["backend"] != "vllm"
            or result["prediction_method"] not in ("roofline", "midlayer")
            or result["main_chain_status"]["status"] != "executed"):
        raise ValueError("Serving summary requires an executed virtual-forward Roofline or Midlayer result")
    progress = result["scheduler_progress"]["summary"]
    counts = result["execution_artifact"]["counts"]
    if (progress["request_count"] <= 0 or progress["completed_requests"] != progress["request_count"]
            or counts["kernel_calls"] != result["num_kernel_calls"]
            or counts["simulation_steps"] != result["num_steps"]
            or counts["trace_transfers"] != result["num_transfers"]):
        raise ValueError("Serving result completion or aggregate counts disagree")
    runtimes = tuple(result["scheduler_runtime"].values())
    model_version = result["execution_provenance"]["runtime_inputs"]["model_execution_version"]
    source_versions = {
        "cuda_eager_inplace": ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_30_0"),
        "vllm_ascend_eager": ("vllm_0_18_0", "vllm_0_30_0"),
        "vllm_0_11_glm4_sync": ("vllm_0_11_0",),
        "vllm_0_30_v4_sync_fp8_ds_mla": ("vllm_0_30_0",),
        "vllm_0_30_v32_bf16_sync_explicit_allreduce": ("vllm_0_30_0",),
        "vllm_0_30_v32_fp8_sync_explicit_allreduce": ("vllm_0_30_0",),
        "vllm_0_30_k2_6_int4_sync_explicit_allreduce": ("vllm_0_30_0",),
        "vllm_0_27_1_deepseek_v3_sync_fp8": ("vllm_0_27_1",),
    }
    if source not in source_versions or model_version not in source_versions[source]:
        raise ValueError("Serving summary source and explicit model execution version disagree")
    scheduler_version = {"vllm_0_11_0": "0.11.0", "vllm_0_18_0": "0.11.0", "vllm_0_24_0": "0.11.0", "vllm_0_30_0": "0.30.0",
        "vllm_0_27_1": "0.30.0"}[model_version]
    if not runtimes or any(runtime["version"] != scheduler_version for runtime in runtimes):
        raise ValueError("summary scope requires matching CPU Schedulers without model or GPU execution")
    if scheduler_version == "0.30.0":
        if any(runtime["model_executed"] or runtime["weights_loaded"]
                or runtime["gpu_worker_processes_started"] != 0 for runtime in runtimes):
            raise ValueError("summary scope requires CPU Schedulers without model or GPU execution")
    elif any(runtime["platform"] != "cpu" or runtime["device_guard"] != "python_tripwire+cuda_state+procfs"
            or runtime["bootstrap"] != ("version_pinned_engine_args_mtp_cpu" if runtime["mtp_num_speculative_tokens"]
                else "version_pinned_scheduler_only_cpu") for runtime in runtimes):
        raise ValueError("summary scope requires guarded CPU0.11 bootstrap without model or GPU execution")
    repository = result["execution_provenance"]["repository"]
    pools = result["kv_capacity"]["scheduler_pools"]
    if (len({pool["pool_name"] for pool in pools}) != len(pools)
            or {pool["pool_name"] for pool in pools} != set(result["scheduler_runtime"])):
        raise ValueError("Serving summary capacity pools must match every Scheduler")
    placements = {pool["name"]: tuple(pool["ranks"]) for pool in result["deployment_plan"]["worker_pools"]}
    budget_rows, policy_rows = [], []
    rank_count = 0
    for pool in pools:
        pool_name, budgets = pool["pool_name"], pool["rank_budgets"]
        runtime = result["scheduler_runtime"][pool_name]
        if pool["runtime_memory_policy"] is None:
            if (scheduler_version != "0.11.0" or pool["source_mode"] != "debug_explicit_block_count"
                    or budgets or pool["num_kv_blocks"] != runtime["num_kv_blocks"]):
                raise ValueError("debug summary requires an explicit CPU0.11 test block budget without rank budgets")
            policy_rows.append(f"| {pool_name} | Not provided | Test block budget only, not normal capacity | {pool['num_kv_blocks']} |")
            budget_rows.extend(f"| {pool_name} / {rank} | Not computed | Not provided | Declared test blocks only |"
                for rank in placements[pool_name])
            rank_count += len(placements[pool_name])
            continue
        if (not budgets or len(budgets) != len(placements[pool_name])
                or {budget["rank"] for budget in budgets} != set(range(len(budgets)))):
            raise ValueError("Serving summary requires all pool-local rank memory budgets")
        policy = pool["runtime_memory_policy"]
        activation = next(item for item in policy["non_kv_reserves"] if item["kind"] == "activation")
        num_blocks = (runtime["native_allocation"]["num_kv_blocks"] if scheduler_version == "0.30.0"
            else runtime["num_kv_blocks"])
        policy_rows.append(f"| {pool_name} | {activation['provenance']['source_kind']} | {policy['evidence_status']} | {num_blocks} |")
        budget_rows.extend(f"| {pool_name} / {placements[pool_name][budget['rank']]} | {budget['model_weight_resident_bytes']} | {budget['non_kv_reserve_bytes']} | {budget['available_kv_bytes']} |"
            for budget in budgets)
        rank_count += len(budgets)
    lines = ["# Virtual model full-request simulation", "",
        f"Original results: [CLI JSON]({result_path.resolve()}). Repository commit observed at run end: `{repository['git_commit']}`, state: `{repository['worktree_state']}`. This does not bind the exact snapshot of imported code or prove that source remained frozen during execution.", "",
        "| Item | Result |", "|---|---|",
        f"| Model architecture / dtype | {result['model_config_provenance']['architecture']} / {result['model_compute_dtype']} |",
        f"| Request source | {result['request_source']['source_path'] if 'request_source' in result else 'workload hash: ' + result['workload_hash']} |",
        f"| Completed requests | {progress['completed_requests']} / {progress['request_count']} |",
        f"| Prefill tokens / scheduled request rows | {progress['prefill_tokens']} / {progress['scheduled_request_rows']} |",
        f"| Scheduling steps / operator calls | {result['num_steps']} / {result['num_kernel_calls']} |",
        f"| Predicted completion time | {result['schedule_end_time_s']:.6f} s |",
        f"| Predicted request / output-token throughput | {result['metrics']['request_throughput_per_s']:.3f} req/s / {result['metrics']['output_token_throughput_per_s']:.3f} token/s |",
        f"| CPU simulation time / peak RAM | {result['simulation_wall_time_s']:.3f} s / {result['process_peak_rss_bytes'] / 10**9:.3f} GB |",
        f"| Declared forward branch | {source} |",
        f"| Declared model source version | {model_version} |",
        f"| Scheduler pools / physical ranks | {len(pools)} / {rank_count} |",
        f"| Total deployed physical ranks (including FFN pool) | {sum(len(ranks) for ranks in placements.values())} |",
        f"| Scheduling-trajectory transfers (including AF activations) | {result['num_transfers']} |",
        f"| Host KV swap-in/out transfers | {counts['modeled_kv_offload_transfers']} |", "",
        "| pool | Activation source | Policy evidence level | Scheduler configured blocks |", "|---|---|---|---|",
        *policy_rows, "",
        "| pool / physical rank | Weights B | Non-KV reserve B | Available KV B |", "|---|---|---|---|",
        *budget_rows, "",
        f"The request stream uses a real vLLM{scheduler_version} CPU Scheduler, full virtual forward, existing lowering, {result['prediction_method']} prediction, and global timeline. Every step must cover all calls in its KernelPlan. Streaming execution retains counts and result-content digests after coverage validation; this complete execution succeeded. Simulation creates no real model Tensor, loads no weights, performs no numerical forward, and generates no real model tokens.", "",
        "Capacity follows the runtime memory policy explicitly listed in JSON or an explicit test block budget; activation source and policy evidence level are listed above. Test block counts do not prove normal model fit in device memory; explicit source policies do not become observed point estimates. Throughput and completion time predict the declared source scenario; normal capacity, actual device/provider, and performance accuracy remain unaccepted.", "",
        "Remaining work: complete startup storage, persistent state, workspace, and 10% log comparisons for comparable execution settings, then accept the production activation budget; extend unsupported models and modes.", ""]
    html_disabled = ("interactive_report" in result and "status" in result["interactive_report"]
        and result["interactive_report"]["status"] == "disabled")
    if not html_disabled and ("interactive_report" not in result or "path" not in result["interactive_report"]
            or not Path(result["interactive_report"]["path"]).is_file()):
        result["interactive_report"] = _regenerate_saved_serving_report(result, result_path, output.with_suffix(".html"))
    if "interactive_report" in result and "path" in result["interactive_report"]:
        lines.extend(["Offline interactive report: [Timeline HTML](" + result["interactive_report"]["path"] + ") · [Reproducible data](" +
            result["interactive_report"]["data_path"] + ")" + (" · [Full cost receipt](" + result["interactive_report"]["cost_receipt_path"] + ")" if "cost_receipt_path" in result["interactive_report"] else ""), ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    if result["deployment"] in ("af_disagg", "pd_af_disagg"):
        lines.extend(["AF transfers use the existing attention/FFN service contract; no actual AF runtime/provider executes here. FFN-pool storage from the same forward is listed separately; attention Scheduler KV budgets do not establish full FFN capacity.", ""])
    if counts["modeled_kv_offload_transfers"]:
        lines.extend(["Host KV transfers follow explicit lifecycle inputs, price full blocks per rank, and wait for completion. Synthetic lifecycles do not represent actual checkpoint/offload-provider behavior. Existing tier managers manage persistent Host/HBM pages; undeclared DMA workspace remains unknown.", ""])
    if scheduler_version == "0.11.0" and any(runtime["mtp_num_speculative_tokens"] for runtime in runtimes):
        lines.extend(["MTP includes the complete source chain for target selection, greedy rejection, and padded proposer. Accepted counts come from explicit JSON scenarios; token feedback comes from Scheduler simulation rather than real model generation or GPU execution.", ""])
    if "source_activation_memory" in result:
        memory = result["source_activation_memory"]
        lines.extend(["Storage statistics from the same virtual forward:", "",
            f"Reported steps: {memory['reported_steps']}/{memory['total_simulation_steps']}. Statistics come from the same forward without reexecuting the model. Saved external-request results retain only pool/rank peaks, omitting step details.", "",
            "| pool / rank | Represented activation peak B |", "|---|---|",
            *(f"| {pool_name} / {rank} | {record['known_activation_peak_bytes']} |"
                for pool_name, pool_memory in memory["pools"].items() for rank, record in pool_memory["ranks"].items()), "",
            "These independent source snapshots neither establish nor bound the complete worker peak. Missing workspace/runtime history remains unknown; normal capacity budgets are not replaced.", ""])
    output.write_text("\n".join(lines))
    return {"completed_requests": progress["completed_requests"], "kernel_calls": result["num_kernel_calls"],
        "normal_capacity_accepted": scheduler_version == "0.30.0" and all(runtime["normal_capacity_accepted"] for runtime in runtimes),
        "runtime_acceptance": scheduler_version == "0.30.0" and all(runtime["runtime_acceptance"] for runtime in runtimes)}


def _captured_executor_backend(case, *, h200_v030):
    """Bind the captured runtime's discrete executor choice, not its times."""
    manifest = case.manifest
    hardware = manifest["identity"]["hardware_id"]
    version = manifest["runtime"]["runtime_version"]
    ascend = hardware == "ascend_910b" and version == "0.18.0"
    h200 = hardware == "nvidia_h200" and version == "0.30.0" and h200_v030
    if not (ascend or h200):
        return None
    executor_class = manifest["runtime"]["scheduler_pipeline_depth_evidence"]["executor_class"]
    classes = {
        "vllm.v1.executor.uniproc_executor.UniProcExecutor": "uni",
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor": "mp",
    }
    if executor_class not in classes:
        raise ValueError(f"unsupported captured vLLM executor class: {executor_class!r}")
    return classes[executor_class]


def _validate_virtual_execution_workers(execution_workers):
    if (isinstance(execution_workers, bool)
            or not isinstance(execution_workers, int) or execution_workers < 0):
        raise ValueError(
            "virtual Serving execution_workers must be a non-negative integer; "
            "0 selects the CPU-aware default"
        )


def prepare_virtual_serving_case(root, config, cell, case, directory, sdk_python, source_root,
        *, serving_timing_mode="device_only", execution_mode="accelerated", state_audit=False,
        execution_workers=1, h200_scenario=None):
    """Keep the official execution contract and require a migrated source."""
    from . import sharegpt_compact_evaluation as compact
    from evaluation.serving.simulation.simulation_runner import _parse_args

    from .compact_run_environment import (
        resolve_host_launch_profile_paths, resolve_scheduler_host_profile_paths,
        case_host_profile_paths, host_profile_variant_hardware)
    from src.serving.host_timing import validate_serving_timing_mode

    validate_serving_timing_mode(serving_timing_mode)
    if h200_scenario not in (None, "analytical_bf16_v1"):
        raise ValueError("unknown H200 Serving scenario")
    launch_path = scheduler_path = None
    hardware_id = cell.hardware.hardware_id
    if serving_timing_mode != "device_only":
        if hardware_id in host_profile_variant_hardware():
            profiles = case_host_profile_paths(root, hardware_id, case.manifest["runtime"]["runtime_version"])
            launch_path = profiles["host_launch"]
            if serving_timing_mode == "device_plus_launch_plus_scheduler":
                scheduler_path = profiles["scheduler_host"]
        else:
            launch_profiles = resolve_host_launch_profile_paths(root, (), {hardware_id})
            if launch_profiles is None or hardware_id not in launch_profiles:
                raise NotImplementedError(f"no canonical Host launch profile for {hardware_id}")
            launch_path = launch_profiles[hardware_id]
            if serving_timing_mode == "device_plus_launch_plus_scheduler":
                scheduler_profiles = resolve_scheduler_host_profile_paths(root, (), {hardware_id})
                if scheduler_profiles is None or hardware_id not in scheduler_profiles:
                    raise NotImplementedError(f"no canonical Scheduler Host profile for {hardware_id}")
                scheduler_path = scheduler_profiles[hardware_id]
    argv = compact.build_simulation_argv(root, config, cell, case,
        execution_mode=execution_mode, serving_timing_mode=serving_timing_mode,
        host_launch_profile_path=launch_path, scheduler_host_profile_path=scheduler_path)
    executor_backend = _captured_executor_backend(case,
        h200_v030=config.prediction.values["scheduler_version"] == "0.30.0")
    if executor_backend is not None:
        argv += ["--vllm-executor-backend", executor_backend]
    model_path = compact._model_config_path(root, cell.model.simulation_model_path)
    model = load_model_config(model_path)
    prediction = config.prediction.values
    policy_path = Path(compact._expand_runtime_memory_policy_path(root, config, cell, cell.model.dataset_model_id))
    directory.mkdir(parents=True, exist_ok=True)
    requests = case.manifest["requests"]
    _validate_virtual_execution_workers(execution_workers)
    # Emit the declared value verbatim: 0 keeps the CLI's unified "auto"
    # semantics so the child both normalizes its pool and installs the
    # adaptive coordinator callback; positive values stay manual overrides.
    argv += [
        "--canonical-requests-sha256", requests["sha256"],
        "--execution-workers", str(execution_workers),
        "--step-retention", "streaming_aggregate", "--output", str(directory / "result.json"),
    ]
    if type(state_audit) is not bool:
        raise TypeError("virtual Serving state_audit must be boolean")
    if state_audit:
        argv += ["--state-audit-output", str(directory / "state_audit.json")]
    explicit_model_execution_version = None
    if prediction["scheduler_version"] == "0.30.0":
        # The existing ordinary H200 async source is an explicit CLI domain;
        # the synchronous default-source normalizer cannot select it. Preserve
        # the case's graph/depth/cache settings and let the CLI enforce them.
        from src.serving.workload.model_config import is_validated_qwen3_moe_ep2_model_scope
        ordinary_dense = (model.model_type in ("qwen3", "llama") and model.num_experts == 0
            and prediction["tensor_parallel_size"] in (1, 2) and prediction["expert_parallel_size"] == 1)
        ordinary_moe = (is_validated_qwen3_moe_ep2_model_scope(model)
            and prediction["tensor_parallel_size"] == prediction["expert_parallel_size"] == 2)
        if (hardware_id == "nvidia_h200" and (ordinary_dense or ordinary_moe)
                and model.compute_dtype == model.weight_dtype == "bf16"
                and prediction["pipeline_parallel_size"] == 1
                and prediction["mtp_num_speculative_tokens"] == 0):
            if prediction["enable_prefix_caching"]:
                # Canonical validation binds observed or independently proved
                # scheduler-equivalent counts. Unknown hits remain None and
                # must never be promoted to a zero-hit request stream.
                if any(request.cached_hit_tokens != 0 for request in case.requests):
                    raise NotImplementedError("ordinary H2000.30 source requires canonical zero-hit evidence for every request")
                argv += ["--prefix-cache-mode", "trace_no_hit"]
            argv += ["--symbolic-forward-source", "cuda_eager_inplace",
                "--target-sampling-source", "vllm_0_30_plain_greedy_bf16"]
            explicit_model_execution_version = "vllm_0_30_0"
        if sdk_python is None or source_root is None:
            raise NotImplementedError("the original0.30 Scheduler contract requires explicit genuine SDK/source inputs")
        argv += ["--vllm-sdk-python", str(sdk_python), "--vllm-source-root", str(source_root),
            "--vllm-cache-layout", "BLHNC"]
    if h200_scenario is not None and hardware_id == "nvidia_h200":
        if explicit_model_execution_version != "vllm_0_30_0":
            raise NotImplementedError("H200 analytical BF16 scenario requires the ordinary 0.30 source")
        # The scenario is an explicit analytical input, not a change to the
        # captured case or a certification of its native provider/content.
        overrides = {"--launch-policy": "analytical_launch",
            "--norm-source-batch-invariant": "disabled", "--bf16-gemm-model": "tiled_reference"}
        if model.model_type == "qwen3_moe":
            overrides["--bf16-expert-model"] = "triton_reference"
        if serving_timing_mode != "device_only":
            overrides.update({"--host-launch-profile": str(root / "configs/host_timing/host_launch_profiles/nvidia_h200_vllm030.json"),
                "--host-request-path-profile": str(root / "configs/host_timing/host_request_path_profiles/nvidia_h200_vllm030.json"),
                "--host-runtime-prepare-profile": str(root / "configs/host_timing/host_runtime_prepare_profiles/nvidia_h200_vllm030_v2_state.json")})
            if serving_timing_mode == "device_plus_launch_plus_scheduler":
                overrides["--scheduler-host-profile"] = str(root / "configs/host_timing/scheduler_host_profiles/nvidia_h200_vllm030_native_cpu.json")
        if prediction["tensor_parallel_size"] == 2:
            overrides.update({"--collective-backend": "analytical_roofline_v1",
                "--dense-source-input-policy": "declared_zero_tp2"})
            for flag in ("--tensor-all-reduce-profile-sha256", "--tensor-all-gather-profile-sha256"):
                if flag in argv:
                    index = argv.index(flag)
                    del argv[index:index + 2]
        for flag, value in overrides.items():
            if flag in argv:
                argv[argv.index(flag) + 1] = value
            else:
                argv += [flag, value]
    try:
        parsed = _parse_args(argv)
    except SystemExit as error:
        raise NotImplementedError(
            f"prediction Scheduler {prediction['scheduler_version']} source CLI rejects "
            f"the declared {serving_timing_mode} case contract; no legacy execution") from error
    source_profile = parsed.symbolic_forward_source
    routing = compact._expected_expert_routing(root, cell)
    contract = {"workload_path": "virtual_tensor", "prediction_method": parsed.prediction_method,
        "execution_mode": parsed.execution_mode, "launch_policy": parsed.launch_policy,
        "rotary_source_program": vllm_rope_program(cell.hardware.hardware_id, model.rope_scaling_type),
        "hardware_config_sha256": hashlib.sha256((root / cell.hardware.hardware_config_path).read_bytes()).hexdigest(),
        "captured_hardware_config_sha256": cell.hardware.hardware_config_sha256,
        "symbolic_forward_source": source_profile,
        "model_execution_version": (explicit_model_execution_version
            if explicit_model_execution_version is not None else
            None if source_profile is None else
            parsed.model_source_normalization["model_execution_version"]),
        "target_sampling_source": parsed.target_sampling_source,
        "collective_backend": parsed.collective_backend,
        "scheduler_version": parsed.vllm_scheduler_version,
        "scheduler_source_commit": prediction["scheduler_source_commit"],
        "captured_runtime_version": case.manifest["runtime"]["runtime_version"],
        "serving_timing_mode": parsed.serving_timing_mode,
        "graph_mode": parsed.graph_mode, "graph_capture_sizes": list(parsed.graph_execution_config.capture_sizes),
        "graph_piecewise_eager_workload_kinds": list(parsed.graph_execution_config.piecewise_eager_workload_kinds),
        "graph_uniform_decode_query_len": parsed.graph_uniform_decode_query_len,
        "scheduler_pipeline_depth": parsed.scheduler_pipeline_depth, "async_scheduling": parsed.vllm_async_scheduling,
        "vllm_executor_backend": parsed.vllm_executor_backend,
        "enable_prefix_caching": parsed.enable_prefix_caching, "enable_chunked_prefill": parsed.enable_chunked_prefill,
        "tensor_parallel_size": parsed.prefill_tp_size, "pipeline_parallel_size": parsed.prefill_pp_size,
        "expert_parallel_size": prediction["expert_parallel_size"], "mtp_num_speculative_tokens": parsed.mtp_num_speculative_tokens,
        "expert_routing": routing.manifest(),
        "reserve_policy_source": str(policy_path), "reserve_policy_effective": str(policy_path),
        "reserve_bytes_changed": False, "normal_capacity_accepted": False,
        "source_selection_status": "unsupported" if source_profile is None else "selected",
        "source_selection_reason": ("no migrated shared forward for the original method/launch/graph/Scheduler contract; legacy fallback is forbidden"
            if source_profile is None else None)}
    if h200_scenario is not None and hardware_id == "nvidia_h200":
        contract["h200_serving_scenario"] = h200_scenario
        contract["norm_source_batch_invariant"] = parsed.norm_source_batch_invariant
        contract["bf16_gemm_model"] = parsed.bf16_gemm_model
        contract["dense_source_input_policy"] = parsed.dense_source_input_policy
        contract["scenario_scope"] = ("analytical launch; Norm default explicitly disabled; tiled BF16 reference; "
            "MoE Triton expert reference; explicit vLLM0.30 native-CPU Scheduler Host and V2-state prepare profiles; "
            "TP2 analytical collectives and declared zero unknown IDs/graph tails with approximate embedding locality; "
            "known backend IDs retained; no observed content or native provider certification")
    return argv, contract


def compare_virtual_serving_aggregate(observed, result, result_path, request_count):
    """Compare saved anonymous distributions without inventing request pairs."""
    from . import sharegpt_compact_evaluation as compact
    from .production_ground_truth_comparison import _simulation_distribution
    metrics = result["metrics"]
    latencies = {}
    for measured, predicted in compact._LATENCY_FIELDS.items():
        distribution = _simulation_distribution(metrics["latency_distributions"][predicted],
            name=predicted, request_count=request_count, path=result_path)
        if distribution["eligible_request_count"] != request_count:
            raise NotImplementedError("virtual full-stream comparison requires matching eligible request populations")
        statistics = {"mean": distribution["mean_ms"],
            **{f"p{percentile}": distribution["percentiles_ms"][percentile] for percentile in (50, 90, 99)}}
        latencies[measured] = {name: compact._comparison(observed["latencies"][measured][name], value)
            for name, value in statistics.items()}
    return {"comparison_kind": "measured_vs_predicted_anonymous_distributions",
        "latency": latencies, "throughput": {measured: compact._comparison(
            observed["throughputs"][measured], metrics[predicted])
            for measured, predicted in compact._THROUGHPUT_FIELDS.items()},
        "paired_request_comparison": "not_available_saved_result_omits_request_mapping"}


def run_paired_virtual_case(argv, case, comparison_output):
    """Pair canonical indexes while the producer still holds request identities.

    The simulation result keeps its request-safe aggregate projection. Only the
    existing evaluation comparison contract crosses this separate boundary.
    """
    from evaluation.serving.simulation.simulation_runner import run_cli_args
    from . import sharegpt_compact_evaluation as compact

    executor_backend = _captured_executor_backend(case,
        h200_v030=("--vllm-scheduler-version" in argv
            and argv[argv.index("--vllm-scheduler-version") + 1] == "0.30.0"))
    if executor_backend is not None:
        if (argv.count("--vllm-executor-backend") != 1
                or argv[argv.index("--vllm-executor-backend") + 1] != executor_backend):
            raise ValueError("virtual case executor backend differs from captured runtime")

    if argv.count("--output") != 1:
        raise ValueError("paired virtual case requires exactly one persisted simulation output")
    result_path = Path(argv[argv.index("--output") + 1])
    payload = run_cli_args(argv)
    expected_sha = case.manifest["requests"]["sha256"]
    if (payload["request_source"]["kind"] != "canonical_requests"
            or payload["request_source"]["source_sha256"] != expected_sha):
        raise ValueError("virtual prediction request source differs from comparison input")
    metrics = compact._compare_prediction(compact._observed_metrics(case), payload, len(case.requests))
    metrics.update(comparison_kind="measured_vs_predicted_by_request_idx",
                   paired_request_comparison="paired_before_anonymous_projection")
    record = {"request_source_sha256": expected_sha,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(), "metrics": metrics}
    Path(comparison_output).write_text(json.dumps(record, indent=2) + "\n")
    return record


def _load_virtual_case_comparison(path, result_path, expected_sha):
    record = json.loads(path.read_text())
    if (record["request_source_sha256"] != expected_sha
            or record["result_sha256"] != hashlib.sha256(result_path.read_bytes()).hexdigest()):
        raise ValueError("paired virtual comparison does not bind the saved result and canonical input")
    metrics = record["metrics"]
    if (metrics["comparison_kind"] != "measured_vs_predicted_by_request_idx"
            or metrics["paired_request_comparison"] != "paired_before_anonymous_projection"):
        raise ValueError("paired virtual comparison has an unknown identity contract")
    return metrics


_VIRTUAL_RELOCATABLE_INPUTS = frozenset({
    "--canonical-requests", "--model", "--hardware-config", "--runtime-memory-policy",
    "--host-launch-profile", "--scheduler-host-profile", "--host-request-path-profile",
    "--host-runtime-prepare-profile", "--expert-routing-profile-manifest",
})


def _snapshot_input_sha256(root, commit, relative):
    """Use retained input bytes, including LFS identities, rather than checkout location."""
    import subprocess
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root)
    if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        import re
        match = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize [0-9]+\n", blob)
        if match is None:
            raise ValueError("historical input has an invalid LFS identity")
        return match[1].decode()
    return hashlib.sha256(blob).hexdigest()


def _validate_virtual_reuse_command(saved, requested, root, snapshot_commit):
    """Check bounded relocation; None is historical-failure display without input equivalence."""
    if len(saved) != len(requested) or any(not isinstance(item, str) for item in (*saved, *requested)):
        raise ValueError("saved virtual Serving command differs from the requested execution")
    if (Path(saved[0]).resolve(strict=True) != Path(requested[0]).resolve(strict=True)
            or Path(saved[0]).parent.resolve(strict=True) != Path(requested[0]).parent.resolve(strict=True)):
        raise ValueError("saved virtual Serving Python environment differs")
    if saved.count("--canonical-requests") != 1 or requested.count("--canonical-requests") != 1:
        raise ValueError("reuse requires one canonical request binding")
    index = requested.index("--canonical-requests") + 1
    relative_requests = Path(requested[index]).relative_to(root)
    old_requests = Path(saved[index])
    if old_requests.parts[-len(relative_requests.parts):] != relative_requests.parts:
        raise ValueError("saved canonical request path refers to a different case")
    old_root = old_requests.parents[len(relative_requests.parts) - 1]
    output_index = requested.index("--output") + 1
    old_output, new_output = Path(saved[output_index]), Path(requested[output_index])
    if old_output.name != "result.json" or new_output.name != "result.json" or old_output.parent.name != new_output.parent.name:
        raise ValueError("saved output path refers to a different case")
    for index in range(1, len(requested)):
        flag = requested[index - 1]
        if flag in _VIRTUAL_RELOCATABLE_INPUTS or flag == "--case-path":
            relative = Path(requested[index]).relative_to(root)
            if Path(saved[index]) != old_root / relative:
                raise ValueError(f"saved virtual Serving input path differs: {flag}")
            bound = relative / "manifest.json" if flag == "--case-path" else relative
            if snapshot_commit is not None:
                current_sha = hashlib.sha256((root / bound).read_bytes()).hexdigest()
                if current_sha != _snapshot_input_sha256(root, snapshot_commit, bound):
                    raise ValueError(f"saved virtual Serving input content differs: {flag}")
        elif flag in ("--output", "--comparison-output"):
            filename = "result.json" if flag == "--output" else "comparison.json"
            if Path(saved[index]) != old_output.parent / filename or Path(requested[index]) != new_output.parent / filename:
                raise ValueError("saved virtual Serving output binding differs")
        elif saved[index] != requested[index]:
            raise ValueError("saved virtual Serving command differs from the requested execution")


def _virtual_reuse_execution_commit(previous_cell):
    """Bind input validation to the case's recorded execution, never its summary commit."""
    if "execution_provenance" not in previous_cell:
        if previous_cell["prediction_status"] == "failed":
            return None  # Historical failure display only; no input equivalence claim.
        raise ValueError("completed reuse requires per-case execution provenance")
    repository = previous_cell["execution_provenance"]["repository"]
    if (repository["worktree_state"] != "clean_reproducible_by_commit"
            or repository["changed_path_count"] != 0 or repository["git_status_short"] != []):
        raise ValueError("saved execution repository was not clean")
    if "identity_changed_between_boundaries" in repository and repository["identity_changed_between_boundaries"] is not False:
        raise ValueError("saved execution repository identity changed")
    if "run_start_identity" in repository:
        start = repository["run_start_identity"]
        for key in ("repository_root", "git_commit", "git_diff_hash", "status_porcelain_sha256",
                    "worktree_state", "changed_path_count", "git_status_short"):
            if start[key] != repository[key]:
                raise ValueError("saved execution repository entry/exit identities differ")
    return repository["git_commit"]



def _recorded_latency_mean(result, field):
    """Use recorded requests or an exact empirical CDF, never quantile guesses."""
    metrics = result["metrics"]
    if "requests" in metrics:
        values = [row[field] for row in metrics["requests"].values() if field in row and row[field] is not None]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("recorded latency values must be finite nonnegative seconds")
        return math.fsum(values) / len(values) if values else None
    if "latency_distributions" not in metrics or field not in metrics["latency_distributions"]:
        return None
    distribution = metrics["latency_distributions"][field]
    if distribution["availability"] == "unavailable_no_eligible_requests":
        return None
    if distribution["availability"] != "available":
        raise ValueError("unknown recorded latency availability")
    if "empirical_cdf" not in distribution:
        return None  # Old quantiles alone do not determine an exact mean.
    cdf = distribution["empirical_cdf"]
    values, probabilities = cdf["values_s"], cdf["cumulative_probabilities"]
    if not values or len(values) != len(probabilities):
        raise ValueError("recorded empirical CDF must contain aligned points")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("recorded CDF values must be finite nonnegative seconds")
    if list(values) != sorted(values):
        raise ValueError("recorded CDF values must be sorted")
    prior, weighted = 0.0, []
    for value, probability in zip(values, probabilities):
        if type(probability) not in (int, float) or not math.isfinite(probability) or not prior < probability <= 1:
            raise ValueError("recorded CDF probabilities must increase to one")
        weighted.append(value * (probability - prior))
        prior = probability
    if prior != 1:
        raise ValueError("recorded CDF probabilities must end at one")
    return math.fsum(weighted)

def _regenerate_saved_serving_report(result, result_path, output):
    from evaluation.reporting.serving_interactive import write_status_report, regenerate_report
    receipt = result["interactive_report"] if "interactive_report" in result else {}
    projection_path = Path(receipt["data_path"]) if "data_path" in receipt else result_path.with_suffix(".visualization.json")
    if not projection_path.is_file():
        projection_path = result_path.with_suffix(".visualization.json")
    if projection_path.is_file():
        return regenerate_report(json.loads(projection_path.read_text()), output)
    return write_status_report({**result,
        "main_chain_status": {"status": "partial", "reason": "Historical results retained no visualization projection; only recorded metrics/configuration are shown, and active intervals are unavailable."},
        "report_facts": [{"label": "Completed requests", "value": result["scheduler_progress"]["summary"]["completed_requests"]},
            {"label": "Total forwards", "value": result["num_steps"]},
            {"label": "Operator calls", "value": result["num_kernel_calls"]},
            {"label": "Mean time to first token TTFT (s)", "value": _recorded_latency_mean(result, "ttft_s")},
            {"label": "Mean time per output token TPOT (s)", "value": _recorded_latency_mean(result, "tpot_s")}]}, output)


def _adaptive_tail_workers_for(case_directory):
    """Resolve the late-tail inner-worker grant for this batch boundary.

    Called by the serving simulation at every parallel-tail boundary so a
    case that is already running can widen its inner pool when enough
    sibling cases have finished.  Returns None when the adaptive
    coordinator is not active for this run (manual override, reuse, or a
    single-case selection), letting the simulation fall back to the static
    CLI value.  The grant is a plain function of (host CPUs, selected-case
    count, currently-active case count) read from the shared run file; it
    never uses case identity, hardware name, or predicted/observed
    durations, and it never exceeds the host CPU headroom left by the
    outer one-worker-per-case pool.
    """
    state_path = Path(case_directory) / "adaptive_case_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text())
    spec_count = state["spec_count"]
    logical_cpus = state["logical_cpus"]
    active_cases = len(state["active_cases"])
    if active_cases <= 0 or spec_count <= 1:
        return None
    budget = max(0, logical_cpus - spec_count)
    if budget <= 0:
        return None
    # Promote as soon as every survivor can take at least one extra worker;
    # the spare budget is unused while cases run serially, so grant early.
    threshold = max(2, budget // 8)
    if active_cases > threshold:
        return None
    return max(2, min(16, budget // active_cases))


def _run_virtual_serving_case(spec, *, h200_scenario=None):
    from . import sharegpt_compact_evaluation as compact
    import subprocess
    execution_workers = 1
    adaptive_state = None
    adaptive_index = None
    adaptive_auto = False
    interactive_report = False
    if len(spec) == 11:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, previous_cell) = spec
        execution_mode = "accelerated"
        state_audit = False
    elif len(spec) == 12 and type(spec[11]) is bool:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, previous_cell, interactive_report) = spec
        execution_mode = "accelerated"
        state_audit = False
    elif len(spec) == 12:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, previous_cell) = spec
        state_audit = False
    elif len(spec) == 13:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, previous_cell) = spec
    elif len(spec) == 14:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, execution_workers, previous_cell) = spec
    elif len(spec) == 15:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, execution_workers, previous_cell,
         adaptive_state) = spec
    elif len(spec) == 16:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, execution_workers, previous_cell,
         adaptive_state, adaptive_index) = spec
    elif len(spec) == 17:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, execution_workers, previous_cell,
         adaptive_state, adaptive_index, adaptive_auto) = spec
    elif len(spec) == 18:
        (root, config, cell, case, directory, python, sdk, source, reuse_completed,
         serving_timing_mode, execution_mode, state_audit, execution_workers, previous_cell,
         adaptive_state, adaptive_index, adaptive_auto, interactive_report) = spec
    else:
        raise ValueError("virtual Serving case spec has an invalid arity")
    if execution_mode not in ("accelerated", "fast", "reference"):
        raise ValueError("virtual Serving execution_mode is invalid: " + repr(execution_mode))
    if any(type(flag) is not bool for flag in (state_audit, adaptive_auto, interactive_report)):
        raise ValueError("virtual Serving audit/adaptive/report flags must be boolean")
    if type(execution_workers) is not int or execution_workers < 0:
        raise ValueError("virtual Serving execution_workers must be a nonnegative integer")
    # The spec carries the run-level resolved inner-worker count.  A value
    # of 0 means "auto" (CPU-aware default for the host and selection size).
    if execution_workers == 0:
        execution_workers = min(4, (os.cpu_count() or 1) // 2)
    if adaptive_state is not None and not reuse_completed:
        # Seed this case's coordination record before the subprocess starts
        # so the serving simulation can re-read the current active-case
        # grant at every legal parallel-tail boundary.
        from evaluation.serving.metrics import symbolic_model as _self_module
        if "started_cases" in adaptive_state:
            seed_active = list(adaptive_state["started_cases"])
        elif "active_cases" in adaptive_state:
            seed_active = list(adaptive_state["active_cases"])
        else:
            raise ValueError("adaptive coordinator state is missing its case list")
        seed_state = {"spec_count": adaptive_state["spec_count"],
            "logical_cpus": adaptive_state["logical_cpus"],
            "active_cases": seed_active}
        _self_module._write_adaptive_case_state(directory, seed_state, adaptive_index)
    record = {"cell_id": cell.cell_id, "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id, "case_id": cell.case_id, "target_qps": cell.qps,
        "request_count": len(case.requests), "captured_prediction_contract": dict(config.prediction.values),
        "captured_runtime": dict(case.manifest["runtime"]), "workload_path": "virtual_tensor",
        "equivalence_accepted": False, "normal_capacity_accepted": False}
    interactive_enabled = bool(interactive_report)

    def status_report(status, reason):
        if not interactive_enabled:
            return {"status": "disabled", "reason": "offline HTML report not requested"}
        from evaluation.reporting.serving_interactive import write_status_report
        return write_status_report({"main_chain_status": {"status": status, "reason": reason},
            "model_config_provenance": {"model_id": cell.model.dataset_model_id},
            "hardware_ids": [cell.hardware.hardware_id], "request_source": {"case": record}}, directory / ("reuse-status.html" if reuse_completed else "result.html"))

    try:
        argv, contract = prepare_virtual_serving_case(root, config, cell, case, directory, sdk, source,
            serving_timing_mode=serving_timing_mode, execution_mode=execution_mode, state_audit=state_audit,
            execution_workers=0 if adaptive_auto else execution_workers,
            **({"h200_scenario": h200_scenario} if h200_scenario is not None else {}))
    except NotImplementedError as error:
        return {**record, "prediction_status": "unsupported", "prediction_reason": str(error),
            "interactive_report": status_report("unsupported", str(error))}
    if interactive_enabled:
        argv = [*argv, "--interactive-report"]
    legacy_command = [str(python), "scripts/run_serving_simulation.py", *argv]
    command = legacy_command
    if contract["symbolic_forward_source"] is None:
        return {**record, "effective_prediction_contract": contract, "command": command,
            "prediction_status": "unsupported", "prediction_reason": contract["source_selection_reason"],
            "subprocess_started": False, "interactive_report": status_report("unsupported", contract["source_selection_reason"])}
    comparison_path = directory / "comparison.json"
    command = [str(python), "scripts/evaluation/serving/run_paired_virtual_case.py",
        "--case-path", str(compact.compact_case_path(root, config, cell)),
        "--comparison-output", str(comparison_path), "--", *argv]
    started = perf_counter()
    if reuse_completed:
        saved_command = json.loads((directory / "command.json").read_text())
        if previous_cell is None or saved_command != previous_cell["command"]:
            raise ValueError("saved command is not bound to the previous campaign case")
        for key in ("cell_id", "request_count", "captured_prediction_contract", "captured_runtime"):
            if record[key] != previous_cell[key]:
                raise ValueError(f"saved virtual Serving case differs: {key}")
        requested_command = legacy_command if saved_command[1] == legacy_command[1] else command
        snapshot_commit = _virtual_reuse_execution_commit(previous_cell)
        _validate_virtual_reuse_command(saved_command, requested_command, root, snapshot_commit)
        command = saved_command
        status = previous_cell["prediction_status"]
        if status not in ("completed", "failed", "unsupported"):
            raise ValueError(f"unknown executed virtual Serving status: {status!r}")
        returncode = previous_cell["exit_code"]
        if type(returncode) is not int or (status == "completed") != (returncode == 0):
            raise ValueError("saved virtual Serving status and exit code disagree")
    else:
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        from evaluation.reporting.serving_interactive import default_report_path
        if interactive_enabled:
            report_override = argv[argv.index("--interactive-output") + 1] if "--interactive-output" in argv else None
            runtime_report_path = default_report_path(directory / "result.json", report_override)
            runtime_projection = runtime_report_path.with_suffix(".visualization.json")
        else:
            runtime_report_path = None
            runtime_projection = directory / "disabled.visualization.json"
        prior_projection_stamp = runtime_projection.stat().st_mtime_ns if runtime_projection.is_file() else None
        with (directory / "run.log").open("w") as log:
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = "0"
            completed = subprocess.run(command, cwd=root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        returncode = completed.returncode
    record.update(effective_prediction_contract=contract, command=command,
        evaluation_wall_time_seconds=perf_counter() - started, exit_code=returncode,
        reused_simulation_result=reuse_completed)
    if returncode != 0:
        if reuse_completed:
            return {**previous_cell, "reused_simulation_result": False, "reused_historical_failure": True,
                    "reuse_input_validation": ("historical_failure_only_inputs_not_revalidated"
                        if snapshot_commit is None else "matched_recorded_execution_commit"),
                    "report_regeneration_wall_time_seconds": perf_counter() - started}
        if runtime_projection.is_file() and runtime_projection.stat().st_mtime_ns != prior_projection_stamp:
            runtime_status = json.loads(runtime_projection.read_text())
            if (runtime_status["visualization_view"] == "status"
                    and runtime_status["main_chain_status"]["status"] in ("unsupported", "failed")):
                status = runtime_status["main_chain_status"]
                return {**record, "prediction_status": status["status"], "prediction_reason": status["reason"],
                    "interactive_report": ({"status": status["status"], "path": str(runtime_report_path), "data_path": str(runtime_projection)}
                        if interactive_enabled else {"status": "disabled", "reason": "offline HTML report not requested"})}
        return {**record, "prediction_status": "failed", "prediction_reason": str(directory / "run.log"),
            "interactive_report": status_report("failed", "Virtual Serving subprocess failed; see run.log")}
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    progress = result["scheduler_progress"]["summary"]
    if "execution_mode" not in result and not reuse_completed:
        raise ValueError("new virtual Serving result omitted its execution mode")
    if "execution_mode" in result and result["execution_mode"] != execution_mode:
        raise ValueError("virtual Serving result differs from the requested execution mode")
    if progress["request_count"] != len(case.requests):
        raise ValueError("virtual Serving did not execute the full selected request stream")
    if result["execution_provenance"]["runtime_inputs"]["symbolic_forward_source"] != contract["symbolic_forward_source"]:
        raise ValueError("virtual Serving forward source differs from the effective contract")
    if (result["request_source"]["source_sha256"] != cell.requests_sha256
            or result["model_config_provenance"]["config_sha256"] != cell.model.model_config_sha256
            or result["execution_provenance"]["hardware_configs"]["prefill"]["sha256"] != contract["hardware_config_sha256"]):
        raise ValueError("saved virtual Serving input identities differ from the selected case")
    legacy_result = command[1] == legacy_command[1]
    if legacy_result:
        comparison = compare_virtual_serving_aggregate(compact._observed_metrics(case), result, result_path, len(case.requests))
    else:
        comparison = _load_virtual_case_comparison(comparison_path, result_path, cell.requests_sha256)
    from .compact_case_evaluation import _observation_quality
    observation_quality = _observation_quality(case.manifest)
    accuracy_claim_status = (
        "provisional_observer_effect"
        if observation_quality["status"] == "provisional"
        else "source_projected_unvalidated")
    report_context = {}
    if not legacy_result:
        from .compact_case_evaluation import _cell_context
        from .compact_execution_payload_validation import _validate_midlayer_wall_time
        from .compact_metric_comparison import _validated_host_timing_payload
        from .scheduler_progress import validate_scheduler_progress_manifest
        context = _cell_context(config, cell, case, argv, {
            "alignment_stage_ran": False, "alignment_checked": False,
            "status": "not_checked_virtual_source", "formal_conclusion_allowed": False, "phases": {}})
        scheduler_progress = validate_scheduler_progress_manifest(result["scheduler_progress"])
        expected_axis = f"predicted_{serving_timing_mode}_virtual_time"
        if scheduler_progress["time_axis"] != expected_axis:
            raise ValueError("virtual Scheduler progress time axis differs from requested Host mode")
        if "ttft_breakdown" not in comparison:
            comparison["ttft_breakdown"] = None
        report_context = {**context, "serving_timing_mode": serving_timing_mode,
            "host_timing": _validated_host_timing_payload(result, serving_timing_mode, cell.hardware.hardware_id),
            "forward_provenance": {
                "validation_status": "source_projected_unvalidated",
                "scheduler_progress": scheduler_progress,
                "midlayer_wall_time": _validate_midlayer_wall_time(result),
                "execution_provenance": result["execution_provenance"]}}
    if reuse_completed:
        # A historical stage split used unrelated clock origins. Validate
        # all prediction metrics, then project that unusable split away in
        # memory; never rewrite the bound saved comparison/result files.
        saved_metrics = previous_cell["metrics"]
        checked_comparison = comparison
        if not legacy_result:
            saved_metrics = {key:value for key,value in saved_metrics.items()
                if key != "ttft_breakdown"}
            checked_comparison = {key:value for key,value in comparison.items()
                if key != "ttft_breakdown"}
        if (previous_cell["execution_provenance"] != result["execution_provenance"]
                or previous_cell["predicted_metrics"] != result["metrics"]
                or saved_metrics != checked_comparison):
            raise ValueError("saved result/comparison differs from the previous campaign")
        if interactive_enabled:
            reused_report = _regenerate_saved_serving_report(result, result_path, directory / "reused-report.html")
        else:
            reused_report = {"status": "disabled", "reason": "offline HTML report not requested"}
        write_symbolic_serving_report(result_path, directory / "report.md", interactive_report=reused_report)
        return {**previous_cell, **report_context,
                "manifest_sha256": cell.manifest_sha256,
                "requests_sha256": cell.requests_sha256,
                "captured_runtime": dict(case.manifest["runtime"]),
                "capture_contract_status": f"schema_v{case.manifest['schema_version']}_validated",
                "observation_quality": observation_quality,
                "accuracy_claim_status": accuracy_claim_status,
                "reused_simulation_result": True, "result_path": str(result_path),
                "metrics": comparison if legacy_result else {**comparison, "ttft_breakdown": None},
                "interactive_report": reused_report,
                "report_regeneration_wall_time_seconds": perf_counter() - started}
    receipt = write_symbolic_serving_report(result_path, directory / "report.md")
    return {**record, **report_context, "prediction_status": "completed", "prediction_reason": None,
        "performance_timing_status": result["performance_timing_status"],
        "observation_quality": observation_quality,
        "accuracy_claim_status": accuracy_claim_status,
        "process_peak_rss_bytes": result["process_peak_rss_bytes"],
        "result_path": str(result_path), "receipt": receipt,
        **{key: result[key] for key in ("interactive_report", "report_output_costs") if key in result},
        "execution_provenance": result["execution_provenance"],
        "metrics": comparison,
        "predicted_metrics": result["metrics"], "schedule_end_time_s": result["schedule_end_time_s"]}


def _simulation_batch_timing(cells):
    rows = []
    for cell in cells:
        if cell["prediction_status"] != "completed":
            continue
        result = json.loads(Path(cell["result_path"]).read_text())
        if ("simulation_started_monotonic_s" not in result
                or "simulation_finished_monotonic_s" not in result):
            return {"schema_version": 1,
                "availability": "legacy_absolute_case_boundaries_unavailable",
                "case_count": 0, "cases": []}
        started = result["simulation_started_monotonic_s"]
        finished = result["simulation_finished_monotonic_s"]
        wall = result["simulation_wall_time_s"]
        if (not isinstance(started, (int, float)) or isinstance(started, bool)
                or not isinstance(finished, (int, float)) or isinstance(finished, bool)
                or not isinstance(wall, (int, float)) or isinstance(wall, bool)
                or started < 0.0 or finished < started or wall < 0.0
                or abs(finished - started - wall) > max(1e-9, wall * 1e-9)):
            raise ValueError(f"invalid simulation timing for {cell['cell_id']}")
        rows.append({"cell_id": cell["cell_id"], "started_monotonic_s": started,
            "finished_monotonic_s": finished, "simulation_wall_time_s": wall})
    if not rows:
        return {"schema_version": 1, "availability": "no_completed_simulation",
            "case_count": 0, "cases": []}
    started = min(row["started_monotonic_s"] for row in rows)
    finished = max(row["finished_monotonic_s"] for row in rows)
    return {"schema_version": 1, "availability": "available",
        "scope": "earliest_case_simulation_start_to_latest_case_simulation_finish",
        "postprocessing": "outside_this_boundary_may_overlap_earlier_case_simulation",
        "started_monotonic_s": started, "finished_monotonic_s": finished,
        "wall_time_s": finished - started, "case_count": len(rows), "cases": rows}


def _initial_inner_workers(spec_count, logical_cpus):
    """Resolve the per-case inner execution-worker count for this run.

    Positive overrides are honored by the caller; here 0 selects the
    CPU-aware automatic default.  A single selected case gets a wider
    inner pool because no other outer worker shares the host.  Small
    subsets get a modest fixed pool.  The full matrix starts from the
    historical single-worker default so every selected case owns exactly
    one outer driver process; late survivors then coordinate through
    the shared filesystem-backed promotion protocol.  The result is a
    plain function of selected-case count and host CPU facts, never of
    case identity, hardware name, or predicted/observed durations.
    """
    if spec_count <= 1:
        return min(4, logical_cpus // 2)
    if spec_count <= 8:
        return 2
    return 1


def _adaptive_inner_pool_facts(spec_count, logical_cpus):
    """Shared budget and promotion threshold for the late-tail pool."""
    if spec_count <= 1:
        return 0, 0
    budget = max(0, logical_cpus - spec_count)
    # Promote as soon as every survivor can take at least one extra worker;
    # the spare budget is unused while cases run serially, so grant early.
    threshold = max(2, budget // 8)
    return budget, threshold


def _write_json_atomic(path, payload):
    """Publish JSON atomically so concurrent readers never see a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
        suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _adaptive_locked_update(path, epoch, coordination_state, update):
    """Single-writer update of the shared run file under an exclusive lock.

    The parent coordinator and the case children all funnel their
    read-modify-write of adaptive_started_cases.json through this one
    locked helper, so updates cannot overwrite each other and a stale
    epoch fails closed instead of silently corrupting the run state.
    """
    import fcntl
    path = Path(path)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text())
            if "epoch" not in state or state["epoch"] != epoch:
                raise ValueError("adaptive inner-worker state is missing or belongs to a different run")
            for field, expected_value in coordination_state.items():
                if state[field] != expected_value:
                    raise ValueError("adaptive inner-worker state is inconsistent with this run")
            started_cases = update(set(state["started_cases"]))
            next_state = {**coordination_state, "epoch": epoch,
                "started_cases": sorted(started_cases)}
            _write_json_atomic(path, next_state)
            return next_state
        finally:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)


_completion_update_hook_for_testing = None


def _read_adaptive_started(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        raise ValueError("adaptive inner-worker state is being updated; retry the read")


def _write_adaptive_case_state(case_directory, state, index):
    """Record this case's live coordination state for the serving simulation.

    The serving simulation re-reads this file at every parallel-tail
    batch boundary (via _adaptive_tail_workers_for) so an already-running
    case can widen its inner pool when enough sibling cases have finished.
    """
    path = Path(case_directory) / "adaptive_case_state.json"
    _write_json_atomic(path, {"spec_count": state["spec_count"], "logical_cpus": state["logical_cpus"],
        "active_cases": sorted(state["active_cases"]), "index": index})


def _mark_adaptive_case_finished(case_directory):
    """Signal that this case has finished so siblings can be promoted."""
    path = Path(case_directory) / "adaptive_case_finished.json"
    _write_json_atomic(path, {"finished": True})


def _run_virtual_serving_case_with_promotion(args, *, h200_scenario=None):
    """Track this case in the coordinator and run it (top-level for pickling).

    The adaptive coordinator in run_virtual_serving_evaluation owns the
    shared state; this worker-side helper records its own start in the
    shared run file, seeds its per-case coordination record, and marks
    itself finished when done.  The serving simulation reads the per-case
    record at every parallel-tail boundary and grants extra inner workers
    when the active-case count drops to the promotion threshold.
    """
    (spec_index, spec, submission_order, spec_count, extra_pool_budget,
     promotion_threshold, started_path, epoch, coordination_state) = args
    state = _adaptive_locked_update(started_path, epoch, coordination_state,
        lambda started_cases: started_cases | {spec_index})
    started_cases = set(state["started_cases"])
    (root_, config_, cell_, case_, directory_, python_, sdk_, source_, reuse_,
     timing_, mode_, audit_, _inner, previous_, adaptive_state,
     adaptive_index, _auto_flag, interactive_report) = spec
    child_state = {**coordination_state, "started_cases": sorted(started_cases)}
    adaptive_spec = (root_, config_, cell_, case_, directory_, python_, sdk_, source_, reuse_,
        timing_, mode_, audit_, 0, previous_, child_state, spec_index, True, interactive_report)
    try:
        cell = _run_virtual_serving_case(adaptive_spec,
            **({"h200_scenario": h200_scenario} if h200_scenario is not None else {}))
        result_path = directory_ / "result.json"
        observed_grants = []
        if result_path.is_file():
            result_payload = json.loads(result_path.read_text())
            if "adaptive_tail_worker_grants" in result_payload:
                raw = result_payload["adaptive_tail_worker_grants"]
                if (isinstance(raw, list)
                        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in raw)):
                    observed_grants = raw
                else:
                    raise ValueError("adaptive_tail_worker_grants has an invalid schema")
        # Only count genuinely promoted cases: a grant strictly above the
        # static resolved inner count.  When no adaptive grant fired the
        # observed list is empty and the run's static inner count is
        # reported unchanged, so telemetry distinguishes "no observation"
        # from "observed the baseline".
        granted_workers = max(observed_grants) if observed_grants else None
        return spec_index, granted_workers, cell
    finally:
        _mark_adaptive_case_finished(directory_)


def run_virtual_serving_evaluation(root, directory, selected_ids, jobs, python, sdk, source,
        load_validated_case, *, h200_version="0.30.0", h200_dataset_root=None, reuse_completed=False,
        serving_timing_mode="device_only", execution_mode="accelerated", state_audit=False,
        execution_workers=1, interactive_report=False, h200_scenario=None):
    """Run every selected canonical request, keeping unsupported cells visible.

    ``interactive_report`` is explicit opt-in: only when true do cases add
    ``--interactive-report`` to the simulation argv and produce HTML pages
    plus visualization projections.
    """
    from . import sharegpt_compact_evaluation as compact
    from .compact_case_contract import validate_case_data_pins
    if type(interactive_report) is not bool:
        raise TypeError("virtual Serving interactive_report must be boolean")
    if h200_scenario not in (None, "analytical_bf16_v1"):
        raise ValueError("unknown H200 Serving scenario")
    case_worker = _run_virtual_serving_case
    promotion_worker = _run_virtual_serving_case_with_promotion
    if h200_scenario is not None:
        case_worker = partial(case_worker, h200_scenario=h200_scenario)
        promotion_worker = partial(promotion_worker, h200_scenario=h200_scenario)
    if type(state_audit) is not bool:
        raise TypeError("virtual Serving state_audit must be boolean")
    _validate_virtual_execution_workers(execution_workers)
    if type(jobs) is not int or jobs < 0:
        raise ValueError("virtual Serving jobs must be a nonnegative integer; 0 uses one worker per selected case")
    if h200_version not in ("0.30.0", None):
        raise ValueError("virtual Serving currently requires captured H200 vLLM0.30 scenarios")
    if h200_dataset_root is not None:
        raise ValueError("virtual Serving candidate dataset roots are not yet supported")
    if execution_mode not in ("accelerated", "fast"):
        raise ValueError("virtual Serving execution mode must be accelerated or fast")
    all_cells = {cell.cell_id: (config, cell) for config, cells in compact.official_metrics_suite_runs(
        root, nvidia_h200_vllm_version=h200_version) for cell in cells}
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) - set(all_cells):
        raise ValueError("virtual Serving selected cells are unknown or duplicated")
    selected = tuple(selected_ids) if selected_ids else tuple(all_cells)
    if type(reuse_completed) is not bool:
        raise TypeError("virtual Serving reuse must be an explicit boolean")
    # Keep the explicit per-case execution_workers override contract intact:
    # callers that pass a positive value retain full control.  A value of 0
    # selects the CPU-aware default.  One outer driver process is started
    # for every selected case by default (jobs==0); cases that cannot use
    # the colocated parallel tail (H200 v0.30 source models with 8 ranks)
    # receive a bounded static inner pool, while tail-eligible cases start
    # at the historical single-worker default and promote late survivors
    # through the shared filesystem coordinator below.
    spec_count = len(selected_ids) if selected_ids else len(all_cells)
    logical_cpus = os.cpu_count() or 1
    automatic_inner = execution_workers == 0
    if not automatic_inner:
        # Explicit override always wins, preserving the existing contract.
        initial_inner_workers = execution_workers
    else:
        initial_inner_workers = _initial_inner_workers(spec_count, logical_cpus)
    extra_pool_budget, promotion_threshold = _adaptive_inner_pool_facts(spec_count, logical_cpus)
    directory.mkdir(parents=True, exist_ok=reuse_completed)
    previous = None
    if reuse_completed:
        previous = json.loads((directory / "results.json").read_text())
        if {cell["cell_id"] for cell in previous["cells"]} != set(selected):
            raise ValueError("virtual report reuse must preserve the original case set and wall-time scope")
    old_cells = {} if previous is None else {cell["cell_id"]: cell for cell in previous["cells"]}
    specs = []
    hardware_dataset_roots = {}
    for key in selected:
        config, cell = all_cells[key]
        dataset_root = str((root / config.dataset_root).resolve())
        hardware_id = cell.hardware.hardware_id
        if hardware_id in hardware_dataset_roots and hardware_dataset_roots[hardware_id] != dataset_root:
            raise ValueError("selected hardware cases use inconsistent canonical dataset roots")
        hardware_dataset_roots[hardware_id] = dataset_root
        case = load_validated_case(compact.compact_case_path(root, config, cell))
        compact._validate_case_binding(config, cell, case)
        validate_case_data_pins(cell, case)
        per_case_inner = initial_inner_workers
        specs.append((root, config, cell, case, directory / key.replace("/", "__"), python, sdk, source, reuse_completed,
                      serving_timing_mode, execution_mode, state_audit, per_case_inner,
                      None if previous is None else old_cells[key], None, None, False, interactive_report))
    started = perf_counter()
    # Low arrival rates expose longer serial decode sequences. Start those
    # cases first so later hardware groups do not wait behind short cases;
    # keep the requested order in the report and never use measured durations.
    submission_order = sorted(range(len(specs)), key=lambda index: specs[index][2].qps)
    worker_count = len(specs) if jobs == 0 else min(jobs, len(specs))
    coordination_state = None
    started_path = None
    if automatic_inner and extra_pool_budget > 0 and not reuse_completed:
        # Late-tail promotion: the serving simulation re-reads the shared
        # run state at every parallel-tail boundary.  When the number of
        # cases that have not yet signalled completion drops to the
        # promotion threshold, each surviving case's next tail batch is
        # granted extra inner workers.  The budget, threshold and grant
        # size are plain functions of the selected-case count and host CPU
        # facts, never of case identity, hardware name, or
        # predicted/observed durations.
        #
        # Coordination channel: a shared run file inside the report
        # directory tracks which cases have started; each case also owns a
        # per-case file recording the currently-active set, refreshed by
        # the parent every time any case finishes.  The per-case file is
        # the only channel the serving simulation reads.
        epoch = hex(int.from_bytes(os.urandom(8), "big"))[2:]
        coordination_state = {"spec_count": spec_count, "logical_cpus": logical_cpus,
            "extra_pool_budget": extra_pool_budget, "promotion_threshold": promotion_threshold}
        started_path = directory / "adaptive_started_cases.json"
        _write_json_atomic(started_path, {**coordination_state, "epoch": epoch, "started_cases": []})
        adaptive_index_to_dir = {index: specs[index][4] for index in submission_order}

        def note_started(spec_index):
            _adaptive_locked_update(started_path, epoch, coordination_state,
                lambda started_cases: started_cases | {spec_index})

        def remove_finished(index_set):
            finished = set()
            for index in index_set:
                flag = adaptive_index_to_dir[index] / "adaptive_case_finished.json"
                if flag.is_file():
                    finished.add(index)
            return index_set - finished

        def prune_finished(active_cases):
            for index in list(active_cases):
                if (adaptive_index_to_dir[index] / "adaptive_case_finished.json").is_file():
                    active_cases.discard(index)

        def write_active_snapshot(active_cases):
            snapshot = {"spec_count": spec_count, "logical_cpus": logical_cpus,
                "active_cases": active_cases}
            for index in submission_order:
                _write_adaptive_case_state(adaptive_index_to_dir[index], snapshot, index)

        def refresh_active_cases(changed_index=None):
            state = _read_adaptive_started(started_path)
            started_cases = set(state["started_cases"])
            active_cases = remove_finished(started_cases)
            if changed_index is not None and changed_index not in active_cases:
                active_cases = active_cases | {changed_index}
            write_active_snapshot(active_cases)
            return active_cases

        # The parent's early call to note_started seeds the record with the
        # first batch so a very fast first finisher cannot shrink the
        # observed active count to zero before any other case launches.
        for spec_index in submission_order[:worker_count]:
            note_started(spec_index)
        active_cases = refresh_active_cases()
        futures = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for spec_index in submission_order:
                if spec_index not in submission_order[:worker_count]:
                    note_started(spec_index)
                    active_cases = refresh_active_cases(spec_index)
                future = executor.submit(promotion_worker,
                    (spec_index, specs[spec_index], submission_order, spec_count, extra_pool_budget,
                     promotion_threshold, started_path, epoch, coordination_state))
                futures[future] = spec_index

            coordination_lock = threading.Lock()
            completion_update_errors = []

            def refresh_on_completion(future):
                completed_index = futures[future]
                try:
                    with coordination_lock:
                        active_cases.discard(completed_index)
                        prune_finished(active_cases)
                        write_active_snapshot(active_cases)
                        if _completion_update_hook_for_testing is not None:
                            _completion_update_hook_for_testing(set(active_cases))
                except BaseException as error:
                    completion_update_errors.append(error)

            for future in futures:
                future.add_done_callback(refresh_on_completion)
            cells_by_index = {}
            actual_grants = {}
            coordination_errors = []
            for future in futures:
                try:
                    finished_index, granted_workers, cell = future.result()
                except BaseException as error:
                    coordination_errors.append(error)
                    continue
                cells_by_index[finished_index] = cell
                actual_grants[finished_index] = granted_workers
                with coordination_lock:
                    active_cases.discard(finished_index)
                    prune_finished(active_cases)
                    write_active_snapshot(active_cases)
            executor.shutdown(wait=True, cancel_futures=True)
            # Done-callback exceptions are only logged by concurrent.futures;
            # they must surface here even when every worker itself succeeded.
            if completion_update_errors:
                raise RuntimeError(f"adaptive completion update failed for {len(completion_update_errors)} case(s)") from completion_update_errors[0]
            if coordination_errors:
                raise RuntimeError(f"adaptive case coordination failed for {len(coordination_errors)} case(s)") from coordination_errors[0]
            if len(cells_by_index) != len(specs):
                raise RuntimeError(f"adaptive case coordination lost results: {len(cells_by_index)} of {len(specs)} cases returned")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            completed = executor.map(case_worker,
                (specs[index] for index in submission_order))
            cells_by_index = dict(zip(submission_order, completed, strict=True))
        actual_grants = {}
    if coordination_state is not None and started_path is not None and started_path.is_file():
        started_path.unlink()
    if coordination_state is not None and started_path is not None:
        lock_path = started_path.with_suffix(".lock")
        if lock_path.is_file():
            lock_path.unlink()
    cells = [cells_by_index[index] for index in range(len(specs))]
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
        check=True, capture_output=True, text=True).stdout.strip()
    # Only cases with a genuine adaptive grant observation contribute to the
    # grant histogram; runs where the coordinator stayed silent report an
    # empty observation set rather than back-filling the static baseline.
    observed_grants = [grant for grant in actual_grants.values() if grant is not None]
    grant_counts = Counter(observed_grants) if observed_grants else Counter()
    payload = {"schema_version": 1, "workload_path": "virtual_tensor", "source_commit": commit,
        "simulation_snapshots": sorted({cell["execution_provenance"]["repository"]["git_commit"]
            for cell in cells if cell["prediction_status"] == "completed"}),
        "hardware_dataset_roots": hardware_dataset_roots,
        "cell_count": len(cells), "evaluation_wall_time_seconds": perf_counter() - started,
        "serving_execution_mode": execution_mode,
        "serving_execution_workers": initial_inner_workers,
        "serving_execution_workers_auto": automatic_inner,
        "serving_initial_inner_workers": initial_inner_workers,
        "serving_extra_pool_budget": extra_pool_budget,
        "serving_promotion_threshold": promotion_threshold,
        "serving_promoted_cases": sum(1 for grant in observed_grants if grant > initial_inner_workers),
        "serving_worker_grant_counts": dict(sorted(grant_counts.items())),
        "serving_state_audit": state_audit,
        "simulation_batch_timing": _simulation_batch_timing(cells),
        "cells": cells, "goal_complete": False}
    if previous is not None:
        payload["report_regeneration_wall_time_seconds"] = payload["evaluation_wall_time_seconds"]
        payload["report_regeneration_source_commit"] = commit
        payload["source_commit"] = previous["source_commit"]
        payload["evaluation_wall_time_seconds"] = previous["evaluation_wall_time_seconds"]
    if all(cell["prediction_status"] != "completed" or "forward_provenance" in cell for cell in cells):
        payload["report_layout"] = "standard_paired"
        payload["serving_timing_mode"] = serving_timing_mode
    result_path = directory / "results.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return result_path


def write_virtual_serving_evaluation_report(result_path, output):
    """Render the actual path and limits; never relabel the captured contract."""
    payload = json.loads(Path(result_path).read_text())
    if payload["workload_path"] != "virtual_tensor":
        raise ValueError("virtual Serving report requires virtual-path results")
    from evaluation.reporting.standard_evaluation_report import _serving_scenario_note
    lines = ["# Serving evaluation: virtual Tensor route", "",
        f"[Full results]({Path(result_path).resolve()}); commit observed during aggregation: `{payload['source_commit']}`; commits observed at simulation end: `{payload['simulation_snapshots']}`. Exact imported-code snapshots are unbound; these end observations cannot establish frozen execution snapshots.", "",
        "Each completed case executes its full canonical request stream. Unmigrated cases are listed explicitly, without falling back to the old graph.", "",
        *_serving_scenario_note(payload["cells"]),
        "| Model / hardware / case | Requests | Execution status | Details |", "|---|---:|---|---|"]
    for cell in payload["cells"]:
        detail = (f"[New-route results]({cell['result_path']})" if cell["prediction_status"] == "completed"
            else cell["prediction_reason"])
        lines.append(f"| {cell['cell_id']} | {cell['request_count']} | {cell['prediction_status']} | {detail} |")
    lines.extend(["", "Each case retains its selected timing mode and original prediction contract; simulation does not start unless the shared source is selected.", "",
        "| case | Method / launch | Scheduler / depth / async | graph / prefix | TP / EP | source |",
        "|---|---|---|---|---|---|"])
    for cell in payload["cells"]:
        if "effective_prediction_contract" in cell:
            contract = cell["effective_prediction_contract"]
            lines.append(f"| {cell['cell_id']} | {contract['prediction_method']} / {contract['launch_policy']} | "
                f"{contract['scheduler_version']} / {contract['scheduler_pipeline_depth']} / {contract['async_scheduling']} | "
                f"{contract['graph_mode']} / {contract['enable_prefix_caching']} | "
                f"{contract['tensor_parallel_size']} / {contract['expert_parallel_size']} | {contract['symbolic_forward_source']} |")
    lines.extend(["", "The table predicts declared source scenarios; it does not prove old/new route equivalence.", "",
        "| case | Mean TTFT ms | Mean TPOT ms | Request throughput req/s |", "|---|---:|---:|---:|"])
    for cell in payload["cells"]:
        if cell["prediction_status"] == "completed":
            metrics = cell["metrics"]
            lines.append(f"| {cell['cell_id']} | {metrics['latency']['ttft_ms']['mean']['predicted']:.6f} | "
                f"{metrics['latency']['tpot_ms']['mean']['predicted']:.6f} | "
                f"{metrics['throughput']['request_throughput_per_s']['predicted']:.6f} |")
    lines.extend(["", "The table records effective simulation settings; explicit analytical-scenario changes are described separately. Full results retain captured_prediction_contract and effective_prediction_contract independently. Original combinations not covered by shared forward are explicitly unsupported; successful or rejected prechecks are not full-request execution. Host mode is explicitly selected; the standard entry defaults to device_plus_launch_plus_scheduler. A single run is not a three-mode comparison.", "",
        "Original normal-policy paths, bindings, and reserve bytes remain unchanged. Normal capacity, startup-log 10%, and actual-provider accuracy remain unaccepted. Only completed cases compare observed metrics, for evaluation only and never Forward.", "",
        "This report establishes listed request-execution statuses, without proving old/new graph, scheduling, or capacity equivalence. Differences require individual comparison and user judgment. New runs pair canonical request_idx before anonymous persistence and retain evaluation metrics; reused older anonymous results compare distribution statistics only, without inferring request identity.", ""])
    Path(output).write_text("\n".join(lines))
