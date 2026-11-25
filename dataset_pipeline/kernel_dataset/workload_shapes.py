"""Hardware-independent expansion of canonical kernel workload shapes."""

import json
from typing import Callable, Mapping

from dataset_pipeline.kernel_dataset.config import (
    require_non_empty_list,
    require_non_empty_string,
    require_mapping,
    require_positive_int,
)


def require_staging_baseline_keys(
        baseline: object, expected_keys, path,
) -> Mapping[str, object]:
    """Validate a staging baseline against the current workload contract.

    The baseline is a canonical dataset document keyed by case filename.  A
    key the current contract no longer produces, or one it requires and the
    baseline lacks, means the baseline was captured under a different workload
    contract, so it fails loud rather than being silently intersected.
    """

    if not isinstance(baseline, Mapping):
        raise TypeError(f"canonical dataset must be a mapping: {path}")
    missing = sorted(set(expected_keys) - set(baseline))
    unexpected = sorted(set(baseline) - set(expected_keys))
    if missing or unexpected:
        raise ValueError(
            "staging baseline does not match the current workload contract: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}")
    return baseline


def build_fused_moe_shapes(
        kernel_cfg: Mapping[str, object],
        *,
        argument_names: Mapping[str, str],
        include_layer: bool,
) -> list:
    """Expand the canonical FusedMoE workload contract.

    The case identity, the attribute set and the required fields are the
    dataset contract's and are the same on every backend; only the runner's
    CLI flag spelling differs, and one backend records the layer name in the
    attributes.  Both differences are supplied by the caller as data, while
    the YAML keys are the canonical ones every backend's config uses.
    """

    shapes = []
    workloads = require_non_empty_list(
        kernel_cfg, "workload_configs", "fused_moe config")
    layers = require_non_empty_list(kernel_cfg, "layers", "fused_moe config")
    for workload in workloads:
        require_mapping(workload, "each fused_moe workload")
        tokens = require_positive_int(
            workload, "num_tokens", "fused_moe workload")
        for layer in layers:
            require_mapping(layer, "each fused_moe layer")
            experts = require_positive_int(
                layer, "num_experts", "fused_moe layer")
            top_k = require_positive_int(layer, "top_k", "fused_moe layer")
            hidden = require_positive_int(
                layer, "hidden_size", "fused_moe layer")
            intermediate = require_positive_int(
                layer, "intermediate_size", "fused_moe layer")
            dtype = require_non_empty_string(
                layer, "dtype", "fused_moe layer")
            attributes = {
                "TOK": tokens,
                "E": experts,
                "TOPK": top_k,
                "H": hidden,
                "INT": intermediate,
            }
            if include_layer:
                attributes["layer"] = require_non_empty_string(
                    layer, "name", "fused_moe layer")
            shapes.append({
                "filename": (
                    f"FUSEDMOE_TOK{tokens}_E{experts}_TOPK{top_k}"
                    f"_H{hidden}_INT{intermediate}"
                ),
                "type": "FUSEDMOE",
                "attr": attributes,
                "script_args": [
                    argument_names["tokens"], str(tokens),
                    argument_names["experts"], str(experts),
                    argument_names["top_k"], str(top_k),
                    argument_names["hidden"], str(hidden),
                    argument_names["intermediate"], str(intermediate),
                    "--dtype", dtype,
                ],
            })
    return shapes


def expand_kernel_shapes(
        kernel: str,
        kernel_cfg: Mapping[str, object],
        expansions: Mapping[str, Callable[[Mapping[str, object]], list]],
        *,
        unknown_kernel_message: str,
) -> list:
    """Dispatch one canonical kernel name to its shape expansion.

    The GEMM corpus is hardware independent, so ``gemm`` is always the shared
    expansion; the other families are supplied by the caller because each
    backend attaches its own runtime kernel names to the canonical shapes.
    ``unknown_kernel_message`` carries the caller's own wording for an
    unrecognized name so the failure stays attributable.
    """

    if kernel == "gemm":
        return build_gemm_shapes(kernel_cfg)
    if kernel in expansions:
        expansion = expansions[kernel]
    else:
        raise ValueError(unknown_kernel_message.format(kernel=kernel))
    return expansion(kernel_cfg)


_DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


def _layer_index(kernel_cfg):
    layers = require_non_empty_list(kernel_cfg, "layers", "workload config")
    result = {}
    for layer in layers:
        require_mapping(layer, "each workload layer")
        name = require_non_empty_string(layer, "name", "workload layer")
        if name in result:
            raise ValueError(f"duplicate workload layer name: {name}")
        result[name] = layer
    return result


def _selected_layers(kernel_cfg, workload):
    layers_by_name = _layer_index(kernel_cfg)
    if "layer" in workload:
        layer_name = require_non_empty_string(workload, "layer", "workload")
        if layer_name not in layers_by_name:
            raise ValueError(f"unknown workload layer: {layer_name}")
        return [layers_by_name[layer_name]]

    selected = []
    for layer in layers_by_name.values():
        override_only = layer["override_only"]
        if not isinstance(override_only, bool):
            raise TypeError(
                f"layer {layer['name']}.override_only must be bool"
            )
        if not override_only:
            selected.append(layer)
    if not selected:
        raise ValueError("workload has no selectable layers")
    return selected


def _dtype(layer):
    dtype = require_non_empty_string(
        layer, "dtype", f"layer {layer['name']}")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported workload dtype: {dtype}")
    return dtype


def flash_attn_request_lengths(batch, query_length, kv_length,
                               kv_layout, kv_distribution):
    """Return the exact per-request lengths executed by every FA runner.

    ``Q_LEN`` and ``KV_LEN`` remain maxima used for case bucketing.  The
    returned lists are the workload contract: runners must consume them
    verbatim and predictors must price them verbatim.  Keeping this expansion
    hardware-independent prevents A800, 910B and C500 from silently generating
    different tensors for the same case ID.
    """
    for name, value in (
            ("batch", batch), ("query_length", query_length),
            ("kv_length", kv_length)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"flash_attn {name} must be an integer")
        if value <= 0:
            raise ValueError(f"flash_attn {name} must be positive")
    if kv_layout not in (
            "contiguous", "paged", "paged_varlen", "mla",
            "chunked_prefill"):
        raise ValueError(f"unsupported flash_attn kv_layout: {kv_layout!r}")
    if kv_distribution not in ("skew", "uniform"):
        raise ValueError(
            f"unsupported flash_attn kv_dist: {kv_distribution!r}")

    def distributed_lengths(maximum):
        # 128 is the canonical minimum serving context represented by this
        # case grid.  It is a workload choice, not the hardware page size.
        minimum = min(maximum, max(128, maximum // 4))
        if kv_distribution == "uniform":
            if batch == 1:
                return [maximum]
            step = (maximum - minimum) // (batch - 1)
            return [
                minimum + index * step
                for index in range(batch)
            ]
        elif kv_distribution == "skew":
            populations = [int(batch * 0.60), int(batch * 0.25)]
            populations.append(batch - sum(populations))
            lengths = []
            for population, multiplier in zip(
                    populations, (0.25, 0.5, 1.0)):
                length = min(
                    maximum, max(128, int(maximum * multiplier)))
                lengths.extend([length] * population)
            return lengths[:batch]
        else:
            raise AssertionError(
                f"unhandled flash_attn kv_dist: {kv_distribution!r}")

    if kv_layout in ("paged_varlen", "chunked_prefill"):
        key_value_lengths = distributed_lengths(kv_length)
    elif kv_layout in ("contiguous", "paged", "mla"):
        key_value_lengths = [kv_length] * batch
    else:
        raise AssertionError(
            f"unhandled flash_attn kv_layout: {kv_layout!r}")

    if kv_layout == "chunked_prefill":
        # Q in the case ID is the scheduler's whole-batch chunk budget.
        # Requests receive deterministic serving-like chunks whose total is
        # bounded by that budget; the exact list is the executable contract.
        per_request_maximum = max(1, query_length // batch)
        query_lengths = distributed_lengths(per_request_maximum)
        if sum(query_lengths) > query_length:
            raise ValueError(
                "chunked_prefill query lengths exceed the scheduler budget: "
                f"sum={sum(query_lengths)}, budget={query_length}")
    elif kv_layout == "paged_varlen" and query_length > 1:
        # A serving varlen prefill request contains one full prompt: its query
        # and KV histories have the same per-request length.  Q_LEN/KV_LEN stay
        # maxima for case bucketing; these lists are the tensors actually run.
        query_lengths = list(key_value_lengths)
    elif kv_layout in ("contiguous", "paged", "paged_varlen", "mla"):
        query_lengths = [query_length] * batch
    else:
        raise AssertionError(
            f"unhandled flash_attn kv_layout: {kv_layout!r}")

    if len(query_lengths) != batch:
        raise AssertionError("flash_attn query expansion lost a request")
    if len(key_value_lengths) != batch:
        raise AssertionError("flash_attn KV expansion lost a request")
    invalid_requests = [
        index for index, (request_query, request_kv) in enumerate(
            zip(query_lengths, key_value_lengths))
        if request_query > request_kv
    ]
    if invalid_requests:
        raise ValueError(
            "flash_attn request query length cannot exceed its KV length: "
            f"request_indices={invalid_requests}, q_lens={query_lengths}, "
            f"kv_lens={key_value_lengths}")
    return query_lengths, key_value_lengths


def build_gemm_shapes(kernel_cfg):
    """Expand the hardware-independent GEMM workload contract."""
    shapes = []
    workloads = require_non_empty_list(
        kernel_cfg, "workload_configs", "gemm config")
    layers = require_non_empty_list(kernel_cfg, "layers", "gemm config")
    for workload in workloads:
        require_mapping(workload, "each gemm workload")
        batch = require_positive_int(workload, "batch", "gemm workload")
        sequence = require_positive_int(
            workload, "seq_len", "gemm workload")
        for layer in layers:
            require_mapping(layer, "each gemm layer")
            k = require_positive_int(layer, "k", "gemm layer")
            n = require_positive_int(layer, "n", "gemm layer")
            layer_name = require_non_empty_string(
                layer, "name", "gemm layer")
            dtype = _dtype(layer)
            shapes.append({
                "filename": f"GEMM_B{batch}_M{sequence}_K{k}_N{n}",
                "type": "GEMM",
                "attr": {
                    "B": batch,
                    "M": sequence,
                    "K": k,
                    "N": n,
                    "dtype": dtype,
                    "layer": layer_name,
                },
                "script_args": [
                    "--bs", str(batch),
                    "--sl", str(sequence),
                    "--in_feat", str(k),
                    "--out_feat", str(n),
                    "--dtype", dtype,
                ],
            })
    return shapes


def build_flash_attn_shapes(kernel_cfg):
    shapes = []
    for workload in require_non_empty_list(
            kernel_cfg, "workload_configs", "flash_attn config"):
        require_mapping(workload, "each flash_attn workload")
        for layer in _selected_layers(kernel_cfg, workload):
            batch = require_positive_int(
                workload, "batch", "flash_attn workload")
            sequence = require_positive_int(
                workload, "seq_len_cache", "flash_attn workload")
            query_heads = require_positive_int(
                layer, "num_heads_q", f"layer {layer['name']}")
            kv_heads = require_positive_int(
                layer, "num_heads_kv", f"layer {layer['name']}")
            head_dim = require_positive_int(
                layer, "head_dim", f"layer {layer['name']}")
            splits = require_positive_int(
                workload, "num_splits", "flash_attn workload")
            split_policy = require_non_empty_string(
                workload, "split_policy", "flash_attn workload")
            if split_policy == "forced":
                pass
            elif split_policy == "serving":
                if splits != 1:
                    raise ValueError(
                        "flash_attn serving split policy requires the "
                        "backend request value "
                        f"num_splits=1, got {splits}")
            else:
                raise ValueError(
                    f"unsupported flash_attn split_policy: {split_policy!r}")
            causal = workload["causal"]
            if not isinstance(causal, bool):
                raise TypeError("flash_attn workload.causal must be bool")
            dtype = _dtype(layer)
            kv_layout = require_non_empty_string(
                workload, "kv_layout", "flash_attn workload")
            kv_distribution = require_non_empty_string(
                workload, "kv_dist", "flash_attn workload")
            if kv_distribution not in ("skew", "uniform"):
                raise ValueError(
                    f"unsupported flash_attn kv_dist: {kv_distribution}")
            query_length = require_positive_int(
                workload, "seq_len_q", "flash_attn workload")
            query_lengths, key_value_lengths = flash_attn_request_lengths(
                batch, query_length, sequence, kv_layout, kv_distribution)
            executed_query_maximum = max(query_lengths)
            executed_kv_maximum = max(key_value_lengths)
            is_prefill = executed_query_maximum > 1
            if kv_layout == "contiguous":
                layout_tag = ""
            elif kv_layout == "paged":
                layout_tag = "_PAGED"
            elif kv_layout == "paged_varlen":
                layout_tag = f"_PAGEDVARLEN_{kv_distribution.upper()}"
            elif kv_layout == "mla":
                layout_tag = "_MLA"
            elif kv_layout == "chunked_prefill":
                layout_tag = f"_CHUNKED_{kv_distribution.upper()}"
            else:
                raise ValueError(f"unsupported flash_attn kv_layout: {kv_layout}")
            phase_tag = "_PREFILL" if is_prefill else ""
            if kv_layout == "chunked_prefill":
                query_tag = f"_Q{query_length}"
            elif kv_layout in ("contiguous", "paged", "paged_varlen", "mla"):
                query_tag = ""
            else:
                raise ValueError(
                    f"unsupported flash_attn kv_layout: {kv_layout}")
            filename = (
                f"FLASHATTN{layout_tag}{phase_tag}{query_tag}_B{batch}"
                f"_S{sequence}_HQ{query_heads}_HKV{kv_heads}_D{head_dim}"
                f"_C{1 if causal else 0}"
            )
            shapes.append({
                "filename": filename,
                "type": "FLASHATTN",
                "attr": {
                    "B": batch,
                    "S": sequence,
                    "Q_LEN": executed_query_maximum,
                    "Q_REQUESTED": query_length,
                    "KV_LEN": executed_kv_maximum,
                    "KV_REQUESTED": sequence,
                    "HQ": query_heads,
                    "HKV": kv_heads,
                    "D": head_dim,
                    "SP": None,
                    "SP_REQUESTED": splits,
                    "Q_LENS": query_lengths,
                    "KV_LENS": key_value_lengths,
                    "split_policy": split_policy,
                    "kv_layout": kv_layout,
                    "kv_dist": kv_distribution,
                    "causal": causal,
                    "layer": layer["name"],
                },
                "script_args": [
                    "--batch", str(batch),
                    "--seq_len_cache", str(executed_kv_maximum),
                    "--num_heads_q", str(query_heads),
                    "--num_heads_kv", str(kv_heads),
                    "--head_dim", str(head_dim),
                    "--dtype", dtype,
                    "--kv_layout", kv_layout,
                    "--kv_dist", kv_distribution,
                    "--seq_len_q", str(executed_query_maximum),
                    "--q-lens-json", json.dumps(query_lengths),
                    "--kv-lens-json", json.dumps(key_value_lengths),
                    "--causal", str(1 if causal else 0),
                ],
                "split_policy": split_policy,
            })
    return shapes


def build_vector_shapes(kernel_cfg):
    """Expand canonical vector/operator groups and reject duplicate shapes."""
    operator_types = {
        "rmsnorm": "NORM",
        "add_rmsnorm": "NORM",
        "rope": "ROTARY",
        "silu": "ACTIVATION",
        "elementwise": "ELEMENTWISE",
        "kvcache": "KV_CACHE",
    }
    has_groups = "op_groups" in kernel_cfg
    has_workloads = "workload_configs" in kernel_cfg
    if has_groups == has_workloads:
        raise ValueError(
            "vector workload config must define exactly one of op_groups or "
            "workload_configs"
        )

    workloads = []
    if has_groups:
        for group in require_non_empty_list(
                kernel_cfg, "op_groups", "vector config"):
            require_mapping(group, "each vector op group")
            operator = require_non_empty_string(
                group, "op", "vector op group")
            if operator not in operator_types:
                raise ValueError(f"unsupported vector operator: {operator}")
            tokens = require_non_empty_list(
                group, "toks", f"vector op group {operator}")
            secondary_ops = group["op2"]
            secondary_ops = (
                secondary_ops if isinstance(secondary_ops, list)
                else [secondary_ops])
            if not secondary_ops:
                raise ValueError(
                    f"vector op group {operator}.op2 must not be empty")
            batches = group["batch"]
            batches = batches if isinstance(batches, list) else [batches]
            if not batches:
                raise ValueError(
                    f"vector op group {operator}.batch must not be empty")
            for layer_name in require_non_empty_list(
                    group, "layers", f"vector op group {operator}"):
                if not isinstance(layer_name, str):
                    raise TypeError(
                        f"vector op group {operator}.layers entries must be strings"
                    )
                if not layer_name:
                    raise ValueError(
                        f"vector op group {operator}.layers entries must not be empty"
                    )
                for token_count in tokens:
                    for secondary_op in secondary_ops:
                        for batch in batches:
                            token_mapping = {"tok": token_count}
                            batch_mapping = {"batch": batch}
                            op_mapping = {"op2": secondary_op}
                            workload = {
                                "op": operator,
                                "tok": require_positive_int(
                                    token_mapping, "tok",
                                    f"vector op group {operator}"),
                                "layer": layer_name,
                                "op2": require_non_empty_string(
                                    op_mapping, "op2",
                                    f"vector op group {operator}"),
                                "batch": require_positive_int(
                                    batch_mapping, "batch",
                                    f"vector op group {operator}"),
                            }
                            workloads.append(workload)
    else:
        workloads = require_non_empty_list(
            kernel_cfg, "workload_configs", "vector config")

    shapes = []
    for workload in workloads:
        require_mapping(workload, "each vector workload")
        operator = require_non_empty_string(
            workload, "op", "vector workload")
        if operator not in operator_types:
            raise ValueError(f"unsupported vector operator: {operator}")
        token_count = require_positive_int(
            workload, "tok", "vector workload")
        if "layer" in workload:
            layer_name = require_non_empty_string(
                workload, "layer", "vector workload")
        else:
            layer_name = None
        layers = _selected_layers(kernel_cfg, workload)
        for layer in layers:
            dtype = _dtype(layer)
            hidden = require_positive_int(
                layer, "hidden", f"layer {layer['name']}")
            num_heads = require_positive_int(
                layer, "num_heads", f"layer {layer['name']}")
            head_dim = require_positive_int(
                layer, "head_dim", f"layer {layer['name']}")
            intermediate = require_positive_int(
                layer, "intermediate", f"layer {layer['name']}")
            num_kv_heads = require_positive_int(
                layer, "num_kv_heads", f"layer {layer['name']}")
            batch = require_positive_int(
                workload, "batch", "vector workload")
            secondary_op = require_non_empty_string(
                workload, "op2", "vector workload")
            if operator == "elementwise":
                if secondary_op not in ("add", "mul"):
                    raise ValueError(
                        f"unsupported elementwise operation: {secondary_op}")
            elif operator in (
                    "rmsnorm", "add_rmsnorm", "silu", "rope", "kvcache"):
                if secondary_op != "add":
                    raise ValueError(
                        f"op2 is only valid for elementwise, got {operator}: "
                        f"{secondary_op}"
                    )
            else:
                raise ValueError(f"unsupported vector operator: {operator}")
            dimension_tags = {
                "rmsnorm": f"H{hidden}",
                "add_rmsnorm": f"H{hidden}",
                "elementwise": f"H{hidden}",
                "silu": f"I{intermediate}",
                "rope": f"NH{num_heads}D{head_dim}",
                "kvcache": f"KVH{num_kv_heads}D{head_dim}",
            }
            dimension_tag = dimension_tags[operator]
            if operator == "elementwise":
                operation_tag = "_" + secondary_op.upper()
            elif operator in (
                    "rmsnorm", "add_rmsnorm", "silu", "rope", "kvcache"):
                operation_tag = ""
            else:
                raise ValueError(f"unsupported vector operator: {operator}")
            filename = (
                f"VECTOR_{operator.upper()}_T{token_count}_{dimension_tag}"
                f"_B{batch}{operation_tag}"
            )
            arguments = [
                "--op", operator,
                "--tok", str(token_count),
                "--dtype", dtype,
                "--batch", str(batch),
                "--op2", secondary_op,
            ]
            for option, value in (
                ("--hidden", hidden),
                ("--num_heads", num_heads),
                ("--head_dim", head_dim),
                ("--intermediate", intermediate),
                ("--num_kv_heads", num_kv_heads),
            ):
                arguments += [option, str(value)]
            if operator in ("rmsnorm", "add_rmsnorm", "elementwise"):
                element_count = token_count * hidden
            elif operator == "silu":
                element_count = token_count * intermediate
            elif operator == "rope":
                element_count = token_count * num_heads * head_dim
            elif operator == "kvcache":
                element_count = token_count * num_kv_heads * head_dim
            else:
                raise ValueError(f"unsupported vector operator: {operator}")
            shapes.append({
                "filename": filename,
                "type": operator_types[operator],
                "attr": {
                    "op": operator,
                    "op2": secondary_op,
                    "tok": token_count,
                    "hidden": hidden,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "intermediate": intermediate,
                    "num_kv_heads": num_kv_heads,
                    "batch": batch,
                    "element_count": element_count,
                    "dtype_bytes": _DTYPE_BYTES[dtype],
                    "layer": layer_name,
                },
                "script_args": arguments,
            })

    seen = set()
    for shape in shapes:
        if shape["filename"] in seen:
            raise ValueError(
                f"duplicate vector workload shape: {shape['filename']}")
        seen.add(shape["filename"])
    return shapes
