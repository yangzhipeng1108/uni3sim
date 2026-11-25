"""Finite source-call inventory; scoped receipts never imply whole-model acceptance."""

from collections import Counter


_COMPOSITE = {"model", "decoder", "attention", "input_projections", "prepare_attention",
    "output_projection", "ffn", "ffn_runner", "ffn_shared_mlp", "main_compressor", "indexer_compressor"}

_MATH_RECEIPTS = {
    "embedding": ("prepare",), "kv_insert": ("prepare",), "short_topk": ("prepare",), "indexer_query": ("prepare",),
    "fused_qkv_projection": ("fp8_dataflow", "quantizer"), "query_projection": ("fp8_dataflow", "quantizer"),
    "indexer_query_projection": ("fp8_dataflow", "quantizer"), "output_linear_projection": ("fp8_dataflow", "quantizer"),
    "compressor_projection": ("mixed_projection",), "indexer_projection": ("mixed_projection",),
    "indexer_weights_projection": ("mixed_projection",), "qkv_norm": ("qkv_norm",),
    "main_save_partial": ("partial_state",), "indexer_save_partial": ("partial_state",),
    "main_finish": ("finish",), "indexer_finish": ("finish",), "main_compress": ("finish",), "main_store": ("finish",),
    "scored_topk": ("indexer",), "sparse_attention": ("attention_math",),
    "inverse_rope": ("output",), "grouped_output_projection": ("output",),
    "output_all_reduce": ("reduction",), "ffn_all_reduce": ("reduction",),
    "ffn_router": ("ffn",), "ffn_route": ("routing_math", "routing_objects"), "ffn_routed": ("ffn",),
    "ffn_shared_gate_up": ("ffn", "fp8_dataflow", "quantizer"),
    "ffn_shared_down": ("ffn", "fp8_dataflow", "quantizer"),
    "ffn_shared_activation": ("ffn_vector",), "ffn_add": ("ffn_vector",),
    "mhc_pre": ("mhc",), "mhc_post_pre": ("mhc",), "mhc_post": ("mhc",), "hc_head": ("mhc",),
    "norm": ("final_norm",), "row_select": ("target",), "logits": ("target",), "greedy": ("target", "argmax"),
}


def v4_source_scope_inventory(reference, result):
    """Every executed source role gets a row, including unknown new roles.

    Math receipts compare independent complete typed programs in their own
    owner. This inventory connects those scopes to the original call tree.
    It intentionally does not upgrade math receipts to private storage proofs.
    """
    counts = Counter(row["role"] for row in reference["calls"])
    caller = result["source_v4_caller_comparison"]
    caller_rows = {(r["rank"], r["role"], r["ordinal"]): r for r in caller["checks"]}
    rows = []
    for role in sorted(counts):
        source_calls = [r for r in reference["calls"] if r["role"] == role]
        present = [caller_rows[(r["rank"], role, r["ordinal"])]
            for r in source_calls if (r["rank"], role, r["ordinal"]) in caller_rows]
        caller_match = len(present) == len(source_calls) and all(r["status"] == "matched" for r in present)
        receipts = []
        if role in _COMPOSITE:
            math_status = "decomposed"
        elif role not in _MATH_RECEIPTS:
            math_status = "unchecked"
        else:
            for name in _MATH_RECEIPTS[role]:
                field = "source_v4_" + name + "_comparison"
                account = result[field] if field in result else None
                matched = account is not None and account["total"] > 0 and account["matched"] == account["total"]
                if account is not None:
                    matched = matched and account["mismatched"] == account["not_checked"] == 0
                receipts.append({"field": field, "matched": matched,
                    "checks": None if account is None else account["total"]})
            math_status = "matched" if all(r["matched"] for r in receipts) else "unchecked"
        rows.append({"role": role, "source_calls": counts[role], "source": reference["source_bindings"][role],
            "caller_geometry_dtype_storage_strides_offsets": "matched" if caller_match else "mismatched",
            "logical_math_and_typed_payload": math_status, "receipts": receipts})
    unknown = [row["role"] for row in rows if row["logical_math_and_typed_payload"] == "unchecked"]
    storage_receipts = {}
    if "source_v4_wrapper_storage_comparison" in result:
        for index, check in enumerate(result["source_v4_wrapper_storage_comparison"]["checks"]):
            key = check["rank"], check["role"], check["ordinal"]
            if key in storage_receipts:
                raise ValueError("wrapper storage receipt repeats a source invocation")
            storage_receipts[key] = (check, "source_v4_wrapper_storage_comparison", index)
    if "routing_object_work" in reference and "source_v4_routing_objects_comparison" in result:
        routing = {(check["rank"], check["layer"]): (check, index) for index, check in
            enumerate(result["source_v4_routing_objects_comparison"]["checks"])}
        for rank in sorted({r["rank"] for r in reference["routing_object_work"]}):
            work = sorted((r for r in reference["routing_object_work"] if r["rank"] == rank), key=lambda r: r["layer"])
            calls = sorted((r for r in reference["calls"] if r["rank"] == rank and r["role"] == "ffn_route"), key=lambda r: r["ordinal"])
            if len(work) != len(calls):
                raise ValueError("source route storage work does not cover its own caller inventory")
            for source, call in zip(work, calls, strict=True):
                key = rank, "ffn_route", call["ordinal"]
                if (rank, source["layer"]) in routing:
                    check, index = routing[rank, source["layer"]]
                    storage_receipts[key] = (check, "source_v4_routing_objects_comparison", index)
    if "source_v4_terminal_storage_comparison" in result:
        stages = result["source_v4_terminal_storage_comparison"]["checks"]
        for call in reference["calls"]:
            role, rank = call["role"], call["rank"]
            if role not in ("embedding", "norm", "logits", "greedy"):
                continue
            if call["ordinal"] != 0:
                raise ValueError("declared target source supports one terminal/embedding invocation per forward")
            expected_stages = [r["call"] for r in reference["terminal_storage_work"] if r["rank"] == rank and (
                role == "embedding" and r["call"] in ("embedding", "embedding:all-reduce")
                or role == "norm" and r["call"] == "final-norm"
                or role == "logits" and r["call"].startswith("lm-head")
                or role == "greedy" and r["call"].startswith("target-sampler"))]
            selected = [(index, row) for index, row in enumerate(stages)
                if row["rank"] == rank and row["stage"] in expected_stages]
            complete = bool(expected_stages) and sorted(r["stage"] for _, r in selected) == sorted(expected_stages)
            matched = complete and all(r["status"] == "matched" for _, r in selected)
            storage_receipts[rank, role, 0] = ({"status": "matched" if matched else "mismatched"},
                "source_v4_terminal_storage_comparison", [index for index, _ in selected])
    storage_obligations = []
    if "source_v4_expert_storage_comparison" in result:
        field = "source_v4_expert_storage_comparison"
        for index, check in enumerate(result[field]["checks"]):
            storage_receipts[check["rank"], "ffn_routed", check["ordinal"]] = (check, field, index)
    for call in reference["calls"]:
        role = call["role"]
        if role in _COMPOSITE:
            continue
        key = call["rank"], role, call["ordinal"]
        if key in storage_receipts:
            check, field, index = storage_receipts[key]
            status = "matched" if check["status"] == "matched" else "mismatch"
            receipt = {"field": field, "check_index": index}
        else:
            status, receipt = "unchecked", None
        storage_obligations.append({"rank": key[0], "role": role, "ordinal": key[2],
            "facet": "source_internal_allocation_alias_lifetime", "status": status,
            "source": reference["source_bindings"][role], "receipt": receipt,
            "required": "source wrapper local allocation/view requests, backing aliases and lifetime versus automatic storage records"})
    storage_counts = Counter(r["status"] for r in storage_obligations)
    return {"scope": "complete executed source caller-role inventory and declared useful Roofline math/typed payload receipt coverage",
        "source_calls": sum(counts.values()), "source_role_families": len(rows),
        "primitive_role_families": sum(row["role"] not in _COMPOSITE for row in rows),
        "unchecked_math_role_families": unknown,
        "caller_mismatched_role_families": [row["role"] for row in rows
            if row["caller_geometry_dtype_storage_strides_offsets"] != "matched"],
        "logical_math_role_inventory_complete": not unknown and all(
            row["caller_geometry_dtype_storage_strides_offsets"] == "matched" for row in rows),
        "rows": rows, "whole_model_source_alignment_accepted": False,
        "source_storage_obligations": storage_obligations,
        "source_storage_counts": {**{name: storage_counts[name] for name in
            ("matched", "mismatch", "unchecked")}, "total": len(storage_obligations)},
        "unchecked_storage_by_role": dict(Counter(row["role"] for row in storage_obligations if row["status"] == "unchecked")),
        "remaining_required_facets": [
            {"facet": "source_internal_materialization", "status": "unchecked",
             "required": "join all wrapper/private source temporary allocation and alias receipts to actual automatic storage records; caller/math equality alone is insufficient"},
            {"facet": "consecutive_logical_state", "status": "unchecked",
             "required": "explicit initialized-prefix and consecutive request sequence receipt; separate single-step reports cannot prove this"}],
        "outside_first_stage": ["actual GPU/provider/compiler precision and traffic", "normal capacity and MTP",
            "comparison/index instruction issue rates"], "runtime_accepted": False}
