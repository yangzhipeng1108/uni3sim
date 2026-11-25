"""Independent embedding/short-index/packed-insert/indexer-Q source accounts."""

import json
from pathlib import Path
import re

from ..simulation.v4_flash_reference import (attach_embedding_accounts,
    prepare_source_bindings, short_topk_source_reference, swa_insert_work_reference,
    _indexer_q_backend_reference)


def prepare_work_reference(raw, source_root, *, tokens, kv_lengths, token_ids, tp):
    if (tp != 4 or not isinstance(token_ids, tuple) or len(token_ids) != tokens
            or any(type(value) is not int or not 0 <= value < raw["vocab_size"] for value in token_ids)):
        raise ValueError("V4 prepare reference requires TP4 and explicit valid scheduled token IDs")
    heads = raw["num_attention_heads"] // tp
    padded = 64 if heads <= 64 else 128
    if (raw["head_dim"], raw["qk_rope_head_dim"]) != (512, 64):
        raise ValueError("V4 packed prepare source requires the exact512/64 geometry")
    if len(raw["compress_ratios"]) < raw["num_hidden_layers"]:
        raise ValueError("V4 prepare source needs every ordinary layer's compression ratio")
    bindings = prepare_source_bindings(Path(source_root))
    rows = []
    for rank in range(tp):
        embedding = {"node_id": "step:embedding", "rank": rank, "variant": "embedding"}
        attach_embedding_accounts(raw, [{"phase": "step"}], [embedding], {"step": list(token_ids)})
        rows.append({"rank": rank, "layer": None, "role": "embedding", "source": "embedding-cuda",
            "work": {**embedding["math_account"]["semantic_fields"], "operands": embedding["byte_account"]["operands"]}})
        # Official config also carries a draft-layer ratio; MTP is off.
        for layer, ratio in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]]):
            rows.append({"rank": rank, "layer": layer, "role": "kv_insert", "source": "qnorm-rope-insert",
                "work": swa_insert_work_reference(tokens, heads, padded)})
            if ratio != 4:
                continue
            if max(kv_lengths) // 4 <= raw["index_topk"]:
                rows.append({"rank": rank, "layer": layer, "role": "short_topk", "source": "short-topk",
                    "work": short_topk_source_reference(tokens, raw["index_topk"])})
            else:
                width, rope, count = raw["index_head_dim"], raw["qk_rope_head_dim"], tokens * raw["index_n_heads"]
                work = _indexer_q_backend_reference(tokens, raw["index_n_heads"], width, rope)
                work = {name: work[name] for name in ("mathematical_primitives", "mathematical_primitives_condition",
                    "backend_alternatives", "selected_backend", "numerical_equivalence_asserted")}
                shapes = (("projected_q", "bf16", count * width, "read"),
                    ("indexer_weights", "bf16", count, "read"), ("positions", "int64", tokens, "read"),
                    ("rotary_cache", "fp32", tokens * rope, "read"),
                    ("quantized_q", "fp8", count * width, "write"), ("folded_weights", "fp32", count, "write"))
                work["operands"] = [{"role": role, "dtype": dtype, "elements": size,
                    "bytes": size * {"bf16": 2, "int64": 8, "fp32": 4, "fp8": 1}[dtype], "access": access}
                    for role, dtype, size, access in shapes]
                rows.append({"rank": rank, "layer": layer, "role": "indexer_query", "source": "indexer-q-quant", "work": work})
    return {"rows": rows, "sources": bindings, "token_ids": token_ids,
        "conditions": "TP4 pure-text BF16, all declared slots valid; FP8 indexer, both CUDA source alternatives retained"}


def compare_v4_prepare_work(reference, plan, hardware):
    from src.kernel.kernel_base import create_predictor
    from src.kernel.predictors.kv_cache_update import dsv4_swa_insert_source_work
    from src.kernel.predictors.vector import (dsv4_short_context_topk_source_work,
        vocab_parallel_embedding_logical_work, dsv4_indexer_q_rope_quant_logical_work)
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {(row["rank"], row["layer"], row["role"]): row for row in reference["rows"]}
    if len(expected) != len(reference["rows"]):
        raise ValueError("V4 prepare reference repeats a source invocation")
    actual = {}
    variants = {"vocab_parallel_embedding": "embedding", "dsv4_qnorm_rope_kv_insert": "kv_insert",
        "dsv4_short_context_topk": "short_topk", "dsv4_indexer_q_rope_quant": "indexer_query"}
    for call in plan.calls:
        spec = call.kernel_spec
        if spec.variant not in variants:
            continue
        role = variants[spec.variant]
        layers = {int(match[1]) for name in call.source_node_ids
            if (match := re.search(r"(?:^|:)layer-(\d+):", name)) is not None}
        if (role != "embedding" and len(layers) != 1) or (role == "embedding" and layers):
            raise ValueError("V4 prepare call has an ambiguous layer identity")
        key = call.rank, None if role == "embedding" else next(iter(layers)), role
        if key in actual:
            raise ValueError("V4 prepare actual repeats a source invocation")
        if role == "embedding":
            owner = vocab_parallel_embedding_logical_work(spec)
            work = {name: owner[name] for name in ("org_vocab_interval", "added_vocab_interval", "owned_token_rows", "source_program")}
            work["operands"] = [{name: row[name] for name in ("role", "dtype", "elements", "bytes", "access")}
                for row in owner["operands"]]
        elif role == "short_topk":
            work = dsv4_short_context_topk_source_work(spec)
        elif role == "indexer_query":
            owner = dsv4_indexer_q_rope_quant_logical_work(spec)
            work = {name: owner[name] for name in ("mathematical_primitives", "mathematical_primitives_condition",
                "backend_alternatives", "selected_backend", "numerical_equivalence_asserted", "operands")}
        else:
            owner = dsv4_swa_insert_source_work(spec)
            work = {name: owner[name] for name in ("objects", "demand", "typed_actions", "read_write_bytes")}
            inp, _ = create_predictor(spec.kind).resource_input(spec, hardware_for_call(call, hardware))
            work["demand"] = {name: inp.demand_per_resource[name] for name in work["demand"]}
            work["read_write_bytes"] = {access: sum(node.bytes for path in inp.memory_paths if path.access_kind == access
                for node in path.nodes if node.resource_role == "l2_cache") for access in ("read", "write")}
        actual[key] = json.loads(json.dumps(work))
    checks = []
    for key in sorted(set(expected) | set(actual), key=str):
        want = None if key not in expected else json.loads(json.dumps(expected[key]["work"]))
        got = actual[key] if key in actual else None
        differences = (["unexpected"] if want is None else ["missing"] if got is None else
            [name for name in want if want[name] != got[name]])
        checks.append({"rank": key[0], "layer": key[1], "role": key[2], "differences": differences,
            "status": "matched" if not differences else "mismatched", "expected": want, "actual": got,
            "source": expected[key]["source"] if key in expected else None})
    return {"scope": reference["conditions"], "total": len(checks),
        "matched": sum(not row["differences"] for row in checks), "mismatched": sum(bool(row["differences"]) for row in checks),
        "not_checked": 0, "checks": checks, "sources": reference["sources"],
        "limitations": ("helper object programs and logical typed work, not physical pages or native launch/provider",
            "indexer-Q alternatives stay conditional; numerical equivalence is not assumed",
            "source typed comparisons/conversions and address/control issue services remain unpriced"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
