"""Original V4 attention math, independent of production workload/pricing."""

import ast
import re

from .v4_indexer_source import _definition


class AttentionMathSource:
    def __init__(self, source, raw, rows, *, indexer_selection_source):
        self.raw, self.rows, self.selection = raw, rows, indexer_selection_source
        function = _definition(source, "indexer_math_reference.py", "_flash_mla_eager")
        expected = {
            "scores": ("torch.einsum('thd,tkd->thk', q_f32, kv) * scale",
                "scores.masked_fill(~valid.unsqueeze(1), float('-inf'))"),
            "sink": ("attn_sink.view(1, num_heads, 1).expand(num_tokens, num_heads, 1)",),
            "scores_full": ("torch.cat([scores, sink], dim=-1)",),
            "probs": ("torch.softmax(scores_full, dim=-1)",),
            "out": ("torch.einsum('thk,tkd->thd', probs[..., :-1], kv)",)}
        for target, expressions in expected.items():
            values = tuple(ast.unparse(node.value) for node in function.body
                if isinstance(node, ast.Assign) and any(ast.unparse(name) == target for name in node.targets))
            if values != expressions:
                raise ValueError("V4 scaled QK/sink softmax/PV mathematical reference changed: " + target)
        self.binding = {"math": {"path": str(source.paths["indexer_math_reference.py"]),
            "symbol": function.name, "line": function.lineno}, "wrappers": []}
        for method, callee, scale in (("_forward_prefill", "flash_mla_sparse_fwd", "sm_scale"),
                ("_forward_decode", "flash_mla_with_kvcache", "softmax_scale")):
            body = source.method("DeepseekV4FlashMLAAttention", method)
            call, = (node for node in ast.walk(body) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee)
            keywords = {item.arg: ast.unparse(item.value) for item in call.keywords}
            if keywords[scale] != "self.scale" or keywords["attn_sink"] != "self.attn_sink":
                raise ValueError("V4 attention wrapper must pass its scale and learned sink")
            self.binding["wrappers"].append({"path": str(source.paths["flashmla.py"]),
                "symbol": "DeepseekV4FlashMLAAttention." + method, "callee": callee, "line": call.lineno})

    def record(self, rank, layer, ratio, heads, query_lengths, kv_lengths, phases):
        raw = self.raw
        if ratio == 4 and max(kv_lengths) // 4 > raw["index_topk"] and self.selection != "declared_causal_topk":
            raise ValueError("attention math requires explicit causal selected cardinality for long C4")
        for phase in ("prefill", "decode"):
            requests = [(q, length) for q, length, value in zip(query_lengths, kv_lengths, phases, strict=True)
                if value == phase]
            if not requests:
                continue
            positions = [position for q, length in requests for position in range(length - q + 1, length + 1)]
            window = sum(min(position, raw["sliding_window"]) for position in positions)
            compressed = sum(min(raw["index_topk"], position // 4) if ratio == 4 else position // ratio
                for position in positions) if ratio > 1 else 0
            pairs = (window + compressed) * heads
            rows = len(positions) * heads
            softmax = pairs + rows
            gathered = (sum(q + min(length - q, raw["sliding_window"] - 1) for q, length in requests)
                + (sum(length // ratio for _, length in requests) if ratio > 1 else 0)) if phase == "prefill" else 0
            gather_multiply = gathered * (raw["head_dim"] - raw["qk_rope_head_dim"])
            primitives = {"qk_dot_flops": 2 * pairs * raw["head_dim"], "qk_scale_multiply": pairs,
                "softmax_shift_subtract": softmax, "softmax_exp": softmax,
                "softmax_sum_add": pairs, "softmax_normalize_divide": softmax,
                "pv_dot_flops": 2 * pairs * raw["head_dim"], "gather_dequant_multiply": gather_multiply}
            self.rows.append({"rank": rank, "layer": layer, "phase": phase,
                "pair_capacities": {"compressed": compressed, "swa": window},
                "score_pairs": pairs, "softmax_rows": rows, "softmax_values_including_sink": softmax,
                "mathematical_primitives": primitives,
                "typed_actions": {"fp32_softmax_max_comparisons": pairs},
                "demand": {"compute_tensor": 4 * pairs * raw["head_dim"], "compute_vector": 0,
                    "compute_reduction": 0, "compute_fma": 4 * pairs + 2 * rows + gather_multiply,
                    "compute_sfu": softmax, "compute_alu": 0},
                "source": self.binding,
                "scope": "declared causal useful mathematical program including sink; stable-softmax decomposition, not numerical library or native issue equivalence"})


def compare_v4_attention_math(reference, plan):
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.sparse_attn import dsv4_attention_roofline_work

    names = ("pair_capacities", "score_pairs", "softmax_rows", "softmax_values_including_sink",
        "mathematical_primitives", "typed_actions", "demand")
    expected, actual, sources = {}, {}, {}
    for row in reference["attention_math_work"]:
        key = row["rank"], row["layer"], row["phase"]
        if key in expected:
            raise ValueError("source attention repeats a rank/layer/phase")
        expected[key], sources[key] = {name: row[name] for name in names}, row["source"]
    for call in plan.calls:
        if call.kernel_spec.kind is not KernelKind.SPARSE_ATTN or call.kernel_spec.variant not in (
                "dsv4_sparse_prefill", "dsv4_sparse_decode"):
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:sparse(?::\d+)?$", name) for name in call.source_node_ids]
        match, = (match for match in matches if match is not None)
        key = call.rank, int(match[1]), call.kernel_spec.variant.removeprefix("dsv4_sparse_")
        if key in actual:
            raise ValueError("actual attention repeats a rank/layer/phase")
        work = dsv4_attention_roofline_work(call.kernel_spec)
        actual[key] = {name: work[name] for name in names}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [name for name in names if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "phase": key[2], "differences": differences,
            "status": "mismatched" if differences else "matched", "expected": expected[key] if key in expected else None,
            "actual": actual[key] if key in actual else None, "source": sources[key] if key in sources else None})
    return {"scope": "scaled QK, one learned sink per softmax row and non-sink PV mathematical baseline",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("not FlashMLA online/split/padded issue or numerical equivalence",
            "cache/indices/metadata and scratch byte completeness remain separate",
            "floating comparisons unpriced; no actual provider acceptance"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
