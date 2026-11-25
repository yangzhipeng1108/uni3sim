"""Independent useful indexer stages bound to the original source program.

Only mathematical role uses are accepted here. Query partition, score backing,
selector refinement, physical pages and a loaded CUDA provider stay unbound.
"""

import ast
from copy import deepcopy
import re

from evaluation.serving.simulation.v4_flash_indexer_reference import indexer_baseline_reference
from evaluation.serving.simulation.v4_flash_reference import _cpp_anchor, _false_constexpr_body


INDEXER_SOURCE_FILES = {
    "sparse_attn_indexer.py": "vllm/model_executor/layers/sparse_attn_indexer.py",
    "indexer_math_reference.py": "tests/kernels/test_deepseek_v4_cpu_kernels.py",
    "indexer_topk.py": "vllm/model_executor/layers/indexer_topk.py",
    "indexer_kernel_config.py": "vllm/config/kernel.py",
    "indexer_score_api.py": "vllm/utils/deep_gemm.py",
}


def _definition(source, file, name):
    node, = (node for node in source.trees[file].body
        if isinstance(node, ast.FunctionDef) and node.name == name)
    return node


class IndexerWorkSource:
    def __init__(self, source, raw, rows):
        self.raw, self.rows = raw, rows
        function = _definition(source, "sparse_attn_indexer.py", "sparse_attn_indexer")
        math = _definition(source, "indexer_math_reference.py", "_reference_paged_mqa_logits")
        loops = [node for node in math.body if isinstance(node, ast.For)]
        if len(loops) != 1 or ast.unparse(loops[0].iter) != "range(batch_size)":
            raise ValueError("indexer mathematical reference batch program changed")
        expected = (
            ("k", "ref_k[b].float()"), ("scale", "ref_scale[b]"),
            ("q", "q_fp8[b].float()"), ("w", "weight[b]"),
            ("dot", "torch.einsum('jd,hd->jh', k, q).clamp(min=0.0)"),
            ("score", "(dot * w.unsqueeze(0)).sum(dim=1)"),
            ("out[b, :k.shape[0]]", "score * scale"))
        statements = loops[0].body
        if len(statements) != len(expected) or any(not isinstance(node, ast.Assign)
                or len(node.targets) != 1 or (ast.unparse(node.targets[0]), ast.unparse(node.value)) != pair
                for node, pair in zip(statements, expected, strict=True)):
            raise ValueError("indexer weighted ReLU/head sum/key-scale mathematical program changed")
        forward = source.method("SparseAttnIndexer", "forward_cuda")
        cuda_call, = (node for node in ast.walk(forward) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "torch.ops.vllm.sparse_attn_indexer")
        parameters = {parameter.arg: argument for parameter, argument in
            zip(function.args.args, cuda_call.args)}
        if ast.literal_eval(parameters["skip_topk_buffer_clear"]) is not False:
            raise ValueError("V4 CUDA indexer must retain the independent sentinel producer")
        clear, = (node for node in ast.walk(function) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not skip_topk_buffer_clear")
        if (len(clear.body) != 1 or not isinstance(clear.body[0], ast.Assign)
                or ast.unparse(clear.body[0].targets[0]) != "topk_indices_buffer[:hidden_states.shape[0]]"
                or ast.literal_eval(clear.body[0].value) != -1):
            raise ValueError("indexer sentinel initialization region changed")
        score_lines = []
        for symbol, bounds in (("fp8_fp4_mqa_logits", ("cu_seqlen_ks", "cu_seqlen_ke")),
                ("fp8_fp4_paged_mqa_logits", ("seq_lens",))):
            call, = (node for node in ast.walk(function) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == symbol)
            if (tuple(ast.unparse(arg) for arg in call.args[3:3 + len(bounds)]) != bounds
                    or {item.arg: ast.unparse(item.value) for item in call.keywords}["clean_logits"] != "False"):
                raise ValueError("indexer FP32 score bounds/cleanup interface changed")
            score_lines.append(call.lineno)
        for symbol in ("ops.top_k_per_row_prefill", "get_indexer_topk(topk_backend)"):
            call, = (node for node in ast.walk(function) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == symbol)
            if ast.unparse(call.args[0]) != "logits":
                raise ValueError("indexer selector must consume the original FP32 score object")
        empty, = (node for node in ast.walk(function) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0")
        if tuple(ast.unparse(node) for node in empty.body) != (
                "logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)",
                "topk_indices.fill_(-1)"):
            raise ValueError("indexer zero-context prefill producer changed")
        shortcut = _cpp_anchor(source.root, "csrc/libtorch_stable/sampler.cu", "if (rowLen <= topK)")
        body = _false_constexpr_body(shortcut["source"], "multipleBlocksPerRow")
        if re.search(r"\b(?:logits|outLogits)\b", body) or "return;" not in body:
            raise ValueError("indexer prefill shortcut must return without reading scores")
        kernel = _cpp_anchor(source.root, "csrc/libtorch_stable/sampler.cu",
            "static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(")
        job = _cpp_anchor(source.root, "csrc/libtorch_stable/sampler.cu",
            "template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,")
        if ("bool multipleBlocksPerRow = false" not in job["source"]
                or "topKPerRowJob<kNumThreadsPerBlock,kNumBins,useRadixSort>(" not in re.sub(r"\s+", "", kernel["source"])):
            raise ValueError("indexer prefill no longer specializes the unsplit shortcut")
        self.binding = {"math": {"path": str(source.paths["indexer_math_reference.py"]),
            "symbol": math.name, "line": math.lineno, "expression_lines": [node.lineno for node in statements]},
            "wrapper": {"path": str(source.paths["sparse_attn_indexer.py"]), "symbol": function.name,
                "line": function.lineno, "clear_line": clear.lineno, "score_lines": score_lines,
                "empty_context_line": empty.lineno},
            "selector_shortcut": {name: shortcut[name] for name in ("path", "line")}}

    def record(self, rank, layer, query_lengths, kv_lengths, phases):
        for phase in ("prefill", "decode"):
            requests = [{"query": q, "computed": length - q}
                for q, length, value in zip(query_lengths, kv_lengths, phases, strict=True) if value == phase]
            if requests:
                self.rows.append({"rank": rank, "layer": layer, "phase": phase,
                    **indexer_baseline_reference(self.raw, requests, phase), "source": self.binding})


def compare_v4_indexer_work(reference, plan, hardware):
    """Original caller inventory vs production mathematical stages/payload."""
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.sparse_attn_indexer import dsv4_indexer_logical_work

    names = ("compressed_row_ends", "active_pairs", "arithmetic", "stages", "demand", "read_write_bytes",
        "logical_bytes", "prefill_gather", "prefill_selection", "score_metadata", "score_materialization", "source_semantics")
    expected, sources = {}, {}
    for row in reference["indexer_work"]:
        key = row["rank"], row["layer"], row["phase"]
        if key in expected:
            raise ValueError("source indexer work repeats a rank/layer/phase")
        expected[key], sources[key] = {name: row[name] for name in names}, row["source"]
    actual = {}
    for call in plan.calls:
        if call.kernel_spec.kind is not KernelKind.SPARSE_ATTN_INDEXER or call.kernel_spec.variant not in (
                "dsv4_indexer_prefill", "dsv4_indexer_decode"):
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:indexer-score-topk(?::\d+)?$", name)
            for name in call.source_node_ids]
        match, = (match for match in matches if match is not None)
        key = call.rank, int(match[1]), call.kernel_spec.variant.removeprefix("dsv4_indexer_")
        if key in actual:
            raise ValueError("actual indexer repeats a rank/layer/phase")
        work = dsv4_indexer_logical_work(call.kernel_spec)
        actual[key] = {name: deepcopy(work[name]) for name in names}
    rows = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [name for name in names if expected[key][name] != actual[key][name]])
        rows.append({"rank": key[0], "layer": key[1], "phase": key[2],
            "status": "mismatched" if differences else "matched", "differences": differences,
            "expected": expected[key] if key in expected else None,
            "actual": actual[key] if key in actual else None, "source": sources[key] if key in sources else None})
    return {"scope": "scored indexer useful mathematical stages and typed logical payload only",
        "total": len(rows), "matched": sum(not row["differences"] for row in rows),
        "mismatched": sum(bool(row["differences"]) for row in rows), "not_checked": 0, "checks": rows,
        "limitations": ("declared causal selected cardinality does not prove actual top-k IDs",
            "physical score backing, query partition and page metadata remain unbound",
            "selector refinement, native instruction service and provider remain unbound"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
