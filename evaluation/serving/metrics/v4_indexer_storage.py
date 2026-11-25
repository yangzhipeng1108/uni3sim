"""Common CUDA auto-selector workspace without choosing a score stride/backend."""

import ast
import re
from types import SimpleNamespace

from .qwen3_source import _Tensor


def _integer_expression(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _integer_expression(node.left) * _integer_expression(node.right)
    raise ValueError("indexer source workspace constant is not static integer multiplication")


def indexer_decode_storage_reference(source, raw, inputs):
    rows = sum(q for q, phase in zip(inputs["query_lengths"], inputs["request_phases"], strict=True) if phase == "decode")
    if not rows or max(inputs["kv_lengths"]) // 4 <= raw["index_topk"]:
        return []
    configured, = (node for node in source.classes["KernelConfig"].body if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "sparse_indexer_topk_backend")
    if ast.literal_eval(configured.value) != "auto":
        raise ValueError("V4 declared default indexer top-k source backend changed")
    constant, = (node for node in source.trees["indexer_topk.py"].body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RADIX_TOPK_WORKSPACE_SIZE" for target in node.targets))
    size = _integer_expression(constant.value)
    if size <= 0 or raw["index_topk"] not in (512, 1024, 2048):
        raise ValueError("V4 common CUDA auto-selector workspace domain changed")
    api, = (node for node in source.trees["indexer_score_api.py"].body
        if isinstance(node, ast.FunctionDef) and node.name == "fp8_fp4_paged_mqa_logits")
    documentation = re.sub(r"\s+", "", ast.get_docstring(api))
    if "Logitstensorofshape[B*next_n,max_model_len],dtype`torch.float32`" not in documentation:
        raise ValueError("DeepGEMM public FP32 score shape contract changed")
    path = source.root / "vllm/models/deepseek_v4/attention.py"
    tree = ast.parse(path.read_text())
    indexer, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4Indexer")
    constructor, = (node for node in indexer.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment, = (node for node in constructor.body if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.max_model_len" for target in node.targets))
    if ast.unparse(assignment.value) != "vllm_config.model_config.max_model_len // self.compress_ratio":
        raise ValueError("V4 indexer runtime context-width binding changed")
    maximum = inputs["source_max_model_len"]
    if type(maximum) is not int or maximum < max(inputs["kv_lengths"]):
        raise ValueError("decode score geometry needs an explicit runtime maximum covering all contexts")
    columns = maximum // 4
    outputs, requests, branches = [], [], []

    def allocate(*layouts):
        requests.append(tuple(layouts))
        if layouts != (((size,), "uint8"),):
            raise ValueError("CUDA auto top-k source common workspace request changed")
        tensor = _Tensor((size,), "uint8")
        outputs.append(tensor)
        return (tensor,)

    def consume(branch):
        def call(logits, lens, indices, workspace, topk, extent):
            if (logits.dtype != "fp32" or logits.shape != (rows, columns)
                    or lens.shape != (rows, 1) or indices.shape != (rows, raw["index_topk"])
                    or workspace is not outputs[-1] or topk != raw["index_topk"]):
                raise ValueError("source auto-selector FP32/INT32 storage handoff changed")
            branches.append(branch)
        return call

    globals = {"torch": SimpleNamespace(uint8="uint8", ops=SimpleNamespace(_C=SimpleNamespace(
        cooperative_topk=consume("cooperative"), persistent_topk=consume("persistent")))),
        "current_workspace_manager": lambda: SimpleNamespace(get_simultaneous=allocate),
        "RADIX_TOPK_WORKSPACE_SIZE": size}
    selector = SimpleNamespace(_backend="auto", _is_cuda=True, _cooperative_capable=True)
    for name in ("resolve_backend", "_resolve_auto", "_cooperative_constraints", "forward"):
        setattr(selector, name, source.forward("SparseIndexerTopk", name, metadata_globals=globals).__get__(selector))
    # For H200-source CUDA/auto and supported K the fallback is persistent,
    # regardless of which cooperative constraint failed. Exercise both row
    # stride residue classes; neither is asserted to be the loaded ABI.
    for row_stride in (columns + (-columns % 4), columns + (-columns % 4) + 1):
        selector.forward(_Tensor((rows, columns), "fp32", strides=(row_stride, 1)),
            _Tensor((rows, 1), "int32"), 1, _Tensor((rows, raw["index_topk"]), "int32"), raw["index_topk"],
            max(inputs["kv_lengths"]) // 4)
    possible = tuple(sorted(set(branches)))
    if possible != (("cooperative", "persistent") if rows <= 64 else ("persistent",)) or len(requests) != 2:
        raise ValueError("CUDA auto-selector branches no longer share the declared source workspace")
    bindings = [{"path": str(source.paths["indexer_topk.py"]), "symbol": "SparseIndexerTopk.forward",
        "line": source.method("SparseIndexerTopk", "forward").lineno},
        {"path": str(source.paths["indexer_kernel_config.py"]), "symbol": "KernelConfig.sparse_indexer_topk_backend", "line": configured.lineno},
        {"path": str(source.paths["indexer_score_api.py"]), "symbol": api.name, "line": api.lineno},
        {"path": str(path), "symbol": "DeepseekV4Indexer.__init__.max_model_len", "line": assignment.lineno}]
    return [{"rank": rank, "layer": layer, "shape": (size,), "dtype": "uint8", "bytes": size,
        "logical_score_shape": (rows, columns), "logical_score_dtype": "fp32", "possible_backends": possible,
        "source": bindings, "condition": "declared CUDA/H200 source, default auto, CP1, single-token decode and supported K; score stride/backend unbound"}
        for rank in range(inputs["TP"]) for layer, ratio in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]]) if ratio == 4]


def compare_v4_decode_selector_storage(reference, traces, graph):
    expected = {(row["rank"], row["layer"]): row for row in reference}
    if len(expected) != len(reference):
        raise ValueError("source decode selector storage repeats layer/rank")
    actual = {}
    for trace in traces:
        views = {view.view_id: view for view in trace.views}
        allocations = {row.allocation_id: row for row in trace.allocations}
        events = {event: i for i, event in enumerate(trace.event_order)}
        for node in graph.nodes:
            if node.rank != trace.rank or node.variant != "dsv4_indexer_decode":
                continue
            match = re.search(r"(?:^|:)layer-(\d+):attn:indexer-score-topk(?::\d+)?$", node.node_id)
            if match is None:
                continue
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("actual decode selector repeats layer/rank")
            request = node.attributes["symbolic_decode_selector_workspace"]
            got = None
            if request is not None:
                view = views[request["view_id"]]
                allocation = allocations[view.allocation_id]
                ends = [i for event, i in events.items() if event.endswith("return:" + node.node_id)]
                live = len(ends) == 1 and events[allocation.allocated_event] < ends[0]
                live = live and (allocation.freed_event is None or events[allocation.freed_event] > ends[0])
                got = {"shape": view.shape, "dtype": view.dtype, "bytes": allocation.size_bytes,
                    "offset_bytes": view.offset_bytes, "live_through_selector": live,
                    "logical_score_shape": node.attributes["symbolic_score_logical_shape"],
                    "logical_score_dtype": node.attributes["score_accumulator_dtype"],
                    "possible_backends": request["possible_backends"], "configured_backend": request["configured_backend"]}
            actual[key] = got
    checks = []
    for key in sorted(set(expected) | set(actual)):
        want = None
        if key in expected:
            row = expected[key]
            want = {field: row[field] for field in ("shape", "dtype", "logical_score_shape", "logical_score_dtype", "possible_backends")}
            want.update(offset_bytes=0, live_through_selector=True, configured_backend="auto", bytes_at_least=row["bytes"])
        got = actual[key] if key in actual else None
        if got is not None and want is not None:
            got = {**got, "bytes_at_least": got["bytes"]}
            differences = [name for name in want if (want[name] > got[name] if name == "bytes_at_least" else want[name] != got[name])]
        else:
            differences = ["unexpected"] if want is None else ["unbound_or_missing"]
        checks.append({"rank": key[0], "layer": key[1], "expected": want, "actual": got,
            "source": None if key not in expected else expected[key]["source"],
            "status": "mismatched" if differences else "matched", "differences": differences})
    return {"scope": "common CUDA auto decode-selector arena request and documented FP32 score logical geometry; score backing/strides and private kernel scratch remain unbound",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "complete_scored_storage_accepted": False, "runtime_accepted": False}
