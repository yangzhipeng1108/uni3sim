"""Original sparse prefill chunk/port program, excluding missing FlashMLA callees."""

import ast
from math import prod
import re
from types import SimpleNamespace

from .qwen3_source import _Tensor
from ..simulation.v4_flash_reference import c128_metadata_width_reference


SPARSE_STORAGE_SOURCE_FILES = {
    "flashmla.py": "vllm/models/deepseek_v4/nvidia/flashmla.py",
    "attention.py": "vllm/models/deepseek_v4/attention.py",
    "sparse_swa_storage.py": "vllm/v1/attention/backends/mla/sparse_swa.py",
    "workspace_manager_storage.py": "vllm/v1/worker/workspace.py",
    "decode_cache_storage.py": "vllm/models/deepseek_v4/common/ops/cache_utils.py",
    "sparse_mla_storage.py": "vllm/models/deepseek_v4/sparse_mla.py",
}


class _Integer(int):
    def item(self):
        return int(self)


class _Requests:
    """Only request-sized integer metadata; no tensor payload or model math."""
    def __init__(self, values):
        self.values = tuple(values)

    def __getitem__(self, index):
        return _Requests(self.values[index]) if isinstance(index, slice) else _Integer(self.values[index])

    def __sub__(self, other):
        return _Requests(a - b for a, b in zip(self.values, other.values, strict=True))

    def __add__(self, other):
        return _Requests(a + b for a, b in zip(self.values, other.values, strict=True))


def sparse_prefill_storage_reference(source, raw, inputs):
    queries = tuple(q for q, p in zip(inputs["query_lengths"], inputs["request_phases"], strict=True) if p == "prefill")
    lengths = tuple(n for n, p in zip(inputs["kv_lengths"], inputs["request_phases"], strict=True) if p == "prefill")
    if not queries:
        return []
    maximum, batch = inputs["source_max_model_len"], inputs["max_batched_tokens"]
    if type(maximum) is not int or maximum < max(lengths):
        raise ValueError("sparse prefill source storage needs its explicit runtime maximum")
    constant, = (node for node in source.classes["DeepseekV4Attention"].body if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "PREFILL_CHUNK_SIZE")
    chunk_size = ast.literal_eval(constant.value)
    function = source.method("DeepseekV4FlashMLAAttention", "_forward_prefill")
    callee, = (node for node in ast.walk(function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "flash_mla_sparse_fwd")
    arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in callee.keywords}
    if any(arguments[name] != expression for name, expression in (
            ("kv", "kv.view(-1, 1, q.shape[-1])"), ("indices", "combined_indices.unsqueeze(1)"),
            ("topk_length", "combined_lens"))):
        raise ValueError("sparse source callee view program changed")
    combine_call, = (node for node in ast.walk(function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "combine_topk_swa_indices")
    passed_out, = (keyword.value for keyword in combine_call.keywords if keyword.arg == "out")
    if not isinstance(passed_out, ast.Tuple) or tuple(ast.unparse(node) for node in passed_out.elts) != (
            "combined_indices_out", "combined_lens_out"):
        raise ValueError("sparse caller no longer passes its workspace views to combine")
    combine, = (node for node in source.trees["decode_cache_storage.py"].body
        if isinstance(node, ast.FunctionDef) and node.name == "combine_topk_swa_indices")
    out_branch, = (node for node in combine.body if isinstance(node, ast.If) and ast.unparse(node.test) == "out is None")
    supplied = out_branch.orelse[0]
    returned = combine.body[-1]
    if (not isinstance(supplied, ast.Assign) or len(supplied.targets) != 1
            or not isinstance(supplied.targets[0], ast.Tuple)
            or tuple(ast.unparse(node) for node in supplied.targets[0].elts) != ("combined_indices", "combined_lens")
            or ast.unparse(supplied.value) != "out" or not isinstance(returned, ast.Return)
            or not isinstance(returned.value, ast.Tuple)
            or tuple(ast.unparse(node) for node in returned.value.elts) != ("combined_indices", "combined_lens")):
        raise ValueError("sparse combine helper no longer returns supplied storage")
    request, = (node for node in ast.walk(function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "workspace_manager.get_simultaneous")
    combined, = (node.value for node in function.body if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "combined_topk" for target in node.targets))
    manager = source.method("WorkspaceManager", "get_simultaneous")
    align, = (node for node in ast.walk(manager) if isinstance(node, ast.Call) and ast.unparse(node.func) == "round_up")
    alignment = ast.literal_eval(align.args[1])
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("source workspace alignment must be a positive literal")
    def divide(value, ratio, *, rounding_mode):
        if rounding_mode != "floor":
            raise ValueError("sparse source compressed lengths need floor division")
        return _Requests(v // ratio for v in value.values)
    torch = SimpleNamespace(clamp=lambda value, *, min, max: _Requests(
        max if v > max else min if v < min else v for v in value.values),
        div=divide,
        zeros_like=lambda value: _Requests(0 for _ in value.values))
    planner = source.forward("DeepseekSparseSWAMetadata", "get_prefill_chunk_plan", metadata_globals={
        "torch": torch, "cdiv": lambda n, d: -(-n // d)})
    metadata = SimpleNamespace(num_prefills=len(queries), prefill_seq_lens_cpu=_Requests(lengths),
        prefill_query_lens_cpu=_Requests(queries), prefill_max_model_len=maximum,
        prefill_window_size=raw["sliding_window"], prefill_max_num_batched_tokens=batch)
    programs = {}
    for configured in sorted(set(raw["compress_ratios"][:raw["num_hidden_layers"]])):
        ratio = max(1, configured)
        if ratio in programs:
            continue
        plan = planner(metadata, ratio, chunk_size)
        topk = (0 if ratio == 1 else raw["index_topk"] if ratio == 4 else
            c128_metadata_width_reference(max(inputs["kv_lengths"]), maximum))
        namespace = {"self": SimpleNamespace(window_size=raw["sliding_window"], max_image_tokens=0,
            max_num_batched_tokens=batch), "q": SimpleNamespace(shape=(sum(queries), 64, raw["head_dim"])),
            "top_k": topk, "round_up": lambda n, d: -(-n // d) * d,
            "torch": SimpleNamespace(bfloat16="bf16", int32="int32")}
        namespace["combined_topk"] = eval(compile(ast.Expression(combined), "<source-combined-topk-shape>", "eval"),
            {"__builtins__": {}}, namespace)
        chunks = []
        for start, end, compressed, width in plan:
            captured = []
            namespace.update(chunk_size=end-start, chunk_M=width,
                workspace_manager=SimpleNamespace(get_simultaneous=lambda *ports: captured.append(ports)))
            eval(compile(ast.Expression(request), "<source-sparse-workspace-ports>", "eval"), {"__builtins__": {}}, namespace)
            if len(captured) != 1:
                raise ValueError("sparse source chunk must issue one simultaneous request")
            ports, = captured
            offsets, total = [], 0
            for shape, dtype in ports:
                offsets.append(total)
                size = prod(shape) * {"bf16": 2, "int32": 4}[dtype]
                total += -(-size // alignment) * alignment
            active = sum(queries[start:end])
            callee_shapes = (((end-start) * width, 1, raw["head_dim"]),
                (active, 1, namespace["combined_topk"]), (active,))
            chunks.append({"request_range": (start, end), "ports": ports, "offsets": tuple(offsets),
                "minimum_backing_bytes": total, "compressed_width": compressed, "gathered_width": width,
                "port_strides_bytes": tuple(tuple(s * {"bf16": 2, "int32": 4}[dtype]
                    for s in _Tensor._dense_strides(shape)) for shape, dtype in ports),
                "callee_views": callee_shapes,
                "callee_strides_bytes": tuple(tuple(s * size for s in _Tensor._dense_strides(shape))
                    for shape, size in zip(callee_shapes, (2, 4, 4), strict=True))})
        programs[ratio] = chunks
    bindings = [{"path": str(source.paths[file]), "symbol": symbol, "line": node.lineno}
        for file, symbol, node in (("sparse_swa_storage.py", "DeepseekSparseSWAMetadata.get_prefill_chunk_plan",
            source.method("DeepseekSparseSWAMetadata", "get_prefill_chunk_plan")),
            ("flashmla.py", "DeepseekV4FlashMLAAttention._forward_prefill", function),
            ("workspace_manager_storage.py", "WorkspaceManager.get_simultaneous", manager))]
    return [{"rank": rank, "layer": layer, "chunks": programs[max(1, ratio)], "source": bindings,
        "condition": "declared text-only prefill, no image visibility, synchronous source and explicit manager history"}
        for rank in range(inputs["TP"]) for layer, ratio in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]])]


def compare_v4_sparse_prefill_storage(reference, traces, graph):
    expected = {(row["rank"], row["layer"]): row for row in reference}
    if len(expected) != len(reference):
        raise ValueError("sparse prefill storage repeats layer/rank")
    actual = {}
    for trace in traces:
        views = {view.view_id: view for view in trace.views}
        allocations = {a.allocation_id: a for a in trace.allocations}
        events = {event: i for i, event in enumerate(trace.event_order)}
        for node in graph.nodes:
            if node.rank != trace.rank or node.variant != "dsv4_sparse_prefill":
                continue
            match = re.search(r"(?:^|:)layer-(\d+):attn:sparse(?::\d+)?$", node.node_id)
            if match is None:
                continue
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("actual sparse prefill repeats layer/rank")
            requests = node.attributes["symbolic_prefill_workspace_requests"]
            chunks, previous = [], None
            for request in (() if requests is None else requests):
                ports = tuple(views[v] for v in request["view_ids"])
                backing = {port.allocation_id for port in ports}
                shared = len(backing) == 1
                allocation = allocations[ports[0].allocation_id]
                retained = True
                if previous is not None and previous.allocation_id != allocation.allocation_id:
                    retained = previous.freed_event is not None and events[previous.freed_event] > events[allocation.allocated_event]
                aliases = tuple(views[v] for v in request["callee_view_ids"])
                chunks.append({"request_range": request["request_range"], "ports": tuple((p.shape, p.dtype) for p in ports),
                    "offsets": tuple(p.offset_bytes for p in ports), "backing_bytes": allocation.size_bytes,
                    "port_strides_bytes": tuple(p.strides_bytes for p in ports),
                    "shared_backing": shared, "prior_chunk_retained_during_growth": retained,
                    "compressed_width": request["compressed_width"], "gathered_width": request["gathered_width"],
                    "callee_views": tuple(v.shape for v in aliases),
                    "callee_strides_bytes": tuple(v.strides_bytes for v in aliases),
                    "callee_aliases": all(v.allocation_id == p.allocation_id and v.offset_bytes == p.offset_bytes
                        and v.dtype == p.dtype for v, p in zip(aliases, ports, strict=True))})
                previous = allocation
            if chunks:
                ends = [i for event, i in events.items() if event.endswith("return:" + node.node_id)]
                chunks[-1]["live_through_call"] = len(ends) == 1 and events[previous.allocated_event] < ends[0]
                chunks[-1]["live_through_call"] = chunks[-1]["live_through_call"] and (
                    previous.freed_event is None or events[previous.freed_event] > ends[0])
            actual[key] = None if requests is None else chunks
    checks = []
    for key in sorted(set(expected) | set(actual)):
        want = expected[key]["chunks"] if key in expected else None
        got = actual[key] if key in actual else None
        differences = []
        if want is None or got is None or len(want) != len(got):
            differences.append("missing_unbound_or_extra_chunks")
        else:
            for index, (a, b) in enumerate(zip(want, got, strict=True)):
                fields = [name for name in a if name != "minimum_backing_bytes" and a[name] != b[name]]
                fields += [name for name in ("shared_backing", "prior_chunk_retained_during_growth", "callee_aliases") if not b[name]]
                if b["backing_bytes"] < a["minimum_backing_bytes"]:
                    fields.append("backing_bytes")
                if index == len(want) - 1 and not b["live_through_call"]:
                    fields.append("live_through_call")
                differences.extend(f"chunk-{index}:{name}" for name in fields)
        checks.append({"rank": key[0], "layer": key[1], "expected": want, "actual": got,
            "source": None if key not in expected else expected[key]["source"],
            "differences": differences, "status": "mismatched" if differences else "matched"})
    return {"scope": "original sparse prefill chunk/typed ports, shared backing/offsets, callee views and growth retention; FlashMLA-returned statistics/planner storage remains separate",
        "total": len(checks), "matched": sum(not r["differences"] for r in checks),
        "mismatched": sum(bool(r["differences"]) for r in checks), "not_checked": 0, "checks": checks,
        "complete_sparse_storage_accepted": False, "runtime_accepted": False}


def sparse_decode_view_reference(source, raw, inputs):
    count = sum(q for q, phase in zip(inputs["query_lengths"], inputs["request_phases"], strict=True) if phase == "decode")
    if not count:
        return []
    function = source.method("DeepseekV4FlashMLAAttention", "_forward_decode")
    q_assignments = [ast.unparse(node.value) for node in function.body if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "q" for target in node.targets)]
    call, = (node for node in ast.walk(function) if isinstance(node, ast.Call) and ast.unparse(node.func) == "flash_mla_with_kvcache")
    arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    c4 = [node for node in ast.walk(function) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "global_indices.view"]
    c128 = [node.value for node in ast.walk(source.trees["sparse_mla_storage.py"]) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
            and target.slice.value == "c128a_global_decode_topk_indices" for target in node.targets)]
    if (q_assignments != ["q.unsqueeze(1)"] or arguments["q"] != "q" or arguments["out"] != "output.unsqueeze(1)"
            or len(c4) != 1 or ast.unparse(c4[0]) != "global_indices.view(num_decode_tokens, 1, -1)"
            or len(c128) != 1 or ast.unparse(c128[0]) != "global_decode.view(num_decode_tokens, 1, -1)"):
        raise ValueError("sparse decode Q/output/compressed-index view program changed")
    padded = source.forward("DeepseekV4FlashMLAAttention", "get_padded_num_q_heads")(
        None, raw["num_attention_heads"] // inputs["TP"])
    dimension = raw["head_dim"]
    maximum = inputs["source_max_model_len"]
    if type(maximum) is not int or maximum < max(inputs["kv_lengths"]):
        raise ValueError("sparse decode metadata views need an explicit runtime maximum")
    width128 = c128_metadata_width_reference(max(inputs["kv_lengths"]), maximum)
    states128 = -(-maximum // 128)
    capacity128 = -(-states128 // 128) * 128
    records = []
    for rank in range(inputs["TP"]):
        for layer, ratio in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]]):
            views = {role: {"shape": (count, 1, padded, dimension), "dtype": "bf16",
                "strides_bytes": (padded * dimension * 2, padded * dimension * 2, dimension * 2, 2)}
                for role in ("query", "output")}
            if ratio > 1:
                width, stride = (raw["index_topk"], raw["index_topk"]) if ratio == 4 else (width128, capacity128)
                views["compressed_indices"] = {"shape": (count, 1, width), "dtype": "int32",
                    "strides_bytes": (stride * 4, width * 4, 4)}
            records.append({"rank": rank, "layer": layer, "views": views,
                "source": [{"path": str(source.paths["flashmla.py"]), "symbol": "DeepseekV4FlashMLAAttention._forward_decode", "line": function.lineno},
                    {"path": str(source.paths["sparse_mla_storage.py"]), "symbol": "c128a_global_decode_topk_indices.view", "line": c128[0].lineno}]})
    return records


def compare_v4_sparse_decode_views(reference, traces, graph):
    expected = {(row["rank"], row["layer"]): row for row in reference}
    if len(expected) != len(reference):
        raise ValueError("sparse decode view reference repeats layer/rank")
    actual = {}
    for trace in traces:
        views = {view.view_id: view for view in trace.views}
        for node in graph.nodes:
            if node.rank != trace.rank or node.variant != "dsv4_sparse_decode":
                continue
            match = re.search(r"(?:^|:)layer-(\d+):attn:sparse(?::\d+)?$", node.node_id)
            if match is None:
                continue
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("actual sparse decode views repeat layer/rank")
            parents = {"query": views[node.attributes["symbolic_tensor_inputs"][0]],
                "output": views[node.attributes["symbolic_tensor_outputs"][0]]}
            ratio = node.shape["compress_ratio"]
            if ratio > 1:
                parent = (node.attributes["symbolic_decode_index_buffers"][0] if ratio == 4 else
                    node.attributes["symbolic_tensor_inputs"][-1])
                parents["compressed_indices"] = views[parent]
            got = {}
            for role, name in node.attributes["symbolic_decode_callee_views"].items():
                view = views[name]
                got[role] = {"shape": view.shape, "dtype": view.dtype, "strides_bytes": view.strides_bytes,
                    "aliases_parent": role in parents and view.allocation_id == parents[role].allocation_id
                        and view.offset_bytes == parents[role].offset_bytes}
            actual[key] = got
    checks = []
    for key in sorted(set(expected) | set(actual)):
        want = None if key not in expected else {role: {**value, "aliases_parent": True}
            for role, value in expected[key]["views"].items()}
        got = actual[key] if key in actual else None
        differences = (["unexpected"] if want is None else ["missing"] if got is None else
            ["view_roles"] if set(want) != set(got) else [role for role in want if want[role] != got[role]])
        checks.append({"rank": key[0], "layer": key[1], "expected": want, "actual": got,
            "source": None if key not in expected else expected[key]["source"], "differences": differences,
            "status": "mismatched" if differences else "matched"})
    return {"scope": "decode Q/output singleton and C4/C128 index views, typed shape/stride/backing aliases; physical paged cache axes and missing FlashMLA statistics/planner are separate",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "complete_sparse_storage_accepted": False, "runtime_accepted": False}
