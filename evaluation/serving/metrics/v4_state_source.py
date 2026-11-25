"""Independent V4 logical cache ports and TP reduction contracts.

Logical cache rows are an explicit input domain. They do not describe a live
allocator's pages, copy-on-write, GPU fences or numerical cache contents.
"""

from math import prod
from pathlib import Path
import ast
import re

from ..simulation.v4_flash_reference import _function_anchor, _definition


def logical_state_reference(raw, source_root, *, query_lengths, kv_lengths,
                            request_phases, capacity, tp):
    """Enumerate source cache owners without production graphs or predictors."""
    if (tp not in (4, 8) or not query_lengths or len(query_lengths) != len(kv_lengths)
            or len(request_phases) != len(kv_lengths)
            or any(q <= 0 or n < q for q, n in zip(query_lengths, kv_lengths, strict=True))
            or capacity < sum(kv_lengths) or capacity % 128):
        raise ValueError("V4 logical state needs TP4/8 and complete, aligned declared cache rows")
    root = Path(source_root)
    bindings = {
        "compressor": _function_anchor(root, "vllm/models/deepseek_v4/compressor.py", "DeepseekCompressor.__init__"),
        "save_finish": _function_anchor(root, "vllm/models/deepseek_v4/compressor.py", "DeepseekCompressor.forward"),
        "sparse": _function_anchor(root, "vllm/models/deepseek_v4/nvidia/flashmla.py", "DeepseekV4FlashMLAAttention.forward_mqa"),
        "embedding_reduce": _function_anchor(root, "vllm/model_executor/layers/vocab_parallel_embedding.py", "VocabParallelEmbedding.forward"),
        "output_reduce": _function_anchor(root, "vllm/model_executor/layers/linear.py", "RowParallelLinear.forward"),
        "ffn_reduce": _function_anchor(root, "vllm/model_executor/layers/fused_moe/runner/moe_runner.py", "MoERunner._maybe_reduce_final_output"),
    }
    anchor = bindings["compressor"]
    constructor = re.sub(r"\s+", "", ast.unparse(_definition(
        ast.parse(Path(anchor["path"]).read_text()), anchor["symbol"])))
    if any(fragment not in constructor for fragment in (
            "self.overlap=compress_ratio==4", "self.coff=1+self.overlap", "state_dtype=torch.float32",
            "state_dim=2*self.coff*self.head_dim", "dtype=state_dtype")):
        raise ValueError("V4 compressor state geometry/type changed")
    active_phases = tuple(p for p in ("prefill", "decode") if p in request_phases)
    resources = []
    for rank in range(tp):
        for layer, configured in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]]):
            ratio = max(1, configured)
            prefix = f"layer-{layer}:attn:"
            sparse = [prefix + "sparse" + (f":{i}" if i else "") for i in range(len(active_phases))]

            def resource(role, shape, dtype, writers, readers):
                resources.append({"rank": rank, "layer": layer, "role": role,
                    "name": prefix + role, "shape": tuple(shape), "dtype": dtype,
                    "writers": sorted(writers), "readers": sorted(readers),
                    "persistent": True, "source": "sparse" if role == "swa-cache" else "save_finish"})

            resource("swa-cache", (capacity, 584), "uint8", [prefix + "swa-insert"], sparse)
            if ratio == 1:
                continue
            boundaries = sum(n // ratio - (n - q) // ratio
                for q, n in zip(query_lengths, kv_lengths, strict=True))
            block, coff = (4, 2) if ratio == 4 else (8, 1)
            finish = ratio == 4 or boundaries > 0
            main = prefix + "compressor:"
            finish_name = main + "finish" + (":store" if ratio == 128 else "")
            state_reader = main + "finish" + (":compress" if ratio == 128 else "")
            resource("partial-state", (capacity // block, block, 2 * coff * raw["head_dim"]), "fp32",
                [main + "save-partial"], [state_reader] if finish else [])
            resource("compressed-cache", (capacity // ratio, 584), "uint8", [finish_name] if finish else [], sparse)
            if ratio == 4:
                indexer = prefix + "indexer-compressor:"
                resource("indexer-partial-state", (capacity // 4, 4, 4 * raw["index_head_dim"]), "fp32",
                    [indexer + "save-partial"], [indexer + "finish"])
                long = max(kv_lengths) // 4 > raw["index_topk"]
                score = [prefix + "indexer-score-topk" + (f":{i}" if i else "")
                    for i, phase in enumerate(active_phases)
                    if long and (phase != "prefill" or any(n // 4 for n, p in zip(kv_lengths, request_phases, strict=True) if p == phase))]
                resource("indexer-cache", (capacity // 4, 132), "uint8", [indexer + "finish"], score)
    tokens, hidden = sum(query_lengths), raw["hidden_size"]
    reductions = [{"name": "embedding:all-reduce", "source": "embedding_reduce"}]
    reductions += [{"name": f"layer-{layer}:" + suffix, "source": source}
        for layer in range(raw["num_hidden_layers"])
        for suffix, source in (("attn:out:all-reduce", "output_reduce"), ("ffn:all-reduce", "ffn_reduce"))]
    for row in reductions:
        row["contract"] = {"operation": "all_reduce", "participants": tuple(range(tp)),
            "reduction": "sum", "bytes": tokens * hidden * 2, "bytes_semantics": "per_rank_input",
            "symbolic_collective_storage": "fresh", "dtype": "bf16",
            "rank_input_shapes": ((tokens, hidden),) * tp, "rank_output_shapes": ((tokens, hidden),) * tp,
            "rank_output_aliases_input": (False,) * tp}
    starts = tuple(n - q for q, n in zip(query_lengths, kv_lengths, strict=True))
    return {"resources": resources, "reductions": reductions, "sources": bindings,
        "position_intervals": tuple(zip(starts, kv_lengths, strict=True)),
        "scope": "declared logical-row cache layouts and same-forward access ports; TP logical sum payload/storage",
        "physical_cache_or_provider_accepted": False}


def compare_v4_logical_state(reference, traces, graph):
    """Read recorded views/ports, including untouched persistent state owners."""
    actual = {}
    view_index = {view.view_id: view for trace in traces for view in trace.views}
    accesses = {}
    for trace in traces:
        accesses[trace.rank] = ({}, {})
    for node in graph.nodes:
        if node.rank not in accesses:
            continue
        namespace = next(t.namespace for t in traces if t.rank == node.rank)
        for field, index in (("symbolic_tensor_inputs", 0), ("symbolic_tensor_outputs", 1)):
            if field not in node.attributes:
                continue
            for allocation in {view_index[v].allocation_id for v in node.attributes[field]}:
                account = accesses[node.rank][index]
                if allocation not in account:
                    account[allocation] = []
                account[allocation].append(node.node_id.removeprefix(namespace + ":"))
    for trace in traces:
        activation = {a.allocation_id for a in trace.activation_allocations}
        for expected in reference["resources"]:
            if expected["rank"] != trace.rank:
                continue
            identifier = trace.qualify(expected["name"])
            if identifier not in view_index:
                continue
            view = view_index[identifier]
            readers, writers = tuple(account[view.allocation_id] if view.allocation_id in account else []
                for account in accesses[trace.rank])
            actual[trace.rank, expected["name"]] = {"shape": view.shape, "dtype": view.dtype,
                "readers": sorted(readers), "writers": sorted(writers),
                "persistent": view.allocation_id not in activation}
    checks = []
    fields = ("shape", "dtype", "readers", "writers", "persistent")
    for expected in reference["resources"]:
        key = expected["rank"], expected["name"]
        want = {name: expected[name] for name in fields}
        got = actual[key] if key in actual else None
        differences = ["missing"] if got is None else [name for name in fields if want[name] != got[name]]
        checks.append({"rank": key[0], "name": key[1], "expected": want, "actual": got,
            "differences": differences, "status": "mismatched" if differences else "matched",
            "source": reference["sources"][expected["source"]]})
    return _comparison(reference["scope"], checks)


def compare_v4_reductions(reference, traces, graph):
    views = {view.view_id: view for trace in traces for view in trace.views}
    actual = {}
    for node in graph.nodes:
        if node.kind.value != "collective" or node.attributes["operation"] != "all_reduce":
            continue
        # Joined collective IDs use the step namespace, not a rank namespace.
        name = node.node_id.split(":", 1)[1]
        ports = tuple(tuple(views[v] for v in node.attributes[field])
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"))
        incoming, outgoing = ports
        actual[name] = {field: node.attributes[field] for field in (
            "operation", "participants", "reduction", "bytes", "bytes_semantics", "symbolic_collective_storage")}
        actual[name].update(dtype=node.dtype, rank_input_shapes=tuple(v.shape for v in incoming),
            rank_output_shapes=tuple(v.shape for v in outgoing),
            rank_output_aliases_input=tuple(a.allocation_id == b.allocation_id for a, b in zip(incoming, outgoing, strict=True)))
        if any(prod(v.shape) * v.element_bytes != node.attributes["bytes"] for v in (*incoming, *outgoing)):
            actual[name]["bytes"] = None
    expected = {r["name"]: r for r in reference["reductions"]}
    checks = []
    for name in sorted(set(expected) | set(actual)):
        want = expected[name]["contract"] if name in expected else None
        got = actual[name] if name in actual else None
        differences = (["unexpected"] if want is None else ["missing"] if got is None else
            [field for field in want if want[field] != got[field]])
        checks.append({"name": name, "expected": want, "actual": got, "differences": differences,
            "status": "mismatched" if differences else "matched",
            "source": None if name not in expected else reference["sources"][expected[name]["source"]]})
    return _comparison("all source TP sum collectives, per-rank input/output shape, dtype, payload and fresh storage", checks)


def _comparison(scope, checks):
    return {"scope": scope, "total": len(checks), "matched": sum(not r["differences"] for r in checks),
        "mismatched": sum(bool(r["differences"]) for r in checks), "not_checked": 0, "checks": checks,
        "whole_model_source_alignment_accepted": False, "physical_cache_or_provider_accepted": False,
        "runtime_accepted": False}


def compare_v4_logical_sequence(declarations, results, *, initial_prefixes):
    """Check consecutive declared requests against actual source port reports.

    Step-local row numbers may change when the scheduler packs requests.
    Identity here is (request, position, rank, layer, cache role); physical
    block tables and transfers between physical slots remain external inputs.
    """
    if len(declarations) != len(results) or not declarations:
        raise ValueError("logical sequence needs one result per declared nonempty step")
    ends = dict(initial_prefixes)
    checks, prior_resources = [], None
    for declaration, result in zip(declarations, results, strict=True):
        requests = declaration["requests"]
        identities = [r["request_id"] for r in requests]
        if not requests or len(set(identities)) != len(identities):
            raise ValueError("logical sequence repeats a request within a step")
        query = tuple(r["query"] for r in requests)
        context = tuple(r["computed"] for r in requests)
        differences = []
        for request in requests:
            identity = request["request_id"]
            expected_start = ends[identity] if identity in ends else 0
            if request["computed"] != expected_start:
                differences.append("request_continuity:" + identity)
            ends[identity] = request["computed"] + request["query"]
        if query != tuple(result["query_lengths"]) or context != tuple(result["context_lengths"]):
            differences.append("actual_step_inputs")
        state = result["source_v4_logical_state_comparison"]
        if state["mismatched"] or state["not_checked"] or not state["total"]:
            differences.append("cache_layout_or_ports")
        # Layout identities and types must survive re-created metadata graphs;
        # changing a capacity is allowed, changing the logical owner is not.
        resources = sorted((r["rank"], r["name"], r["actual"]["dtype"],
            tuple(r["actual"]["shape"])[1:]) for r in state["checks"] if r["actual"] is not None)
        if prior_resources is not None and resources != prior_resources:
            differences.append("logical_cache_owner_or_type_changed")
        prior_resources = resources
        checks.append({"step": declaration["step_id"], "requests": requests,
            "status": "mismatched" if differences else "matched", "differences": differences,
            "expected": "consecutive request positions and persistent logical cache owner/type; every step's source access-port check passes",
            "actual": {"query_lengths": query, "context_lengths": context,
                "logical_cache_resources": len(resources), "checked_cache_ports": state["matched"]}})
    comparison = _comparison("consecutive logical request/cache dataflow on explicitly initialized prefixes", checks)
    comparison.update(initial_prefixes=initial_prefixes, final_prefixes=ends,
        physical_slot_continuity_accepted=False, numerical_cache_contents_checked=False,
        limitation="logical state only; physical mappings, remap/copy-on-write, provider and normal capacity are not inferred")
    return comparison
