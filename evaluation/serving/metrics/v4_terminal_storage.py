"""Source Tensor API programs for TP embedding, CUDA norm and greedy suffix."""

from math import prod
from types import SimpleNamespace

from src.kernel.memory_access import region_end

from .qwen3_source import _Ledger, _Tensor


def terminal_storage_reference(source, raw, reference):
    """Execute original wrappers on metadata, never production graph builders."""
    tp = reference["declared_inputs"]["TP"]
    hidden, vocab = raw["hidden_size"], raw["vocab_size"]
    # TargetSource already checked the genuine pad_vocab_size constructor.
    # This branch has no added/padded vocabulary, so every shard is equal.
    if vocab % tp or vocab % 64:
        raise ValueError("source terminal storage requires the declared unpadded TP vocabulary")
    programs = list(reference["target_work"])
    for call in reference["calls"]:
        if call["role"] != "embedding":
            continue
        rank, ledger = call["rank"], _Ledger()
        count = call["inputs"][0]["shape"][0]
        shard = vocab // tp
        weight = _Tensor((shard, hidden), "bf16")

        def embed(ids, w, start, end, padding, added_start, added_end):
            if (w is not weight or (start, end, padding, added_start, added_end)
                    != (rank * shard, (rank + 1) * shard, 0, vocab, vocab)):
                raise ValueError("source fused embedding shard/storage domain changed")
            partial = _Tensor((count, hidden), "bf16")
            ledger.emit("embedding", "gather", {}, "bf16", "embedding", (ids,), (partial,), {})
            return partial

        def reduce(partial):
            output = _Tensor(partial.shape, partial.dtype)
            ledger.emit("embedding:all-reduce", "collective", {}, "bf16", "all_reduce", (partial,), (output,), {})
            return output

        function = source.forward("VocabParallelEmbedding", metadata_globals={
            "ops": SimpleNamespace(vocab_parallel_embedding=embed),
            "tensor_model_parallel_all_reduce": reduce})
        owner = SimpleNamespace(tp_size=tp, use_fused_embedding=True, embedding_dim=hidden,
            weight=weight, parallel_group=None, shard_indices=SimpleNamespace(
                org_vocab_start_index=rank * shard, org_vocab_end_index=(rank + 1) * shard,
                num_org_vocab_padding=0, added_vocab_start_index=vocab, added_vocab_end_index=vocab))
        out = function(owner, _Tensor((count,), "int32"))
        if len(ledger.calls) != 2 or out.storage.writer != "embedding:all-reduce":
            raise ValueError("source embedding must retain partial through its fresh TP reduction")
        symbol = source.method("VocabParallelEmbedding", "forward")
        programs.extend({"rank": rank, **row, "source": {
            "path": str(source.paths["vocab_parallel_embedding.py"]), "symbol": "VocabParallelEmbedding.forward",
            "line": symbol.lineno}, "condition": "declared fused CUDA BF16 embedding, no added/padded vocabulary, TP>1"}
            for row in ledger.calls)
    for row in reference["final_norm_work"]:
        if row["source_input_layout"]["strides_elements"][-1] != 1:
            raise ValueError("declared CUDA norm storage needs its source unit-final-stride domain")
        rank, ledger, allocations = row["rank"], _Ledger(), []
        count = row["reference_allocations"][0]["shape"][0]
        value, weight = _Tensor((count, hidden), "bf16"), _Tensor((hidden,), "bf16")

        def empty(shape, *, device, dtype):
            output = _Tensor(tuple(shape), dtype)
            allocations.append(output)
            return output

        def kernel(output, x, gamma, epsilon):
            if (x is not value or gamma is not weight or epsilon != row["epsilon"]
                    or output.shape != value.shape or output.dtype != value.dtype):
                raise ValueError("source CUDA RMSNorm storage/operand program changed")
            ledger.emit("final-norm", "norm", {}, "bf16", "rmsnorm", (x, gamma), (output,),
                {"source_storage_profile": "vllm_0_30_vllm_c_cuda"})

        function = source.function("rms_cuda_storage.py", "rms_norm", metadata_globals={"IS_ROCM": False,
            "torch": SimpleNamespace(empty=empty, ops=SimpleNamespace(_C=SimpleNamespace(rms_norm=kernel)))})
        output = function(value, weight, row["epsilon"])
        if len(allocations) != 1 or output is not allocations[0] or len(ledger.calls) != 1:
            raise ValueError("declared CUDA RMSNorm source must allocate only its dense output")
        definition = next(n for n in source.trees["rms_cuda_storage.py"].body
            if hasattr(n, "name") and n.name == "rms_norm")
        programs.extend({"rank": rank, **call, "source": {"path": str(source.paths["rms_cuda_storage.py"]),
            "symbol": "rms_norm", "line": definition.lineno},
            "condition": "declared vllm_c CUDA storage, unit final stride, BF16 gamma and no variance override; not loaded-provider acceptance"}
            for call in ledger.calls)
    return programs


def compare_v4_terminal_storage(reference, traces, graph):
    """Compare exposed storage and liveness, including rank-major gather views."""
    trace_by_rank = {trace.rank: trace for trace in traces}
    accounts = {trace.rank: ({v.view_id: v for v in trace.views},
        {a.allocation_id: a for a in trace.allocations},
        {event: i for i, event in enumerate(trace.event_order)}) for trace in traces}
    locals_by_rank = {}
    for trace in traces:
        owned = {}
        parameters = trace.parameter_allocation_ids
        for allocation in trace.allocations:
            if allocation.allocation_id in parameters:
                continue
            prefix = allocation.allocation_id
            while prefix not in trace._nodes and ":" in prefix:
                prefix = prefix.rsplit(":", 1)[0]
            if prefix in trace._nodes:
                if prefix not in owned:
                    owned[prefix] = []
                owned[prefix].append(allocation.allocation_id)
        locals_by_rank[trace.rank] = owned
    checks = []
    seen = set()
    for row in reference:
        rank, call = row["rank"], row["call"]
        key = rank, call
        if key in seen:
            raise ValueError("source terminal storage repeats a stage")
        seen.add(key)
        trace = trace_by_rank[rank]
        views, allocations, events = accounts[rank]
        candidates = [n for n in graph.nodes if n.node_id.endswith(":" + call)
            and (n.rank == rank or n.kind.value == "collective" and rank in n.attributes["participants"])]
        local = [n for n in trace._nodes.values() if n.node_id.endswith(":" + call)]
        differences, actual = [], None
        if len(candidates) != 1 or len(local) != 1:
            differences.append("missing" if not candidates else "ambiguous_stage")
        else:
            node, local_node = candidates[0], local[0]
            incoming, outgoing = tuple(tuple(views[v] for v in node.attributes[field] if v in views)
                for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs"))
            want_inputs = list(row["inputs"])
            want_alias = [list(values) for values in row["output_alias_inputs"]]
            # Parameters remain public read operands even though the weight
            # ledger owns their backing; compare their geometry and aliases.
            actual = {"inputs": [(v.shape, v.dtype) for v in incoming],
                "outputs": [(v.shape, v.dtype) for v in outgoing],
                "output_alias_inputs": [[out.allocation_id == src.allocation_id for src in incoming] for out in outgoing]}
            for field, want in (("inputs", want_inputs), ("outputs", list(row["outputs"])), ("output_alias_inputs", want_alias)):
                if actual[field] != want:
                    differences.append(field)
            if row["call"] == "final-norm":
                if ("source_storage_profile" not in node.attributes
                        or node.attributes["source_storage_profile"] != row["attributes"]["source_storage_profile"]):
                    differences.append("declared_norm_storage_profile")
                if not incoming or incoming[0].strides_bytes[-1] != incoming[0].element_bytes:
                    differences.append("declared_unit_final_stride")
            ends = [i for event, i in events.items() if event.endswith("return:" + local_node.node_id)]
            if len(ends) != 1:
                raise ValueError("terminal source stage needs one actual call boundary")
            for inp in incoming:
                allocation = allocations[inp.allocation_id]
                if (allocation.size_bytes < region_end(inp.region())
                        or events[allocation.allocated_event] >= ends[0]
                        or allocation.freed_event is not None and events[allocation.freed_event] <= ends[0]):
                    differences.append("input_backing_or_lifetime")
            for out in outgoing:
                allocation = allocations[out.allocation_id]
                aliases = any(out.allocation_id == inp.allocation_id for inp in incoming)
                payload = prod(out.shape) * out.element_bytes
                if ((allocation.size_bytes < payload if aliases else allocation.size_bytes != payload)
                        or events[allocation.allocated_event] >= ends[0]
                        or allocation.freed_event is not None and events[allocation.freed_event] <= ends[0]):
                    differences.append("output_backing_or_lifetime")
            # A new unaccounted local Tensor is an error, even when public
            # ports, math and byte traffic still agree.
            owned = (locals_by_rank[rank][local_node.node_id]
                if local_node.node_id in locals_by_rank[rank] else [])
            fresh = {out.allocation_id for out in outgoing if not any(out.allocation_id == inp.allocation_id for inp in incoming)}
            if set(owned) != fresh:
                differences.append("extra_or_missing_stage_local")
            actual["fresh_backing_bytes"] = sorted(allocations[name].size_bytes for name in fresh)
        checks.append({"rank": rank, "stage": call, "source": row["source"],
            "condition": row["condition"] if "condition" in row else "declared ordinary TP head and all-greedy source program",
            "expected": {field: row[field] for field in ("inputs", "outputs", "output_alias_inputs")},
            "actual": actual, "differences": differences, "status": "mismatched" if differences else "matched"})
    return {"scope": "source embedding, declared CUDA norm, TP head and greedy storage API, fresh/alias outputs, per-rank backing and lifetime",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False,
        "limitations": ["native IR RMS reference allocations remain a separate conditional program; this selects the declared CUDA source storage only",
            "library/compiler scratch and actual loaded provider remain outside this Tensor API program"]}
