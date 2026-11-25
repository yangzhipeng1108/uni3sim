"""Join independent wrapper allocations to the unobserved storage trace.

Expected requests come from the source interpreters, not a production memory
builder. Device compiler scratch is outside these Python wrapper requests.
"""

from collections import defaultdict
from math import prod

from src.kernel.memory_access import region_end


# Tensor API storage of the declared source primitives. None means that the
# caller supplies the written backing; these are not zero-cost operations.
# Their math/traffic and original caller aliases have independent receipts.
_DIRECT_OUTPUTS = {
    "kv_insert": ("output",),
    "qkv_norm": ("qr", "kv"), "indexer_query": ("q_fp8", "weights"),
    "grouped_output_projection": ("output",),
    "ffn_router": ("output",), "compressor_projection": ("output",),
    "indexer_projection": ("output",), "indexer_weights_projection": ("output",),
    "ffn_shared_activation": ("output",), "ffn_add": ("output",),
    "row_select": ("output",), "main_compress": ("output",),
    "output_all_reduce": ("output",), "ffn_all_reduce": ("output",),
    "short_topk": None, "main_save_partial": None, "indexer_save_partial": None,
    "main_finish": None, "indexer_finish": None, "main_store": None,
}


def _tensor_ports(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(port for member in value for port in _tensor_ports(member))
    return (value,)


def wrapper_storage_reference(reference):
    callers = defaultdict(list)
    for row in reference["calls"]:
        callers[row["rank"], row["role"]].append(row)
    expected = []
    groups = defaultdict(list)
    for row in (*reference["ffn_fp8_dataflow_work"], *reference["output_fp8_work"], *reference["input_fp8_work"]):
        groups[row["rank"], row["role"]].append(row)
    for key, work in groups.items():
        work = sorted(work, key=lambda row: row["layer"])
        calls = sorted(callers[key], key=lambda row: row["ordinal"])
        if len(work) != len(calls):
            raise ValueError("FP8 source storage requests must cover every source wrapper call")
        for source, call in zip(work, calls, strict=True):
            objects = []
            for role, descriptor, returned in (
                    ("quant_input", source["quantizer"]["quantized_input"], False),
                    ("quant_scales", source["quantizer"]["activation_scales"], False),
                    ("output", source["matmul"]["output"], True)):
                objects.append({"role": role, "shape": tuple(descriptor["shape"]), "dtype": descriptor["dtype"],
                    "bytes": descriptor["bytes"], "returned": returned})
            expected.append({"rank": key[0], "role": key[1], "ordinal": call["ordinal"], "layer": source["layer"],
                "objects": objects, "quantization_allocates_before_matmul_output": True,
                "source": reference["source_bindings"][key[1]], "condition": source["source_condition"]})
    for row in reference["mhc_work"]:
        key = row["rank"], row["role"]
        calls = [call for call in callers[key] if call["ordinal"] == row["ordinal"]]
        if len(calls) != 1:
            raise ValueError("mHC source storage request has no unique source caller")
        call, = calls
        objects = []
        if row["role"] in ("mhc_pre", "mhc_post_pre"):
            if not isinstance(call["output"], (tuple, list)) or len(call["output"]) != 4:
                raise ValueError("mHC source wrapper must return four tensor objects")
            for role, descriptor in zip(("residual", "post_mix", "res_mix", "layer_input"), call["output"], strict=True):
                objects.append({"role": role, "shape": tuple(descriptor["shape"]), "dtype": descriptor["dtype"],
                    "bytes": prod(descriptor["shape"]) * {"bf16": 2, "fp32": 4}[descriptor["dtype"]], "returned": True})
            for source_role, role in (("projection_partial", "prenorm_mixes"), ("sqrsum_partial", "prenorm_sqrsum")):
                writes = [operand for stage in row["stages"] for operand in stage["operands"]
                    if operand["role"] == source_role and operand["access"] == "write"]
                if len(writes) != 1:
                    raise ValueError("mHC source partial storage must have one writer")
                descriptor, = writes
                objects.append({"role": role, "shape": tuple(descriptor["shape"]), "dtype": descriptor["dtype"],
                    "bytes": descriptor["bytes"], "returned": False})
        else:
            descriptor = call["output"]
            objects.append({"role": "output", "shape": tuple(descriptor["shape"]), "dtype": descriptor["dtype"],
                "bytes": prod(descriptor["shape"]) * 2, "returned": True})
        expected.append({"rank": row["rank"], "role": row["role"], "ordinal": row["ordinal"],
            "objects": objects, "quantization_allocates_before_matmul_output": False,
            "source": row["source"], "condition": "declared source mHC wrapper and selected source fusion branch"})
    for call in reference["calls"]:
        if call["role"] not in _DIRECT_OUTPUTS:
            continue
        roles = _DIRECT_OUTPUTS[call["role"]]
        objects = []
        if roles is not None:
            outputs = _tensor_ports(call["output"])
            if len(roles) != len(outputs):
                raise ValueError("source primitive tensor return arity changed: " + call["role"])
            inputs = _tensor_ports(call["inputs"])
            for role, descriptor in zip(roles, outputs, strict=True):
                if any(descriptor["storage"] == port["storage"] for port in inputs):
                    raise ValueError("source primitive no longer creates independent output: " + call["role"])
                objects.append({"role": role, "shape": tuple(descriptor["shape"]), "dtype": descriptor["dtype"],
                    "bytes": prod(descriptor["shape"]) * {"bf16": 2, "fp32": 4, "fp8": 1, "int32": 4}[descriptor["dtype"]],
                    "returned": True})
        expected.append({"rank": call["rank"], "role": call["role"], "ordinal": call["ordinal"],
            "objects": objects, "quantization_allocates_before_matmul_output": False,
            "source": reference["source_bindings"][call["role"]],
            "condition": "source primitive Tensor API and declared output/passed-in storage; compiler/library-private workspace separate"})
    inverse = defaultdict(list)
    for row in reference["inverse_rope_work"]:
        inverse[row["rank"]].append(row)
    for rank, work in inverse.items():
        calls = sorted(callers[rank, "inverse_rope"], key=lambda row: row["ordinal"])
        work = sorted(work, key=lambda row: row["layer"])
        if len(calls) != len(work):
            raise ValueError("inverse-RoPE source storage work must cover every call")
        for call, row in zip(calls, work, strict=True):
            layouts = row["source_buffer_layouts"]
            scale = layouts["scales"]
            scale_shape = (scale["shape"][0], scale["shape"][2], scale["strides_elements"][2])
            objects = [{"role": "values", "shape": tuple(layouts["values"]["shape"]), "dtype": "fp8",
                "bytes": row["elements"], "returned": True},
                {"role": "scales", "shape": scale_shape, "dtype": "fp32",
                    "bytes": row["write_bytes"] - row["elements"], "returned": True}]
            expected.append({"rank": rank, "role": "inverse_rope", "ordinal": call["ordinal"], "objects": objects,
                "quantization_allocates_before_matmul_output": False,
                "source": reference["source_bindings"]["inverse_rope"],
                "condition": "source group-major FP8 backing and aligned FP32 scale backing; returned transpose views checked at caller"})
    return expected


def compare_v4_wrapper_storage(reference, observation, traces, graph):
    """Use real allocation intervals, not the profile-hook replay lifetimes."""
    expected = {(row["rank"], row["role"], row["ordinal"]): row for row in reference}
    if len(expected) != len(reference):
        raise ValueError("source wrapper storage repeats an invocation")
    nodes = {node.node_id: node for node in graph.nodes}
    storage_by_rank = {trace.rank: (
        {view.view_id: view for view in trace.views},
        {a.allocation_id: a for a in trace.allocations},
        {event: i for i, event in enumerate(trace.event_order)},
        trace.parameter_allocation_ids) for trace in traces}
    actual = {}
    for index, row in enumerate(observation.calls.rows):
        key = row["rank"], row["role"], row["ordinal"]
        if key not in expected:
            continue
        names = ([] if index not in observation.call_nodes else
            [(qualified, qualified) if qualified in nodes else (original, qualified)
                for qualified, original in observation.call_nodes[index] if qualified in nodes or original in nodes])
        names = sorted(set(names))
        if len(names) != 1:
            actual[key] = {"error": "wrapper must map to exactly one compound graph node", "nodes": names}
            continue
        (graph_name, name), = names
        node = nodes[graph_name]
        views, allocations, order, parameters = storage_by_rank[row["rank"]]
        end = [i for event, i in order.items() if event.endswith("return:" + name)]
        if len(end) != 1:
            raise ValueError("wrapper node needs one recorded source call interval")
        outputs = {views[v].allocation_id for v in node.attributes["symbolic_tensor_outputs"] if v in views}
        inputs = {views[v].allocation_id for v in node.attributes["symbolic_tensor_inputs"] if v in views}
        objects = []
        intervals, storage_ids = {}, []
        for storage, allocation in allocations.items():
            # Explicit readonly parameter ports belong to the weight ledger;
            # their node-prefixed names do not make them wrapper temporaries.
            if storage in parameters:
                continue
            if storage == name:
                role = "output"
            elif storage.startswith(name + ":"):
                role = storage[len(name) + 1:]
            else:
                continue
            view = views[storage]
            allocated = order[allocation.allocated_event]
            freed = None if allocation.freed_event is None else order[allocation.freed_event]
            intervals[role] = (allocated, freed)
            storage_ids.append(storage)
            objects.append({"role": role, "shape": view.shape, "dtype": view.dtype,
                "bytes": allocation.size_bytes, "returned": storage in outputs})
        if len(objects) == len(expected[key]["objects"]) == 1 and expected[key]["objects"][0]["role"] == "output":
            # The canonical allocation ID may use an arbitrary output-role
            # suffix. Source identity is the backing, never that suffix.
            previous = objects[0]["role"]
            objects[0]["role"] = "output"
            intervals["output"] = intervals[previous]
        # Every declared local must overlap the compound operation, and none
        # may accidentally use its borrowed input or another local's backing.
        lifetime_valid = all(intervals[obj["role"]][0] < end[0] and (
            intervals[obj["role"]][1] is None and obj["returned"] or
            intervals[obj["role"]][1] is not None and intervals[obj["role"]][1] > end[0]) for obj in objects)
        lifetime_valid = lifetime_valid and all(order[allocations[storage].allocated_event] < end[0]
            and (allocations[storage].freed_event is None or order[allocations[storage].freed_event] > end[0])
            for storage in inputs.intersection(parameters))
        # Borrowed weights need backing for every public read view, even
        # though their allocations are outside the wrapper's fresh locals.
        lifetime_valid = lifetime_valid and all(
            region_end(views[view_id].region()) <= allocations[views[view_id].allocation_id].size_bytes
            for view_id in node.attributes["symbolic_tensor_inputs"]
            if view_id in views and views[view_id].allocation_id in parameters)
        output_roles = [obj["role"] for obj in objects if obj["returned"]]
        if len(set(storage_ids)) != len(storage_ids) or inputs.intersection(storage_ids):
            lifetime_valid = False
        for obj in objects:
            if not obj["returned"]:
                temporary_end = intervals[obj["role"]][1]
                if temporary_end is None or any(intervals[role][1] is not None and temporary_end > intervals[role][1]
                        for role in output_roles):
                    lifetime_valid = False
        quant_before_output = True
        if expected[key]["quantization_allocates_before_matmul_output"]:
            quant_before_output = all(role in intervals for role in ("quant_input", "quant_scales", "output"))
            if quant_before_output:
                quant_before_output = intervals["quant_input"][0] < intervals["quant_scales"][0] < intervals["output"][0]
        actual[key] = {"objects": sorted(objects, key=lambda obj: obj["role"]),
            "all_locals_overlap_call_and_temporaries_release_before_outputs": lifetime_valid,
            "quantization_allocates_before_matmul_output": quant_before_output}
    checks = []
    for key, row in sorted(expected.items()):
        want = {"objects": sorted(row["objects"], key=lambda obj: obj["role"]),
            "all_locals_overlap_call_and_temporaries_release_before_outputs": True,
            "quantization_allocates_before_matmul_output": True}
        got = actual[key] if key in actual else None
        differences = ["missing"] if got is None else [field for field in want if field not in got or want[field] != got[field]]
        checks.append({"rank": key[0], "role": key[1], "ordinal": key[2], "source": row["source"],
            "condition": row["condition"], "expected": want, "actual": got, "differences": differences,
            "status": "mismatched" if differences else "matched"})
    return {"scope": "source wrapper/primitive tensor shape/type/bytes, passed-in versus fresh output, backing and source-scope lifetime",
        "total": len(checks), "matched": sum(not c["differences"] for c in checks),
        "mismatched": sum(bool(c["differences"]) for c in checks), "not_checked": 0, "checks": checks,
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False,
        "limitations": ["source wrapper allocations only; selected native library workspace and compiler storage remain separate"]}
