"""Declared synchronous Marlin Tensor requests, independent of prediction.

Allocation capacity is not reduction traffic, allocator reserve or GPU lifetime.
"""

from math import prod
from pathlib import Path
import re

from ..simulation.v4_flash_reference import _cpp_anchor, _function_anchor, _marlin_moe_reference, _marlin_scratch_reference


def expert_storage_reference(root, raw, inputs):
    root = Path(root)
    cpp = _cpp_anchor(root, "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu",
        "if (use_fp32_reduce && !use_atomic_add)")
    compact = re.sub(r"\s+", "", cpp["source"])
    for expression in ("longmax_c_tmp_size=min((long)size_n*sorted_token_ids.size(0),",
            "(long)sms*4*moe_block_size*MARLIN_NAMESPACE_NAME::max_thread_n",
            "if(moe_block_size==8)max_c_tmp_size*=2;",
            "c_tmp=torch::stable::new_empty(a,{max_c_tmp_size},kFloat)"):
        if expression not in compact:
            raise ValueError("Marlin FP32 storage source program changed")
    header = root / "csrc/libtorch_stable/quantization/marlin/marlin.cuh"
    constant = re.search(r"static constexpr int max_thread_n\s*=\s*(\d+)\s*;", header.read_text())
    if constant is None or int(constant[1]) != 256:
        raise ValueError("Marlin scratch source maximum N tile changed")
    bindings = [_function_anchor(root, "vllm/model_executor/layers/fused_moe/" + file, symbol)
        for file, symbol in (("experts/marlin_moe.py", "_fused_marlin_moe"),
            ("experts/marlin_moe.py", "fused_marlin_moe"),
            ("modular_kernel.py", "FusedMoEKernelModularImpl._allocate_buffers"),
            ("moe_align_block_size.py", "moe_align_block_size"))]
    bindings.append(cpp)
    bindings.append(_function_anchor(root, "vllm/model_executor/layers/quantization/utils/marlin_utils.py", "marlin_make_workspace_new"))
    bindings = [{field: row[field] for field in ("path", "symbol", "line")} for row in bindings]
    program = _marlin_moe_reference(raw, sum(inputs["query_lengths"]), inputs["TP"])
    sm = inputs["specification_multiprocessors"]
    scratch = _marlin_scratch_reference(program, sm)
    names = {"sorted_token_ids": "sorted-token-ids", "expert_ids": "expert-ids",
        "num_tokens_post_padded": "num-tokens-post-padded", "cumsum": "alignment-cumsum"}
    objects = [{"role": names[row["role"]], "shape": tuple(row["shape"]), "dtype": row["dtype"],
        "bytes": 4 * prod(row["shape"])} for row in program["tensor_object_program"]["routing_buffers"]]
    objects += [{"role": "locks", "shape": (sm * 4,), "dtype": "int32", "bytes": sm * 16}]
    objects += [{"role": ("w2" if row["stage"] == "w2_with_router_weight" else "w13") + ":fp32-reduce",
        "shape": (row["requested_elements"],), "dtype": "fp32", "bytes": row["requested_bytes"]}
        for row in scratch["stages"]]
    tokens, hidden = sum(inputs["query_lengths"]), raw["hidden_size"]
    objects += [{"role": "output", "shape": (tokens, hidden), "dtype": "bf16", "bytes": tokens * hidden * 2}]
    return {"objects": sorted(objects, key=lambda row: row["role"]),
        "arena": program["modular_buffer_program"], "bindings": bindings,
        "sm_count": sm, "layers": raw["num_hidden_layers"], "tp": inputs["TP"],
        "condition": program["source_condition"], "zero_extent_objects": program["tensor_object_program"]["zero_extent_objects"],
        "scope": "source Tensor allocation/view requests and synchronous Python lifetime; reduction accesses remain parameterized"}


def compare_v4_expert_storage(reference, traces, graph):
    actual = {}
    for trace in traces:
        views = {view.view_id: view for view in trace.views}
        allocations = {row.allocation_id: row for row in trace.allocations}
        events = {event: i for i, event in enumerate(trace.event_order)}
        for node in graph.nodes:
            if node.rank != trace.rank or node.variant != "full" or node.kind.value != "fused_moe":
                continue
            match = re.search(r"(?:^|:)layer-(\d+):ffn:routed$", node.node_id)
            if match is None:
                continue
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("Marlin storage repeats layer/rank")
            prefix = node.node_id
            local = {("output" if name == prefix else name[len(prefix) + 1:]): row
                for name, row in allocations.items() if name == prefix or name.startswith(prefix + ":")}
            objects = sorted(({"role": role, "shape": views[row.allocation_id].shape,
                "dtype": views[row.allocation_id].dtype, "bytes": row.size_bytes}
                for role, row in local.items()), key=lambda row: row["role"])
            markers = {role: events[trace.checkpoints[prefix + ":" + role]]
                for role in ("alignment-complete", "locks-zeroed", "w13:consume", "activation-complete",
                    "w2:consume", "reduction-complete") if prefix + ":" + role in trace.checkpoints}
            ends = [i for event, i in events.items() if event.endswith("return:" + prefix)]
            lifetime = len(ends) == 1 and len(markers) == 6
            intervals = {role: (events[row.allocated_event], None if row.freed_event is None else events[row.freed_event])
                for role, row in local.items()}
            if lifetime and set(intervals) == {row["role"] for row in reference["objects"]}:
                end, = ends
                def alive(role, at):
                    start, finish = intervals[role]
                    return start < at and (finish is None or finish > at)
                lifetime = all(alive(role, end) and intervals[role][1] is not None
                    for role in ("sorted-token-ids", "expert-ids", "num-tokens-post-padded"))
                lifetime = lifetime and alive("output", end)
                lifetime = lifetime and alive("locks", markers["w13:consume"]) and alive("locks", markers["w2:consume"])
                lifetime = lifetime and intervals["locks"][1] is not None and intervals["locks"][1] < markers["reduction-complete"]
                if "alignment-cumsum" in intervals:
                    lifetime = lifetime and alive("alignment-cumsum", markers["alignment-complete"])
                    lifetime = lifetime and intervals["alignment-cumsum"][1] is not None and intervals["alignment-cumsum"][1] < markers["locks-zeroed"]
                for stage, next_marker in (("w13", "activation-complete"), ("w2", "reduction-complete")):
                    role = stage + ":fp32-reduce"
                    lifetime = lifetime and alive(role, markers[stage + ":consume"])
                    lifetime = lifetime and intervals[role][1] is not None and intervals[role][1] < markers[next_marker]
                lifetime = lifetime and intervals["w13:fp32-reduce"][1] < intervals["w2:fp32-reduce"][0]
                lifetime = lifetime and intervals["output"][0] < intervals["sorted-token-ids"][0]
            else:
                lifetime = False
            arena_valid = "symbolic_marlin_arena_views" in node.attributes
            names = node.attributes["symbolic_marlin_arena_views"] if arena_valid else ()
            expected_views = reference["arena"]["view_regions"]
            arena_valid = arena_valid and len(names) == len(expected_views)
            if arena_valid:
                by_role = {name.removeprefix(prefix + ":"): views[name] for name in names}
                arena_valid = set(by_role) == {row["role"] for row in expected_views}
                if arena_valid:
                    shared = {view.allocation_id for view in by_role.values()}
                    arena_valid = len(shared) == 1 and not shared.intersection(row.allocation_id for row in local.values())
                    for row in expected_views:
                        view = by_role[row["role"]]
                        arena_valid = arena_valid and (view.shape, view.dtype, view.offset_bytes, view.strides_bytes) == (
                            tuple(row["shape"]), "bf16", row["offset_bytes"], tuple(v * 2 for v in row["strides_elements"]))
                    backing = allocations[by_role["gate_up"].allocation_id]
                    arena_valid = arena_valid and backing.size_bytes >= reference["arena"]["requested_workspace_bytes"]
                    for role, base in (("activation-input", "gate_up"), ("reduction-input", "weighted_expert_output")):
                        name = prefix + ":" + role
                        arena_valid = arena_valid and name in views and views[name].allocation_id == by_role[base].allocation_id
                        if name in views:
                            arena_valid = arena_valid and views[name].offset_bytes == by_role[base].offset_bytes
            actual[key] = {"objects": objects, "source_lifetimes": lifetime, "arena_views": arena_valid,
                "sm_count": node.attributes["symbolic_marlin_multiprocessors"] if "symbolic_marlin_multiprocessors" in node.attributes else None}
    checks = []
    expected_keys = {(rank, layer) for rank in range(reference["tp"]) for layer in range(reference["layers"])}
    want = {"objects": reference["objects"], "source_lifetimes": True, "arena_views": True, "sm_count": reference["sm_count"]}
    for key in sorted(expected_keys | set(actual)):
        got = actual[key] if key in actual else None
        differences = ["unexpected"] if key not in expected_keys else ["missing"] if got is None else [field for field in want if want[field] != got[field]]
        checks.append({"rank": key[0], "layer": key[1], "role": "ffn_routed", "ordinal": key[1],
            "expected": want if key in expected_keys else None, "actual": got, "differences": differences,
            "status": "mismatched" if differences else "matched", "condition": reference["condition"], "source": reference["bindings"]})
    return {"scope": reference["scope"], "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "normal_capacity_accepted": False, "runtime_accepted": False}
