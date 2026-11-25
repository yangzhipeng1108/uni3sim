"""Original TP logits and plain Sampler wrappers on payload-free metadata.

The source communicator suffix describes layout, not an available collective
provider. Argmax remains a logical primitive; its comparisons are not FLOPs.
"""

import ast
from math import prod
from pathlib import Path
import re
from types import SimpleNamespace

from .qwen3_source import _Ledger, _TargetOps, _Tensor
from ..simulation.v4_flash_reference import _argmax_anchor, argmax_pair_source_work


TARGET_SOURCE_FILES = {
    "vocab_parallel_embedding.py": "vllm/model_executor/layers/vocab_parallel_embedding.py",
    "base_device_communicator.py": "vllm/distributed/device_communicators/base_device_communicator.py",
}


def argmax_source_reference(torch_include, target_work):
    """Bind the declared ATen source program without loading its provider."""
    anchor = _argmax_anchor(torch_include)
    text = re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", Path(anchor["path"]).read_text()))
    required = (
        "usingindex_t=int64_t;", "usingarg_t=detail::pair<scalar_t,index_t>;",
        "returncomp_t{}(arg.first,val,arg.second,idx)?arg:arg_t(val,idx);",
        "returncomp_t{}(a.first,b.first,a.second,b.second)?a:b;",
        "structArgReductionOps:publicMinMaxReductionOps<comp_t>",
        "staticC10_DEVICEindex_tproject(arg_targ){returnarg.second;}",
        "structArgMaxOps:publicdetail::ArgReductionOps<detail::GreaterOrNan<scalar_t>>")
    if any(fragment not in text for fragment in required):
        raise ValueError("ATen ArgMaxOps pair/index wiring changed")
    rows = []
    for call in target_work:
        if call["variant"] != "vllm_plain_greedy_sample":
            continue
        stage = call["attributes"]["stage"]
        if stage not in ("argmax", "output_int32"):
            raise ValueError("argmax reference has an unknown source sampler stage")
        count, vocab = call["shape"]["rows"], call["shape"]["vocab"]
        program = argmax_pair_source_work(count, vocab) if stage == "argmax" else None
        if program is not None:
            program.update(order="NaN wins over any non-NaN; otherwise greatest numeric value; equal values or two NaNs choose lower original index",
                signed_zero="-0.0 equals +0.0; choose lower original index", logical_order_complete=True)
        rows.append({"rank": call["rank"], "stage": stage, "call": call["call"],
            "mathematical_value_comparisons": count * (vocab - 1) if stage == "argmax" else 0,
            "mathematical_floating_point_arithmetic": 0,
            "sampled_long_aliases_int64_argmax": True, "output_unsqueeze_is_view": True,
            "argmax_index_bounds": [0, vocab - 1], "int32_cast_preserves_valid_indices": vocab <= 2 ** 31,
            "logical_pair_program": program, "source": (anchor, call["source"]),
            "condition": "declared Torch2.11 source semantics; actual library/provider is not selected"})
    return rows


def compare_v4_argmax_source(reference, plan):
    from src.kernel.predictors.vector import vllm_plain_greedy_sampling_logical_work

    expected = {(row["rank"], row["stage"]): row for row in reference}
    if len(expected) != len(reference):
        raise ValueError("argmax source repeats a rank/stage")
    fields = ("mathematical_value_comparisons", "mathematical_floating_point_arithmetic",
        "sampled_long_aliases_int64_argmax", "output_unsqueeze_is_view", "logical_pair_program",
        "argmax_index_bounds", "int32_cast_preserves_valid_indices")
    actual = {}
    for call in plan.calls:
        spec = call.kernel_spec
        if spec.variant != "vllm_plain_greedy_sample":
            continue
        key = call.rank, spec.attributes["stage"]
        if key in actual:
            raise ValueError("argmax actual repeats a rank/stage")
        work = vllm_plain_greedy_sampling_logical_work(spec)
        work["argmax_index_bounds"] = [0, spec.shape["vocab"] - 1]
        work["int32_cast_preserves_valid_indices"] = spec.shape["vocab"] <= 2 ** 31
        if work["logical_pair_program"] is not None:
            pair_fields = ("logical_pair_merges", "value_dtype", "index_dtype", "maximum_fp32_predicates",
                "logical_primitive_count_bounds", "typed_primitive_resource_mapping_complete",
                "compiler_or_actual_library_work_complete", "branches_per_pair_merge", "order", "signed_zero", "logical_order_complete")
            program = {name: work["logical_pair_program"][name] for name in pair_fields}
            program["branches_per_pair_merge"] = list(program["branches_per_pair_merge"])
            program["logical_primitive_count_bounds"] = {name: list(bounds)
                for name, bounds in program["logical_primitive_count_bounds"].items()}
            work["logical_pair_program"] = program
        actual[key] = {name: work[name] for name in fields}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [name for name in fields if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "stage": key[1], "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "status": "mismatched" if differences else "matched"})
    return {"scope": "anchored ATen GreaterOrNan/ArgMaxOps typed logical program and source sampler ID views",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("branch counts are value-dependent bounds; no particular branch is chosen",
            "minimal pair tree is not padded/library work; predicates and index issue are outside FLOPs/bytes pricing",
            "source identity does not select the loaded Torch library, compiler or GPU provider"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


class _GatherTensor(_Tensor):
    """Only the original rank-major -> movedim -> row-major program."""
    def __init__(self, shape, ledger, tp, rows, shard, *, storage=None, strides=None):
        super().__init__(shape, "bf16", storage, strides=strides)
        self.ledger, self.tp, self.rows, self.shard = ledger, tp, rows, shard

    def size(self):
        return self.shape

    def reshape(self, shape):
        p, r, n = self.tp, self.rows, self.shard
        if self.shape == (p * r, n) and shape == (p, r, n):
            return _GatherTensor(shape, self.ledger, p, r, n, storage=self.storage)
        if self.shape != (r, p, n) or shape != (r, p * n):
            raise ValueError("target communicator reshape program changed")
        # The moved rank axis can merge with the shard axis only for one row.
        if r == 1:
            return _Tensor(shape, "bf16", self.storage)
        output = _Tensor(shape)
        self.ledger.emit("lm-head", "vector", {"element_count": r * p * n, "rows": r,
            "vocab": p * n, "tensor_parallel_size": p}, "bf16", "vllm_logits_gather_reorder",
            (self,), (output,), {})
        return output

    def movedim(self, first, last):
        if (first, last, self.shape) != (0, 1, (self.tp, self.rows, self.shard)):
            raise ValueError("target communicator movedim program changed")
        return _GatherTensor((self.rows, self.tp, self.shard), self.ledger, self.tp, self.rows,
            self.shard, storage=self.storage, strides=(self.shard, self.rows * self.shard, 1))


class TargetSource:
    def __init__(self, source, calls, config, tp, rank, work, query_lengths):
        self.source, self.calls, self.rank, self.work = source, calls, rank, work
        self.tp, self.queries, self.ledger = tp, query_lengths, _Ledger()
        fields = source.fields("DeepseekV4ForCausalLM", {"config": config})
        head, _, _ = fields.call("self.lm_head", "ParallelLMHead")
        args, keywords, _ = fields.call("self.logits_processor", "LogitsProcessor")
        if len(args) != 1 or keywords or args[0] != head[0]:
            raise ValueError("V4 target requires its ordinary bias-free logits processor")
        self.vocab, self.hidden = head
        constants = {node.targets[0].id: ast.literal_eval(node.value)
            for node in source.trees["vocab_parallel_embedding.py"].body if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DEFAULT_VOCAB_PADDING_SIZE"}
        pad = source.function("vocab_parallel_embedding.py", "pad_vocab_size", metadata_globals=constants)
        padded = pad(self.vocab)
        if padded != self.vocab or tp <= 1 or padded % tp or padded // tp <= 1:
            raise NotImplementedError("target reference currently requires an unpadded TP vocabulary")
        self.shard = padded // tp
        defaults = source.default_parameters("LogitsProcessor", {"vocab_size": args[0]})
        fields = source.fields("LogitsProcessor", defaults)
        processor = type("ReferenceLogitsProcessor", (), {"__call__": source.forward("LogitsProcessor")})()
        for name in ("logits_as_input", "scale", "soft_cap", "org_vocab_size"):
            setattr(processor, name, fields.value(fields.expression("self." + name)))
        # Explicit ordinary NVIDIA source program; this is not provider dispatch.
        processor.head_dtype, processor.use_all_gather = None, True
        for name in ("_get_logits", "_apply_head"):
            setattr(processor, name, source.forward("LogitsProcessor", name).__get__(processor))
        processor._gather_logits = source.forward("LogitsProcessor", "_gather_logits", metadata_globals={
            "tensor_model_parallel_all_gather": self.gather}).__get__(processor)
        self.root = SimpleNamespace(logits_processor=processor,
            lm_head=SimpleNamespace(tp_size=tp, quant_method=SimpleNamespace(apply=self.head)))
        self.root.compute_logits = source.forward("DeepseekV4ForCausalLM", "compute_logits").__get__(self.root)
        self.gather_source = source.forward("DeviceCommunicatorBase", "all_gather", metadata_globals={
            "torch": SimpleNamespace(empty=self.empty),
            "dist": SimpleNamespace(all_gather_into_tensor=self.collective)})

    def empty(self, shape, *, dtype, device):
        if dtype != "bf16" or shape != (self.tp * len(self.queries), self.shard):
            raise ValueError("target all-gather output allocation changed")
        return _GatherTensor(shape, self.ledger, self.tp, len(self.queries), self.shard)

    def collective(self, output, value, *, group):
        self.ledger.emit("lm-head:all-gather", "collective", {"elements": value.numel(),
            "world_size": self.tp}, value.dtype, "all_gather", (value,), (output,), {})

    def gather(self, value):
        return self.gather_source(SimpleNamespace(world_size=self.tp, device_group=None), value)

    def head(self, layer, hidden, bias=None):
        if bias is not None or hidden.dtype != "bf16" or hidden.shape[-1] != self.hidden:
            raise ValueError("target head requires the original BF16 bias-free geometry")
        rows = hidden.shape[0]
        output = _GatherTensor((rows, self.shard), self.ledger, self.tp, rows, self.shard)
        weight = _Tensor((self.shard, self.hidden))
        self.ledger.emit("lm-head:projection", "lm_head", {"b": 1, "m": rows, "n": self.shard,
            "k": self.hidden}, "bf16", "gemm", (hidden, weight), (output,), {})
        return output

    def select(self, hidden, indices):
        rows, width = len(self.queries), self.hidden
        if hidden.shape != (sum(self.queries), width) or indices.shape != (rows,) or indices.dtype != "int32":
            raise ValueError("target worker selection disagrees with the source-packed queries")
        output, end, offsets = _Tensor((rows, width)), 0, []
        for count in self.queries:
            end += count
            offsets.append(end - 1)
        self.ledger.emit("logits-row-select", "vector", {"element_count": rows * width, "rows": rows,
            "hidden": width, "source_rows": hidden.shape[0]}, "bf16", "vllm_logits_row_select",
            (hidden, indices), (output,), {"selected_indices": tuple(offsets)})
        return output

    def logits(self, hidden):
        return self.root.compute_logits(hidden)

    def greedy(self, logits):
        rows = len(self.queries)
        logits.target_ops = _TargetOps(self.ledger, rows, self.vocab)
        metadata_globals = {"torch": SimpleNamespace(float32="fp32", int32="int32", int64="int64"),
            "SamplerOutput": SimpleNamespace}
        source = self.source
        sampler = type("ReferenceSampler", (), {"__call__": source.forward("Sampler", metadata_globals=metadata_globals)})()
        sampler.logprobs_mode = "raw_logprobs"
        for name in ("sample", "apply_logits_processors"):
            setattr(sampler, name, source.forward("Sampler", name, metadata_globals=metadata_globals).__get__(sampler))
        for name in ("greedy_sample", "apply_penalties"):
            setattr(sampler, name, source.forward("Sampler", name, metadata_globals=metadata_globals))
        metadata = SimpleNamespace(all_greedy=True, all_random=False, no_penalties=True,
            max_num_logprobs=None, logprob_token_ids=None, allowed_token_ids_mask=None, bad_words_token_ids={},
            output_token_ids=(), spec_token_ids=(), logitsprocs=SimpleNamespace(non_argmax_invariant=()),
            thinking_budget_state_holder=None)
        output = sampler(logits, metadata).sampled_token_ids
        bindings = {
            "logits-row-select": ("gpu_model_runner.py", "GPUModelRunner", "execute_model"),
            "lm-head:projection": ("logits_processor.py", "LogitsProcessor", "_apply_head"),
            "lm-head:all-gather": ("base_device_communicator.py", "DeviceCommunicatorBase", "all_gather"),
            "lm-head": ("base_device_communicator.py", "DeviceCommunicatorBase", "all_gather"),
            "target-sampler": ("sampler.py", "Sampler", "forward"),
            "target-sampler:1": ("sampler.py", "Sampler", "greedy_sample"),
            "target-sampler:2": ("sampler.py", "Sampler", "forward")}
        for call in self.ledger.calls:
            inputs, outputs = call["inputs"], call["outputs"]
            typed = [(dtype, prod(shape), access) for access, tensors in (("read", inputs), ("write", outputs))
                for shape, dtype in tensors]
            if call["call"] == "logits-row-select":
                # Advanced indexing reads the selected rows, not the full
                # backing described by the input tensor's logical shape.
                typed[0] = inputs[0][1], prod(outputs[0][0]), "read"
            traffic = {access: sum(elements * {"bf16": 2, "fp32": 4, "int64": 8, "int32": 4}[dtype]
                for dtype, elements, action in typed if action == access) for access in ("read", "write")}
            file, cls, method = bindings[call["call"]]
            self.work.append({"rank": self.rank, **call, "typed_operands": typed,
                "read_bytes": traffic["read"], "write_bytes": traffic["write"],
                "tensor_flops": 2 * prod(call["shape"].values()) if call["kind"] == "lm_head" else 0,
                "source": {"path": str(source.paths[file]), "symbol": cls + "." + method,
                    "line": source.method(cls, method).lineno},
                "condition": "BF16 unpadded TP head, declared all-gather suffix, greedy without processors/penalties/logprobs"})
        return output


def compare_v4_target_work(reference, plan, hardware):
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.gemm import (
        unquantized_bf16_gemm_logical_work, unquantized_fp16_gemm_logical_work)
    from src.kernel.predictors.vector import (vllm_logits_row_selection_logical_work,
        vllm_logits_gather_reorder_logical_work, vllm_logits_fp32_cast_logical_work,
        vllm_plain_greedy_roofline_work)
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {(row["rank"], row["call"]): row for row in reference["target_work"]}
    if len(expected) != len(reference["target_work"]):
        raise ValueError("target reference repeats a rank/stage")
    actual = {}
    for call in plan.calls:
        matches = {match[1] for name in call.source_node_ids if (match := re.search(
            r"(?:^|:)(lm-head(?::.*)?|target-sampler(?::.*)?|logits-row-select(?::.*)?)$", name)) is not None}
        if not matches:
            continue
        name, = matches
        ranks = (tuple(call.kernel_spec.transfer.participants)
            if name == "lm-head:all-gather" else (call.rank,))
        for rank in ranks:
            key = rank, name
            if key in actual:
                raise ValueError("target actual repeats a rank/stage")
            actual[key] = call
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = []
        observed = None
        if key in actual:
            call = actual[key]
            spec = call.kernel_spec
            observed = {"kind": spec.kind.value, "variant": spec.variant, "input_dtype": spec.dtype,
                "shape": dict(spec.shape), "call_id": call.call_id, "dependencies": call.dependencies}
        if key not in expected or key not in actual:
            differences.append("unexpected" if key not in expected else "missing")
        else:
            row, call = expected[key], actual[key]
            spec = call.kernel_spec
            if row["kind"] != "collective" and spec.dtype != row["dtype"]:
                differences.append("input_dtype")
            if row["kind"] == "collective":
                observed["collective"] = {name: getattr(spec.transfer, name) for name in
                    ("operation", "bytes", "bytes_semantics", "participants")}
                if spec.kind is not KernelKind.COMMUNICATION or spec.transfer.operation != "all_gather":
                    differences.append("collective")
                if (spec.transfer.bytes != row["read_bytes"]
                        or spec.transfer.bytes_semantics != "per_rank_input"
                        or len(spec.transfer.participants) != row["shape"]["world_size"]):
                    differences.append("collective_geometry")
            else:
                wanted = {k: v for k, v in row["shape"].items() if k not in ("elements", "hidden", "source_rows")}
                if "elements" in row["shape"]:
                    wanted["element_count"] = row["shape"]["elements"]
                if any(k not in spec.shape or spec.shape[k] != v for k, v in wanted.items()):
                    differences.append("geometry")
                if row["kind"] == "lm_head":
                    if (spec.kind is not KernelKind.GEMM or spec.attributes["output_dtype"] not in (None, "bf16")
                            or ("weight_dtype" in spec.attributes and spec.attributes["weight_dtype"] != "bf16")):
                        differences.append("head_dtype")
                elif spec.variant != row["variant"]:
                    differences.append("variant")
                if "stage" in row["attributes"] and spec.attributes["stage"] != row["attributes"]["stage"]:
                    differences.append("stage")
                if "selected_indices" in row["attributes"] and spec.attributes["selected_indices"] != row["attributes"]["selected_indices"]:
                    differences.append("selected_indices")
                if row["kind"] == "lm_head":
                    logical_programs = {
                        "unquantized_bf16_flops_bytes_v1": unquantized_bf16_gemm_logical_work,
                        "unquantized_fp16_flops_bytes_v1": unquantized_fp16_gemm_logical_work,
                    }
                    if ("mathematical_program" in spec.attributes
                            and spec.attributes["mathematical_program"] in logical_programs):
                        work = logical_programs[spec.attributes["mathematical_program"]](spec)
                        tensor_flops = work["mathematical_tensor_flops"]
                        traffic = {access: sum(operand["bytes"] for operand in work["operands"]
                            if operand["access"] == access) for access in ("read", "write")}
                    else:
                        owner, compulsory = create_predictor(spec.kind).resource_input(spec, hardware_for_call(call, hardware))
                        # Source arithmetic does not depend on Tensor/SIMT mapping.
                        tensor_flops = sum(owner.demand_per_resource[resource]
                            for resource in ("compute_tensor", "compute_fma"))
                        write = sum(node.bytes for path in owner.memory_paths if path.access_kind == "write"
                            for node in path.nodes if node.resource_role == "dram")
                        traffic = {"read": compulsory - write, "write": write}
                else:
                    if spec.variant == "vllm_plain_greedy_sample":
                        math = vllm_plain_greedy_roofline_work(spec)
                        work, demand = math["logical_work"], math["demand"]
                    elif spec.variant in ("vllm_logits_row_select", "vllm_logits_gather_reorder", "vllm_logits_fp32_cast"):
                        function = {"vllm_logits_row_select": vllm_logits_row_selection_logical_work,
                            "vllm_logits_gather_reorder": vllm_logits_gather_reorder_logical_work,
                            "vllm_logits_fp32_cast": vllm_logits_fp32_cast_logical_work}[spec.variant]
                        work = function(spec)
                        demand = work["demand"]
                    else:
                        raise ValueError("target stage has no logical owner: " + str(spec.variant))
                    tensor_flops = demand["compute_tensor"]
                    traffic = {access: sum(operand["bytes"] for operand in work["operands"]
                        if operand["access"] == access) for access in ("read", "write")}
                    if any(value != 0 for value in demand.values()):
                        differences.append("copy_or_comparison_charged_as_arithmetic")
                    if row["typed_operands"] != [(operand["dtype"], operand["elements"], operand["access"]) for operand in work["operands"]]:
                        differences.append("typed_operands")
                if tensor_flops != row["tensor_flops"]:
                    differences.append("tensor_flops")
                for access in ("read", "write"):
                    if traffic[access] != row[access + "_bytes"]:
                        differences.append(access + "_bytes")
                observed.update(tensor_flops=tensor_flops, read_bytes=traffic["read"], write_bytes=traffic["write"])
            for parent in row["parents"]:
                parent_key = key[0], parent
                if parent_key not in actual or actual[parent_key].call_id not in call.dependencies:
                    differences.append("dependency:" + parent)
        checks.append({"rank": key[0], "stage": key[1], "differences": differences,
            "status": "mismatched" if differences else "matched",
            "expected": expected[key] if key in expected else None, "actual": observed})
    return {"scope": "original V4 compute_logits/LogitsProcessor/communicator layout/Sampler and typed logical payload",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("argmax is a logical primitive; numerical NaN/tie order and library internal work not checked here",
            "communicator suffix geometry does not select a provider or price its network service",
            "logical operand bytes, not physical traffic, scratch or complete lifetime acceptance"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
