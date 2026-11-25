"""Genuine V4 modular FFN control and constructor geometry on metadata.

Routing values and native expert execution remain opaque. Declared modular,
serial, TP-only conditions are explicit; no runtime/SDK is imported or
registered and no numeric model tensor or checkpoint value is created.
"""

import ast
from contextlib import nullcontext
import re
from types import SimpleNamespace

from .qwen3_source import _Fields, _Tensor
from .v4_fp8_source import FP8_SOURCE_FILES, FP8LinearSource
from .v4_activation_source import FFNVectorSource
from .v4_routing_source import ROUTING_SOURCE_FILES, RoutingObjectsSource
from .v4_marlin_source import MARLIN_SOURCE_FILES, MarlinSource, marlin_projection


FFN_SOURCE_FILES = {
    **FP8_SOURCE_FILES,
    **ROUTING_SOURCE_FILES,
    **MARLIN_SOURCE_FILES,
    "moe_runner.py": "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
    "shared_experts.py": "vllm/model_executor/layers/fused_moe/runner/shared_experts.py",
    "activation.py": "vllm/model_executor/layers/activation.py",
    "moe_layer.py": "vllm/model_executor/layers/fused_moe/layer.py",
    "moe_config.py": "vllm/model_executor/layers/fused_moe/config.py",
    "moe_activation.py": "vllm/model_executor/layers/fused_moe/activation.py",
}


class _Port(SimpleNamespace):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _SharedOutput(_Tensor):
    def __init__(self, shape, dtype, add):
        super().__init__(shape, dtype)
        self.add = add

    def __add__(self, other):
        return self.add(self, other)


class _UnknownStreamLimit:
    def __ge__(self, tokens):
        raise ValueError("FFN source requires an undeclared shared-stream threshold")


class MoESource:
    def __init__(self, source, calls, config, tp_size, rank, linear_work, routed_work, fp8_work, vector_work):
        self.source, self.calls, self.config = source, calls, config
        self.tp, self.rank, self.linear_work, self.routed_work = tp_size, rank, linear_work, routed_work
        self.fp8_source, self.fp8_work = FP8LinearSource(source), fp8_work
        self.vector = FFNVectorSource(source, rank, vector_work)
        self.routing_object_work = []
        self.routing_objects = RoutingObjectsSource(source, config, rank, self.routing_object_work)
        self.marlin = MarlinSource(source, config, tp_size)

    def layer(self, layer):
        source, calls, config, rank = self.source, self.calls, self.config, self.rank
        if config["n_shared_experts"] != 1:
            raise ValueError("FFN source requires one shared expert and pure text")
        fields = source.fields("DeepseekV4MoE", {"config": config, "num_hash_layers": config["num_hash_layers"],
            "n_routed_experts": None, "n_activated_experts": None, "torch": {"float32": "fp32"}})
        if fields.value(fields.expression("self.image_sentinel_lo")) != 0:
            raise ValueError("FFN source requires pure-text routing")
        if ast.unparse(fields.expression("is_hash_moe")) != "extract_layer_index(prefix) < num_hash_layers":
            raise ValueError("source V4 hash-layer selection changed")
        hidden = fields.value(fields.expression("self.hidden_size"))
        experts = fields.value(fields.expression("self.n_routed_experts"))
        topk = fields.value(fields.expression("self.n_activated_experts"))
        gate_args, gate_kwargs, _ = fields.call("self.gate", "GateLinear")
        # GateLinear is keyword-constructed in this source.
        if gate_args or fields.keyword(gate_kwargs, "bias") or fields.keyword(gate_kwargs, "out_dtype") != "fp32":
            raise ValueError("source GateLinear must remain unbiased FP32 output")
        gate_in, gate_out = (fields.keyword(gate_kwargs, name) for name in ("input_size", "output_size"))
        if (gate_in, gate_out) != (hidden, experts):
            raise ValueError("source GateLinear geometry changed")

        def addition(left, right):
            self.vector.addition(layer, left, right)
            return _Tensor(left.shape, left.dtype)
        add = calls.bind(rank, "ffn_add", addition)
        def linear(role, k, n, weight_dtype, output_dtype="bf16", shared_result=False):
            def invoke(x):
                if x.shape[-1] != k or x.dtype != "bf16":
                    raise ValueError("source FFN linear requires its BF16 constructor input")
                m = x.shape[0]
                if weight_dtype == "fp8":
                    self.fp8_work.append({"rank": rank, "layer": layer, "role": role,
                        **self.fp8_source.analyze(x, n)})
                self.linear_work.append({"rank": rank, "layer": layer, "role": role,
                    "shape": {"b": 1, "m": m, "n": n, "k": k}, "source_input_dtype": x.dtype,
                    "weight_dtype": weight_dtype, "output_dtype": output_dtype, "tensor_flops": 2 * m * n * k,
                    "plain_A_bytes": 2 * m * k if weight_dtype == "bf16" else None,
                    "plain_B_bytes": 2 * n * k if weight_dtype == "bf16" else None,
                    "C_bytes": (4 if output_dtype == "fp32" else 2) * m * n})
                out = _SharedOutput((m, n), output_dtype, add) if shared_result else _Tensor((m, n), output_dtype)
                return out, None
            return calls.bind(rank, role, invoke)

        gate = _Port(forward=linear("ffn_router", gate_in, gate_out, "bf16", "fp32"),
            tid2eid=None, e_score_correction_bias=None, bias_vl=None)
        hashed = layer < config["num_hash_layers"]
        if hashed:
            gate.tid2eid = _Tensor((config["vocab_size"], topk), "int32")
        elif config["topk_method"] == "noaux_tc":
            gate.e_score_correction_bias = _Tensor((experts,), "fp32")
        if (config["quantization_config"]["quant_method"] != "fp8"
                or config["quantization_config"]["activation_scheme"] != "dynamic"
                or config["quantization_config"]["weight_block_size"] != [128, 128]):
            raise ValueError("reference shared FFN requires the source checkpoint FP8 linear configuration")
        quant = SimpleNamespace(weight_block_size=tuple(config["quantization_config"]["weight_block_size"]))
        def merged(k, sizes, *, bias, quant_config, disable_tp, prefix):
            if bias or disable_tp or quant_config is not quant or any(n % self.tp for n in sizes):
                raise ValueError("source shared merged projection disagrees with TP/block-FP8 domain")
            return _Port(forward=linear("ffn_shared_gate_up", k, sum(sizes) // self.tp, "fp8"))
        def row(k, n, *, bias, quant_config, reduce_results, disable_tp, prefix):
            if bias or disable_tp or reduce_results or quant_config is not quant or k % self.tp:
                raise ValueError("source shared down projection must remain TP-partial and unreduced")
            return _Port(forward=linear("ffn_shared_down", k // self.tp, n, "fp8", shared_result=True))
        def activation(limit):
            defaults = source.default_parameters("SiluAndMulWithClamp", {"swiglu_limit": limit})
            if defaults["alpha"] != 1.0 or defaults["beta"] != 0.0:
                raise ValueError("source FFN clamp requires alpha=1 and beta=0")
            op = _Port(swiglu_limit=limit, alpha=defaults["alpha"], beta=defaults["beta"])
            def apply_clamp(out, x, clamp, alpha, beta):
                if (clamp, alpha, beta) != (limit, 1.0, 0.0):
                    raise ValueError("source shared activation kernel parameters changed")
                if out.shape != (*x.shape[:-1], x.shape[-1] // 2) or out.dtype != x.dtype or out.storage is x.storage:
                    raise ValueError("source shared activation output allocation changed")
                self.vector.activation(layer, "ffn_shared_activation", out.shape[0], out.shape[1], out.dtype, clamp)
            op.op = apply_clamp
            raw = source.forward("SiluAndMulWithClamp", "forward_cuda", metadata_globals={"torch": SimpleNamespace(
                empty=lambda shape, *, dtype, device: _Tensor(shape, dtype))}).__get__(op)
            def call(x):
                index = calls.enter(rank, "ffn_shared_activation", (x,))
                calls.rows[index]["activation_parameters"] = {"swiglu_limit": limit}
                output = raw(x)
                calls.leave(index, output)
                return output
            op.forward = call
            return op
        _, shared_args, _ = fields.call("self.shared_experts", "DeepseekV4MLP")
        if (ast.unparse(shared_args["reduce_results"]) != "self.use_mega_moe"
                or ast.unparse(shared_args["is_sequence_parallel"]) != "use_sequence_parallel"):
            raise ValueError("source shared MLP reduction/SP binding changed")
        cls = source.metadata_constructor("DeepseekV4MLP", metadata_globals={
            "get_tensor_model_parallel_world_size": lambda: self.tp, "cdiv": lambda a, b: -(-a // b),
            "MergedColumnParallelLinear": merged, "RowParallelLinear": row, "SiluAndMulWithClamp": activation})
        shared = cls(**{name: fields.keyword(shared_args, name) for name in ("hidden_size", "intermediate_size", "hidden_act", "swiglu_limit")},
            quant_config=quant, reduce_results=False, is_sequence_parallel=False)
        shared.forward = source.forward("DeepseekV4MLP").__get__(shared)
        shared_call = calls.bind(rank, "ffn_shared_mlp", shared.forward)

        order_class = source.classes["SharedExpertsOrder"]
        order = SimpleNamespace(**{node.targets[0].id: ast.literal_eval(node.value)[0]
            for node in order_class.body if isinstance(node, ast.Assign)})
        globals = {"torch": SimpleNamespace(Tensor=_Tensor, float16="fp16"), "cast": lambda kind, value: value,
            "nullcontext": nullcontext, "SharedExpertsOrder": order,
            "UnfinalizedMoEOutput": type("UnboundDeferredOutput", (), {}),
            "ZeroExpertRouter": type("ExcludedZeroRouter", (), {}),
            "tensor_model_parallel_all_reduce": calls.bind(rank, "ffn_all_reduce", lambda x: _Tensor(x.shape, x.dtype)),
            "get_forward_context": lambda: SimpleNamespace(dp_metadata=None),
            "_unpack": lambda value: value}
        methods = ("apply_routed_input_transform", "apply_routed_output_transform", "_maybe_pad_hidden_states",
            "_maybe_apply_shared_experts", "_apply_quant_method", "_sequence_parallel_context", "_maybe_dispatch",
            "_maybe_combine", "_maybe_reduce_routed_output_before_transform", "_maybe_reduce_shared_expert_output",
            "_maybe_apply_routed_scale_to_output", "_maybe_reduce_final_output", "_maybe_add_zero_expert_output", "_forward_impl")
        runner_type = type("SourceModularRunner", (), {"__call__": source.forward("MoERunner", metadata_globals=globals),
            **{name: source.forward("MoERunner", name, metadata_globals=globals) for name in methods},
            "shared_experts": property(source.forward("MoERunner", "shared_experts")),
            "do_naive_dispatch_combine": property(source.forward("MoERunner", "do_naive_dispatch_combine")),
            "_fused_output_is_reduced": property(source.forward("MoERunner", "_fused_output_is_reduced"))})
        runner = runner_type()
        runner.moe_config = SimpleNamespace(hidden_dim=hidden, hidden_dim_unpadded=hidden, dp_size=1, pcp_size=1,
            tp_size=self.tp, ep_size=1, is_sequence_parallel=False, skip_final_all_reduce=False,
            moe_parallel_config=SimpleNamespace(use_all2all_kernels=False, enable_eplb=False,
                all2all_backend=None, use_fi_nvl_two_sided_kernels=False))
        runner.gate, runner.shared_expert_gate, runner._fse_fuse_gate = gate, None, False
        runner.routed_input_transform = runner.routed_output_transform = None
        factory, = (node for node in source.trees["moe_layer.py"].body if isinstance(node, ast.FunctionDef) and node.name == "FusedMoEFactory")
        model_factory, = (node for node in ast.walk(source.method("DeepseekV4MoE", "_init_fused_moe_experts"))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "FusedMoEFactory")
        model_args = {item.arg: item.value for item in model_factory.keywords}
        if None in model_args:
            raise ValueError("source FFN factory has undeclared expanded arguments")
        for name, expression in (("gate", "self.gate"), ("shared_experts", "self.shared_experts"),
                ("num_experts", "self.n_routed_experts"), ("top_k", "self.n_activated_experts"),
                ("intermediate_size", "config.moe_intermediate_size")):
            if ast.unparse(model_args[name]) != expression:
                raise ValueError("source FFN factory binding changed: " + name)
        names = factory.args.args[-len(factory.args.defaults):]
        activation_defaults = {name.arg: ast.literal_eval(value) for name, value in zip(names, factory.args.defaults, strict=True)
            if name.arg in ("activation", "swiglu_alpha", "swiglu_beta", "activation_situ_beta", "activation_situ_linear_beta")}
        if any(name in model_args for name in activation_defaults) or activation_defaults["activation"] != "silu":
            raise ValueError("source V4 factory no longer declares default SiLU activation")
        if ast.unparse(model_args["swiglu_limit"]) != "self.swiglu_limit":
            raise ValueError("source V4 factory no longer passes its clamp limit")
        activation_config = source.forward("ApplyMoEActivationConfig", "from_configs")(
            lambda **kwargs: SimpleNamespace(**kwargs),
            SimpleNamespace(swiglu_limit=config["swiglu_limit"], **{name: activation_defaults[name] for name in
                ("swiglu_alpha", "swiglu_beta", "activation_situ_beta", "activation_situ_linear_beta")}),
            SimpleNamespace(gemm1_clamp_limit=None, gemm1_alpha=None, gemm1_beta=None))
        if (activation_config.alpha, activation_config.beta) != (1.0, 0.0):
            raise ValueError("source routed activation requires declared alpha=1 beta=0")
        apply_scale, = (ast.literal_eval(value) for name, value in zip(names, factory.args.defaults, strict=True)
            if name.arg == "apply_routed_scale_to_output")
        if apply_scale is not False:
            raise ValueError("source factory output-scale default changed")
        if "apply_routed_scale_to_output" in model_args:
            apply_scale = fields.value(model_args["apply_routed_scale_to_output"])
        factory_fields = _Fields(factory, {"apply_routed_scale_to_output": apply_scale,
            "routed_scaling_factor": config["routed_scaling_factor"]})
        _, runner_args, _ = factory_fields.call("runner", "runner_cls")
        runner.routed_scaling_factor = factory_fields.keyword(runner_args, "routed_scaling_factor")
        if runner.routed_scaling_factor != 1.0:
            raise ValueError("source modular factory now requires a post-output scale")
        quant_method = SimpleNamespace(is_monolithic=False, topk_indices_dtype="int32", skip_forward_padding=False,
            has_unpadded_output=False, moe_kernel=SimpleNamespace(output_is_reduced=lambda: False))
        runner._quant_method = quant_method
        def select(x, logits, ids):
            if logits.shape != (x.shape[0], experts) or logits.dtype != "fp32":
                raise ValueError("source FFN routing requires FP32 projected logits")
            if hashed and (ids is None or ids.dtype != "int32" or ids.shape != (x.shape[0],)):
                raise ValueError("source hash FFN requires INT32 token IDs")
            return self.routing_objects.run(layer, x, logits, ids, gate.tid2eid, gate.e_score_correction_bias)
        route = calls.bind(rank, "ffn_route", select)
        runner.router = SimpleNamespace(select_experts=lambda *, hidden_states, router_logits, topk_indices_dtype, input_ids:
            route(hidden_states, router_logits, input_ids))
        def routed(*, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            intermediate = fields.value(fields.expression("self.moe_intermediate_size"))
            if intermediate % self.tp:
                raise ValueError("source routed FFN requires complete TP shards")
            post_init = source.method("FusedMoEConfig", "__post_init__")
            partition, = (node.value for node in ast.walk(post_init) if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == "self.intermediate_size_per_partition" for target in node.targets))
            if ast.unparse(partition) != "self.intermediate_size // tp_size":
                raise ValueError("source routed FFN TP partition changed")
            rows = x.shape[0] * topk
            self.vector.activation(layer, "ffn_routed_activation", rows, intermediate // self.tp,
                x.dtype, activation_config.clamp_limit)
            def experts(h, w, ids):
                output, marlin = self.marlin.run(h, w, ids, activation_config)
                self.routed_work.append({"rank": rank, "layer": layer, "routed_rows": rows,
                    "hidden": hidden, "intermediate": intermediate // self.tp,
                    "useful_w13_tensor_flops": marlin["mathematical_primitives"]["useful_w13_tensor_flops"],
                    "useful_w2_tensor_flops": marlin["mathematical_primitives"]["useful_w2_tensor_flops"],
                    "marlin": marlin})
                return output
            return calls.bind(rank, "ffn_routed", experts)(x, topk_weights, topk_ids)
        runner.routed_experts = SimpleNamespace(quant_method=quant_method, forward_modular=routed,
            _ensure_moe_quant_config_init=lambda: None)
        shared_globals = {"SharedExpertsOrder": order, "current_platform": SimpleNamespace(is_cuda_alike=lambda: True),
            "envs": SimpleNamespace(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=_UnknownStreamLimit())}
        shared_type = type("SourceSerialShared", (), {"__call__": source.forward("SharedExperts", metadata_globals=shared_globals),
            **{name: source.forward("SharedExperts", name, metadata_globals=shared_globals)
                for name in ("maybe_forward_async", "_determine_shared_experts_order")},
            **{name: property(source.forward("SharedExperts", name, metadata_globals=shared_globals))
                for name in ("_output_idx", "output", "_disable_shared_experts_overlap")}})
        wrapped = shared_type()
        wrapped._layer, wrapped._moe_config = shared_call, runner.moe_config
        wrapped._output, wrapped.enable_dbo, wrapped._stream = [None, None], False, None
        wrapped._mk_can_overlap_shared_experts = lambda: False
        runner._shared_experts = wrapped
        runner._encode_layer_name = lambda: "source-metadata-layer"
        runner._forward_entry = lambda h, logits, shared, ids, name, unpadded: runner._forward_impl(h, logits, shared, ids)
        moe = _Port(gate=gate, use_mega_moe=False)
        raw_forward = source.forward("DeepseekV4MoE").__get__(moe)
        moe._forward_fused_moe = source.forward("DeepseekV4MoE", "_forward_fused_moe").__get__(moe)
        moe.experts = lambda **args: calls.bind(rank, "ffn_runner", lambda x, logits, ids:
            runner(x, logits, ids))(args["hidden_states"], args["router_logits"], args["input_ids"])
        return calls.bind(rank, "ffn", raw_forward)


def compare_v4_ffn_work(reference, plan, hardware):
    """Source expectations versus actual public owner accounts, no duration.

    The original Marlin host controls cache views and serial custom-op ports.
    Its independent useful math/typed payload account is compared with the
    production owner. Checkpoint assignments, conditional scratch traffic and
    compiler services remain outside this projection.
    """
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.gemm import bf16_fp32_gemm_logical_work
    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work, block_fp8_linear_logical_work
    from src.kernel.predictors.fused_moe import dsv4_marlin_moe_source_work

    expected = {}
    for row in reference["ffn_linear_work"]:
        fields = ("shape", "weight_dtype", "output_dtype", "tensor_flops", "C_bytes")
        work = {name: row[name] for name in fields}
        work["source_input_dtype"] = row["source_input_dtype"]
        if row["role"] == "ffn_router":
            work["read_bytes"] = row["plain_A_bytes"] + row["plain_B_bytes"]
        key = row["rank"], row["layer"], row["role"]
        if key in expected:
            raise ValueError("source FFN work has duplicate identities")
        expected[key] = work
    for row in reference["ffn_routed_work"]:
        key = row["rank"], row["layer"], "ffn_routed"
        if key in expected:
            raise ValueError("source routed work has duplicate identities")
        expected[key] = {name: row[name] for name in ("routed_rows", "hidden", "intermediate",
            "useful_w13_tensor_flops", "useful_w2_tensor_flops", "marlin")}
    actual = {}
    roles = {"router": "ffn_router", "shared:gate-up": "ffn_shared_gate_up",
        "shared:down": "ffn_shared_down", "routed": "ffn_routed"}
    for call in plan.calls:
        matches = [re.search(r"(?:^|:)layer-(\d+):ffn:(router|shared:gate-up|shared:down|routed)$", name)
            for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("actual FFN comparison cannot collapse multiple source operators")
        match, = matches
        role, kernel = roles[match[2]], call.kernel_spec
        kind = KernelKind.FUSED_MOE if role == "ffn_routed" else KernelKind.GEMM
        # The FP8 quantization producer is a separate non-matmul call.
        # Its absence/work is not accepted by this mathematical projection.
        if kernel.kind is not kind:
            continue
        key = call.rank, int(match[1]), role
        if key in actual:
            raise ValueError("actual FFN work has duplicate lowered calls")
        if role == "ffn_routed":
            owner = dsv4_marlin_moe_source_work(kernel)
            actual[key] = {"routed_rows": owner["routed_rows"],
                **{name: kernel.shape[name] for name in ("hidden", "intermediate")},
                **{name: owner["mathematical_primitives"][name] for name in
                    ("useful_w13_tensor_flops", "useful_w2_tensor_flops")}, "marlin": marlin_projection(owner)}
        elif role == "ffn_router":
            owner = bf16_fp32_gemm_logical_work(kernel)
            actual[key] = {"shape": {name: kernel.shape[name] for name in ("b", "m", "n", "k")},
                "source_input_dtype": kernel.dtype, "weight_dtype": kernel.attributes["weight_dtype"],
                "output_dtype": kernel.attributes["output_dtype"], "tensor_flops": owner["demand"]["compute_tensor"],
                "read_bytes": owner["logical_bytes"] - owner["write_bytes"], "C_bytes": owner["write_bytes"]}
        else:
            owner = (block_fp8_linear_logical_work(kernel) if kernel.launch is None
                else block_fp8_gemm_roofline_work(kernel))
            actual[key] = {"shape": {name: kernel.shape[name] for name in ("b", "m", "n", "k")},
                "source_input_dtype": kernel.dtype, "weight_dtype": "fp8", "output_dtype": "bf16",
                "tensor_flops": owner["mathematical_flops"], "C_bytes": owner["operand_payload_bytes"]["output"]}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual
            else [name for name in expected[key] if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "source": ({"caller": reference["source_bindings"][key[2]],
                "typed_leaves": reference["ffn_marlin_typed_source_binding"]} if key[2] == "ffn_routed"
                else reference["source_bindings"][key[2]]) if key in expected else None})
    marlin_checks = [row for row in checks if row["role"] == "ffn_routed"]
    return {"scope": "source constructor/TP matrix work; original Marlin host cache/serial program and useful typed math/payload/stage accounts",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "routed_marlin": {"total": len(marlin_checks), "matched": sum(not row["differences"] for row in marlin_checks),
            "mismatched": sum(bool(row["differences"]) for row in marlin_checks),
            "scope": "original host views/serial ports plus useful math, typed uses and production stage templates; not full expert kernel internals"},
        "limitations": ("shared quantization/scales/provider not checked", "Marlin custom-op leaves use independently source-bound math; no numeric tensor execution",
            "routed checkpoint assignments, conditional scratch traffic, dequant issue and actual workspace capacity remain unverified",
            "no routing values, finite-precision equality, physical collective allocation, native launch or measured traffic acceptance"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
