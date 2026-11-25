"""Shared source operators for symbolic model forwards (no real tensors/GPU).

The initial profile is explicitly synchronous CUDA-source RMSNorm/RoPE with
in-place residual/RoPE and a source SiluAndMul output. It does not select an
actual provider, compiler, allocator or library workspace implementation.
"""

from src.kernel.prediction_cache import runtime_lru_cache

from dataclasses import replace
from math import isfinite, prod
from types import MappingProxyType

from src.kernel.kernel_spec import PAGED_KV_CACHE_ATTRIBUTES, linear_expert_partition
from src.kernel.launch_config.block_fp8_gemm import BLOCK_FP8_WEIGHT_QUANTIZATION
from src.kernel.launch_config.select_launch_config import select_mhc_stage_launches
from src.serving.deployment.deployment_config import CollectiveConfig
from .traced_forward import ForwardCollectiveGroup, ForwardTrace, ForwardWorkspace
from .workload_graph import WorkloadKind, WorkloadNode
from .rope_dispatch import ROPE_PROGRAM_FUSED, ROPE_PROGRAM_NATIVE_COMPOSITE
from .model_config import is_glm_fp8_tensor_model_scope, is_deepseek_v4_parallel_model_scope


def _maca_native_copy_dispatch(tensor, runtime_source):
    """Derive existing Copy facts from a declared native owning allocation.

    Contiguity alone does not prove alignment. Offset aliases, including a
    two-byte BF16 slice, remain outside this deliberately owning-only scope.
    The output of the caller's Copy is a fresh native allocation.
    """
    from .runtime_inputs import MACA_COPY_RUNTIME_SOURCE
    if runtime_source != MACA_COPY_RUNTIME_SOURCE:
        raise ValueError("unknown MACA native Copy runtime source")
    if (not tensor.is_contiguous or tensor.view.offset_bytes != 0
            or tensor.view.view_id != tensor.view.allocation_id):
        raise NotImplementedError("MACA native Copy requires a contiguous owning input at offset zero")
    return {"disable_unroll_float_opt": False,
            "input_alignment_bytes": 4, "output_alignment_bytes": 4}


@runtime_lru_cache(maxsize=2048)
def _node_geometry_template(kind, pool, items, dtype, variant, rank):
    # Only native immutable key values reach this cache. The constructor owns
    # all validation; failed constructions never populate a proof template.
    return WorkloadNode("geometry-template", kind, pool, dict(items), dtype, variant, rank, {})


def _node_geometry_validator_unchanged():
    node_type, namespace, methods, globals_owner, globals_snapshot = _NODE_GEOMETRY_VALIDATION
    return (WorkloadNode is node_type and vars(node_type) == namespace
        and globals_owner == globals_snapshot
        and all(function.__code__ is code and function.__defaults__ == defaults
            and function.__kwdefaults__ == kwdefaults
            for function, code, defaults, kwdefaults in methods))


class SymbolicOperator:
    def __init__(self, trace, name, pool_name):
        self.trace, self.name, self.pool_name = trace, name, pool_name

    def node(self, kind, shape, dtype, variant, attributes=None):
        identity = self.trace.unique_name(self.name)
        annotations = {"launch_policy": "analytical_launch",
            **({} if attributes is None else attributes)}
        if type(shape) in (dict, MappingProxyType):
            items = tuple(shape.items())
            if (type(kind) is WorkloadKind and type(self.pool_name) is str
                    and type(dtype) in (str, type(None)) and type(variant) in (str, type(None))
                    and type(self.trace.rank) in (int, type(None))
                    and all(type(key) is str and type(value) is int for key, value in items)
                    and _node_geometry_validator_unchanged()):
                template = _node_geometry_template(kind, self.pool_name, items, dtype, variant, self.trace.rank)
                return template.with_attributes(annotations, node_id=identity)
        return WorkloadNode(identity, kind, self.pool_name, shape,
            dtype=dtype, variant=variant, rank=self.trace.rank, attributes=annotations)


class SymbolicLinear(SymbolicOperator):
    """Unquantized rank-local linear; optional FP32 C retains BF16 A/B."""
    def __init__(self, trace, name, pool_name, in_features, out_features, *, bias=False,
                 kind=WorkloadKind.LINEAR, output_dtype=None):
        super().__init__(trace, name, pool_name)
        if output_dtype not in (None, "bf16", "fp32"):
            raise NotImplementedError("symbolic linear output must be BF16 or FP32")
        self.in_features, self.out_features, self.bias, self.kind = in_features, out_features, bias, kind
        self.output_dtype = output_dtype

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("symbolic linear input width disagrees with its constructor")
        dtype = x.dtype if self.output_dtype is None else self.output_dtype
        if dtype != x.dtype and (x.dtype != "bf16" or dtype != "fp32" or self.bias):
            raise NotImplementedError("mixed symbolic linear requires BF16 inputs, FP32 output and no bias")
        rows = prod(x.shape[:-1])
        # Keep the existing GEMM workload's matrix/vector descriptor. This
        # selects a discrete observation key, not a different source formula.
        variant = "gemv" if self.kind is WorkloadKind.LM_HEAD and rows == 1 else "gemm"
        layout = ({"a_layout": "row_major", "b_layout": "row_major", "c_layout": "row_major",
            "trans_a": False, "trans_b": True, "epilogue": "bias" if self.bias else "none"}
            if x.is_contiguous else {})
        node = self.node(self.kind, {"b": 1, "m": rows,
            "n": self.out_features, "k": self.in_features}, x.dtype, variant,
            {"has_bias": self.bias, "weight_dtype": x.dtype, "weight_group_size": 0, **layout,
                **({"source_component": "symbolic_fp32_linear"} if x.dtype == "fp32" else {}),
                **({"output_dtype": dtype}
                if self.output_dtype is not None else {})})
        weight = self.trace.parameter_view(self.name + ":weight",
            (self.out_features, self.in_features), x.dtype)
        if x.dtype == "fp16":
            # Publish the actual source views; a runtime label alone cannot
            # establish BLAS operand strides or alignment of an alias.
            node = node.with_attributes({**node.attributes,
                "source_gemm_input_shape": tuple(x.shape),
                "source_gemm_input_strides_bytes": tuple(x.view.strides_bytes),
                "source_gemm_input_offset_bytes": x.view.offset_bytes,
                "source_gemm_weight_shape": tuple(weight.shape),
                "source_gemm_weight_strides_bytes": tuple(weight.strides_bytes),
                "source_gemm_weight_offset_bytes": weight.offset_bytes})
        return self.trace.call(node, (x,), shape=(*x.shape[:-1], self.out_features),
            dtype=dtype, parameters=(weight,)), None


class SymbolicBatchedLinear(SymbolicOperator):
    """BF16 bmm with distinct rank-local weights per batch and optional out.

    Canonical GEMM strides describe logical operand footprints for Roofline;
    the actual source strides/alias are retained in the input/output views.
    No packing copy or actual library/launch binding is inferred.
    """

    def __init__(self, trace, name, pool_name, batches, in_features, out_features):
        super().__init__(trace, name, pool_name)
        if any(type(value) is not int or value <= 0 for value in (batches, in_features, out_features)):
            raise ValueError("symbolic batched linear dimensions must be positive integers")
        self.batches, self.in_features, self.out_features = batches, in_features, out_features

    def __call__(self, x, *, out=None, controls=()):
        self.trace._check_tensor(x)
        if len(x.shape) != 3 or x.shape[0] != self.batches or x.shape[-1] != self.in_features or x.dtype != "bf16":
            raise ValueError("symbolic batched linear requires matching BF16 [batch,rows,in_features]")
        batches, rows, width = x.shape
        if out is not None:
            self.trace._check_tensor(out)
            if out.shape != (batches, rows, self.out_features) or out.dtype != x.dtype:
                raise ValueError("symbolic batched linear out must match its BF16 result")
            if out.view.allocation_id == x.view.allocation_id:
                raise NotImplementedError("symbolic batched linear requires independent input/output storage")
            if (not out.is_contiguous
                    and out.view.strides_bytes != (self.out_features * 2, batches * self.out_features * 2, 2)):
                raise NotImplementedError("symbolic batched linear out requires dense or token/head-interleaved storage")
        node = self.node(WorkloadKind.LINEAR, {"b": batches, "m": rows,
            "n": self.out_features, "k": width}, "bf16", "batched_gemm", {
                "source_component": "symbolic_batched_linear",
                "batch_layout": "strided", "a_batch_stride_elements": rows * width,
                "b_batch_stride_elements": width * self.out_features,
                "c_batch_stride_elements": rows * self.out_features,
                "weight_dtype": "bf16", "weight_group_size": 0, "has_bias": False})
        return self.trace.call(node, (x,), shape=(batches, rows, self.out_features), dtype=x.dtype, out=out, controls=controls)


class SymbolicConcat(SymbolicOperator):
    """Last-axis concatenation into fresh or caller-provided strided storage."""

    def __call__(self, *inputs, out=None):
        if len(inputs) < 2:
            raise ValueError("symbolic concat requires at least two tensors")
        for tensor in inputs:
            self.trace._check_tensor(tensor)
        first = inputs[0]
        if any(tensor.shape[:-1] != first.shape[:-1] or tensor.dtype != first.dtype for tensor in inputs):
            raise ValueError("symbolic concat requires equal leading axes and dtype")
        shape = (*first.shape[:-1], sum(tensor.shape[-1] for tensor in inputs))
        if out is not None:
            self.trace._check_tensor(out)
            if any(out.view.allocation_id == tensor.view.allocation_id for tensor in inputs):
                raise NotImplementedError("symbolic concat requires separate output backing")
        node = self.node(WorkloadKind.GATHER, {"elements": prod(shape), "rows": prod(shape[:-1]),
            "hidden": shape[-1]}, first.dtype, "gather", {"gather_factor": 1, "index_count": 0,
                "index_bytes": 8, "scattered": False, "materialized_output": True,
                "source_operation": "cat_last_dim"})
        return self.trace.call(node, inputs, shape=shape, dtype=first.dtype, out=out)


class SymbolicFloat32Cast(SymbolicOperator):
    """Aten BF16->FP32 materialization; FP32 inputs keep their backing."""

    def __call__(self, x, *, vllm011_router_topk=None):
        self.trace._check_tensor(x)
        if x.dtype == "fp32":
            return x
        if x.dtype != "bf16":
            raise NotImplementedError("symbolic float32 cast requires BF16 or FP32 input")
        attributes = {"operation": "cast_bf16_float32", "reads": 1, "writes": 1}
        if vllm011_router_topk is not None:
            if (len(x.shape) != 2 or type(vllm011_router_topk) is not int
                    or not 1 <= vllm011_router_topk <= x.shape[1]):
                raise ValueError("0.11 router cast requires its actual logits matrix and topk")
            attributes.update(source_component="vllm_0_11_symbolic_router_logits_cast",
                model_execution_version="vllm_0_11_0",
                source_moe_router_shape=(*x.shape, vllm011_router_topk),
                source_input_layout=("contiguous_aligned_16"
                    if x.is_contiguous and x.view.offset_bytes % 16 == 0 else "unbound"))
        node = self.node(WorkloadKind.ELEMENTWISE, {"elements": prod(x.shape)}, "bf16", "elementwise", attributes)
        return self.trace.call(node, (x,), shape=x.shape, dtype="fp32")


class SymbolicPositionZeroMask(SymbolicOperator):
    """Version-bound MTP position mask: indexed fill or fresh where output."""

    def __call__(self, positions, embeddings, *, position_values, model_execution_version="vllm_0_11_0"):
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_27_1"):
            raise ValueError("MTP position mask requires its declared source version")
        rows, width = embeddings.shape
        if (positions.shape != (rows,) or positions.dtype != "int64" or embeddings.dtype != "bf16"
                or not isinstance(position_values, tuple) or len(position_values) != rows
                or any(type(p) is not int or p < 0 for p in position_values)):
            raise ValueError("MTP position mask requires explicit INT64 positions and BF16 embedding rows")
        fresh_where = model_execution_version == "vllm_0_27_1"
        attrs = {"source_component": "vllm_0_27_1_symbolic_mtp_position_where" if fresh_where else "symbolic_mtp_position_mask",
            "model_execution_version": model_execution_version,
            "masked_rows": position_values.count(0), "symbolic_workspace_complete": False}
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * width, "rows": rows, "hidden": width},
            "int64", "vllm_mtp_position_mask", {**attrs, "stage": "compare"})
        mask = self.trace.call(node, (positions,), shape=(rows,), dtype="bool")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * width, "rows": rows, "hidden": width},
            "bf16", "vllm_mtp_position_mask", {**attrs, "stage": "where" if fresh_where else "fill"})
        if fresh_where:
            return self.trace.call(node, (mask, embeddings), shape=embeddings.shape, dtype=embeddings.dtype)
        return self.trace.call(node, (mask,), shape=embeddings.shape, dtype=embeddings.dtype, out=embeddings)


class SymbolicV32EHNorm(SymbolicOperator):
    """vLLM0.30 fused masked embedding/previous-hidden RMSNorm + concat.

    Persistent weights and all operands are metadata only. Position zero
    still reads its embedding row, as in the Triton source. No separate
    mask, norm or concatenation allocation/launch is invented.
    """

    def __init__(self, trace, name, pool_name, width, epsilon, *, legacy_dummy_retained_inputs=False):
        super().__init__(trace, name, pool_name)
        if (type(width) is not int or width <= 0 or type(epsilon) not in (int, float)
                or not isfinite(epsilon) or epsilon <= 0):
            raise ValueError("V32 fused EH requires positive width and finite source epsilon")
        self.width, self.epsilon = width, epsilon
        if type(legacy_dummy_retained_inputs) is not bool:
            raise TypeError("V32 EH retained-input source must be explicit")
        self.legacy_dummy_retained_inputs = legacy_dummy_retained_inputs
        self.enorm = trace.tensor(name + ":enorm.weight", (width,), "bf16", persistent=True)
        self.hnorm = trace.tensor(name + ":hnorm.weight", (width,), "bf16", persistent=True)

    def _positions(self, positions, previous_hidden, position_values):
        self.trace._check_tensor(positions)
        self.trace._check_tensor(previous_hidden)
        rows = positions.shape[0]
        if (positions.shape != (rows,) or positions.dtype != "int64"
                or previous_hidden.shape != (rows, self.width) or previous_hidden.dtype != "bf16"
                or previous_hidden.view.strides_bytes[-1] != 2
                or (position_values is not None if self.legacy_dummy_retained_inputs else
                    not isinstance(position_values, tuple) or len(position_values) != rows
                    or any(type(p) is not int or p < 0 for p in position_values))):
            raise ValueError("V32 fused EH requires INT64 positions, explicit values and BF16 hidden rows")
        return rows

    def _fused(self, inputs, rows, position_values, *, embedded):
        node = self.node(WorkloadKind.NORM, {"rows": rows, "hidden": self.width,
            "dtype_bytes": 2, "position_bytes": 8, **({"input_id_bytes": 4} if embedded else {})},
            "bf16", "dsv32_embed_eh_norm" if embedded else "dsv32_eh_norm", {
                "source_component": "vllm_0_30_symbolic_v32_eh_norm",
                "model_execution_version": "vllm_0_30_0", "epsilon": self.epsilon,
                "masked_rows": None if self.legacy_dummy_retained_inputs else position_values.count(0),
                **({"input_values_source": "legacy_dummy_retained_buffer"} if self.legacy_dummy_retained_inputs else {}),
                "symbolic_workspace_complete": False,
                "input_roles": (("positions", "input_ids", "embedding_table", "previous_hidden", "enorm_weight", "hnorm_weight")
                    if embedded else ("positions", "inputs_embeds", "previous_hidden", "enorm_weight", "hnorm_weight"))})
        return self.trace.call(node, inputs, shape=(rows, 2 * self.width), dtype="bf16")

    def __call__(self, positions, inputs_embeds, previous_hidden, *, position_values):
        rows = self._positions(positions, previous_hidden, position_values)
        self.trace._check_tensor(inputs_embeds)
        if (inputs_embeds.shape != previous_hidden.shape or inputs_embeds.dtype != "bf16"
                or inputs_embeds.view.strides_bytes[-1] != 2):
            raise ValueError("V32 fused EH requires matching BF16 embedding rows")
        return self._fused((positions, inputs_embeds, previous_hidden, self.enorm, self.hnorm),
            rows, position_values, embedded=False)


class SymbolicV32EmbedEHNorm(SymbolicV32EHNorm):
    """TP1 full-vocabulary gather folded into the same EH source operation."""

    def __init__(self, trace, name, pool_name, width, vocab, epsilon, *, legacy_dummy_retained_inputs=False):
        if type(vocab) is not int or vocab <= 0:
            raise ValueError("V32 fused embedding EH requires a positive full vocabulary")
        super().__init__(trace, name, pool_name, width, epsilon, legacy_dummy_retained_inputs=legacy_dummy_retained_inputs)
        self.vocab = vocab
        self.embedding_table = trace.tensor(name + ":embed_tokens.weight", (vocab, width), "bf16", persistent=True)

    def __call__(self, input_ids, positions, previous_hidden, *, position_values, input_token_ids):
        rows = self._positions(positions, previous_hidden, position_values)
        self.trace._check_tensor(input_ids)
        if (input_ids.shape != (rows,) or input_ids.dtype != "int32"
                or (input_token_ids is not None if self.legacy_dummy_retained_inputs else
                    not isinstance(input_token_ids, tuple) or len(input_token_ids) != rows
                    or any(type(token) is not int or not 0 <= token < self.vocab for token in input_token_ids))):
            raise ValueError("V32 fused embedding EH requires explicit in-vocabulary INT32 token IDs")
        return self._fused((positions, input_ids, self.embedding_table, previous_hidden, self.enorm, self.hnorm),
            rows, position_values, embedded=True)


class SymbolicBiasedSigmoidTopK(SymbolicOperator):
    """GLM4's single-group logical routing, with separate unbiased weights.

    This mathematical projection does not select fused CUDA or torch,
    compute expert IDs, or assert their finite-precision equivalence.
    """

    def __init__(self, trace, name, pool_name, model):
        super().__init__(trace, name, pool_name)
        if (model.model_type != "glm4_moe" or model.moe_group_count != 1
                or model.moe_topk_group_count != 1
                or type(model.moe_normalize_topk_probability) is not bool):
            raise NotImplementedError("GLM4 symbolic sigmoid routing requires its single-group source")
        self.model = model
        self.bias = trace.tensor(name + ":bias", (model.num_experts,), "fp32", persistent=True)

    def __call__(self, x, logits):
        model, rows = self.model, x.shape[0]
        if (x.shape != (rows, model.hidden_size) or x.dtype != "bf16"
                or logits.shape != (rows, model.num_experts) or logits.dtype != "fp32"):
            raise ValueError("GLM4 routing requires BF16 hidden states and FP32 logits/bias")
        node = self.node(WorkloadKind.VECTOR, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size}, "fp32", "moe_topk", {
                "source_component": "symbolic_glm4_sigmoid_topk",
                "source_program": "vllm_0_11_fp32_sigmoid_bias_topk",
                "mathematical_program": "fp32_sigmoid_bias_topk_flops_bytes_v1",
                "normalize_topk_probability": model.moe_normalize_topk_probability})
        return self.trace.call_outputs(node, (logits, self.bias), outputs=(
            ("weights", (rows, model.num_experts_per_tok), "fp32"),
            ("ids", (rows, model.num_experts_per_tok), "int32")))


class SymbolicCopy(SymbolicOperator):
    """Source copy_ into an existing view; no extra output allocation."""

    def __call__(self, source, destination):
        self.trace._check_tensor(source)
        self.trace._check_tensor(destination)
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("symbolic copy requires matching shapes and dtypes")
        if source.view.allocation_id == destination.view.allocation_id:
            raise NotImplementedError("symbolic overlapping copy needs explicit alias semantics")
        if any(size > 1 and stride == 0 for size, stride in zip(
                destination.shape, destination.view.strides_bytes, strict=True)):
            raise ValueError("symbolic copy destination cannot have broadcast write axes")
        node = self.node(WorkloadKind.GATHER,
            {"elements": prod(source.shape), "rows": source.shape[0], "hidden": prod(source.shape[1:])},
            source.dtype, "gather", {"gather_factor": 1, "index_count": 0,
                "index_bytes": 8, "scattered": False, "materialized_output": True,
                "symbolic_tensor_input_allocations": (source.view.allocation_id,),
                "symbolic_tensor_output_allocations": (destination.view.allocation_id,)})
        return self.trace.call(node, (source,), shape=destination.shape, dtype=destination.dtype, out=destination)



class SymbolicFillZero(SymbolicOperator):
    """Typed zero writes into an existing source-owned view."""
    def __call__(self, destination, *, controls=()):
        self.trace._check_tensor(destination)
        node = self.node(WorkloadKind.ELEMENTWISE, {"elements": prod(destination.shape)}, destination.dtype,
            "elementwise", {"operation": "fill", "reads": 0, "writes": 1, "source_constant": 0})
        return self.trace.call(node, (), shape=destination.shape, dtype=destination.dtype, out=destination, controls=controls)

class SymbolicSiluAndMul(SymbolicOperator):
    def __init__(self, trace, name, pool_name, *, swiglu_limit=None, model_execution_version=None):
        super().__init__(trace, name, pool_name)
        if swiglu_limit is not None and (type(swiglu_limit) not in (int, float)
                or not isfinite(swiglu_limit) or swiglu_limit <= 0):
            raise ValueError("symbolic SwiGLU clamp must be finite and positive")
        self.swiglu_limit = swiglu_limit
        if model_execution_version is not None and model_execution_version not in (
                "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1", "vllm_0_30_0"):
            raise ValueError("symbolic SiLU requires its declared model source version")
        self.model_execution_version = model_execution_version

    def __call__(self, x):
        if x.shape[-1] % 2:
            raise ValueError("SiluAndMul requires an even gate/up width")
        width, rows = x.shape[-1] // 2, prod(x.shape[:-1])
        node = self.node(WorkloadKind.ACTIVATION,
            {"elements": rows * width, "tok": rows, "intermediate": width}, x.dtype, "silu",
            {**({} if self.swiglu_limit is None else {"swiglu_limit": self.swiglu_limit}),
             **({"model_execution_version": self.model_execution_version}
                 if self.model_execution_version is not None else {}),
             **({"source_input_layout": "contiguous"} if x.is_contiguous else {})})
        return self.trace.call(node, (x,), shape=(*x.shape[:-1], width), dtype=x.dtype)


class SymbolicBlockFP8Linear(SymbolicOperator):
    """Declared group128 FP32-scale source, BF16 input/output.

    Per-token quantized input and its scales live inside the linear wrapper.
    The logical profile permits a final partial output block and describes
    only live scale payload, not actual row/column-major allocation padding.
    Neither profile selects an installed FP8 library.
    """

    def __init__(self, trace, name, pool_name, in_features, out_features, *, source_profile):
        super().__init__(trace, name, pool_name)
        if source_profile not in ("vllm_0_30_row_major_group128_fp32", "vllm_0_30_logical_group128_fp32"):
            raise NotImplementedError("block-FP8 symbolic linear requires an explicit supported scale layout")
        self.logical_only = source_profile == "vllm_0_30_logical_group128_fp32"
        if (any(type(d) is not int or d <= 0 for d in (in_features, out_features))
                or in_features % 128 or (not self.logical_only and out_features % 128)):
            raise ValueError("block-FP8 symbolic linear requires whole128 weight blocks")
        self.in_features, self.out_features = in_features, out_features

    def __call__(self, x):
        if len(x.shape) != 2 or x.shape[-1] != self.in_features or x.dtype != "bf16":
            raise NotImplementedError("block-FP8 symbolic linear requires matching BF16 matrix rows")
        # per_token_group_quant_fp8 reads x.stride(0) explicitly and only
        # requires contiguous groups along the last axis. A latent prefix
        # sliced from MLA's wider KV workspace therefore needs no copy.
        unit_stride_rows = (x.view.strides_bytes[1] == x.view.element_bytes
            and x.view.strides_bytes[0] >= self.in_features * x.view.element_bytes)
        if not x.is_contiguous and not (self.logical_only and unit_stride_rows):
            raise NotImplementedError("block-FP8 symbolic linear requires contiguous BF16 groups")
        rows = x.shape[0]
        node = self.node(WorkloadKind.LINEAR, {"b": 1, "m": rows,
            "n": self.out_features, "k": self.in_features}, x.dtype, "gemm", {
                "has_bias": False, "weight_dtype": "fp8", "weight_group_size": 128,
                "weight_quantization": BLOCK_FP8_WEIGHT_QUANTIZATION,
                "symbolic_scale_layout": "logical_group128_fp32" if self.logical_only else "row_major_group128_fp32",
                **({"source_component": "symbolic_block_fp8_linear", "symbolic_workspace_complete": False}
                    if self.logical_only else {})})
        if self.logical_only:
            # apply_weights quantizes before apply_block_scaled_mm allocates
            # C. Ordinary wrapper locals retain both temporaries until return.
            quant_input = self.trace.tensor(node.node_id + ":quant_input", x.shape, "fp8")
            quant_scales = self.trace.tensor(node.node_id + ":quant_scales",
                (rows, self.in_features // 128), "fp32")
            result = self.trace.call(node, (x,), shape=(rows, self.out_features), dtype="bf16")
            return result, None
        # The original QuantFP8 wrapper returns Q/scales before the matrix
        # allocates C. Keep their source order and ordinary local lifetimes.
        quant_input = self.trace.tensor(node.node_id + ":quant_input", x.shape, "fp8")
        quant_scales = self.trace.tensor(node.node_id + ":quant_scales",
            (rows, self.in_features // 128), "fp32")
        output = self.trace.tensor(node.node_id + ":output", (rows, self.out_features), "bf16")
        result = self.trace.call(node, (x,), shape=output.shape, dtype=output.dtype, out=output)
        return result, None


class SymbolicAdd(SymbolicOperator):
    def __call__(self, left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("symbolic add requires equal input shape and dtype")
        node = self.node(WorkloadKind.ELEMENTWISE, {"elements": prod(left.shape)}, left.dtype, "add",
            {"operation": "add", "reads": 2, "writes": 1})
        return self.trace.call(node, (left, right), shape=left.shape, dtype=left.dtype)


class SymbolicScalarMultiply(SymbolicOperator):
    """One tensor/Python-scalar multiply, preserving source in-place output."""

    def __init__(self, trace, name, pool, scalar, *, inplace=True):
        super().__init__(trace, name, pool)
        if type(scalar) not in (float, int) or not isfinite(scalar) or scalar <= 0:
            raise ValueError("symbolic routed output scale must be a finite positive scalar")
        self.scalar = float(scalar)
        if type(inplace) is not bool:
            raise TypeError("symbolic scalar multiply requires an explicit boolean storage contract")
        self.inplace = inplace

    def __call__(self, x):
        self.trace._check_tensor(x)
        if self.scalar == 1.0 and self.inplace:
            return x
        node = self.node(WorkloadKind.ELEMENTWISE, {"elements": prod(x.shape)}, x.dtype, "elementwise", {
            "operation": "unary", "reads": 1, "writes": 1, "scalar_value": self.scalar})
        return self.trace.call(node, (x,), shape=x.shape, dtype=x.dtype, out=x if self.inplace else None)


class SymbolicRowParallelLinear:
    """A rank-local projection followed by its declared TP all-reduce.

    Keeps reduction inside the Linear call frame, preserving the original
    model forward and releasing the partial when the helper returns. The
    ordinary fresh-output collective branch does not bind a CUDA provider.
    """

    def __init__(self, projection, group, config, *, inplace=False):
        if (not isinstance(projection, (SymbolicLinear, SymbolicBlockFP8Linear))
                or not isinstance(group, ForwardCollectiveGroup)
                or not any(projection.trace is trace for trace in group.traces)
                or not isinstance(config, CollectiveConfig) or config.operation != "all_reduce"):
            raise ValueError("row-parallel reduction requires its projection's rank group and all-reduce config")
        if type(inplace) is not bool:
            raise TypeError("row-parallel collective alias requires an explicit boolean")
        self.projection, self.group, self.config, self.inplace = projection, group, config, inplace

    def __call__(self, x):
        partial, bias = self.projection(x)
        output = self.group.call(self.projection.name + ":all-reduce", partial,
            pool_name=self.projection.pool_name, config=self.config, inplace=self.inplace)
        return output, bias


VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT = "vllm_0_30_gpu_model_runner_set_forward_context"


class SymbolicSoftmaxTopK(SymbolicOperator):
    """vLLM FusedTopKRouter public tensors; no routing values are computed."""

    def __init__(self, trace, name, pool_name, model, *, model_execution_version="vllm_0_30_0",
                 source_forward_context=None):
        super().__init__(trace, name, pool_name)
        if model.model_type != "qwen3_moe" or type(model.moe_normalize_topk_probability) is not bool:
            raise ValueError("softmax top-k requires explicit Qwen3 MoE normalization semantics")
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise NotImplementedError("softmax top-k requires its version-bound source program")
        self.model = model
        self.model_execution_version = model_execution_version
        if source_forward_context is not None and (model_execution_version != "vllm_0_30_0"
                or source_forward_context != VLLM030_GPU_MODEL_RUNNER_TOPK_CONTEXT):
            raise ValueError("TopK padding context requires its exact version-bound runner source")
        self.source_forward_context = source_forward_context

    def __call__(self, x, logits):
        model, rows = self.model, x.shape[0]
        if (x.shape != (rows, model.hidden_size) or x.dtype != "bf16"
                or logits.shape != (rows, model.num_experts) or logits.dtype != "bf16"):
            raise ValueError("Qwen3 router requires BF16 hidden states and gate logits")
        fp32_source = self.model_execution_version == "vllm_0_11_0"
        source024 = self.model_execution_version == "vllm_0_24_0"
        source018 = self.model_execution_version == "vllm_0_18_0"
        if fp32_source:
            logits = SymbolicFloat32Cast(self.trace, self.name + ":logits-float", self.pool_name)(
                logits, vllm011_router_topk=model.num_experts_per_tok)
        # GPUModelRunner.execute_model omits is_padding when setting the
        # forward context; set_forward_context's source default is None.
        # This is distinct from GPU-v2's gpu/model_runner.py row mask, and
        # does not depend on the VLLM_MOE_SKIP_PADDING environment value.
        context_inputs = ({} if self.source_forward_context is None else {
            "source_topk_padding_mask_program": self.source_forward_context,
            "source_topk_padding_mask": None})
        node = self.node(WorkloadKind.VECTOR, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size}, "fp32" if fp32_source else "bf16", "moe_topk", {
                "source_component": "symbolic_softmax_topk",
                "source_program": "vllm_0_11_fp32_softmax_topk" if fp32_source else "vllm_0_18_bf16_softmax_topk" if source018 else "vllm_0_24_bf16_softmax_topk" if source024 else "vllm_0_30_bf16_softmax_topk",
                "mathematical_program": "fp32_softmax_topk_flops_bytes_v1" if fp32_source else "bf16_softmax_topk_public_math_v1" if source024 or source018 else "bf16_softmax_topk_flops_bytes_v1",
                "normalize_topk_probability": model.moe_normalize_topk_probability,
                **context_inputs,
                **({"stage": "route", "discarded_score_output_allocation_known": False} if source018 else {}),
                **({"symbolic_workspace_complete": False} if source024 or source018 else {})})
        # fused_topk returns three allocations; _compute_routing discards the
        # INT32 token_expert_indices local when returning weights and IDs.
        outputs = (
            ("weights", (rows, model.num_experts_per_tok), "fp32"),
            ("ids", (rows, model.num_experts_per_tok), "int32"),
            ("token_expert_indices", (rows, model.num_experts_per_tok), "int32"))
        if source018:
            # out_flag=False suppresses writes to the third FP32 score port.
            # Its host allocation is unbound, not an INT32 token-index matrix
            # and not a known zero-sized allocation.
            outputs = (("weights", (rows, model.num_experts_per_tok), "bf16"), outputs[1])
        if fp32_source and model.moe_normalize_topk_probability:
            #0.11 Python divides freshly allocated raw weights by their row
            # sums. These public temporaries coexist with the result and the
            # outer fused_topk's FP32 logits. This is not a fused-launch claim.
            outputs += (("raw_weights", (rows, model.num_experts_per_tok), "fp32"),
                ("selected_weight_sum", (rows, 1), "fp32"))
        result = self.trace.call_outputs(node, (logits,), outputs=outputs)
        return result[0], result[1]


class SymbolicExpertOwnerMask(SymbolicOperator):
    """Ascend AllGather dispatcher: map[ids] != -1, then fresh BF16 multiply."""

    def __init__(self, trace, name, pool, model, *, expert_rank, expert_parallel_size):
        super().__init__(trace, name, pool)
        if expert_parallel_size != 2 or type(expert_rank) is not int or expert_rank not in (0, 1) or model.num_experts % 2:
            raise ValueError("Ascend owner mask requires linear EP2 ownership")
        self.model = model
        self.expert_rank = expert_rank
        self.expert_map = trace.tensor(name + ":expert-map", (model.num_experts,), "int32", persistent=True)

    def __call__(self, weights, ids):
        rows, topk = weights.shape
        model = self.model
        if weights.dtype != "bf16" or ids.dtype != "int32" or ids.shape != weights.shape or topk != model.num_experts_per_tok:
            raise ValueError("Ascend owner mask requires BF16 weights and INT32 top-k IDs")
        node = self.node(WorkloadKind.VECTOR, {"tokens": rows, "experts": model.num_experts,
            "topk": topk, "hidden": model.hidden_size, "intermediate": model.moe_intermediate_size}, "bf16", "moe_topk", {
                "source_component": "symbolic_expert_owner_mask", "source_program": "vllm_0_18_bf16_softmax_topk",
                "mathematical_program": "bf16_softmax_topk_public_math_v1", "stage": "owner_mask",
                "normalize_topk_probability": model.moe_normalize_topk_probability,
                "expert_rank": self.expert_rank,
                "symbolic_workspace_complete": False})
        outputs = self.trace.call_outputs(node, (weights, ids, self.expert_map), outputs=(
            ("masked_weights", weights.shape, "bf16"), ("mapped_ids", ids.shape, "int32"),
            ("owner_mask", ids.shape, "bool")))
        return outputs[0]


class SymbolicK3AttentionResidual(SymbolicOperator):
    """AttnRes's fresh output, prefix mutation and block-buffer alias."""

    def __init__(self, trace, name, pool, model, *, blocks, write=-1, output_norm=True, capacity=None):
        super().__init__(trace, name, pool)
        self.width = model.hidden_size
        self.capacity = (-(-model.num_hidden_layers // model.kimi_k3_semantics.attention_residual_block_size)
            if capacity is None else capacity)
        self.blocks, self.write, self.output_norm = blocks, write, output_norm

    def __call__(self, prefix, delta, blocks):
        rows = prefix.shape[0]
        if (prefix.shape != (rows, self.width) or prefix.dtype != "bf16"
                or blocks.shape != (rows, self.capacity, self.width) or blocks.dtype != "bf16"
                or not 0 <= self.blocks <= self.capacity or not -1 <= self.write < self.capacity
                or (delta is not None and (delta.shape != prefix.shape or delta.dtype != prefix.dtype))):
            raise ValueError("K3 AttnRes requires matching BF16 prefix/delta and block backing: "
                f"prefix={prefix.shape}/{prefix.dtype}, blocks={blocks.shape}/{blocks.dtype}, "
                f"capacity={self.capacity}, num_blocks={self.blocks}, write={self.write}")
        node = self.node(WorkloadKind.VECTOR, {"rows": rows, "hidden": self.width,
            "elements": rows * self.width}, "bf16", "kimi_k3_attn_res", {
                "source_component": "vllm_0_30_kimi_k3_attention_residual",
                "num_blocks": self.blocks, "block_write_idx": self.write,
                "block_capacity": self.capacity, "has_delta": delta is not None,
                "apply_output_norm": self.output_norm, "symbolic_workspace_complete": False})
        inputs = (prefix,) + (() if delta is None else (delta,))
        if self.blocks:
            inputs += (blocks.narrow(1, 0, self.blocks),)
        writes = (() if delta is None else (prefix,))
        if self.write >= 0:
            writes += (blocks.narrow(1, self.write, 1),)
        return self.trace.call(node, inputs, shape=prefix.shape, dtype="bf16", writes=writes)


class SymbolicK3SiTU(SymbolicOperator):
    def __init__(self, trace, name, pool, model):
        super().__init__(trace, name, pool)
        self.source = model.kimi_k3_semantics

    def __call__(self, x):
        rows, packed = x.shape
        if x.dtype != "bf16" or packed % 2:
            raise ValueError("K3 SiTU requires BF16 packed gate/up rows")
        width = packed // 2
        node = self.node(WorkloadKind.ACTIVATION, {"rows": rows, "intermediate": width,
            "elements": rows * width}, "bf16", "kimi_k3_situ", {
                "source_component": "vllm_0_30_kimi_k3_ordinary_mlp",
                "activation_beta": self.source.activation_situ_beta,
                "activation_linear_beta": self.source.activation_situ_linear_beta,
                "symbolic_workspace_complete": False})
        return self.trace.call(node, (x,), shape=(rows, width), dtype="bf16")


class SymbolicK3Routing(SymbolicOperator):
    def __init__(self, trace, name, pool, model):
        super().__init__(trace, name, pool)
        self.model = model

    def __call__(self, logits):
        model, rows = self.model, logits.shape[0]
        if logits.shape != (rows, model.num_experts) or logits.dtype != "fp32":
            raise ValueError("K3 routing requires the FP32 gate output")
        node = self.node(WorkloadKind.VECTOR, {"rows": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "elements": rows * model.num_experts}, "fp32", "kimi_k3_moe_route", {
                "source_component": "vllm_0_30_kimi_k3_native_marlin_moe",
                "expert_id_selection": "score_topk", "score_function": "sigmoid",
                "expert_group_count": 1, "topk_group_count": 1,
                "normalize_topk_probability": True, "routed_scaling_factor": model.routed_scaling_factor,
                "symbolic_workspace_complete": False})
        return self.trace.call_outputs(node, (logits,), outputs=(
            ("weights", (rows, model.num_experts_per_tok), "fp32"),
            ("ids", (rows, model.num_experts_per_tok), "int32")))


class SymbolicK3Experts(SymbolicOperator):
    def __init__(self, trace, name, pool, model, *, resolution, invocation_id, layer_index,
                 tp_size=1, tensor_rank=0, expert_parallel_size=1, data_parallel_size=1, data_rank=0):
        super().__init__(trace, name, pool)
        self.model, self.resolution, self.invocation_id = model, resolution, invocation_id
        self.module = f"language_model.model.layers.{layer_index}.block_sparse_moe.experts"
        if model.weight_dtype_for_module(self.module, "RoutedExperts") != "mxfp4":
            raise NotImplementedError("K3 symbolic experts require the checkpoint MXFP4 source branch")
        if (type(tp_size) is not int or tp_size < 1 or model.moe_intermediate_size % tp_size
                or type(tensor_rank) is not int or not 0 <= tensor_rank < tp_size
                or type(data_parallel_size) is not int or data_parallel_size < 1
                or type(data_rank) is not int or not 0 <= data_rank < data_parallel_size
                or type(expert_parallel_size) is not int or expert_parallel_size not in (1, tp_size * data_parallel_size)
                or model.num_experts % expert_parallel_size):
            raise ValueError("K3 experts require whole logical TP intermediate shards and a valid lane")
        from .shape_inference import infer_gemm_dimensions
        self.intermediate = infer_gemm_dimensions(model, "experts.w2_weight", 1,
            tp_size=tp_size if expert_parallel_size == 1 else 1)[2]
        expert_tp = tp_size * data_parallel_size if expert_parallel_size == 1 else 1
        if expert_parallel_size == 1:
            if (model.moe_intermediate_size % expert_tp or self.intermediate % data_parallel_size
                    or self.intermediate // data_parallel_size % 64):
                raise ValueError("K3 DP experts require whole logical and physical intermediate shards")
            self.intermediate //= data_parallel_size
        self.tp_size, self.tensor_rank = tp_size, tensor_rank
        self.data_size, self.data_rank, self.expert_tp = data_parallel_size, data_rank, expert_tp
        self.ep_size = expert_parallel_size
        self.expert_rank = 0 if expert_parallel_size == 1 else data_rank * tp_size + tensor_rank
        self.owned = linear_expert_partition(model.num_experts, expert_parallel_size, self.expert_rank)

    def __call__(self, x, weights, ids):
        model, source, rows = self.model, self.model.kimi_k3_semantics, x.shape[0]
        counts = self.resolution.tokens_per_expert
        if (x.shape != (rows, source.routed_expert_hidden_size) or x.dtype != "bf16"
                or weights.shape != (rows, model.num_experts_per_tok) or weights.dtype != "fp32"
                or ids.shape != weights.shape or ids.dtype != "int32"
                or len(counts) != model.num_experts or sum(counts) != rows * model.num_experts_per_tok):
            raise ValueError("K3 experts require the latent, typed route outputs and a conserved explicit histogram")
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": source.routed_expert_hidden_size,
            "intermediate": self.intermediate}, "bf16", "full", {
                "source_component": "vllm_0_30_kimi_k3_marlin_experts", "module_path": self.module,
                "moe_invocation_id": self.invocation_id, "routing_provenance": dict(self.resolution.routing_provenance),
                "tokens_per_expert": counts,
                "local_tokens_per_expert": tuple(count if expert in self.owned else 0 for expert, count in enumerate(counts)),
                "expert_parallel_size": self.ep_size, "expert_rank": self.expert_rank, "owned_experts": self.owned,
                "expert_weight_tile_rows": None, "activation_beta": source.activation_situ_beta,
                "activation_linear_beta": source.activation_situ_linear_beta,
                "routed_scaling_factor": model.routed_scaling_factor, "symbolic_workspace_complete": False,
                **({"tensor_parallel_size": self.tp_size,
                    "source_tensor_parallel_rank": self.tensor_rank,
                    "expert_tensor_parallel_size": self.expert_tp,
                    "expert_tensor_rank": 0 if self.ep_size > 1 else self.data_rank * self.tp_size + self.tensor_rank,
                    "logical_intermediate": model.moe_intermediate_size // self.expert_tp,
                    "expert_output_is_reduced": False} if self.tp_size > 1 or self.data_size > 1 else {}),
                **({"source_data_parallel_size": self.data_size, "source_data_parallel_rank": self.data_rank}
                    if self.data_size > 1 else {})})
        return self.trace.call(node, (x, weights, ids), shape=x.shape, dtype=x.dtype)


class SymbolicK3Stage(SymbolicOperator):
    """Existing source primitive with explicit public output/mutation ports.

    This records tensor boundaries, not a prebuilt model graph. Dependencies
    and lifetime come from the actual forward's handles; callee scratch stays
    incomplete wherever the source contract does not describe it.
    """

    def __init__(self, trace, name, pool, *, variant, shape, attributes):
        super().__init__(trace, name, pool)
        self.variant, self.shape, self.attributes = variant, shape, attributes

    def __call__(self, *inputs, shape, dtype="bf16", out=None, writes=(), workspace=(), controls=()):
        node = self.node(WorkloadKind.VECTOR, self.shape, "bf16", self.variant,
            {**self.attributes, "symbolic_workspace_complete": False})
        scratch = tuple(self.trace.tensor(self.trace.unique_name(self.name + ":" + role), layout, dt)
            for role, layout, dt in workspace)
        return self.trace.call(node, inputs, shape=shape, dtype=dtype, out=out, writes=(*writes, *scratch), controls=controls)

    def trace_vllm030_chunk_source_body(self, q, k, v, raw_gate, raw_beta, initial_state,
            query_start, chunk_indices, chunk_offsets, final_state, *, source_spec, source_attributes):
        """Explicit nine-submission KDA source storage; no automatic lowering.

        Caller supplies actual cached metadata and the existing fresh final
        target. Its earlier caller allocation is retained; this is not an
        assertion of upstream allocator timing, compiler scratch or native ABI.
        """
        from src.kernel.kernel_spec import KernelSpec, KernelKind
        from src.kernel.predictors.vector import (kimi_k3_kda_prefill_logical_work,
            kimi_k3_kda_mixed_logical_work, kimi_k3_kda_context_prefill_logical_work)
        owners = {"kimi_k3_kda_prefill": kimi_k3_kda_prefill_logical_work,
            "kimi_k3_kda_mixed": kimi_k3_kda_mixed_logical_work,
            "kimi_k3_kda_context_prefill": kimi_k3_kda_context_prefill_logical_work}
        if (type(source_spec) is not KernelSpec or source_spec.kind is not KernelKind.VECTOR
                or source_spec.hardware_id != "nvidia_h200" or source_spec.dtype != "bf16"
                or source_spec.launch is not None or source_spec.variant not in owners
                or source_spec.variant != self.variant or source_spec.attributes["stage"] != "chunk"
                or any(source_spec.shape[name] != self.shape[name] for name in ("rows", "heads", "head_dim"))
                or any(source_spec.attributes[name] != self.attributes[name] for name in
                    ("stage", "state_indices", "state_pool_slots", "query_start_loc", "has_initial_state", "gate_lower_bound"))):
            raise ValueError("KDA source body requires its exact existing unbound parent and caller")
        logical = owners[source_spec.variant](source_spec)
        rows, heads, dim = (source_spec.shape[name] for name in ("rows", "heads", "head_dim"))
        lengths, chunks = logical["query_lengths"], logical["chunks"]
        tensors = (q, k, v, raw_gate, raw_beta, initial_state, query_start, chunk_indices, chunk_offsets, final_state)
        for tensor in tensors:
            self.trace._check_tensor(tensor)
        expected_indices = tuple((sequence, chunk) for sequence, length in enumerate(lengths)
            for chunk in range((length + 63) // 64))
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + (length + 63) // 64)
        expected = {"model_execution_version": "vllm_0_30_0",
            "source_norm_backend": "l2norm_fwd_kernel2", "source_use_qk_l2norm_in_kernel": True,
            "source_output_final_state": True, "source_out": "original_v",
            "source_metadata_state": "cached_actual_cu_seqlens_identity",
            "source_query_start_view_id": query_start.view.view_id,
            "source_chunk_indices": expected_indices, "source_chunk_offsets": tuple(offsets)}
        if (type(source_attributes) not in (dict, MappingProxyType) or set(source_attributes) != set(expected)
                or any(type(source_attributes[name]) is not type(value) or source_attributes[name] != value
                    for name, value in expected.items()) or source_spec.attributes["gate_lower_bound"] != -5):
            raise ValueError("KDA source body requires explicit prepared safe-gate/norm/metadata caller facts")
        dense = (1, rows, heads, dim)
        if (any(tensor.shape != dense or tensor.dtype != "bf16" or not tensor.is_contiguous
                for tensor in (q, k, v, raw_gate))
                or raw_beta.shape != (1, rows, heads) or raw_beta.dtype != "bf16"
                or initial_state.shape != (len(lengths), heads, dim, dim) or initial_state.dtype != "fp32"
                or not initial_state.is_contiguous or final_state.shape != initial_state.shape
                or final_state.dtype != "fp32" or not final_state.is_contiguous
                or query_start.shape != (len(lengths) + 1,) or query_start.dtype != "int32"
                or chunk_indices.shape != (chunks, 2) or chunk_indices.dtype != "int32"
                or chunk_offsets.shape != (len(lengths) + 1,) or chunk_offsets.dtype != "int64"
                or any(not tensor.is_contiguous for tensor in (query_start, chunk_indices, chunk_offsets))
                or v.view.allocation_id in {tensor.view.allocation_id for tensor in tensors if tensor is not v}
                or final_state.view.allocation_id in {tensor.view.allocation_id for tensor in tensors[:-1]}
                or self.trace._writers[final_state.view.allocation_id] is not None
                or self.trace._readers[final_state.view.allocation_id]):
            raise ValueError("KDA source body requires exact dense source ports, cached typed metadata and fresh FP32 final target")
        parent = {name: source_spec.attributes[name] for name in source_spec.attributes}
        dynamic = {**parent, **source_attributes, "source_component": "symbolic_vllm030_kda_chunk_stage",
            "source_parent_variant": self.variant, "source_parent_name": self.name,
            "symbolic_workspace_complete": False, "source_native_memory_ABI_bound": False,
            "source_compiler_resources_bound": False, "source_complete_worker_peak": False}

        def node(role, dtype, reads, writes):
            operator = SymbolicOperator(self.trace, self.name + ":" + role, self.pool_name)
            regions = {"intra_sub_chunk": "Aqk_diagonal16_blocks_and_FP32_Akkd_live_rows",
                "inter_solve": "Aqk_lower_off_diagonal16_blocks_and_Akk_lower_block_triangle"}
            return operator.node(WorkloadKind.VECTOR, self.shape, dtype, "kda_chunk_" + role,
                {**dynamic, "source_stage_role": role, "memory_input_roles": reads, "memory_output_roles": writes,
                    **({"source_partial_write_regions": regions[role]} if role in regions else {})})

        qn = self.trace.call(node("q_l2norm", "bf16", ("q",), ("q_normalized",)),
            (q,), shape=dense, dtype="bf16")
        kn = self.trace.call(node("k_l2norm", "bf16", ("k",), ("k_normalized",)),
            (k,), shape=dense, dtype="bf16", controls=(qn,))
        parameters = (self.trace.parameter_view(self.name + ":A_log", (heads,), "fp32"),
            self.trace.parameter_view(self.name + ":dt_bias", (heads * dim,), "fp32"))
        gate, beta = self.trace.call_outputs(node("gate_cumsum_beta", "fp32",
            ("raw_gate", "raw_beta", "query_start", "chunk_indices", "A_log", "dt_bias"),
            ("cumulative_gate_log2", "sigmoid_beta")), (raw_gate, raw_beta, query_start, chunk_indices),
            outputs=(("gate", dense, "fp32"), ("beta", (1, rows, heads), "fp32")),
            controls=(kn,), parameters=parameters)
        Aqk = self.trace.tensor(self.trace.unique_name(self.name + ":Aqk"), (1, rows, heads, 64), "bf16")
        Akk = self.trace.tensor(self.trace.unique_name(self.name + ":Akk"), Aqk.shape, "bf16")
        Akkd = self.trace.tensor(self.trace.unique_name(self.name + ":Akkd"), (1, rows, heads, 16), "fp32")
        self.trace.call(node("zero_Akk", "bf16", (), ("Akk",)), (),
            shape=Akk.shape, dtype="bf16", out=Akk, controls=(gate, beta))
        self.trace.call(node("intra_sub_chunk", "fp32",
            ("q_normalized", "k_normalized", "gate", "beta", "query_start", "chunk_indices"), ("Akkd", "Aqk")),
            (qn, kn, gate, beta, query_start, chunk_indices), shape=Akkd.shape, dtype="fp32",
            out=Akkd, writes=(Aqk,), controls=(Akk,))
        self.trace.call(node("inter_solve", "bf16",
            ("q_normalized", "k_normalized", "gate", "beta", "Akkd", "query_start", "chunk_indices"), ("Akk", "Aqk")),
            (qn, kn, gate, beta, Akkd, query_start, chunk_indices), shape=Akk.shape, dtype="bf16", out=Akk, writes=(Aqk,))
        del Akkd
        w, u, kg = self.trace.call_outputs(node("recompute_w_u", "bf16",
            ("k_normalized", "v", "beta", "Akk", "gate", "query_start", "chunk_indices"), ("w", "u", "kg")),
            (kn, v, beta, Akk, gate, query_start, chunk_indices),
            outputs=(("w", dense, "bf16"), ("u", dense, "bf16"), ("kg", dense, "bf16")))
        del Akk
        h = self.trace.tensor(self.trace.unique_name(self.name + ":chunk_h"), (1, chunks, heads, dim, dim), "bf16")
        v_new = self.trace.tensor(self.trace.unique_name(self.name + ":v_new"), dense, "bf16")
        self.trace.call(node("state", "bf16",
            ("kg", "w", "u", "gate", "initial_state", "query_start", "chunk_offsets"), ("chunk_h", "v_new", "final_state")),
            (kg, w, u, gate, initial_state, query_start, chunk_offsets), shape=h.shape, dtype="bf16",
            out=h, writes=(v_new, final_state))
        del w, u, kg
        output = self.trace.call(node("output", "bf16",
            ("q_normalized", "v_new", "gate", "chunk_h", "Aqk", "query_start", "chunk_indices"), ("output_original_v",)),
            (qn, v_new, gate, h, Aqk, query_start, chunk_indices), shape=dense, dtype="bf16", out=v)
        return output, final_state



def unquantized_expert_source_metadata(model, resolution, invocation_id, *, rows,
        expert_parallel_size, expert_rank):
    """Pure current histogram/placement metadata shared by source and binder."""
    from .runtime_inputs import ExpertRoutingResolution
    if (not isinstance(resolution, ExpertRoutingResolution) or type(rows) is not int or rows <= 0
            or type(expert_parallel_size) is not int or expert_parallel_size < 1
            or type(expert_rank) is not int or not 0 <= expert_rank < expert_parallel_size):
        raise ValueError("expert source metadata requires its typed routing and placement")
    counts = resolution.tokens_per_expert
    if len(counts) != model.num_experts or sum(counts) != rows * model.num_experts_per_tok:
        raise ValueError("expert histogram must conserve every top-k assignment: "
            f"invocation={invocation_id!r}, rows={rows}, topk={model.num_experts_per_tok}, "
            f"experts={len(counts)}, assignments={sum(counts)}, "
            f"provenance={dict(resolution.routing_provenance)!r}")
    placement = {}
    if expert_parallel_size > 1:
        owned = linear_expert_partition(model.num_experts, expert_parallel_size, expert_rank)
        local_counts = [0] * len(counts)
        for expert in owned:
            local_counts[expert] = counts[expert]
        placement = {"expert_rank": expert_rank, "owned_experts": owned,
            "local_tokens_per_expert": tuple(local_counts)}
    return {"tokens_per_expert": counts, **placement, "moe_invocation_id": invocation_id,
        "routing_provenance": resolution.routing_provenance}


class SymbolicUnquantizedExperts(SymbolicOperator):
    """Shared BF16 expert boundary with an explicit histogram and TP shard."""

    def __init__(self, trace, name, pool_name, model, *, tp_size, resolution, invocation_id, model_execution_version,
                 expert_parallel_size=1, expert_rank=0, inplace=False, routing_weight_dtype="fp32"):
        from .runtime_inputs import ExpertRoutingResolution
        super().__init__(trace, name, pool_name)
        if (model.compute_dtype != "bf16" or model.weight_dtype != "bf16"
                or type(tp_size) is not int or tp_size < 1
                or type(expert_parallel_size) is not int or expert_parallel_size < 1
                or (expert_parallel_size > 1 and expert_parallel_size != tp_size)
                or model.moe_intermediate_size * expert_parallel_size % tp_size
                or type(expert_rank) is not int or not 0 <= expert_rank < expert_parallel_size
                or not isinstance(resolution, ExpertRoutingResolution) or type(inplace) is not bool
                or routing_weight_dtype not in ("bf16", "fp32")):
            raise ValueError("unquantized experts require BF16 TP shards and explicit normalized routing")
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise ValueError("unquantized expert parameter source requires its execution version")
        self.model_execution_version = model_execution_version
        if model_execution_version != "vllm_0_30_0" and (expert_parallel_size not in (1, 2) or model.num_experts % expert_parallel_size):
            raise NotImplementedError("other model versions retain their existing EP1/EP2 source domain")
        self.model, self.tp_size, self.resolution, self.invocation_id = model, tp_size, resolution, invocation_id
        self.ep_size, self.ep_rank = expert_parallel_size, expert_rank
        self.inplace = inplace
        self.routing_weight_dtype = routing_weight_dtype

    def __call__(self, x, weights, ids):
        model, rows = self.model, x.shape[0]
        if (x.shape != (rows, model.hidden_size) or x.dtype != "bf16"
                or weights.shape != (rows, model.num_experts_per_tok) or weights.dtype != self.routing_weight_dtype
                or ids.shape != weights.shape or ids.dtype != "int32"):
            raise ValueError("unquantized experts require BF16 inputs, source-typed routing weights and INT32 IDs")
        dynamic = unquantized_expert_source_metadata(model, self.resolution, self.invocation_id,
            rows=rows, expert_parallel_size=self.ep_size, expert_rank=self.ep_rank)
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size * self.ep_size // self.tp_size}, "bf16", "full", {
                "source_component": "symbolic_unquantized_experts", **dynamic,
                "expert_parallel_size": self.ep_size, "expert_placement_strategy": "linear",
                "routing_weight_dtype": self.routing_weight_dtype,
                "model_execution_version": self.model_execution_version,
                "memory_input_roles": ("hidden_states", "routing_weights", "expert_ids", "w13_weight", "w2_weight"),
                "memory_output_roles": ("output",),
                "expert_parameter_layout": "unquantized_checkpoint_expert_major_nk",
                "symbolic_workspace_complete": False})
        # UnquantizedFusedMoEMethod.create_weights declares these persistent
        # public parameters. A backend-formatted representation is a separate
        # callee fact; this trace does not infer one or invent its workspace.
        local_experts = len(linear_expert_partition(model.num_experts, self.ep_size, self.ep_rank))
        if not local_experts:
            node = node.with_attributes({**node.attributes,
                "memory_input_roles": ("hidden_states", "routing_weights", "expert_ids"),
                "symbolic_zero_extent_parameter_roles": ("w13_weight", "w2_weight")})
            return self.trace.call(node, (x, weights, ids), shape=x.shape, dtype=x.dtype)
        intermediate = model.moe_intermediate_size * self.ep_size // self.tp_size
        parameters = (
            self.trace.parameter_view(self.name + ":w13_weight",
                (local_experts, 2 * intermediate, model.hidden_size), "bf16"),
            self.trace.parameter_view(self.name + ":w2_weight",
                (local_experts, model.hidden_size, intermediate), "bf16"))
        return self.trace.call(node, (x, weights, ids), shape=x.shape, dtype=x.dtype,
            out=x if self.inplace else None, parameters=parameters)


class SymbolicGroupedRouting(SymbolicOperator):
    """Declared GLM/Kimi grouped top-k, without evaluating score values.

    Gate logits retain their BF16/FP32 source type; correction bias and
    output weights are FP32, and exported expert IDs are INT32.
    The explicit useful-math program prices arithmetic/public payload only.
    Complete native selection, finite-precision equivalence and actual
    expert IDs are not inferred here.
    """

    def __init__(self, trace, name, pool_name, model, *, apply_routed_scale_to_output=False):
        super().__init__(trace, name, pool_name)
        if (model.model_type not in ("glm_moe_dsa", "kimi_k2") or model.target_semantics is None
                or type(apply_routed_scale_to_output) is not bool
                or apply_routed_scale_to_output != (model.model_type == "kimi_k2")):
            raise NotImplementedError("symbolic grouped routing requires its GLM/Kimi scale-placement source branch")
        self.model = model
        self.variant = "kimi_moe_route" if model.model_type == "kimi_k2" else "dsv32_moe_route"
        self.logits_dtype = model.target_semantics.moe_router_dtype
        self.scale = 1.0 if apply_routed_scale_to_output else model.routed_scaling_factor
        self.bias = trace.tensor(name + ":correction-bias", (model.num_experts,), "fp32", persistent=True)

    def __call__(self, hidden_states, logits):
        model, rows = self.model, hidden_states.shape[0]
        if (hidden_states.shape != (rows, model.hidden_size) or hidden_states.dtype != "bf16"
                or logits.shape != (rows, model.num_experts) or logits.dtype != self.logits_dtype):
            raise ValueError("grouped routing requires BF16 hidden rows and checkpoint/source gate logits")
        semantics = model.target_semantics
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * model.num_experts,
            "rows": rows, "experts": model.num_experts, "topk": model.num_experts_per_tok}, self.logits_dtype, self.variant, {
                "source_component": "vllm_0_30_symbolic_grouped_routing", "expert_id_selection": "score_topk",
                "routed_scaling_factor": self.scale, "score_function": semantics.moe_score_function,
                "expert_group_count": semantics.moe_group_count, "topk_group_count": semantics.moe_topk_group_count,
                "normalize_topk_probability": semantics.moe_normalize_topk_probability,
                "mathematical_program": "biased_sigmoid_topk_flops_bytes_v1"})
        return self.trace.call_outputs(node, (logits, self.bias), outputs=(
            ("topk_weights", (rows, model.num_experts_per_tok), "fp32"),
            ("topk_ids", (rows, model.num_experts_per_tok), "int32")))


class SymbolicRoutedExperts(SymbolicOperator):
    """EP-off rank-local modular experts with declared route histogram.

    The public input/output storage and existing canonical expert recipe are
    reused. Unknown modular/callee workspaces are not replaced with zeros.
    """

    def __init__(self, trace, name, pool_name, model, *, layer_index, tp_size, resolution, invocation_id, routing_source_stage,
                 expert_parallel_size=1, expert_rank=0, pcp_source=False, expert_tp_size=None):
        from .runtime_inputs import ExpertRoutingResolution
        super().__init__(trace, name, pool_name)
        stage = {"glm_moe_dsa": "dsv32_moe_route", "kimi_k2": "kimi_moe_route"}
        if (model.model_type not in stage or model.compute_dtype != "bf16" or type(tp_size) is not int
                or tp_size <= 0 or (expert_parallel_size == 1 and model.moe_intermediate_size % tp_size)
                or routing_source_stage != stage[model.model_type]):
            raise NotImplementedError("symbolic grouped experts require BF16-compute TP shards and explicit grouped routing")
        expert_tp = (tp_size if expert_parallel_size == 1 else 1) if expert_tp_size is None else expert_tp_size
        if (type(expert_tp) is not int or expert_tp < 1 or model.moe_intermediate_size % expert_tp
                or (expert_parallel_size > 1 and expert_tp != 1)
                or (expert_tp_size is not None and model.model_type != "kimi_k2")
                or (model.model_type == "kimi_k2" and model.moe_intermediate_size // expert_tp % model.weight_group_size)):
            raise ValueError("expert tensor shards must preserve the exact checkpoint quantization groups")
        ordinary = type(layer_index) is int and 0 <= layer_index < model.num_hidden_layers and model.layer_uses_moe(layer_index)
        appended_glm = (type(layer_index) is int and layer_index == model.num_hidden_layers
            and model.model_type == "glm_moe_dsa" and model.mtp_architecture == "glm_moe_dsa_mtp"
            and model.num_mtp_layers == 1 and model.mtp_layer_uses_moe(0))
        if not ordinary and not appended_glm:
            raise ValueError("symbolic experts require an accurate checkpoint MoE layer")
        if type(pcp_source) is not bool or (pcp_source and model.model_type != "glm_moe_dsa"):
            raise ValueError("PCP routed expert source requires an explicit GLM source")
        if (type(expert_parallel_size) is not int or type(expert_rank) is not int
                or expert_parallel_size < 1 or not 0 <= expert_rank < expert_parallel_size
                or (not pcp_source and not (model.model_type == "glm_moe_dsa" and (model.weight_dtype == "bf16"
                        or model.weight_dtype == "fp8" and model.num_experts % expert_parallel_size == 0))
                    and not (model.model_type == "kimi_k2" and (expert_parallel_size in (1, tp_size) or expert_tp_size is not None)
                        and model.num_experts % expert_parallel_size == 0)
                    and (expert_parallel_size not in (1, 8)
                    or (expert_parallel_size == 8 and (model.model_type not in ("kimi_k2", "glm_moe_dsa") or tp_size != 8))
                    or model.num_experts % expert_parallel_size))):
            raise ValueError("symbolic experts require EP1 or GLM/K2.6 TP=EP8 with linear expert ownership")
        if not isinstance(resolution, ExpertRoutingResolution) or not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("symbolic grouped experts require explicit normalized routing and invocation identity")
        self.model, self.tp_size, self.resolution = model, tp_size, resolution
        self.expert_tp_size = expert_tp
        self.expert_parallel_size, self.expert_rank = expert_parallel_size, expert_rank
        from src.kernel.kernel_spec import linear_expert_partition
        self.owned_experts = linear_expert_partition(model.num_experts, expert_parallel_size, expert_rank)
        root = "language_model.model" if model.model_type == "kimi_k2" else "model"
        self.weight_dtype = model.weight_dtype_for_module(f"{root}.layers.{layer_index}.mlp.experts.0.gate_proj", "Linear")
        if self.weight_dtype not in ("bf16", "fp8", "int4"):
            raise NotImplementedError("symbolic experts require BF16, FP8 or INT4 checkpoint storage")
        self.group_size = model.weight_group_size if self.weight_dtype == "int4" else 0
        self.invocation_id, self.routing_source_stage = invocation_id, routing_source_stage

    def routing_metadata(self, rows):
        """Current physical routing fields, also consumed by source binding."""
        counts, model = self.resolution.tokens_per_expert, self.model
        if len(counts) != model.num_experts or sum(counts) != rows * model.num_experts_per_tok:
            raise ValueError("expert histogram must conserve all token/top-k assignments")
        return {"tokens_per_expert": counts,
            "local_tokens_per_expert": (counts if self.expert_parallel_size == 1 else
                tuple(count if expert in self.owned_experts else 0 for expert, count in enumerate(counts))),
            "moe_invocation_id": self.invocation_id,
            "routing_provenance": dict(self.resolution.routing_provenance)}

    def __call__(self, x, weights, ids):
        model, rows = self.model, x.shape[0]
        if (x.shape != (rows, model.hidden_size) or x.dtype != "bf16"
                or weights.shape != (rows, model.num_experts_per_tok) or weights.dtype != "fp32"
                or ids.shape != weights.shape or ids.dtype != "int32"):
            raise ValueError("modular experts require BF16 hidden, FP32 weights and INT32 IDs")
        routing = self.routing_metadata(rows)
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": model.num_experts_per_tok, "hidden": model.hidden_size,
            "intermediate": model.moe_intermediate_size // self.expert_tp_size}, "bf16", "full", {
                "source_component": "symbolic_grouped_routed_experts", "expert_parallel_size": self.expert_parallel_size, "expert_rank": self.expert_rank,
                "owned_experts": self.owned_experts, **routing, "expert_placement_strategy": "linear",
                "output_is_reduced": False,
                "routed_scaling_factor": 1.0 if model.model_type == "kimi_k2" else model.routed_scaling_factor,
                "routed_weight_dtype": self.weight_dtype,
                "checkpoint_weight_dtype": self.weight_dtype,
                "weight_group_size": self.group_size, "routing_source_stage": self.routing_source_stage,
                "symbolic_workspace_complete": False})
        if not self.owned_experts:
            # An empty linear EP owner still participates in the real routed
            # combine. Reuse the public zero-contribution contract: there are
            # no parameter allocations, and lowering prices an output fill.
            if self.weight_dtype != "bf16":
                raise NotImplementedError("empty grouped expert source requires the BF16 public parameter layout")
            node = node.with_attributes({**node.attributes,
                "model_execution_version": "vllm_0_30_0",
                "memory_input_roles": ("hidden_states", "routing_weights", "expert_ids"),
                "memory_output_roles": ("output",),
                "expert_parameter_layout": "unquantized_checkpoint_expert_major_nk",
                "symbolic_zero_extent_parameter_roles": ("w13_weight", "w2_weight")})
        return self.trace.call(node, (x, weights, ids), shape=x.shape, dtype=x.dtype)


def _v4_source_block_geometry(model, tp_size, *, expert_parallel):
    """Low-level shared blocks admit the separate, validated draft geometry."""
    from .model_config import ModelConfig
    if not isinstance(model, ModelConfig):
        return False
    if model.is_dspark_draft:
        from src.serving.parallelism.parallel_config import ParallelConfig
        from src.serving.parallelism.tp import validate_dspark_parallelism
        # Shared/insert callers ask about full-expert geometry even for TP1;
        # the enclosing standalone entry separately validates its actual EP flag.
        validate_dspark_parallelism(model, ParallelConfig(tensor_parallel_size=tp_size,
            enable_expert_parallel=expert_parallel and tp_size > 1))
        return True
    return is_deepseek_v4_parallel_model_scope(model, tp_size, 1, expert_parallel=expert_parallel)


class SymbolicV4Routing(SymbolicOperator):
    """Pure-text modular top6 source; no logits or hash table are evaluated."""

    def __init__(self, trace, name, pool_name, model, *, layer_index, valid_rows=None):
        super().__init__(trace, name, pool_name)
        if model.model_type != "deepseek_v4" or model.num_experts_per_tok != 6:
            raise NotImplementedError("symbolic V4 routing requires the checkpoint pure-text top6 source")
        layers = model.dspark_layer_indices if model.is_dspark_draft else range(model.num_hidden_layers)
        if type(layer_index) is not int or layer_index not in layers:
            raise ValueError("symbolic routing layer must belong to its actual target or draft model")
        self.model, self.layer_index = model, layer_index
        if valid_rows is not None and (type(valid_rows) is not int or valid_rows < 0 or layer_index >= model.num_hash_layers):
            raise ValueError("V4 padding-mask source is restricted to hash routing with an explicit valid-row prefix")
        self.valid_rows = valid_rows

    def __call__(self, hidden_states, router_logits, input_ids):
        model, rows = self.model, hidden_states.shape[0]
        for tensor in (hidden_states, router_logits):
            self.trace._check_tensor(tensor)
        if input_ids is not None:
            self.trace._check_tensor(input_ids)
        if (hidden_states.shape != (rows, model.hidden_size) or hidden_states.dtype != "bf16"
                or router_logits.shape != (rows, model.num_experts) or router_logits.dtype != "fp32"
                or input_ids is None or input_ids.shape != (rows,) or input_ids.dtype != "int32"):
            raise ValueError("V4 modular routing requires BF16 hidden, FP32 logits and INT32 token IDs")
        hashed = self.layer_index < model.num_hash_layers
        padding = None
        if self.valid_rows is not None:
            if self.valid_rows > rows:
                raise ValueError("V4 valid routing rows cannot exceed the physical SP shard")
            padding = self.trace.tensor(self.trace.unique_name(self.name + ":padding-mask"), (rows,), "bool")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * model.num_experts,
            "rows": rows, "experts": model.num_experts, "topk": 6}, "fp32", "dsv4_moe_route", {
                "source_component": "vllm_0_30_symbolic_v4_routing",
                "expert_id_selection": "hash_table" if hashed else "score_topk",
                "routed_scaling_factor": model.routed_scaling_factor,
                **({"symbolic_padding_valid_prefix": self.valid_rows,
                    "source_padding_expert_ids": -1, "source_padding_weights": 0} if padding is not None else {}),
                "routing_source": "vllm_0_30_cuda_hash_top6_pure_text" if hashed else "vllm_0_30_triton_top6_pure_text"})
        inputs = (router_logits, input_ids) if hashed else (router_logits,)
        if padding is not None:
            inputs += (padding,)
        return self.trace.call_outputs(node, inputs, outputs=(
            ("topk_weights", (rows, 6), "fp32"), ("topk_ids", (rows, 6), "int32")),
            workspace=(("token_expert_indices", (rows, 6), "int32"),) if hashed else ())


class SymbolicDSANormRope(SymbolicOperator):
    """V32 common/kernels.py fused_norm_rope, BF16 MLA/FP8 index cache.

    Caller-allocated latent/rope outputs remain live even in decode. The
    shared-indexer branch neither clears nor overwrites the shared top-k.
    Metadata/cache slots are declared valid; no physical pages are invented.
    """

    def __init__(self, trace, name, pool_name, model, *, has_indexer, cache, index_cache,
                 topk_buffer, rotary_cache, index_rotary_cache, slot_mapping, outputs_only=False, no_metadata=False):
        super().__init__(trace, name, pool_name)
        if model.model_type != "glm_moe_dsa" or type(has_indexer) is not bool:
            raise NotImplementedError("symbolic DSA fused norm requires GLM and an explicit indexer branch")
        if type(outputs_only) is not bool:
            raise TypeError("DSA outputs-only mode must be an explicit boolean")
        if type(no_metadata) is not bool or no_metadata and not outputs_only:
            raise ValueError("DSA metadata=None must explicitly omit all cache and slot pointers")
        self.no_metadata = no_metadata
        if outputs_only and any(value is not None for value in (cache, index_cache, slot_mapping)):
            raise ValueError("PCP norm outputs-only branch must not receive cache or slot pointers")
        if not outputs_only and has_indexer != (index_cache is not None):
            raise ValueError("DSA full indexer requires its cache and shared layers must not insert into it")
        for tensor in (topk_buffer, rotary_cache, index_rotary_cache, *((cache, slot_mapping) if not outputs_only else ())):
            trace._check_tensor(tensor)
        if ((not outputs_only and (len(cache.shape) != 2 or cache.dtype != "bf16"
                or cache.shape[-1] != model.kv_lora_rank + model.qk_rope_head_dim
                or len(slot_mapping.shape) != 1 or slot_mapping.dtype != "int64"))
                or len(topk_buffer.shape) != 2 or topk_buffer.dtype != "int32"
                or topk_buffer.shape[-1] != model.index_topk
                or any(len(table.shape) != 2 or table.dtype != "bf16"
                    or table.shape[-1] != model.qk_rope_head_dim for table in (rotary_cache, index_rotary_cache))):
            raise ValueError("DSA fused norm requires typed MLA cache, top-k, RoPE and slot metadata")
        if index_cache is not None:
            trace._check_tensor(index_cache)
            if (len(index_cache.shape) != 2 or index_cache.dtype != "uint8"
                    or index_cache.shape[-1] != model.index_head_dim + model.index_head_dim // 128 * 4):
                raise ValueError("DSA index cache requires FP8 values and per-group FP32 scale backing")
        self.model, self.has_indexer = model, has_indexer
        self.outputs_only = outputs_only
        self.cache, self.index_cache, self.topk_buffer = cache, index_cache, topk_buffer
        self.rotary_cache, self.index_rotary_cache, self.slot_mapping = rotary_cache, index_rotary_cache, slot_mapping

    def __call__(self, positions, q_c, kv_c, k_pe, index_k, *, kv_c_out, k_pe_out, index_k_out=None):
        model, rows = self.model, positions.shape[0]
        if ((q_c.shape, kv_c.shape, k_pe.shape) != ((rows, model.q_lora_rank),
                (rows, model.kv_lora_rank), (rows, model.qk_rope_head_dim))
                or any(x.dtype != "bf16" for x in (q_c, kv_c, k_pe, kv_c_out, k_pe_out))
                or kv_c_out.shape != kv_c.shape or k_pe_out.shape != k_pe.shape
                or positions.dtype != "int64" or (not self.outputs_only and self.slot_mapping.shape != (rows,))
                or self.topk_buffer.shape[0] < rows):
            raise ValueError("DSA fused norm inputs/outputs disagree with the declared BF16 source")
        inputs = (positions, q_c, kv_c, k_pe, self.rotary_cache, *((self.slot_mapping,) if not self.outputs_only else ()))
        writes = (kv_c_out, k_pe_out, *((self.cache,) if not self.outputs_only else ()))
        if self.has_indexer:
            if index_k is None or index_k.shape != (rows, model.index_head_dim) or index_k.dtype != "bf16":
                raise ValueError("DSA full indexer requires BF16 index K")
            inputs += (index_k, self.index_rotary_cache)
            if self.outputs_only:
                if self.no_metadata and index_k_out is not None:
                    raise ValueError("CP1 metadata=None norm cannot invent a PCP index-K output")
                if not self.no_metadata and (index_k_out is None or index_k_out.shape != index_k.shape or index_k_out.dtype != "bf16"):
                    raise ValueError("PCP indexer normalization requires its real BF16 gathered-key output")
                writes += ((self.topk_buffer.narrow(0, 0, rows),) if self.no_metadata else (index_k_out, self.topk_buffer))
            else:
                if index_k_out is not None:
                    raise ValueError("cache-writing norm cannot invent a separate index-K output")
                writes += (self.index_cache, self.topk_buffer)
        elif index_k is not None:
            raise ValueError("shared DSA layer must not run indexer K norm/cache insertion")
        elif index_k_out is not None:
            raise ValueError("shared DSA layer must not invent an index-K output")
        node = self.node(WorkloadKind.VECTOR, {"rows": rows, "q_hidden": model.q_lora_rank,
            "kv_hidden": model.kv_lora_rank, "rope_dim": model.qk_rope_head_dim,
            "index_dim": model.index_head_dim, "topk": model.index_topk, "dtype_bytes": 2,
            "position_bytes": 8, "slot_bytes": 8, "rope_cache_bytes": 2, "index_affine_bytes": 4},
            "bf16", "dsv32_fused_norm_rope", {
                "source_component": "vllm_0_30_symbolic_dsa", "has_indexer": self.has_indexer,
                "index_rope_interleave": model.target_semantics.indexer_rope_layout == "interleaved",
                "mla_cache_dtype": "bf16", "cache_slot_scope": (
                    "no_metadata_no_cache_slots" if self.no_metadata else
                    "outputs_only_no_cache_slots" if self.outputs_only else "all_rows_valid_slots_projection"),
                **({"source_no_attention_metadata": True, "index_k_output_present": False,
                    "topk_active_fill": -1 if self.has_indexer else None} if self.no_metadata else {})})
        return self.trace.call(node, inputs, shape=q_c.shape, dtype="bf16", writes=writes)


class SymbolicDSAQuery(SymbolicOperator):
    """V32 fused_q with BF16 unquantized MQA and FP8 indexer outputs.

    Even the shared branch allocates its two tiny dummy-shaped outputs, as
    the wrapper does before HAS_INDEXER skips all corresponding payload work.
    """

    def __init__(self, trace, name, pool_name, model, *, has_indexer, rotary_cache,
                 index_rotary_cache, dummy_index_q, dummy_index_weights):
        super().__init__(trace, name, pool_name)
        if model.model_type != "glm_moe_dsa" or type(has_indexer) is not bool:
            raise NotImplementedError("symbolic DSA query requires GLM and an explicit indexer branch")
        for tensor in (rotary_cache, index_rotary_cache, dummy_index_q, dummy_index_weights):
            trace._check_tensor(tensor)
        if (dummy_index_q.shape != (1, 1, 1) or dummy_index_q.dtype != "bf16"
                or dummy_index_weights.shape != (1, 1) or dummy_index_weights.dtype != "fp32"):
            raise ValueError("shared DSA query requires the typed cached source dummy shapes")
        self.model, self.has_indexer = model, has_indexer
        self.rotary_cache, self.index_rotary_cache = rotary_cache, index_rotary_cache
        self.dummy_index_q, self.dummy_index_weights = dummy_index_q, dummy_index_weights

    def __call__(self, positions, q_pe, ql_nope, index_q, index_weights):
        for tensor in (positions, q_pe, ql_nope):
            self.trace._check_tensor(tensor)
        model, (rows, heads, rope) = self.model, q_pe.shape
        if (q_pe.dtype != "bf16" or rope != model.qk_rope_head_dim
                or ql_nope.shape != (rows, heads, model.kv_lora_rank) or ql_nope.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"):
            raise ValueError("DSA fused query requires the unquantized BF16 MQA source")
        inputs = (positions, q_pe, self.rotary_cache)
        if self.has_indexer:
            if (index_q is None or index_weights is None
                    or index_q.shape != (rows, model.index_n_heads, model.index_head_dim)
                    or index_weights.shape != (rows, model.index_n_heads)
                    or index_q.dtype != "bf16" or index_weights.dtype != "bf16"):
                raise ValueError("DSA full indexer requires BF16 Q and weights")
            inputs += (index_q, index_weights, self.index_rotary_cache)
        elif index_q is not None or index_weights is not None:
            raise ValueError("shared DSA query must not evaluate an indexer")
        else:
            index_q, index_weights = self.dummy_index_q, self.dummy_index_weights
        node = self.node(WorkloadKind.VECTOR, {"rows": rows, "q_heads": heads, "rope_dim": rope,
            "index_heads": model.index_n_heads, "index_dim": model.index_head_dim,
            "dtype_bytes": 2, "position_bytes": 8, "rope_cache_bytes": 2,
            "index_weight_bytes": 2, "q_scale_bytes": 4}, "bf16", "dsv32_fused_q", {
                "source_component": "vllm_0_30_symbolic_dsa", "has_indexer": self.has_indexer,
                "index_rope_interleave": model.target_semantics.indexer_rope_layout == "interleaved",
                "quantize_mqa": False})
        # ql_nope is passed to the source wrapper but not read without MQA
        # quantization. Keep its ready-order as a control, not a payload read.
        if not self.has_indexer:
            mqa_q = self.trace.tensor(node.node_id + ":mqa_q", q_pe.shape, "bf16")
            index_fp8 = self.trace.tensor(node.node_id + ":index_q_fp8", index_q.shape, "fp8")
            weights_fp32 = self.trace.tensor(node.node_id + ":index_weights_out", index_weights.shape, "fp32")
            self.trace.call(node, inputs, shape=mqa_q.shape, dtype=mqa_q.dtype, out=mqa_q, controls=(ql_nope,))
            return index_fp8, weights_fp32, mqa_q
        mqa_q, index_fp8, weights_fp32 = self.trace.call_outputs(node, inputs, outputs=(
            ("mqa_q", q_pe.shape, "bf16"), ("index_q_fp8", index_q.shape, "fp8"),
            ("index_weights_out", index_weights.shape, "fp32")), controls=(ql_nope,))
        return index_fp8, weights_fp32, mqa_q


def _dsa_source_phase(q_lens, kv_lens, phases, prefill_source, num_speculative_tokens=0,
                      *, flattened_decode=False):
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise NotImplementedError("DSA source requires an explicit nonnegative integer speculation depth")
    threshold = num_speculative_tokens + 1
    if (not isinstance(q_lens, tuple) or not q_lens or not isinstance(kv_lens, tuple)
            or not isinstance(phases, tuple) or not len(q_lens) == len(kv_lens) == len(phases)
            or any(type(q) is not int or type(kv) is not int or not 0 < q <= kv
                for q, kv in zip(q_lens, kv_lens))
            or any(p not in ("decode", "prefill") or (p == "decode" and q > threshold)
                or (num_speculative_tokens > 0 and p == "prefill" and q <= threshold
                    and not any(previous > threshold for previous in q_lens[:index]))
                for index, (p, q) in enumerate(zip(phases, q_lens)))
            or phases != tuple(sorted(phases))):
        raise ValueError("DSA source requires valid request lengths and decode-prefix attention order")
    if prefill_source not in (None, "force_sparse_mqa", "dense_mha_fresh"):
        raise NotImplementedError("unknown DSA prefill attention source")
    decode_lengths = tuple(q for q, phase in zip(q_lens, phases) if phase == "decode")
    if num_speculative_tokens > 0 and len(set(decode_lengths)) > 1 and not flattened_decode:
        raise NotImplementedError("DSA speculative mathematical source requires uniform decode query lengths")
    if prefill_source == "dense_mha_fresh" and ("prefill" not in phases
            or any(q != kv for q, kv, phase in zip(q_lens, kv_lens, phases, strict=True) if phase == "prefill")):
        raise NotImplementedError("dense MHA source requires fresh prefill rows and an optional decode prefix")
    if "prefill" in phases and prefill_source not in ("force_sparse_mqa", "dense_mha_fresh"):
        raise NotImplementedError("DSA all-decode source needs explicit force_sparse_mqa for prefill/mixed")
    return "mixed" if len(set(phases)) == 2 else phases[0]


def _dsa_sm90_k5_flattening(model, num_speculative_tokens):
    # Pinned vLLM0.30 deep_gemm.native_next_n_supported on SM90 is {1,2,4}.
    # next_n=6 therefore always takes IndexerMetadataBuilder._use_flattening,
    # not pack_seq padding or a rectangular native six-query invocation.
    return (model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"
        and num_speculative_tokens == 5)


def _dsa_flattened_decode_metadata(model, q_lens, kv_lens, phases, num_speculative_tokens):
    decode = tuple(q for q, phase in zip(q_lens, phases, strict=True) if phase == "decode")
    if len(set(decode)) <= 1:
        return q_lens, kv_lens, phases, {}
    if not _dsa_sm90_k5_flattening(model, num_speculative_tokens):
        raise NotImplementedError("DSA heterogeneous decode metadata requires the pinned SM90 FP8 K5 flattening source")
    # _prepare_decode_tensors repeats each real request's block-table row,
    # then expands seq_len-q + [1..q]. Q and top-k remain flat in the original
    # row order. This changes metadata, never the target/draft row extents,
    # cache arena, packing order, or output layout; no padded rows are made.
    queries, lengths, physical_phases = [], [], []
    for q, kv, phase in zip(q_lens, kv_lens, phases, strict=True):
        queries.extend((1,) * q if phase == "decode" else (q,))
        lengths.extend(range(kv - q + 1, kv + 1) if phase == "decode" else (kv,))
        physical_phases.extend((phase,) * q if phase == "decode" else (phase,))
    return tuple(queries), tuple(lengths), tuple(physical_phases), {
        "source_decode_metadata_program": "vllm_0_30_sm90_k5_flattened_decode",
        "source_logical_q_lens": q_lens, "source_logical_kv_lens": kv_lens,
        "source_logical_request_phases": phases,
        "source_metadata_scope": "prepared per-query context/block-table inputs; metadata producer internals remain outside the existing DSA stage boundary"}


def _fp8_indexer_gather_workspace(trace, workspace, name, *, max_model_len, head_dim, compress_ratio):
    """Shared sparse_attn_indexer FP8 ports; only metadata is allocated.

    Both V32 and V4 use get_max_prefill_buffer_size(max_model_len); V4
    divides that constructor capacity by its compression ratio. The UINT8
    scale port is reinterpreted as FP32, retaining its arena and offset.
    Chunk accesses, score backing and selector scratch are separate facts.
    """
    capacity = max_model_len * 40 // compress_ratio
    specs = (((capacity, head_dim), "fp8"), ((capacity, 4), "uint8"))
    values, scale_bytes = workspace.get_simultaneous(*specs)
    scale_fp32 = trace.alias(scale_bytes, trace.unique_name(name + ":gather-scale-fp32"),
        shape=(capacity,), strides_bytes=(4,), offset_bytes=scale_bytes.view.offset_bytes, dtype="fp32")
    request = {"ports": specs, "view_ids": (values.view.view_id, scale_bytes.view.view_id),
        "scale_fp32_view_id": scale_fp32.view.view_id,
        "capacity_rows": capacity, "source_max_model_len": max_model_len}
    return (values, scale_bytes), scale_fp32, request


def _glm53_indexer_decode_source_fields(model, phase, spec_decode, max_model_len):
    if model.weight_dtype != "fp8" or phase != "decode":
        return {}
    if max_model_len is None:
        raise NotImplementedError("GLM FP8 DSA decode source pricing requires explicit maximum context metadata")
    fields = {"model_execution_version": "vllm_0_30_0", "source_deepgemm_snapshot": "vllm_0_30_cmake_pin",
        "source_q_scale": "none_folded_into_fp32_weights", "source_input_layout": "contiguous_aligned_16",
        "clean_logits": False, "source_num_sms": 132, "source_context_lens_layout": "contiguous_int32_2d",
        "source_indices": "none", "source_kv_cache_layout": "paged_fp8_128_plus_fp32_scale",
        "source_block_table_inner_stride": 1, "source_schedule_shape": (133, 2),
        "source_schedule_metadata": "provided_current_context",
        "source_indexer_input_stage": "quantized_q_weights_and_cache_ready",
        "source_decode_layout": "native_contiguous", "source_candidate_blocks": "none",
        "source_parallel_mode": "pcp1_dcp1", "source_dense_mha_metadata_layer": "none",
        "source_topk_clear_state": "upstream_fused_norm_rope_ready", "source_prefill_chunks": (),
        "indexer_topk_backend": "auto", "source_block_size": 64, "source_max_model_len": max_model_len,
        "source_cost_model": "vllm_0_30_bounded_stage_roofline_v1",
        "source_tie_refinement_policy": "capacity_bound", "source_sm_budget": "h200_132_sm_declared_v1"}
    if spec_decode:
        # Verify windows keep the declared speculative depth; launch-owned
        # layout facts (native contiguous decode body, upstream clear state)
        # are unchanged by the verify window itself.
        fields["source_decode_layout"] = "native_contiguous"
    return fields


def _glm53_indexer_prefill_source_fields(model, phase, spec_decode, max_model_len,
        q_lens, kv_lens, request_phases):
    if model.weight_dtype != "fp8" or phase == "decode" or spec_decode or max_model_len is None:
        return {}
    common = _glm53_indexer_decode_source_fields(model, "decode", False, max_model_len)
    cursor = sum(q for q, item in zip(q_lens, request_phases, strict=True) if item == "decode")
    chunks = []
    for q, kv, item in zip(q_lens, kv_lens, request_phases, strict=True):
        if item != "prefill":
            continue
        chunks.append({"token_start": cursor, "token_end": cursor + q, "kv_rows": kv,
            "skip_kv_gather": False, "gather_input_state": "not_materialized"})
        cursor += q
    return {**common, "source_parallel_mode": "pcp1_dcp1", "source_dense_mha_metadata_layer": "none",
        "source_topk_clear_state": "upstream_fused_norm_rope_ready",
        "source_prefill_chunks": tuple(chunks), "source_chunk_policy": "one_request_per_chunk_analytical"}


def _glm53_sparse_source_fields(model, query):
    if model.weight_dtype != "fp8":
        return {}
    fields = {"model_execution_version": "vllm_0_30_0", "source_flashmla_snapshot": "vllm_0_30_cmake_pin",
        "source_attention_backend": "flashmla_sparse",
        "source_kv_cache_layout": "paged_bf16_rows_576", "source_index_inputs_layout": "contiguous_int32",
        "source_metadata_layout": "contiguous_int32_request_ids_and_block_table", "source_attn_sink": "none",
        "source_extra_cache": "none", "source_hisparse": False, "source_parallel_mode": "pcp1_dcp1",
        "tensor_iterator_source": "pytorch_2_13_sm90", "source_q_input_layout": "contiguous_unpadded",
        "source_q_backing_heads": 64, "source_index_remap_state": "build_current_indices",
        "source_index_remap_stream": "caller_stream", "source_block_size": 64, "source_block_stride_rows": 64,
        "source_cost_model": "vllm_0_30_bounded_stage_roofline_v1",
        "source_tie_refinement_policy": "capacity_bound"}
    # Read the actual caller view: ordinary q1/prefill may use the same
    # padded q_concat_buffer as a verify window, especially at TP8.
    _, heads, dim = query.shape
    padded = -(-heads // 64) * 64
    strides = query.view.strides_bytes
    if strides == (heads * dim * 2, dim * 2, 2):
        fields["source_q_backing_heads"] = heads
    elif strides == (padded * dim * 2, dim * 2, 2):
        fields["source_q_input_layout"] = "caller_padded_head_backing"
        fields["source_q_backing_heads"] = padded
    else:
        raise ValueError("DSA sparse source query has unsupported physical view strides")
    return fields


class SymbolicDSAIndexer(SymbolicOperator):
    """Unpadded CP1 useful indexer, with clear/cache insertion owned upstream."""

    def __init__(self, trace, name, pool_name, model, *, q_lens, kv_lens, request_phases,
                 cache, topk_buffer, selection_source, prefill_attention_source=None,
                 workspace=None, source_max_model_len=None, num_speculative_tokens=0):
        super().__init__(trace, name, pool_name)
        self.phase = _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens,
            flattened_decode=_dsa_sm90_k5_flattening(model, num_speculative_tokens))
        self.num_speculative_tokens = num_speculative_tokens
        self.spec_decode = any(q > 1 for q, phase in zip(q_lens, request_phases) if phase == "decode")
        if selection_source != "declared_causal_topk":
            raise NotImplementedError("DSA symbolic indexer needs an explicit causal top-k cardinality declaration")
        if (model.model_type != "glm_moe_dsa" or not isinstance(kv_lens, tuple) or not kv_lens
                or any(type(length) is not int or length <= 0 for length in kv_lens)):
            raise ValueError("DSA decode indexer needs GLM and one positive context length per row")
        trace._check_tensor(cache)
        trace._check_tensor(topk_buffer)
        if (cache.dtype != "uint8" or len(cache.shape) != 2 or cache.shape[-1] != 132
                or topk_buffer.dtype != "int32" or len(topk_buffer.shape) != 2
                or topk_buffer.shape[0] < sum(q_lens) or topk_buffer.shape[-1] != model.index_topk):
            raise ValueError("DSA indexer requires its typed FP8 cache and shared INT32 top-k buffer")
        self.model, self.kv_lens, self.cache, self.topk_buffer = model, kv_lens, cache, topk_buffer
        self.q_lens, self.phases = q_lens, request_phases
        self.prefill_attention_source = prefill_attention_source
        if (workspace is None) != (source_max_model_len is None):
            raise ValueError("DSA gather workspace and source maximum length must be supplied together")
        if workspace is not None:
            if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
                raise ValueError("DSA indexer workspace must belong to this physical-rank source trace")
            if (type(source_max_model_len) is not int
                    or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings):
                raise ValueError("DSA indexer workspace needs an explicit maximum length covering the contexts")
        self.workspace, self.max_model_len = workspace, source_max_model_len

    def __call__(self, query, weights):
        model, rows = self.model, sum(self.q_lens)
        self.trace._check_tensor(query)
        self.trace._check_tensor(weights)
        if (query.shape != (rows, model.index_n_heads, model.index_head_dim) or query.dtype != "fp8"
                or weights.shape != (rows, model.index_n_heads) or weights.dtype != "fp32"):
            raise ValueError("DSA indexer consumes FP8 Q and folded FP32 head weights")
        gather_views, scale_fp32, gather_request = (), None, None
        if self.phase != "decode" and self.workspace is not None:
            gather_views, scale_fp32, gather_request = _fp8_indexer_gather_workspace(
                self.trace, self.workspace, self.name, max_model_len=self.max_model_len,
                head_dim=model.index_head_dim, compress_ratio=1)
        q_lens, kv_lens, phases, flattening = _dsa_flattened_decode_metadata(
            model, self.q_lens, self.kv_lens, self.phases, self.num_speculative_tokens)
        spec_decode = self.spec_decode and not flattening
        node = self.node(WorkloadKind.EAGER_ATTN, {"batch": len(q_lens), "q_len": max(q_lens),
            "kv_len": max(kv_lens), "num_query_heads": model.index_n_heads,
            "head_dim": model.index_head_dim, "topk": min(model.index_topk, max(kv_lens))},
            "int32", "dsa_indexer_" + self.phase, {"source_component": "vllm_0_30_symbolic_dsa_indexer",
                "causal": True, "score_accumulator_dtype": "fp32", "score_function": "weighted_dot_product_topk",
                "cache_layout": "paged_mla_indexer", "cache_quantization": "fp8_e4m3_with_ue8m0_scale",
                "q_lens": q_lens, "kv_lens": kv_lens, "request_phases": phases,
                "topk_capacity": model.index_topk, "topk_index_dtype": "int32", "topk_sentinel": -1,
                "mathematical_program": ("dsa_weighted_relu_specdecode_v1" if spec_decode else
                    "dsa_weighted_relu_topk_v1" if self.phase == "decode" else
                    "dsa_weighted_relu_dense_mha_v1" if self.prefill_attention_source == "dense_mha_fresh" else
                    "dsa_weighted_relu_force_mqa_v1"),
                "prefill_attention_source": self.prefill_attention_source, "context_parallel_size": 1,
                **({"num_speculative_tokens": self.num_speculative_tokens,
                    "decode_threshold": self.num_speculative_tokens + 1} if spec_decode else {}),
                "skip_topk_buffer_clear": True, "selection_cardinality_source": "declared_causal_topk",
                "symbolic_prefill_gather_workspace": gather_request,
                **_glm53_indexer_decode_source_fields(model, self.phase, spec_decode, self.max_model_len),
                **_glm53_indexer_prefill_source_fields(model, self.phase, spec_decode, self.max_model_len,
                    q_lens, kv_lens, phases), **flattening,
                "symbolic_workspace_complete": False,
                "score_backing_bound": model.weight_dtype == "fp8" and not spec_decode})
        active = self.topk_buffer if self.topk_buffer.shape[0] == rows else self.topk_buffer.narrow(0, 0, rows)
        return self.trace.call(node, (query, weights, self.cache), shape=active.shape, dtype="int32",
            out=active, writes=gather_views)


class SymbolicAttentionStateMerge(SymbolicOperator):
    """Attention states with nonempty suffix; real-arithmetic projection.

    The source merge kernel's rounded branches, instruction program and
    launch are not asserted by this useful mathematical program.
    Empty prefix rows explicitly declare zero O and negative-infinity LSE.
    """

    def __call__(self, prefix, prefix_lse, suffix, suffix_lse, *, out, out_lse=None, inplace=False, empty_prefix_rows=0):
        for value in (prefix, prefix_lse, suffix, suffix_lse, out):
            self.trace._check_tensor(value)
        if out_lse is not None:
            self.trace._check_tensor(out_lse)
        if (len(prefix.shape) != 3 or prefix.dtype != "bf16" or suffix.shape != prefix.shape
                or suffix.dtype != "bf16" or out.shape != prefix.shape or out.dtype != "bf16"):
            raise ValueError("attention state merge requires matching BF16 row/head/value ports")
        rows, heads, width = prefix.shape
        if type(empty_prefix_rows) is not int or not 0 <= empty_prefix_rows <= rows:
            raise ValueError("attention merge empty-prefix rows must be within the output rows")
        if (prefix_lse.shape != (heads, rows) or suffix_lse.shape != (heads, rows)
                or prefix_lse.dtype != "fp32" or suffix_lse.dtype != "fp32"):
            raise ValueError("attention state merge requires FP32 head/row LSE ports")
        if (out_lse is not None and (out_lse.shape != prefix_lse.shape or out_lse.dtype != "fp32")):
            raise ValueError("attention state merge LSE output must match its FP32 head/row inputs")
        if type(inplace) is not bool:
            raise TypeError("attention state merge in-place contract must be boolean")
        if inplace:
            if (out_lse is None or out.view != prefix.view or out_lse.view != prefix_lse.view
                    or out.view.allocation_id == suffix.view.allocation_id
                    or out_lse.view.allocation_id == suffix_lse.view.allocation_id):
                raise ValueError("context accumulation requires exact prefix O/LSE aliases and independent suffix states")
        elif (out.view.allocation_id in (prefix.view.allocation_id, suffix.view.allocation_id)
                or (out_lse is not None and out_lse.view.allocation_id in
                    (prefix_lse.view.allocation_id, suffix_lse.view.allocation_id))):
            raise NotImplementedError("ordinary MLA final merge requires independent output storage")
        node = self.node(WorkloadKind.VECTOR, {"element_count": rows * heads * width,
            "rows": rows, "heads": heads, "head_dim": width, "dtype_bytes": 2}, "bf16",
            "vllm_attention_state_merge", {"source_component": "symbolic_mla_context_merge",
                "mathematical_program": ("sglang_0_5_19_empty_prefix_lse_merge_flops_bytes_v1" if empty_prefix_rows
                    else "vllm_0_30_nonempty_lse_merge_flops_bytes_v1"),
                **({"empty_prefix_rows": empty_prefix_rows} if empty_prefix_rows else {}),
                "output_lse": out_lse is not None, "symbolic_workspace_complete": False})
        inputs = (prefix, prefix_lse, suffix, suffix_lse)
        return self.trace.call(node, inputs, shape=out.shape, dtype="bf16", out=out,
            writes=() if out_lse is None else (out_lse,))


class SymbolicMLACacheUpdate(SymbolicOperator):
    """BF16 latent/RoPE append into a declared logical MLA cache pool."""

    def __init__(self, trace, name, pool, cache, slot_mapping, *, latent_dim, rope_dim, slots, sglang_dummy_rows=0):
        super().__init__(trace, name, pool)
        trace._check_tensor(cache)
        trace._check_tensor(slot_mapping)
        # Native tuples with native integer elements cannot change after the
        # first successful constructor. Identity avoids bool/float equality
        # collisions; capacity and dummy-tail policy remain part of the proof.
        cacheable = (type(trace) is ForwardTrace and type(slots) is tuple and len(slots) > 1
            and type(sglang_dummy_rows) is int and len(cache.shape) == 2
            and type(cache.shape[0]) is int)
        proved = (cacheable and trace._has_mla_slot_proof(
            slots, cache.shape[0], sglang_dummy_rows))
        if (any(type(width) is not int or width <= 0 for width in (latent_dim, rope_dim))
                or cache.dtype != "bf16" or len(cache.shape) != 2 or cache.shape[-1] != latent_dim + rope_dim
                or not isinstance(slots, tuple) or not slots
                or (not proved and any(type(slot) is not int or not 0 <= slot < cache.shape[0] for slot in slots))
                or slot_mapping.shape != (len(slots),) or slot_mapping.dtype != "int64"):
            raise ValueError("MLA cache update requires BF16 latent/RoPE storage and distinct declared INT64 slots")
        if type(sglang_dummy_rows) is not int or not 0 <= sglang_dummy_rows < len(slots):
            raise ValueError("SGLang graph cache update requires a bounded explicit dummy-row count")
        if not proved:
            active = slots[:-sglang_dummy_rows] if sglang_dummy_rows else slots
            if (len(set(active)) != len(active) or (sglang_dummy_rows and (
                    slots[-sglang_dummy_rows:] != (0,) * sglang_dummy_rows or 0 in active))):
                raise ValueError("MLA cache slots must be distinct except declared SGLang dummy tail writes to reserved0")
            if cacheable:
                trace._remember_mla_slot_proof(slots, cache.shape[0], sglang_dummy_rows)
        self.cache, self.slot_mapping, self.slots = cache, slot_mapping, slots
        self.sglang_dummy_rows = sglang_dummy_rows
        self.latent_dim, self.rope_dim = latent_dim, rope_dim

    def __call__(self, latent, rope):
        rows, width = len(self.slots), self.latent_dim + self.rope_dim
        for value in (latent, rope):
            self.trace._check_tensor(value)
        if (latent.dtype != "bf16" or latent.shape != (rows, self.latent_dim)
                or rope.dtype != "bf16" or rope.shape != (rows, 1, self.rope_dim)):
            raise ValueError("MLA cache append requires matching BF16 normalized latent and rotated key")
        node = self.node(WorkloadKind.KV_UPDATE, {"elements": rows * width, "seq_len": rows,
            "compressed_dim": width}, "bf16", "mla_kv_cache", {
                "source_component": "symbolic_mla_cache_update", "cache_layout": "paged_mla",
                "cache_write_semantics": "append", "symbolic_slot_mapping": self.slots,
                "source_dummy_dependency_payload_bytes": 0,
                **({"sglang_graph_dummy_rows": self.sglang_dummy_rows,
                    "dummy_cache_slot": 0, "dummy_cache_disjointness": "caller_declared_not_observed"}
                    if self.sglang_dummy_rows else {}),
                "slot_mapping_scope": "declared_logical_pool_not_native_pages"})
        self.trace.call(node, (latent, rope, self.slot_mapping), shape=self.cache.shape, dtype="bf16", out=self.cache)


class SymbolicDSAIndexCacheUpdate(SymbolicOperator):
    """PCP's separate BF16 index K quantize/cache insert, with real slots."""

    def __init__(self, trace, name, pool, cache, slot_mapping, *, slots, key_dim=128):
        super().__init__(trace, name, pool)
        for tensor in (cache, slot_mapping):
            trace._check_tensor(tensor)
        if (key_dim != 128 or cache.dtype != "uint8" or len(cache.shape) != 2 or cache.shape[1] != 132
                or not isinstance(slots, tuple) or not slots or len(set(slots)) != len(slots)
                or any(type(slot) is not int or not 0 <= slot < cache.shape[0] for slot in slots)
                or slot_mapping.dtype != "int64" or slot_mapping.shape != (len(slots),)):
            raise ValueError("PCP index-cache insert requires actual BF16-128 keys and distinct valid slots")
        self.cache, self.slot_mapping, self.slots, self.key_dim = cache, slot_mapping, slots, key_dim

    def __call__(self, key):
        rows = len(self.slots)
        self.trace._check_tensor(key)
        if key.shape != (rows, self.key_dim) or key.dtype != "bf16":
            raise ValueError("PCP index insert requires matching normalized BF16 rows")
        node = self.node(WorkloadKind.KV_UPDATE,
            {"elements": rows * 132, "seq_len": rows, "key_dim": self.key_dim, "scale_bytes_per_token": 4},
            "uint8", "dsa_fp8_index_cache", {"source_component": "symbolic_dsa_index_cache_update",
                "cache_layout": "paged_mla_indexer", "cache_write_semantics": "append",
                "quantization": "fp8_e4m3_with_ue8m0_scale", "symbolic_slot_mapping": self.slots,
                "slot_mapping_scope": "declared_logical_pool_not_native_pages"})
        self.trace.call(node, (key, self.slot_mapping), shape=self.cache.shape, dtype="uint8", out=self.cache)


class SymbolicMLALatentDecode(SymbolicOperator):
    """Declared BF16 FlashMLA public O/LSE shapes and useful dense math."""

    def __init__(self, trace, name, pool, *, heads, kv_lens, cache, q_lens=None,
                 source_profile=None, source_batch_invariant=None, sglang_dummy_rows=0):
        super().__init__(trace, name, pool)
        trace._check_tensor(cache)
        if (type(heads) is not int or heads <= 0 or not isinstance(kv_lens, tuple) or not kv_lens
                or any(type(length) is not int or length <= 0 for length in kv_lens)
                or cache.dtype != "bf16" or len(cache.shape) != 2 or cache.shape[-1] != 576):
            raise ValueError("MLA decode math requires BF16576 latent cache and positive contexts")
        if type(sglang_dummy_rows) is not int or not 0 <= sglang_dummy_rows < len(kv_lens):
            raise ValueError("SGLang graph attention requires a bounded explicit dummy-row count")
        queries = (1,) * len(kv_lens) if q_lens is None else q_lens
        if (not isinstance(queries, tuple) or len(queries) != len(kv_lens)
                or any(type(q) is not int or not 1 <= q <= 128 for q in queries)
                or len(set(queries)) != 1):
            raise ValueError("MLA decode requires uniform positive queries<=128 within every context")
        if source_profile not in (None, "vllm_0_27_1_causal_flashmla_bf16_source", "sglang_0_5_19_causal_flashmla_bf16_source",
                "vllm_0_30_causal_flashmla_bf16_source"):
            raise NotImplementedError("MLA multiquery requires its explicit FlashMLA source")
        if ((queries[0] > 1 and source_profile is None)
                or (source_profile not in (None, "vllm_0_30_causal_flashmla_bf16_source") and source_batch_invariant is not False)
                or (source_profile == "vllm_0_30_causal_flashmla_bf16_source" and source_batch_invariant is not None)
                or (source_profile is None and source_batch_invariant is not None)):
            raise NotImplementedError("FlashMLA uniform queries require explicit source and BATCH_INVARIANT=False")
        active_count = len(kv_lens) - sglang_dummy_rows
        if any(q > k for q, k in zip(queries[:active_count], kv_lens[:active_count])):
            raise ValueError("MLA active queries must fit within every context")
        if sglang_dummy_rows:
            dummy_lengths = kv_lens[-sglang_dummy_rows:]
            valid_windows = (((1,) * sglang_dummy_rows, (4,) * sglang_dummy_rows)
                if queries[0] == 3 else (((1,) * sglang_dummy_rows,) if queries[0] == 1 else ()))
            if (source_profile != "sglang_0_5_19_causal_flashmla_bf16_source"
                    or dummy_lengths not in valid_windows):
                raise ValueError("SGLang graph dummy attention requires its explicit decode/verify/extend window")
            # Anonymous rows share one reserved slot0, not independent histories.
            # Its backing must still be disjoint from every active cache row.
            required_cache_rows = sum(kv_lens[:-sglang_dummy_rows]) + 64
        else:
            required_cache_rows = sum(kv_lens)
        if cache.shape[0] < required_cache_rows:
            raise ValueError("MLA cache backing must cover active contexts and any reserved dummy slot0")
        self.heads, self.kv_lens, self.cache, self.q_lens = heads, kv_lens, cache, queries
        self.sglang_dummy_rows = sglang_dummy_rows
        self.launch_source_identity = {
            "vllm_0_27_1_causal_flashmla_bf16_source": ("vllm_0_27_1", "vllm_0_27_1_cmake_pin"),
            "vllm_0_30_causal_flashmla_bf16_source": ("vllm_0_30_0", "vllm_0_30_cmake_pin"),
        }[source_profile] if source_profile in (
            "vllm_0_27_1_causal_flashmla_bf16_source", "vllm_0_30_causal_flashmla_bf16_source") else None

    def __call__(self, query, *, public_output_rows=None):
        self.trace._check_tensor(query)
        rows, batch, query_length = sum(self.q_lens), len(self.kv_lens), self.q_lens[0]
        physical_rows = rows if public_output_rows is None else public_output_rows
        if (type(physical_rows) is not int or physical_rows < rows
                or (public_output_rows is not None and (query_length != 1 or self.sglang_dummy_rows))
                or query.dtype != "bf16" or query.shape != (physical_rows, self.heads, 576)):
            raise ValueError("MLA decode consumes all declared BF16 absorbed query rows")
        node = self.node(WorkloadKind.FLASH_ATTN, {"batch": batch, "q_len": query_length,
            "kv_len": max(self.kv_lens), "num_query_heads": self.heads,
            "qk_head_dim": 576, "v_head_dim": 512, "kv_lora_rank": 512}, "bf16", "mla_decode", {
                "source_component": "symbolic_mla_decode_mathematical", "causal": True,
                "source_query_dtype": query.dtype, "source_cache_dtype": self.cache.dtype,
                "source_query_shape": query.shape, "source_cache_shape": self.cache.shape,
                **({"source_flashmla_launch_identity": self.launch_source_identity}
                    if self.launch_source_identity is not None else {}),
                "kv_layout": "mla", "kv_dist": "uniform", "q_lens": self.q_lens, "kv_lens": self.kv_lens,
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
                **({"source_graph_padding_program": "vllm_0_30_full_mla",
                    "public_query_rows": physical_rows, "public_output_rows": physical_rows,
                    "padding_seq_lens": (0,) * (physical_rows - rows),
                    "graph_padding_output_values_known": False,
                    "padding_native_service_bound": False} if public_output_rows is not None else {}),
                **({"sglang_graph_dummy_rows": sum(self.q_lens[-self.sglang_dummy_rows:]),
                    "sglang_graph_dummy_requests": self.sglang_dummy_rows, "dummy_cache_slot": 0,
                    "dummy_cache_disjointness": "caller_declared_not_observed"} if self.sglang_dummy_rows else {}),
                "mathematical_program": ("bf16_mla_causal_empty_rows_softmax_v1"
                    if any(q > k for q, k in zip(self.q_lens, self.kv_lens)) else "bf16_mla_decode_softmax_v1"),
                "symbolic_workspace_complete": False})
        if query_length > 1:
            query_view = query.view_as_shape(batch, query_length, self.heads, 576)
            output, lse = self.trace.call_outputs(node, (query_view, self.cache), outputs=(
                ("latent", (batch, query_length, self.heads, 512), "bf16"),
                ("lse", (batch, self.heads, query_length), "fp32")))
            return output.view_as_shape(rows, self.heads, 512), lse
        return self.trace.call_outputs(node, (query, self.cache), outputs=(
            ("latent", (physical_rows, self.heads, 512), "bf16"), ("lse", (physical_rows, self.heads, 1), "fp32")))


class SymbolicDSASparseAttention(SymbolicOperator):
    """Useful latent-attention math with source public output/LSE backing.

    Logical-to-physical index conversion, ignored backend return, score and
    library scratch remain unbound. This partial storage projection cannot
    stand in for complete backend/workspace or normal-capacity acceptance.
    """

    def __init__(self, trace, name, pool_name, model, *, tp_size, q_lens, kv_lens, request_phases,
                 cache, topk_buffer, indexer_type, selection_source, prefill_attention_source=None,
                 num_speculative_tokens=0, reused_selection_kv_lens=None, pcp_source=False):
        super().__init__(trace, name, pool_name)
        self.phase = _dsa_source_phase(q_lens, kv_lens, request_phases, prefill_attention_source, num_speculative_tokens,
            flattened_decode=_dsa_sm90_k5_flattening(model, num_speculative_tokens))
        self.num_speculative_tokens = num_speculative_tokens
        self.phases = request_phases
        self.spec_decode = any(q > 1 for q, phase in zip(q_lens, request_phases) if phase == "decode")
        if reused_selection_kv_lens is not None and (indexer_type != "shared" or self.phase != "decode"
                or any(q != 1 for q in q_lens) or not isinstance(reused_selection_kv_lens, tuple)
                or len(reused_selection_kv_lens) != len(kv_lens)
                or any(type(old) is not int or not 0 < old <= current
                    for old, current in zip(reused_selection_kv_lens, kv_lens))):
            raise ValueError("DSA reused selection requires shared q1 and explicit prior causal KV lengths")
        self.reused_selection_kv_lens = reused_selection_kv_lens
        if type(pcp_source) is not bool:
            raise TypeError("PCP sparse source flag must be boolean")
        if (model.model_type != "glm_moe_dsa" or type(tp_size) is not int or tp_size < 1
                or model.num_attention_heads % tp_size
                or (tp_size not in (1, 8) and not pcp_source and model.weight_dtype != "bf16"
                    and not is_glm_fp8_tensor_model_scope(model, tp_size))
                or indexer_type not in ("full", "shared")
                or selection_source != "declared_causal_topk"):
            raise NotImplementedError("DSA sparse source requires GLM TP1/TP8 and explicit selected-cardinality applicability")
        if not isinstance(kv_lens, tuple) or not kv_lens or any(type(length) is not int or length <= 0 for length in kv_lens):
            raise ValueError("DSA sparse decode requires one positive context length per row")
        trace._check_tensor(cache)
        trace._check_tensor(topk_buffer)
        if (cache.dtype != "bf16" or len(cache.shape) != 2
                or cache.shape[-1] != model.kv_lora_rank + model.qk_rope_head_dim
                or topk_buffer.dtype != "int32" or len(topk_buffer.shape) != 2
                or topk_buffer.shape[-1] != model.index_topk or topk_buffer.shape[0] < sum(q_lens)):
            raise ValueError("DSA sparse decode requires BF16 latent cache and a shared INT32 top-k buffer")
        self.model, self.heads, self.kv_lens = model, model.num_attention_heads // tp_size, kv_lens
        self.q_lens = q_lens
        self.prefill_attention_source = prefill_attention_source
        self.cache, self.topk_buffer, self.indexer_type = cache, topk_buffer, indexer_type

    def __call__(self, query):
        model, rows = self.model, sum(self.q_lens)
        width = model.kv_lora_rank + model.qk_rope_head_dim
        if query.shape != (rows, self.heads, width) or query.dtype != "bf16":
            raise ValueError("DSA sparse decode consumes the concatenated BF16 latent/RoPE query")
        if self.spec_decode:
            output = self.trace.tensor(self.trace.unique_name(self.name + ":latent"), (rows, 64, model.kv_lora_rank), "bf16")
            lse = self.trace.tensor(self.trace.unique_name(self.name + ":lse"), (rows, 64), "fp32")
            offset = 0
            for phase in ("decode", "prefill"):
                indexes = tuple(i for i, item in enumerate(self.phases) if item == phase)
                if not indexes:
                    continue
                queries = tuple(self.q_lens[i] for i in indexes)
                lengths = tuple(self.kv_lens[i] for i in indexes)
                tokens = sum(queries)
                self._group(query.narrow(0, offset, tokens), queries, lengths, phase,
                    output.narrow(0, offset, tokens), lse.narrow(0, offset, tokens), offset=offset)
                offset += tokens
            return output.narrow(1, 0, self.heads)
        return self._group(query, self.q_lens, self.kv_lens, self.phase)

    def _group(self, query, q_lens, kv_lens, phase, output=None, lse=None, *, offset=0):
        model, rows = self.model, sum(q_lens)
        q_lens, kv_lens, _, flattening = _dsa_flattened_decode_metadata(
            model, q_lens, kv_lens, (phase,) * len(q_lens), self.num_speculative_tokens)
        width = model.kv_lora_rank + model.qk_rope_head_dim
        spec_decode = phase == "decode" and any(q > 1 for q in q_lens)
        selected = tuple(min(model.index_topk, length) for length in (
            kv_lens if self.reused_selection_kv_lens is None else self.reused_selection_kv_lens))
        node = self.node(WorkloadKind.EAGER_ATTN, {"batch": len(q_lens), "q_len": max(q_lens), "kv_len": max(kv_lens),
            "num_query_heads": self.heads, "qk_head_dim": width, "v_head_dim": model.kv_lora_rank,
            "dtype_bytes": 2, "topk": max(selected)}, "bf16", "dsa_sparse_" + phase, {
                "source_component": "vllm_0_30_symbolic_dsa_sparse", "causal": True,
                "source_query_strides_bytes": query.view.strides_bytes,
                "source_query_offset_bytes": query.view.offset_bytes,
                "indexer_type": self.indexer_type, "kv_layout": "mla", "kv_dist": "uniform",
                "q_lens": q_lens, "kv_lens": kv_lens,
                "selected_kv_lens": selected,
                "topk_capacity": model.index_topk, "topk_index_dtype": "int32",
                "selection_cardinality_source": "declared_causal_topk", "symbolic_workspace_complete": False,
                **({"mathematical_program": "bf16_latent_specdecode_v1", "num_speculative_tokens": self.num_speculative_tokens,
                    "decode_threshold": self.num_speculative_tokens + 1}
                    if spec_decode else
                    {"mathematical_program": "bf16_latent_reused_topk_v1", "reused_selection_kv_lens": self.reused_selection_kv_lens}
                    if self.reused_selection_kv_lens is not None else
                    {"mathematical_program": "bf16_latent_force_mqa_v1", "prefill_attention_source": self.prefill_attention_source}
                    if phase != "decode" else {}),
                **_glm53_sparse_source_fields(model, query), **flattening,
                "backend_index_conversion_bound": model.weight_dtype == "fp8" and not spec_decode,
                "backend_ignored_return_bound": model.weight_dtype == "fp8" and not spec_decode})
        topk = self.topk_buffer if self.topk_buffer.shape[0] == rows and offset == 0 else self.topk_buffer.narrow(0, offset, rows)
        node = node.with_attributes({**node.attributes,
            "source_topk_allocation_id": topk.view.allocation_id,
            "source_topk_input_writer": self.trace._writers[topk.view.allocation_id]})
        if output is not None:
            self.trace.call(node, (query, self.cache, topk), shape=output.shape, dtype=output.dtype,
                out=output, writes=(lse,))
            return output
        # flashmla_sparse's Hopper wrapper returns a 64-head backing and
        # narrows it to the actual heads; the view keeps the padded storage.
        output, lse = self.trace.call_outputs(node, (query, self.cache, topk), outputs=(
            ("latent", (rows, 64, model.kv_lora_rank), "bf16"), ("lse", (rows, 64), "fp32")))
        return output.narrow(1, 0, self.heads)


class SymbolicV4QKVNorm(SymbolicOperator):
    """Fused two-output RMSNorm over split q-lora/KV views, no input copy."""

    def __init__(self, trace, name, pool_name, model):
        super().__init__(trace, name, pool_name)
        if model.model_type != "deepseek_v4":
            raise NotImplementedError("fused Q/KV symbolic norm requires the V4 source widths")
        self.model = model

    def __call__(self, projection):
        model, rows = self.model, projection.shape[0]
        if projection.shape != (rows, model.q_lora_rank + model.head_dim) or projection.dtype != "bf16":
            raise ValueError("V4 Q/KV norm input must be the fused BF16 projection")
        qr, kv = projection.split((model.q_lora_rank, model.head_dim), dim=-1)
        node = self.node(WorkloadKind.NORM, {"rows": rows, "q_hidden": model.q_lora_rank,
            "kv_hidden": model.head_dim}, "bf16", "dsv4_q_kv_rmsnorm", {
                "source_component": "vllm_0_30_symbolic_v4_qkv_norm",
                "epsilon": model.target_semantics.rms_norm_eps})
        qr, kv = self.trace.call_outputs(node, (qr, kv), outputs=(
            ("qr", (rows, model.q_lora_rank), "bf16"), ("kv", (rows, model.head_dim), "bf16")))
        return qr, None, kv


class SymbolicV4InverseRope(SymbolicOperator):
    """SM90 inverse-RoPE/FP8 quant outputs with source transpose/scale views."""

    def __init__(self, trace, name, pool_name, model, *, tp_size):
        super().__init__(trace, name, pool_name)
        if not _v4_source_block_geometry(model, tp_size, expert_parallel=True):
            raise NotImplementedError("V4 inverse-RoPE source requires whole SM90 output groups")
        self.model, self.tp_size = model, tp_size

    def __call__(self, o, positions, rotary_cache):
        model, rows = self.model, o.shape[0]
        heads = model.num_attention_heads // self.tp_size
        groups = model.o_groups // self.tp_size
        width = model.num_attention_heads // model.o_groups * model.head_dim
        if (o.shape != (rows, heads, model.head_dim) or o.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"
                or len(rotary_cache.shape) != 2 or rotary_cache.shape[-1] != model.qk_rope_head_dim
                or rotary_cache.dtype != "fp32"):
            raise ValueError("V4 inverse-RoPE requires BF16 heads, INT64 positions and FP32 source RoPE cache")
        aligned, blocks = -(-rows // 4) * 4, width // 128
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * heads * model.head_dim,
            "seq_len": rows, "num_query_heads": heads, "head_dim": model.head_dim,
            "rope_dim": model.qk_rope_head_dim}, "fp8", "dsv4_inv_rope_fp8_quant", {
                "source_component": "vllm_0_30_symbolic_v4_inverse_rope", "quant_group_size": 128})
        values, scales = self.trace.call_outputs(node, (o, positions, rotary_cache), outputs=(
            ("values", (groups, rows, width), "fp8"), ("scales", (groups, blocks, aligned), "fp32")))
        # Source buffers are group-major, returned as token-major transpose
        # views. Padding rows retain their storage and receive explicit zeros.
        values = self.trace.alias(values, self.trace.unique_name(self.name + ":value-view"),
            shape=(rows, groups, width), strides_bytes=(width, rows * width, 1))
        scales = self.trace.alias(scales, self.trace.unique_name(self.name + ":scale-view"),
            shape=(rows, groups, blocks), strides_bytes=(4, blocks * aligned * 4, aligned * 4))
        return values, scales


class SymbolicV4GroupedOutput(SymbolicOperator):
    """fp8_einsum('bhr,hdr->bhd') with fresh interleaved BF16 output."""

    def __init__(self, trace, name, pool_name, model, *, tp_size):
        super().__init__(trace, name, pool_name)
        if not _v4_source_block_geometry(model, tp_size, expert_parallel=True):
            raise NotImplementedError("V4 FP8 grouped output source requires whole SM90 output groups")
        self.model, self.tp_size = model, tp_size

    def __call__(self, values, scales):
        model, rows = self.model, values.shape[0]
        groups = model.o_groups // self.tp_size
        width = model.num_attention_heads // model.o_groups * model.head_dim
        if (values.shape != (rows, groups, width) or values.dtype != "fp8"
                or scales.shape != (rows, groups, width // 128) or scales.dtype != "fp32"):
            raise ValueError("V4 grouped output requires its FP8 values and FP32 scale views")
        node = self.node(WorkloadKind.LINEAR, {"b": groups, "m": rows,
            "n": model.o_lora_rank, "k": width}, "bf16", "batched_gemm", {
                "source_component": "vllm_0_30_symbolic_v4_grouped_output",
                "batch_layout": "einsum_bhr_hdr_bhd", "a_batch_stride_elements": rows * width,
                "b_batch_stride_elements": width * model.o_lora_rank,
                "c_batch_stride_elements": model.o_lora_rank,
                "has_bias": False, "weight_dtype": "fp8", "weight_group_size": 128})
        return self.trace.call(node, (values, scales), shape=(rows, groups, model.o_lora_rank), dtype="bf16")


class SymbolicV4RoutedExperts(SymbolicOperator):
    """BF16/MXFP4 modular expert call with explicit routing and ownership.

    The existing owner prices its logical Marlin program. An explicit workspace
    binds the BF16 arena; a declared SM count also describes alignment, locks
    and the serial FP32 scratch requests. Allocation extents are not traffic
    or proof of a loaded provider or complete worker capacity.
    """

    def __init__(self, trace, name, pool_name, model, *, tp_size, resolution, invocation_id, workspace=None,
                 source_multiprocessors=None, expert_parallel_size=1, expert_rank=0, expert_tp_size=None, valid_tokens=None):
        from .traced_forward import ForwardWorkspace
        from .runtime_inputs import ExpertRoutingResolution
        super().__init__(trace, name, pool_name)
        if (not _v4_source_block_geometry(model, tp_size, expert_parallel=expert_parallel_size > 1)
                or not isinstance(resolution, ExpertRoutingResolution)):
            raise NotImplementedError("symbolic V4 routed experts require whole-block TP/EP and explicit normalized routing")
        expert_tp = (1 if expert_parallel_size > 1 else tp_size) if expert_tp_size is None else expert_tp_size
        if (type(expert_parallel_size) is not int or expert_parallel_size < 1
                or type(expert_rank) is not int or not 0 <= expert_rank < expert_parallel_size
                or model.num_experts % expert_parallel_size or type(expert_tp) is not int or expert_tp < 1
                or expert_parallel_size > 1 and expert_tp != 1
                or model.moe_intermediate_size % expert_tp or model.moe_intermediate_size // expert_tp % 32):
            raise ValueError("V4 expert source requires whole intermediate shards and uniform linear EP ownership")
        self.model, self.tp_size, self.resolution, self.invocation_id = model, tp_size, resolution, invocation_id
        self.expert_parallel_size, self.expert_rank, self.expert_tp_size = expert_parallel_size, expert_rank, expert_tp
        self.logical_intermediate = model.moe_intermediate_size // expert_tp
        # Mxfp4MoEMethod.maybe_roundup_sizes delegates to the Marlin oracle:
        # local intermediate rounds to128; checkpoint scale groups stay32.
        self.intermediate = -(-self.logical_intermediate // 128) * 128
        if valid_tokens is not None and (type(valid_tokens) is not int or valid_tokens < 0):
            raise ValueError("V4 expert valid tokens require an explicit nonnegative count")
        self.valid_tokens = valid_tokens
        if workspace is not None and (not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace):
            raise ValueError("V4 expert workspace must belong to this physical-rank source trace")
        self.workspace = workspace
        if source_multiprocessors is not None and (type(source_multiprocessors) is not int or source_multiprocessors <= 0):
            raise ValueError("Marlin source temporaries need an explicit positive integer SM specification")
        self.source_multiprocessors = source_multiprocessors

    def _alignment_storage(self, node, routes, experts, block):
        capacity = routes + experts * (block - 1)
        if routes < experts:
            capacity = min(capacity, routes * block)
        sorted_ids = self.trace.tensor(node.node_id + ":sorted-token-ids", (capacity,), "int32")
        expert_ids = self.trace.tensor(node.node_id + ":expert-ids", (-(-capacity // block),), "int32")
        padded = self.trace.tensor(node.node_id + ":num-tokens-post-padded", (1,), "int32")
        if not (routes < 1024 and experts <= 64):
            # moe_align_block_size owns cumsum only through its two CUDA calls.
            cumsum = self.trace.tensor(node.node_id + ":alignment-cumsum", (experts + 1,), "int32")
        self.trace.checkpoint(node.node_id + ":alignment-complete")
        return (sorted_ids, expert_ids, padded), capacity

    def _matrix_scratch_storage(self, node, stage, columns, sorted_capacity, block):
        # Host new_empty request from moe_wna16_marlin_gemm: max_thread_n=256,
        # four CTAs per SM, and the M8 path doubles the request after min().
        elements = min(columns * sorted_capacity, self.source_multiprocessors * 4 * block * 256)
        if block == 8:
            elements *= 2
        partial = self.trace.tensor(node.node_id + ":" + stage + ":fp32-reduce", (elements,), "fp32")
        self.trace.checkpoint(node.node_id + ":" + stage + ":consume")
        # Ordinary helper scope releases partial before the next matrix call.

    def _matrix_program_storage(self, node, scratch, rows, hidden, width, capacity, block):
        # Locks belong to _fused_marlin_moe, which returns before outer moe_sum.
        locks = self.trace.tensor(node.node_id + ":locks", (self.source_multiprocessors * 4,), "int32")
        self.trace.checkpoint(node.node_id + ":locks-zeroed")
        cache_views = ()
        if scratch:
            common_view, gate_view = scratch
            cache_views = tuple(self.trace.alias(base, node.node_id + ":" + role,
                shape=shape, strides_bytes=(shape[-1] * 2, 2), offset_bytes=base.view.offset_bytes)
                for role, base, shape in (("gate_up", gate_view, (rows * 6, 2 * width)),
                    ("weighted_expert_output", gate_view, (rows * 6, hidden)),
                    ("activated", common_view, (rows * 6, width)),
                    ("provisional_fused_out", common_view, (rows, hidden))))
        self._matrix_scratch_storage(node, "w13", 2 * width, capacity, block)
        if cache_views:
            activation_input = self.trace.alias(cache_views[0], node.node_id + ":activation-input",
                shape=(rows * 6, 2 * width), strides_bytes=(4 * width, 2), offset_bytes=cache_views[0].view.offset_bytes)
        self.trace.checkpoint(node.node_id + ":activation-complete")
        self._matrix_scratch_storage(node, "w2", hidden, capacity, block)
        return (None if not cache_views else cache_views[1]), tuple(view.view.view_id for view in cache_views)

    def __call__(self, x, topk_weights, topk_ids):
        model, rows = self.model, x.shape[0]
        if (x.shape != (rows, model.hidden_size) or x.dtype != "bf16"
                or topk_weights.shape != (rows, 6) or topk_weights.dtype != "fp32"
                or topk_ids.shape != (rows, 6) or topk_ids.dtype != "int32"):
            raise ValueError("modular Marlin expert inputs disagree with the typed source contract")
        counts = self.resolution.tokens_per_expert
        valid = rows if self.valid_tokens is None else self.valid_tokens
        if valid > rows or len(counts) != model.num_experts or sum(counts) != valid * 6:
            raise ValueError("explicit routing histogram must conserve all top6 assignments")
        owned_count = model.num_experts // self.expert_parallel_size
        owned = tuple(range(self.expert_rank * owned_count, (self.expert_rank + 1) * owned_count))
        node = self.node(WorkloadKind.FUSED_MOE, {"tokens": rows, "experts": model.num_experts,
            "topk": 6, "hidden": model.hidden_size, "intermediate": self.intermediate},
            "bf16", "full", {"source_component": "vllm_0_30_symbolic_v4_marlin",
                "expert_parallel_size": self.expert_parallel_size, "expert_rank": self.expert_rank, "owned_experts": owned,
                "tokens_per_expert": counts, "local_tokens_per_expert": tuple(counts[index] for index in owned),
                "source_expert_tensor_parallel_size": self.expert_tp_size,
                "source_logical_intermediate": self.logical_intermediate,
                "source_marlin_padding_columns": self.intermediate - self.logical_intermediate,
                **({"source_valid_tokens": valid, "source_padding_routing": "hash_topk_masked_ids_minus_one"}
                    if self.valid_tokens is not None else {}),
                "expert_placement_strategy": "linear", "moe_invocation_id": self.invocation_id,
                "output_is_reduced": False, "routed_scaling_factor": model.routed_scaling_factor,
                "routed_weight_dtype": "mxfp4", "weight_group_size": 32, "swiglu_limit": model.swiglu_limit,
                "routing_provenance": dict(self.resolution.routing_provenance),
                "symbolic_workspace_complete": False,
                "symbolic_modular_arena_bound": self.workspace is not None,
                "symbolic_marlin_multiprocessors": self.source_multiprocessors})
        # FusedMoEKernel.apply allocates output before _prepare/_allocate_buffers.
        output = self.trace.tensor(node.node_id, x.shape, x.dtype)
        scratch = ()
        h, n = model.hidden_size, self.intermediate
        if self.workspace is not None:
            common = max(rows * 6 * max(n, h), rows * h)
            scratch = self.workspace.get_simultaneous(((common,), "bf16"),
                ((rows * 6 * max(2 * n, h),), "bf16"))
        private = ()
        cache_view_ids = ()
        if self.source_multiprocessors is not None:
            block = 64
            estimated_rows = -(-(rows * owned_count) // model.num_experts)
            for candidate in (8, 16, 32, 48, 64):
                if estimated_rows * 6 / owned_count / candidate < 0.9:
                    block = candidate
                    break
            aligned, capacity = self._alignment_storage(node, rows * 6, model.num_experts, block)
            weighted_output, cache_view_ids = self._matrix_program_storage(node, scratch, rows, h, n, capacity, block)
            if weighted_output is not None:
                reduction_input = self.trace.alias(weighted_output, node.node_id + ":reduction-input",
                    shape=(rows, 6, h), strides_bytes=(12 * h, 2 * h, 2), offset_bytes=weighted_output.view.offset_bytes)
            self.trace.checkpoint(node.node_id + ":reduction-complete")
            private = aligned
        node = node.with_attributes({**node.attributes,
            "symbolic_marlin_arena_views": cache_view_ids})
        return self.trace.call(node, (x, topk_weights, topk_ids), shape=x.shape, dtype=x.dtype,
            out=output, writes=(*scratch, *private))


class SymbolicV4Compressor(SymbolicOperator):
    """V4 save-partial then finish, with explicit cache objects and CPU lengths.

    State slots/pages are caller-declared logical arenas. Their existence does
    not bind a native allocator, cross-layer page sharing or actual state
    values. Forward returns None, as the source mutates its two caches.
    """

    def __init__(self, trace, name, pool_name, *, head_dim, compress_ratio, q_lens, kv_lens,
                 state_cache, kv_cache, graph_runtime_mode, slot_mapping, dummy_run=False):
        super().__init__(trace, name, pool_name)
        if (head_dim, compress_ratio) not in ((512, 4), (512, 128), (128, 4)):
            raise NotImplementedError("symbolic V4 compressor requires the main C4/C128 or indexer C4 source")
        if (not isinstance(q_lens, tuple) or not isinstance(kv_lens, tuple) or not q_lens
                or len(q_lens) != len(kv_lens)
                or any(type(q) is not int or type(kv) is not int or q <= 0 or kv < q
                    for q, kv in zip(q_lens, kv_lens, strict=True))):
            raise ValueError("compressor requires aligned explicit positive query/context lengths")
        if graph_runtime_mode not in ("NONE", "PIECEWISE", "FULL"):
            raise ValueError("compressor requires an explicit graph runtime mode")
        coff = 2 if compress_ratio == 4 else 1
        record_bytes = 584 if head_dim == 512 else 132
        trace._check_tensor(state_cache)
        trace._check_tensor(kv_cache)
        if (state_cache.dtype != "fp32" or state_cache.shape[-1] != 2 * coff * head_dim
                or kv_cache.dtype != "uint8" or kv_cache.shape[-1] != record_bytes):
            raise ValueError("compressor requires its FP32 partial and UINT8 compressed cache layouts")
        if type(dummy_run) is not bool:
            raise TypeError('V4 dummy compressor requires an explicit boolean')
        if (not isinstance(slot_mapping, tuple) or len(slot_mapping) != sum(q_lens)
                or any(type(slot) is not int or (slot!=-1 if dummy_run else not 0<=slot<prod(state_cache.shape[:-1])) for slot in slot_mapping)):
            raise NotImplementedError("symbolic compressor requires explicit valid partial-state slots")
        self.dummy_run=dummy_run
        self.head_dim, self.ratio, self.coff = head_dim, compress_ratio, coff
        self.q_lens, self.kv_lens = q_lens, kv_lens
        self.state_cache, self.kv_cache = state_cache, kv_cache
        self.graph_runtime_mode = graph_runtime_mode
        self.slot_mapping = slot_mapping
        self.save = SymbolicOperator(trace, name + ":save-partial", pool_name)
        self.finish = SymbolicOperator(trace, name + ":finish", pool_name)

    def __call__(self, kv_score, positions, rotary_cache):
        rows = sum(self.q_lens)
        width = self.coff * self.head_dim
        if (kv_score.shape != (rows, 2 * width) or kv_score.dtype != "fp32"
                or positions.shape != (rows,) or positions.dtype != "int64"
                or len(rotary_cache.shape) != 2 or rotary_cache.shape[-1] != 64 or rotary_cache.dtype != "fp32"):
            raise ValueError("compressor source requires FP32 projected KV/score and RoPE cache with INT64 positions")
        kv, score = kv_score.split((width, width), dim=-1)
        partial_bytes = 2 * width * 4
        node = self.save.node(WorkloadKind.KV_UPDATE,
            {"elements": rows * partial_bytes, "seq_len": rows,
             "state_bytes": partial_bytes, "dtype_bytes": 4}, "fp32", "dsv4_compressor_state_cache", {
                "source_component": "vllm_0_30_symbolic_v4_compressor",
                "cache_layout": "paged_dsv4_compressor_state", "cache_write_semantics": "all_PAD_no_store" if self.dummy_run else "append",
                "symbolic_slot_mapping": self.slot_mapping,
                "source_dummy_run":self.dummy_run,"cache_write_valid_rows":0 if self.dummy_run else rows,
                "slot_mapping_scope": "declared_logical_pool_not_native_physical_pages"})
        if self.dummy_run:
            self.trace.call_effect(node,(kv,score,positions))
        else:
            self.trace.call(node, (kv, score, positions), shape=self.state_cache.shape, dtype="fp32", out=self.state_cache)
        old_lens = tuple(kv - q for q, kv in zip(self.q_lens, self.kv_lens, strict=True))
        # _get_c128_boundary: any(start %128 + query >=128). CPU lengths
        # are explicit here, so this source branch never guesses a boundary.
        states = sum(new // self.ratio - old // self.ratio for old, new in zip(old_lens, self.kv_lens, strict=True))
        if self.ratio == 128 and self.graph_runtime_mode != "FULL" and states == 0:
            return None
        if self.dummy_run:
            # Dummy positions are all zero and partial slots all PAD. C4's
            # finish launch still occurs, but no lane reads/writes cache.
            if self.ratio==128:
                raise NotImplementedError('V4 dummy C128 boundary execution needs its explicit positions')
            node=self.finish.node(WorkloadKind.KV_UPDATE,{'elements':0,'seq_len':rows,'compress_ratio':self.ratio,
                'state_bytes':584 if self.head_dim==512 else 132,'dtype_bytes':1},'uint8','dsv4_compressed_state_cache',{
                'source_component':'vllm_0_30_symbolic_v4_compressor','source_dummy_run':True,
                'cache_write_valid_rows':0,'cache_write_semantics':'all_PAD_no_store'})
            self.trace.call_effect(node,(positions,rotary_cache))
            return None
        record_bytes = 584 if self.head_dim == 512 else 132
        node = self.finish.node(WorkloadKind.KV_UPDATE,
            {"elements": states * record_bytes, "seq_len": rows, "compress_ratio": self.ratio,
             "state_bytes": record_bytes, "dtype_bytes": 1}, "uint8", "dsv4_compressed_state_cache", {
                "source_component": "vllm_0_30_symbolic_v4_compressor",
                "cache_layout": "paged_dsv4_compressed", "cache_write_semantics": "append",
                "quantization": "fp8_ds_mla" if self.head_dim == 512 else "fp8_e4m3_with_ue8m0_scale",
                "state_old_lens": old_lens, "state_new_lens": self.kv_lens})
        if self.ratio == 128:
            # SparseAttnCompressC128Block8Kernel.__call__ allocates a fresh
            # FP32[num_actual,512], including non-boundary rows. Its store
            # consumer reads completed rows, then this source scope releases it.
            from dataclasses import replace
            compress_node = replace(node, node_id=node.node_id + ":compress", dtype="fp32",
                shape={**node.shape, "elements": states * 512, "dtype_bytes": 4},
                attributes={**node.attributes, "finish_stage": "compress"})
            compressed = self.trace.call(compress_node, (self.state_cache, positions),
                shape=(rows, 512), dtype="fp32")
            store_node = node.with_attributes({**node.attributes, "finish_stage": "store"}, node_id=node.node_id + ":store")
            self.trace.call(store_node, (compressed, positions, rotary_cache),
                shape=self.kv_cache.shape, dtype="uint8", out=self.kv_cache)
            return None
        self.trace.call(node, (self.state_cache, positions, rotary_cache),
            shape=self.kv_cache.shape, dtype="uint8", out=self.kv_cache)
        return None


class SymbolicV4KVInsert(SymbolicOperator):
    """Fused query norm/RoPE and UINT8 SWA insert, fresh padded BF16 Q."""

    def __init__(self, trace, name, pool_name, model, *, tp_size, cache, slot_mapping, slot_tensor=None,
            masked_dspark_context=False, physical_query_rows=None, dummy_run=False):
        super().__init__(trace, name, pool_name)
        trace._check_tensor(cache)
        if (not _v4_source_block_geometry(model, tp_size, expert_parallel=True)
                or cache.dtype != "uint8" or cache.shape[-1] != 584):
            raise NotImplementedError("symbolic V4 insert requires legal TP and the fp8_ds_mla SWA cache")
        if (type(masked_dspark_context) is not bool or masked_dspark_context and not model.is_dspark_draft):
            raise ValueError("masked context insert is restricted to explicit DSpark context updates")
        if type(dummy_run) is not bool:
            raise TypeError('V4 dummy Q/KV insert requires an explicit boolean')
        if (not isinstance(slot_mapping, tuple) or not slot_mapping
                or dummy_run and any(slot!=-1 for slot in slot_mapping)
                or any(type(slot) is not int or not (-1 if masked_dspark_context or dummy_run else 0) <= slot < prod(cache.shape[:-1])
                    for slot in slot_mapping)):
            raise NotImplementedError("symbolic V4 insert requires explicit valid SWA slots")
        self.masked_dspark_context = masked_dspark_context
        self.dummy_run=dummy_run
        if slot_tensor is not None:
            trace._check_tensor(slot_tensor)
            if slot_tensor.shape != (len(slot_mapping),) or slot_tensor.dtype != "int64":
                raise ValueError("V4 insert slot port requires the actual INT64 metadata tensor")
        self.slot_tensor = slot_tensor
        if physical_query_rows is None:
            physical_query_rows = len(slot_mapping)
        if (type(physical_query_rows) is not int or physical_query_rows < len(slot_mapping)
                or physical_query_rows != len(slot_mapping)
                and (not model.is_dspark_draft or slot_tensor is None or masked_dspark_context)):
            raise ValueError("DP-padded Q requires DSpark's actual active slot port and explicit physical rows")
        self.query_rows = physical_query_rows
        self.model, self.tp_size, self.cache, self.slots = model, tp_size, cache, slot_mapping

    def __call__(self, q, kv, positions, rotary_cache):
        model, rows = self.model, self.query_rows
        heads = model.num_attention_heads // self.tp_size
        padded = 64 if heads <= 64 else 128
        if (q.shape != (rows, heads, model.head_dim) or q.dtype != "bf16"
                or kv.shape != (rows, model.head_dim) or kv.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"
                or len(rotary_cache.shape) != 2 or rotary_cache.shape[-1] != model.qk_rope_head_dim
                or rotary_cache.dtype != "fp32"):
            raise ValueError("V4 insert requires source Q/KV, INT64 positions and FP32 RoPE cache")
        node = self.node(WorkloadKind.KV_UPDATE, {"elements": (rows * heads + (0 if self.dummy_run else len(self.slots))) * model.head_dim,
            "seq_len": rows, "num_query_heads": heads, "padded_query_heads": padded,
            "head_dim": model.head_dim, "rope_dim": model.qk_rope_head_dim, "cache_token_bytes": 584},
            "bf16", "dsv4_qnorm_rope_kv_insert", {
                "source_component": "vllm_0_30_symbolic_v4_swa_insert",
                "cache_layout": "paged_dsv4_swa", "cache_write_semantics": ("all_PAD_no_store" if self.dummy_run else
                    "masked_target_context_overwrite" if self.masked_dspark_context else "append"),
                "quantization": "fp8_ds_mla", "slot_mapping_all_valid": -1 not in self.slots,
                "masked_dspark_context": self.masked_dspark_context,
                "source_dummy_run":self.dummy_run,
                "query_physical_rows": rows, "cache_insert_rows": len(self.slots),
                "dp_padding_KV_insert_skipped": rows > len(self.slots),
                "cache_write_valid_rows": sum(slot >= 0 for slot in self.slots),
                "symbolic_slot_mapping": self.slots,
                "slot_mapping_scope": "declared_logical_pool_not_native_physical_pages"})
        inputs = (q, kv, positions, rotary_cache)
        if self.slot_tensor is not None:
            inputs += (self.slot_tensor,)
        return self.trace.call(node, inputs,
            shape=(rows, padded, model.head_dim), dtype="bf16", writes=() if self.dummy_run else (self.cache,))


class SymbolicV4ShortTopK(SymbolicOperator):
    def __init__(self, trace, name, pool_name, buffer):
        super().__init__(trace, name, pool_name)
        trace._check_tensor(buffer)
        if len(buffer.shape) != 2 or buffer.dtype != "int32":
            raise ValueError("V4 short-context source requires the persistent INT32 top-k buffer")
        self.buffer = buffer

    def __call__(self, positions, *, controls=()):
        rows = positions.shape[0]
        if positions.shape != (rows,) or positions.dtype != "int64" or rows > self.buffer.shape[0]:
            raise ValueError("short-context source positions must fit the top-k buffer")
        output = self.buffer.narrow(0, 0, rows)
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * output.shape[-1],
            "seq_len": rows, "topk_capacity": output.shape[-1], "compress_ratio": 4},
            "int32", "dsv4_short_context_topk", {
                "source_component": "vllm_0_30_symbolic_v4_short_topk", "position_bytes": 8})
        return self.trace.call(node, (positions,), shape=output.shape, dtype="int32", out=output, controls=controls)


class SymbolicV4IndexerQuery(SymbolicOperator):
    """Source fused RoPE/FP8 Q with the quantization scale folded into weights."""

    def __init__(self, trace, name, pool_name, model):
        super().__init__(trace, name, pool_name)
        if model.model_type != "deepseek_v4" or model.index_head_dim != 128:
            raise NotImplementedError("V4 FP8 indexer query requires its128-wide source heads")
        self.model = model

    def __call__(self, q, weights, positions, rotary_cache):
        model, rows = self.model, q.shape[0]
        if (q.shape != (rows, model.index_n_heads, model.index_head_dim) or q.dtype != "bf16"
                or weights.shape != (rows, model.index_n_heads) or weights.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"
                or rotary_cache.dtype != "fp32" or rotary_cache.shape[-1] != model.qk_rope_head_dim):
            raise ValueError("V4 indexer query requires BF16 Q/weights, INT64 positions and FP32 RoPE")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * model.index_n_heads * model.index_head_dim,
            "seq_len": rows, "num_query_heads": model.index_n_heads,
            "head_dim": model.index_head_dim, "rope_dim": model.qk_rope_head_dim},
            "bf16", "dsv4_indexer_q_rope_quant", {
                "source_component": "vllm_0_30_symbolic_v4_indexer_query", "q_scale_folded_into_weights": True})
        return self.trace.call_outputs(node, (q, weights, positions, rotary_cache), outputs=(
            ("q_fp8", q.shape, "fp8"), ("weights", weights.shape, "fp32")))


class SymbolicV4ScoredTopK(SymbolicOperator):
    """CP1 score/selector call, with an explicit output-cardinality domain.

    The existing predictor supplies its useful mathematical/byte baseline.
    The optional source manager binds the FP8/UINT8 prefill gather arena and
    common CUDA/auto decode-selector workspace request. Score backing,
    private refinement scratch and integer selection service remain
    unbound. The declared output has min(K, causal completed states) distinct
    nonnegative indices; this is caller input, not a conclusion from lengths,
    finite logits, checkpoint weights, or an installed selector's behavior.
    """

    def __init__(self, trace, name, pool_name, model, buffer, cache, *, q_lens, kv_lens,
                 request_phases, selection_source, workspace=None, source_max_model_len=None):
        super().__init__(trace, name, pool_name)
        if selection_source != "declared_causal_topk":
            raise NotImplementedError("scored C4 requires an explicit selected-cardinality declaration")
        if (not isinstance(q_lens, tuple) or not q_lens or not isinstance(kv_lens, tuple)
                or not isinstance(request_phases, tuple) or len(q_lens) != len(kv_lens)
                or len(q_lens) != len(request_phases)
                or any(type(q) is not int or type(kv) is not int or q <= 0 or kv < q
                    for q, kv in zip(q_lens, kv_lens, strict=True))
                or any(phase not in ("decode", "prefill") for phase in request_phases)
                or tuple(sorted(request_phases, key=lambda phase: phase == "prefill")) != request_phases):
            raise ValueError("scored C4 requires complete source-packed query/context/phase metadata")
        trace._check_tensor(buffer)
        trace._check_tensor(cache)
        if (buffer.dtype != "int32" or buffer.shape[-1] != model.index_topk or buffer.shape[0] < sum(q_lens)
                or cache.dtype != "uint8" or cache.shape[-1] != model.index_head_dim + 4):
            raise ValueError("V4 scored indexer requires typed shared top-k and packed indexer cache")
        self.model, self.buffer, self.cache = model, buffer, cache
        self.q_lens, self.kv_lens, self.phases = q_lens, kv_lens, request_phases
        if source_max_model_len is not None and (type(source_max_model_len) is not int
                or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings):
            raise ValueError("V4 indexer maximum length must cover the declared contexts")
        if workspace is not None:
            if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
                raise ValueError("V4 indexer workspace must belong to this physical-rank source trace")
            if (type(source_max_model_len) is not int
                    or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings):
                raise ValueError("V4 indexer workspace needs an explicit maximum length covering the contexts")
        self.workspace, self.max_model_len = workspace, source_max_model_len

    def __call__(self, q, weights, *, controls=()):
        model, rows = self.model, sum(self.q_lens)
        if (q.shape != (rows, model.index_n_heads, model.index_head_dim) or q.dtype != "fp8"
                or weights.shape != (rows, model.index_n_heads) or weights.dtype != "fp32"):
            raise ValueError("V4 scored indexer requires prepared FP8 Q and folded FP32 weights")
        decode_rows = sum(n for n, phase in zip(self.q_lens, self.phases, strict=True) if phase == "decode")
        # sparse_attn_indexer keeps these full gather buffers (and its last
        # chunk views) alive through the subsequent decode branch. Locals,
        # rather than module attributes, release them at this call's return.
        gather_views, scale_fp32 = (), None
        shared_clear = None
        shared_prefix = (self.buffer.narrow(0, 0, rows)
            if decode_rows and decode_rows < rows else None)
        for phase in ("prefill", "decode"):
            indices = tuple(i for i, value in enumerate(self.phases) if value == phase)
            if not indices:
                continue
            queries, lengths = tuple(self.q_lens[i] for i in indices), tuple(self.kv_lens[i] for i in indices)
            count, offset = sum(queries), decode_rows if phase == "prefill" else 0
            output = self.buffer.narrow(0, offset, count)
            gather_request = None
            if phase == "prefill" and self.workspace is not None:
                # DeepseekV4Indexer.max_total_seq_len is source helper
                # get_max_prefill_buffer_size(config)//4 = max_model_len*40//4.
                # The wrapper reserves that capacity, not active KV rows.
                gather_views, scale_fp32, gather_request = _fp8_indexer_gather_workspace(
                    self.trace, self.workspace, self.name, max_model_len=self.max_model_len,
                    head_dim=model.index_head_dim, compress_ratio=4)
            node = self.node(WorkloadKind.EAGER_ATTN, {"batch": len(indices), "q_len": max(queries),
                "kv_len": max(lengths), "num_query_heads": model.index_n_heads,
                "head_dim": model.index_head_dim, "topk": min(model.index_topk, max(lengths) // 4),
                "compress_ratio": 4, "dtype_bytes": 4}, "int32", "dsv4_indexer_" + phase, {
                    "source_component": "vllm_0_30_symbolic_v4_scored_indexer", "causal": True,
                    "cache_layout": "paged_mla_indexer", "cache_quantization": "fp8_e4m3_with_ue8m0_scale",
                    "q_lens": queries, "kv_lens": lengths, "request_phases": (phase,) * len(indices),
                    "score_accumulator_dtype": "fp32", "score_function": "weighted_dot_product_topk",
                    "topk_capacity": model.index_topk, "topk_index_dtype": "int32", "topk_sentinel": -1,
                    "symbolic_workspace_complete": False, "score_backing_bound": False,
                    "symbolic_prefill_gather_workspace": gather_request,
                    "selection_cardinality_source": "declared_causal_topk"})
            if shared_prefix is not None and phase == "prefill":
                shared_clear = {"owner_node_id": node.node_id,
                    "view_id": shared_prefix.view.view_id, "rows": rows}
            node = node.with_attributes({**node.attributes,
                "symbolic_shared_topk_clear": shared_clear,
                "symbolic_topk_token_range": (offset, offset + count)})
            empty = phase == "prefill" and all(length // 4 == 0 for length in lengths)
            selector_views, selector_request = (), None
            if phase == "decode" and self.workspace is not None and model.index_topk in (512, 1024, 2048):
                # vLLM0.30 SparseIndexerTopk auto on the declared CUDA source:
                # cooperative (if eligible) and otherwise persistent both
                # request RADIX_TOPK_WORKSPACE_SIZE. No score stride or actual
                # backend is selected to infer this common allocation.
                selector_views = self.workspace.get_simultaneous(((1024 * 1024,), "uint8"))
                selector_request = {"view_id": selector_views[0].view.view_id,
                    "configured_backend": "auto", "possible_backends": (("cooperative", "persistent")
                        if count <= 64 else ("persistent",))}
            node = node.with_attributes({**node.attributes,
                "symbolic_decode_selector_workspace": selector_request,
                "symbolic_score_logical_shape": ((count, self.max_model_len // 4)
                    if phase == "decode" and self.max_model_len is not None else None)})
            inputs = (() if empty else
                (q.narrow(0, offset, count), weights.narrow(0, offset, count), self.cache))
            self.trace.call(node, inputs, shape=output.shape, dtype="int32", out=output,
                writes=((shared_prefix,) if shared_prefix is not None and phase == "prefill" else ())
                    + (gather_views if phase == "prefill" and not empty else ()) + selector_views,
                controls=(*controls, q, weights, self.cache) if empty else controls)
        return None


class SymbolicV4SparseAttention(SymbolicOperator):
    """Source FlashMLA wrapper over all-completed compressed states and SWA.

    Existing SPARSE_ATTN pricing includes prefill gathers/index materialization
    and decode index conversion. Source workspace-manager storage remains
    unbound; this is not a complete activation-memory or provider claim.
    """

    def __init__(self, trace, name, pool_name, model, *, compress_ratio, q_lens, kv_lens,
                 request_phases, swa_cache, compressed_cache, topk_buffer, c128_indices,
                 source_max_model_len, indexer_selection_source=None, workspace=None, source_max_batched_tokens=None,dummy_run=False):
        super().__init__(trace, name, pool_name)
        if compress_ratio not in (1, 4, 128):
            raise ValueError("V4 sparse source requires SWA, C4 or C128")
        if (not isinstance(q_lens, tuple) or not q_lens or not isinstance(kv_lens, tuple)
                or not isinstance(request_phases, tuple) or len(q_lens) != len(kv_lens)
                or len(q_lens) != len(request_phases)
                or any(type(q) is not int or type(kv) is not int or q <= 0 or kv < q
                    for q, kv in zip(q_lens, kv_lens, strict=True))
                or any(phase not in ("decode", "prefill") for phase in request_phases)):
            raise ValueError("V4 sparse source requires complete query/context/phase metadata")
        if tuple(sorted(request_phases, key=lambda phase: phase == "prefill")) != request_phases:
            raise ValueError("V4 source rows must pack decode requests before prefill")
        if type(source_max_model_len) is not int or not max(kv_lens) <= source_max_model_len <= model.max_position_embeddings:
            raise ValueError("V4 source maximum length must contain the declared contexts")
        long_c4 = compress_ratio == 4 and max(kv_lens) // 4 > model.index_topk
        if long_c4 and indexer_selection_source != "declared_causal_topk":
            raise NotImplementedError("long C4 source requires an explicit selected-cardinality declaration")
        trace._check_tensor(swa_cache)
        if swa_cache.dtype != "uint8" or swa_cache.shape[-1] != 584:
            raise ValueError("V4 sparse source requires the UINT8 SWA cache")
        if compress_ratio != 1:
            trace._check_tensor(compressed_cache)
            if compressed_cache.dtype != "uint8" or compressed_cache.shape[-1] != 584:
                raise ValueError("V4 sparse source requires the UINT8 compressed cache")
        if compress_ratio == 4:
            trace._check_tensor(topk_buffer)
            if topk_buffer.dtype != "int32" or topk_buffer.shape[-1] != model.index_topk or topk_buffer.shape[0] < sum(q_lens):
                raise ValueError("C4 attention requires the shared INT32 top-k buffer")
        if compress_ratio == 128:
            from .runtime_inputs import dsv4_c128_metadata_width
            capacity = dsv4_c128_metadata_width(max(kv_lens), source_max_model_len)
            if not isinstance(c128_indices, tuple) or len(c128_indices) != 2:
                raise ValueError("C128 source requires separate prefill/decode metadata input buffers")
            for phase, tensor in zip(("prefill", "decode"), c128_indices, strict=True):
                count = sum(q for q, p in zip(q_lens, request_phases, strict=True) if p == phase)
                if count:
                    trace._check_tensor(tensor)
                    if tensor.dtype != "int32" or tensor.shape != (count, capacity):
                        raise ValueError("C128 metadata input shape/dtype disagrees with source width")
        else:
            capacity = model.index_topk
        self.model, self.ratio, self.capacity = model, compress_ratio, capacity
        if type(dummy_run) is not bool or dummy_run and (len(q_lens)!=1 or q_lens[0] not in ((5,) if model.is_dspark_draft else (1,6))
                or kv_lens!=tuple(q+1 if model.is_dspark_draft else q for q in q_lens) or request_phases!=('decode',)):
            raise ValueError('V4 dummy attention requires one declared uniform decode batch with no context')
        self.dummy_run=dummy_run
        self.q_lens, self.kv_lens, self.phases = q_lens, kv_lens, request_phases
        self.swa_cache, self.compressed_cache = swa_cache, compressed_cache
        self.topk_buffer, self.c128_indices = topk_buffer, c128_indices
        if workspace is not None:
            if not isinstance(workspace, ForwardWorkspace) or workspace.trace is not trace:
                raise ValueError("V4 prefill workspace must belong to this physical-rank source trace")
            if type(source_max_batched_tokens) is not int or source_max_batched_tokens < sum(q_lens):
                raise ValueError("V4 prefill workspace needs an explicit batch capacity covering all rows")
        self.workspace, self.max_batched_tokens = workspace, source_max_batched_tokens
        self.source_max_model_len = source_max_model_len
        self.selection = ("declared_causal_topk" if long_c4 else
            "all_completed_states" if compress_ratio > 1 else "candidate_upper_bound")

    def _prefill_workspace(self, queries, lengths):
        """Source sparse_swa chunk plan and FlashMLA typed workspace ports.

        PREFILL_CHUNK_SIZE=4 bounds area, not request count. Chunk planning
        and descriptor sizes use only request metadata and declared limits.
        This does not bind other manager users, FlashMLA callee scratch or
        an actual worker's incoming arena history.
        """
        if self.workspace is None:
            return (), None
        ratio, model = self.ratio, self.model
        compressed = tuple(length // ratio if ratio > 1 else 0 for length in lengths)
        gathers = tuple(q + min(length - q, model.window_size - 1)
            for q, length in zip(queries, lengths, strict=True))
        max_area = 4 * ((-(-self.source_max_model_len // ratio) if ratio > 1 else 0)
            + model.window_size + self.max_batched_tokens)
        topk = 0 if ratio == 1 else self.capacity
        combined = -(-(topk + model.window_size) // 128) * 128
        requests, views = [], ()
        start = 0
        while start < len(queries):
            maximum_compressed, maximum_gather = compressed[start], gathers[start]
            end = start + 1
            while end < len(queries):
                next_compressed, next_gather = max(maximum_compressed, compressed[end]), max(maximum_gather, gathers[end])
                if (end - start + 1) * (next_compressed + next_gather) > max_area:
                    break
                maximum_compressed, maximum_gather = next_compressed, next_gather
                end += 1
            width = maximum_compressed + maximum_gather
            ports = (((end - start, width, model.head_dim), "bf16"),
                ((self.max_batched_tokens, combined), "int32"), ((self.max_batched_tokens,), "int32"))
            # Retain the old views while the next get_simultaneous executes,
            # as the source loop's previous workspace tuple does.
            views = self.workspace.get_simultaneous(*ports)
            active_rows = sum(queries[start:end])
            active_indices = views[1].narrow(0, 0, active_rows).unsqueeze(1)
            active_lengths = views[2].narrow(0, 0, active_rows)
            gathered = views[0].view_as_shape((end - start) * width, 1, model.head_dim)
            requests.append({"request_range": (start, end), "compressed_width": maximum_compressed,
                "gathered_width": width, "ports": ports, "view_ids": tuple(view.view.view_id for view in views),
                "callee_view_ids": tuple(view.view.view_id for view in (gathered, active_indices, active_lengths))})
            start = end
        return views, tuple(requests)

    def __call__(self, q, positions, out, *, controls=()):
        model, rows = self.model, sum(self.q_lens)
        if (len(q.shape) != 3 or q.shape[0] != rows or q.shape[1] not in (64, 128)
                or q.shape[-1] != model.head_dim or q.dtype != "bf16"
                or out.shape != q.shape or out.dtype != "bf16"
                or positions.shape != (rows,) or positions.dtype != "int64"):
            raise ValueError("V4 sparse source requires padded BF16 Q/output and INT64 positions")
        if self.dummy_run:
            # The real metadata marks every PAD slot invalid. Its indexer
            # clears indices/lens, and FlashMLA launches with empty KV lists.
            index=self.node(WorkloadKind.VECTOR,{'elements':rows*self.model.window_size,'rows':rows},'int32',
                'dsv4_dummy_swa_indices',{'source_component':'vllm_0_30_symbolic_v4_sparse',
                    'source_dummy_run':True,'all_slot_mapping_PAD':True,'indices_fill':-1,'used_lengths_fill':0})
            indices,lengths=self.trace.call_outputs(index,(positions,),outputs=(
                ('indices',(rows,1,self.model.window_size),'int32'),('used_lengths',(rows,),'int32')))
            node=self.node(WorkloadKind.EAGER_ATTN,{'batch':1,'q_len':rows,'kv_len':rows,
                'num_query_heads':q.shape[1],'qk_head_dim':model.head_dim,'v_head_dim':model.head_dim,
                'topk':0,'window_size':model.window_size,'compress_ratio':self.ratio,'dtype_bytes':2},'bf16','dsv4_sparse_decode',{
                    'source_component':'vllm_0_30_symbolic_v4_sparse','source_dummy_run':True,
                    'all_slot_mapping_PAD':True,'effective_KV_rows':0,'cache_read_bytes':0,
                    'output_value':0,'native_provider_bound':False,'complete_priced_request':False})
            self.trace.call(node,(q,indices,lengths),shape=out.shape,dtype='bf16',out=out,controls=controls)
            return None
        decode_rows = sum(n for n, phase in zip(self.q_lens, self.phases, strict=True) if phase == "decode")
        for phase in ("prefill", "decode"):
            indices = tuple(i for i, p in enumerate(self.phases) if p == phase)
            if not indices:
                continue
            queries, lengths = tuple(self.q_lens[i] for i in indices), tuple(self.kv_lens[i] for i in indices)
            count, offset = sum(queries), decode_rows if phase == "prefill" else 0
            query, output = q.narrow(0, offset, count), out.narrow(0, offset, count)
            selected = (tuple(min(self.capacity, length // self.ratio) for length in lengths)
                if self.ratio > 1 else (0,) * len(lengths))
            inputs = (query, self.swa_cache)
            if self.ratio != 1:
                inputs += (self.compressed_cache,)
            if self.ratio == 4:
                inputs += (self.topk_buffer.narrow(0, offset, count),)
            elif self.ratio == 128:
                inputs += (self.c128_indices[0 if phase == "prefill" else 1],)
            workspace_views, workspace_requests = (self._prefill_workspace(queries, lengths)
                if phase == "prefill" else ((), None))
            node = self.node(WorkloadKind.EAGER_ATTN, {"batch": len(indices), "q_len": max(queries),
                "kv_len": max(lengths), "num_query_heads": q.shape[1], "qk_head_dim": model.head_dim,
                "v_head_dim": model.head_dim, "topk": max(selected), "window_size": model.window_size,
                "compress_ratio": self.ratio, "compressed_state_bytes": 584, "swa_token_bytes": 584,
                "dtype_bytes": 2}, "bf16", "dsv4_sparse_" + phase, {
                    "source_component": "vllm_0_30_symbolic_v4_sparse", "causal": True,
                    "indexer_type": "shared", "kv_layout": "mla", "kv_dist": "uniform",
                    "q_lens": queries, "kv_lens": lengths, "selected_kv_lens": selected,
                    "swa_kv_lens": tuple(min(model.window_size, n) for n in lengths),
                    "topk_capacity": self.capacity, "topk_index_dtype": "int32",
                    # SWA has no compressed tier; its existing capacity
                    # contract is exactly zero, with no selected-ID ambiguity.
                    "compressed_selection": self.selection,
                    "symbolic_prefill_workspace_requests": workspace_requests,
                    "symbolic_workspace_complete": False})
            decode_buffers = ()
            if phase == "decode" and self.ratio == 4:
                # compute_global_topk_indices_and_lens allocates these two
                # tensors before FlashMLA consumes their views. Python locals
                # retain both until this source decode scope returns.
                global_topk = self.trace.tensor(node.node_id + ":decode-global-topk",
                    (count, self.capacity), "int32")
                topk_lens = self.trace.tensor(node.node_id + ":decode-topk-lens", (count,), "int32")
                decode_buffers = (global_topk, topk_lens)
            callee_views = {}
            if phase == "decode":
                callee_views = {"query": query.unsqueeze(1), "output": output.unsqueeze(1)}
                if self.ratio == 4:
                    callee_views["compressed_indices"] = global_topk.unsqueeze(1)
                elif self.ratio == 128:
                    callee_views["compressed_indices"] = self.c128_indices[1].unsqueeze(1)
            node = node.with_attributes({**node.attributes,
                "symbolic_decode_index_buffers": tuple(value.view.view_id for value in decode_buffers),
                "symbolic_decode_callee_views": {role: value.view.view_id for role, value in callee_views.items()}})
            self.trace.call(node, inputs, shape=output.shape, dtype="bf16", out=output,
                writes=(*workspace_views, *decode_buffers), controls=controls)
        return None


class SymbolicDSparkNoncausalSWA(SymbolicOperator):
    """Block-anchored prefix window plus every query, including future rows.

    Adapt sparse_swa.py ComputeDSparkNoncausalSWAIndicesKernel; logical slots
    are explicit declarations, not native block-table or provider evidence.
    """
    def __init__(self, trace, name, pool_name, model, *, q_lens, context_lens, cache,
            context_slots, query_slots, prepared_metadata=None,dummy_run=False):
        super().__init__(trace, name, pool_name)
        trace._check_tensor(cache)
        if (not model.is_dspark_draft or type(q_lens) is not tuple or type(context_lens) is not tuple
                or len(q_lens) != len(context_lens) or not q_lens
                or any(type(q) is not int or q != model.dspark_block_size for q in q_lens)
                or any(type(c) is not int or c < 1 for c in context_lens)
                or any(c + q > model.max_position_embeddings for c, q in zip(context_lens, q_lens, strict=True))
                or cache.dtype != "uint8" or len(cache.shape) != 2 or cache.shape[-1] != 584
                or cache.shape[0] < 128 or cache.shape[0] % 8):
            raise ValueError("DSpark attention requires complete all-valid query blocks and positive fresh context")
        if type(dummy_run) is not bool:
            raise TypeError('DSpark dummy SWA requires an explicit boolean')
        if (type(context_slots) is not tuple or type(query_slots) is not tuple
                or len(context_slots) != sum(context_lens) or len(query_slots) != sum(q_lens)
                or any(type(slot) is not int or (slot!=-1 if dummy_run else not 0<=slot<cache.shape[0]) for slot in context_slots + query_slots)
                or not dummy_run and len(set(context_slots + query_slots)) != len(context_slots) + len(query_slots)):
            raise ValueError("DSpark attention requires every distinct explicit valid context and query slot")
        self.dummy_run=dummy_run
        self.model, self.q_lens, self.context_lens, self.cache = model, q_lens, context_lens, cache
        self.index_width = -(-(model.window_size + model.dspark_block_size) // 64) * 64
        visible, cbase, qbase = [], 0, 0
        for context, query in zip(context_lens, q_lens, strict=True):
            visible.append(() if dummy_run else context_slots[cbase + max(context - model.window_size, 0):cbase + context]
                + query_slots[qbase:qbase + query])
            cbase += context
            qbase += query
        self.visible_slots = tuple(visible)
        self.prepared_metadata = () if prepared_metadata is None else prepared_metadata
        if dummy_run and (prepared_metadata is None or len(prepared_metadata)!=5):
            raise ValueError('DSpark dummy SWA requires actual prepared slot-validity metadata')
        if prepared_metadata is not None:
            if type(prepared_metadata) is not tuple or len(prepared_metadata) not in (3, 5):
                raise ValueError("DSpark prepared attention requires starts, lengths, table and optional mapping/validity pair")
            starts, lengths, table = prepared_metadata[:3]
            for value in prepared_metadata:
                trace._check_tensor(value)
            if (starts.shape != (len(q_lens) + 1,) or starts.dtype != "int32"
                    or lengths.shape != (len(q_lens),) or lengths.dtype != "int32"
                    or len(table.shape) != 2 or table.shape[0] != len(q_lens) or table.dtype != "int32"):
                raise ValueError("DSpark prepared attention metadata differs from its current request rows")
            if len(prepared_metadata) == 5:
                mapping, validity = prepared_metadata[3:]
                if (mapping.shape != (sum(q_lens),) or mapping.dtype != "int32"
                        or validity.shape != mapping.shape or validity.dtype != "bool"
                        or not mapping.is_contiguous or not validity.is_contiguous):
                    raise ValueError("DSpark prepared attention requires active INT32 request mapping and bool validity")

    def __call__(self, q, positions, output, *, controls=()):
        rows, model = sum(self.q_lens), self.model
        for value in (q, positions, output):
            self.trace._check_tensor(value)
        if (len(q.shape) != 3 or q.shape != output.shape or q.shape[0] < rows or q.shape[1] not in (64, 128)
                or q.dtype != "bf16" or output.dtype != "bf16"
                or q.shape[-1] != model.head_dim or positions.shape != (q.shape[0],) or positions.dtype != "int64"
                or q.shape[0] > rows and len(self.prepared_metadata) != 5
                or controls):
            raise ValueError("DSpark SWA requires padded BF16 queries/output and no compressed tier")
        physical_rows = q.shape[0]
        # FlashMLA consumes metadata.num_actual_tokens only. torch.empty's
        # DP tail remains unwritten, although the following O projection reads it.
        if physical_rows > rows:
            q, positions, output = q.narrow(0, 0, rows), positions.narrow(0, 0, rows), output.narrow(0, 0, rows)
        lengths = tuple(len(slots) for slots in self.visible_slots)
        attributes = {"source_component": "vllm_0_30_dspark_noncausal_swa",
            "causal": False, "q_lens": self.q_lens, "prefix_lens": self.context_lens,
            "visible_kv_lens": lengths, "visible_logical_slot_ids_by_request": self.visible_slots,
            "index_capacity": self.index_width, "future_query_keys_visible": True,
            "invalid_index_fill": -1, "slot_mapping_scope": "declared_logical_pool_not_native_physical_pages",
            "native_provider_bound": False, "prepared_query_lengths_and_table_bound": bool(self.prepared_metadata),
            "source_dummy_run":self.dummy_run,"effective_KV_rows":max(lengths),
            "dummy_output_zero":self.dummy_run,
            "physical_output_rows": physical_rows, "active_output_rows": rows,
            "unwritten_output_tail_rows": physical_rows - rows,
            "output_tail_initialization": "unwritten_torch_empty",
            "token_to_request_and_validity_metadata_bound": len(self.prepared_metadata) == 5}
        index_node = self.node(WorkloadKind.VECTOR, {"elements": rows * self.index_width,
            "rows": rows, "index_width": self.index_width}, "int32", "dspark_noncausal_swa_indices", attributes)
        indices, used = self.trace.call_outputs(index_node, (positions, *self.prepared_metadata), outputs=(
            ("indices", (rows, 1, self.index_width), "int32"), ("used_lens", (rows,), "int32")))
        node = self.node(WorkloadKind.EAGER_ATTN, {"batch": len(self.q_lens), "q_len": max(self.q_lens),
            "kv_len": max(c + n for c, n in zip(self.context_lens, self.q_lens, strict=True)),
            "num_query_heads": q.shape[1], "num_kv_heads": 1, "qk_head_dim": model.head_dim,
            "v_head_dim": model.head_dim, "window_size": model.window_size, "dtype_bytes": 2},
            "bf16", "dspark_noncausal_swa", attributes)
        self.trace.call(node, (q, indices, used) if self.dummy_run else (q, self.cache, indices, used),
            shape=output.shape, dtype="bf16", out=output)
        return None


class SymbolicV4AuxHiddenMean(SymbolicOperator):
    """Actual target auxiliary port: mhc_post(...).mean(dim=1), BF16."""

    def __call__(self, expanded):
        if expanded.dtype != "bf16" or len(expanded.shape) != 3 or not expanded.is_contiguous:
            raise ValueError("V4 auxiliary mean requires contiguous BF16 [rows,hc,hidden]")
        rows, streams, hidden = expanded.shape
        if rows <= 0 or streams <= 1 or hidden <= 0:
            raise ValueError("V4 auxiliary mean requires positive rows/hidden and multiple streams")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * streams * hidden,
            "rows": rows, "streams": streams, "hidden": hidden}, "bf16", "v4_aux_hidden_mean",
            {"operation": "torch_mean", "reduction_axis": 1, "output_elements": rows * hidden,
             "source_component": "vllm_0_30_v4_target_aux_mhc_mean", "source_projection_only": True})
        return self.trace.call(node, (expanded,), shape=(rows, hidden), dtype="bf16")


class SymbolicMHC(SymbolicOperator):
    """vLLM 0.30 TileLang mHC wrappers, including source-owned FP32 scratch.

    Reuses the existing source split derivation; it does not infer installed
    DeepGEMM support, library workspace, asynchronous retirement or compilation.
    Model/decoder forwards retain and pass the four outputs normally.
    """

    def __init__(self, trace, name, pool_name, model, hardware, *, variant, source_use_deep_gemm,
            workspace_scope="source_launch"):
        super().__init__(trace, name, pool_name)
        if (model.model_type != "deepseek_v4" or model.compute_dtype != "bf16"
                or variant not in ("mhc_pre", "mhc_post_pre", "mhc_post", "hc_head")):
            raise NotImplementedError("symbolic mHC requires the BF16 V4 TileLang source wrappers")
        if workspace_scope not in ("source_launch", "logical_math_only"):
            raise ValueError("mHC requires a declared source or mathematical workspace scope")
        if workspace_scope == "logical_math_only":
            if hardware is not None or source_use_deep_gemm is not None:
                raise ValueError("mathematical mHC cannot claim a native hardware/backend binding")
        elif type(source_use_deep_gemm) is not bool:
            raise TypeError("symbolic mHC requires an explicit source DeepGEMM branch")
        self.workspace_scope = workspace_scope
        self.model, self.hardware = model, hardware
        self.variant, self.source_use_deep_gemm = variant, source_use_deep_gemm

    def __call__(self, x, residual=None, post_mix=None, res_mix=None):
        h, c = self.model.hidden_size, self.model.hc_mult
        rows = x.shape[0]
        self.trace._check_tensor(x)
        if x.dtype != "bf16" or not x.is_contiguous:
            raise ValueError("mHC source requires contiguous BF16 input")
        state = (residual, post_mix, res_mix)
        if self.variant in ("mhc_pre", "hc_head"):
            if any(tensor is not None for tensor in state):
                raise ValueError("broadcast/head mHC does not consume previous sublayer state")
            expected = (rows, h) if self.variant == "mhc_pre" else (rows, c, h)
            inputs = (x,)
        else:
            if any(tensor is None for tensor in state):
                raise ValueError("mHC post/pre requires residual and both FP32 mix tensors")
            for tensor, shape, dtype in zip(state,
                    ((rows, c, h), (rows, c, 1), (rows, c, c)), ("bf16", "fp32", "fp32"), strict=True):
                self.trace._check_tensor(tensor)
                if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous:
                    raise ValueError("mHC carried state disagrees with source shape/dtype")
            expected, inputs = (rows, h), (x, *state)
        if x.shape != expected:
            raise ValueError("mHC source input shape disagrees with wrapper")
        shape = {"elements": rows * c * h, "hidden": h, "hc_mult": c,
            "dtype_bytes": x.view.element_bytes}
        attributes = {"source_component": "vllm_0_30_symbolic_mhc",
            "sinkhorn_iters": self.model.hc_sinkhorn_iters,
            "source_use_deep_gemm": self.source_use_deep_gemm}
        node = self.node(WorkloadKind.VECTOR, shape, x.dtype, self.variant, attributes)
        outputs = (("residual", (rows, c, h), "bf16"),
            ("post_mix", (rows, c, 1), "fp32"), ("res_mix", (rows, c, c), "fp32"),
            ("layer_input", (rows, h), "bf16"))
        if self.variant in ("mhc_post", "hc_head"):
            output_shape = (rows, c, h) if self.variant == "mhc_post" else (rows, h)
            return self.trace.call(node, inputs, shape=output_shape, dtype=x.dtype)
        if self.workspace_scope == "logical_math_only":
            node = node.with_attributes({**node.attributes,
                "mathematical_program": "vllm_mhc_wrapper_math_v1",
                "symbolic_workspace_complete": False, "source_scratch_shape_bound": False})
            return self.trace.call_outputs(node, inputs, outputs=outputs)
        # tilelang.py creates these wrapper outputs before calling prenorm.
        # Fused post/prenorm returns the same scratch layout as the unfused
        # source helper. The existing selector owns their exact split count.
        stages = select_mhc_stage_launches("nvidia_h200", {"element_count": shape["elements"],
            **{key: value for key, value in shape.items() if key != "elements"}},
            x.dtype, self.variant, attributes, self.hardware)
        splits = stages[-1][1]["n_splits"]
        return self.trace.call_outputs(node, inputs, outputs=outputs, workspace=(
            ("prenorm_mixes", (splits, rows, c * (c + 2)), "fp32"),
            ("prenorm_sqrsum", (splits, rows), "fp32")))


class SymbolicMHCPipelineInput:
    """V4 nonbroadcast stage-entry mhc_pre_tilelang, optional fallback branch.

    The expanded input itself is the residual; only mixes and layer input are
    fresh. The explicit env-disabled TileLang source uses one partial. GPU
    provider selection remains unbound, as for the existing prenorm stages.
    """

    def __init__(self, trace, name, pool, model, *, source_use_deep_gemm, source_context="pipeline"):
        if source_context not in ("pipeline", "dspark") or source_context == "dspark" and not model.is_dspark_draft:
            raise ValueError("nonbroadcast mHC requires its actual pipeline or DSpark source context")
        self.source_context = source_context
        if model.model_type != "deepseek_v4" or source_use_deep_gemm is not False:
            raise NotImplementedError("V4 PP entry currently requires its declared optional TileLang prenorm branch")
        self.trace, self.model = trace, model
        self.prenorm = SymbolicOperator(trace, name + ":prenorm", pool)
        self.normalize = SymbolicOperator(trace, name + ":normalize", pool)

    def __call__(self, x):
        self.trace._check_tensor(x)
        rows, c, h = x.shape[0], self.model.hc_mult, self.model.hidden_size
        if x.shape != (rows, c, h) or x.dtype != "bf16" or not x.is_contiguous:
            raise ValueError("V4 PP prenorm requires its contiguous expanded BF16 hidden tensor")
        shape = {"elements": rows * c * h, "hidden": h, "hc_mult": c, "dtype_bytes": 2, "n_splits": 1}
        attrs = {"source_component": "vllm_0_30_symbolic_v4_" + self.source_context + "_prenorm",
            "source_use_deep_gemm": False}
        projection = self.prenorm.node(WorkloadKind.VECTOR, shape, "bf16", "mhc_prenorm", attrs)
        mixes, squared = self.trace.call_outputs(projection, (x,), outputs=(
            ("prenorm_mixes", (1, rows, c * (c + 2)), "fp32"),
            ("prenorm_sqrsum", (1, rows), "fp32")))
        normalize = self.normalize.node(WorkloadKind.VECTOR,
            {**shape, "sinkhorn_iters": self.model.hc_sinkhorn_iters}, "bf16", "mhc_pre_norm", attrs)
        post_mix, res_mix, layer_input = self.trace.call_outputs(normalize, (x, mixes, squared), outputs=(
            ("post_mix", (rows, c, 1), "fp32"), ("res_mix", (rows, c, c), "fp32"),
            ("layer_input", (rows, h), "bf16")))
        return x, post_mix, res_mix, layer_input


class SymbolicGather(SymbolicOperator):
    def __call__(self, x, rows, *, indices=None, source_operation=None, indices_tensor=None):
        if source_operation is not None and (source_operation != "cat_last_dim" or indices is not None):
            raise ValueError("source cat materialization requires an unindexed copy")
        if indices is not None and (len(indices) != rows or any(type(i) is not int or not 0 <= i < x.shape[0] for i in indices)):
            raise ValueError("symbolic gather row indices are invalid")
        if indices_tensor is not None:
            self.trace._check_tensor(indices_tensor)
            if indices is None or indices_tensor.shape != (rows,) or indices_tensor.dtype != "int64":
                raise ValueError("symbolic gather requires an actual INT64 index port and its declared row projection")
        shape = x.shape if indices is None else (rows, *x.shape[1:])
        width = prod(shape[1:])
        node = self.node(WorkloadKind.GATHER,
            {"elements": prod(shape), "rows": rows, "hidden": width}, x.dtype, "gather",
            {"gather_factor": 1, "index_count": 0 if indices is None else rows,
             "index_bytes": 8, "scattered": indices is not None, "materialized_output": True,
             **({"source_operation": source_operation} if source_operation is not None else {})})
        return self.trace.call(node, (x,) if indices_tensor is None else (x, indices_tensor), shape=shape, dtype=x.dtype)


class SymbolicRMSNorm(SymbolicOperator):
    """Source RMSNorm with an explicit CUDA or Ascend storage program.

    New CUDA wrappers accept unit-final-stride rows; the vLLM0.11 Python
    wrapper first materializes every noncontiguous input. Output allocation
    precedes either copy. Fused residual updates are in-place.
    """
    def __init__(self, trace, name, pool_name, width, epsilon, *, storage_program="vllm_c_cuda",
                 model_execution_version=None):
        super().__init__(trace, name, pool_name)
        if epsilon is None or epsilon <= 0:
            raise ValueError("symbolic RMSNorm requires checkpoint epsilon")
        if storage_program not in ("vllm_c_cuda", "vllm_c_cuda_0_11", "vllm_ascend_cann"):
            raise ValueError("symbolic RMSNorm requires a supported source storage program")
        self.width, self.epsilon = width, epsilon
        self.storage_program = storage_program
        if model_execution_version is not None:
            if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1", "vllm_0_30_0"):
                raise ValueError("symbolic RMSNorm requires its declared model source version")
            if storage_program == "vllm_c_cuda_0_11" and model_execution_version != "vllm_0_11_0":
                raise ValueError("symbolic RMSNorm source version disagrees with its storage program")
        self.model_execution_version = model_execution_version
        self.materialize = SymbolicGather(trace, name + ":input-contiguous", pool_name)

    def __call__(self, x, residual=None):
        self.trace._check_tensor(x)
        if len(x.shape) not in (2, 3, 4):
            raise NotImplementedError("symbolic RMSNorm source supports rank2/3/4 inputs")
        if x.shape[-1] != self.width:
            raise ValueError("symbolic RMSNorm normalized width disagrees with input")
        node = self.node(WorkloadKind.NORM, {"rows": prod(x.shape[:-1]), "hidden": self.width},
            x.dtype, "rmsnorm" if residual is None else "add_rmsnorm", {"epsilon": self.epsilon,
                "source_storage_profile": "vllm_0_30_vllm_c_cuda" if (self.storage_program == "vllm_c_cuda"
                    and self.model_execution_version in (None, "vllm_0_30_0"))
                    else self.storage_program,
                **({"model_execution_version": self.model_execution_version}
                    if self.model_execution_version is not None else {})})
        if residual is not None:
            self.trace._check_tensor(residual)
            if residual.shape != x.shape or residual.dtype != x.dtype:
                raise ValueError("fused RMSNorm residual shape/dtype must match input")
        weight = self.trace.parameter_view(self.name + ":weight", (self.width,), x.dtype)
        if self.storage_program == "vllm_ascend_cann":
            # AscendRMSNorm.forward_oot returns fresh npu_{add_,}rms_norm
            # outputs. CANN additionally writes one FP32 rstd per row. Keep
            # that returned local alive until this helper returns, including
            # the discarded `_` of the residual branch. aclnn materializes
            # non-dense inputs; these buffers belong to the source call.
            contiguous = self.materialize(x, x.shape[0]) if not x.is_contiguous else x
            inputs = (contiguous,)
            outputs = (("normalized", x.shape, x.dtype), ("rstd", (*x.shape[:-1], 1), "fp32"))
            if residual is not None:
                dense_residual = self.materialize(residual, residual.shape[0]) if not residual.is_contiguous else residual
                inputs = (contiguous, dense_residual)
                outputs += (("residual", residual.shape, residual.dtype),)
            results = self.trace.call_outputs(node, inputs, outputs=outputs, parameters=(weight,))
            return results[0] if residual is None else (results[0], results[2])
        if residual is not None:
            # The symbolic view proves strides, not allocator alignment or
            # the runtime's batch-invariant branch. Keep those facts explicit.
            if x.is_contiguous and residual.is_contiguous:
                node = node.with_attributes({**node.attributes, "source_input_layout": "contiguous"})
            self.trace.call(node, (x, residual), shape=x.shape, dtype=x.dtype, out=x, writes=(residual,), parameters=(weight,))
            return x, residual
        # kernels/vllm_c.py's flatten/reshape branch is ROCm-only. CUDA's
        # csrc/libtorch_stable/layernorm_kernels.cu supports strided leading
        # axes directly, including the grouped views of packed QKV output.
        # Only a non-unit final stride triggers its internal contiguous copy.
        output = self.trace.tensor(node.node_id, x.shape, x.dtype)
        if self.storage_program == "vllm_c_cuda_0_11" and not x.is_contiguous:
            # vLLM0.11 _custom_ops.rms_norm calls input.contiguous() before
            # the C++ op, unlike the newer unit-final-stride CUDA wrapper.
            if (len(x.shape) not in (2, 3) or x.view.strides_bytes[-1] != x.view.element_bytes
                    or (len(x.shape) == 3 and x.view.strides_bytes[-2] != self.width * x.view.element_bytes)):
                raise NotImplementedError("vLLM0.11 RMSNorm copy requires its grouped row-strided input ABI")
            groups, rows = x.shape[0], 1 if len(x.shape) == 2 else x.shape[1]
            copy = self.materialize.node(WorkloadKind.ELEMENTWISE,
                {"elements": prod(x.shape)}, x.dtype, "elementwise",
                {"reads": 1, "writes": 1, "operation": "copy",
                 "model_execution_version": "vllm_0_11_0",
                 "input_materialization": "grouped_strided_to_contiguous",
                 "input_outer_groups": groups, "input_rows_per_group": rows,
                 "input_group_stride_elements": x.view.strides_bytes[0] // x.view.element_bytes,
                 "input_row_stride_elements": self.width})
            contiguous = self.trace.call(copy, (x,), shape=x.shape, dtype=x.dtype)
        else:
            contiguous = self.materialize(x, x.shape[0]) if x.view.strides_bytes[-1] != x.view.element_bytes else x
        node = node.with_attributes({**node.attributes, "source_input_layout": (
            "contiguous" if self.storage_program == "vllm_c_cuda_0_11" else "unit_final_stride")})
        return self.trace.call(node, (contiguous,), shape=x.shape, dtype=x.dtype, out=output, parameters=(weight,))


class SymbolicAllReduceRMSNorm:
    """Ordinary source branch of fused_allreduce_rms_norm, with its scope.

    The reduced local lives in this helper frame, not across subsequent
    attention/MLP calls in the decoder. The optional fused provider is not
    selected or emulated by this source composition.
    """

    def __init__(self, norm, group, config, *, name, pool_name, moe_invocation_id=None):
        if (not isinstance(norm, SymbolicRMSNorm) or not isinstance(group, ForwardCollectiveGroup)
                or not isinstance(config, CollectiveConfig) or config.operation != "all_reduce"
                or not any(norm.trace is trace for trace in group.traces)):
            raise ValueError("source all-reduce/norm requires this rank's norm and canonical all-reduce group")
        self.norm, self.group, self.config, self.name, self.pool_name = norm, group, config, name, pool_name
        self.moe_invocation_id = moe_invocation_id

    def __call__(self, hidden_states, residual):
        reduced = self.group.call(self.name, hidden_states,
            pool_name=self.pool_name, config=self.config, inplace=False,
            moe_invocation_id=self.moe_invocation_id)
        return self.norm(reduced, residual)


class SymbolicEmbedding(SymbolicOperator):
    def __init__(self, trace, name, pool_name, width, dtype, *, model_execution_version="vllm_0_30_0",
                 model_input_order=None, vocab_size=None, source_rows=None, copy_dispatch=None,
                 maca_copy_runtime_source=None):
        super().__init__(trace, name, pool_name)
        self.width, self.dtype = width, dtype
        if model_execution_version not in ("vllm_0_30_0", "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1"):
            raise ValueError("symbolic embedding requires its declared source version")
        self.source = {"vllm_0_11_0": "vllm_0_11", "vllm_0_18_0": "vllm_0_18", "vllm_0_24_0": "vllm_0_24",
            "vllm_0_27_1": "vllm_0_27_1", "vllm_0_30_0": "vllm_0_30"}[model_execution_version]
        self.model_input_order = model_input_order
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= 0):
            raise ValueError("embedding vocabulary size must be a positive integer")
        if source_rows is not None and (type(source_rows) is not int or source_rows <= 0
                or vocab_size is not None or self.source not in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1")):
            raise ValueError("explicit embedding table rows require a positive0.11/0.24/0.27.1 physical extent, separate from global vocabulary")
        self.vocab_size = vocab_size
        # A TP shard is divided AFTER global padding. Re-padding a local
        # shard (e.g.192/2=96) would invent table rows and change addressing.
        self.source_rows = (source_rows if source_rows is not None else
            -(-vocab_size // 64) * 64 if vocab_size is not None else None)
        self.sharded_table = source_rows is not None
        if copy_dispatch is not None and (self.source != "vllm_0_24" or self.sharded_table):
            raise ValueError("embedding Copy dispatch requires ordinary0.24 INT32 source IDs")
        self.copy_dispatch = copy_dispatch
        if maca_copy_runtime_source is not None:
            from .runtime_inputs import MACA_COPY_RUNTIME_SOURCE
            if maca_copy_runtime_source != MACA_COPY_RUNTIME_SOURCE:
                raise ValueError("unknown MACA native Copy runtime source")
        if maca_copy_runtime_source is not None and (
                self.source != "vllm_0_24" or self.sharded_table or copy_dispatch is not None):
            raise ValueError("MACA runtime Copy requires an ordinary0.24 embedding without separate dispatch facts")
        self.maca_copy_runtime_source = maca_copy_runtime_source

    def __call__(self, ids):
        rows = prod(ids.shape)
        if ids.dtype == "int32":
            if self.sharded_table:
                raise ValueError("masked shard embedding requires its already-INT64 mask output")
            dispatch = (self.copy_dispatch if self.maca_copy_runtime_source is None else
                _maca_native_copy_dispatch(ids, self.maca_copy_runtime_source))
            cast = self.node(WorkloadKind.VECTOR, {"rows": rows, "elements": rows}, "int32",
                "vllm_embedding_id_widen", {"source_component": self.source + "_symbolic_embedding",
                    "embedding_source": self.source + "_unquantized_tp1",
                    **({"copy_dispatch": dispatch} if dispatch is not None else {})})
            ids = self.trace.call(cast, (ids,), shape=ids.shape, dtype="int64")
        elif ids.dtype != "int64":
            raise ValueError("ordinary vLLM embedding requires INT32/INT64 source IDs")
        elif self.copy_dispatch is not None:
            raise ValueError("embedding Copy dispatch cannot describe an INT64 alias")
        node = self.node(WorkloadKind.GATHER,
            {"elements": rows * self.width, "rows": rows, "hidden": self.width,
             **({"source_rows": self.source_rows}
                if (self.source in ("vllm_0_11", "vllm_0_24") or self.source == "vllm_0_27_1" and self.sharded_table
                    or self.source == "vllm_0_30" and self.dtype in ("bf16", "fp16") and len(ids.shape) == 1)
                and self.source_rows is not None else {})}, self.dtype, "embedding",
            {"gather_factor": 1, "index_count": rows, "index_bytes": ids.view.element_bytes,
             "scattered": True, "materialized_output": True,
             **({"index_select_source": "vllm_0_11_embedding"}
                if self.source == "vllm_0_11" and self.source_rows is not None else {}),
             **({"embedding_source": "vllm_0_24_unquantized_index_select",
                 "source_operation": "index_select_dim0"}
                if self.source == "vllm_0_24" else {}),
             **({"embedding_source": "vllm_0_30_unquantized_tp1",
                 "model_execution_version": "vllm_0_30_0", "source_operation": "embedding"}
                if self.source == "vllm_0_30" and self.dtype in ("bf16", "fp16")
                and len(ids.shape) == 1 and self.source_rows is not None else {}),
             **({"source_component": "vllm_0_27_1_symbolic_masked_embedding_gather",
                 "model_execution_version": "vllm_0_27_1",
                 "embedding_source": "vllm_0_27_1_masked_index_select",
                 "source_operation": "index_select_dim0"}
                if self.source == "vllm_0_27_1" and self.sharded_table else {}),
             **({"symbolic_model_input_order": self.model_input_order} if self.model_input_order is not None else {})})
        return self.trace.call(node, (ids,), shape=(rows, self.width), dtype=self.dtype)


class SymbolicVocabParallelEmbedding(SymbolicOperator):
    """Versioned unquantized TP embedding and out-of-place all-reduce.

    Explicit scheduled IDs determine CUDA fused-branch shard ownership.
    The unquantized 16-bit/no-added-vocabulary branch reads the original INT32
    IDs and writes all partial rows, including zeros on non-owner ranks.
    vLLM0.11 CUDA and non-CUDA sources preserve mask/gather/fill stages.
    The TP2 public mask/gather/fill program processes every input row;
    its logical argument payloads permit unknown IDs, without binding cache
    traffic, compiled instructions or value-dependent conditional stores.
    """

    def __init__(self, trace, name, pool_name, width, vocab, group, config, *, token_ids, model_input_order=None,
                 fused=True, mask_program="vllm_0_30", dtype="bf16", model_execution_version=None,
                 reduction_inplace=False, input_values_source=None):
        super().__init__(trace, name, pool_name)
        if not isinstance(group, ForwardCollectiveGroup) or not isinstance(config, CollectiveConfig):
            raise TypeError("TP embedding requires a virtual collective group and canonical config")
        if config.operation != "all_reduce" or not any(trace is item for item in group.traces):
            raise ValueError("TP embedding requires this rank's all-reduce group")
        if (type(vocab) is not int or vocab <= 0 or (-(-vocab // 64) * 64) % len(group.traces)
                or type(width) is not int or width <= 0):
            raise ValueError("TP embedding requires positive width and evenly padded vocabulary shards")
        ascend018 = model_execution_version == "vllm_0_18_0"
        if ascend018 and (mask_program != "vllm_ascend_no_added_vocab" or dtype != "bf16" or fused):
            raise ValueError("Ascend0.18 TP embedding requires its ordinary BF16 masked source")
        if type(reduction_inplace) is not bool:
            raise TypeError("embedding reduction alias requires an explicit boolean")
        if mask_program == "vllm_0_27_1" and (
                model_execution_version != "vllm_0_27_1" or fused
                or dtype != "bf16" or len(group.traces) not in (4, 8)):
            raise ValueError("vLLM0.27.1 TP embedding requires its BF16 TP4/8 masked source")
        self.ascend018, self.reduction_inplace = ascend018, reduction_inplace
        unknown_masked_ids = (token_ids is None and fused is False and (mask_program in ("vllm_0_11", "vllm_0_24", "vllm_0_30") or ascend018)
            and len(group.traces) == 2)
        if input_values_source not in (None, "legacy_dummy_retained_buffer") or (
                input_values_source is not None and (token_ids is not None or not fused or mask_program != "vllm_0_30")):
            raise ValueError("unknown fused embedding IDs require the explicit retained legacy dummy buffer source")
        unknown_dummy_ids = input_values_source == "legacy_dummy_retained_buffer"
        if not (unknown_masked_ids or unknown_dummy_ids) and (not isinstance(token_ids, tuple) or not token_ids
                or any(type(token) is not int or not 0 <= token < vocab for token in token_ids)):
            raise ValueError("TP embedding requires explicit valid scheduled INT32 token IDs")
        if dtype not in ("bf16", "fp16"):
            raise ValueError("unquantized TP embedding requires BF16 or FP16 weights")
        self.width, self.vocab, self.token_ids, self.dtype = width, vocab, token_ids, dtype
        self.shared_weight = None
        self.group, self.config = group, config
        self.model_input_order = model_input_order
        self.input_values_source = input_values_source
        if type(fused) is not bool:
            raise TypeError("TP embedding source branch must be an explicit boolean")
        self.fused = fused
        if mask_program not in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1", "vllm_0_30", "vllm_ascend_no_added_vocab"):
            raise ValueError("TP embedding requires its supported source mask program")
        if mask_program in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1", "vllm_ascend_no_added_vocab") and fused:
            raise ValueError("this masked embedding source cannot select the CUDA fused branch")
        self.mask_program = mask_program

    def _masked_embedding(self, ids):
        """Native mask/gather branch (including CUDA in vLLM0.11).

        The compiled mask helper promotes bool*Python-int offsets to INT64,
        so masked_input.long() aliases its already-INT64 result. This describes
        public helper outputs, not compiler temporaries or an actual provider.
        """
        rows, tp = ids.shape[0], len(self.group.traces)
        logical_rank = self.group.participants.index(self.trace.rank)
        attributes = {"source_component": ("vllm_0_18_symbolic_masked_embedding" if self.ascend018
            else "vllm_0_11_symbolic_masked_embedding" if self.mask_program == "vllm_0_11"
            else "vllm_0_24_symbolic_masked_embedding" if self.mask_program == "vllm_0_24"
            else "vllm_0_27_1_symbolic_masked_embedding" if self.mask_program == "vllm_0_27_1"
            else "vllm_0_30_symbolic_masked_embedding"),
            "vocab_size": self.vocab, "tensor_parallel_size": tp, "tensor_parallel_rank": logical_rank,
            "mask_program": self.mask_program,
            **({"model_execution_version": "vllm_0_27_1"} if self.mask_program == "vllm_0_27_1" else {})}
        mask = SymbolicOperator(self.trace, self.name + ":mask-ids", self.pool_name)
        node = mask.node(WorkloadKind.VECTOR, {"rows": rows, "hidden": self.width, "elements": rows},
            "int32", "vllm_masked_embedding", {**attributes, "stage": "mask_ids"})
        masked_ids, input_mask = self.trace.call_outputs(node, (ids,), outputs=(
            ("masked_ids", (rows,), "int64"), ("input_mask", (rows,), "bool")))
        # No extra widening/copy: UnquantizedEmbeddingMethod receives INT64.
        output_parallel = SymbolicEmbedding(self.trace, self.name + ":gather", self.pool_name,
            self.width, self.dtype, model_execution_version=("vllm_0_18_0" if self.ascend018
                else "vllm_0_11_0" if self.mask_program == "vllm_0_11"
                else "vllm_0_24_0" if self.mask_program == "vllm_0_24"
                else "vllm_0_27_1" if self.mask_program == "vllm_0_27_1"
                else "vllm_0_30_0"),
            source_rows=(-(-self.vocab // 64) * 64 // tp
                if self.mask_program in ("vllm_0_11", "vllm_0_24", "vllm_0_27_1") else None))(masked_ids)
        fill = SymbolicOperator(self.trace, self.name + ":mask-output", self.pool_name)
        node = fill.node(WorkloadKind.VECTOR,
            {"rows": rows, "hidden": self.width, "elements": rows * self.width}, self.dtype,
            "vllm_masked_embedding", {**attributes, "stage": "mask_output"})
        return self.trace.call(node, (output_parallel, input_mask.unsqueeze(-1)),
            shape=output_parallel.shape, dtype=self.dtype, out=output_parallel)

    def __call__(self, ids):
        self.trace._check_tensor(ids)
        if len(ids.shape) != 1 or ids.dtype != "int32" or not ids.is_contiguous:
            raise NotImplementedError("TP embedding source requires contiguous one-dimensional INT32 input")
        rows = ids.shape[0]
        if self.token_ids is not None and len(self.token_ids) != rows:
            raise ValueError("TP embedding token IDs must cover every source input row")
        if not self.fused:
            output_parallel = self._masked_embedding(ids)
            return self.group.call(self.name + ":all-reduce", output_parallel,
                pool_name=self.pool_name, config=self.config, inplace=self.reduction_inplace)
        node = self.node(WorkloadKind.GATHER,
            {"elements": rows * self.width, "rows": rows, "hidden": self.width}, self.dtype, "embedding", {
                "source_component": "vllm_0_30_symbolic_vocab_embedding",
                "embedding_source": "vllm_0_30_cuda_unquantized_tp",
                "tensor_parallel_size": len(self.group.traces), "vocab_size": self.vocab,
                "embedding_token_ids": self.token_ids, "index_bytes": 4,
                **({"embedding_input_values_source": self.input_values_source,
                    "value_dependent_shard_traffic_bound": False} if self.input_values_source is not None else {}),
                "index_count": rows, "gather_factor": 1, "scattered": True,
                "materialized_output": True,
                **({"symbolic_model_input_order": self.model_input_order} if self.model_input_order is not None else {})})
        output_parallel = self.trace.call(node, (ids,), shape=(rows, self.width), dtype=self.dtype,
            parameters=() if self.shared_weight is None else (self.shared_weight,))
        # GroupCoordinator.all_reduce explicitly returns an out-of-place result
        # for TP>1. Internal communicator buffers remain a separate boundary.
        return self.group.call(self.name + ":all-reduce", output_parallel,
            pool_name=self.pool_name, config=self.config, inplace=False)


class SymbolicRotary(SymbolicOperator):
    def __init__(self, trace, name, pool_name, heads, kv_heads, head_dim, cache, *, program=ROPE_PROGRAM_FUSED,
                 operand_semantics="aten_tensor_iterator", rotary_dim=None, model_execution_version=None):
        super().__init__(trace, name, pool_name)
        self.heads, self.kv_heads, self.head_dim, self.cache = heads, kv_heads, head_dim, cache
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if (type(self.rotary_dim) is not int or not 0 < self.rotary_dim <= head_dim
                or self.rotary_dim % 2):
            raise ValueError("symbolic rotary width must be positive, even and no wider than its head")
        if len(cache.shape) != 2 or cache.shape[1] != self.rotary_dim:
            raise ValueError("symbolic rotary cache width must match its active rotation width")
        if program not in (ROPE_PROGRAM_FUSED, ROPE_PROGRAM_NATIVE_COMPOSITE):
            raise ValueError("symbolic rotary requires its declared source program")
        if operand_semantics not in ("aten_tensor_iterator", "aclnn_materialized_operands"):
            raise ValueError("symbolic rotary requires supported source operand semantics")
        self.program = program
        if model_execution_version is not None and model_execution_version not in (
                "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1", "vllm_0_30_0"):
            raise ValueError("symbolic rotary requires its declared model source version")
        self.model_execution_version = model_execution_version
        self.operand_semantics = operand_semantics
        self.materialize = SymbolicGather(trace, name + ":input-contiguous", pool_name)

    def _binary(self, left, right, operation):
        self.trace._check_tensor(left)
        self.trace._check_tensor(right)
        if (len(left.shape) != 3 or right.shape not in (left.shape, (left.shape[0], 1, left.shape[-1]))
                or left.dtype != right.dtype):
            raise ValueError("native RoPE arithmetic requires equal dtype and explicit head broadcast")
        rows, heads, width = left.shape
        if operation not in ("mul", "add", "sub"):
            raise ValueError("unknown native RoPE arithmetic")
        if self.operand_semantics == "aclnn_materialized_operands":
            left = self.materialize(left, rows) if not left.is_contiguous else left
            right = self.materialize(right, rows) if not right.is_contiguous else right
        node = SymbolicOperator(self.trace, self.name + ":" + operation, self.pool_name).node(
            WorkloadKind.ELEMENTWISE, {"elements": prod(left.shape)}, left.dtype, "elementwise", {
                "operation": "add" if operation == "sub" else operation, "reads": 2, "writes": 1,
                "source_operation": operation, "source_component": "vllm_symbolic_native_rotary",
                "iteration_tokens": rows, "iteration_rows": heads, "iteration_inner": width,
                **({"operand_layout": "contiguous" if self.operand_semantics == "aclnn_materialized_operands"
                    else "dim0_contiguous_dim1_broadcast"} if operation == "mul" else {})})
        return self.trace.call(node, (left, right), shape=left.shape, dtype=left.dtype)

    def _apply_native(self, x, cos, sin):
        # ApplyRotaryEmb.forward_static, BF16/neox active prefix.
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        x1, x2 = x.split((self.rotary_dim // 2, self.rotary_dim // 2))
        o1 = self._binary(self._binary(x1, cos, "mul"), self._binary(x2, sin, "mul"), "sub")
        o2 = self._binary(self._binary(x2, cos, "mul"), self._binary(x1, sin, "mul"), "add")
        return SymbolicConcat(self.trace, self.name + ":rotate-concat", self.pool_name)(o1, o2)

    def _forward_native(self, positions, query, key):
        # Preserve forward_static's locals across both Q and K. Its empty
        # full-width passthrough views have no payload; aten::cat still has
        # a fresh output contract, so retain the outer copies explicitly.
        for tensor in (positions, query, key, self.cache):
            self.trace._check_tensor(tensor)
        rows = positions.shape[0]
        if (positions.shape != (rows,) or positions.dtype != "int64"
                or query.shape != (rows, self.heads * self.head_dim)
                or key.shape != (rows, self.kv_heads * self.head_dim)
                or query.dtype != "bf16" or key.dtype != "bf16" or self.cache.dtype != "bf16"
                or len(self.cache.shape) != 2 or self.cache.shape[1] != self.rotary_dim):
            raise NotImplementedError("native RoPE requires BF16 Q/K/active-width cache and INT64 positions")
        gather_shape = {"elements": rows * self.rotary_dim, "rows": rows, "hidden": self.rotary_dim}
        if self.cache.is_contiguous and positions.is_contiguous:
            # This is the actual cache extent, not the number of selected
            # rows.  Together with the existing contiguous index-select
            # contract it permits a source-derived addressing-domain check.
            gather_shape["source_rows"] = self.cache.shape[0]
        node = SymbolicOperator(self.trace, self.name + ":cos-sin", self.pool_name).node(
            WorkloadKind.GATHER, gather_shape,
            self.cache.dtype, "gather", {"gather_factor": 1, "index_count": rows, "index_bytes": 8,
                "scattered": True, "materialized_output": True, "source_component": "vllm_symbolic_native_rotary"})
        cos_sin = self.trace.call(node, (self.cache, positions), shape=(rows, self.rotary_dim), dtype=self.cache.dtype)
        cos, sin = cos_sin.split((self.rotary_dim // 2, self.rotary_dim // 2))
        query_shape = query.shape
        query = query.view_as_shape(rows, self.heads, self.head_dim)
        query_rot = self._apply_native(query.narrow(-1, 0, self.rotary_dim), cos, sin)
        query = (SymbolicGather(self.trace, self.name + ":query-empty-suffix-cat", self.pool_name)(
            query_rot, rows, source_operation="cat_last_dim")
            if self.rotary_dim == self.head_dim else
            SymbolicConcat(self.trace, self.name + ":query-suffix-cat", self.pool_name)(
                query_rot, query.narrow(-1, self.rotary_dim, self.head_dim - self.rotary_dim)))
        query = query.view_as_shape(query_shape)
        key_shape = key.shape
        key = key.view_as_shape(rows, self.kv_heads, self.head_dim)
        key_rot = self._apply_native(key.narrow(-1, 0, self.rotary_dim), cos, sin)
        key = (SymbolicGather(self.trace, self.name + ":key-empty-suffix-cat", self.pool_name)(
            key_rot, rows, source_operation="cat_last_dim")
            if self.rotary_dim == self.head_dim else
            SymbolicConcat(self.trace, self.name + ":key-suffix-cat", self.pool_name)(
                key_rot, key.narrow(-1, self.rotary_dim, self.head_dim - self.rotary_dim)))
        key = key.view_as_shape(key_shape)
        return query, key

    def __call__(self, positions, query, key):
        if self.program == ROPE_PROGRAM_NATIVE_COMPOSITE:
            start = len(self.trace._nodes)
            result = self._forward_native(positions, query, key)
            # The capture hooks this module once; preserve every internal
            # operator for lowering while retaining its invocation boundary.
            for position, identifier in enumerate(tuple(self.trace._nodes)[start:]):
                node = self.trace._nodes[identifier]
                self.trace._nodes[identifier] = node.with_attributes({**node.attributes,
                    "rope_composite_position": position, "rope_composite_tokens": positions.shape[0],
                    "rope_composite_full_width": self.rotary_dim == self.head_dim})
            return result
        if self.operand_semantics == "aclnn_materialized_operands":
            # Both full-head Ascend wrappers (Triton and torch_npu) make Q/K
            # contiguous, then update those buffers in place. This storage
            # fact does not assert either provider's launch was selected.
            query = self.materialize(query, query.shape[0]) if not query.is_contiguous else query
            key = self.materialize(key, key.shape[0]) if not key.is_contiguous else key
        rows = query.shape[0]
        layout = ({"source_input_layout": "contiguous"} if query.is_contiguous and key.is_contiguous
            else {"source_input_layout": "unit_final_stride"}
                if all(tensor.view.strides_bytes[-1] == tensor.view.element_bytes for tensor in (query, key)) else {})
        if (positions.shape != (rows,) or positions.dtype != "int64"
                or not positions.is_contiguous or not self.cache.is_contiguous):
            # The CUDA scalar helper indexes these two ports densely. Q/K
            # row/head gaps do not prove the positions/cache ABI.
            layout = {}
        node = self.node(WorkloadKind.ROTARY,
            {"elements": rows * (self.heads + self.kv_heads) * self.rotary_dim,
             "seq_len": rows, "num_query_heads": self.heads, "num_kv_heads": self.kv_heads,
             "head_dim": self.head_dim, **({"rotary_dim": self.rotary_dim}
                 if self.rotary_dim != self.head_dim else {})}, query.dtype,
            "rope" if self.rotary_dim != self.head_dim else "rotary",
            {**({"key_present": True, "is_neox": True} if self.rotary_dim != self.head_dim else {}),
             "source_rope_program": "vllm_fused_rotary",
             **({"model_execution_version": self.model_execution_version}
                 if self.model_execution_version is not None else {}),
             **layout})
        self.trace.call(node, (positions, query, key, self.cache), shape=query.shape,
            dtype=query.dtype, out=query, writes=(key,))
        return query, key


def gqa_cache_shape(capacity_tokens, kv_heads, head_dim):
    """Pure persistent GQA layout shared by actual constructors and binders."""
    if any(type(value) is not int or value <= 0 for value in (capacity_tokens, kv_heads, head_dim)):
        raise ValueError("GQA cache geometry requires positive integer extents")
    return (2, capacity_tokens, kv_heads, head_dim)


def gqa_full_source_metadata(q_lens, kv_lens, slot_mapping, token_shape):
    """Pure CUDA FULL attention metadata, not an operator or a model graph."""
    if token_shape.runtime_mode != "FULL" or token_shape.logical_tokens != sum(q_lens):
        raise ValueError("GQA metadata requires its declared FULL token domain")
    shape = token_shape.attention_shape(q_lens, kv_lens, uniform_decode_query_len=1)
    return {"attention_shape": shape, "kv_len": max(kv_lens),
        "kv_lens": kv_lens, "slot_mapping": slot_mapping}


class SymbolicAttention(SymbolicOperator):
    def __init__(self, trace, name, pool_name, heads, kv_heads, head_dim, cache, q_lens, kv_lens, slot_mapping,
                 *, profile_no_metadata=False, model_execution_version="vllm_0_30_0", legacy_dummy_run=False):
        super().__init__(trace, name, pool_name)
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise ValueError("attention source requires a supported model execution version")
        self.model_execution_version = model_execution_version
        if type(profile_no_metadata) is not bool or ((cache is None) != profile_no_metadata):
            raise ValueError("attention profiling must explicitly omit the unused KV arena")
        if type(legacy_dummy_run) is not bool or legacy_dummy_run and (
                not profile_no_metadata or model_execution_version != "vllm_0_30_0"):
            raise ValueError("legacy dummy attention requires its0.30 metadata=None source")
        self.legacy_dummy_run = legacy_dummy_run
        self.heads, self.kv_heads, self.head_dim, self.cache = heads, kv_heads, head_dim, cache
        self.q_lens, self.kv_lens = q_lens, kv_lens
        self.slot_mapping = slot_mapping
        self.slot_tensor = None
        self.profile_no_metadata = profile_no_metadata
        self.graph_token_shape = None
        self.graph_attention_source = "cuda_graph_v1"
        self.fa3_full_aot = None
        self.update = SymbolicOperator(trace, name + ":kv-update", pool_name)

    def __call__(self, q, k, v):
        rows = sum(self.q_lens)
        physical_rows = rows
        graph_attributes = {"graph_runtime_mode": "NONE"}
        dynamic_metadata = None
        full_output = None
        source_inputs = None
        if self.graph_attention_source not in ("cuda_graph_v1", "ascend_0_18_full_fia", "ascend_0_18_piecewise_fia"):
            raise ValueError("unknown source attention graph program")
        ascend_piecewise = self.graph_attention_source == "ascend_0_18_piecewise_fia"
        if ascend_piecewise and (self.model_execution_version != "vllm_0_18_0"
                or self.graph_token_shape is not None and self.graph_token_shape.runtime_mode != "PIECEWISE"):
            raise ValueError("Ascend PIECEWISE FIA source requires explicit0.18 PIECEWISE or overflow NONE")
        if ascend_piecewise:
            graph_attributes["source_attention_program"] = "ascend_0_18_piecewise_fia"
        ascend_full = self.graph_token_shape is not None and self.graph_attention_source == "ascend_0_18_full_fia"
        if ascend_full and self.graph_token_shape.runtime_mode != "FULL":
            raise ValueError("Ascend FIA padding source requires FULL execution")
        if self.graph_token_shape is not None:
            token_shape = self.graph_token_shape
            if (token_shape.runtime_mode not in ("FULL", "PIECEWISE") or token_shape.logical_tokens != rows
                    or (token_shape.runtime_mode == "FULL" and not ascend_full and any(length != 1 for length in self.q_lens))
                    or any(value.shape[0] != token_shape.model_input_tokens for value in (q, k, v))):
                raise ValueError("captured GQA requires physical padded inputs and logical single-token decode")
            physical_rows = token_shape.model_input_tokens
            # GPUModelRunner pads attention metadata only for FULL. PIECEWISE
            # keeps the custom attention/KV ops eager with active lengths,
            # while Attention.forward's output allocation follows padded Q.
            if token_shape.runtime_mode == "FULL":
                if ascend_full:
                    from .graph_padding import GraphAttentionShape
                    fresh = self.q_lens == self.kv_lens
                    attention_shape = GraphAttentionShape.from_ascend_full(token_shape, self.q_lens, self.kv_lens,
                        prefill_no_cache=fresh)
                    source_inputs = (q, k, v) if fresh else (q, self.cache)
                else:
                    dynamic_metadata = gqa_full_source_metadata(
                        self.q_lens, self.kv_lens, self.slot_mapping, token_shape)
                    attention_shape = dynamic_metadata["attention_shape"]
                graph_attributes = {"graph_runtime_mode": "FULL",
                    "model_input_q_lens": attention_shape.model_input_q_lens,
                    "model_input_kv_lens": attention_shape.model_input_kv_lens}
                if ascend_full:
                    graph_attributes.update({"mathematical_program": "ascend_0_18_full_fia_softmax_v1",
                        "ascend_fia_prefill_no_cache": fresh, "symbolic_workspace_complete": False})
            full_output = self.trace.tensor(self.trace.unique_name(self.name + ":physical-output"),
                (physical_rows, self.heads * self.head_dim), q.dtype)
            q, k, v = (value.narrow(0, 0, rows) for value in (q, k, v))
        if self.profile_no_metadata:
            if self.legacy_dummy_run:
                output = self.trace.tensor(self.trace.unique_name(self.name + ":dummy-output"),
                    (rows, self.heads * self.head_dim), q.dtype)
                return SymbolicFillZero(self.trace, self.name + ":dummy-output-zero", self.pool_name)(
                    output, controls=(q, k, v))
            # Attention.forward allocates output; FlashAttentionImpl.forward
            # fills it when attn_metadata is None. The dummy runner's -1 slots
            # skip KV writes. This allocation trace has no timing contract.
            node = self.node(WorkloadKind.VECTOR, {"elements": rows * self.heads * self.head_dim},
                q.dtype, "symbolic_storage_only", {"source_operation": "flash_attention_profile_fill"})
            return self.trace.call(node, (q, k, v), shape=(rows, self.heads * self.head_dim), dtype=q.dtype)
        eager_frontend_output = None
        if ascend_piecewise:
            eager_frontend_output = (full_output.narrow(0, 0, rows) if full_output is not None else
                self.trace.tensor(self.trace.unique_name(self.name + ":frontend-output"),
                    (rows, self.heads * self.head_dim), q.dtype))
        elif (full_output is None and self.model_execution_version == "vllm_0_18_0"
                and (any(n != 1 for n in self.q_lens) or self.q_lens == self.kv_lens)):
            eager_frontend_output = self.trace.tensor(self.trace.unique_name(self.name + ":frontend-output"),
                (rows, self.heads * self.head_dim), q.dtype)
        update = self.update.node(WorkloadKind.KV_UPDATE,
            {"elements": rows * self.kv_heads * self.head_dim,
             "seq_len": rows, "num_kv_heads": self.kv_heads, "head_dim": self.head_dim},
            q.dtype, "paged_kv_cache", {**PAGED_KV_CACHE_ATTRIBUTES,
                "model_execution_version": self.model_execution_version,
                **({"source_input_layout": "contiguous"} if k.is_contiguous and v.is_contiguous else
                    {"source_input_layout": "unit_final_stride"} if all(
                        tensor.view.strides_bytes[-1] == tensor.view.element_bytes
                        and (len(tensor.shape) == 2 or len(tensor.shape) == 3
                            and tensor.view.strides_bytes[-2] == self.head_dim * tensor.view.element_bytes)
                        for tensor in (k, v)) else {}),
                "symbolic_slot_mapping": (self.slot_mapping if dynamic_metadata is None else dynamic_metadata["slot_mapping"]),
                "slot_mapping_scope": "declared_logical_pool_not_native_physical_pages",
                **({"symbolic_physical_token_extent": physical_rows, "graph_padding_slot": -1}
                    if self.graph_token_shape is not None else {})})
        inputs = (k, v)
        if self.slot_tensor is not None:
            if self.slot_tensor.shape != (physical_rows,) or self.slot_tensor.dtype != "int64":
                raise ValueError("symbolic GQA slot port requires one INT64 slot per query row")
            inputs += (self.slot_tensor if physical_rows == rows else self.slot_tensor.narrow(0, 0, rows),)
        self.trace.call(update, inputs, shape=self.cache.shape, dtype=self.cache.dtype, out=self.cache)
        node = self.node(WorkloadKind.FLASH_ATTN,
            {"batch": (len(attention_shape.model_input_q_lens) if ascend_full else physical_rows if self.graph_token_shape is not None
                       and self.graph_token_shape.runtime_mode == "FULL" else len(self.q_lens)),
             "q_len": max(attention_shape.model_input_q_lens) if ascend_full else max(self.q_lens),
             "kv_len": (dynamic_metadata["kv_len"] if dynamic_metadata is not None else
                 max(attention_shape.model_input_kv_lens) if ascend_full else max(self.kv_lens)),
             "head_dim": self.head_dim, "num_query_heads": self.heads, "num_kv_heads": self.kv_heads},
            q.dtype, "decode" if all(n == 1 for n in self.q_lens) else "prefill",
            {"causal": True, "kv_layout": "paged_varlen", "kv_dist": "uniform",
             "source_component": "symbolic_gqa_mathematical",
             "source_model_execution_version": self.model_execution_version,
             "source_output_rows": physical_rows,
             "q_lens": self.q_lens, "kv_lens": (self.kv_lens if dynamic_metadata is None else dynamic_metadata["kv_lens"]),
             "fa_ksteps": None, "fa_eff_mfrac": 1.0, "split_policy": "serving", **graph_attributes})
        if ascend_full:
            # FIA.out receives the full physical query/output. Only the
            # preceding KV producer consumes the active K/V prefix. The
            # fresh path reads direct K/V; cache paths read the persistent
            # arena. Keep source ordering without inventing a cache read.
            # full_graph_fia also supplies a fresh one-element BF16 LSE
            # placeholder to the same .out call. Its unknown library workspace
            # is a separate boundary, not a zero-byte scratch declaration.
            lse = self.trace.tensor(self.trace.unique_name(self.name + ":softmax-lse"), (1,), q.dtype)
            return self.trace.call(node, source_inputs, shape=full_output.shape, dtype=q.dtype,
                out=full_output, writes=(lse,), controls=(self.cache,) if fresh else ())
        if eager_frontend_output is not None:
            # forward_fused_infer_attention returns a fresh FIA result,
            # then assigns it into Attention.forward's owning output.
            # Allocate the destination before FIA, retaining both live
            # buffers and the source-visible copy dependency.
            output = eager_frontend_output
            result = self.trace.call(node, (q, k, v, self.cache), shape=output.shape, dtype=q.dtype)
            copy = SymbolicOperator(self.trace, self.name + ":output-copy", self.pool_name)
            copy_node = copy.node(WorkloadKind.ELEMENTWISE, {"elements": rows * self.heads * self.head_dim},
                q.dtype, "elementwise", {"operation": "copy", "reads": 1, "writes": 1,
                    "source_component": "symbolic_ascend_fia_output_copy",
                    **({"source_attention_program": "ascend_0_18_piecewise_fia"} if ascend_piecewise else {})})
            copied = self.trace.call(copy_node, (result,), shape=output.shape, dtype=q.dtype, out=output)
            return full_output if ascend_piecewise and full_output is not None else copied
        if self.fa3_full_aot is not None:
            metadata,semaphore,cu,seq,layout,prepare_id,copy_id,clear_id=self.fa3_full_aot
            operands=(q,k,v,self.cache,cu,seq,metadata)
            node=node.with_attributes({**node.attributes,"fa3_full_aot_prepare_node":prepare_id,
                "fa3_full_aot_copy_node":copy_id,"fa3_full_aot_clear_node":clear_id,
                "fa3_full_aot_layout":layout,"fa3_full_aot_group":metadata.view.allocation_id,
                "fa3_full_aot_input_producers":tuple(self.trace._writers[t.view.allocation_id] for t in operands)})
            if full_output is None:
                raise ValueError("FA3 native AOT metadata requires its full captured output parent")
            self.trace.call(node,operands,shape=(rows,self.heads*self.head_dim),dtype=q.dtype,
                out=full_output.narrow(0,0,rows),writes=(semaphore,))
            return full_output
        if full_output is None:
            return self.trace.call(node, (q, k, v, self.cache), shape=(rows, self.heads * self.head_dim), dtype=q.dtype)
        self.trace.call(node, (q, k, v, self.cache), shape=(rows, self.heads * self.head_dim), dtype=q.dtype,
            out=full_output.narrow(0, 0, rows))
        return full_output


class SymbolicLogitsRowSelect(SymbolicOperator):
    """GPUModelRunner's materialized 16-bit indexing with INT32 caller IDs.

    The index producer is an explicit metadata input. Unknown ATen internal
    index conversion, scratch and launch are not assigned zero storage.
    """

    def __call__(self, hidden_states, indices, *, selected_indices, model_execution_version="vllm_0_30_0",
                 hardware_id="nvidia_a800", disable_index_broadcast=None):
        self.trace._check_tensor(hidden_states)
        self.trace._check_tensor(indices)
        if (len(hidden_states.shape) != 2 or hidden_states.dtype not in ("bf16", "fp16") or len(indices.shape) != 1
                or indices.dtype != "int32" or not isinstance(selected_indices, tuple)
                or len(selected_indices) != indices.shape[0]
                or any(type(index) is not int or not 0 <= index < hidden_states.shape[0] for index in selected_indices)
                or any(a >= b for a, b in zip(selected_indices, selected_indices[1:]))):
            raise ValueError("plain target row selection requires BF16/FP16 hidden and increasing explicit INT32 indices")
        rows, width = indices.shape[0], hidden_states.shape[1]
        if disable_index_broadcast is not None:
            if type(disable_index_broadcast) is not bool:
                raise TypeError("index broadcast flag must be boolean")
            if hardware_id != "metax_c500" or model_execution_version != "vllm_0_24_0":
                raise ValueError("index broadcast dispatch requires its C5000.24 source")
        if model_execution_version not in ("vllm_0_30_0", "vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_27_1"):
            raise ValueError("symbolic logits selection requires its declared source version")
        if hidden_states.dtype == "fp16" and model_execution_version in ("vllm_0_18_0", "vllm_0_27_1"):
            raise NotImplementedError("FP16 target selection requires the ordinary0.11 or0.30 source")
        source = {"vllm_0_11_0": "vllm_0_11", "vllm_0_18_0": "vllm_0_18", "vllm_0_24_0": "vllm_0_24",
            "vllm_0_27_1": "vllm_0_27_1", "vllm_0_30_0": "vllm_0_30"}[model_execution_version]
        index_source = "query_start_loc_int32"
        if model_execution_version in ("vllm_0_11_0", "vllm_0_24_0"):
            if not hidden_states.is_contiguous or hidden_states.view.offset_bytes % 16:
                raise NotImplementedError("bound0.11 row indexing requires contiguous aligned hidden rows")
            # TensorAdvancedIndexingUtils.h::make_info materializes Int
            # indices as Long before advanced indexing. Keep the caller's
            # INT32 input and the fresh internal allocation as distinct ports.
            cast = self.node(WorkloadKind.ELEMENTWISE, {"elements": rows}, "int32", "elementwise",
                {"operation": "cast_int32_int64", "reads": 1, "writes": 1,
                 "tensor_iterator_source": (("pytorch_2_8_sm80" if hardware_id == "nvidia_a800" else "pytorch_2_8_sm90")
                    if hardware_id in ("nvidia_a800", "nvidia_h200") and model_execution_version == "vllm_0_11_0"
                    else "aten_advanced_indexing_long")})
            indices = self.trace.call(cast, (indices,), shape=indices.shape, dtype="int64")
            index_source = "query_start_loc_int32_to_int64"
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * width, "rows": rows, "hidden": width,
            "source_rows": hidden_states.shape[0]}, hidden_states.dtype, "vllm_logits_row_select", {
                "source_component": source + "_symbolic_logits_boundary", "model_execution_version": model_execution_version,
                "index_source": index_source, "selected_indices": selected_indices,
                **({"disable_index_broadcast": disable_index_broadcast} if disable_index_broadcast is not None else {}),
                "symbolic_workspace_complete": False})
        return self.trace.call(node, (hidden_states, indices), shape=(rows, width), dtype=hidden_states.dtype)


class SymbolicTPLogitsProcessor(SymbolicOperator):
    """16-bit ParallelLMHead + CUDA all-gather + source reshape suffix.

    Both base and PyNccl communicator suffixes move the rank-major axis to
    the vocabulary dimension. Multi-row nontrivial shards require a fresh
    row-major materialization; one row is a view of the collective backing.
    Provider/allocator and library-internal workspace remain unbound.
    """

    def __init__(self, trace, name, pool_name, *, hidden_size, vocab_size, group, config,
                 model_execution_version="vllm_0_30_0"):
        super().__init__(trace, name, pool_name)
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_24_0", "vllm_0_30_0", "vllm_0_27_1"):
            raise ValueError("TP logits require their explicitly supported source version")
        if (not isinstance(group, ForwardCollectiveGroup) or not isinstance(config, CollectiveConfig)
                or config.operation != "all_gather" or not any(trace is item for item in group.traces)):
            raise ValueError("TP logits require this rank's canonical all-gather group")
        if type(hidden_size) is not int or hidden_size <= 0 or type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("TP logits require positive checkpoint hidden/vocabulary sizes")
        padded = -(-vocab_size // 64) * 64
        if padded % len(group.traces) or padded // len(group.traces) <= 1:
            raise NotImplementedError("TP logits require evenly sharded nontrivial padded vocabulary")
        self.hidden_size, self.vocab_size, self.padded_vocab = hidden_size, vocab_size, padded
        self.group, self.config = group, config
        self.model_execution_version = model_execution_version
        self.source_component = ("vllm_0_27_1_symbolic_logits_boundary" if model_execution_version == "vllm_0_27_1"
            else "vllm_0_18_symbolic_logits_boundary" if model_execution_version == "vllm_0_18_0"
            else "vllm_0_24_symbolic_logits_boundary" if model_execution_version == "vllm_0_24_0"
            else "vllm_0_11_symbolic_logits_boundary" if model_execution_version == "vllm_0_11_0"
            else "vllm_0_30_symbolic_logits_boundary")
        self.lm_head = SymbolicLinear(trace, name + ":projection", pool_name, hidden_size,
            padded // len(group.traces), kind=WorkloadKind.LM_HEAD)

    def __call__(self, hidden_states):
        if (len(hidden_states.shape) != 2 or hidden_states.shape[-1] != self.hidden_size
                or hidden_states.dtype not in ("bf16", "fp16")):
            raise ValueError("TP logits source requires 16-bit selected hidden rows")
        if hidden_states.dtype == "fp16" and self.model_execution_version in ("vllm_0_18_0", "vllm_0_27_1"):
            raise NotImplementedError("FP16 TP logits require the ordinary0.11 or0.30 source")
        logits, _ = self.lm_head(hidden_states)
        rows, shard = logits.shape
        # The network writes the raw rank-major buffer along dimension0;
        # the API's requested last-dimension order is restored below.
        gathered = self.group.call(self.name + ":all-gather", logits,
            pool_name=self.pool_name, config=self.config, inplace=False, dim=0)
        moved = gathered.view_as_shape(len(self.group.traces), rows, shard).transpose(0, 1)
        if rows == 1:
            logits = moved.view_as_shape(rows, self.padded_vocab)
        else:
            node = self.node(WorkloadKind.VECTOR, {"elements": rows * self.padded_vocab,
                "rows": rows, "vocab": self.padded_vocab, "tensor_parallel_size": len(self.group.traces)},
                logits.dtype, "vllm_logits_gather_reorder", {
                    "source_component": self.source_component, "model_execution_version": self.model_execution_version})
            logits = self.trace.call(node, (moved,), shape=(rows, self.padded_vocab), dtype=logits.dtype)
        return logits if self.padded_vocab == self.vocab_size else logits.narrow(1, 0, self.vocab_size)


class SymbolicGreedy(SymbolicOperator):
    """Version-bound logits preparation, optionally followed by plain greedy."""
    def __call__(self, logits, *, model_execution_version="vllm_0_30_0", sample_logits=True,
                 copy_dispatch=None, maca_copy_runtime_source=None):
        if type(sample_logits) is not bool:
            raise TypeError("greedy sample_logits must be boolean")
        rows, vocab = logits.shape
        versions = {"vllm_0_30_0": "vllm_0_30", "vllm_0_11_0": "vllm_0_11",
                    "vllm_0_18_0": "vllm_0_18",
                    "vllm_0_24_0": "vllm_0_24",
                    "vllm_0_27_1": "vllm_0_27_1"}
        if model_execution_version not in versions:
            raise ValueError("symbolic greedy requires its declared source version")
        if maca_copy_runtime_source is not None:
            if model_execution_version != "vllm_0_24_0" or copy_dispatch is not None:
                raise ValueError("MACA runtime Copy requires0.24 sampler without separate dispatch facts")
            copy_dispatch = _maca_native_copy_dispatch(logits, maca_copy_runtime_source)
        version = versions[model_execution_version]
        source = version + "_plain_greedy_" + logits.dtype
        binding = {"source_component": version + "_symbolic_plain_greedy"}
        if logits.dtype not in ("bf16", "fp16") or (logits.dtype == "fp16" and model_execution_version in ("vllm_0_18_0", "vllm_0_27_1")):
            raise NotImplementedError("ordinary greedy source requires its supported 16-bit logits")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * vocab, "rows": rows, "vocab": vocab},
            logits.dtype, "vllm_logits_fp32_cast", {**binding, "sampling_source": version + "_plain_sampler_" + logits.dtype,
                "copy_input_contiguous": logits.is_contiguous,
                **({"copy_dispatch": copy_dispatch} if copy_dispatch is not None else {})})
        if copy_dispatch is not None and not logits.is_contiguous:
            raise NotImplementedError("explicit copy dispatch requires contiguous logits")
        logits_fp32 = self.trace.call(node, (logits,), shape=logits.shape, dtype="fp32")
        if not sample_logits:
            return logits_fp32
        node = self.node(WorkloadKind.VECTOR, {"elements": rows * vocab, "rows": rows, "vocab": vocab},
            "fp32", "vllm_plain_greedy_sample", {**binding, "sampling_source": source, "stage": "argmax"})
        sampled = self.trace.call(node, (logits_fp32,), shape=(rows,), dtype="int64")
        node = self.node(WorkloadKind.VECTOR, {"elements": rows, "rows": rows, "vocab": vocab},
            "int64", "vllm_plain_greedy_sample", {**binding, "sampling_source": source, "stage": "output_int32"})
        output = self.trace.call(node, (sampled,), shape=(rows,), dtype="int32")
        return output.view_as_shape(rows, 1)


# Capture the real geometry constructor/validator owner once. Unknown
# constructor, validator or owner-global changes use the ordinary constructor.
_NODE_GEOMETRY_VALIDATION = (
    WorkloadNode, dict(vars(WorkloadNode)),
    tuple((function, function.__code__, function.__defaults__,
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__))
        for function in (WorkloadNode.__init__, WorkloadNode.__post_init__, WorkloadNode.with_attributes)),
    WorkloadNode.__post_init__.__globals__, dict(WorkloadNode.__post_init__.__globals__))


"""Append this class to traced_operators.py; original SymbolicV4KVInsert stays unchanged."""


class SymbolicDSparkPartialContextInsert(SymbolicV4KVInsert):
    """Full six-row DSpark context insert with a declared valid cache prefix.

    Actual Q/KV/positions/RoPE and INT64 slot-tensor ports are preserved.
    Logical slots declare the prefix scenario; native physical pages are unbound.
    """

    def __init__(self, trace, name, pool_name, model, *, tp_size, cache,
                 slot_mapping, slot_tensor, valid_token_count):
        if (type(valid_token_count) is not int or not 1 <= valid_token_count <= 6
                or type(tp_size) is not int or tp_size != 8
                or model.num_attention_heads % tp_size
                or model.num_attention_heads // tp_size not in (8, 16)
                or model.head_dim != 512 or model.qk_rope_head_dim != 64):
            raise NotImplementedError("DSpark partial context insert requires TP8, localheads8/16,512/64 and prefix1..6")
        if (not isinstance(slot_mapping, tuple) or len(slot_mapping) != 6
                or any(type(slot) is not int for slot in slot_mapping)
                or any(slot < 0 for slot in slot_mapping[:valid_token_count])
                or any(slot != -1 for slot in slot_mapping[valid_token_count:])):
            raise ValueError("DSpark context slots require six rows: valid prefix then exact -1 tail")
        trace._check_tensor(slot_tensor)
        if slot_tensor.shape != (6,) or slot_tensor.dtype != "int64":
            raise ValueError("DSpark partial context insert requires its actual INT64 six-slot tensor")
        # Reuse the unchanged ordinary owner's cache/model/slot-bound checks.
        # Only the valid prefix is passed to that all-valid constructor; its
        # runtime call below still sees and validates the complete six rows.
        super().__init__(trace, name, pool_name, model, tp_size=tp_size,
            cache=cache, slot_mapping=slot_mapping[:valid_token_count], slot_tensor=None)
        self.slots = slot_mapping
        self.slot_tensor = slot_tensor
        self.valid_token_count = valid_token_count

    def node(self, kind, shape, dtype, variant, attributes=None):
        if (kind is not WorkloadKind.KV_UPDATE or variant != "dsv4_qnorm_rope_kv_insert"
                or dtype != "bf16" or shape["seq_len"] != 6
                or attributes is None
                or attributes["source_component"] != "vllm_0_30_symbolic_v4_swa_insert"):
            raise ValueError("DSpark partial writer must retain the original six-row fused source operation")
        return super().node(kind, shape, dtype, variant, {
            **attributes,
            "mathematical_program": "vllm_0_30_dspark_partial_context_insert_v1",
            "valid_token_count": self.valid_token_count,
            "slot_mapping_all_valid": self.valid_token_count == 6,
            "symbolic_slot_mapping": self.slots,
            "symbolic_valid_cache_write_slots": self.slots[:self.valid_token_count],
            "slot_mapping_scope": "declared_logical_valid_prefix_not_native_physical_pages",
        })
