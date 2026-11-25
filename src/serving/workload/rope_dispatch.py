"""Source-pinned dispatch of the RoPE program vLLM actually executes.

vLLM resolves a rotary embedding to one of two very different physical
programs, and which one runs is decided by a *class-name* lookup, not by the
model architecture:

* ``vllm/model_executor/layers/rotary_embedding/__init__.py::get_rope``
  instantiates one concrete class per ``rope_parameters["rope_type"]``
  (``"default"`` -> ``RotaryEmbedding``, ``"llama3"`` ->
  ``Llama3RotaryEmbedding``, ``"yarn"`` -> ``YaRNScalingRotaryEmbedding``, ...).
* ``vllm/model_executor/custom_op.py::CustomOp.__new__`` looks the *exact*
  ``cls.__name__`` up in the out-of-tree registry ``op_registry_oot`` and, only
  on a hit, instantiates the plugin subclass instead.
* ``CustomOp.dispatch_forward`` returns ``forward_oot`` on an out-of-tree
  platform, and the ``CustomOp.forward_oot`` default is ``forward_native``.

So on an out-of-tree platform a rope class whose name the plugin did not
register falls all the way back to ``RotaryEmbedding.forward_native``, which is
a composite of ~31 framework tensor ops per call rather than one fused device
kernel.  ``Llama3RotaryEmbedding`` is exactly such a class: neither
vllm-ascend nor vllm-metax registers that name, while both register
``RotaryEmbedding``.  In-tree CUDA is unaffected because ``forward_cuda`` is
reached through normal class inheritance, not through a name registry.

Source anchors (versioned vLLM snapshots, pinned by ``runtime.source_commit``
in the Serving manifests:
``vllm=0.18.0;...;vllm-ascend=5673759a...cdd3fd`` for ``ascend_910b``):

* ``v0.18.0_ascend/model_executor/layers/rotary_embedding/__init__.py`` lines 94-326
  (``get_rope`` scaling-type dispatch) and ``base.py`` lines 137-192
  (``RotaryEmbedding.forward_static`` / ``forward_native``).
* ``v0.18.0_ascend/model_executor/layers/rotary_embedding/common.py`` lines 137-175
  (``ApplyRotaryEmb.forward_static``).
* ``vllm_ascend_plugin/utils.py`` lines 633-660 (``REGISTERED_ASCEND_OPS``)
  and ``vllm_ascend_plugin/ops/rotary_embedding.py`` lines 231-249
  (``AscendRotaryEmbedding.forward_oot`` -> ``npu_rotary_embedding``) and lines
  151-213 (``rope_forward_oot`` -> ``rope_forward_triton``).
* ``v0.24.0_c500_sdist/vllm/model_executor/custom_op.py`` lines 109-128
  (``CustomOp.__new__``), 168-172 (``forward_oot`` default) and 173-200
  (``dispatch_forward``); the same file is not shipped in the partial 0.18
  snapshot, and the 0.24 text is the pinned reference for that mechanism.
* ``vllm_metax_0.24.0/platform.py`` line 162 (``_enum = PlatformEnum.OOT``)
  and ``vllm_metax_0.24.0/customized/ops/rotary_embedding.py`` lines 11-32
  (the four registered names).
"""

from dataclasses import dataclass


# ``get_rope`` scaling type -> concrete class name.  Exhaustive over the
# branches of the pinned ``get_rope``; the value is the class the branch
# instantiates, which is the key ``CustomOp.__new__`` looks up.
VLLM_ROPE_CLASS_BY_SCALING_TYPE = {
    "default": "RotaryEmbedding",
    "llama3": "Llama3RotaryEmbedding",
    "linear": "LinearScalingRotaryEmbedding",
    "ntk": "NTKScalingRotaryEmbedding",
    "dynamic": "DynamicNTKScalingRotaryEmbedding",
    "xdrope": "XDRotaryEmbedding",
    "yarn": "YaRNScalingRotaryEmbedding",
    "deepseek_yarn": "DeepseekScalingRotaryEmbedding",
    "deepseek_llama_scaling": "DeepseekScalingRotaryEmbedding",
    "longrope": "Phi3LongRoPEScaledRotaryEmbedding",
    "mllama4": "Llama4VisionRotaryEmbedding",
    "openpangu": "MRotaryEmbeddingInterleaved",
}

# Out-of-tree rope class names each runtime plugin registers.
_OOT_ROTARY_REGISTRY = {
    # vllm_ascend/utils.py::register_ascend_customop REGISTERED_ASCEND_OPS.
    "vllm_ascend": frozenset({
        "RotaryEmbedding",
        "MRotaryEmbedding",
        "YaRNScalingRotaryEmbedding",
        "DeepseekScalingRotaryEmbedding",
    }),
    # vllm_metax/customized/ops/rotary_embedding.py register_oot decorators.
    "vllm_metax": frozenset({
        "RotaryEmbedding",
        "MRotaryEmbedding",
        "XDRotaryEmbedding",
        "DeepseekScalingRotaryEmbedding",
    }),
}

# ``current_platform.is_out_of_tree()`` per hardware.  ``vllm_cuda_in_tree``
# reaches ``forward_cuda`` by inheritance, so every rope class is fused there.
ROPE_RUNTIME_BY_HARDWARE = {
    "ascend_910b": "vllm_ascend",
    "metax_c500": "vllm_metax",
    "nvidia_a800": "vllm_cuda_in_tree",
    "nvidia_h200": "vllm_cuda_in_tree",
}

# How the runtime's kernel library executes the composite.  vllm-ascend calls
# aclnn ops, which require contiguous inputs and therefore materialize every
# `chunk` view with its own Slice kernel; ATen's TensorIterator consumes the
# strided views directly, and lowers `sub` onto the same CUDAFunctor_add with a
# negated alpha.  The two programs are 31 and 15 kernels per call.
ROPE_COMPOSITE_SEMANTICS_BY_RUNTIME = {
    "vllm_ascend": "aclnn_materialized_operands",
    "vllm_metax": "aten_tensor_iterator",
}

ROPE_PROGRAM_FUSED = "fused_backend_custom_op"
ROPE_PROGRAM_NATIVE_COMPOSITE = "vllm_native_composite"

# Runtimes whose native-composite rope Midlayer can currently lower.  A runtime
# missing here still *runs* the composite program -- ``vllm_rope_program``
# always reports the source truth -- but Midlayer cannot yet build its kernel
# sequence, so ``modelled_rope_program`` hands the caller a named divergence
# instead of pretending the composite does not exist.
_COMPOSITE_LOWERING_RUNTIMES = frozenset({"vllm_ascend", "vllm_metax"})
_COMPOSITE_LOWERING_GAPS = {}

# ``aclnnCat`` over a single non-empty input (the trailing ``cat((rot, pass))``
# whose ``pass`` slice is empty when rotary_dim == head_size) degenerates to a
# TensorMove kernel, and CANN launches no kernel at all at or below this output
# element count.  This is a discrete launch fact read from the launch columns
# of the three ShareGPT llama ``op_profile.json`` captures (q crosses it between
# T=48 and T=49 at 32 heads, k between T=192 and T=193 at 8 heads, zero
# violations over 375 distinct token counts); no duration was consulted.
CAT_SINGLE_INPUT_TENSOR_MOVE_NUMEL = 196608


@dataclass(frozen=True)
class RopeCompositeCall:
    """One framework tensor op of ``RotaryEmbedding.forward_native``."""

    name: str
    operation: str
    elements: int
    source_symbol: str
    # Operand layout as the kernel library sees it.  "contiguous" covers every
    # aclnn call (its operands were materialized first) and the ATen reductions
    # over freshly written multiply outputs; "dim0_contiguous_dim1_broadcast"
    # is the ATen multiply over a strided rotate-half view times a broadcast
    # cos/sin row; "concatenated_last_dim" is the cat of the two halves.
    operand_layout: str = "contiguous"
    # Iteration geometry the launch selector needs beyond element_count.
    tokens: int = 0
    rows: int = 0
    inner: int = 0


def pinned_rope_runtime(hardware_id: str) -> str:
    """Return the rope runtime the pinned vLLM build uses on this hardware.

    A hardware without a pinned runtime is unsupported rather than absent, so
    the failure is a ValueError naming the hardware.
    """

    if hardware_id not in ROPE_RUNTIME_BY_HARDWARE:
        raise ValueError(
            f"no pinned vLLM rope runtime for hardware {hardware_id!r}")
    return ROPE_RUNTIME_BY_HARDWARE[hardware_id]


def vllm_rope_program(hardware_id: str, rope_scaling_type: str) -> str:
    """Return which physical rope program the pinned runtime executes."""

    runtime = pinned_rope_runtime(hardware_id)
    if rope_scaling_type not in VLLM_ROPE_CLASS_BY_SCALING_TYPE:
        raise ValueError(
            f"unsupported vLLM rope scaling type: {rope_scaling_type!r}")
    rope_class = VLLM_ROPE_CLASS_BY_SCALING_TYPE[rope_scaling_type]
    if runtime == "vllm_cuda_in_tree":
        # CustomOp.dispatch_forward returns forward_cuda, which every
        # RotaryEmbedding subclass inherits.
        return ROPE_PROGRAM_FUSED
    if runtime not in _OOT_ROTARY_REGISTRY:
        raise ValueError(f"unhandled vLLM rope runtime: {runtime!r}")
    if rope_class in _OOT_ROTARY_REGISTRY[runtime]:
        return ROPE_PROGRAM_FUSED
    return ROPE_PROGRAM_NATIVE_COMPOSITE


def native_composite_rope_calls(
        model_input_tokens: int, query_heads: int, kv_heads: int,
        head_dim: int, semantics: str) -> tuple:
    """Enumerate the framework tensor ops of one forward_native rope call.

    Mirrors ``RotaryEmbedding.forward_static`` composed with
    ``ApplyRotaryEmb.forward_static`` for the neox-style, full-width
    (``rotary_dim == head_size``) case that the Serving models in scope use.

    ``x1``/``x2`` are ``chunk`` views of the fused QKV buffer and ``cos``/
    ``sin`` are ``chunk`` views of the gathered ``[T, head_dim]`` table.  Under
    ``aclnn_materialized_operands`` every aclnn call needs contiguous inputs,
    so each multiply materializes both operands through a Slice kernel first;
    the cos/sin view is already contiguous when there is exactly one token (its
    only outer dimension has extent one) and CANN then launches no Slice.
    Under ``aten_tensor_iterator`` the strided views are consumed in place, so
    no Slice exists at all, and ``torch.sub`` shares ``CUDAFunctor_add``.
    """

    for name, value in (
            ("model_input_tokens", model_input_tokens),
            ("query_heads", query_heads),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"composite rope {name} must be an integer")
        if value <= 0:
            raise ValueError(f"composite rope {name} must be positive")
    if head_dim % 2:
        raise ValueError("composite rope head_dim must be even")
    if semantics not in ROPE_COMPOSITE_SEMANTICS_BY_RUNTIME.values():
        raise ValueError(f"unknown rope composite semantics: {semantics!r}")
    half = head_dim // 2
    tokens = model_input_tokens

    calls = [RopeCompositeCall(
        name="cos-sin-gather",
        operation="gather",
        elements=tokens * head_dim,
        source_symbol="vllm.RotaryEmbedding.forward_static.index_select",
        tokens=tokens,
        rows=tokens,
        inner=head_dim,
    )]
    materialize_operands = semantics == "aclnn_materialized_operands"
    materialize_cos_sin = materialize_operands and tokens > 1
    for role, heads in (("q", query_heads), ("k", kv_heads)):
        rotated = tokens * heads * half
        # o1 = x1 * cos - x2 * sin ; o2 = x2 * cos + x1 * sin
        for term, factor in (
                ("x1-cos", "cos"), ("x2-sin", "sin"),
                ("x2-cos", "cos"), ("x1-sin", "sin")):
            if materialize_operands:
                calls.append(RopeCompositeCall(
                    name=f"{role}-{term}-operand-slice",
                    operation="copy",
                    elements=rotated,
                    source_symbol="torch.chunk(qkv)",
                    tokens=tokens, rows=heads, inner=half,
                ))
            if materialize_cos_sin:
                calls.append(RopeCompositeCall(
                    name=f"{role}-{term}-{factor}-slice",
                    operation="copy",
                    elements=tokens * half,
                    source_symbol="torch.chunk(cos_sin)",
                    tokens=tokens, rows=1, inner=half,
                ))
            calls.append(RopeCompositeCall(
                name=f"{role}-{term}-mul",
                operation="mul",
                elements=rotated,
                source_symbol="vllm.ApplyRotaryEmb.forward_static.mul",
                operand_layout=(
                    "contiguous" if materialize_operands
                    else "dim0_contiguous_dim1_broadcast"),
                tokens=tokens, rows=heads, inner=half,
            ))
        calls.append(RopeCompositeCall(
            name=f"{role}-o1-sub",
            operation="binary",
            elements=rotated,
            source_symbol="vllm.ApplyRotaryEmb.forward_static.sub",
            tokens=tokens, rows=heads, inner=half,
        ))
        calls.append(RopeCompositeCall(
            name=f"{role}-o2-add",
            operation="binary",
            elements=rotated,
            source_symbol="vllm.ApplyRotaryEmb.forward_static.add",
            tokens=tokens, rows=heads, inner=half,
        ))
        calls.append(RopeCompositeCall(
            name=f"{role}-rotate-concat",
            operation="copy",
            elements=tokens * heads * head_dim,
            source_symbol="vllm.ApplyRotaryEmb.forward_static.cat",
            operand_layout="concatenated_last_dim",
            tokens=tokens, rows=heads, inner=head_dim,
        ))
        # cat((rot, pass)) back in forward_static; `pass` is empty because
        # rotary_dim == head_size.  ATen returns the already materialized
        # rotate output, and CANN degenerates it to a single-input TensorMove
        # that it elides below the threshold above.
        passthrough_concat = tokens * heads * head_dim
        if (materialize_operands
                and passthrough_concat > CAT_SINGLE_INPUT_TENSOR_MOVE_NUMEL):
            calls.append(RopeCompositeCall(
                name=f"{role}-passthrough-concat",
                operation="copy",
                elements=passthrough_concat,
                source_symbol="vllm.RotaryEmbedding.forward_static.cat",
                operand_layout="concatenated_last_dim",
                tokens=tokens, rows=heads, inner=head_dim,
            ))
    return tuple(calls)


def rope_composite_semantics(hardware_id: str) -> str:
    """Return how the hardware's runtime executes a native-composite rope."""

    runtime = pinned_rope_runtime(hardware_id)
    if runtime not in ROPE_COMPOSITE_SEMANTICS_BY_RUNTIME:
        raise ValueError(
            f"runtime {runtime!r} has no composite rope semantics")
    return ROPE_COMPOSITE_SEMANTICS_BY_RUNTIME[runtime]


def modelled_rope_program(
        hardware_id: str, rope_scaling_type: str) -> tuple[str, str | None]:
    """Return the rope program Midlayer emits plus any declared divergence.

    The second element is ``None`` when the emitted program is the one the
    runtime really executes, and otherwise names why Midlayer still models the
    fused custom op.  Callers must handle both, so a lowering gap can never be
    mistaken for source truth.
    """

    program = vllm_rope_program(hardware_id, rope_scaling_type)
    if program == ROPE_PROGRAM_FUSED:
        return program, None
    if program != ROPE_PROGRAM_NATIVE_COMPOSITE:
        raise AssertionError(f"unhandled rope program: {program!r}")
    runtime = pinned_rope_runtime(hardware_id)
    if runtime in _COMPOSITE_LOWERING_RUNTIMES:
        return program, None
    if runtime not in _COMPOSITE_LOWERING_GAPS:
        raise NotImplementedError(
            f"composite rope lowering is undeclared for runtime {runtime!r}")
    return ROPE_PROGRAM_FUSED, _COMPOSITE_LOWERING_GAPS[runtime]
