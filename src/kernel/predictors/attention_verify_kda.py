"""SGLang 0.5.19 projected-input KDA T4 chain source accounting.

This explicitly selects the non-fused Triton fallback, BF16 inputs/history,
FP32 recurrent state, H64/D128, bounded gate -5 and in-kernel QK L2 norm.
Source operand bytes and mathematical work do not bind a native provider,
compiler issue rate, cache residency or measured duration.
"""

_ROOT = "sglang/v0.5.19/python/sglang/"
_BACKEND = _ROOT + "srt/layers/attention/linear/kda_backend.py"
_CONV = _ROOT + "kernels/ops/mamba/causal_conv1d_triton.py"
_RECURRENT = _ROOT + "kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"
_NORM = _ROOT + "kernels/ops/attention/fla/fused_norm_gate.py"
_POOL = _ROOT + "srt/mem_cache/memory_pool.py"
_RESOURCES = ("compute_tensor", "compute_vector", "compute_reduction",
              "compute_fma", "compute_sfu", "compute_alu")
_STAGES = ("convolution", "layout_views", "recurrent", "output_norm")
_HEADS, _DIM, _TOKENS, _PARTITIONS = 64, 128, 4, 4
_VALUES = _HEADS * _DIM
_CHANNELS = 3 * _VALUES
_LIMITATIONS = (
    "Explicit generic Triton target_verify linear chain; fused K3, CuTeDSL, tree, ragged verify and ReplaySSM are disabled.",
    "BF16 contiguous projected QKV/gates/history/norm weights; FP32 recurrent state, conv weights, A_log and dt_bias; no observed runtime/backend or native accuracy claim.",
    "Source operand-role bytes include repeated program loads, not measured physical HBM transactions; compiler/local/shared traffic is unbound.",
    "Mathematical FP32 operations and ideal transcendental counts reuse the existing owner; conversion, address/control and compiled issue costs are unpriced.",
    "Output projection and subsequent acceptance/rollback/commit are outside projected-input to normalized-attention-output scope.",
)


def verify_kda_stages():
    """Ordered source stages; layout_views performs no device submission."""
    return _STAGES


def _scalar_work(stage, hardware_id):
    # These stages have exactly the same per-token scalar equations as the
    # ordinary-decode owner. Its T1 state/history traffic and launch geometry
    # are deliberately discarded: rows=4 here supplies only arithmetic.
    from src.kernel.kernel_spec import KernelKind, KernelSpec
    from src.kernel.predictors.vector import kimi_k3_kda_decode_logical_work

    spec = KernelSpec(
        kind=KernelKind.VECTOR, hardware_id=hardware_id, dtype="bf16",
        variant="kimi_k3_kda_decode", execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable",
        shape={"element_count": _TOKENS * _VALUES, "rows": _TOKENS,
               "heads": _HEADS, "head_dim": _DIM},
        attributes={"stage": stage, "state_indices": (1, 2, 3, 4),
                    "state_pool_slots": 5, "gate_lower_bound": -5.0},
    )
    return dict(kimi_k3_kda_decode_logical_work(spec)["demand"])


def verify_kda_work(stage, context, *, hardware_id):
    """One BS1 T4 verification segment ending at ``context`` tokens.

    Context does not change the recurrent state extent. Input projection
    GEMMs are outside this prepared-input boundary. No four-independent-T1
    state reads or committed-SSM stores are substituted for the T4 loop.
    """
    if stage not in _STAGES:
        raise ValueError("unknown KDA T4 source stage")
    if type(context) is not int or context < _TOKENS:
        raise ValueError("KDA T4 context must be an integer >=4")
    operands = []

    def operand(role, dtype, elements, access):
        width = 4 if dtype in ("fp32", "int32") else 2
        operands.append({"role": role, "dtype": dtype, "elements": elements,
                         "bytes": width * elements, "access": access})

    values = _TOKENS * _VALUES
    head_rows = _TOKENS * _HEADS
    programs = _PARTITIONS * _HEADS
    demand = dict.fromkeys(_RESOURCES, 0)
    flags = {}
    aliases = ()
    if stage == "convolution":
        # state_len=3 < T=4: history shift has no old-history reload.
        # The final history is the last three raw projected inputs; the
        # four-step dot uses a separate source input load at each step.
        operand("packed_qkv_for_final_history", "bf16", 3 * _CHANNELS, "read")
        operand("packed_qkv_for_dot", "bf16", _TOKENS * _CHANNELS, "read")
        operand("committed_conv_history", "bf16", 3 * _CHANNELS, "read")
        operand("conv_weight", "fp32", 4 * _CHANNELS, "read")
        operand("conv_state_indices", "int32", _CHANNELS // 256, "read")
        operand("intermediate_state_indices", "int32", _CHANNELS // 256, "read")
        operand("updated_committed_conv_history", "bf16", 3 * _CHANNELS, "write")
        operand("intermediate_conv_windows", "bf16", _TOKENS * 3 * _CHANNELS, "write")
        operand("post_conv_qkv", "bf16", _TOKENS * _CHANNELS, "write")
        demand = _scalar_work(stage, hardware_id)
        source = [_BACKEND + ":980-1003", _CONV + ":579-991,995-1214"]
        launch = {"grid": (1, 96), "block_n": 256,
                  "triton_default_warps_not_assigned": True}
        flags = {"SAVE_INTERMEDIATE": True, "IS_SPEC_DECODING": False,
                 "HAS_EAGLE_TREE_CUSTOM_ATTN_MASK": False,
                 "SILU_ACTIVATION": True, "HAS_BIAS": False,
                 "IS_CONTINUOUS_BATCHING": True, "state_len": 3}
        aliases = (("committed_conv_history", "updated_committed_conv_history"),)
    elif stage == "layout_views":
        # Contiguous [4,C] -> [1,4,C] -> [1,C,4] is non-overlapping
        # dense. empty_like preserves that transpose's strides; transposing
        # its output back and splitting/unflattening QKV are views. The
        # executable wrappers have no input_guard/contiguous decorator.
        source = [_BACKEND + ":958-1022", _CONV + ":1083-1092"]
        launch = None
        flags = {"input_projected_qkv_contiguous": True,
                 "conv_output_preserves_input_strides": True,
                 "required_device_materializations": 0}
    elif stage == "recurrent":
        operand("post_conv_qkv", "bf16", (2 * _PARTITIONS + 1) * values, "read")
        operand("raw_gate", "bf16", _PARTITIONS * values, "read")
        operand("raw_beta", "bf16", _PARTITIONS * head_rows, "read")
        operand("A_log", "fp32", _PARTITIONS * head_rows, "read")
        operand("dt_bias", "fp32", _PARTITIONS * values, "read")
        operand("state_indices", "int32", programs, "read")
        operand("intermediate_state_indices", "int32", programs, "read")
        operand("cu_seqlens", "int32", 2 * programs, "read")
        operand("committed_recurrent_state", "fp32", _VALUES * _DIM, "read")
        operand("intermediate_recurrent_states", "fp32", _TOKENS * _VALUES * _DIM, "write")
        operand("recurrent_output", "bf16", values, "write")
        demand = _scalar_work(stage, hardware_id)
        source = [_ROOT + "srt/layers/attention/linear/kernels/kda_triton.py:158-217",
                  _RECURRENT + ":95-348,390-519"]
        launch = {"grid": (4, 1, 64), "bk": 128, "bv": 32,
                  "num_warps": 1, "num_stages": 3, "ordered_token_loop": 4}
        flags = {"IS_KDA": True, "USE_INITIAL_STATE": True,
                 "USE_QK_L2NORM_IN_KERNEL": True, "IS_VARLEN": True,
                 "USE_LOWER_BOUND": True, "DISABLE_STATE_UPDATE": True,
                 "CACHE_INTERMEDIATE_STATES": True, "CACHE_RING": False,
                 "HAS_EAGLE_TREE_CUSTOM_ATTN_MASK": False}
    elif stage == "output_norm":
        operand("recurrent_output", "bf16", values, "read")
        operand("output_gate", "bf16", values, "read")
        operand("norm_weight", "bf16", (head_rows // 32) * _DIM, "read")
        operand("normalized_attention_output", "bf16", values, "write")
        operand("rstd", "fp32", head_rows, "write")
        demand = _scalar_work(stage, hardware_id)
        source = [_ROOT + "srt/models/kimi_k3.py:1642-1644,1859-1902",
                  _NORM + ":26-110,186-265,281-322"]
        launch = {"grid": (8,), "bt": 32, "bd": 128, "num_warps": 4}
        flags = {"IS_RMS_NORM": True, "ACTIVATION": "sigmoid",
                 "HAS_WEIGHT": True, "HAS_BIAS": False,
                 "HAS_RESIDUAL": False, "STORE_RESIDUAL_OUT": False}
        aliases = (("recurrent_output", "normalized_attention_output"),)
    else:
        raise AssertionError("non-exhaustive KDA T4 stage dispatch")
    return {"stage": stage, "demand": demand, "operands": operands,
            "source": source, "limitations": list(_LIMITATIONS),
            "source_launch": launch, "source_flags": flags,
            "in_place_roles": aliases, "tensor_dtype": "bf16",
            "arithmetic_dtype": "fp32", "query_tokens": _TOKENS,
            "fixed_source_constants": {"conv_width": 4, "gate_lower_bound": -5.0,
                "qk_l2_epsilon": 1e-6, "output_norm_epsilon": 1e-5,
                "query_scale": _DIM ** -0.5}}


def verify_kda_state():
    """Active one-request/layer state, excluding whole-pool padding/reserve.

    Four post-state snapshots and four dense conv windows are distinct
    physical buffers. SSM snapshots are scratch, not four permanent states.
    KDA explicitly disables the pool's overlapping conv-window deduplication.
    External prepared inputs/parameters are not state or operator workspace.
    """
    components = {
        "committed_fp32_recurrent_state": _VALUES * _DIM * 4,
        "committed_bf16_conv_history": 3 * _CHANNELS * 2,
        "intermediate_fp32_recurrent_states": _TOKENS * _VALUES * _DIM * 4,
        "intermediate_bf16_conv_windows": _TOKENS * 3 * _CHANNELS * 2,
    }
    persistent = (components["committed_fp32_recurrent_state"]
                  + components["committed_bf16_conv_history"])
    workspace = (components["intermediate_fp32_recurrent_states"]
                 + components["intermediate_bf16_conv_windows"])
    # Target-verify holds post-conv QKV until recurrence returns. Norm then
    # aliases recurrence output and allocates rstd after conv's local dies.
    transient_components = {"post_conv_qkv": _TOKENS * _CHANNELS * 2,
                            "recurrent_output": _TOKENS * _VALUES * 2,
                            "norm_rstd": _TOKENS * _HEADS * 4}
    transient_peak = max(transient_components["post_conv_qkv"]
                         + transient_components["recurrent_output"],
                         transient_components["recurrent_output"]
                         + transient_components["norm_rstd"])
    return {"base_persistent_bytes": persistent, "verify_workspace_bytes": workspace,
            "components": components, "total_live_bytes": persistent + workspace,
            "transient_buffer_components": transient_components,
            "transient_buffer_peak_bytes": transient_peak,
            "peak_including_operator_temporaries_bytes": persistent + workspace + transient_peak,
            "source": [_POOL + ":117-132,738-750,779-826", _BACKEND + ":980-1077"],
            "limitations": [*_LIMITATIONS,
                "Active slot extents exclude null slots, spare pool capacity, metadata, parameters and prepared input storage; not a whole-model capacity claim.",
                "Committed SSM is read-only during verify; committed conv history is speculatively mutated and resolved by later acceptance/rollback.",
                "total_live_bytes sums persistent and verification-state buffers; the separately reported phase peak includes source operator temporaries, not allocator/runtime reserve."]}
