"""Complete pure K3 speculative verification core from vLLM0.30 source.

causal_conv1d_update consumes the first column of the2-D state-ID tensor as
a strided view, and accepted_tokens-1 selects the conv-history offset.
fused_recurrent_kda instead reads initial state from that accepted column,
then writes each query token's state into its actual declared ID. Acceptance
comes from the previous step and is not capped by this step's query length.

The general recurrent wrapper's H*N->BV/stages rule is unconditional source
dispatch, including on SM90. Its comment says tuned on GB300; no GB300
resource facts or timing are transferred here. Existing H200 observations
independently check the actual source geometry. Registers/shared resources
remain unbound; this is not the packed one-token recurrent specialization.
"""

import math
from dataclasses import replace

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.nvidia_h200.kda_launch import select_kda_decode_launch
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_positive_integer, require_shape


def select_kda_spec_decode_recipe(shape, dtype, attributes, hardware):
    """Return conv, fused general recurrence and in-place output norm.

    Pure verification writes directly into core_attn_out. Mixed speculative
    and ordinary rows, RecoverSSM/checkpoints and padded/null slots need
    their distinct source programs and never gain an inferred copy/scatter.
    """
    return _select_kda_spec_program(shape, dtype, attributes, hardware, caller="pure")


def _select_kda_spec_program(shape, dtype, attributes, hardware, *, caller):
    """Shared source validation/core, with explicit pure or mixed caller.

    Mixed callers receive only convolution/recurrence. Their one outer norm
    and any indexed restore belong to the complete mixed wrapper, not each
    subgroup. No pure direct-output alias is fabricated for an allocation.
    """
    sources = {
        "pure": ("vllm_0_30_kimi_k3_spec_decode", "pure_spec_triton_direct_output"),
        "mixed_decode": ("vllm_0_30_kimi_k3_mixed_decode", "mixed_spec_ordinary_triton"),
        "mixed_prefill": ("vllm_0_30_kimi_k3_mixed_prefill", "mixed_spec_prefill"),
    }
    component, program = sources[caller]
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 speculative source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != component
            or attributes["source_batch_program"] != program
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_input_layout"] != "packed_projection_and_dense_conv_output"
            or attributes["source_raw_gate_layout"] != "contiguous_bf16"
            or attributes["source_conv_weight_layout"] != "channels_taps_fp32"
            or attributes["source_conv_state_dtype"] != "bf16"
            or attributes["source_recurrent_state_dtype"] != "fp32"
            or attributes["source_state_inner_layout"] != "heads_value_key_contiguous"
            or attributes["source_state_indices_dtype"] != "int32"
            or attributes["source_accepted_tokens_dtype"] != "int32"
            or attributes["source_query_start_dtype"] != "int32"
            or attributes["source_recoverssm"] is not False
            or attributes["source_checkpoint_state"] != "disabled"
            or attributes["source_graph_padding"] != "none"
            or (caller == "pure" and attributes["source_output_copy_alias"] != "same_view")
            or attributes["source_output_norm_implementation"] != "forward_cuda"
            or attributes["source_output_norm_alias"] != "same_view"):
        raise NotImplementedError("K3 verification requires its exact0.30 declared Triton/FP32/CUDA-norm source")
    rows, heads, dim, width, max_query = require_shape(shape,
        ("rows", "heads", "head_dim", "conv_kernel", "max_query_len"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128 or width != 4 or max_query < 2:
        raise NotImplementedError("K3 verification requires multi-token128-wide/four-tap source geometry")
    configured_spec = require_positive_integer(attributes, "source_configured_num_spec")
    if max_query != configured_spec + 1:
        raise ValueError("K3 verification state columns must preserve the actual configured speculative extent")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) < 2
            or any(type(value) is not int for value in starts) or starts[0] != 0 or starts[-1] != rows
            or any(not 0 < b - a <= max_query for a, b in zip(starts, starts[1:]))):
        raise ValueError("K3 verification requires complete positive query segments within its declared extent")
    sequences = len(starts) - 1
    if max(rows, sequences, max_query) >= 2**31:
        raise NotImplementedError("K3 verification host fields exceed signed int32")
    accepted = attributes["num_accepted_tokens"]
    if (not isinstance(accepted, (tuple, list)) or len(accepted) != sequences
            or any(type(value) is not int or not 1 <= value <= max_query for value in accepted)):
        raise ValueError("K3 verification requires actual previous accepted counts within its state columns")
    slots = require_positive_integer(attributes, "state_pool_slots")
    indices = attributes["state_indices"]
    if (not isinstance(indices, (tuple, list)) or len(indices) != sequences
            or any(not isinstance(row, (tuple, list)) or len(row) != max_query
                or any(type(index) is not int or not -2**31 <= index < min(slots, 2**31) for index in row) for row in indices)):
        raise ValueError("K3 verification requires every actual INT32 state-ID column")
    owners = set()
    for seq, (begin, end) in enumerate(zip(starts, starts[1:])):
        live_writes = tuple(indices[seq][:end - begin])
        initial = indices[seq][accepted[seq] - 1]
        # Unaccessed columns may be null. Previous acceptance can select a
        # column beyond today's query, but that actual read must remain live.
        if initial <= 0 or any(index <= 0 for index in live_writes):
            raise ValueError("K3 verification requires live initial/current-token state slots")
        accessed = set((*live_writes, initial))
        if len(set(live_writes)) != len(live_writes) or accessed & owners:
            raise ValueError("K3 verification requires non-overlapping live per-token state slots")
        owners.update(accessed)
    index_stride = require_positive_integer(attributes, "source_state_indices_row_stride_elements")
    column_stride = require_positive_integer(attributes, "source_state_indices_column_stride_elements")
    accepted_stride = require_positive_integer(attributes, "source_accepted_tokens_stride_elements")
    query_stride = require_positive_integer(attributes, "source_query_start_stride_elements")
    state_stride = require_positive_integer(attributes, "source_state_slot_stride_elements")
    if (index_stride < max_query or state_stride < heads * dim * dim
            or column_stride != 1 or accepted_stride != 1 or query_stride != 1):
        raise ValueError("K3 verification state/index rows must have dense inner dimensions without overlap")
    if int(float(index_stride)) != index_stride:
        raise NotImplementedError("K3 verification index stride is not exactly representable in launch parameters")
    state_length = require_positive_integer(attributes, "source_conv_state_length")
    conv_stride = require_positive_integer(attributes, "source_conv_slot_stride_elements")
    if state_length < max_query + 2 or conv_stride < 3 * heads * dim * state_length:
        raise ValueError("K3 verification conv state must retain its full configured speculative history")
    layout = attributes["source_conv_state_layout"]
    if layout not in ("SD", "DS"):
        raise NotImplementedError("K3 verification requires actual SD/DS conv-state layout")
    expected = (1, 3 * heads * dim) if layout == "SD" else (state_length, 1)
    channel_stride = require_positive_integer(attributes, "source_conv_channel_stride_elements")
    tap_stride = require_positive_integer(attributes, "source_conv_tap_stride_elements")
    if (channel_stride, tap_stride) != expected:
        raise ValueError("K3 verification convolution strides disagree with the actual history layout")
    lower_bound = attributes["gate_lower_bound"]
    if lower_bound is not None and (type(lower_bound) not in (int, float)
            or not math.isfinite(lower_bound) or not -5 <= lower_bound < 0):
        raise ValueError("K3 verification gate bound must be absent or finite in[-5,0)")
    if type(attributes["source_norm_eps"]) not in (int, float) or attributes["source_norm_eps"] != 1e-5:
        raise ValueError("K3 verification requires the source gated-norm epsilon1e-5")
    if max(4 * sequences * index_stride, 4 * slots * state_stride, 2 * slots * conv_stride) >= 2**63:
        raise NotImplementedError("K3 verification metadata/cache addressing exceeds signed int64")
    head_sequences = heads * sequences
    if head_sequences <= 48:
        bv, stages = 4, 4
    elif head_sequences <= 96:
        bv, stages = 8, 3
    elif head_sequences <= 192:
        bv, stages = 16, 3
    else:
        bv, stages = 8, 3
    programs = ceil_div(dim, bv) * head_sequences
    if programs > 2**31 - 1:
        raise NotImplementedError("K3 verification recurrent grid exceeds CUDA grid.x")
    history = width - 1 + max_query - 1
    np2_history = 1 << (history - 1).bit_length()
    conv = LaunchConfig("vllm_0_30_sm90_kda_spec_causal_conv1d_update",
        (sequences, ceil_div(3 * heads * dim, 256), 1), (128, 1, 1), parameters={
            "source_np2_state_length": float(np2_history), "source_programmatic_stream_serialization": 1.0})
    recurrent = LaunchConfig(f"vllm_0_30_sm90_kda_spec_recurrent_bv{bv}", (programs, 1, 1), (32, 1, 1),
        parameters={"source_BV": float(bv), "source_num_stages": float(stages),
            "source_state_indices_row_stride": float(index_stride), "source_programmatic_stream_serialization": 1.0})
    if caller != "pure":
        return (("spec_convolution", conv), ("spec_recurrent", recurrent))
    # Both source branches end at the same CUDA in-place o_norm call. Reuse
    # that physical stage only; no ordinary conv/recurrent fact is borrowed.
    norm = select_kda_decode_launch(shape, dtype, "kimi_k3_kda_decode",
        {"source_component": "vllm_0_30_kimi_k3_triton_decode", "stage": "output_norm"}, hardware)
    return (("spec_convolution", conv), ("spec_recurrent", recurrent),
        ("output_norm", replace(norm, execution_variant=None)))
