"""Complete ordinary K3 decode source branches, pinned to vLLM0.30/SM90.

Native fused_kda_decode_kernel.cu launches(B,H),256 threads,dynamic shared0
and PDL. It incorporates convolution, gate/beta, state recurrence and output
norm in that one submission. Its many-head and SD/DS specializations are
fixed by the host wrapper; compiler registers/static shared remain unbound.

The alternative source branch is packed Triton conv -> recurrent -> actual
output transfer/no-op -> in-place gated RMSNorm. Normal backend selection
retains runtime op availability and preloaded weight-buffer readiness. It
does not assume a wheel contains an operator merely because source exists.
"""

import math

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_positive_integer, require_shape


def _ordinary_decode_domain(shape, dtype, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 ordinary decode source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_input_layout"] != "aligned_16_k3_packed_qkv_beta_gate"
            or attributes["source_raw_gate_layout"] != "contiguous_bf16"
            or attributes["source_conv_state_dtype"] != "bf16"
            or attributes["source_recurrent_state_dtype"] != "fp32"
            or attributes["source_state_inner_layout"] != "heads_value_key_contiguous"
            or attributes["source_state_indices_dtype"] != "int32"
            or attributes["source_output_layout"] != "contiguous_bf16"
            or attributes["source_speculation_state"] != "disabled"
            or attributes["source_checkpoint_state"] != "disabled"
            or attributes["source_graph_padding"] != "none"):
        raise NotImplementedError("K3 ordinary decode requires its exact unpadded0.30 BF16/FP32 source")
    rows, heads, dim, width = require_shape(shape, ("rows", "heads", "head_dim", "conv_kernel"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128 or width != 4:
        raise NotImplementedError("K3 ordinary decode binds its128-wide TP1/2/4/8/16 four-tap source")
    if rows >= 2**31:
        raise NotImplementedError("K3 decode batch must fit its signed int32 source field")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) != rows + 1
            or any(type(value) is not int or value != i for i, value in enumerate(starts))):
        raise ValueError("K3 ordinary decode requires one actual query token per request")
    slots = require_positive_integer(attributes, "state_pool_slots")
    indices = attributes["state_indices"]
    if (not isinstance(indices, (tuple, list)) or len(indices) != rows
            or any(type(index) is not int or not 0 < index < min(slots, 2**31) for index in indices)
            or len(set(indices)) != rows):
        raise ValueError("K3 ordinary decode requires distinct positive INT32 live state IDs")
    lower_bound = attributes["gate_lower_bound"]
    if lower_bound is not None and (type(lower_bound) not in (int, float)
            or not math.isfinite(lower_bound) or not -5 <= lower_bound < 0):
        raise ValueError("K3 decode gate bound must be absent or finite in[-5,0)")
    norm_eps = attributes["source_norm_eps"]
    if type(norm_eps) not in (int, float) or norm_eps != 1e-5:
        raise ValueError("K3 decode requires its source gated-norm epsilon1e-5")
    layout = attributes["source_conv_state_layout"]
    if layout not in ("SD", "DS"):
        raise NotImplementedError("K3 decode requires its actual SD or DS conv-state layout")
    channel = require_positive_integer(attributes, "source_conv_channel_stride_elements")
    tap = require_positive_integer(attributes, "source_conv_tap_stride_elements")
    expected = (1, 3 * heads * dim) if layout == "SD" else (3, 1)
    if (channel, tap) != expected:
        raise ValueError("K3 decode conv-state inner strides disagree with the actual source layout")
    conv_slot = require_positive_integer(attributes, "source_conv_slot_stride_elements")
    state_slot = require_positive_integer(attributes, "source_state_slot_stride_elements")
    if conv_slot < 9 * heads * dim or state_slot < heads * dim * dim or state_slot % 4:
        raise ValueError("K3 decode cache slots must be non-overlapping and state float4 aligned")
    if max(2 * slots * conv_slot, 4 * slots * state_slot) >= 2**63:
        raise NotImplementedError("K3 decode cache slot addressing exceeds signed int64")
    return rows, heads, dim


def select_kda_native_decode_launch(shape, dtype, attributes, hardware):
    """The complete ordinary fused native call with preloaded FP32 weights.

    Bias is absent in K3. Beta sigmoid, conv-state update and output norm
    are always enabled by its caller. Pool row gaps and packed projection
    row strides remain caller facts, not inferred alignment/copy launches.
    """
    rows, heads, dim = _ordinary_decode_domain(shape, dtype, attributes, hardware)
    if heads == 6:
        raise NotImplementedError("K3 native fused decode has no six-head specialization")
    if (attributes["source_component"] != "vllm_0_30_kimi_k3_native_decode"
            or attributes["source_decode_backend"] != "native"
            or type(attributes["source_configured_num_spec"]) is not int
            or attributes["source_configured_num_spec"] != 0
            or attributes["source_native_decode_op_available"] is not True
            or attributes["source_decode_weight_buffers_ready"] is not True
            or attributes["source_decode_weight_layout"] != "planes_taps_channels_fp32"
            or attributes["source_decode_norm_weight_layout"] != "contiguous_fp32_128"
            or attributes["source_output_copy_alias"] != "same_view"):
        raise NotImplementedError("K3 native decode requires the actual op,loaded source buffers and direct output")
    x_stride = require_positive_integer(attributes, "source_x_row_stride_elements")
    beta_stride = require_positive_integer(attributes, "source_beta_row_stride_elements")
    onorm_stride = require_positive_integer(attributes, "source_output_gate_row_stride_elements")
    if x_stride < 3 * heads * dim or (x_stride, beta_stride) != (beta_stride, onorm_stride):
        raise ValueError("K3 native inputs must retain the actual shared projection row stride")
    if 2 * rows * x_stride >= 2**63:
        raise NotImplementedError("K3 native projection addressing exceeds signed int64")
    bounded = attributes["gate_lower_bound"] is not None
    layout = attributes["source_conv_state_layout"].lower()
    decay = "bounded" if bounded else "softplus"
    return LaunchConfig(f"vllm_0_30_sm90_kda_native_decode_h{heads}_{layout}_{decay}",
        (rows, heads, 1), (256, 1, 1), parameters={"dynamic_shared_bytes": 0.0,
            "source_programmatic_stream_serialization": 1.0,
            "source_update_conv_state": 1.0, "source_apply_beta_sigmoid": 1.0,
            "source_apply_output_norm": 1.0, "source_use_lower_bound": float(bounded),
            "source_conv_channel_stride": float(attributes["source_conv_channel_stride_elements"]),
            "source_conv_tap_stride": float(attributes["source_conv_tap_stride_elements"])})


def select_kda_ordinary_decode_recipe(shape, dtype, attributes, hardware):
    """Resolve the normal backend and return its complete physical core.

    vLLM returns early for explicitly requested Triton, and rejects native
    six-head/configured-spec variants before probing the installed op. The
    availability key is therefore required only when that source probe is
    actually reached. This permits a forced-Triton observation to be checked
    without inventing the unrecorded wheel's native-op availability.
    """
    from dataclasses import replace

    from src.kernel.launch_config.nvidia_h200.kda_launch import (
        select_kda_decode_launch, select_kda_output_copy_transfer,
    )

    rows, heads, _ = _ordinary_decode_domain(shape, dtype, attributes, hardware)
    requested = attributes["source_requested_decode_backend"]
    if requested not in ("auto", "native", "triton", "flashinfer"):
        raise ValueError("unknown K3 source decode backend")
    configured_spec = attributes["source_configured_num_spec"]
    if type(configured_spec) is not int or configured_spec < 0:
        raise ValueError("K3 configured speculative token count must be a non-negative integer")
    if requested == "flashinfer":
        raise NotImplementedError("K3 FlashInfer fused decode requires SM100/103,not H200")
    native = False
    if requested != "triton" and heads != 6 and configured_spec == 0:
        available = attributes["source_native_decode_op_available"]
        if type(available) is not bool:
            raise TypeError("K3 installed native op availability must be an actual boolean")
        native = available
    if requested == "native" and not native:
        raise NotImplementedError("K3 requested native backend is unavailable for the actual runtime/source domain")
    if native:
        ready = attributes["source_decode_weight_buffers_ready"]
        if type(ready) is not bool:
            raise TypeError("K3 decoded weight buffer readiness must be an actual boolean")
        native = ready
    actual = "native" if native else "triton"
    if attributes["source_decode_backend"] != actual:
        raise ValueError("K3 executed decode backend contradicts its source dispatch facts")
    if native:
        return (("fused_native_decode", select_kda_native_decode_launch(shape, dtype, attributes, hardware)),)
    if (attributes["source_component"] != "vllm_0_30_kimi_k3_triton_decode"
            or attributes["source_output_norm_implementation"] != "forward_cuda"
            or attributes["source_output_norm_alias"] != "same_view"
            or attributes["source_conv_weight_layout"] != "channels_taps_fp32"):
        raise NotImplementedError("K3 Triton decode requires its actual CUDA-norm/conv-weight source")
    if rows * heads > 65535:
        raise NotImplementedError("K3 packed recurrence grid.y exceeds CUDA axis capacity")
    variant = "kimi_k3_kda_decode"
    launches = tuple((stage, replace(select_kda_decode_launch(shape, dtype, variant,
        {**attributes, "stage": stage}, hardware), execution_variant=None))
        for stage in ("convolution", "recurrent", "output_norm"))
    transfer = select_kda_output_copy_transfer(shape, dtype,
        {**attributes, "stage": "output_copy", "source_copy_alias": attributes["source_output_copy_alias"],
         "source_input_layout": "contiguous_aligned_16"}, hardware)
    copy = (("output_copy", transfer),) if transfer is not None else ()
    return (*launches[:2], *copy, launches[2])
