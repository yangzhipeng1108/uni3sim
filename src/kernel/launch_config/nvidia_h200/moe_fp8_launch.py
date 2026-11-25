"""vLLM0.30 Triton FP8 block128 expert source sequences on H200.

fused_moe.py::fused_experts_impl quantizes the input, prepares assignment,
runs W1, activation, quantizes the routed activation, runs W2 and sums.
utils.py::_fp8_quantize selects per_token_group_quant_fp8; contiguous inputs
take the CUDA wrapper in libtorch_stable/quantization/w8a8/fp8/
per_token_group_quant.cu. This is the ordinary shared-memory quantizer, not
the separate packed-register/DeepGEMM quantizer or a vLLM0.24 launch rule.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.nvidia_h200.moe_aux_launch import (
    select_source_moe_aux_launch, source_alignment_stages)
from src.kernel.launch_config.nvidia_h200.moe_source_launch import select_source_moe_launch
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def select_source_fp8_group_quant_launch(shape, dtype, attributes, hardware):
    """One row-major, contiguous group128 CUDA quantization submission.

    The source rounds policy rows to even before choosing the largest
    divisor among16/8/4/2/1 groups per block. The actual group count still
    controls the grid and CHECK_BOUNDS specialization. Shared memory holds
    each block's source values; register/static shared remain unbound.
    The SM90 source enables programmatic stream serialization. No device
    overlap, synchronization duration or new service price is inferred.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 FP8 group quantization requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_scale_layout"] != "row_major"
            or attributes["source_quant_output_dtype"] != "fp8_e4m3"):
        raise NotImplementedError("FP8 quantization requires its explicit0.30 contiguous row-major source")
    ue8m0 = attributes["source_scale_ue8m0"]
    if type(ue8m0) is not bool:
        raise TypeError("source_scale_ue8m0 must identify the actual scale rounding branch")
    rows, columns, group = require_shape(shape, ("rows", "columns", "group_size"))
    if group != 128 or columns % group:
        raise NotImplementedError("source FP8 group quantization requires complete128-value groups")
    groups_per_row = columns // group
    groups = rows * groups_per_row
    if groups >= 2**31:
        raise NotImplementedError("CUDA source num_groups must fit its signed-int host parameter")
    from src.kernel.launch_config.nvidia_h200.fp8_quant_launch import group_quant_cuda_geometry
    blocks, threads, shared, per_block = group_quant_cuda_geometry(rows, groups_per_row, group, 2)
    bounds = "bounds" if groups % per_block else "full"
    scale = "ue8m0" if ue8m0 else "fp32"
    return LaunchConfig(f"vllm_0_30_sm90_group128_fp8_quant_{dtype}_{scale}_g{per_block}_{bounds}_pdl",
        (blocks, 1, 1), (threads, 1, 1),
        parameters={"dynamic_shared_bytes": float(shared)},
        execution_variant="fp8_group_quant")


def select_source_fp8_silu_quant_launch(shape, dtype, attributes, hardware):
    """TritonExperts' fused SiLU+multiply+row-major FP8 quantization branch."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 fused SiLU FP8 source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_scale_layout"] != "row_major"
            or attributes["source_quant_output_dtype"] != "fp8_e4m3"
            or attributes["source_scale_ue8m0"] is not False):
        raise NotImplementedError("fused SiLU quantization requires its0.30 FP32-scale source branch")
    rows, columns, group = require_shape(shape, ("rows", "columns", "group_size"))
    if group != 128 or columns % group or columns >= 2**31 or rows >= 2**31 or columns // group > 65535:
        raise NotImplementedError("fused SiLU quantization requires source128-value groups and CUDA grid bounds")
    # fused_silu_mul_block_quant.cu launches128 scalar threads per
    # (routed row,128-column group), with no dynamic shared memory.
    return LaunchConfig(f"vllm_0_30_sm90_silu_mul_group128_fp8_quant_{dtype}",
        (rows, columns // group, 1), (128, 1, 1), parameters={"dynamic_shared_bytes": 0.0},
        execution_variant="fp8_group_quant")


def select_source_fp8_moe_recipe(shape, dtype, attributes, hardware):
    """Complete selected Triton body, optionally including routing.

    The caller identifies fused_experts_impl or modular TritonExperts with
    NoDPEP prepare/finalize. Modular LoRA/deferred preparation and distinct
    finalizer output copies require additional producers. DeepGEMM and
    FlashInfer have other programs. A physical stage tuple is not an
    aggregate KernelSpec launch or source-aware typed price acceptance.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("FP8 expert source recipe requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    implementation = attributes["source_moe_implementation"]
    if (implementation not in ("fused_experts_impl", "modular_triton_no_dp_ep")
            or attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["routed_weight_dtype"] != "fp8"
            or attributes["source_quantization_emulation"] is not False
            or attributes["source_per_channel_quant"] is not False
            or attributes["source_activation"] != "silu"
            or attributes["source_apply_router_weight_on_input"] is not False
            or attributes["source_bias"] != "none"):
        raise NotImplementedError("FP8 recipe requires the explicit functional block128 SiLU source program")
    if implementation == "modular_triton_no_dp_ep":
        if (dtype != "bf16" or attributes["source_lora"] != "none"
                or attributes["source_defer_input_quant"] is not False
                or attributes["source_fused_output_alias"] != "same_tensor"):
            raise NotImplementedError("modular FP8 recipe requires BF16,no LoRA,eager input quant and proven same-tensor finalizer")
    weight_block = attributes["source_weight_block_shape"]
    if (not isinstance(weight_block, (tuple, list)) or len(weight_block) != 2
            or any(type(value) is not int for value in weight_block)
            or tuple(weight_block) != (128, 128)):
        raise NotImplementedError("FP8 recipe requires128x128 weight scale blocks")
    has_map = attributes["source_has_expert_map"]
    ep_size = attributes["expert_parallel_size"]
    if type(has_map) is not bool or type(ep_size) is not int or ep_size <= 0:
        raise ValueError("FP8 source requires explicit boolean expert map and positive integer EP size")
    if has_map != (ep_size > 1):
        raise ValueError("FP8 source expert map must match actual EP ownership")
    if implementation == "modular_triton_no_dp_ep" and has_map:
        raise NotImplementedError("modular NoDPEP source cannot carry an expert map")
    tokens, experts, topk, hidden, intermediate = require_shape(shape, (
        "tokens", "experts", "topk", "hidden", "intermediate"))
    if topk > experts:
        raise ValueError("topk exceeds global expert count")
    if hidden % 128 or intermediate % 128:
        raise NotImplementedError("complete FP8 recipe needs both W1/W2 activation widths divisible by128")
    input_quant = select_source_fp8_group_quant_launch(
        {"rows": tokens, "columns": hidden, "group_size": 128}, dtype, attributes, hardware)
    router_state = attributes["source_router_state"]
    if router_state == "softmax_logits_ready":
        router = (("moe_topk", select_source_moe_aux_launch(shape,
            attributes["routing_input_dtype"], "moe_topk", attributes, hardware)),)
    elif router_state == "prepared_topk_weights_and_ids":
        router = ()
    else:
        raise NotImplementedError("FP8 recipe requires explicit router producer ownership")
    # The naive branch's torch.full writes this unique source-owned scalar.
    assignment_attrs = {**attributes, "fill_role": "num_tokens_post_padded"}
    assignment = tuple((stage, select_source_moe_aux_launch(shape, dtype, stage,
        assignment_attrs, hardware)) for stage in source_alignment_stages(tokens, topk, experts, has_map))
    w1 = select_source_moe_launch(shape, dtype, "w1", attributes, hardware)
    intermediate_shape = {"rows": tokens * topk, "columns": intermediate, "group_size": 128}
    if implementation == "modular_triton_no_dp_ep" and attributes["source_scale_ue8m0"] is False:
        activation_stages = (("moe_silu_fp8_quant", select_source_fp8_silu_quant_launch(
            intermediate_shape, dtype, attributes, hardware)),)
    else:
        activation = select_source_moe_aux_launch(shape, dtype, "moe_act_and_mul", attributes, hardware)
        intermediate_quant = select_source_fp8_group_quant_launch(intermediate_shape, dtype, attributes, hardware)
        activation_stages = (("moe_act_and_mul", activation), ("intermediate_fp8_quant", intermediate_quant))
    w2 = select_source_moe_launch(shape, dtype, "w2", attributes, hardware)
    reduction = select_source_moe_aux_launch(shape, dtype, "moe_sum", attributes, hardware)
    if has_map:
        from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch
        output_zero = (("moe_zero_output", select_torch213_fill_launch(
            {"element_count": tokens * topk * hidden}, dtype, attributes, hardware)),)
    else:
        output_zero = ()
    return (*router, ("input_fp8_quant", input_quant), *assignment, ("w1", w1),
        *activation_stages, *output_zero, ("w2", w2), ("moe_sum", reduction))
