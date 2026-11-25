"""Versioned vLLM SM90 expert alignment, activation and sum host launches.

Source: csrc/libtorch_stable/moe/moe_align_sum_kernels.cu and
activation_kernels.cu. Shape dtype follows the existing MoE activation ABI;
the alignment's actual INT32 index dtype is a separate mandatory source fact.
"""

import math
import struct

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


SOURCE_MOE_AUX_VARIANTS = frozenset({
    "moe_align", "moe_align_small_batch", "moe_count_and_sort",
    "moe_act_and_mul", "moe_sum", "moe_fill",
    "moe_topk",
    "moe_reduce",
})
SOURCE_TOPK_EXPERT_COUNTS = frozenset({1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                     192, 320, 384, 448, 576})


def situ_plain_geometry(rows, intermediate):
    """activation_kernels.cu::situ_and_mul's scalar host launch."""
    if rows >= 2**31:
        raise NotImplementedError("plain SiTU rows exceed CUDA grid.x")
    return (rows, 1, 1), (min(intermediate, 1024), 1, 1)


def select_situ_launch(shape, dtype, attributes, hardware):
    """Complete selected SiTU CUDA op,including masked and fused FP8 forms.

    Source activation.py and fused_moe/activation.py own the actual entry;
    activation_kernels.cu owns host geometry. Masked direct and modular
    entries differ: only the latter adds up to32 token lanes. Counts change
    live row work,not host geometry. Fused quant rounds each activation to
    the input dtype in registers before absmax,does not materialize BF16
    intermediates,and fills only skipped FP32 scales with1 inside the same
    kernel. Skipped FP8/output rows are not initialized by this operation.
    Compiler registers/static shared and typed service pricing are unbound.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SiTU source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_input_layout"] != "contiguous_aligned_16"):
        raise NotImplementedError("SiTU requires exact0.30 contiguous gate/up source")
    intermediate, = require_shape(shape, ("intermediate",))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("SiTU rows must be the non-negative physical buffer row extent")
    beta, linear = attributes["source_situ_beta"], attributes["source_situ_linear_beta"]
    if (isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta)
            or beta <= 0 or beta > 3.4028234663852886e38):
        raise ValueError("SiTU beta must be a positive finite FP32-representable source value")
    if linear is not None and (isinstance(linear, bool) or not isinstance(linear, (int, float))
            or not math.isfinite(linear) or abs(linear) > 3.4028234663852886e38):
        raise ValueError("SiTU linear beta must be an actual finite value orNone")
    beta32 = struct.unpack("f", struct.pack("f", beta))[0]
    linear32 = -1.0 if linear is None else struct.unpack("f", struct.pack("f", linear))[0]
    if beta32 <= 0:
        raise ValueError("SiTU beta must remain positive after the host FP32 conversion")
    if intermediate >= 2**30 or rows * 2 * intermediate >= 2**63:
        raise NotImplementedError("SiTU input exceeds bound signed source addressing")
    component, counts = attributes["source_component"], attributes["source_valid_counts_layout"]
    if component not in ("vllm_0_30_situ_and_mul_cuda", "vllm_0_30_apply_moe_activation_situ",
                         "vllm_0_30_masked_situ_and_mul", "vllm_0_30_situ_and_mul_quant"):
        raise NotImplementedError("SiTU requires an exact supported source entry")
    if counts not in ("none", "contiguous_int32_flat_scalar", "contiguous_int32_per_expert"):
        raise NotImplementedError("SiTU requires actual absent orcontiguousINT32 valid-row counts")
    if counts != "none":
        count_shape = attributes["source_valid_counts_shape"]
        if (not isinstance(count_shape, (tuple, list))
                or any(type(value) is not int for value in count_shape)):
            raise ValueError("SiTU valid counts require their actual integer tensor shape")
        if counts == "contiguous_int32_per_expert":
            experts = shape["num_experts"]
            capacity = shape["max_num_tokens"]
            if (type(experts) is not int or type(capacity) is not int or experts < 0 or capacity < 0
                    or experts * capacity != rows or tuple(count_shape) != (experts,)):
                raise ValueError("SiTU batched counts must match E*T physical rows andE counts")
        elif tuple(count_shape) != (1,):
            raise ValueError("SiTU flat valid count must be an actual INT32 scalar tensor")
    if component == "vllm_0_30_situ_and_mul_quant":
        if (dtype not in ("bf16", "fp16") or attributes["source_output_dtype"] != "fp8_e4m3fn"
                or attributes["source_scale_layout"] != "contiguous_fp32_rows_groups"
                or counts == "contiguous_int32_per_expert"
                or attributes["source_group_size"] != 128 or intermediate % 128):
            raise NotImplementedError("SiTU quant requires BF16/FP16,group128 FP8e4m3fn andFP32 row-major scales")
        topk, = require_shape(shape, ("topk",))
        if topk >= 2**31 or rows >= 2**63 // topk:
            raise NotImplementedError("SiTU quant valid-row bound exceeds source signed multiplication")
        if rows == 0:
            return None
        pipelined = intermediate == 3072 and beta32 == 4.0 and linear32 == 25.0
        # This is the original host's fixed SM_COUNT*BLOCKS_PER_SM launch,
        # not an occupancy estimate or hardware-count-derived grid.
        return LaunchConfig("vllm_0_30_sm90_situ_quant_group128_" + ("pipelined_d3072" if pipelined else "scalar"),
            (132 * 8, 1, 1), (256 if pipelined else min(intermediate // 128, 32) * 32, 1, 1),
            parameters={"dynamic_shared_bytes": float(8 * 4 * 256 * 2 if pipelined else 0)})
    if component == "vllm_0_30_situ_and_mul_cuda" and counts != "none":
        raise ValueError("SituAndMul.forward_cuda has no valid-count argument")
    if component == "vllm_0_30_masked_situ_and_mul" and counts != "contiguous_int32_per_expert":
        raise ValueError("direct masked SiTU consumes a3D expert buffer andper-expert counts")
    if rows == 0:
        return None
    if counts == "none":
        grid, block = situ_plain_geometry(rows, intermediate)
        suffix = "plain"
    else:
        capacity = shape["max_num_tokens"] if counts == "contiguous_int32_per_expert" else rows
        if capacity >= 2**31:
            raise NotImplementedError("masked SiTU token capacity exceeds source signed int")
        if counts == "contiguous_int32_per_expert" and shape["num_experts"] > 65535:
            raise NotImplementedError("masked SiTU expert count exceeds CUDA grid.y")
        lanes = 1 if component == "vllm_0_30_masked_situ_and_mul" else min(capacity, 32)
        grid = ((ceil_div(intermediate, 256), shape["num_experts"], lanes)
                if counts == "contiguous_int32_per_expert" else (ceil_div(intermediate, 256), lanes, 1))
        block, suffix = (256, 1, 1), "masked_batched" if counts == "contiguous_int32_per_expert" else "masked_flat"
        if component == "vllm_0_30_masked_situ_and_mul":
            suffix += "_direct"
    return LaunchConfig("vllm_0_30_sm90_situ_" + suffix, grid, block,
        parameters={"dynamic_shared_bytes": 0.0})


def select_clamped_swiglu_launch(shape, dtype, attributes, hardware):
    """Complete source clamped SwiGLU,with actual caller/mapping dispatch.

    activation.silu_and_mul_with_clamp selects the pad-aware Triton path
    only when both IDs and expert_map exist. utils.swiglu_limit_func selects
    it whenever IDs exist,including no-map calls. Native CUDA on SM90 uses
    128-bit vectors;the256-bit path is explicitly gated to SM100+. rows is
    the flattened token/expert-slot extent,not the original request batch.
    Return None for the source's zero-row no-op. No missing resource price
    or output write is inferred for masked/non-local slots.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("clamped SwiGLU source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    intermediate, = require_shape(shape, ("intermediate",))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("clamped SwiGLU rows must be a non-negative integer")
    component = attributes["source_component"]
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or component not in ("vllm_0_30_apply_moe_activation", "vllm_0_30_swiglu_limit_func")
            or attributes["source_input_layout"] != "contiguous_aligned_16"):
        raise NotImplementedError("clamped SwiGLU requires its exact0.30 contiguous gate/up source")
    ids, mapping = attributes["source_topk_ids"], attributes["source_expert_map"]
    if ids not in ("none", "contiguous_int32") or mapping not in ("none", "contiguous_int32"):
        raise NotImplementedError("clamped SwiGLU requires actual absent orcontiguousINT32 IDs/map")
    if mapping != "none" and ids == "none":
        raise ValueError("clamped SwiGLU expert map requires the actual routed IDs")
    limit = attributes["source_clamp_limit"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        raise ValueError("clamped SwiGLU requires the actual finite clamp limit")
    if rows == 0:
        return None
    pad_aware = ids != "none" and (component == "vllm_0_30_swiglu_limit_func" or mapping != "none")
    if pad_aware:
        return LaunchConfig("vllm_0_30_sm90_swiglu_limit_pad_aware", (min(rows, 256), ceil_div(intermediate, 1024), 1),
            (128, 1, 1), parameters={"source_has_expert_map": float(mapping != "none"),
                "source_has_limit": float(limit > 0)})
    vector = 8 if intermediate % 8 == 0 else 1
    return LaunchConfig(f"vllm_0_30_sm90_silu_mul_clamp_vec{vector}", (rows, 1, 1),
        (min(intermediate // vector, 1024), 1, 1), parameters={"dynamic_shared_bytes": 0.0},
        execution_variant="silu")


COUNT_SORT_VALID_GLOBAL_IDS = "vllm_0_30_topk_softmax_global_int32_no_padding"
COUNT_SORT_VALID_GLOBAL_ALGORITHM = "vllm_0_30_sm90_count_sort_valid_global_int32"


def source_alignment_stages(tokens, topk, experts, has_expert_map):
    """Actual submission sequence for unquantized _prepare_expert_assignment."""
    if type(has_expert_map) is not bool:
        raise TypeError("source_has_expert_map must be boolean")
    if not has_expert_map and tokens * topk * 4 <= experts:
        return ("moe_fill",)
    if tokens * topk < 1024 and experts <= 64:
        return ("moe_align_small_batch",)
    return ("moe_align", "moe_count_and_sort")


def select_source_moe_aux_launch(shape, dtype, variant, attributes, hardware):
    if attributes["model_execution_version"] == "vllm_0_11_0" and variant != "moe_topk":
        from src.kernel.launch_config.nvidia_h200.moe_legacy_launch import select_legacy_moe_aux_launch
        return select_legacy_moe_aux_launch(shape, dtype, variant, attributes, hardware)
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 MoE auxiliary source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32") if variant == "moe_topk" else ("bf16", "fp16"))
    if attributes["model_execution_version"] not in ("vllm_0_30_0", "vllm_0_11_0"):
        raise NotImplementedError("MoE auxiliary source runtime is not bound")
    if variant not in SOURCE_MOE_AUX_VARIANTS:
        raise NotImplementedError("MoE auxiliary source variant is not bound")
    if variant == "moe_reduce":
        raise NotImplementedError("vLLM0.30 uses native CUDA MoE sum rather than Torch reduction")
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    if variant == "moe_topk":
        if attributes["source_routing"] != "softmax":
            raise NotImplementedError("source topk requires its ordinary softmax routing program")
        input_dtype = attributes["routing_input_dtype"]
        if attributes["model_execution_version"] == "vllm_0_11_0" and input_dtype != "fp32":
            raise NotImplementedError("vLLM0.11 topk source requires FP32 gating logits")
        require_dtype(input_dtype, ("bf16", "fp16", "fp32"))
        power_two = experts <= 512 and experts & (experts - 1) == 0
        if experts not in SOURCE_TOPK_EXPERT_COUNTS:
            raise NotImplementedError("source topk requires separate softmax/topk launches for this expert count")
        dtype_bytes = 4 if input_dtype == "fp32" else 2
        maximum_load = 16 if power_two else 8 if input_dtype == "fp32" else 4
        values_per_load = min(maximum_load, dtype_bytes * experts) // dtype_bytes
        values_per_thread = max(1, experts // (values_per_load * 32)) * values_per_load
        threads_per_row = experts // values_per_thread
        rows_per_warp = 32 // threads_per_row
        grid, block = (ceil_div(ceil_div(tokens, rows_per_warp), 4), 1, 1), (32, 4, 1)
    elif variant in ("moe_align", "moe_align_small_batch", "moe_count_and_sort", "moe_fill"):
        if attributes["source_topk_id_dtype"] != "int32":
            raise NotImplementedError("MoE alignment source requires INT32 routed IDs")
        stages = source_alignment_stages(tokens, topk, experts, attributes["source_has_expert_map"])
        if variant not in stages:
            raise ValueError("MoE alignment stage does not belong to the source submission sequence")
        if variant != "moe_fill" and ceil_div(experts, 32) * 32 >= 1024:
            raise ValueError("source alignment requires padded expert count below1024")
        if variant == "moe_fill":
            if attributes["fill_role"] != "num_tokens_post_padded":
                raise ValueError("naive assignment fills only num_tokens_post_padded")
            grid, block = (1, 1, 1), (128, 1, 1)
        elif variant == "moe_align_small_batch":
            grid, block = (1, 1, 1), (256 + max(experts, 32), 1, 1)
        elif variant == "moe_align":
            grid, block = (2, 1, 1), (1024, 1, 1)
        else:
            grid, block = (1, min(ceil_div(tokens * topk, 256), 65535), 1), (256, 1, 1)
    else:
        if attributes["source_input_layout"] != "contiguous_aligned_16":
            raise NotImplementedError("MoE vector auxiliary source requires aligned contiguous operands")
        if variant == "moe_act_and_mul":
            intermediate, = require_shape(shape, ("intermediate",))
            vector_width = 8 if intermediate % 8 == 0 else 1
            grid = (tokens * topk, 1, 1)
            block = (min(intermediate // vector_width, 1024), 1, 1)
        else:
            hidden, = require_shape(shape, ("hidden",))
            if hidden % 8:
                grid, block = (tokens, 1, 1), (min(hidden, 1024), 1, 1)
            else:
                grid = (min(ceil_div(tokens * (hidden // 8), 256), 65535), 1, 1)
                block = (256, 1, 1)
    runtime = "0_11" if attributes["model_execution_version"] == "vllm_0_11_0" else "0_30"
    parameters = {}
    if runtime == "0_11":
        from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_moe_aux_resources
        parameters = select_legacy_moe_aux_resources(shape, dtype, variant, attributes)
    algorithm = f"vllm_{runtime}_sm90_source_{variant}"
    if variant == "moe_count_and_sort" and "source_topk_id_domain" in attributes:
        if (attributes["model_execution_version"] != "vllm_0_30_0"
                or attributes["source_topk_id_domain"] != COUNT_SORT_VALID_GLOBAL_IDS
                or attributes["source_has_expert_map"] is not False):
            raise ValueError("count-sort valid-global source requires exact0.30 unmasked INT32 IDs without an expert map")
        algorithm = COUNT_SORT_VALID_GLOBAL_ALGORITHM
    return LaunchConfig(algorithm, grid, block,
                        parameters=parameters, execution_variant=variant)
