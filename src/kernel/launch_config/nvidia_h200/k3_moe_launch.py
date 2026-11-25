"""Prepared K3 routing and portable latent-tail suffix on vLLM0.30 SM90.

KimiK3MoE supplies FP32 gate logits and correction bias. GroupedTopKRouter
enters grouped_topk with its actual fused flag;grouped_topk_kernels.cu
owns the single-group896/top16 tier and PDL decision. Gate/down-projection,
expert kernels and latent/shared-output transforms are separate producers.
SM90 excludes the SM100-only tcgen05 tail. Its portable tier choice and
prepared local merge are bound below, independently of GEMM/collectives.
"""

import math

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def select_k3_latent_moe_tail_tier(shape, dtype, attributes, hardware):
    """Exact portable H200 tier from K3's LatentMoERunner.

    __init__ enables tcgen05 tail fusion only for capability family100.
    H200 is SM90, so even TP8/16 and tiny batches cannot enter that tier.
    The actual shared-expert stream flags select overlap or column-parallel;
    neither tier proves any collective/GEMM/compiler resource or price.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 latent MoE tail requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_latent_moe_portable_tail"
            or attributes["source_tail_runner"] != "fused_path_with_replicated_routed_transform"):
        raise NotImplementedError("K3 latent MoE tail needs exact0.30 portable runner source")
    rows = shape["rows"]
    if type(rows) is not int or not 0 <= rows < 2**31:
        raise ValueError("K3 tail needs actual nonnegative physical token rows")
    threshold, disabled = attributes["source_shared_stream_token_threshold"], attributes["source_disable_shared_stream"]
    if type(threshold) is not int or type(disabled) is not bool:
        raise ValueError("K3 tail stream flags must be actual integer threshold/boolean disable values")
    return "allreduce_overlap" if rows <= threshold and not disabled else "column_parallel"


def select_k3_latent_moe_tail_epilogue_recipe(shape, dtype, attributes, hardware):
    """Complete local merge after this tier's projection/reduction producers.

    Overlap tier does result.add_(reduced_shared_output), then only a trim
    view. Column-parallel adds the shared shard inside addmm_'s beta epilogue,
    so has no separate local merge submission; its final reduction is a
    distinct caller requirement. Prepared norm/latent/shared collectives and
    projection GEMM are NOT described by this local suffix. No SM100 kernel
    is selected and no resource/rate is inferred for external producers.
    """
    tier = select_k3_latent_moe_tail_tier(shape, dtype, attributes, hardware)
    if (attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"
            or attributes["source_tail_merge_inputs"] != "actual_tier_projection_and_collective_outputs"
            or attributes["source_tail_merge_layout"] != "contiguous_bf16_full_hidden_rows"):
        raise NotImplementedError("K3 tail merge requires exact prepared Torch2.13 tier outputs")
    hidden, = require_shape(shape, ("hidden",))
    rows, elements = shape["rows"], shape["rows"] * hidden
    if tier == "column_parallel" or rows == 0:
        return ()
    alignments = attributes["source_tail_add_address_alignments"]
    if (not isinstance(alignments, (tuple, list)) or len(alignments) != 3
            or any(type(value) is not int or value not in (2, 4, 8, 16) for value in alignments)
            or tuple(alignments[:2]) != (16, 16)):
        raise ValueError("K3 in-place tail add needs fresh16B GEMM output/left and actual right BF16 pointer alignments")
    if elements >= 2**31 or 2 * (elements - 1) >= 2**31 - 1:
        raise NotImplementedError("K3 tail add needs every source split-iterator launch")
    # Binary alpha=1 add uses same-dtype no-cast CUDALoops:6B I/O,
    # eight values/thread,128 threads and SM90's BF16 vec8 ceiling.
    vec = min(alignments) // 2
    return (("reduced_shared_output_add", LaunchConfig(f"pytorch_2_13_sm90_bf16_add_vec{vec}",
        (ceil_div(elements, 1024), 1, 1), (128, 1, 1),
        parameters={"dynamic_shared_bytes": 0, "source_values_per_thread": 8})),)


def select_k3_modular_router_recipe(shape, dtype, attributes, hardware):
    """Complete selected route after prepared gate projection,not full MoE.

    Sigmoid,biased selection,unbiased weight normalization and scaling all
    execute in one custom CUDA op. The source ordered tier list chooses
    <896,16>;UseTunedBlockPath keeps it on256 threads even above1024 rows.
    It outputs FP32 weights/INT32 IDs directly,no scalar scale/cast helpers.
    The unfused torch.compile branch needs its actual generated program.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 routing source requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    rows, experts, topk = require_shape(shape, ("rows", "experts", "topk"))
    if (experts, topk) != (896, 16):
        raise NotImplementedError("K3 router binds its actual896-expert/top16 source domain")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_modular_router"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_correction_bias"] != "contiguous_fp32"
            or attributes["source_score_function"] != "sigmoid"
            or attributes["source_num_expert_group"] != 1
            or type(attributes["source_num_expert_group"]) is not int
            or attributes["source_topk_group"] != 1
            or type(attributes["source_topk_group"]) is not int
            or attributes["source_renormalize"] is not True
            or attributes["source_output_ids_dtype"] != "int32"
            or attributes["source_output_weights_dtype"] != "fp32"
            or attributes["source_eplb_and_shared_slots"] != "none"):
        raise NotImplementedError("K3 routing requires exact0.30 modular sigmoid/bias/single-group source facts")
    fused = attributes["source_use_fused_grouped_topk"]
    if type(fused) is not bool:
        raise TypeError("K3 grouped routing requires the actual fused environment flag")
    if not fused:
        raise NotImplementedError("K3 unfused routing requires its actual Torch compiled program")
    scale = attributes["source_routed_scaling_factor"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("K3 routing requires the actual finite scaling factor")
    if rows * experts >= 2**31:
        raise NotImplementedError("K3 routing exceeds the bound source signed-index domain")
    return (("sigmoid_bias_top16", LaunchConfig("vllm_0_30_sm90_k3_sigmoid_bias_single_group_e896_k16_block",
        (rows, 1, 1), (256, 1, 1), parameters={"dynamic_shared_bytes": 0.0,
            "source_programmatic_stream_serialization": float(rows <= 16)})),)
