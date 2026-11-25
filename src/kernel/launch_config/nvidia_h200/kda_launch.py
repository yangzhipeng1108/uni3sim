"""Pinned Kimi-K3 ordinary Triton decode launch stages on H200.

The packed recurrence explicitly uses four warps, BV32 and two stages in
models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py. Gated norm in
third_party/flash_linear_attention/ops/kda.py explicitly uses BT16/eight warps.
Convolution's non-autotuned128-thread compiler default is bound by all27
retained kc_kda_models__v030 CONV_DECODE window_01 launch observations.
Its grid independently follows causal_conv1d.py::causal_conv1d_update.
This is observation-backed geometry, not independent compiler-resource
verification. Copy.cu supplies the separate output-copy transfer selector;
its same-view branch has no device submission and never receives a fake CTA.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_shape


def select_kda_decode_launch(shape, dtype, variant, attributes, hardware):
    # These fixed host rules do not read any device properties. The public
    # dispatcher validates HWSpec; contract replay may call the owner without it.
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 SM90 decode source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (variant != "kimi_k3_kda_decode"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_triton_decode"):
        raise NotImplementedError("K3 decode source runtime is not bound")
    rows, heads, dim = require_shape(shape, ("rows", "heads", "head_dim"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128:
        raise NotImplementedError("K3 decode source requires its128-wide TP1/2/4/8/16 heads")
    stage = attributes["stage"]
    if stage == "recurrent":
        bv = min(1 << (dim - 1).bit_length(), 32)
        grid, threads = (-(-dim // bv), rows * heads, 1), 128
        algorithm = "vllm_0_30_sm90_kda_packed_decode"
    elif stage == "output_norm":
        grid, threads = (-(-(rows * heads) // 16), 1, 1), 256
        algorithm = "vllm_0_30_sm90_kda_gated_rmsnorm"
    elif stage == "convolution":
        # Packed q/k/v contributes3*heads*head_dim channels. The wrapper
        # launches (batch,cdiv(dim,BLOCK_N)) with fixed BLOCK_N256 and no
        # num_warps autotuning. Versioned H200 observation binds128 threads;
        # do not copy its register count to unobserved compiler specializations.
        grid, threads = (rows, -(-(3 * heads * dim) // 256), 1), 128
        algorithm = "vllm_0_30_sm90_kda_causal_conv1d_update"
    elif stage == "output_copy":
        raise NotImplementedError("K3 output_copy is an ATen MEMCPY/no-op; use select_kda_output_copy_transfer")
    else:
        raise ValueError("unknown K3 decode stage")
    return LaunchConfig(algorithm, grid, (threads, 1, 1), execution_variant=variant)


def select_kda_output_copy_transfer(shape, dtype, attributes, hardware):
    """Torch2.11/2.13 Copy.cu::copy_device_to_device, with explicit alias proof.

    Ordinary decode allocates non_spec_out separately when non_spec_token_start
    is absent, then copies its complete live prefix to core_attn_out. A direct
    out=core_attn_out view instead produces the same-view no-op. Gated norm is
    in-place and its trailing copy_ is also a no-op. The caller supplies the
    proven alias relationship; shape equality cannot establish it.
    """
    from src.kernel.kernel_spec import PhysicalTransferConfig

    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 output-copy source requires H200 hardware")
    require_dtype(dtype, ("bf16",))
    components = ("vllm_0_30_kimi_k3_triton_decode", "vllm_0_30_kimi_k3_triton_fresh_prefill",
                  "vllm_0_30_kimi_k3_triton_mixed", "vllm_0_30_kimi_k3_triton_context_prefill")
    if (attributes["source_component"] not in components or attributes["stage"] != "output_copy"
            or attributes["tensor_iterator_source"] not in ("pytorch_2_11_sm90", "pytorch_2_13_sm90")):
        raise NotImplementedError("K3 output-copy software source is not bound")
    if attributes["source_input_layout"] != "contiguous_aligned_16":
        raise NotImplementedError("K3 output-copy source requires contiguous live prefixes")
    rows, heads, dim = require_shape(shape, ("rows", "heads", "head_dim"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128:
        raise NotImplementedError("K3 output-copy source requires128-wide TP1/2/4/8/16 heads")
    alias = attributes["source_copy_alias"]
    if alias == "same_view":
        return None
    if alias != "distinct_contiguous":
        raise NotImplementedError("K3 output-copy source requires an exact same-view or distinct-contiguous proof")
    return PhysicalTransferConfig(algorithm="stream_copy", bytes=rows * heads * dim * 2,
                                  direction="d2d", link="dram")


def select_kda_prefill_launch(shape, dtype, variant, attributes, hardware):
    """Source-bound outer stages; the multi-kernel chunk body remains unbound.

    causal_conv1d.py fixes BM8/BN256/stages2. The existing36 CONV_PREFILL
    windows contain three128-thread convolution launches each, binding the
    compiler warp default without claiming an independent compiler query.
    gather_initial_states.py explicitly selects1024 values/CTA and8 warps.
    Gated output norm uses the same BT16/eight-warps wrapper as decode.
    Scheduler/metadata setup lies outside these individually named stages.
    """
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 prefill source requires H200 hardware")
    require_dtype(dtype, ("bf16",))
    components = {
        "kimi_k3_kda_prefill": ("vllm_0_30_kimi_k3_triton_fresh_prefill", "vllm_0_30_kimi_k3_flashkda_fresh_prefill"),
        "kimi_k3_kda_mixed": ("vllm_0_30_kimi_k3_triton_mixed", "vllm_0_30_kimi_k3_flashkda_mixed"),
        "kimi_k3_kda_context_prefill": ("vllm_0_30_kimi_k3_triton_context_prefill", "vllm_0_30_kimi_k3_flashkda_context_prefill"),
    }
    if variant not in components or attributes["source_component"] not in components[variant]:
        raise NotImplementedError("K3 prefill source wrapper is not bound")
    rows, heads, dim = require_shape(shape, ("rows", "heads", "head_dim"))
    if heads not in (6, 12, 24, 48, 96) or dim != 128:
        raise NotImplementedError("K3 prefill source requires128-wide TP1/2/4/8/16 heads")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) < 2
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != rows
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError("K3 prefill requires complete strictly increasing query segments")
    sequences = len(starts) - 1
    if variant == "kimi_k3_kda_mixed" and sequences == 1:
        raise ValueError("K3 source phase disagrees with its query segments")
    stage = attributes["stage"]
    if stage in ("q_convolution", "k_convolution", "v_convolution"):
        programs = sum(-(-(b - a) // 8) for a, b in zip(starts, starts[1:]))
        grid, threads = (programs, -(-(heads * dim) // 256), 1), 128
        algorithm = "vllm_0_30_sm90_kda_causal_conv1d_fwd"
    elif stage == "initial_state":
        state_row = heads * dim * dim
        grid, threads = (-(-state_row // 1024), sequences, 1), 256
        algorithm = "vllm_0_30_sm90_kda_gather_initial_states"
    elif stage == "output_norm":
        grid, threads = (-(-(rows * heads) // 16), 1, 1), 256
        algorithm = "vllm_0_30_sm90_kda_gated_rmsnorm"
    elif stage == "output_copy":
        raise NotImplementedError("K3 output_copy is an ATen MEMCPY/no-op; use select_kda_output_copy_transfer")
    elif stage == "chunk":
        raise NotImplementedError("K3 chunk requires all selected autotune configs and physical kernel submissions")
    elif stage == "final_state_scatter":
        raise NotImplementedError("K3 final-state scatter requires its exact Torch advanced-index assignment program")
    else:
        raise ValueError("unknown K3 prefill stage")
    return LaunchConfig(algorithm, grid, (threads, 1, 1), execution_variant=variant)


def _kda_prepared_prefill_frame(shape, dtype, attributes, hardware, backend, *, mixed=False):
    """The unconditional outer frame shared by the source's two backends."""
    from dataclasses import replace
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 full prefill recipe requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_prefill_backend"] != backend
            or attributes["source_output_norm_implementation"] != "forward_cuda"
            or attributes["source_input_layout"] != "packed_k3_projection_with_dense_conv_outputs"
            or attributes["source_raw_gate_layout"] != "contiguous"
            or attributes["source_conv_metadata_state"] != "prepared_actual_query_segments"
            or attributes["source_conv_weight_dtype"] != "fp32"
            or attributes["source_conv_state_dtype"] != "bf16"
            or attributes["source_checkpoint_state"] != "disabled"
            or attributes["source_speculation_state"] != ("mixed_prefill_branch" if mixed else "disabled")
            or attributes["source_output_norm_alias"] != "same_view"):
        raise NotImplementedError("K3 full prefill binds the exact prepared ordinary selected-backend/CUDA-norm source")
    rows, heads, dim, width = require_shape(shape, ("rows", "heads", "head_dim", "conv_kernel"))
    if width != 4:
        raise NotImplementedError("K3 full prefill requires its4-tap convolution source")
    component = attributes["source_component"]
    variants_by_backend = {
        "triton": {
            "vllm_0_30_kimi_k3_triton_fresh_prefill": "kimi_k3_kda_prefill",
            "vllm_0_30_kimi_k3_triton_context_prefill": "kimi_k3_kda_context_prefill",
            "vllm_0_30_kimi_k3_triton_mixed": "kimi_k3_kda_mixed"},
        "flashkda": {
            "vllm_0_30_kimi_k3_flashkda_fresh_prefill": "kimi_k3_kda_prefill",
            "vllm_0_30_kimi_k3_flashkda_context_prefill": "kimi_k3_kda_context_prefill",
            "vllm_0_30_kimi_k3_flashkda_mixed": "kimi_k3_kda_mixed"},
    }
    variants = variants_by_backend[backend]
    if component not in variants:
        raise NotImplementedError("K3 full prefill source component is not bound")
    variant = variants[component]
    starts = attributes["query_start_loc"]
    # Outer selection validates all segments before any metadata arithmetic.
    outer = select_kda_prefill_launch(shape, dtype, variant,
        {**attributes, "stage": "q_convolution"}, hardware)
    sequences = len(starts) - 1
    flags = attributes["has_initial_state"]
    if (not isinstance(flags, (tuple, list)) or len(flags) != sequences
            or any(type(flag) is not bool for flag in flags)):
        raise ValueError("K3 prefill requires the actual initial-state flag for every request")
    if ((variant == "kimi_k3_kda_prefill" and any(flags))
            or (variant == "kimi_k3_kda_context_prefill" and not all(flags))):
        raise ValueError("K3 prefill phase disagrees with its actual initial-state flags")
    conv_programs = attributes["source_conv_num_programs"]
    if type(conv_programs) is not int or conv_programs != outer.grid[0]:
        raise ValueError("K3 prefill prepared convolution metadata disagrees with actual query segments")
    gather = select_kda_prefill_launch(shape, dtype, variant,
        {**attributes, "stage": "initial_state"}, hardware)
    norm = select_kda_prefill_launch(shape, dtype, variant,
        {**attributes, "stage": "output_norm"}, hardware)
    # The outer selector's aggregate execution_variant belongs to the existing
    # partial public contract. Each recipe entry instead names one physical
    # operation and must not impersonate that entire aggregate KernelSpec.
    return sequences, variant, replace(outer, execution_variant=None), replace(gather, execution_variant=None), replace(norm, execution_variant=None)


def select_kda_triton_prefill_recipe(shape, dtype, attributes, hardware):
    """Complete prepared Triton body after projections, with real transfers.

    All chunk selected configs and cached metadata identities are mandatory.
    Cold setup and automatic backend selection are separate entry points.
    """
    return _select_kda_triton_prefill_program(shape, dtype, attributes, hardware, mixed=False)


def _select_kda_triton_prefill_program(shape, dtype, attributes, hardware, *, mixed):
    from src.kernel.launch_config.nvidia_h200.kda_chunk_launch import select_kda_chunk_recipe
    from src.kernel.launch_config.nvidia_h200.kda_state_launch import select_kda_final_state_scatter_recipe

    sequences, _, conv, gather, norm = _kda_prepared_prefill_frame(shape, dtype, attributes, hardware, "triton", mixed=mixed)
    aliases = ("group_output_unrestored",) if mixed else ("same_view", "distinct_contiguous")
    if attributes["source_copy_alias"] not in aliases:
        raise NotImplementedError("K3 prefill requires its actual output alias relationship")
    chunk = select_kda_chunk_recipe(shape, dtype,
        {**attributes, "source_component": "vllm_0_30_kimi_k3_triton_chunk",
         "source_input_layout": "contiguous_qkv_gate_initial"}, hardware)
    scatter = select_kda_final_state_scatter_recipe(
        {"rows": sequences, "heads": shape["heads"], "head_dim": shape["head_dim"], "state_pool_slots": attributes["state_pool_slots"]},
        "fp32", {**attributes, "stage": "final_state_scatter"}, hardware)
    core = (("q_convolution", conv), ("k_convolution", conv), ("v_convolution", conv),
            ("initial_state", gather), *chunk, *scatter)
    if mixed:
        return core
    transfer = select_kda_output_copy_transfer(shape, dtype,
        {**attributes, "stage": "output_copy", "source_input_layout": "contiguous_aligned_16"}, hardware)
    copy_stage = (("output_copy", transfer),) if transfer is not None else ()
    return (*core, *copy_stage, ("output_norm", norm))


def select_kda_flashkda_prefill_recipe(shape, dtype, attributes, hardware):
    """Complete prepared FlashKDA K3 prefill body, directly writing output."""
    return _select_kda_flashkda_prefill_program(shape, dtype, attributes, hardware, mixed=False)


def _select_kda_flashkda_prefill_program(shape, dtype, attributes, hardware, *, mixed):
    from src.kernel.launch_config.nvidia_h200.flashkda_launch import select_flashkda_prefill_recipe
    from src.kernel.launch_config.nvidia_h200.kda_state_launch import select_kda_final_state_scatter_recipe

    sequences, _, conv, gather, norm = _kda_prepared_prefill_frame(shape, dtype, attributes, hardware, "flashkda", mixed=mixed)
    if attributes["source_copy_alias"] != ("group_output_unrestored" if mixed else "same_view"):
        raise NotImplementedError("K3 FlashKDA writes directly to its actual output view")
    main = select_flashkda_prefill_recipe(shape, dtype,
        {**attributes, "source_component": "vllm_0_30_kimi_k3_flashkda",
         "source_input_layout": "contiguous_aligned_16_qkvg_out_and_state"}, hardware)
    scatter = select_kda_final_state_scatter_recipe(
        {"rows": sequences, "heads": shape["heads"], "head_dim": shape["head_dim"], "state_pool_slots": attributes["state_pool_slots"]},
        "fp32", {**attributes, "stage": "final_state_scatter"}, hardware)
    core = (("q_convolution", conv), ("k_convolution", conv), ("v_convolution", conv),
            ("initial_state", gather), *main, *scatter)
    return core if mixed else (*core, ("output_norm", norm))


def select_kda_prepared_prefill_recipe(shape, dtype, attributes, hardware):
    """Resolve vLLM0.30's actual H200 config before selecting the full body.

    H200 is outside FlashInfer's SM100/103 domain. Auto prefers FlashKDA
    with the source's BF16/128-wide/bounded-gate/FP32-state facts. It never
    silently falls back when the selected library's build proof is missing.
    """
    return _select_kda_prepared_prefill_program(shape, dtype, attributes, hardware, mixed=False)


def _select_kda_prepared_prefill_program(shape, dtype, attributes, hardware, *, mixed):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 prefill backend source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if attributes["model_execution_version"] != "vllm_0_30_0":
        raise NotImplementedError("K3 prefill backend binds only its0.30 source")
    if attributes["source_recurrent_state_dtype"] != "fp32" or shape["head_dim"] != 128:
        raise NotImplementedError("K3 complete prepared source currently binds FP32 state/head128")
    requested = attributes["source_requested_prefill_backend"]
    if requested not in ("auto", "triton", "flashkda", "flashinfer"):
        raise ValueError("unknown K3 source prefill backend")
    bounded = attributes["gate_lower_bound"] is not None
    if requested == "flashinfer" or (requested == "flashkda" and not bounded):
        raise NotImplementedError("K3 requested prefill backend is not supported by this H200 source domain")
    if bounded and requested in ("auto", "flashkda"):
        return _select_kda_flashkda_prefill_program(shape, dtype, attributes, hardware, mixed=mixed)
    return _select_kda_triton_prefill_program(shape, dtype, attributes, hardware, mixed=mixed)
