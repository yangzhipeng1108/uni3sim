"""H200 versioned CUDA host dispatch, derived without duration observations.

Source: vLLM v0.30.0 csrc/libtorch_stable/{layernorm_kernels,
activation_kernels,pos_encoding_kernels,cache_kernels}.cu. This owner covers
BF16/FP16 inputs with each host program's actual layout requirements.
New normalization requires its ordinary (non batch-invariant) runtime branch;
SiLU,RoPE andcache host geometry does not consult that flag. Compiler
resources are deliberately absent until independently bound.
The SM90 0.30 activation branch uses 16-byte vectors; the32-byte path needs
SM100. v0.11.0/csrc/{layernorm_kernels,activation_kernels,
pos_encoding_kernels,cache_kernels}.cu supplies the explicitly selected0.11
scalar rules (https://github.com/vllm-project/vllm/tree/v0.11.0/csrc).
No hardware package or architecture-specific compiler facts are borrowed.
The exact v0.27.1 tag's csrc/libtorch_stable host wrappers independently
bind the same SM90 vector8/gcd8 and row-threshold branches as0.30.
"""

import math

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    require_dtype, require_paged_kv_cache_contract, require_shape)


SOURCE_VECTOR_VARIANTS = frozenset({
    "rmsnorm", "add_rmsnorm", "rotary", "rope", "activation", "silu",
    "kv_cache", "kvcache",
})

LEGACY_FUSED_NORM_GEOMETRY = "vllm_0_11_sm90_add_rmsnorm_geometry_partial"


def legacy_fused_norm_geometry(shape, dtype):
    """0.11 host geometry shared by aligned and scalar CUDA templates.

    Pointer alignment selects the compiled template, but never grid/block.
    This identity intentionally carries no compiled register/shared facts.
    """
    require_dtype(dtype, ("bf16", "fp16"))
    rows, hidden = require_shape(shape, ("rows", "hidden"))
    return LaunchConfig(LEGACY_FUSED_NORM_GEOMETRY, (rows, 1, 1),
        (min(hidden, 1024 if rows < 256 else 256), 1, 1), execution_variant="add_rmsnorm")


def select_kimi_k3_situ_launch(shape, dtype, variant, attributes, hardware):
    """v0.30 SituAndMul.forward_cuda -> situ_and_mul scalar CUDA wrapper.

    csrc/libtorch_stable/activation_kernels.cu::situ_and_mul selects one
    token per CTA and min(d,1024) threads. Unlike ordinary SiLU on SM90, it
    does not use vector8 launch geometry. Compiler resources remain unbound.
    """
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 SiTU source requires H200 hardware")
    require_dtype(dtype, ("bf16",))
    if (variant != "kimi_k3_situ"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_ordinary_mlp"):
        raise NotImplementedError("K3 SiTU source program is not bound")
    if (type(attributes["activation_beta"]) not in (int, float)
            or attributes["activation_beta"] != 4.0
            or type(attributes["activation_linear_beta"]) not in (int, float)
            or attributes["activation_linear_beta"] != 25.0):
        raise ValueError("K3 SiTU requires its beta4 and linear beta25 source contract")
    rows, width, elements = require_shape(shape, ("rows", "intermediate", "element_count"))
    if elements != rows * width:
        raise ValueError("K3 SiTU requires complete output rows")
    from src.kernel.launch_config.nvidia_h200.moe_aux_launch import situ_plain_geometry
    grid, block = situ_plain_geometry(rows, width)
    return LaunchConfig("vllm_0_30_sm90_kimi_k3_situ_scalar", grid, block,
                        execution_variant=variant)


def select_torch_vector_launch(shape, dtype, variant, attributes, hardware):
    """SM90 Torch2.11 binary TensorIterator, 128 threads and eight values/lane.

    The independent versioned non-duration geometry evidence is retained in
    references/bf16_binary_tensor_iterator_task_floor_curve_sm90.json under
    provenance.launch_geometry_evidence; timing fields are not read by selection.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("Torch SM90 vector selection requires H200 HWSpec")
    if attributes["tensor_iterator_source"] == "pytorch_2_8_sm90":
        from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch28_vector_launch
        return select_torch28_vector_launch(shape, dtype, variant, attributes, hardware)
    require_dtype(dtype, ("bf16",))
    if (variant != "elementwise"
            or attributes["tensor_iterator_source"] != "pytorch_2_11_sm90"
            or attributes["source_input_layout"] != "contiguous_aligned_16"):
        raise NotImplementedError("Torch vector program is not bound")
    if attributes["operation"] not in ("add", "mul"):
        raise NotImplementedError("Torch vector source requires binary add/mul")
    if (type(attributes["reads"]) is not int or type(attributes["writes"]) is not int
            or attributes["reads"] != 2 or attributes["writes"] != 1):
        raise ValueError("Torch binary source requires two reads and one write")
    elements, = require_shape(shape, ("element_count",))
    if 2 * elements >= 2**31:
        raise NotImplementedError("Torch vector source requires an unsplit32-bit iterator")
    return LaunchConfig("pytorch_2_11_sm90_binary_vector8",
        grid=(-(-elements // 1024), 1, 1), block=(128, 1, 1),
        execution_variant=variant)


def select_source_vector_launch(shape, dtype, variant, attributes, hardware):
    """Select a pinned host program, never infer runtime from tensor shape."""
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("H200 source vector selection requires H200 HWSpec")
    runtime = attributes["model_execution_version"]
    if runtime not in ("vllm_0_30_0", "vllm_0_27_1", "vllm_0_11_0"):
        raise NotImplementedError("source vector runtime is not bound")
    require_dtype(dtype, ("bf16", "fp16"))
    if variant not in SOURCE_VECTOR_VARIANTS:
        raise NotImplementedError(f"unbound H200 source vector variant: {variant}")
    if (runtime == "vllm_0_11_0" and variant == "add_rmsnorm"
            and attributes["source_input_layout"] == "contiguous"):
        return legacy_fused_norm_geometry(shape, dtype)
    # Require only layout/branch facts read by this exact host program.
    # Scalar0.11 RMSNorm/SiLU and scalar RoPE never dispatch by16B alignment.
    # New RMSNorm handles per-row alignment internally and arbitrary leading
    # strides; newer SiLU dispatches by d%8 on SM90,not by pointer alignment.
    layouts = {"contiguous_aligned_16"}
    if variant == "rmsnorm":
        layouts.add("contiguous")
        if runtime != "vllm_0_11_0":
            layouts.add("unit_final_stride")
    elif variant in ("silu", "activation"):
        layouts.add("contiguous")
    elif variant == "add_rmsnorm" and runtime != "vllm_0_11_0":
        # Pointer alignment changes its internal vector/scalar template,
        # not the host CTA geometry. No compiler resources are bound here.
        layouts.add("contiguous")
    elif variant in ("rope", "rotary"):
        layouts.update(("contiguous", "unit_final_stride"))
    elif variant in ("kv_cache", "kvcache"):
        # reshape_and_cache_flash reads each K/V token through its own
        # leading stride; the heads*head_dim payload within a token is dense.
        # Packed QKV views can retain gaps between those complete rows.
        layouts.update(("contiguous", "unit_final_stride"))
    if attributes["source_input_layout"] not in layouts:
        raise NotImplementedError("source vector input layout is not bound")
    # Only the0.27/0.30 normalization wrappers consult this runtime flag.
    # Do not require (or invent) it for source programs that never read it.
    if (runtime != "vllm_0_11_0" and variant in ("rmsnorm", "add_rmsnorm")
            and attributes["source_batch_invariant"] is not False):
        raise NotImplementedError("batch-invariant source vector program is not bound")
    if variant in ("rmsnorm", "add_rmsnorm"):
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        vector_width = (1 if runtime == "vllm_0_11_0" else math.gcd(8, hidden)) if variant == "rmsnorm" else 8
        if variant == "add_rmsnorm" and hidden % 8:
            vector_width = 1
        maximum = 1024 if rows < 256 or runtime == "vllm_0_11_0" and variant == "rmsnorm" else 256
        block = min(hidden // vector_width, maximum) if variant == "rmsnorm" else min(hidden, maximum)
    elif variant in ("silu", "activation"):
        rows, intermediate = require_shape(shape, ("tok", "intermediate"))
        vector_width = 8 if runtime != "vllm_0_11_0" and intermediate % 8 == 0 else 1
        block = min(intermediate // vector_width, 1024)
    elif variant in ("rope", "rotary"):
        rows, query_heads, kv_heads, head_dim = require_shape(
            shape, ("seq_len", "num_query_heads", "num_kv_heads", "head_dim"))
        if query_heads % kv_heads:
            raise ValueError("RoPE requires query heads divisible by KV heads")
        if variant == "rope":
            rotary_dim, = require_shape(shape, ("rotary_dim",))
            if type(attributes["is_neox"]) is not bool or type(attributes["key_present"]) is not bool:
                raise TypeError("RoPE source flags must be booleans")
        else:
            rotary_dim = head_dim
        if rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError("RoPE requires an even rotary dimension within head_dim")
        block = min(query_heads * rotary_dim // 2, 512)
    else:
        require_paged_kv_cache_contract(attributes)
        rows, heads, head_dim = require_shape(shape, ("seq_len", "num_kv_heads", "head_dim"))
        block = min(heads * head_dim, 512)
    if rows <= 0 or block <= 0:
        raise ValueError("source vector launch requires positive dimensions")
    parameters = {}
    if runtime == "vllm_0_11_0" and variant in ("rmsnorm", "add_rmsnorm", "silu", "rope", "kv_cache", "kvcache"):
        from src.kernel.launch_config.nvidia_h200.legacy_vector_resources import select_legacy_vector_resources
        parameters = select_legacy_vector_resources(shape, dtype, variant, attributes)
    return LaunchConfig(
        algorithm=f"{runtime[:-2]}_sm90_source_{variant}",
        grid=(rows, 1, 1), block=(block, 1, 1),
        parameters=parameters, execution_variant=variant)


def select_mla_rope_launch(shape, dtype, attributes, hardware):
    """The same pinned scalar rotary_embedding wrapper over MLA's RoPE view.

    YaRN/default cache generation is outside this call. The prepared cache
    supplies the active width; Q's leading/head strides may span its NoPE
    prefix. Host grid/block do not depend on those strides or the cache
    values. No compiler-resource binding is inferred for this target ABI.
    """
    if (attributes["source_rope_program"] != "vllm_fused_rotary"
            or attributes["rope_layout"] != "interleaved"
            or attributes["rope_type"] not in ("default", "deepseek_yarn")):
        raise NotImplementedError("MLA RoPE requires its exact fused interleaved source program")
    heads, = require_shape(shape, ("num_key_heads",))
    launch = select_source_vector_launch({**shape, "num_kv_heads": heads}, dtype,
        "rotary", attributes, hardware)
    return LaunchConfig(f"{attributes['model_execution_version'][:-2]}_sm90_source_mla_rope",
        launch.grid, launch.block, execution_variant="mla_rope")
