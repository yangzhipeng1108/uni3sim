"""Versioned H200 attention-state helpers,excluding FA3 implementations.

v1/attention/ops/merge_attn_states.py selects the0.30 CUDA op for supported
input dtype/head width andotherwise uses its Triton merge. Their kernels
derive token strides from heads*head_stride,while LSE uses actual strides.
No data-dependent LSE branch,register/static resource orprice is inferred.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _output_strides(attributes, key, rows, heads, dim):
    value = attributes[key]
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(item) is not int for item in value)
            or value[2] != 1 or value[1] < dim or value[0] != heads * value[1]):
        raise NotImplementedError("merge output views require unit features andtoken stride heads*head_stride")
    if (rows * heads - 1) * value[1] + dim >= 2**31:
        raise NotImplementedError("merge output view exceeds the bound source signed index domain")
    return value[1]


def _lse_strides(attributes, key, rows, heads):
    value = attributes[key]
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int or item < 0 for item in value)
            or (heads > 1 and value[0] == 0) or (rows > 1 and value[1] == 0)
            or (heads > 1 and rows > 1 and not (value[0] >= rows * value[1] or value[1] >= heads * value[0]))):
        raise ValueError("merge requires actual nonoverlapping head-major ortransposed/gapped FP32 LSE strides")
    if (heads - 1) * value[0] + (rows - 1) * value[1] >= 2**31:
        raise NotImplementedError("merge LSE view exceeds bound source signed indexing")


def select_vllm030_merge_attn_states_launch(shape, dtype, attributes, hardware):
    """Complete selected public state-merge call onprepared partials.

    Source always submits one merge kernel,including context-free suffix
    rows;no separate suffix copy,normalization orLSE kernel is inserted.
    CUDA threads each process one128-bit input pack;Triton processes one
    token/head. Optional FP8 output andLSE stay inside that same operation.
    Exact prefix/output aliases may be passed by the source,but launch
    derivation alone doesnot prove native numerical/alias correctness.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("state merge source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16", "fp32"))
    rows, heads, dim = require_shape(shape, ("rows", "num_heads", "head_dim"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_merge_attn_states"
            or attributes["source_suffix_dtype"] != dtype
            or attributes["source_lse_dtype"] != "fp32"):
        raise NotImplementedError("state merge requires exact0.30 matching input dtypes andFP32 LSE")
    prefix_tokens = attributes["source_prefill_tokens_with_context"]
    if prefix_tokens is not None and (type(prefix_tokens) is not int or not 0 <= prefix_tokens <= rows):
        raise ValueError("state merge prefix extent must be actual0..rows orNone")
    prefix_stride = _output_strides(attributes, "source_prefix_output_strides", rows, heads, dim)
    suffix_stride = _output_strides(attributes, "source_suffix_output_strides", rows, heads, dim)
    output_stride = _output_strides(attributes, "source_output_strides", rows, heads, dim)
    if prefix_stride != suffix_stride:
        raise ValueError("merge requires matching prefix/suffix head strides")
    _lse_strides(attributes, "source_prefix_lse_strides", rows, heads)
    _lse_strides(attributes, "source_suffix_lse_strides", rows, heads)
    lse = attributes["source_output_lse"]
    if lse not in ("none", "fp32"):
        raise NotImplementedError("merge output LSE must be explicitly absent orFP32")
    if lse == "fp32":
        _lse_strides(attributes, "source_output_lse_strides", rows, heads)
    out_dtype = attributes["source_output_dtype"]
    scale = attributes["source_output_scale"]
    fp8 = out_dtype == "fp8_e4m3fn"
    if fp8:
        if scale != "contiguous_fp32_scalar":
            raise NotImplementedError("FP8 merge output requires its actual FP32 scalar scale tensor")
    elif out_dtype != dtype or scale != "none":
        raise NotImplementedError("ordinary merge requires matching output dtype andno FP8 scale")
    width = 4 if dtype == "fp32" else 2
    pack = 16 // width
    if dim % pack == 0:
        # The Python API decides CUDA from input dtype/head width. A bad
        # pointer/head stride is not a source reason to switch to Triton.
        output_pack_bytes = pack if fp8 else 16
        addresses = tuple(attributes[key] for key in (
            "source_prefix_address_alignment", "source_suffix_address_alignment", "source_output_address_alignment"))
        if any(type(value) is not int or value not in (1, 2, 4, 8, 16) for value in addresses):
            raise ValueError("merge requires actual capped16-byte address alignments")
        if (addresses[0] < 16 or addresses[1] < 16 or addresses[2] < output_pack_bytes
                or prefix_stride * width % 16
                or output_stride * (1 if fp8 else width) % output_pack_bytes):
            raise NotImplementedError("selected CUDA merge requires its actual packed pointer/stride alignment")
        return LaunchConfig(f"vllm_0_30_sm90_merge_attn_states_cuda_{dtype}_to_{out_dtype}",
            (ceil_div(rows * heads * (dim // pack), 128), 1, 1), (128, 1, 1),
            parameters={"dynamic_shared_bytes": 0.0})
    if heads > 65535 or rows >= 2**31:
        raise NotImplementedError("Triton merge exceeds CUDA grid limits")
    return LaunchConfig(f"vllm_0_30_sm90_merge_attn_states_triton_{dtype}_to_{out_dtype}",
        (rows, heads, 1), (128, 1, 1))
