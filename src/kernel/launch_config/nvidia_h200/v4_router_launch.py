"""Complete prepared V4 modular pure-text top6 routing on H200.

fused_topk_bias_router.py normalizes input IDs only for a present hash
table,then dsv4_topk.py handles ordinary score selection. Hash routes enter
topk_softplus_sqrt_kernels.cu's FP32 fast path,which uses four rows/CTA.
Its unused token_expert_indices allocation has no kernel or initialization.
Gate projection,expert execution,EPLB/shared-slot append andVL are separate
programs;no duration,SM80 resource or aggregate typed price is supplied.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_integer_cast_launch


def select_v4_modular_router_recipe(shape, dtype, attributes, hardware):
    """Return every routing submission after the FP32 gate projection.

    Actual hash IDs are caller-owned and must fit their declared dtype/table
    domain. A narrowing cast is source behavior,not permission to fabricate
    or wrap IDs. Source modular tables/outputs are INT32;MegaMoE's INT64
    table belongs to a different backend and is not relabeled here.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("V4 modular router source requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    experts, topk = require_shape(shape, ("experts", "topk"))
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("V4 route rows must be a non-negative integer")
    if (experts not in (256, 384) or topk != 6
            or attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_dsv4_modular_router"
            or attributes["source_input_layout"] != "contiguous_aligned_16"
            or attributes["source_score_function"] != "sqrtsoftplus"
            or attributes["source_renormalize"] is not True
            or attributes["source_output_ids_dtype"] != "int32"
            or attributes["source_vision_bias"] != "none"
            or attributes["source_eplb_and_shared_slots"] != "none"):
        raise NotImplementedError("V4 router binds exact0.30 modular pure-text FP32 top6 without EPLB/shared-slot append")
    if rows >= 2**31 or rows * experts >= 2**31:
        raise NotImplementedError("V4 router source row addressing exceeds its signed int domain")
    table = attributes["source_hash_table_dtype"]
    if table not in ("none", "int32"):
        raise NotImplementedError("V4 modular source table is INT32;other backend/table dtype is unbound")
    if rows == 0:
        return ()
    if table == "none":
        if attributes["source_correction_bias"] != "contiguous_fp32":
            raise NotImplementedError("V4 normal routing requires the actual contiguousFP32 correction bias")
        pdl = attributes["source_platform_pdl"]
        if type(pdl) is not bool:
            raise TypeError("V4 source platform PDL probe must be an actual boolean")
        launch = LaunchConfig("vllm_0_30_sm90_dsv4_top6_triton_int32", (rows, 1, 1), (32, 1, 1),
            parameters={"source_programmatic_stream_serialization": float(pdl)})
        return (("score_top6", launch),)
    if (attributes["source_input_ids_layout"] != "contiguous_aligned_16"
            or attributes["source_hash_table_layout"] != "contiguous_vocab_top6"
            or attributes["source_correction_bias"] != "none"):
        raise NotImplementedError("V4 hash route requires real contiguous IDs/table andno score bias")
    token_dtype = attributes["source_input_ids_dtype"]
    if token_dtype not in ("int32", "int64"):
        raise NotImplementedError("V4 hash input IDs require their actual INT32/INT64 dtype")
    prefix = ()
    if token_dtype != table:
        bounds = attributes["source_input_id_range"]
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not -(2**31) <= bounds[0] <= bounds[1] < 2**31):
            raise ValueError("V4 narrowing cast requires the actual input ID range to fit INT32")
        cast = select_torch213_integer_cast_launch({"element_count": rows}, token_dtype, {
            "tensor_iterator_source": attributes["tensor_iterator_source"],
            "source_input_layout": attributes["source_input_ids_layout"], "source_output_dtype": table}, hardware)
        prefix = (("hash_input_ids_cast", cast),)
    # launchDsv4HashTopk hardcodes PDL1,independently of the platform probe.
    # It does not pass token_expert_indices into the kernel. torch.empty's
    # rows*6*4B request must not become a fill or payload traffic stage.
    launch = LaunchConfig("vllm_0_30_sm90_dsv4_hash_top6_int32_int32",
        (ceil_div(rows, 4), 1, 1), (128, 1, 1), parameters={
            "dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0})
    return (*prefix, ("hash_top6", launch))
