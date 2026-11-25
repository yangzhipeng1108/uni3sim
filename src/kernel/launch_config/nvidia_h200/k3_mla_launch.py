"""Kimi-K3 MLA fused BF16/FP16 epilogues from the exact vLLM0.30 source.

fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu::launchPdlSlots fixes256
threads/eight warps. Cache insertion adds one cache-slot row per token;
decode up to64 tokens splits each slot across three warps. Context KV pack
uses two head rows/warp and caps blocks atSMs*8,retaining its grid-stride
loop. No GEMM/FA3 implementation,compiler register estimate or duration is
borrowed. These complete fused operators are individual physical stages.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


K3_MLA_STAGES = frozenset({"decode_q_concat_cache", "prefill_key_concat_cache", "context_key_concat"})


def select_k3_mla_fused_launch(shape, dtype, attributes, hardware):
    """Return the source stage's launch,or None for its explicit zero-row no-op.

    Actual outer strides are caller inputs;aligned unit last dimension is
    sufficient for the source8-value vector loads. This includes BMM's
    transposed absorbed query and split K/V,Q/RoPE views,without inventing
    contiguous copies. NoPE omits both position IDs and the RoPE cache.
    Slot IDs may include negative graph padding: those rows still produce
    Q/K output,while the cache-slot warp skips its write.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("K3 MLA fused source requires H200 HWSpec")
    require_dtype(dtype, ("bf16", "fp16"))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_mla"
            or attributes["source_input_layout"] != "aligned_16_unit_last_dim"
            or attributes["source_rope_state"] != "disabled_nope"
            or attributes["source_cache_dtype"] != dtype
            or attributes["source_slot_mapping_dtype"] != "int64"):
        raise NotImplementedError("K3 MLA binds its exact0.30 NoPE,matching cache dtype and aligned strided source")
    stage = attributes["stage"]
    if stage not in K3_MLA_STAGES:
        raise NotImplementedError("K3 MLA fused physical stage is not bound")
    heads, latent, nope, rope, value, page = require_shape(shape, (
        "num_heads", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "block_size"))
    if (latent, nope, rope, value) != (512, 128, 64, 128):
        raise NotImplementedError("K3 fused MLA requires its512 latent,128 NoPE/V and64 RoPE dimensions")
    rows = shape["rows"]
    if type(rows) is not int or rows < 0:
        raise ValueError("K3 MLA rows must be a non-negative integer source extent")
    if rows >= 2**31 or heads >= 2**31 or page >= 2**31:
        raise NotImplementedError("K3 MLA source token/head/page fields must fit signed int")
    if rows == 0:
        return None
    if stage == "decode_q_concat_cache":
        split = 3 if rows <= 64 else 1
        slots = (heads + 1) * split
        rows_per_warp, cap = 1, None
        algorithm = f"vllm_0_30_sm90_k3_mla_decode_concat_cache_{dtype}_split{split}"
    elif stage == "prefill_key_concat_cache":
        slots, rows_per_warp, cap = heads + 1, 1, None
        algorithm = f"vllm_0_30_sm90_k3_mla_prefill_key_cache_{dtype}"
    elif stage == "context_key_concat":
        slots, rows_per_warp = heads, 2
        cap = int(hardware.num_compute_units) * 8
        algorithm = f"vllm_0_30_sm90_k3_mla_context_key_concat_{dtype}"
    else:
        raise AssertionError("K3 MLA stage dispatch is exhaustive")
    if slots >= 2**31:
        raise NotImplementedError("K3 MLA source slots_per_token must fit signed int")
    grid = ceil_div(ceil_div(rows * slots, rows_per_warp), 8)
    if cap is not None:
        grid = min(grid, cap)
    if grid >= 2**31:
        raise NotImplementedError("K3 MLA source grid exceeds CUDA grid.x capacity")
    return LaunchConfig(algorithm, (grid, 1, 1), (256, 1, 1),
        parameters={"dynamic_shared_bytes": 0.0, "source_programmatic_stream_serialization": 1.0})
