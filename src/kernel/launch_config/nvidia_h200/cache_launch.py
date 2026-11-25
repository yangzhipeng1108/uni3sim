"""H200 vLLM 0.30 MLA/indexer and V4 compressor host launch programs.

CUDA source: csrc/libtorch_stable/cache_kernels.cu. CuTeDSL source:
vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py.
The indexer C4 source is common/ops/fused_compress_quant_cache.py.
These are static launch facts; resources and timings are not fabricated.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import require_dtype, require_positive_integer, require_shape


CACHE_VARIANTS = frozenset({
    "mla_kv_cache", "dsa_fp8_index_cache", "dsv4_compressed_state_cache",
    "dsv4_compressor_state_cache",
})


def select_v4_compressor_recipe(shape, dtype, attributes, hardware):
    """Complete prepared packed-FP8 compressor body,after its projection GEMM.

    The projection's FP32 split views,APE and paged state already exist.
    Metadata is caller-owned: in particular a missing CPU boundary is None,
    not False,and FULL graph capture retains all finish CTAs. Slots may be
    padded; neither live slots nor completed rows determine the launch grid.
    State and output cache have independent mappings and actual page strides.

    Main C128's fresh FP32[rows,512] is allocation extent,not initialized
    payload or physical traffic. No register/shared estimate,typed aggregate
    service,full-cache scale setup or page-layout inference is supplied.
    """
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("V4 compressor source requires H200 HWSpec")
    require_dtype(dtype, ("fp32",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_deepseek_v4_compressor"
            or attributes["source_input_layout"] != "split_fp32_unit_last_dim"
            or attributes["source_state_and_ape_dtype"] != "fp32"
            or attributes["source_metadata_dtypes"] != ("int64", "int64", "int32", "int32")):
        raise NotImplementedError("V4 compressor requires exact0.30 split FP32 inputs and position/state-slot/request/table dtypes")
    rows, head, rope, ratio, state_block = require_shape(shape, (
        "seq_len", "head_dim", "rope_head_dim", "compress_ratio", "state_block_size"))
    if (head, rope, ratio) not in ((512, 64, 4), (512, 64, 128), (128, 64, 4)):
        raise NotImplementedError("V4 compressor source covers main C4/C128 and FP8 indexer C4")
    width = head * (2 if ratio == 4 else 1)
    projection_stride = require_positive_integer(attributes, "source_projection_row_stride")
    state_row = require_positive_integer(attributes, "source_state_row_stride")
    state_page = require_positive_integer(attributes, "source_state_page_stride")
    if projection_stride < 2 * width or state_row < 2 * width or state_page < state_block * state_row:
        raise ValueError("V4 source split/state strides overlap their packed rows")
    if ratio == 4 and state_block % 4:
        raise NotImplementedError("C4 source state pages require whole compression groups")
    # SavePartialStatesKernel uses one CTA/token and the bound Triton default
    # of four warps for both512/1024 main widths and256 indexer width.
    partial = LaunchConfig("vllm_0_30_sm90_v4_save_partial_states", (rows, 1, 1), (128, 1, 1),
        parameters={"source_head_size": float(width), "source_state_width": float(width),
            "source_programmatic_stream_serialization": 0.0})
    stages = (("save_partial_states", partial),)
    if head == 512 and ratio == 128:
        mode, boundary = attributes["source_cudagraph_runtime_mode"], attributes["source_c128_boundary"]
        if mode not in ("none", "piecewise", "full"):
            raise ValueError("V4 C128 requires the actual CUDA graph runtime mode")
        if boundary is not None and type(boundary) is not bool:
            raise TypeError("V4 C128 CPU boundary must be a boolean or explicitly unknown")
        # This return precedes lookup/validation of the output cache in source.
        if mode != "full" and boundary is False:
            return stages
        if state_block != 8:
            raise NotImplementedError("CuTeDSL C128 finish requires resolved state block_size=8")
    if (attributes["source_norm_weight_dtype"] != "bf16"
            or attributes["source_rope_cache_layout"] != "fp32_gptj_last64"
            or attributes["source_output_slot_dtype"] != "int64"):
        raise NotImplementedError("V4 finish requires prepared BF16 gamma,FP32 GPT-J cache and independent INT64 output slots")
    cache_format = "fp8_ds_mla" if head == 512 else "fp8_indexer"
    if attributes["source_cache_format"] != cache_format:
        raise NotImplementedError("V4 prepared recipe covers packed uint8 FP8 caches;full-cache/FP4 setup is unbound")
    page_rows = require_positive_integer(attributes, "source_cache_page_rows")
    page_stride = require_positive_integer(attributes, "source_cache_page_stride_bytes")
    record_bytes = 584 if head == 512 else 132
    if page_stride < page_rows * record_bytes:
        raise ValueError("V4 packed cache page stride truncates its separate data/scale regions")
    if head == 512 and (state_row % 16 or state_page % 16 or page_stride % 16):
        raise NotImplementedError("CuTeDSL V4 main requires its actual aligned state/cache page strides")
    if ratio == 4:
        algorithm, threads = (("vllm_dsv4_c4_norm_rope_store_cutedsl", 64) if head == 512
            else ("vllm_dsv4_c4_indexer_norm_rope_store_triton", 32))
        return (*stages, ("compress_norm_rope_store", LaunchConfig(algorithm, (rows, 1, 1), (threads, 1, 1))))
    # Both kernels retain CTAs for non-boundary/padded rows and return early.
    compress = LaunchConfig("vllm_dsv4_c128_compress_block8_cutedsl", (rows * 8, 1, 1), (256, 1, 1),
        parameters={"source_workspace_allocation_bytes": float(rows * 512 * 4)})
    store = LaunchConfig("vllm_dsv4_c128_norm_rope_store_cutedsl", (rows, 1, 1), (256, 1, 1))
    return (*stages, ("compress", compress), ("norm_rope_store", store))


def select_cache_launch(shape, dtype, variant, attributes, hardware):
    if hardware.hardware_id != "nvidia_h200":
        raise ValueError("SM90 cache source requires H200 HWSpec")
    parameters = {}
    if variant == "dsv4_compressor_state_cache":
        require_dtype(dtype, ("fp32",))
        rows, state_bytes = require_shape(shape, ("seq_len", "state_bytes"))
        if (attributes["cache_layout"] != "paged_dsv4_compressor_state"
                or attributes["cache_write_semantics"] != "append"):
            raise ValueError("V4 partial-state source requires paged append semantics")
        if state_bytes not in (2048, 4096, 8192):
            raise ValueError("V4 partial state must contain 2048, 4096 or 8192 bytes")
        # SavePartialStatesKernel.__call__: one program per new token.
        # FP32 packed KV/score use widths256 (indexer C4),512 (main
        # C128) and1024 (main C4),with the non-autotuned four-warp
        # Triton default retained by the prepared compressor recipe.
        algorithm, grid, threads = "vllm_0_30_sm90_v4_save_partial_states", (rows, 1, 1), 128
    elif variant == "mla_kv_cache":
        require_dtype(dtype, ("bf16",))
        rows, width = require_shape(shape, ("seq_len", "compressed_dim"))
        # The existing contract combines latent and RoPE. The model domain
        # here has latent512+RoPE64; a generic combined width cannot identify
        # kv_lora_rank and must not be mistaken for the CUDA block argument.
        if width != 576:
            raise NotImplementedError("MLA cache source needs the latent512/RoPE64 domain")
        if attributes["cache_layout"] != "paged_mla" or attributes["cache_write_semantics"] != "append":
            raise ValueError("MLA source cache requires paged append semantics")
        if (("kv_cache_dtype" in attributes and attributes["kv_cache_dtype"] != "auto")
                or ("cache_storage_matches_input" in attributes and attributes["cache_storage_matches_input"] is not True)
                or ("source_cache_dtype" in attributes and attributes["source_cache_dtype"] != "bf16")):
            raise NotImplementedError("MLA cache source requires matching BF16 storage,not quantized cache")
        if "source_model_execution_version" in attributes:
            runtime = attributes["source_model_execution_version"]
            algorithms = {
                "vllm_0_11_0": "vllm_0_11_sm90_concat_and_cache_mla",
                "vllm_0_27_1": "vllm_0_27_sm90_concat_and_cache_mla",
                "vllm_0_30_0": "vllm_0_30_sm90_concat_and_cache_mla",
            }
            if not isinstance(runtime, str) or runtime not in algorithms:
                raise NotImplementedError("MLA cache source runtime is not bound")
            if "model_execution_version" in attributes and attributes["model_execution_version"] != runtime:
                raise ValueError("MLA cache source runtime declarations disagree")
            algorithm = algorithms[runtime]
        else:
            # Preserve the existing0.30-only unstamped source contract.
            # An older runtime must supply its actual caller stamp.
            algorithm = "vllm_0_30_sm90_concat_and_cache_mla"
        # Each pinned tag's matching-storage host wrapper independently uses
        # grid(slot_mapping.size(0)),block(min(kv_lora_rank,512)). Their FP8
        # DS-MLA branches differ and are deliberately outside this BF16 ABI.
        grid, threads = (rows, 1, 1), 512
    elif variant == "dsa_fp8_index_cache":
        require_dtype(dtype, ("uint8",))
        rows, key_dim, scale_bytes = require_shape(
            shape, ("seq_len", "key_dim", "scale_bytes_per_token"))
        if scale_bytes % 4 or key_dim % (scale_bytes // 4):
            raise ValueError("indexer scale extent must identify integral FP32 quantization blocks")
        quant_block = key_dim // (scale_bytes // 4)
        if attributes["cache_layout"] != "paged_mla_indexer" or attributes["cache_write_semantics"] != "append":
            raise ValueError("indexer source cache requires paged append semantics")
        if attributes["quantization"] != "fp8_e4m3_with_ue8m0_scale":
            raise NotImplementedError("indexer source cache requires its UE8M0 scale format")
        return LaunchConfig("vllm_0_30_sm90_indexer_k_quant_and_cache",
            grid=(rows, -(-key_dim // (quant_block * 4)), 1), block=(32, 4, 1),
            parameters=parameters, execution_variant=variant)
    elif variant == "dsv4_compressed_state_cache":
        parameters = {}
        rows, ratio, state_bytes = require_shape(shape, ("seq_len", "compress_ratio", "state_bytes"))
        if attributes["cache_layout"] != "paged_dsv4_compressed" or attributes["cache_write_semantics"] != "append":
            raise ValueError("V4 compressor source requires paged append semantics")
        if ratio == 4:
            require_dtype(dtype, ("uint8",))
            if "finish_stage" in attributes and attributes["finish_stage"] is not None:
                raise ValueError("C4 compress/norm/store is one fused source launch")
            if state_bytes == 584 and attributes["quantization"] == "fp8_ds_mla":
                algorithm, threads = "vllm_dsv4_c4_norm_rope_store_cutedsl", 64
            elif state_bytes == 132 and attributes["quantization"] == "fp8_e4m3_with_ue8m0_scale":
                algorithm, threads = "vllm_dsv4_c4_indexer_norm_rope_store_triton", 32
            else:
                raise ValueError("C4 source cache record and quantization disagree")
            grid = (rows, 1, 1)
        elif ratio == 128 and state_bytes == 584:
            if attributes["quantization"] != "fp8_ds_mla":
                raise ValueError("C128 source cache requires the packed MLA record")
            if "finish_stage" not in attributes or attributes["finish_stage"] is None:
                raise NotImplementedError("C128 compressor has two physical stages; select each finish_stage")
            stage = attributes["finish_stage"]
            if stage == "compress":
                require_dtype(dtype, ("fp32",))
                algorithm, grid, threads = "vllm_dsv4_c128_compress_block8_cutedsl", (rows * 8, 1, 1), 256
            elif stage == "store":
                require_dtype(dtype, ("uint8",))
                algorithm, grid, threads = "vllm_dsv4_c128_norm_rope_store_cutedsl", (rows, 1, 1), 256
            else:
                raise ValueError("C128 finish_stage must be compress or store")
        else:
            raise NotImplementedError("V4 source compressor requires C4 main/indexer or C128 main")
        # CuTeDSL/Triton kernels use shared reductions. Their compiler decides
        # allocation, so neither static nor dynamic shared bytes is inferred.
    else:
        raise NotImplementedError(f"unbound SM90 cache variant: {variant}")
    return LaunchConfig(algorithm, grid, (threads, 1, 1),
                        parameters=parameters, execution_variant=variant)
