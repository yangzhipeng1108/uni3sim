"""GEMM and vector/elementwise observation normalization for kernel
evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from typing import Mapping, Optional

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    DTYPE_BYTES,
    PAGED_KV_CACHE_ATTRIBUTES,
)
from src.kernel.launch_config import GpuCtaRasterProgram, TilingSpec
from src.kernel.launch_config.ascend_910b.gemm_launch import (
    launch_from_cann_gemm_observation,
)
from src.kernel.launch_config.nvidia_a800.gemm_launch import (
    cublaslt_output_dtype_reducer_source_parameters,
    observed_gemm_cta_raster,
)
from src.kernel.launch_config.nvidia_h200.gemm_launch import (
    cublaslt_fp32_partial_reducer_source_parameters,
    cublaslt_narrow_tensor_instruction_parameters,
    cublaslt_uniform_split_k_parameters,
)
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.tiling_spec import (
    find_gemm_swizzle_convention,
    parse_nvjet_gemm_symbol,
)
from evaluation.kernel.observation_parser import (
    is_non_tensor,
    observed_launch_config,
)
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from src.kernel.model_contract import SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER

from evaluation.json_shape import require_positive_metric as _positive_int

from evaluation.kernel.observation_readers import (
    _non_empty_string,
    observed_nvjet_gemm_split_k,
    legacy_h200_bf16_aggregate_projection,
)


# The vector variants whose Ascend observation is bound to the retained CANN
# source program.  A800 binds every vector observation and the MetaX backends
# bind the variants their typed flags mark; this constant holds only the
# Ascend side, which is a fixed set rather than a per-observation fact.
_ASCEND_SOURCE_BOUND_VECTOR_VARIANTS = frozenset({
    "rmsnorm", "add_rmsnorm", "activation", "silu", "rotary", "rope",
    "paged_kv_cache", "elementwise", "qkv_rmsnorm_rope",
})


def _is_source_bound_vector_observation(
        hardware_id: str, variant: str, *, c500_typed) -> bool:
    """Whether this vector observation is bound to a retained source program.

    A800 binds every observation; the MetaX backends bind the variants their
    typed flags mark (their symbols were checked against the retained
    artifact above); Ascend binds the fixed CANN program set.
    """

    if hardware_id == "nvidia_a800":
        return True
    if any(c500_typed):
        return True
    return (hardware_id == "ascend_910b"
            and variant in _ASCEND_SOURCE_BOUND_VECTOR_VARIANTS)


# The runtime symbol each backend's fused SiLU (activation) kernel publishes.
# A hardware absent from this map has no bound SILU symbol, so its observation
# is not checked against one.
_SILU_RUNTIME_SYMBOLS = {
    "nvidia_a800": (
        "void vllm::act_and_mul_kernel<c10::BFloat16, "
        "&vllm::silu_kernel<c10::BFloat16>, (bool)1>"
        "(T1 *, const T1 *, int)"
    ),
    "metax_c500": (
        "void vllm::act_and_mul_kernel_bd_opt<c10::BFloat16, "
        "float4, 8, &(c10::BFloat16 vllm::silu_kernel<"
        "c10::BFloat16>(c10::BFloat16 const&)), true>"
        "(c10::BFloat16*, c10::BFloat16 const*, int, int)"
    ),
    "ascend_910b": "SwiGlu_3_high_performance_27",
}


_ASCEND_ELEMENTWISE_RUNTIME_SYMBOLS = {
        "add": (
            "Add_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210000000",
            "Add_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210010000",
        ),
        "mul": (
            "Mul_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210020000",
        ),
    }


_METAX_VECTOR_RUNTIME_SYMBOLS = {
        "rmsnorm": (
            "void vllm::rms_norm_kernel<c10::BFloat16>"
            "(c10::BFloat16*, c10::BFloat16 const*, "
            "c10::BFloat16 const*, float, int, int)"),
        "add_rmsnorm": (
            "std::enable_if<((8)==(0))||"
            "(!_typeConvert<c10::BFloat16>::exists), void>::type "
            "vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>"
            "(c10::BFloat16*, c10::BFloat16*, "
            "c10::BFloat16 const*, float, int, int)"),
        "paged_kv_cache": (
            "void vllm::reshape_and_cache_flash_kernel<"
            "__maca_bfloat16, __maca_bfloat16, "
            "(vllm::Fp8KVCacheDataType)0>"
            "(__maca_bfloat16 const*, __maca_bfloat16 const*, "
            "__maca_bfloat16*, __maca_bfloat16*, long const*, "
            "long, long, long, long, long, int, int, int, "
            "float const*, float const*)"),
    }


# The typed PyTorch vec4 elementwise symbol the A800 capture publishes, keyed
# by the Attr operation.
_A800_ELEMENTWISE_RUNTIME_SYMBOLS = {
    "add": (
        "void at::native::vectorized_elementwise_kernel<(int)4, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, "
        "std::array<char *, (unsigned long)3>>(int, T2, T3)"
    ),
    "mul": (
        "void at::native::vectorized_elementwise_kernel<(int)4, "
        "at::native::BinaryFunctor<c10::BFloat16, "
        "c10::BFloat16, c10::BFloat16, "
        "at::native::binary_internal::MulFunctor<float>>, "
        "std::array<char *, (unsigned long)3>>(int, T2, T3)"
    ),
}


# The vLLM 0.11.0 runtime symbol each A800 Vector kernel publishes, keyed by
# the canonical variant.  A variant absent here has no retained symbol, so its
# observation is not bound to one.
_A800_VECTOR_RUNTIME_SYMBOLS = {
    "rmsnorm": (
        "void vllm::rms_norm_kernel<c10::BFloat16>"
        "(T1 *, const T1 *, long, const T1 *, float, int, int)"),
    "add_rmsnorm": (
        "std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, "
        "void>::type vllm::fused_add_rms_norm_kernel<"
        "c10::BFloat16, (int)8>(T1 *, long, T1 *, const T1 *, "
        "float, int, int)"),
    "rope": (
        "void vllm::rotary_embedding_kernel<c10::BFloat16, "
        "(bool)1>(const long *, T1 *, T1 *, const T1 *, int, "
        "long, long, long, int, int, int)"),
    "silu": (
        "void vllm::act_and_mul_kernel<c10::BFloat16, "
        "&vllm::silu_kernel<c10::BFloat16>, (bool)1>"
        "(T1 *, const T1 *, int)"),
    "paged_kv_cache": (
        "void vllm::reshape_and_cache_flash_kernel<"
        "__nv_bfloat16, __nv_bfloat16, "
        "(vllm::Fp8KVCacheDataType)0>(const T1 *, const T1 *, "
        "T2 *, T2 *, const long *, long, long, long, long, long, "
        "int, int, int, const float *, const float *)"),
}


_A800_CUBLASLT_OUTPUT_DTYPE_REDUCER = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __half, "
    "__half, float, __half, (bool)0, __half, __half, __half, (bool)1, "
    "(bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, "
    "const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, "
    "const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 "
    "*, const T6 *, const T6 *, const T6 *, const T6 *)"
)

_H200_CUBLASLT_FP32_PARTIAL_REDUCER = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, "
    "__half, float, __half, (bool)0, float, __half, __half, (bool)1, "
    "(bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, "
    "const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, "
    "const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 "
    "*, const T6 *, const T6 *, const T6 *, const T6 *)"
)


_GEMM_ATTR_DTYPES = frozenset({"bf16", "fp16", "float16", "fp8", "int8"})


def _normalize_gemm(op_attributes: Mapping, kdata: Mapping,
                    hardware_id: str, hardware,
                    case_split_k: Optional[int],
                    case_partial_bytes: Optional[int],
                    case_partial_dram_bytes: Optional[int]) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    lowered = runtime_name.lower()
    dtype = _non_empty_string(
        op_attributes, "dtype", "GEMM Attr").lower()
    if dtype not in _GEMM_ATTR_DTYPES:
        raise ValueError(f"unsupported GEMM Attr.dtype: {dtype!r}")
    dtype_bytes = DTYPE_BYTES[dtype]
    shape = {
        "b": _positive_int(op_attributes, "B", "GEMM Attr"),
        "m": _positive_int(op_attributes, "M", "GEMM Attr"),
        "n": _positive_int(op_attributes, "N", "GEMM Attr"),
        "k": _positive_int(op_attributes, "K", "GEMM Attr"),
    }
    if hardware_id == "nvidia_h200":
        aggregate = legacy_h200_bf16_aggregate_projection(op_attributes, metrics, hardware)
        if aggregate is not None:
            if case_split_k is not None:
                raise ValueError("legacy BF16 aggregate must not claim an observed effective split")
            if aggregate.execution_variant == "gemm":
                return KernelSpec(KernelKind.GEMM, hardware_id, shape, dtype="bf16", variant="gemm",
                    launch=aggregate, attributes={"output_dtype": None})
            return KernelSpec(KernelKind.VECTOR, hardware_id,
                {"element_count": shape["b"] * shape["m"] * shape["n"], "dtype_bytes": 2},
                dtype="bf16", variant="gemm_reduce", launch=aggregate,
                attributes={"source_gemm_shape": shape,
                    "split_k": int(aggregate.parameters["modeled_split_k"])})
    non_tensor = is_non_tensor(
        "GEMM", runtime_name, ascend_op_type=metrics["op_type"])
    if non_tensor and "gemv" not in lowered:
        is_split_reducer = (
            "splitkreduce" in lowered or "elementwiseadd" in lowered)
        if is_split_reducer:
            if case_split_k is None:
                raise ValueError(
                    "GEMM split reducer requires the sibling tensor "
                    "launch split_k")
            if case_split_k <= 1:
                raise ValueError(
                    "GEMM split reducer requires tensor split_k > 1")
            variant = "gemm_reduce"
            attributes = {"split_k": case_split_k}
            observed_aux_algorithm = (
                "observed_mcblas_gemm_aux"
                if "elementwiseadd" in lowered
                else "observed_gemm_aux")
        elif "reduce" in lowered:
            raise ValueError(
                "GEMM reduction identity lacks a source-backed split count")
        elif any(token in lowered for token in (
                "elementwise", "vectorized", "permute")):
            variant = "gemm_elementwise"
            attributes = {}
        else:
            raise ValueError(
                f"unknown GEMM auxiliary implementation: {runtime_name}")
        launch = observed_launch_config(
            metrics, KernelKind.VECTOR,
            {"element_count": shape["b"] * shape["m"] * shape["n"],
             "dtype_bytes": dtype_bytes},
            algorithm=(
                observed_aux_algorithm
                if is_split_reducer else "observed_gemm_aux"),
            variant=variant)
        if is_split_reducer:
            # Both retained cuBLAS splitKreduce and mcBLAS
            # DevDstElementwiseAdd ABIs consume FP32 accumulator partials
            # before storing the requested output dtype.  The width is a
            # discrete runtime-template fact, not a duration-derived value.
            if case_partial_bytes is None or case_partial_bytes <= 0:
                raise ValueError(
                    "GEMM split reducer requires a source-backed partial "
                    "element width")
            if (case_partial_dram_bytes is None
                    or case_partial_dram_bytes not in (
                        0, case_partial_bytes)):
                raise ValueError(
                    "GEMM split reducer requires an explicit zero or full "
                    "partial DRAM width")
            launch = dataclasses.replace(
                launch,
                parameters={
                    "split_k_partial_bytes_per_element": float(
                        case_partial_bytes),
                    "split_k_partial_dram_bytes_per_element": float(
                        case_partial_dram_bytes),
                },
            )
            if (hardware_id == "nvidia_a800"
                    and runtime_name == _A800_CUBLASLT_OUTPUT_DTYPE_REDUCER):
                source_parameters = (
                    cublaslt_output_dtype_reducer_source_parameters(
                        element_count=shape["b"] * shape["m"] * shape["n"],
                        split_k=case_split_k,
                        grid=launch.grid,
                        block=launch.block,
                    ))
                launch = dataclasses.replace(
                    launch,
                    parameters={**launch.parameters, **source_parameters},
                )
            elif (hardware_id == "nvidia_h200"
                    and runtime_name == _H200_CUBLASLT_FP32_PARTIAL_REDUCER):
                if dtype not in ("fp16", "float16") or case_partial_bytes != 4:
                    raise ValueError("retained SM90 reducer requires FP32 partials and FP16 output")
                source_parameters = cublaslt_fp32_partial_reducer_source_parameters(
                    element_count=shape["b"] * shape["m"] * shape["n"],
                    split_k=case_split_k, grid=launch.grid, block=launch.block)
                launch = dataclasses.replace(launch,
                    parameters={**launch.parameters, **source_parameters})
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={
                "element_count": shape["b"] * shape["m"] * shape["n"],
                "dtype_bytes": dtype_bytes,
            },
            dtype=dtype,
            variant=variant,
            launch=launch,
            attributes=attributes,
        )

    variant = "gemv" if non_tensor else "gemm"
    if lowered.startswith("mcblas__"):
        observed_algorithm = "observed_mcblas_gemm"
    else:
        observed_algorithm = "observed_gemm"
    launch = observed_launch_config(
        metrics, KernelKind.GEMM, shape,
        algorithm=observed_algorithm, variant=variant)
    nvjet = parse_nvjet_gemm_symbol(runtime_name)
    if nvjet is None:
        launch_split_k = metrics["grid_z"]
    else:
        # Hopper nvjet keeps grid.z == 1; its K split is the cluster-padded
        # tile multiplicity of the launched grid (a launch-geometry fact).
        launch_split_k = observed_nvjet_gemm_split_k(op_attributes, metrics)
        launch = dataclasses.replace(
            launch,
            # NVJet's symbol names the logical output-tile cluster. The
            # retained WP0 cuLaunchKernelEx observations flatten that shape
            # onto physical CUDA x (e.g. logical 4x2 -> physical 8x1x1).
            # Keep the two logical axes below for operand ownership, rather
            # than imposing their layout on the observed physical grid.
            cluster=(nvjet.cluster_n_tiles * nvjet.cluster_m_tiles, 1, 1),
            tiling=dataclasses.replace(
                launch.tiling, split_k=launch_split_k))
    if case_split_k is not None and launch_split_k != case_split_k:
        raise ValueError(
            "GEMM tensor launch disagrees with case split_k: "
            f"launch={launch_split_k}, case={case_split_k}")
    parameters = {
        "compulsory_traffic": 0.0,
        "weight_tile_rows": 0.0,
        "mte2_tile_bytes": 0.0,
        "aiv_mixed": 0.0,
        "n_major_traffic": 0.0,
        "split_k_partial_bytes_per_element": (
            float(case_partial_bytes)
            if launch_split_k > 1 and case_partial_bytes is not None
            else 0.0),
        "split_k_partial_dram_bytes_per_element": (
            float(case_partial_dram_bytes)
            if (launch_split_k > 1
                and case_partial_dram_bytes is not None)
            else 0.0),
    }
    if nvjet is not None:
        # NVJet is the Hopper TMA-fed GEMM family. This records transport,
        # not a multicast operand, persistent raster, or a duration price.
        parameters["operand_transport_tma"] = 1.0
        parameters["cluster_output_m_tiles"] = float(nvjet.cluster_m_tiles)
        parameters["cluster_output_n_tiles"] = float(nvjet.cluster_n_tiles)
        parameters.update(cublaslt_narrow_tensor_instruction_parameters(runtime_name))
        parameters.update(cublaslt_uniform_split_k_parameters(runtime_name, split_k=launch_split_k))
    tiling = launch.tiling
    if not non_tensor and hardware_id == "ascend_910b":
        # The Dataset owns the exact duration-free CANN payload emitted by its
        # measured rank-two operation.  Decode that observation directly;
        # Serving independently selects a payload-derived program from
        # launch_data, and both paths meet at CannGemmLaunchProgram.
        selection_shape = {
            "b": 1,
            "m": shape["b"] * shape["m"],
            "n": shape["n"],
            "k": shape["k"],
        }
        observation = metrics["ascend"]["gemm_launch"]
        expected_observation_fields = {
            "schema_version", "software_revision", "input_rank",
            "kernel_family", "kernel_symbol", "block_dim",
            "tiling_payload_hex", "tiling_payload_sha256",
            "capture_baseline_commit", "capture_artifact_sha256",
            "capture_sha256",
            "duration_consumed", "device_kernel_submitted",
        }
        if (not isinstance(observation, Mapping)
                or set(observation) != expected_observation_fields):
            raise ValueError(
                "Ascend 910B GEMM Dataset launch observation fields differ")
        if (observation["schema_version"] != 1
                or observation["software_revision"] != "cann_8.5.1"
                or observation["input_rank"] != 2
                or observation["duration_consumed"] is not False
                or observation["device_kernel_submitted"] is not False):
            raise ValueError(
                "Ascend 910B GEMM Dataset launch observation contract differs")
        if runtime_name not in (
                observation["kernel_symbol"],
                f'{observation["kernel_symbol"]}_mix_aic'):
            raise ValueError(
                "Ascend 910B GEMM Dataset payload symbol disagrees with its "
                "profiler record")
        if observation["block_dim"] != launch.grid_size:
            raise ValueError(
                "Ascend 910B GEMM Dataset payload blockDim disagrees with its "
                "profiler record")
        for field in (
                "tiling_payload_sha256", "capture_artifact_sha256",
                "capture_sha256"):
            value = observation[field]
            if (not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None):
                raise ValueError(
                    f"Ascend 910B GEMM Dataset {field} is not SHA256")
        baseline = observation["capture_baseline_commit"]
        if (not isinstance(baseline, str)
                or re.fullmatch(r"[0-9a-f]{40}", baseline) is None):
            raise ValueError(
                "Ascend 910B GEMM Dataset capture baseline is not a commit")
        payload_hex = observation["tiling_payload_hex"]
        if not isinstance(payload_hex, str):
            raise TypeError(
                "Ascend 910B GEMM Dataset tiling payload must be hexadecimal")
        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError as error:
            raise ValueError(
                "Ascend 910B GEMM Dataset tiling payload is not hexadecimal") \
                from error
        if hashlib.sha256(payload).hexdigest() != (
                observation["tiling_payload_sha256"]):
            raise ValueError(
                "Ascend 910B GEMM Dataset tiling payload SHA256 changed")
        projected_launch = launch_from_cann_gemm_observation(
            payload, selection_shape, observation["block_dim"],
            observation["kernel_symbol"], observation["kernel_family"],
            variant, hardware)
        tiling = projected_launch.tiling
        parameters = dict(projected_launch.parameters)
        if tiling is None:
            raise ValueError(
                "Ascend 910B GEMM launch projection requires a macro tile")
        source_program = projected_launch.source_program
        launch = dataclasses.replace(
            launch,
            algorithm=projected_launch.algorithm,
            grid=projected_launch.grid,
            tiling=tiling,
            parameters=parameters,
            execution_variant=projected_launch.execution_variant,
            source_program=source_program,
        )
    elif not non_tensor and hardware.aic_aiv_separated:
        raise NotImplementedError(
            f"{hardware_id} tensor GEMM requires an explicit macro tile")
    elif not non_tensor and tiling is None:
        if hardware.kernel_memory_model not in (
                "pim_bank", "systolic_scratchpad"):
            raise ValueError(
                "tensor GEMM observation has no parseable tile on a tiled "
                f"architecture: {runtime_name!r}")
        # Cacheless architectures do not encode a GPU tile in their runtime
        # symbol. A concrete structural tile satisfies the common launch
        # contract; their architecture path prices traffic independently of it.
        tiling = TilingSpec(
            tile_shape={"m": 16, "n": 16, "k": 16},
            split_k=metrics["grid_z"],
        )

    if not non_tensor and observed_algorithm == "observed_mcblas_gemm":
        if tiling is None:
            raise ValueError("observed mcBLAS GEMM requires a concrete tile")
        m_tiles = math.ceil(
            (shape["b"] * shape["m"]) / tiling.tile_shape["m"])
        n_tiles = math.ceil(shape["n"] / tiling.tile_shape["n"])
        swizzle = find_gemm_swizzle_convention(
            m_tiles, n_tiles, launch.grid[0], launch.grid[1])
        parameters["n_major_traffic"] = float(swizzle is None)

    # One observed XMMa launch has no valid swizzled LRU schedule in the
    # historical model. Stamp compulsory traffic as an explicit fact of this
    # exact observed implementation and launch geometry.
    if ("xmma" in lowered
            and shape == {"b": 512, "m": 1, "n": 5120, "k": 5120}
            and tiling is not None
            and tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            == (128, 192, 32)):
        parameters["compulsory_traffic"] = 1.0
    cta_raster = None
    # No verified CTA raster word layout exists for nvjet (H200 WP0 launch
    # report): its launch keeps the schedule-free traffic envelope instead of
    # decoding a CUTLASS swizzle from a cluster-shaped grid by coincidence.
    if (nvjet is None and not non_tensor and tiling is not None
            and not hardware.aic_aiv_separated):
        # A launch whose selection retains the vendor CTA map uses it.  A
        # launch that does not can still decode one when its grid shape is
        # itself the grouped encoding: a grid of ``tiles * group`` by
        # ``ceil(other / group)`` states the group, and a grid that is one CTA
        # per tile states a direct raster.  A grid that is neither mounts no
        # map and keeps the schedule-free envelope.
        if observed_algorithm == "observed_gemm":
            cta_raster = observed_gemm_cta_raster(
                shape, dtype, variant,
                tuple(tiling.tile_shape[name] for name in ("m", "n", "k")),
                tiling.split_k)
        if cta_raster is None:
            cta_raster = _decoded_grid_cta_raster(
                shape, tiling, launch.grid, metrics["grid_z"])
    launch = dataclasses.replace(launch, tiling=tiling, parameters=parameters)
    if cta_raster is not None:
        launch = dataclasses.replace(launch, source_program=cta_raster)
    return KernelSpec(
        kind=KernelKind.GEMM,
        hardware_id=hardware_id,
        shape=shape,
        dtype=dtype,
        variant=variant,
        launch=launch,
    )


def _decoded_grid_cta_raster(shape, tiling, grid, grid_z):
    """Decode the CTA map a tensor GEMM grid shape states, when it states one."""

    m_tiles = math.ceil((shape["b"] * shape["m"]) / tiling.tile_shape["m"])
    n_tiles = math.ceil(shape["n"] / tiling.tile_shape["n"])
    swizzle = find_gemm_swizzle_convention(m_tiles, n_tiles, grid[0], grid[1])
    if swizzle is None or grid[2] != grid_z:
        return None
    axis, width = swizzle
    if width & (width - 1):
        return None
    convention = "cutlass_n_group" if axis == "M" else "cutlass_m_group"
    return GpuCtaRasterProgram(
        convention, width.bit_length() - 1, False, grid_z)


def _normalize_eager_attn(op_attributes: Mapping, kdata: Mapping,
                          hardware_id: str, hardware) -> KernelSpec:
    """Normalize one decomposed QK-softmax-PV observation.

    Eager attention deliberately has no FlashAttention tile, split, layout, or
    fused-pipeline attributes in its production contract. Dataset-only fields
    are still validated here so malformed observations fail at the boundary.
    """

    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    if is_non_tensor(
            "EAGERATTN", runtime_name,
            ascend_op_type=metrics["op_type"]):
        raise ValueError(
            "EAGERATTN dataset entry must describe its decomposed tensor path")

    _positive_int(op_attributes, "S", "EAGERATTN Attr")
    _positive_int(op_attributes, "SP", "EAGERATTN Attr")
    _non_empty_string(op_attributes, "kv_layout", "EAGERATTN Attr")
    _non_empty_string(op_attributes, "kv_dist", "EAGERATTN Attr")
    causal = op_attributes["causal"]
    if not isinstance(causal, bool):
        raise ValueError("EAGERATTN causal must be boolean")

    shape = {
        "batch": _positive_int(op_attributes, "B", "EAGERATTN Attr"),
        "q_len": _positive_int(
            op_attributes, "Q_LEN", "EAGERATTN Attr"),
        "kv_len": _positive_int(
            op_attributes, "KV_LEN", "EAGERATTN Attr"),
        "head_dim": _positive_int(
            op_attributes, "D", "EAGERATTN Attr"),
        "num_query_heads": _positive_int(
            op_attributes, "HQ", "EAGERATTN Attr"),
        "num_kv_heads": _positive_int(
            op_attributes, "HKV", "EAGERATTN Attr"),
    }
    launch = observed_launch_config(
        metrics, KernelKind.EAGER_ATTN, shape,
        algorithm="observed_eager_attn", variant="decomposed")
    return KernelSpec(
        kind=KernelKind.EAGER_ATTN,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant="decomposed",
        launch=launch,
        attributes={},
    )


def _validate_observed_vector_element_count(
        op_attributes: Mapping, label: str, canonical_count: int,
        legacy_count: int) -> None:
    """Accept only the documented legacy or canonical vector count schema.

    Existing protected datasets stored ``tok * hidden`` for SILU and KV-cache,
    while their structured fields describe the actual launched shape.  Resolve
    that known schema difference here at the observation boundary so production
    predictors never select a shape based on whichever keys happen to exist.
    """

    observed = _positive_int(
        op_attributes, "element_count", f"{label} Attr")
    allowed = {canonical_count, legacy_count}
    if observed not in allowed:
        raise ValueError(
            f"{label} Attr.element_count {observed} matches neither canonical "
            f"structured count {canonical_count} nor documented legacy count "
            f"{legacy_count}")


def _validate_canonical_vector_element_count(
        op_attributes: Mapping, label: str, canonical_count: int) -> None:
    """Require the observation count to match its structured shape exactly."""

    observed = _positive_int(
        op_attributes, "element_count", f"{label} Attr")
    if observed != canonical_count:
        raise ValueError(
            f"{label} Attr.element_count {observed} disagrees with canonical "
            f"structured count {canonical_count}")


def _require_source_bound_vector_observation(
        hardware_id: str, variant: str, runtime_name: str,
        observed_launch, canonical_launch, expected_runtime_name: str) -> None:
    """Prove a runtime launch before admitting source-bound parameters."""

    if runtime_name != expected_runtime_name:
        raise ValueError(
            f"{hardware_id} {variant} runtime symbol disagrees with the "
            "source-bound implementation: "
            f"expected={expected_runtime_name!r}, observed={runtime_name!r}")
    if (observed_launch.grid != canonical_launch.grid
            or observed_launch.block != canonical_launch.block):
        raise ValueError(
            f"{hardware_id} {variant} profiler geometry disagrees with the "
            "source-bound launch: "
            f"expected={canonical_launch.grid}/{canonical_launch.block}, "
            f"observed={observed_launch.grid}/{observed_launch.block}")


def _normalize_vector(op_type: str, op_attributes: Mapping, kdata: Mapping,
                      hardware_id: str, hardware) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    typed_a800_rope = False
    typed_c500_rope = False
    typed_ascend_qkv_rmsnorm_rope = False
    if op_type == "NORM":
        kind = KernelKind.VECTOR
        variant = _non_empty_string(
            op_attributes, "op", "NORM Attr").lower()
    elif op_type == "KV_CACHE":
        kind = KernelKind.KV_CACHE
        variant = "paged_kv_cache"
    elif op_type in (
            "ROTARY", "ACTIVATION", "ELEMENTWISE",
            "REDUCE", "GATHER"):
        kind = KernelKind.VECTOR
        variants = {
            "ROTARY": "rotary",
            "ACTIVATION": "activation",
            "ELEMENTWISE": "elementwise",
            "REDUCE": "reduce",
            "GATHER": "gather",
        }
        variant = variants[op_type]
        if op_type == "ACTIVATION":
            variant = _non_empty_string(
                op_attributes, "op", "ACTIVATION Attr").lower()
            if variant not in ("activation", "silu"):
                raise ValueError(
                    f"unsupported ACTIVATION Attr.op: {variant!r}")
    else:
        raise ValueError(f"unknown VECTOR observation type: {op_type!r}")
    if op_type == "NORM":
        shape = {
            "rows": _positive_int(op_attributes, "tok", "NORM Attr"),
            "hidden": _positive_int(
                op_attributes, "hidden", "NORM Attr"),
        }
    elif op_type == "ROTARY":
        operator = _non_empty_string(
            op_attributes, "op", "ROTARY Attr").lower()
        if operator not in ("rope", "rotary", "qkv_rmsnorm_rope"):
            raise ValueError(
                f"unsupported ROTARY Attr.op: {operator!r}")
        seq_len = _positive_int(op_attributes, "tok", "ROTARY Attr")
        num_query_heads = _positive_int(
            op_attributes, "num_heads", "ROTARY Attr")
        num_kv_heads = _positive_int(
            op_attributes, "num_kv_heads", "ROTARY Attr")
        head_dim = _positive_int(
            op_attributes, "head_dim", "ROTARY Attr")
        typed_ascend_qkv_rmsnorm_rope = (
            operator == "qkv_rmsnorm_rope")
        if typed_ascend_qkv_rmsnorm_rope:
            if (hardware_id != "ascend_910b"
                    or hardware.kernel_dtype != "bf16"):
                raise ValueError(
                    "qkv_rmsnorm_rope observation requires the pinned "
                    "Ascend 910B BF16 program")
            rotary_dim = _positive_int(
                op_attributes, "rotary_dim", "ROTARY Attr")
            canonical_count = (
                seq_len * (num_query_heads + 2 * num_kv_heads) * head_dim)
            _validate_canonical_vector_element_count(
                op_attributes, "QKV_RMSNORM_ROPE", canonical_count)
            shape = {
                "seq_len": seq_len,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "rotary_dim": rotary_dim,
            }
            variant = operator
        else:
            # Dataset element_count historically denotes the Q tensor. K is
            # represented independently by num_kv_heads and is included when
            # the production KernelSpec derives its total rotary work.
            canonical_count = seq_len * num_query_heads * head_dim
            _validate_canonical_vector_element_count(
                op_attributes, "ROTARY", canonical_count)
            shape = {
                "seq_len": seq_len,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        # The A800 dataset runner is source-pinned to vLLM's full-dimension
        # NeoX call with both Q and K.  Only the matching BF16 template identity
        # is enriched into the typed ``rope`` contract; all other ROTARY
        # observations retain the cross-hardware ``rotary`` semantics and do
        # not borrow its SASS lowering.
        source_identity = (
            "vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>")
        typed_a800_rope = (
            hardware_id == "nvidia_a800"
            and hardware.kernel_dtype == "bf16"
            and operator == "rope"
            and metrics["kernel_name"].count(source_identity) == 1
        )
        c500_source_identity = (
            "void vllm::rotary_embedding_kernel<c10::BFloat16, true>"
            "(long const*, c10::BFloat16*, c10::BFloat16*, "
            "c10::BFloat16 const*, int, long, long, int, int, int)")
        typed_c500_rope = (
            hardware_id == "metax_c500"
            and hardware.kernel_dtype == "bf16"
            and operator == "rope"
            and metrics["kernel_name"] == c500_source_identity
        )
        if typed_a800_rope or typed_c500_rope:
            variant = "rope"
            shape["rotary_dim"] = head_dim
    elif op_type == "ACTIVATION":
        tokens = _positive_int(op_attributes, "tok", "ACTIVATION Attr")
        intermediate = _positive_int(
            op_attributes, "intermediate", "ACTIVATION Attr")
        canonical_count = tokens * intermediate
        legacy_count = tokens * _positive_int(
            op_attributes, "hidden", "ACTIVATION Attr")
        _validate_observed_vector_element_count(
            op_attributes, "ACTIVATION", canonical_count, legacy_count)
        shape = {
            "tok": tokens,
            "intermediate": intermediate,
            "element_count": canonical_count,
        }
    elif op_type == "KV_CACHE":
        batch = _positive_int(op_attributes, "batch", "KV_CACHE Attr")
        if batch != 1:
            raise ValueError(
                "KV_CACHE Attr.batch is a legacy label, not a physical "
                f"dimension; expected 1, got {batch}")
        seq_len = _positive_int(op_attributes, "tok", "KV_CACHE Attr")
        num_kv_heads = _positive_int(
            op_attributes, "num_kv_heads", "KV_CACHE Attr")
        head_dim = _positive_int(
            op_attributes, "head_dim", "KV_CACHE Attr")
        canonical_count = seq_len * num_kv_heads * head_dim
        legacy_count = seq_len * _positive_int(
            op_attributes, "hidden", "KV_CACHE Attr")
        _validate_observed_vector_element_count(
            op_attributes, "KV_CACHE", canonical_count, legacy_count)
        shape = {
            "seq_len": seq_len,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
    elif op_type in ("ELEMENTWISE", "REDUCE", "GATHER"):
        shape = {"element_count": _positive_int(
            op_attributes, "element_count", f"{op_type} Attr")}
    else:
        raise ValueError(f"unknown VECTOR observation type: {op_type!r}")
    launch = observed_launch_config(
        metrics, kind, shape, algorithm="observed_vector", variant=variant)
    attributes = {
        name: value for name, value in op_attributes.items()
        if name in (
            "reads", "writes", "reduction_factor",
            "gather_factor", "index_count", "index_bytes", "scattered")
    }
    if typed_ascend_qkv_rmsnorm_rope:
        bias = op_attributes["bias"]
        is_neox = op_attributes["is_neox"]
        if not isinstance(bias, bool) or not isinstance(is_neox, bool):
            raise TypeError(
                "QKV_RMSNORM_ROPE bias and is_neox must be boolean")
        attributes = {"bias": bias, "is_neox": is_neox}
    elif typed_a800_rope or typed_c500_rope:
        attributes = {"is_neox": True, "key_present": True}
    elif variant == "elementwise":
        # The Vector dataset runner executes a binary add/mul: two input reads
        # and one output write. Keep the semantic traffic contract independent
        # of profiler counters.
        if "op2" in op_attributes:
            operation = _non_empty_string(
                op_attributes, "op2", "ELEMENTWISE Attr").lower()
        elif (hardware_id == "samsung_hbm_pim"
              and op_attributes["op"] == "elementwise"):
            # The protected historical UniNDP dataset predates the add/mul
            # label, but its ``elewise`` primitive explicitly contains one
            # ordinary arithmetic stage.  Both add and mul map to the same
            # single-operation demand, so normalize that legacy simulator
            # contract without changing target-hardware behavior.
            operation = "add"
        else:
            raise ValueError(
                "ELEMENTWISE Attr requires explicit op2 operation identity")
        if operation not in ("add", "mul"):
            raise ValueError(
                f"unsupported ELEMENTWISE Attr.op2: {operation!r}")
        attributes = {
            "reads": 2, "writes": 1, "operation": operation}
        if hardware_id == "nvidia_a800":
            expected_symbol = _A800_ELEMENTWISE_RUNTIME_SYMBOLS[operation]
            if metrics["kernel_name"] != expected_symbol:
                raise ValueError(
                    "A800 ELEMENTWISE runtime symbol disagrees with the "
                    "typed PyTorch vec4 implementation: "
                    f"expected={expected_symbol!r}, "
                    f"observed={metrics['kernel_name']!r}")
            variant = "elementwise_pytorch_vec4"
            launch = dataclasses.replace(
                launch, execution_variant=variant)
    elif variant == "reduce":
        attributes = {"reduction_factor": op_attributes["reduction_factor"]}
    elif variant == "gather":
        attributes = {
            name: op_attributes[name]
            for name in (
                "gather_factor", "index_count", "index_bytes", "scattered")
        }
    elif variant in (
            "rmsnorm", "add_rmsnorm", "rotary", "rope", "activation",
            "silu", "paged_kv_cache", "qkv_rmsnorm_rope"):
        if variant == "paged_kv_cache":
            # All three pinned runners use a paged cache with the same storage
            # dtype as K/V and an arange slot mapping.  These are runner/source
            # semantics, not profiler-duration observations.
            attributes = PAGED_KV_CACHE_ATTRIBUTES
    else:
        raise ValueError(f"unknown normalized VECTOR variant: {variant!r}")
    if variant == "silu":
        if hardware_id in _SILU_RUNTIME_SYMBOLS:
            expected_symbol = _SILU_RUNTIME_SYMBOLS[hardware_id]
            if metrics["kernel_name"] != expected_symbol:
                raise ValueError(
                    "SILU observation runtime symbol disagrees with its "
                    f"operator identity: hardware={hardware_id!r}, "
                    f"expected={expected_symbol!r}, "
                    f"observed={metrics['kernel_name']!r}")
    if hardware_id == "nvidia_a800" and variant == "rmsnorm":
        expected_symbol = _A800_VECTOR_RUNTIME_SYMBOLS["rmsnorm"]
        if metrics["kernel_name"] != expected_symbol:
            raise ValueError(
                "A800 RMSNorm observation runtime symbol disagrees with "
                "the retained vLLM BF16 implementation: "
                f"expected={expected_symbol!r}, "
                f"observed={metrics['kernel_name']!r}")
    if hardware_id == "metax_c500" and variant in _METAX_VECTOR_RUNTIME_SYMBOLS:
        expected_symbol = _METAX_VECTOR_RUNTIME_SYMBOLS[variant]
        if metrics["kernel_name"] != expected_symbol:
            raise ValueError(
                "MetaX C500 MACA 3.7.2 Vector observation runtime symbol "
                "disagrees with the retained plugin artifact: "
                f"variant={variant!r}, expected={expected_symbol!r}, "
                f"observed={metrics['kernel_name']!r}")
    if hardware_id == "ascend_910b" and variant == "elementwise":
        operation = attributes["operation"]
        expected_symbols = _ASCEND_ELEMENTWISE_RUNTIME_SYMBOLS[operation]
        if metrics["kernel_name"] not in expected_symbols:
            raise ValueError(
                "Ascend 910B ELEMENTWISE runtime symbol disagrees with the "
                "CANN 8.5.1 BF16 BroadcastSch implementation: "
                f"operation={operation!r}, "
                f"expected_one_of={expected_symbols!r}, "
                f"observed={metrics['kernel_name']!r}")
    # C500 NORM, SiLU, and typed RoPE observations have exact retained source
    # symbols; their observed geometry is checked against the MACA 3.7.2
    # selector before source-semantic parameters are admitted. Other C500
    # observations remain self-contained. A800 and 910B observations below are
    # likewise bound to the same source revision as their production selectors.
    typed_c500_norm = (
        hardware_id == "metax_c500"
        and variant in ("rmsnorm", "add_rmsnorm"))
    typed_c500_silu = (
        hardware_id == "metax_c500" and variant == "silu")
    typed_c500_kv_cache = (
        hardware_id == "metax_c500" and variant == "paged_kv_cache")
    if _is_source_bound_vector_observation(
            hardware_id, variant,
            c500_typed=(typed_c500_norm, typed_c500_silu, typed_c500_rope,
                        typed_c500_kv_cache)):
        selection_attributes = dict(attributes)
        selection_attributes["launch_policy"] = "injected_launch"
        if hardware_id == "ascend_910b" and variant == "silu":
            selection_attributes["injected_grid_size"] = launch.grid_size
        canonical_launch = select_launch_config(
            hardware_id, kind, shape, hardware.kernel_dtype, variant,
            selection_attributes, hardware)
        if hardware_id == "nvidia_a800":
            if variant == "elementwise_pytorch_vec4":
                expected_runtime_name = expected_symbol
            elif variant in _A800_VECTOR_RUNTIME_SYMBOLS:
                expected_runtime_name = _A800_VECTOR_RUNTIME_SYMBOLS[variant]
            else:
                expected_runtime_name = None
            if expected_runtime_name is not None:
                _require_source_bound_vector_observation(
                    hardware_id, variant, metrics["kernel_name"], launch,
                    canonical_launch, expected_runtime_name)
        if hardware_id == "metax_c500" and typed_c500_rope:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, c500_source_identity)
        if typed_c500_norm:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _METAX_VECTOR_RUNTIME_SYMBOLS[variant])
        if typed_c500_silu:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _SILU_RUNTIME_SYMBOLS[hardware_id])
        if typed_c500_kv_cache:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _METAX_VECTOR_RUNTIME_SYMBOLS[variant])
        if (hardware_id == "ascend_910b"
                and variant == "paged_kv_cache"):
            tiling_key = int(
                canonical_launch.parameters["cann_tiling_key"])
            expected_kernel_name = f"reshape_and_cache_{tiling_key}"
            if metrics["kernel_name"] != expected_kernel_name:
                raise ValueError(
                    "Ascend 910B KV_CACHE source selector disagrees with "
                    "profiler kernel identity: "
                    f"expected={expected_kernel_name!r}, "
                    f"observed={metrics['kernel_name']!r}")
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, expected_kernel_name)
        if hardware_id == "ascend_910b" and variant == "silu":
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _SILU_RUNTIME_SYMBOLS[hardware_id])
        if typed_ascend_qkv_rmsnorm_rope:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, "split_qkv_rmsnorm_rope_kernel")
        launch_replacements = {
            "parameters": canonical_launch.parameters,
        }
        if typed_c500_norm or typed_ascend_qkv_rmsnorm_rope:
            # Runtime symbol and launch geometry were proved against the
            # retained source program above. The observation label is an
            # evaluator ingestion detail, not a second physical algorithm:
            # publish exactly the launch identity selected for Serving.
            launch_replacements["algorithm"] = canonical_launch.algorithm
            launch_replacements["execution_variant"] = (
                canonical_launch.execution_variant)
            launch_replacements["source_program"] = (
                canonical_launch.source_program)
        elif (hardware_id == "ascend_910b"
              and variant in ("rmsnorm", "add_rmsnorm")):
            tiling_key = int(
                canonical_launch.parameters["cann_tiling_key"])
            expected_kernel_names = {
                ("rmsnorm", 0): (
                    "RmsNorm_normal_all_bf16_high_performance_0"),
                ("rmsnorm", 1): (
                    "RmsNorm_normal_all_bf16_high_performance_1"),
                ("add_rmsnorm", 30): (
                    "AddRmsNorm_3_high_performance_30"),
                ("add_rmsnorm", 31): (
                    "AddRmsNorm_3_high_performance_31"),
                ("add_rmsnorm", 33): (
                    "AddRmsNorm_3_high_performance_33"),
            }
            selector_key = (variant, tiling_key)
            if selector_key not in expected_kernel_names:
                raise ValueError(
                    "Ascend 910B NORM source selector produced an unknown "
                    f"variant/key pair: {selector_key!r}")
            expected_kernel_name = expected_kernel_names[selector_key]
            if (metrics["kernel_name"] != expected_kernel_name
                    or launch.grid != canonical_launch.grid
                    or launch.block != canonical_launch.block):
                raise ValueError(
                    "Ascend 910B NORM source selector disagrees with "
                    "profiler launch identity: "
                    f"expected_kernel={expected_kernel_name!r}, "
                    f"observed_kernel={metrics['kernel_name']!r}, "
                    f"expected_grid={canonical_launch.grid}, "
                    f"observed_grid={launch.grid}, "
                    f"expected_block={canonical_launch.block}, "
                    f"observed_block={launch.block}")
            # After the runtime identity and geometry agree with the fixed
            # CANN selector, normalize the evaluator-only observed label to
            # the canonical production algorithm.  This lets the production
            # predictor enforce that source-bound DAGs cannot be attached to
            # an unverified generic observation.
            launch_replacements["algorithm"] = canonical_launch.algorithm
            launch_replacements["execution_variant"] = (
                canonical_launch.execution_variant)
            launch_replacements["source_program"] = (
                canonical_launch.source_program)
        launch = dataclasses.replace(launch, **launch_replacements)
        if (hardware_id == "nvidia_a800"
                and variant == "add_rmsnorm"):
            expected_symbol = (
                "std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, "
                "void>::type vllm::fused_add_rms_norm_kernel<"
                "c10::BFloat16, (int)8>(T1 *, long, T1 *, const T1 *, "
                "float, int, int)"
            )
            if metrics["kernel_name"] == expected_symbol:
                if (launch.grid != canonical_launch.grid
                        or launch.block != canonical_launch.block):
                    raise ValueError(
                        "A800 typed Add-RMS vec8 observation disagrees with "
                        "the source-defined launch geometry")
                parameters = dict(launch.parameters)
                parameters[
                    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 1.0
                launch = dataclasses.replace(
                    launch, parameters=parameters)
    # H200 NORM source binding: the pinned vLLM SM90 RMSNorm kernel uses
    # CUB BlockReduce with the same independently-measured reduction chain
    # as the A800 service.  Bind the source marker only for the measured
    # block sizes (256, 1024) with the exact vLLM kernel identity.
    if hardware_id == "nvidia_h200" and variant in ("rmsnorm", "add_rmsnorm"):
        h200_rmsnorm_symbols = {
            "rmsnorm": "void vllm::rms_norm_kernel<c10::BFloat16, (int)8, (int)2, (bool)1>",
            "add_rmsnorm": (
                "std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type "
                "vllm::fused_add_rms_norm_kernel<c10::BFloat16, (int)8, (bool)1>"),
        }
        if (metrics["kernel_name"].startswith(h200_rmsnorm_symbols[variant])
                and launch.grid == (shape["rows"], 1, 1)
                and launch.block in ((256, 1, 1), (1024, 1, 1))
                and shape["hidden"] % 256 == 0):
            parameters = dict(launch.parameters)
            parameters["source_bound_block_reduction"] = 1.0
            launch = dataclasses.replace(launch, parameters=parameters)
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )
