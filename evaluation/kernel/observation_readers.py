"""Observed launch-evidence readers for kernel profiler observations:
observation sequence validation and split-k / FlashAttention split
facts, plus the scalar validators they share."""

from __future__ import annotations

import math
from typing import Mapping, Optional

from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.select_launch_config import (
    select_fused_moe_reduce_work_fractions,
)
from src.kernel.launch_config.tiling_spec import parse_nvjet_gemm_symbol
from evaluation.kernel.observation_parser import is_non_tensor
from evaluation.kernel.profiler_observation import extract_ncu_metrics

from evaluation.json_shape import require_positive_metric as _positive_int


_A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL = (
    "void at::native::reduce_kernel<(int)128, (int)4, "
    "at::native::ReduceOp<c10::BFloat16, "
    "at::native::func_wrapper_t<c10::BFloat16, "
    "at::native::sum_functor<c10::BFloat16, float, "
    "c10::BFloat16>::operator ()(at::TensorIterator &)::"
    "[lambda(float, float) (instance 1)]>, unsigned int, "
    "c10::BFloat16, (int)4, (int)4>>(T3)")


def _optional_positive_int(mapping: Mapping, name: str, context: str):
    value = mapping[name]
    if value is None:
        return None
    return _positive_int(mapping, name, context)


def _non_empty_string(mapping: Mapping, name: str, context: str) -> str:
    value = mapping[name]
    if not isinstance(value, str):
        raise TypeError(
            f"{context}.{name} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{context}.{name} must not be empty")
    return value


def _finite_number(mapping: Mapping, name: str, context: str):
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{context}.{name} must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(f"{context}.{name} must be finite, got {value!r}")
    return value


def _positive_int_sequence(mapping: Mapping, name: str, context: str,
                           expected_length: int, expected_maximum: int):
    value = mapping[name]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context}.{name} must be a list or tuple")
    result = tuple(value)
    if len(result) != expected_length:
        raise ValueError(
            f"{context}.{name} must contain {expected_length} values, got "
            f"{len(result)}")
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in result):
        raise TypeError(f"{context}.{name} must contain integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{context}.{name} must contain positive values")
    if max(result) != expected_maximum:
        raise ValueError(
            f"{context}.{name} maximum must equal {expected_maximum}, got "
            f"{max(result)}")
    return result


def validate_kernel_observation_sequence(
        kernel_records, hardware_id: str, *, op_info: Optional[Mapping] = None,
        hardware=None):
    """Reject per-invocation observations with ambiguous counter provenance.

    MetaX mcTracer and mcProfiler execute the runner separately.  A historical
    name-wise trace union could therefore attach one numbered layer-1 payload
    to two mutually exclusive kernel variants.  Two genuinely distinct kernel
    identities cannot share the entire nested per-invocation counter payload;
    seeing that exact duplication is evidence of the known invalid join, not a
    second kernel execution.
    """

    if not isinstance(hardware_id, str) or not hardware_id:
        raise TypeError("observation hardware_id must be a non-empty string")
    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "profiler case must contain a non-empty kernel-record sequence")
    if (op_info is None) != (hardware is None):
        raise ValueError(
            "observation sequence validation requires op_info and hardware "
            "together")
    if hardware_id == "nvidia_a800" and op_info is not None:
        _validate_nvidia_a800_moe_reduce_sequence(
            op_info, kernel_records, hardware)
        return
    if hardware_id != "metax_c500":
        return

    observations = []
    for ordinal, record in enumerate(kernel_records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"MetaX kernel record {ordinal} must be a mapping")
        payload = record["MetaX"]
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"MetaX kernel record {ordinal} payload must be a mapping")
        name = extract_ncu_metrics(record)["kernel_name"]
        for previous_ordinal, previous_name, previous_payload in observations:
            if name != previous_name and payload == previous_payload:
                raise ValueError(
                    "distinct MetaX kernel identities share an identical "
                    "per-invocation counter payload; trace/layer-1 provenance "
                    "is ambiguous: "
                    f"ordinals=({previous_ordinal}, {ordinal}), "
                    f"kernels=({previous_name!r}, {name!r})")
        observations.append((ordinal, name, payload))


def _validate_nvidia_a800_moe_reduce_sequence(
        op_info: Mapping, kernel_records, hardware) -> None:
    """Validate the exact source-derived reduction suffix and partition."""

    op_type = _non_empty_string(op_info, "Type", "op_info").upper()
    if op_type != "FUSEDMOE":
        return
    attributes = op_info["Attr"]
    shape = {
        "tokens": _positive_int(attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            attributes, "INT", "FUSED_MOE Attr"),
    }
    runtime_names = [
        extract_ncu_metrics(record)["kernel_name"]
        for record in kernel_records
    ]
    topk = shape["topk"]
    if shape["topk"] in (2, 3, 4):
        expected_symbols = (_a800_moe_sum_symbol(topk),)
        expected_launch = select_launch_config(
            "nvidia_a800", KernelKind.VECTOR, shape,
            hardware.kernel_dtype, "moe_sum",
            {"work_fraction": 1.0}, hardware)
        expected_launches = ((expected_launch.grid, expected_launch.block),)
    else:
        if hardware.kernel_dtype != "bf16":
            raise NotImplementedError(
                "A800 PyTorch 2.8 moe_reduce symbol identity is only bound "
                "for the retained BF16 binary")
        fractions = select_fused_moe_reduce_work_fractions(
            "nvidia_a800", shape, hardware)
        expected_symbols = tuple(
            _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL for _ in fractions)
        expected_launches = []
        for fraction in fractions:
            launch = select_launch_config(
                "nvidia_a800", KernelKind.VECTOR, shape,
                hardware.kernel_dtype, "moe_reduce",
                {"work_fraction": fraction}, hardware)
            expected_launches.append((launch.grid, launch.block))
        expected_launches = tuple(expected_launches)
    if len(runtime_names) < len(expected_symbols):
        raise ValueError(
            "A800 FUSEDMOE reduction suffix is incomplete: "
            f"expected={len(expected_symbols)}, actual_records={len(runtime_names)}")
    reduction_like = tuple(
        index for index, name in enumerate(runtime_names)
        if "reduce_kernel" in name.lower() or "moe_sum" in name.lower()
    )
    suffix_start = len(runtime_names) - len(expected_symbols)
    if (tuple(runtime_names[suffix_start:]) != expected_symbols
            or reduction_like != tuple(range(suffix_start, len(runtime_names)))):
        raise ValueError(
            "A800 FUSEDMOE reduction suffix disagrees with the exact "
            "vLLM 0.11.0/PyTorch 2.8 source recipe")
    observed_launches = [
        (
            (metrics["grid_x"], metrics["grid_y"], metrics["grid_z"]),
            (metrics["block_x"], metrics["block_y"], metrics["block_z"]),
        )
        for metrics in (
            extract_ncu_metrics(record)
            for record in kernel_records[suffix_start:]
        )
    ]
    if tuple(observed_launches) != expected_launches:
        raise ValueError(
            "A800 FUSEDMOE reduction physical sequence disagrees with the "
            "source-derived recipe/32-bit indexing partition: "
            f"expected={expected_launches}, observed={observed_launches}")


def _a800_moe_sum_symbol(topk: int) -> str:
    if topk not in (2, 3, 4):
        raise ValueError("A800 specialized moe_sum requires top-k 2, 3, or 4")
    return (
        "void vllm::moe::moe_sum_kernel<c10::BFloat16, "
        f"(int){topk}>(T1 *, const T1 *, int)")


def observed_flash_attention_splits(
        op_attributes: Mapping, kernel_records, hardware_id: str) -> int:
    """Derive the executed FA split count from one profiler case.

    Canonical ``SP`` is the executed count. NVIDIA exposes it in the split main
    kernel grid, while MetaX records the public API request or FlashMLA metadata
    because its grid axes are implementation-specific.  H200 FA3 runs a
    persistent varlen scheduler whose grid carries no split axis; its ``SP`` is
    the largest per-request ``num_splits_dynamic`` of the recorded FA3
    scheduler metadata (``SP_PER_REQUEST``), and ``FA3_SPLIT_COMBINE`` (static
    split bound > 1) must match the presence of the combine launch.
    """

    recorded_splits = _positive_int(
        op_attributes, "SP", "FLASH_ATTN Attr")
    batch = _positive_int(op_attributes, "B", "FLASH_ATTN Attr")
    kv_heads = _positive_int(op_attributes, "HKV", "FLASH_ATTN Attr")
    query_length = _positive_int(
        op_attributes, "Q_LEN", "FLASH_ATTN Attr")
    kv_layout = _non_empty_string(
        op_attributes, "kv_layout", "FLASH_ATTN Attr")
    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "FLASH_ATTN profiler case must contain a non-empty kernel sequence")
    if hardware_id == "nvidia_h200":
        return _observed_h200_fa3_splits(
            op_attributes, kernel_records, recorded_splits, batch)
    if query_length > 1:
        if recorded_splits != 1:
            raise ValueError("FLASH_ATTN prefill observation requires SP=1")
        return 1
    if hardware_id == "metax_c500":
        # MetaX uses different grid axes across implementations. The C500
        # collector records the scalar passed to the public API, or for MLA the
        # actual device metadata value, so the grid is not a second SP source.
        return recorded_splits
    if hardware_id != "nvidia_a800":
        raise ValueError(
            "observed GPU FlashAttention split validation supports only "
            f"nvidia_a800 and metax_c500, got {hardware_id!r}")
    candidates = set()
    stacked_kv_heads = batch * kv_heads
    for record in kernel_records:
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        if metrics["op_type"] is not None:
            raise ValueError(
                "NVIDIA A800 FLASH_ATTN record cannot contain Ascend Op Type")
        if "flash_fwd_splitkv_kernel" in lowered:
            grid_y = metrics["grid_y"]
            grid_z = metrics["grid_z"]
            if grid_y == batch and grid_z == kv_heads:
                candidates.add(1)
            elif grid_z == stacked_kv_heads:
                candidates.add(grid_y)
            elif grid_y == stacked_kv_heads:
                candidates.add(grid_z)
            elif metrics["grid_x"] == stacked_kv_heads and grid_z == 1:
                candidates.add(grid_y)
            else:
                raise ValueError(
                    "FLASH_ATTN split-kv grid does not contain B*HKV: "
                    f"grid=({metrics['grid_x']}, {grid_y}, {grid_z}), "
                    f"B*HKV={stacked_kv_heads}")
        elif "flash_fwd_kernel" in lowered and "splitkv" not in lowered:
            candidates.add(1)
        elif (kv_layout == "mla"
              and lowered == "_fwd_grouped_kernel_stage1"):
            # vLLM Triton MLA launches (batch, query-head groups, KV splits).
            # grid_y=ceil(HQ/BLOCK_H); only grid_z represents executed SP.
            candidates.add(metrics["grid_z"])

    if len(candidates) > 1:
        raise ValueError(
            "FLASH_ATTN profiler case contains conflicting observed split "
            f"counts: {sorted(candidates)}")
    if candidates:
        observed_splits = next(iter(candidates))
        if observed_splits <= 0:
            raise ValueError(
                "FLASH_ATTN observed split count must be positive")
        if observed_splits != recorded_splits:
            raise ValueError(
                "FLASH_ATTN recorded SP disagrees with profiler launch: "
                f"recorded={recorded_splits}, observed={observed_splits}")
        return recorded_splits
    raise ValueError(
        "FLASH_ATTN GPU profiler case has no recognized main-kernel launch")


def _observed_h200_fa3_splits(
        op_attributes, kernel_records, recorded_splits, batch) -> int:
    per_request = op_attributes["SP_PER_REQUEST"]
    if (not isinstance(per_request, list) or len(per_request) != batch
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in per_request)):
        raise ValueError(
            "H200 FLASH_ATTN Attr.SP_PER_REQUEST must hold one positive "
            f"split count per request: {per_request!r}")
    if max(per_request) != recorded_splits:
        raise ValueError(
            "H200 FLASH_ATTN SP must be the largest per-request split: "
            f"SP={recorded_splits}, per_request_max={max(per_request)}")
    combine_expected = op_attributes["FA3_SPLIT_COMBINE"]
    if not isinstance(combine_expected, bool):
        raise TypeError("H200 FLASH_ATTN Attr.FA3_SPLIT_COMBINE must be bool")
    names = [extract_ncu_metrics(record)["kernel_name"]
             for record in kernel_records]
    if not any("FlashAttnFwdSm90" in name for name in names):
        raise ValueError(
            "H200 FLASH_ATTN case has no FA3 FlashAttnFwdSm90 main launch")
    combine_launched = any("FlashAttnFwdCombine" in name for name in names)
    if combine_launched != combine_expected:
        raise ValueError(
            "H200 FLASH_ATTN combine launch disagrees with the FA3 scheduler "
            f"metadata: launched={combine_launched}, "
            f"FA3_SPLIT_COMBINE={combine_expected}")
    if recorded_splits > 1 and not combine_launched:
        raise ValueError("H200 FLASH_ATTN SP > 1 requires a combine launch")
    return recorded_splits


_LEGACY_H200_BF16_REDUCER_SYMBOL = 'void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)'


def legacy_h200_bf16_aggregate_projection(op_attributes, metrics, hardware=None):
    """Project only the exact six captured CUDA12.9 program/geometry pairs.

    The returned launch states modeled_split_k; effective split stays unknown.
    No counter or duration participates. Hardware-aware callers must be H200.
    """
    if (not isinstance(op_attributes, Mapping) or "dtype" not in op_attributes
            or op_attributes["dtype"] != "bf16"):
        return None
    shape = {name: _positive_int(op_attributes, field, "GEMM Attr")
             for name, field in (("b", "B"), ("m", "M"), ("n", "N"), ("k", "K"))}
    if shape["b"] != 1 or shape["k"] != 7168:
        return None
    if (shape["m"], shape["n"]) == (12, 4608):
        main_symbol = "nvjet_tst_64x16_64x16_4x1_v_bz_splitK_TNT"
    elif shape["n"] == 20480 and shape["m"] in (1, 2, 4, 6, 8):
        main_symbol = "nvjet_tst_192x8_64x8_1x1_h_bz_splitK_TNT"
    else:
        return None
    name = metrics["kernel_name"]
    if name not in (main_symbol, _LEGACY_H200_BF16_REDUCER_SYMBOL):
        return None
    ordinary = {"has_bias": False, "a_layout": "row_major", "b_layout": "row_major",
                "c_layout": "row_major", "layout": "row_major", "trans_a": False,
                "trans_b": True, "epilogue": "none", "alpha": 1}
    for key, expected in ordinary.items():
        if key in op_attributes:
            actual = op_attributes[key]
            if actual != expected or (type(expected) is bool and type(actual) is not bool):
                raise ValueError("legacy BF16 aggregate has conflicting explicit " + key)
    if "beta" in op_attributes and (isinstance(op_attributes["beta"], bool)
            or op_attributes["beta"] not in (None, "", "0", 0)):
        raise ValueError("legacy BF16 aggregate has conflicting explicit beta")
    if "output_dtype" in op_attributes and op_attributes["output_dtype"] not in (None, "bf16"):
        raise ValueError("legacy BF16 aggregate has conflicting explicit output_dtype")
    from src.kernel.launch_config.nvidia_h200.gemm_launch import select_legacy_bf16_external_aggregate_launch
    reducer = name == _LEGACY_H200_BF16_REDUCER_SYMBOL
    launch = select_legacy_bf16_external_aggregate_launch(shape, "bf16", "gemm", {
        "has_bias": False, "launch_policy": "injected_launch", "a_layout": "row_major",
        "b_layout": "row_major", "c_layout": "row_major", "trans_a": False,
        "trans_b": True, "epilogue": "none", "beta": 0, "output_dtype": None}, hardware, reducer=reducer)
    actual_grid = tuple(metrics["grid_" + axis] for axis in ("x", "y", "z"))
    actual_block = tuple(metrics["block_" + axis] for axis in ("x", "y", "z"))
    if launch.grid != actual_grid or launch.block != actual_block:
        raise ValueError("legacy BF16 aggregate observation disagrees with its published exact geometry")
    return launch


def observed_nvjet_gemm_split_k(
        op_attributes: Mapping, metrics: Mapping) -> Optional[int]:
    """Return an observed K-split count, or None for six explicit aggregate rows.

    Hopper ``nvjet`` kernels keep ``grid.z == 1`` and launch the output tile
    grid in thread-block clusters; the symbol states the tile, the cluster
    shape and whether the program is split-K (``_splitK``).  A split launch
    enumerates every cluster-padded output tile once per K slice, so the
    slice count is the launched CTA count over the padded tile count.  A
    non-split launch is either one CTA per padded tile or a persistent grid
    with fewer CTAs than tiles.  Only launch geometry and the symbol enter
    this derivation; no duration or counter is read.
    """

    if legacy_h200_bf16_aggregate_projection(op_attributes, metrics) is not None:
        return None  # Effective K partitions are not observed for this model.
    symbol = parse_nvjet_gemm_symbol(metrics["kernel_name"])
    if symbol is None:
        raise ValueError("nvjet split-K derivation requires an nvjet symbol")
    if not isinstance(op_attributes, Mapping):
        raise ValueError(
            "nvjet split-K derivation requires the GEMM Attr shape")
    rows = (_positive_int(op_attributes, "B", "GEMM Attr")
            * _positive_int(op_attributes, "M", "GEMM Attr"))
    columns = _positive_int(op_attributes, "N", "GEMM Attr")
    n_tiles = math.ceil(columns / symbol.tile_n)
    m_tiles = math.ceil(rows / symbol.tile_m)
    padded_tiles = (
        math.ceil(n_tiles / symbol.cluster_n_tiles) * symbol.cluster_n_tiles
        * math.ceil(m_tiles / symbol.cluster_m_tiles)
        * symbol.cluster_m_tiles)
    if metrics["grid_z"] != 1:
        raise ValueError("nvjet GEMM launch must keep grid.z == 1")
    grid_size = metrics["grid_x"] * metrics["grid_y"]
    if symbol.split_k:
        splits, remainder = divmod(grid_size, padded_tiles)
        if remainder or splits < 2:
            raise NotImplementedError(
                "nvjet split-K launch grid is not an integer multiple (>= 2) "
                "of its cluster-padded output tiles: "
                f"grid={grid_size}, padded_tiles={padded_tiles}")
        return splits
    if grid_size > padded_tiles:
        raise ValueError(
            "non-split nvjet launch has more CTAs than padded output tiles: "
            f"grid={grid_size}, padded_tiles={padded_tiles}")
    return 1


def _gemm_tensor_split(metrics: Mapping, op_attributes) -> Optional[int]:
    if parse_nvjet_gemm_symbol(metrics["kernel_name"]) is not None:
        return observed_nvjet_gemm_split_k(op_attributes, metrics)
    return metrics["grid_z"]


def observed_gemm_split_k(kernel_records, op_attributes=None) -> Optional[int]:
    """Return observed split count; exact six aggregate cases keep it unknown.

    A split-K reducer's own grid does not encode how many partial matrices it
    consumes.  The immediately preceding tensor launch does, so evaluation
    normalizes that launch fact into the same production ``split_k`` contract
    used by Serving.  No duration or counter is consumed.  Hopper ``nvjet``
    launches state the split through their symbol and cluster-padded grid
    (``observed_nvjet_gemm_split_k``) and require ``op_attributes``.
    """

    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "GEMM split-K derivation requires a non-empty kernel sequence")
    exact = [legacy_h200_bf16_aggregate_projection(op_attributes, extract_ncu_metrics(record))
             for record in kernel_records]
    if any(launch is not None for launch in exact):
        if (len(exact) != 2 or any(launch is None for launch in exact)
                or exact[0].execution_variant != "gemm" or exact[1].execution_variant != "gemm_reduce"):
            raise ValueError("legacy BF16 aggregate requires its exact main then reducer pair")
        return None  # Not modeled_split_k: this API reports observed partitions.
    tensor_splits = []
    for ordinal, record in enumerate(kernel_records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"GEMM kernel record {ordinal} must be a mapping")
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        non_tensor = is_non_tensor(
            "GEMM", metrics["kernel_name"],
            ascend_op_type=metrics["op_type"])
        # MetaX GEMV-family kernels are implemented on ordinary vector lanes,
        # so the observation classifier correctly reports them as non-tensor.
        # They are nevertheless the case's primary GEMM launch (and are
        # normalized to KernelKind.GEMM below), not an auxiliary reducer.
        if not non_tensor or "gemv" in lowered:
            tensor_splits.append(_gemm_tensor_split(metrics, op_attributes))
    if not tensor_splits:
        raise ValueError("GEMM profiler case has no tensor launch")
    unique = set(tensor_splits)
    if len(unique) != 1:
        raise ValueError(
            "GEMM tensor launches disagree on split_k: "
            f"{sorted(unique)}")
    return tensor_splits[0]


def observed_gemm_split_k_partial_bytes(
        kernel_records, op_attributes=None) -> int:
    """Return a source-visible split-K partial width, or zero if unknown.

    ``grid_z`` proves that the tensor launch is split, but not whether the
    implementation uses a materialized partial matrix.  A directly following
    cuBLAS ``splitKreduce`` exposes its partial type as the fourth runtime
    template argument; mcBLAS ``DevDstElementwiseAdd`` exposes an FP32 partial
    pointer.  No timing or profiler counter participates in this derivation.
    """

    split_k = observed_gemm_split_k(kernel_records, op_attributes)
    if split_k is None:
        return 4  # Exact pair above exposes FP32 partial type; extent remains modeled.
    if split_k == 1:
        return 0
    tensor_ordinals = []
    reducer_ordinals = []
    reducer_names = []
    for ordinal, record in enumerate(kernel_records):
        metrics = extract_ncu_metrics(record)
        name = metrics["kernel_name"]
        lowered = name.lower()
        non_tensor = is_non_tensor(
            "GEMM", name, ascend_op_type=metrics["op_type"])
        if ((not non_tensor or "gemv" in lowered)
                and _gemm_tensor_split(metrics, op_attributes) > 1):
            tensor_ordinals.append(ordinal)
        if "splitkreduce" in lowered or "elementwiseadd" in lowered:
            reducer_ordinals.append(ordinal)
            reducer_names.append(name)
    if not reducer_ordinals:
        return 0
    if len(tensor_ordinals) != 1 or len(reducer_ordinals) != 1:
        raise ValueError(
            "GEMM split-K partial ABI requires one tensor main followed by "
            "one reducer")
    if reducer_ordinals[0] != tensor_ordinals[0] + 1:
        raise ValueError(
            "GEMM split-K reducer must immediately follow its tensor main")
    reducer_name = reducer_names[0]
    lowered_reducer = reducer_name.lower()
    if "splitkreduce" in lowered_reducer:
        marker = "splitKreduce_kernel<"
        if marker not in reducer_name or ">(" not in reducer_name:
            raise ValueError(
                "cuBLAS split-K reducer name does not expose its template ABI")
        template_text = reducer_name.split(marker, 1)[1].split(">(", 1)[0]
        template_arguments = tuple(
            value.strip() for value in template_text.split(","))
        if len(template_arguments) < 4:
            raise ValueError(
                "cuBLAS split-K reducer template ABI is incomplete")
        partial_type = template_arguments[3]
        if partial_type == "__half":
            return 2
        elif partial_type == "float":
            return 4
        else:
            raise ValueError(
                "cuBLAS split-K reducer has an unsupported partial type: "
                f"{partial_type!r}")
    elif "elementwiseadd" in lowered_reducer:
        if "float const*" not in reducer_name:
            raise ValueError(
                "mcBLAS split-K reducer ABI does not expose an FP32 partial")
        return 4
    else:
        raise AssertionError("unhandled recognized split-K reducer")


def observed_gemm_split_k_partial_dram_bytes(
        kernel_records, op_attributes=None) -> int:
    """Return the frozen complete-partial DRAM width for a split launch.

    The retained mcBLAS ``DevDstElementwiseAdd`` family has 70/70 exact
    main/reducer observations whose non-duration physical global write/read
    counters cover a complete FP32 partial buffer.  The retained cuBLAS
    ``splitKreduce`` family does not: all 26 main launches write less than the
    complete partial buffer to DRAM.  Map those versioned launch families to
    the source-visible width and 0 bytes/element respectively; individual
    counter magnitudes do not enter Forward.
    """

    partial_bytes = observed_gemm_split_k_partial_bytes(
        kernel_records, op_attributes)
    if partial_bytes == 0:
        return 0
    reducer_names = []
    for record in kernel_records:
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        if "splitkreduce" in lowered or "elementwiseadd" in lowered:
            reducer_names.append(lowered)
    if len(reducer_names) != 1:
        raise ValueError(
            "GEMM split-K DRAM partial contract requires one reducer")
    reducer_name = reducer_names[0]
    if "elementwiseadd" in reducer_name:
        return partial_bytes
    if "splitkreduce" in reducer_name:
        return 0
    raise AssertionError("unhandled recognized split-K reducer")
