"""Adapt the external PipeWeave baseline to structured :class:`KernelSpec` inputs.

PipeWeave is an operator-level comparison method.  This adapter does not accept
profiler dictionaries.  GEMM/MoE launch geometry normally comes from the
``KernelSpec``.  A matrix-vector implementation is instead priced as GEMM
using PipeWeave's own configuration lookup for the same matrix shape.
"""

from __future__ import annotations

import copy
import functools
import math
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from src.kernel.kernel_spec import KernelSpec
from src.kernel.predictors.common import prediction_from_ns


_PIPEWEAVE_ROOT = (
    Path(__file__).resolve().parents[3] / "pipeweave"
).resolve()
_PIPEWEAVE_CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "configs/baselines/pipeweave"
).resolve()
_PIPEWEAVE_HARDWARE = {
    "nvidia_a800": "nvidia_a800",
    "metax_c500": "metax_c500",
    "ascend_910b": "ascend_910b",
    "samsung_hbm_pim": "samsung_hbm_pim",
    "ucb_gemmini": "ucb_gemmini",
    "groq_lpu": "groq_lpu",
}
_MODEL_DIRECTORY = {
    "nvidia_a800": "mlp_models",
    "metax_c500": "mlp_models_c500",
    "ascend_910b": "mlp_models_910b",
}
_MODEL_SOURCE_ALIASES = {
    "A100": "nvidia_a800",
    "C500": "metax_c500",
    "910B": "ascend_910b",
}


def _model_source(hardware_name: str, model_hardware: Optional[str]) -> str:
    if model_hardware is None:
        source = hardware_name
    elif model_hardware in _MODEL_SOURCE_ALIASES:
        source = _MODEL_SOURCE_ALIASES[model_hardware]
    elif model_hardware in _MODEL_DIRECTORY:
        source = model_hardware
    else:
        raise ValueError(f"unknown PipeWeave model hardware: {model_hardware!r}")
    if source not in _MODEL_DIRECTORY:
        raise NotImplementedError(
            f"PipeWeave has no trained model directory for {source!r}; "
            "pass an explicit --pipeweave-model-hw")
    return source


def _install_external_import_paths() -> None:
    for path in (_PIPEWEAVE_ROOT, _PIPEWEAVE_ROOT / "analytical_model"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


_PREDICTOR_LOAD_LOCK = threading.Lock()
_PREDICTOR_INFERENCE_LOCK = threading.Lock()


@functools.lru_cache(maxsize=8)
def _get_predictor_cached(
        hardware_name: str, model_hardware: Optional[str] = None):
    """Load the external model; launch selection remains outside this adapter."""

    _install_external_import_paths()
    model_source = _model_source(hardware_name, model_hardware)
    model_dir = _PIPEWEAVE_ROOT / _MODEL_DIRECTORY[model_source]
    if not model_dir.exists():
        raise FileNotFoundError(
            f"PipeWeave model directory does not exist: {model_dir}")
    from aggregator import DurationPredictor

    return DurationPredictor(
        hardware_name=hardware_name,
        model_dir=str(model_dir),
        dataset_dir=str(_PIPEWEAVE_ROOT / "dataset"),
        hardware_dir=str(_PIPEWEAVE_CONFIG_DIR),
        device="cpu",
    )


def _get_predictor(
        hardware_name: str, model_hardware: Optional[str] = None):
    """Load each external model set once even under threaded evaluation.

    ``functools.lru_cache`` protects its internal mapping but deliberately may
    evaluate the wrapped function more than once when concurrent callers miss
    the same key.  Loading the external models and 96k-row lookup dataset is
    expensive, so serialize only that cache-miss boundary.
    """

    with _PREDICTOR_LOAD_LOCK:
        return _get_predictor_cached(hardware_name, model_hardware)


def _get_vector_predictor(
        hardware_name: str, model_hardware: Optional[str] = None):
    """Give pure Vector operators their own physical resource domain.

    Ascend910B2.ini declares 24 Cube cores, 48 Vector cores, and 192 KiB
    per-Vector-core UB. The cached base profile remains the Cube view.
    Copy only the profile containers, retaining the same pretrained models.
    Chip-wide memory bandwidths must not grow with the Vector core count.

    The JSON supplies theoretical Vector arithmetic capacities. Upstream's
    aggregate instruction counts and trained networks remain unchanged. This
    resource port does not reconstruct CANN's scalar handoffs or request
    latency, including the scalar normalization path of RMSNorm key 0.
    """

    predictor = _get_predictor(hardware_name, model_hardware)
    if hardware_name != "ascend_910b":
        return predictor
    vector = copy.copy(predictor)
    vector.hardware_spec = predictor.hardware_spec.copy()
    vector.hardware_spec_obj = copy.copy(predictor.hardware_spec_obj)
    vector.hardware_spec["num_sms"] = 48
    vector.hardware_spec["shared_memory_size"] = 192
    vector.hardware_spec_obj.num_sms = 48
    vector.hardware_spec_obj.shared_memory_size = 192
    return vector


def _hardware_name(kernel: KernelSpec) -> str:
    if kernel.hardware_id not in _PIPEWEAVE_HARDWARE:
        raise NotImplementedError(
            f"PipeWeave has no hardware adapter for {kernel.hardware_id}")
    return _PIPEWEAVE_HARDWARE[kernel.hardware_id]


def _predict_features(predictor, operator: str, features: np.ndarray) -> float:
    if operator in ("rmsnorm", "siluandmul"):
        cycle_sum = float(features[1] + features[5])
    elif operator in ("gemm", "attn"):
        cycle_sum = float(features[1])
    else:
        raise ValueError(f"unknown PipeWeave operator: {operator!r}")
    try:
        # The external baseline uses one PyTorch model set whose CPU kernels
        # already parallelize internally.  Concurrent calls from the kernel
        # evaluator otherwise oversubscribe those pools and are not documented
        # as thread-safe by upstream.
        with _PREDICTOR_INFERENCE_LOCK:
            milliseconds = predictor.predict_operator_duration(
                operator, features, cycle_sum)
    except ZeroDivisionError as exc:
        raise NotImplementedError(
            f"PipeWeave has no positive {operator} estimate for this hardware") from exc
    latency = float(milliseconds) * 1e6
    if not np.isfinite(latency) or latency <= 0:
        raise NotImplementedError(
            f"PipeWeave has no positive {operator} estimate for this hardware")
    return latency


def _gemm_features(kernel: KernelSpec, predictor) -> np.ndarray:
    b, m, n, k = (int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if kernel.variant.lower() == "gemv":
        # GEMV has no tensor-core GEMM tile to pass through.  Use the same
        # shape/configuration interface as PipeWeave's predict_workload.
        # Its lookup supplies only discrete launch parameters, not latency.
        config = predictor.find_nearest_gemm_config(b * m, n, k)
        if config is None:
            raise NotImplementedError(
                "PipeWeave has no GEMM configuration for this matrix shape")
        return predictor.generate_gemm_features(
            b * m, n, k,
            config["tile_M"], config["tile_N"], config["tile_K"],
            config["cta_count"], config["is_split_k"],
        )
    tile = kernel.launch.tiling
    if tile is None or not {"m", "n", "k"}.issubset(tile.tile_shape):
        raise ValueError("PipeWeave GEMM requires m/n/k tiling in LaunchConfig")
    return predictor.generate_gemm_features(
        b * m,
        n,
        k,
        int(tile.tile_shape["m"]),
        int(tile.tile_shape["n"]),
        int(tile.tile_shape["k"]),
        kernel.launch.grid_size,
        bool(tile.split_k and tile.split_k > 1),
    )


def _attention_features(kernel: KernelSpec, predictor) -> np.ndarray:
    shape = kernel.shape
    batch = int(shape["batch"])
    layout_value = kernel.attributes["kv_layout"]
    if layout_value is None:
        raise ValueError("PipeWeave FLASH_ATTN requires kv_layout")
    layout = str(layout_value).lower()
    if layout.startswith("paged"):
        layout = "paged"
    elif layout in ("ragged", "contiguous", "chunked_prefill", "mla"):
        layout = "ragged"
    else:
        raise ValueError(f"unknown PipeWeave FLASH_ATTN layout: {layout!r}")
    # MLA uses the same FA2 shape approximation on every platform.  There is
    # no dedicated MLA model in PipeWeave, and an implementation's name must
    # not determine whether an otherwise identical MLA problem is supported.
    # Other attention implementations retain the existing FA2/FA3 dispatch.
    algorithm = kernel.launch.algorithm.lower()
    if str(layout_value).lower() == "mla":
        family = "fa2"
    elif "fa3" in algorithm:
        family = "fa3"
    elif any(name in algorithm for name in (
            "flash_attn", "flash_attention", "observed_flash")):
        family = "fa2"
    else:
        raise NotImplementedError(
            f"unknown PipeWeave attention launch family: {algorithm!r}")
    return predictor.generate_attn_features(
        bs=batch,
        q_lengths=[int(value) for value in kernel.attributes["q_lens"]],
        kv_lengths=[int(value) for value in kernel.attributes["kv_lens"]],
        nh=int(shape["num_query_heads"]),
        nkv=int(shape["num_kv_heads"]),
        hd=int(shape["head_dim"]),
        attention_type=f"{family}_{layout}",
    )


def _attention_resource_features(
        hardware_name: str, features: np.ndarray) -> np.ndarray:
    """Map each baseline Cube group's padded XU work onto its two AIVs.

    Keep the original 24-group tile selection, total instruction counts,
    matrix work, and memory features. Its padded attention tiles admit two
    equal row partitions, matching the 1:2 Cube/Vector resource grouping.
    Only XU per-worker work and its cycles change. This is a resource-level
    mapping of the upstream schedule, not a reconstruction of CANN launches.
    """

    if hardware_name != "ascend_910b":
        return features
    if features.shape != (15,):
        raise ValueError("PipeWeave attention requires exactly 15 features")
    mapped = features.copy()
    # [4] remains total XU work. [5] is total work / (48 * XU capacity).
    # [6:8] describe one AIV rather than the enclosing Cube/Vector group.
    mapped[5:8] /= 2.0
    return mapped


def _moe_launch_geometry(kernel: KernelSpec):
    """Return one projection's geometry and the grid PipeWeave prices it on.

    PipeWeave is a GPU model, and the cross-architecture setting asks what it
    predicts for a problem it would run as a Triton grouped GEMM.  On a GPU the
    caller's launch already is that, so its own padded rows and grid are used
    unchanged.  On an NPU the same problem runs as a persistent CANN grouped
    matmul whose grid is the core count and which pads no rows, so there is no
    Triton grid to read; the baseline is instead asked what it would do with
    this problem at this tile, and both quantities come from PipeWeave's own
    round-robin routing model rather than from anything invented here.
    """

    tile = kernel.launch.tiling
    if tile is None or not {"m", "n", "k"}.issubset(tile.tile_shape):
        raise ValueError("PipeWeave FUSED_MOE requires m/n/k tiling in LaunchConfig")
    projection = kernel.variant.lower()
    if projection not in ("w1", "w2"):
        raise NotImplementedError(
            "PipeWeave FUSED_MOE supports separate w1/w2 projections only")

    shape = kernel.shape
    block_m = int(tile.tile_shape["m"])
    block_n = int(tile.tile_shape["n"])
    if projection == "w1":
        output_columns = 2 * int(shape["intermediate"])
        reduction_columns = int(shape["hidden"])
        pass_type = "w13"
    elif projection == "w2":
        output_columns = int(shape["hidden"])
        reduction_columns = int(shape["intermediate"])
        pass_type = "w2"
    else:
        raise AssertionError(f"unhandled PipeWeave MoE projection: {projection!r}")

    output_tiles = math.ceil(output_columns / block_n)
    if output_tiles <= 0:
        raise ValueError(
            "PipeWeave MoE has no structured N tiles: "
            f"n_tiles={output_tiles}")
    if "triton" in kernel.launch.algorithm.lower():
        if kernel.launch.grid_size % output_tiles:
            raise ValueError(
                "PipeWeave MoE grid is not divisible by its structured N tiles: "
                f"grid={kernel.launch.grid_size}, n_tiles={output_tiles}")
        padded_rows = int(kernel.launch.parameters["padded_rows"])
        grid_size = kernel.launch.grid_size
        expected_grid = math.ceil(padded_rows / block_m) * output_tiles
        if expected_grid != grid_size:
            raise ValueError(
                "PipeWeave MoE launch is inconsistent with its structured padded rows: "
                f"grid={grid_size}, calculated={expected_grid}")
        return padded_rows, output_columns, reduction_columns, pass_type, grid_size
    _install_external_import_paths()
    from triton_moe_calculator import calculate_token_distribution_per_expert_rr

    shape = kernel.shape
    _tokens_per_expert, _padded_per_expert, padded_rows = (
        calculate_token_distribution_per_expert_rr(
            m=int(shape["tokens"]),
            e=int(shape["experts"]),
            top_k=int(shape["topk"]),
            block_size_m=block_m,
        ))
    if padded_rows <= 0:
        raise ValueError(
            "PipeWeave MoE routing model returned no padded rows for "
            f"{kernel.variant!r}")
    grid_size = math.ceil(padded_rows / block_m) * output_tiles
    return padded_rows, output_columns, reduction_columns, pass_type, grid_size


def _moe_features(kernel: KernelSpec, predictor) -> np.ndarray:
    """Generate one projection's features using the supplied launch tile."""

    padded_rows, output_columns, reduction_columns, pass_type, grid_size = (
        _moe_launch_geometry(kernel))
    tile = kernel.launch.tiling
    block_m = int(tile.tile_shape["m"])
    block_n = int(tile.tile_shape["n"])

    _install_external_import_paths()
    from pipes import MoeProblemConfig, TensorPipe
    from triton_moe_calculator import (
        calculate_memory_pipe,
        schedule_uniform_ctas,
    )

    shape = kernel.shape
    dtype = kernel.dtype.lower()
    if dtype in ("fp8", "int8", "uint8"):
        data_size_bytes = 1
    elif dtype in ("fp16", "bf16", "float16"):
        data_size_bytes = 2
    else:
        raise NotImplementedError(
            f"PipeWeave MoE does not support dtype {kernel.dtype!r}")
    problem = MoeProblemConfig(
        m=int(shape["tokens"]),
        e=int(shape["experts"]),
        top_k=int(shape["topk"]),
        h=int(shape["hidden"]),
        n=int(shape["intermediate"]),
        data_size_bytes=data_size_bytes,
    )
    total_flops = (
        2.0 * padded_rows * output_columns * reduction_columns)

    per_sm = schedule_uniform_ctas(
        grid_size, predictor.hardware_spec_obj.num_sms)
    if not per_sm:
        raise ValueError("PipeWeave returned an empty CTA schedule")
    maximum_ctas = max(per_sm)
    if maximum_ctas <= 0:
        raise ValueError("PipeWeave returned a non-positive CTA schedule")
    per_cta_flops = total_flops / grid_size
    tensor = TensorPipe(
        all_ops=total_flops,
        all_cycle=total_flops / predictor.hardware_spec_obj.tc_bf16
        / predictor.hardware_spec_obj.num_sms,
        sm_max_ops=per_cta_flops * maximum_ctas,
        sm_max_cycle=per_cta_flops * maximum_ctas / predictor.hardware_spec_obj.tc_bf16,
    )
    memory = calculate_memory_pipe(
        problem, block_m, block_n, grid_size, per_sm,
        predictor.hardware_spec_obj, pass_type)
    return np.array([
        tensor.all_ops,
        tensor.all_cycle,
        tensor.sm_max_ops,
        tensor.sm_max_cycle,
        memory.global_in_flight,
        memory.global_cycle,
        memory.local_cycle,
        memory.sm_max_in_flight,
        memory.sm_max_global_cycle,
        memory.sm_max_shared_cycle,
        memory.sm_max_local_cycle,
    ], dtype=np.float32)


def _prediction(predictor, operator: str, features: np.ndarray):
    latency = _predict_features(predictor, operator, features)
    return prediction_from_ns(latency, "pipeweave")


def predict_gemm(kernel: KernelSpec, hardware, *, model_hardware=None):
    del hardware  # KernelPredictor already validated the hardware identity.
    if kernel.variant.lower() not in ("gemm", "gemv"):
        raise NotImplementedError(
            f"PipeWeave GEMM does not support variant {kernel.variant!r}")
    predictor = _get_predictor(_hardware_name(kernel), model_hardware)
    return _prediction(predictor, "gemm", _gemm_features(kernel, predictor))


def predict_flash_attn(kernel: KernelSpec, hardware, *, model_hardware=None):
    del hardware
    hardware_name = _hardware_name(kernel)
    predictor = _get_predictor(hardware_name, model_hardware)
    features = _attention_resource_features(
        hardware_name, _attention_features(kernel, predictor))
    return _prediction(predictor, "attn", features)


def predict_fused_moe(kernel: KernelSpec, hardware, *, model_hardware=None):
    del hardware
    predictor = _get_predictor(_hardware_name(kernel), model_hardware)
    return _prediction(predictor, "gemm", _moe_features(kernel, predictor))


def predict_norm(kernel: KernelSpec, hardware, *, model_hardware=None):
    del hardware
    predictor = _get_vector_predictor(_hardware_name(kernel), model_hardware)
    features = predictor.generate_rmsnorm_features(
        seq=int(kernel.shape["rows"]), dim=int(kernel.shape["hidden"]))
    return _prediction(predictor, "rmsnorm", features)


def predict_vector_activation(kernel: KernelSpec, hardware, *, model_hardware=None):
    """Price a SiLU-and-multiply launch through PipeWeave's siluandmul model.

    PipeWeave ships one trained model for this operator, so every launch that
    is a SiLU-and-multiply belongs in its domain whatever the caller names the
    variant.  The two callers differ only in how many rows the operator runs
    over: the dense launch carries that count directly in ``tok``, while the
    fused-MoE launch runs it once per routed row, which is the token count
    times the number of experts each token is routed to.
    """

    del hardware
    variant = kernel.variant.lower()
    shape = kernel.shape
    if variant in ("activation", "silu"):
        rows = int(shape["tok"])
    elif variant in ("moe_act_and_mul", "moe_swiglu"):
        rows = int(shape["tokens"]) * int(shape["topk"])
    else:
        raise NotImplementedError(f"PipeWeave does not support vector/{kernel.variant}")
    predictor = _get_vector_predictor(_hardware_name(kernel), model_hardware)
    features = predictor.generate_siluandmul_features(
        seq=rows,
        dim=int(shape["intermediate"]),
    )
    return _prediction(predictor, "siluandmul", features)
