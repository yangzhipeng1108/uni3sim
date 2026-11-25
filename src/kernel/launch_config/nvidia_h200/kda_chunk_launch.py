"""Prepared vLLM0.30 Triton KDA chunk operator, independently bound to SM90.

The exact K3 third_party/kda/chunk.py call includes Q/K normalization,
fused gate/beta cumsum, Akk zeroing, intra/solve, WY recompute, chunk state
and output. Autotuned configurations are mandatory caller compiler facts;
source candidate sets validate them but never select a winner by shape or
duration. Triton v3.7.1 CUDAOptions/Config independently bind the default
four warps and three stages when the wrapper leaves an option unspecified.

Indices/offsets must already be cached for the actual cu_seqlens tensor
identity. Cold metadata construction includes CPU synchronization and H2D
and is outside this prepared operator. Convolutions, state gather/scatter,
output-copy and output norm are separate operations, not omitted stages of
this body. Registers and shared memory of Triton specializations are unbound.
"""

import math
from dataclasses import replace

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.nvidia_h200.torch_vector_launch import select_torch213_fill_launch
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


def _selected_config(config, fields):
    """Admit an actual selected source candidate, without choosing one."""
    if not isinstance(config, dict) or set(config) != set(fields):
        raise ValueError("KDA requires the complete selected compiler config fields")
    for key, allowed in fields.items():
        if type(config[key]) is not int or config[key] not in allowed:
            raise ValueError("KDA selected compiler config is not a source candidate: " + key)
    return config


def _launch(stage, grid, config, **tiles):
    if grid[0] > 2**31 - 1 or any(axis > 65535 for axis in grid[1:]):
        raise NotImplementedError("KDA chunk source grid exceeds CUDA axis capacity")
    return LaunchConfig("vllm_0_30_sm90_kda_chunk_" + stage, grid,
        (config["num_warps"] * 32, 1, 1), parameters={
            "source_num_stages": float(config["num_stages"]),
            **{"source_" + key: float(value) for key, value in tiles.items()}})


def select_kda_chunk_recipe(shape, dtype, attributes, hardware):
    """Return every submission of the prepared packed-varlen chunk call.

    source_selected_configs comes from the exact compiled/autotune cache,
    not an evaluator-chosen configuration. Query segments, chunk metadata,
    normalized contiguous inputs and live cached tensor identity are caller
    facts. One packed tensor batch may contain multiple unequal requests;
    state grid uses request count while other grids use total chunk count.
    """
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("KDA chunk source requires H200 HWSpec")
    require_dtype(dtype, ("bf16",))
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_component"] != "vllm_0_30_kimi_k3_triton_chunk"
            or attributes["source_compiler"] != "triton_3_7_1_cuda_sm90"
            or attributes["source_input_layout"] != "contiguous_qkv_gate_initial"
            or attributes["source_norm_backend"] != "l2norm_fwd_kernel2"
            or attributes["source_metadata_state"] != "cached_actual_cu_seqlens_identity"
            or attributes["source_initial_state_dtype"] != "fp32"
            or attributes["source_output_final_state"] is not True
            or attributes["tensor_iterator_source"] != "pytorch_2_13_sm90"):
        raise NotImplementedError("KDA chunk requires its exact prepared0.30 Triton3.7.1 source")
    rows, heads, dim = require_shape(shape, ("rows", "heads", "head_dim"))
    if heads not in (6, 12, 24, 32, 48, 96) or dim != 128:
        raise NotImplementedError("KDA chunk binds128-wide K3 TP1/2/4/8/16 or KimiLinear32-head source")
    starts = attributes["query_start_loc"]
    if (not isinstance(starts, (tuple, list)) or len(starts) < 2
            or any(type(value) is not int for value in starts)
            or starts[0] != 0 or starts[-1] != rows
            or any(a >= b for a, b in zip(starts, starts[1:]))):
        raise ValueError("KDA chunk requires complete strictly increasing packed query segments")
    sequences = len(starts) - 1
    # Both metadata and kernel token indices are converted to signed int32.
    if rows >= 2**31 or sequences > 2**20:
        raise NotImplementedError("KDA chunk source token/binary-search domain exceeded")
    fill = select_torch213_fill_launch({"element_count": rows * heads * 64}, dtype,
        {"tensor_iterator_source": attributes["tensor_iterator_source"],
         "source_input_layout": "contiguous_aligned_16"}, hardware)
    chunks_per_sequence = tuple(ceil_div(b - a, 64) for a, b in zip(starts, starts[1:]))
    if max(sum(chunks_per_sequence), sequences) * heads * dim * dim > 2**31:
        raise NotImplementedError("KDA chunk source state offsets require bounded signed int32 intermediates")
    expected_indices = tuple((seq, chunk) for seq, count in enumerate(chunks_per_sequence)
        for chunk in range(count))
    indices = attributes["source_chunk_indices"]
    offsets = attributes["source_chunk_offsets"]
    expected_offsets = [0]
    for count in chunks_per_sequence:
        expected_offsets.append(expected_offsets[-1] + count)
    if (not isinstance(indices, (tuple, list))
            or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(value) is not int for value in pair) for pair in indices)
            or tuple(tuple(pair) for pair in indices) != expected_indices
            or not isinstance(offsets, (tuple, list))
            or any(type(value) is not int for value in offsets)
            or tuple(offsets) != tuple(expected_offsets)):
        raise ValueError("KDA chunk cached indices/offsets disagree with actual query segments")
    chunks = expected_offsets[-1]
    lower_bound = attributes["gate_lower_bound"]
    if lower_bound is not None and (type(lower_bound) not in (int, float)
            or not math.isfinite(lower_bound) or not -5 <= lower_bound < 0):
        raise ValueError("K3 chunk gate lower bound must be absent or in[-5,0)")
    safe_gate = lower_bound is not None
    selected = attributes["source_selected_configs"]
    if not isinstance(selected, dict) or set(selected) != {
            "gate", "intra", "inter_solve", "recompute", "state", "output"}:
        raise ValueError("KDA chunk requires all actual selected source configurations")
    gate = _selected_config(selected["gate"],
        {"BS": (32, 64), "num_warps": (2, 4, 8), "num_stages": (3,)})
    if safe_gate:
        intra = _selected_config(selected["intra"],
            {"num_warps": (1, 2, 4, 8), "num_stages": (2, 3, 4)})
    else:
        intra = _selected_config(selected["intra"],
            {"BH": (1, 2, 4, 8), "num_warps": (1, 2, 4, 8), "num_stages": (3,)})
    inter = _selected_config(selected["inter_solve"],
        {"BK": (32, 64), "num_warps": (1, 2, 4), "num_stages": (3,)})
    recompute = _selected_config(selected["recompute"],
        {"num_warps": (2, 4, 8), "num_stages": (2, 3, 4)})
    state = _selected_config(selected["state"],
        {"BV": (32, 64), "num_warps": (2, 4), "num_stages": (2, 3, 4)})
    output = _selected_config(selected["output"],
        {"BK": (32, 64), "BV": (64, 128), "num_warps": (2, 4, 8), "num_stages": (2, 3, 4)})
    norm = _launch("l2norm", (ceil_div(rows * heads, 32), 1, 1),
        {"num_warps": 4, "num_stages": 3}, MBLOCK=32, BD=128)
    intra_stage = "intra_sub_chunk" if safe_gate else "intra_token_parallel"
    intra_grid = (chunks, 4, heads) if safe_gate else (rows, ceil_div(heads, intra["BH"]), 1)
    intra_tiles = {"BT": 64, "BC": 16, "BK": 128} if safe_gate else {
        "BT": 64, "BC": 16, "BH": intra["BH"]}
    return (
        ("q_l2norm", norm), ("k_l2norm", norm),
        ("gate_cumsum_beta", _launch("gate_cumsum_beta",
            (ceil_div(dim, gate["BS"]) + 1, chunks, heads), gate,
            BT=64, BS=gate["BS"], USE_LOWER_BOUND=int(safe_gate))),
        ("zero_Akk", fill),
        (intra_stage, _launch(intra_stage, intra_grid, intra, **intra_tiles)),
        ("inter_solve", _launch("inter_solve", (chunks, heads, 1), inter,
            BT=64, BC=16, BK=inter["BK"], USE_SAFE_GATE=int(safe_gate))),
        ("recompute_w_u", _launch("recompute_w_u", (chunks, heads, 1), recompute, BT=64, BK=64, BV=64)),
        ("state", _launch("state", (ceil_div(dim, state["BV"]), sequences * heads, 1),
            state, BT=64, BV=state["BV"])),
        ("output", _launch("output", (ceil_div(dim, output["BV"]), chunks, heads),
            output, BT=64, BK=output["BK"], BV=output["BV"])),
    )


def select_kda_chunk_stage_launch(shape, dtype, variant, attributes, hardware):
    """Bind a prepared source child explicitly; no FP8 None exception reuse."""
    if variant not in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill"):
        raise ValueError("KDA chunk stage requires its exact parent execution variant")
    recipe = select_kda_chunk_recipe(shape, dtype, attributes, hardware)
    for stage, launch in recipe:
        if attributes["source_stage"] == stage:
            return replace(launch, execution_variant=variant)
    raise ValueError("KDA chunk requires one of its exact nine source stages")
