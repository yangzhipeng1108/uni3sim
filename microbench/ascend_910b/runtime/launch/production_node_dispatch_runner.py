#!/usr/bin/env python3
"""Price each charged node class through the operator production dispatches.

``configs/host_timing/host_runtime_prepare_profiles/ascend_910b_eager.json``
charges every eager-step node one ``compiled_fx_*_node_dispatch`` service.
Those prices come from ``aclnn_eager_dispatch_20260926_r2``, which submits a
*bare* operator: ``torch.nn.functional.linear``, ``torch_npu.npu_rms_norm``,
``torch_npu.npu_scatter_nd_update_`` and so on.

An operator census of a real above-capture-limit step shows that five of the
six bare calls are not the operator the engine dispatches:

    class          bare call (priced today)     dispatched by production
    linear         F.linear                     vllm::unquantized_gemm
    norm           torch_npu.npu_rms_norm       AscendRMSNorm.forward_oot,
                                                which issues
                                                vllm::maybe_chunk_residual then
                                                _C_ascend::npu_add_rms_norm_bias
    fia            torch_npu.npu_fused_...      npu::npu_fused_... packet
    reshape_cache  torch_npu.npu_scatter_nd_..  atb::_npu_reshape_and_cache
    rope           torch_npu.npu_rotary_mul     vllm::npu_rotary_embedding
    copy           Tensor.copy_                 aten::copy_          (identical)

The production forms are registered custom ops reached through
``torch.ops.<ns>.<op>`` and, for the norm, through the installed ``CustomOp``
module; each carries Python wrapper frames the bare call never enters.  Which
operator is dispatched is a launch fact and carries no duration.

``copy`` is the control: its two arms are the same call, so its two arms must
agree, and any ratio it shows is protocol noise rather than wrapper cost.

The node table charges its single norm price to three different kernels --
RmsNorm, AddRmsNormBias and SwiGlu -- which production reaches through three
different branches.  All three are measured so that whether one price can
stand for all three is a measured question rather than an assumption.

``vllm::qkv_rmsnorm_rope`` is measured as its own class, but it is *not* part
of the charged sequence.  The container build carries a QKNormRope fusion pass
that the canonical captures do not exercise: the canonical launch observation
under ``serving_dataset/sharegpt/ascend_910b/`` contains 73 RmsNorm, 72
AddRmsNormBias and 36 ``_triton_rope`` per step and no fused signature, which
is why ``lower_workload._ascend_qkv_rmsnorm_rope_fusions`` is deliberately
inert.  Its price is reported for the record; the interleaving census is the
canonical one so that every charged class is priced in the neighbourhood the
canonical build actually submits.

Two classes are reported ``partial``.  Production reaches the attention packet
through ``vllm::unified_attention_with_output``, which requires a live forward
context; constructing a fake one would price an invention rather than the
engine, so the packet itself is priced and the wrapper is recorded as
uncovered.  ``copy`` has no wrapper to cover by construction.

Protocol.  Classes are issued round-robin in the proportion the node table
charges them, which is how a decoder layer interleaves them, and every single
dispatch is timed with no synchronization inside the timed region, so each
sample is one Host submit with the caches in the state the preceding
*distinct* operator left them.  Per-class medians over many passes give the
price of one dispatch of that class under production-representative
conditions.  This is the protocol ``aclnn_eager_dispatch_runner``'s
``_per_class_in_sequence`` uses, so its numbers and these are comparable.

No Serving workload, model or production duration is read, and no value here
is selected against an error metric.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import torch
import torch_npu

SCHEMA_VERSION = 1
HARDWARE_ID = "ascend_910b"
ROUNDS = 60
SEED = 20260926
HIDDEN = 4096
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = KV_HEADS * HEAD_DIM
TOKENS = 544
BLOCK = 16
BLOCKS = 4096
EPS = 1e-6
SCALE = HEAD_DIM ** -0.5
FUSED = "compiled_fx_qkv_rmsnorm_rope_node_dispatch"
NORM_PLAIN = "compiled_fx_norm_no_residual_node_dispatch"
SWIGLU = "compiled_fx_swiglu_node_dispatch"
# The dtype the engine runs; the fused triton op requires operands to agree.
DTYPE = torch.bfloat16
# Kernels one canonical above-capture-limit step launches, per the canonical
# launch observation recorded at src/serving/kernel_lowering/lower_workload.py
# :1500-1508 and the node table at src/serving/runtime_prepare.py:788-838 for a
# 36-layer dense model: 145 aclnnMatmul, 73 RmsNorm, 72 AddRmsNormBias, 36
# SwiGlu, 217 copy-shape TensorMove/Slice, 36 _triton_rope, 36
# FusedInferAttentionScore, 36 ReshapeAndCache -- 651 charged nodes.  Used only
# to interleave the classes the way the engine issues them and to set the queue
# depth each submit sees.  It is a launch count and carries no duration.
#
# The fused class is issued once per sequence for the record only: it does not
# appear in any canonical capture and is not charged.
CENSUS = {
    "compiled_fx_aten_linear_node_dispatch": 145,
    NORM_PLAIN: 73,
    "compiled_fx_torch_npu_norm_node_dispatch": 72,
    "compiled_fx_aten_copy_node_dispatch": 217,
    SWIGLU: 36,
    "compiled_fx_rope_node_dispatch": 36,
    "compiled_fx_fia_node_dispatch": 36,
    "compiled_fx_reshape_cache_node_dispatch": 36,
    FUSED: 1,
}
CHARGED = tuple(name for name in CENSUS if name != FUSED)
COVERAGE = {
    "compiled_fx_aten_linear_node_dispatch": "covered",
    "compiled_fx_torch_npu_norm_node_dispatch": "covered",
    "compiled_fx_aten_copy_node_dispatch": "control_no_wrapper",
    "compiled_fx_rope_node_dispatch": "covered",
    "compiled_fx_fia_node_dispatch": "partial_unified_attention_wrapper_uncovered",
    "compiled_fx_reshape_cache_node_dispatch": "covered",
    FUSED: "container_build_only_not_charged",
    NORM_PLAIN: "production_only_new_class",
    SWIGLU: "production_only_new_class",
}


def _operands(device: str) -> dict:
    """Allocate every tensor outside any timed region."""
    half = dict(dtype=DTYPE, device=device)
    return {
        "activation": torch.randn(TOKENS, HIDDEN, **half),
        "weight": torch.randn(HIDDEN, HIDDEN, **half),
        "residual": torch.randn(TOKENS, HIDDEN, **half),
        "destination": torch.empty(TOKENS, HIDDEN, **half),
        "gain": torch.randn(HIDDEN, **half),
        "gate_up": torch.randn(TOKENS, 2 * HIDDEN, **half),
        "query": torch.randn(1, HEADS, TOKENS, HEAD_DIM, **half),
        "key": torch.randn(1, KV_HEADS, TOKENS, HEAD_DIM, **half),
        "value": torch.randn(1, KV_HEADS, TOKENS, HEAD_DIM, **half),
        "cache_key": torch.zeros(BLOCKS, BLOCK, KV_HEADS, HEAD_DIM, **half),
        "cache_value": torch.zeros(BLOCKS, BLOCK, KV_HEADS, HEAD_DIM, **half),
        "entry_key": torch.randn(TOKENS, KV_HEADS, HEAD_DIM, **half),
        "entry_value": torch.randn(TOKENS, KV_HEADS, HEAD_DIM, **half),
        "slots": torch.arange(TOKENS, dtype=torch.int32, device=device),
        "scatter_cache": torch.zeros(BLOCKS, KV_HEADS, HEAD_DIM, **half),
        "scatter_entries": torch.randn(TOKENS, KV_HEADS, HEAD_DIM, **half),
        "scatter_slots": torch.arange(
            TOKENS, dtype=torch.int32, device=device).view(-1, 1),
        "rotary": torch.randn(TOKENS, HEADS, 1, HEAD_DIM, **half),
        "cosine": torch.randn(TOKENS, 1, 1, HEAD_DIM, **half),
        "sine": torch.randn(TOKENS, 1, 1, HEAD_DIM, **half),
        # AscendRotaryEmbedding.forward_oot hands npu_rotary_embedding a flat
        # [tokens, heads * head_dim] pair and views it inside.
        "rope_query": torch.randn(TOKENS, HIDDEN, **half),
        "rope_key": torch.randn(TOKENS, KV_HIDDEN, **half),
        "qkv": torch.randn(TOKENS, HIDDEN + 2 * KV_HIDDEN, **half),
        "cos_sin_cache": torch.randn(8192, HEAD_DIM, **half),
        "positions": torch.arange(TOKENS, dtype=torch.int64, device=device),
        "q_norm": torch.randn(HEAD_DIM, **half),
        "k_norm": torch.randn(HEAD_DIM, **half),
    }


def _bare_calls(tensors: dict) -> dict:
    """The call forms the prices in the config were measured through."""
    return {
        "compiled_fx_aten_linear_node_dispatch":
            lambda: torch.nn.functional.linear(
                tensors["activation"], tensors["weight"]),
        "compiled_fx_torch_npu_norm_node_dispatch":
            lambda: torch_npu.npu_rms_norm(
                tensors["activation"], tensors["gain"], EPS),
        "compiled_fx_aten_copy_node_dispatch":
            lambda: tensors["destination"].copy_(tensors["activation"]),
        "compiled_fx_rope_node_dispatch":
            lambda: torch_npu.npu_rotary_mul(
                tensors["rotary"], tensors["cosine"], tensors["sine"]),
        "compiled_fx_fia_node_dispatch":
            lambda: torch_npu.npu_fused_infer_attention_score(
                tensors["query"], tensors["key"], tensors["value"],
                num_heads=HEADS, num_key_value_heads=KV_HEADS,
                input_layout="BNSD", scale=SCALE),
        "compiled_fx_reshape_cache_node_dispatch":
            lambda: torch_npu.npu_scatter_nd_update_(
                tensors["scatter_cache"], tensors["scatter_slots"],
                tensors["scatter_entries"]),
    }


def _production_calls(tensors: dict, norm_module) -> dict:
    """The call forms a real above-capture-limit step dispatches."""
    return {
        "compiled_fx_aten_linear_node_dispatch":
            lambda: torch.ops.vllm.unquantized_gemm(
                tensors["activation"], tensors["weight"], None),
        "compiled_fx_torch_npu_norm_node_dispatch":
            lambda: norm_module(tensors["activation"], tensors["residual"]),
        "compiled_fx_aten_copy_node_dispatch":
            lambda: tensors["destination"].copy_(tensors["activation"]),
        "compiled_fx_rope_node_dispatch":
            lambda: torch.ops.vllm.npu_rotary_embedding(
                tensors["positions"], tensors["rope_query"],
                tensors["rope_key"], tensors["cos_sin_cache"],
                HEAD_DIM, HEAD_DIM, True),
        "compiled_fx_fia_node_dispatch":
            lambda: torch.ops.npu.npu_fused_infer_attention_score(
                tensors["query"], tensors["key"], tensors["value"],
                num_heads=HEADS, num_key_value_heads=KV_HEADS,
                input_layout="BNSD", scale=SCALE),
        "compiled_fx_reshape_cache_node_dispatch":
            lambda: torch.ops.atb._npu_reshape_and_cache(
                tensors["entry_key"], tensors["entry_value"],
                tensors["cache_key"], tensors["cache_value"],
                tensors["slots"]),
        NORM_PLAIN:
            lambda: norm_module(tensors["activation"]),
        SWIGLU:
            lambda: torch.ops.npu.npu_swiglu(tensors["gate_up"], -1),
        FUSED:
            lambda: torch.ops.vllm.qkv_rmsnorm_rope(
                tensors["qkv"], tensors["cos_sin_cache"],
                tensors["positions"], tensors["q_norm"], tensors["k_norm"],
                HIDDEN, KV_HIDDEN, HEAD_DIM, EPS),
    }


def _sequence() -> list:
    """The class multiset one eager step issues, in charged proportion."""
    order, remaining = [], dict(CENSUS)
    while any(remaining.values()):
        for mechanism in CENSUS:
            if remaining[mechanism] > 0:
                order.append(mechanism)
                remaining[mechanism] -= 1
    return order


def _measure(bare: dict, production: dict, order: list) -> dict:
    """Time both forms of every scheduled class, order randomized per round.

    Where a submit sits in the stream moves its cost by as much as a Python
    wrapper does: an operator repeated back to back is cheap, one that follows
    a different operator is not, and an arm whose other classes are expensive
    runs at a different queue depth throughout.  Two fixed layouts were tried
    first and moved the copy control -- whose two forms are literally the same
    call -- by x1.44 one way and x0.54 the other, which is the size of the
    effect being measured.  No fixed layout is neutral, so the order is
    shuffled every round from a fixed seed instead.  Both forms of every class
    then see the same distribution of predecessors and queue depths, and the
    copy control -- whose two forms are the same call -- must come out equal;
    the spread it does show is the protocol's noise floor, and no ratio
    smaller than it can be read.
    """
    pairs = [(mechanism, form, calls[mechanism])
             for mechanism in order
             for form, calls in (("bare", bare), ("production", production))
             if mechanism in calls]
    for _, _, call in pairs:
        call()
    torch.npu.synchronize()
    shuffler = random.Random(SEED)
    samples: dict = {}
    for _ in range(ROUNDS):
        shuffler.shuffle(pairs)
        torch.npu.synchronize()
        for mechanism, form, call in pairs:
            start = time.perf_counter_ns()
            call()
            samples.setdefault((form, mechanism), []).append(
                time.perf_counter_ns() - start)
        torch.npu.synchronize()
    measured: dict = {}
    for (form, mechanism), values in samples.items():
        measured.setdefault(form, {})[mechanism] = {
            "median_ns": statistics.median(values),
            "mean_ns": statistics.fmean(values),
            "p10_ns": statistics.quantiles(values, n=10)[0],
            "p90_ns": statistics.quantiles(values, n=10)[-1],
            "sample_count": len(values),
        }
    return measured


def _norm_module(device: str, model: Path):
    """The installed production RMSNorm module, not a re-implementation."""
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops.layernorm import AscendRMSNorm, enable_custom_op
    from vllm_ascend.utils import set_weight_prefetch_method

    if not enable_custom_op():
        raise SystemExit(
            "enable_custom_op() is False: this build does not take the "
            "npu_add_rms_norm_bias branch production was observed to take")
    # The engine's own configuration for this model: the norm branches read
    # the model config, so an empty one would take a different path.
    configuration = EngineArgs(model=str(model)).create_engine_config()
    with set_current_vllm_config(configuration):
        module = AscendRMSNorm(HIDDEN, eps=EPS)
        # The model runner publishes this at init (model_runner_v1.py:265);
        # the no-residual norm branch calls into it.
        ascend_configuration = init_ascend_config(configuration)
        set_weight_prefetch_method(
            ascend_configuration.weight_prefetch_config)
    return module.to(device=device, dtype=DTYPE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is None:
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES must select an idle NPU")

    # These imports are what register the production custom ops; the engine
    # reaches them the same way.
    import vllm  # noqa: F401
    import vllm_ascend  # noqa: F401
    import vllm_ascend.ops  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    import vllm_ascend.patch.worker.patch_unquantized_gemm  # noqa: F401
    import vllm_ascend as _ascend_package

    device = "npu:0"
    torch.npu.set_device(device)
    # The worker does this at device init (worker.py:299); the fused triton
    # op reads the core counts it publishes.
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    tensors = _operands(device)
    order = _sequence()
    arms = _measure(
        _bare_calls(tensors),
        _production_calls(tensors, _norm_module(device, arguments.model)),
        order)

    comparison = {}
    for mechanism in CENSUS:
        production = arms["production"][mechanism]["median_ns"]
        bare = arms["bare"].get(mechanism)
        comparison[mechanism] = {
            "coverage": COVERAGE[mechanism],
            "bare_us": None if bare is None else bare["median_ns"] / 1000,
            "production_us": production / 1000,
            "production_over_bare": (
                None if bare is None else production / bare["median_ns"]),
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "Host submit cost of one node dispatch of each charged class, "
            "measured through the operator the engine actually dispatches and "
            "through the bare operator the current prices were measured "
            "through, under one interleaved-sequence protocol"),
        "forward_eligibility": "forward_eligible",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "container": arguments.container_name,
        "visible_devices": visible,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "framework": {"name": "torch_npu", "version": torch_npu.__version__},
        "vllm_version": vllm.__version__,
        "vllm_ascend_path": _ascend_package.__file__,
        "config_model": str(arguments.model),
        "decoder_shape": {
            "hidden": HIDDEN, "heads": HEADS, "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM, "tokens": TOKENS, "block_size": BLOCK,
            "dtype": str(DTYPE)},
        "census": dict(CENSUS),
        "charged_classes": list(CHARGED),
        "charged_node_count": sum(CENSUS[name] for name in CHARGED),
        "rounds": ROUNDS,
        "arms": arms,
        "comparison": comparison,
    }, indent=2) + "\n")

    for mechanism, row in comparison.items():
        bare = ("     n/a" if row["bare_us"] is None
                else f"{row['bare_us']:8.3f}")
        ratio = ("  n/a" if row["production_over_bare"] is None
                 else f"x{row['production_over_bare']:.2f}")
        print(f"{mechanism:48s} bare {bare} us  "
              f"production {row['production_us']:8.3f} us  {ratio}  "
              f"{row['coverage']}")


if __name__ == "__main__":
    main()
