#!/usr/bin/env python3
"""Measure source-derived 910B MoE Host-control work without Device kernels.

This deliberately does not run a FusedMoE operator.  It invokes the actual
vLLM/vLLM-Ascend wrapper functions with every Device child replaced by a
synthetic immediate result.  The measurement therefore isolates framework
control flow (custom-op entry, layer lookup, argument construction and
wrapper nesting) from the already-modelled NPU APIs and kernels.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.fused_moe.runner import default_moe_runner
from vllm_ascend.ops.fused_moe import fused_moe as ascend_fused_moe
from vllm_ascend.ops.fused_moe import moe_comm_method as ascend_moe_comm_method
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.ops.fused_moe import token_dispatcher as ascend_token_dispatcher
from vllm_ascend.quantization.quant_type import QuantType


CHAIN_LENGTHS = (1, 8, 64, 512, 2048)
FIT_CHAIN_LENGTHS = (1, 8, 64, 512)
COMPILED_CHAIN_LENGTHS = (1, 8, 64, 512)
COMPILED_FIT_CHAIN_LENGTHS = (1, 8, 64)
TRIALS = 8
WARMUP_REPEATS = 3
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    if count < 2:
        raise ValueError("linear fit requires at least two points")
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError("linear fit has degenerate inputs")
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in points
    ) / denominator
    intercept = mean_y - slope * mean_x
    if slope <= 0:
        raise ValueError("wrapper service slope must be positive")
    return intercept, slope


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        location - lower)


class _FakeStream:
    def record_event(self) -> object:
        return _SENTINEL

    def wait_event(self, event: object) -> None:
        if event is not _SENTINEL:
            raise ValueError("synthetic event identity differs")

    def wait_stream(self, stream: object) -> None:
        if not isinstance(stream, _FakeStream):
            raise TypeError("synthetic stream differs")


class _FakeDispatcher:
    def __init__(self) -> None:
        self._dispatch_output = SimpleNamespace(
            hidden_states=_SENTINEL,
            group_list=_SENTINEL,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            combine_metadata=_SENTINEL,
        )

    def token_dispatch(self, *, token_dispatch_input: object) -> object:
        return self._dispatch_output

    def token_combine(self, *, hidden_states: object,
                      combine_metadata: object) -> object:
        if hidden_states is not _SENTINEL:
            raise ValueError("synthetic MLP output differs")
        if combine_metadata is not _SENTINEL:
            raise ValueError("synthetic combine metadata differs")
        return _SENTINEL


class _FakeComm:
    def __init__(self) -> None:
        self.token_dispatcher = _FakeDispatcher()
        self.use_fusion_ops = False
        self._prepare_output = SimpleNamespace(
            hidden_states=_TENSOR,
            router_logits=_TENSOR,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )
        self._fused_experts_result = ascend_fused_moe.FusedExpertsResult(
            routed_out=_SENTINEL)

    def prepare(self, **kwargs: object) -> object:
        return self._prepare_output

    def finalize(self, *, hidden_states: object, **kwargs: object) -> object:
        if hidden_states is not _SENTINEL:
            raise ValueError("synthetic routed output differs")
        return _SENTINEL

    def fused_experts(self, *, fused_experts_input: object) -> object:
        if fused_experts_input is None:
            raise ValueError("synthetic fused-expert input is missing")
        return self._fused_experts_result

    def _apply_mlp(self, input_value: object) -> object:
        if input_value is None:
            raise ValueError("synthetic MLP input is missing")
        return _SENTINEL


class _FakeQuantMethod:
    def __init__(self) -> None:
        self._result = ascend_fused_moe.FusedExpertsResult(
            routed_out=_SENTINEL)

    def apply(self, **kwargs: object) -> object:
        return self._result


class _FakeLayer:
    def __init__(self) -> None:
        self.runner = SimpleNamespace(forward_impl=self.forward_impl)

    def ensure_moe_quant_config_init(self) -> None:
        return None

    def forward_impl(self, *args: object, **kwargs: object) -> object:
        return _OUTPUT_TENSOR


_SENTINEL = object()


class _SyntheticDispatchTensor:
    """A no-Device tensor-shaped value for source-only dispatch wrappers."""

    shape = torch.Size((1, 1))

    def to(self, dtype: torch.dtype) -> "_SyntheticDispatchTensor":
        if dtype is not torch.int64:
            raise ValueError("synthetic dispatch conversion must be int64")
        return self


_SYNTHETIC_DISPATCH_TENSOR = _SyntheticDispatchTensor()
# The registered vLLM custom op has a PrivateUse1 (NPU) implementation only.
# These are allocated once before measurement solely to select that dispatcher;
# every child reached from its source body remains patched to an immediate value.
_TENSOR = torch.empty((1, 1), dtype=torch.float32, device="npu:0")
_OUTPUT_TENSOR = torch.empty((1, 1), dtype=torch.float32, device="npu:0")
# A custom op may not return an input or a tensor returned by an earlier call
# in the same compiled chain.  Allocate a ring before measurement, then select
# one distinct output per wrapper call; this keeps allocation out of the timed
# boundary.
_COMPILED_OUTPUTS = tuple(
    torch.empty_like(_OUTPUT_TENSOR)
    for _ in range(max(COMPILED_CHAIN_LENGTHS)))
_COMPILED_OUTPUT_INDEX = 0
_ORIGINAL_ASCEND_FUSED_FORWARD = ascend_fused_moe.AscendFusedMoE.forward_impl


def _build_fused_fixture() -> object:
    fused = object.__new__(ascend_fused_moe.AscendFusedMoE)
    fused.quant_method = _FakeQuantMethod()
    fused.enable_npugraph_ex_static_kernel = False
    fused.multistream_overlap_gate = False
    fused.dynamic_eplb = False
    fused.enable_shared_expert_dp = False
    fused.quant_type = QuantType.NONE
    fused.top_k = 1
    fused.renormalize = False
    fused.use_grouped_topk = False
    fused.global_num_experts = 1
    fused._expert_map = None
    fused.topk_group = None
    fused.num_expert_group = None
    fused.custom_routing_function = None
    fused.scoring_func = "softmax"
    fused.routed_scaling_factor = 1.0
    fused.e_score_correction_bias = None
    fused.activation = "silu"
    fused.apply_router_weight_on_input = False
    fused.log2phy = None
    fused.global_redundant_expert_num = 0
    fused.reduce_results = False
    return fused


_FAKE_LAYER = _FakeLayer()
_FAKE_RUNNER = object.__new__(ascend_fused_moe.AscendMoERunner)
_FAKE_SHARED = object.__new__(ascend_fused_moe.AscendSharedFusedMoE)
_FAKE_SHARED.multistream_overlap_gate = False
_FAKE_SHARED._shared_experts = _SENTINEL
_FAKE_SHARED._forward_shared_experts = lambda *args: _SENTINEL
_FAKE_FUSED = _build_fused_fixture()
_FAKE_METHOD = object.__new__(ascend_fused_moe.AscendUnquantizedFusedMoEMethod)
_FAKE_METHOD.moe = SimpleNamespace(has_bias=False)
_FAKE_METHOD.dynamic_eplb = False
_FAKE_EXPERT_LAYER = SimpleNamespace(
    zero_expert_num=0,
    zero_expert_type=None,
    vllm_config=SimpleNamespace(model_config=None),
    w13_weight=_TENSOR,
    w2_weight=_TENSOR,
    w13_bias=None,
    w2_bias=None,
)
_FAKE_COMM = _FakeComm()
_FAKE_EXTRA_CONTEXT = SimpleNamespace(
    in_profile_run=False,
    moe_comm_method=_FAKE_COMM,
    moe_comm_type=None,
    flash_comm_v1_enabled=False,
)
_FAKE_FORWARD_CONTEXT = SimpleNamespace(
    moe_layer_index=0,
    all_moe_layers=("synthetic",),
)
_FAKE_FUSED_INPUT = SimpleNamespace(
    hidden_states=_TENSOR,
    topk_weights=_TENSOR,
    topk_ids=_SENTINEL,
    routing=SimpleNamespace(log2phy=None),
    weights=_SENTINEL,
    quant=SimpleNamespace(is_mxfp=False, quant_type=QuantType.NONE),
    activation="silu",
    need_trans=False,
    dynamic_eplb=False,
)
_FAKE_UNQUANT_MLP_INPUT = SimpleNamespace(
    hidden_states=_TENSOR,
    group_list=_TENSOR,
    group_list_type=1,
    dynamic_scale=None,
    topk_scales=None,
    weights=SimpleNamespace(
        w1=_TENSOR,
        w2=_TENSOR,
        w1_bias=None,
        w2_bias=None,
        w1_scale=None,
        w2_scale=None,
        w1_scale_bias=None,
        w2_scale_bias=None,
        w1_offset=None,
        w2_offset=None,
    ),
    quant=SimpleNamespace(is_quant=False, is_mxfp=False),
    fusion=False,
    activation="silu",
    need_trans=False,
    dynamic_eplb=False,
)
_FAKE_ALLGATHER_DISPATCHER = object.__new__(
    ascend_token_dispatcher.TokenDispatcherWithAllGather)
_FAKE_ALLGATHER_DISPATCHER.top_k = 8
_FAKE_ALLGATHER_DISPATCHER.num_experts_local = 64
_FAKE_ALLGATHER_TOKEN_DISPATCH_INPUT = SimpleNamespace(
    hidden_states=_SYNTHETIC_DISPATCH_TENSOR,
    topk_weights=_SYNTHETIC_DISPATCH_TENSOR,
    topk_ids=_SYNTHETIC_DISPATCH_TENSOR,
    routing=SimpleNamespace(
        expert_map=None,
        pertoken_scale=None,
        global_redundant_expert_num=0,
        apply_router_weight_on_input=False,
    ),
    quant=SimpleNamespace(is_int_quant=False),
)
_FAKE_SHARED_RESULT = ascend_fused_moe.FusedExpertsResult(
    routed_out=_SENTINEL)


def _measure_custom_op_entry() -> None:
    result = default_moe_runner._moe_forward(
        _TENSOR, _TENSOR, None, "synthetic-layer")
    if result is not _OUTPUT_TENSOR:
        raise ValueError("synthetic custom-op entry result differs")


def _measure_registered_custom_op_dispatch() -> None:
    """Exercise torch.ops dispatch plus the same immediate source body."""

    result = torch.ops.vllm.moe_forward.default(
        _TENSOR, _TENSOR, None, "synthetic-layer")
    if result is not _OUTPUT_TENSOR:
        raise ValueError("synthetic registered custom-op result differs")


def _measure_runner_forward() -> None:
    result = ascend_fused_moe.AscendMoERunner.forward_impl(
        _FAKE_RUNNER, _FAKE_LAYER, _TENSOR, _TENSOR, None)
    if result is not _OUTPUT_TENSOR:
        raise ValueError("synthetic runner result differs")


def _measure_shared_forward() -> None:
    result = ascend_fused_moe.AscendSharedFusedMoE.forward_impl(
        _FAKE_SHARED, _TENSOR, _TENSOR)
    if result != (_SENTINEL, _SENTINEL):
        raise ValueError("synthetic shared MoE result differs")


def _measure_fused_forward() -> None:
    result = _ORIGINAL_ASCEND_FUSED_FORWARD(
        _FAKE_FUSED, _TENSOR, _TENSOR, return_with_event=True)
    if result.routed_out is not _SENTINEL:
        raise ValueError("synthetic fused MoE result differs")


def _measure_expert_apply() -> None:
    result = ascend_fused_moe.AscendUnquantizedFusedMoEMethod.apply(
        _FAKE_METHOD,
        _FAKE_EXPERT_LAYER,
        _TENSOR,
        False,
        1,
        _TENSOR,
        False,
        global_num_experts=1,
    )
    if (not isinstance(result, ascend_fused_moe.FusedExpertsResult)
            or result.routed_out is not _SENTINEL):
        raise ValueError("synthetic expert-apply result differs")


def _measure_fused_experts() -> None:
    result = ascend_moe_comm_method.MoECommMethod.fused_experts(
        _FAKE_COMM, _FAKE_FUSED_INPUT)
    if result.routed_out is not _SENTINEL:
        raise ValueError("synthetic fused-experts result differs")


def _measure_unified_unquant_mlp_control() -> None:
    """Measure only the source wrapper around the three MoE MLP children."""

    result = moe_mlp.unified_apply_mlp(
        mlp_compute_input=_FAKE_UNQUANT_MLP_INPUT)
    if result is not _TENSOR:
        raise ValueError("synthetic unquantized MLP result differs")


def _measure_allgather_token_dispatch_control() -> None:
    """Measure the source wrapper around the routed-token NPU child."""

    result = ascend_token_dispatcher.TokenDispatcherWithAllGather.token_dispatch(
        _FAKE_ALLGATHER_DISPATCHER,
        _FAKE_ALLGATHER_TOKEN_DISPATCH_INPUT,
    )
    if (result.hidden_states is not _SYNTHETIC_DISPATCH_TENSOR
            or result.group_list is not _SYNTHETIC_DISPATCH_TENSOR):
        raise ValueError("synthetic all-gather dispatch result differs")


MEASUREMENTS = {
    "custom_op_entry": _measure_custom_op_entry,
    "registered_custom_op_dispatch": _measure_registered_custom_op_dispatch,
    "runner_forward": _measure_runner_forward,
    "shared_forward": _measure_shared_forward,
    "fused_forward": _measure_fused_forward,
    "expert_apply": _measure_expert_apply,
    "fused_experts_control": _measure_fused_experts,
    "unified_unquant_mlp_control": _measure_unified_unquant_mlp_control,
    "allgather_token_dispatch_control": _measure_allgather_token_dispatch_control,
}
COMPOSITE_MEASUREMENTS = (
    "custom_op_entry",
    "runner_forward",
    "shared_forward",
    "fused_forward",
    "expert_apply",
    "fused_experts_control",
)


@torch.library.custom_op(
    "midlayer_moe_wrapper_probe::composite", mutates_args=())
def _compiled_wrapper_composite(source: torch.Tensor) -> torch.Tensor:
    """Execute the source-control composite beneath one opaque FX entry.

    The child calls have already been replaced by immediate synthetic values
    by ``_synthetic_source_environment``.  Returning a preallocated NPU
    tensor keeps the probe asynchronous and excludes NPU tensor allocation
    and Device work. Python control objects remain part of this boundary.
    """

    _measure_custom_op_entry()
    _measure_runner_forward()
    _measure_shared_forward()
    _measure_fused_forward()
    _measure_expert_apply()
    _measure_fused_experts()
    global _COMPILED_OUTPUT_INDEX
    if source.device != _OUTPUT_TENSOR.device:
        raise ValueError("compiled wrapper probe device differs")
    if _COMPILED_OUTPUT_INDEX >= len(_COMPILED_OUTPUTS):
        raise RuntimeError("compiled wrapper output ring is exhausted")
    output = _COMPILED_OUTPUTS[_COMPILED_OUTPUT_INDEX]
    _COMPILED_OUTPUT_INDEX += 1
    return output


@torch.library.register_fake("midlayer_moe_wrapper_probe::composite")
def _compiled_wrapper_composite_fake(source: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(source)


class _CompiledWrapperChain(torch.nn.Module):
    """A count-swept opaque FX graph with one wrapper composite per node."""

    def __init__(self, chain_length: int) -> None:
        super().__init__()
        if chain_length not in COMPILED_CHAIN_LENGTHS:
            raise ValueError("compiled wrapper chain length is unsupported")
        self.chain_length = chain_length

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.chain_length):
            value = torch.ops.midlayer_moe_wrapper_probe.composite(value)
        return value


def _compiled_wrapper_backend(name: str):
    """Select and identify the compiler whose Host control is being measured.

    Ordinary torch.compile defaults to Inductor. Ascend Serving's npugraph_ex
    path explicitly selects TorchAir reduce-overhead, so an unspecified
    default cannot establish coverage of that path.
    """
    if name == "inductor":
        from torch._dynamo.backends.registry import lookup_backend

        compiler = lookup_backend("inductor")
        return "inductor", {
            "name": name,
            "compiler_module": compiler.__module__,
            "compiler_qualname": compiler.__qualname__,
            "compiler_source": _source_identity(compiler),
        }
    if name == "torchair_npugraph_ex":
        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        return torchair.get_npu_backend(compiler_config=config), {
            "name": name,
            "mode": "reduce-overhead",
            "run_eagerly": True,
            "disable_reinplace_inplaceable_ops_pass": True,
            "static_kernel_requested": False,
            "backend_factory_source": _source_identity(torchair.get_npu_backend),
        }
    raise ValueError(f"unsupported compiled wrapper backend: {name}")


def _measure_compiled_wrapper_composite(backend_name: str) -> dict[str, object]:
    """Measure the complete ``FX entry → source wrapper`` boundary.

    Comparing this slope with the separately measured generic compiled-FX
    slope resolves whether the source-wrapper composite can be appended as a
    runtime-prepare event or would double count the generic entry.
    """

    global _COMPILED_OUTPUT_INDEX
    backend, backend_identity = _compiled_wrapper_backend(backend_name)
    samples = {length: [] for length in COMPILED_CHAIN_LENGTHS}
    for chain_length in COMPILED_CHAIN_LENGTHS:
        compiled = torch.compile(
            _CompiledWrapperChain(chain_length).eval(),
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        for _ in range(WARMUP_REPEATS):
            _COMPILED_OUTPUT_INDEX = 0
            compiled(_TENSOR)
        torch.npu.synchronize()
        for _ in range(TRIALS):
            _COMPILED_OUTPUT_INDEX = 0
            started = time.perf_counter_ns()
            result = compiled(_TENSOR)
            elapsed = time.perf_counter_ns() - started
            if result.device != _TENSOR.device:
                raise ValueError("compiled wrapper probe changed the device")
            if elapsed <= 0:
                raise ValueError("compiled wrapper interval must be positive")
            samples[chain_length].append(elapsed)
        torch.npu.synchronize()
    medians = {
        length: statistics.median(samples[length])
        for length in COMPILED_CHAIN_LENGTHS
    }
    intercept, slope = _linear_fit([
        (length, medians[length])
        for length in COMPILED_FIT_CHAIN_LENGTHS
    ])
    holdout = medians[512]
    return {
        "compiler_backend": backend_identity,
        "measurement_boundary": (
            "compiled opaque FX entry through the complete source-wrapper "
            "control composite with all Device children replaced by "
            "immediate values"),
        "wall_fixed_intercept_ns": intercept,
        "wall_slope_ns_per_composite_call": slope,
        "holdout_wall_measured_ns": holdout,
        "holdout_wall_predicted_ns": intercept + slope * 512,
        "holdout_wall_signed_error_percent": 100.0 * (
            (intercept + slope * 512) / holdout - 1.0),
        "predeclared": {
            "chain_lengths": list(COMPILED_CHAIN_LENGTHS),
            "fit_chain_lengths": list(COMPILED_FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 512,
        },
        "cells": [
            {
                "chain_length": length,
                "sample_count": len(samples[length]),
                "wall_ns": {
                    "p10": _percentile(samples[length], 0.1),
                    "median": medians[length],
                    "p90": _percentile(samples[length], 0.9),
                },
            }
            for length in COMPILED_CHAIN_LENGTHS
        ],
    }


def _synthetic_source_environment():
    """Install stubs once, outside the timed loop."""

    patches = ExitStack()
    patches.enter_context(patch.object(
        default_moe_runner, "get_layer_from_name", return_value=_FAKE_LAYER))
    patches.enter_context(patch.object(
        ascend_fused_moe.AscendFusedMoE,
        "forward_impl",
        return_value=_FAKE_SHARED_RESULT))
    patches.enter_context(patch.object(
        ascend_fused_moe, "get_forward_context",
        return_value=_FAKE_FORWARD_CONTEXT))
    patches.enter_context(patch.object(
        ascend_fused_moe, "_EXTRA_CTX", _FAKE_EXTRA_CONTEXT))
    patches.enter_context(patch.object(
        ascend_fused_moe, "select_experts",
        return_value=(_TENSOR, _TENSOR)))
    patches.enter_context(patch.object(
        ascend_moe_comm_method, "_EXTRA_CTX", _FAKE_EXTRA_CONTEXT))
    patches.enter_context(patch.object(
        torch.npu, "current_stream", return_value=_FakeStream()))
    patches.enter_context(patch.object(
        moe_mlp.torch_npu,
        "npu_grouped_matmul",
        return_value=[_TENSOR],
    ))
    patches.enter_context(patch.object(
        moe_mlp.torch_npu,
        "npu_swiglu",
        return_value=_TENSOR,
    ))
    patches.enter_context(patch.object(
        ascend_token_dispatcher.DeviceOperator,
        "npu_moe_init_routing",
        return_value=(
            _SYNTHETIC_DISPATCH_TENSOR,
            _SYNTHETIC_DISPATCH_TENSOR,
            _SYNTHETIC_DISPATCH_TENSOR,
            None,
        ),
    ))
    return patches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--compiled-backend", choices=("inductor", "torchair_npugraph_ex"),
        default="inductor", help="Compiler used for the opaque FX control chain")
    args = parser.parse_args()
    if len(args.baseline_commit) != 40:
        parser.error("--baseline-commit must be a full Git commit")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    args.output_dir = args.output_dir.resolve()
    if "profiler_output" not in args.output_dir.parts:
        parser.error("--output-dir must be inside profiler_output")
    return args


def _source_identity(module: object) -> dict[str, str]:
    source_path = Path(inspect.getsourcefile(module)).resolve()
    return {"path": str(source_path), "sha256": _sha256(source_path)}


def main() -> None:
    args = _parse_args()
    runner_path = Path(__file__).resolve()
    runner_sha256 = _sha256(runner_path)
    if runner_sha256 != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "baseline_commit": args.baseline_commit,
        "container_name": args.container_name,
        "runner": {"path": str(runner_path), "sha256": runner_sha256},
        "compiled_backend_requested": args.compiled_backend,
        "source_modules": {
            "vllm_default_moe_runner": _source_identity(default_moe_runner),
            "vllm_ascend_fused_moe": _source_identity(ascend_fused_moe),
        },
        "contract": {
            "boundary": (
                "source wrapper entry to return with all Device child calls "
                "replaced by immediate synthetic values"),
            "patch_installation": (
                "all synthetic patches are installed once before warmup and "
                "remain installed through every timed sample; patch setup and "
                "teardown are outside the measurement boundary"),
            "excluded": (
                "all NPU kernels, CANN API submits, collectives, NPU tensor allocations "
                "and Serving/model duration; NPU inputs are allocated once "
                "before timing only to select the registered dispatch backend"),
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": list(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 2048,
            "trials_per_cell": TRIALS,
            "host_clock": "time.perf_counter_ns",
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        with _synthetic_source_environment():
            samples: dict[tuple[str, int], list[int]] = {
                (name, length): []
                for name in MEASUREMENTS for length in CHAIN_LENGTHS
            }
            for measurement in MEASUREMENTS.values():
                for _ in range(WARMUP_REPEATS):
                    measurement()
            sequence = 0
            cells = tuple(
                (name, length)
                for name in MEASUREMENTS for length in CHAIN_LENGTHS
            )
            for trial in range(TRIALS):
                offset = trial % len(cells)
                ordered = cells[offset:] + cells[:offset]
                if trial % 2:
                    ordered = tuple(reversed(ordered))
                for order, (name, length) in enumerate(ordered, start=1):
                    measurement = MEASUREMENTS[name]
                    started = time.perf_counter_ns()
                    for _ in range(length):
                        measurement()
                    elapsed = time.perf_counter_ns() - started
                    samples[(name, length)].append(elapsed)
                    sequence += 1
                    _append_json_line(trials_path, {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "trial": trial + 1,
                        "order": order,
                        "measurement": name,
                        "chain_length": length,
                        "wall_ns": elapsed,
                    })
            compiled_wrapper_composite = _measure_compiled_wrapper_composite(
                args.compiled_backend)
        fits = []
        for name in MEASUREMENTS:
            medians = {
                length: statistics.median(samples[(name, length)])
                for length in CHAIN_LENGTHS
            }
            intercept, slope = _linear_fit([
                (length, medians[length])
                for length in FIT_CHAIN_LENGTHS
            ])
            holdout = medians[2048]
            fits.append({
                "measurement": name,
                "wall_fixed_intercept_ns": intercept,
                "wall_slope_ns_per_call": slope,
                "holdout_wall_measured_ns": holdout,
                "holdout_wall_predicted_ns": intercept + slope * 2048,
                "holdout_wall_signed_error_percent": 100.0 * (
                    (intercept + slope * 2048) / holdout - 1.0),
                "cells": [
                    {
                        "chain_length": length,
                        "sample_count": len(samples[(name, length)]),
                        "wall_ns": {
                            "p10": _percentile(samples[(name, length)], 0.1),
                            "median": medians[length],
                            "p90": _percentile(samples[(name, length)], 0.9),
                        },
                    }
                    for length in CHAIN_LENGTHS
                ],
            })
        fit_by_measurement = {
            entry["measurement"]: entry for entry in fits
        }
        total_slope = sum(
            fit_by_measurement[name]["wall_slope_ns_per_call"]
            for name in COMPOSITE_MEASUREMENTS
        )
        registered_dispatch_increment = (
            fit_by_measurement["registered_custom_op_dispatch"]
            ["wall_slope_ns_per_call"]
            - fit_by_measurement["custom_op_entry"]
            ["wall_slope_ns_per_call"]
        )
        if registered_dispatch_increment <= 0:
            raise ValueError(
                "registered custom-op dispatch must exceed direct source entry")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "fits": fits,
            "source_wrapper_control_total_ns_per_composite_call": total_slope,
            "compiled_wrapper_composite": compiled_wrapper_composite,
            "registered_custom_op_dispatch_increment_ns_per_call": (
                registered_dispatch_increment),
            "interpretation": (
                "The composite total excludes registered_custom_op_dispatch, "
                "which is an alternate path used only to isolate the dispatcher "
                "increment over direct source entry. Neither value may enter "
                "Forward unless a source-agnostic composite runtime descriptor "
                "and a separate production contract are proved."),
        }
        _write_json(summary_path, summary)
        manifest["status"] = "pass"
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__, "message": str(error)}
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
