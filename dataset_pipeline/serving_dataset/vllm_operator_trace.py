"""Fail-closed vLLM operator and Scheduler sidecar capture for NVIDIA raw cases.

This is the active re-implementation of the capture that produced the A800
``serve_exp`` raw cases (``plugin_trace.json_scheduler.jsonl`` plus
``plugin_trace.json_operators.jsonl.rank*``).  It writes the same record
schema the ShareGPT raw adapter reads, for vLLM 0.11.0 (Model Runner V1) and
vLLM 0.30.0 (Model Runner V2), and adds one NVTX range per formal forward so
an attached Nsight Systems capture can attribute every kernel launch to its
Scheduler iteration exactly instead of through a sampler-kernel heuristic.

Step identity is carried by the Scheduler output object itself: the Scheduler
wrapper stamps each output with its iteration number and the model-runner
wrapper reads that stamp.  Runner calls without the stamp (Model Runner V2
warmup, profile and dummy runs) are therefore never confused with Scheduler
iterations, which a call counter on the runner cannot guarantee.

Installation is opt-in through two environment variables and is performed by
``instrumentation_bootstrap/sitecustomize.py`` in every spawned vLLM process.
"""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from dataset_pipeline.serving_dataset.json_shape import layer_index as _layer_index


SCHEDULER_SIDECAR_ENV = "MIDLAYER_SCHEDULER_SIDECAR_PATH"
OPERATOR_TRACE_ENV = "MIDLAYER_OPERATOR_TRACE_PATH"
NVTX_STEP_PREFIX = "midlayer-op-step:"
NVTX_SAMPLE_PREFIX = "midlayer-op-sample:"
_ITER_ATTRIBUTE = "_midlayer_operator_trace_iter_id"

_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_RUNNER_V1_MODULE = "vllm.v1.worker.gpu_model_runner"
_RUNNER_V2_MODULE = "vllm.v1.worker.gpu.model_runner"
_COMM_MODULE = "vllm.distributed.communication_op"
_LINEAR_MODULE = "vllm.model_executor.layers.linear"
_PARALLEL_STATE_MODULE = "vllm.distributed.parallel_state"
_MOE_LAYER_V011_MODULE = "vllm.model_executor.layers.fused_moe.layer"
_MOE_ROUTER_V030_MODULE = (
    "vllm.model_executor.layers.fused_moe.router.fused_moe_router"
)
_TARGET_MODULES = (
    _SCHEDULER_MODULE,
    _RUNNER_V1_MODULE,
    _RUNNER_V2_MODULE,
    _COMM_MODULE,
    _LINEAR_MODULE,
    _PARALLEL_STATE_MODULE,
    _MOE_LAYER_V011_MODULE,
    _MOE_ROUTER_V030_MODULE,
)

# MoE routing is observed at a release-specific entry point: vLLM 0.11.0 routes
# through the FusedMoE.select_experts staticmethod, vLLM 0.30.0 through
# FusedMoERouter.select_experts.
_MOE_ROUTING_MODULE_BY_RELEASE = {
    "0.11.0": _MOE_LAYER_V011_MODULE,
    "0.30.0": _MOE_ROUTER_V030_MODULE,
}

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_STEP_LOCK = threading.Lock()


class _StepState:
    """Per-process forward state; one formal forward runs at a time."""

    def __init__(self) -> None:
        self.hooks_installed = False
        self.rank: int | None = None
        self.step_id: int | None = None
        self.last_step_id: int | None = None
        self.op_idx = 0
        self.buffer: list[str] = []
        self.pending_topk_ids: object | None = None


_STATE = _StepState()
_SIDECAR_COUNTER = [0]
_SIDECAR_STREAM: list[Any] = []


def _required_path(name: str) -> Path:
    if name not in os.environ:
        raise RuntimeError(f"{name} is required for vLLM operator capture")
    path = Path(os.environ[name])
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"capture parent directory is absent: {path.parent}")
    return path


def _close_sidecar() -> None:
    for stream in _SIDECAR_STREAM:
        stream.close()
    _SIDECAR_STREAM.clear()


atexit.register(_close_sidecar)


def _append_sidecar(record: dict[str, Any]) -> None:
    if not _SIDECAR_STREAM:
        _SIDECAR_STREAM.append(
            _required_path(SCHEDULER_SIDECAR_ENV).open("a", encoding="utf-8"))
    stream = _SIDECAR_STREAM[0]
    stream.write(json.dumps(record) + "\n")
    stream.flush()


def _flush_operator_buffer() -> None:
    if not _STATE.buffer:
        return
    if _STATE.rank is None:
        raise RuntimeError("operator records were buffered before rank binding")
    path = Path(f"{_required_path(OPERATOR_TRACE_ENV)}.rank{_STATE.rank}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("".join(_STATE.buffer))
    _STATE.buffer.clear()


def _emit_operator(
    op_type: str,
    position: str,
    layer_idx: int,
    attrs: dict[str, object],
) -> None:
    if _STATE.step_id is None:
        raise RuntimeError(
            f"{op_type} operator ran outside a formal Scheduler forward")
    record = {
        "step_id": _STATE.step_id,
        "op_idx": _STATE.op_idx,
        "op_type": op_type,
        "position": position,
        "layer_idx": layer_idx,
        "attrs": attrs,
        "ts_us": time.perf_counter_ns() / 1000.0,
    }
    _STATE.op_idx += 1
    _STATE.buffer.append(json.dumps(record) + "\n")


def _first_tensor(args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    import torch

    for value in args:
        if isinstance(value, torch.Tensor):
            return value
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return value
    return None




def _classify(module: object) -> tuple[str, str] | None:
    """Map one live module to the legacy A800 operator vocabulary."""

    name = type(module).__name__
    if name.endswith("QKVParallelLinear"):
        return "linear", "qkv_proj"
    if name.endswith("RowParallelLinear"):
        return "linear", "row"
    if name.endswith("MergedColumnParallelLinear") or name.endswith("ColumnParallelLinear"):
        return "linear", "col"
    if name.endswith("ParallelLMHead"):
        return "linear", "lm_head"
    if "RMSNorm" in name:
        return "rms_norm", "norm"
    if "RotaryEmbedding" in name:
        return "rotary", ""
    if name == "Attention":
        return "attention", ""
    if "SiluAndMul" in name or "GeluAndMul" in name or name.endswith("AndMul"):
        return "activation", ""
    if "LogitsProcessor" in name:
        return "linear", "lm_head"
    # vLLM 0.11.0 exposes the MoE layer as FusedMoE; vLLM 0.30.0 builds it
    # through FusedMoEFactory as a MoERunner.
    if name in ("FusedMoE", "MoERunner"):
        return "moe", ""
    return None


def _linear_output_width(module: object) -> int:
    if hasattr(module, "output_size_per_partition"):
        width = module.output_size_per_partition
    elif hasattr(module, "vocab_size"):
        width = module.vocab_size
    else:
        raise AttributeError(
            f"linear capture module has no output width: {type(module).__name__}")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise TypeError("linear output width must be a positive integer")
    return width


def _moe_expert_count(module: object) -> int:
    if hasattr(module, "global_num_experts"):
        count = module.global_num_experts
    elif hasattr(module, "moe_config"):
        count = module.moe_config.num_logical_experts
    else:
        raise AttributeError(
            f"MoE capture module has no global expert count: {type(module).__name__}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise TypeError("MoE global expert count must be a positive integer")
    return count


def _pre_hook(module_name: str, op_type: str, position: str):
    layer_idx = _layer_index(module_name)

    def record(module: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        tensor = _first_tensor(args, kwargs)
        if tensor is None:
            return
        import torch

        shape = [int(value) for value in tensor.shape]
        tokens = shape[0] if shape else 1
        actual_op_type = op_type
        if op_type == "rms_norm":
            if len(args) >= 2:
                residual = args[1]
            elif "residual" in kwargs:
                residual = kwargs["residual"]
            else:
                residual = None
            if isinstance(residual, torch.Tensor):
                actual_op_type = "fused_add_rms_norm"
        attrs: dict[str, object] = {"input_shape": shape}
        if actual_op_type == "linear":
            attrs.update({
                "B": 1, "M": tokens, "N": _linear_output_width(module),
                "K": shape[-1],
            })
        elif actual_op_type == "attention":
            attrs.update({"batch": 1, "seq_len": tokens, "is_decode": tokens <= 1})
        else:
            attrs.update({"batch": 1, "seq_len": tokens})
        _emit_operator(actual_op_type, position, layer_idx, attrs)

    return record


def _moe_post_hook(module_name: str, position: str):
    layer_idx = _layer_index(module_name)

    def record(
        module: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        _output: object,
    ) -> None:
        tensor = _first_tensor(args, kwargs)
        if tensor is None:
            raise RuntimeError("MoE capture found no hidden-state tensor")
        import torch

        topk_ids = _STATE.pending_topk_ids
        _STATE.pending_topk_ids = None
        if not isinstance(topk_ids, torch.Tensor):
            raise RuntimeError(
                "MoE forward completed without an observed select_experts call")
        shape = [int(value) for value in tensor.shape]
        tokens = shape[0] if shape else 1
        flat = topk_ids.detach().reshape(-1).to(torch.int64)
        histogram = torch.bincount(
            flat, minlength=_moe_expert_count(module)).cpu().tolist()
        if len(histogram) != _moe_expert_count(module):
            raise RuntimeError("MoE routing references an expert outside the model")
        _emit_operator("moe", position, layer_idx, {
            "input_shape": shape,
            "batch": 1,
            "seq_len": tokens,
            "is_decode": tokens <= 1,
            "tok_per_expert": histogram,
        })

    return record


def _install_module_hooks(model: object) -> None:
    if not hasattr(model, "named_modules"):
        raise AttributeError("model runner model lacks named_modules")
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():
        _STATE.rank = int(distributed.get_rank())
    else:
        _STATE.rank = 0
    for module_name, module in model.named_modules():
        classified = _classify(module)
        if classified is None:
            continue
        op_type, position = classified
        if op_type == "moe":
            module.register_forward_hook(
                _moe_post_hook(module_name, position), with_kwargs=True)
        else:
            module.register_forward_pre_hook(
                _pre_hook(module_name, op_type, position), with_kwargs=True)
    _STATE.hooks_installed = True


def _scheduler_iter_id(scheduler_output: object) -> int | None:
    if not hasattr(scheduler_output, _ITER_ATTRIBUTE):
        return None
    value = getattr(scheduler_output, _ITER_ATTRIBUTE)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("operator trace Scheduler iteration stamp is invalid")
    return value


def _patch_runner(runner_cls: type) -> None:
    original = runner_cls.execute_model
    if hasattr(original, "_midlayer_operator_trace_patched"):
        return

    def execute_model(self: object, scheduler_output: object, *args: object, **kwargs: object) -> object:
        dummy_run = "dummy_run" in kwargs and kwargs["dummy_run"] is True
        iter_id = None if dummy_run else _scheduler_iter_id(scheduler_output)
        if iter_id is None:
            _STATE.last_step_id = None
            return original(self, scheduler_output, *args, **kwargs)
        import torch

        with _STEP_LOCK:
            if not _STATE.hooks_installed:
                if not hasattr(self, "model"):
                    raise AttributeError("GPUModelRunner lacks model")
                _install_module_hooks(self.model)
            if _STATE.step_id is not None:
                raise RuntimeError("operator trace observed overlapping forwards")
            _STATE.step_id = iter_id
            _STATE.last_step_id = iter_id
            _STATE.op_idx = 0
            torch.cuda.nvtx.range_push(f"{NVTX_STEP_PREFIX}{iter_id}")
            try:
                return original(self, scheduler_output, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                _STATE.step_id = None
                if _STATE.pending_topk_ids is not None:
                    raise RuntimeError("MoE routing was observed without its layer")
                _flush_operator_buffer()

    execute_model._midlayer_operator_trace_patched = True
    runner_cls.execute_model = execute_model
    if hasattr(runner_cls, "sample_tokens"):
        _patch_sample_tokens(runner_cls)


def _patch_sample_tokens(runner_cls: type) -> None:
    original = runner_cls.sample_tokens
    if hasattr(original, "_midlayer_operator_trace_patched"):
        return

    def sample_tokens(self: object, *args: object, **kwargs: object) -> object:
        iter_id = _STATE.last_step_id
        if iter_id is None:
            return original(self, *args, **kwargs)
        import torch

        # Model Runner V2 computes logits here, so the step stays open and its
        # operator index continues from the forward.
        with _STEP_LOCK:
            _STATE.last_step_id = None
            if _STATE.step_id is not None:
                raise RuntimeError("sampler started inside an open forward")
            _STATE.step_id = iter_id
            torch.cuda.nvtx.range_push(f"{NVTX_SAMPLE_PREFIX}{iter_id}")
            try:
                return original(self, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                _STATE.step_id = None
                _flush_operator_buffer()

    sample_tokens._midlayer_operator_trace_patched = True
    runner_cls.sample_tokens = sample_tokens


def _patch_scheduler(scheduler_cls: type) -> None:
    original = scheduler_cls.schedule
    if hasattr(original, "_midlayer_operator_trace_patched"):
        return

    def schedule(self: object, *args: object, **kwargs: object) -> object:
        output = original(self, *args, **kwargs)
        iter_id = _SIDECAR_COUNTER[0]
        _SIDECAR_COUNTER[0] += 1
        setattr(output, _ITER_ATTRIBUTE, iter_id)
        _append_sidecar({
            "iter_id": iter_id,
            "ts_us": time.time_ns() // 1000,
            "scheduled": {
                str(request_id): int(tokens)
                for request_id, tokens in output.num_scheduled_tokens.items()
            },
            "total_tokens": int(output.total_num_scheduled_tokens),
            "finished": sorted(str(value) for value in output.finished_req_ids),
        })
        return output

    schedule._midlayer_operator_trace_patched = True
    scheduler_cls.schedule = schedule


def _all_reduce_wrapper(function: Any) -> Any:
    if hasattr(function, "_midlayer_operator_trace_patched"):
        return function

    def all_reduce(input_: Any) -> Any:
        import torch

        if _STATE.step_id is not None and isinstance(input_, torch.Tensor):
            shape = [int(value) for value in input_.shape]
            _emit_operator("all_reduce", "tp", -1, {
                "batch": 1,
                "seq_len": shape[0] if shape else 1,
                "msg_bytes": int(input_.numel() * input_.element_size()),
                "input_shape": shape,
            })
        return function(input_)

    all_reduce._midlayer_operator_trace_patched = True
    return all_reduce


def _patch_all_reduce_binding(module: object) -> None:
    if not hasattr(module, "tensor_model_parallel_all_reduce"):
        raise AttributeError(
            f"{module.__name__} lacks tensor_model_parallel_all_reduce")
    module.tensor_model_parallel_all_reduce = _all_reduce_wrapper(
        module.tensor_model_parallel_all_reduce)


def _patch_group_coordinator(module: object) -> None:
    coordinator = module.GroupCoordinator
    for method_name, position in (("dispatch", "dispatch"), ("combine", "combine")):
        if not hasattr(coordinator, method_name):
            continue
        original = getattr(coordinator, method_name)
        if hasattr(original, "_midlayer_operator_trace_patched"):
            continue

        def wrapper(self: object, hidden_states: Any, *args: object,
                    _original: Any = original, _position: str = position,
                    **kwargs: object) -> Any:
            import torch

            if _STATE.step_id is not None and isinstance(hidden_states, torch.Tensor):
                shape = [int(value) for value in hidden_states.shape]
                _emit_operator("all_to_all", _position, -1, {
                    "batch": 1,
                    "seq_len": shape[0] if shape else 1,
                    "msg_bytes": int(
                        hidden_states.numel() * hidden_states.element_size()),
                    "input_shape": shape,
                })
            return _original(self, hidden_states, *args, **kwargs)

        wrapper._midlayer_operator_trace_patched = True
        setattr(coordinator, method_name, wrapper)


def _record_topk_ids(topk_ids: object) -> None:
    if _STATE.step_id is None:
        return
    if _STATE.pending_topk_ids is not None:
        raise RuntimeError("two MoE routing calls occurred inside one MoE layer")
    _STATE.pending_topk_ids = topk_ids


def _patch_v011_moe_layer(module: object) -> None:
    fused_moe = module.FusedMoE
    original = fused_moe.__dict__["select_experts"]
    if not isinstance(original, staticmethod):
        raise TypeError("vLLM 0.11.0 FusedMoE.select_experts must be a staticmethod")
    function = original.__func__
    if hasattr(function, "_midlayer_operator_trace_patched"):
        return

    def select_experts(*args: object, **kwargs: object) -> object:
        result = function(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("FusedMoE.select_experts returned an unexpected value")
        _record_topk_ids(result[1])
        return result

    select_experts._midlayer_operator_trace_patched = True
    fused_moe.select_experts = staticmethod(select_experts)


def _patch_v030_moe_router(module: object) -> None:
    router = module.FusedMoERouter
    original = router.select_experts
    if hasattr(original, "_midlayer_operator_trace_patched"):
        return

    def select_experts(self: object, *args: object, **kwargs: object) -> object:
        result = original(self, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("FusedMoERouter.select_experts returned an unexpected value")
        _record_topk_ids(result[1])
        return result

    select_experts._midlayer_operator_trace_patched = True
    router.select_experts = select_experts


def _patch_module(fullname: str, module: object) -> None:
    if fullname == _SCHEDULER_MODULE:
        _patch_scheduler(module.Scheduler)
    elif fullname in (_RUNNER_V1_MODULE, _RUNNER_V2_MODULE):
        _patch_runner(module.GPUModelRunner)
    elif fullname in (_COMM_MODULE, _LINEAR_MODULE):
        _patch_all_reduce_binding(module)
    elif fullname == _PARALLEL_STATE_MODULE:
        _patch_group_coordinator(module)
    elif fullname in (_MOE_LAYER_V011_MODULE, _MOE_ROUTER_V030_MODULE):
        if fullname != _MOE_ROUTING_MODULE_BY_RELEASE[_installed_vllm_release()]:
            return
        if fullname == _MOE_LAYER_V011_MODULE:
            _patch_v011_moe_layer(module)
        else:
            _patch_v030_moe_router(module)
    else:
        raise AssertionError(f"unexpected operator trace target: {fullname}")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(fullname, spec.loader)
        return spec


class _Loader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: Any) -> None:
        self.fullname = fullname
        self.wrapped = wrapped

    def create_module(self, spec: object) -> object:
        return self.wrapped.create_module(spec)

    def exec_module(self, module: object) -> None:
        self.wrapped.exec_module(module)
        _patch_module(self.fullname, module)


def _installed_vllm_release() -> str:
    from importlib.metadata import version

    return version("vllm").split("+", 1)[0]


def install_from_environment() -> bool:
    """Install the capture when both raw output paths are explicitly supplied."""

    global _INSTALLED
    present = [
        name for name in (SCHEDULER_SIDECAR_ENV, OPERATOR_TRACE_ENV)
        if name in os.environ
    ]
    if not present:
        return False
    if len(present) != 2:
        raise RuntimeError(
            "vLLM operator capture requires both "
            f"{SCHEDULER_SIDECAR_ENV} and {OPERATOR_TRACE_ENV}")
    _required_path(SCHEDULER_SIDECAR_ENV)
    _required_path(OPERATOR_TRACE_ENV)
    release = _installed_vllm_release()
    if release not in _MOE_ROUTING_MODULE_BY_RELEASE:
        raise RuntimeError(f"vLLM operator capture does not support vLLM {release}")
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        for fullname in _TARGET_MODULES:
            if fullname in sys.modules:
                _patch_module(fullname, sys.modules[fullname])
        sys.meta_path.insert(0, _Finder())
        _INSTALLED = True
    return True
