"""Fail-closed vLLM V1 Scheduler and operator capture for serving raw evidence."""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, TextIO

from dataset_pipeline.serving_dataset.json_shape import layer_index as _layer_index


_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_SCHEDULER_PATH_ENV = "CUDA_SIDECAR_PATH"
_OPERATOR_PATH_ENV = "CUDA_OPERATOR_TRACE_PATH"
_RUNNER_STEP_PATH_ENV = "CUDA_RUNNER_STEP_PATH"
_SPAWN_WRAPPER_ENV = "MIDLAYER_C500_CAPTURE_SPAWN_WRAPPER"
_SPAWN_PYTHON_ENV = "MIDLAYER_C500_CAPTURE_SPAWN_PYTHON"
_SPAWN_PYTHONPATH_ENV = "MIDLAYER_C500_CAPTURE_SPAWN_PYTHONPATH"
_STATE = threading.local()
_INSTALLED = False
_WRITER_LOCK = threading.Lock()
_WRITERS: dict[Path, TextIO] = {}


def _required_path(name: str) -> Path:
    if name not in os.environ:
        raise RuntimeError(f"{name} is required for vLLM capture")
    path = Path(os.environ[name])
    if not path.parent.is_dir():
        raise FileNotFoundError(f"capture parent directory is absent: {path.parent}")
    return path


def _append_jsonl(path: Path, record: dict[str, Any], *, flush: bool) -> None:
    serialized = json.dumps(record, sort_keys=True) + "\n"
    with _WRITER_LOCK:
        if path in _WRITERS:
            stream = _WRITERS[path]
        else:
            stream = path.open("a", encoding="utf-8", buffering=1024 * 1024)
            _WRITERS[path] = stream
        stream.write(serialized)
        if flush:
            stream.flush()


def _close_writers() -> None:
    with _WRITER_LOCK:
        for stream in _WRITERS.values():
            stream.close()
        _WRITERS.clear()


atexit.register(_close_writers)


def _capture_enabled() -> bool:
    return _SCHEDULER_PATH_ENV in os.environ and _OPERATOR_PATH_ENV in os.environ


def _configure_spawn_bootstrap() -> None:
    """Keep V1 EngineCore workers outside the parent mcTracer executable."""

    configured = tuple(
        name for name in (
            _SPAWN_WRAPPER_ENV,
            _SPAWN_PYTHON_ENV,
            _SPAWN_PYTHONPATH_ENV,
        ) if name in os.environ
    )
    if not configured:
        return
    if len(configured) != 3:
        raise RuntimeError(
            "C500 capture spawn bootstrap is incomplete: "
            + ", ".join(configured)
        )
    wrapper = Path(os.environ[_SPAWN_WRAPPER_ENV]).resolve()
    python = Path(os.environ[_SPAWN_PYTHON_ENV]).resolve()
    pythonpath = os.environ[_SPAWN_PYTHONPATH_ENV]
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise RuntimeError("C500 capture spawn wrapper is not executable")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("C500 capture spawn Python is not executable")
    if not pythonpath:
        raise ValueError("C500 capture spawn Python path is empty")
    multiprocessing.set_executable(str(wrapper))


def _rank() -> int:
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():
        return int(distributed.get_rank())
    return 0


def _operator_path() -> Path:
    base = _required_path(_OPERATOR_PATH_ENV)
    return Path(f"{base}.rank{_rank()}")


def _classify(module: object) -> tuple[str, str] | None:
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
    return None




def _first_tensor(args: tuple[object, ...], kwargs: dict[str, object]) -> object | None:
    import torch

    for value in args:
        if isinstance(value, torch.Tensor):
            return value
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return value
    return None


def _hook(module: object, module_name: str, op_type: str, position: str):
    def record(_module: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        tensor = _first_tensor(args, kwargs)
        if tensor is None:
            return
        if not hasattr(_STATE, "step_id"):
            raise RuntimeError("operator hook ran outside execute_model")
        shape = [int(value) for value in tensor.shape]
        tokens = shape[0] if shape else 1
        attrs: dict[str, object] = {"input_shape": shape}
        if op_type == "linear":
            if not hasattr(module, "output_size_per_partition"):
                raise AttributeError(
                    f"linear capture module lacks output_size_per_partition: {type(module).__name__}"
                )
            output_size = module.output_size_per_partition
            if isinstance(output_size, bool) or not isinstance(output_size, int) or output_size < 1:
                raise TypeError("linear output_size_per_partition must be a positive integer")
            attrs.update({"B": 1, "M": tokens, "N": output_size, "K": shape[-1]})
        else:
            attrs.update({"batch": 1, "seq_len": tokens})
        record_data = {
            "step_id": _STATE.step_id,
            "op_idx": _STATE.op_idx,
            "op_type": op_type,
            "position": position,
            "layer_idx": _layer_index(module_name),
            "attrs": attrs,
            "ts_us": time.time_ns() / 1000.0,
        }
        _append_jsonl(_operator_path(), record_data, flush=False)
        _STATE.op_idx += 1

    return record


def _install_module_hooks(model: object) -> None:
    if not hasattr(model, "named_modules"):
        raise AttributeError("GPUModelRunner model lacks named_modules")
    for module_name, module in model.named_modules():
        classified = _classify(module)
        if classified is None:
            continue
        op_type, position = classified
        if not hasattr(module, "register_forward_pre_hook"):
            raise AttributeError(f"captured module lacks forward hook API: {module_name}")
        module.register_forward_pre_hook(_hook(module, module_name, op_type, position), with_kwargs=True)


def _patch_runner(runner_cls: type[object]) -> None:
    original = runner_cls.execute_model
    if hasattr(original, "_midlayer_capture_patched"):
        return

    def execute_model(self: object, *args: object, **kwargs: object) -> object:
        if not hasattr(_STATE, "hooks_installed"):
            if not hasattr(self, "model"):
                raise AttributeError("GPUModelRunner lacks model")
            _install_module_hooks(self.model)
            _STATE.hooks_installed = True
            _STATE.next_step_id = 0
        _STATE.step_id = _STATE.next_step_id
        _STATE.next_step_id += 1
        _STATE.op_idx = 0
        if _RUNNER_STEP_PATH_ENV not in os.environ:
            return original(self, *args, **kwargs)
        if args:
            scheduler_output = args[0]
        elif "scheduler_output" in kwargs:
            scheduler_output = kwargs["scheduler_output"]
        else:
            raise ValueError("captured execute_model call lacks Scheduler output")
        if not hasattr(scheduler_output, "num_scheduled_tokens"):
            raise AttributeError("runner Scheduler output lacks request tokens")
        if not hasattr(scheduler_output, "total_num_scheduled_tokens"):
            raise AttributeError("runner Scheduler output lacks total tokens")
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        if sum(scheduled.values()) != scheduler_output.total_num_scheduled_tokens:
            raise ValueError("runner Scheduler output token total differs")
        started_ns = time.time_ns()
        result = original(self, *args, **kwargs)
        ended_ns = time.time_ns()
        _append_jsonl(
            _required_path(_RUNNER_STEP_PATH_ENV),
            {
                "runner_step_index": _STATE.step_id,
                "start_ns": started_ns,
                "end_ns": ended_ns,
                "scheduled": scheduled,
                "total_tokens": scheduler_output.total_num_scheduled_tokens,
            },
            flush=True,
        )
        return result

    execute_model._midlayer_capture_patched = True
    runner_cls.execute_model = execute_model


def _patch_scheduler(scheduler_cls: type[object]) -> None:
    original = scheduler_cls.schedule
    if hasattr(original, "_midlayer_capture_patched"):
        return
    counter = [0]

    def schedule(self: object, *args: object, **kwargs: object) -> object:
        output = original(self, *args, **kwargs)
        if not hasattr(output, "num_scheduled_tokens"):
            raise AttributeError("Scheduler output lacks num_scheduled_tokens")
        if not hasattr(output, "total_num_scheduled_tokens"):
            raise AttributeError("Scheduler output lacks total_num_scheduled_tokens")
        if not hasattr(output, "finished_req_ids"):
            raise AttributeError("Scheduler output lacks finished_req_ids")
        scheduled = dict(output.num_scheduled_tokens)
        record = {
            "iter_id": counter[0],
            "ts_us": int(time.time_ns() // 1000),
            "scheduled": scheduled,
            "total_tokens": output.total_num_scheduled_tokens,
            "finished": sorted(output.finished_req_ids),
        }
        _append_jsonl(_required_path(_SCHEDULER_PATH_ENV), record, flush=True)
        counter[0] += 1
        return output

    schedule._midlayer_capture_patched = True
    scheduler_cls.schedule = schedule


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        if fullname not in (_SCHEDULER_MODULE, _RUNNER_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot find vLLM capture target: {fullname}")
        spec.loader = _Loader(fullname, spec.loader)
        return spec


class _Loader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: object) -> None:
        self.fullname = fullname
        self.wrapped = wrapped

    def create_module(self, spec: object) -> object:
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)
        return None

    def exec_module(self, module: object) -> None:
        self.wrapped.exec_module(module)
        if self.fullname == _SCHEDULER_MODULE:
            if not hasattr(module, "Scheduler"):
                raise AttributeError("vLLM scheduler module lacks Scheduler")
            _patch_scheduler(module.Scheduler)
            return
        if not hasattr(module, "GPUModelRunner"):
            raise AttributeError("vLLM runner module lacks GPUModelRunner")
        _patch_runner(module.GPUModelRunner)


def install_from_environment() -> bool:
    """Install capture patches when both raw output paths are explicitly supplied."""

    global _INSTALLED
    if not _capture_enabled():
        return False
    _required_path(_SCHEDULER_PATH_ENV)
    _required_path(_OPERATOR_PATH_ENV)
    if _INSTALLED:
        return True
    _configure_spawn_bootstrap()
    if _SCHEDULER_MODULE in sys.modules:
        module = sys.modules[_SCHEDULER_MODULE]
        if not hasattr(module, "Scheduler"):
            raise AttributeError("loaded vLLM scheduler module lacks Scheduler")
        _patch_scheduler(module.Scheduler)
    if _RUNNER_MODULE in sys.modules:
        module = sys.modules[_RUNNER_MODULE]
        if not hasattr(module, "GPUModelRunner"):
            raise AttributeError("loaded vLLM runner module lacks GPUModelRunner")
        _patch_runner(module.GPUModelRunner)
    sys.meta_path.insert(0, _Finder())
    _INSTALLED = True
    return True
