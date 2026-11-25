"""Serving-stack GEMM launch identity recorder (loaded in every vLLM process).

Launch facts only: kernel name, grid and block of the kernels each model
forward launches, plus the forward's runtime Graph mode and token counts.
No duration is kept.  Active only when ``MACA_LAUNCHID_DIR`` names an output
directory (the ``MACA_`` prefix is what the serving launcher forwards).
A torch profiler (CUDA activity) runs in each model-runner process while the
flag file ``$MACA_LAUNCHID_DIR/profile.on`` exists; every recorded forward is
bracketed by two tiny marker kernels so the trace splits per forward.
"""

import importlib.abc
import importlib.util
import json
import os
import sys

OUT = os.environ.get("MACA_LAUNCHID_DIR")
_STATE = {"tokens": None, "reqs": None, "prof": None, "done": False, "meta": []}


def _flag():
    return os.path.exists(os.path.join(OUT, "profile.on"))


def _finish():
    import torch

    prof = _STATE["prof"]
    torch.cuda.synchronize()
    prof.__exit__(None, None, None)
    pid = os.getpid()
    prof.export_chrome_trace(os.path.join(OUT, f"trace.{pid}.json"))
    with open(os.path.join(OUT, f"meta.{pid}.json"), "w") as handle:
        json.dump(_STATE["meta"], handle)
    open(os.path.join(OUT, f"done.{pid}"), "w").close()
    _STATE["done"] = True


def _patch(module):
    import torch
    from vllm.forward_context import get_forward_context

    runner = module.GPUModelRunner
    original_execute = runner.execute_model
    original_forward = runner._model_forward

    def execute_model(self, scheduler_output, *args, **kwargs):
        _STATE["tokens"] = scheduler_output.total_num_scheduled_tokens
        _STATE["reqs"] = sorted(scheduler_output.num_scheduled_tokens.values())
        return original_execute(self, scheduler_output, *args, **kwargs)

    def model_forward(self, *args, **kwargs):
        if _STATE["done"]:
            return original_forward(self, *args, **kwargs)
        if _STATE["prof"] is None:
            if not _flag():
                return original_forward(self, *args, **kwargs)
            _STATE["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA])
            _STATE["prof"].__enter__()
        elif not _flag():
            _finish()
            return original_forward(self, *args, **kwargs)
        context = get_forward_context()
        descriptor = context.batch_descriptor
        torch.cuda._sleep(100)
        output = original_forward(self, *args, **kwargs)
        torch.cuda._sleep(100)
        _STATE["meta"].append({
            "mode": str(context.cudagraph_runtime_mode).rsplit(".", 1)[-1],
            "padded_tokens": None if descriptor is None else descriptor.num_tokens,
            "scheduled_tokens": _STATE["tokens"],
            "per_request_tokens": _STATE["reqs"],
        })
        return output

    runner.execute_model = execute_model
    runner._model_forward = model_forward


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    TARGET = "vllm.v1.worker.gpu_model_runner"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if OUT:
    if "vllm.v1.worker.gpu_model_runner" in sys.modules:
        _patch(sys.modules["vllm.v1.worker.gpu_model_runner"])
    else:
        sys.meta_path.insert(0, _Finder())
