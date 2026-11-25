"""CPU-only observer installed as sitecustomize in a bounded diagnostic.

Only the uncompiled runner entry points are wrapped. Detailed observation
uses CPython profiling after server warmup; compiled model function objects
and code stay intact. Numeric arrays avoid retaining a Python object graph
per sample. These timings are diagnostic evidence, never Forward prices.
"""
from array import array
import atexit
import functools
import gc
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

FIELDS = ("label", "start_ns", "end_ns", "thread_cpu_ns", "step", "batch",
          "tokens", "qmax", "thread_id", "object_id")
NAMES = frozenset(("gather_batch_req_state", "prepare_inputs", "prepare_attn",
    "_compute_logits_and_sample", "run_fullgraph", "run_pw_graph", "replay",
    "capture_begin", "capture_end", "instantiate", "__enter__", "__exit__"))
SUFFIXES = ("/vllm/v1/attention/backends/flash_attn.py",
            "/vllm/vllm_flash_attn/flash_attn_interface.py")
_rows = array("q")
_labels = {0: "GPUModelRunner.execute_model", 1: "GPUModelRunner.sample_tokens",
           10: "gc.generation0", 11: "gc.generation1", 12: "gc.generation2"}
_identities = {}
_frames = {}
_gc_starts = {}
_mode = "disabled"
_step = -1
_shape = (0, 0, 0)
_directory = None
_metadata = {}
_partial_gc_stops = 0


def _append(label, start, stop, cpu, shape, step, identity=0):
    _rows.extend((label, start, stop, cpu, step, *shape,
                  threading.get_ident(), identity))


def _selected(code):
    path = code.co_filename.replace("\\", "/")
    return (path.endswith(SUFFIXES) or
            code.co_name in NAMES and ("/vllm/" in path or "/torch/cuda/graphs.py" in path) or
            code.co_name == "__call__" and path.endswith("/vllm/compilation/cuda_graph.py"))


def _profile(frame, event, argument):
    del argument
    if event == "call" and _selected(frame.f_code):
        code = frame.f_code
        if code not in _identities:
            label = 100 + len(_identities)
            _identities[code] = label
            _labels[label] = f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}"
        key = id(frame)
        if key in _frames:
            raise RuntimeError("observer frame identity overlapped")
        _frames[key] = (time.perf_counter_ns(), time.thread_time_ns(),
                        _shape, _step, _identities[code])
    elif event == "return" and id(frame) in _frames:
        start, cpu, shape, step, label = _frames.pop(id(frame))
        _append(label, start, time.perf_counter_ns(),
                time.thread_time_ns() - cpu, shape, step)


def _gc_event(phase, info):
    global _partial_gc_stops
    key = threading.get_ident(), info["generation"]
    if phase == "start":
        if _mode in ("reference", "detailed"):
            _gc_starts[key] = (time.perf_counter_ns(), time.thread_time_ns(), _shape, _step)
    elif phase == "stop":
        if key not in _gc_starts:
            # Enabling observation during an existing collection gives no
            # start boundary. Count the missing boundary, never invent time.
            if _mode in ("reference", "detailed"):
                _partial_gc_stops += 1
            return
        start, cpu, shape, step = _gc_starts.pop(key)
        _append(10 + info["generation"], start, time.perf_counter_ns(),
                time.thread_time_ns() - cpu, shape, step)
    else:
        raise ValueError("unknown CPython GC callback phase")


def _dump():
    if _directory is None:
        return
    # Main worker is idle when the controller requests a dump after all responses.
    target = _directory / str(os.getpid())
    target.mkdir(parents=True, exist_ok=True)
    (target / "events.bin").write_bytes(_rows.tobytes())
    metadata = target / "metadata.json"
    temporary = target / "metadata.tmp"
    temporary.write_text(json.dumps({**_metadata,
        "fields": FIELDS, "itemsize": _rows.itemsize, "byteorder": sys.byteorder,
        "rows": len(_rows) // len(FIELDS), "labels": _labels,
        "pid": os.getpid(), "unclosed_profile_frames": len(_frames),
        "partial_gc_stops": _partial_gc_stops,
        "unclosed_gc_starts": len(_gc_starts)}, indent=2) + "\n")
    temporary.replace(metadata)


def _watch(flag):
    global _mode
    while True:
        mode = flag.read_text().strip()
        if mode not in ("disabled", "reference", "detailed", "dump", "stop"):
            raise ValueError("unknown diagnostic observer mode")
        if mode == "dump":
            _mode = "disabled"
            _dump()
        elif mode == "stop":
            _mode = "disabled"
            _dump()
            return
        else:
            _mode = mode
        time.sleep(0.1)


def _wrap(function, label):
    @functools.wraps(function)
    def observed(*args, **kwargs):
        global _step, _shape
        mode = _mode
        if mode not in ("reference", "detailed"):
            return function(*args, **kwargs)
        if label == 0:
            output = args[1] if len(args) > 1 else kwargs["scheduler_output"]
            counts = output.num_scheduled_tokens
            _step += 1
            _shape = (len(counts), sum(counts.values()), max(counts.values()) if counts else 0)
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        prior = sys.getprofile()
        if mode == "detailed":
            if prior is not None:
                raise RuntimeError("an unrelated CPython profiler is already active")
            sys.setprofile(_profile)
        try:
            return function(*args, **kwargs)
        finally:
            if mode == "detailed":
                sys.setprofile(prior)
            _append(label, start, time.perf_counter_ns(),
                    time.thread_time_ns() - cpu, _shape, _step, id(args[0]))
    return observed


def _patch_runner(module):
    import torch
    import vllm
    if vllm.__version__.split("+", 1)[0] != "0.30.0":
        raise ValueError("observer requires installed vLLM0.30")
    GPUModelRunner = module.GPUModelRunner
    methods = (GPUModelRunner.execute_model, GPUModelRunner.sample_tokens)
    _metadata.update({"runner_loaded": True, "torch": torch.__version__,
        "vllm": vllm.__version__, "source_method_sha256": {
            method.__name__: hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
            for method in methods}})
    GPUModelRunner.execute_model = _wrap(methods[0], 0)
    GPUModelRunner.sample_tokens = _wrap(methods[1], 1)


class _RunnerLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_runner(module)


class _RunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm.v1.worker.gpu.model_runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("actual V2 runner source could not be found")
        spec.loader = _RunnerLoader(spec.loader)
        return spec


def install():
    global _directory, _metadata
    # Do not force a GPU-model import into API or compiler subprocesses.
    # Patch the real runner only when the application's normal import loads it.
    directory = Path(os.environ["MIDLAYER_BOUNDARY_DIRECTORY"])
    flag = Path(os.environ["MIDLAYER_BOUNDARY_FLAG"])
    initial_mode = flag.read_text().strip()
    if not directory.is_dir() or initial_mode not in ("disabled", "reference", "detailed", "dump", "stop"):
        raise ValueError("observer requires a prepared campaign and known mode")
    _metadata = {"kind": "host_runner_boundary_diagnostic", "runner_loaded": False,
        "torch": None, "vllm": None, "source_method_sha256": None,
        "clock": {"name": "time.perf_counter_ns",
                  "implementation": time.get_clock_info("perf_counter").implementation},
        "initial_observer_mode": initial_mode,
        "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiled_model_function_objects_modified": False,
        "compiled_guard_equivalence_certified": False,
        "limitations": ["detailed CPython profiling perturbs execution; no price or canonical truth",
                        "numeric observer buffers and a polling thread are additional Host work"]}
    if "vllm.v1.worker.gpu.model_runner" in sys.modules:
        _patch_runner(sys.modules["vllm.v1.worker.gpu.model_runner"])
    else:
        sys.meta_path.insert(0, _RunnerFinder())
    _directory = directory
    gc.callbacks.append(_gc_event)
    atexit.register(_dump)
    threading.Thread(target=_watch, args=(flag,), daemon=True).start()


if __name__ == "sitecustomize":
    if not any("torch._inductor.compile_worker" in argument or
               "torch/_inductor/compile_worker" in argument for argument in sys.orig_argv):
        install()
