"""Native CUDA Host-API observer; leaves compiled model callables intact."""
import atexit
import ctypes
import functools
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

_mode = 'disabled'
_step = -1
_shape = (0, 0, 0)
_library = None
_directory = None
_metadata = {}
_dump_lock = threading.Lock()
_dumped = False


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f'native API boundary {operation} failed with status{status}')


def _dump():
    global _dumped
    with _dump_lock:
        if _library is None or _dumped:
            return
        _dump_native()


def _dump_native():
    global _dumped
    if _library is None:
        return
    status = _library.midlayer_boundary_dump(os.fsencode(_directory))
    native = json.loads((_directory / 'native_metadata.json').read_text())
    payload = {**_metadata, **native, 'partial_gc_stops': 0, 'unclosed_gc_starts': 0,
        'clock': {'name': 'clock_gettime(CLOCK_MONOTONIC)',
            'implementation': 'clock_gettime(CLOCK_MONOTONIC)'},
        'thread_id_source': 'Linux gettid', 'native_dump_status': status}
    temporary = _directory / 'metadata.tmp'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(_directory / 'metadata.json')
    _dumped = True
    _check(status, 'dump')


def _watch(flag):
    global _mode
    while True:
        mode = flag.read_text().strip()
        if mode not in ('disabled', 'reference', 'detailed', 'dump', 'stop'):
            raise ValueError('unknown native API observer mode')
        if mode in ('dump', 'stop'):
            _mode = 'disabled'
            _dump()
            if mode == 'stop':
                return
        else:
            _mode = mode
        time.sleep(0.1)


def _wrap(function, phase):
    @functools.wraps(function)
    def observed(*args, **kwargs):
        global _step, _shape
        if _mode not in ('reference', 'detailed'):
            return function(*args, **kwargs)
        if phase == 0:
            output = args[1] if len(args) > 1 else kwargs['scheduler_output']
            counts = output.num_scheduled_tokens
            _step += 1
            _shape = (len(counts), sum(counts.values()), max(counts.values()) if counts else 0)
        _check(_library.midlayer_boundary_begin(phase, _step, *_shape), 'begin')
        try:
            return function(*args, **kwargs)
        finally:
            _check(_library.midlayer_boundary_end(), 'end')
    return observed


def _patch_runner(module):
    global _library, _directory
    import torch
    import vllm
    if vllm.__version__.split('+', 1)[0] != '0.30.0' or sys.byteorder != 'little':
        raise ValueError('native API boundary requires its vLLM0.30/little-endian domain')
    root = Path(os.environ['MIDLAYER_BOUNDARY_DIRECTORY'])
    stage = root.parent.name
    if stage not in ('reference', 'detailed'):
        raise ValueError('native observer stage must be explicit reference/detailed')
    _directory = root / str(os.getpid())
    _directory.mkdir(parents=True, exist_ok=True)
    library_path = Path(os.environ['MIDLAYER_API_BOUNDARY_LIBRARY'])
    _library = ctypes.CDLL(str(library_path))
    _library.midlayer_boundary_provider.argtypes = ()
    _library.midlayer_boundary_provider.restype = ctypes.c_char_p
    _library.midlayer_boundary_init.argtypes = (ctypes.c_size_t, ctypes.c_int)
    _library.midlayer_boundary_init.restype = ctypes.c_int
    _library.midlayer_boundary_begin.argtypes = (ctypes.c_int64,) * 5
    _library.midlayer_boundary_begin.restype = ctypes.c_int
    _library.midlayer_boundary_end.argtypes = ()
    _library.midlayer_boundary_end.restype = ctypes.c_int
    _library.midlayer_boundary_dump.argtypes = (ctypes.c_char_p,)
    _library.midlayer_boundary_dump.restype = ctypes.c_int
    status = _library.midlayer_boundary_init(1000000, int(stage == 'detailed'))
    _metadata['native_init_status'] = status
    _check(status, 'initialize')
    provider = _library.midlayer_boundary_provider()
    _metadata['active_cupti_library'] = None if provider is None else os.fsdecode(provider)
    _metadata['active_cupti_library_sha256'] = (None if provider is None else
        hashlib.sha256(Path(os.fsdecode(provider)).read_bytes()).hexdigest())
    runner = module.GPUModelRunner
    methods = runner.execute_model, runner.sample_tokens
    _metadata.update({'runner_loaded': True, 'torch': torch.__version__, 'vllm': vllm.__version__,
        'source_method_sha256': {method.__name__: hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
            for method in methods}, 'native_library_sha256': hashlib.sha256(library_path.read_bytes()).hexdigest(),
        'api_callbacks_enabled_during_warmup': stage == 'detailed'})
    runner.execute_model = _wrap(methods[0], 0)
    runner.sample_tokens = _wrap(methods[1], 1)


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
        if fullname != 'vllm.v1.worker.gpu.model_runner':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError('actual V2 runner source could not be found')
        spec.loader = _RunnerLoader(spec.loader)
        return spec


def install():
    flag = Path(os.environ['MIDLAYER_BOUNDARY_FLAG'])
    if flag.read_text().strip() not in ('disabled', 'reference', 'detailed', 'dump', 'stop'):
        raise ValueError('unknown initial native API observer mode')
    _metadata.update({'kind': 'host_runner_boundary_diagnostic', 'runner_loaded': False,
        'initial_observer_mode': flag.read_text().strip(),
        'hook_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'compiled_model_function_objects_modified': False,
        'compiled_guard_equivalence_certified': False,
        'observer': 'native_cupti_host_api_callbacks_and_outer_markers',
        'limitations': ['CUPTI callbacks and native markers still perturb execution; no Forward prices.',
            'CUDA Graph launch callbacks do not enumerate kernels captured inside a graph.',
            'Kernel symbol availability is recorded explicitly; no-symbol is not a known identity.',
            'Reference shares outer markers; detailed additionally records CUDA Host APIs from warmup.']})
    if 'vllm.v1.worker.gpu.model_runner' in sys.modules:
        _patch_runner(sys.modules['vllm.v1.worker.gpu.model_runner'])
    else:
        sys.meta_path.insert(0, _RunnerFinder())
    atexit.register(_dump)
    threading.Thread(target=_watch, args=(flag,), daemon=True).start()


if __name__ == 'sitecustomize':
    if not any('torch._inductor.compile_worker' in argument or 'torch/_inductor/compile_worker' in argument
            for argument in sys.orig_argv):
        install()
