"""PEP669 local-code Host observation; no function replacement or global profile.

Standalone sitecustomize for one bounded diagnostic. Recorded call spans are
evaluation evidence, never Forward services. Actual code identities and source
hashes bind what was observed; missing source coverage remains a failure.
"""
from array import array
import atexit
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

FIELDS = ('label', 'start_ns', 'end_ns', 'thread_cpu_ns', 'step', 'batch',
          'tokens', 'qmax', 'thread_id', 'object_id')
TARGETS = {
    'vllm.v1.worker.gpu.model_runner': (
        ('GPUModelRunner.execute_model', 0), ('GPUModelRunner.sample_tokens', 1)),
    'vllm.model_executor.layers.fused_moe.modular_kernel': (
        ('FusedMoEKernel.apply', None), ('FusedMoEKernelModularImpl.apply', None),
        ('FusedMoEKernelModularImpl._prepare', None),
        ('FusedMoEKernelModularImpl._allocate_buffers', None),
        ('FusedMoEKernelModularImpl._fused_experts', None),
        ('FusedMoEKernelModularImpl._finalize', None)),
    'vllm.model_executor.layers.fused_moe.experts.triton_moe': (('TritonExperts.apply', None),),
    'vllm.model_executor.layers.fused_moe.fused_moe': (('invoke_fused_moe_triton_kernel', None),),
    'vllm.compilation.cuda_graph': (('CUDAGraphWrapper.__call__', None),),
}


class Recorder:
    def __init__(self, api):
        self.api, self.tool = api, 2
        if api.get_tool(self.tool) is not None:
            raise RuntimeError('monitoring tool2 is already owned')
        api.use_tool_id(self.tool, 'midlayer-local-code-observer')
        self.rows, self.labels, self.functions = array('q'), {}, []
        self.codes, self.outer, self.stacks, self.step_epochs = {}, {}, {}, []
        self.mode, self.epoch, self.step, self.shape = 'disabled', -1, -1, (0, 0, 0)
        self.errors, self.off_scope_calls = 0, 0
        self.global_profile_seen = False
        self.references = []
        self.rank = None
        api.register_callback(self.tool, api.events.PY_START, self.start)
        api.register_callback(self.tool, api.events.PY_RETURN, self.stop)

    def register(self, function, name, outer):
        original = inspect.unwrap(function)
        code = original.__code__
        if code in self.codes:
            raise ValueError('selected source functions share one code object')
        label = outer if outer is not None else 100 + len(self.functions)
        if label in self.labels:
            raise ValueError('duplicate source label')
        self.codes[code], self.labels[label] = label, name
        if outer is not None:
            self.outer[code] = outer
        self.functions.append({'name': name, 'label': label,
            'source_sha256': hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
            'code_sha256': hashlib.sha256(code.co_code).hexdigest(),
            'function_id': id(function), 'unwrapped_function_id': id(original),
            'code_id': id(code), 'filename': code.co_filename})
        self.references.append((function, original, code, code.co_code))
        self._enable(code)

    def _enable(self, code):
        enabled = self.mode == 'selective' or self.mode == 'reference' and code in self.outer
        # PY_UNWIND is not a local event in CPython3.12. Exceptions leave an
        # unclosed stack and fail capture qualification; no global hook is used.
        events = (self.api.events.PY_START | self.api.events.PY_RETURN) if enabled else 0
        self.api.set_local_events(self.tool, code, events)

    def set_mode(self, mode, epoch):
        if mode not in ('disabled', 'reference', 'selective') or type(epoch) is not int:
            raise ValueError('unknown selective-monitor mode/epoch')
        if any(self.stacks.values()):
            self.errors += 1
            raise RuntimeError('cannot change code-event mode across an active source call')
        self.mode, self.epoch = mode, epoch
        for code in self.codes:
            self._enable(code)

    def start(self, code, offset):
        del offset
        if code not in self.codes:
            self.errors += 1
            raise RuntimeError('event outside registered local code')
        frame = sys._getframe(1)
        if frame.f_code is not code:
            self.errors += 1
            raise RuntimeError('local event has no matching source frame')
        self.global_profile_seen |= sys.getprofile() is not None
        thread = threading.get_ident()
        if thread not in self.stacks:
            self.stacks[thread] = []
        stack = self.stacks[thread]
        if code not in self.outer and not any(entry[0] in self.outer for entry in stack):
            self.off_scope_calls += 1
            stack.append((code, None, None, None, None))
            return
        if code in self.outer and self.outer[code] == 0:
            rank = frame.f_locals['self'].parallel_config.rank
            if self.rank is not None and self.rank != rank:
                self.errors += 1
                raise RuntimeError('one recorder observed different worker ranks')
            self.rank = rank
            output = frame.f_locals['scheduler_output']
            counts = output.num_scheduled_tokens
            self.step += 1
            self.shape = (len(counts), sum(counts.values()), max(counts.values()) if counts else 0)
            self.step_epochs.append((self.step, self.epoch))
        stack.append((code, time.perf_counter_ns(), time.thread_time_ns(), self.step, self.shape))

    def stop(self, code, offset, value):
        del offset, value
        thread = threading.get_ident()
        if thread not in self.stacks or not self.stacks[thread] or self.stacks[thread][-1][0] is not code:
            self.errors += 1
            raise RuntimeError('unmatched local source return')
        _, start, cpu, step, shape = self.stacks[thread].pop()
        if start is None:
            return
        self.rows.extend((self.codes[code], start, time.perf_counter_ns(),
            time.thread_time_ns() - cpu, step, *shape, thread, id(code)))

    def dump(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        events = directory / 'events.tmp'
        events.write_bytes(self.rows.tobytes())
        events.replace(directory / 'events.bin')
        metadata = {'kind': 'pep669_local_code_host_observer', 'fields': FIELDS,
            'itemsize': self.rows.itemsize, 'byteorder': sys.byteorder,
            'rows': len(self.rows) // len(FIELDS), 'labels': self.labels,
            'registered_functions': self.functions, 'step_epochs': self.step_epochs,
            'pid': os.getpid(), 'rank': self.rank, 'mode': self.mode, 'errors': self.errors,
            'off_scope_calls': self.off_scope_calls,
            'unclosed_profile_frames': sum(len(v) for v in self.stacks.values()),
            'function_objects_replaced': False, 'global_profile_installed': self.global_profile_seen,
            'registered_function_code_unchanged': all(inspect.unwrap(function) is original
                and original.__code__ is code and code.co_code == bytecode
                for function, original, code, bytecode in self.references),
            'exceptions_are_not_timed': True,
            'object_id_role': 'actual Python code object identity; not native batch/future identity',
            'clock': {name: {'implementation': time.get_clock_info(name).implementation,
                'resolution_s': time.get_clock_info(name).resolution} for name in ('perf_counter', 'thread_time')},
            'hook_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        metadata['software'] = {name: sys.modules[name].__version__ if name in sys.modules else None
            for name in ('torch', 'vllm')}
        target = directory / 'metadata.tmp'
        target.write_text(json.dumps(metadata, indent=2) + '\n')
        target.replace(directory / 'metadata.json')

    def close(self):
        self.set_mode('disabled', self.epoch)
        for event in (self.api.events.PY_START, self.api.events.PY_RETURN):
            self.api.register_callback(self.tool, event, None)
        self.api.free_tool_id(self.tool)


_recorder = None


def register_module(module):
    for qualified, outer in TARGETS[module.__name__]:
        value = module
        for part in qualified.split('.'):
            value = getattr(value, part)
        _recorder.register(value, module.__name__ + '.' + qualified, outer)


class SourceLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        register_module(module)


class SourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError('selected source module is missing')
        spec.loader = SourceLoader(spec.loader)
        return spec


def install():
    global _recorder
    if not hasattr(sys, 'monitoring') or sys.getprofile() is not None:
        raise RuntimeError('requires Python3.12 monitoring and no unrelated global profiler')
    _recorder = Recorder(sys.monitoring)
    directory = Path(os.environ['MIDLAYER_SELECTIVE_DIRECTORY']) / str(os.getpid())
    flag = Path(os.environ['MIDLAYER_SELECTIVE_FLAG'])
    for name in TARGETS:
        if name in sys.modules:
            register_module(sys.modules[name])
    sys.meta_path.insert(0, SourceFinder())

    def watch():
        previous = ''
        try:
            while True:
                text = flag.read_text()
                if text != previous:
                    state = json.loads(text)
                    if set(state) != {'mode', 'epoch'}:
                        raise ValueError('unknown monitor-control fields')
                    if state['mode'] == 'stop':
                        _recorder.set_mode('disabled', state['epoch'])
                        _recorder.dump(directory)
                        return
                    _recorder.set_mode(state['mode'], state['epoch'])
                    if state['mode'] == 'disabled' and _recorder.rows:
                        _recorder.dump(directory)
                    previous = text
                time.sleep(.02)
        except BaseException:
            _recorder.errors += 1
            _recorder.dump(directory)
            raise

    atexit.register(lambda: _recorder.dump(directory))
    threading.Thread(target=watch, daemon=True).start()


if __name__ == 'sitecustomize':
    if not any('torch._inductor.compile_worker' in argument or
               'torch/_inductor/compile_worker' in argument for argument in sys.orig_argv):
        install()
