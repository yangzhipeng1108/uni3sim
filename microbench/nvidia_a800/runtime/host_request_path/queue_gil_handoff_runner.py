"""Independent CPU queue/GIL handoff controls; no Serving, Torch, or GPU work.

Native controls use the documented CDLL (releases GIL) / PyDLL (holds GIL)
calling conventions. Fixed native-work/burst/Python-work grids and two switch
intervals classify handoff sensitivity; no measured latency becomes a price.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time

C_SOURCE = r'''
#include <stdint.h>
#include <time.h>
static uint64_t mono(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return ((uint64_t)t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
uint64_t busy_ns(uint64_t budget) {
    uint64_t start = mono(), end;
    do { end = mono(); } while (end - start < budget);
    return end - start;
}
'''
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(exist_ok=False)
source = output / 'busy.c'
source.write_text(C_SOURCE)
library = output / 'busy.so'
compiler = ['gcc', '-O2', '-fPIC', '-shared', str(source), '-o', str(library)]
subprocess.run(compiler, check=True)
functions = {}
for release in (False, True):
    lib = ctypes.CDLL(str(library)) if release else ctypes.PyDLL(str(library))
    function = lib.busy_ns
    function.argtypes, function.restype = (ctypes.c_uint64,), ctypes.c_uint64
    functions[release] = function
inbox, outbox = queue.Queue(), queue.Queue()
ready = threading.Event()


def consume():
    while True:
        ready.set()
        item = inbox.get()
        if item is None:
            return
        index, enqueued = item
        resumed = time.perf_counter_ns()
        outbox.put((index, resumed - enqueued))


thread = threading.Thread(target=consume)
thread.start()
original_interval = sys.getswitchinterval()
rows, index = [], 0
started = time.perf_counter()
try:
    for interval in (0.005, 0.001):
        sys.setswitchinterval(interval)
        for release in (False, True):
            native = functions[release]
            for native_ns in (1000, 5000, 20000):
                for python_work in (0, 64):
                    for burst_calls in (8, 128, 512):
                        samples = []
                        native_min = None
                        for sample in range(40):
                            ready.wait()
                            ready.clear()
                            enqueued = time.perf_counter_ns()
                            inbox.put((index, enqueued))
                            checksum = 0
                            for call in range(burst_calls):
                                for value in range(python_work):
                                    checksum ^= value
                                elapsed = native(native_ns)
                                if elapsed < native_ns:
                                    raise ValueError('native work did not meet its declared budget')
                                native_min = elapsed if native_min is None else min(native_min, elapsed)
                            returned_index, latency = outbox.get(timeout=10)
                            if returned_index != index or latency < 0 or checksum != 0:
                                raise ValueError('queue handoff identity/clock/checksum mismatch')
                            index += 1
                            if sample >= 8:
                                samples.append(latency)
                        ordered = sorted(samples)
                        rows.append({'switch_interval_s': interval, 'native_releases_gil': release,
                            'native_work_ns': native_ns, 'python_xor_count_per_call': python_work,
                            'burst_calls': burst_calls, 'sample_count': len(samples),
                            'native_work_min_ns': native_min, 'enqueue_to_consumer_python_ns': samples,
                            'median_ns': statistics.median(samples), 'p95_ns': ordered[30], 'max_ns': max(samples)})
finally:
    sys.setswitchinterval(original_interval)
    inbox.put(None)
    thread.join(timeout=10)
if thread.is_alive() or index != 72 * 40:
    raise RuntimeError('queue consumer did not finish the full independent grid')
meta = {'scope': 'independent CPU queue/GIL handoff controls; no model, Serving requests, Torch, GPU, or Modal',
    'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
    'host_cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
        if line.startswith('model name')),
    'cpu_affinity_count': len(os.sched_getaffinity(0)), 'original_switch_interval_s': original_interval,
    'restored_switch_interval_s': sys.getswitchinterval(), 'wall_s': time.perf_counter() - started,
    'compiler': compiler, 'compiler_version': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'native_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'native_library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
    'ctypes_source_path': ctypes.__file__, 'ctypes_source_sha256': hashlib.sha256(Path(ctypes.__file__).read_bytes()).hexdigest(),
    'cells': len(rows), 'validated_handoffs_including_warmup': index,
    'adoption': 'none; unknown target GIL hold/release sites are not inferred from these controls',
    'new_modal_executions': 0, 'gpu_hours': 0, 'rows': rows}
(output / 'raw.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({key: value for key, value in meta.items() if key != 'rows'}))
