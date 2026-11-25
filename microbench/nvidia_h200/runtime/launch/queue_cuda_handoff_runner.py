"""Independent queue/GIL handoff during PyTorch native submission bursts.

A32-element add supplies minimal independent CUDA work. CPU fixture mode is
only local protocol validation; its timings are never hardware prices.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
if output.exists():
    raise FileExistsError(output)
cuda = args.device == 'cuda'
if cuda:
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('exactly one authorized H200 required')
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
value = torch.zeros(32, dtype=torch.float32, device=device)
eager = partial(value.add_, 1)
if cuda:
    for _ in range(20):
        eager()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value.add_(1)
    torch.cuda.synchronize()
    natives = {'graph_replay': graph.replay, 'eager_add': eager}
else:
    natives = {'cpu_fixture_add': eager}
intervals = (0.005, 0.001) if cuda else (0.005,)
bursts = (1, 8, 32, 128, 512) if cuda else (1, 8)
warmup, samples = (8, 63) if cuda else (2, 4)
inbox, outbox, ready = queue.Queue(), queue.Queue(), threading.Event()


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
    for interval in intervals:
        sys.setswitchinterval(interval)
        for mode, native in natives.items():
            for python_work in (0, 64):
                for burst in bursts:
                    value.zero_()
                    if cuda:
                        torch.cuda.synchronize()
                    timings = []
                    for sample in range(warmup + samples):
                        if cuda:
                            torch.cuda.synchronize()
                        ready.wait()
                        ready.clear()
                        begin = time.perf_counter_ns()
                        inbox.put((index, begin))
                        checksum = 0
                        for call in range(burst):
                            for number in range(python_work):
                                checksum ^= number
                            native()
                        producer_end = time.perf_counter_ns()
                        returned, latency = outbox.get(timeout=30)
                        if returned != index or latency < 0 or checksum != 0:
                            raise ValueError('handoff identity/clock/checksum mismatch')
                        index += 1
                        if sample >= warmup:
                            timings.append({'enqueue_to_consumer_python_ns': latency,
                                'producer_burst_wall_ns': producer_end - begin})
                    if cuda:
                        torch.cuda.synchronize()
                    expected = burst * (warmup + samples)
                    actual = value.cpu().tolist()
                    if actual != [float(expected)] * 32:
                        raise ValueError('independent native-work count/checksum mismatch')
                    rows.append({'switch_interval_s': interval, 'native': mode,
                        'python_xor_count_per_call': python_work, 'burst_calls': burst,
                        'warmup': warmup, 'sample_count': samples,
                        'validated_native_calls': expected, 'timings': timings})
finally:
    sys.setswitchinterval(original_interval)
    inbox.put(None)
    thread.join(timeout=30)
if thread.is_alive() or index != len(intervals) * len(natives) * 2 * len(bursts) * (warmup + samples):
    raise RuntimeError('consumer or independent handoff grid incomplete')
result = {'scope': 'independent queue/GIL native-submission mechanism; no model, Serving workload, or GPU duration pricing',
    'device': args.device, 'cpu_fixture_only_not_price': not cuda,
    'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
    'torch_version': torch.__version__, 'torch_git_version': torch.version.git_version,
    'cuda_version': torch.version.cuda,
    'gpu_name': torch.cuda.get_device_name(0) if cuda else None,
    'host_cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
        if line.startswith('model name')),
    'cpu_affinity_count': len(os.sched_getaffinity(0)),
    'original_switch_interval_s': original_interval, 'restored_switch_interval_s': sys.getswitchinterval(),
    'wall_s': time.perf_counter() - started, 'cells': len(rows), 'validated_handoffs_including_warmup': index,
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'adoption': 'none; active vLLM wrapper hold/release windows and full delivery path remain unbound',
    'rows': rows}
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: item for key, item in result.items() if key != 'rows'}))
