#!/usr/bin/env python3
"""Measure independent BF16 ATen Host API submits, never a composite operator.

All operands are fixed synthetic domains. The strided/broadcast control
matches a generic tensor-view mechanism, not a Serving shape or duration.
No complete rotary helper is timed. Synchronization and result destruction
are outside the per-call timed boundary. Parameters are not selected by error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
import torch_npu


DOMAINS = ((17, 4), (257, 4), (17, 16))
TRIALS = 120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--main-cpus', required=True)
    parser.add_argument('--acl-cpu', type=int, required=True)
    parser.add_argument('--release-cpu', type=int, required=True)
    parser.add_argument('--reverse-order', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    owner = ('profiler_output', 'microbench', 'ascend_910b')
    if sum(output.parts[i:i + 3] == owner for i in range(len(output.parts) - 2)) != 1:
        raise ValueError('output must belong to the 910B microbenchmark owner')
    device = os.environ['ASCEND_RT_VISIBLE_DEVICES']
    if not device.isdecimal():
        raise ValueError('exactly one physical NPU must be selected')
    first, last = (int(value) for value in args.main_cpus.split('-'))
    cpus = set(range(first, last + 1))
    allowed = os.sched_getaffinity(0)
    if not cpus or not cpus <= allowed or not {args.acl_cpu, args.release_cpu} <= allowed:
        raise ValueError('requested task-thread CPU domain unavailable')
    torch.npu.set_device('npu:0')
    os.sched_setaffinity(0, cpus)
    with torch.inference_mode():
        warm = torch.ones((17, 128), dtype=torch.bfloat16, device='npu:0')
        warm.add_(1)
        torch.npu.synchronize()
    bindings = []
    for thread in Path('/proc/self/task').iterdir():
        name = (thread / 'comm').read_text().strip()
        if name == 'acl_thread':
            target = {args.acl_cpu}
        elif name == 'release_thread':
            target = {args.release_cpu}
        else:
            continue
        os.sched_setaffinity(int(thread.name), target)
        bindings.append({'tid': int(thread.name), 'name': name, 'cpus': sorted(target)})
    if {item['name'] for item in bindings} != {'acl_thread', 'release_thread'}:
        raise RuntimeError('source worker ACL/release thread names unavailable')
    records = []
    domains = tuple(reversed(DOMAINS)) if args.reverse_order else DOMAINS
    with torch.inference_mode():
        for rows, heads in domains:
            full = torch.ones((rows, heads, 128), dtype=torch.bfloat16, device='npu:0')
            half = full[..., :64]
            cosine_parent = torch.ones((rows, 128), dtype=torch.bfloat16, device='npu:0')
            cosine = cosine_parent[:, :64].unsqueeze(1)
            left = half.contiguous()
            right = torch.ones_like(left)
            cosine_contiguous = cosine.contiguous()
            empty = full[..., 128:]
            flat = full.view(rows, heads * 128)
            table = torch.ones((512, 128), dtype=torch.bfloat16, device='npu:0')
            indices = torch.arange(rows, dtype=torch.int64, device='npu:0')
            if half.is_contiguous() or cosine.is_contiguous():
                raise ValueError('strided mechanism operands unexpectedly contiguous')
            calls = {
                'mul_strided_broadcast': lambda: torch.ops.aten.mul.Tensor(half, cosine),
                'mul_contiguous_broadcast': lambda: torch.ops.aten.mul.Tensor(left, cosine_contiguous),
                'add_contiguous': lambda: torch.ops.aten.add.Tensor(left, right),
                'sub_contiguous': lambda: torch.ops.aten.sub.Tensor(left, right),
                'cat_pair': lambda: torch.ops.aten.cat.default([left, right], -1),
                'cat_empty_suffix': lambda: torch.ops.aten.cat.default([full, empty], -1),
                'index_select_bf16_i64': lambda: torch.ops.aten.index_select.default(table, 0, indices),
                'reshape_metadata': lambda: torch.ops.aten.reshape.default(flat, [rows, -1, 128]),
                'split_2d_metadata': lambda: torch.ops.aten.split.Tensor(cosine_parent, 64, -1),
                'split_3d_metadata': lambda: torch.ops.aten.split.Tensor(full, 64, -1),
                'unsqueeze_metadata': lambda: torch.ops.aten.unsqueeze.default(cosine_parent[:, :64], -2),
                'slice_empty_metadata': lambda: torch.ops.aten.slice.Tensor(full, 2, 128, 9223372036854775807),
                'sym_size_metadata': lambda: torch.ops.aten.sym_size.int(full, 2),
                'noop_control': lambda: full,
            }
            # Construct the unsqueezed operand outside timing. The measured
            # entry is one unsqueeze, not an additional Python slice operation.
            cosine_2d = cosine_parent[:, :64]
            calls['unsqueeze_metadata'] = lambda: torch.ops.aten.unsqueeze.default(cosine_2d, -2)
            for _ in range(8):
                for call in calls.values():
                    result = call()
                    del result
            torch.npu.synchronize()
            samples = {name: [] for name in calls}
            order = list(calls)
            randomizer = random.Random(20261104)
            for _ in range(TRIALS):
                randomizer.shuffle(order)
                torch.npu.synchronize()
                for name in order:
                    call = calls[name]
                    cpu_start = time.thread_time_ns()
                    start = time.perf_counter_ns()
                    result = call()
                    elapsed = time.perf_counter_ns() - start
                    cpu = time.thread_time_ns() - cpu_start
                    samples[name].append({'wall_ns': elapsed, 'thread_cpu_ns': cpu})
                    del result
                torch.npu.synchronize()
            records.append({'rows': rows, 'heads': heads, 'half_width': 64,
                            'strided_left_strides': list(half.stride()),
                            'strided_rhs_strides': list(cosine.stride()),
                            'samples': samples,
                            'median_wall_ns': {name: statistics.median(v['wall_ns'] for v in values)
                                               for name, values in samples.items()}})
            print(rows, heads, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 1, 'kind': 'independent_aten_bf16_host_submits',
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'physical_npu': int(device), 'main_cpus': sorted(cpus), 'thread_bindings': bindings,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'boundary': 'one exact ATen OpOverload call to asynchronous return; no synchronization or output destruction inside timing',
        'selection_policy': 'retain all predeclared domains and two reversed-order fresh-process runs; no production error selects values',
        'production_duration_consumed': False, 'complete_composite_operator_timed': False,
        'global_irq_settings_changed': False, 'records': records,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
