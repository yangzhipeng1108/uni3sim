#!/usr/bin/env python3
"""Summarise nsys decode windows of the real vLLM DSV3-L32 TP4 server.

For each ``window_<n>.nsys-rep`` this exports the CUDA GPU trace (graph nodes
traced individually), keeps device 0, splits it into engine steps at gaps
longer than ``--step-gap-us`` or at the sampler kernel, and reports per-step
wall time, GPU-busy time, idle time and busy time by kernel family.  Output
is one JSON per window plus a printed table; no timing is fed back to any
model parameter (evaluation only).
"""
import argparse
import collections
import csv
import json
import re
import subprocess
from pathlib import Path

FAMILIES = (
    ('allreduce', r'allreduce|all_reduce|AllReduce|lamport|oneshot|twoshot|nccl'),
    ('moe_grouped_gemm', r'fused_moe_kernel|moe_gemm|grouped_gemm|MoeGemm'),
    ('moe_routing', r'moe_align|topk|grouped_topk|moe_sum|count_and_sort|moe_permute|moe_unpermute|expert'),
    ('dense_fp8_gemm', r'deep_gemm|DeepGemm|fp8_gemm|sm90_fp8|cutlass.*fp8|gemm.*fp8|fp8.*gemm|blockscale|w8a8'),
    ('attention', r'flash|mla|attention|Attention|fmha|FlashMLA|paged|decode_kernel|merge_attn|BatchMLA'),
    ('quant', r'quant|per_token_group|scaled_fp8|fp8_quant'),
    ('norm_rope_act', r'rms|norm|rotary|rope|silu|act_and_mul|gelu'),
    ('gemm_other', r'gemm|Gemm|cublas|cutlass|sm90_xmma|nvjet|matmul'),
    ('copy_misc', r'copy|Copy|fill|elementwise|reduce|index|gather|scatter|cat|triton'),
)


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name):
            return label
    return 'other'


def export(rep: Path) -> Path:
    prefix = rep.with_suffix('')
    csv_path = Path(f'{prefix}_cuda_gpu_trace.csv')
    if not csv_path.exists():
        subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv',
                        '--output', str(prefix), str(rep)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return csv_path


def load_kernels(csv_path: Path, device: int):
    kernels = []
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            if not row['Device'].endswith(f'({device})'):
                continue
            name = row['Name']
            if 'memcpy' in name.lower() or 'memset' in name.lower():
                continue
            start = int(row['Start (ns)'])
            kernels.append((start, start + int(row['Duration (ns)']), name))
    kernels.sort()
    return kernels


MOE_LAYERS_TIMES_PROJECTIONS = 29 * 2  # 32 layers, first 3 dense; W1 and W2


def union_ns(kernels):
    total, cur_start, cur_end = 0, None, None
    for start, end, _ in kernels:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('window_dir', type=Path)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    labels = [line.split()[0] for line in (args.window_dir / 'windows.txt').read_text().splitlines()
              if line.split()[1] == 'start']
    reps = sorted(args.window_dir.glob('window_*.nsys-rep'),
                  key=lambda p: int(re.search(r'_(\d+)\.nsys-rep$', p.name).group(1)))
    summary = {}
    for label, rep in zip(labels, reps):
        kernels = load_kernels(export(rep), args.device)
        # vLLM 0.24 runs engine steps back to back (async scheduling), so
        # steps are counted by the once-per-MoE-projection grouped GEMM.
        moe_calls = sum(1 for k in kernels if k[2] == 'fused_moe_kernel')
        steps = moe_calls / MOE_LAYERS_TIMES_PROJECTIONS
        span = kernels[-1][1] - kernels[0][0]
        busy_union = union_ns(kernels)
        fam = collections.defaultdict(float)
        names = collections.defaultdict(lambda: [0, 0.0])
        for start, end, name in kernels:
            fam[family(name)] += (end - start)
            names[name][0] += 1
            names[name][1] += (end - start)
        summary[label] = {
            'rep': rep.name, 'steps': steps, 'kernels': len(kernels),
            'kernels_per_step': len(kernels) / steps,
            'step_wall_us': span / steps / 1e3,
            'step_busy_union_us': busy_union / steps / 1e3,
            'step_idle_us': (span - busy_union) / steps / 1e3,
            'family_us_per_step': {k: v / steps / 1e3
                                   for k, v in sorted(fam.items(), key=lambda kv: -kv[1])},
            'top_kernels': sorted(
                ({'name': n[:200], 'calls_per_step': c / steps,
                  'us_per_call': t / c / 1e3, 'us_per_step': t / steps / 1e3}
                 for n, (c, t) in names.items()),
                key=lambda d: -d['us_per_step'])[:30],
        }
        s = summary[label]
        print(f"== {label}: steps {s['steps']:.1f} kernels/step {s['kernels_per_step']:.0f} "
              f"wall {s['step_wall_us']:.0f}us busy(union) {s['step_busy_union_us']:.0f}us "
              f"idle {s['step_idle_us']:.0f}us")
        for k, v in s['family_us_per_step'].items():
            print(f"     {k:<18} {v:8.1f} us/step")
        for t in s['top_kernels'][:12]:
            print(f"       {t['us_per_step']:8.1f}us/step {t['calls_per_step']:6.1f}x {t['us_per_call']:7.2f}us  {t['name'][:90]}")
    (args.window_dir / 'decode_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
