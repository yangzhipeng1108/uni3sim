#!/usr/bin/env python3
"""Extract real rank-0 attention-family kernel calls from H200 nsys traces.

Evaluation only.  For each ``window_<n>_cuda_gpu_trace.csv`` this keeps
device 0 (rank 0), selects kernels whose demangled name matches the same
``attention`` family regex as ``summarize_nvidia_h200_nsys_decode_windows.py``, and
groups calls by the physical launch (name, grid, block, registers, static and
dynamic shared memory) and by the token count of the step the call belongs to
(grid X of the most recent ``concat_and_cache_mla`` launch).  Per group it reports the call count, calls per step
(steps counted by the once-per-MoE-projection grouped GEMM, as in the
summariser) and the median / mean / p10 / p90 duration.  Runs with the plain
system ``python3`` (standard library only) so it can execute next to the
traces on the capture host.  No duration feeds any model parameter.

Usage:
    python3 extract_nvidia_h200_attention_kernel_calls.py --output calls.json \
        C=1:<dir>/window_1_cuda_gpu_trace.csv ...
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ATTENTION = re.compile(
    r'flash|mla|attention|Attention|fmha|FlashMLA|paged|decode_kernel|'
    r'merge_attn|BatchMLA')
# Families matched before ``attention`` in the summariser's first-match order.
EARLIER = (
    re.compile(r'allreduce|all_reduce|AllReduce|lamport|oneshot|twoshot|nccl'),
    re.compile(r'fused_moe_kernel|moe_gemm|grouped_gemm|MoeGemm'),
    re.compile(r'moe_align|topk|grouped_topk|moe_sum|count_and_sort|'
               r'moe_permute|moe_unpermute|expert'),
    re.compile(r'deep_gemm|DeepGemm|fp8_gemm|sm90_fp8|cutlass.*fp8|'
               r'gemm.*fp8|fp8.*gemm|blockscale|w8a8'),
)
MOE_LAYERS_TIMES_PROJECTIONS = 29 * 2


def is_attention(name):
    return (not any(p.search(name) for p in EARLIER)
            and ATTENTION.search(name) is not None)


def extract(label, csv_path, device):
    groups = {}
    moe_calls = 0
    # Token count of the step the call belongs to: the grid X of the most
    # recent concat_and_cache_mla launch (one CTA per written token).
    step_tokens = None
    with open(csv_path) as stream:
        for row in csv.DictReader(stream):
            if not row['Device'].endswith(f'({device})'):
                continue
            name = row['Name']
            if name == 'fused_moe_kernel':
                moe_calls += 1
                continue
            if 'memcpy' in name.lower() or 'memset' in name.lower():
                continue
            if not is_attention(name):
                continue
            if 'concat_and_cache_mla' in name:
                step_tokens = int(row['GrdX'])
            key = (
                step_tokens,
                name,
                tuple(int(row[c]) for c in ('GrdX', 'GrdY', 'GrdZ')),
                tuple(int(row[c]) for c in ('BlkX', 'BlkY', 'BlkZ')),
                int(row['Reg/Trd']), row['StcSMem (MB)'],
                row['DymSMem (MB)'])
            if key not in groups:
                groups[key] = []
            groups[key].append(int(row['Duration (ns)']))
    if moe_calls % MOE_LAYERS_TIMES_PROJECTIONS:
        raise ValueError(f'{label}: MoE calls {moe_calls} not a whole step count')
    steps = moe_calls // MOE_LAYERS_TIMES_PROJECTIONS
    rows = []
    for (tokens, name, grid, block, regs, stc, dyn), durations in groups.items():
        durations.sort()
        count = len(durations)
        rows.append({
            'step_tokens_from_kv_write': tokens,
            'name': name, 'grid': list(grid), 'block': list(block),
            'registers_per_thread': regs,
            'static_shared_memory_mb_nsys': stc,
            'dynamic_shared_memory_mb_nsys': dyn,
            'calls': count, 'calls_per_step': count / steps,
            'median_us': statistics.median(durations) / 1e3,
            'mean_us': sum(durations) / count / 1e3,
            'p10_us': durations[count // 10] / 1e3,
            'p90_us': durations[(9 * count) // 10] / 1e3,
            'min_us': durations[0] / 1e3, 'max_us': durations[-1] / 1e3,
            'us_per_step': sum(durations) / steps / 1e3,
        })
    rows.sort(key=lambda r: -r['us_per_step'])
    return {'source_csv': str(csv_path), 'device': device, 'steps': steps,
            'attention_us_per_step': sum(r['us_per_step'] for r in rows),
            'launches': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('windows', nargs='+', help='LABEL:path/to/trace.csv')
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'evaluation_only': True,
              'production_or_evaluation_duration_consumed_by_model': False,
              'windows': {}}
    for item in args.windows:
        label, path = item.split(':', 1)
        report['windows'][label] = extract(label, path, args.device)
        window = report['windows'][label]
        print(f'== {label}: steps {window["steps"]} attention '
              f'{window["attention_us_per_step"]:.1f} us/step')
        for r in window['launches']:
            print(f'   {r["us_per_step"]:8.1f} us/step {r["calls_per_step"]:6.2f}x '
                  f'med {r["median_us"]:8.2f} grid {r["grid"]} blk {r["block"]} '
                  f'reg {r["registers_per_thread"]} dyn {r["dynamic_shared_memory_mb_nsys"]} '
                  f'tok {r["step_tokens_from_kv_write"]} '
                  f'{r["name"][:70]}')
    args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
