#!/usr/bin/env python3
"""Extract real H200 FP8 block-scaled dense GEMM calls from nsys GPU traces.

Evaluation only: per-call durations printed here are compared with Midlayer
predictions in WP_REPORT.md and are never read by a predictor, HWSpec or
launch selector.

For every ``--window LABEL:CSV:M`` (an nsys ``cuda_gpu_trace`` CSV) this
keeps one device, walks its kernels in start order and records

* ``deep_gemm::fp8_gemm_kernel_swapAB<N, K, ...>`` (TensorRT-LLM/FlashInfer
  swap-AB DeepGEMM, M < 32 path) together with the
  ``tensorrt_llm::...::scale_1x128_kernel`` launched immediately before it on
  the same CUDA stream by the same ``fp8_gemm_run`` call;
* ``deep_gemm::sm90_fp8_gemm_1d2d_impl<...>`` (DeepGEMM ``fp8_gemm_nt``,
  M >= 32 path) with its GEMM M recovered from the launch geometry of the
  ``per_token_group_quant_8bit_kernel`` that quantised its activation.

Kernels are grouped by (kernel, N, K, template tuple, launch geometry); the
output holds count, median, p10 and p90 of the per-call duration and the
launch geometry columns.  Standard library only (runs with the remote system
python3).
"""
import argparse
import csv
import json
import re
import statistics
import sys

SWAP_AB = 'deep_gemm::fp8_gemm_kernel_swapAB<'
ONE_D_TWO_D = 'deep_gemm::sm90_fp8_gemm_1d2d_impl<'
SCALE = 'fp8_blockscale_gemm::scale_1x128_kernel<'
GROUP_QUANT = 'per_token_group_quant_8bit_kernel<'
UINT = re.compile(r'\(unsigned int\)(\d+)')
BOOL = re.compile(r'\(bool\)(\d)')


def template_values(name):
    return [int(v) for v in UINT.findall(name)]


def swap_ab_key(name):
    v = template_values(name)
    # <SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages,
    #  kNumTMAThreads, kNumMathThreadsPerGroup, kNumTMAMulticast, Scheduler<..>>
    return {
        'kernel': 'trtllm_deepgemm_fp8_gemm_kernel_swapAB',
        'n': v[0], 'k': v[1], 'block_m': v[2], 'block_n': v[3],
        'block_k': v[4], 'num_groups': v[5], 'num_stages': v[6],
        'num_tma_threads': v[7], 'num_math_threads_per_group': v[8],
        'num_tma_multicast': v[9],
    }


def one_d_two_d_key(name):
    v = template_values(name)
    b = BOOL.findall(name)
    # <major_sfb, SHAPE_M, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N,
    #  BLOCK_K, kSwizzleA, kSwizzleB, kSwizzleD, kNumStages, kNumTMAThreads,
    #  kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, ...>
    return {
        'kernel': 'deepgemm_sm90_fp8_gemm_1d2d_impl',
        'shape_m_template': v[0], 'n': v[1], 'k': v[2], 'num_groups': v[3],
        'block_m': v[4], 'block_n': v[5], 'block_k': v[6],
        'swizzle_a': v[7], 'swizzle_b': v[8], 'swizzle_d': v[9],
        'num_stages': v[10], 'num_tma_threads': v[11],
        'num_math_threads': v[12], 'num_tma_multicast': v[13],
        'tma_multicast_on_a': int(b[0]), 'num_sms': v[14],
    }


def geometry(row):
    return {
        'grid': [int(row['GrdX']), int(row['GrdY']), int(row['GrdZ'])],
        'block': [int(row['BlkX']), int(row['BlkY']), int(row['BlkZ'])],
        'registers_per_thread': int(row['Reg/Trd']),
        'static_shared_memory_mb_nsys': row['StcSMem (MB)'],
        'dynamic_shared_memory_mb_nsys': row['DymSMem (MB)'],
    }


def group_quant_m(row, k):
    """Return the GEMM M implied by one activation-quant launch.

    vLLM 0.24 ``per_token_group_quant_8bit`` runs 16 threads per 128-channel
    group, so ``groups_per_block = BlkX / 16`` and the launch covers
    ``GrdX * groups_per_block = M * K / 128`` groups.
    """
    block_x = int(row['BlkX'])
    if block_x % 16:
        raise ValueError(f'unexpected group-quant block {block_x}')
    groups = int(row['GrdX']) * (block_x // 16)
    if groups % (k // 128):
        raise ValueError(f'group-quant grid {row["GrdX"]} does not tile K={k}')
    return groups // (k // 128)


def summarize(values):
    ordered = sorted(values)
    count = len(ordered)
    return {
        'count': count,
        'median_ns': statistics.median(ordered),
        'p10_ns': ordered[int(0.1 * (count - 1))],
        'p90_ns': ordered[int(0.9 * (count - 1))],
    }


def extract(csv_path, device):
    kernels = []
    with open(csv_path, newline='') as stream:
        for row in csv.DictReader(stream):
            if not row['Device'].endswith(f'({device})'):
                continue
            if not row['GrdX']:
                continue  # memcpy / memset rows
            kernels.append(row)
    kernels.sort(key=lambda row: int(row['Start (ns)']))
    groups = {}
    previous_on_stream = {}
    quant_on_stream = {}
    for row in kernels:
        stream_id = row['Strm']
        preceding = tuple(
            previous_on_stream[key] for key in (stream_id,)
            if key in previous_on_stream)
        previous_on_stream[stream_id] = row
        name = row['Name']
        if GROUP_QUANT in name:
            quant_on_stream[stream_id] = row
            continue
        if SWAP_AB in name:
            key = swap_ab_key(name)
        elif ONE_D_TWO_D in name:
            key = one_d_two_d_key(name)
        else:
            continue
        key.update(geometry(row))
        key['stream'] = int(stream_id)
        if ONE_D_TWO_D in name:
            # The activation quant of this call is the last group-quant on
            # the stream (a scale-layout copy may sit in between); each quant
            # feeds exactly one GEMM.
            if stream_id not in quant_on_stream:
                raise ValueError(
                    f'1D2D launch at {row["Start (ns)"]} has no preceding '
                    'per_token_group_quant_8bit_kernel on its stream')
            key['gemm_m'] = group_quant_m(
                quant_on_stream.pop(stream_id), key['k'])
        identity = json.dumps(key, sort_keys=True)
        if identity not in groups:
            groups[identity] = {
                'key': key, 'durations': [], 'scale_1x128': [],
                'scale_geometry': None}
        entry = groups[identity]
        entry['durations'].append(int(row['Duration (ns)']))
        if SWAP_AB in name:
            if len(preceding) != 1 or SCALE not in preceding[0]['Name']:
                raise ValueError(
                    f'swapAB launch at {row["Start (ns)"]} is not preceded '
                    'by scale_1x128_kernel')
            previous = preceding[0]
            entry['scale_1x128'].append(int(previous['Duration (ns)']))
            scale_geometry = geometry(previous)
            if entry['scale_geometry'] not in (None, scale_geometry):
                raise ValueError('scale_1x128 geometry changed within group')
            entry['scale_geometry'] = scale_geometry
    result = []
    for entry in groups.values():
        item = {'launch': entry['key'],
                'duration': summarize(entry['durations'])}
        if entry['scale_1x128']:
            item['scale_1x128_launch'] = entry['scale_geometry']
            item['scale_1x128_duration'] = summarize(entry['scale_1x128'])
        result.append(item)
    result.sort(key=lambda item: (item['launch']['kernel'],
                                  item['launch']['n'], item['launch']['k'],
                                  item['launch']['block_m'],
                                  item['launch']['block_n']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', action='append', required=True,
                        help='LABEL:CSV_PATH:M (M = tokens per GEMM call)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {
        'artifact_type': 'evaluation_only_real_kernel_durations',
        'production_or_evaluation_duration_consumed_by_model': False,
        'device': args.device,
        'windows': {},
    }
    for spec in args.window:
        label, path, m = spec.split(':')
        report['windows'][label] = {
            'source_csv': path, 'gemm_m': int(m),
            'calls': extract(path, args.device)}
        print(label, len(report['windows'][label]['calls']), 'groups',
              file=sys.stderr)
    with open(args.output, 'w') as stream:
        json.dump(report, stream, indent=1)
        stream.write('\n')


if __name__ == '__main__':
    main()
