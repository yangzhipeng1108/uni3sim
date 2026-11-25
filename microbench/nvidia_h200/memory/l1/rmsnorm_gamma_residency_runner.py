#!/usr/bin/env python3
"""Capture H200 shared-gamma L1 residency counters (owner copy of the A800 runner).

Same probe cases, grid/block, three trials per case, 5 % counter-spread gate and
summary fields as ``microbench/nvidia_a800/memory/l1/rmsnorm_gamma_residency_runner.py``,
plus the launch-lifetime derivation of ``rmsnorm_gamma_launch_lifetime_sm80.json``:

    l2_read_bytes_saved              = private_l2(H2048) - shared_l2(H2048)
    inferred_shared_gamma_l2_bytes   = logical_gamma_bytes(H2048) - saved
    inferred_shared_gamma_l2_copies  = inferred bytes / 4096
    launch_lifetime_maximum_gamma_bytes = 4096 iff copies == active SMs (132)
    maximum_evidenced_gamma_bytes    = largest shared-gamma size whose L2 read
                                       bytes stay below its private control

Measurement-tool coupling: Nsight Compute is not installed on the H200 host, so
the probe is built with ``-DH200_CUPTI_RANGE_PROFILER`` and reads the same five
metric names through the CUPTI 12.9 Range Profiler.  STATUS: on the current
target host container the driver restricts GPU performance counters to admin
(``RmProfilingAdminOnly: 1``) and CUPTI returns
``CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`` even for container root, so this runner
fails at the first case until counter access is granted.

usage (H200 host): python3 rmsnorm_gamma_residency_runner.py --run-id <id> \
    --out-root <dir> --device 3 --baseline-commit <sha>
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'compute'))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('rmsnorm_gamma_residency.cu')
CUDA = capture.NVCC.parents[1]
SM_COUNT = 132
TRIALS = 3
MAX_RELATIVE_SPREAD = 0.05
CASES = ('shared_gamma_h2048', 'shared_gamma_h8192', 'shared_gamma_h16384',
         'private_gamma_h16384_control', 'private_gamma_h2048_control',
         'private_gamma_h8192_control')
METRICS = ('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
           'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
           'l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum',
           'lts__t_requests_srcunit_tex_op_read.sum',
           'lts__t_sectors_srcunit_tex_op_read.sum')


def run_case(binary, case_id, device):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device),
                       LD_LIBRARY_PATH=str(CUDA / 'lib64'))
    result = subprocess.run([str(binary), str(case_id)], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=environment)
    lines = result.stdout.strip().splitlines()
    metric_line = [line for line in lines if line.startswith('metrics,')]
    native_line = [line for line in lines if not line.startswith('metrics,')]
    if len(metric_line) != 1 or len(native_line) != 1:
        raise ValueError(f'case {case_id}: unexpected output {result.stdout}')
    metrics = dict(item.split('=') for item in metric_line[0].split(',')[1:])
    fields = native_line[0].split(',')
    return {name: int(metrics[name]) for name in METRICS}, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--baseline-commit', required=True)
    args = parser.parse_args()
    out = args.out_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    identity = capture.device_identity(args.device)
    build = capture.build(SOURCE_PATH, out / 'build' / 'rmsnorm_gamma_residency_cupti',
                          ('-DH200_CUPTI_RANGE_PROFILER', f'-I{CUDA / "include"}',
                           f'-L{CUDA / "lib64"}', '-lcupti', '-lnvperf_host',
                           '-lnvperf_target', '-lcuda'))
    summaries = {}
    for case_id, name in enumerate(CASES):
        trials, native = [], None
        for _ in range(TRIALS):
            metrics, native = run_case(build['binary'], case_id, args.device)
            trials.append(metrics)
        medians, spreads = {}, {}
        for metric in METRICS:
            values = sorted(trial[metric] for trial in trials)
            medians[metric] = values[len(values) // 2]
            spreads[metric] = ((values[-1] - values[0]) / medians[metric]
                               if medians[metric] else 0.0)
            if spreads[metric] > MAX_RELATIVE_SPREAD:
                raise ValueError(f'{name}: unstable {metric}: {spreads[metric]}')
        logical_read_bytes = int(native[11])
        if medians[METRICS[1]] != logical_read_bytes:
            raise ValueError(f'{name}: L1 requested bytes disagree with source accounting')
        summaries[name] = {
            'hidden': int(native[2]), 'native': native, 'trials': trials,
            'median_metrics': medians, 'relative_spread': spreads,
            'logical_read_bytes': logical_read_bytes,
            'l2_read_bytes': medians[METRICS[4]] * 32,
            'l1_hit_fraction': (medians[METRICS[1]] - medians[METRICS[2]])
            / medians[METRICS[1]],
            'resident_ctas_per_sm': int(native[12]),
        }
    shared, private = summaries['shared_gamma_h2048'], summaries['private_gamma_h2048_control']
    logical_gamma = 1024 * 2048 * 2
    saved = private['l2_read_bytes'] - shared['l2_read_bytes']
    inferred = logical_gamma - saved
    copies = inferred / 4096
    evidenced = [hidden * 2 for hidden in (2048, 8192, 16384)
                 if summaries[f'shared_gamma_h{hidden}']['l2_read_bytes']
                 < summaries[f'private_gamma_h{hidden}_control']['l2_read_bytes']]
    result = {
        'schema': 'nvidia_h200_rmsnorm_gamma_residency_capture_v1',
        'hardware_id': 'nvidia_h200', 'device': identity,
        'baseline_commit': args.baseline_commit, 'build': build,
        'runner_sha256': capture.sha256(RUNNER_PATH), 'cases': summaries,
        'h2048_shared_private_control': {
            'gamma_bytes_per_cta': 4096, 'logical_gamma_bytes': logical_gamma,
            'shared_gamma_l2_read_bytes': shared['l2_read_bytes'],
            'private_gamma_l2_read_bytes': private['l2_read_bytes'],
            'l2_read_bytes_saved': saved,
            'inferred_shared_gamma_l2_bytes': inferred,
            'inferred_shared_gamma_l2_copies': copies},
        'h2048_one_gamma_fill_per_active_sm': copies == SM_COUNT,
        'shared_gamma_bytes_with_l2_savings': evidenced,
        'production_or_evaluation_duration_consumed': False,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['h2048_shared_private_control']))


if __name__ == '__main__':
    main()
