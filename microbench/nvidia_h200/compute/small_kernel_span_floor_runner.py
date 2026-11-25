#!/usr/bin/env python3
"""Run the H200 small-kernel span-floor probe and publish its reference.

Same mechanism, bodies, launch patterns and selection rule as the A800
reference ``small_kernel_span_floor_sm80.json``: each (threads, body, launch
pattern, grid) point launches ``LAUNCHES`` kernels under nsys, the kernel
durations come from the ``cuda_gpu_trace`` report, and the published floor at
each grid is the smallest one-store duration across trials and both launch
patterns.  SM clock is sampled every 50 ms during every capture; a capture in
which any busy sample is below the boost clock is rejected, since the clock
cannot be locked in this container.

Usage (on the H200 host, binaries built from
microbench/nvidia_h200/compute/small_kernel_span_floor.cu):
    python3 small_kernel_span_floor_runner.py --bin-dir <dir> --out-dir <dir>
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

GRIDS = (1, 16, 48, 108, 132, 216, 264)
THREADS = (128, 512)
LAUNCHES = 30
TRIALS = 3
BOOST_MHZ = 1980


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(binary: Path, grid: int, out: Path, device: int):
    rep = out.with_suffix('')
    clock_log = out.with_suffix('.clock.csv')
    sampler = subprocess.Popen(
        ['nvidia-smi', '-i', str(device), '--query-gpu=clocks.sm,utilization.gpu',
         '--format=csv,noheader,nounits', '-lms', '50'],
        stdout=clock_log.open('w'))
    try:
        subprocess.run(['nsys', 'profile', '-t', 'cuda', '-o', str(rep),
                        '--force-overwrite', 'true', str(binary), '0', str(grid),
                        str(LAUNCHES)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device)))
    finally:
        time.sleep(0.2)
        sampler.terminate()
        sampler.wait()
    subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv',
                    '--output', str(rep), f'{rep}.nsys-rep'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(f'{rep}_cuda_gpu_trace.csv')))
    durations = [int(r['Duration (ns)']) for r in rows
                 if 'memcpy' not in r['Name'].lower() and 'memset' not in r['Name'].lower()]
    # The first five launches are the probe's own warm-up.
    measured = durations[5:]
    if len(measured) != LAUNCHES:
        raise RuntimeError(f'{binary.name} grid {grid}: {len(measured)} launches')
    clocks = [int(line.split(',')[0]) for line in clock_log.read_text().splitlines()
              if line.strip() and int(line.split(',')[1]) > 0]
    return measured, clocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--source', type=Path, required=True,
                        help='probe source path in the repository (for provenance)')
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    points = []
    rejected = []
    for threads in THREADS:
        for body in (0, 1):
            for sync in (1, 0):
                binary = args.bin_dir / f'skf_t{threads}_b{body}_s{sync}'
                for grid in GRIDS:
                    for trial in range(TRIALS):
                        out = args.out_dir / f't{threads}_b{body}_s{sync}_g{grid}_r{trial}.csv'
                        durations, clocks = capture(binary, grid, out, args.device)
                        point = {'threads': threads, 'body': 'one_store' if body else 'empty',
                                 'launch_pattern': 'isolated' if sync else 'back_to_back',
                                 'grid_ctas': grid, 'trial': trial,
                                 'durations_ns': durations,
                                 'busy_clock_mhz_min': min(clocks) if clocks else None}
                        if not clocks or min(clocks) < BOOST_MHZ:
                            rejected.append(point)
                        else:
                            points.append(point)
    floor = []
    for grid in GRIDS:
        store = [min(p['durations_ns']) for p in points
                 if p['grid_ctas'] == grid and p['body'] == 'one_store']
        empty = [min(p['durations_ns']) for p in points
                 if p['grid_ctas'] == grid and p['body'] == 'empty']
        iso = [d for p in points if p['grid_ctas'] == grid and p['body'] == 'one_store'
               and p['launch_pattern'] == 'isolated' for d in p['durations_ns']]
        b2b = [d for p in points if p['grid_ctas'] == grid and p['body'] == 'one_store'
               and p['launch_pattern'] == 'back_to_back' for d in p['durations_ns']]
        if not store or not empty:
            raise RuntimeError(f'grid {grid}: no clock-accepted capture')
        floor.append({'grid_ctas': grid, 'minimum_span_ns': float(min(store)),
                      'isolated_median_ns': float(sorted(iso)[len(iso) // 2]) if iso else None,
                      'back_to_back_median_ns': float(sorted(b2b)[len(b2b) // 2]) if b2b else None,
                      'empty_cta_minimum_ns': float(min(empty))})
    reference = {
        'schema': 'small_kernel_span_floor_reference_v1',
        'hardware_id': 'nvidia_h200',
        'mechanism': 'small_kernel_span_floor',
        'production_or_evaluation_duration_consumed': False,
        'allowed_use': ('A lower bound on the profiler-reported duration of a launch '
                        'that performs at least one global memory access, given its '
                        'CTA grid. It composes with an execution time by maximum and '
                        'never adds to one.'),
        'forbidden_use': ('An additive launch cost, a typical or median duration, a host '
                          'launch latency, a CUDA-event interval, a per-shape correction, '
                          'or a floor for a launch with no global memory access.'),
        'derived_model': {
            'floor_by_grid_ctas': floor,
            'selection_rule': ('At each measured grid retain the smallest one-store '
                               'duration seen across trials, block sizes and both launch '
                               'patterns, isolated and back-to-back (same rule as the '
                               'A800 reference). Captures whose busy SM clock fell below '
                               f'{BOOST_MHZ} MHz are rejected.'),
        },
        'scope': {'timer_domain': 'nsys_gpu_kernel', 'threads_per_cta_measured': list(THREADS),
                  'minimum_grid_ctas': GRIDS[0], 'maximum_grid_ctas': GRIDS[-1]},
        'provenance': {'source': str(args.source), 'source_sha256': args.source_sha256,
                       'launches_per_point': LAUNCHES, 'trials': TRIALS,
                       'device_index': args.device, 'profiler': 'nsys 2025.6.1',
                       'report': 'cuda_gpu_trace',
                       'accepted_captures': len(points), 'rejected_captures': len(rejected)},
    }
    (args.out_dir / 'raw_points.json').write_text(json.dumps(
        {'accepted': points, 'rejected': rejected}) + '\n')
    (args.out_dir / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    for row in floor:
        print(row)
    print('accepted', len(points), 'rejected', len(rejected))


if __name__ == '__main__':
    main()
