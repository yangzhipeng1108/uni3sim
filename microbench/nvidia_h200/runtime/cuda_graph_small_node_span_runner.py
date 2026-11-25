#!/usr/bin/env python3
"""Capture the H200 CUDA-graph small-node span probe (three rotations).

Runs on the H200 host with the system ``python3`` (standard library only).
Each rotation is one independent process of the binary built from
``microbench/nvidia_h200/runtime/cuda_graph_small_node_span.cu`` with the A800
arguments (9 trials, 64 nodes per batch) over the A800 grid list plus the
H200 full-chip multiples 132 and 264.  The A800 capture alternated rotations
between two devices; this worker owns one GPU, so the three rotations are
three independent processes on it.  SM clock and utilization are sampled every
50 ms during each rotation; a rotation with no busy sample or any busy sample
below 1980 MHz is recorded as rejected.

Usage:
    python3 cuda_graph_small_node_span_runner.py --binary <bin> --out-dir <dir>
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

TRIALS = 9
NODES = 64
GRIDS = (1, 2, 4, 8, 16, 32, 48, 80, 108, 132, 216, 264)
ROTATIONS = 3
BOOST_MHZ = 1980


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    raw = args.out_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    status = []
    for rotation in range(ROTATIONS):
        clock_path = raw / f'rotation_{rotation}.clock.csv'
        with clock_path.open('w') as clock_file:
            sampler = subprocess.Popen(
                ['nvidia-smi', '-i', str(args.device),
                 '--query-gpu=clocks.sm,utilization.gpu',
                 '--format=csv,noheader,nounits', '-lms', '50'], stdout=clock_file)
            try:
                with (raw / f'rotation_{rotation}.csv').open('w') as out:
                    subprocess.run([str(args.binary), str(TRIALS), str(NODES),
                                    ','.join(map(str, GRIDS))],
                                   check=True, stdout=out, env=env)
            finally:
                time.sleep(0.2)
                sampler.terminate()
                sampler.wait()
        busy = [int(line.split(',')[0]) for line in clock_path.read_text().splitlines()
                if line.strip() and int(line.split(',')[1]) > 0]
        status.append({'rotation': rotation, 'busy_samples': len(busy),
                       'busy_clock_mhz_min': min(busy) if busy else None,
                       'busy_clock_mhz_max': max(busy) if busy else None,
                       'accepted': bool(busy) and min(busy) >= BOOST_MHZ})
    loadavg = Path('/proc/loadavg').read_text().split()[:3]
    (args.out_dir / 'capture.json').write_text(json.dumps({
        'device_index': args.device, 'trials': TRIALS, 'nodes_per_batch': NODES,
        'grids': list(GRIDS), 'rotations': status, 'host_loadavg_after': loadavg,
    }, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
