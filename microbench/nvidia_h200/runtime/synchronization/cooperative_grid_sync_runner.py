#!/usr/bin/env python3
"""Capture the cooperative grid-barrier service (cooperative_grid_sync.cu) on H200.

Mechanism probe (no production workload).  Runs the prebuilt binary alone on
one GPU for each grid in GRIDS with 256 threads and the one-CTA-per-SM
147456-B dynamic shared-memory envelope, REPEATS barriers per launch, SAMPLES
launches per grid (3 warm-up launches discarded inside the binary) and
TRIALS rotations.  nvidia-smi samples the SM clock every 50 ms; a trial whose
busy samples are not all at 1980 MHz is rejected.  Output: <out>/raw/*.csv
and <out>/grid_sync_summary.json with the median ns per barrier per grid.
Usage: python cooperative_grid_sync_runner.py --binary <path> --out <dir> --gpu <index>
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess

BOOST_MHZ = 1980
GRIDS = (1, 2, 4, 8, 16, 32, 66, 132)
REPEATS = 1024
SAMPLES = 32
TRIALS = 3
SMEM = 147456


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    raw = args.out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    summary = {'mechanism': 'cooperative_grid_sync', 'threads_per_cta': 256,
               'dynamic_smem_bytes': SMEM, 'repeats': REPEATS,
               'samples': SAMPLES, 'trials': TRIALS,
               'production_or_evaluation_duration_consumed': False,
               'grids': {}}
    for grid in GRIDS:
        accepted = []
        for trial in range(TRIALS):
            monitor = subprocess.Popen(
                ['nvidia-smi', '-i', str(args.gpu), '-lms', '50',
                 '--query-gpu=clocks.sm,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, text=True)
            result = subprocess.run(
                [str(args.binary), str(grid), str(REPEATS), str(SAMPLES),
                 str(SMEM)], env=env, capture_output=True, text=True,
                check=True)
            monitor.terminate()
            clocks, _ = monitor.communicate()
            busy = [int(line.split(',')[0]) for line in clocks.splitlines()
                    if line.strip() and int(line.split(',')[1]) > 0]
            (raw / f'grid{grid}_trial{trial}.csv').write_text(result.stdout)
            clock_ok = bool(busy) and all(mhz == BOOST_MHZ for mhz in busy)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if clock_ok:
                accepted.append({name: statistics.median(
                    float(row[name]) for row in rows) for name in (
                        'ns_per_barrier', 'cooperative_span_ns',
                        'regular_span_ns')})
        summary['grids'][str(grid)] = {
            'accepted_trials': len(accepted),
            **{f'median_{name}': (
                statistics.median(trial[name] for trial in accepted)
                if accepted else None)
               for name in ('ns_per_barrier', 'cooperative_span_ns',
                            'regular_span_ns')},
        }
    (args.out / 'grid_sync_summary.json').write_text(
        json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary['grids'], indent=1))


if __name__ == '__main__':
    main()
