#!/usr/bin/env python3
"""Capture the H200 CTA dispatch-cost probe: three independent trials.

Runs on the H200 host with the system ``python3``.  Each trial is one process
of the binary built from ``cta_dispatch_cost.cu`` (A800 used three trials of
the same probe: trials 1-2 derive, trial 3 validates).  SM clock/utilization
are sampled every 50 ms during each trial; a trial with no busy sample or any
busy sample below 1980 MHz is rejected in ``capture.json``.

Usage:
    python3 cta_dispatch_cost_runner.py --binary <bin> --out-dir <dir> --device 0
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

TRIALS = (1, 2, 3)
BOOST_MHZ = 1980


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    status = []
    for trial in TRIALS:
        clock_path = args.out_dir / f'trial_{trial}.clock.csv'
        loadavg = Path('/proc/loadavg').read_text().split()[:3]
        with clock_path.open('w') as clock_file:
            sampler = subprocess.Popen(
                ['nvidia-smi', '-i', str(args.device),
                 '--query-gpu=clocks.sm,utilization.gpu',
                 '--format=csv,noheader,nounits', '-lms', '50'], stdout=clock_file)
            try:
                with (args.out_dir / f'trial_{trial}.csv').open('w') as out:
                    # The probe calls cudaSetDevice(argv[1]); device 0 is the
                    # only visible device under CUDA_VISIBLE_DEVICES.
                    subprocess.run([str(args.binary), '0'], check=True, stdout=out, env=env)
            finally:
                time.sleep(0.2)
                sampler.terminate()
                sampler.wait()
        busy = [int(line.split(',')[0]) for line in clock_path.read_text().splitlines()
                if line.strip() and int(line.split(',')[1]) > 0]
        status.append({'trial': trial, 'busy_samples': len(busy),
                       'busy_clock_mhz_min': min(busy) if busy else None,
                       'busy_clock_mhz_max': max(busy) if busy else None,
                       'accepted': bool(busy) and min(busy) >= BOOST_MHZ,
                       'host_loadavg_before': loadavg})
    (args.out_dir / 'capture.json').write_text(json.dumps(
        {'device_index': args.device, 'trials': status}, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
