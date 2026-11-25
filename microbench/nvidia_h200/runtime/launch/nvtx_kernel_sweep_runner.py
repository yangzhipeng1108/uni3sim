#!/usr/bin/env python3
"""Capture an NVTX-labelled launch sweep under nsys on the H200 host.

Shared capture step of the two NVTX sweep probes of this directory,
``empty_kernel_duration.cu`` (launch_latency_model) and
``memory_dependency_task_duration.cu`` (memory_dependency_task_latency_model).
Both probes push one NVTX range ``<prefix>_grid_<g>_block_<b>`` around every
labelled launch; the A800 reports consume a per-(trial, grid, block) summary of
the nsys GPU-kernel durations of those launches.  This runner reproduces that
summary on H200 (system ``python3``, standard library only):

1. for each trial, sample SM clock/utilization every 50 ms with nvidia-smi and
   run ``nsys profile -t cuda,nvtx`` on ``<binary> --sweep``;
2. export the report to SQLite and join every NVTX range to the one
   ``cudaLaunchKernel`` call inside it and, via ``correlationId``, to its
   CUPTI kernel record; the grid/block of every kernel record must equal its
   label;
3. write ``summary_by_trial.csv`` (the A800 fields ``trial, grid, block,
   sample_count, median_duration_ns``), ``cell_minimum_by_trial.csv`` (the
   per-cell minimum needed by the memory-dependency lower envelope) and
   ``capture.json`` with the clock gate (a trial with no busy sample or any
   busy sample below 1980 MHz is rejected).

Usage:
    python3 nvtx_kernel_sweep_runner.py --binary <bin> --label-prefix <p> \
        --kernel-name <n> --trials 3 --samples-per-cell 200 --out-dir <dir> --device 0
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import shutil
import time

NSYS = shutil.which('nsys') or 'nsys'
BOOST_MHZ = 1980


def extract(sqlite_path: Path, prefix: str, kernel_name: str):
    label = re.compile(rf'^{re.escape(prefix)}_grid_(\d+)_block_(\d+)$')
    connection = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True)
    try:
        strings = dict(connection.execute('SELECT id, value FROM StringIds'))
        ranges = []
        for text, text_id, start, end in connection.execute(
                'SELECT text, textId, start, end FROM NVTX_EVENTS '
                'WHERE end IS NOT NULL ORDER BY start'):
            name = text if text is not None else strings[text_id]
            match = label.match(name)
            if match:
                ranges.append((start, end, int(match.group(1)), int(match.group(2))))
        launches = [
            (start, end, correlation) for correlation, start, end, name_id in
            connection.execute(
                'SELECT correlationId, start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME '
                'ORDER BY start')
            if strings[name_id].startswith('cudaLaunchKernel')]
        kernels = {}
        for correlation, start, end, gx, gy, gz, bx, by, bz, short in connection.execute(
                'SELECT correlationId, start, end, gridX, gridY, gridZ, blockX, blockY, '
                'blockZ, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            if correlation in kernels:
                raise ValueError(f'duplicate kernel correlation {correlation}')
            kernels[correlation] = (start, end, (gx, gy, gz), (bx, by, bz), strings[short])
    finally:
        connection.close()
    rows = []
    cursor = 0
    for range_start, range_end, grid, block in ranges:
        while cursor < len(launches) and launches[cursor][0] < range_start:
            cursor += 1
        if cursor == len(launches) or launches[cursor][1] > range_end:
            raise ValueError(f'NVTX range at {range_start} holds no launch')
        correlation = launches[cursor][2]
        cursor += 1
        start, end, grid_dim, block_dim, name = kernels[correlation]
        if name != kernel_name or grid_dim != (grid, 1, 1) or block_dim != (block, 1, 1):
            raise ValueError(f'kernel record disagrees with its label: {name} {grid_dim} '
                             f'{block_dim} vs grid {grid} block {block}')
        rows.append((grid, block, end - start))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--label-prefix', required=True)
    parser.add_argument('--kernel-name', required=True)
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--samples-per-cell', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    cells = {}
    trial_status = []
    for trial in range(1, args.trials + 1):
        report = args.out_dir / f'trial{trial}'
        clock_path = args.out_dir / f'trial{trial}.clock.csv'
        loadavg_before = Path('/proc/loadavg').read_text().split()[:3]
        with clock_path.open('w') as clock_file:
            sampler = subprocess.Popen(
                ['nvidia-smi', '-i', str(args.device),
                 '--query-gpu=clocks.sm,utilization.gpu',
                 '--format=csv,noheader,nounits', '-lms', '50'], stdout=clock_file)
            try:
                subprocess.run([NSYS, 'profile', '-t', 'cuda,nvtx', '-s', 'none',
                                '--cpuctxsw=none', '-o', str(report),
                                '--force-overwrite', 'true', str(args.binary), '--sweep'],
                               check=True, env=env, stdout=subprocess.DEVNULL)
            finally:
                time.sleep(0.2)
                sampler.terminate()
                sampler.wait()
        subprocess.run([NSYS, 'export', '--type', 'sqlite', '--force-overwrite', 'true',
                        '--output', f'{report}.sqlite', f'{report}.nsys-rep'],
                       check=True, stdout=subprocess.DEVNULL)
        rows = extract(Path(f'{report}.sqlite'), args.label_prefix, args.kernel_name)
        busy = [int(line.split(',')[0]) for line in clock_path.read_text().splitlines()
                if line.strip() and int(line.split(',')[1]) > 0]
        accepted = bool(busy) and min(busy) >= BOOST_MHZ
        trial_status.append({
            'trial': trial, 'labelled_launches': len(rows), 'busy_samples': len(busy),
            'busy_clock_mhz_min': min(busy) if busy else None,
            'busy_clock_mhz_max': max(busy) if busy else None,
            'accepted': accepted, 'host_loadavg_before': loadavg_before})
        if not accepted:
            continue
        for grid, block, duration in rows:
            identity = (trial, grid, block)
            if identity not in cells:
                cells[identity] = []
            cells[identity].append(duration)
    for identity, durations in cells.items():
        if len(durations) != args.samples_per_cell:
            raise ValueError(f'cell {identity} holds {len(durations)} samples')
    with (args.out_dir / 'summary_by_trial.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('trial', 'grid', 'block', 'sample_count', 'median_duration_ns'))
        for (trial, grid, block), durations in sorted(cells.items()):
            writer.writerow((trial, grid, block, len(durations),
                             repr(float(statistics.median(durations)))))
    with (args.out_dir / 'cell_minimum_by_trial.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('trial', 'grid', 'block', 'minimum_duration_ns'))
        for (trial, grid, block), durations in sorted(cells.items()):
            writer.writerow((trial, grid, block, min(durations)))
    driver = subprocess.run(['nvidia-smi', '-i', str(args.device),
                             '--query-gpu=name,uuid,driver_version',
                             '--format=csv,noheader'], check=True,
                            capture_output=True, text=True).stdout.strip()
    (args.out_dir / 'capture.json').write_text(json.dumps({
        'binary': str(args.binary), 'device_index': args.device, 'device': driver,
        'nsys': NSYS, 'nsys_command': 'nsys profile -t cuda,nvtx -s none --cpuctxsw=none',
        'label_prefix': args.label_prefix, 'kernel_name': args.kernel_name,
        'samples_per_cell': args.samples_per_cell, 'trials': trial_status,
    }, indent=2) + '\n')
    print(json.dumps(trial_status))


if __name__ == '__main__':
    main()
