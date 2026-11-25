#!/usr/bin/env python3
"""Capture the H200 FA2 source-task outer envelope probe under nsys.

The A800 probe ``fa_task_gap_geometry.cu`` ships without a runner; its yaml
entry (``fa2_source_task_outer_envelope``) states the procedure: the NSYS
whole-kernel span minus the maximum CTA-internal globaltimer envelope,
published as the maximum over grids 8..64 of 128-thread CTAs, and (probe
header) accepted only when it is invariant under the separately selected
in-CTA spin length.  This runner scripts that procedure on H200 (system
``python3``):

for every rotation, geometry (main = 80 KiB dynamic shared memory, combine =
static only), grid in 8..64 and spin length, one ``nsys profile -t cuda``
process runs ``SAMPLES`` synchronised launches (after the probe's 10 warm-up
launches); the nsys ``cuda_gpu_trace`` kernel durations are paired in launch
order with the per-CTA globaltimer rows the probe writes.  Output
``samples.csv`` holds, per launch, the nsys duration, the maximum per-CTA
globaltimer envelope and the launch-wide union envelope.  SM clock and
utilization are sampled every 50 ms during every capture (clock gate: no busy
sample or a busy sample below 1980 MHz rejects the capture).

Usage:
    python3 fa_task_gap_geometry_runner.py --binary <bin> --out-dir <dir> --device 0
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

NSYS = shutil.which('nsys') or 'nsys'
GRIDS = (8, 16, 32, 48, 64)
BODY_CYCLES = (4096, 32768)
GEOMETRIES = (1, 0)
SAMPLES = 50
WARMUPS = 10
ROTATIONS = 2
BOOST_MHZ = 1980


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device))
    captures = []
    with (args.out_dir / 'samples.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('rotation', 'main_geometry', 'grid', 'body_cycles', 'sample',
                         'nsys_duration_ns', 'max_cta_globaltimer_envelope_ns',
                         'union_globaltimer_envelope_ns', 'accepted'))
        for rotation in range(ROTATIONS):
            for geometry in GEOMETRIES:
                for grid in GRIDS:
                    for body in BODY_CYCLES:
                        tag = f'r{rotation}_m{geometry}_g{grid}_b{body}'
                        report = args.out_dir / tag
                        cta_csv = args.out_dir / f'{tag}.cta.csv'
                        clock_path = args.out_dir / f'{tag}.clock.csv'
                        with clock_path.open('w') as clock_file:
                            sampler = subprocess.Popen(
                                ['nvidia-smi', '-i', str(args.device),
                                 '--query-gpu=clocks.sm,utilization.gpu',
                                 '--format=csv,noheader,nounits', '-lms', '50'],
                                stdout=clock_file)
                            try:
                                subprocess.run(
                                    [NSYS, 'profile', '-t', 'cuda', '-s', 'none',
                                     '--cpuctxsw=none', '-o', str(report),
                                     '--force-overwrite', 'true', str(args.binary), '0',
                                     str(geometry), str(grid), str(body), str(SAMPLES),
                                     str(cta_csv)],
                                    check=True, env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                            finally:
                                time.sleep(0.2)
                                sampler.terminate()
                                sampler.wait()
                        subprocess.run([NSYS, 'stats', '--report', 'cuda_gpu_trace',
                                        '--format', 'csv', '--output', str(report),
                                        f'{report}.nsys-rep'], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        trace = list(csv.DictReader(open(f'{report}_cuda_gpu_trace.csv')))
                        durations = [int(row['Duration (ns)']) for row in trace
                                     if 'task_gap_geometry' in row['Name']]
                        if len(durations) != WARMUPS + SAMPLES:
                            raise RuntimeError(f'{tag}: {len(durations)} kernel records')
                        measured = durations[WARMUPS:]
                        per_sample = {}
                        for row in csv.DictReader(cta_csv.open()):
                            sample = int(row['sample'])
                            begin = int(row['global_begin_ns'])
                            end = int(row['global_end_ns'])
                            if sample not in per_sample:
                                per_sample[sample] = []
                            per_sample[sample].append((begin, end))
                        busy = [int(line.split(',')[0])
                                for line in clock_path.read_text().splitlines()
                                if line.strip() and int(line.split(',')[1]) > 0]
                        accepted = bool(busy) and min(busy) >= BOOST_MHZ
                        captures.append({'tag': tag, 'busy_samples': len(busy),
                                         'busy_clock_mhz_min': min(busy) if busy else None,
                                         'accepted': accepted})
                        for sample in range(SAMPLES):
                            ctas = per_sample[sample]
                            if len(ctas) != grid:
                                raise RuntimeError(f'{tag}: sample {sample} CTA count')
                            max_cta = max(end - begin for begin, end in ctas)
                            union = max(end for _, end in ctas) - min(begin for begin, _ in ctas)
                            writer.writerow((rotation, geometry, grid, body, sample,
                                             measured[sample], max_cta, union, int(accepted)))
                        cta_csv.unlink()
    (args.out_dir / 'capture.json').write_text(json.dumps({
        'device_index': args.device, 'grids': list(GRIDS), 'body_cycles': list(BODY_CYCLES),
        'samples': SAMPLES, 'rotations': ROTATIONS, 'captures': captures,
        'host_loadavg_after': Path('/proc/loadavg').read_text().split()[:3],
    }, indent=1) + '\n')
    print('accepted', sum(c['accepted'] for c in captures), 'of', len(captures))


if __name__ == '__main__':
    main()
