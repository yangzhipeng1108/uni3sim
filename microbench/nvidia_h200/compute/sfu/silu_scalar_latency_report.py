#!/usr/bin/env python3
"""Derive the H200 ``silu_scalar_round`` service from the scalar-SiLU probe.

A800 procedure (scripted here; A800 derived it by hand): take the
``threads=1024 iterations=64`` line of ``silu_scalar_latency`` and read
``dependent_cycles_per_scalar`` (249.125 SM cycles on A800); the Forward rate is
model_clock / cycles (A800: 1.14e9 / 249.125).  On H200 the cycles are converted
with the measured, clock-gated SM clock (1980 MHz).  The runner records
``ROTATIONS`` independent processes; the published value is the median of their
64-iteration results (with two rotations, their mean) and the spread is recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

REPORT_PATH = Path(__file__).resolve()
LINE = re.compile(
    r'threads=(\d+) iterations=(\d+) dependent_cycles=(\d+) '
    r'dependent_cycles_per_scalar=([0-9.]+) independent_cycles=(\d+) '
    r'independent_cycles_per_scalar=([0-9.]+)')
SELECTED_THREADS = 1024
SELECTED_ITERATIONS = 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(path):
    text = Path(path).read_text()
    header = text.splitlines()[0]
    if 'name=NVIDIA H200' not in header:
        raise ValueError(f'{path}: not an H200 capture: {header}')
    rows = [
        {'threads': int(m[0]), 'iterations': int(m[1]),
         'dependent_cycles': int(m[2]), 'dependent_cycles_per_scalar': float(m[3]),
         'independent_cycles': int(m[4]),
         'independent_cycles_per_scalar': float(m[5])}
        for m in LINE.findall(text)]
    if len(rows) != 4 * 7:
        raise ValueError(f'{path}: expected 28 result lines, got {len(rows)}')
    return header, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    clock_mhz = manifest['clock_gate']['measured_sm_clock_mhz']
    rotations = []
    for raw in manifest['raw']:
        path = args.run_dir / Path(raw['path']).name
        if sha256(path) != raw['sha256']:
            raise ValueError(f'{path}: raw SHA256 mismatch')
        header, rows = parse(path)
        selected = [r for r in rows if r['threads'] == SELECTED_THREADS
                    and r['iterations'] == SELECTED_ITERATIONS]
        rotations.append({'rotation': raw['rotation'], 'header': header,
                          'raw_sha256': raw['sha256'], 'rows': rows,
                          'selected_dependent_cycles_per_scalar':
                              selected[0]['dependent_cycles_per_scalar']})
    values = [r['selected_dependent_cycles_per_scalar'] for r in rotations]
    cycles = statistics.median(values)
    spread = (max(values) - min(values)) / cycles
    if spread > 0.10:
        raise ValueError(f'rotations disagree by {spread:.3%}')
    reference = {
        'schema': 'nvidia_h200_silu_scalar_round_reference_v1',
        'hardware_id': 'nvidia_h200',
        'production_or_evaluation_duration_consumed': False,
        'mechanism': 'register_resident_dependent_scalar_silu_round_block1024',
        'a800_counterpart': {
            'value_cycles_per_round': 249.125,
            'artifact': 'profiler_output/microbench/nvidia_a800/'
                        'silu_scalar_latency_0ea587f0_20260825_r1/run/silu_scalar_latency.txt',
            'rate_rule': '1.14e9 / 249.125 = 4576016.056196689 (A800 model clock)',
        },
        'selection_rule': (
            'dependent_cycles_per_scalar of the threads=1024, iterations=64 line '
            '((dependent - zero-iteration baseline) / 64, median of 1000 launches); '
            'median over independent process rotations'),
        'selected_services': {
            'silu_scalar_round': {
                'cycles_per_round': cycles,
                'measured_sm_clock_mhz': clock_mhz,
                'ns_per_round': cycles * 1000.0 / clock_mhz,
                'operations_per_second_at_1_98ghz': clock_mhz * 1.0e6 / cycles,
            },
        },
        'rotations': [{k: v for k, v in r.items() if k != 'rows'} for r in rotations],
        'rotation_relative_spread': spread,
        'curves_1024_threads': [r for r in rotations[0]['rows']
                                if r['threads'] == SELECTED_THREADS],
        'provenance': {
            'source': 'microbench/nvidia_h200/compute/sfu/silu_scalar_latency.cu',
            'source_sha256': manifest['source_sha256'],
            'runner': 'microbench/nvidia_h200/compute/sfu/silu_scalar_latency_runner.py',
            'runner_sha256': manifest['runner_sha256'],
            'reporter': 'microbench/nvidia_h200/compute/sfu/silu_scalar_latency_report.py',
            'reporter_sha256': sha256(REPORT_PATH),
            'baseline_commit': manifest['baseline_commit'],
            'device': manifest['device'],
            'nvcc_version': manifest['build']['nvcc_version'],
            'compile_command': manifest['build']['compile_command'],
            'remote_raw_dir': 'profiler_output/microbench/nvidia_h200/'
                              + manifest['run_id'],
            'local_raw_dir': 'profiler_output/microbench/nvidia_h200/' + manifest['run_id'],
            'clock_gate': manifest['clock_gate'],
        },
    }
    args.output.write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps(reference['selected_services'], indent=1), spread)


if __name__ == '__main__':
    main()
