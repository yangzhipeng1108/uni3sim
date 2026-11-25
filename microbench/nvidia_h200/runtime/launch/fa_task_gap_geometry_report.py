#!/usr/bin/env python3
"""Derive the H200 FA2 source-task outer envelope from the probe capture.

Input: ``samples.csv`` + ``capture.json`` of ``fa_task_gap_geometry_runner.py``.
Procedure (A800 yaml ``fa2_source_task_outer_envelope``): per launch, the
outer envelope is the nsys whole-kernel span minus the maximum CTA-internal
globaltimer envelope.  Per (geometry, grid, spin length) cell the value is the
median over the accepted launches of both rotations; the probe header admits
it as a launch-wide fill/drain span only when it is invariant under the spin
length, which is reported per (geometry, grid) as the difference between the
two spin lengths.  The published ``maximum_outer_envelope_ns`` is the maximum
over every cell with grid 8..64, i.e. over both FA2 geometries (main, combine)
and both spin lengths.

Output keys: those of the yaml entry (model, duration_unit, timer_domain,
software_revision, threads_per_cta, minimum/maximum_grid_size,
maximum_outer_envelope_ns, source_sha256, binary_sha256) plus the per-cell
table and provenance; the A800 value has no JSON reference.

Usage:
    python fa_task_gap_geometry_report.py --run-dir <dir> --run-path <p> \
        --binary-sha256 <sha> --output <json>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[4]
SOURCE = 'microbench/nvidia_h200/runtime/launch/fa_task_gap_geometry.cu'
RUNNER = 'microbench/nvidia_h200/runtime/launch/fa_task_gap_geometry_runner.py'
MIN_GRID = 8
MAX_GRID = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-path', required=True)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads((args.run_dir / 'capture.json').read_text())
    accepted = sum(c['accepted'] for c in capture['captures'])
    cells = {}
    union = {}
    for row in csv.DictReader((args.run_dir / 'samples.csv').open()):
        if row['accepted'] != '1':
            continue
        key = (int(row['main_geometry']), int(row['grid']), int(row['body_cycles']))
        if key not in cells:
            cells[key] = []
            union[key] = []
        nsys = int(row['nsys_duration_ns'])
        cells[key].append(nsys - int(row['max_cta_globaltimer_envelope_ns']))
        union[key].append(nsys - int(row['union_globaltimer_envelope_ns']))
    expected = {(g, grid, b) for g in (0, 1) for grid in capture['grids']
                for b in capture['body_cycles']}
    if set(cells) != expected:
        raise ValueError(f'cells without an accepted capture: {sorted(expected - set(cells))}')
    table = []
    for key in sorted(cells):
        geometry, grid, body = key
        table.append({
            'geometry': 'main_80KiB_dynamic_smem' if geometry else 'combine_static_smem',
            'grid': grid, 'body_cycles': body, 'launches': len(cells[key]),
            'median_outer_envelope_ns': float(statistics.median(cells[key])),
            'minimum_outer_envelope_ns': float(min(cells[key])),
            'maximum_outer_envelope_ns': float(max(cells[key])),
            'median_nsys_minus_union_globaltimer_envelope_ns': float(statistics.median(union[key])),
        })
    bodies = capture['body_cycles']
    invariance = []
    for geometry in (1, 0):
        for grid in capture['grids']:
            values = [statistics.median(cells[(geometry, grid, b)]) for b in bodies]
            invariance.append({'geometry': 'main_80KiB_dynamic_smem' if geometry
                               else 'combine_static_smem', 'grid': grid,
                               'spin_length_spread_ns': float(max(values) - min(values))})
    in_range = [row for row in table if MIN_GRID <= row['grid'] <= MAX_GRID]
    maximum = max(row['median_outer_envelope_ns'] for row in in_range)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    reference = {
        'schema': 'nvidia_h200_fa2_source_task_outer_envelope_reference_v1',
        'hardware_id': 'nvidia_h200',
        'model': 'nsys_source_task_outer_envelope_v1',
        'duration_unit': 'ns_per_launch',
        'timer_domain': 'nsys_gpu_kernel_and_globaltimer_cta_envelope',
        'software_revision': 'cuda_12_2_sm90',
        'threads_per_cta': 128,
        'minimum_grid_size': MIN_GRID,
        'maximum_grid_size': MAX_GRID,
        'maximum_outer_envelope_ns': maximum,
        'source_sha256': sha256(REPO / SOURCE),
        'binary_sha256': args.binary_sha256,
        'production_or_evaluation_duration_consumed': False,
        'derivation_rule': (
            'per launch: nsys kernel duration minus max over CTAs of (globaltimer end - '
            'begin); per (geometry, grid, spin length): median over accepted launches '
            'of both rotations; published: maximum over grids 8..64, both geometries '
            'and both spin lengths'),
        'cells': table,
        'spin_length_invariance': invariance,
        'provenance': {
            'capture': capture,
            'accepted_captures': accepted,
            'rejected_captures': len(capture['captures']) - accepted,
            'clock_gate': ('nvidia-smi -lms 50 per capture; reject a capture with no '
                           'busy sample or a busy sample below 1980 MHz'),
            'derivation': 'microbench/nvidia_h200/runtime/launch/fa_task_gap_geometry_report.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'git_commit': commit,
            'host_contention': ('GPUs 1-3 of the same host were used concurrently by '
                                'other workers'),
            'nvcc': 'CUDA 12.2, -O3 -std=c++17 -arch=sm_90 -cudart static',
            'raw_path': args.run_path,
            'remote_path': args.remote_path,
            'runner': RUNNER,
            'runner_sha256': sha256(REPO / RUNNER),
            'samples_sha256': sha256(args.run_dir / 'samples.csv'),
            'a800_procedure_note': (
                'The A800 probe has no runner or report and its grid list, spin '
                'lengths and sample counts are not recorded; the grids 8..64, the '
                'two spin lengths and the median-per-cell / maximum-over-cells rule '
                'are this worker\'s scripted reading of the yaml comment and probe '
                'header.'),
        },
    }
    args.output.write_text(json.dumps(reference, indent=1) + '\n')
    for row in table:
        print(row)
    print('invariance', invariance)
    print('maximum_outer_envelope_ns', maximum)


if __name__ == '__main__':
    main()
