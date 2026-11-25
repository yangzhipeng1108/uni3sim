#!/usr/bin/env python3
"""Derive the H200 CUDA-graph small-node span reference in the A800 schema.

Reads ``raw/rotation_{0,1,2}.csv`` (+ ``.clock.csv``) written by the probe
``microbench/nvidia_h200/runtime/cuda_graph_small_node_span.cu`` and applies
the rule of ``microbench/nvidia_a800/references/
cuda_graph_small_node_span_sm80.json``: per node span = batch time / 64;
``minimum_graph_node_span_ns`` is the minimum graph span across trials and
rotations, the two medians are over every trial of every rotation.  (Applied
to the retained A800 raw this rule reproduces every A800 published number.)
Rotations failing the clock gate (no busy sample, or a busy sample below
1980 MHz) are excluded and counted.

Usage:
    python cuda_graph_small_node_span_reference.py --run-dir <dir> \
        --run-path <repo-relative run dir> --output <json>
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[3]
SOURCE = 'microbench/nvidia_h200/runtime/cuda_graph_small_node_span.cu'
RUNNER = 'microbench/nvidia_h200/runtime/cuda_graph_small_node_span_runner.py'
ROTATIONS = 3
BOOST_MHZ = 1980


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rotation(path: Path, nodes_expected: int):
    lines = path.read_text().splitlines()
    if lines[0] != 'device,NVIDIA H200':
        raise ValueError(f'{path}: unexpected device line {lines[0]!r}')
    if lines[1] != 'threads_per_cta,128' or lines[2] != f'nodes_per_batch,{nodes_expected}':
        raise ValueError(f'{path}: probe geometry changed')
    if lines[3] != 'grid_ctas,mode,trial_index,batch_duration_ms':
        raise ValueError(f'{path}: header changed')
    rows = []
    for line in lines[4:]:
        grid, mode, trial, ms = line.split(',')
        if mode not in ('graph', 'standalone'):
            raise ValueError(f'{path}: unknown mode {mode}')
        rows.append((int(grid), mode, int(trial), float(ms) * 1.0e6 / nodes_expected))
    return rows


def clock_status(path: Path):
    busy = [int(line.split(',')[0]) for line in path.read_text().splitlines()
            if line.strip() and int(line.split(',')[1]) > 0]
    return {'busy_samples': len(busy),
            'busy_clock_mhz_min': min(busy) if busy else None,
            'busy_clock_mhz_max': max(busy) if busy else None,
            'accepted': bool(busy) and min(busy) >= BOOST_MHZ}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-path', required=True,
                        help='repository-relative path recorded in provenance')
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--capture-note', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    nodes = 64
    captures = []
    clocks = []
    rows = []
    rejected = 0
    for rotation in range(ROTATIONS):
        csv_path = args.run_dir / 'raw' / f'rotation_{rotation}.csv'
        status = clock_status(args.run_dir / 'raw' / f'rotation_{rotation}.clock.csv')
        clocks.append({'rotation': rotation, **status})
        captures.append({
            'device_index': 0,
            'path': f'{args.run_path}/raw/rotation_{rotation}.csv',
            'raw_sha256': sha256(csv_path), 'rotation': rotation})
        if not status['accepted']:
            rejected += 1
            continue
        rows.extend(read_rotation(csv_path, nodes))
    grids = sorted({row[0] for row in rows})
    trials = sorted({row[2] for row in rows})
    table = []
    for grid in grids:
        graph = [r[3] for r in rows if r[0] == grid and r[1] == 'graph']
        standalone = [r[3] for r in rows if r[0] == grid and r[1] == 'standalone']
        if len(graph) != len(standalone) or not graph:
            raise ValueError(f'grid {grid}: unpaired modes')
        table.append({
            'grid_ctas': grid,
            'median_graph_node_span_ns': float(statistics.median(graph)),
            'median_standalone_span_ns': float(statistics.median(standalone)),
            'minimum_graph_node_span_ns': float(min(graph)),
        })
    floor = json.loads((REPO / 'microbench/nvidia_h200/references/'
                        'small_kernel_span_floor_sm90.json').read_text())
    floor_by_grid = {row['grid_ctas']: row['minimum_span_ns']
                     for row in floor['derived_model']['floor_by_grid_ctas']}
    ratios = [row['minimum_graph_node_span_ns'] / floor_by_grid[row['grid_ctas']]
              for row in table if row['grid_ctas'] in floor_by_grid]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    reference = {
        'allowed_use': (
            'A lower bound on the span of a small memory-touching kernel that '
            'executes as a node of a replayed CUDA graph, given its CTA grid. It '
            'composes with an execution time by maximum and never adds. It replaces '
            'the standalone small_kernel_span_floor only for graph-replayed '
            'launches; a launch issued singly into a stream keeps that floor.'),
        'captures': captures,
        'compute_capability': 'sm_90',
        'derived_model': {
            'note': (
                'On H200 the minimum graph-node span is '
                f'{min(ratios):.2f}-{max(ratios):.2f} times the standalone nsys '
                'small_kernel_span_floor_sm90 minimum at the grids both references '
                'measure (the two use different timers: CUDA-event batch / 64 here, '
                'nsys kernel duration there).'),
            'span_ns_by_grid_ctas': table,
        },
        'device_name': 'NVIDIA H200',
        'forbidden_use': (
            'An additive launch cost, a typical or median duration, a host launch '
            'latency, a floor for a standalone launch, or a floor for a launch with '
            'no global memory access.'),
        'hardware_id': 'nvidia_h200',
        'measurement': {
            'devices': [0],
            'grids_measured': grids,
            'nodes_per_batch': nodes,
            'rotations': ROTATIONS,
            'threads_per_cta': 128,
            'trials_per_cell': len(trials),
            'warmup_rounds': 200,
        },
        'mechanism': 'cuda_graph_small_node_span',
        'method': (
            'One identical trivial memory-touching kernel (one aligned 16-byte load, '
            'one dependent integer op, one aligned 16-byte store per active thread) '
            'is timed two ways at the same grid and block: 64 separate launches into '
            'a stream, and the same 64 launches captured once into a CUDA graph and '
            'replayed. Both are warmed 200 rounds and the two modes alternate inside '
            'every trial so they see the same clock state. The published value is '
            'the minimum graph span across trials and rotations, so it is a floor '
            'rather than a typical duration, which is what composing by maximum '
            'requires. The A800 two-device rotation becomes three independent '
            'processes on the single owned H200 GPU.'),
        'production_or_evaluation_duration_consumed': False,
        'production_predictor_consumption_allowed': True,
        'provenance': {
            'accepted_rotations': ROTATIONS - rejected,
            'capture_note': args.capture_note,
            'clock_gate': (f'nvidia-smi -lms 50 per rotation; reject a rotation with '
                           f'no busy sample or a busy sample below {BOOST_MHZ} MHz'),
            'rotation_clock': clocks,
            'derivation': 'microbench/nvidia_h200/runtime/cuda_graph_small_node_span_reference.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'git_commit': commit,
            'host_contention': ('GPUs 1-3 of the same host were used concurrently by '
                                'other workers; the published value is a minimum'),
            'nvcc': 'CUDA 12.2, -O3 -arch=sm_90 -cudart static',
            'rejected_rotations': rejected,
            'remote_path': args.remote_path,
            'runner': RUNNER,
            'runner_sha256': sha256(REPO / RUNNER),
            'source': SOURCE,
            'source_sha256': sha256(REPO / SOURCE),
        },
        'schema': 'nvidia_h200_cuda_graph_small_node_span_reference_v1',
    }
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + '\n')
    for row in table:
        print(row)
    print('rejected rotations', rejected)


if __name__ == '__main__':
    main()
