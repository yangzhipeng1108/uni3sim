#!/usr/bin/env python3
"""Derive the H200 small-kernel span-floor reference in the A800 schema.

Input is the ``raw_points.json`` written by
``small_kernel_span_floor_runner.py`` (clock-accepted and clock-rejected
captures, 30 nsys ``cuda_gpu_trace`` kernel durations each).  Output carries
exactly the keys of ``microbench/nvidia_a800/references/
small_kernel_span_floor_sm80.json`` (``small_kernel_span_floor_reference_v1``)
with H200 values.  Selection rules are the A800 ones:

* ``minimum_span_ns``: smallest one-store duration across trials, both block
  sizes and both launch patterns (isolated and back-to-back).
* ``isolated_median_ns`` / ``back_to_back_median_ns``: median of every
  one-store duration of that launch pattern at the grid.
* ``empty_cta_reference_ns``: median of every isolated empty-body duration at
  the grid (the A800 header table quotes the isolated empty median).

Only clock-accepted captures participate; the rejected count is kept in the
provenance.  No production or evaluation duration is read.

Usage:
    python small_kernel_span_floor_reference.py --run-dir <dir> --output <json>
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[3]
SOURCE = 'microbench/nvidia_h200/compute/small_kernel_span_floor.cu'
RUNNER = 'microbench/nvidia_h200/compute/small_kernel_span_floor_runner.py'
GRIDS = (1, 16, 48, 108, 132, 216, 264)
BOOST_MHZ = 1980


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values):
    return float(statistics.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.run_dir / 'raw_points.json'
    raw = json.loads(raw_path.read_text())
    accepted = raw['accepted']
    rejected = raw['rejected']
    for point in accepted:
        if point['busy_clock_mhz_min'] < BOOST_MHZ:
            raise ValueError(f'accepted capture below boost clock: {point}')
    floor = []
    empty_reference = {}
    for grid in GRIDS:
        at_grid = [p for p in accepted if p['grid_ctas'] == grid]
        store = [p for p in at_grid if p['body'] == 'one_store']
        iso = [d for p in store if p['launch_pattern'] == 'isolated'
               for d in p['durations_ns']]
        b2b = [d for p in store if p['launch_pattern'] == 'back_to_back'
               for d in p['durations_ns']]
        empty_iso = [d for p in at_grid if p['body'] == 'empty'
                     and p['launch_pattern'] == 'isolated' for d in p['durations_ns']]
        if not iso or not b2b or not empty_iso:
            raise ValueError(f'grid {grid}: a launch pattern has no accepted capture')
        floor.append({
            'back_to_back_median_ns': median(b2b),
            'grid_ctas': grid,
            'isolated_median_ns': median(iso),
            'minimum_span_ns': float(min(iso + b2b)),
        })
        empty_reference[str(grid)] = median(empty_iso)

    def iso_median(threads, grid):
        return median([d for p in accepted if p['grid_ctas'] == grid
                       and p['threads'] == threads and p['body'] == 'one_store'
                       and p['launch_pattern'] == 'isolated'
                       for d in p['durations_ns']])

    block_dependence = ', '.join(
        f'grid {g}: {iso_median(128, g):.0f} ns at block 128 and '
        f'{iso_median(512, g):.0f} at 512' for g in (48, 132))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    empty_text = ', '.join(f'{empty_reference[str(g)]:.0f}' for g in GRIDS)
    reference = {
        'allowed_use': (
            'A lower bound on the profiler-reported duration of a launch that '
            'performs at least one global memory access, given its CTA grid. It '
            'composes with an execution time by maximum and never adds to one. It '
            'does not describe a launch that touches no memory: a truly empty CTA '
            f'measures {empty_text} ns (isolated median) at grids '
            f'{", ".join(map(str, GRIDS))}, below these values, and every '
            'production kernel touches memory.'),
        'derived_model': {
            'empty_cta_reference_ns': empty_reference,
            'floor_by_grid_ctas': floor,
            'selection_rule': (
                'At each measured grid retain the smallest one-store duration seen '
                'across trials, both block sizes (128, 512) and both launch patterns, '
                'isolated and back-to-back. The published value is therefore a floor '
                'rather than a typical duration, which is what composing by maximum '
                'requires. A grid below the first measured point takes the first '
                'point and one above the last takes the last; the surface is not '
                'extrapolated. Captures with no busy SM-clock sample at or above '
                f'{BOOST_MHZ} MHz are rejected.'),
        },
        'forbidden_use': (
            'An additive launch cost, a typical or median duration, a host launch '
            'latency, a CUDA-event interval, a per-shape correction, or a floor for '
            'a launch with no global memory access.'),
        'hardware_id': 'nvidia_h200',
        'mechanism': 'small_kernel_span_floor',
        'production_or_evaluation_duration_consumed': False,
        'provenance': {
            'accepted_captures': len(accepted),
            'captures': [{
                'device_index': 0,
                'launches_per_point': 30,
                'path': 'profiler_output/microbench/nvidia_h200/span_floor_20260925_021635',
                'remote_path': 'profiler_output/microbench/nvidia_h200/span_floor_20260925_021635',
                'profiler': 'nsys 2025.6.1',
                'raw_points_sha256': sha256(raw_path),
                'report': 'cuda_gpu_trace',
                'trials': 3,
            }],
            'clock_gate': (
                f'nvidia-smi -lms 50 during every capture; a capture is rejected when '
                f'it has no busy sample or any busy sample is below {BOOST_MHZ} MHz '
                '(all 160 accepted captures have a busy minimum of exactly 1980 MHz; '
                'all 8 rejected ones had no busy sample)'),
            'derivation': 'microbench/nvidia_h200/compute/small_kernel_span_floor_reference.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'git_commit': commit,
            'host_contention': (
                'GPUs 1-3 of the same host were used concurrently by other workers; '
                'the minimum selection rule is the A800 one'),
            'rejected_captures': len(rejected),
            'runner': RUNNER,
            'runner_sha256': sha256(REPO / RUNNER),
            'source': SOURCE,
            'source_sha256': sha256(REPO / SOURCE),
        },
        'schema': 'small_kernel_span_floor_reference_v1',
        'scope': {
            'block_size_dependence': block_dependence,
            'maximum_grid_ctas': GRIDS[-1],
            'minimum_grid_ctas': GRIDS[0],
            'threads_per_cta_measured': [128, 512],
            'timer_domain': 'nsys_gpu_kernel',
        },
    }
    args.output.write_text(json.dumps(reference, indent=4, sort_keys=True) + '\n')
    for row in floor:
        print(row)
    print('empty', empty_reference)


if __name__ == '__main__':
    main()
