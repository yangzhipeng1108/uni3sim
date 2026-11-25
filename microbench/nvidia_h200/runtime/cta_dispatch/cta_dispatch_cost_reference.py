#!/usr/bin/env python3
"""Derive the H200 CTA dispatch-cost reference in the A800 schema.

Input: ``trial_{1,2,3}.csv`` + ``capture.json`` from
``cta_dispatch_cost_runner.py``.  Rule of ``microbench/nvidia_a800/references/
cta_dispatch_cost_sm80.json`` (``median_over_block_sizes_minus_single_cta``):

* only the chain-64, zero-dynamic-shared-memory rows are used (shortest CTA,
  so the cost is not hidden by execution);
* for every derivation trial (1, 2) and block size, the cost at a grid is the
  per-launch time at that grid minus the per-launch time of the single-CTA
  launch at the same block and trial;
* the published cost at a grid is the median of those differences over block
  sizes and derivation trials, clamped at zero and made monotone by a running
  maximum over increasing grid;
* the tabled grids are the A800 grids with the full-chip point 108 -> 132
  (1, 2, 4, 8, 16, 32, 64, 132); the extra H200 capture point 108 is not
  tabled (it is not an A800 grid) and is reported in provenance, because on
  H200 it measures *above* grid 132 and a running maximum would carry it into
  the full-chip entry;
* the table stops at the full-chip width (132); trial 3 validates
  (``max_cross_trial_spread_ns`` = largest spread of the three per-trial
  median-over-blocks costs at any tabled grid).

Usage:
    python cta_dispatch_cost_reference.py --run-dir <dir> --run-path <p> --output <json>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[4]
PROBE = 'microbench/nvidia_h200/runtime/cta_dispatch/cta_dispatch_cost.cu'
RUNNER = 'microbench/nvidia_h200/runtime/cta_dispatch/cta_dispatch_cost_runner.py'
FULL_CHIP = 132
CHAIN = 64
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
BLOCKS = (128, 256, 512, 1024)
SMEMS = (0, 16384)
GRIDS = (1, 2, 4, 8, 16, 32, 64, 108, 132, 264)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trial(path: Path):
    table = {}
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames) != ('grid', 'block', 'smem', 'chain', 'per_launch_ns'):
            raise ValueError(f'{path}: header changed')
        for row in reader:
            key = (int(row['grid']), int(row['block']), int(row['smem']), int(row['chain']))
            if key in table:
                raise ValueError(f'{path}: duplicate {key}')
            table[key] = float(row['per_launch_ns'])
    expected = {(g, b, s, c) for g in GRIDS for b in BLOCKS for s in SMEMS
                for c in (64, 1024, 16384, 262144)}
    if set(table) != expected:
        raise ValueError(f'{path}: sweep matrix incomplete')
    return table


def cost(table, grid, block, smem):
    return table[(grid, block, smem, CHAIN)] - table[(1, block, smem, CHAIN)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-path', required=True)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads((args.run_dir / 'capture.json').read_text())
    if not all(t['accepted'] for t in capture['trials']):
        raise ValueError('a trial failed the 1980 MHz clock gate')
    trials = {t: read_trial(args.run_dir / f'trial_{t}.csv') for t in (1, 2, 3)}
    tabled = [g for g in GRIDS if g <= FULL_CHIP and g != 108]
    table = {}
    running = 0.0
    for grid in tabled:
        value = statistics.median(cost(trials[t], grid, b, 0)
                                  for t in DERIVATION_TRIALS for b in BLOCKS)
        running = max(running, value, 0.0)
        table[str(grid)] = round(running, 1)
    per_trial = {t: {g: statistics.median(cost(trials[t], g, b, 0) for b in BLOCKS)
                     for g in tabled} for t in (1, 2, 3)}
    spread = max(max(per_trial[t][g] for t in per_trial) - min(per_trial[t][g] for t in per_trial)
                 for g in tabled)
    validation_error = {str(g): round(per_trial[VALIDATION_TRIAL][g] - table[str(g)], 1)
                        for g in tabled}
    full_chip_by_block = [statistics.mean(cost(trials[t], FULL_CHIP, b, 0)
                                          for t in DERIVATION_TRIALS) for b in BLOCKS]
    smem_ratio = [statistics.mean(cost(trials[t], FULL_CHIP, b, 16384) for t in DERIVATION_TRIALS)
                  / statistics.mean(cost(trials[t], FULL_CHIP, b, 0) for t in DERIVATION_TRIALS)
                  - 1.0 for b in BLOCKS]
    two_wave = {str(b): round(statistics.mean(cost(trials[t], 264, b, 0)
                                              for t in DERIVATION_TRIALS), 1) for b in BLOCKS}
    long_cta = {str(c): round(statistics.median(
        trials[t][(FULL_CHIP, b, 0, c)] - trials[t][(1, b, 0, c)]
        for t in DERIVATION_TRIALS for b in BLOCKS), 1) for c in (64, 1024, 16384, 262144)}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    reference = {
        'chain_length_used': CHAIN,
        'cost_ns_by_grid': table,
        'derivation_rule': 'median_over_block_sizes_minus_single_cta',
        'derivation_trials': list(DERIVATION_TRIALS),
        'domain_note': (
            'the table stops at the full-chip width because beyond one wave the cost '
            'becomes residency-dependent, which cta_round_ns already models; larger '
            'grids reuse the last entry via the step-down rule'),
        'duration_unit': 'ns_per_launch',
        'full_chip_width': FULL_CHIP,
        'geometry_independence': {
            'block_sizes_swept': list(BLOCKS),
            'cost_range_over_blocks_at_full_chip_ns': [min(full_chip_by_block),
                                                       max(full_chip_by_block)],
            'dynamic_smem_bytes_swept': list(SMEMS),
            'note': (
                'cost at the full-chip grid varies by this much across block sizes 128 '
                'to 1024; 16 KiB of dynamic shared memory changes it by '
                f'{100 * min(smem_ratio):+.1f} to {100 * max(smem_ratio):+.1f} percent '
                'across the four block sizes'),
        },
        'hardware_id': 'nvidia_h200',
        'lookup_rule': 'step_down_largest_sampled_grid_not_exceeding_request',
        'max_cross_trial_spread_ns': round(spread, 1),
        'model': 'geometry_independent_cta_dispatch_cost_v1',
        'monotonicity': ('running maximum applied: dispatching more CTAs cannot cost '
                         'less, so noise is not allowed to invert the table'),
        'probe': PROBE,
        'probe_sha256': sha256(REPO / PROBE),
        'production_or_evaluation_duration_consumed': False,
        'provenance': {
            'capture': capture,
            'clock_gate': ('nvidia-smi -lms 50 per trial; reject a trial with no busy '
                           'sample or a busy sample below 1980 MHz'),
            'derivation': 'microbench/nvidia_h200/runtime/cta_dispatch/cta_dispatch_cost_reference.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'full_chip_cost_by_chain_length_ns': long_cta,
            'git_commit': commit,
            'host_contention': ('GPUs 1-3 of the same host were used concurrently by '
                                'other workers'),
            'nvcc': 'CUDA 12.2, -O3 -arch=sm_90 -cudart static',
            'raw_path': args.run_path,
            'remote_path': args.remote_path,
            'runner': RUNNER,
            'runner_sha256': sha256(REPO / RUNNER),
            'two_wave_grid_264_cost_by_block_ns': two_wave,
            'untabled_grid_108_cost_by_trial_ns': {
                str(t): round(statistics.median(cost(trials[t], 108, b, 0) for b in BLOCKS), 1)
                for t in (1, 2, 3)},
            'untabled_grid_108_note': (
                'grid 108 (the A800 full-chip width, a partial wave on 132 SMs) costs '
                'more than the H200 full-chip grid 132 at every block size and trial; '
                'it is kept out of the step-down table because it is not an A800 grid, '
                'and including it would, through the running maximum, raise every '
                'entry from 108 upward to its value'),
            'validation_trial_minus_published_ns': validation_error,
        },
        'raw_sha256': {f'trial_{t}.csv': sha256(args.run_dir / f'trial_{t}.csv')
                       for t in (1, 2, 3)},
        'schema_version': 1,
        'software_revision': 'cuda_12.2_sm90',
        'target_operator_used': False,
        'timer_domain': 'paired_differential_cuda_graph_replay',
        'validation_trial': VALIDATION_TRIAL,
    }
    args.output.write_text(json.dumps(reference, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'table': table, 'spread': spread, 'full_chip_by_block': full_chip_by_block,
                      'smem_ratio': smem_ratio, 'validation': validation_error,
                      'two_wave': two_wave, 'long_cta': long_cta}, indent=1))


if __name__ == '__main__':
    main()
