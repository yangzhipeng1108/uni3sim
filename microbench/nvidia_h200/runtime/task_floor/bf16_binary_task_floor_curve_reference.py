#!/usr/bin/env python3
"""Derive the H200 BF16 binary TensorIterator task-floor curve (A800 schema).

Input: ``curve2_trial_{1..6}.json`` written by ``bf16_binary_task_floor_runner.py``
(one process per trial, elements 2^11..2^25, repeats 100 and 500, so two
paired-differential samples per element per trial).  Rule of
``microbench/nvidia_a800/references/
bf16_binary_tensor_iterator_task_floor_curve_sm80.json``
(``minimum_all_binary_graph_replay_differentials_per_element``): trials 1-5
derive, the floor at an element count is the minimum of its 10 derivation
samples; trial 6 validates (samples below the floor are counted, the largest
shortfall reported).  A trial whose busy SM clock has no sample or a sample
below 1980 MHz is rejected.

Usage:
    python bf16_binary_task_floor_curve_reference.py --run-dir <dir> --run-path <p> \
        --launch-geometry <json> --output <json>
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNNER = 'microbench/nvidia_h200/runtime/task_floor/bf16_binary_task_floor_runner.py'
DERIVATION_TRIALS = (1, 2, 3, 4, 5)
VALIDATION_TRIAL = 6
L2_BYTES = 62914560
BOOST_MHZ = 1980


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-path', required=True)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--launch-geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trials = {}
    clock = []
    for trial in DERIVATION_TRIALS + (VALIDATION_TRIAL,):
        path = args.run_dir / f'curve2_trial_{trial}.json'
        document = json.loads(path.read_text())
        if document['hardware_id'] != 'nvidia_h200' or document['device_name'] != 'NVIDIA H200':
            raise ValueError(f'{path}: not an H200 capture')
        accepted = (document['busy_clock_samples'] > 0
                    and document['busy_clock_mhz_min'] >= BOOST_MHZ)
        clock.append({'trial': trial, 'busy_clock_samples': document['busy_clock_samples'],
                      'busy_clock_mhz_min': document['busy_clock_mhz_min'],
                      'busy_clock_mhz_max': document['busy_clock_mhz_max'],
                      'accepted': accepted})
        if not accepted:
            raise ValueError(f'trial {trial} failed the {BOOST_MHZ} MHz clock gate')
        trials[trial] = document
    torch_versions = {d['torch_version'] for d in trials.values()}
    if len(torch_versions) != 1:
        raise ValueError(f'torch version changed across trials: {torch_versions}')
    torch_version = torch_versions.pop()
    elements = sorted({s['elements'] for s in trials[1]['samples']})
    derivation = {e: [s['per_task_ns'] for t in DERIVATION_TRIALS
                      for s in trials[t]['samples'] if s['elements'] == e] for e in elements}
    samples_per_element = {len(v) for v in derivation.values()}
    if len(samples_per_element) != 1:
        raise ValueError('uneven derivation samples')
    floor = {e: min(v) for e, v in derivation.items()}
    monotone = all(floor[a] <= floor[b] for a, b in zip(elements, elements[1:]))
    spread = max((max(v) - min(v)) / min(v) for v in derivation.values())
    validation = [s for s in trials[VALIDATION_TRIAL]['samples']]
    below = [s for s in validation if s['per_task_ns'] < floor[s['elements']]]
    shortfall = max([(floor[s['elements']] - s['per_task_ns']) / floor[s['elements']]
                     for s in below] + [0.0])
    crossing = [e for e in elements if 3 * 2 * e > L2_BYTES]
    ratio = max((floor[b] / floor[a], a) for a, b in zip(elements, elements[1:]))
    steepest = (ratio[1], ratio[0])
    geometry_evidence = json.loads(args.launch_geometry.read_text())
    geometry = {key: geometry_evidence[key]
                for key in ('block_threads', 'grid_rule', 'kernel_template', 'note')}
    reference = {
        'derivation_rule': 'minimum_all_binary_graph_replay_differentials_per_element',
        'derivation_samples_per_element': samples_per_element.pop(),
        'derivation_trials': list(DERIVATION_TRIALS),
        'device_name': 'NVIDIA H200',
        'duration_unit': 'ns_per_task',
        'floor_ns_by_elements': {str(e): round(floor[e], 4) for e in elements},
        'hardware_id': 'nvidia_h200',
        'l2_capacity_transition': {
            'elements': crossing[0],
            'note': (
                'three bf16 streams put the working set at '
                f'{6 * crossing[0] // 2 / 1e6:.0f} MB at {crossing[0] // 2} elements and '
                f'{6 * crossing[0] / 1e6:.0f} MB at {crossing[0]}, crossing H200\'s 60 MiB '
                f'({L2_BYTES / 1e6:.1f} MB) L2. On H200 the largest per-doubling step of '
                f'the floor ({steepest[1]:.2f}x) is already between '
                f'{steepest[0]} and {steepest[0] * 2} elements '
                f'({6 * steepest[0] / 1e6:.0f} -> {12 * steepest[0] / 1e6:.0f} MB), one '
                'sample below the nominal capacity crossing; the curve is a measured '
                'table and no breakpoint is chosen from it'),
        },
        'lookup_rule': 'step_down_largest_sampled_element_not_exceeding_request',
        'lookup_rule_note': (
            'step_down_largest_sampled_element_not_exceeding_request; each tabled value is '
            'the minimum measured at that element count, so the value at the largest '
            'sampled size at or below the request is used and no interpolation shape is '
            'chosen. ' + ('The measured curve is monotone non-decreasing in elements.'
                          if monotone else
                          'The measured curve is NOT monotone: ' + ', '.join(
                              f'{b} ({floor[b]:.1f} ns) < {a} ({floor[a]:.1f} ns)'
                              for a, b in zip(elements, elements[1:])
                              if floor[b] < floor[a])
                          + '; no running maximum is applied (the A800 rule has none).')),
        'maximum_cross_sample_spread_fraction': spread,
        'maximum_validation_shortfall_fraction': round(shortfall, 4),
        'model': 'warm_bf16_binary_tensor_iterator_task_floor_curve_v1',
        'monotone_non_decreasing': monotone,
        'probe_launch_geometry': geometry,
        'production_or_evaluation_duration_consumed': False,
        'profiler_attached': False,
        'provenance': {
            'clock_gate': clock,
            'launch_geometry_evidence': {
                'observed': geometry_evidence['observed'],
                'trace': geometry_evidence['trace'],
                'trace_sha256': geometry_evidence['trace_sha256']},
            'clock_gate_rule': ('nvidia-smi -lms 50 during each trial process; reject a '
                                'trial with no busy sample or a busy sample below 1980 MHz'),
            'derivation': 'microbench/nvidia_h200/runtime/task_floor/bf16_binary_task_floor_curve_reference.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'host_contention': ('GPUs 1-3 of the same host were used concurrently by other '
                                'workers; the per-element value is a minimum'),
            'python': 'python',
            'remote_path': args.remote_path,
        },
        'raw_sha256': {f'curve2_trial_{t}.json': sha256(args.run_dir / f'curve2_trial_{t}.json')
                       for t in DERIVATION_TRIALS + (VALIDATION_TRIAL,)},
        'runner': RUNNER,
        'runner_sha256': sha256(REPO / RUNNER),
        'schema_version': 1,
        'software_revision': f'torch_{torch_version}_cuda_sm90',
        'source_capture': args.run_path,
        'target_operator_used': False,
        'task_class': 'pytorch_bf16_binary_tensor_iterator',
        'timer_domain': 'paired_differential_events_around_cuda_graph_replay',
        'torch_version': torch_version,
        'validation_below_floor_count': len(below),
        'validation_sample_count': len(validation),
        'validation_trial': VALIDATION_TRIAL,
        'warm_probe_caveat': (
            'the probe replays the same tensors, so mid-size samples are L2-warm while '
            'production elementwise reads are colder. A warm measurement is faster than a '
            'cold one, so this curve is a conservative lower bound on production task '
            'time, which is the direction a floor requires; it must not be read as a '
            'prediction of production duration'),
    }
    args.output.write_text(json.dumps(reference, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'floor': reference['floor_ns_by_elements'], 'spread': spread,
                      'below': len(below), 'shortfall': shortfall, 'monotone': monotone},
                     indent=1))


if __name__ == '__main__':
    main()
