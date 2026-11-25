#!/usr/bin/env python3
"""Capture and derive the H200 block-256 Add-RMSNorm full+tail pair services.

The A800 probe ``add_rmsnorm_block256_full_tail_pair.cu`` had no runner; its
reference (``add_rmsnorm_block256_full_tail_pair_sm80.json``) was derived by
hand from one ``<binary> <device> <csv>`` run with this rule, scripted here:

    span[cell]       = median of the five samples of maximum_sm_span_ns
    first_pair_ns    = span[first] / 8192
    resident_pair_ns = (span[resident] - span[first]) / ((8 - 1) * 8192)
    Forward rate     = 2 / pair_ns   (one pair = two source-round operations)

(A800 check: add_square tail 64 first = 3466240 / 8192 = 423.125 ns and
resident = (53952512 - 3466240) / 57344 = 880.4107 ns.)

Hardware coupling changed: cells first = 132 and resident = 1056 CTAs, H200
identity, physical GPU 3 of the H200 host, clock-gated capture.  The timer is
``%globaltimer`` (ns), so no clock conversion is involved.

usage (H200 host): python3 add_rmsnorm_block256_full_tail_pair_runner.py \
    --run-id <id> --out-root <dir> --device 3 --baseline-commit <sha>
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('add_rmsnorm_block256_full_tail_pair.cu')
ROUNDS = 8192
SAMPLES = 5
RESIDENT_CTAS = 8
FUNCTIONS = ('add_square', 'scale_multiply')
TAILS = (64, 128)
CELLS = ('first', 'resident')


def derive(path):
    rows = list(csv.DictReader(Path(path).open()))
    if len(rows) != len(FUNCTIONS) * len(TAILS) * len(CELLS) * SAMPLES:
        raise ValueError(f'unexpected row count {len(rows)}')
    services, spreads = {}, {}
    for function in FUNCTIONS:
        for tail in TAILS:
            spans = {}
            for cell in CELLS:
                values = [int(r['maximum_sm_span_ns']) for r in rows
                          if r['function'] == function and int(r['tail_threads']) == tail
                          and r['cell'] == cell]
                if len(values) != SAMPLES:
                    raise ValueError('incomplete cell')
                spans[cell] = median(values)
                spreads[f'{function}_{cell}_{tail}'] = (max(values) - min(values)) / spans[cell]
            first = spans['first'] / ROUNDS
            resident = (spans['resident'] - spans['first']) / ((RESIDENT_CTAS - 1) * ROUNDS)
            if min(first, resident) <= 0:
                raise ValueError('non-positive pair service')
            services[f'{function}_first_{tail}'] = first
            services[f'{function}_resident_{tail}'] = resident
    return services, spreads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--baseline-commit', required=True)
    args = parser.parse_args()
    out = args.out_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    identity = capture.device_identity(args.device)
    build = capture.build(SOURCE_PATH, out / 'build' / 'add_rmsnorm_block256_full_tail_pair')
    warmup = capture.build_warmup(out / 'build')
    raw_path = out / 'raw.csv'
    verdict = capture.run_gated(
        'capture', [([build['binary'], 0, raw_path], out / 'native.stdout')],
        out, args.device, warmup)
    services, spreads = derive(raw_path)
    clock = capture.clock_summary([verdict])
    reference = {
        'schema': 'nvidia_h200_add_rmsnorm_block256_full_tail_pair_reference_v1',
        'hardware_id': 'nvidia_h200',
        'software_revision': 'vllm_0.11.0_source_cuda_12.9_sm90',
        'production_or_evaluation_duration_consumed': False,
        'source_contract': {
            'dtype': 'bf16', 'vector_width_elements': 8, 'block_threads': 256,
            'tail_threads': list(TAILS), 'registers_per_thread': 32,
            'static_shared_memory_bytes': 168, 'resident_ctas_per_sm': RESIDENT_CTAS,
            'operation_unit': 'one_source_round_equivalent',
            'operations_per_measured_pair': 2},
        'artifacts': {
            'tracked_source': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_block256_full_tail_pair.cu',
            'compiled_source_snapshot_sha256': build['source_sha256'],
            'binary_sha256': build['binary_sha256'],
            'raw_csv_sha256': capture.sha256(raw_path),
            'raw_output': f'profiler_output/microbench/nvidia_h200/{args.run_id}',
            'remote_raw_output': str(out),
            'runner': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_block256_full_tail_pair_runner.py',
            'runner_sha256': capture.sha256(RUNNER_PATH),
            'helper_sha256': capture.sha256(capture.__file__),
            'baseline_commit': args.baseline_commit,
            'nvcc_version': build['nvcc_version'], 'ptxas_verbose': build['ptxas_verbose'],
            'device': identity},
        'campaign': {
            'device': args.device, 'timer_domain': 'globaltimer_same_cta',
            'rounds_per_sample': ROUNDS, 'samples_per_cell': SAMPLES, 'cells': list(CELLS),
            'maximum_cross_sample_relative_spread': max(spreads.values()),
            'clock_gate': clock, 'accepted_captures': clock['accepted_captures'],
            'rejected_captures': clock['rejected_captures']},
        'service_parameters': {'unit': 'ns_per_full_tail_pair'},
        'composition_contract': {
            'domain': 'one_complete_256_thread_source_round_followed_by_one_exact_64_or_128_thread_tail',
            'replacement': 'the measured pair replaces the complete round and tail rather than stacking with either service',
            'queued_ctas_supported': False, 'interpolation_allowed': False},
        'decision': {
            'status': 'accepted', 'forward_eligible': True,
            'reason': ('The independent loop isolates two adjacent source iterations at the '
                       'exact production widths while preserving the static CTA resource '
                       'identity; it does not implement or time a complete AddRMSNorm operator.')},
    }
    for tail in TAILS:
        for function in FUNCTIONS:
            for cell in CELLS:
                reference['service_parameters'][f'{function}_{cell}_{tail}'] = \
                    services[f'{function}_{cell}_{tail}']
    (out / 'add_rmsnorm_block256_full_tail_pair_sm90.json').write_text(
        json.dumps(reference, indent=2) + '\n')
    print(json.dumps(reference['service_parameters']), json.dumps(clock))


if __name__ == '__main__':
    main()
