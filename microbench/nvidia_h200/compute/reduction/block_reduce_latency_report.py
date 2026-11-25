#!/usr/bin/env python3
"""Derive the H200 block-reduction contract (owner copy of the A800 reporter).

Selection rule, formula and gates are those of
``microbench/nvidia_a800/compute/reduction/block_reduce_latency_report.py``:

* trials 0-1 train, trial 2 validates;
* first-block latency L_B = median grid-1 ``block_median_cycles``;
* resident service = (median same-SM span at grid 132*R_B - L_B) / (R_B - 1);
* queued service = (median span at grid 2112 - resident span) / (16 - R_B);
* validation MAPE <= 10 % and maximum APE <= 30 % on trial 2;
* cycles -> ns with the measured SM clock of the clock-gated capture.

Hardware coupling changed: 132 SMs, H200 SM limits (65,536 registers,
233,472 B shared memory, 2,048 threads, 32 CTAs, 1 KiB reserved shared memory
per CTA) for the resident-CTA count derived from the ptxas resource usage of the
built probe, and the clock gate of the H200 capture replaces the A800 per-cell
clock-equality check.  The resource usage is recorded rather than pinned to the
A800's 24 registers / 168 B because the probe is rebuilt with CUDA 12.9 for
sm_90; the resident counts must still be 8 (256 threads) and 2 (1024 threads)
for the A800 formula to apply unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REPORT_PATH = Path(__file__).resolve()
SM_COUNT = 132
BLOCK_THREADS = (256, 1024)
GRID_MULTIPLIERS = (0, 1, 2, 4, 8, 16)
GRIDS = tuple(1 if m == 0 else SM_COUNT * m for m in GRID_MULTIPLIERS)
TRIALS = (0, 1, 2)
TRAINING_TRIALS = (0, 1)
VALIDATION_TRIALS = (2,)
SAMPLES_PER_CELL = 200
MAX_VALIDATION_MAPE_PERCENT = 10.0
MAX_VALIDATION_MAX_APE_PERCENT = 30.0
SM_REGISTERS = 65_536
SM_SHARED_BYTES = 233_472
CTA_RESERVED_SHARED_BYTES = 1_024
SM_THREADS = 2_048
SM_BLOCKS = 32
EXPECTED_RESIDENT = {'256': 8, '1024': 2}
PTXAS_USAGE = re.compile(r'Used (\d+) registers, (?:used \d+ barriers, )?(\d+) bytes smem')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_usage(ptxas_text):
    matches = PTXAS_USAGE.findall(ptxas_text)
    if len(matches) != 1:
        raise ValueError(f'expected one kernel resource line, got {matches}')
    registers, shared = (int(value) for value in matches[0])
    resident = {}
    for threads in BLOCK_THREADS:
        warps = threads // 32
        registers_per_warp = math.ceil(registers * 32 / 256) * 256
        by_registers = (SM_REGISTERS // registers_per_warp) // warps
        by_shared = SM_SHARED_BYTES // (shared + CTA_RESERVED_SHARED_BYTES)
        resident[str(threads)] = min(by_registers, by_shared,
                                     SM_THREADS // threads, SM_BLOCKS)
    if resident != EXPECTED_RESIDENT:
        raise ValueError(f'resident CTA limits changed: {resident}')
    return {'registers_per_thread': registers,
            'shared_memory_bytes_per_cta': shared,
            'sm_register_capacity': SM_REGISTERS,
            'sm_shared_memory_bytes': SM_SHARED_BYTES,
            'sm_thread_capacity': SM_THREADS, 'sm_block_capacity': SM_BLOCKS,
            'resident_ctas_by_threads': resident}


def read_raw(path):
    rows = []
    seen = set()
    with Path(path).open(newline='') as stream:
        for raw in csv.DictReader(stream):
            row = {key: int(raw[key]) for key in (
                'trial', 'position', 'sample', 'sm_count', 'threads', 'grid',
                'block_median_cycles', 'sm_local_span_cycles', 'grid_span_ns')}
            if raw['validation_status'] != 'pass' or row['sm_count'] != SM_COUNT:
                raise ValueError(f'invalid raw row {raw}')
            key = (row['trial'], row['threads'], row['grid'], row['sample'])
            if key in seen:
                raise ValueError(f'duplicate sample {key}')
            seen.add(key)
            rows.append(row)
    expected = {(t, b, g, s) for t in TRIALS for b in BLOCK_THREADS for g in GRIDS
                for s in range(SAMPLES_PER_CELL)}
    if seen != expected:
        raise ValueError('raw matrix incomplete')
    return rows


def parameters(rows, resident_by_threads, clock_mhz):
    training = [row for row in rows if row['trial'] in TRAINING_TRIALS]
    maximum_grid = GRIDS[-1]
    maximum_rounds = maximum_grid // SM_COUNT
    result = []
    for threads in BLOCK_THREADS:
        resident = resident_by_threads[str(threads)]
        resident_grid = SM_COUNT * resident
        first = statistics.median(r['block_median_cycles'] for r in training
                                  if r['threads'] == threads and r['grid'] == 1)
        resident_span = statistics.median(
            r['sm_local_span_cycles'] for r in training
            if r['threads'] == threads and r['grid'] == resident_grid)
        maximum_span = statistics.median(
            r['sm_local_span_cycles'] for r in training
            if r['threads'] == threads and r['grid'] == maximum_grid)
        resident_service = (resident_span - first) / (resident - 1)
        queued_service = (maximum_span - resident_span) / (maximum_rounds - resident)
        if not 0 < resident_service < first:
            raise ValueError(f'resident service not positive/hidden: {threads}: '
                             f'{first} {resident_service}')
        if not 0 < queued_service < first:
            raise ValueError(f'queued service not positive: {threads}: {queued_service}')
        latency_ns = first * 1000.0 / clock_mhz
        resident_ns = resident_service * 1000.0 / clock_mhz
        queued_ns = queued_service * 1000.0 / clock_mhz
        result.append({
            'threads': threads, 'resident_ctas_per_sm': resident,
            'first_block_latency_cycles': first,
            'measured_sm_clock_mhz': clock_mhz,
            'first_block_latency_ns': latency_ns,
            'resident_block_service_cycles_per_sm': resident_service,
            'resident_block_service_ns_per_sm': resident_ns,
            'resident_block_rate_per_sm_s': 1.0e9 / resident_ns,
            'queued_block_service_cycles_per_sm': queued_service,
            'queued_block_service_ns_per_sm': queued_ns,
            'queued_block_rate_per_sm_s': 1.0e9 / queued_ns,
        })
    return result


def predict(row, by_threads):
    p = by_threads[row['threads']]
    n = 1 if row['grid'] == 1 else math.ceil(row['grid'] / SM_COUNT)
    r = p['resident_ctas_per_sm']
    return (p['first_block_latency_cycles']
            + (min(n, r) - 1) * p['resident_block_service_cycles_per_sm']
            + max(0, n - r) * p['queued_block_service_cycles_per_sm'])


def metrics(rows, by_threads):
    signed, absolute, percentages, total = [], [], [], 0.0
    for row in rows:
        observed = row['sm_local_span_cycles']
        error = predict(row, by_threads) - observed
        signed.append(error / observed)
        absolute.append(abs(error))
        percentages.append(abs(error) / observed)
        total += observed
    return {'sample_count': len(rows),
            'mape_percent': statistics.fmean(percentages) * 100.0,
            'wmae_percent': sum(absolute) / total * 100.0,
            'signed_mean_percentage_error': statistics.fmean(signed) * 100.0,
            'maximum_absolute_percentage_error': max(percentages) * 100.0}


def build_report(raw_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['schema'] != 'h200_block_reduce_latency_capture_v1':
        raise ValueError('capture manifest schema changed')
    if manifest['raw_sha256'] != sha256(raw_path):
        raise ValueError('raw SHA256 mismatch')
    if manifest['grids'] != list(GRIDS) or manifest['samples_per_cell'] != SAMPLES_PER_CELL:
        raise ValueError('capture domain changed')
    usage = {domain: resource_usage(build['ptxas_verbose'])
             for domain, build in manifest['builds'].items()}
    if usage['clock64'] != usage['globaltimer']:
        raise ValueError('timer-domain binaries differ in resources')
    clock_mhz = manifest['clock_gate']['measured_sm_clock_mhz']
    rows = read_raw(raw_path)
    params = parameters(rows, usage['clock64']['resident_ctas_by_threads'], clock_mhz)
    by_threads = {p['threads']: p for p in params}
    diagnostics = {
        'training': metrics([r for r in rows if r['trial'] in TRAINING_TRIALS], by_threads),
        'validation': metrics([r for r in rows if r['trial'] in VALIDATION_TRIALS], by_threads),
    }
    validation = diagnostics['validation']
    passed = (validation['mape_percent'] <= MAX_VALIDATION_MAPE_PERCENT
              and validation['maximum_absolute_percentage_error']
              <= MAX_VALIDATION_MAX_APE_PERCENT)
    if not passed:
        raise ValueError(f'validation gate failed: {validation}')
    return {
        'schema_version': 2,
        'status': 'candidate',
        'hardware_id': 'nvidia_h200',
        'mechanism': 'source_bound_add_rmsnorm_reduction_and_publication',
        'duration_source': 'clock64_same_smid_span_with_globaltimer_diagnostic',
        'production_or_evaluation_duration_consumed': False,
        'validity_domain': {'dtype': 'bfloat16', 'vector_width_elements': 8,
                            'block_threads': list(BLOCK_THREADS),
                            'hidden_size': 4096, 'normal_fdiv_path': True},
        'input': {
            'path': str(raw_path), 'sha256': sha256(raw_path),
            'manifest_path': str(manifest_path),
            'manifest_sha256': sha256(manifest_path),
            'baseline_commit': manifest['baseline_commit'],
            'toolchain_nvcc_version': manifest['builds']['clock64']['nvcc_version'],
            'cub_version': 200802,
            'trials': list(TRIALS), 'training_trials': list(TRAINING_TRIALS),
            'validation_trials': list(VALIDATION_TRIALS),
            'samples_per_cell': SAMPLES_PER_CELL,
            'block_threads': list(BLOCK_THREADS), 'grids': list(GRIDS),
        },
        'formula': (
            'first_block_latency_cycles[threads] + '
            '(min(max(1, ceil(grid / 132)), resident_ctas) - 1) * '
            'resident_block_service_cycles_per_sm[threads] + '
            'max(0, ceil(grid / 132) - resident_ctas) * '
            'queued_block_service_cycles_per_sm[threads]; then multiply '
            'by 1000 / measured_sm_clock_mhz for nanoseconds'),
        'derivation': (
            'first-block latency is the pooled clock64 median at grid=1; '
            'resident service is the incremental same-SMID clock64 span from '
            'one CTA to the static-resource occupancy boundary, and queued '
            'service is the increment from that boundary to 16 CTAs/SM. '
            'Resident CTA counts 8/2 follow only from the ptxas resource usage '
            'and H200 SM capacities. CUDA blocks do not migrate between SMs. '
            'Cycles are converted by the measured (clock-gated, 1980 MHz) SM '
            'clock. Globaltimer is retained only as a quantized diagnostic. '
            'No production/evaluation duration or residual is used'),
        'parameters': params,
        'diagnostics': diagnostics,
        'publication_gate': {
            'maximum_validation_mape_percent': MAX_VALIDATION_MAPE_PERCENT,
            'maximum_validation_maximum_ape_percent': MAX_VALIDATION_MAX_APE_PERCENT,
            'passed': passed,
        },
        'static_equivalence': {
            'schema': 'h200_owner_copy_same_source_v1',
            'status': 'not_run_production_sass_unavailable',
            'resource_usage': usage['clock64'],
        },
        'clock_gate': manifest['clock_gate'],
        'provenance': {
            'source': 'microbench/nvidia_h200/compute/reduction/block_reduce_latency.cu',
            'source_sha256': sha256(REPORT_PATH.with_name('block_reduce_latency.cu')),
            'runner': 'microbench/nvidia_h200/compute/reduction/block_reduce_latency_runner.py',
            'runner_sha256': sha256(REPORT_PATH.with_name('block_reduce_latency_runner.py')),
            'reporter': 'microbench/nvidia_h200/compute/reduction/block_reduce_latency_report.py',
            'reporter_sha256': sha256(REPORT_PATH),
            'captured_source_sha256': manifest['source_sha256'],
            'captured_runner_sha256': manifest['runner_sha256'],
            'baseline_commit': manifest['baseline_commit'],
            'device': manifest['device'],
            'remote_raw_dir': 'profiler_output/microbench/nvidia_h200/'
                              + manifest['run_id'],
            'local_raw_dir': 'profiler_output/microbench/nvidia_h200/' + manifest['run_id'],
            'accepted_captures': manifest['clock_gate']['accepted_captures'],
            'rejected_captures': manifest['clock_gate']['rejected_captures'],
            'a800_counterpart': 'profiler_output/microbench/nvidia_a800/'
                                'a800_reduce_daa756d_20260818/report.json (report schema; '
                                'no A800 references/ JSON exists for this mechanism)',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.input, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for p in report['parameters']:
        print(p)
    print(report['diagnostics'])


if __name__ == '__main__':
    main()
