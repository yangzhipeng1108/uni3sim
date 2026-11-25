#!/usr/bin/env python3
"""Build the H200 depth-three CTA-barrier service reference in the A800 schema.

Input: ``report.json`` and ``manifest.json`` written by
``cta_barrier_service_runner.py``.  Output keys are those of
``microbench/nvidia_a800/references/sm80_depth3_cta_barrier_service.json``.
``software`` and ``production_source`` are copied verbatim from the A800
reference: they are static facts of the CUDA 12.2 cuBLASLt sm_80 depth-three
kernel whose 256-thread / 147456-B one-CTA-per-SM envelope the probe
reproduces (the H200 config keeps that envelope under the key
``sm80_depth3_cta_barrier_service_cycles``); only ``microbenchmark`` carries
H200 measurements.  ``barrier_cycles`` is the report's median over trials of
(median barrier envelope - median baseline envelope) / repetitions, in SM
clock64 cycles.

Usage:
    python cta_barrier_service_reference.py --run-dir <dir> --run-path <p> --output <json>
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
A800_REFERENCE = 'microbench/nvidia_a800/references/sm80_depth3_cta_barrier_service.json'
SOURCE = 'microbench/nvidia_h200/runtime/synchronization/cta_barrier_service.cu'
RUNNER = 'microbench/nvidia_h200/runtime/synchronization/cta_barrier_service_runner.py'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--run-path', required=True)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a800 = json.loads((REPO / A800_REFERENCE).read_text())
    report = json.loads((args.run_dir / 'report.json').read_text())
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    if report['hardware_id'] != 'nvidia_h200' or manifest['device']['name'] != 'NVIDIA H200':
        raise ValueError('capture is not an H200 capture')
    if report['source_sha256'] != sha256(REPO / SOURCE):
        raise ValueError('capture source differs from the repository probe')
    if report['runner_sha256'] != sha256(REPO / RUNNER):
        raise ValueError('capture runner differs from the repository runner')
    gates = report['gates']
    if not (gates['all_trials_cover_each_sm_once_per_sample']
            and gates['busy_clock_at_least_1980_mhz']
            and gates['relative_spread_at_most_0_01']):
        raise ValueError(f'capture gates failed: {gates}')
    reference = {
        'schema': 'nvidia_h200_sm80_depth3_cta_barrier_service_reference_v1',
        'hardware_id': 'nvidia_h200',
        'software': a800['software'],
        'production_source': a800['production_source'],
        'microbenchmark': {
            'source': SOURCE,
            'runner': RUNNER,
            'source_sha256': report['source_sha256'],
            'runner_sha256': report['runner_sha256'],
            'capture': args.run_path,
            'capture_manifest_sha256': report['capture_manifest_sha256'],
            'raw_sha256': report['raw_sha256'],
            'report_sha256': sha256(args.run_dir / 'report.json'),
            'sass_sha256': report['sass_sha256'],
            'timer_domain': report['timer_domain'],
            'threads_per_cta': report['threads_per_cta'],
            'total_shared_memory_bytes': report['total_shared_bytes'],
            'dynamic_shared_memory_bytes': report['dynamic_shared_bytes'],
            'resident_ctas_per_sm': report['resident_ctas_per_sm'],
            'grid_ctas': report['grid'],
            'repetitions': report['repeats'],
            'samples_per_trial': report['samples_per_trial'],
            'trials': len(report['trial_barrier_cycles']),
            'barrier_cycles': report['barrier_cycles'],
            'trial_relative_spread': report['trial_relative_spread'],
        },
        'decision': {
            'status': 'accepted_mechanism_service',
            'forward_eligible': True,
            'reason': (
                'The probe has no global-memory traffic or matrix operation, exactly '
                'preserves the 256-thread one-CTA/SM 147456-B shared-memory envelope '
                '(one CTA on each of the 132 H200 SMs), and measures the one-barrier '
                'service independently of every production duration. The envelope is '
                'the A800 cuBLASLt sm_80 source envelope; the H200 production GEMM '
                'kernels were not re-audited here.'),
        },
        'excluded_conclusions': a800['excluded_conclusions'],
        'production_or_evaluation_duration_consumed': False,
        'provenance': {
            'baseline_commit': manifest['baseline_commit'],
            'binary_sha256': manifest['binary_sha256'],
            'clock_gate': report['clock_gate'],
            'clock_gate_rule': ('nvidia-smi -lms 50 during each trial x mode process; '
                                'reject a process with no busy sample or a busy sample '
                                'below 1980 MHz'),
            'compile_command': manifest['compile_command'],
            'derivation': 'microbench/nvidia_h200/runtime/synchronization/cta_barrier_service_reference.py',
            'derivation_sha256': sha256(Path(__file__).resolve()),
            'device': manifest['device'],
            'host_contention': ('GPUs 1-3 of the same host were used concurrently by '
                                'other workers; the timer is same-CTA clock64'),
            'host_loadavg_after': manifest['host_loadavg_after'],
            'remote_path': args.remote_path,
            'trial_barrier_cycles': report['trial_barrier_cycles'],
            'report_gates': gates,
        },
    }
    args.output.write_text(json.dumps(reference, indent=2) + '\n')
    print(report['trial_barrier_cycles'], report['barrier_cycles'], gates)


if __name__ == '__main__':
    main()
