#!/usr/bin/env python3
"""Capture the H200 scalar-SiLU dependency probe (A800 had no runner).

The A800 value (``silu_scalar_round`` = 249.125 SM cycles/round) came from one
manual run of ``silu_scalar_latency.cu`` recorded in
``profiler_output/microbench/nvidia_a800/silu_scalar_latency_0ea587f0_20260825_r1``:
the 1024-thread, 64-iteration ``dependent_cycles_per_scalar`` line.  This
runner scripts that procedure on H200: build the owner copy for sm_90, run it
under the clock gate in ``ROTATIONS`` independent processes on the owned GPU,
and keep every raw stdout.  Derivation lives in ``silu_scalar_latency_report.py``.

usage (H200 host): python3 silu_scalar_latency_runner.py --run-id <id> \
    --out-root <dir> --device 3 --baseline-commit <sha>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('silu_scalar_latency.cu')
ROTATIONS = 2


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
    build = capture.build(SOURCE_PATH, out / 'build' / 'silu_scalar_latency')
    warmup = capture.build_warmup(out / 'build')
    verdicts = []
    for rotation in range(ROTATIONS):
        verdicts.append(capture.run_gated(
            f'rotation{rotation}', [([build['binary']], out / f'raw_r{rotation}.txt')],
            out, args.device, warmup))
    manifest = {
        'schema': 'h200_silu_scalar_latency_capture_v1',
        'hardware_id': 'nvidia_h200', 'run_id': args.run_id,
        'baseline_commit': args.baseline_commit, 'device': identity,
        'physical_device': args.device,
        'production_or_evaluation_duration_consumed': False,
        'source_sha256': capture.sha256(SOURCE_PATH),
        'runner_sha256': capture.sha256(RUNNER_PATH),
        'helper_sha256': capture.sha256(capture.__file__),
        'build': build,
        'raw': [{'rotation': rotation, 'path': v['stdout_paths'][0],
                 'sha256': capture.sha256(v['stdout_paths'][0])}
                for rotation, v in enumerate(verdicts)],
        'clock_gate': capture.clock_summary(verdicts), 'captures': verdicts,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['clock_gate']))


if __name__ == '__main__':
    main()
