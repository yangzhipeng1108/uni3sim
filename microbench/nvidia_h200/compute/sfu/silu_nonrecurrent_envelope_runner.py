#!/usr/bin/env python3
"""Capture the H200 non-recurrent SiLU envelope (owner copy of the A800 runner).

Same probe, cells, eight trials per cell, cyclic schedule and three independent
runs (rotations 0, 1, 2) as
``microbench/nvidia_a800/compute/sfu/silu_nonrecurrent_envelope_runner.py``.
Each rotation is a separate process, as on A800 (three run IDs).

Hardware coupling changed:
  - grids {132, 264, 313, 1056} (see the owner-copy probe header);
  - physical GPU 3 of the H200 host; the probe sees it as device 0;
  - the A800 production-SASS identity gate (vLLM 0.11 sm_80 ``_C.abi3.so``)
    has no sm_90 counterpart on this host; the probe binary is copied back and
    its SASS static gates are checked locally by the report instead;
  - each rotation runs under the clock gate (busy SM clock >= 1980 MHz).

usage (H200 host): python3 silu_nonrecurrent_envelope_runner.py --run-id <id> \
    --out-root <dir> --device 3 --baseline-commit <sha>
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('silu_nonrecurrent_envelope.cu')
REPORT_PATH = RUNNER_PATH.with_name('silu_nonrecurrent_envelope_report.py')
GATES_PATH = RUNNER_PATH.with_name('silu_nonrecurrent_envelope_gates.json')
ROTATIONS = (0, 1, 2)


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
    build = capture.build(SOURCE_PATH, out / 'build' / 'silu_nonrecurrent_envelope')
    warmup = capture.build_warmup(out / 'build')
    verdicts = []
    for rotation in ROTATIONS:
        run_id = f'{args.run_id}_r{rotation}'
        verdicts.append(capture.run_gated(
            f'rotation{rotation}',
            [([build['binary'], 0, run_id, rotation], out / f'raw_r{rotation}.csv')],
            out, args.device, warmup))
    manifest = {
        'schema': 'nvidia_h200_silu_nonrecurrent_envelope_capture_v1',
        'hardware_id': 'nvidia_h200', 'run_id': args.run_id,
        'baseline_commit': args.baseline_commit, 'device': identity,
        'physical_device': args.device,
        'production_or_evaluation_duration_consumed': False,
        'source_sha256': capture.sha256(SOURCE_PATH),
        'runner_sha256': capture.sha256(RUNNER_PATH),
        'report_sha256': capture.sha256(REPORT_PATH),
        'gates_sha256': capture.sha256(GATES_PATH),
        'helper_sha256': capture.sha256(capture.__file__),
        'build': build,
        'raw': [{'rotation': rotation, 'run_id': f'{args.run_id}_r{rotation}',
                 'path': v['stdout_paths'][0],
                 'sha256': capture.sha256(v['stdout_paths'][0])}
                for rotation, v in zip(ROTATIONS, verdicts)],
        'clock_gate': capture.clock_summary(verdicts), 'captures': verdicts,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['clock_gate']))


if __name__ == '__main__':
    main()
