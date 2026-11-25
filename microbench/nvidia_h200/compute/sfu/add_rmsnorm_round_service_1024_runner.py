#!/usr/bin/env python3
"""Capture the H200 block-1024 Add-RMSNorm source-round probes (all widths).

Owner copy of the A800 per-width runners
(``add_rmsnorm_round_service_1024_h{4096,5120}_runner.py``,
``add_rmsnorm_h8192/add_rmsnorm_round_service_1024_h8192_runner.py`` and the
H2048 runner at commit fc2c2dbb).  Every A800 runner builds one exact-width
probe and runs it once as ``<binary> <device> 0.875``; the probe itself runs
three position-rotated trials of 200 samples per (function, grid) cell.  This
runner does the same for each requested width, one process per width.

Hardware coupling changed:
  - probes are the H200 owner copies (grid cell 108 -> 132, H200 identity);
  - physical GPU 3 of the H200 host, seen as device 0 by the probe;
  - the A800 static-contract step (vLLM 0.11 sm_80 production SASS/resource
    identity, cuobjdump) has no sm_90 counterpart on this host; the probe's own
    runtime check (32 registers, 168 B static shared memory, 2 resident CTAs)
    and the recorded ptxas usage remain;
  - each width runs under the clock gate (busy SM clock >= 1980 MHz).

usage (H200 host): python3 add_rmsnorm_round_service_1024_runner.py --run-id <id> \
    --out-root <dir> --device 3 --baseline-commit <sha> [--hidden-sizes 2048 ...]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
HERE = RUNNER_PATH.parent
SOURCES = {
    2048: HERE / 'add_rmsnorm_round_service_1024.cu',
    2560: HERE / 'add_rmsnorm_round_service_1024_h2560.cu',
    4096: HERE / 'add_rmsnorm_round_service_1024_h4096.cu',
    5120: HERE / 'add_rmsnorm_round_service_1024_h5120.cu',
    8192: HERE / 'add_rmsnorm_h8192' / 'add_rmsnorm_round_service_1024_h8192.cu',
}
VARIANCE_SCALE = '0.875'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=sorted(SOURCES))
    args = parser.parse_args()
    out = args.out_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    identity = capture.device_identity(args.device)
    warmup = capture.build_warmup(out / 'build')
    widths = []
    verdicts = []
    for hidden in args.hidden_sizes:
        source = SOURCES[hidden]
        build = capture.build(source, out / 'build' / f'add_rmsnorm_round_service_1024_h{hidden}')
        verdict = capture.run_gated(
            f'h{hidden}', [([build['binary'], 0, VARIANCE_SCALE], out / f'raw_h{hidden}.csv')],
            out, args.device, warmup)
        verdicts.append(verdict)
        widths.append({'hidden_size': hidden,
                       'source': str(source.relative_to(HERE.parents[3])),
                       'build': build, 'raw_path': verdict['stdout_paths'][0],
                       'raw_sha256': capture.sha256(verdict['stdout_paths'][0]),
                       'native_command': f'{build["binary"]} 0 {VARIANCE_SCALE}',
                       'clock': verdict})
    manifest = {
        'schema': 'nvidia_h200_add_rmsnorm_round_service_1024_capture_v1',
        'hardware_id': 'nvidia_h200', 'run_id': args.run_id,
        'baseline_commit': args.baseline_commit, 'device': identity,
        'physical_device': args.device, 'runtime_variance_scale': 0.875,
        'production_or_evaluation_duration_consumed': False,
        'runner_sha256': capture.sha256(RUNNER_PATH),
        'helper_sha256': capture.sha256(capture.__file__),
        'widths': widths, 'clock_gate': capture.clock_summary(verdicts),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['clock_gate']))


if __name__ == '__main__':
    main()
