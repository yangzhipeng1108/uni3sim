#!/usr/bin/env python3
"""Capture the H200 block-reduction mechanism (owner copy of the A800 runner).

Same probe (``block_reduce_latency.cu``, two timer-domain binaries), cells,
schedule and sample count as
``microbench/nvidia_a800/compute/reduction/block_reduce_latency_runner.py``:
block widths 256/1024, grids 1 and 1/2/4/8/16 CTAs per SM, three trials
rotated with stride 7, 200 samples per cell and binary.

Hardware coupling changed (stated here as required by the porting contract):
  - grids are multiples of 132 SMs (1, 132, 264, 528, 1056, 2112);
  - target is physical GPU 3 of the H200 host, ``NVIDIA H200``;
  - the A800 runner's CUDA 12.8.93 / CUB 2.7.0 pin, the live vLLM 0.11 sm_80
    extension check and the retained A800 production SASS/CFG artifact cannot
    exist on this host (the artifact is also missing on A800).  The probe is
    built natively for sm_90 with the host's CUDA 12.9 toolkit (CUB 2.8.2) and
    both versions plus the ptxas resource usage are recorded instead;
  - the A800 per-cell ``nvidia-smi`` before/after clock equality check is
    replaced by the clock-gated capture helper: each trial is one gated window
    (warm-up spin, then every cell of that trial) whose busy SM-clock samples
    must all read >= 1980 MHz; rejected windows are re-run and counted.

usage (on the H200 host, from the remote repository copy):
    python3 block_reduce_latency_runner.py --run-id <id> --out-root <dir> \
        --device 3 --baseline-commit <sha>
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('block_reduce_latency.cu')
REPORT_PATH = RUNNER_PATH.with_name('block_reduce_latency_report.py')
SM_COUNT = 132
BLOCK_THREADS = (256, 1024)
GRID_MULTIPLIERS = (0, 1, 2, 4, 8, 16)
SAMPLES = 200
TRIALS = 3
ROTATION_STRIDE = 7
GLOBAL_FIELDS = ('sample', 'device_ordinal', 'sm_count', 'threads', 'grid',
                 'block_median_ns', 'grid_span_ns', 'validation_checksum',
                 'validation_status')
CYCLE_FIELDS = ('sample', 'device_ordinal', 'sm_count', 'threads', 'grid',
                'block_median_cycles', 'sm_local_span_cycles',
                'validation_checksum', 'validation_status')
RAW_FIELDS = ('run_id', 'trial', 'position', 'physical_device', 'device_uuid',
              'sample', 'sm_count', 'threads', 'grid', 'block_median_cycles',
              'sm_local_span_cycles', 'grid_span_ns', 'validation_checksum',
              'validation_status')


def grids():
    return tuple(1 if m == 0 else SM_COUNT * m for m in GRID_MULTIPLIERS)


def schedule(trial):
    cells = tuple((threads, grid) for threads in BLOCK_THREADS for grid in grids())
    offset = (trial * ROTATION_STRIDE) % len(cells)
    for position in range(len(cells)):
        threads, grid = cells[(position + offset) % len(cells)]
        yield position, threads, grid


def read_binary_csv(path, fields):
    rows = list(csv.DictReader(io.StringIO(Path(path).read_text())))
    if not rows or tuple(rows[0].keys()) != fields:
        raise ValueError(f'{path}: unexpected probe CSV schema')
    if len(rows) != SAMPLES or any(r['validation_status'] != 'pass' for r in rows):
        raise ValueError(f'{path}: incomplete or failed samples')
    return rows


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
    build = out / 'build'
    builds = {
        'globaltimer': capture.build(SOURCE_PATH, build / 'block_reduce_globaltimer'),
        'clock64': capture.build(SOURCE_PATH, build / 'block_reduce_clock64',
                                 ('-DA800_REDUCTION_CLOCK64',)),
    }
    warmup = capture.build_warmup(build)
    raw_rows = []
    verdicts = []
    for trial in range(TRIALS):
        commands = []
        cells = list(schedule(trial))
        for position, threads, grid in cells:
            for domain in ('globaltimer', 'clock64'):
                commands.append((
                    [builds[domain]['binary'], threads, grid, SAMPLES],
                    out / f't{trial}_p{position}_{threads}_{grid}_{domain}.csv'))
        verdict = capture.run_gated(f'trial{trial}', commands, out, args.device,
                                    warmup)
        verdicts.append(verdict)
        paths = iter(verdict['stdout_paths'])
        for position, threads, grid in cells:
            global_rows = read_binary_csv(next(paths), GLOBAL_FIELDS)
            cycle_rows = read_binary_csv(next(paths), CYCLE_FIELDS)
            for g, c in zip(global_rows, cycle_rows):
                if (int(g['threads']), int(g['grid']), int(g['sm_count'])) != (
                        threads, grid, SM_COUNT) or g['sample'] != c['sample']:
                    raise ValueError('probe output disagrees with the schedule')
                raw_rows.append({
                    'run_id': args.run_id, 'trial': trial, 'position': position,
                    'physical_device': args.device, 'device_uuid': identity['uuid'],
                    'sample': int(c['sample']), 'sm_count': SM_COUNT,
                    'threads': threads, 'grid': grid,
                    'block_median_cycles': int(c['block_median_cycles']),
                    'sm_local_span_cycles': int(c['sm_local_span_cycles']),
                    'grid_span_ns': int(g['grid_span_ns']),
                    'validation_checksum': c['validation_checksum'],
                    'validation_status': 'pass',
                })
    with (out / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest = {
        'schema': 'h200_block_reduce_latency_capture_v1',
        'hardware_id': 'nvidia_h200', 'run_id': args.run_id,
        'baseline_commit': args.baseline_commit,
        'production_duration_consumed': False,
        'device': identity, 'physical_device': args.device,
        'block_threads': list(BLOCK_THREADS),
        'grid_multipliers': list(GRID_MULTIPLIERS), 'grids': list(grids()),
        'samples_per_cell': SAMPLES, 'trials': TRIALS, 'sm_count': SM_COUNT,
        'timer_domains': {'block_median_cycles': 'clock64_sm_cycles',
                          'sm_local_span_cycles': 'clock64_same_smid_cycles',
                          'grid_span_ns': 'globaltimer_ns'},
        'source_sha256': capture.sha256(SOURCE_PATH),
        'runner_sha256': capture.sha256(RUNNER_PATH),
        'reporter_sha256': capture.sha256(REPORT_PATH),
        'helper_sha256': capture.sha256(capture.__file__),
        'builds': builds, 'raw_sha256': capture.sha256(out / 'raw.csv'),
        'clock_gate': capture.clock_summary(verdicts), 'captures': verdicts,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['clock_gate']))


if __name__ == '__main__':
    main()
