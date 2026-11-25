#!/usr/bin/env python3
"""Build and clock-gate-capture the H200 FlashAttention-3 MLA mechanism probes.

Runs on target host with the system ``python3`` (standard library only).  Three
independent mechanism probes, none of which executes FlashAttention:

  fa3_wgmma_service      wgmma.mma_async service (BF16 -> FP32) for the FA3
                         MLA instruction shapes, 1 and 2 issuing warpgroups;
  fa3_softmax_step       register-resident FA3 online-softmax step service
                         for the prefill (2x32 scores, 2x32 O) and decode
                         (2x16 scores, 2x64 O) fragments;
  fa3_paged_kv_stream    paged-MLA-cache cp.async two-stage operand ring
                         service versus streaming SM count and stage hold.

Every capture goes through ``compute/clock_gated_capture.run_gated`` (busy
nvidia-smi samples >= 1980 MHz, rejected attempts re-run and counted).  The
binaries are built with the remote CUDA 12.9 nvcc; wgmma needs sm_90a.

usage: fa3_mechanism_runner.py --device 2 --out-dir <dir> --plan base
       fa3_mechanism_runner.py --device 2 --out-dir <dir> --plan decode_chain
       fa3_mechanism_runner.py --device 2 --out-dir <dir> --plan decode_ring \
           --stream-hold <cycles>

``base`` is the full sweep above.  ``decode_chain`` measures the 36-k-step
m64n64k16 SS gemm (the decode QK+QV issue sequence of the source softmax
warpgroup).  ``decode_ring`` re-measures the paged ring at one stage hold
(the decode per-block consumer occupancy derived from the base and
decode_chain references, passed explicitly) for every streaming SM count
1..132, so the decode template reads the ring service by exact lookup.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'compute'))
import clock_gated_capture as gate  # noqa: E402

PROBES = {
    'fa3_wgmma_service': ('fa3_wgmma_service.cu', 'sm_90a'),
    'fa3_softmax_step': ('fa3_softmax_step.cu', 'sm_90'),
    'fa3_paged_kv_stream': ('fa3_paged_kv_stream.cu', 'sm_90'),
}
WGMMA_SHAPES = ('ss_n64_kmajor', 'ss_n128_kmajor', 'rs_n128_mnmajor',
                'ss_n256_mnmajor', 'rs_n256_mnmajor')
WGMMA_KSTEPS = (1, 2, 4, 8, 12, 32)
WGMMA_REPS = 2000
SOFTMAX_GEOMETRIES = ((16, 64), (32, 32))
SOFTMAX_REPS = 4000
FULL_GRID = 132
STREAM_BLOCKS = 32
STREAM_PAGE_SIZE = 16
STREAM_TRIALS = 3
STREAM_HOLD_GRIDS = (1, 32, 64, 108, 128, 132)
STREAM_HOLDS = (1000, 2000, 3000, 4000, 5000, 6000)


def build(name, out_dir):
    source, arch = PROBES[name]
    binary = Path(out_dir) / 'bin' / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [str(gate.NVCC), '-O3', '-std=c++17',
               f'-gencode=arch=compute_{arch[3:]},code={arch}', '-lineinfo',
               '-Xptxas=-v', str(HERE / source), '-o', str(binary)]
    result = subprocess.run(command, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (binary.parent / f'{name}.ptxas.txt').write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'nvcc failed for {source}:\n{result.stderr}')
    return binary, {
        'source': source, 'source_sha256': gate.sha256(HERE / source),
        'binary_sha256': gate.sha256(binary), 'compile_command': ' '.join(command),
        'nvcc_version': gate.nvcc_version()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--plan', choices=('base', 'decode_chain', 'decode_ring'),
                        required=True)
    parser.add_argument('--stream-hold', type=int)
    parser.add_argument('--stream-page-size', type=int, default=STREAM_PAGE_SIZE,
                        help='paged-cache rows per page for the decode_ring plan '
                             '(16 = vLLM block; 1 = SGLang page_size 1)')
    args = parser.parse_args()
    if args.stream_page_size != STREAM_PAGE_SIZE and args.plan != 'decode_ring':
        parser.error('--stream-page-size applies only to decode_ring')
    if (args.plan == 'decode_ring') != (args.stream_hold is not None):
        parser.error('--stream-hold is required by, and only by, decode_ring')
    out = args.out_dir
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    identity = gate.device_identity(args.device)
    warmup = gate.build_warmup(out / 'bin')
    builds = {}
    binaries = {}
    for name in PROBES:
        binaries[name], builds[name] = build(name, out)
    verdicts = []
    if args.plan == 'decode_chain':
        commands = [
            ([binaries['fa3_wgmma_service'], 'ss_n64_kmajor', 36, active,
              FULL_GRID, WGMMA_REPS],
             raw / f'wgmma_ss_n64_kmajor_k36_wg{active}.csv')
            for active in (1, 2)]
        verdicts.append(('fa3_wgmma_service_k36',
                         gate.run_gated('fa3_wgmma_service_k36', commands, raw,
                                        args.device, warmup)))
    elif args.plan == 'decode_ring':
        for chunk in range(0, FULL_GRID, 33):
            commands = [
                ([binaries['fa3_paged_kv_stream'], grid, STREAM_BLOCKS,
                  args.stream_hold, args.stream_page_size, STREAM_TRIALS],
                 raw / f'stream_g{grid}_b{STREAM_BLOCKS}_h{args.stream_hold}'
                       f'{"" if args.stream_page_size == STREAM_PAGE_SIZE else f"_p{args.stream_page_size}"}.csv')
                for grid in range(chunk + 1, min(chunk + 33, FULL_GRID) + 1)]
            name = f'fa3_paged_kv_stream_h{args.stream_hold}_{chunk}'
            verdicts.append((name, gate.run_gated(
                name, commands, raw, args.device, warmup)))
    else:
        verdicts.extend(base_plan(binaries, raw, args.device, warmup))
    write_manifest(out, identity, builds, verdicts, args)


def base_plan(binaries, raw, device, warmup):
    verdicts = []
    commands = []
    for shape in WGMMA_SHAPES:
        for ksteps in WGMMA_KSTEPS:
            for active in (1, 2):
                commands.append((
                    [binaries['fa3_wgmma_service'], shape, ksteps, active,
                     FULL_GRID, WGMMA_REPS],
                    raw / f'wgmma_{shape}_k{ksteps}_wg{active}.csv'))
    verdicts.append(('fa3_wgmma_service',
                     gate.run_gated('fa3_wgmma_service', commands, raw,
                                    device, warmup)))

    commands = []
    for values, ovalues in SOFTMAX_GEOMETRIES:
        for mode in (0, 1):
            for active in (1, 2):
                commands.append((
                    [binaries['fa3_softmax_step'], mode, values, ovalues,
                     active, FULL_GRID, SOFTMAX_REPS],
                    raw / f'softmax_m{mode}_v{values}_o{ovalues}_wg{active}.csv'))
    verdicts.append(('fa3_softmax_step',
                     gate.run_gated('fa3_softmax_step', commands, raw,
                                    device, warmup)))

    # Hold-0 sweep over every streaming SM count (exact lookup domain), then
    # the hold sweep that validates the two-stage ring law on six grids.
    for chunk in range(0, FULL_GRID, 33):
        commands = [
            ([binaries['fa3_paged_kv_stream'], grid, STREAM_BLOCKS, 0,
              STREAM_PAGE_SIZE, STREAM_TRIALS],
             raw / f'stream_g{grid}_b{STREAM_BLOCKS}_h0.csv')
            for grid in range(chunk + 1, min(chunk + 33, FULL_GRID) + 1)]
        verdicts.append((f'fa3_paged_kv_stream_h0_{chunk}',
                         gate.run_gated(f'fa3_paged_kv_stream_h0_{chunk}',
                                        commands, raw, device, warmup)))
    commands = [
        ([binaries['fa3_paged_kv_stream'], grid, STREAM_BLOCKS, hold,
          STREAM_PAGE_SIZE, STREAM_TRIALS],
         raw / f'stream_g{grid}_b{STREAM_BLOCKS}_h{hold}.csv')
        for grid in STREAM_HOLD_GRIDS for hold in STREAM_HOLDS]
    verdicts.append(('fa3_paged_kv_stream_holds',
                     gate.run_gated('fa3_paged_kv_stream_holds', commands, raw,
                                    device, warmup)))
    return verdicts


def write_manifest(out, identity, builds, verdicts, args):
    manifest = {
        'plan': args.plan, 'stream_hold': args.stream_hold,
        'decode_ring_page_size': args.stream_page_size,
        'device': identity, 'builds': builds,
        'captures': {name: verdict for name, verdict in verdicts},
        'clock': gate.clock_summary([verdict for _, verdict in verdicts]),
        'constants': {
            'wgmma_shapes': WGMMA_SHAPES, 'wgmma_ksteps': WGMMA_KSTEPS,
            'wgmma_reps': WGMMA_REPS, 'softmax_geometries': SOFTMAX_GEOMETRIES,
            'softmax_reps': SOFTMAX_REPS, 'full_grid': FULL_GRID,
            'stream_blocks': STREAM_BLOCKS, 'stream_page_size': STREAM_PAGE_SIZE,
            'stream_trials': STREAM_TRIALS, 'stream_hold_grids': STREAM_HOLD_GRIDS,
            'stream_holds': STREAM_HOLDS},
        'production_or_evaluation_duration_consumed': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['clock'], indent=1))


if __name__ == '__main__':
    main()
