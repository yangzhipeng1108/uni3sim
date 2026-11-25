#!/usr/bin/env python3
"""Run the H200 FP8 block-scaled GEMM mechanism probes (remote, GPU-gated).

Two independent mechanism probes back the Hopper FP8 block-scaled dense GEMM
template (TensorRT-LLM swap-AB DeepGEMM and DeepGEMM sm90 1D2D):

* ``fp8_tma_operand_stream.cu`` -- the TMA producer/consumer shared-memory
  ring without MMA: launch shell (kblocks = 0), ring fill and per-stage TMA
  ingress service as a function of active CTAs, ring depth and stage bytes.
  Timed by nsys (``cuda_gpu_trace`` kernel durations, the timer domain of the
  production traces), one marker launch (grid.x = 1000 + config) per config.
* ``fp8_wgmma_kblock_issue.cu`` -- the per-BLOCK_K=128 FP8 wgmma issue/wait
  (+ FFMA promotion) service of 1 or 2 math warp-groups per SM, timed by
  clock64/globaltimer inside the kernel.

Every timed capture goes through ``clock_gated_capture.run_gated`` (busy SM
clock must read 1980 MHz).  No production kernel, library or duration is
involved; the probe geometry values (stages, box rows, threads, shared memory)
are the launch facts of the in-scope kernels, not their timings.

Usage (remote, system python3):
    python3 fp8_gemm_mechanisms_runner.py --device 1 --out <dir>
Output: ``<dir>/fp8_gemm_mechanisms_raw.json`` (small; nsys reports stay on
the remote host).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'compute'))
import clock_gated_capture as gate  # noqa: E402

NSYS = Path(shutil.which('nsys') or 'nsys')
TMA_SOURCE = HERE / 'fp8_tma_operand_stream.cu'
WGMMA_SOURCE = HERE / 'fp8_wgmma_kblock_issue.cu'
LAUNCHES_PER_CONFIG = 120
DISCARDED_LEADING_LAUNCHES = 20
WGMMA_KBLOCKS = 512

# Launch-shell geometries (grid 132; threads, dynamic shared memory bytes,
# cluster) of the in-scope kernels, with and without the output-tile store.
SHELL_GEOMETRIES = (
    (384, 152928, 1, 4096),    # swap-AB BLOCK_M=128 BLOCK_N=16, 8 stages
    (256, 170464, 1, 2048),    # 1D2D BLOCK_M=64 BLOCK_N=16, 16 stages
    (256, 225280, 1, 8192),    # 1D2D BLOCK_M=64 BLOCK_N=64, 13 stages
    (384, 216544, 2, 65536),   # 1D2D BLOCK_M=256 BLOCK_N=128, 3 stages, cluster 2
)
# k-blocks (BLOCK_K = 128) per CTA; each (family, active) line is fitted over
# these points by the reference derivation.
RING_KBLOCKS = (4, 8, 16, 32, 56)
SWAP_AB_ACTIVE_CTAS = (1, 2, 4, 8, 12, 17, 24, 32, 40, 48, 56, 64, 72, 88,
                       104, 132)
# (weight rows = BLOCK_N, stages) of the DeepGEMM 1D2D BLOCK_M=64 rings.
BM64_RINGS = ((16, 16), (48, 15), (64, 13), (80, 11))
BM64_ACTIVE_CTAS = (8, 16, 32, 64, 96, 112, 116, 128, 132)


def build_sm90a(source, binary):
    """Compile for sm_90a (wgmma / TMA need the architecture-specific target).

    Same provenance record as ``clock_gated_capture.build`` but with the
    ``sm_90a`` target instead of its fixed ``sm_90``.
    """
    binary = Path(binary)
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [str(gate.NVCC), '-O3', '-std=c++17', '-gencode',
               'arch=compute_90a,code=sm_90a',
               '-lineinfo', '-Xptxas=-v', str(source), '-o', str(binary)]
    result = subprocess.run(command, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (binary.parent / f'{binary.name}.ptxas.txt').write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'nvcc failed for {source}:\n{result.stderr}')
    return {
        'source': str(source), 'source_sha256': gate.sha256(source),
        'binary': str(binary), 'binary_sha256': gate.sha256(binary),
        'compile_command': ' '.join(command),
        'nvcc_version': gate.nvcc_version(),
    }


def smem_bytes(stages, rows_stream, rows_shared, store_bytes):
    return (stages * (rows_stream + rows_shared) * 128
            + -(-store_bytes // 128) * 128 + 2 * 16 * 8)


def tma_configs():
    configs = []
    for threads, smem, cluster, store in SHELL_GEOMETRIES:
        configs.append({'family': 'shell', 'grid': 132, 'active': 132,
                        'threads': threads, 'stages': 1, 'rows_stream': 16,
                        'rows_shared': 0, 'kblocks': 0, 'cluster': cluster,
                        'store': store, 'smem': smem})
        configs.append({'family': 'shell_no_store', 'grid': 132,
                        'active': 132, 'threads': threads, 'stages': 1,
                        'rows_stream': 16, 'rows_shared': 0, 'kblocks': 0,
                        'cluster': cluster, 'store': 0, 'smem': smem})
    # swap-AB ring: 8 stages of a 128-row DRAM weight box and a 16-row
    # L2-resident activation box, 384 threads, grid 132; streaming-CTA curve.
    for active in SWAP_AB_ACTIVE_CTAS:
        for kblocks in RING_KBLOCKS:
            configs.append({
                'family': 'ring_s8_w128_a16_t384', 'grid': 132,
                'active': active, 'threads': 384, 'stages': 8,
                'rows_stream': 128, 'rows_shared': 16, 'kblocks': kblocks,
                'cluster': 1, 'store': 2 * 16 * 128,
                'smem': max(152928, smem_bytes(8, 128, 16, 4096))})
    # 1D2D BLOCK_M=64 rings: a BLOCK_N-row DRAM weight box and a 64-row
    # L2-resident activation box, 256 threads; streaming-CTA curves.
    for rows, stages in BM64_RINGS:
        for active in BM64_ACTIVE_CTAS:
            for kblocks in RING_KBLOCKS:
                configs.append({
                    'family': f'ring_s{stages}_w{rows}_a64_t256', 'grid': 132,
                    'active': active, 'threads': 256, 'stages': stages,
                    'rows_stream': rows, 'rows_shared': 64,
                    'kblocks': kblocks, 'cluster': 1, 'store': 2 * 64 * rows,
                    'smem': smem_bytes(stages, rows, 64, 2 * 64 * rows)})
    # 1D2D BLOCK_M=256 BLOCK_N=128 per-SM ingress (3 stages, 48 KiB/stage).
    for kblocks in (8, 16, 32, 56):
        configs.append({
            'family': 'ring_s3_w128_a256', 'grid': 132, 'active': 132,
            'threads': 384, 'stages': 3, 'rows_stream': 128,
            'rows_shared': 256, 'kblocks': kblocks, 'cluster': 1,
            'store': 0, 'smem': smem_bytes(3, 128, 256, 0)})
    return configs


def export_trace(report):
    prefix = report.with_suffix('')
    subprocess.run([str(NSYS), 'stats', '--report', 'cuda_gpu_trace',
                    '--format', 'csv', '--force-export=true', '--output',
                    str(prefix), str(report)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Path(f'{prefix}_cuda_gpu_trace.csv')


def parse_trace(csv_path, config_count):
    durations = {index: [] for index in range(config_count)}
    current = None
    rows = []
    with csv_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['GrdX']:
                rows.append(row)
    rows.sort(key=lambda row: int(row['Start (ns)']))
    for row in rows:
        name = row['Name']
        if name.startswith('marker_kernel'):
            current = int(row['GrdX']) - 1000
            continue
        if name.startswith('tma_stream_kernel'):
            if current is None:
                raise ValueError('probe launch before the first marker')
            durations[current].append(int(row['Duration (ns)']))
    summary = []
    for index in range(config_count):
        values = durations[index]
        if len(values) != LAUNCHES_PER_CONFIG:
            raise ValueError(
                f'config {index}: {len(values)} launches, expected '
                f'{LAUNCHES_PER_CONFIG}')
        kept = sorted(values[DISCARDED_LEADING_LAUNCHES:])
        summary.append({
            'launches_kept': len(kept),
            'median_ns': statistics.median(kept),
            'p10_ns': kept[int(0.1 * (len(kept) - 1))],
            'p90_ns': kept[int(0.9 * (len(kept) - 1))],
            'min_ns': kept[0],
        })
    return summary


def parse_wgmma(stdout_path):
    rows = []
    for line in Path(stdout_path).read_text().splitlines():
        fields = line.split()
        values = dict(zip(fields[0::2], fields[1::2]))
        rows.append({
            'n': int(values['n']), 'waves': int(values['waves']),
            'promote': int(values['promote']),
            'warpgroups': int(values['warpgroups']),
            'kblocks': int(values['kblocks']),
            'cycles_per_kblock': float(values['cycles_per_kblock']),
            'ns_per_kblock': float(values['ns_per_kblock']),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ['HOME'] = (
        os.environ['MIDLAYER_H200_HOME']
        if 'MIDLAYER_H200_HOME' in os.environ else str(Path.home())
    )
    identity = gate.device_identity(args.device)
    build_dir = out / 'build'
    tma_build = build_sm90a(TMA_SOURCE, build_dir / 'fp8_tma_operand_stream')
    wgmma_build = build_sm90a(WGMMA_SOURCE,
                              build_dir / 'fp8_wgmma_kblock_issue')
    warmup = gate.build_warmup(build_dir)
    configs = tma_configs()
    config_path = out / 'tma_configs.txt'
    config_path.write_text(''.join(
        f"{c['grid']} {c['active']} {c['threads']} {c['stages']} "
        f"{c['rows_stream']} {c['rows_shared']} {c['kblocks']} "
        f"{c['cluster']} {c['store']} {c['smem']}\n" for c in configs))
    trials = []
    verdicts = []
    for trial in range(args.trials):
        report = out / f'tma_trial{trial}.nsys-rep'
        verdict = gate.run_gated(
            f'tma_trial{trial}',
            [([NSYS, 'profile', '--trace=cuda', '--sample=none',
               '--cpuctxsw=none', '--force-overwrite=true', '--output',
               report.with_suffix(''), tma_build['binary'], config_path,
               LAUNCHES_PER_CONFIG], out / f'tma_trial{trial}.txt')],
            out, args.device, warmup)
        verdicts.append(verdict)
        tma_summary = parse_trace(export_trace(report), len(configs))
        wgmma_verdict = gate.run_gated(
            f'wgmma_trial{trial}',
            [([wgmma_build['binary'], WGMMA_KBLOCKS],
              out / f'wgmma_trial{trial}.txt')],
            out, args.device, warmup)
        verdicts.append(wgmma_verdict)
        trials.append({
            'trial': trial,
            'tma': tma_summary,
            'wgmma': parse_wgmma(wgmma_verdict['stdout_paths'][0]),
            'tma_clock': verdict,
            'wgmma_clock': wgmma_verdict,
        })
    raw = {
        'artifact_type': 'h200_fp8_gemm_mechanism_probe_raw',
        'hardware_id': 'nvidia_h200',
        'device': identity,
        'production_or_evaluation_duration_consumed': False,
        'tma_configs': configs,
        'launches_per_config': LAUNCHES_PER_CONFIG,
        'discarded_leading_launches': DISCARDED_LEADING_LAUNCHES,
        'wgmma_kblocks': WGMMA_KBLOCKS,
        'builds': {'tma': tma_build, 'wgmma': wgmma_build},
        'clock': gate.clock_summary(verdicts),
        'trials': trials,
    }
    (out / 'fp8_gemm_mechanisms_raw.json').write_text(
        json.dumps(raw, indent=1) + '\n')
    print('done', out / 'fp8_gemm_mechanisms_raw.json')


if __name__ == '__main__':
    main()
