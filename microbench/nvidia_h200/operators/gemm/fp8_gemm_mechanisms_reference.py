#!/usr/bin/env python3
"""Derive the H200 FP8 block-GEMM TMA-ring mechanism reference (local).

Input: ``fp8_gemm_mechanisms_raw.json`` written on the H200 host by
``fp8_gemm_mechanisms_runner.py`` (three clock-gated trials of the TMA
operand-ring probe and of the FP8 wgmma k-block probe).

Published model (``derived_model``):

* ``ring_service_surfaces``: for each ring geometry (threads, stages, rows of
  the per-CTA DRAM weight box and of the shared L2-resident activation box,
  per-CTA output-tile store bytes) and each measured number of streaming CTAs,
  the nsys kernel duration at each measured k-block count (median over the
  trials of the per-trial median over 100 kept launches) -- the evidence.
* ``ring_service_model``: the geometry-free ring form fitted to those
  surfaces only (``fit_ring_service_model``), which consumers evaluate.
* ``launch_shells``: the kblocks = 0 duration (barrier init, synchronisation,
  optional output-tile bulk store, exit) of the four in-scope launch shells.
* ``launch_shell_ns``: median kblocks = 0, store-free shell duration over
  the in-scope launch geometries (spread recorded).
* ``full_chip_tile_store_bytes_per_ns``: the aggregate rate of the largest
  per-CTA output-tile bulk store (64 KiB x 132 CTAs), from the shell pair.

The FP8 wgmma k-block probe is reported but not published: its isolated
shared-memory-operand throughput exceeds the public dense FP8 tensor rate and
changes by 1.7x with the operand-buffer rotation, so it does not isolate a
production mechanism (see ``wgmma_probe_not_published``).

No production kernel, library call or duration is read.

Usage:
    python3 fp8_gemm_mechanisms_reference.py <raw.json> <reference.json>
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SOURCES = ('fp8_tma_operand_stream.cu', 'fp8_wgmma_kblock_issue.cu',
           'fp8_gemm_mechanisms_runner.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def steady_line(by_kblocks):
    """Steady-state ring line between the two largest k-block samples."""
    (k0, t0), (k1, t1) = ((p['kblocks'], p['duration_ns'])
                          for p in by_kblocks[-2:])
    slope = (t1 - t0) / (k1 - k0)
    return slope, t1 - slope * k1


def fit_ring_service_model(surfaces):
    """Fit the geometry-free ring service form to the probe surfaces.

    Per-stage interval of one CTA streaming ``W`` DRAM bytes per stage while
    ``X = streaming CTAs * W`` bytes stream chip-wide per stage round::

        interval = (floor**p + (X / bandwidth)**p) ** (1 / p)

    (a per-SM TMA-ring stage service ``floor`` composed with the chip DRAM
    stream by a p-norm).  The ring duration is
    ``fill_base + fill_per_store_byte * store_bytes + kblocks * interval``.
    Parameters minimise the squared log error of the steady per-stage slopes
    (grid search) and, for the fill, the least-squares line over the
    intercepts; only probe data enter.
    """

    samples = []
    for surface in surfaces:
        weight_bytes = surface['weight_box_rows'] * surface['box_inner_bytes']
        for point in surface['points']:
            slope, intercept = steady_line(point['by_kblocks'])
            samples.append({
                'weight_bytes': weight_bytes,
                'streaming_ctas': point['streaming_ctas'],
                'store_bytes': surface['output_store_bytes'],
                'slope_ns': slope, 'intercept_ns': intercept})

    def interval(sample, floor, bandwidth, power):
        stream = sample['streaming_ctas'] * sample['weight_bytes'] / bandwidth
        return (floor ** power + stream ** power) ** (1.0 / power)

    best = None
    for power in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        for floor in (100.0 + 0.5 * i for i in range(31)):
            for bandwidth in (3500.0 + 25.0 * i for i in range(57)):
                error = sum(
                    math.log(interval(x, floor, bandwidth, power)
                             / x['slope_ns']) ** 2 for x in samples)
                if best is None or error < best[0]:
                    best = (error, floor, bandwidth, power)
    _, floor, bandwidth, power = best
    count = len(samples)
    mean_store = sum(x['store_bytes'] for x in samples) / count
    mean_intercept = sum(x['intercept_ns'] for x in samples) / count
    fill_per_byte = (
        sum((x['store_bytes'] - mean_store)
            * (x['intercept_ns'] - mean_intercept) for x in samples)
        / sum((x['store_bytes'] - mean_store) ** 2 for x in samples))
    fill_base = mean_intercept - fill_per_byte * mean_store
    residuals = []
    for x in samples:
        model_slope = interval(x, floor, bandwidth, power)
        model_fill = fill_base + fill_per_byte * x['store_bytes']
        residuals.append({
            'weight_bytes': x['weight_bytes'],
            'streaming_ctas': x['streaming_ctas'],
            'store_bytes': x['store_bytes'],
            'measured_stage_ns': x['slope_ns'],
            'model_stage_ns': model_slope,
            'stage_relative_error': model_slope / x['slope_ns'] - 1.0,
            'measured_fill_ns': x['intercept_ns'],
            'model_fill_ns': model_fill,
        })
    return {
        'form': 'interval = (floor^p + (streaming_ctas * weight_box_bytes / '
                'bandwidth)^p)^(1/p); duration = fill_base + '
                'fill_per_store_byte * store_bytes + kblocks * interval',
        'stage_floor_ns': floor,
        'dram_stream_bytes_per_ns': bandwidth,
        'combination_power': power,
        'fill_base_ns': fill_base,
        'fill_per_store_byte_ns': fill_per_byte,
        'fit_rule': 'grid search minimising the squared log error of the '
                    'steady (two largest k-block) per-stage slopes; fill by '
                    'least squares over the steady intercepts',
        'max_abs_stage_relative_error': max(
            abs(r['stage_relative_error']) for r in residuals),
        'residuals': residuals,
    }


def main():
    raw_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    raw = json.loads(raw_path.read_text())
    if raw['production_or_evaluation_duration_consumed'] is not False:
        raise ValueError('raw probe consumed a production duration')
    if raw['clock']['busy_clock_mhz_min'] != 1980:
        raise ValueError('a busy clock sample fell below 1980 MHz')
    configs = raw['tma_configs']
    trials = raw['trials']
    medians = []
    for index in range(len(configs)):
        per_trial = [trial['tma'][index]['median_ns'] for trial in trials]
        medians.append({
            'median_ns': statistics.median(per_trial),
            'trial_medians_ns': per_trial,
            'cross_trial_spread_ns': max(per_trial) - min(per_trial),
        })
    surfaces = {}
    shells = []
    for config, measured in zip(configs, medians):
        if config['family'] in ('shell', 'shell_no_store'):
            shells.append({
                'threads': config['threads'], 'cluster': config['cluster'],
                'dynamic_smem_bytes': config['smem'],
                'store_bytes': config['store'],
                'grid_ctas': config['grid'],
                'duration_ns': measured['median_ns'],
                'trial_medians_ns': measured['trial_medians_ns'],
            })
            continue
        if not config['family'].startswith('ring_s') or (
                config['family'] == 'ring_s3_w128_a256'):
            continue
        key = (config['threads'], config['stages'], config['rows_stream'],
               config['rows_shared'], config['store'])
        if key not in surfaces:
            surfaces[key] = {}
        by_active = surfaces[key]
        if config['active'] not in by_active:
            by_active[config['active']] = []
        by_active[config['active']].append({
            'kblocks': config['kblocks'],
            'duration_ns': measured['median_ns'],
            'cross_trial_spread_ns': measured['cross_trial_spread_ns'],
        })
    published = []
    for key in sorted(surfaces):
        threads, stages, weight_rows, activation_rows, store = key
        points = []
        for active in sorted(surfaces[key]):
            row = sorted(surfaces[key][active], key=lambda p: p['kblocks'])
            points.append({'streaming_ctas': active, 'by_kblocks': row})
        published.append({
            'threads': threads, 'stages': stages,
            'weight_box_rows': weight_rows,
            'activation_box_rows': activation_rows,
            'box_inner_bytes': 128,
            'output_store_bytes': store,
            'grid_ctas': 132,
            'points': points,
        })
    ring_model = fit_ring_service_model(published)
    # Full-chip TMA tile-store (DRAM write) rate: the largest-store shell
    # minus its store-free twin (same threads, cluster, shared memory).
    largest = max(shells, key=lambda shell: shell['store_bytes'])
    twin = [shell for shell in shells
            if (shell['threads'], shell['cluster'],
                shell['dynamic_smem_bytes'], shell['store_bytes'])
            == (largest['threads'], largest['cluster'],
                largest['dynamic_smem_bytes'], 0)]
    if len(twin) != 1:
        raise ValueError('largest-store shell has no store-free twin')
    store_rate = (largest['grid_ctas'] * largest['store_bytes']
                  / (largest['duration_ns'] - twin[0]['duration_ns']))
    store_free = sorted(shell['duration_ns'] for shell in shells
                        if shell['store_bytes'] == 0)
    wgmma = {}
    for trial in trials:
        for row in trial['wgmma']:
            key = (row['n'], row['waves'], row['promote'], row['warpgroups'])
            if key not in wgmma:
                wgmma[key] = []
            wgmma[key].append(row['cycles_per_kblock'])
    wgmma_rows = [
        {'n': key[0], 'waves': key[1], 'promote': key[2],
         'warpgroups': key[3],
         'cycles_per_kblock': statistics.median(values),
         'fp8_flop_per_cycle_per_sm': (
             2 * 64 * key[0] * 128 * key[1] * key[3]
             / statistics.median(values))}
        for key, values in sorted(wgmma.items())]
    reference = {
        'schema': 'nvidia_h200_block_fp8_gemm_tma_ring_reference_v1',
        'hardware_id': 'nvidia_h200',
        'mechanism': 'block_fp8_gemm_tma_ring_service',
        'production_or_evaluation_duration_consumed': False,
        'production_predictor_consumption_allowed': True,
        'timer_domain': 'nsys_gpu_kernel',
        'probe_sources': {
            name: {'path': f'microbench/nvidia_h200/operators/gemm/{name}',
                   'sha256': sha256(HERE / name)} for name in SOURCES},
        'derivation': {
            'path': 'microbench/nvidia_h200/operators/gemm/'
                    'fp8_gemm_mechanisms_reference.py',
            'sha256': sha256(Path(__file__).resolve())},
        'raw_capture': {
            'remote_path': 'profiler_output/microbench/nvidia_h200/wp12_deepgemm_fp8/mech_run2/fp8_gemm_mechanisms_raw.json',
            'sha256': sha256(raw_path),
            'device': raw['device'],
            'builds': raw['builds'],
            'trials': len(trials),
            'launches_per_config': raw['launches_per_config'],
            'discarded_leading_launches': raw['discarded_leading_launches'],
        },
        'clock': raw['clock'],
        'derived_model': {
            'rule': 'median over trials of the per-trial median nsys kernel '
                    'duration; consumers interpolate piecewise-linearly in '
                    'streaming CTAs and k-blocks (no fitted form)',
            'ring_service_model': ring_model,
            'ring_service_surfaces': published,
            'launch_shells': shells,
            'launch_shell_ns': statistics.median(store_free),
            'launch_shell_spread_ns': store_free[-1] - store_free[0],
            'launch_shell_note': (
                'median of the store-free shells over the four in-scope '
                'launch geometries (grid 132; 256/384 threads; 150-225 KiB '
                'dynamic shared memory; cluster 1/2)'),
            'full_chip_tile_store_bytes_per_ns': store_rate,
            'full_chip_tile_store_note': (
                f"{largest['grid_ctas']} CTAs x {largest['store_bytes']} B "
                'bulk shared->global store: shell with store minus the '
                'store-free shell of the same geometry'),
        },
        'wgmma_probe_not_published': {
            'reason': (
                'isolated shared-memory-operand FP8 wgmma throughput exceeds '
                'the public H200 dense FP8 rate (1978.9 TFLOP/s, 7571 '
                'FLOP/cycle/SM at 1.98 GHz) and a smoke variant rotating '
                'the operands over 4-6 stage buffers ran 1.7x slower than '
                'the fixed-buffer probe, so the probe does not isolate the '
                'production tensor mechanism; the template uses the public '
                'FP8 rate of the HWSpec tensor_core component instead'),
            'rows': wgmma_rows,
        },
    }
    out_path.write_text(json.dumps(reference, indent=1, sort_keys=True) + '\n')
    print(out_path, {name: ring_model[name] for name in (
        'stage_floor_ns', 'dram_stream_bytes_per_ns', 'combination_power',
        'fill_base_ns', 'fill_per_store_byte_ns',
        'max_abs_stage_relative_error')})


if __name__ == '__main__':
    main()
