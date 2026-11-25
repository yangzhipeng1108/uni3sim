#!/usr/bin/env python3
"""Derive the H200 FA3 MLA mechanism reference from fa3_mechanism_runner output.

Standard library only (runs next to the raw captures on target host).  Every
statistic is a median over the CTAs / warpgroups of the accepted clock-gated
capture; cycles are SM cycles (clock64), converted to ns elsewhere with the
measured 1980-MHz busy clock recorded in the manifest.

  wgmma_gemm_cycles[shape][ksteps][active_wg]
      median cycles from wgmma.fence to wgmma.wait_group 0 of one gemm of
      ``ksteps`` back-to-back instructions, per issuing warpgroup, with 1 or
      2 warpgroups issuing concurrently on the SM;
  softmax_step_cycles[values x ovalues][active_wg]
      median cycles of one FA3 online-softmax step (mode 1) minus the
      dependency-update baseline (mode 0);
  paged_kv_stream[hold][grid] = {first_full_cycles, steady_ii_cycles}
      median over CTAs and trials of the two-stage paged cp.async ring.

usage: fa3_mechanism_reference.py --capture-dir <dir> [--capture-dir ...] --output <json>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def median(values):
    if not values:
        raise ValueError('empty sample')
    return statistics.median(values)


def rows(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def accepted_paths(capture_dir, manifest):
    paths = []
    for verdict in manifest['captures'].values():
        paths.extend(Path(p) for p in verdict['stdout_paths'])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    wgmma, softmax_raw, stream = {}, {}, {}
    provenance = []
    clocks = []
    for capture_dir in args.capture_dir:
        manifest = json.loads((capture_dir / 'manifest.json').read_text())
        clocks.append(manifest['clock'])
        digest = hashlib.sha256()
        for path in sorted(accepted_paths(capture_dir, manifest)):
            digest.update(path.read_bytes())
            name = path.name
            data = rows(path)
            if name.startswith('wgmma_'):
                for row in data:
                    shape, k, wg = row['shape'], int(row['ksteps']), int(row['active_wg'])
                    wgmma.setdefault(shape, {}).setdefault(k, {}).setdefault(wg, []).append(
                        int(row['cycles']) / int(row['reps']))
            elif name.startswith('softmax_'):
                for row in data:
                    key = (int(row['values']), int(row['ovalues']), int(row['active_wg']), int(row['mode']))
                    softmax_raw.setdefault(key, []).append(int(row['cycles']) / int(row['reps']))
            elif name.startswith('stream_'):
                for row in data:
                    hold, grid = int(row['hold']), int(row['grid'])
                    cell = stream.setdefault(hold, {}).setdefault(grid, {'first': [], 'ii': []})
                    cell['first'].append(int(row['first_full_cycles']))
                    cell['ii'].append(float(row['steady_ii_cycles']))
            else:
                raise ValueError(f'unexpected capture file {name}')
        provenance.append({
            'capture_dir': str(capture_dir),
            'accepted_stdout_sha256': digest.hexdigest(),
            'builds': manifest['builds'], 'clock': manifest['clock'],
            'device': manifest['device'], 'constants': manifest['constants']})
    softmax = {}
    for (values, ovalues, wg, mode), samples in softmax_raw.items():
        if mode != 1:
            continue
        base = median(softmax_raw[(values, ovalues, wg, 0)])
        softmax.setdefault(f'{values}x{ovalues}', {})[str(wg)] = median(samples) - base
    reference = {
        'schema': 'nvidia_h200_fa3_mla_mechanisms_reference_v1',
        'hardware_id': 'nvidia_h200',
        'mechanism': 'fa3_mla_mechanisms',
        'production_or_evaluation_duration_consumed': False,
        'production_predictor_consumption_allowed': True,
        'timer_domain': 'clock64_sm_cycles',
        'measured_sm_clock_mhz': sorted(c['measured_sm_clock_mhz'] for c in clocks)[len(clocks) // 2],
        'source_revision': 'vllm-flash-attention 803020a8fa15407871341d41eba4919ade2ee1ee (hopper/)',
        'wgmma_gemm_cycles': {
            shape: {str(k): {str(wg): median(v) for wg, v in sorted(by_wg.items())}
                    for k, by_wg in sorted(by_k.items())}
            for shape, by_k in sorted(wgmma.items())},
        'softmax_step_cycles': softmax,
        'paged_kv_stream': {
            str(hold): {str(grid): {'first_full_cycles': median(cell['first']),
                                    'steady_ii_cycles': median(cell['ii']),
                                    'samples': len(cell['ii'])}
                        for grid, cell in sorted(by_grid.items())}
            for hold, by_grid in sorted(stream.items())},
        'provenance': provenance,
    }
    # The decode ring was re-measured at one stage hold: the per-KV-block
    # occupancy of the source softmax warpgroup (QK+QV 36-k-step m64n64k16
    # SS gemm alone, one 2x16 / 2x64 softmax step alone, then the two
    # warpgroups' concurrent m64n256k16 PV gemms of 4 k-steps).
    chain = (reference['wgmma_gemm_cycles']['ss_n64_kmajor']['36']['1']
             + reference['softmax_step_cycles']['16x64']['1']
             + reference['wgmma_gemm_cycles']['rs_n256_mnmajor']['4']['2'])
    ring_holds = [int(h) for h in reference['paged_kv_stream']
                  if int(h) not in (0, 1000, 2000, 3000, 4000, 5000, 6000)]
    if ring_holds != [round(chain)]:
        raise ValueError(f'decode ring hold {ring_holds} != derived chain {chain:.2f}')
    reference['decode_ring'] = {
        'hold_cycles': round(chain),
        'derivation': ('wgmma ss_n64_kmajor k36 1wg + softmax 16x64 1wg + '
                       'wgmma rs_n256_mnmajor k4 2wg'),
        'derived_chain_cycles': chain,
    }
    args.output.write_text(json.dumps(reference, indent=1, sort_keys=False) + '\n')


if __name__ == '__main__':
    main()
