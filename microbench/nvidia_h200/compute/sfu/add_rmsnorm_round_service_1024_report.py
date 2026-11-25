#!/usr/bin/env python3
"""Derive the H200 block-1024 Add-RMSNorm source-round services (all widths).

Selection rule of the A800 per-width reporters
(``add_rmsnorm_round_service_1024_h{4096,5120}_report.py`` and
``add_rmsnorm_h8192/..._report.py``), unchanged:

    trial median  = median over the 200 samples of the per-CTA clock64 span
                    (``median_block_clock64_cycles``) of one (function, grid) cell
    cell service  = median of the three trial medians
    gate          = (max - min) / median of the trial medians <= 10 % per cell

and the pipeline reference rule of ``add_rmsnorm_block1024_pipeline_sm80.json``:
the grid-1 cell is the CTA-local source-round service Forward uses; the larger
grids are validation cells.  Forward rates are ``clock / cycles``; on H200 the
clock is the measured, clock-gated 1980 MHz (A800 used its 1.14-GHz model clock).

Hardware coupling changed: grid cell 108 -> 132 (one full wave on 132 SMs);
H200 identity; the A800 production-SASS identity gate is replaced by the probe's
runtime resource check (32 registers, 168 B, 2 resident CTAs), which every raw
row records and this reporter re-checks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPORT_PATH = Path(__file__).resolve()
SM_COUNT = 132
TRIALS = 3
SAMPLES = 200
MAXIMUM_RELATIVE_SPREAD = 0.10
FUNCTIONS = ('add_square', 'scale_multiply')
GRIDS = {2048: (1, 4, 16, 132), 2560: (1, 4, 16, 132), 4096: (1, 4, 16, 132),
         5120: (1, 4, 8, 16, 132), 8192: (1, 4, 8, 16, 64, 132)}
PUBLISHED_ACTIVE = {2048: (4, 16, 132), 2560: (4, 16, 132), 4096: (4, 16, 132),
                    5120: (4, 8, 16, 132), 8192: (4, 8, 16, 132)}
A800_CAPTURE = {
    2048: 'add-rmsnorm-round-1024-clock64-fc2c2dbb-capture',
    2560: 'add-rmsnorm-h2560-block1024-1adea53d-device6-20260904',
    4096: 'add-rmsnorm-h4096-block1024-dc4b36cd-device7-20260905-r2',
    5120: 'add-rmsnorm-h5120-block1024-1c004f02-device7-20260905-r1 + '
          'add_rmsnorm_h5120_grid8_2cf25b66_20260906_gpu7',
    8192: 'add_rmsnorm_h8192_7d79c3db_20260907_gpu7',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_width(path, hidden):
    rows = list(csv.DictReader(Path(path).open()))
    grids = GRIDS[hidden]
    if len(rows) != TRIALS * SAMPLES * len(FUNCTIONS) * len(grids):
        raise ValueError(f'H{hidden}: row count {len(rows)}')
    resources = set()
    for row in rows:
        grid = int(row['grid_blocks'])
        if (int(row['hidden_size']) != hidden or int(row['block_threads']) != 1024
                or int(row['active_threads']) != hidden // 8
                or int(row['active_sms']) != grid
                or int(row['minimum_ctas_per_active_sm']) != 1
                or int(row['maximum_ctas_per_active_sm']) != 1
                or grid not in grids):
            raise ValueError(f'H{hidden}: row contract violated {row}')
        resources.add((int(row['registers_per_thread']), int(row['static_shared_bytes']),
                       int(row['resident_cta_limit'])))
    if resources != {(32, 168, 2)}:
        raise ValueError(f'H{hidden}: resource identity changed {resources}')
    cells = {}
    for function in FUNCTIONS:
        for grid in grids:
            trial_medians = []
            for trial in range(TRIALS):
                samples = [int(r['median_block_clock64_cycles']) for r in rows
                           if r['function'] == function and int(r['grid_blocks']) == grid
                           and int(r['trial']) == trial]
                if len(samples) != SAMPLES:
                    raise ValueError('incomplete trial cell')
                trial_medians.append(statistics.median(samples))
            center = statistics.median(trial_medians)
            spread = (max(trial_medians) - min(trial_medians)) / center
            if spread > MAXIMUM_RELATIVE_SPREAD:
                raise ValueError(f'H{hidden} {function} grid {grid}: spread {spread}')
            cells[(function, grid)] = {'median_block_clock64_cycles': center,
                                       'trial_median_block_clock64_cycles': trial_medians,
                                       'relative_spread': spread}
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    clock_mhz = manifest['clock_gate']['measured_sm_clock_mhz']
    captures, services, validation, selected = {}, {}, {}, {}
    services = {function: {} for function in FUNCTIONS}
    for width in manifest['widths']:
        hidden = width['hidden_size']
        active = hidden // 8
        raw = args.run_dir / Path(width['raw_path']).name
        if sha256(raw) != width['raw_sha256']:
            raise ValueError(f'{raw}: SHA256 mismatch')
        cells = derive_width(raw, hidden)
        captures[f'active_{active}'] = {
            'path': f'profiler_output/microbench/nvidia_h200/{manifest["run_id"]}',
            'hidden_size': hidden, 'baseline_commit': manifest['baseline_commit'],
            'device_index': manifest['physical_device'],
            'device_uuid': manifest['device']['uuid'],
            'manifest_sha256': sha256(args.run_dir / 'manifest.json'),
            'source': width['source'], 'source_sha256': width['build']['source_sha256'],
            'binary_sha256': width['build']['binary_sha256'], 'raw_sha256': width['raw_sha256'],
            'ptxas_verbose': width['build']['ptxas_verbose'],
            'a800_capture': A800_CAPTURE[hidden],
            'clock_gate': {k: width['clock'][k] for k in (
                'accepted', 'busy_samples', 'busy_clock_mhz_min', 'busy_clock_mhz_max')},
            'rejected_attempts': len(width['clock']['rejected_attempts']),
            'report_sha256': 'same_as_this_reference',
            'resource_usage_sha256': hashlib.sha256(
                width['build']['ptxas_verbose'].encode()).hexdigest(),
            'sass_sha256': 'not_recorded_no_cuobjdump_on_h200_host',
            'static_contract_sha256': 'not_applicable_on_h200',
        }
        if hidden == 5120:
            captures['active_640_grid8_extension'] = {
                **captures['active_640'],
                'note': 'grid 8 is a cell of the same H200 H5120 capture'}
        spreads = []
        validation[f'active_{active}'] = {}
        for function in FUNCTIONS:
            services[function][f'active_{active}'] = cells[(function, 1)][
                'median_block_clock64_cycles']
            validation[f'active_{active}'][f'{function}_cycles'] = {
                f'grid_{grid}': cells[(function, grid)]['median_block_clock64_cycles']
                for grid in GRIDS[hidden]}
            spreads += [cells[(function, grid)]['relative_spread'] for grid in GRIDS[hidden]]
            base = f'bf16_{function}_first_{active}'
            cycles = cells[(function, 1)]['median_block_clock64_cycles']
            selected[f'{base}_block1024'] = {'cycles': cycles,
                                             'rate_per_s_at_1_98ghz': clock_mhz * 1e6 / cycles}
            for grid in PUBLISHED_ACTIVE[hidden]:
                cycles = cells[(function, grid)]['median_block_clock64_cycles']
                selected[f'{base}_active{grid}_block1024'] = {
                    'cycles': cycles, 'rate_per_s_at_1_98ghz': clock_mhz * 1e6 / cycles}
        validation[f'active_{active}']['maximum_trial_relative_spread'] = max(spreads)
    widths = sorted(GRIDS)
    reference = {
        'schema': 'nvidia_h200_add_rmsnorm_block1024_pipeline_reference_v6',
        'hardware_id': 'nvidia_h200',
        'software_revision': 'vllm_0.11.0_source_cuda_12.9_sm90',
        'production_or_evaluation_duration_consumed': False,
        'captures': captures,
        'source_contract': {
            'dtype': 'bf16', 'hidden_sizes': widths, 'vector_width_elements': 8,
            'active_threads_per_source_round': [w // 8 for w in widths],
            'block_threads': 1024, 'registers_per_thread': 32,
            'static_shared_memory_bytes': 168, 'resident_ctas_per_sm': 2,
            'one_wave_grid_blocks': {
                'minimum': 1, 'maximum': SM_COUNT,
                'derivation': ('The source launches one identical CTA per row. Any positive '
                               'grid no larger than the 132-SM hardware width is one '
                               'source-identical, non-queued wave; aggregate L2 and DRAM '
                               'capacity remain separate Forward bounds.')},
            'production_function_sass_sha256': 'not_applicable_on_h200',
        },
        'service_parameters': {
            'timer_domain': 'clock64_same_cta', 'unit': 'sm_cycles_per_source_round',
            'measured_sm_clock_mhz': clock_mhz,
            'derivation': ('For each active-lane class, the grid-one cell isolates one '
                           'CTA-local source round. Larger-grid clock64 captures include '
                           'launch-wide cache and memory contention and therefore validate '
                           'the aggregate envelope rather than defining alternate CTA-local '
                           'rates. No complete production-operator duration selects either '
                           'service.'),
            'production_cta_local_service': 'grid_one_intrinsic_round_only',
            'larger_grid_captures': 'validation_only_against_launch_wide_resource_envelope',
            **services,
        },
        'composition_contract': {
            'ordered_stages': ['add_square_source_round',
                               'existing_block_reduction_and_publication',
                               'scale_multiply_source_round'],
            'each_stage_composed_once': True, 'generic_fma_alu_replaced': True,
            'generic_complete_memory_task_replaced': True,
            'aggregate_l2_dram_capacity_envelope_retained': True,
            'reduction_reference': 'microbench/nvidia_h200/references/block_reduce_latency_sm90.json',
            'reduction_report_sha256': sha256(
                REPORT_PATH.parents[2] / 'references' / 'block_reduce_latency_sm90.json'),
        },
        'validation_cells': {
            'measured_grid_blocks': [1, 4, 16, SM_COUNT],
            'active_640_extension_grid_blocks': [8],
            'active_1024_extension_grid_blocks': [8, 64],
            **validation,
            'all_trial_cells_stable_within_10_percent': True,
            'composition_has_no_reduction_double_count': True,
            'probe_and_production_resource_identity_exact': (
                'probe_runtime_identity_32_registers_168_bytes_2_ctas'),
        },
        'selected_services': selected,
        'yaml_key_rename_needed': {
            f'bf16_{f}_first_{w // 8}_active108_block1024':
                f'bf16_{f}_first_{w // 8}_active132_block1024'
            for w in widths for f in FUNCTIONS},
        'clock_gate': manifest['clock_gate'],
        'provenance': {
            'runner': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_round_service_1024_runner.py',
            'runner_sha256': manifest['runner_sha256'],
            'reporter': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_round_service_1024_report.py',
            'reporter_sha256': sha256(REPORT_PATH),
            'helper_sha256': manifest['helper_sha256'],
            'remote_raw_dir': ('profiler_output/microbench/nvidia_h200/'
                               + manifest['run_id']),
            'accepted_captures': manifest['clock_gate']['accepted_captures'],
            'rejected_captures': manifest['clock_gate']['rejected_captures'],
            'h2560_source_note': ('The A800 H2560 probe source is not retained; the H200 H2560 '
                                  'probe is the H2048 source with kHiddenSize = 2560.'),
            'h8192_note': ('The A800 H8192 class was bound separately through its capture '
                           'report (yaml fourth_additional_reference_sha256); on H200 it is '
                           'folded into this reference as active_1024.'),
        },
        'decision': {
            'status': 'accepted_with_grid_contention_separated_from_cta_service',
            'forward_eligible': True,
            'reason': ('The five active-lane classes isolate the same two source phases with '
                       'the probe resource signature. Forward uses only each grid-one intrinsic '
                       'round service, composes it with reduction, and prices grid-dependent '
                       'L2/DRAM contention through the separate aggregate resource envelope.'),
        },
        'excluded_conclusions': [
            'a complete AddRMSNorm duration', 'rows outside the one-wave grid domain',
            'hidden sizes other than 2048, 2560, 4096, 5120, and 8192',
            'a grid-dependent CTA-local service rate',
            'using larger-grid validation captures as a replacement for the aggregate memory envelope',
            'interpolation to unmeasured active-lane widths', 'block widths other than 1024',
            'resident or queued CTA source-round service',
            'per-shape timing interpolation or residual correction'],
    }
    args.output.write_text(json.dumps(reference, indent=2) + '\n')
    for key, value in selected.items():
        print(key, value['cycles'], round(value['rate_per_s_at_1_98ghz'], 3))


if __name__ == '__main__':
    main()
