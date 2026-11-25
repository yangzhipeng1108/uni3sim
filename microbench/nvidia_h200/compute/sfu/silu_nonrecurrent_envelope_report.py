#!/usr/bin/env python3
"""Audit the H200 non-recurrent SiLU envelope and publish its segmented services.

Owner copy of ``microbench/nvidia_a800/compute/sfu/silu_nonrecurrent_envelope_report.py``
with the same gates and selection rule:

* per run and cell: median / interquartile spread over eight trials of the
  maximum same-SM clock64 span (within-run IQR spread <= 0.2);
* cross-run spread of the per-run medians <= 0.2;
* calibration = median over rotations 0-1 of the per-run medians of source SiLU:
  first = T(132), resident = T(264) - T(132),
  queued = (T(1056) - T(264)) / (ceil(1056/132) - 2) = (...)/6
  (A800: T(1024) with divisor ceil(1024/108) - 2 = 8);
* affine entry/round per segment from the width endpoints 6144 and 53248;
* held-out rotation 2 must predict every grid/width within 10 % APE;
* source SiLU must be slower than the arithmetic control in every cell.

Hardware coupling changed: 132 SMs, grids {132,264,313,1056}, rate conversion
with the measured H200 SM clock (1980 MHz) instead of the A800 1.14-GHz model
clock, and the production-SASS identity check is replaced by the probe-only
SASS gates (MUFU.EX2/MUFU.RCP per loop, no local memory) evaluated on the
captured H200 binary with ``cuobjdump -sass``.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

REPORT_PATH = Path(__file__).resolve()
GATES_PATH = REPORT_PATH.with_name('silu_nonrecurrent_envelope_gates.json')
SM_COUNT = 132
THREADS = 1024
TRIALS = 8
WIDTHS = (6144, 9728, 12288, 14336, 25600, 28672, 53248)
GRIDS = (132, 264, 313, 1056)
VARIANTS = ('arithmetic_control', 'source_silu')
CELLS = tuple((variant, grid, width)
              for width in WIDTHS for grid in GRIDS for variant in VARIANTS)
KEY_PREFIX = 'silu_nonrecurrent'
CUOBJDUMP = Path('/usr/local/cuda/bin/cuobjdump')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_index_for(trial, position, rotation):
    return (position + 7 * trial + 19 * rotation) % len(CELLS)


def spread(values):
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def read_native(path):
    metadata, lines = {}, []
    for line in Path(path).read_text().splitlines():
        kind, _, rest = line.partition(',')
        if kind == 'meta':
            key, _, value = rest.partition(',')
            metadata[key] = value
        elif kind == 'raw':
            lines.append(rest)
        else:
            raise ValueError(f'{path}: unexpected line {line!r}')
    rows = []
    for raw in csv.DictReader(io.StringIO('\n'.join(lines))):
        row = {key: (raw[key] if key in ('run_id', 'variant') else
                     float(raw[key]) if key == 'checksum' else int(raw[key]))
               for key in raw}
        rows.append(row)
    return metadata, rows


def audit_run(path, rotation, run_id, gates):
    metadata, rows = read_native(path)
    expected = {
        'device_index': '0', 'device_name': 'NVIDIA H200',
        'compute_capability': '9.0', 'sm_count': str(SM_COUNT),
        'threads_per_cta': str(THREADS), 'resident_cta_limit_per_sm': '2',
        'run_id': run_id, 'run_rotation': str(rotation),
        'schedule': 'silu_nonrecurrent_cyclic_v1', 'cycle_unit': gates['cycle_unit'],
        'work_unit': gates['work_unit'],
        'production_or_evaluation_duration_consumed': 'false'}
    if metadata != expected:
        raise ValueError(f'{path}: metadata {metadata}')
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError(f'{path}: row count {len(rows)}')
    by_cell = {cell: [] for cell in CELLS}
    placement = {}
    seen = set()
    for row in rows:
        trial = row['trial'] - 1
        position = row['order_position']
        index = cell_index_for(trial, position, rotation)
        cell = (row['variant'], row['grid_ctas'], row['intermediate_elements_per_cta'])
        if (row['cell_index'] != index or cell != CELLS[index]
                or (trial, position) in seen or row['run_id'] != run_id
                or row['loop_rounds_per_cta'] != math.ceil(cell[2] / THREADS)
                or row['active_sms'] != SM_COUNT
                or row['maximum_sm_span_cycles'] <= 0):
            raise ValueError(f'{path}: row contract violated {row}')
        seen.add((trial, position))
        if cell[1] not in placement:
            placement[cell[1]] = set()
        placement[cell[1]].add((row['minimum_ctas_per_sm'], row['maximum_ctas_per_sm']))
        by_cell[cell].append(row['maximum_sm_span_cycles'])
    limit = gates['capture_success_gates']['within_run_interquartile_relative_spread_max']
    replication = {}
    for cell, values in by_cell.items():
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
        iqr = (q3 - q1) / statistics.median((q1, q3))
        replication[cell] = {'median_sm_clock_cycles': statistics.median(values),
                             'q1_sm_clock_cycles': q1, 'q3_sm_clock_cycles': q3,
                             'interquartile_relative_spread': iqr,
                             'stability_gate': iqr <= limit}
    if not all(r['stability_gate'] for r in replication.values()):
        raise ValueError(f'{path}: within-run stability failed')
    return {'replication': replication,
            'observed_ctas_per_sm_min_max': {
                str(grid): sorted(values) for grid, values in placement.items()}}


def _loop_opcodes(block):
    """Opcodes inside the single backward-branch loop of a SASS function."""
    instructions = [(int(address, 16), opcode, operands) for address, opcode, operands in
                    re.findall(r'/\*([0-9a-f]{4})\*/\s+(?:@!?U?P\w+\s+)?'
                               r'([A-Z][A-Z0-9_.]*)\s*([^;]*);', block)]
    loops = [(int(operands.strip().split()[-1], 16), address)
             for address, opcode, operands in instructions
             if opcode == 'BRA' and int(operands.strip().split()[-1], 16) < address]
    if len(loops) != 1:
        raise ValueError(f'expected one backward branch, found {loops}')
    begin, end = loops[0]
    return [opcode for address, opcode, _ in instructions if begin <= address <= end]


def sass_gates(binary, ptxas_text):
    sass = subprocess.run([str(CUOBJDUMP), '-sass', str(binary)], check=True,
                          text=True, stdout=subprocess.PIPE).stdout
    functions = {}
    for block in re.split(r'\n\s*Function : ', sass)[1:]:
        name = block.split('\n', 1)[0].strip()
        if 'silu_nonrecurrent_envelope' not in name:
            continue
        variant = 'source_silu' if 'VariantE1' in name else 'arithmetic_control'
        loop = _loop_opcodes(block)
        functions[variant] = {
            'loop_mufu_ex2': loop.count('MUFU.EX2'),
            'loop_mufu_rcp': loop.count('MUFU.RCP'),
            'loop_instruction_count': len(loop),
            'local_memory_instructions': len(re.findall(r'\b(?:LDL|STL)\b', block)),
        }
    registers = {}
    for name, count in re.findall(
            r"Compiling entry function '(\S+)'.*?Used (\d+) registers", ptxas_text,
            flags=re.S):
        variant = 'source_silu' if 'VariantE1' in name else 'arithmetic_control'
        registers[variant] = int(count)
    if set(functions) != set(VARIANTS) or set(registers) != set(VARIANTS):
        raise ValueError(f'SASS/ptxas functions missing: {sorted(functions)} {registers}')
    gates = {
        'source_silu_mufu_ex2_per_loop': functions['source_silu']['loop_mufu_ex2'],
        'source_silu_fast_path_mufu_rcp_per_loop': functions['source_silu']['loop_mufu_rcp'],
        'arithmetic_control_mufu_ex2_per_loop':
            functions['arithmetic_control']['loop_mufu_ex2'],
        'arithmetic_control_fast_path_mufu_rcp_per_loop':
            functions['arithmetic_control']['loop_mufu_rcp'],
        'local_memory_instructions': sum(f['local_memory_instructions']
                                         for f in functions.values()),
    }
    return {'functions': functions, 'registers_per_thread': registers,
            'gates': gates, 'cuobjdump': str(CUOBJDUMP)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True,
                        help='the captured probe binary copied back from the host')
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--reference-output', type=Path, required=True)
    args = parser.parse_args()
    gates = json.loads(GATES_PATH.read_text())
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    if sha256(args.binary) != manifest['build']['binary_sha256']:
        raise ValueError('binary does not match the captured build')
    clock_mhz = manifest['clock_gate']['measured_sm_clock_mhz']
    static = sass_gates(args.binary, manifest['build']['ptxas_verbose'])
    expected_static = {k: v for k, v in gates['static_success_gates'].items()
                       if k in static['gates']}
    if static['gates'] != expected_static:
        raise ValueError(f'SASS static gates failed: {static["gates"]}')
    audits = []
    for raw in manifest['raw']:
        path = args.run_dir / Path(raw['path']).name
        if sha256(path) != raw['sha256']:
            raise ValueError(f'{path}: raw SHA256 mismatch')
        audits.append(audit_run(path, raw['rotation'], raw['run_id'], gates))
    cross_limit = gates['capture_success_gates']['cross_run_median_relative_spread_max']
    cross = {cell: spread([a['replication'][cell]['median_sm_clock_cycles']
                           for a in audits]) for cell in CELLS}
    if max(cross.values()) > cross_limit:
        raise ValueError('cross-run stability failed')

    def calibration(grid, width):
        return statistics.median(
            a['replication'][('source_silu', grid, width)]['median_sm_clock_cycles']
            for a in audits[:2])

    maximum_ctas = math.ceil(GRIDS[-1] / SM_COUNT)

    def services_at(width):
        first = calibration(132, width)
        two = calibration(264, width)
        full = calibration(1056, width)
        return {'first': first, 'resident': two - first,
                'queued': (full - two) / (maximum_ctas - 2)}

    endpoints = (WIDTHS[0], WIDTHS[-1])
    endpoint_services = {w: services_at(w) for w in endpoints}
    endpoint_rounds = tuple(math.ceil(w / THREADS) for w in endpoints)
    parameters = {}
    for segment in ('first', 'resident', 'queued'):
        lower = endpoint_services[endpoints[0]][segment]
        upper = endpoint_services[endpoints[1]][segment]
        round_cycles = (upper - lower) / (endpoint_rounds[1] - endpoint_rounds[0])
        entry_cycles = lower - endpoint_rounds[0] * round_cycles
        if entry_cycles <= 0 or round_cycles <= 0:
            raise ValueError(f'{segment} affine service not positive: '
                             f'{entry_cycles} {round_cycles}')
        parameters[segment] = {'entry_cycles_per_cta': entry_cycles,
                               'round_cycles_per_cta': round_cycles,
                               'round_service_rate_per_sm_cycle': 1.0 / round_cycles}

    def predict(grid, width):
        n = math.ceil(grid / SM_COUNT)
        rounds = math.ceil(width / THREADS)
        service = {s: p['entry_cycles_per_cta'] + rounds * p['round_cycles_per_cta']
                   for s, p in parameters.items()}
        return (service['first'] + (min(n, 2) - 1) * service['resident']
                + max(0, n - 2) * service['queued'])

    held_out = {}
    for grid in GRIDS:
        for width in WIDTHS:
            observed = audits[2]['replication'][('source_silu', grid, width)][
                'median_sm_clock_cycles']
            predicted = predict(grid, width)
            held_out[f'grid{grid}/width{width}'] = {
                'observed_sm_clock_cycles': observed,
                'predicted_sm_clock_cycles': predicted,
                'absolute_percentage_error': abs(predicted - observed) / observed}
    errors = [v['absolute_percentage_error'] for v in held_out.values()]
    heldout_limit = gates['capture_success_gates']['held_out_segmented_duration_ape_max']
    gate_pass = max(errors) <= heldout_limit
    ratios = []
    for grid in GRIDS:
        for width in WIDTHS:
            source = statistics.median(a['replication'][('source_silu', grid, width)][
                'median_sm_clock_cycles'] for a in audits)
            control = statistics.median(a['replication'][('arithmetic_control', grid, width)][
                'median_sm_clock_cycles'] for a in audits)
            if source <= control:
                raise ValueError(f'source not slower than control at {grid}/{width}')
            ratios.append(source / control)
    within = max(r['interquartile_relative_spread']
                 for a in audits for r in a['replication'].values())
    audit = {
        'schema': 'nvidia_h200_silu_nonrecurrent_envelope_audit_v2',
        'hardware_id': 'nvidia_h200',
        'run_ids': [raw['run_id'] for raw in manifest['raw']],
        'grid_ctas': list(GRIDS), 'intermediate_elements_per_cta': list(WIDTHS),
        'maximum_within_run_interquartile_relative_spread': within,
        'maximum_cross_run_median_relative_spread': max(cross.values()),
        'observed_ctas_per_sm_min_max_by_run': [
            a['observed_ctas_per_sm_min_max'] for a in audits],
        'segmented_service_model': {
            'service_parameters': parameters,
            'calibration_grid_ctas': [132, 264, 1056],
            'validation_grid_ctas': [313], 'queued_divisor': maximum_ctas - 2},
        'held_out_validation': held_out,
        'held_out_mean_absolute_percentage_error': statistics.mean(errors),
        'held_out_maximum_absolute_percentage_error': max(errors),
        'segmented_service_gate': gate_pass,
        'source_to_arithmetic_control_duration_ratio_minimum': min(ratios),
        'source_to_arithmetic_control_duration_ratio_maximum': max(ratios),
        'static_sass': static,
        'per_run_cell_medians': [
            {f'{v}/grid{g}/width{w}': a['replication'][(v, g, w)]['median_sm_clock_cycles']
             for (v, g, w) in CELLS} for a in audits],
    }
    args.audit_output.write_text(json.dumps(audit, indent=2) + '\n')
    if not gate_pass:
        raise ValueError(f'held-out gate failed: max APE {max(errors)}')
    selected = {}
    for segment in ('first', 'resident', 'queued'):
        for part, field in (('entry', 'entry_cycles_per_cta'),
                            ('round', 'round_cycles_per_cta')):
            cycles = parameters[segment][field]
            selected[f'{KEY_PREFIX}_{segment}_{part}_block1024'] = {
                'cycles_per_cta': cycles,
                'operations_per_second_at_1_98ghz': clock_mhz * 1.0e6 / cycles}
    reference = {
        'schema': 'nvidia_h200_silu_nonrecurrent_segmented_service_reference_v1',
        'hardware_id': 'nvidia_h200',
        'device_name': manifest['device']['name'],
        'device_uuid': manifest['device']['uuid'],
        'compute_capability': 'sm_90',
        'physical_sm_count': SM_COUNT,
        'source_baseline_commit': manifest['baseline_commit'],
        'captures': [
            {'path': f'profiler_output/microbench/nvidia_h200/{manifest["run_id"]}',
             'run_id': raw['run_id'], 'rotation': raw['rotation'],
             'manifest_sha256': sha256(args.run_dir / 'manifest.json'),
             'probe_binary_sha256': manifest['build']['binary_sha256'],
             'raw_sha256': raw['sha256']} for raw in manifest['raw']],
        'audit': {
            'path': str(args.audit_output), 'sha256': sha256(args.audit_output),
            'calibration_rotations': [0, 1], 'held_out_rotation': 2,
            'within_run_maximum_interquartile_relative_spread': within,
            'cross_run_maximum_relative_median_spread': max(cross.values()),
            'held_out_duration_ape_limit': heldout_limit,
            'held_out_mean_duration_ape': statistics.mean(errors),
            'held_out_maximum_duration_ape': max(errors),
            'segmented_service_gate': 'pass',
        },
        'tool_identity': {
            'source_sha256': manifest['source_sha256'],
            'runner_sha256': manifest['runner_sha256'],
            'report_sha256': sha256(REPORT_PATH),
            'gates_sha256': manifest['gates_sha256'],
        },
        'production_binary_identity': {
            'path': 'not_applicable_on_h200', 'binary_sha256': 'not_applicable_on_h200',
            'sm80_cubin_sha256': 'not_applicable_on_h200',
            'silu_function_sass_sha256': 'not_applicable_on_h200',
            'status': 'not_applicable_on_h200',
            'reason': ('The A800 reference bound the probe math spine to the vLLM 0.11 '
                       'sm_80 act_and_mul SASS. No vLLM 0.11 sm_90 build exists on the '
                       'H200 host; the probe keeps the vLLM 0.11 source expression and '
                       'its own SASS gates (MUFU.EX2 = 1 and MUFU.RCP = 1 per source loop, '
                       'none in the control, no local memory) are checked instead.'),
            'probe_static_sass_gates': static['gates'],
            'probe_registers_per_thread': static['registers_per_thread'],
        },
        'mechanism_contract': {
            'threads_per_cta': THREADS, 'resident_ctas_per_sm': 2,
            'calibration_grid_ctas': [132, 264, 1056],
            'validation_grid_ctas': [313],
            'calibration_endpoint_widths': [WIDTHS[0], WIDTHS[-1]],
            'source_round_elements': THREADS,
            'formula': ('service(segment, rounds) = entry + rounds * round; total = '
                        'first + (min(ceil(grid/132), 2) - 1) * resident + '
                        'max(0, ceil(grid/132) - 2) * queued'),
        },
        'selected_services': selected,
        'measured_sm_clock_mhz': clock_mhz,
        'clock_gate': manifest['clock_gate'],
        'forward_domain': {
            'dtype': 'bf16', 'threads_per_cta': THREADS, 'grid_ctas': 313,
            'resident_ctas_per_sm': 2, 'maximum_ctas_per_sm': 3, 'cohort_refills': 2},
        'decision': {
            'status': 'positive_mechanism_candidate', 'mechanism_publishable': True,
            'forward_eligible': True,
            'reason': ('The predeclared first/resident/queued affine services pass the '
                       'frozen held-out 10 percent duration gate without production or '
                       'evaluation durations.')},
        'hardware_coupling_changes': gates['hardware_coupling_changes'] + [
            'rates converted with the measured H200 SM clock (1980 MHz); the A800 key '
            'operations_per_second_at_1_14ghz is renamed operations_per_second_at_1_98ghz',
            'forward_domain grid_ctas 256 -> 313 (same 2.37 CTAs/SM, 2 cohort refills)'],
        'production_or_evaluation_duration_consumed': False,
        'production_predictor_consumption_allowed': True,
    }
    args.reference_output.write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps(selected, indent=1))
    print('held-out mean/max APE', statistics.mean(errors), max(errors))


if __name__ == '__main__':
    main()
