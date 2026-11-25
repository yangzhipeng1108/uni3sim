"""Validate an independent masked byte-ring program; no operator observations."""
import csv
import io
import hashlib
import math
from statistics import median
import zipfile

from finite_strided_panel_analysis import median_confidence_interval


METRICS = (
    'sm__cycles_elapsed.avg.per_second', 'lts__cycles_elapsed.avg.per_second',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'lts__t_sectors_srcunit_tex_op_read.sum',
    'lts__t_sectors_srcunit_ltcfabric_op_read.sum', 'lts__t_sectors_op_read.sum',
    'dram__bytes_read.sum', 'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum',
    'sm__ctas_launched.sum',
)


def evidence_bundle(directory):
    """Archive immutable mounted inputs even when their file time is1970."""
    output = io.BytesIO()
    members = []
    with zipfile.ZipFile(output, 'w') as archive:
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            data = path.read_bytes()
            members.append({'path': path.name, 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest()})
            entry = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return output.getvalue(), members


def counter_values(output):
    lines = output.splitlines()
    start, = (index for index, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    rows = list(reader)
    fields = reader.fieldnames
    if fields is None or len(fields) != len(set(fields)):
        raise ValueError('counter CSV requires unique named fields')
    units = dict(zip(METRICS, ('hz', 'hz', 'sector', 'sector', 'sector', 'sector', 'byte', 'inst', 'block')))
    values = {}
    if 'Metric Name' in fields:
        for row in rows:
            name = row['Metric Name']
            if name not in METRICS:
                continue
            if name in values:
                raise ValueError('counter metric repeats across more than one launch')
            if row['Metric Unit'] != units[name]:
                raise ValueError('counter metric unit differs from the frozen base unit')
            values[name] = float(row['Metric Value'].replace(',', ''))
    elif set(METRICS) <= set(fields):
        unit_rows = [row for row in rows if row['ID'] == '']
        data_rows = [row for row in rows if row['ID'] != '']
        if len(unit_rows) != 1 or len(data_rows) != 1:
            raise ValueError('wide counter CSV requires one unit row and one kernel launch')
        for name in METRICS:
            if unit_rows[0][name] != units[name]:
                raise ValueError('wide counter metric unit differs from the frozen base unit')
            values[name] = float(data_rows[0][name].replace(',', ''))
    else:
        raise ValueError('unknown counter CSV layout')
    if set(values) != set(METRICS):
        raise ValueError('counter result does not cover every frozen metric')
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError('counter metric must be finite and nonnegative')
    return values


def counter_checks(values, cell):
    stride, grid, groups, _suffix = cell
    active, rounds = grid * groups // 8, stride // 256
    upstream = active * rounds * (16384 + 256)
    unique = active * rounds * 16384 + stride
    def agrees(actual, expected):
        return abs(actual / expected - 1) <= .01
    tex = values[METRICS[3]] * 32
    fabric = values[METRICS[4]] * 32
    total = values[METRICS[5]] * 32
    return {
        'upstream_source_bytes_1pct': agrees(values[METRICS[2]] * 32, upstream),
        'LTS_TEX_matches_upstream_1pct': agrees(tex, upstream),
        'LTS_total_conserves_TEX_plus_fabric_1pct': agrees(total, tex + fabric),
        'DRAM_matches_unique_B_plus_broadcast_A_1pct': agrees(values[METRICS[6]], unique),
        'unpredicated_LDGSTS_warp_count_exact': values[METRICS[7]] == active * rounds * 40,
        'full_physical_CTA_count_exact': values[METRICS[8]] == grid,
        'nominal_SM_clock_3pct': abs(values[METRICS[0]] / 1.98e9 - 1) <= .03,
    }


def summarize(raw, cells):
    by_key = {}
    for item in raw['timing']:
        key = tuple(item['cell'])
        if key not in by_key:
            by_key[key] = {}
        process = item['process']
        if process in by_key[key]:
            raise ValueError('timing process cell repeated')
        by_key[key][process] = item['measurement']
    if set(by_key) != set(tuple(cell) for cell in cells):
        raise ValueError('timing matrix incomplete')
    counters = {tuple(item['cell']): item for item in raw['counters']}
    if len(counters) != len(raw['counters']) or set(counters) != set(by_key):
        raise ValueError('counter matrix incomplete or repeated')
    results = []
    for cell in cells:
        key = tuple(cell)
        if set(by_key[key]) != {0, 1}:
            raise ValueError('both reverse-order processes required')
        medians, intervals, clocks = [], [], []
        for process in (0, 1):
            measurement = by_key[key][process]
            stride, grid, groups, suffix = key
            if (measurement['stride'], measurement['grid'], measurement['active_groups'], measurement['suffix']) != key:
                raise ValueError('timing identity differs from frozen cell')
            if (measurement['threads'] != 128 or measurement['shared_bytes'] != 61440
                    or measurement['resident_capacity'] != 3
                    or measurement['active_ctas'] != grid * groups // 8
                    or measurement['rounds'] != stride // 256
                    or measurement['all_thread_checksum_pass'] is not True
                    or len(measurement['samples']) != 63):
                raise ValueError('timing resource, count or numerical contract failed')
            samples = [sample['body_ns'] for sample in measurement['samples']]
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError('invalid independent body time')
            clocks.extend(sample['weighted_mhz'] for sample in measurement['samples'])
            medians.append(median(samples))
            intervals.append(median_confidence_interval(samples))
        checks = {
            'both_median_CI_3pct': all(max(abs(ci['lower_ns'] / center - 1), abs(ci['upper_ns'] / center - 1)) <= .03
                                         for ci, center in zip(intervals, medians)),
            'reverse_process_3pct': max(medians) / min(medians) <= 1.03,
            'all_weighted_SM_clocks_3pct': all(math.isfinite(value) and abs(value / 1980 - 1) <= .03 for value in clocks),
        }
        values = counter_values(counters[key]['stdout'])
        checks.update(counter_checks(values, key))
        b_bytes = key[1] * key[2] // 8 * 16384 * (key[0] // 256)
        results.append({'cell': list(key), 'process_medians_ns': medians,
                        'median_intervals': intervals, 'checks': checks,
                        'qualified': all(checks.values()), 'B_logical_bytes': b_bytes,
                        'slower_process_complete_body_ns': max(medians),
                        'logical_B_progress_bytes_s': b_bytes / (max(medians) * 1e-9),
                        'counter_values': values})
    lts = [point['counter_values'][METRICS[1]] for point in results]
    lts_stable = min(lts) > 0 and max(lts) / min(lts) <= 1.03
    return {'status': 'complete_independent_masked_byte_program',
            'qualified': all(point['qualified'] for point in results) and lts_stable,
            'LTS_clock_across_cells_3pct': lts_stable, 'points': results,
            'production_or_Serving_duration_consumed': False,
            'automatic_Forward_parameter_adoption': False,
            'limitations': ['Complete byte-ring includes scalar checksum consumption; this is not a generic DRAM bandwidth or pure copy latency.',
                           'Same resident capacity does not establish matching native instruction mix or CTA placement.',
                           'No Tensor, routing, activation or full expert operator executes. Transfer to Forward requires its own source/physical-domain check.']}
