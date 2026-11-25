"""Summarize native Host API spans without converting them to Forward prices."""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct

from .nvidia_h200_serving_boundary_probe import EVENT_FIELDS, observer_summary


def _kernel_api(function):
    return ('LaunchKernel' in function or 'LaunchCooperativeKernel' in function
        or function in ('cudaLaunch', 'cuLaunch'))


def union_ns(intervals):
    stop, total = -1, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(stop, start))
        stop = max(stop, end)
    return total


def summarize_records(records):
    """Keep PID/thread containment and union nested runtime/driver API spans."""
    starts, outers, children = {}, {}, defaultdict(list)
    first_symbol = {}
    for pid, labels, rows in records:
        for row in rows:
            if row[2] < row[1] or row[3] < 0:
                raise ValueError('native span has invalid clocks')
            if row[0] in (0, 1):
                key = pid, row[8]
                if key not in outers:
                    outers[key] = []
                outers[key].append(row)
            else:
                parts = labels[row[0]].split(':', 4)
                if len(parts) != 5 or parts[0] != 'cuda_api':
                    raise ValueError('native span has unknown label semantics')
                _, domain, cbid, function, symbol = parts
                int(domain), int(cbid)
                if symbol != '<no-symbol>' and _kernel_api(function):
                    key = pid, symbol
                    if key not in first_symbol or row[1] < first_symbol[key]:
                        first_symbol[key] = row[1]
    for key, rows in outers.items():
        rows.sort(key=lambda row: row[1])
        if any(left[2] > right[1] for left, right in zip(rows, rows[1:])):
            raise ValueError('outer calls overlap on the same PID/thread')
        starts[key] = tuple(row[1] for row in rows)
    outside = 0
    api_counts = Counter()
    for pid, labels, rows in records:
        for row in rows:
            if row[0] in (0, 1):
                continue
            thread = pid, row[8]
            position = bisect_right(starts[thread], row[1]) - 1 if thread in starts else -1
            if position < 0 or row[2] > outers[thread][position][2]:
                outside += 1
                continue
            parts = labels[row[0]].split(':', 4)
            child = {'domain': int(parts[1]), 'callback_id': int(parts[2]),
                'function': parts[3], 'symbol': (parts[4]
                    if _kernel_api(parts[3]) and parts[4] != '<no-symbol>' else None),
                'start_ns': row[1], 'end_ns': row[2], 'wall_ms': (row[2] - row[1]) / 1e6,
                'thread_cpu_ms': row[3] / 1e6, 'correlation_id': row[9]}
            children[pid, row[8], position].append(child)
            api_counts[int(parts[1]), parts[3]] += 1
    results = []
    for (pid, thread), rows in outers.items():
        for index, row in enumerate(rows):
            current = children[pid, thread, index]
            covered = union_ns((child['start_ns'], child['end_ns']) for child in current)
            if covered > row[2] - row[1]:
                raise ValueError('native API union exceeds its actual outer call')
            symbols = {child['symbol'] for child in current if child['symbol'] is not None
                and _kernel_api(child['function']) and first_symbol[pid, child['symbol']] >= row[1]}
            results.append({'pid': pid, 'thread_id': thread, 'phase': 'execute_model' if row[0] == 0 else 'sample_tokens',
                'step': row[4], 'requests': row[5], 'tokens': row[6], 'max_query_tokens': row[7],
                'start_ns': row[1], 'end_ns': row[2], 'wall_ms': (row[2] - row[1]) / 1e6,
                'thread_cpu_ms': row[3] / 1e6, 'api_union_ms': covered / 1e6,
                'unattributed_outer_ms': (row[2] - row[1] - covered) / 1e6,
                'first_observed_launch_symbols': sorted(symbols),
                'longest_api_calls': sorted(current, key=lambda child: child['wall_ms'], reverse=True)[:8]})
    return {'outer_call_count': len(results), 'outside_outer_api_spans': outside,
        'api_counts': [{'domain': domain, 'function': function, 'calls': count}
            for (domain, function), count in api_counts.most_common()],
        'top_outer_calls': sorted(results, key=lambda row: row['wall_ms'], reverse=True)[:30],
        'max_outer_wall_ms': max((row['wall_ms'] for row in results), default=None),
        'known_launch_symbols': len(first_symbol),
        'limitations': ['Runtime/driver API counts overlap and are not physical GPU kernel counts.',
            'Nested API wall is unioned only inside the same PID/thread outer call.',
            'Symbols identify observed names per PID, not unique module binaries or cross-run identity.',
            'CUDA Graph launch API spans do not expose captured inner kernels.',
            'Host API spans may include Device/peer waits and observer overhead; never Forward prices.']}


def analyze_stage(directory):
    directory = Path(directory)
    counts, metadata = observer_summary(directory / 'events')
    records = []
    for path in sorted((directory / 'events').glob('*/metadata.json')):
        item = json.loads(path.read_text())
        if item['observer'] != 'native_cupti_host_api_callbacks_and_outer_markers':
            raise ValueError('native analysis received a different observer')
        rows = tuple(struct.iter_unpack('<10q', path.with_name('events.bin').read_bytes()))
        labels = {int(key): value for key, value in item['labels'].items()}
        records.append((item['pid'], labels, rows))
    summary = summarize_records(records)
    enabled = {item['api_callbacks_enabled_during_warmup'] for item in metadata}
    if len(enabled) != 1 or any(type(value) is not bool for value in enabled):
        raise ValueError('stage mixes different API observation contracts')
    observing = enabled == {True}
    summary['api_observation_status'] = 'enabled_complete' if observing else 'disabled_reference'
    if not observing:
        for row in summary['top_outer_calls']:
            row.update(api_union_ms=None, unattributed_outer_ms=None,
                first_observed_launch_symbols=None, longest_api_calls=None)
        summary.update(api_counts=None, known_launch_symbols=None, outside_outer_api_spans=None)
    return {'observer_counts': counts, 'native': summary,
        'providers': [{**{name: item[name] for name in ('pid', 'torch', 'vllm', 'native_library_sha256',
            'active_cupti_library', 'active_cupti_library_sha256', 'cupti_version',
            'api_callbacks_enabled_during_warmup')}, 'cupti_version': (item['cupti_version']
                if item['api_callbacks_enabled_during_warmup'] else None)} for item in metadata]}


def inspect_overflow_prefix(directory):
    """Inspect only retained positive count/name facts; never a valid full trace."""
    directory = Path(directory)
    client = json.loads((directory / 'client_result.json').read_text())
    if (client['clock']['implementation'] != 'clock_gettime(CLOCK_MONOTONIC)'
            or type(client['clock']['origin_ns']) is not int):
        raise ValueError('overflow prefix requires the recorded common client clock')
    origin = client['clock']['origin_ns']
    summaries = []
    for path in sorted((directory / 'events').glob('*/metadata.json')):
        item = json.loads(path.read_text())
        if (item['error_code'] != 2 or item['native_dump_status'] != 2
                or item['native_init_status'] != 0 or item['attempted_rows'] <= item['rows']
                or tuple(item['fields']) != EVENT_FIELDS or item['itemsize'] != 8
                or item['byteorder'] != 'little' or item['unclosed_profile_frames'] != 0
                or item['observer'] != 'native_cupti_host_api_callbacks_and_outer_markers'
                or item['clock']['implementation'] != client['clock']['implementation']):
            raise ValueError('overflow-prefix inspection requires an explicit failed overflow trace')
        payload = path.with_name('events.bin').read_bytes()
        if len(payload) != item['rows'] * 80:
            raise ValueError('retained overflow prefix extent differs')
        counts, warmup_symbols, benchmark_rows, markers = Counter(), set(), 0, 0
        before_client, latest_end = 0, None
        for row in struct.iter_unpack('<10q', payload):
            if row[2] < row[1] or row[3] < 0:
                raise ValueError('invalid retained prefix clocks')
            benchmark_rows += int(row[4] >= 0)
            markers += int(row[0] in (0, 1))
            before_client += int(row[2] < origin)
            latest_end = row[2] if latest_end is None else max(latest_end, row[2])
            if row[0] in (0, 1):
                continue
            parts = item['labels'][str(row[0])].split(':', 4)
            if len(parts) != 5 or parts[0] != 'cuda_api':
                raise ValueError('unknown retained native API label')
            counts[parts[3]] += 1
            if row[2] < origin and _kernel_api(parts[3]) and parts[4] != '<no-symbol>':
                warmup_symbols.add(parts[4])
        summaries.append({'pid': item['pid'], 'retained_rows': item['rows'],
            'attempted_rows': item['attempted_rows'], 'error_code': item['error_code'],
            'retained_benchmark_rows': benchmark_rows, 'retained_outer_markers': markers,
            'retained_rows_before_client_origin': before_client,
            'all_retained_rows_before_client_origin': before_client == item['rows'],
            'latest_retained_end_relative_to_client_s': (None if latest_end is None else
                (latest_end - origin) * 1e-9),
            'api_name_counts_in_prefix': counts.most_common(),
            'observed_warmup_launch_symbols': sorted(warmup_symbols)})
    if not summaries:
        raise ValueError('no overflow trace exists')
    return {'status': 'failed_overflow_prefix_only', 'valid_complete_trace': False,
        'processes': summaries, 'production_price_eligible': False,
        'limitations': ['Only positive retained prefix counts and launch names are reported.',
            'No missed/first-use/coverage/steady-state frequency or timing inference from lost data.',
            'Nonlaunch symbol fields are ignored under the CUPTI callback contract.']}
