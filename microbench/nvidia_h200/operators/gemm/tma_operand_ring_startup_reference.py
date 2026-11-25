"""Derive a conservative TMA startup from independent transport controls.

Reads mechanism raw and hardware latency only; never reads evaluation cases
or production durations. The selection rule is fixed before comparison.
"""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'profiler_output/microbench/nvidia_h200/gemm_ring_service_20261107_startup/raw.json'
REFERENCE = ROOT / 'microbench/nvidia_h200/references/tma_operand_ring_startup_sm90.json'
BOUNDARY = ROOT / 'profiler_output/microbench/nvidia_h200/tma_ring_boundary_20261107/raw.json'


def boundary_prefix():
    """Joint-minus-empty lower envelope, fixed before production comparison."""
    payload = BOUNDARY.read_bytes()
    raw = json.loads(payload)
    if (raw['build_exit_code'] != 0 or raw['run_exit_code'] != 0
            or raw['production_duration_consumed'] is not False):
        raise ValueError('independent boundary capture failed')
    source = BOUNDARY.parent / 'tma_cluster_ring_service.cu'
    runner = BOUNDARY.parent / 'tma_cluster_ring_service_modal_runner.py'
    if (hashlib.sha256(source.read_bytes()).hexdigest() != raw['source_sha256']
            or hashlib.sha256(runner.read_bytes()).hexdigest() != raw['runner_sha256']):
        raise ValueError('independent boundary source identity drifted')
    rows = [json.loads(line) for line in raw['stdout'].splitlines()]
    for row in rows:
        if any(type(row[name]) is not int for name in ('grid', 'cluster', 'depth', 'mode')):
            raise ValueError('boundary geometry must contain integers')
        for name in ('median_event_ns', 'median_cta_interval_ns', 'median_max_cta_interval_ns'):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('boundary intervals must be finite nonnegative times')
    expected = {(g, c, d, m) for g, c, d, m in itertools.product(
        (32, 72, 132), (1, 2, 4, 8), (4, 16), (0, 1, 2, 3)) if g % c == 0}
    indexed = {(r['grid'], r['cluster'], r['depth'], r['mode']): r for r in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError('independent boundary sweep incomplete or duplicated')
    if not all(r['timestamp_check_ok'] is True and r['threads'] == 384
            and r['dynamic_smem_bytes'] == 196608 for r in rows):
        raise ValueError('boundary capture protocol drifted')
    controls = []
    for grid, cluster, depth in sorted({key[:3] for key in indexed}):
        empty, joint = indexed[(grid, cluster, depth, 0)], indexed[(grid, cluster, depth, 3)]
        event = joint['median_event_ns'] - empty['median_event_ns']
        cta = joint['median_cta_interval_ns'] - empty['median_cta_interval_ns']
        if event <= 0 or cta <= 0:
            raise ValueError('no uniform positive boundary prefix; do not select positive points')
        controls.append({'grid': grid, 'cluster': cluster, 'depth': depth,
            'joint_minus_empty_event_ns': event, 'joint_minus_empty_cta_ns': cta})
    prefix = min(min(r['joint_minus_empty_event_ns'], r['joint_minus_empty_cta_ns']) for r in controls)
    capture = {'path': str(BOUNDARY.relative_to(ROOT)),
        'raw_sha256': hashlib.sha256(payload).hexdigest(),
        'source_snapshot_directory': str(BOUNDARY.parent.relative_to(ROOT)),
        'source': str(source.relative_to(ROOT)), 'source_sha256': raw['source_sha256'],
        'runner': str(runner.relative_to(ROOT)), 'runner_sha256': raw['runner_sha256']}
    return prefix, controls, capture

def derive():
    raw_bytes = RAW.read_bytes()
    raw = json.loads(raw_bytes)
    if raw['run_exit_code'] != 0 or raw['production_duration_consumed'] is not False:
        raise ValueError('independent mechanism capture is not valid')
    rows = [json.loads(line) for line in raw['stdout'].splitlines()]
    if len(rows) != 96:
        raise ValueError('incomplete zero/single/dual descriptor sweep')
    config = yaml.safe_load((ROOT / 'configs/hardware/nvidia_h200.yaml').read_text())
    memory = config['memory_subsystem']
    first_request_ns = 1e9 * (
        memory['l2_hit_latency_local_cycles'] + memory['dram_miss_extra_cycles']) / config['clock_domains']['sm']
    groups = collections.defaultdict(dict)
    for row in rows:
        if row['same_operand_replay'] != 0:
            continue
        key = (row['grid'], row['depth'], row['shared_rows'])
        if row['rounds'] in groups[key]:
            raise ValueError('duplicate mechanism point')
        groups[key][row['rounds']] = row['median_ns']
    details = []
    for (grid, depth, shared_rows), samples in sorted(groups.items()):
        if set(samples) != {0, 8, 32, 128}:
            raise ValueError('mechanism secant endpoints are absent')
        steady = (samples[128] - samples[32]) / 96
        residual = samples[8] - samples[0] - 8 * steady
        details.append({'grid': grid, 'depth': depth, 'shared_rows': shared_rows,
            'steady_ns_per_round': steady, 'finite_residual_ns': residual,
            'additional_startup_ns': residual - first_request_ns})
    transport_fill = min(row['additional_startup_ns'] for row in details)
    if transport_fill <= 0:
        raise ValueError('no independently positive startup envelope')
    initialization, boundary_controls, boundary_capture = boundary_prefix()
    startup = transport_fill + initialization
    scope = {'dtype_element_bytes': 2, 'threads_per_cta': 384, 'tile_k': 64,
        'minimum_pipeline_depth': 4, 'maximum_pipeline_depth': 16,
        'minimum_rounds': 8, 'maximum_rounds': 128, 'one_wave_grid_ctas': 132,
        'transport': 'tma', 'minimum_grid_ctas': 32}
    reference = {'schema': 'finite_mixed_operand_stream_startup_reference_v1',
        'hardware_id': 'nvidia_h200', 'device_name': 'NVIDIA H200',
        'mechanism': 'finite_mixed_operand_stream_startup',
        'production_or_evaluation_duration_consumed': False,
        'source': str((RAW.parent / 'tma_operand_ring_service.cu').relative_to(ROOT)),
        'source_sha256': raw['sources']['microbench/nvidia_h200/operators/gemm/tma_operand_ring_service.cu'],
        'runner': str((RAW.parent / 'tma_operand_ring_service_modal_runner.py').relative_to(ROOT)),
        'runner_sha256': raw['runner_sha256'],
        'captures': [{'path': str(RAW.relative_to(ROOT)),
            'raw_sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'source_snapshot_directory': str(RAW.parent.relative_to(ROOT))}, boundary_capture],
        'probe_contract': {'description': 'MMA-free two-dimensional TMA operand ring, one 64-row private box and zero/one shared 16-row box, 32-node graph replay, cold-intended eviction before each graph; plus an independent no-TMA/no-math full/empty-barrier initialization and cluster-sync prefix, paired with the same timestamp-writing empty control. No matrix, library, output epilogue or production shape.',
            'clock_status': 'provisional: two-second transport warmup, endpoint clocks recorded; no per-configuration busy trace',
            'scope_approximation': 'launch-local transport startup envelope, not native ABI equivalence; no transfer of steady byte rates or multicast behavior',
            'derivation': str(Path(__file__).resolve().relative_to(ROOT))},
        'derived_model': {'selection_rule': 'Transport fill: minimum over all cold-intended grid/depth/single-or-dual-descriptor controls of t8-t0-8*(t128-t32)/96, minus the existing independent full DRAM first-request latency. The discarded zero-round program is separately represented by the predeclared minimum of joint-minus-empty event and same-CTA intervals across every boundary geometry; every difference must be positive. Add the two disjoint program phases once per launch, never per K tile or split. Neither rule reads production data.',
            'existing_first_request_ns': first_request_ns, 'controls': details,
            'operand_transport_fill_ns': transport_fill,
            'ring_initialization_prefix_ns': initialization,
            'ring_initialization_controls': boundary_controls,
            'one_wave_mixed_tier_startup_ns': startup},
        'scope': scope,
        'allowed_use': 'One additive launch-local startup for a declared TMA K64 two-byte operand ring inside the measured concurrency/depth/round domain; aggregate transport approximation.',
        'forbidden_use': 'No production duration/APE fitting; no per-shape time price, native multicast, raster, reducer ABI or steady-bandwidth certification.'}
    REFERENCE.write_text(json.dumps(reference, indent=2) + '\n')
    return reference

if __name__ == '__main__':
    result = derive()
    print(result['derived_model']['one_wave_mixed_tier_startup_ns'])
