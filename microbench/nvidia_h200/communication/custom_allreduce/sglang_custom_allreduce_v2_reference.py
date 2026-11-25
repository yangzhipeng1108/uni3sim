#!/usr/bin/env python3
"""Publish the SGLang CustomAllReduceV2 graph-node reference and YAML table.

Reads one runner report.json (sglang_custom_allreduce_v2_latency_runner.py),
writes microbench/nvidia_h200/references/sglang_custom_allreduce_v2_graph_node_sm90.json
(the report plus derivation metadata) and prints the
``interconnect.sglang_custom_all_reduce_service`` block: per participant
count, the slowest-rank median service of every sampled payload below
``max_payload_bytes``.  No production or evaluation duration is used.
Usage: python sglang_custom_allreduce_v2_reference.py <report.json>
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'microbench/nvidia_h200/references/sglang_custom_allreduce_v2_graph_node_sm90.json'


def main():
    src = Path(sys.argv[1])
    report = json.loads(src.read_text())
    if report['mechanism'] != 'sglang_custom_all_reduce_v2_cuda_graph_node_service':
        raise ValueError('not an SGLang custom all-reduce v2 report')
    if report['production_or_evaluation_duration_consumed'] is not False:
        raise ValueError('report consumed production durations')
    tables = {}
    for world, block in report['world_sizes'].items():
        tables[world] = {str(row['payload_bytes']): round(row['custom_all_reduce_ns'], 1)
                         for row in block['summary']
                         if row['payload_bytes'] < report['custom_all_reduce_max_bytes']}
    published = dict(report)
    published['schema'] = 'nvidia_h200_sglang_custom_allreduce_v2_graph_node_reference_v1'
    published['derivation'] = {
        'source_report_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
        'rule': 'slowest-rank median of 30 replays / 64 chained nodes per payload',
        'service_ns_by_participants': tables,
    }
    OUT.write_text(json.dumps(published, indent=1) + '\n')
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    env = report['environment']
    print('  sglang_custom_all_reduce_service:')
    print('    model: sglang_custom_all_reduce_v2_graph_node_v1')
    print('    duration_unit: ns_per_collective')
    print('    timer_domain: cuda_event_around_graph_replay_divided_by_chain')
    print('    lookup_rule: step_down_largest_sampled_payload_not_exceeding_request_floor_at_smallest_sample')
    print(f"    software_revision: sglang_{env['sglang_version']}_torch_{env['torch_version']}_driver_{env['driver_version']}")
    print(f"    max_payload_bytes: {report['custom_all_reduce_max_bytes']}")
    print(f"    reference_path: {OUT.relative_to(ROOT)}")
    print(f'    reference_sha256: {sha}')
    print('    production_or_evaluation_duration_consumed: false')
    print('    service_ns_by_participants:')
    for world, table in tables.items():
        print(f'      {world}:')
        for payload, ns in table.items():
            print(f'        {payload}: {ns}')


if __name__ == '__main__':
    main()
