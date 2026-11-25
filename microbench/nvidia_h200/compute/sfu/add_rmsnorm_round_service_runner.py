#!/usr/bin/env python3
"""Capture and derive the H200 block-256 Add-RMSNorm source-round service.

Owner copy of ``microbench/nvidia_a800/compute/sfu/add_rmsnorm_round_service_runner.py``:
one process runs ``<binary> <device> 5`` (five trials of the first and
resident cells for both source loops, 512 rounds each), and the candidate is
derived with the A800 rule:

    span[cell]   = median over trials of maximum_sm_span_ns
    first_ns     = span[first] / 512
    resident_ns  = (span[resident] - span[first]) / ((8 - 1) * 512)
    rate         = 1e9 / ns          (every cell spread <= 10 %)

Hardware coupling changed: cells first = 132 CTAs and resident = 8 x 132 =
1056 CTAs (were 108 / 864); physical GPU 3 of the H200 host; the A800
production-artifact / SASS identity step (vLLM 0.11 sm_80 ``_C.abi3.so``) has
no sm_90 counterpart and is replaced by the recorded ptxas resource usage and
the probe's runtime 8-CTA occupancy check; the capture is clock-gated.  The
timer is ``%globaltimer`` (ns), so no clock conversion is involved.

The runner writes ``add_rmsnorm_round_service_sm90.json`` and
``add_rmsnorm_block256_pipeline_sm90.json`` candidates (A800 reference
schemas) into the run directory; they are copied to
``microbench/nvidia_h200/references/`` unchanged.

usage (H200 host): python3 add_rmsnorm_round_service_runner.py --run-id <id> \
    --out-root <dir> --device 3 --baseline-commit <sha>
"""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as capture  # noqa: E402

RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name('add_rmsnorm_round_service.cu')
SM_COUNT = 132
THREADS = 256
RESIDENT_CTAS = 8
ROUNDS = 512
TRIALS = 5
FUNCTIONS = ('add_square', 'scale_multiply')
CELLS = {'first': SM_COUNT, 'resident': RESIDENT_CTAS * SM_COUNT}


REDUCTION_REFERENCE_SHA256 = capture.sha256(
    RUNNER_PATH.parents[2] / 'references' / 'block_reduce_latency_sm90.json')


def probe_resources(ptxas_text):
    resources = {}
    for block in ptxas_text.split("Compiling entry function '")[1:]:
        name = block.split("'", 1)[0]
        used = re.search(r'Used (\d+) registers[^\n]*', block).group(0)
        shared = re.search(r'(\d+) bytes smem', used)
        spills = re.search(r'(\d+) bytes spill stores', block).group(1)
        function = 'add_square' if 'RoundFunction0' in name else 'scale_multiply'
        resources[function] = {'registers_per_thread': int(used.split()[1]),
                               'static_shared_bytes': int(shared.group(1)) if shared else 0,
                               'local_bytes': int(spills)}
    if set(resources) != set(FUNCTIONS):
        raise ValueError(f'ptxas resource lines missing: {resources}')
    return resources


def load(path):
    rows = list(csv.DictReader(Path(path).open()))
    if len(rows) != TRIALS * len(FUNCTIONS) * len(CELLS):
        raise ValueError(f'unexpected row count {len(rows)}')
    for row in rows:
        grid = CELLS[row['cell']]
        if (int(row['grid_blocks']) != grid or int(row['block_threads']) != THREADS
                or int(row['rounds']) != ROUNDS
                or int(row['resident_cta_limit']) != RESIDENT_CTAS
                or int(row['active_sms']) != SM_COUNT
                or int(row['minimum_ctas_per_sm']) != grid // SM_COUNT
                or int(row['maximum_ctas_per_sm']) != grid // SM_COUNT):
            raise ValueError(f'row contract violated: {row}')
    return rows


def derive(rows):
    functions = {}
    stable = True
    for function in FUNCTIONS:
        spans, summaries = {}, {}
        for cell in CELLS:
            values = [int(r['maximum_sm_span_ns']) for r in rows
                      if r['function'] == function and r['cell'] == cell]
            center = median(values)
            spread = (max(values) - min(values)) / center
            stable = stable and spread <= 0.10
            spans[cell] = center
            summaries[cell] = {'median_maximum_sm_span_ns': center,
                               'maximum_relative_spread': spread}
        first = spans['first'] / ROUNDS
        resident = (spans['resident'] - spans['first']) / ((RESIDENT_CTAS - 1) * ROUNDS)
        if min(first, resident) <= 0:
            raise ValueError(f'{function}: non-positive increment')
        functions[function] = {'cell_summaries': summaries,
                               'first_ns': first, 'resident_ns': resident}
    if not stable:
        raise ValueError('a cell exceeds the 10 % spread gate')
    return functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--baseline-commit', required=True)
    args = parser.parse_args()
    out = args.out_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    identity = capture.device_identity(args.device)
    build = capture.build(SOURCE_PATH, out / 'build' / 'add_rmsnorm_round_service')
    warmup = capture.build_warmup(out / 'build')
    verdict = capture.run_gated('capture', [([build['binary'], 0, TRIALS], out / 'raw.csv')],
                                out, args.device, warmup)
    raw_path = Path(verdict['stdout_paths'][0])
    functions = derive(load(raw_path))
    clock = capture.clock_summary([verdict])
    raw_dir = f'profiler_output/microbench/nvidia_h200/{args.run_id}'
    provenance = {
        'source': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_round_service.cu',
        'source_sha256': build['source_sha256'],
        'runner': 'microbench/nvidia_h200/compute/sfu/add_rmsnorm_round_service_runner.py',
        'runner_sha256': capture.sha256(RUNNER_PATH),
        'helper_sha256': capture.sha256(capture.__file__),
        'baseline_commit': args.baseline_commit,
        'probe_binary_sha256': build['binary_sha256'],
        'nvcc_version': build['nvcc_version'], 'ptxas_verbose': build['ptxas_verbose'],
        'raw_dir': raw_dir,
        'remote_raw_dir': str(out), 'raw_sha256': capture.sha256(raw_path),
        'clock_gate': clock,
        'accepted_captures': clock['accepted_captures'],
        'rejected_captures': clock['rejected_captures'],
    }
    measurements = {'time_unit': 'ns_per_cta_round',
                    'service_rate_unit': 'cta_rounds_per_sm_second'}
    for function, value in functions.items():
        measurements[function] = {
            cell: {'incremental_ns_per_cta_round': value[f'{cell}_ns'],
                   'service_rate_per_sm_s': 1e9 / value[f'{cell}_ns'],
                   'maximum_relative_spread':
                       value['cell_summaries'][cell]['maximum_relative_spread']}
            for cell in CELLS}
    round_service = {
        'schema': 'nvidia_h200_add_rmsnorm_round_service_reference_v1',
        'hardware_id': 'nvidia_h200', 'device_name': identity['name'],
        'device_uuid': identity['uuid'], 'compute_capability': 'sm_90',
        'physical_sm_count': SM_COUNT, 'source_baseline_commit': args.baseline_commit,
        'capture': {'path': raw_dir, 'raw_sha256': provenance['raw_sha256'],
                    'trials': TRIALS,
                    'candidate_sha256': 'not_applicable_reference_written_directly',
                    'provenance_sha256': 'not_applicable_provenance_embedded',
                    'preflight_cleanup': 'target_device_no_compute_processes',
                    'postflight_cleanup': 'runner_exits_after_single_probe_process'},
        'tool_identity': {**{key: 'not_recorded_on_h200_host' for key in (
                              'cuobjdump_sha256', 'cuobjdump_version', 'nvcc_sha256',
                              'nvidia_smi_sha256', 'probe_resource_dump_sha256',
                              'probe_sass_dump_sha256')},
                          'source_sha256': build['source_sha256'],
                          'runner_sha256': provenance['runner_sha256'],
                          'probe_binary_sha256': build['binary_sha256'],
                          'nvcc_version': build['nvcc_version'],
                          'driver_version': identity['driver_version']},
        'production_artifact': {
            **{key: 'not_applicable_on_h200' for key in (
                'binary_path', 'binary_sha256', 'cubin_name', 'cubin_sha256',
                'function_sass_sha256', 'instruction_count', 'local_bytes',
                'registers_per_thread', 'rejected_activation_cubin', 'required_opcodes',
                'resource_dump_sha256', 'static_shared_bytes', 'symbol')},
            'status': 'not_applicable_on_h200',
            'reason': ('No vLLM 0.11.0 sm_90 production binary exists on the H200 host; '
                       'the probe keeps the vLLM 0.11.0 source loops.')},
        'probe_contract': {
            'block_threads': THREADS, 'rounds': ROUNDS, 'resident_ctas_per_sm': RESIDENT_CTAS,
            'cells': {cell: {'grid_blocks': grid, 'ctas_per_sm': grid // SM_COUNT}
                      for cell, grid in CELLS.items()},
            'supported_grid_scope': 'grid_blocks <= 1056',
            'supported_service_scope': ['first_cta_source_round',
                                        'resident_cta_source_round'],
            'probe_ptxas_resource_usage': build['ptxas_verbose'],
            'probe_resources': probe_resources(build['ptxas_verbose']),
            'required_probe_opcodes': 'not_checked_no_cuobjdump_on_h200_host',
            'forbidden_probe_opcodes_absent': 'not_checked_no_cuobjdump_on_h200_host',
        },
        'measurements': measurements,
        'allowed_conclusion': ('The independent block-256 probe characterizes first and '
                               'resident CTA Add-RMSNorm source-round service on the captured '
                               'H200 through grid 1056.'),
        'gates': {'all_cells_stable_within_10_percent': True,
                  'probe_resource_and_sass_contract_exact': 'ptxas_resources_recorded_sass_not_checked',
                  'all_incremental_services_positive': True,
                  'production_artifact_identity_exact': 'not_applicable_on_h200',
                  'clock_gate_passed': True},
        'decision': {'status': 'mechanism_candidate_only', 'mechanism_publishable': True,
                     'forward_eligible': False,
                     'reason': ('Supports only block-256 first and resident source-round '
                                'service through grid 1056.')},
        'excluded_conclusions': [
            'a complete Add-RMSNorm duration',
            'a reduction or reciprocal-square-root service',
            'a production-specific correction factor',
            'queued CTA service or grids above 1056',
            'direct Forward consumption before composition review'],
        'provenance': provenance,
        'production_or_evaluation_duration_consumed': False,
        'production_predictor_consumption_allowed': False,
    }
    pipeline = {
        'schema': 'nvidia_h200_add_rmsnorm_block256_pipeline_reference_v1',
        'hardware_id': 'nvidia_h200',
        'software_revision': 'vllm_0.11.0_source_cuda_12.9_sm90',
        'production_or_evaluation_duration_consumed': False,
        'source_service_reference': {
            'path': 'microbench/nvidia_h200/references/add_rmsnorm_round_service_sm90.json',
            'sha256': 'see_provenance_raw_sha256_same_capture',
            'time_unit': 'ns_per_cta_round'},
        'source_contract': {
            'dtype': 'bf16', 'vector_width_elements': 8, 'source_round_elements': 2048,
            'block_threads': THREADS, 'registers_per_thread': 32,
            'static_shared_memory_bytes': 168, 'resident_ctas_per_sm': RESIDENT_CTAS,
            'minimum_grid_blocks': 1, 'maximum_grid_blocks': RESIDENT_CTAS * SM_COUNT,
            'hidden_domain': 'hidden_size % source_round_elements == 0',
            'production_function_sass_sha256': 'not_applicable_on_h200',
            'grid_derivation': ('The source launches one CTA per row. At most eight '
                                '256-thread CTAs reside on each of 132 SMs from the static '
                                'thread, warp, register, shared-memory, and block limits, '
                                'so grids through 1056 contain no queued CTA.')},
        'service_parameters': {
            'timer_domain': 'globaltimer_same_cta', 'unit': 'ns_per_cta_source_round',
            'add_square_first': functions['add_square']['first_ns'],
            'add_square_resident': functions['add_square']['resident_ns'],
            'scale_multiply_first': functions['scale_multiply']['first_ns'],
            'scale_multiply_resident': functions['scale_multiply']['resident_ns']},
        'composition_contract': {
            'ordered_stages': ['add_square_source_round',
                               'existing_block_reduction_and_publication',
                               'scale_multiply_source_round'],
            'resident_cta_source_and_reduction_stages_share_warp_issue': True,
            'independent_vector_reduction_stage_overlap_allowed': False,
            'each_stage_composed_once': True, 'generic_fma_alu_replaced': True,
            'generic_complete_memory_task_replaced': True,
            'aggregate_l2_dram_capacity_envelope_retained': True,
            'reduction_reference': 'microbench/nvidia_h200/references/block_reduce_latency_sm90.json',
            'reduction_report_sha256': REDUCTION_REFERENCE_SHA256},
        'decision': {'status': 'accepted_after_pipeline_composition_review',
                     'forward_eligible': True,
                     'reason': ('Same composition as the A800 reference; the H200 reference '
                                'supplies only first/resident source-round service and no '
                                'production duration.')},
        'excluded_conclusions': [
            'a complete AddRMSNorm duration', 'hidden sizes with a partial source round',
            'block widths other than 256', 'queued CTA service or grids above 1056',
            'per-shape timing interpolation or residual correction'],
        'provenance': provenance,
    }
    (out / 'add_rmsnorm_round_service_sm90.json').write_text(
        json.dumps(round_service, indent=1) + '\n')
    (out / 'add_rmsnorm_block256_pipeline_sm90.json').write_text(
        json.dumps(pipeline, indent=2) + '\n')
    print(json.dumps(pipeline['service_parameters']), json.dumps(clock))


if __name__ == '__main__':
    main()
