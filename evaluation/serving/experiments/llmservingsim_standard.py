"""Run standard TP1 cases with existing, accepted profiles; no hardware collection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from evaluation.serving.experiments.llmservingsim import SOURCE, git_revision, read_json, write_json
from evaluation.serving.experiments.llmservingsim_targets import convert_requests
from evaluation.serving.metrics.llmservingsim_comparison import compare_requests, normalize_native_csv, summarize


HARDWARE = ("nvidia_a800", "metax_c500", "ascend_910b")
MODELS = ("qwen3-4b", "llama-3.1-8b")



def adapter_snapshot() -> dict:
    root = Path(__file__).resolve().parents[3]
    sources = (Path(__file__), root / 'evaluation/serving/experiments/llmservingsim.py',
               root / 'evaluation/serving/experiments/llmservingsim_targets.py',
               root / 'evaluation/serving/metrics/llmservingsim_comparison.py',
               root / 'scripts/evaluation/serving/run_llmservingsim.py')
    return {'adapter_commit': git_revision(root),
            'adapter_sources': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources},
            'adapter_worktree_dirty': bool(subprocess.check_output(
                ['git', 'status', '--porcelain', '--', *[str(path.relative_to(root)) for path in sources]],
                cwd=root, text=True).strip())}

def validate_profile_reuse(cell: dict, accepted_case: dict, accepted_run: dict) -> None:
    contract = cell['prediction_contract']
    settings = accepted_case['prediction_settings']
    for key in ('compute_dtype', 'tensor_parallel_size', 'pipeline_parallel_size',
                'expert_parallel_size', 'max_batched_tokens', 'max_model_len',
                'max_requests', 'kv_block_size_tokens', 'enable_prefix_caching',
                'enable_chunked_prefill'):
        if contract[key] != settings[key]:
            raise ValueError(f'existing LUT/runtime cannot be reused: {key}')
    provenance = cell['forward_provenance']
    capacity = provenance['kv_capacity']
    old = accepted_run['capacity']
    if (capacity['num_kv_blocks'] != old['midlayer_blocks']
            or capacity['available_kv_bytes_by_rank'] != [old['available_kv_bytes']]
            or provenance['model_identity']['config_sha256'] != old['runtime_memory_policy']['binding']['model_config_sha256']
            or not old['equal_block_bytes'] or not old['equal_block_count']):
        raise ValueError('existing LUT case differs in model identity or logical KV capacity')


def _execute_case(cell: dict, payload: dict, accepted: Path, runtime: Path,
                  python: Path, output: Path, load_case) -> dict:
    hardware, model, case_id = cell['hardware_id'], cell['model_id'], cell['case_id']
    original = accepted / 'targets' / hardware / model
    old_comparison = read_json(original / 'comparison.json')
    run_name = Path(old_comparison['native_run_manifest']).parent.name
    old_run_path = original / 'llmservingsim' / run_name
    old_run = read_json(old_run_path / 'run.json')
    old_case = read_json(original / 'case.json')
    validate_profile_reuse(cell, old_case, old_run)
    profile = runtime / 'profiler/perf' / hardware / old_case['native_model'] / 'bf16/tp1'
    for group in ('dense', 'per_sequence', 'attention'):
        if hashlib.sha256((profile / f'{group}.csv').read_bytes()).hexdigest() != old_run['profile_tables'][group]['sha256']:
            raise ValueError('installed LUT differs from accepted profile binding')
    if read_json(runtime / 'configs/model' / (old_case['native_model'] + '.json')) != old_case['model_config']:
        raise ValueError('native model configuration differs from accepted profile')
    if old_run['status'] != 'completed' or old_run['exit_code'] != 0:
        raise ValueError('profile reuse requires a completed accepted native simulation')
    canonical = Path(payload['dataset_root']) / hardware / model / case_id
    case = load_case(canonical)
    evidence = case.manifest['requests']['cached_hit_value_evidence']
    if (evidence['kind'] != 'scheduler_equivalent_representative'
            or evidence['representative_value'] != 0
            or evidence['net_prefix_work_reduction_tokens'] != 0):
        raise ValueError('native bridge requires independently evidenced zero-work-reduction representative')
    rows = [dict(request_id=str(i), arrival_time_ms=r.arrival_time_ms,
                 input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                 cached_hit_tokens=r.cached_hit_tokens) for i, r in enumerate(case.requests)]
    if len(rows) != cell['request_count'] or any(r['cached_hit_tokens'] != 0 for r in rows):
        raise ValueError('standard request count/cache semantics differ')
    native, mapping = convert_requests(rows)
    for i, row in enumerate(native):
        row['input_tok_ids'] = [i + 1] * row['input_toks']
        row['output_tok_ids'] = [i + 1] * row['output_toks']
    directory = output / 'targets' / hardware / model / case_id
    runs = directory / 'llmservingsim'
    simulation = runs / f'run-{len(list(runs.glob("run-*"))) + 1:04d}'
    simulation.mkdir(parents=True, exist_ok=False)
    link = directory / 'canonical'
    if link.is_symlink():
        if link.resolve() != canonical.resolve():
            raise ValueError('canonical retry binding differs')
    else:
        link.symlink_to(canonical.resolve(), target_is_directory=True)
    write_json(directory / 'request_mapping.json', {'requests': mapping})
    shutil.copyfile(old_run_path / 'config.json', simulation / 'config.json')
    workload = simulation / 'workload.jsonl'
    workload.write_text(''.join(json.dumps(row) + '\n' for row in native))
    settings = cell['prediction_contract']
    command = [str(python.absolute()), '-m', 'serving', '--cluster-config', str(simulation / 'config.json'),
               '--dataset', os.path.relpath(workload, runtime), '--output', str(simulation / 'sim.csv'),
               '--inputs-root', str(simulation / 'astra-inputs'), '--num-reqs', str(len(native)),
               '--max-num-seqs', str(settings['max_requests']),
               '--max-num-batched-tokens', str(settings['max_batched_tokens']),
               '--block-size', str(settings['kv_block_size_tokens']),
               '--network-backend', 'analytical', '--log-level', 'WARNING']
    manifest = {'case': cell['cell_id'], 'command': command, 'cwd': str(runtime),
                **adapter_snapshot(),
                'reused_accepted_run': str(old_run_path / 'run.json'),
                'standard_results': str(output / 'standard-input.json'),
                'profile_tables': old_run['profile_tables'], 'profile_metadata': old_run['profile_metadata'],
                'capacity': old_run['capacity'], 'cache_evidence': evidence,
                'source_manifest': str(SOURCE), 'compute_mode': 'LUT', 'network_mode': 'analytical',
                'status': 'running', 'exit_code': None, 'started_unix_s': time.time()}
    write_json(simulation / 'run.json', manifest)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='', ASCEND_RT_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1', TERM='dumb')
    library_path = str(accepted / 'build-env/lib')
    if 'LD_LIBRARY_PATH' in environment:
        library_path += ':' + environment['LD_LIBRARY_PATH']
    environment['LD_LIBRARY_PATH'] = library_path
    linkage = subprocess.check_output(['ldd', str(runtime / 'astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra')], env=environment, text=True)
    if 'not found' in linkage:
        raise ValueError('native ASTRA shared library dependency is unavailable')
    manifest['dynamic_library_path'] = library_path
    write_json(simulation / 'run.json', manifest)
    with (simulation / 'stdout.log').open('w') as stdout, (simulation / 'stderr.log').open('w') as stderr:
        process = subprocess.run(command, cwd=runtime, env=environment, stdout=stdout, stderr=stderr, check=False)
    manifest.update(status='executed' if process.returncode == 0 else 'failed', exit_code=process.returncode,
                    finished_unix_s=time.time())
    write_json(simulation / 'run.json', manifest)
    if process.returncode:
        raise RuntimeError(f'native standard simulation failed: {simulation}')
    expected = [{'request_id': row['request_id'], **request} for row, request in zip(mapping, native, strict=True)]
    normalized = normalize_native_csv(simulation / 'sim.csv', expected, old_case['native_model'])
    hits = re.findall(r'^NPU prefix hit prompt tokens:\s+([0-9]+)\s*$',
                      (simulation / 'stdout.log').read_text(), re.MULTILINE)
    if hits != ['0']:
        raise ValueError('native prefix-cache work does not match canonical representative')
    write_json(simulation / 'normalized.json', {'requests': normalized, 'summary': summarize(normalized)})
    truth, predicted, prediction_requests = [], [], {}
    paired = cell['metrics']['paired_request_latency']['metrics']
    for i, request in enumerate(mapping):
        base = {'request_id': str(i), 'input_tokens': request['input_tokens'],
                'output_tokens': request['output_tokens'], 'arrival_time_s': request['arrival_time_ms'] / 1000,
                'itl_s': []}
        observed, modeled = dict(base), dict(base)
        for field_ms, field_s in (('ttft_ms', 'ttft_s'), ('tpot_ms', 'tpot_s'), ('e2e_latency_ms', 'latency_s')):
            row = paired[field_ms][i]
            if row['request_idx'] != i:
                raise ValueError('standard paired request identity differs')
            observed[field_s] = row['measured_ms'] / 1000
            modeled[field_s] = row['predicted_ms'] / 1000
        observed['end_time_s'] = observed['arrival_time_s'] + observed['latency_s']
        modeled['end_time_s'] = modeled['arrival_time_s'] + modeled['latency_s']
        truth.append(observed)
        predicted.append(modeled)
        prediction_requests[f'request-{i:012d}'] = {field: modeled[field] for field in ('ttft_s', 'tpot_s', 'latency_s')}
    (directory / 'midlayer-request-metrics').mkdir()
    write_json(directory / 'midlayer-request-metrics/prediction.json', {'requests': prediction_requests,
               'source': 'unchanged paired metrics from standard-input.json; no Midlayer rerun'})
    comparison = {'case': cell['cell_id'], 'llmservingsim_status': 'completed',
                  'native_run_manifest': str(simulation / 'run.json'),
                  'midlayer_vs_truth': compare_requests(predicted, truth),
                  'llmservingsim_vs_truth': compare_requests(normalized, truth),
                  'profile_evidence': {**old_comparison['profile_evidence'], 'truth_runtime': case.manifest['runtime']},
                  'reused_accepted_run': str(old_run_path / 'run.json')}
    write_json(directory / 'comparison.json', comparison)
    manifest.update(status='completed', native_prefix_hit_prompt_tokens=0)
    write_json(simulation / 'run.json', manifest)
    print(f"completed {cell['cell_id']}", flush=True)
    return {'case': cell['cell_id'], 'status': 'completed', 'directory': str(directory)}


def run_standard_suite(runtime: Path, python: Path, accepted: Path, results: Path,
                       output: Path, jobs: int, load_case) -> dict:
    accepted, output, runtime = accepted.resolve(), output.resolve(), runtime.resolve()
    if jobs < 1:
        raise ValueError('standard baseline CPU jobs must be positive')
    source = read_json(SOURCE)
    if git_revision(runtime) != source['commit']:
        raise ValueError('baseline source differs from pinned version')
    payload = read_json(results)
    selected = [c for c in payload['cells'] if c['hardware_id'] in HARDWARE and c['model_id'] in MODELS
                and c['case_id'] in ('tp1_qps4', 'tp1_qps64')]
    if len(selected) != 12 or len({c['cell_id'] for c in selected}) != 12:
        raise ValueError('expected complete 12-case TP1 standard expansion')
    repository = Path(__file__).resolve().parents[3]
    if not output.is_relative_to(repository / 'experiment_logs'):
        raise ValueError('standard baseline outputs must remain in this checkout experiment_logs')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'standard-input.json', payload)
    isolated = output / 'runtime'
    if isolated.exists():
        raise ValueError('use a fresh output directory to preserve prior runs')
    shutil.copytree(runtime, isolated, symlinks=True, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    for original_name, link_name in (('report.md', 'reused-six-case-report.md'),
                                     ('main-acceptance.json', 'reused-six-case-acceptance.json')):
        (output / link_name).symlink_to(accepted / original_name)
    for hardware in HARDWARE:
        for model in MODELS:
            directory = output / 'targets' / hardware / model
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'tp1_qps16').symlink_to(accepted / 'targets' / hardware / model, target_is_directory=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        cases = list(pool.map(lambda cell: _execute_case(cell, payload, accepted, isolated,
                                                        python, output, load_case), selected))
    summary = {'accepted_lut_results': str(accepted), 'standard_results': str(results.resolve()),
               'new_cases': cases, 'reused_cases': 6, 'jobs': jobs, 'hardware_execution': False}
    write_standard_suite_report(output, summary)
    return summary


def write_standard_suite_report(output: Path, summary: dict) -> None:
    """Record new execution separately from the six reused, previously accepted cases."""
    write_json(output / 'standard-suite.json', summary)
    lines = ['# LLMServingSim standard TP1 workload expansion', '',
             '12 new CPU simulations reuse the six accepted hardware/model BF16 TP1 LUTs; '
             '6 QPS16 cases are reused unchanged. No hardware profiling or weight download. '
             'This execution report is not a main-agent acceptance receipt.', '',
             '[Original six-case acceptance](reused-six-case-acceptance.json) · '
             '[Original LUT comparison and limitations](reused-six-case-report.md)', '',
             '| Case | Requests | Status | TTFT mean error | TPOT mean error | E2E mean error |',
             '|---|---:|---|---:|---:|---:|']
    for entry in summary['new_cases']:
        directory = output / 'targets' / entry['case']
        comparison = read_json(directory / 'comparison.json')
        run_name = Path(comparison['native_run_manifest']).parent.name
        run = read_json(directory / 'llmservingsim' / run_name / 'run.json')
        if run['status'] != 'completed' or run['exit_code'] != 0:
            raise ValueError('expansion summary contains incomplete native run')
        metrics = comparison['llmservingsim_vs_truth']
        errors = ' | '.join(f"{metrics[field]['mean_relative_error_percent']:+.2f}%"
                            for field in ('ttft_s', 'tpot_s', 'latency_s'))
        lines.append(f"| [{entry['case']}](targets/{entry['case']}/comparison.json) | {metrics['ttft_s']['prediction']['count']} | [completed](targets/{entry['case']}/llmservingsim/{run_name}/run.json) | {errors} |")
    lines += ['', 'Per-case comparison.json binds request-normalized outputs, standard measured/Midlayer '
              'metrics, and native run.json. Run manifests identify commands, reused LUT source, '
              'dynamic-library environment, cache-equivalence evidence, and the executed adapter source snapshot.', '',
              'Limits: native eager dummy-weight LUT sampling and interpolation/extrapolation remain; '
              'host/Graph runtime equivalence is not claimed. QPS changes arrival rate, not the LUT ownership '
              'or model/TP identity. The native network backend is analytical; compute remains LUT-based.', '']
    (output / 'report.md').write_text('\n'.join(lines))
