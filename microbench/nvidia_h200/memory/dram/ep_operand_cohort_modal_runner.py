"""One bounded H200 capture of an independent masked byte-ring program."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] if modal.is_local() else Path('/tmp')
CAMPAIGN = ROOT / 'profiler_output/microbench/nvidia_h200/serving_accuracy_ep_operand_cohort_20261109'
TASK = Path('/tmp/midlayer-ep-operand-cohort-20261109')
NAMES = ('ep_operand_cohort.cu', 'ep_operand_cohort_analysis.py', 'finite_strided_panel_analysis.py', 'plan.json')
IMAGE = modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12').apt_install('cuda-nsight-compute-12-9')
if modal.is_local():
    SOURCES = (HERE / NAMES[0], HERE / NAMES[1], HERE / NAMES[2], CAMPAIGN / NAMES[3])
    for source, name in zip(SOURCES, NAMES):
        IMAGE = IMAGE.add_local_file(source, remote_path=str(TASK / name))
APP = modal.App('midlayer-h200-ep-byte-cohort-20261109')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(arguments, timeout):
    try:
        result = subprocess.run(arguments, cwd=TASK, capture_output=True, text=True, timeout=timeout)
        return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'command': arguments}
    except subprocess.TimeoutExpired as error:
        return {'exit_code': 124, 'command': arguments, 'timeout': timeout,
                'stdout': error.stdout.decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
                'stderr': error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr}


@APP.function(gpu='H200', cpu=4, memory=8192, image=IMAGE, timeout=300,
              retries=0, single_use_containers=True)
def capture(expected_hashes):
    started = time.monotonic()
    raw = {'status': 'started', 'timing': [], 'counters': []}
    payload = b''
    sys.path.insert(0, str(TASK))
    from ep_operand_cohort_analysis import METRICS, summarize, evidence_bundle
    try:
        if [sha(TASK / name) for name in NAMES] != expected_hashes:
            raise ValueError('frozen capture inputs differ')
        plan = json.loads((TASK / 'plan.json').read_text())
        raw['plan'] = plan
        raw['GPU'] = command(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], 10)
        if raw['GPU']['exit_code'] or len(raw['GPU']['stdout'].splitlines()) != 1 or 'H200' not in raw['GPU']['stdout']:
            raise ValueError('exactly one authorized H200 required')
        raw['compile'] = command(['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-Xptxas=-v', NAMES[0], '-o', 'probe'], 30)
        if raw['compile']['exit_code']:
            raise RuntimeError('byte program compilation failed')
        raw['sass'] = command(['cuobjdump', '--dump-sass', 'probe'], 15)
        raw['resources'] = command(['cuobjdump', '--dump-resource-usage', 'probe'], 15)
        if raw['sass']['exit_code'] or raw['resources']['exit_code']:
            raise RuntimeError('byte program export failed')
        sass = raw['sass']['stdout']
        if ('LDGSTS.E.BYPASS.128' not in sass or any(opcode in sass for opcode in ('HGMMA.', 'HMMA.', 'LDL.', 'STL.'))
                or re.search(r'[1-9][0-9]* bytes (?:stack frame|spill stores|spill loads)', raw['compile']['stderr'])
                or '0 bytes spill stores, 0 bytes spill loads' not in raw['compile']['stderr']):
            raise ValueError('source/zero-spill boundary failed')
        cells = plan['physical_cells']
        for process in (0, 1):
            ordered = cells if process == 0 else list(reversed(cells))
            for cell in ordered:
                if time.monotonic() - started > 250:
                    raise TimeoutError('capture approaching its frozen function bound')
                result = command([str(TASK / 'probe'), *[str(value) for value in cell], '63'], 15)
                if result['exit_code']:
                    raw['failed_child'] = result
                    raise RuntimeError('byte-ring timing/numerical boundary failed')
                raw['timing'].append({'cell': cell, 'process': process,
                                      'measurement': json.loads(result['stdout']), 'execution': result})
            print(json.dumps({'process': process, 'timing_cells': len(ordered)}), flush=True)
        for cell in cells:
            if time.monotonic() - started > 250:
                raise TimeoutError('capture approaching its frozen function bound')
            result = command(['ncu', '--csv', '--page', 'raw', '--print-units', 'base',
                              '--replay-mode', 'application', '--cache-control', 'none', '--clock-control', 'none',
                              '--kernel-name', 'regex:cohort_copy', '--launch-skip', '3', '--launch-count', '1',
                              '--metrics', ','.join(METRICS), str(TASK / 'probe'),
                              *[str(value) for value in cell], '1'], 20)
            raw['counters'].append({'cell': cell, **result})
            if result['exit_code']:
                raise RuntimeError('byte-ring counter boundary failed')
        raw['analysis'] = summarize(raw, cells)
        raw['status'] = 'complete_qualified' if raw['analysis']['qualified'] else 'complete_unqualified'
    except Exception as error:
        raw['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raw['status'] = 'failed_retained'
    finally:
        raw['GPU_function_wall_s'] = time.monotonic() - started
        (TASK / 'raw.json').write_text(json.dumps(raw, indent=2) + '\n')
        print(json.dumps({'capture_status': raw['status'],
                          'capture_failure': raw['failure'] if 'failure' in raw else None}), flush=True)
        try:
            payload, members = evidence_bundle(TASK)
        finally:
            shutil.rmtree(TASK)
    return {'status': raw['status'], 'artifact_bundle': payload, 'members': members,
            'GPU_function_wall_s': time.monotonic() - started, 'temporary_directory_removed': not TASK.exists(),
            'persistent_storage': False}


@APP.local_entrypoint()
def main(budget_key: str, budget_tool: str, attempt: int = 1):
    if attempt not in (1, 2):
        raise ValueError('only the initial call and one explicit technical retry are authorized here')
    target = CAMPAIGN / f'attempt_{attempt}'
    target.mkdir(exist_ok=False)
    tool = Path(budget_tool)
    launch = {'app_id': APP.app_id, 'budget_key': budget_key,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in SOURCES},
              'runner_sha256': sha(Path(__file__)),
              'resource_bound': {'GPU': 'H200', 'cards': 1, 'seconds_including_startup_warmup_cleanup': 360,
                                 'GPU_hours': .1, 'USD_reserved': .75, 'automatic_retries': 0, 'Volumes': 0}}
    subprocess.run([sys.executable, str(tool), 'bind', '--key', budget_key, '--app-id', APP.app_id], check=True)
    (target / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(launch), flush=True)
    try:
        result = capture.remote([sha(path) for path in SOURCES])
        payload = result.pop('artifact_bundle')
        (target / 'evidence.zip').write_bytes(payload)
        result['bundle_sha256'] = hashlib.sha256(payload).hexdigest()
        result['app_id'] = APP.app_id
        (target / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'status': result['status'], 'app_id': APP.app_id,
                          'GPU_function_wall_s': result['GPU_function_wall_s']}), flush=True)
        subprocess.run([sys.executable, str(tool), 'terminal', '--key', budget_key], check=True)
    except Exception as error:
        (target / 'client_failure.json').write_text(json.dumps({'app_id': APP.app_id,
            'type': type(error).__name__, 'message': str(error),
            'must_verify_App_terminal_before_budget_terminal': True}, indent=2) + '\n')
        raise
