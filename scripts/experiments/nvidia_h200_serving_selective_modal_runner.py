"""One bounded2xH200 full MoE local-code observation, with shared-pool accounting."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import zipfile

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path('/root')
ROOT = Path('/tmp') if REMOTE else HERE.parents[2]
CAMPAIGN = ROOT / 'profiler_output/serving/nvidia_h200/serving_accuracy_moe_selective_boundary_20261109'
PROBE = Path('/tmp/selective_probe.py') if REMOTE else ROOT / 'dataset_pipeline/profiler/nvidia_h200_serving_boundary_probe.py'
HOOKS = Path('/tmp/selective_hooks.py') if REMOTE else ROOT / 'dataset_pipeline/profiler/nvidia_h200_serving_selective_hooks.py'
GEOMETRY = Path('/tmp/selective_geometry.json') if REMOTE else CAMPAIGN / 'geometry.json'
IMAGE = (modal.Image.from_registry('vllm/vllm-openai:v0.30.0', add_python='3.12').entrypoint([])
    .add_local_file(PROBE, remote_path='/tmp/selective_probe.py')
    .add_local_file(HOOKS, remote_path='/tmp/selective_hooks.py')
    .add_local_file(GEOMETRY, remote_path='/tmp/selective_geometry.json'))
APP = modal.App('midlayer-serving-moe-local-code-20261109')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(data, directory):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate evidence member')
        for name in names:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe evidence member')
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def record_cleanup(owner, process, errors, log):
    """Cleanup failures invalidate the capture but cannot discard its raw."""
    try:
        owner.stop_tree(process)
    except BaseException as exception:
        errors.append({'type': type(exception).__name__, 'message': str(exception)})
        traceback.print_exc(file=log)


@APP.function(gpu='H200:2', cpu=8, memory=98304, image=IMAGE, timeout=540,
    retries=0, max_containers=1, single_use_containers=True)
def capture(plan):
    started = time.monotonic()
    sources = (Path('/tmp/selective_probe.py'), Path('/tmp/selective_hooks.py'), Path('/tmp/selective_geometry.json'))
    with tempfile.TemporaryDirectory(prefix='midlayer_moe_selective_', dir='/tmp') as temporary:
        work = Path(temporary)
        environment = dict(os.environ)
        environment.update(HF_HOME=str(work / 'hf_home'), VLLM_NO_USAGE_STATS='1')
        command = ['/usr/bin/python3', str(sources[0]), '--selective-moe', '--geometry', str(sources[2]),
            '--hooks', str(sources[1]), '--output', str(work / 'candidate')]
        process, error, timed_out, identity, owner = None, None, False, None, None
        cleanup_errors = []
        with (work / 'collector.log').open('w') as log:
            try:
                if [sha(path) for path in sources] != plan['uploaded_source_sha256']:
                    raise ValueError('uploaded source or input differs from frozen plan')
                identity = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                    check=True, capture_output=True, text=True, timeout=10).stdout
                if len(identity.strip().splitlines()) != 2 or any('H200' not in line for line in identity.strip().splitlines()):
                    raise ValueError('requires exactly the authorized twoH200 GPUs')
                (work / 'bootstrap.json').write_text(json.dumps({'gpu_identity': identity,
                    'cpu_model': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')),
                    'affinity': sorted(os.sched_getaffinity(0)), 'source_sha256': [sha(path) for path in sources]}, indent=2) + '\n')
                spec = importlib.util.spec_from_file_location('selective_lifecycle_owner', sources[0])
                owner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(owner)
                process = subprocess.Popen(command, cwd=work, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 480
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        owner.stop_tree(process)
                        break
                    try:
                        process.wait(timeout=min(30, remaining))
                    except subprocess.TimeoutExpired:
                        print(json.dumps({'elapsed_s': time.monotonic() - started,
                            'server_log_exists': (work / 'candidate/server.stdout.log').is_file(),
                            'completed_client_result_files': len(list((work / 'candidate').glob('client_*.json')))}), flush=True)
            except BaseException as exception:
                error = {'type': type(exception).__name__, 'message': str(exception)}
                traceback.print_exc(file=log)
            finally:
                if process is not None:
                    record_cleanup(owner, process, cleanup_errors, log)
        files = []
        archive_data = io.BytesIO()
        with zipfile.ZipFile(archive_data, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob('*')):
                relative = path.relative_to(work)
                if not path.is_file() or any(part in ('hf', 'hf_home', 'compile_cache', '__pycache__') for part in relative.parts):
                    continue
                if path.suffix not in ('.json', '.log', '.bin'):
                    continue
                data = path.read_bytes()
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
                files.append({'path': relative.as_posix(), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()})
        receipt_path = work / 'candidate/receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        success = (process is not None and process.returncode == 0 and not timed_out and error is None and not cleanup_errors
            and receipt is not None and receipt['status'] == 'complete_candidate' and len(receipt['runs']) == 3)
        result = {'status': 'complete_candidate' if success else 'failed_retained', 'identity': identity,
            'exit_code': None if process is None else process.returncode, 'timed_out': timed_out,
            'error': error, 'cleanup_errors': cleanup_errors, 'command': command, 'files': files, 'plan': plan,
            'function_wall_s': time.monotonic() - started,
            'artifact_bundle': archive_data.getvalue(), 'no_volume': True}
    result['temporary_directory_removed'] = not Path(temporary).exists()
    result['function_wall_s'] = time.monotonic() - started
    return result


@APP.local_entrypoint()
def main(budget_tool: str, attempt: int = 1):
    if attempt not in (1, 2):
        raise ValueError('only the original capture and one technical retry are allowed')
    plan = json.loads((CAMPAIGN / ('plan.json' if attempt == 1 else 'retry_plan.json')).read_text())
    key = {1: '1107-serving-moe-selective-20261109-01',
           2: '1107-serving-moe-selective-20261109-02'}[attempt]
    if plan['key'] != key:
        raise ValueError('reservation key differs from explicit attempt')
    if [sha(path) for path in (PROBE, HOOKS, GEOMETRY)] != plan['uploaded_source_sha256'] or sha(HERE) != plan['modal_runner_sha256']:
        raise ValueError('local source/input differs from frozen plan')
    app_id = APP.app_id
    subprocess.run(['python', budget_tool, 'bind', '--key', key, '--app-id', app_id], check=True)
    directory = CAMPAIGN / f'attempt_{attempt}'
    directory.mkdir(exist_ok=False)
    (directory / 'app.json').write_text(json.dumps({'key': key, 'app_id': app_id}, indent=2) + '\n')
    try:
        result = capture.remote(plan)
        data = result.pop('artifact_bundle')
        (directory / 'evidence.zip').write_bytes(data)
        unpack(data, directory / 'extracted')
        for row in result['files']:
            path = directory / 'extracted' / row['path']
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('returned evidence extent or SHA differs')
        result.update(app_id=app_id, bundle_sha256=hashlib.sha256(data).hexdigest(), bundle_bytes=len(data))
        (directory / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({name: result[name] for name in ('status', 'app_id', 'function_wall_s', 'bundle_bytes')}))
    finally:
        subprocess.run(['python', budget_tool, 'terminal', '--key', key], check=True)


if __name__ == '__main__':
    main()
