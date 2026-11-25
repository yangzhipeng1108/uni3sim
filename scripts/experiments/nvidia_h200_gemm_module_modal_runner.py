"""One bounded, untimed H200 GEMM module/launch observation."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import modal

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if modal.is_local() else Path('/tmp')
CAMPAIGN_NAME = (os.environ['MIDLAYER_GEMM_MODULE_CAMPAIGN']
    if 'MIDLAYER_GEMM_MODULE_CAMPAIGN' in os.environ else 'serving_accuracy_gemm_modules_20261109')
if Path(CAMPAIGN_NAME).name != CAMPAIGN_NAME or CAMPAIGN_NAME in ('.', '..'):
    raise ValueError('module campaign must be one directory name')
CAMPAIGN = ROOT / 'profiler_output/serving/nvidia_h200' / CAMPAIGN_NAME
INPUT = Path('/tmp/midlayer_gemm_module_inputs')
NAMES = ('nvidia_h200_gemm_module.cc', 'nvidia_h200_gemm_module_probe.py', 'plan.json', 'sdk_manifest.json')
IMAGE = modal.Image.from_registry('vllm/vllm-openai:v0.30.0', add_python='3.12').entrypoint([])
if modal.is_local():
    SOURCES = (ROOT / 'dataset_pipeline/profiler' / NAMES[0],
        ROOT / 'dataset_pipeline/profiler' / NAMES[1], CAMPAIGN / NAMES[2], CAMPAIGN / NAMES[3])
    for source, name in zip(SOURCES, NAMES, strict=True):
        IMAGE = IMAGE.add_local_file(source, remote_path=str(INPUT / name))
    IMAGE = IMAGE.add_local_dir(ROOT / 'profiler_output/serving/nvidia_h200/serving_accuracy_native_api_boundary_20261108/sdk_input',
        remote_path=str(INPUT / 'sdk_wheels'))
APP = modal.App('midlayer-h200-gemm-modules-20261109')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


@APP.function(gpu='H200', cpu=4, memory=16384, image=IMAGE, timeout=180,
    retries=0, single_use_containers=True)
def capture(hashes, source_commit, reservation_key):
    started = time.monotonic()
    try:
        if [sha(INPUT / name) for name in NAMES] != hashes:
            raise ValueError('uploaded frozen inputs differ')
        identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
            '--format=csv,noheader'], text=True, timeout=10)
        if len(identity.splitlines()) != 1 or 'H200' not in identity:
            raise ValueError('exactly one authorized H200 required')
        with tempfile.TemporaryDirectory(prefix='midlayer-gemm-module-', dir='/tmp') as temporary:
            work = Path(temporary)
            failure = None
            exit_code = None
            try:
                for name in NAMES:
                    shutil.copyfile(INPUT / name, work / name)
                sdk = work / 'sdk'
                for item in json.loads((INPUT / 'sdk_manifest.json').read_text()):
                    name = item['filename']
                    if Path(name).name != name:
                        raise ValueError('invalid SDK filename')
                    wheel = INPUT / 'sdk_wheels' / name
                    if wheel.stat().st_size != item['bytes'] or sha(wheel) != item['sha256']:
                        raise ValueError('SDK wheel content binding differs')
                    with zipfile.ZipFile(wheel) as archive:
                        for member in archive.infolist():
                            if not (sdk / member.filename).resolve().is_relative_to(sdk.resolve()):
                                raise ValueError('SDK archive member leaves task directory')
                        archive.extractall(sdk)
                sdk_root = sdk / 'nvidia/cu13'
                library = sdk_root / 'lib/libcupti.so.13'
                output = work / 'module_observer.so'
                command = ['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
                    '-Wall', '-Wextra', '-Werror', '-I' + str(sdk_root / 'include'),
                    str(INPUT / NAMES[0]), str(library), '-Wl,-rpath,' + str(library.parent), '-o', str(output)]
                (work / 'compiler.json').write_text(json.dumps({'command': command,
                    'sdk_library_sha256': sha(library),
                    'compiler_version': subprocess.check_output(['c++', '--version'], text=True),
                    'model_runtime_environment_modified': False}, indent=2) + '\n')
                with (work / 'compile.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=25)
                environment = dict(os.environ)
                environment['CUDA_CACHE_PATH'] = str(work / 'cuda_cache')
                command = ['/usr/bin/python3', str(INPUT / NAMES[1]), '--library', str(output),
                    '--output', str(work / 'candidate'), '--plan', str(INPUT / 'plan.json')]
                (work / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
                with (work / 'capture.log').open('w') as log:
                    run = subprocess.run(command, cwd=work, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, timeout=120)
                    exit_code = run.returncode
            except Exception as error:
                failure = {'type': type(error).__name__, 'message': str(error)}
            receipt_path = work / 'candidate/receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
            complete = failure is None and exit_code == 0 and receipt is not None and receipt['status'] == 'complete_candidate'
            payload = io.BytesIO()
            retained = []
            with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(work.rglob('*')):
                    if not path.is_file() or path.relative_to(work).parts[0] in ('sdk', 'cuda_cache'):
                        continue
                    relative = path.relative_to(work).as_posix()
                    retained.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
                    archive.write(path, relative)
            return {'status': 'complete_candidate' if complete else 'capture_failed', 'identity': identity,
                'source_commit': source_commit, 'source_sha256': hashes, 'reservation_key': reservation_key,
                'exit_code': exit_code, 'failure': failure, 'files': retained, 'artifact_bundle': payload.getvalue(),
                'gpu_function_wall_s': time.monotonic() - started, 'duration_recorded': False,
                'persistent_storage': False, 'cloud_temporaries_retained': False}
    finally:
        shutil.rmtree(INPUT)


@APP.local_entrypoint()
def main(reservation_key: str, budget_tool: str):
    output = CAMPAIGN / 'attempt_1'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    helper = Path(budget_tool).resolve(strict=True)
    subprocess.run([sys.executable, str(helper), 'bind', '--key', reservation_key,
                    '--app-id', APP.app_id], check=True)
    (output / 'launch.json').write_text(json.dumps({'app_id': APP.app_id,
        'reservation_key': reservation_key, 'source_commit': commit,
        'input_sha256': [sha(path) for path in SOURCES],
        'runner_sha256': sha(HERE)}, indent=2) + '\n')
    result = capture.remote([sha(path) for path in SOURCES], commit, reservation_key)
    payload = result.pop('artifact_bundle')
    result.update(app_id=APP.app_id, client_wall_s=time.monotonic() - started, runner_sha256=sha(HERE),
        bundle_sha256=hashlib.sha256(payload).hexdigest(),
        resource_bound={'gpu': 'H200', 'cards': 1, 'total_seconds': 240, 'gpu_hours': 240 / 3600,
            'usd_reserved': 0.50, 'automatic_retries': 0, 'volume': False})
    (output / 'evidence.zip').write_bytes(payload)
    (output / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
    subprocess.run([sys.executable, str(helper), 'terminal', '--key', reservation_key], check=True)
    print(json.dumps({'status': result['status'], 'path': str(output), 'gpu_function_wall_s': result['gpu_function_wall_s']}), flush=True)
    if result['status'] != 'complete_candidate':
        raise RuntimeError('bounded module capture failed; retained returned evidence; no automatic retry')
