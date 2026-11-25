"""One bounded H200 native Host-API diagnostic; original model/CLI retained."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path('/root')
ROOT = Path('/tmp') if REMOTE else HERE.parents[2]
CAMPAIGN = ROOT / 'profiler_output/serving/nvidia_h200/serving_accuracy_native_api_boundary_20261108'
LOCAL_SOURCES = (ROOT / 'dataset_pipeline/profiler/nvidia_h200_serving_boundary_probe.py',
    ROOT / 'dataset_pipeline/profiler/nvidia_h200_serving_api_boundary_hooks.py',
    ROOT / 'dataset_pipeline/profiler/nvidia_h200_serving_api_boundary.cc',
    CAMPAIGN / 'requests_geometry.json', CAMPAIGN / 'sdk_input_manifest.json')
REMOTE_SOURCES = tuple(Path('/tmp') / name for name in (
    'midlayer_api_probe.py', 'midlayer_api_hooks.py', 'midlayer_api_boundary.cc', 'midlayer_api_geometry.json',
    'midlayer_api_sdk_manifest.json'))
REMOTE_SDK = Path('/tmp/midlayer_api_sdk_wheels')
IMAGE = modal.Image.from_registry('vllm/vllm-openai:v0.30.0', add_python='3.12').entrypoint([])
for local, remote in zip(LOCAL_SOURCES, REMOTE_SOURCES, strict=True):
    IMAGE = IMAGE.add_local_file(remote if REMOTE else local, remote_path=str(remote))
IMAGE = IMAGE.add_local_dir(REMOTE_SDK if REMOTE else CAMPAIGN / 'sdk_input', remote_path=str(REMOTE_SDK))
APP = modal.App('midlayer-h200-serving-native-api-20261108')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sdk(work):
    """Compile with pinned isolated public SDK inputs; never install into Torch."""
    entries = json.loads(REMOTE_SOURCES[4].read_text())
    if {(item['package'], item['version']) for item in entries} != {
            ('nvidia-cuda-cupti', '13.0.85'), ('nvidia-cuda-runtime', '13.0.96'), ('nvidia-cuda-crt', '13.0.88')}:
        raise ValueError('unexpected isolated SDK package set')
    sdk = work / 'sdk'
    sdk.mkdir()
    for item in entries:
        name = item['filename']
        if Path(name).name != name:
            raise ValueError('invalid SDK wheel filename')
        path = REMOTE_SDK / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('SDK wheel differs from its public content address')
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (sdk / member.filename).resolve()
                if not target.is_relative_to(sdk.resolve()):
                    raise ValueError('SDK archive member leaves the task directory')
            archive.extractall(sdk)
    root = sdk / 'nvidia/cu13'
    include, library = root / 'include', root / 'lib/libcupti.so.13'
    if not all((include / name).is_file() for name in ('cupti.h', 'cuda.h', 'crt/host_defines.h')) or not library.is_file():
        raise ValueError('isolated SDK is incomplete')
    compiler = shutil.which('c++')
    if compiler is None:
        raise RuntimeError('native recorder needs the existing C++ compiler')
    target = work / 'api_boundary.so'
    command = [compiler, '-std=c++17', '-O2', '-shared', '-fPIC', '-pthread',
        '-Wall', '-Wextra', '-Werror', '-I' + str(include), str(REMOTE_SOURCES[2]),
        str(library), '-Wl,-rpath,' + str(library.parent), '-o', str(target)]
    (work / 'compiler.json').write_text(json.dumps({'command': command,
        'compiler_version': subprocess.check_output([compiler, '--version'], text=True),
        'sdk_header_sha256': {name: sha(include / name) for name in ('cupti.h', 'cuda.h', 'crt/host_defines.h')},
        'intended_cupti_library': str(library), 'intended_cupti_library_sha256': sha(library),
        'model_runtime_environment_modified': False}, indent=2) + '\n')
    with (work / 'compile.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    return target


def retained_bundle(work):
    bundle, retained = io.BytesIO(), []
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(work.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(work)
            if any(part in ('hf', 'hf_home', 'compile_cache', 'hooks', 'sdk') for part in relative.parts):
                continue
            if path.suffix in ('.log', '.json', '.bin', '.so'):
                archive.write(path, relative.as_posix())
                retained.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return bundle.getvalue(), retained


@APP.function(cpu=1, memory=4096, image=IMAGE, timeout=120, retries=0,
    max_containers=1, single_use_containers=True)
def cpu_sdk(hashes, commit):
    started = time.monotonic()
    if [sha(path) for path in REMOTE_SOURCES] != hashes:
        raise ValueError('CPU SDK source/input binding changed')
    with tempfile.TemporaryDirectory(prefix='midlayer_native_sdk_', dir='/tmp') as temporary:
        work = Path(temporary)
        failure = None
        try:
            target = prepare_sdk(work)
            import ctypes
            library = ctypes.CDLL(str(target))
            library.midlayer_boundary_init.argtypes = (ctypes.c_size_t, ctypes.c_int)
            library.midlayer_boundary_begin.argtypes = (ctypes.c_int64,) * 5
            library.midlayer_boundary_end.argtypes = ()
            library.midlayer_boundary_dump.argtypes = (ctypes.c_char_p,)
            # No subscription or GPU query: exercise the recorder's shared ABI.
            if library.midlayer_boundary_init(16, 0) != 0:
                raise RuntimeError('CPU recorder initialization failed')
            for phase in (0, 1):
                if library.midlayer_boundary_begin(phase, 0, 1, 1, 1) != 0 or library.midlayer_boundary_end() != 0:
                    raise RuntimeError('CPU recorder marker pair failed')
            if library.midlayer_boundary_dump(os.fsencode(work)) != 0:
                raise RuntimeError('CPU recorder dump failed')
            metadata = json.loads((work / 'native_metadata.json').read_text())
            if metadata['rows'] != 2 or metadata['error_code'] != 0:
                raise RuntimeError('CPU recorder conservation failed')
        except Exception as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
        payload, files = retained_bundle(work)
        return {'status': 'sdk_and_cpu_recorder_ready' if failure is None else 'sdk_preflight_failed',
            'failure': failure, 'files': files, 'artifact_bundle': payload, 'source_commit': commit,
            'source_sha256': hashes, 'function_wall_s': time.monotonic() - started,
            'gpu_cards': 0, 'gpu_hours': 0, 'gpu_workload_run': False,
            'persistent_storage': False, 'cloud_temporaries_retained': False}


@APP.function(gpu='H200', cpu=8, memory=32768, image=IMAGE, timeout=600,
    retries=0, max_containers=1, single_use_containers=True)
def capture(hashes, commit):
    started = time.monotonic()
    try:
        if [sha(path) for path in REMOTE_SOURCES] != hashes:
            raise ValueError('uploaded frozen source/input differs')
        identity = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
            check=True, capture_output=True, text=True).stdout
        if 'H200' not in identity:
            raise ValueError('authorized H200 required')
        with tempfile.TemporaryDirectory(prefix='midlayer_native_api_', dir='/tmp') as temporary:
            work = Path(temporary)
            failure, process, exit_code, timed_out, command = None, None, 1, False, []
            try:
                target = prepare_sdk(work)
                environment = dict(os.environ)
                environment['HF_HOME'] = str(work / 'hf_home')
                environment['VLLM_NO_USAGE_STATS'] = '1'
                environment['MIDLAYER_API_BOUNDARY_LIBRARY'] = str(target)
                command = ['/usr/bin/python3', str(REMOTE_SOURCES[0]), '--hooks', str(REMOTE_SOURCES[1]),
                    '--geometry', str(REMOTE_SOURCES[3]), '--output', str(work / 'candidate')]
                with (work / 'collector.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        exit_code = process.wait(timeout=540)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        raise
            except Exception as error:
                failure = {'type': type(error).__name__, 'message': str(error)}
            finally:
                if process is not None and process.poll() is None:
                    import psutil
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    _, alive = psutil.wait_procs(children, timeout=5)
                    for child in alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            payload, retained = retained_bundle(work)
            receipt_path = work / 'candidate/receipt.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
            complete = (failure is None and exit_code == 0 and receipt is not None and len(receipt['runs']) == 2
                and all(run['completed_requests'] == 128 for run in receipt['runs']))
            return {'status': 'complete_candidate' if complete else 'capture_failed',
                'identity': identity, 'source_commit': commit, 'source_sha256': hashes,
                'command': command, 'exit_code': exit_code, 'failure': failure, 'timed_out': timed_out,
                'gpu_function_wall_s': time.monotonic() - started, 'files': retained,
                'artifact_bundle': payload, 'persistent_storage': False,
                'cloud_temporaries_and_weights_retained': False}
    finally:
        for path in REMOTE_SOURCES:
            path.unlink()


@APP.local_entrypoint()
def main(attempt: str = 'initial', sdk_only: bool = False):
    if attempt not in ('initial', 'attempt_2'):
        raise ValueError('only the initial attempt and one necessary GPU retry are configured')
    output = CAMPAIGN / 'cpu_sdk' if sdk_only else CAMPAIGN if attempt == 'initial' else CAMPAIGN / attempt
    if (output / 'execution.json').exists():
        raise FileExistsError('single native API campaign attempt already exists')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = (cpu_sdk if sdk_only else capture).remote([sha(path) for path in LOCAL_SOURCES], commit)
    payload = result.pop('artifact_bundle')
    result.update(client_wall_s=time.monotonic() - started,
        bundle_sha256=hashlib.sha256(payload).hexdigest(), runner_sha256=sha(HERE),
        billing=('shared80; key1107-serving-native-api-sdk-20261108-01; CPU-only/GPUh0/US$0.05 reservation' if sdk_only
            else 'shared80; key1107-serving-native-api-20261108-02; one necessary retry H200x1<=600s/0.166667GPUh/US$1.20 reservation; no automatic retry/Volume; invoice pending'))
    (output / 'evidence.zip').write_bytes(payload)
    (output / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'path': str(output), 'new_gpu_call': not sdk_only}), flush=True)
    expected = 'sdk_and_cpu_recorder_ready' if sdk_only else 'complete_candidate'
    if result['status'] != expected:
        raise RuntimeError('native API diagnostic failed; all returned receipts and raw have been retained')
