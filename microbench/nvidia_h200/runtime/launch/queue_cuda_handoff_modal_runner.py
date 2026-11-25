"""One budgeted1xH200 native-submission/GIL handoff mechanism capture."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path('/root')
SOURCE = Path('/tmp/midlayer_queue_cuda.py') if REMOTE else HERE.with_name('queue_cuda_handoff_runner.py')
ROOT = Path('/tmp') if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / 'profiler_output/microbench/nvidia_h200/serving_accuracy_queue_gil_20261108'
KEY = '1107-serving-queue-gil-20261108-01'
IMAGE = (modal.Image.from_registry('vllm/vllm-openai:v0.30.0', add_python='3.12')
    .entrypoint([]).add_local_file(SOURCE, remote_path='/tmp/midlayer_queue_cuda.py'))
APP = modal.App('midlayer-h200-serving-queue-gil-20261108')


@APP.function(gpu='H200', image=IMAGE, cpu=2, memory=4096, timeout=480, retries=0,
    max_containers=1, single_use_containers=True)
def measure(expected_sha256: str) -> dict:
    started = time.monotonic()
    source = Path('/tmp/midlayer_queue_cuda.py')
    try:
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('uploaded mechanism runner differs')
        identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
            '--format=csv,noheader'], text=True)
        if 'H200' not in identity:
            raise ValueError('authorized H200 required')
        with tempfile.TemporaryDirectory(prefix='midlayer_queue_gil_', dir='/tmp') as directory:
            work = Path(directory)
            output = work / 'raw.json'
            environment = dict(os.environ)
            for name, subdir in (('CUDA_CACHE_PATH', 'cuda-cache'), ('TRITON_CACHE_DIR', 'triton'),
                    ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'temporary')):
                path = work / subdir
                path.mkdir()
                environment[name] = str(path)
            command = ['/usr/bin/python3', str(source), '--device', 'cuda', '--output', str(output)]
            result = subprocess.run(command, cwd=work, env=environment, capture_output=True,
                text=True, check=False, timeout=420)
            return {'status': 'complete' if result.returncode == 0 and output.is_file() else 'probe_failed',
                'gpu_identity': identity, 'command': command, 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr,
                'source_sha256': expected_sha256, 'raw': json.loads(output.read_text()) if output.is_file() else None,
                'gpu_function_wall_s': time.monotonic() - started}
    finally:
        if source.exists():
            source.unlink()


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / 'raw.json').exists():
        raise FileExistsError(CAMPAIGN / 'raw.json')
    CAMPAIGN.mkdir(exist_ok=True)
    started = time.monotonic()
    result = measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result['client_wall_s'] = time.monotonic() - started
    result['runner_sha256'] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result['budget'] = {'shared_cap_usd': 80, 'reservation_key': KEY, 'reserved_usd': '1.10',
        'gpu': 'H200', 'cards': 1, 'seconds_including_startup_warmup_cleanup': 600,
        'planned_gpu_hours_bound': 600 / 3600, 'automatic_retries': 0, 'persistent_volume': False,
        'actual_spend': 'unknown until official billing; hold is not actual cost'}
    path = CAMPAIGN / 'raw.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'status': result['status'], 'gpu_function_wall_s': result['gpu_function_wall_s']}))
