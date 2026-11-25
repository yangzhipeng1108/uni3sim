"""Authorized independent H200 memory-burst capacity capture; no model inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import modal

SOURCE = Path(__file__).with_name('cta_memory_burst.cu')
TASK = Path('/tmp/midlayer-h200-cta-memory-burst-20261107')
IMAGE = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
         .apt_install('cuda-nsight-compute-12-9')
         .add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)))
APP = modal.App('midlayer-h200-moe-memory-burst-20261107')

def run(command, timeout):
    started = time.monotonic()
    result = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=timeout)
    return {'command': command, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr, 'wall_s': time.monotonic() - started}

@APP.function(gpu='H200', image=IMAGE, timeout=780, retries=0, max_containers=1, single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {'boundary': 'independent_two_vector16_reads_one_vector16_write_no_activation_or_matrix',
              'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        commands = (
            ('gpu', ['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20),
            ('nvcc', ['nvcc', '--version'], 20),
            ('build', ['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-Xptxas=-v', SOURCE.name, '-o', 'probe'], 120),
            ('resources', ['cuobjdump', '--dump-resource-usage', 'probe'], 30),
            ('sass', ['cuobjdump', '--dump-sass', 'probe'], 45),
            ('trial_0', [str(TASK / 'probe')], 90),
            ('trial_1', [str(TASK / 'probe')], 90),
            ('counter_burst1', ['ncu', '--csv', '--page', 'raw', '--cache-control', 'none', '--clock-control', 'none', '--kernel-name', 'regex:memory_burst', '--metrics', 'lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum', str(TASK / 'probe'), '96', '1'], 120),
            ('counter_burst8', ['ncu', '--csv', '--page', 'raw', '--cache-control', 'none', '--clock-control', 'none', '--kernel-name', 'regex:memory_burst', '--metrics', 'lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum', str(TASK / 'probe'), '96', '8'], 120),
            ('gpu_after', ['nvidia-smi', '--query-gpu=name,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20))
        for name, command, timeout in commands:
            result[name] = run(command, timeout)
            if result[name]['exit_code']:
                result['status'] = name + '_failed'
                return result
        result['binary_sha256'] = hashlib.sha256((TASK / 'probe').read_bytes()).hexdigest()
        result['status'] = 'complete'
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    campaign = root / 'profiler_output/microbench/nvidia_h200/cta_memory_burst_20261107'
    target = Path(output_dir).resolve()
    if target.parent != campaign.resolve() or target.name != 'attempt_1' or target.exists():
        raise ValueError('output must be the new registered attempt_1')
    started = time.monotonic()
    result = capture.remote()
    result['app_id'] = APP.app_id
    result['client_wall_s'] = time.monotonic() - started
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True)
    (target / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'gpu_function_wall_s': result['gpu_function_wall_s'], 'app_id': result['app_id'], 'raw': str(target / 'raw.json')}))
