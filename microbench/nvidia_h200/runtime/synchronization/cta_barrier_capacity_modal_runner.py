"""Capture pure CTA-barrier cycles, excluding every production operation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import modal

HERE = Path(__file__).resolve().parent
TASK = Path('/tmp/midlayer-h200-cta-barrier-capacity-20261107')
SOURCE = HERE / 'cta_barrier_capacity.cu' if HERE.name == 'synchronization' else TASK / 'cta_barrier_capacity.cu'
IMAGE = modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12').add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App('midlayer-h200-cta-barrier-capacity-20261107')

def run(command, timeout):
    start = time.monotonic()
    result = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=timeout)
    return {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr, 'wall_s': time.monotonic() - start}

@APP.function(gpu='H200', image=IMAGE, timeout=240, retries=0, max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    result = {'boundary': 'pure_barrier_no_global_traffic_no_matrix', 'records': []}
    try:
        for name, command, limit in (
            ('gpu', ['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20),
            ('nvcc', ['nvcc', '--version'], 20)):
            result[name] = run(command, limit)
            if result[name]['exit_code']:
                result['status'] = name + '_failed'
                return result
        for shared in (147456, 196608):
            build = run(['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-Xptxas=-v',
                         '-DTOTAL_SHARED_BYTES=' + str(shared), SOURCE.name, '-o', 'probe'], 60)
            result['records'].append({'shared': shared, 'build': build})
            if build['exit_code']:
                result['status'] = 'build_failed'
                return result
            result['records'][-1]['binary_sha256'] = hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
            result['records'][-1]['sass'] = run(['cuobjdump', '--dump-sass', 'probe'], 30)
            result['records'][-1]['resources'] = run(['cuobjdump', '--dump-resource-usage', 'probe'], 20)
            for trial in range(3):
                for repeats in (256, 1024, 4096):
                    for mode in (1, 2):
                        measured = run([str(TASK/'probe'), str(mode), '132', str(repeats), '8'], 10)
                        result['records'].append({'shared': shared, 'trial': trial,
                            'repeats': repeats, 'mode': mode, 'measurement': measured})
                        if measured['exit_code']:
                            result['status'] = 'measurement_failed'
                            return result
        result['status'] = 'complete'
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir: str):
    root = HERE.parents[3]
    target = Path(output_dir).resolve()
    campaign = root/'profiler_output/microbench/nvidia_h200/cta_barrier_capacity_20261107'
    if target.parent != campaign.resolve() or target.name != 'attempt_1' or target.exists():
        raise ValueError('output must be a new registered attempt_1')
    start = time.monotonic()
    result = capture.remote()
    result['app_id'] = APP.app_id
    result['client_wall_s'] = time.monotonic() - start
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True)
    (target/'raw.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'app_id': result['app_id'], 'wall_s': result['gpu_function_wall_s']}))
