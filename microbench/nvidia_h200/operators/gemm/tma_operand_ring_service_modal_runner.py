"""Capture a bounded, operator-free H200 TMA ring sweep."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import modal

HERE = Path(__file__).resolve().parent
SOURCES = tuple(HERE / name for name in (
    'tma_operand_ring_service.cu', 'fp8_tma_operand_stream.cu'))
IMAGE = modal.Image.from_registry(
    'nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
for source in SOURCES:
    IMAGE = IMAGE.add_local_file(source, remote_path='/tmp/gemm_ring/' + source.name)
APP = modal.App('midlayer-h200-gemm-ring-service-20261107')

@APP.function(gpu='H200', image=IMAGE, timeout=300, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
        '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout
    if not gpu.startswith('NVIDIA H200,'):
        raise ValueError('one H200 required')
    with tempfile.TemporaryDirectory(prefix='midlayer-gemm-ring-') as temporary:
        binary = Path(temporary) / 'probe'
        command = ['nvcc', '-O3', '-std=c++17', '-arch=sm_90a', '-Xptxas=-v',
            '/tmp/gemm_ring/tma_operand_ring_service.cu', '-o', str(binary)]
        build = subprocess.run(command, capture_output=True, text=True, timeout=90)
        result = {'gpu': gpu, 'compile_command': command,
            'build_exit_code': build.returncode, 'build_stdout': build.stdout,
            'build_stderr': build.stderr}
        if build.returncode == 0:
            result['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            before = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,pstate',
                '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=180)
            after = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,pstate',
                '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout
            result.update({'run_exit_code': run.returncode, 'stdout': run.stdout,
                'stderr': run.stderr, 'clock_before': before, 'clock_after': after})
        result['function_wall_s'] = time.monotonic() - started
    return result

@APP.local_entrypoint()
def main(output_dir: str):
    target = Path(output_dir).resolve()
    expected = HERE.parents[3] / 'profiler_output/microbench/nvidia_h200/gemm_ring_service_20261107'
    if target not in (expected.resolve(), expected.with_name(expected.name + '_graph').resolve(), expected.with_name(expected.name + '_startup').resolve(), expected.with_name(expected.name + '_cluster').resolve()) or target.exists():
        raise ValueError('output must be the new registered campaign')
    started = time.monotonic()
    result = capture.remote()
    result['client_wall_s'] = time.monotonic() - started
    result['sources'] = {str(source.relative_to(HERE.parents[3])):
        hashlib.sha256(source.read_bytes()).hexdigest() for source in SOURCES}
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['production_duration_consumed'] = False
    target.mkdir(parents=True)
    (target / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'raw': str(target / 'raw.json'),
        'function_wall_s': result['function_wall_s'], 'build_exit_code': result['build_exit_code']}))
