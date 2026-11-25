"""Separate independent L2 read/write service from profiler replay traffic."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone
import modal

HERE = Path(__file__).resolve().parent
DIRECTIONAL = HERE / 'directional_bandwidth.cu'
BURST = HERE.parent / 'lsu/cta_memory_burst.cu'
TASK = Path('/tmp/midlayer-h200-l2-directional-replay-20261107')
IMAGE = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
         .apt_install('cuda-nsight-compute-12-9')
         .add_local_file(DIRECTIONAL, remote_path=str(TASK / DIRECTIONAL.name))
         .add_local_file(BURST, remote_path=str(TASK / BURST.name)))
APP = modal.App('midlayer-h200-l2-directional-replay-20261107')

def run(command, timeout):
    start = time.monotonic()
    result = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=timeout)
    return {'command': command, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr, 'wall_s': time.monotonic() - start}

@APP.function(gpu='H200', image=IMAGE, timeout=780, retries=0, max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    result = {'boundary': 'independent_directional_bandwidth_and_profiler_replay_no_production_operator',
              'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        commands = [
            ('gpu', ['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20),
            ('nvcc', ['nvcc', '--version'], 20),
            ('build_directional', ['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-Xptxas=-v',
                '-DDIRECTIONAL_BUFFER_BYTES=15728640', '-DDIRECTIONAL_ITERATIONS=64', '-DDIRECTIONAL_TRIALS=7',
                DIRECTIONAL.name, '-o', 'directional'], 120),
            ('build_burst', ['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-Xptxas=-v', BURST.name, '-o', 'burst'], 120)]
        metrics = 'lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum'
        for trial in (0, 1):
            for mode in ('read', 'write', 'read50_write50'):
                commands.append((f'native_{mode}_{trial}', [str(TASK / 'directional'), '--single', mode, '1056'], 60))
        for replay in ('kernel', 'application'):
            for mode in ('read', 'write', 'read50_write50'):
                commands.append((f'counter_{mode}_{replay}', ['ncu', '--csv', '--page', 'raw',
                    '--replay-mode', replay, '--cache-control', 'none', '--clock-control', 'none',
                    '--kernel-name', f'regex:^a800_dram_{mode}$', '--launch-skip', '1', '--launch-count', '1',
                    '--metrics', metrics, str(TASK / 'directional'), '--single', mode, '1056'], 90))
            commands.append((f'counter_burst_{replay}', ['ncu', '--csv', '--page', 'raw',
                '--replay-mode', replay, '--cache-control', 'none', '--clock-control', 'none',
                '--kernel-name', 'regex:memory_burst', '--metrics', metrics,
                str(TASK / 'burst'), '96', '1'], 90))
        commands.append(('gpu_after', ['nvidia-smi', '--query-gpu=name,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20))
        for name, command, timeout in commands:
            result[name] = run(command, timeout)
            if result[name]['exit_code']:
                result['status'] = name + '_failed'
                return result
        result['binary_sha256'] = {name: hashlib.sha256((TASK / name).read_bytes()).hexdigest()
                                   for name in ('directional', 'burst')}
        result['status'] = 'complete'
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - start
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    campaign = root / 'profiler_output/microbench/nvidia_h200/l2_directional_replay_20261107'
    target = Path(output_dir).resolve()
    if target.parent != campaign.resolve() or target.name != 'attempt_1' or target.exists():
        raise ValueError('output must be the new registered attempt_1')
    start = time.monotonic()
    result = capture.remote()
    result['app_id'] = APP.app_id
    result['client_wall_s'] = time.monotonic() - start
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (DIRECTIONAL, BURST)}
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True)
    (target / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'app_id': result['app_id'],
                      'gpu_function_wall_s': result['gpu_function_wall_s'], 'output': str(target)}))
