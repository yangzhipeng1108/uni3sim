"""Independent finite TMA row-pitch diagnostic;no Attention or parameter admission."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'tma_queue_service.cu'
TASK = Path('/tmp/midlayer-tma-row-pitch-20261108')
IMAGE = modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
IMAGE = IMAGE.add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App('midlayer-tma-row-pitch-20261108')


def code_gate(sass):
    # Integer division in generic grid/validation control may lower through
    # MUFU.RCP. The probe has no Tensor or independent EX2 pressure stream.
    if 'HGMMA.' in sass or 'WG.DP' in sass or 'MUFU.EX2' in sass:
        return False
    if not re.search(r'Function : .*queue_service', sass):
        return False
    return len(re.findall(r'\bUTMALDG\.4D\b', sass)) >= 2


def run_command(command, environment, timeout):
    process = subprocess.Popen(command, cwd=TASK, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    return dict(command=command, exit_code=process.returncode, timed_out=timed_out,
        stdout=out, stderr=err)


@APP.function(gpu='H200', image=IMAGE, timeout=180, retries=0, single_use_containers=True)
def capture():
    result = {'captures': [], 'production_duration_consumed': False}
    started = time.monotonic()
    try:
        environment = dict(os.environ)
        for name in ('NVCC_APPEND_FLAGS', 'NVCC_PREPEND_FLAGS'):
            if name in environment:
                del environment[name]
        for name, folder in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg')):
            path = TASK / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        query = ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,clocks.mem', '--format=csv,noheader']
        identity = subprocess.run(query, capture_output=True, text=True, check=True, timeout=15)
        result['identity'] = identity.stdout
        if len(identity.stdout.strip().splitlines()) != 1 or not identity.stdout.startswith('NVIDIA H200,'):
            raise ValueError('A single H200 is required')
        for row_gap in (1, 4, 8):
            threads = 256
            item = {'threads': threads, 'row_gap': row_gap}
            result['captures'].append(item)
            binary = TASK / f'probe_gap_{row_gap}'
            commands = {
                'build': ['nvcc', '-O3', '-std=c++17', '-arch=sm_90a', '-Xptxas=-v',
                    '-DTMA_BALANCED_PAIRS=1', f'-DTMA_CTA_THREADS={threads}',
                    '-DTMA_ACTIVITY_PRELUDE=2', f'-DTMA_BALANCED_ROW_GAP={row_gap}',
                    '-DTMA_STREAM_PACKETS=64', '-DTMA_BALANCED_STRIDE_SCREEN=1',
                    str(TASK / SOURCE.name), '-o', str(binary)],
                'sass': ['cuobjdump', '--dump-sass', str(binary)],
                'measurement': [str(binary)],
            }
            item['required_commands'] = list(commands)
            for name, command in commands.items():
                remaining = 165 - (time.monotonic() - started)
                if remaining <= 0:
                    item['deadline_exceeded'] = True
                    return result
                maximum = 95 if name == 'measurement' else 30
                process = run_command(command, environment, min(maximum, remaining))
                item[name] = process
                if process['exit_code'] or process['timed_out']:
                    return result
                if name == 'build':
                    spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', process['stderr'])
                    item['zero_spill_gate'] = len(spills) == 2 and all(a == b == '0' for a, b in spills)
                    if not item['zero_spill_gate']:
                        return result
                if name == 'sass':
                    item['code_gate'] = code_gate(process['stdout'])
                    if not item['code_gate']:
                        return result
        result['final_identity'] = subprocess.run(query, capture_output=True,
            text=True, check=True, timeout=15).stdout
        return result
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        result['failure'] = dict(kind=type(error).__name__, message=str(error))
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)
        result['cloud_temporary_directory_removed'] = not TASK.exists()


@APP.local_entrypoint()
def main(output_dir: str):
    out = Path(output_dir)
    if out.exists():
        raise ValueError('A new output directory is required')
    result = capture.remote()
    out.mkdir(parents=True)
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2)+'\n')
    if 'failure' in result:
        raise RuntimeError('Capture failed;raw retained')
    if len(result['captures']) != 3:
        raise RuntimeError('Incomplete capture;raw retained')
    for item in result['captures']:
        if not all(name in item and item[name]['exit_code'] == 0 for name in item['required_commands']):
            raise RuntimeError('Capture command failed;raw retained')
        if not item['zero_spill_gate'] or not item['code_gate']:
            raise RuntimeError('Compiler gate failed;no price admitted')
    print(dict(function_wall_s=result['gpu_function_wall_s'], captures=3))
