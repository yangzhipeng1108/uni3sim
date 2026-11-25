"""Independent signed-input and continuous-conditioning BF16 clock control."""
import ast
import base64
import gzip
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

TASK = Path('/tmp/midlayer-tensor-signed-activity-clock-20261108')
if (TASK / 'tensor_signed_activity_clock.cu').is_file():
    SOURCE = TASK / 'tensor_signed_activity_clock.cu'
    REFERENCE = TASK / 'fa3_wgmma_service.cu'
    GENERATOR = TASK / 'fa3_wgmma_service_gen.py'
    WARMUP = TASK / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.debian_slim()
else:
    HERE = Path(__file__).resolve().parent
    SOURCE = HERE / 'tensor_signed_activity_clock.cu'
    REFERENCE = HERE.parents[1] / 'operators/flash_attn/fa3_wgmma_service.cu'
    GENERATOR = HERE.parents[1] / 'operators/flash_attn/fa3_wgmma_service_gen.py'
    WARMUP = HERE.parent / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11')
    for path in (SOURCE, REFERENCE, GENERATOR, WARMUP):
        IMAGE = IMAGE.add_local_file(path, remote_path=str(TASK / path.name))
APP = modal.App('midlayer-tensor-signed-activity-clock-20261108')


def helper(reference, generator):
    marker = '__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker) != 1:
        raise ValueError('Independent helper preamble changed')
    functions = [n for n in ast.parse(generator).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'wgmma_function']
    if len(functions) != 1:
        raise ValueError('Independent PTX generator changed')
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<primitive-generator>', 'exec'), namespace)
    return (reference[:reference.index(marker)]
            + namespace['wgmma_function']('ss_n128_kmajor', 128, False, 0)
            + namespace['wgmma_function']('rs_n128_mnmajor', 128, True, 1))


def code_gate(sass):
    bodies = [body for body in sass.split('Function : ')[1:]
              if body.startswith('_Z21signed_activity_clock')]
    if len(bodies) != 2 or 'WG.DP' in sass or re.search(r'\b(?:LDL|STL)\b', sass):
        return False
    seen = set()
    for body in bodies:
        match = re.match(r'_Z21signed_activity_clockILb([01])EE', body)
        if match is None:
            return False
        mixed = int(match[1])
        seen.add(mixed)
        matrix = re.findall(r'HGMMA\.64x128x16\.F32\.BF16 (R\d+),', body)
        if len(matrix) != 16 or len(set(matrix)) != 1:
            return False
        if body.count('MUFU.EX2') != 64 * mixed:
            return False
        waits = re.findall(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x([0-9a-f]+)', body)
        if waits != ['0']:
            return False
    return seen == {0, 1}


def run(command, environment, timeout):
    process = subprocess.Popen(command, cwd=TASK, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    return {'command': command, 'exit_code': process.returncode, 'timed_out': timed_out, 'stdout': out, 'stderr': err}


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        environment = dict(os.environ)
        for key, folder in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg')):
            path = TASK / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment['CUDA_VISIBLE_DEVICES'] = '0'
        header = helper(REFERENCE.read_text(), GENERATOR.read_text())
        (TASK / 'signed_activity_wgmma_helper.cuh').write_text(header)
        result['helper_source'] = header
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90a', '-Xptxas=-v', str(TASK / SOURCE.name), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
            'build_warmup': ['nvcc', '-O3', '-arch=sm_90', str(TASK / WARMUP.name), '-o', str(TASK / 'warmup')],
            'warmup': ['./warmup', '600'],
            'before': ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader'],
            'plain': ['./probe'],
            'after': ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader'],
        }
        result['required_commands'] = list(commands)
        for name, command in commands.items():
            remaining = 110 - (time.monotonic() - started)
            if remaining <= 0:
                result['deadline_exceeded'] = True
                break
            item = run(command, environment, min(95, remaining))
            if name == 'plain':
                item['stdout_encoding'] = 'gzip_base64'
                item['stdout_gzip_base64'] = base64.b64encode(gzip.compress(item['stdout'].encode())).decode()
                item['stdout'] = ''
            result[name] = item
            if item['exit_code'] != 0:
                break
            if name == 'identity' and (len(item['stdout'].strip().splitlines()) != 1 or not item['stdout'].startswith('NVIDIA H200,')):
                result['identity_gate'] = False
                break
            if name == 'build':
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', item['stderr'])
                result['zero_spill_gate'] = len(spills) == 3 and all(a == b == '0' for a, b in spills)
                if not result['zero_spill_gate']:
                    break
            if name == 'sass':
                result['code_gate'] = code_gate(item['stdout'])
                if not result['code_gate']:
                    break
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str):
    out = Path(output_dir)
    if out.exists():
        raise ValueError('Tensor data-activity clock capture requires a new attempt directory')
    result = capture.remote()
    out.mkdir(parents=True)
    files = {}
    for path in (SOURCE, REFERENCE, GENERATOR, WARMUP, Path(__file__)):
        content = path.read_bytes()
        (out / path.name).write_bytes(content)
        files[path.name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
    result['source_files'] = files
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    if not all(k in result and result[k]['exit_code'] == 0 for k in result['required_commands']):
        raise RuntimeError('Tensor data-activity clock capture incomplete; raw retained')
    print({'status': 'raw_independent_input_activity_clock', 'wall_s': result['gpu_function_wall_s']})
