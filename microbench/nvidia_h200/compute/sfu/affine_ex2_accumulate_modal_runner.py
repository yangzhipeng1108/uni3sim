"""Independent bounded FFMA/EX2/ADD register-bundle capture on one H200."""
import bz2
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

TASK = Path('/tmp/midlayer-affine-ex2-accumulate-20261108')
if (TASK / 'affine_ex2_accumulate.cu').is_file():
    SOURCE = TASK / 'affine_ex2_accumulate.cu'
    WARMUP = TASK / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.debian_slim()
else:
    SOURCE = Path(__file__).with_name('affine_ex2_accumulate.cu')
    WARMUP = SOURCE.parents[1] / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11')
    for path in (SOURCE, WARMUP):
        IMAGE = IMAGE.add_local_file(str(path), remote_path=str(TASK / path.name))
APP = modal.App('midlayer-affine-ex2-accumulate-20261108')


def code_gate(sass):
    bodies = sass.split('Function : ')[1:]
    if len(bodies) != 18:
        return False
    keys = set()
    for body in bodies:
        match = re.search(r'affine_ex2ILi([012])ELi([28])ELi(16|32|64)EE', body)
        if match is None or re.search(r'\b(?:LDL|STL)\b', body):
            return False
        mode, chains, width = map(int, match.groups())
        keys.add((mode, chains, width))
        if body.count('SR_CLOCKLO') != 2 or body.count('SR_GLOBALTIMERLO') != 2:
            return False
        expected = chains * width
        lines = body.splitlines()
        markers = [i for i, line in enumerate(lines) if 'SR_CLOCKLO' in line]
        timed = '\n'.join(lines[markers[0]:markers[1]])
        if mode == 2 and (timed.count('MUFU.EX2') != expected or timed.count('FFMA') != expected + chains):
            return False
        if mode != 2 and ('MUFU.EX2' in body or 'FFMA' in body):
            return False
        if mode in (1, 2) and timed.count('FADD') != expected:
            return False
    return keys == {(m, c, w) for m in (0, 1, 2) for c in (2, 8) for w in (16, 32, 64)}


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture():
    result = {}
    started = time.monotonic()
    try:
        environment = dict(os.environ)
        for name, folder in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg')):
            path = TASK / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90', '-Xptxas=-v', str(SOURCE), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
            'build_warmup': ['nvcc', '-O3', '-arch=sm_90', str(WARMUP), '-o', str(TASK / 'warmup')],
            'warmup': ['./warmup', '600'],
            'plain': ['./probe']}
        result['required_commands'] = list(commands)
        for name, command in commands.items():
            remaining = 110 - (time.monotonic() - started)
            if remaining <= 0:
                result['deadline_exceeded'] = True
                break
            p = subprocess.run(command, capture_output=True, text=True, cwd=TASK,
                env=environment, timeout=min(80, remaining))
            result[name] = dict(command=command, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
            if p.returncode != 0:
                break
            if name == 'identity' and not p.stdout.startswith('NVIDIA H200,'):
                result['identity_gate'] = False
                break
            if name == 'build':
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', p.stderr)
                result['zero_spill_gate'] = len(spills) == 18 and all(a == b == '0' for a, b in spills)
                if not result['zero_spill_gate']:
                    break
            if name == 'sass':
                result['code_gate'] = code_gate(p.stdout)
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
    out.mkdir(parents=True, exist_ok=False)
    result = capture.remote()
    for name in ('plain', 'sass'):
        if name in result:
            payload = result[name]['stdout'].encode()
            (out / (name + '.txt.bz2')).write_bytes(bz2.compress(payload))
            result[name]['stdout_sha256'] = hashlib.sha256(payload).hexdigest()
            result[name]['stdout_bytes'] = len(payload)
            result[name]['stdout'] = 'Retained in ' + name + '.txt.bz2'
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'gpu_function_wall_s': result['gpu_function_wall_s'],
        'complete': all(k in result and result[k]['exit_code'] == 0 for k in result['required_commands'])}))
