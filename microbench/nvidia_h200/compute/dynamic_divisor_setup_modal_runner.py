"""One bounded H200 register-only reciprocal-integer setup capture."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

SOURCE = Path(__file__).with_name('dynamic_divisor_setup.cu')
TASK = Path('/tmp/midlayer-dynamic-divisor-20261107')
IMAGE = modal.Image.from_registry(
    'nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11'
).add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App('midlayer-dynamic-divisor-20261107')


def code_gate(sass):
    bodies = sass.split('Function : ')[1:]
    if len(bodies) != 4:
        return False
    for body in bodies:
        match = re.search(r'setupILi([01])ELi([18])EE', body)
        if match is None or 'LDL' in body or 'STL' in body:
            return False
        division = 'MUFU.RCP ' in body and 'CALL.REL.NOINC' in body
        if division != (match[1] == '1'):
            return False
        if body.count('CS2R') < 4:
            return False
    return True


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90', '-Xptxas=-v', str(TASK / SOURCE.name), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
            'plain': [str(TASK / 'probe')],
        }
        result['required_commands'] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=TASK)
            result[name] = {'command': command, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
            if p.returncode:
                break
            if name == 'identity' and not p.stdout.startswith('NVIDIA H200,'):
                raise ValueError('Independent setup requires the authorized H200')
            if name == 'build':
                entries = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', p.stderr)
                result['zero_spill_gate'] = len(entries) == 4 and all(a == b == '0' for a, b in entries)
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
    if out.exists():
        raise ValueError('Capture output must be a new attempt directory')
    result = capture.remote()
    out.mkdir(parents=True)
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(SOURCE, out / 'executed_source.cu')
    shutil.copy2(Path(__file__), out / 'executed_runner.py')
    if not all(name in result and result[name]['exit_code'] == 0 for name in result['required_commands']):
        raise RuntimeError('Setup capture incomplete;raw retained')
    print({'wall_s': result['gpu_function_wall_s'], 'status': 'raw_independent_mechanism'})
