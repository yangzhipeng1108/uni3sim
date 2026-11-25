"""Independent notification primitive; requires authorized Modal execution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'mbarrier_handoff.cu'
TASK = Path('/tmp/midlayer-mbarrier-handoff-20261107')
IMAGE = modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11').add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App('midlayer-mbarrier-handoff-20261107')


def code_gate(sass):
    bodies = sass.split('Function : ')[1:]
    return (len(bodies) == 6 and all('SYNCS.PHASECHK' in body and 'SYNCS.ARRIVE' in body
        and 'HGMMA.' not in body and 'MUFU.' not in body and re.search(r'\bLDGSTS\.', body) is None
        for body in bodies))


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture():
    result = {}; started = time.monotonic()
    try:
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90a', '-Xptxas=-v', str(TASK / SOURCE.name), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
            'measurement': [str(TASK / 'probe')],
        }
        result['required_commands'] = list(commands)
        for name, command in commands.items():
            p = subprocess.run(command, capture_output=True, text=True, cwd=TASK, timeout=60)
            result[name] = {'command': command, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
            if p.returncode:
                break
            if name == 'build':
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', p.stderr)
                result['zero_spill_gate'] = len(spills) == 6 and all(a == b == '0' for a, b in spills)
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
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw.json').exists():
        raise ValueError('Capture already exists')
    result = capture.remote()
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    if not all(name in result and result[name]['exit_code'] == 0 for name in result['required_commands']):
        raise RuntimeError('Notification command failed; inspect retained raw')
    if not result['zero_spill_gate'] or not result['code_gate']:
        raise RuntimeError('Notification validity gate failed; inspect retained raw')
    print({'function_wall_s': result['gpu_function_wall_s'], 'validity_gates': True})
