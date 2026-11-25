"""Independent conditional forward-guard capture; no operator inputs or timings.

First generate the include locally with:
python microbench/nvidia_h200/compute/uniform_branch_packets_gen.py
"""
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

HERE = Path(__file__).resolve().parent
FILES = ('uniform_branch_packets.cu', 'uniform_branch_packets.cuh')
TASK = Path('/tmp/midlayer-uniform-branch-packets-20261108')
IMAGE = modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
for filename in FILES:
    IMAGE = IMAGE.add_local_file(HERE / filename, remote_path=str(TASK / filename))
APP = modal.App('midlayer-uniform-branch-packets-20261108')


def code_gate(sass):
    records = []
    expected = {(g, s, True) for g in (16, 32, 48, 64) for s in (8, 12, 16)}
    expected |= {(g, 0, False) for g in (16, 32, 48, 64)}
    for function, body in re.findall(r'Function : (\S+)(.*?)(?=Function :|\Z)', sass, re.S):
        match = re.search(r'guard_packet_kernelILi(\d+)ELi(\d+)ELb([01])EE', function)
        if match is None:
            raise ValueError('Unexpected kernel')
        guards, skip, flag = map(int, match.groups())
        key = (guards, skip, bool(flag))
        if key not in expected:
            raise ValueError('Unexpected or duplicate variant')
        expected.remove(key)
        lines = [line for line in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/', line)]
        clocks = [i for i, line in enumerate(lines) if 'CS2R' in line and 'SR_CLOCKLO' in line]
        if len(clocks) != 4:
            raise ValueError('Missing prelude/packet clocks')
        start, end = clocks[-2:]
        if 'SR_GLOBALTIMERLO' not in lines[start-1] or 'SR_GLOBALTIMERLO' not in lines[end+1]:
            raise ValueError('Timer order changed')
        measured = lines[start+1:end]
        ops = [re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?(\S+)', line)[1] for line in measured]
        if any(op.startswith(('LD', 'UL', 'ST', 'HGMMA', 'MUFU', 'BAR', 'BSYNC', 'BSSY')) for op in ops):
            raise ValueError('Measured region contains another mechanism')
        branches = [line for line, op in zip(measured, ops) if op == 'BRA']
        if len(branches) != (guards if flag else 0):
            raise ValueError('Forward guards were changed or predicated away')
        if ops.count('FSEL') != guards * (2 if flag else 1):
            raise ValueError('Score controls were simplified')
        gaps = []
        for branch in branches:
            m = re.search(r'/\*([0-9a-f]+)\*/\s+@!?P\d+\s+BRA (0x[0-9a-f]+)', branch)
            if m is None:
                raise ValueError('Expected conditional per-thread branch')
            gap = int(m[2], 16) - int(m[1], 16)
            if gap <= 16:
                raise ValueError('Not a conditional forward skip')
            gaps.append(gap)
        records.append(dict(guards=guards, skip_ops=skip, guarded=bool(flag),
            branch_count=len(branches), branch_gaps_bytes=sorted(set(gaps)),
            opcode_counts={op: ops.count(op) for op in sorted(set(ops))}))
    if expected:
        raise ValueError('Incomplete variant coverage')
    return records


def command(argv, environment, timeout):
    process = subprocess.Popen(argv, cwd=TASK, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    return dict(command=argv, exit_code=process.returncode, timed_out=timed_out,
        stdout=out, stderr=err)


@APP.function(gpu='H200', image=IMAGE, timeout=180, retries=0, single_use_containers=True)
def capture():
    result = {'production_duration_consumed': False}
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
        result['identity'] = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10).stdout
        if len(result['identity'].strip().splitlines()) != 1 or not result['identity'].startswith('NVIDIA H200,'):
            raise ValueError('One H200 is required')
        binary = TASK / 'probe'
        commands = {
            'build': ['nvcc', '-O3', '-std=c++17', '-arch=sm_90a', '-Xptxas=-v',
                str(TASK / FILES[0]), '-o', str(binary)],
            'sass': ['cuobjdump', '--dump-sass', str(binary)],
            'measurement': [str(binary)],
        }
        result['required_commands'] = list(commands)
        for name, argv in commands.items():
            remaining = 165 - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError('Capture deadline exhausted')
            item = command(argv, environment, min(remaining, 110 if name == 'measurement' else 40))
            result[name] = item
            if item['exit_code'] or item['timed_out']:
                return result
            if name == 'build':
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', item['stderr'])
                registers = list(map(int, re.findall(r'Used (\d+) registers', item['stderr'])))
                if len(spills) != 16 or any(a != '0' or b != '0' for a, b in spills):
                    raise ValueError('Spill or incomplete compiler variants')
                if len(registers) != 16 or max(registers) > 168:
                    raise ValueError('Register allocation exceeds 384-thread feasibility')
                result['registers'] = registers
            elif name == 'sass':
                result['code_gate'] = code_gate(item['stdout'])
            elif name == 'measurement':
                payload = item['stdout'].encode()
                item['stdout_gzip_base64'] = base64.b64encode(gzip.compress(payload, mtime=0)).decode()
                item['stdout_sha256'] = hashlib.sha256(payload).hexdigest()
                del item['stdout']
        result['final_identity'] = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10).stdout
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
        raise ValueError('A new capture directory is required')
    result = capture.remote()
    out.mkdir(parents=True)
    result['source_sha256'] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2)+'\n')
    if 'failure' in result:
        raise RuntimeError('Capture rejected; raw retained')
    if not all(name in result and result[name]['exit_code'] == 0
               and result[name]['timed_out'] is False for name in result['required_commands']):
        raise RuntimeError('Incomplete capture; raw retained')
    print(dict(function_wall_s=result['gpu_function_wall_s'], compiler_variants=len(result['code_gate'])))
