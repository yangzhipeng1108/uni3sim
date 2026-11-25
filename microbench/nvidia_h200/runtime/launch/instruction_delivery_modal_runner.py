"""Independent fixed-work instruction footprint and No Instruction counters."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'instruction_delivery.cu'
TASK = Path('/tmp/midlayer-instruction-delivery-20261107')
IMAGE = modal.Image.from_registry(
    'nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11'
).apt_install('cuda-nsight-compute-12-6').add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App('midlayer-instruction-delivery-20261107')
METRICS = ('smsp__warp_issue_stalled_no_instruction_per_warp_active.pct',
           'smsp__inst_executed.sum')


def instruction_streams():
    lines = ['template<int Ops,int Tag> __device__ __forceinline__ void stream(Values& v,int iterations,float a,float b);']
    for ops in (64, 256, 512, 1024, 2048):
        for tag in (0, 1):
            lines += [f'template<> __device__ __forceinline__ void stream<{ops},{tag}>(Values& v,int iterations,float a,float b) {{',
                      '#pragma unroll 1', 'for(int i=0;i<iterations;++i) {']
            for i in range(ops):
                j = i % 8
                if tag == 0:
                    lines.append(f'asm volatile("fma.rn.f32 %0,%0,%1,%2;":"+f"(v.v[{j}]):"f"(a),"f"(b));')
                else:
                    lines.append(f'asm volatile("add.rn.f32 %0,%0,%1;":"+f"(v.v[{j}]):"f"(b));')
            lines += ['}', '}']
    return '\n'.join(lines) + '\n'


def code_gate(sass):
    bodies = sass.split('Function : ')[1:]
    if len(bodies) != 20:
        return False
    seen = set()
    for body in bodies:
        match = re.search(r'instruction_deliveryILi(128|384)ELi(64|256|512|1024|2048)ELi([01])EE', body)
        if match is None:
            return False
        threads, ops, split = map(int, match.groups())
        key = threads, ops, split
        if key in seen:
            return False
        seen.add(key)
        fma = len(re.findall(r'\bFFMA\s', body))
        add = len(re.findall(r'\bFADD\s', body))
        if not (split and threads == 128) and fma < ops:
            return False
        if split and threads == 384 and add < ops:
            return False
        # With128 threads the split path is exclusively ADD, so no FFMA
        # stream executes. Its init may still contain a few FFMA sites.
        if split and threads == 128 and add < ops:
            return False
        if any(op in body for op in ('LDL', 'STL', 'HGMMA', 'MUFU', 'LDG.')):
            return False
    return True


def counter_values(stdout, ops):
    lines = stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('"ID",')]
    if len(starts) != 1:
        raise ValueError('Expected one Nsight counter table')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])))
    if reader.fieldnames is None or not set(METRICS).issubset(reader.fieldnames):
        raise ValueError('Requested counter fields missing despite CLI exit status')
    rows = [row for row in reader if row['ID'].isdigit()]
    if len(rows) != 1:
        raise ValueError('Expected one profiled instruction-delivery launch')
    row = rows[0]
    if re.search(rf'instruction_delivery<(?:\(int\))?384,\s*(?:\(int\))?{ops},\s*(?:\(int\))?1>', row['Kernel Name']) is None:
        raise ValueError('Unexpected profiled instruction footprint')
    values = {name: float(row[name]) for name in METRICS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite instruction-delivery counter')
    if not 0 <= values[METRICS[0]] <= 100 or values[METRICS[1]] <= 0:
        raise ValueError('Invalid instruction-delivery counter units/range')
    return values


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture(counter_only: bool = False):
    result = {}
    started = time.monotonic()
    try:
        helper = instruction_streams()
        (TASK / 'instruction_streams.cuh').write_text(helper)
        result['generated_helper'] = helper
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90', '-Xptxas=-v', str(TASK / SOURCE.name), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
        }
        if not counter_only:
            commands['measurement'] = [str(TASK / 'probe')]
        metrics = ','.join(METRICS)
        for ops in (64, 512, 2048):
            commands[f'counter_{ops}'] = [
                'ncu', '--csv', '--page', 'raw', '--cache-control', 'none', '--clock-control', 'none',
                '--kernel-name', 'regex:instruction_delivery', '--launch-skip', '8', '--launch-count', '1',
                '--metrics', metrics, str(TASK / 'probe'), str(ops)]
        result['required_commands'] = list(commands)
        for name, cmd in commands.items():
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=TASK, timeout=90)
            result[name] = {'command': cmd, 'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
            if proc.returncode:
                break
            if name == 'build':
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', proc.stderr)
                result['zero_spill_gate'] = len(spills) == 20 and all(a == b == '0' for a, b in spills)
                if not result['zero_spill_gate']:
                    break
            if name == 'sass':
                result['code_gate'] = code_gate(proc.stdout)
                if not result['code_gate']:
                    break
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str, counter_only: bool = False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw.json').exists():
        raise ValueError('capture already exists')
    result = capture.remote(counter_only)
    result['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    if not all(k in result and result[k]['exit_code'] == 0 for k in result['required_commands']):
        raise RuntimeError('Instruction delivery capture failed; inspect saved raw.json')
    checked = {str(ops): counter_values(result[f'counter_{ops}']['stdout'], ops)
               for ops in (64, 512, 2048)}
    (out / 'counter_validation.json').write_text(json.dumps(checked, indent=2) + '\n')
    print({'wall_s': result['gpu_function_wall_s'], 'code_gate': result['code_gate']})
