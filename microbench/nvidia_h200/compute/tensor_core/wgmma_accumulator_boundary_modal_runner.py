"""Bounded matched-work generic-register/WGMMA boundary mechanism capture."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'wgmma_accumulator_boundary.cu'
REFERENCE=HERE/'../../operators/flash_attn/fa3_wgmma_service.cu'
GENERATOR=HERE/'../../operators/flash_attn/fa3_wgmma_service_gen.py'
TASK=Path('/tmp/midlayer-accumulator-boundary-20261108')
IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
for source in (SOURCE,REFERENCE,GENERATOR):
    IMAGE=IMAGE.add_local_file(source,remote_path=str(TASK/source.name))
APP=modal.App('midlayer-accumulator-boundary-20261108')


def helper(reference,generator):
    marker='__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker)!=1:
        raise ValueError('Primitive helper boundary changed')
    nodes=[n for n in ast.parse(generator).body if isinstance(n,ast.FunctionDef) and n.name=='wgmma_function']
    if len(nodes)!=1:
        raise ValueError('Primitive generator missing')
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<primitive-generator>','exec'),namespace)
    header=reference[:reference.index(marker)]
    for rs in (False,True):
        name='rs_n128_mnmajor' if rs else 'ss_n128_kmajor'
        header+=namespace['wgmma_function'](name,128,rs,int(rs))
    return header


def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=6 or 'WG.DP' in sass or 'MUFU.' in sass:
        return False
    for body in bodies:
        match=re.search(r'accumulator_boundaryILb([01])ELi(4|8|12)EE',body)
        if match is None:
            return False
        steps=int(match[2])
        instructions=[line for line in body.splitlines() if '/*' in line]
        clocks=[i for i,line in enumerate(instructions) if 'SR_CLOCKLO' in line]
        if len(clocks)!=2:
            return False
        timed=instructions[clocks[0]+1:clocks[1]]
        matrices=[line for line in timed if 'HGMMA.64x128x16.' in line]
        # ptxas can unswitch the runtime uniform mode branch into two loops.
        if len(matrices) not in (steps,2*steps):
            return False
        accumulator_matches=[re.search(r'HGMMA\.64x128x16\.\S+\s+(R\d+)',line) for line in matrices]
        if any(match is None for match in accumulator_matches):
            return False
        bases={match[1] for match in accumulator_matches}
        if len(bases)!=1:
            return False
        batches=[part.count('HGMMA.') for part in '\n'.join(timed).split('WARPGROUP.DEPBAR') if 'HGMMA.' in part]
        if any(n!=steps for n in batches):
            return False
        adds=[line for line in timed if re.search(r'\bFADD(?:32I)?\b',line)]
        if len(adds)!=256 or any(re.search(r'\b(?:LDG|LDS|LDL|STL|STG|FFMA|FMUL)\b',line) for line in timed):
            return False
        base=int(next(iter(bases))[1:])
        destinations=[int(re.search(r'FADD(?:32I)?\s+R(\d+)',line)[1]) for line in adds]
        same=[r for r in destinations if base<=r<base+64]
        other=[r for r in destinations if not base<=r<base+64]
        if len(same)!=128 or len(set(same))!=64 or len(other)!=128 or len(set(other))!=64:
            return False
        if any(destinations.count(register)!=2 for register in set(destinations)):
            return False
    return True


@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    result={};start=time.monotonic()
    try:
        header=helper((TASK/REFERENCE.name).read_text(),(TASK/GENERATOR.name).read_text())
        (TASK/'accumulator_wgmma_helper.cuh').write_text(header)
        result['helper_source']=header
        commands={
            'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
            'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-arch=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'measurement':[str(TASK/'probe')],
        }
        result['required_commands']=list(commands)
        for name,command in commands.items():
            p=subprocess.run(command,capture_output=True,text=True,timeout=90,cwd=TASK)
            result[name]=dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
            if p.returncode:
                break
            if name=='build':
                entries=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)
                result['zero_spill_gate']=len(entries)==6 and all(a==b=='0' for a,b in entries)
                if not result['zero_spill_gate']:
                    break
            if name=='sass':
                result['code_gate']=code_gate(p.stdout)
                if not result['code_gate']:
                    break
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-start
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'raw.json').exists():
        raise ValueError('Capture already exists')
    result=capture.remote()
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(name in result and result[name]['exit_code']==0 for name in result['required_commands']):
        raise RuntimeError('Capture command failed;raw retained')
    if not result['zero_spill_gate'] or not result['code_gate']:
        raise RuntimeError('Compiler gate rejected;no GPU measurement')
    print(dict(function_wall_s=result['gpu_function_wall_s'],code_gate=True))
