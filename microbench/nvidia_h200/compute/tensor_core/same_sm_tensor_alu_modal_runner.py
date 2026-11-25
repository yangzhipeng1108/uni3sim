"""Independent same-SM CTA pair; no production predictor or workload inputs."""
import ast
import bz2
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

TASK=Path('/tmp/midlayer-same-sm-max-busy-20261108')
if (TASK/'same_sm_tensor_alu.cu').is_file():
    SOURCE=TASK/'same_sm_tensor_alu.cu'
    REFERENCE=TASK/'fa3_wgmma_service.cu'
    GENERATOR=TASK/'fa3_wgmma_service_gen.py'
    WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'same_sm_tensor_alu.cu'
    REFERENCE=HERE.parents[1]/'operators/flash_attn/fa3_wgmma_service.cu'
    GENERATOR=REFERENCE.with_name('fa3_wgmma_service_gen.py')
    WARMUP=HERE.parent/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    for path in (SOURCE,REFERENCE,GENERATOR,WARMUP,Path(__file__).resolve()):
        IMAGE=IMAGE.add_local_file(str(path),str(TASK/path.name))
APP=modal.App('midlayer-same-sm-max-busy-20261108')


def helper(reference,generator):
    marker='__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker)!=1:raise ValueError('Independent helper preamble changed')
    definitions=[n for n in ast.parse(generator).body
        if isinstance(n,ast.FunctionDef) and n.name=='wgmma_function']
    if len(definitions)!=1:raise ValueError('Independent PTX generator changed')
    namespace={}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<primitive-generator>','exec'),namespace)
    return reference[:reference.index(marker)]+namespace['wgmma_function']('rs_n128_mnmajor',128,True,1)


def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=16 or 'WG.DP' in sass or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.match(r'_Z7same_smILi0ELi([28])ELi([14])ELi(0|32|64|128)EE',body)
        if m is None:return False
        chains,depth,ops=map(int,m.groups());seen.add((chains,depth,ops))
        if body.count('HGMMA.64x128x16.F32.BF16')!=8:return False
        waits=[int(v,16) for v in re.findall(r'WARPGROUP.DEPBAR.LE gsb0, 0x([0-9a-f]+)',body)]
        if waits!=[depth-1,depth-1,0]:return False
        if body.count('FMNMX')<ops or body.count('SR_CLOCKLO')!=4:return False
    return seen=={(c,d,o) for c in (2,8) for d in (1,4) for o in (0,32,64,128)}


def run(command,environment,timeout):
    process=subprocess.Popen(command,cwd=TASK,env=environment,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:out,err=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGKILL);out,err=process.communicate()
    return {'command':command,'exit_code':process.returncode,'timed_out':timed_out,'stdout':out,'stderr':err}


@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    start=time.monotonic();result={}
    try:
        environment=dict(os.environ)
        for name,folder in (('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')):
            path=TASK/folder;path.mkdir(parents=True,exist_ok=True);environment[name]=str(path)
        environment['CUDA_VISIBLE_DEVICES']='0'
        header=helper(REFERENCE.read_text(),GENERATOR.read_text())
        (TASK/'same_sm_wgmma_helper.cuh').write_text(header);result['helper_source']=header
        commands={
            'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
            'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-arch=sm_90a','-Xptxas=-v',str(SOURCE),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'build_warmup':['nvcc','-O3','-arch=sm_90',str(WARMUP),'-o',str(TASK/'warmup')],
            'warmup':['./warmup','600'],
            'before':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader'],
            'plain':['./probe'],
            'after':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader']}
        result['required_commands']=list(commands)
        for name,command in commands.items():
            remaining=110-(time.monotonic()-start)
            if remaining<=0:result['deadline_exceeded']=True;break
            item=run(command,environment,min(80,remaining));result[name]=item
            if item['exit_code']!=0:break
            if name=='identity' and not item['stdout'].startswith('NVIDIA H200,'):
                result['identity_gate']=False;break
            if name=='build':
                spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',item['stderr'])
                result['zero_spill_gate']=len(spills)==16 and all(a==b=='0' for a,b in spills)
                if not result['zero_spill_gate']:break
            if name=='sass':
                result['code_gate']=code_gate(item['stdout'])
                if not result['code_gate']:break
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-start
        if TASK.exists():shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();result=capture.remote()
    for name in result['required_commands']:
        if name in result:
            item=result[name]
            if name in ('plain','sass'):
                payload=item['stdout'].encode()
                (out/(name+'.txt.bz2')).write_bytes(bz2.compress(payload))
                item['stdout_sha256']=hashlib.sha256(payload).hexdigest();item['stdout_bytes']=len(payload)
                item['stdout']='retained in '+name+'.txt.bz2'
    (out/'same_sm_wgmma_helper.cuh').write_text(result['helper_source'])
    result['helper_source']='retained in same_sm_wgmma_helper.cuh'
    result['local_wait_wall_s']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('gpu_function_wall_s','local_wait_wall_s')},indent=2))
