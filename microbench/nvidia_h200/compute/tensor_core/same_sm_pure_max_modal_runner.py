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

TASK=Path('/tmp/midlayer-same-sm-pure-max-20261108')
if (TASK/'same_sm_pure_max.cu').is_file():
    SOURCE=TASK/'same_sm_pure_max.cu'
    REFERENCE=TASK/'fa3_wgmma_service.cu'
    GENERATOR=TASK/'fa3_wgmma_service_gen.py'
    WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'same_sm_pure_max.cu'
    REFERENCE=HERE.parents[1]/'operators/flash_attn/fa3_wgmma_service.cu'
    GENERATOR=REFERENCE.with_name('fa3_wgmma_service_gen.py')
    WARMUP=HERE.parent/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    for path in (SOURCE,REFERENCE,GENERATOR,WARMUP,Path(__file__).resolve()):
        IMAGE=IMAGE.add_local_file(str(path),str(TASK/path.name))
APP=modal.App('midlayer-same-sm-pure-max-20261108')
VARIANTS=36


def helper(reference,generator):
    marker='__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker)!=1:raise ValueError('Independent helper preamble changed')
    definitions=[n for n in ast.parse(generator).body
        if isinstance(n,ast.FunctionDef) and n.name=='wgmma_function']
    if len(definitions)!=1:raise ValueError('Independent PTX generator changed')
    namespace={}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<primitive-generator>','exec'),namespace)
    return reference[:reference.index(marker)]+namespace['wgmma_function']('rs_n128_mnmajor',128,True,1)


def zero_spill_gate(build_log):
    spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',build_log)
    return len(spills)==VARIANTS and all(a==b=='0' for a,b in spills)


def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=VARIANTS or 'WG.DP' in sass or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.search(r'same_smILi0ELi([28])ELi([14])ELi(32|64|128)ELi(1|16|64)EE',body)
        if m is None:return False
        chains,depth,ops,pool=map(int,m.groups());seen.add((chains,depth,ops,pool))
        if body.count('HGMMA.64x128x16.F32.BF16')!=8:return False
        waits=[int(v,16) for v in re.findall(r'WARPGROUP.DEPBAR.LE gsb0, 0x([0-9a-f]+)',body)]
        if waits!=[depth-1,depth-1,0] or body.count('SR_CLOCKLO')!=4:return False
        lines=body.splitlines();markers=[i for i,line in enumerate(lines) if 'SR_CLOCKLO' in line]
        windows=['\n'.join(lines[markers[i]:markers[i+1]]) for i in (0,2)]
        scalar=[w for w in windows if 'HGMMA' not in w]
        if len(scalar)!=1:return False
        timed=scalar[0]
        if timed.count('FMNMX')!=ops:return False
        if re.search(r'\b(?:FMUL|FADD|FFMA|I2F|LDG|LDL|STL)\b',timed):return False
    return seen=={(c,d,o,p) for c in (2,8) for d in (1,4) for o in (32,64,128) for p in (1,16,64)}


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
                result['zero_spill_gate']=zero_spill_gate(item['stderr'])
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
