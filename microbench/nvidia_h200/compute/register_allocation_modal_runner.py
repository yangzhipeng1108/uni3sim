"""Bounded warpgroup register-quota handoff,with paired empty controls."""
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

TASK=Path('/tmp/midlayer-register-allocation-20261107')
if (TASK/'register_allocation.cu').is_file():
    SOURCE=TASK/'register_allocation.cu'
    WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'register_allocation.cu'
    WARMUP=HERE/'gpu_clock_warmup.cu'
    IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
        .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
        .add_local_file(WARMUP,remote_path=str(TASK/WARMUP.name)))
APP=modal.App('midlayer-register-allocation-20261107')


def code_gate(sass):
    from collections import Counter
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=2 or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.match(r'_Z10allocationILi([23])ELi(40|56)ELi(232|256)EE',body)
        if m is None:return False
        groups,producer,consumer=map(int,m.groups());seen.add((groups,producer,consumer))
        ops=re.findall(r'USETMAXREG\.(TRY_ALLOC|DEALLOC)\.CTAPOOL (?:UP0, )?0x([0-9a-f]+)',body)
        base=256 if groups==2 else 168
        expected=Counter([('TRY_ALLOC',base),('TRY_ALLOC',consumer),('DEALLOC',producer),('DEALLOC',base),('TRY_ALLOC',base)])
        if Counter((op,int(value,16)) for op,value in ops)!=expected:return False
        timed=body[body.index('USETMAXREG'):]
        # Both producer and consumer paths use only low registers after the
        # primer,including while the producer quota is reduced to40/56.
        if max(int(x) for x in re.findall(r'\bR(\d+)\b',timed))>=40:return False
    return seen=={(2,56,256),(3,40,232)}


def _run(command,environment,timeout):
    p=subprocess.Popen(command,cwd=TASK,env=environment,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:
        out,err=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
    return {'command':command,'exit_code':p.returncode,'timed_out':timed_out,'stdout':out,'stderr':err}


@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    started=time.monotonic();result={}
    try:
        environment=dict(os.environ)
        for key,folder in (('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')):
            path=TASK/folder;path.mkdir(parents=True,exist_ok=True);environment[key]=str(path)
        environment['CUDA_VISIBLE_DEVICES']='0'
        commands={
            'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
            'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-arch=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'build_warmup':['nvcc','-O3','-arch=sm_90',str(TASK/WARMUP.name),'-o',str(TASK/'warmup')],
            'warmup':['./warmup','600'],
            'before':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader'],
            'plain':['./probe'],
            'after':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader']}
        result['required_commands']=list(commands)
        for name,command in commands.items():
            remaining=110-(time.monotonic()-started)
            if remaining<=0:
                result['deadline_exceeded']=True;break
            item=_run(command,environment,min(75,remaining));result[name]=item
            if item['exit_code']!=0:
                break
            if name=='identity' and (len(item['stdout'].strip().splitlines())!=1 or not item['stdout'].startswith('NVIDIA H200,')):
                result['identity_gate']=False;break
            if name=='build':
                matches=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',item['stderr'])
                result['zero_spill_gate']=len(matches)==2 and all(a==b=='0' for a,b in matches)
                properties=re.findall(r'Function properties for _Z10allocationILi([23])ELi(?:40|56)ELi(?:232|256)EE.*?Used (\d+) registers',item['stderr'],re.S)
                result['initial_quota_gate']=(len(properties)==2
                    and {int(g) for g,_ in properties}=={2,3}
                    and all(int(r)<=(256 if g=='2' else 168) for g,r in properties))
                if not result['zero_spill_gate'] or not result['initial_quota_gate']:break
            if name=='sass':
                result['code_gate']=code_gate(item['stdout'])
                if not result['code_gate']:break
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-started
        if TASK.exists():shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir)
    if out.exists():raise ValueError('Register allocation capture requires a new attempt directory')
    result=capture.remote();out.mkdir(parents=True)
    files={}
    for path in (SOURCE,WARMUP,Path(__file__)):
        content=path.read_bytes();(out/path.name).write_bytes(content)
        files[path.name]={'sha256':hashlib.sha256(content).hexdigest(),'size':len(content)}
    result['source_files']=files
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(k in result and result[k]['exit_code']==0 for k in result['required_commands']):
        raise RuntimeError('Register allocation capture incomplete;raw retained')
    print({'status':'raw_independent_register_allocation','wall_s':result['gpu_function_wall_s']})
