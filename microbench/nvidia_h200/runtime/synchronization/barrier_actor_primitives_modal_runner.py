"""Independent one-actor init/fence primitive groups;no operator or parameter reads in timing."""
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

TASK=Path('/tmp/midlayer-barrier-actor-primitives-20261108')
if (TASK/'barrier_actor_primitives.cu').is_file():
    SOURCE=TASK/'barrier_actor_primitives.cu';WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'barrier_actor_primitives.cu';WARMUP=HERE.parents[1]/'compute/gpu_clock_warmup.cu'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    for p in (SOURCE,WARMUP,Path(__file__).resolve()):IMAGE=IMAGE.add_local_file(str(p),str(TASK/p.name))
APP=modal.App('midlayer-barrier-actor-primitives-20261108')

def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=80 or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.search(r'_Z12actor_groupsILi(128|256)ELi(0|2|4|8|n1)ELi([012468])ELi(1|4|128)EE',body)
        if m is None:return False
        threads=int(m[1]);size=-1 if m[2]=='n1' else int(m[2]);groups=int(m[3]);arrivals=int(m[4]);seen.add((threads,size,groups,arrivals))
        lines=body.splitlines();timers=[i for i,s in enumerate(lines) if 'SR_CLOCKLO' in s]
        if len(timers)!=2:return False
        timed='\n'.join(lines[timers[0]:timers[1]])
        count=22 if size==-1 else size*groups
        if timed.count('SYNCS.EXCH.64')!=count or timed.count('FENCE.VIEW.ASYNC.S')!=(1 if count else 0):return False
        if re.search(r'\b(?:LDG|LDC|ULDC|LDS|STS|LDL|STL|HGMMA|FMUL|FADD|FFMA|MUFU|WARPSYNC|BSSY|BSYNC|BREAK|BRA|SHFL|BAR)\b',timed):return False
        if 'SYNCS.ARRIVE' in timed or 'SYNCS.PHASECHK' in timed:return False
    expected={(t,n,g,a) for t in (128,256) for n in (2,4,8) for g in (1,2,4,8) for a in (1,4,128)}
    expected.update((t,0,0,a) for t in (128,256) for a in (1,4,128));expected.update((t,-1,6,1) for t in (128,256))
    return seen==expected

def zero_spill_gate(log):
    rows=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',log)
    return len(rows)==80 and all(a==b=='0' for a,b in rows)

@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    started=time.monotonic();r={}
    try:
        env=dict(os.environ)
        for name,folder in (('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')):
            p=TASK/folder;p.mkdir(parents=True,exist_ok=True);env[name]=str(p)
        commands={
          'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
          'compiler':['nvcc','--version'],
          'build':['nvcc','-O3','-arch=sm_90','-Xptxas=-v',str(SOURCE),'-o',str(TASK/'probe')],
          'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
          'build_warmup':['nvcc','-O3','-arch=sm_90',str(WARMUP),'-o',str(TASK/'warmup')],
          'warmup':['./warmup','600'],
          'before':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader'],
          'plain':['./probe'],
          'after':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader']}
        r['required_commands']=list(commands)
        for name,cmd in commands.items():
            item=subprocess.run(cmd,cwd=TASK,env=env,text=True,capture_output=True,timeout=80)
            r[name]={'command':cmd,'exit_code':item.returncode,'stdout':item.stdout,'stderr':item.stderr}
            if item.returncode:break
            if name=='identity' and not item.stdout.startswith('NVIDIA H200,'):r['identity_gate']=False;break
            if name=='build':
                r['zero_spill_gate']=zero_spill_gate(item.stderr)
                if not r['zero_spill_gate']:break
            if name=='sass':
                r['code_gate']=code_gate(item.stdout)
                if not r['code_gate']:break
        return r
    finally:
        r['gpu_function_wall_s']=time.monotonic()-started
        if TASK.exists():shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=False)
    r=capture.remote()
    for name in ('sass','plain'):
        if name in r:
            blob=r[name]['stdout'].encode();(p/(name+'.txt.bz2')).write_bytes(bz2.compress(blob))
            r[name]['stdout_sha256']=hashlib.sha256(blob).hexdigest();r[name]['stdout_bytes']=len(blob)
            r[name]['stdout']='retained in '+name+'.txt.bz2'
    (p/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('gpu_function_wall_s',)},indent=2))
