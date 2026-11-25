"""Independent fresh-barrier initialization/fence/CTA sync; no operator or predictor input."""
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

TASK=Path('/tmp/midlayer-barrier-initialization-20261108')
if (TASK/'barrier_initialization.cu').is_file():
    SOURCE=TASK/'barrier_initialization.cu';WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'barrier_initialization.cu';WARMUP=HERE.parents[1]/'compute/gpu_clock_warmup.cu'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    for p in (SOURCE,WARMUP,Path(__file__).resolve()):IMAGE=IMAGE.add_local_file(str(p),str(TASK/p.name))
APP=modal.App('midlayer-barrier-initialization-20261108')

def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=48 or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.search(r'_Z15single_protocolILi(128|256|384|512)ELi([012])ELi([1246])EE',body)
        if m is None:return False
        threads,mode,groups=map(int,m.groups());seen.add((threads,mode,groups))
        lines=body.splitlines();timers=[i for i,s in enumerate(lines) if 'SR_CLOCKLO' in s]
        if len(timers)!=2:return False
        timed='\n'.join(lines[timers[0]:timers[1]])
        if timed.count('SYNCS.EXCH.64')!=(4*groups-2 if mode==1 else 0):return False
        if timed.count('FENCE.VIEW.ASYNC.S')!=((2 if groups==1 else groups) if mode==1 else 0):return False
        if timed.count('BAR.SYNC')!=(1 if mode!=2 else 0):return False
        if re.search(r'\b(?:LDG|LDS|STS|LDL|STL|HGMMA|FMUL|FADD|FFMA)\b',timed):return False
        if 'SYNCS.ARRIVE' in timed or 'SYNCS.PHASECHK' in timed:return False
    return seen=={(t,m,g) for t in (128,256,384,512) for m in (0,1,2) for g in (1,2,4,6)}

def zero_spill_gate(log):
    rows=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',log)
    return len(rows)==48 and all(a==b=='0' for a,b in rows)

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
