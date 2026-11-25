"""Independent unrolled MAX/ADD dependency and issue streams;no operator or predictor input."""
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

TASK=Path('/tmp/midlayer-scalar-dependency-stream-20261108')
if (TASK/'scalar_dependency_stream.cu').is_file():
    SOURCE=TASK/'scalar_dependency_stream.cu';WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'scalar_dependency_stream.cu';WARMUP=HERE/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    for p in (SOURCE,WARMUP,Path(__file__).resolve()):IMAGE=IMAGE.add_local_file(str(p),str(TASK/p.name))
APP=modal.App('midlayer-scalar-dependency-stream-20261108')

def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=96 or re.search(r'\b(?:LDL|STL)\b',sass):return False
    seen=set()
    for body in bodies:
        m=re.search(r'_Z6streamILi(32|128|256)ELi([01])ELi([1248])ELi(16|32|64|128)EE',body)
        if m is None:return False
        threads,kind,chains,levels=map(int,m.groups());seen.add((threads,kind,chains,levels))
        lines=body.splitlines();timers=[i for i,s in enumerate(lines) if 'SR_CLOCKLO' in s]
        if len(timers)!=2:return False
        timed='\n'.join(lines[timers[0]:timers[1]])
        if len(re.findall(r'\bFMNMX\b',timed))!=(chains*levels if kind==0 else 0):return False
        if len(re.findall(r'\bFADD\b',timed))!=chains+(chains*levels if kind==1 else 0):return False
        if re.search(r'\b(?:LDG|LDS|STS|LDL|STL|HGMMA|FMUL|FFMA|MUFU|SHFL)\b',timed):return False
        if timed.count('FSETP.NAN')!=1 or timed.count('BPT.TRAP')!=1:return False
        # Dependency depth,not register ABI:single-chain instructions cannot
        # be transformed into independent rows or a parallel reduction tree.
        op='FMNMX' if kind==0 else 'FADD';depth={};longest=0
        for line in timed.splitlines():
            match=re.search(r'\b'+op+r'\s+([^;]+);',line)
            if match is None:continue
            operands=match[1].split(',');dest=operands[0].strip();sources=re.findall(r'\bR[0-9]+\b',','.join(operands[1:]))
            value=1+max([depth[r] if r in depth else 0 for r in sources]+[0]);depth[dest]=value;longest=max(longest,value)
        if kind==0 and longest!=levels:return False
        if kind==1 and not levels<=longest<=levels+chains:return False
    return seen=={(t,k,c,n) for t in (32,128,256) for k in (0,1) for c in (1,2,4,8) for n in (16,32,64,128)}

def zero_spill_gate(log):
    rows=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',log)
    return len(rows)==96 and all(a==b=='0' for a,b in rows)

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
