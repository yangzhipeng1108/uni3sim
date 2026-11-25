"""Bounded independent packed-conversion/LOP3 capacity diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from collections import Counter

import modal

TASK=Path('/tmp/midlayer-bf16-pack-20261107')
if (TASK/'bf16_pack_throughput.cu').is_file():
    SOURCE=TASK/'bf16_pack_throughput.cu'
    WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'bf16_pack_throughput.cu'
    WARMUP=HERE/'gpu_clock_warmup.cu'
    IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
        .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
        .add_local_file(WARMUP,remote_path=str(TASK/WARMUP.name)))
APP=modal.App('midlayer-bf16-pack-20261107')


def code_gate(sass):
    seen=set()
    for body in sass.split('Function : ')[1:]:
        match=re.match(r'_Z13pack_capacityILi(16|32|44|48|64)ELi([12])EE',body)
        if match is None or re.search(r'\b(?:LDL|STL)\b',body):return False
        pairs,feedback=map(int,match.groups())
        if (pairs,feedback) in seen:return False
        seen.add((pairs,feedback));instructions=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?[UP]*\d+\s+)?([A-Z][A-Z0-9_.]*)\s+(.*?);',line)
            if m:instructions.append((int(m[1],16),m[2],m[3]))
        edges=[(pc,int(args.split(',')[-1].strip(),16)) for pc,op,args in instructions
               if op=='BRA' and int(args.split(',')[-1].strip(),16)<pc]
        if len(edges)!=1:return False
        hi,lo=edges[0]
        counts=Counter(op for pc,op,args in instructions if lo<=pc<=hi)
        expected={'F2FP.BF16.F32.PACK_AB':pairs,'LOP3.LUT':feedback*pairs,'UIADD3':1,'ISETP.LE.AND':1,'BRA':1}
        if dict(counts)!=expected:return False
        clocks=[pc for pc,op,args in instructions if op=='CS2R' and ('SR_CLOCKLO' in args or 'SR_GLOBALTIMERLO' in args)]
        if len(clocks)!=4 or not max(clocks[:2])<lo<=hi<min(clocks[2:]):return False
    return seen=={(p,f) for p in (16,32,44,48,64) for f in (1,2)}


def _run(command,environment,timeout):
    p=subprocess.Popen(command,cwd=TASK,env=environment,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:out,err=p.communicate(timeout=timeout)
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
            'build':['nvcc','-O3','-arch=sm_90','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'build_warmup':['nvcc','-O3','-arch=sm_90',str(TASK/WARMUP.name),'-o',str(TASK/'warmup')],
            'warmup':['./warmup','600'],
            'before':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader'],
            'plain':['./probe'],
            'after':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader']}
        result['required_commands']=list(commands)
        for name,command in commands.items():
            remaining=110-(time.monotonic()-started)
            if remaining<=0:result['deadline_exceeded']=True;break
            item=_run(command,environment,min(75,remaining));result[name]=item
            if item['exit_code']!=0:break
            if name=='identity' and (len(item['stdout'].strip().splitlines())!=1 or not item['stdout'].startswith('NVIDIA H200,')):
                result['identity_gate']=False;break
            if name=='build':
                matches=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',item['stderr'])
                result['zero_spill_gate']=len(matches)==10 and all(a==b=='0' for a,b in matches)
                if not result['zero_spill_gate']:break
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
    if out.exists():raise ValueError('Packed conversion capture requires a new attempt directory')
    result=capture.remote();out.mkdir(parents=True)
    files={}
    for path in (SOURCE,WARMUP,Path(__file__)):
        content=path.read_bytes();(out/path.name).write_bytes(content)
        files[path.name]={'sha256':hashlib.sha256(content).hexdigest(),'size':len(content)}
    result['source_files']=files
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(k in result and result[k]['exit_code']==0 for k in result['required_commands']):
        raise RuntimeError('Packed conversion capture incomplete;raw retained')
    print({'status':'raw_independent_mechanism','wall_s':result['gpu_function_wall_s']})
