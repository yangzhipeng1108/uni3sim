"""Bounded same-address2D/4D TMA readiness and non-duration traffic capture."""
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

TASK=Path('/tmp/midlayer-tma-rank-20261107')
if (TASK/'tma_rank_ready.cu').is_file():
    SOURCE=TASK/'tma_rank_ready.cu'
    WARMUP=TASK/'gpu_clock_warmup.cu'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'tma_rank_ready.cu'
    WARMUP=HERE.parents[1]/'compute/gpu_clock_warmup.cu'
    IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
        .apt_install('cuda-nsight-compute-12-6')
        .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
        .add_local_file(WARMUP,remote_path=str(TASK/WARMUP.name)))
APP=modal.App('midlayer-tma-rank-20261107')


def code_gate(sass):
    bodies=[b for b in sass.split('Function : ')[1:] if b.startswith('_Z10rank_ready')]
    if len(bodies)!=4:
        return False
    domains=set()
    for body in bodies:
        m=re.search(r'rank_readyILi([24])ELi([01])EE',body)
        if m is None or 'LDL' in body or 'STL' in body or body.count('CS2R')<4:
            return False
        rank,prefetch=map(int,m.groups());domains.add((rank,prefetch))
        if (len(re.findall(r'\bUTMALDG\.'+str(rank)+r'D\b',body))!=1
                or ('UTMACCTL.PF' in body)!=(prefetch==1)):
            return False
    return domains=={(2,0),(2,1),(4,0),(4,1)}


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
            'build':['nvcc','-O3','-arch=sm_90','-Xptxas=-v',str(TASK/SOURCE.name),'-L/usr/local/cuda/lib64/stubs','-lcuda','-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'build_warmup':['nvcc','-O3','-arch=sm_90',str(TASK/WARMUP.name),'-o',str(TASK/'warmup')],
            'warmup':['./warmup','600'],
            'before':['nvidia-smi','--query-gpu=clocks.sm','--format=csv,noheader'],
            'plain':['./probe'],
            'rewarm':['./warmup','600'],
            'counter':['ncu','--csv','--page','raw','--cache-control','none','--clock-control','none',
                '--kernel-name','regex:rank_ready','--launch-count','4','--metrics',
                'lts__t_sectors_op_read.sum,dram__bytes_read.sum','./probe','counter'],
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
                result['zero_spill_gate']=len(matches)==6 and all(a==b=='0' for a,b in matches)
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
    if out.exists():raise ValueError('TMA rank capture requires a new attempt directory')
    result=capture.remote();out.mkdir(parents=True)
    files={}
    for path in (SOURCE,WARMUP,Path(__file__)):
        content=path.read_bytes();(out/path.name).write_bytes(content)
        files[path.name]={'sha256':hashlib.sha256(content).hexdigest(),'size':len(content)}
    result['source_files']=files
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(k in result and result[k]['exit_code']==0 for k in result['required_commands']):
        raise RuntimeError('TMA rank capture incomplete;raw retained')
    print({'status':'raw_independent_mechanism','wall_s':result['gpu_function_wall_s']})
