"""Independent named-barrier service diagnostic;no operator timing or price."""
import base64
import gzip
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

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'named_barrier_rendezvous.cu'
TASK=Path('/tmp/midlayer-named-barrier-rendezvous-20261108')
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
IMAGE=IMAGE.add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-named-barrier-rendezvous-20261108')


def code_gate(sass):
    expected={(384,1,0),(384,1,1),(384,2,0),(384,2,1),(384,2,2),
              (512,1,0),(512,1,1),(512,3,0),(512,3,1),(512,3,2)}
    records=[]
    for function,body in re.findall(r'Function : (\S+)(.*?)(?=Function :|\Z)',sass,re.S):
        m=re.search(r'rendezvousILi(\d+)ELi(\d+)ELi(\d+)EE',function)
        if m is None:raise ValueError('Unexpected kernel')
        key=tuple(map(int,m.groups()))
        if key not in expected:raise ValueError('Duplicate/unexpected variant')
        expected.remove(key)
        lines=[l for l in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)]
        clock=[i for i,l in enumerate(lines) if 'CS2R' in l and 'SR_CLOCKLO' in l]
        if len(clock)!=4:raise ValueError('Missing clock endpoints')
        start,end=clock[-2:]
        if 'SR_GLOBALTIMERLO' not in lines[start-1] or 'SR_GLOBALTIMERLO' not in lines[end+1]:
            raise ValueError('Timer order changed')
        instructions=[]
        for l in lines[start+1:end]:
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?(\S+)\s*(.*)',l)
            instructions.append((int(m[1],16),m[2],m[3]))
        backward=[]
        for pc,op,args in instructions:
            m=re.match(r'(0x[0-9a-f]+)',args)
            if op=='BRA' and m is not None and int(m[1],16)<pc:backward.append((int(m[1],16),pc))
        if len(backward)!=1:raise ValueError('Expected one retained repeat loop')
        lo,hi=backward[0]
        loop=[(pc,op,args) for pc,op,args in instructions if lo<=pc<=hi]
        ops=[op for _,op,_ in loop]
        sync=[args for _,op,args in loop if op.startswith('BAR.SYNC')]
        arrive=[args for _,op,args in loop if op=='BAR.ARV']
        count=1 if key[2] else 0
        if len(sync)!=count or len(arrive)!=count or any('0x100' not in a for a in sync+arrive):
            raise ValueError('Named256-participant protocol changed')
        if ops.count('IMAD')!=4:raise ValueError('Feedback control was changed')
        if any(op.startswith(('LD','ST','ULDC','HGMMA','MUFU','ATOM')) for op in ops):
            raise ValueError('Another mechanism in the repeated packet')
        records.append(dict(threads=key[0],groups=key[1],mode=key[2],loop_ops=ops,
            loop_start_pc=lo,loop_end_pc=hi,
            fixed_frame_memory_ops=[op for pc,op,_ in instructions if not lo<=pc<=hi and op.startswith(('LD','ULDC','ST'))]))
    if expected:raise ValueError('Incomplete variants')
    return records


def command(argv,environment,timeout):
    process=subprocess.Popen(argv,cwd=TASK,env=environment,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:out,err=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGKILL);out,err=process.communicate()
    return dict(command=argv,exit_code=process.returncode,timed_out=timed_out,stdout=out,stderr=err)


@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    result={'production_duration_consumed':False};started=time.monotonic()
    try:
        environment=dict(os.environ)
        for name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
            if name in environment:del environment[name]
        for name,folder in (('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')):
            p=TASK/folder;p.mkdir(parents=True,exist_ok=True);environment[name]=str(p)
        query=['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem','--format=csv,noheader']
        result['identity']=subprocess.run(query,capture_output=True,text=True,check=True,timeout=10).stdout
        if len(result['identity'].strip().splitlines())!=1 or not result['identity'].startswith('NVIDIA H200,'):
            raise ValueError('One H200 required')
        binary=TASK/'probe'
        commands={'build':['nvcc','-O3','-std=c++17','-arch=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(binary)],
                  'sass':['cuobjdump','--dump-sass',str(binary)],'measurement':[str(binary)]}
        result['required_commands']=list(commands)
        for name,argv in commands.items():
            remaining=105-(time.monotonic()-started)
            if remaining<=0:raise ValueError('Deadline exhausted')
            item=command(argv,environment,min(remaining,65 if name=='measurement' else 25));result[name]=item
            if item['exit_code'] or item['timed_out']:return result
            if name=='build':
                spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',item['stderr'])
                if len(spills)!=10 or any(a!='0' or b!='0' for a,b in spills):raise ValueError('Spill or missing variants')
                result['zero_spill_gate']=True
            elif name=='sass':result['code_gate']=code_gate(item['stdout'])
            else:
                payload=item['stdout'].encode();item['stdout_gzip_base64']=base64.b64encode(gzip.compress(payload,mtime=0)).decode()
                item['stdout_sha256']=hashlib.sha256(payload).hexdigest();del item['stdout']
        return result
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        result['failure']=dict(kind=type(error).__name__,message=str(error));return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-started
        if TASK.exists():shutil.rmtree(TASK)
        result['cloud_temporary_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir)
    if out.exists():raise ValueError('New output directory required')
    result=capture.remote();out.mkdir(parents=True)
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'failure' in result:raise RuntimeError('Capture failed;raw retained')
    if not all(name in result and result[name]['exit_code']==0 and not result[name]['timed_out'] for name in result['required_commands']):
        raise RuntimeError('Incomplete capture;raw retained')
    print(dict(function_wall_s=result['gpu_function_wall_s'],variants=len(result['code_gate'])))
