"""Independent TMA/static-Tensor overlap diagnostic; no Forward pricing."""
import ast
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

TASK=Path('/tmp/midlayer-tma-tensor-overlap-20261108')
if (TASK/'tma_tensor_overlap.cu').is_file():
    SOURCE=TASK/'tma_tensor_overlap.cu'
    REFERENCE=TASK/'fa3_wgmma_service.cu'
    GENERATOR=TASK/'fa3_wgmma_service_gen.py'
    IMAGE=modal.Image.debian_slim()
else:
    HERE=Path(__file__).resolve().parent
    SOURCE=HERE/'tma_tensor_overlap.cu'
    REFERENCE=HERE.parents[1]/'operators/flash_attn/fa3_wgmma_service.cu'
    GENERATOR=HERE.parents[1]/'operators/flash_attn/fa3_wgmma_service_gen.py'
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
    for path in (SOURCE,REFERENCE,GENERATOR):IMAGE=IMAGE.add_local_file(path,remote_path=str(TASK/path.name))
APP=modal.App('midlayer-tma-tensor-overlap-20261108')


def helper(reference,generator):
    marker='__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker)!=1:raise ValueError('Reference preamble changed')
    functions=[n for n in ast.parse(generator).body if isinstance(n,ast.FunctionDef) and n.name=='wgmma_function']
    if len(functions)!=1:raise ValueError('Missing unique PTX owner')
    namespace={}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<existing-primitive-owner>','exec'),namespace)
    return reference[:reference.index(marker)]+''.join(namespace['wgmma_function'](f'ss_n{n}_kmajor',n,False,0) for n in (128,176))


def code_gate(sass):
    expected={(128,1),(128,4),(176,1),(176,4)};records=[]
    if 'WG.DP' in sass or re.search(r'\b(?:LDL|STL)\b',sass):raise ValueError('Serialized matrix/spill')
    for function,body in re.findall(r'Function : (\S+)(.*?)(?=Function :|\Z)',sass,re.S):
        if function.startswith('_Z10initialize'):continue
        match=re.search(r'overlapILi(128|176)ELi([14])EE',function)
        if match is None:raise ValueError('Unexpected kernel')
        key=tuple(map(int,match.groups()))
        if key not in expected:raise ValueError('Duplicate/unexpected matrix variant')
        expected.remove(key)
        lines=[l for l in body.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)]
        matrix=[i for i,l in enumerate(lines) if 'HGMMA.' in l]
        if len(matrix)!=8 or any(f'HGMMA.64x{key[0]}x16.F32.BF16' not in lines[i] for i in matrix):raise ValueError('Wrong matrix packet')
        clock=[i for i,l in enumerate(lines) if 'CS2R' in l and 'SR_CLOCKLO' in l]
        before=[i for i in clock if i<matrix[0]];after=[i for i in clock if i>matrix[-1]]
        if not before or not after:raise ValueError('Missing component timer')
        lo,hi=max(before),min(after);packet='\n'.join(lines[lo+1:hi])
        if 'SR_GLOBALTIMERLO' not in lines[lo-1] or 'SR_GLOBALTIMERLO' not in lines[hi+1]:raise ValueError('Timer order changed')
        if any(op in packet for op in ('UTMALDG','LDG.','STG.','LDGSTS','MUFU.','SYNCS.')):raise ValueError('Another mechanism in matrix window')
        waits=[int(v,16) for v in re.findall(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x([0-9a-f]+)',packet)]
        if waits!=[key[1]-1,key[1]-1,0]:raise ValueError('Matrix queue changed')
        tma_ops=re.findall(r'UTMALDG\.\S+',body)
        if not tma_ops or any('4D' not in op for op in tma_ops):raise ValueError('Missing balanced4D TMA')
        records.append(dict(columns=key[0],queue_depth=key[1],matrix_instructions=8,
            wait_groups=waits,tma_ops=tma_ops,timer_pairs=len(clock)//2,
            matrix_window_payload_memory=False))
    if expected:raise ValueError('Incomplete matrix variants')
    return records


def command(argv,environment,timeout):
    process=subprocess.Popen(argv,cwd=TASK,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timed_out=False
    try:out,err=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(process.pid,signal.SIGKILL);out,err=process.communicate()
    return dict(command=argv,exit_code=process.returncode,timed_out=timed_out,stdout=out,stderr=err)


@APP.function(gpu='H200',image=IMAGE,timeout=180,retries=0,single_use_containers=True)
def capture():
    result={'production_duration_consumed':False,'price_admitted':False};started=time.monotonic()
    try:
        environment=dict(os.environ)
        for name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
            if name in environment:del environment[name]
        for name,folder in (('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda'),('XDG_CACHE_HOME','xdg')):
            p=TASK/folder;p.mkdir(parents=True,exist_ok=True);environment[name]=str(p)
        query=['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem','--format=csv,noheader']
        result['identity']=subprocess.run(query,capture_output=True,text=True,check=True,timeout=10).stdout
        if len(result['identity'].strip().splitlines())!=1 or not result['identity'].startswith('NVIDIA H200,'):raise ValueError('One H200 required')
        header=helper((TASK/REFERENCE.name).read_text(),(TASK/GENERATOR.name).read_text())
        (TASK/'tma_tensor_helper.cuh').write_text(header);result['helper_source']=header
        commands={'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],'measurement':[str(TASK/'probe')]}
        result['required_commands']=list(commands)
        for name,argv in commands.items():
            remaining=165-(time.monotonic()-started)
            if remaining<=0:raise ValueError('Deadline exhausted')
            item=command(argv,environment,min(remaining,125 if name=='measurement' else 30));result[name]=item
            if item['exit_code'] or item['timed_out']:return result
            if name=='build':
                spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',item['stderr'])
                if len(spills)!=5 or any(a!='0' or b!='0' for a,b in spills):raise ValueError('Spill or missing variants')
                result['zero_spill_gate']=True
            elif name=='sass':result['code_gate']=code_gate(item['stdout'])
            elif name=='measurement':
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
    result['source_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,REFERENCE,GENERATOR,Path(__file__))}
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'failure' in result:raise RuntimeError('Capture failed;raw retained')
    if not all(name in result and result[name]['exit_code']==0 and not result[name]['timed_out'] for name in result['required_commands']):raise RuntimeError('Incomplete capture;raw retained')
    print(dict(function_wall_s=result['gpu_function_wall_s'],variants=len(result['code_gate'])))
