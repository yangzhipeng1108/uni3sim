"""Bounded H200 signed quotient and inclusive-warp-scan mechanism capture."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import modal

TASK=Path('/tmp/midlayer-warp-integer-dependencies-20261108')
if (TASK/'warp_integer_dependencies.cu').is_file():
    SOURCE=TASK/'warp_integer_dependencies.cu'
    IMAGE=modal.Image.debian_slim()
else:
    SOURCE=Path(__file__).with_name('warp_integer_dependencies.cu')
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-warp-integer-dependencies-20261108')

def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=6:
        return False
    keys=set()
    for body in bodies:
        match=re.search(r'dependencyILi([012])ELi([14])EE',body)
        if match is None or 'LDL' in body or 'STL' in body:
            return False
        mode,warps=map(int,match.groups())
        keys.add((mode,warps))
        if ('MUFU.RCP ' in body)!=(mode==1) or body.count('SHFL.UP')!=(5 if mode==2 else 0):
            return False
        if body.count('CS2R')<4:
            return False
    return keys=={(m,w) for m in (0,1,2) for w in (1,4)}

@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    result={};started=time.monotonic()
    try:
        commands={
            'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
            'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-arch=sm_90','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'plain':[str(TASK/'probe')]}
        result['required_commands']=list(commands)
        for name,command in commands.items():
            p=subprocess.run(command,capture_output=True,text=True,timeout=90,cwd=TASK)
            result[name]={'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
            if p.returncode:
                break
            if name=='identity' and not p.stdout.startswith('NVIDIA H200,'):
                raise ValueError('This capture requires the authorized H200')
            if name=='build':
                spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)
                result['zero_spill_gate']=len(spills)==6 and all(a==b=='0' for a,b in spills)
                if not result['zero_spill_gate']:
                    break
            if name=='sass':
                result['code_gate']=code_gate(p.stdout)
                if not result['code_gate']:
                    break
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-started
        if TASK.exists():
            shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir)
    if out.exists():
        raise ValueError('A new attempt directory is required')
    result=capture.remote()
    out.mkdir(parents=True)
    result['source_files']={}
    for path in (SOURCE,Path(__file__)):
        payload=path.read_bytes()
        (out/path.name).write_bytes(payload)
        result['source_files'][path.name]={'sha256':hashlib.sha256(payload).hexdigest(),'size':len(payload)}
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(k in result and result[k]['exit_code']==0 for k in result['required_commands']):
        raise RuntimeError('Capture incomplete; raw retained')
    print({'wall_s':result['gpu_function_wall_s'],'status':'independent_integer_dependencies'})
