"""Capture independent register Tensor activity clocks without operator timing."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent.parent.parent
SOURCES=(HERE/'dense_register_tensor_clock.cu',)
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
for source in SOURCES:
    IMAGE=IMAGE.add_local_file(source,remote_path='/tmp/dense_wgmma/'+source.name)
APP=modal.App('midlayer-h200-register-tensor-clock-20261107')

@APP.function(gpu='H200',image=IMAGE,timeout=300,retries=0,max_containers=1,single_use_containers=True)
def capture():
    started=time.monotonic()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
    if not gpu.startswith('NVIDIA H200,'):
        raise ValueError('one H200 required')
    with tempfile.TemporaryDirectory(prefix='midlayer-dense-wgmma-') as directory:
        command=['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v','-I'+directory,
            '/tmp/dense_wgmma/dense_register_tensor_clock.cu','-o',directory+'/probe']
        build=subprocess.run(command,capture_output=True,text=True,timeout=90)
        result={'gpu':gpu,'command':command,'build_exit_code':build.returncode,
            'build_stdout':build.stdout,'build_stderr':build.stderr}
        if build.returncode==0:
            result['binary_sha256']=hashlib.sha256(Path(directory+'/probe').read_bytes()).hexdigest()
            run=subprocess.run([directory+'/probe'],capture_output=True,text=True,timeout=180)
            result.update({'run_exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
        result['function_wall_s']=time.monotonic()-started
    return result

@APP.local_entrypoint()
def main(output_dir: str):
    target=Path(output_dir).resolve()
    expected=ROOT/'profiler_output/microbench/nvidia_h200/dense_register_tensor_clock_20261107'
    if target not in (expected.resolve(),expected.with_name(expected.name+'_retry').resolve()) or target.exists():
        raise ValueError('output must be the new registered campaign')
    started=time.monotonic();result=capture.remote();result['client_wall_s']=time.monotonic()-started
    result['sources']={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest() for source in SOURCES}
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['production_duration_consumed']=False
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'raw':str(target/'raw.json'),'function_wall_s':result['function_wall_s'],
        'build_exit_code':result['build_exit_code']}))
