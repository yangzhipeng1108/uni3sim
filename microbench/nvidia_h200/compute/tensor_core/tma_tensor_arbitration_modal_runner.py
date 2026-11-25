"""Run independent TMA/Tensor arbitration or data-activity controls on H200."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent.parent.parent
SOURCES=(HERE/'tma_tensor_arbitration.cu',HERE/'tma_tensor_arbitration_gen.py',
    ROOT/'microbench/nvidia_h200/operators/gemm/fp8_tma_operand_stream.cu',
    ROOT/'microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu')
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
for source in SOURCES:
    IMAGE=IMAGE.add_local_file(source,remote_path='/tmp/tma_tensor_arbitration/'+source.name)
APP=modal.App('midlayer-h200-tma-tensor-controls-20261107')

@APP.function(gpu='H200',image=IMAGE,timeout=300,retries=0,max_containers=1,single_use_containers=True)
def capture(activity_controls):
    started=time.monotonic()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
    if not gpu.startswith('NVIDIA H200,'):
        raise ValueError('one H200 required')
    with tempfile.TemporaryDirectory(prefix='midlayer-arbitration-') as directory:
        subprocess.run(['python','/tmp/tma_tensor_arbitration/tma_tensor_arbitration_gen.py',
            '/tmp/tma_tensor_arbitration/fp8_tma_operand_stream.cu',
            '/tmp/tma_tensor_arbitration/fa3_wgmma_service.cu',directory+'/arbitration_helper.cuh'],check=True,timeout=30)
        command=['nvcc','-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a',
            '-Xptxas=-v','-I'+directory,'/tmp/tma_tensor_arbitration/tma_tensor_arbitration.cu','-o',directory+'/probe']
        compile_environment=dict(os.environ)
        removed=[]
        for name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
            if name in compile_environment:
                del compile_environment[name]
                removed.append(name)
        build=subprocess.run(command,capture_output=True,text=True,timeout=90,env=compile_environment)
        result={'gpu':gpu,'command':command,'compiler_flag_environment_removed':removed,
            'build_exit_code':build.returncode,'build_stdout':build.stdout,'build_stderr':build.stderr}
        if build.returncode==0:
            result['binary_sha256']=hashlib.sha256(Path(directory+'/probe').read_bytes()).hexdigest()
            arguments=[directory+'/probe']
            if activity_controls:
                arguments.append('activity-controls')
            run=subprocess.run(arguments,capture_output=True,text=True,timeout=180)
            result.update({'run_exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
        result['function_wall_s']=time.monotonic()-started
    return result

@APP.local_entrypoint()
def main(output_dir: str, activity_controls: bool=False):
    target=Path(output_dir).resolve()
    campaign='tma_tensor_activity_20261107' if activity_controls else 'tma_tensor_arbitration_20261107'
    expected=ROOT/'profiler_output/microbench/nvidia_h200'/campaign
    if target!=expected.resolve() or target.exists():
        raise ValueError('output must be the new registered campaign')
    started=time.monotonic();result=capture.remote(activity_controls);result['client_wall_s']=time.monotonic()-started
    result['sources']={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest() for source in SOURCES}
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['production_duration_consumed']=False
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'raw':str(target/'raw.json'),'function_wall_s':result['function_wall_s'],
        'build_exit_code':result['build_exit_code']}))
