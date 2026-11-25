"""Run independent Tensor width/accumulator activity controls on one H200."""
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
SOURCES=(HERE/"shared_tensor_activity.cu",HERE/"shared_tensor_activity_gen.py",
    ROOT/"microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu")
IMAGE=modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04",add_python="3.12")
for source in SOURCES:
    IMAGE=IMAGE.add_local_file(source,remote_path="/tmp/tensor_activity/"+source.name)
APP=modal.App("midlayer-h200-shared-tensor-activity-20261107")

@APP.function(gpu="H200",image=IMAGE,timeout=300,retries=0,max_containers=1,single_use_containers=True)
def capture(extended_controls):
    started=time.monotonic()
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader"],check=True,capture_output=True,text=True).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    with tempfile.TemporaryDirectory(prefix="midlayer-shared-tensor-") as directory:
        subprocess.run(["python","/tmp/tensor_activity/shared_tensor_activity_gen.py",
            "/tmp/tensor_activity/fa3_wgmma_service.cu",directory+"/shared_tensor_activity_helper.cuh"],
            check=True,timeout=30)
        command=["nvcc","-O3","-std=c++17","-gencode=arch=compute_90a,code=sm_90a",
            "-Xptxas=-v","-I"+directory,"/tmp/tensor_activity/shared_tensor_activity.cu",
            "-o",directory+"/probe"]
        if extended_controls:
            command.insert(1,"-DEXTENDED_CONTROLS")
        environment=dict(os.environ)
        removed=[]
        for name in ("NVCC_APPEND_FLAGS","NVCC_PREPEND_FLAGS"):
            if name in environment:
                del environment[name]
                removed.append(name)
        build=subprocess.run(command,capture_output=True,text=True,timeout=90,env=environment)
        result={"gpu":gpu,"command":command,"compiler_flag_environment_removed":removed,
            "build_exit_code":build.returncode,"build_stdout":build.stdout,"build_stderr":build.stderr}
        if build.returncode==0:
            result["binary_sha256"]=hashlib.sha256(Path(directory+"/probe").read_bytes()).hexdigest()
            run=subprocess.run([directory+"/probe"],capture_output=True,text=True,timeout=180)
            result.update({"run_exit_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
        result["function_wall_s"]=time.monotonic()-started
    return result

@APP.local_entrypoint()
def main(extended_controls: bool=False, retry_fixed_arena: bool=False):
    campaign="shared_tensor_activity_extended_20261107" if extended_controls else "shared_tensor_activity_20261107"
    if retry_fixed_arena:
        if not extended_controls:
            raise ValueError("fixed-arena retry requires the extended controls")
        campaign="shared_tensor_activity_extended_retry_20261107"
    target=ROOT/"profiler_output/microbench/nvidia_h200"/campaign
    if target.exists():
        raise ValueError("refuse to overwrite captured evidence")
    result=capture.remote(extended_controls)
    result["sources"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    result["runner_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"]=False
    target.mkdir(parents=True)
    (target/"raw.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"build_exit_code":result["build_exit_code"],
        "function_wall_s":result["function_wall_s"]}))
