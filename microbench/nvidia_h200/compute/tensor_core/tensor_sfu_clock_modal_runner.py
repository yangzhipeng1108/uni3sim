"""Capture same-CTA active frequency under independent Tensor/SFU pressure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/"tensor_sfu_clock.cu"
REFERENCE=HERE/".."/".."/"operators/flash_attn/fa3_wgmma_service.cu"
TASK=Path("/tmp/midlayer-tensor-sfu-clock-20261107")
IMAGE=(modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04",add_python="3.11")
       .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
       .add_local_file(REFERENCE,remote_path=str(TASK/REFERENCE.name)))
APP=modal.App("midlayer-tensor-sfu-clock-20261107")


def helper(source):
    first="__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    start="__device__ __forceinline__ void wgmma_ss_n128_kmajor"
    end="__device__ __forceinline__ void wgmma_rs_n128_mnmajor"
    if any(source.count(s)!=1 for s in (first,start,end)):
        raise ValueError("WGMMA primitive source boundary changed")
    return source[:source.index(first)]+source[source.index(start):source.index(end)]


@APP.function(gpu="H200",image=IMAGE,timeout=240,retries=0,single_use_containers=True)
def capture(short_bursts:bool):
    result={};start=time.monotonic()
    try:
        reference=(TASK/REFERENCE.name).read_text()
        header=helper(reference);(TASK/"clock_wgmma_helper.cuh").write_text(header)
        result["reference_sha256"]=hashlib.sha256(reference.encode()).hexdigest()
        result["helper_source"]=header
        commands={
            "identity":["nvidia-smi","--query-gpu=name,driver_version,clocks.sm","--format=csv,noheader"],
            "compiler":["nvcc","--version"],
            "build":["nvcc","-O3","-arch=sm_90a","-Xptxas=-v",str(TASK/SOURCE.name),"-o",str(TASK/"probe")],
            "sass":["cuobjdump","--dump-sass",str(TASK/"probe")]}
        if short_bursts:
            for repeats in (1,8,128):
                for idle in (0,50,1000,10000):
                    commands[f"measurement_r{repeats}_idle{idle}"]=[str(TASK/"probe"),str(repeats),str(idle)]
        else:
            commands["measurement"]=[str(TASK/"probe")]
        commands["final_clock"]=["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader"]
        result["required_commands"]=list(commands)
        for name,command in commands.items():
            p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=120)
            result[name]={"command":command,"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
            if p.returncode:break
        return result
    finally:
        result["gpu_function_wall_s"]=time.monotonic()-start
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir:str,short_bursts:bool=False):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/"raw.json").exists():raise ValueError("capture already exists")
    result=capture.remote(short_bursts)
    result["source_sha256"]=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"raw.json").write_text(json.dumps(result,indent=2)+"\n")
    required=result["required_commands"]
    if not all(name in result and result[name]["exit_code"]==0 for name in required):
        raise RuntimeError("Clock capture failed; inspect the saved raw.json")
    print({"wall_s":result["gpu_function_wall_s"],"commands_passed":len(required)})
