"""Paired C++/whole-PTX narrow WGMMA window sweep on one H200."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import modal

IMAGE=modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04",add_python="3.12")
if modal.is_local():
    HERE=Path(__file__).resolve().parent
    ROOT=HERE.parents[3]
    SOURCES=(HERE/"narrow_wgmma_service.cu",HERE/"shared_tensor_activity_gen.py",
        HERE/"narrow_wgmma_service_gen.py",
        ROOT/"microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu")
    for source in SOURCES:
        IMAGE=IMAGE.add_local_file(source,remote_path="/tmp/narrow_wgmma/"+source.name)
APP=modal.App("midlayer-h200-narrow-wgmma-paired-20261107")

@APP.function(gpu="H200",image=IMAGE,timeout=300,retries=0,single_use_containers=True)
def capture():
    started=time.monotonic()
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],
        capture_output=True,text=True,check=True,timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    with tempfile.TemporaryDirectory(prefix="midlayer-narrow-wgmma-") as directory:
        subprocess.run(["python","/tmp/narrow_wgmma/shared_tensor_activity_gen.py",
            "/tmp/narrow_wgmma/fa3_wgmma_service.cu",directory+"/shared_tensor_activity_helper.cuh"],
            check=True,timeout=30)
        subprocess.run(["python","/tmp/narrow_wgmma/narrow_wgmma_service_gen.py",
            directory+"/narrow_wgmma_service_helper.cuh"],check=True,timeout=30)
        environment=dict(os.environ);removed=[]
        for name in ("NVCC_APPEND_FLAGS","NVCC_PREPEND_FLAGS"):
            if name in environment:
                del environment[name];removed.append(name)
        result={"gpu":gpu,"compiler_flag_environment_removed":removed,"modes":{},
            "nvcc_version":subprocess.run(["nvcc","--version"],capture_output=True,text=True,
                check=True,timeout=10).stdout}
        for mode in ("legacy","single_asm"):
            command=["nvcc","-O3","-std=c++17","-gencode=arch=compute_90a,code=sm_90a","-Xptxas=-v",
                "-I"+directory,"/tmp/narrow_wgmma/narrow_wgmma_service.cu","-o",directory+"/probe"]
            if mode=="single_asm":
                command.append("-DSINGLE_ASM_WINDOW")
            build=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=90)
            item={"build_command":command,"build_exit_code":build.returncode,
                "build_stdout":build.stdout,"build_stderr":build.stderr}
            result["modes"][mode]=item
            if build.returncode!=0:
                break
            item["binary_sha256"]=hashlib.sha256(Path(directory+"/probe").read_bytes()).hexdigest()
            item["sass"]={}
            for width,chains,wgs,pending in ((8,1,1,1),(8,1,2,1),(8,4,2,1),(64,1,1,1),(64,1,2,1)):
                symbol=f"_Z18instruction_windowILi{width}ELi{chains}ELi{wgs}ELi{pending}EEviiPmPf"
                dump=subprocess.run(["cuobjdump","--dump-sass","--function",symbol,directory+"/probe"],
                    capture_output=True,text=True,check=True,timeout=15)
                if "Function : "+symbol not in dump.stdout:
                    raise ValueError("missing selected SASS")
                item["sass"][symbol]={"stdout":dump.stdout,"stderr":dump.stderr}
            run=subprocess.run([directory+"/probe"],capture_output=True,text=True,timeout=90)
            item.update({"run_exit_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
            if run.returncode!=0:
                break
        result["function_wall_s"]=time.monotonic()-started
    return result

@APP.local_entrypoint()
def main():
    target=ROOT/"profiler_output/microbench/nvidia_h200/narrow_wgmma_service_paired_20261107"
    if target.exists():
        raise ValueError("refuse to overwrite captured evidence")
    result=capture.remote()
    result["sources"]={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest() for source in SOURCES}
    result["runner_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"]=False
    target.mkdir(parents=True)
    for mode,item in result["modes"].items():
        if "sass" in item:
            for symbol,dump in item["sass"].items():
                filename=mode+"_"+symbol+".sass.txt.gz"
                data=dump["stdout"].encode()
                (target/filename).write_bytes(gzip.compress(data,mtime=0))
                dump["stdout_sha256"]=hashlib.sha256(data).hexdigest()
                dump["path"]=filename
                del dump["stdout"]
    (target/"raw.json").write_text(json.dumps(result,indent=2)+"\n")
    snapshot=target/"source";snapshot.mkdir()
    for source in (*SOURCES,Path(__file__)):
        if source.name=="fa3_wgmma_service.cu":
            (snapshot/(source.name+".gz")).write_bytes(gzip.compress(source.read_bytes(),mtime=0))
        else:
            (snapshot/source.name).write_bytes(source.read_bytes())
    print(json.dumps({"modes":{mode:item["build_exit_code"] for mode,item in result["modes"].items()},
        "function_wall_s":result["function_wall_s"]}))
