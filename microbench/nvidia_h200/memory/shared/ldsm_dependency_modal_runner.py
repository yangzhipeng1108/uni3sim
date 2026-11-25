"""One bounded independent shared-load/use capture, no full operator or automatic retry."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-ldsm-dependency-20261108')
SOURCE=HERE/'ldsm_dependency.cu'
ANALYSIS=HERE/'ldsm_dependency_analysis.py'
PLAN=HERE.parents[1]/'references/ldsm_dependency_plan_sm90.json' if modal.is_local() else TASK/'plan.json'
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
 .apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
 .add_local_file(ANALYSIS,remote_path=str(TASK/ANALYSIS.name)).add_local_file(PLAN,remote_path=str(TASK/'plan.json')))
APP=modal.App('midlayer-ldsm-dependency-20261108')
METRICS=('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,'
 'sm__ctas_launched.sum,sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second')


def run(command,limit):
    start=time.monotonic()
    try:
        p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
        return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)
    except subprocess.TimeoutExpired as exc:
        def decode(v):return '' if v is None else v.decode(errors='replace') if isinstance(v,bytes) else v
        return dict(command=command,exit_code=124,stdout=decode(exc.stdout),stderr=decode(exc.stderr),wall_s=time.monotonic()-start,timed_out=True)


@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture(plan):
    start=time.monotonic();out=dict(status='started',timing=[],counters=[])
    try:
        for name,cmd,limit in (
          ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
          ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],45),
          ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
          ('sass',['cuobjdump','--dump-sass','probe'],15),
          ('cpu_fixture',['./probe','--cpu-only'],10)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if 'H200' not in out['gpu']['stdout']:out['status']='wrong_gpu';return out
        (TASK/'probe.sass').write_text(out['sass']['stdout'])
        (TASK/'resources.txt').write_text(out['resources']['stdout'])
        out['static_proof']=run(['python','-c','import json;from ldsm_dependency_analysis import static_proof,resource_proof;print(json.dumps(dict(instructions=static_proof(open("probe.sass").read()),resources=resource_proof(open("resources.txt").read()))))'],10)
        if out['static_proof']['exit_code']:out['status']='static_proof_failed';return out
        out['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        for process in (0,1):
            for c,w,n in (plan['cells'] if process==0 else list(reversed(plan['cells']))):
                if time.monotonic()-start>205:out['status']='bounded_partial';return out
                item=run(['./probe',str(c),str(w),str(n),'17'],12)
                out['timing'].append(dict(process=process,chains=c,warps=w,inner=n,measurement=item))
                if item['exit_code']:out['status']='timing_failed';return out
            print(f'process{process} completed all frozen timing cells',flush=True)
        for c,w in plan['counter_cells']:
            if time.monotonic()-start>205:out['status']='bounded_partial';return out
            item=run(['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application','--cache-control','none','--clock-control','none',
                '--kernel-name','regex:ldsm_dependency','--launch-skip','6','--launch-count','1','--metrics',METRICS,'./probe',str(c),str(w),'1024','1'],25)
            out['counters'].append(dict(chains=c,warps=w,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
            print(f'{len(out["counters"])} counters complete',flush=True)
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str):
    root=HERE.parents[3];target=Path(output_dir).resolve();tool=Path(budget_tool)
    if target!=(root/'profiler_output/microbench/nvidia_h200/ldsm_dependency_20261108/attempt_1').resolve() or target.exists():raise ValueError('new registered attempt required')
    if not tool.is_file() or not budget_key:raise ValueError('atomic shared reservation required')
    plan=json.loads(PLAN.read_text());sources=(SOURCE,ANALYSIS,Path(__file__))
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if actual!=plan['source_sha256']:raise ValueError('source changed after frozen plan')
    target.mkdir(parents=True)
    meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256=actual,
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    started=time.monotonic()
    try:
        out=capture.remote(plan);out.update(meta);out['client_wall_s']=time.monotonic()-started
        (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as exc:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n');raise
