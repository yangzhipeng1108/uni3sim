"""Bounded fixed-state instruction study; no observer retry or Forward rate."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'mbarrier_query_fanin.cu';ANALYSIS=HERE/'mbarrier_query_fanin_analysis.py'
TASK=Path('/tmp/midlayer-query-fanin-20261108')
PLAN=HERE.parents[1]/'references/mbarrier_query_fanin_plan_sm90.json' if modal.is_local() else TASK/'plan.json'
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9')
 .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)).add_local_file(ANALYSIS,remote_path=str(TASK/ANALYSIS.name)).add_local_file(PLAN,remote_path=str(TASK/'plan.json')))
APP=modal.App('midlayer-query-fanin-20261108')

def run(command,limit):
    started=time.monotonic()
    try:
        p=subprocess.run(command,cwd=TASK,text=True,capture_output=True,timeout=limit)
        return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-started)
    except subprocess.TimeoutExpired as e:
        def decoded(x):return x.decode(errors='replace') if isinstance(x,bytes) else '' if x is None else x
        return dict(command=command,exit_code=124,stdout=decoded(e.stdout),stderr=decoded(e.stderr),wall_s=time.monotonic()-started)

@APP.function(gpu='H200',image=IMAGE,timeout=165,retries=0,max_containers=1,single_use_containers=True)
def capture(plan):
    started=time.monotonic();out=dict(status='started',plan=plan,timing=[],counters=[])
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],10),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],45),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],10),
            ('sass',['cuobjdump','--dump-sass','probe'],10)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if ('NVIDIA H200' not in out['gpu']['stdout'] or any(s in out['sass']['stdout'] for s in ('LDGSTS.','HMMA.','HGMMA.'))):out['status']='instruction_boundary_failed';return out
        (TASK/'probe.sass').write_text(out['sass']['stdout'])
        out['query_proof']=run(['python','-c','import json;from mbarrier_query_fanin_analysis import query_proof;print(json.dumps(query_proof(open("probe.sass").read())))'],10)
        if out['query_proof']['exit_code']:out['status']='query_proof_failed';return out
        for process in (0,1):
            for g,d,r,q in (plan['cells'] if process==0 else list(reversed(plan['cells']))):
                if time.monotonic()-started>120:out['status']='bounded_partial';return out
                m=run([str(TASK/'probe'),str(g),str(d),str(r),str(q),'17'],8)
                out['timing'].append(dict(grid=g,distinct_addresses=d,ready=r,queries=q,process=process,measurement=m))
                if m['exit_code']:out['status']='timing_failed';return out
            print(f'instruction process{process} complete',flush=True)
        for g,d,r,q in plan['counter_cells']:
            if time.monotonic()-started>135:out['status']='bounded_partial';return out
            m=run(['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application','--cache-control','none','--clock-control','none','--kernel-name','regex:query_fanin','--launch-skip','6','--launch-count','1','--metrics','sm__cycles_elapsed.avg.per_second,sm__ctas_launched.sum',str(TASK/'probe'),str(g),str(d),str(r),str(q),'1'],10)
            out['counters'].append(dict(grid=g,distinct_addresses=d,ready=r,queries=q,measurement=m))
            if m['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-started;shutil.rmtree(TASK);out['remote_task_directory_removed']=not TASK.exists()

@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str):
    root=HERE.parents[3];target=Path(output_dir).resolve();tool=Path(budget_tool)
    if target!=(root/'profiler_output/microbench/nvidia_h200/mbarrier_query_fanin_20261108/attempt_1').resolve() or target.exists():raise ValueError('new registered attempt required')
    plan=json.loads(PLAN.read_text());sources=(SOURCE,ANALYSIS,Path(__file__));actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if actual!=plan['source_sha256']:raise ValueError('source changed after freeze')
    if not tool.is_file() or not budget_key:raise ValueError('atomic reservation required')
    target.mkdir(parents=True);meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256=actual,plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    started=time.monotonic()
    try:
        out=capture.remote(plan);out.update(meta);out['client_wall_s']=time.monotonic()-started;(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as e:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(e).__name__,error=str(e)),indent=2)+'\n');raise
