"""Traffic-only heldout cache check, no timing price or parameter selection."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'two_resident_cold_copy.cu'
TASK=Path('/tmp/midlayer-two-resident-cache-check-20261108')
PLAN=(HERE.parents[1]/'references/two_resident_cache_check_plan_sm90.json'
      if modal.is_local() else TASK/'plan.json')
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
    .apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
    .add_local_file(PLAN,remote_path=str(TASK/'plan.json')))
APP=modal.App('midlayer-two-resident-cache-check-20261108')
METRICS=('sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,'
 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,'
 'lts__t_sectors_srcunit_ltcfabric_op_read.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,'
 'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum')

def run(command,limit):
    start=time.monotonic()
    try:
        p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
        return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)
    except subprocess.TimeoutExpired as e:
        def decoded(x):return x.decode(errors='replace') if isinstance(x,bytes) else '' if x is None else x
        return dict(command=command,exit_code=124,stdout=decoded(e.stdout),stderr=decoded(e.stderr),wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture(plan):
    start=time.monotonic();out=dict(status='started',plan=plan,counters=[],
        plan_sha256=hashlib.sha256((TASK/'plan.json').read_bytes()).hexdigest())
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if 'NVIDIA H200' not in out['gpu']['stdout'] or 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout']:
            out['status']='identity_failed';return out
        for c in plan['physical_cells']:
            if time.monotonic()-start>225:out['status']='bounded_partial';return out
            s,g,f=c['stride'],c['grid'],c['fanout']
            item=run(['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application',
                '--cache-control','none','--clock-control','none','--kernel-name','regex:cold_copy',
                '--launch-skip','3','--launch-count','1','--metrics',METRICS,
                str(TASK/'probe'),str(s),str(g),str(f),'1'],30)
            out['counters'].append(dict(stride=s,grid=g,fanout=f,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
            print(f'{len(out["counters"])}/8 heldout counters',flush=True)
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK);out['remote_task_directory_removed']=not TASK.exists()

@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str):
    root=HERE.parents[3];target=Path(output_dir).resolve();tool=Path(budget_tool)
    if target!=(root/'profiler_output/microbench/nvidia_h200/two_resident_cache_check_20261108/attempt_1').resolve() or target.exists():
        raise ValueError('new registered heldout attempt required')
    if not tool.is_file() or not budget_key:raise ValueError('shared atomic reservation required')
    target.mkdir(parents=True);plan=json.loads(PLAN.read_text())
    meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__),PLAN)},
        plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    try:
        out=capture.remote(plan);out.update(meta);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as e:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(e).__name__,error=str(e)),indent=2)+'\n');raise
