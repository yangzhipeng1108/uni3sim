"""Frozen independent two-resident cold request progress; no operator price."""
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
ANALYSIS=HERE/'two_resident_cold_copy_analysis.py'
TASK=Path('/tmp/midlayer-two-resident-cold-copy-20261108')
PLAN=(HERE.parents[1]/'references/two_resident_cold_copy_plan_sm90.json'
      if modal.is_local() else TASK/'plan.json')
CACHE_PLAN=(HERE.parents[1]/'references/two_resident_cold_copy_cache_accounted_plan_sm90.json'
            if modal.is_local() else TASK/'cache_plan.json')
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
       .apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
       .add_local_file(PLAN,remote_path=str(TASK/'plan.json'))
       .add_local_file(CACHE_PLAN,remote_path=str(TASK/'cache_plan.json'))
       .add_local_file(ANALYSIS,remote_path=str(TASK/ANALYSIS.name)))
APP=modal.App('midlayer-two-resident-cold-copy-20261108')
METRICS=('sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,'
 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,'
 'lts__t_sectors_srcunit_ltcfabric_op_read.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,'
 'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum')

def run(command,limit):
    start=time.monotonic()
    try:
        p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
        return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)
    except subprocess.TimeoutExpired as exc:
        def decoded(x):return x.decode(errors='replace') if isinstance(x,bytes) else '' if x is None else x
        return dict(command=command,exit_code=124,stdout=decoded(exc.stdout),stderr=decoded(exc.stderr),wall_s=time.monotonic()-start,timed_out=True)

@APP.function(gpu='H200',image=IMAGE,timeout=840,retries=0,max_containers=1,single_use_containers=True)
def capture(plan):
    if plan['kind']=='independent_two_resident_cold_copy_plan':
        plan_file=TASK/'plan.json';analysis_flags=[]
    elif plan['kind']=='independent_two_resident_cache_accounted_progress_plan':
        plan_file=TASK/'cache_plan.json';analysis_flags=['--cache-accounted']
    else:
        raise ValueError('unknown cold-copy accounting protocol')
    start=time.monotonic();out=dict(status='started',plan=plan,timing=[],counters=[],
        plan_sha256=hashlib.sha256(plan_file.read_bytes()).hexdigest())
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if ('NVIDIA H200' not in out['gpu']['stdout'] or 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout']
                or 'HGMMA.' in out['sass']['stdout'] or 'HMMA.' in out['sass']['stdout']):
            out['status']='physical_boundary_failed';return out
        out['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        def timing(cells,label):
            for process in (0,1):
                ordered=cells if process==0 else tuple(reversed(cells))
                for ordinal,(stride,grid,fanout) in enumerate(ordered):
                    if time.monotonic()-start>780:out['status']='bounded_partial';return False
                    item=run([str(TASK/'probe'),str(stride),str(grid),str(fanout),'63'],15)
                    out['timing'].append(dict(stride=stride,grid=grid,fanout=fanout,process=process,measurement=item))
                    if item['exit_code']:out['status']='timing_failed';return False
                    if (ordinal+1)%20==0:print(f'{label}/process{process}: {ordinal+1}/{len(cells)} timing cells',flush=True)
            return True
        if not timing(plan['counter_cells'],'controls'):return out
        for ordinal,(stride,grid,fanout) in enumerate(plan['counter_cells']):
            if time.monotonic()-start>780:out['status']='bounded_partial';return out
            item=run(['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application',
                '--cache-control','none','--clock-control','none','--kernel-name','regex:cold_copy',
                '--launch-skip','3','--launch-count','1','--metrics',METRICS,
                str(TASK/'probe'),str(stride),str(grid),str(fanout),'1'],30)
            out['counters'].append(dict(stride=stride,grid=grid,fanout=fanout,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
            print(f'{ordinal+1}/{len(plan["counter_cells"])} counters',flush=True)
        out['status']='screen_complete'
        (TASK/'screen_raw.json').write_text(json.dumps(out))
        out['screen_analysis']=run(['python',ANALYSIS.name,'screen_raw.json','screen_summary.json',
            '--plan',str(plan_file),'--control-only',*analysis_flags],30)
        if out['screen_analysis']['exit_code']:out['status']='screen_analysis_failed';return out
        out['screen_result']=json.loads((TASK/'screen_summary.json').read_text())
        if not out['screen_result']['mechanism_screen_qualified']:
            out['status']='mechanism_screen_failed';return out
        controls={tuple(c) for c in plan['counter_cells']}
        remaining=[c for c in plan['physical_cells'] if tuple(c) not in controls]
        if not timing(remaining,'source_endpoints'):return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()

@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str,cache_accounted:bool=False):
    root=HERE.parents[3];target=Path(output_dir).resolve();tool=Path(budget_tool)
    attempt='attempt_3_cache_accounted' if cache_accounted else 'attempt_2'
    if target!=(root/'profiler_output/microbench/nvidia_h200/two_resident_cold_copy_20261108'/attempt).resolve() or target.exists():
        raise ValueError('new registered matching-protocol attempt required')
    if not tool.is_file() or not budget_key:raise ValueError('atomic shared reservation required')
    plan_file=CACHE_PLAN if cache_accounted else PLAN
    plan=json.loads(plan_file.read_text())
    if cache_accounted:
        proof=plan['independent_cache_check']
        for path_key,hash_key in (('path','sha256'),('manifest_path','manifest_sha256')):
            if hashlib.sha256((root/proof[path_key]).read_bytes()).hexdigest()!=proof[hash_key]:
                raise ValueError('independent cache evidence content address mismatch')
        if json.loads((root/proof['path']).read_text())['qualified'] is not True:
            raise ValueError('independent cache check must qualify before protocol2')
    target.mkdir(parents=True)
    meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__),plan_file,ANALYSIS)},plan_sha256=hashlib.sha256(plan_file.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    started=time.monotonic()
    try:
        out=capture.remote(plan);out.update(meta);out['client_wall_s']=time.monotonic()-started
        (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as exc:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n');raise
