"""Independent mbarrier completion observer; no operator timing price."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'async_group_observer.cu'
TASK=Path('/tmp/midlayer-async-group-observer-20261108')
PHYSICAL=((16384,132),(16384,264),(32768,132))
CELLS=tuple((b,g,l) for b,g in PHYSICAL for l in (1,2,4))
METRICS=('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,'
         'lts__t_sectors_srcunit_ltcfabric_op_read.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,'
         'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum,'
         'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second')
SELECTION=('All nine private byte cells:16KiB/grid132or264/shared98304B,32KiB/grid132/shared196608B;'
 '256producerthreads+32independentobserverthreads,lookahead1/2/4,127rounds,six1024-roundwarmkernels.'
 'Each producerwarp marks every group with a separate mbarrier,32deferred noinc arrivals+1ordinary arrival;'
 'the independent observerwarp polls all events concurrently and timestamps completion during continued enqueue.'
 'Per-CTA first-panel/all-panel endpoints join all8producerwarps. Producer timestamps only bound the enqueue interval;'
 'no wait-return timestamp,posthoc enqueue subtraction or timing price. Interior CTA barriers separate independent bursts;'
 'this is not a steady native matrix recurrence. Two independent processes,second reversed,63samples/cell;18matched counters.'
 'Require exact numerical/resources/132SMplacement/CTA/LDGSTS,upstream/TEXlogical1%,DRAM<=1%,source-unitclosure1%,'
 'directSMnominal3%,allSM/LTSstability3%,median95%CIhalfwidth3%,processmedian3%.'
 'Observer resolution:each sample retains maximum poll interval over ALL observed events/CTAs/rounds;itsmedian95%upper'
 'must be<=10%of first-response median95%lower in EVERY cell. No weakened old enqueue gate:the producer wait observer'
 'has been replaced by independent hardware event completion,with added ninth warp/marker overhead explicitly scoped.'
 'Q=(first_L-first_1)/(all_L-first_1);all12contrasts95%upper<=.25supportfirstpriority;all95%lower>=.75supportequalburstslowresponse;'
 'otherwise inconclusive. Allcells/gatesrequired,no selectedcell,rate,Qparameter or Forward policy adoption.'
 'Any future production use still requires compact physical request/response conservation and independent source transfer;'
 'mbarrier instrumentation is not native warpgroup timing or an instruction to revive rejected FIFO/fluid schedules.')
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
 .apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)))
APP=modal.App('midlayer-async-group-observer-20261108')

def run(command,limit):
    start=time.monotonic()
    try:
        p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
        return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)
    except subprocess.TimeoutExpired as exc:
        def decoded(x):
            return x.decode(errors='replace') if isinstance(x,bytes) else '' if x is None else x
        return dict(command=command,exit_code=124,stdout=decoded(exc.stdout),stderr=decoded(exc.stderr),wall_s=time.monotonic()-start,timed_out=True)

@APP.function(gpu='H200',image=IMAGE,timeout=450,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(status='started',selection=SELECTION,physical_cells=CELLS,timing=[],counters=[])
    try:
        for name,cmd,limit in (
          ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
          ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
          ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
          ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:
                out['status']=name+'_failed';return out
        if ('NVIDIA H200' not in out['gpu']['stdout'] or 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout']
            or 'ARRIVES.LDGSTSBAR' not in out['sass']['stdout'] or 'SYNCS.PHASECHK' not in out['sass']['stdout']
            or 'HGMMA.' in out['sass']['stdout'] or 'HMMA.' in out['sass']['stdout']):
            out['status']='identity_or_instruction_boundary_failed';return out
        out['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        for process in (0,1):
            ordered=CELLS if process==0 else tuple(reversed(CELLS))
            for b,g,l in ordered:
                item=run([str(TASK/'probe'),str(b),str(g),str(l),'63'],20)
                out['timing'].append(dict(process=process,B_bytes=b,grid=g,lookahead=l,measurement=item))
                if item['exit_code']:
                    out['status']='timing_failed';return out
            print(f'completed process {process} timing',flush=True)
        for process in (0,1):
            for b,g,l in (CELLS if process==0 else tuple(reversed(CELLS))):
                if time.monotonic()-start>390:
                    out['status']='bounded_partial';return out
                item=run(['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application',
                    '--cache-control','none','--clock-control','none','--kernel-name','regex:group_observer',
                    '--launch-skip','6','--launch-count','1','--metrics',METRICS,str(TASK/'probe'),str(b),str(g),str(l),'1'],25)
                out['counters'].append(dict(process=process,B_bytes=b,grid=g,lookahead=l,measurement=item))
                if item['exit_code']:
                    out['status']='counter_failed';return out
                print(f'completed {len(out["counters"])}/18 counters',flush=True)
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()

@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/microbench/nvidia_h200/async_group_observer_20261108/attempt_1').resolve() or target.exists():
        raise ValueError('requires registered new attempt in this worktree')
    tool=Path(budget_tool)
    if not tool.is_file() or not budget_key:
        raise ValueError('shared budget reservation and local accounting tool required')
    target.mkdir(parents=True)
    meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))})
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    print(json.dumps(dict(app_id=APP.app_id,budget_key=budget_key)),flush=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    started=time.monotonic()
    try:
        out=capture.remote();out.update(meta);out['client_wall_s']=time.monotonic()-started
        (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as exc:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(exc).__name__,error=str(exc),client_wall_s=time.monotonic()-started),indent=2)+'\n')
        raise
