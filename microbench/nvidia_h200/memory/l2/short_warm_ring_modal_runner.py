"""Inline-prewarmed short byte-ring validation; no new service selection."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-short-warm-ring-20261107')
SOURCE=HERE/'short_warm_ring.cu'
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-short-warm-ring-20261107')
SELECTION=('Frozen before capture: independent48KiB/four-slot byte ring,256threads,196608Bshared,132cooperativeCTAs,exactoneCTA/SM. '
 'No matrix,routing,operand indexing,operator output or production duration. Every launch evicts2*deviceL2bytes, then the SAME kernel prewarms all four private panels and synchronizes the grid before the timed body. '
 'Two processes,63samples at rounds8,12,16,32,64; zero-round matched counter control has identical inline prewarm/entrance. All conditions retained. '
 'Numerical checksum and exact SM placement every launch. Median95%distribution-free CI endpoints/process medians within3%; weighted clock64 at nominal1.98GHz within3%. '
 'Counters at rounds0,8,12,16,32,64: exactLDGSTS132*(4+rounds)*49152/512 and132CTAs; netTEXreads minus zero within1%of132*rounds*49152,netDRAM<=1%,TEX+fabric covers total within1%ofbodybytes. '
 'All directSM/LTSclock counters stable within3%and SM nominal within3%. Netprewarm/DRAM subtraction is ONLY for counters,never measured elapsed time. '
 'Existing independently published7.608679884194TB/s capability is unchanged. At EACH process/round, the lower95%CI for full-chip body envelope must be>=97%ofbodybytes/existingCapacity. '
 'This only tests extending that existing approximate competing frontend bound down to8rounds. Any failed cell leaves extension unbound; no new rate,per-round LUT,threshold change or production-error selection.')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(status='started',selection=SELECTION,cells=[],counters=[])
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('nvcc',['nvcc','--version'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout']:out['status']='instruction_scope_failed';return out
        for process in range(2):
            for rounds in (8,12,16,32,64):
                item=run([str(TASK/'probe'),str(rounds),'63',str(process)],20)
                out['cells'].append(dict(process=process,rounds=rounds,measurement=item))
                if item['exit_code']:out['status']='measurement_failed';return out
        for rounds in (0,8,12,16,32,64):
            item=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:short_warm_ring','--launch-skip','3','--launch-count','1','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',
                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_ltcfabric_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum',str(TASK/'probe'),str(rounds),'1','0'],25)
            out['counters'].append(dict(rounds=rounds,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/microbench/nvidia_h200/short_warm_ring_20261107/attempt_1').resolve() or target.exists():raise ValueError('requires new registered attempt1in this worktree')
    start=time.monotonic();out=capture.remote();out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
