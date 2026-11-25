"""Frozen copy/copy sharing discrimination; no production-duration inputs."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-resident-copy-arbitration-20261107')
SOURCE=HERE/'resident_copy_arbitration.cu' if HERE.name=='l2' else TASK/'resident_copy_arbitration.cu'
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-resident-copy-arbitration-20261107')
SELECTION=('Predeclared byte-only sharing test, no production price. Cooperative grid264,256threads,98304B shared,exact2CTAs per each132SMs. '
 'Modes0:32+96KiB;1:32+dummy;2:dummy+96;3:64+64;4:64+dummy. Physical entry roles rotate in two separate processes;128/256bursts,63launch samples per cell. '
 'Grid barriers and a4096ns deadline align independent bursts; they impose a measurement protocol,not a production K barrier. '
 'Every launch retains each CTA sum/min/max latency,mean paired start/finish skew,first/last burst stamps,and outer clock64/globaltimer. No all-burst trace claimed. '
 'All20cells required. Each role median95%distribution-free CI and process median variation<=3%;mean start skew<=10%ofsolo96-solo32;'
 'direct LTS clock spread<=3%;matched5counter cells:read payloadwithin1%,DRAM<=1%,exact LDGSTS bytes/512 and264CTA launches. '
 'Freeze Q=(paired32-solo32)/(solo96-solo32),requiring positive denominator. Ideal fluid sharingQ=0.5;whole-burst atomic FIFO Q=0(shortfirst)or1.5(longfirst). '
 'Fluid admission requires every process/depth Q in[0.25,0.75],and paired64 mean finish-skew/solo64<=0.25. '
 'FIFO admission requires every Q in[-0.25,0.25]or[1.25,1.75],and paired64 normalized finish-skew>=0.75. '
 'Otherwise arbitration unbound;failed gates do not authorize another policy. These coarse policies are falsifiable approximations,not native instruction schedules or new Forward durations.')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(cells=[],counters=[],selection=SELECTION+' Protocol2 amendment, frozen before second capture: role rotation changes physical arbitration position and is a causal contrast, not a repeated process. Keep both rotations, with two independent process repetitions per fixed rotation/depth/mode; apply the unchanged3%process gate within those identical conditions. Require ALL8 fluid/FIFO contrasts, with the original intervals. Add mode5(all dummy) matched counter control for cooperative barrier/timestamp metadata; gate net L2 read bytes after that control within1%ofcopy payload, with exact LDGSTS and CTA counts in every cell including zero-copy. Keep original DRAM1%,clocks3%,alignment and precision rules. No elapsed time is subtracted or published as a Forward price; failed protocol1 remains unbound.',protocol_version=2,status='started')
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('nvcc',['nvcc','--version'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        if 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout']:
            out['status']='instruction_scope_failed';return out
        for process in range(2):
            for rotation in range(2):
                for rounds in (128,256):
                    for mode in range(5):
                        cell=run([str(TASK/'probe'),str(mode),str(rounds),'63',str(rotation)],20)
                        out['cells'].append(dict(mode=mode,rounds=rounds,rotation=rotation,process=process,measurement=cell))
                        if cell['exit_code']:out['status']='measurement_failed';return out
        for mode in range(6):
            counter=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:copy_arbitration','--launch-skip','2','--launch-count','1','--cache-control','none','--clock-control','none','--metrics',
                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum',str(TASK/'probe'),str(mode),'256','3','0'],25)
            out['counters'].append(dict(mode=mode,measurement=counter))
            if counter['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/resident_copy_arbitration_20261107').resolve() or target.name!='attempt_2' or target.exists():
        raise ValueError('output must be a new registered attempt in this worktree')
    start=time.monotonic();out=capture.remote();out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
