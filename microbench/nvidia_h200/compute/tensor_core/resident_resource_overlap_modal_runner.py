"""Independent co-resident byte/matrix resource coexistence, never an operator."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-resident-resource-overlap-20261107')
SOURCE=HERE/'resident_resource_overlap.cu' if HERE.name=='tensor_core' else TASK/'resident_resource_overlap.cu'
GENERATOR=HERE/'wgmma_ss_n128_shortk_issue_gen.py' if HERE.name=='tensor_core' else TASK/'wgmma_ss_n128_shortk_issue_gen.py'
REFERENCE=HERE.parents[1]/'operators/flash_attn/fa3_wgmma_service.cu' if HERE.name=='tensor_core' else TASK/'fa3_wgmma_service.cu'
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9')
for source in (SOURCE,GENERATOR,REFERENCE):
    IMAGE=IMAGE.add_local_file(source,remote_path=str(TASK/source.name))
APP=modal.App('midlayer-resident-resource-overlap-20261107')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture(matrix_pause_ns:int,clock_domains:bool):
    if matrix_pause_ns not in (0,1024):raise ValueError('unknown physical duty contrast')
    start=time.monotonic()
    out=dict(cells=[],counters=[],boundary='Each CTA is pure hot-byte-copy OR pure shared-resident WGMMA; never a combined GEMM CTA, routing or operator output',
        selection='Frozen causal coexistence test, no Forward rate. One identical256-thread kernel reserves98304dynamic shared bytes; occupancy exactly2CTAs/SM. Modes0/1 have264all-copy/all-matrix CTAs; mode2has264CTAs and per-SM atomic role assignment, requiring exactly1copy+1matrix on every SM; modes3/4have132single-copy/single-matrix CTAs. Rounds256/1024,two63-sample processes for every cell. Keep all CTA timelines, actual SMIDs, role/entry indices and clocks. Every point requires95%distribution-free median CI endpoints and process medians within3%. For each recurrence, the mixed copy-role median CTA span divided by single-copy median CTA span must be<1.5 to distinguish permanent half-capacity reservation; copy-role clocks must agree within3%and remain within3%of1980MHz. Each mixed sample must have median per-SM overlap/min(role-span)>=75%; same-SM1+1coexistence checked at capture. No replacement SM/matrix/memory clock or price. Matrix roles intentionally repeat identical descriptors and do not replay a production K layout. Matched mixed counter:DRAM<=1%copy logical bytes,L2/read payload within1%,exact64*132*1024LDGSTSwarp instructions. Four contiguous HGMMA per group and wait1 must survive compilation. Failed checks cannot change gates or choose another mode, recurrence or process. This only reopens work-conserving resource pooling if coupled evidence passes; it does not establish a native CTA permutation or complete production price.')
    out['matrix_pause_ns']=matrix_pause_ns
    out['timeline_scope']='Thread0 internal window;matrix endpoint drains its own warpgroup, not certified final CTA retirement. Copy endpoint has a full CTA join.'
    if matrix_pause_ns:
        out['boundary']='One pure byte CTA plus one shared-resident matrix/explicit-idle CTA per SM; no combined GEMM or operator'
        out['selection']+=' Independent low-duty contrast:matrix roles execute a fixed1024-ns nanosleep after every four-instruction group, chosen before capture to limit power interference. Copy roles are unchanged. All original precision/process/clock/overlap/counter gates remain; no new clock or sleep service is published. This tests capacity of a byte role alongside a physically occupied CTA with matrix work and explicit idle periods; no claim of native matrix issue schedule.'
    if clock_domains:
        out['boundary']='Hardware SM/LTS clock counters of independent byte/matrix roles; no production clock or duration'
        out['selection']='Frozen clock-domain attribution, separate from failed coexistence adoption. All16cells:two process rotations,mode2mixed/mode3single-copy,rounds256/1024,matrix-pause0/1024ns. Query direct sm__cycles_elapsed.avg.per_second and lts__cycles_elapsed.avg.per_second, plus read bytes and exact LDGSTS. The first JSON trial0 is the profiled third kernel; compare its role-specific clock64/globaltimer ratios to direct SM/LTS counters, retain all cells. No cell selected by production error. A reference LTS clock may be the median of ALL16cells only if all are complete, direct LTS frequencies agree within3%,all read-traffic/instruction gates pass,and the contrast independently distinguishes direct SM frequency or clock64 effective progress from LTS frequency. Distinction:at each exact recurrence/pause/rotation pair,clock64copy ratio or direct SM-frequency ratio changes by>=5%while LTS ratio stays within3%. Otherwise no parameter. These clock64 differences are not replacement SM clocks. No failed coexistence gate is waived and no new memory rate or native CTA policy is published.'
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20),
            ('nvcc',['nvcc','--version'],20),
            ('generate',['python',GENERATOR.name,REFERENCE.name,'wgmma_ss_helper.cuh'],20),
            ('build',['nvcc','-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v',SOURCE.name,'-o','probe'],80),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],20),
            ('sass',['cuobjdump','--dump-sass','probe'],20)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        sass=out['sass']['stdout'];matrix=[line for line in sass.splitlines() if 'HGMMA.' in line]
        if len(matrix)!=4 or 'WARPGROUP.DEPBAR.LE gsb0, 0x1' not in sass or 'LDGSTS.E.BYPASS.128' not in sass:
            out['status']='instruction_scope_failed';return out
        start_sass=sass.index(matrix[0]);end_sass=sass.index(matrix[-1])
        if 'WARPGROUP.DEPBAR' in sass[start_sass:end_sass]:out['status']='serialized_matrix_group';return out
        if clock_domains:
            out['clock_counter_cells']=[]
            for rotation in range(2):
                for rounds in (256,1024):
                    for pause in (0,1024):
                        for mode in (2,3):
                            counter=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:resource_overlap','--launch-skip','2','--launch-count','1',
                                '--cache-control','none','--clock-control','none','--metrics',
                                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum',
                                str(TASK/'probe'),str(mode),str(rounds),'3',str(rotation),str(pause)],45)
                            out['clock_counter_cells'].append(dict(rotation=rotation,rounds=rounds,mode=mode,matrix_pause_ns=pause,measurement=counter))
                            if counter['exit_code']:out['status']='clock_counter_failed';return out
            out['status']='complete';return out
        for rotation in range(2):
            for rounds in (256,1024):
                for mode in range(5):
                    cell=run([str(TASK/'probe'),str(mode),str(rounds),'63',str(rotation),str(matrix_pause_ns)],40)
                    out['cells'].append(dict(rotation=rotation,mode=mode,rounds=rounds,measurement=cell))
                    if cell['exit_code']:out['status']='measurement_failed';return out
        counter=run(['ncu','--csv','--page','raw','--kernel-name','regex:resource_overlap','--launch-skip','2','--launch-count','1',
            '--cache-control','none','--clock-control','none','--metrics',
            'lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum',str(TASK/'probe'),'2','1024','3','0',str(matrix_pause_ns)],80)
        out['counters'].append(counter)
        out['status']='complete' if counter['exit_code']==0 else 'counter_failed'
        return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str,matrix_pause_ns:int=0,clock_domains:bool=False):
    if matrix_pause_ns not in (0,1024):raise ValueError('unknown physical duty contrast')
    root=HERE.parents[3];target=Path(output_dir).resolve()
    attempts=('attempt_4',) if clock_domains else ('attempt_1','attempt_2') if matrix_pause_ns==0 else ('attempt_3',)
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/resident_resource_overlap_20261107').resolve() or target.name not in attempts or target.exists():
        raise ValueError('output must be a new registered attempt in this worktree')
    start=time.monotonic();out=capture.remote(matrix_pause_ns,clock_domains);out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,GENERATOR,REFERENCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
