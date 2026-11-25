"""Fill physical/grid holes of the existing independent cold-request probe."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'finite_strided_panel.cu'
TASK=Path('/tmp/midlayer-coalesced-scope-20261108')
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-coalesced-scope-20261108')
# Stride bytes, physical grid, simultaneous B-request fanout. These are all
# missing nominal-clock source endpoints plus designated1/2fanout controls.
CELLS=((1024,6624,1),(1024,14304,2),(1024,14304,4),(1024,24544,1),
       (1024,24544,2),(1024,24544,4),(1024,24544,8),
       (1536,3064,1),(1536,3064,2),(1536,3064,4),
       (4096,1146,1),(4096,2298,2),(4096,2298,4),(4096,3312,1),
       (4096,3834,4),(4096,3834,8),(4096,6906,4),(4096,6906,8),
       (4096,7152,2),(4096,7152,4),(4096,12272,4),(4096,12272,8),
       (4096,13050,8),(4096,13050,16),(4096,25338,1),(4096,25338,2),
       (4096,25338,8),(4096,25338,16))
COUNTERS=tuple((s,max(g for stride,g,f in CELLS if stride==s),f) for s in (1024,1536,4096) for f in (1,2))
SELECTION=('Frozen before capture: unchanged independent cold B-request byte ring,32768B B plus padded128-row A destination/one valid broadcast row,4slots,256threads,196608Bshared/one residentCTA. '
 'No matrix,routing,activation,operator output or production durations. All28physical(stride,grid,fanout)points are missing nominal-clock source endpoints plus designated fanout1/2controls; all retained. '
 'Same scalar request program as accepted finite_strided_panel attempt12;32GiB backing only raises source bounds for25338CTAs and no adjacent grid is borrowed. '
 'Two independent63-sample process sweeps,each launch evicts128MiB. Selection is slower process median complete COPY body,never copy-minus-control. '
 'Every point requires95%medianCI/process/nominalclock3%,exact resource identity and132SMcoverage. Six largest-grid/stride fanout1/2counters require uniqueDRAMbytes/TEXpayload/source-unit closure1%,exactLDGSTSincluding zero-filled A/exactCTA. '
 'All directSM/LTSclocksstable3%,SMnominalwithin3%. Paired fanout1/2progress is reported as a regime diagnostic,not an added eligibility threshold; the existing independent pending-request evidence remains unchanged. '
 'Only qualified exact points can extend the EXISTING adjacent-fanout compulsory-B competing floor; existing rates/points remain untouched. No production-error selection,new interpolation policy,generic bandwidth replacement,CTA permutation or operator body price. Failed points/classes remain unbound.')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

def cell_file(path,cells):
    path.write_text(''.join(f'{s},{g},{f}\n' for s,g,f in cells))

@APP.function(gpu='H200',image=IMAGE,timeout=570,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(status='started',selection=SELECTION,physical_cells=CELLS,counter_cells=COUNTERS,rotations=[],counters=[])
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('compiler',['nvcc','--version'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-DMIDLAYER_COALESCED_SCOPE_ONLY','-DMIDLAYER_FINITE_SOURCE_GIB=32','-DMIDLAYER_FINITE_MAX_GRID=25338','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:out['status']=name+'_failed';return out
        sass=out['sass']['stdout']
        if 'LDGSTS.E.BYPASS.128' not in sass or 'HGMMA.' in sass or 'HMMA.' in sass:out['status']='instruction_scope_failed';return out
        cell_file(TASK/'cells.csv',CELLS)
        for process in range(2):
            item=run([str(TASK/'probe'),'44',str(process),str(TASK/'cells.csv'),'63'],180)
            out['rotations'].append(dict(process=process,measurement=item))
            if item['exit_code']:out['status']='measurement_failed';return out
        for stride,grid,fanout in COUNTERS:
            cell_file(TASK/'counter.csv',((stride,grid,fanout),))
            item=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:copy_panels','--launch-skip','3','--launch-count','1','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',
                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_ltcfabric_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum',str(TASK/'probe'),'44','0',str(TASK/'counter.csv'),'1'],30)
            out['counters'].append(dict(stride=stride,grid=grid,fanout=fanout,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/microbench/nvidia_h200/coalesced_scope_20261108/attempt_1').resolve() or target.exists():raise ValueError('requires new registered attempt1in this worktree')
    start=time.monotonic();out=capture.remote();out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
