"""Independent counter-only hot request merging; no operator or timing rate."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'hot_request_merge.cu'
TASK=Path('/tmp/midlayer-hot-request-merge-20261108')
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
       .apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)))
APP=modal.App('midlayer-hot-request-merge-20261108')
PROGRAMS=((16384,3,12288,2),(16384,3,32768,2),(32768,4,1536,1),(32768,4,4096,1))
CELLS=tuple((b,slots,stride,132*resident*waves,f,skew)
            for b,slots,stride,resident in PROGRAMS for waves in (1,2)
            for f in (1,2,4,8,16) for skew in (0,512))
METRICS=('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,'
         'lts__t_sectors_srcunit_ltcfabric_op_read.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,'
         'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum,'
         'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second')
SELECTION=('Frozen counter-only independent byte program: B16KiB/3slots or B32KiB/4slots,zero-filled16KiB A,256threads, '
           '98304/196608Bshared and2/1residentCTA. Four private warm B column panels per fanout group,257rounds,two copy groups. '
           'No matrix,routing,activation,operator output or production duration. All80physical cells:four row-stride programs, '
           'one/two fully resident waves,fanout1/2/4/8/16,andinitial relative phases0/512SMcyclesacrosssharingCTAs. '
           'Two independent process rotations,second order reversed;160application-replay captures,no timing price. '
           'Every cell requires all-lane B/zero-Achecksum,132SMcoverage,exactresources/CTA/LDGSTS,upstream logical-byte closure1%, '
           'DRAM<=1%logical andTEX/source-unit closure1%. DirectSM/LTSclocksstable3%,SMnominal3%. '
           'For each program/fanout the downstream TEX/upstream ratio must agree within3%acrossbothwaves,phases,andprocesses; '
           'fanout1must conserveTEX/logicalwithin1%. Only if ALLprogram gates pass may the largestremaining-request ratio '
           'acrossits eightconditions represent a conservative independent coalescing fraction. Failed classes remain unbound; '
           'no relaxation,production-error selection,point borrowing,new bandwidth or nativephasecertification.')


def run(command,limit):
    start=time.monotonic()
    p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)


@APP.function(gpu='H200',image=IMAGE,timeout=570,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic()
    out=dict(status='started',selection=SELECTION,physical_cells=CELLS,cells=[])
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],60),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],15),
            ('sass',['cuobjdump','--dump-sass','probe'],15)):
            out[name]=run(cmd,limit)
            if out[name]['exit_code']:
                out['status']=name+'_failed'
                return out
        if 'LDGSTS.E.BYPASS.128' not in out['sass']['stdout'] or 'HGMMA.' in out['sass']['stdout'] or 'HMMA.' in out['sass']['stdout']:
            out['status']='instruction_boundary_failed'
            return out
        for process in (0,1):
            for b,slots,stride,grid,fanout,skew in (CELLS if process==0 else tuple(reversed(CELLS))):
                # Return partial evidence before the remote570s hard timeout.
                if time.monotonic()-start>500:
                    out['status']='budget_stop_partial'
                    return out
                cmd=['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application',
                     '--cache-control','none','--clock-control','none','--kernel-name','regex:hot_requests',
                     '--launch-skip','3','--launch-count','1','--metrics',METRICS,str(TASK/'probe'),
                     str(b),str(stride),str(grid),str(fanout),str(skew)]
                item=run(cmd,20)
                out['cells'].append(dict(process=process,B_bytes=b,slots=slots,stride=stride,grid=grid,fanout=fanout,skew_cycles=skew,measurement=item))
                if item['exit_code']:
                    out['status']='counter_failed'
                    return out
                if len(out['cells'])%20==0:
                    print(f'completed {len(out["cells"])}/160 counter cells',flush=True)
        out['status']='complete'
        return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3]
    target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/microbench/nvidia_h200/hot_request_merge_20261108/attempt_1').resolve() or target.exists():
        raise ValueError('requires new registered attempt in this worktree')
    start=time.monotonic()
    out=capture.remote()
    out['client_wall_s']=time.monotonic()-start
    out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True)
    (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
