"""Independent partial-cohort evidence for source-known single-panel classes."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-single-panel-partial-20261107')
SOURCE=HERE/'single_panel_partial.cu'
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-single-panel-partial-20261107')
# Config,panel bytes,A destination rows,row bytes,threads,shared reservation,residency.
CONFIGS=((0,16384,16,256,128,26624,8),(1,32768,16,512,128,40960,5),
         (2,32768,16,256,128,49152,4),(3,65536,32,256,128,98304,2),
         (4,65536,16,256,256,98304,2))
# Source-populated CTA counts, not case names or observed durations. Existing
# measured1272/1696points are retained in their old reference, not recaptured.
TARGETS={(0,4096):(768,2784,2976),(0,1536):(1024,3712,3968),
         (1,4096):(2112,2784),(1,1536):(2816,3712),
         (2,4096):(1548,2124),(2,1536):(2064,2832),
         (3,4096):(768,1086),(3,1536):(1024,1448),
         (4,4096):(744,774,2084),(4,1536):(992,1032),(4,1024):(4168,)}
SELECTION=('Frozen before capture: byte-only single-panel copy/consume with padded A destinations(one valid broadcast row),strided unique B and twoCTAbarriers/round. '
 'No matrix,routing,activation,operator output or production duration. Five physical classes retain panel/row/thread shape; shared reservation fixes2/4/5/8residentCTAsinstead of native register use. '
 'All eleven config/stride programs and all frozen source-populated grids retained. Each program adds full cohort W=132*residency and designated partial5W/4controls. '
 'Two independent63-sample processes at every grid; each launch evicts2*deviceL2bytes. No control duration subtraction or new body price. '
 'Require numerical checksum,132SMcoverage,CUDA occupancy,95%medianCI endpoints/process medians/clock64nominal within3%. '
 'For EACH program, partial upper95%CI must be<1.8*full-cohort lower95%CI,distinguishing capacity reclamation from an extra permanently reserved full wave. '
 'Matched counter at each designated partial grid:exactLDGSTS including zero-filled A and exactCTAs,TEXlogicalreadbytes and uniqueDRAMbytes within1%,TEX+fabric cover total within1%. '
 'All directSM/LTSclocksstable3%,SMnominal within3%. Class admission requires its contrast/counter/globalclock gates; a grid additionally requires both processes precision/clock/process checks. Failed points remain unbound without neighboring lookup. '
 'Eligible points may only admit the EXISTING coarse aggregate operand/matrix-shared/output envelope,using unchanged hardware rates/work. Measured body durations are not operator prices or new Forward rates; no arbitrary CTA order or Tensor release fraction.')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=420,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(status='started',selection=SELECTION,configurations=CONFIGS,
        programs=[dict(config=c,stride=s,target_grids=grids) for (c,s),grids in TARGETS.items()],cells=[],counters=[])
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
            for (config,stride),targets in TARGETS.items():
                full=132*CONFIGS[config][-1]
                for grid in sorted({full,full*5//4,*targets}):
                    item=run([str(TASK/'probe'),str(config),str(grid),str(stride),'63',str(process)],20)
                    out['cells'].append(dict(config=config,stride=stride,grid=grid,process=process,measurement=item))
                    if item['exit_code']:out['status']='measurement_failed';return out
        for config,stride in TARGETS:
            grid=132*CONFIGS[config][-1]*5//4
            item=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:partial_single_panel','--launch-skip','3','--launch-count','1','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',
                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_ltcfabric_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum',str(TASK/'probe'),str(config),str(grid),str(stride),'1','0'],25)
            out['counters'].append(dict(config=config,stride=stride,grid=grid,measurement=item))
            if item['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/microbench/nvidia_h200/single_panel_partial_20261107/attempt_1').resolve() or target.exists():raise ValueError('requires new registered attempt1in this worktree')
    start=time.monotonic();out=capture.remote();out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
