"""Independent hot/cold burst all-return experiment; no operator pricing."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-resident-burst-join-20261107')
SOURCE=HERE/'resident_burst_join.cu' if HERE.name=='l2' else TASK/'resident_burst_join.cu'
IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12').apt_install('cuda-nsight-compute-12-9').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-resident-burst-join-20261107')
SELECTION=('Frozen before capture: pure-byte cp.async bursts with wait0 and CTA join, no matrix/routing/operator recurrence/output. '
 'Cooperative264CTAs/132SMs,256threads,98304B shared,exact2resident CTAs/SM. Each burst has H=4096 or16384hot bytes and1024companion bytes. '
 'Modes0:allH+1024hot;1:Hhot+1024cold;2:allH+1024cold;3:only1024cold;4:zero-copy metadata counter control. '
 'Cold addresses are unique perCTA/burst; before EVERY launch evict with writes to2*deviceL2bytes then explicitly warm all hot addresses with ld.global.cg. '
 'Two independent processes,256synchronized bursts/launch,63launch samples/condition,all16cells retained. '
 'PerCTA latency sum/min/max plus first/last stamps and outer clock64/globaltimer; no full internal trace claimed. '
 'Every cell median95%distribution-free CI and fixed-condition process medians within3%. '
 'Application replay repeats eviction/warm for every counter pass. Matched counters for all eight nonzero conditions plus zero:H200 directSM/LTS clocks within3%,exactLDGSTS logicalbytes/512,exact264CTAs. '
 'NetL2read after zero-copy metadata within1%ofcopy logical bytes; mixed/allcold DRAM within1%ofcold logicalbytes; allhot DRAM<=1%oflogicalbytes. '
 'At BOTH sizes/processes mixed lower95%median bound must be>=97%ofmaximum(allhot lowerbound,cold-only lowerbound), testing an all-return join. '
 'The designated4096B contrast distinguishes a mean-only latency interpolation: p=1024/5120; D=(mixed-hot)/(allcold-hot),positive denominator. '
 'Its conservative95%D lowerbound=(mixedLower-hotUpper)/(allcoldUpper-hotLower)must>=2*p in BOTH processes. '
 '16384B is a predeclared bandwidth-hiding contrast,not an alternate selected point. Failure means unbound; no new price/rate/latency or threshold selected by production error.')

def run(command,limit):
    start=time.monotonic();p=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)

@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic();out=dict(cells=[],counters=[],selection=SELECTION+' Counter protocol2: retain first attempt failure. Total L2 read sectors include TEX and LTC-fabric origins. Query both source units; the unchanged1%issued-payload gate applies to TEX minus zero-copy TEX metadata. Require TEX+LTC-fabric to explain total read within1%oflogicalcopybytes at all nonzero cells. Every other original gate and physical point is unchanged; all16normal cells and9counter cells are repeated. No fabric price or inferred production traffic multiplier. Execution protocol3 removes the repeated grid barriers and global polling between bursts, retaining one synchronized entrance and every CTA-local wait0/join. This eliminates duration-dependent observer reads from the burst loop. No collective burst phase is imposed. Zero-copy metadata control spins only on globaltimer for512ns per recurrence, keeping its clock window measurable without global loads; its elapsed time is never subtracted. Original16cells,9counters,precision/process/clock/byte gates and4KiB Q>=0.4 criterion all remain unchanged. Earlier two attempts remain unbound.',status='started',counter_protocol=2,execution_protocol=3)
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
            for hot in (4096,16384):
                for mode in range(4):
                    cell=run([str(TASK/'probe'),str(mode),str(hot),'256','63',str(process)],25)
                    out['cells'].append(dict(mode=mode,hot_bytes=hot,process=process,measurement=cell))
                    if cell['exit_code']:out['status']='measurement_failed';return out
        conditions=[(m,h) for h in (4096,16384) for m in range(4)]+[(4,4096)]
        for mode,hot in conditions:
            counter=run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:burst_join','--launch-skip','2','--launch-count','1','--replay-mode','application','--cache-control','none','--clock-control','none','--metrics',
                'sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,lts__t_sectors_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_ltcfabric_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum',str(TASK/'probe'),str(mode),str(hot),'256','3','0'],25)
            out['counters'].append(dict(mode=mode,hot_bytes=hot,measurement=counter))
            if counter['exit_code']:out['status']='counter_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/resident_burst_join_20261107').resolve() or target.name!='attempt_3' or target.exists():
        raise ValueError('output must be a new registered attempt in this worktree')
    start=time.monotonic();out=capture.remote();out['client_wall_s']=time.monotonic()-start;out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
