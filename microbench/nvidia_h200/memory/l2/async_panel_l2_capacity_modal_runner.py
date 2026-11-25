"""Pure async-byte-copy L2 capacity and counter attribution, not a GEMM."""
import hashlib,json,shutil,subprocess,time,re
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-async-panel-l2-capacity-20261107')
SOURCE=HERE/'async_panel_l2_capacity.cu' if HERE.name=='l2' else TASK/'async_panel_l2_capacity.cu'
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    .apt_install('cuda-nsight-compute-12-6').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)))
APP=modal.App('midlayer-async-panel-l2-capacity-20261107')
def run(cmd,limit):
    t=time.monotonic();p=subprocess.run(cmd,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-t)
@APP.function(gpu='H200',image=IMAGE,timeout=540,retries=0,max_containers=1,single_use_containers=True)
def capture(experiment: str):
    if experiment not in ("capacity", "entropy"):raise ValueError("unknown independent experiment")
    t=time.monotonic();result={'boundary':'isolated_async_ingress_capacity_no_matrix_routing_or_operator_output','cells':[],'counters':[],
        'selection':'Frozen isolated ingress:2separate16KiB A/B copy groups,256threads,3slots,98304shared bytes and2residentCTAs/SM. Two63-sample processes per physical cell; private four-panel A/B cache footprint, row128bytes and strides512/4096/5760/12288/32768; grids132/264/528,rounds256/1024, consumer0/1. Consumer0 does only copies/synchronization in timing; final shared checksum is after timer. Consumer1 is causal screen, never subtracted. Publish solely Consumer0/G264/R1024:slower process median logical-byte rate at each of the five independently measured row strides; publish every qualified stride, never select by production error, no alternative grid/round fallback. Each published stride must also have G264/G528 plateau capacities within3%atR1024. Each selected point:95%distribution-free median CI endpoints within3%,process medians within3%,SM clock within3%of1980MHz,CUDA occupancy exactly2CTA/SM; matching counter DRAM<=1%logical,L2logical bytes within1%,exact64*G*R issuedLDGSTSwarp instructions. Preserve all failed/unselected points. No production duration/error parameter selection, no complete GEMM or MoE.'}
    if experiment == 'entropy':
        result['boundary']='identical_async_byte_kernel_with_input_entropy_contrast'
        result['selection']='Frozen causal screen only; no Forward rate. Identical copy kernel, no timed consumer, no matrix/routing/operator output. Data patterns:0hash32,1approximately normal BF16 from12uniform bytes,2constant packed BF16 one. Strides512/12288,grids132/264/528,rounds256/1024,two63-sample processes. Report all points; designated contrastG264/R1024 at both strides uses slower process median. Require95%medianCI endpoints/process medians/nominal1980MHz within3%,G264/G528 rate plateau3%,DRAM<=1%logical,L2/logical within1%,exact64*G*RLDGSTS. Failed points retained; never borrow another point or select a pattern by production error. No service published without a separate source/data applicability contract.'
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20),
            ('nvcc',['nvcc','--version'],20),('ncu',['ncu','--version'],20),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],80),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],20),
            ('sass',['cuobjdump','--dump-sass','probe'],30)):
            result[name]=run(cmd,limit)
            if result[name]['exit_code']:result['status']=name+'_failed';return result
        sass=result['sass']['stdout']
        if 'LDGSTS.E.BYPASS.128' not in sass or 'HGMMA.' in sass or 'HMMA.' in sass:
            result['status']='sass_boundary_failed';return result
        windows=[]
        for chunk in sass.split('Function :')[1:]:
            name=chunk.splitlines()[0].strip()
            if 'copy_capacity' not in name or 'Lb0E' not in name:
                continue
            lines=chunk.splitlines()
            timers=[i for i,line in enumerate(lines) if 'GLOBALTIMER' in line]
            loads=[i for i,line in enumerate(lines) if re.search(r'\bLDS(?:\.|\s)',line) and '@!PT' not in line]
            if len(timers)!=2 or any(timers[0]<i<timers[1] for i in loads) or not any(i>timers[1] for i in loads):
                result['status']='consumer_window_failed';return result
            windows.append(name)
        result['no_consumer_timing_windows']=windows
        if len(windows)!=5:result['status']='consumer_scope_failed';return result
        strides=(512,4096,5760,12288,32768) if experiment=='capacity' else (512,12288)
        choices=(0,1) if experiment=='capacity' else (0,1,2)
        for rotation in range(2):
            for stride in strides:
                for grid in (132,264,528):
                    for rounds in (256,1024):
                        for choice in choices:
                            cell=run([str(TASK/'probe'),experiment,str(grid),str(stride),str(rounds),str(choice),'63'],35)
                            result['cells'].append(dict(rotation=rotation,measurement=cell))
                            if cell['exit_code']:result['status']='measurement_failed';return result
        counter_choices=(0,) if experiment=='capacity' else (0,1,2)
        for stride in strides:
            for choice in counter_choices:
                counter=run(['ncu','--csv','--page','raw','--cache-control','none','--clock-control','none',
                    '--kernel-name','regex:copy_capacity','--launch-skip','2','--launch-count','1','--metrics',
                    'lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum',str(TASK/'probe'),experiment,'264',str(stride),'1024',str(choice),'3'],75)
                result['counters'].append(dict(row_stride_bytes=stride,pattern=choice,measurement=counter))
                if counter['exit_code']:result['status']='counter_failed';return result
        result['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        result['status']='complete'
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-t
        shutil.rmtree(TASK)
@APP.local_entrypoint()
def main(output_dir:str, experiment:str="capacity"):
    if experiment not in ("capacity","entropy"):raise ValueError("unknown experiment")
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('capture must stay inside the assigned worktree')
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/async_panel_l2_capacity_20261107').resolve() or target.name!=('attempt_2' if experiment=='capacity' else 'attempt_3') or target.exists():
        raise ValueError('output must be a new registered attempt matching experiment')
    t=time.monotonic();result=capture.remote(experiment);result['client_wall_s']=time.monotonic()-t;result['app_id']=APP.app_id
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'app_id':result['app_id'],'gpu_function_wall_s':result['gpu_function_wall_s']}))
