"""Capture finite byte-request progress without a production operator."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'finite_strided_panel.cu'
TASK=Path('/tmp/midlayer-finite-strided-panel-20261107')
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
       .apt_install('cuda-nsight-compute-12-6')
       .add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)))
APP=modal.App('midlayer-finite-strided-panel-20261107')

def run(command,timeout):
    start=time.monotonic()
    result=subprocess.run(command,cwd=TASK,capture_output=True,text=True,timeout=timeout)
    return {'command':command,'exit_code':result.returncode,'stdout':result.stdout,
            'stderr':result.stderr,'wall_s':time.monotonic()-start}

@APP.function(gpu='H200',image=IMAGE,timeout=300,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic()
    result={'scope':'independent finite async byte requests; no matrix/routing/operator output',
            'rotations':[],'counters':[],
            'frozen_checks':'Independent same-address cold-request fanout. B32768bytes plus padded128-row A(one valid broadcast row),4slots,256threads,196608shared bytes/one resident CTA. K strides1024/1536/4096, grids132/1530/2040/4592/9184, request fanout1/2/4/8/16 adjacent CTAs use the same cold B strip. All sources evicted before each measurement. No MMA/routing/operator output. Two63-sample process rotations. Slower process complete-copy median;95%median CI/process/nominalclock3%; no cell dropping or borrowing failed endpoints. Six exact LDGSTS counts and DRAM payload within1%of ceil(grid/fanout)*Bpanel*rounds plus one broadcast A row. Captured byte capacities can only price compulsory simultaneous B request work under matching topology; true cache-hit replays remain free in the optimistic floor. Fanout interpolation is only across adjacent independently measured fanouts at exact measured grid/row signature. No duration consumption from production, generic bandwidth replacement, pure barrier fee or cache-traffic modification.'}
    try:
        for name,command,timeout in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20),
            ('compiler',['nvcc','--version'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],90),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],20),
            ('sass',['cuobjdump','--dump-sass','probe'],30)):
            result[name]=run(command,timeout)
            if result[name]['exit_code']!=0:
                result['status']=name+'_failed'
                return result
        sass=result['sass']['stdout']
        if 'LDGSTS.E.BYPASS.128' not in sass or 'HGMMA.' in sass or 'HMMA.' in sass:
            result['status']='instruction_boundary_failed'
            return result
        for profile_code in (37,):
            for rotation in (0,1):
                sample=run([str(TASK/'probe'),str(profile_code),str(rotation)],120)
                result['rotations'].append({'profile_code':profile_code,'rotation':rotation,'measurement':sample})
                if sample['exit_code']!=0:
                    result['status']='measurement_failed'
                    return result
        for code in (38,39,40,41,42,43):
            counter=run(['ncu','--csv','--page','raw','--cache-control','none','--clock-control','none',
                '--kernel-name','regex:copy_panels','--launch-skip','2','--launch-count','1',
                '--metrics','lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__sass_inst_executed_op_ldgsts_cache_bypass.sum',str(TASK/'probe'),str(code),'0'],90)
            result['counters'].append({'cell_code':code,'measurement':counter})
            if counter['exit_code']!=0:
                result['status']='counter_failed'
                return result
        result['gpu_after']=run(['nvidia-smi','--query-gpu=name,clocks.sm,clocks.mem','--format=csv,noheader'],15)
        result['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        result['status']='complete'
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3]
    target=Path(output_dir).resolve()
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/finite_strided_panel_20261107').resolve() or target.name!='attempt_12' or target.exists():
        raise ValueError('output must be the new registered attempt_12')
    target.mkdir(parents=True)
    start=time.monotonic()
    result=capture.remote()
    result['client_wall_s']=time.monotonic()-start
    result['app_id']=APP.app_id
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    (target/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'app_id':result['app_id'],
                      'gpu_function_wall_s':result['gpu_function_wall_s']}))
