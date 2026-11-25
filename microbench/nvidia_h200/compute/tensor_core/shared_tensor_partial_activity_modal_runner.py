"""Independent active-CTA activity domain, preserving the sustained protocol."""
import gzip
import hashlib
import json
from pathlib import Path
import modal

IMAGE=modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04',add_python='3.12')
APP=modal.App('midlayer-h200-partial-tensor-activity-20261107')
FLAGS=['-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a',
    '-DMANTISSA_ACTIVITY_CONTROLS','-DSUSTAINED_DEVICE_ACTIVITY','-DTENSOR_ACTIVE_CTAS=128']

@APP.function(image=IMAGE,serialized=True,include_source=False,memory=2048,
              timeout=60,startup_timeout=30,retries=0,single_use_containers=True)
def precompile(source,flags):
    import hashlib,subprocess,tempfile
    from pathlib import Path
    results=[]
    with tempfile.TemporaryDirectory(prefix='midlayer-partial-tensor-cpu-') as directory:
        p=Path(directory)/'probe.cu';p.write_bytes(source)
        for extended in (False,True):
            command=['nvcc',*flags,*(['-DEXTENDED_CONTROLS'] if extended else []),str(p),'-o',directory+'/probe']
            run=subprocess.run(command,capture_output=True,text=True,timeout=25)
            results.append({'extended_controls':extended,'command':command,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
            if run.returncode!=0:break
    return {'gpu':None,'source_sha256':hashlib.sha256(source).hexdigest(),'builds':results}

@APP.function(gpu='H200',image=IMAGE,serialized=True,include_source=False,memory=2048,
              timeout=330,startup_timeout=30,retries=0,single_use_containers=True)
def capture(source,flags):
    import hashlib,os,subprocess,tempfile,time
    from pathlib import Path
    started=time.monotonic()
    query=['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit,clocks.sm,temperature.gpu','--format=csv,noheader']
    gpu=subprocess.run(query,capture_output=True,text=True,check=True,timeout=20).stdout
    if not gpu.startswith('NVIDIA H200,'):raise ValueError('one H200 required')
    environment=dict(os.environ)
    for name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
        if name in environment:del environment[name]
    results=[]
    with tempfile.TemporaryDirectory(prefix='midlayer-partial-tensor-gpu-') as directory:
        p=Path(directory)/'probe.cu';p.write_bytes(source)
        for extended in (False,True):
            command=['nvcc',*flags,*(['-DEXTENDED_CONTROLS'] if extended else []),'-Xptxas=-v',str(p),'-o',directory+'/probe']
            build=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=35)
            result={'extended_controls':extended,'command':command,'build_exit_code':build.returncode,'build_stdout':build.stdout,'build_stderr':build.stderr,
                'source_sha256':hashlib.sha256(source).hexdigest(),'sustained_device_warmup':True,'active_ctas':128,'launched_grid_ctas':132}
            results.append(result)
            if build.returncode!=0:break
            result['binary_sha256']=hashlib.sha256(Path(directory+'/probe').read_bytes()).hexdigest()
            result['sass']=subprocess.run(['cuobjdump','--dump-sass',directory+'/probe'],capture_output=True,text=True,check=True,timeout=15).stdout
            run=subprocess.run([directory+'/probe'],capture_output=True,text=True,timeout=140 if not extended else 175)
            result.update(run_exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr)
            result['gpu_after']=subprocess.run(query,capture_output=True,text=True,check=True,timeout=20).stdout
            if run.returncode!=0:break
    return {'gpu':gpu,'captures':results,'function_wall_s':time.monotonic()-started}

@APP.local_entrypoint()
def main():
    import subprocess,sys,tempfile
    here=Path(__file__).resolve().parent;root=here.parents[3]
    target=root/'profiler_output/microbench/nvidia_h200/shared_tensor_partial_activity_20261107'
    if {p.name for p in target.iterdir()}!={'preflight.json','local_build.log.gz'}:raise ValueError('refuse to overwrite captured evidence')
    primary=here/'shared_tensor_activity.cu';generator=here/'shared_tensor_activity_gen.py';primitive=root/'microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu'
    with tempfile.TemporaryDirectory(prefix='midlayer-partial-helper-') as directory:
        helper=Path(directory)/'helper.cuh';subprocess.run([sys.executable,str(generator),str(primitive),str(helper)],check=True,timeout=15);generated=helper.read_text()
    source=primary.read_text().replace('#include "shared_tensor_activity_helper.cuh"',generated).encode()
    snapshot=target/'source';snapshot.mkdir();(snapshot/'translation_unit.cu').write_bytes(source)
    for p in (primary,generator,primitive,Path(__file__)):
        if p==primitive:(snapshot/(p.name+'.gz')).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
        else:(snapshot/p.name).write_bytes(p.read_bytes())
    (snapshot/'shared_tensor_activity_helper.cuh').write_text(generated)
    smoke=precompile.remote(source,FLAGS);(target/'cpu_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
    if (len(smoke['builds'])!=2 or any(b['exit_code']!=0 for b in smoke['builds'])
            or smoke['source_sha256']!=hashlib.sha256(source).hexdigest()):raise ValueError('CPU compile/source failed; GPU not called')
    result=capture.remote(source,FLAGS)
    for i,row in enumerate(result['captures']):
        row['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();row['production_duration_consumed']=False
        for key in ('sass','stdout'):
            if key in row:
                data=row[key].encode();name=('extended' if i else 'base')+'_'+key+'.txt.gz';(target/name).write_bytes(gzip.compress(data,mtime=0));row[key+'_path']=name;row[key+'_sha256']=hashlib.sha256(data).hexdigest();del row[key]
        (target/('extended_raw.json' if i else 'base_raw.json')).write_text(json.dumps(row,indent=2)+'\n')
    (target/'capture_record.json').write_text(json.dumps({'gpu':result['gpu'],'function_wall_s':result['function_wall_s'],'subcaptures':len(result['captures'])},indent=2)+'\n')
    print(json.dumps({'build_exit_codes':[r['build_exit_code'] for r in result['captures']],
        'run_exit_codes':[r['run_exit_code'] if 'run_exit_code' in r else None for r in result['captures']],'function_wall_s':result['function_wall_s']}))
