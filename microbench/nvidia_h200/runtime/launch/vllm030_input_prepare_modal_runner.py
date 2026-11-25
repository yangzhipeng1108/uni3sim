"""One budgeted H200 run of independent synthetic worker-input preparation."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
import modal
HERE=Path(__file__).resolve();REMOTE=HERE.parent==Path('/root')
ROOT=Path('/tmp') if REMOTE else HERE.parents[4]
SOURCE=Path('/tmp/midlayer_input_prepare.py') if REMOTE else HERE.with_name('vllm030_input_prepare_runner.py')
CAMPAIGN=ROOT/'profiler_output/microbench/nvidia_h200/serving_accuracy_input_prepare_20261107'
IMAGE=modal.Image.from_registry('vllm/vllm-openai:v0.30.0',add_python='3.12').entrypoint([]).add_local_file(SOURCE,remote_path='/tmp/midlayer_input_prepare.py')
APP=modal.App('midlayer-h200-serving-input-prepare-20261107')

@APP.function(gpu='H200',image=IMAGE,timeout=240,retries=0,max_containers=1,single_use_containers=True)
def measure(digest:str)->dict:
    started=time.monotonic();source=Path('/tmp/midlayer_input_prepare.py')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('uploaded probe changed')
    identity=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
    if 'H200' not in identity:raise ValueError('authorized H200 required')
    try:
        with tempfile.TemporaryDirectory(prefix='midlayer_input_prepare_',dir='/tmp') as temporary:
            output=Path(temporary)/'raw.json'
            locate=subprocess.run(['/usr/bin/python3','-c','import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent.parent)'],check=True,capture_output=True,text=True)
            command=['/usr/bin/python3',str(source),'--source-root',locate.stdout.strip().splitlines()[-1],'--device','cuda','--samples','64','--output',str(output)]
            completed=subprocess.run(command,check=False,capture_output=True,text=True,timeout=180)
            return {'status':'complete' if completed.returncode==0 else 'probe_failed','gpu_identity':identity,'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'source_sha256':digest,'raw':json.loads(output.read_text()) if output.is_file() else None,'gpu_function_wall_s':time.monotonic()-started}
    finally:source.unlink()

@APP.local_entrypoint()
def main():
    if (CAMPAIGN/'modal_raw.json').exists():raise FileExistsError(CAMPAIGN/'modal_raw.json')
    start=time.monotonic();result=measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result['client_wall_s']=time.monotonic()-start;result['runner_sha256']=hashlib.sha256(HERE.read_bytes()).hexdigest()
    result['budget_scope']='Serving independent US$25; this reservation <=US$1.20, <=6min1xH200 including warmup; no retry; cumulative reservations<=US$2.40'
    CAMPAIGN.mkdir(parents=True,exist_ok=True);(CAMPAIGN/'modal_raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(CAMPAIGN/'modal_raw.json'),'status':result['status'],'gpu_function_wall_s':result['gpu_function_wall_s']}))
