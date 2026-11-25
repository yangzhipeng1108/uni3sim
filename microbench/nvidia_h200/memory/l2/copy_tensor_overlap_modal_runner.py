"""Measure independent copy/Tensor concurrency; no production operator."""
import hashlib,json,re,shutil,subprocess,time
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'copy_tensor_overlap.cu'
REFERENCE=HERE/'../../operators/flash_attn/fa3_wgmma_service.cu'
TASK=Path('/tmp/midlayer-copy-tensor-overlap-20261107')
IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11').apt_install('cuda-nsight-compute-12-6').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)).add_local_file(REFERENCE,remote_path=str(TASK/REFERENCE.name))
APP=modal.App('midlayer-copy-tensor-overlap-20261107')

def helper(source):
 first='__device__ __forceinline__ void wgmma_ss_n64_kmajor'
 start='__device__ __forceinline__ void wgmma_ss_n128_kmajor'
 end='__device__ __forceinline__ void wgmma_ss_n256_mnmajor'
 if any(source.count(x)!=1 for x in (first,start,end)):raise ValueError('WGMMA helper boundary changed')
 return source[:source.index(first)]+source[source.index(start):source.index(end)]

def code_gate(sass):
 names=re.findall(r'Function : (_Z\S+)',sass)
 if len(names)!=18 or 'WARPGROUP.DEPBAR' not in sass or 'LDGSTS' not in sass:return False
 for body in sass.split('Function : ')[1:]:
  name=body.splitlines()[0]
  dimensions=re.search(r'ILi(\d+)ELi(\d+)ELi(\d+)EE',name)
  if dimensions is None:return False
  matrices=int(dimensions[2])
  batches=[window.count('HGMMA.') for window in body.split('WARPGROUP.DEPBAR') if 'HGMMA.' in window]
  if not batches or any(n<matrices or n%matrices for n in batches):return False
 return not any('WG.DP' in part or 'STL ' in part or 'LDL ' in part for part in sass.splitlines())

@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
 result={};started=time.monotonic()
 try:
  reference=(TASK/REFERENCE.name).read_text();header=helper(reference)
  (TASK/'copy_tensor_wgmma_helper.cuh').write_text(header)
  result['helper_source']=header;result['reference_sha256']=hashlib.sha256(reference.encode()).hexdigest()
  commands={
   'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
   'compiler':['nvcc','--version'],
   'build':['nvcc','-O3','-arch=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
   'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
   'measurement':[str(TASK/'probe')],
  }
  metrics='lts__t_sectors_op_read.sum,dram__bytes_read.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'
  for mode in (0,1,2):commands['counter_'+str(mode)]=['ncu','--csv','--page','raw','--cache-control','none','--clock-control','none','--kernel-name','regex:copy_tensor','--launch-skip','8','--launch-count','1','--metrics',metrics,str(TASK/'probe'),str(mode)]
  result['required_commands']=list(commands)
  for name,command in commands.items():
   p=subprocess.run(command,capture_output=True,text=True,timeout=60,cwd=TASK)
   result[name]={'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
   if p.returncode:break
   if name=='build':
    spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)
    result['zero_spill_gate']=len(spills)==18 and all(a==b=='0' for a,b in spills)
    if not result['zero_spill_gate']:break
   if name=='sass':
    result['code_gate']=code_gate(p.stdout)
    if not result['code_gate']:break
  return result
 finally:
  result['gpu_function_wall_s']=time.monotonic()-started
  if TASK.exists():shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 if (out/'raw.json').exists():raise ValueError('capture already exists')
 result=capture.remote();result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest();result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
 if not all(name in result and result[name]['exit_code']==0 for name in result['required_commands']):raise RuntimeError('Copy/Tensor capture failed; inspect raw')
 print({'wall_s':result['gpu_function_wall_s'],'zero_spill_gate':result['zero_spill_gate'],'code_gate':result['code_gate']})
