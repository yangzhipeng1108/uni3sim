"""Bind only the fetched installed-source snapshot; imports no device packages."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bootstrap'))
from memory_capture import ast_hash,method_hash
TARGETS={
 'vllm.utils.mem_utils':[(None,'memory_profiling'),('MemorySnapshot','measure'),('DeviceMemoryProfiler','current_memory_usage')],
 'vllm_ascend.worker.model_runner_v1':[('NPUModelRunner','load_model'),('NPUModelRunner','capture_model'),('NPUModelRunner','initialize_kv_cache'),(None,'_torch_cuda_wrapper')],
 'vllm_ascend.worker.worker':[('NPUWorker','_init_device'),('NPUWorker','init_device'),('NPUWorker','determine_available_memory'),('NPUWorker','initialize_from_config'),('NPUWorker','compile_or_warm_up_model'),('NPUWorker','_warm_up_atb')],
 'vllm_ascend.patch.platform.patch_torch_accelerator':[(None,'patch_empty_cache')],
 'vllm_ascend.platform':[('NPUPlatform','get_current_memory_usage')],
 'vllm.v1.worker.gpu_model_runner':[('GPUModelRunner','capture_model')],
 'vllm_ascend.utils':[(None,'refresh_block_size')],
 'vllm.v1.kv_cache_interface':[]}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,default=HERE/'bindings.json');args=parser.parse_args()
 raw=json.loads((args.source_snapshot/'source_snapshot.stdout.json').read_text())
 manifest={'identity':'ascend018-installed-startup-native-bytes-reset-epochs-v2','software_versions':raw['versions'],
  'source_snapshot_reference':'profiler_output/serving/ascend_910b/startup_native_memory_20261026/installed_source_readonly','modules':{}}
 for name,methods in TARGETS.items():
  source=(args.source_snapshot/(name+'.py')).read_text()
  manifest['modules'][name]={'module_ast_sha256':ast_hash(source),'methods':[{'owner':owner,'name':method,'method_ast_sha256':method_hash(source,owner,method)}for owner,method in methods]}
 args.output.write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'modules':len(TARGETS),'bound_functions':sum(map(len,TARGETS.values())),'device_imports':[name for name in('torch','torch_npu','vllm','vllm_ascend')if name in sys.modules]}))
if __name__=='__main__':main()
