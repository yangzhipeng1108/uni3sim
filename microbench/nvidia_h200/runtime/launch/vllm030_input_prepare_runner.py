"""Independent vLLM input preparation, with synthetic state and no model.

The four unmodified methods are compiled from the installed source. CPU mode
checks the fixture only and explicitly excludes GPU/block-table measurements.
CUDA mode uses the real CpuGpuBuffer and MultiGroupBlockTable implementations.
No Serving cases, model shapes, observed durations or predicted times are read.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,platform,statistics,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

METHODS=('_get_cumsum_and_arange','_compute_prev_positions','_prepare_input_ids','_prepare_inputs')


def source_methods(path):
    body=path.read_text();tree=ast.parse(body)
    owner,=tuple(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GPUModelRunner')
    definitions={node.name:ast.get_source_segment(body,node) for node in owner.body if isinstance(node,ast.FunctionDef) and node.name in METHODS}
    if set(definitions)!=set(METHODS):raise ValueError('required native preparation methods missing')
    # Source segments retain their class indentation after the first line.
    import textwrap
    code='from __future__ import annotations\n'+ '\n\n'.join(textwrap.dedent('    '+definitions[name]) for name in METHODS)
    namespace={'np':np,'torch':torch,'PIN_MEMORY':True}
    exec(compile(code,str(path),'exec'),namespace)
    return type('SyntheticInputPreparation',(),{name:namespace[name] for name in METHODS}),hashlib.sha256(body.encode()).hexdigest()


class CpuFixtureBuffer:
    def __init__(self,*shape,dtype,device,with_numpy=True):
        self.cpu=torch.zeros(shape,dtype=dtype);self.gpu=self.cpu.clone();self.np=self.cpu.numpy()
    def copy_to_gpu(self,n=None):
        if n is None:self.gpu.copy_(self.cpu)
        else:self.gpu[:n].copy_(self.cpu[:n])


class CpuFixtureBlockTable:
    def commit_block_table(self,num_reqs):pass
    def compute_slot_mapping(self,num_reqs,query_start_loc,positions):pass


def fixture(cls,requests,query_tokens,retained,device):
    cuda=device.type=='cuda';capacity=256;tokens=8192;max_len=8192
    if cuda:
        from vllm.v1.utils import CpuGpuBuffer
        from vllm.v1.worker.block_table import MultiGroupBlockTable
        buffer=CpuGpuBuffer
        blocks=MultiGroupBlockTable(capacity,tokens,True,device,[16],[16],[max_len//16])
        for i in range(requests):blocks.add_row((list(range(i*512,(i+1)*512)),),i)
    else:buffer=CpuFixtureBuffer;blocks=CpuFixtureBlockTable()
    runner=cls();runner.device=device;runner.arange_np=np.arange(tokens,dtype=np.int64)
    for name,size,dtype in (('input_ids',tokens,torch.int32),('query_pos',tokens,torch.int64),('query_start_loc',capacity+1,torch.int32),('prev_positions',capacity,torch.int64),('req_indices',tokens,torch.int64),('num_scheduled_tokens',capacity,torch.int32),('num_accepted_tokens',capacity,torch.int32),('discard_request_mask',capacity,torch.bool)):
        setattr(runner,name,buffer(size,dtype=dtype,device=device))
    runner.positions=torch.zeros(tokens,dtype=torch.int64,device=device)
    runner.seq_lens=torch.zeros(capacity,dtype=torch.int32,device=device)
    runner.num_computed_tokens=torch.zeros(capacity,dtype=torch.int32,device=device)
    runner.optimistic_seq_lens_cpu=torch.zeros(capacity,dtype=torch.int32,pin_memory=cuda)
    computed=torch.full((capacity,),1024,dtype=torch.int32,pin_memory=cuda)
    ids=torch.arange(capacity*max_len,dtype=torch.int32,pin_memory=cuda).reshape(capacity,max_len)
    names=[f'synthetic-{i}' for i in range(requests)]
    runner.input_batch=SimpleNamespace(num_reqs=requests,block_table=blocks,num_computed_tokens_cpu=computed.numpy(),num_computed_tokens_cpu_tensor=computed,token_ids_cpu=ids.numpy(),token_ids_cpu_tensor=ids,req_prompt_embeds={},prev_req_id_to_index={name:i for i,name in enumerate(names)} if retained else {},req_ids=names,prev_sampled_token_ids=torch.full((requests,1),17,dtype=torch.int32,device=device) if retained else None)
    runner.requests={name:SimpleNamespace(num_tokens=1024+query_tokens) for name in names}
    runner.enable_prompt_embeds=False;runner.uses_mrope=False;runner.num_accepted_tokens_event=None
    runner.use_async_scheduling=True;runner.cache_config=SimpleNamespace(mamba_cache_mode='none')
    runner.mamba_prev_last_scheduled_idx=None;runner.use_async_spec_decode=False
    runner.valid_sampled_token_count_gpu=None;runner._pp_recv_work=None;runner.lora_config=None
    runner.prev_num_spec_tokens=0;runner._draft_token_ids=None
    output=SimpleNamespace(total_num_scheduled_tokens=requests*query_tokens,scheduled_spec_decode_tokens={})
    counts=np.full(requests,query_tokens,dtype=np.int32)
    return runner,output,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--samples',type=int,default=64)
    args=parser.parse_args()
    if args.output.exists() or args.samples<4:raise ValueError('output exists or insufficient samples')
    path=args.source_root/'vllm/v1/worker/gpu_model_runner.py'
    cls,digest=source_methods(path);device=torch.device(args.device)
    if device.type=='cuda' and 'H200' not in torch.cuda.get_device_name(0):raise ValueError('requires authorized H200')
    rows=[]
    for requests,query,retained in ((1,1,True),(8,1,True),(32,1,True),(128,1,True),(1,32,False),(8,32,False),(32,32,False),(1,256,False),(8,256,False)):
        runner,output,counts=fixture(cls,requests,query,retained,device)
        host=[];complete=[];gpu=[]
        for sample in range(12+args.samples):
            if device.type=='cuda':
                torch.cuda.synchronize();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record()
            start=time.perf_counter_ns();indices,spec,max_sampled=runner._prepare_inputs(output,counts);enqueued=time.perf_counter_ns()
            if device.type=='cuda':end_event.record();end_event.synchronize()
            ended=time.perf_counter_ns()
            if sample>=12:
                host.append(enqueued-start);complete.append(ended-start)
                if device.type=='cuda':gpu.append(start_event.elapsed_time(end_event)*1e6)
        assert spec is None and max_sampled==1
        assert indices.tolist()==[(i+1)*query-1 for i in range(requests)]
        assert runner.positions[:requests*query].tolist()==[1024+j for i in range(requests) for j in range(query)]
        expected=([17]*requests if retained else [i*8192+1024+j for i in range(requests) for j in range(query)])
        assert runner.input_ids.gpu[:requests*query].tolist()==expected
        rows.append({'requests':requests,'query_tokens_each':query,'retained_async_tokens':retained,'host_enqueue_ns':host,'synchronized_completion_ns':complete,'cuda_event_ns':gpu,'median_host_enqueue_ns':statistics.median(host),'median_completion_ns':statistics.median(complete)})
        print(json.dumps({k:v for k,v in rows[-1].items() if not isinstance(v,list)}),flush=True)
    payload={'kind':'independent_vllm030_worker_input_prepare','measurement_boundary':'four exact native methods, synthetic state, no model forward or weights; CUDA buffers and one real block-table group when device=cuda','limitations':['excludes worker update_state and attention-metadata builder','completion and CUDA-event include CPU enqueue idle; do not label as GPU kernel service','not full model runner or production workload'],'cpu_preflight_only':device.type=='cpu','production_or_evaluation_duration_consumed':False,'source_path':str(path),'source_sha256':digest,'method_names':METHODS,'torch':torch.__version__,'host':{'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'cpuinfo_first':Path('/proc/cpuinfo').read_text().split('\n\n',1)[0]},'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__':main()
