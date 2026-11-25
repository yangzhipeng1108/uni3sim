"""Native non-duration request-layer attribution, never a Forward rate."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CASE = HERE / 'fused_moe_counter_case.py'
TASK = Path('/tmp/midlayer-moe-request-layers-20261108')
IMAGE = (modal.Image.from_registry('vllm/vllm-openai:v0.30.0',add_python='3.12').entrypoint([])
         .apt_install('cuda-nsight-compute-13-0','cuda-cuobjdump-13-0','cuda-nvdisasm-13-0')
         .add_local_file(CASE,remote_path=str(TASK/CASE.name)))
APP = modal.App('midlayer-moe-request-layers-20261108')
CELLS = ((4096,2,'e8_h6144_i16384'),(8192,16,'e128_h2048_i768'))
METRICS = ('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,'
           'lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,'
           'lts__t_sectors_srcunit_ltcfabric_op_read.sum,lts__t_sectors_srcunit_ltcfabric_op_write.sum,'
           'lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,'
           'sm__ctas_launched.sum,sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second')
SELECTION = ('Frozen diagnostic only: two synthetic vLLM0.30profiles with native BM128/N128/depth3/twoCTA and '
             'BM128/N256/depth4/oneCTA signatures, both W1/W2projections. Read only upstream L1TEX sectors, '
             'downstream LTS TEX/fabric/read/write,DRAM,CTA and clock counters. No production duration requested, '
             'no observed ratio or rate may enter Forward. Application replay,cache/clock control none. '
             'Triton metadata/SASS binds resource/instruction identity; a mismatch is partial,not native equivalence. '
             'Synthetic seed42routing is not claimed identical to canonical; histograms/source/output validity retained. '
             'All four projections retained. This determines whether source logical L2work conflates request layers '
             'and whether hot merging merits a separate independent mechanism experiment,not a fitted correction.')


def run(command, limit, env):
    start=time.monotonic()
    p=subprocess.run(command,cwd=TASK,env=env,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)


@APP.function(gpu='H200',image=IMAGE,timeout=570,retries=0,max_containers=1,single_use_containers=True)
def capture():
    start=time.monotonic()
    out=dict(status='started',selection=SELECTION,cells=[],compiler_artifacts=[],compiler_export_commands=[])
    env=os.environ.copy()
    env['TRITON_CACHE_DIR']=str(TASK/'triton')
    env['CUDA_CACHE_PATH']=str(TASK/'cuda')
    env['XDG_CACHE_HOME']=str(TASK/'cache')
    try:
        out['gpu']=run(['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20,env)
        if out['gpu']['exit_code'] or 'H200' not in out['gpu']['stdout']:
            out['status']='gpu_identity_failed'
            return out
        for name,command in (('cuobjdump',['cuobjdump','--version']),('nvdisasm',['nvdisasm','--version'])):
            out[name]=run(command,15,env)
            if out[name]['exit_code']:
                out['status']=name+'_preflight_failed'
                return out
        for tokens,topk,profile in CELLS:
            cmd=['ncu','--csv','--page','raw','--print-units','base','--replay-mode','application',
                 '--cache-control','none','--clock-control','none','--kernel-name','regex:fused_moe_kernel',
                 '--launch-skip','2','--launch-count','2','--metrics',METRICS,'/usr/bin/python3',str(TASK/CASE.name),
                 '--tokens',str(tokens),'--topk',str(topk),'--profile',profile]
            result=run(cmd,240,env)
            out['cells'].append(dict(tokens=tokens,topk=topk,profile=profile,measurement=result))
            if result['exit_code']:
                out['status']='counter_failed'
                return out
        for metadata in sorted((TASK/'triton').rglob('fused_moe_kernel*.json')):
            binary=metadata.with_suffix('.cubin')
            if not binary.is_file():
                raise ValueError('compiler metadata missing cubin')
            sass=run(['cuobjdump','--dump-sass',str(binary)],30,env)
            out['compiler_export_commands'].append(sass)
            out['compiler_artifacts'].append(dict(metadata=json.loads(metadata.read_text()),metadata_path=str(metadata.relative_to(TASK)),
                cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),sass=sass['stdout']))
            if sass['exit_code']:
                out['status']='sass_failed'
                return out
        if len(out['compiler_artifacts'])!=4:
            out['status']='compiler_scope_failed'
            return out
        out['status']='complete'
        return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[2]
    target=Path(output_dir).resolve()
    if target!=(root/'profiler_output/kernel_dataset/nvidia_h200/moe_request_layers_20261108/attempt_1').resolve() or target.exists():
        raise ValueError('requires new registered destination in this worktree')
    start=time.monotonic()
    out=capture.remote()
    out['client_wall_s']=time.monotonic()-start
    out['app_id']=APP.app_id
    out['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (CASE,Path(__file__))}
    target.mkdir(parents=True)
    (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))


NATIVE_STALL_METRICS=(
    'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,'
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,'
    'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,'
    'smsp__warp_issue_stalled_barrier_per_warp_active.pct,'
    'smsp__warp_issue_stalled_wait_per_warp_active.pct,'
    'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,'
    'smsp__warps_eligible.avg.per_cycle_active,smsp__warps_active.avg.per_cycle_active,'
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,'
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,'
    'sm__ctas_launched.sum,sm__cycles_elapsed.avg.per_second')


def native_hmma_identity(artifacts, protocol, binary_pins):
    """Separate legacy exact identity from bound new-source counter scope."""
    import re
    if protocol not in (1,2):raise ValueError('unknown native counter protocol')
    if len(artifacts)!=2:raise ValueError('exactly two compiled projection artifacts required')
    rows=[]
    for artifact in artifacts:
        metadata=artifact['metadata'];sass=artifact['sass'];resources=artifact['resources']
        facts=re.findall(r'Function [^\n]+:\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',resources)
        if len(facts)!=1:raise ValueError('one compiled kernel resource record required')
        registers,stack,shared,local=map(int,facts[0])
        counts=dict(hmma=len(re.findall(r'\bHMMA\.16816\.F32\.BF16\b',sass)),
            ldsm=len(re.findall(r'\bLDSM\.16\.M88(?:\.MT88)?\.4\b',sass)),
            ldgsts=len(re.findall(r'\bLDGSTS\.E\.BYPASS\.128\b',sass)))
        common=(stack==0 and local==0 and metadata['shared']==61440 and metadata['num_warps']==4
                and counts==dict(hmma=16,ldsm=16,ldgsts=40) and 'HGMMA.' not in sass)
        if protocol==1:admitted=common and registers==64 and shared==0
        else:
            # These are SOURCE/binary pins from a separately closed value/bank
            # graph, not measured durations or canonical binary equality.
            admitted=common and artifact['cubin_sha256'] in binary_pins
        rows.append(dict(metadata_path=artifact['metadata_path'],registers=registers,stack=stack,cuobjdump_reported_shared=shared,local=local,
            dynamic_shared=metadata['shared'],warps=metadata['num_warps'],counts=counts,
            canonical_register_identity_matches=registers==64,user_static_shared_bytes=None,
            shared_allocation_scope='cuobjdump field is not NCU user-static;postcapture launch facts required',
            counter_source_domain_matches=admitted))
    if protocol==2 and {a['cubin_sha256'] for a in artifacts}!=set(binary_pins):raise ValueError('bound binary set changed')
    return rows


@APP.function(gpu='H200',image=IMAGE,timeout=270,retries=0,max_containers=1,single_use_containers=True)
def capture_native_stalls(plan):
    import zlib
    start=time.monotonic();out=dict(status='started',plan=plan,compiler_artifacts=[],preflight_commands=[],counter_runs=[])
    env=os.environ.copy();env['TRITON_CACHE_DIR']=str(TASK/'native_triton');env['CUDA_CACHE_PATH']=str(TASK/'native_cuda');env['XDG_CACHE_HOME']=str(TASK/'native_cache')
    def bounded(command,limit):
        try:return run(command,limit,env)
        except subprocess.TimeoutExpired as exc:
            def decoded(v):return '' if v is None else v.decode(errors='replace') if isinstance(v,bytes) else v
            return dict(command=command,exit_code=124,stdout=decoded(exc.stdout),stderr=decoded(exc.stderr),timed_out=True)
    try:
        out['gpu']=bounded(['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15)
        if out['gpu']['exit_code'] or 'H200' not in out['gpu']['stdout']:out['status']='gpu_identity_failed';return out
        out['ncu_help']=bounded(['ncu','--help'],15)
        if out['ncu_help']['exit_code'] or '--print-source' not in out['ncu_help']['stdout']:out['status']='source_export_cli_unavailable';return out
        cmd=['/usr/bin/python3',str(TASK/CASE.name),'--tokens','8','--topk','2','--profile','e8_h6144_i16384']
        out['compile_case']=bounded(cmd,70)
        if out['compile_case']['exit_code']:out['status']='compile_case_failed';return out
        for metadata in sorted((TASK/'native_triton').rglob('fused_moe_kernel*.json')):
            binary=metadata.with_suffix('.cubin')
            if not binary.is_file():raise ValueError('metadata missing compiled cubin')
            sass=bounded(['cuobjdump','--dump-sass',str(binary)],15);resources=bounded(['cuobjdump','--dump-resource-usage',str(binary)],15)
            out['preflight_commands'].extend((sass,resources))
            if sass['exit_code'] or resources['exit_code']:out['status']='binary_export_failed';return out
            out['compiler_artifacts'].append(dict(metadata=json.loads(metadata.read_text()),metadata_path=str(metadata.relative_to(TASK)),
                cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cubin_zlib_hex=zlib.compress(binary.read_bytes()).hex(),sass=sass['stdout'],resources=resources['stdout']))
        try:out['native_identity']=native_hmma_identity(out['compiler_artifacts'],plan['protocol'],plan['binary_pins'])
        except ValueError as exc:
            out['status']='source_identity_unbound';out['identity_error']=str(exc);return out
        if not all(r['counter_source_domain_matches'] for r in out['native_identity']):out['status']='physical_identity_mismatch';return out
        for repeat in (0,1):
            if time.monotonic()-start>180:out['status']='bounded_partial';return out
            report_prefix=TASK/f'native_shared_{repeat}'
            report=report_prefix.with_suffix('.ncu-rep')
            item=dict(repeat=repeat)
            if plan['replay'] not in ('application','kernel'):raise ValueError('unknown profiler replay')
            item['measurement']=bounded(['ncu','--csv','--page','raw','--print-units','base','--replay-mode',plan['replay'],'--cache-control','none','--clock-control','none',
                '--kernel-name','regex:fused_moe_kernel','--launch-skip','2','--launch-count','2','--section','SourceCounters','--metrics',NATIVE_STALL_METRICS,'--export',str(report_prefix),*cmd],65)
            out['counter_runs'].append(item)
            if report.is_file():
                payload=report.read_bytes();item['report_sha256']=hashlib.sha256(payload).hexdigest();item['report_bytes']=len(payload);item['report_zlib_hex']=zlib.compress(payload).hex()
                item['source_export']=bounded(['ncu','--import',str(report),'--page','source','--print-source','sass','--csv'],20)
            if item['measurement']['exit_code']:out['status']='counter_failed';return out
            if 'source_export' not in item or item['source_export']['exit_code']:out['status']='source_export_failed';return out
        out['status']='complete';return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK);out['remote_task_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def native_shared_stalls(output_dir:str,budget_key:str,budget_tool:str,plan_path:str):
    import sys
    root=HERE.parents[2];target=Path(output_dir).resolve();tool=Path(budget_tool);path=Path(plan_path).resolve()
    expected=root/'profiler_output/kernel_dataset/nvidia_h200/moe_native_stalls_20261108'
    if target!=(expected/'attempt_3_kernel_replay').resolve() or target.exists() or path!=(expected/'plan_kernel_replay.json').resolve():raise ValueError('new registered source-bound destination and frozen plan required')
    if not tool.is_file() or not budget_key:raise ValueError('shared atomic reservation required')
    plan=json.loads(path.read_text());actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (CASE,Path(__file__))}
    if actual!=plan['source_sha256']:raise ValueError('source changed after native counter plan freeze')
    target.mkdir(parents=True)
    meta=dict(app_id=APP.app_id,budget_key=budget_key,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_sha256=actual,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    started=time.monotonic()
    try:
        out=capture_native_stalls.remote(plan);out.update(meta);out['client_wall_s']=time.monotonic()-started
        (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=APP.app_id,gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as exc:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n');raise
