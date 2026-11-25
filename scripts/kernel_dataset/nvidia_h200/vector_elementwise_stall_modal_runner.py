"""Native non-duration Vector kernel stall attribution, never a Forward rate."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
CASE = HERE / 'vector_elementwise_case_runner.py'
TASK = Path('/tmp/midlayer-vector-stalls-20261108')
IMAGE = (modal.Image.from_registry('vllm/vllm-openai:v0.30.0',add_python='3.12').entrypoint([])
         .apt_install('cuda-nsight-compute-13-0','cuda-cuobjdump-13-0','cuda-nvdisasm-13-0')
         .add_local_file(CASE,remote_path=str(TASK/CASE.name)))
APP = modal.App('midlayer-vector-stalls-20261108')

# Representative case: ELEMENTWISE T=8 H=2048 (grid=16, block=128)
# Matches the 256ns-gap representative case from checkpoint22
METRICS = (
    'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,'
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,'
    'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,'
    'smsp__warp_issue_stalled_barrier_per_warp_active.pct,'
    'smsp__warp_issue_stalled_wait_per_warp_active.pct,'
    'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,'
    'smsp__warp_issue_stalled_drain_per_warp_active.pct,'
    'smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,'
    'smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct,'
    'smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,'
    'smsp__warps_eligible.avg.per_cycle_active,'
    'smsp__warps_active.avg.per_cycle_active,'
    'sm__ctas_launched.sum,'
    'sm__cycles_elapsed.avg.per_second,'
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
    'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,'
    'lts__t_sectors_op_read.sum,'
    'lts__t_sectors_op_write.sum,'
    'dram__bytes_read.sum,'
    'dram__bytes_write.sum'
)

SELECTION = (
    'Frozen diagnostic only: ELEMENTWISE T=8 H=2048 representative case. '
    'Read only non-duration stall reasons, warp states, memory sector counters, '
    'and clock counters. No production duration requested, no observed ratio or '
    'rate may enter Forward. Application replay, cache/clock control none. '
    'This identifies which hardware mechanisms (memory latency, barrier, drain, '
    'math throttle) dominate the 256ns unexplained residual for small-grid '
    'vector kernels, supporting or refuting model hypotheses.'
)


def run(command, limit, env):
    start=time.monotonic()
    p=subprocess.run(command,cwd=TASK,env=env,capture_output=True,text=True,timeout=limit)
    return dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-start)


@APP.function(gpu='H200',image=IMAGE,timeout=570,retries=0,max_containers=1,single_use_containers=True)
def capture():
    import zlib
    start=time.monotonic()
    out=dict(status='started',selection=SELECTION,counter_runs=[],compiler_artifacts=[])
    env=os.environ.copy()
    env['TRITON_CACHE_DIR']=str(TASK/'triton')
    env['CUDA_CACHE_PATH']=str(TASK/'cuda')
    env['XDG_CACHE_HOME']=str(TASK/'cache')
    try:
        out['gpu']=run(['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20,env)
        if out['gpu']['exit_code'] or 'H200' not in out['gpu']['stdout']:
            out['status']='gpu_identity_failed'
            return out
        out['ncu_help']=run(['ncu','--help'],15,env)
        if out['ncu_help']['exit_code'] or '--print-source' not in out['ncu_help']['stdout']:
            out['status']='source_export_cli_unavailable'
            return out

        # Compile and run the vector case once to generate binary
        cmd=['/usr/bin/python3',str(TASK/CASE.name),'--tok','8','--hidden','2048','--operation','add']
        out['compile_case']=run(cmd,70,env)
        if out['compile_case']['exit_code']:
            out['status']='compile_case_failed'
            return out

        # Find compiled binary (PyTorch JIT or Triton)
        for pattern in ('*.cubin','*.so'):
            for binary in sorted(TASK.rglob(pattern)):
                if binary.is_file() and binary.stat().st_size > 1000:
                    sass=run(['cuobjdump','--dump-sass',str(binary)],30,env)
                    resources=run(['cuobjdump','--dump-resource-usage',str(binary)],30,env)
                    out['compiler_artifacts'].append(dict(
                        binary_path=str(binary.relative_to(TASK)),
                        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                        sass=sass['stdout'][:50000],  # Limit SASS size
                        resources=resources['stdout']))

        # Capture with NCU kernel replay
        for repeat in (0,1):
            if time.monotonic()-start > 180:
                out['status']='bounded_partial'
                return out
            report_prefix=TASK/f'vector_stalls_{repeat}'
            report=report_prefix.with_suffix('.ncu-rep')
            item=dict(repeat=repeat)
            item['measurement']=run(['ncu','--csv','--page','raw','--print-units','base',
                '--replay-mode','kernel','--cache-control','none','--clock-control','none',
                '--kernel-name','regex:elementwise','--launch-skip','2','--launch-count','2',
                '--section','SourceCounters','--metrics',METRICS,'--export',str(report_prefix),
                *cmd],65,env)
            out['counter_runs'].append(item)
            if report.is_file():
                payload=report.read_bytes()
                item['report_sha256']=hashlib.sha256(payload).hexdigest()
                item['report_bytes']=len(payload)
                item['report_zlib_hex']=zlib.compress(payload).hex()
                item['source_export']=run(['ncu','--import',str(report),'--page','source','--print-source','sass','--csv'],20,env)
            if item['measurement']['exit_code']:
                out['status']='counter_failed'
                return out
            if 'source_export' not in item or item['source_export']['exit_code']:
                out['status']='source_export_failed'
                return out

        out['status']='complete'
        return out
    finally:
        out['gpu_function_wall_s']=time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed']=not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str,budget_key:str,budget_tool:str):
    import sys
    root=HERE.parents[2]
    target=Path(output_dir).resolve()
    expected=root/'profiler_output/vector_stalls_20261108/attempt_1_kernel_replay'
    if target!=expected.resolve() or target.exists():
        raise ValueError('requires new registered destination in this worktree')
    tool=Path(budget_tool)
    if not tool.is_file() or not budget_key:
        raise ValueError('shared atomic reservation required')
    target.mkdir(parents=True)
    meta=dict(app_id=APP.app_id,budget_key=budget_key,
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (CASE,Path(__file__))})
    subprocess.run([sys.executable,str(tool),'bind','--key',budget_key,'--app-id',APP.app_id],check=True)
    (target/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)
    started=time.monotonic()
    try:
        out=capture.remote();out.update(meta);out['client_wall_s']=time.monotonic()-started
        (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
    except Exception as exc:
        (target/'client_failure.json').write_text(json.dumps(dict(**meta,error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n')
        raise
