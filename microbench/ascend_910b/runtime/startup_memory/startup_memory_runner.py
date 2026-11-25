"""Plan by default. A future expressly authorized run captures existing vLLM startup bytes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
REQUIRED_PHASES={'worker_init.end','model_load.begin','model_load.end','profile.begin','profile.end',
                 'available_KV.end','runtime_peak_reset.after','KV_allocate.end','all_capture.begin','all_capture.end','startup_warmup.end'}


@contextmanager
def fresh_task_tmpdir(task_root):
    """Own one short fresh IPC directory until this startup has stopped."""
    temporary=Path(tempfile.mkdtemp(prefix=task_root.name+'-',dir='/tmp'))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary)


def command_for(model,port):
    return [sys.executable,'-B','-c','from vllm.entrypoints.cli.main import main; main()',
        'serve',str(model),'--host','127.0.0.1','--port',str(port),'--served-model-name','Qwen3-8B',
        '--dtype','bfloat16','--seed','42','--tensor-parallel-size','1','--gpu-memory-utilization','0.8',
        '--max-model-len','8192','--max-num-batched-tokens','8192','--max-num-seqs','256',
        '--enable-chunked-prefill','--enable-prefix-caching']


def read_phases(raw):
    phases_by_pid={}
    for path in raw.glob('memory.rank-process-*.jsonl'):
        for line in path.read_text().split('\n')[:-1]:
            record=json.loads(line)
            if record['record_type']=='native_startup_memory_bytes':
                pid=record['pid']
                if pid not in phases_by_pid: phases_by_pid[pid]=set()
                phases_by_pid[pid].add(record['phase'])
    return phases_by_pid


def read_bound_modules(raw):
    result={}
    for path in raw.glob('memory.rank-process-*.jsonl'):
        for line in path.read_text().split('\n')[:-1]:
            record=json.loads(line)
            if record['record_type']=='installed_source_binding':
                pid=record['pid']
                if pid not in result:result[pid]=set()
                result[pid].add(record['module'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--raw-output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--startup-timeout-seconds',type=int,default=300)
    parser.add_argument('--task-root',type=Path,
        help='assigned remote task root; required only for authorized execution')
    parser.add_argument('--execute-authorized-device-startup',action='store_true')
    args=parser.parse_args()
    if args.device_index<0 or not 1024<=args.port<=65535 or not 1<=args.startup_timeout_seconds<=600:
        raise ValueError('Explicit device/port and bounded startup timeout required')
    command=command_for(args.model,args.port)
    if not args.execute_authorized_device_startup:
        print(json.dumps({'mode':'plan_only_no_model_or_device_import','command':command,'required_phases':sorted(REQUIRED_PHASES),
                         'raw_output':str(args.raw_output),'device_index':args.device_index},indent=2));return 0
    # This branch is not exercised during CPU preparation. Owner must first
    # authorize/coordinate an idle NPU and stage this bundle in its task dir.
    if args.task_root is None:
        raise ValueError('--task-root is required for authorized execution')
    raw=args.raw_output.resolve()
    allowed=args.task_root.resolve()
    if allowed not in raw.parents or not raw.relative_to(allowed).parts[0].startswith('wt-'):
        raise ValueError('Raw output must be in the assigned remote task directory')
    task_root=allowed/raw.relative_to(allowed).parts[0]
    config=args.model/'config.json'
    if not config.is_file():raise FileNotFoundError(config)
    if raw.exists():raise FileExistsError('Fresh raw campaign directory required: '+str(raw))
    raw.mkdir(parents=True)
    identity={'model':str(args.model),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
              'device_index':args.device_index,'command':command,'requested_block_size_tokens':None,'provider_or_capacity_acceptance':False}
    (raw/'capture_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    environment=dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1',ASCEND_RT_VISIBLE_DEVICES=str(args.device_index),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        MIDLAYER_STARTUP_MEMORY_RAW=str(raw),MIDLAYER_STARTUP_MEMORY_BINDINGS=str(HERE/'bindings.json'),
        MIDLAYER_STARTUP_MEMORY_IDENTITY=str(raw/'capture_identity.json'),
        ASCEND_PROCESS_LOG_PATH=str(raw/'ascend_logs'),ASCEND_CACHE_PATH=str(raw/'ascend_cache'),
        VLLM_CACHE_ROOT=str(raw/'vllm_cache'),XDG_CACHE_HOME=str(raw/'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(raw/'inductor_cache'),TRITON_CACHE_DIR=str(raw/'triton_cache'),TORCH_HOME=str(raw/'torch_cache'))
    for name in ('ascend_logs','ascend_cache','vllm_cache','cache','inductor_cache','triton_cache','torch_cache'):(raw/name).mkdir()
    bootstrap=str(HERE/'bootstrap')
    environment['PYTHONPATH']=bootstrap if 'PYTHONPATH' not in environment else bootstrap+os.pathsep+environment['PYTHONPATH']
    expected_modules=set(json.loads((HERE/'bindings.json').read_text())['modules'])
    with fresh_task_tmpdir(task_root) as temporary:
        environment['TMPDIR']=str(temporary)
        identity['runtime_tmpdir']=str(temporary)
        (raw/'capture_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        process=None;phases=set();phase_sets={};bound_sets={};complete=False;started=time.monotonic()
        with (raw/'startup.stdout.log').open('w') as stdout,(raw/'startup.stderr.log').open('w') as stderr:
            try:
                process=subprocess.Popen(command,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
                while time.monotonic()-started<args.startup_timeout_seconds:
                    phase_sets=read_phases(raw)
                    phases=set().union(*phase_sets.values())
                    bound_sets=read_bound_modules(raw)
                    if any(REQUIRED_PHASES<=values and pid in bound_sets and expected_modules<=bound_sets[pid] for pid,values in phase_sets.items()):complete=True;break
                    if process.poll() is not None:break
                    time.sleep(0.25)
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    (raw/'collection_receipt.json').write_text(json.dumps({'collected_required_phases':complete,
        'seen_phases':sorted(phases),'seen_phases_by_pid':{str(pid):sorted(values) for pid,values in phase_sets.items()},
        'missing_phases':sorted(REQUIRED_PHASES-phases),
        'seen_source_modules_by_pid':{str(pid):sorted(values)for pid,values in bound_sets.items()},
        'runtime_tmpdir':str(temporary),'runtime_tmpdir_removed':not temporary.exists(),
        'normal_capacity_accepted':False,'timing_values_for_prediction':False},indent=2)+'\n')
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(main())
