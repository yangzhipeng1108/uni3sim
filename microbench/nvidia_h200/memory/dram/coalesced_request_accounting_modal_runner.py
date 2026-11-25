"""Independent upstream/downstream accounting contrast; no timing publication."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'finite_strided_panel.cu'
TASK = Path('/tmp/midlayer-coalesced-accounting-20261108')
IMAGE = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
         .apt_install('cuda-nsight-compute-12-9')
         .add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name)))
APP = modal.App('midlayer-coalesced-accounting-20261108')
CELLS = tuple((s, g, f) for s, g in ((1024,24544),(1536,3064),(4096,25338)) for f in (1,2))
METRICS = ('sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg.per_second,'
           'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,'
           'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,'
           'lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_op_read.sum,'
           'lts__t_sectors_srcunit_ltcfabric_op_read.sum,dram__bytes_read.sum,'
           'sm__sass_inst_executed_op_ldgsts_cache_bypass.sum,sm__ctas_launched.sum')
SELECTION = ('Counter-only causal contrast, frozen before capture: same independent copy kernel at all six largest-grid/stride fanout1/2points. '
             'No new body duration or rate selected. Upstream L1TEX sectors must conserve all source logical B+validA bytes within1%, '
             'both fanouts must conserve unique DRAM within1% and exact CTA/LDGSTS counts, downstream TEX must lie between unique and logical bytes within1%. '
             'Every stride must show fanout2 TEX<97%logical while upstream remains within1%, proving counter-layer distinction. '
             'Direct SM/LTS clocks stable3%,SM nominal3%. Every frozen cell required. Prior failed TEX==logical gate remains failed. '
             'Only if this independent conservation proof passes may the earlier qualified process/precision points be reconsidered under '
             'upstream request conservation, preserving every old capability and formula. No production data,duration or error is consumed.')


def run(command, timeout):
    start = time.monotonic()
    p = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=timeout)
    return dict(command=command, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr, wall_s=time.monotonic()-start)


@APP.function(gpu='H200', image=IMAGE, timeout=270, retries=0, max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    out = dict(status='started', selection=SELECTION, physical_cells=CELLS, counters=[])
    try:
        for name, command, timeout in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],15),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-DMIDLAYER_COALESCED_SCOPE_ONLY','-DMIDLAYER_FINITE_SOURCE_GIB=32','-DMIDLAYER_FINITE_MAX_GRID=25338',SOURCE.name,'-o','probe'],60)):
            out[name] = run(command, timeout)
            if out[name]['exit_code']:
                out['status'] = name+'_failed'
                return out
        for stride, grid, fanout in CELLS:
            (TASK/'cell.csv').write_text(f'{stride},{grid},{fanout}\n')
            result = run(['ncu','--csv','--page','raw','--print-units','base','--kernel-name','regex:copy_panels',
                          '--launch-skip','3','--launch-count','1','--replay-mode','application','--cache-control','none',
                          '--clock-control','none','--metrics',METRICS,str(TASK/'probe'),'44','0',str(TASK/'cell.csv'),'1'],30)
            out['counters'].append(dict(stride=stride,grid=grid,fanout=fanout,measurement=result))
            if result['exit_code']:
                out['status'] = 'counter_failed'
                return out
        out['status'] = 'complete'
        return out
    finally:
        out['gpu_function_wall_s'] = time.monotonic()-start
        shutil.rmtree(TASK)
        out['remote_task_directory_removed'] = not TASK.exists()


@APP.local_entrypoint()
def main(output_dir:str):
    root = HERE.parents[3]
    target = Path(output_dir).resolve()
    if target != (root/'profiler_output/microbench/nvidia_h200/coalesced_request_accounting_20261108/attempt_1').resolve() or target.exists():
        raise ValueError('requires new registered accounting destination')
    start = time.monotonic()
    out = capture.remote()
    out['client_wall_s'] = time.monotonic()-start
    out['app_id'] = APP.app_id
    out['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    out['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))}
    target.mkdir(parents=True)
    (target/'raw.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],app_id=out['app_id'],gpu_function_wall_s=out['gpu_function_wall_s'])))
