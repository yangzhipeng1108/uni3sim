"""Independent hardware-clock sweep for a generic panel/WGMMA program."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone
import modal

HERE = Path(__file__).resolve().parent
TASK = Path('/tmp/midlayer-h200-panel-clock-20261107')
if HERE.name == 'tensor_core':
    SOURCE = HERE / 'panel_wgmma_clock.cu'
    GENERATOR = HERE / 'wgmma_ss_wait1_issue_gen.py'
    REFERENCE = HERE.parents[1] / 'operators/flash_attn/fa3_wgmma_service.cu'
else:
    SOURCE = TASK / 'panel_wgmma_clock.cu'
    GENERATOR = TASK / 'wgmma_ss_wait1_issue_gen.py'
    REFERENCE = TASK / 'fa3_wgmma_service.cu'
IMAGE = modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
for local in (SOURCE, GENERATOR, REFERENCE):
    IMAGE = IMAGE.add_local_file(local, remote_path=str(TASK / local.name))
APP = modal.App('midlayer-h200-panel-clock-20261107')

def run(command, timeout):
    start = time.monotonic()
    result = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=timeout)
    return {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr, 'wall_s': time.monotonic() - start}

@APP.function(gpu='H200', image=IMAGE, timeout=780, retries=0, max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    result = {'boundary': 'independent_panel_ring_clock64_globaltimer_no_production_operator',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'frozen_rule':'Independent physical active-clock interval, not a stable typical-clock claim: retain all54cells across copy/matrix/combined modes, grids132/2112/65536, rounds8/32/128,16/128MiB. Two63-sample processes,2s matrix warmup each. Every matrix/combined cell requires95% distribution-free median CI within3% and process medians within3%; no cell dropping. For a conservative optimistic cycle conversion, use maximum upper median-CI endpoint across all matrix/combined cells and processes; upper endpoints from the two processes must agree within3%. Across-cell clock range is reported uncertainty, never a per-grid parameter table. Copy controls do not select the active clock. No production/evaluation duration or error selection; old failed constant-typical-clock gate is not changed.'}
    try:
        commands = (
            ('gpu', ['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20),
            ('nvcc', ['nvcc', '--version'], 20),
            ('generate', ['python3', GENERATOR.name, REFERENCE.name, 'wgmma_ss_helper.cuh'], 20),
            ('build', ['nvcc', '-O3', '-std=c++17', '-gencode=arch=compute_90a,code=sm_90a', '-Xptxas=-v', SOURCE.name, '-o', 'probe'], 120),
            ('resources', ['cuobjdump', '--dump-resource-usage', 'probe'], 30),
            ('sass', ['cuobjdump', '--dump-sass', 'probe'], 45),
            ('rotation0', [str(TASK / 'probe'),'0'], 300),
            ('rotation1', [str(TASK / 'probe'),'1'], 300),
            ('gpu_after', ['nvidia-smi', '--query-gpu=name,clocks.sm,clocks.mem,driver_version', '--format=csv,noheader'], 20))
        for name, command, timeout in commands:
            result[name] = run(command, timeout)
            if result[name]['exit_code']:
                result['status'] = name + '_failed'
                return result
        result['binary_sha256'] = hashlib.sha256((TASK / 'probe').read_bytes()).hexdigest()
        result['generated_header_sha256'] = hashlib.sha256((TASK / 'wgmma_ss_helper.cuh').read_bytes()).hexdigest()
        result['status'] = 'complete'
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - start
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    campaign = root / 'profiler_output/microbench/nvidia_h200/panel_wgmma_clock_20261107'
    target = Path(output_dir).resolve()
    if target.parent != campaign.resolve() or target.name not in ('attempt_3',) or target.exists():
        raise ValueError('output must be a new registered attempt_3')
    start = time.monotonic()
    result = capture.remote()
    result['app_id'] = APP.app_id
    result['client_wall_s'] = time.monotonic() - start
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (SOURCE, GENERATOR, REFERENCE)}
    result['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True)
    (target / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'app_id': result['app_id'],
                      'gpu_function_wall_s': result['gpu_function_wall_s'], 'output': str(target)}))
