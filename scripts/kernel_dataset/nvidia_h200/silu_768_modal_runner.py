"""Modal runner for 768-thread SiLU scalar latency measurement on H200.

Re-measures the register-resident dependent SiLU chain with 768-thread CTAs
using clock64 timing (not CUDA events). Properly binds source SHA.
Budget-bound via shared atomic reservation tool.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
TASK = Path('/tmp/midlayer-silu-768-20261108')

def _resolve_repo_root(here: Path):
    """Walk ancestors to find repo root (contains microbench/ and src/).

    Returns None when no ancestor qualifies (e.g. container /root flat layout).
    """
    for parent in [here, *here.parents]:
        try:
            if (parent / 'microbench').is_dir() and (parent / 'src').is_dir():
                return parent
        except PermissionError:
            continue
    return None


_REPO_ROOT = _resolve_repo_root(HERE)

if _REPO_ROOT is not None:
    REPO_ROOT = _REPO_ROOT
    SOURCE = REPO_ROOT / 'microbench/nvidia_h200/compute/sfu/silu_scalar_latency.cu'
else:
    REPO_ROOT = None
    SOURCE = TASK / 'silu_scalar_latency.cu'

IMAGE = (modal.Image.from_registry('pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel', add_python='3.12')
         .add_local_file(SOURCE, remote_path=str(TASK / 'silu_scalar_latency.cu')))
APP = modal.App('midlayer-silu-768-20261108-v6')

SELECTION = (
    'Independent 768-thread SiLU scalar latency measurement. '
    'Re-measures the register-resident dependent SiLU chain with 768-thread CTAs '
    'using clock64 timing (not CUDA events). '
    'No production operator, no production duration. '
    'Evidence for silu_scalar_round_768 service binding.'
)


def _validate_result(result):
    """Validate the probe result before reporting completion."""
    if not isinstance(result, dict):
        raise ValueError('result must be an object')
    status = result['status']
    if status not in ('complete', 'probe_failed', 'build_failed'):
        raise ValueError(f"unknown result status: {status}")
    if status == 'complete':
        if 'silu_768_result' not in result:
            raise ValueError('complete result missing silu_768_result')
        if 'threads=768' not in result['silu_768_result']:
            raise ValueError('silu_768_result does not contain 768-thread data')
    return True


def run(command, limit, env):
    start = time.monotonic()
    p = subprocess.run(command, cwd=TASK, env=env, capture_output=True, text=True, timeout=limit)
    return dict(command=command, exit_code=p.returncode,
                stdout=p.stdout, stderr=p.stderr,
                wall_s=time.monotonic() - start)


@APP.function(gpu='H200', image=IMAGE, timeout=570, retries=0,
              max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    out = dict(status='started', selection=SELECTION)
    env = os.environ.copy()
    env['CUDA_CACHE_PATH'] = str(TASK / 'cuda')
    env['XDG_CACHE_HOME'] = str(TASK / 'cache')

    try:
        out['gpu'] = run(['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version',
                          '--format=csv,noheader'], 20, env)
        if out['gpu']['exit_code'] or 'H200' not in out['gpu']['stdout']:
            out['status'] = 'gpu_identity_failed'
            return out

        out['nvcc_check'] = run(['nvcc', '--version'], 15, env)
        out['has_nvcc'] = out['nvcc_check']['exit_code'] == 0
        if not out['has_nvcc']:
            out['status'] = 'nvcc_unavailable'
            return out

        # Modify source to include 768 threads
        src = TASK / 'silu_scalar_latency.cu'
        src_text = src.read_text()
        if '{32, 128, 512, 1024}' not in src_text:
            out['status'] = 'source_unexpected'
            out['error'] = 'source thread list not found'
            return out
        src_text = src_text.replace(
            'for (int threads : {32, 128, 512, 1024}) {',
            'for (int threads : {32, 128, 512, 768, 1024}) {')
        src.write_text(src_text)
        out['source_modified'] = True
        out['modified_source_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()

        # Compile
        build = run(['nvcc', '-O3', '-arch=sm_90a', '-o', str(TASK / 'silu_probe'),
                     str(src), '-lcudart', '-lcuda'], 60, env)
        out['build'] = build
        if build['exit_code']:
            out['status'] = 'build_failed'
            return out

        # Run the probe
        probe = run([str(TASK / 'silu_probe')], 300, env)
        out['probe'] = probe
        if probe['exit_code']:
            out['status'] = 'probe_failed'
            return out

        # Parse output for 768-thread line
        for line in probe['stdout'].split('\n'):
            if 'threads=768 iterations=64' in line:
                out['silu_768_result'] = line.strip()
                out['status'] = 'complete'
                break
        else:
            out['status'] = 'result_not_found'

    except Exception as e:
        out['status'] = 'exception'
        out['error'] = f'{type(e).__name__}: {e}'
    finally:
        out['gpu_function_wall_s'] = time.monotonic() - start
        try:
            shutil.rmtree(TASK)
        except OSError as exc:
            out['status'] = 'cleanup_failed'
            out['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        out['remote_task_directory_removed'] = not TASK.exists()

    return out


@APP.local_entrypoint()
def main(output_dir: str, budget_key: str, budget_tool: str):
    if REPO_ROOT is None:
        raise RuntimeError('local entrypoint requires repo context')

    target = Path(output_dir).resolve()
    expected_root = REPO_ROOT / 'profiler_output/vector_silu_768_20261108'
    if not str(target).startswith(str(expected_root.resolve()) + '/'):
        raise ValueError(f'output must be under {expected_root}/')
    if target.exists():
        raise ValueError(f'destination already exists: {target}')

    # Validate source exists and record its SHA
    if not SOURCE.is_file():
        raise ValueError(f'source missing: {SOURCE}')
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    tool = Path(budget_tool)
    if not tool.is_file() or not budget_key:
        raise ValueError('shared atomic reservation required')

    target.mkdir(parents=True)
    meta = dict(
        app_id=APP.app_id,
        budget_key=budget_key,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=REPO_ROOT, text=True).strip(),
        source_sha256={
            'silu_768_modal_runner.py': hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            'silu_scalar_latency.cu': source_sha256,
        },
    )

    subprocess.run([sys.executable, str(tool), 'bind',
                    '--key', budget_key, '--app-id', APP.app_id], check=True)
    (target / 'launch.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)

    started = time.monotonic()
    try:
        result = capture.remote()
        result.update(meta)
        result['client_wall_s'] = time.monotonic() - started
        (target / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')

        # Validate result
        _validate_result(result)
        if result['status'] != 'complete':
            raise RuntimeError(f"capture failed: {result['status']}")
        if result['remote_task_directory_removed'] is not True:
            raise RuntimeError('remote task cleanup did not complete')

        summary = dict(
            status=result['status'],
            app_id=result['app_id'],
            gpu_function_wall_s=result['gpu_function_wall_s'],
            silu_768_result=result['silu_768_result'],
        )
    except Exception as exc:
        (target / 'client_failure.json').write_text(json.dumps(
            dict(**meta, error_type=type(exc).__name__, error=str(exc)),
            indent=2) + '\n')
        raise
    finally:
        # Mark reservation terminal with error propagation
        try:
            subprocess.run([sys.executable, str(tool), 'terminal',
                            '--key', budget_key], check=True)
        except Exception as exc:
            (target / 'terminal_failure.json').write_text(json.dumps(
                dict(**meta, error_type=type(exc).__name__, error=str(exc)),
                indent=2) + '\n')
            raise
    print(json.dumps(summary))
