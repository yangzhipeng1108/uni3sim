"""Modal runner for independent occupancy-latency probe on H200.

Runs a pure-hardware microbenchmark measuring L2 latency exposure
at varying warp-per-SMSP occupancy. No production operator timing.
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
PROBE = HERE / 'occupancy_latency_probe_runner.py'
PROBE_TYPE = 'occupancy_latency'
TASK = Path('/tmp/midlayer-occupancy-latency-20261108')
IMAGE = (modal.Image.from_registry('pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel', add_python='3.12')
         .add_local_file(PROBE, remote_path=str(TASK / PROBE.name)))
APP = modal.App('midlayer-occupancy-latency-20261108')

SELECTION = (
    'Independent occupancy-latency microbenchmark only. '
    'Measures L2 latency exposure of 1/2/4 independent LDG.E.128 at '
    'varying warps-per-SMSP occupancy on H200. '
    'No production operator, no production duration. '
    'Closed units: cycles, nanoseconds. '
    'Evidence for low-occupancy latency exposure model in vector predictor.'
)


def _validate_probe_result(result):
    """Validate the probe's explicit result before reporting a completed capture."""
    if not isinstance(result, dict):
        raise ValueError('probe result must be an object')
    if result['probe_type'] != PROBE_TYPE or result['gpu'] != 'H200':
        raise ValueError('probe result identity mismatch')
    if result['status'] not in ('complete', 'compile_failed'):
        raise ValueError('unknown probe result status')
    if not isinstance(result['measurements'], list):
        raise ValueError('probe measurements must be an array')
    if result['status'] == 'complete':
        if not result['measurements'] or any(
                not isinstance(row, dict) or not row
                for row in result['measurements']):
            raise ValueError('complete probe requires nonempty measurement objects')
    elif not isinstance(result['error'], str) or not result['error']:
        raise ValueError('failed compilation requires its original error')


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
    out = dict(status='started', selection=SELECTION, results=None)
    env = os.environ.copy()
    env['CUDA_CACHE_PATH'] = str(TASK / 'cuda')
    env['XDG_CACHE_HOME'] = str(TASK / 'cache')

    try:
        # Verify GPU identity
        out['gpu'] = run(['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version',
                          '--format=csv,noheader'], 20, env)
        if out['gpu']['exit_code'] or 'H200' not in out['gpu']['stdout']:
            out['status'] = 'gpu_identity_failed'
            return out

        # Check nvcc
        out['nvcc_check'] = run(['nvcc', '--version'], 15, env)
        out['has_nvcc'] = out['nvcc_check']['exit_code'] == 0

        if not out['has_nvcc']:
            out['status'] = 'nvcc_unavailable'
            return out

        # Run the probe
        output_file = TASK / 'results.json'
        out['probe_run'] = run([
            'python3', str(TASK / PROBE.name),
            '--output', str(output_file),
            '--sm-count', '132',
            '--sm-per-smsp', '4',
        ], 300, env)

        if out['probe_run']['exit_code']:
            out['status'] = 'probe_failed'
            return out

        if output_file.is_file():
            payload = output_file.read_bytes()
            out['results_sha256'] = hashlib.sha256(payload).hexdigest()
            out['results_bytes'] = len(payload)
            out['results'] = json.loads(payload)
            _validate_probe_result(out['results'])
            out['status'] = out['results']['status']
        else:
            out['status'] = 'output_missing'

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
    root = HERE.parents[2]
    target = Path(output_dir).resolve()
    expected_root = root / 'profiler_output/vector_occupancy_latency_20261108'
    if not target.is_relative_to(expected_root.resolve()):
        raise ValueError(f'output must be under {expected_root}')
    if target.exists():
        raise ValueError(f'destination already exists: {target}')

    # Verify probe script exists and has valid syntax
    if not PROBE.is_file():
        raise ValueError(f'probe script missing: {PROBE}')
    try:
        import ast
        ast.parse(PROBE.read_text())
    except SyntaxError as e:
        raise ValueError(f'probe script syntax error: {e}')

    tool = Path(budget_tool)
    if not tool.is_file() or not budget_key:
        raise ValueError('shared atomic reservation required')

    target.mkdir(parents=True)
    meta = dict(
        app_id=APP.app_id,
        budget_key=budget_key,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=root, text=True).strip(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROBE, Path(__file__))
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
        if result['status'] != 'complete':
            raise RuntimeError(f"capture failed: {result['status']}")
        _validate_probe_result(result['results'])
        if result['results']['status'] != 'complete':
            raise RuntimeError(f"probe failed: {result['results']['status']}")
        if result['remote_task_directory_removed'] is not True:
            raise RuntimeError('remote task cleanup did not complete')
        summary = dict(
            status=result['status'],
            app_id=result['app_id'],
            gpu_function_wall_s=result['gpu_function_wall_s'],
            measurements_count=len(result['results']['measurements']),
        )
    except Exception as exc:
        (target / 'client_failure.json').write_text(json.dumps(
            dict(**meta, error_type=type(exc).__name__, error=str(exc)),
            indent=2) + '\n')
        raise
    finally:
        # Mark reservation terminal regardless of outcome (cannot cancel after execution)
        try:
            subprocess.run([sys.executable, str(tool), 'terminal',
                            '--key', budget_key], check=True)
        except Exception as exc:
            (target / 'terminal_failure.json').write_text(json.dumps(
                dict(**meta, error_type=type(exc).__name__, error=str(exc)),
                indent=2) + '\n')
            raise
    print(json.dumps(summary))
