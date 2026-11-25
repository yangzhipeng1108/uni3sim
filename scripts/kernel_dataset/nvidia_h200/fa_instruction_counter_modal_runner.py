"""Three source-matched native FA non-duration instruction counter points.

Raw diagnostic capture only. Canonical timings and production predictors
are neither imported nor changed. The selected main-kernel names/Dim3 and
input arguments come from the retained fixed-case source plans.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import modal

TASK = Path('/tmp/midlayer-fa-instruction-20261107')
REMOTE = TASK / 'repo'
ROOT = REMOTE if REMOTE.is_dir() else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (
    FA_INSTRUCTION_COUNTER_METRICS, extract_fa_instruction_counters)

CAMPAIGN = ROOT / 'profiler_output/kernel_dataset/nvidia_h200/fa_instruction_delivery_20261107'
if REMOTE.is_dir():
    IMAGE = modal.Image.debian_slim()
else:
    IMAGE = (
        modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
        .entrypoint([])
        .apt_install('nsight-systems-2025.6.3', 'cuda-nsight-compute-12-9')
        .run_commands(
            "python -m pip install --no-cache-dir "
            "'torch==2.13.0+cu129' 'vllm==0.30.0+cu129' "
            "'triton==3.7.1' 'flashinfer-python==0.6.18.post1' pandas tqdm zstandard "
            "--extra-index-url https://download.pytorch.org/whl/cu129 "
            "--extra-index-url https://wheels.vllm.ai/0.30.0/cu129/"
        )
        .run_commands(
            "python -m pip install --no-cache-dir --force-reinstall --no-deps "
            "'torchcodec==0.16.0+cu129' --index-url https://download.pytorch.org/whl/cu129"
        )
    )
    local_manifest = json.loads((CAMPAIGN / 'source_manifest.json').read_text())
    for item in local_manifest['files']:
        path = ROOT / item['path']
        if (Path(item['path']).is_absolute() or '..' in Path(item['path']).parts
                or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']):
            raise ValueError('Native counter source snapshot changed before launch')
        IMAGE = IMAGE.add_local_file(path, remote_path=str(REMOTE / item['path']))
    IMAGE = IMAGE.add_local_file(CAMPAIGN / 'plan.json', remote_path=str(TASK / 'plan.json'))
    IMAGE = IMAGE.add_local_file(CAMPAIGN / 'source_manifest.json', remote_path=str(TASK / 'source_manifest.json'))

APP = modal.App('midlayer-fa-instruction-counters-20261107')


def _run(command, environment, timeout):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=REMOTE, env=environment, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return {'command': command, 'exit_code': process.returncode,
            'timed_out': timed_out, 'stdout': stdout, 'stderr': stderr}


@APP.function(gpu='H200', image=IMAGE, timeout=360, retries=0, single_use_containers=True)
def capture():
    import zstandard

    started = time.monotonic()
    result = {'cases': [], 'reports': {}, 'source_files': {}}
    try:
        manifest = json.loads((TASK / 'source_manifest.json').read_text())
        plan_bytes = (TASK / 'plan.json').read_bytes()
        if hashlib.sha256(plan_bytes).hexdigest() != manifest['plan_sha256']:
            raise ValueError('Native instruction counter input changed')
        plan = json.loads(plan_bytes)
        if (len(plan['cases']) != 3 or [c['case_id'] for c in plan['cases']] != manifest['case_ids']
                or plan['service_metrics'] != list(FA_INSTRUCTION_COUNTER_METRICS)
                or plan['traffic_metrics'] != []):
            raise ValueError('Native instruction counter scope changed')
        for item in manifest['files']:
            path = REMOTE / item['path']
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Native instruction counter source changed')
            result['source_files'][item['path']] = content
        result['plan_sha256'] = manifest['plan_sha256']
        versions = {name: importlib.metadata.version(name) for name in (
            'torch', 'vllm', 'triton', 'flashinfer-python', 'nvidia-cublas-cu12')}
        result['versions'] = versions
        if versions != {'torch': '2.13.0+cu129', 'vllm': '0.30.0+cu129',
                        'triton': '3.7.1', 'flashinfer-python': '0.6.18.post1',
                        'nvidia-cublas-cu12': '12.9.1.4'}:
            raise ValueError('Native FA software differs from retained campaign')
        environment = dict(os.environ)
        for key, folder in (
                ('TORCHINDUCTOR_CACHE_DIR', 'torch'), ('TRITON_CACHE_DIR', 'triton'),
                ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg'),
                ('VLLM_CACHE_ROOT', 'vllm'), ('HF_HOME', 'hf'), ('TMPDIR', 'tmp')):
            path = TASK / 'cache' / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment['CUDA_VISIBLE_DEVICES'] = '0'
        environment['VLLM_NO_USAGE_STATS'] = '1'
        environment['DO_NOT_TRACK'] = '1'
        result['identity'] = _run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], environment, 20)
        if result['identity']['exit_code'] != 0 or not result['identity']['stdout'].startswith('NVIDIA H200,'):
            raise ValueError('Native instruction counter expected one H200')
        result['profiler'] = _run(['ncu', '--version'], environment, 20)
        for case in plan['cases']:
            name = case['case_id']
            directory = TASK / 'raw' / name
            directory.mkdir(parents=True)
            item = {'case_id': name, 'status': 'started'}
            result['cases'].append(item)
            stem = directory / 'profile'
            command = [
                'ncu', '--nvtx', '--nvtx-include', case['plan']['marker'] + '/',
                '--kernel-name-base', 'demangled', '--kernel-name', 'regex:FlashAttnFwdSm90',
                '--clock-control', 'none', '--cache-control', 'none', '--section', 'SourceCounters',
                '--csv', '--page', 'raw', '--metrics', ','.join(FA_INSTRUCTION_COUNTER_METRICS),
                '--export', str(stem), sys.executable, str(REMOTE / case['plan']['runner_relpath']),
                *case['plan']['runner_args'], '--h200-gpu-lock', str(TASK / 'gpu.lock'),
                '--h200-telemetry-output', str(directory / 'telemetry.json'), '--h200-physical-gpu', '0']
            item['capture'] = _run(command, environment, 90)
            report = stem.with_suffix('.ncu-rep')
            if report.is_file():
                result['reports'][name + '/profile.ncu-rep.zst'] = zstandard.ZstdCompressor(level=3).compress(report.read_bytes())
            telemetry = directory / 'telemetry.json'
            if telemetry.is_file():
                item['telemetry'] = json.loads(telemetry.read_text())
            if item['capture']['exit_code'] != 0 or not report.is_file():
                item['status'] = 'capture_failed'
                break
            try:
                item['non_duration_launches'] = extract_fa_instruction_counters(item['capture']['stdout'], case)
            except (ValueError, TypeError) as error:
                item['status'] = 'counter_validation_failed'
                item['validation_error'] = str(error)
                break
            item['source_export'] = _run(
                ['ncu', '--import', str(report), '--page', 'source', '--print-source', 'sass'], environment, 20)
            if item['source_export']['exit_code'] != 0:
                item['status'] = 'source_export_failed'
                break
            item['status'] = 'raw_diagnostic_complete'
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str):
    out = Path(output_dir).resolve()
    if not out.is_relative_to(CAMPAIGN.resolve()) or out.exists():
        raise ValueError('Native instruction counters require a new campaign attempt directory')
    result = capture.remote()
    out.mkdir(parents=True)
    reports = {}
    for name, content in result['reports'].items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        reports[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
    result['reports'] = reports
    sources = {}
    for name, content in result['source_files'].items():
        path = out / 'source_snapshot' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        sources[name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
    result['source_files'] = sources
    result['modal_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    if len(result['cases']) != 3 or any(c['status'] != 'raw_diagnostic_complete' for c in result['cases']):
        raise RuntimeError('Native instruction counter capture incomplete; raw retained')
    print({'wall_s': result['gpu_function_wall_s'], 'cases': [c['case_id'] for c in result['cases']], 'status': 'raw_only'})
