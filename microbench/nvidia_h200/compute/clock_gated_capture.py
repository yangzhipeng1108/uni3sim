#!/usr/bin/env python3
"""Clock-gated capture helper for the H200 WP5 vector/reduction probe runners.

The H200 container cannot lock the SM clock, so every timed capture of the
owner-copy probes goes through ``run_gated``:

1. require the physical GPU to be an ``NVIDIA H200`` with no compute process;
2. start ``nvidia-smi -i <gpu> --query-gpu=timestamp,clocks.sm,utilization.gpu
   -lms 50`` into ``<name>.clock.csv``;
3. spin ``gpu_clock_warmup`` for ``WARMUP_MS`` so the clock leaves the 345-MHz
   idle state, then run the probe command(s) with ``CUDA_VISIBLE_DEVICES`` set to
   the owned GPU;
4. keep only samples whose timestamp lies inside the probe window.  The capture
   is accepted when at least one busy sample (utilization > 0) exists and every
   busy sample reads >= 1980 MHz.  A window without a busy sample (a very short
   probe) is accepted only under the stricter rule that every sample in the
   window, busy or not, reads >= 1980 MHz.  Rejected attempts are kept on disk
   and counted; ``MAX_ATTEMPTS`` rejections of the same capture fail loudly.

Only the Python standard library is used because the H200 host's system
``python3`` has no third-party packages.
"""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

BOOST_MHZ = 1980
EXPECTED_DEVICE_NAME = 'NVIDIA H200'
EXPECTED_SM_COUNT = 132
WARMUP_MS = 600
MAX_ATTEMPTS = 4
NVCC = Path(shutil.which('nvcc') or 'nvcc')
HELPER_DIR = Path(__file__).resolve().parent
WARMUP_SOURCE = HELPER_DIR / 'gpu_clock_warmup.cu'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nvcc_version():
    result = subprocess.run([str(NVCC), '--version'], check=True, text=True,
                            stdout=subprocess.PIPE)
    return result.stdout.strip().splitlines()[-2]


def build(source, binary, extra_flags=()):
    """Compile ``source`` natively for sm_90 and return build provenance."""
    binary = Path(binary)
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [str(NVCC), '-O3', '-std=c++17', '-arch=sm_90', '-lineinfo',
               '-Xptxas=-v', *extra_flags, str(source), '-o', str(binary)]
    result = subprocess.run(command, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (binary.parent / f'{binary.name}.ptxas.txt').write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'nvcc failed for {source}:\n{result.stderr}')
    return {
        'source': str(source), 'source_sha256': sha256(source),
        'binary': str(binary), 'binary_sha256': sha256(binary),
        'compile_command': ' '.join(command), 'nvcc_version': nvcc_version(),
        'ptxas_verbose': result.stderr,
    }


def build_warmup(build_dir):
    binary = Path(build_dir) / 'gpu_clock_warmup'
    if not binary.exists():
        build(WARMUP_SOURCE, binary)
    return binary


def device_identity(device):
    result = subprocess.run(
        ['nvidia-smi', '-i', str(device),
         '--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.used',
         '--format=csv,noheader,nounits'],
        check=True, text=True, stdout=subprocess.PIPE)
    fields = [value.strip() for value in result.stdout.strip().split(',')]
    identity = {'index': int(fields[0]), 'name': fields[1], 'uuid': fields[2],
                'pci_bus_id': fields[3], 'driver_version': fields[4],
                'memory_used_mib': int(fields[5])}
    if identity['index'] != device or identity['name'] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f'unexpected target GPU: {identity}')
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
         '--format=csv,noheader,nounits'],
        check=True, text=True, stdout=subprocess.PIPE).stdout
    for line in apps.splitlines():
        if line.split(',')[0].strip() == identity['uuid']:
            raise RuntimeError(f'target GPU has a compute process: {line}')
    return identity


def _parse_clock_log(path, begin, end):
    samples = []
    for row in csv.reader(Path(path).read_text().splitlines()):
        if len(row) != 3:
            continue
        stamp = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
        if begin <= stamp <= end:
            samples.append((int(row[1]), int(row[2])))
    return samples


def _clock_verdict(samples):
    busy = [clock for clock, utilization in samples if utilization > 0]
    window = [clock for clock, _ in samples]
    if busy:
        accepted = min(busy) >= BOOST_MHZ
        rule = 'busy_samples_at_boost'
    else:
        accepted = bool(window) and min(window) >= BOOST_MHZ
        rule = 'no_busy_sample_every_window_sample_at_boost'
    return {
        'accepted': accepted, 'rule': rule,
        'window_samples': len(window), 'busy_samples': len(busy),
        'busy_clock_mhz_min': min(busy) if busy else None,
        'busy_clock_mhz_max': max(busy) if busy else None,
        'busy_clock_mhz_median': sorted(busy)[len(busy) // 2] if busy else None,
        'window_clock_mhz_min': min(window) if window else None,
    }


def _run_once(commands, device, warmup_binary, clock_log, timeout):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    with Path(clock_log).open('w') as log:
        sampler = subprocess.Popen(
            ['nvidia-smi', '-i', str(device),
             '--query-gpu=timestamp,clocks.sm,utilization.gpu',
             '--format=csv,noheader,nounits', '-lms', '50'], stdout=log)
    try:
        time.sleep(0.2)
        subprocess.run([str(warmup_binary), str(WARMUP_MS)], check=True,
                       env=environment, timeout=60)
        begin = datetime.datetime.now()
        for argv, stdout_path in commands:
            with Path(stdout_path).open('w') as stdout, \
                    Path(f'{stdout_path}.stderr').open('w') as stderr:
                subprocess.run([str(item) for item in argv], check=True,
                               env=environment, stdout=stdout, stderr=stderr,
                               timeout=timeout)
        end = datetime.datetime.now()
        time.sleep(0.15)
    finally:
        sampler.terminate()
        sampler.wait()
    verdict = _clock_verdict(_parse_clock_log(clock_log, begin, end))
    verdict['wall_seconds'] = (end - begin).total_seconds()
    return verdict


def run_gated(name, commands, out_dir, device, warmup_binary, timeout=3600):
    """Run ``commands`` ([(argv, stdout_path)]) until one attempt passes the gate.

    Returns the accepted attempt's verdict with the rejected attempts attached.
    Each attempt writes into ``out_dir/<name>_a<k>``-suffixed paths; callers pass
    stdout paths built with ``attempt_path``.
    """
    rejected = []
    for attempt in range(MAX_ATTEMPTS):
        attempt_commands = [
            (argv, Path(out_dir) / f'{Path(path).stem}.a{attempt}{Path(path).suffix}')
            for argv, path in commands]
        clock_log = Path(out_dir) / f'{name}.a{attempt}.clock.csv'
        verdict = _run_once(attempt_commands, device, warmup_binary, clock_log,
                            timeout)
        verdict['attempt'] = attempt
        verdict['stdout_paths'] = [str(path) for _, path in attempt_commands]
        verdict['clock_log'] = str(clock_log)
        if verdict['accepted']:
            verdict['rejected_attempts'] = rejected
            return verdict
        rejected.append(verdict)
        time.sleep(2.0)
    raise RuntimeError(f'{name}: {MAX_ATTEMPTS} clock-rejected attempts: '
                       f'{json.dumps(rejected)}')


def clock_summary(verdicts):
    """Aggregate accepted/rejected counts and busy-clock statistics."""
    accepted = len(verdicts)
    rejected = sum(len(verdict['rejected_attempts']) for verdict in verdicts)
    busy_min = [v['busy_clock_mhz_min'] for v in verdicts
                if v['busy_clock_mhz_min'] is not None]
    busy_max = [v['busy_clock_mhz_max'] for v in verdicts
                if v['busy_clock_mhz_max'] is not None]
    return {
        'gate': f'every busy nvidia-smi sample (50 ms) >= {BOOST_MHZ} MHz; a '
                'window without busy samples requires every sample >= boost',
        'accepted_captures': accepted, 'rejected_captures': rejected,
        'captures_without_busy_sample': sum(
            1 for v in verdicts if v['busy_samples'] == 0),
        'busy_samples_total': sum(v['busy_samples'] for v in verdicts),
        'busy_clock_mhz_min': min(busy_min) if busy_min else None,
        'busy_clock_mhz_max': max(busy_max) if busy_max else None,
        'measured_sm_clock_mhz': measured_clock_mhz(verdicts),
    }


def measured_clock_mhz(verdicts):
    """Median busy SM clock over accepted captures (the cycle<->ns clock)."""
    medians = sorted(v['busy_clock_mhz_median'] for v in verdicts
                     if v['busy_clock_mhz_median'] is not None)
    if not medians:
        raise RuntimeError('no busy clock sample in any accepted capture')
    return medians[len(medians) // 2]
