"""Run cuda_graph_node_gap under nsys node tracing and reduce the dependent-node gap.

Mechanism probe (no production workload): builds cuda_graph_node_gap.cu,
runs it under ``nsys profile --cuda-graph-trace=node``, exports sqlite and,
per configuration (grid CTAs, threads per CTA, busy-wait ns), reads every
replay's node sequence from CUPTI_ACTIVITY_KIND_KERNEL and reports the
dependent-node gap start(i+1) - end(i) (inside a replay only) and the node
duration.  The first 3 replays of each configuration are warm-up and
discarded.  SM clock is sampled before and after; windows whose samples are
not 1980 MHz are marked.  Output: <out>/gap_summary.json + raw sqlite/csv.
Usage: python cuda_graph_node_gap_runner.py --out <dir> --gpu <physical index>
"""
import argparse, json, os, shutil, sqlite3, statistics, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYS = shutil.which('nsys') or 'nsys'
NVCC = shutil.which('nvcc') or 'nvcc'
GRIDS = '1,16,66,132,264,528,1056'
THREADS = '128,256,384'
WAITS = '0,1000,2000,5000,10000,20000'
# Kernel-switch pass: alternate two distinct kernels, the second with these
# dynamic shared-memory sizes (bytes).
SWITCH_GRIDS = '16,132,528'
SWITCH_THREADS = '128,384'
SWITCH_WAITS = '2000,10000'
SWITCH_SMEM = '0,16384,65536,131072,200000'
NODES, REPLAYS = 64, 20


def clock(gpu):
    out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=clocks.sm,clocks.mem',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    sm, mem = (int(x) for x in out.stdout.strip().split(','))
    return {'sm_clock_mhz': sm, 'memory_clock_mhz': mem}


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--switch', action='store_true',
                    help='run the kernel-switch pass instead of identical nodes')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    binary = a.out / 'cuda_graph_node_gap'
    subprocess.run([NVCC, '-O3', '-arch=sm_90', '-cudart', 'static', '-o', str(binary),
                    str(HERE / 'cuda_graph_node_gap.cu')], check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    before = clock(a.gpu)
    sampler = subprocess.Popen(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=timestamp,clocks.sm',
                                '--format=csv,noheader,nounits', '-lms', '50'],
                               stdout=open(a.out / 'clock_samples.csv', 'w'))
    rep = a.out / 'trace'
    run = subprocess.run([NSYS, 'profile', '-t', 'cuda', '--cuda-graph-trace=node', '--sample=none',
                          '--cpuctxsw=none', '-o', str(rep), '--force-overwrite', 'true',
                          str(binary), str(NODES), str(REPLAYS),
                          *((SWITCH_GRIDS, SWITCH_THREADS, SWITCH_WAITS, SWITCH_SMEM) if a.switch
                            else (GRIDS, THREADS, WAITS))],
                         env=env, capture_output=True, text=True, check=True)
    sampler.terminate()
    sampler.wait()
    after = clock(a.gpu)
    samples = [int(line.split(',')[1]) for line in (a.out / 'clock_samples.csv').read_text().splitlines()
               if line.count(',') == 1]
    (a.out / 'stdout.csv').write_text(run.stdout)
    subprocess.run([NSYS, 'export', '--type', 'sqlite', '--force-overwrite', 'true',
                    '--output', str(rep) + '.sqlite', str(rep) + '.nsys-rep'], check=True,
                   capture_output=True)
    configs = [line.split(',') for line in run.stdout.splitlines()[1:] if line and line[0].isdigit()]
    db = sqlite3.connect(str(rep) + '.sqlite')
    kernels = db.execute('select start, end, gridX, blockX, dynamicSharedMemory from '
                         'CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
    per_config = NODES * (3 + REPLAYS)
    if len(kernels) != per_config * len(configs):
        raise RuntimeError(f'{len(kernels)} kernels, expected {per_config * len(configs)}')
    rows = []
    for i, (idx, grid, block, wait, nodes, replays, alt_smem) in enumerate(configs):
        chunk = kernels[i * per_config:(i + 1) * per_config]
        if {(k[2], k[3]) for k in chunk} != {(int(grid), int(block))}:
            raise RuntimeError(f'config {idx}: trace geometry does not match')
        gaps, durs = [], []
        for r in range(3, 3 + REPLAYS):
            seq = chunk[r * NODES:(r + 1) * NODES]
            durs += [e - s for s, e, *_ in seq]
            gaps += [seq[j + 1][0] - seq[j][1] for j in range(NODES - 1)]
        rows.append({'grid_ctas': int(grid), 'threads_per_cta': int(block), 'wait_ns': int(wait),
                     'alt_smem_bytes': int(alt_smem),
                     'samples': len(gaps), 'gap_ns_median': statistics.median(gaps),
                     'gap_ns_p10': pct(gaps, 0.1), 'gap_ns_p90': pct(gaps, 0.9),
                     'node_duration_ns_median': statistics.median(durs)})
    summary = {'mechanism': 'cuda_graph_dependent_node_gap', 'hardware_id': 'nvidia_h200',
               'timer_domain': 'nsys_cuda_graph_node_trace', 'nodes_per_graph': NODES,
               'traced_replays_per_config': REPLAYS, 'warmup_replays_discarded': 3,
               'clock_before': before, 'clock_after': after,
               'clock_samples_mhz_counts': {str(v): samples.count(v) for v in sorted(set(samples))},
               'production_or_evaluation_duration_consumed': False, 'rows': rows}
    (a.out / 'gap_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
