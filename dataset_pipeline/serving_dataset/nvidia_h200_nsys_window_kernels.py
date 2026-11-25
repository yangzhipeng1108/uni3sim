"""Read-only aggregation of one nsys window (sqlite export), device 0, for the family tables.
Run beside the operator-selected trace inputs.

Same output contract as the TP4 ``aggregate_remote_trace_v2.py`` but read
straight from CUPTI_ACTIVITY_KIND_KERNEL: per (name, grid, block) calls, raw
ns, overlap-shared ns (each busy instant split equally among the kernels
running then; sums to the busy union), streams; idle gaps between busy
segments; and the window span.  Evaluation only.
Usage: python aggregate_nvidia_h200_nsys_window_kernels.py <window.sqlite> > window_dev0_share.json
"""
import collections, json, sqlite3, sys



def aggregate_window(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    names = dict(db.execute('select id, value from StringIds'))
    rows = db.execute(
        'select start, end, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId, deviceId '
        'from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    devices = sorted({r[10] for r in rows})
    dev = devices[0]
    kernels = []
    for s, e, n, gx, gy, gz, bx, by, bz, st, d in rows:
        if d != dev:
            continue
        key = '|'.join([names[n], f'{gx}x{gy}x{gz}', f'{bx}x{by}x{bz}'])
        kernels.append((s, e, key, st))
    kernels.sort()
    events = []
    for i, (s, e, k, st) in enumerate(kernels):
        events.append((s, 1, i)); events.append((e, 0, i))
    events.sort()
    share = collections.defaultdict(float)
    active = set(); last_t = None
    idle = collections.defaultdict(lambda: [0, 0])
    last_end_key = None
    for t, typ, i in events:
        if last_t is not None and active:
            dt = (t - last_t) / len(active)
            for j in active:
                share[kernels[j][2]] += dt
        if typ == 1:
            if not active and last_t is not None and last_end_key is not None and t > last_t:
                gk = last_end_key.split('|')[0][:80] + ' -> ' + kernels[i][2].split('|')[0][:80]
                idle[gk][0] += 1; idle[gk][1] += t - last_t
            active.add(i)
        else:
            active.discard(i)
            if not active:
                last_end_key = kernels[i][2]
        last_t = t
    agg = collections.OrderedDict()
    for s, e, k, st in kernels:
        if k not in agg:
            agg[k] = [0, 0, set()]
        agg[k][0] += 1; agg[k][1] += e - s; agg[k][2].add(st)
    return {'sqlite': sqlite_path, 'device': dev, 'devices': devices, 'n_kernels': len(kernels),
               'span_ns': kernels[-1][1] - kernels[0][0], 'first_start_ns': kernels[0][0],
               'fused_moe_calls': sum(1 for k in kernels if k[2].startswith('fused_moe_kernel|')),
               'agg': [[k, c, t, share[k], sorted(sts)] for k, (c, t, sts) in agg.items()],
               'idle': sorted([[k, c, t] for k, (c, t) in idle.items()], key=lambda r: -r[2])[:60]}



def main():
    json.dump(aggregate_window(sys.argv[1]), sys.stdout)
