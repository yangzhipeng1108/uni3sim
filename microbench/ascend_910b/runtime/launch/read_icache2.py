"""Warm vs cold AIV launch cost, from msprof task time plus the icache PMU.

WARM issues one opcode 30 times in a row; COLD interleaves it with a second
opcode whose body lies elsewhere in the same text section.  Launch geometry,
element count and iteration count are identical, so the only difference is
which code the AIV must fetch.  aiv_icache_miss_rate arbitrates whether the
interleaving actually evicted anything.
"""
import sqlite3, glob, statistics, sys
from collections import defaultdict


def rows(root):
    p = glob.glob(root + "/device_*/sqlite/ai_core_op_summary.db")[0]
    con = sqlite3.connect("file:" + p + "?mode=ro", uri=True)
    shape = {}
    for tid, sid, bid, ish in con.execute(
            "select task_id,stream_id,batch_id,input_shapes from ge_summary"):
        shape[(tid, sid, bid)] = ish
    met = {}
    for r in con.execute("select task_id,stream_id,batch_id,"
                         "aiv_total_time,aiv_icache_miss_rate "
                         "from ai_core_metrics"):
        met[(r[0], r[1], r[2])] = (r[3], r[4])
    out = []
    for tid, sid, bid, st, dur in con.execute(
            "select task_id,stream_id,batch_id,start_time,duration_time "
            "from task_time"):
        k = (tid, sid, bid)
        if k in met and k in shape:
            out.append((float(st), shape[k], float(dur)) + met[k])
    out.sort()
    return out


def report(label, root):
    data = rows(root)
    byshape = defaultdict(list)
    for st, sh, dur, tot, ic in data:
        byshape[sh].append((dur, tot, ic))
    print("== %s   launches=%d" % (label, len(data)))
    for sh, v in sorted(byshape.items(), key=lambda kv: -len(kv[1])):
        tail = v[2:]  # drop the process's own first cold pair
        if not tail:
            continue
        print("   shape=%-16s n=%3d  task_ns=%8.1f  aiv_total=%7.3f  "
              "icache=%9.6f"
              % (sh[:16], len(tail),
                 statistics.median(x[0] for x in tail),
                 statistics.median(x[1] for x in tail),
                 statistics.median(x[2] for x in tail)))
    first = data[0]
    print("   process first launch: task_ns=%.1f icache=%.6f"
          % (first[2], first[4]))
    return byshape


if __name__ == "__main__":
    w = report("WARM (same opcode repeated)", sys.argv[1])
    c = report("COLD (two opcodes interleaved)", sys.argv[2])
