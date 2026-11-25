"""Cold vs warm complete non-empty AIV memory task floor.

One msprof capture per process: launch 0 executes the kernel text for the
first time, launches 1.. re-execute it.  Geometry and work are identical, so
the pair isolates the instruction-fetch state; aiv_icache_miss_rate arbitrates.
"""
import glob
import statistics
import sqlite3
import sys
from collections import defaultdict

PUBLISHED_SUBMIT_NS = 895.9751773049647
PUBLISHED_DESC_NS = 91.4567954841511


def launches(root):
    p = glob.glob(root + "/PROF_*/device_*/sqlite/ai_core_op_summary.db")
    if not p:
        return []
    con = sqlite3.connect("file:" + p[0] + "?mode=ro", uri=True)
    met = {}
    for r in con.execute("select task_id,stream_id,batch_id,"
                         "aiv_icache_miss_rate from ai_core_metrics"):
        met[(r[0], r[1], r[2])] = r[3]
    out = []
    for tid, sid, bid, st, dur in con.execute(
            "select task_id,stream_id,batch_id,start_time,duration_time "
            "from task_time"):
        k = (tid, sid, bid)
        if k in met:
            out.append((float(st), float(dur), met[k]))
    out.sort()
    return out


def main(base):
    cold, warm, icold, iwarm = (defaultdict(list) for _ in range(4))
    for d in sorted(glob.glob(base + "/bd*_r*")):
        bd = int(d.rsplit("/", 1)[-1].split("_")[0][2:])
        rows = launches(d)
        if len(rows) < 3:
            continue
        cold[bd].append(rows[0][1])
        icold[bd].append(rows[0][2])
        for r in rows[1:]:
            warm[bd].append(r[1])
            iwarm[bd].append(r[2])
    print("%4s %5s %12s %12s %5s %12s %12s %10s"
          % ("bd", "n", "cold_ns", "cold_icache", "n", "warm_ns",
             "warm_icache", "published"))
    pts = []
    for bd in sorted(cold):
        c = statistics.median(cold[bd])
        w = statistics.median(warm[bd])
        pub = PUBLISHED_SUBMIT_NS + PUBLISHED_DESC_NS * bd
        pts.append((bd, c))
        print("%4d %5d %12.1f %12.6f %5d %12.1f %12.6f %10.1f"
              % (bd, len(cold[bd]), c, statistics.median(icold[bd]),
                 len(warm[bd]), w, statistics.median(iwarm[bd]), pub))
    if len(pts) >= 2:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        num = sum((p[0] - mx) * (p[1] - my) for p in pts)
        den = sum((p[0] - mx) ** 2 for p in pts)
        slope = num / den
        intercept = my - slope * mx
        print("\ncold-regime affine fit: submit_ns=%.4f  descriptor_ns_per_aiv=%.4f"
              % (intercept, slope))
        print("published (warm)      : submit_ns=%.4f  descriptor_ns_per_aiv=%.4f"
              % (PUBLISHED_SUBMIT_NS, PUBLISHED_DESC_NS))
        for bd, c in pts:
            print("   bd=%2d cold=%9.1f fit=%9.1f ape=%6.2f%%"
                  % (bd, c, intercept + slope * bd,
                     abs(intercept + slope * bd - c) / c * 100))


if __name__ == "__main__":
    main(sys.argv[1])
