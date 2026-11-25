"""Sweep DRAM bandwidth via msprof-measured kernel Duration (physically reliable).

Wraps aiv_dram_bw ops but reads the TRUE kernel Task Duration from msprof op
output CSV (OpBasicInfo), not wall-clock (which is launch-overhead-dominated for
small kernels) and not active_bw (found unreliable). BW = passes*size / duration.

Usage: python3 dram_bw_sweep.py --op copy --dtype fp16
Runs each size as its own msprof op invocation, parses OpBasicInfo Task Duration.
"""
import argparse
import csv
import glob
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "aiv_dram_bw.py")
PASSES = {"add": 3, "mul": 3, "copy": 2, "cast": 2}


def measure(op, dtype, mb, dev=0):
    if op not in PASSES:
        raise ValueError(f"unsupported DRAM bandwidth operation: {op!r}")
    if mb <= 0 or dev < 0:
        raise ValueError("DRAM bandwidth size must be positive and device non-negative")
    with tempfile.TemporaryDirectory(
            prefix=f"dram_bw_{op}_{dtype}_{mb}_") as out:
        app = f"python3 {DRIVER} --op {op} --dtype {dtype} --mb {mb}"
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(dev))
        subprocess.run(
            ["msprof", "op", "--aic-metrics=PipeUtilization,Memory",
             "--launch-count=3", "--launch-skip-before-match=12",
             f"--output={out}", f"--application={app}"],
            env=env, check=True, timeout=600)
        # Parse the largest-duration OpBasicInfo (the compute kernel, not tiny casts)
        durs, bd = [], None
        files = glob.glob(os.path.join(
            out, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
        if not files:
            raise FileNotFoundError(f"msprof produced no OpBasicInfo CSV under {out}")
        for path in files:
            with open(path) as stream:
                for row in csv.DictReader(stream):
                    durs.append(float(row["Task Duration(us)"]))
                    bd = int(float(row["Block Dim"]))
    if not durs or bd is None:
        raise ValueError("msprof OpBasicInfo contains no kernel measurements")
    dur = max(durs)  # the streaming kernel dominates
    bytes_moved = mb * 1024 * 1024 * PASSES[op]
    bw = bytes_moved / (dur * 1e-6) / 1e9
    return dur, bw, bd


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=sorted(PASSES), default="copy")
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--sizes", default="16,32,64,128,256,512")
    a = p.parse_args()
    print(f"# {a.op} {a.dtype}: DRAM BW via msprof Duration")
    print(f"{'MB':>6} {'dur_us':>10} {'BW_GB/s':>10} {'BlockDim':>9}")
    for mb in [int(x) for x in a.sizes.split(",")]:
        r = measure(a.op, a.dtype, mb)
        print(f"{mb:6d} {r[0]:10.1f} {r[1]:10.0f} {r[2]:9d}", flush=True)
