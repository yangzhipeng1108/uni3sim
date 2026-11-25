"""AIV-domain DRAM bandwidth microbench via a pure element-wise op.

Physical principle (NPU, AIV path): a large 1-D element-wise op streams
DRAM (Ascend GM) -> UB -> Vector -> UB -> DRAM. For a memory-bound op with negligible compute
(a single add/mul), wall time is set by GM traffic through MTE2(GM->UB) +
MTE3(UB->GM). Effective DRAM BW = total_bytes / kernel_duration.

We drive torch_npu element-wise ops of increasing size, measure the kernel's
measured duration via msprof op (reliable — NOT the active_bw column, which was
found unreliable), and report achieved DRAM BW. Sweeping size shows the
saturation curve; the plateau is the sustained AIV DRAM bandwidth.

Run under msprof op:
  msprof op --aic-metrics=PipeUtilization,Memory --launch-count=N \
     --output=<dir> --application="python3 aiv_dram_bw.py --op mul --mb 64"
Or standalone (torch.npu.Event timing) for a quick read.
"""
import argparse, time
import torch, torch_npu

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def bench(op, nbytes, dtype, iters=50):
    if op not in ("add", "mul", "copy", "cast"):
        raise ValueError(f"unsupported DRAM bandwidth operation: {op!r}")
    if dtype not in DT:
        raise ValueError(f"unsupported DRAM bandwidth dtype: {dtype!r}")
    if nbytes <= 0 or iters <= 0:
        raise ValueError("nbytes and iters must be positive")
    torch.npu.set_device(0)
    dt = DT[dtype]
    elem_bytes = torch.tensor([], dtype=dt).element_size()
    n = nbytes // elem_bytes
    x = torch.randn(n, dtype=dt, device="npu")
    y = torch.randn(n, dtype=dt, device="npu")

    def run():
        if op == "add":
            return x + y            # read 2, write 1  -> 3 GM passes
        elif op == "mul":
            return x * y
        elif op == "copy":
            return x.clone()        # read 1, write 1  -> 2 GM passes
        elif op == "cast":
            return x.to(torch.float32 if dt != torch.float32 else torch.float16)
        else:
            raise AssertionError(f"validated operation became unknown: {op!r}")

    for _ in range(10):
        run()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        run()
    torch.npu.synchronize()
    dur_s = (time.time() - t0) / iters

    passes = {"add": 3, "mul": 3, "copy": 2, "cast": 2}[op]
    dram_bytes = nbytes * passes  # rough: reads + write, ignoring dtype change size
    bw = dram_bytes / dur_s / 1e9
    return dur_s * 1e6, bw


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="mul", choices=["add", "mul", "copy", "cast"])
    p.add_argument("--dtype", choices=sorted(DT), default="fp16")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--mb", type=float, default=64)
    a = p.parse_args()
    if a.sweep:
        print(f"{'size_MB':>8} {'dur_us':>10} {'DRAM_BW_GB/s':>12}")
        for mb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            us, bw = bench(a.op, int(mb * 1024 * 1024), a.dtype)
            print(f"{mb:8.0f} {us:10.1f} {bw:12.0f}")
    else:
        us, bw = bench(a.op, int(a.mb * 1024 * 1024), a.dtype)
        print(f"op={a.op} dtype={a.dtype} size={a.mb}MB dur={us:.1f}us DRAM_BW={bw:.0f} GB/s")
