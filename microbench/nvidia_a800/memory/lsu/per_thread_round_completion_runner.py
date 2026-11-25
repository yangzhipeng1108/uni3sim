"""Capture the per-thread round-completion probe on a verified-idle A800.

The probe occupies one CTA on one SM.  Its L1-resident cells are private to
that SM and are reproducible while other tenants run, but every L2-resident
cell shares the device L2 with whatever else is on the GPU.  A busy device
does not make those cells noisy -- it makes them steadily wrong, so a
trial-spread gate cannot detect it.  A capture taken on 2026-09-23 while
eight other processes held all eight A800s at 100% read 534 ns for the same
cell an idle device reads at 196 ns, with per-cell spread still at 1.02.

This runner therefore refuses to capture unless the selected device is an
idle 108-SM compute-capability-8.0 A800, the same precondition the SiLU
load-pair runner enforces.  It is a device precondition, not a tolerance: no
measurement, production duration, or model error selects it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).with_name("per_thread_round_completion.cu")
MAX_IDLE_MEMORY_MIB = 8


def _smi(query, device):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{query[0]}={query[1]}",
         "--format=csv,noheader,nounits", "-i", str(device)],
        check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def require_idle_device(device):
    """Raise unless the selected device is an idle A800 with no tenants."""

    row = _smi(("gpu", "name,memory.used,utilization.gpu"), device)[0]
    name, memory_used, utilization = [f.strip() for f in row.split(",")]
    problems = []
    if "A800" not in name:
        problems.append(f"device {device} is {name!r}, not an A800")
    if int(utilization) != 0:
        problems.append(
            f"device {device} utilization is {utilization}%, not 0%")
    if int(memory_used) > MAX_IDLE_MEMORY_MIB:
        problems.append(
            f"device {device} holds {memory_used} MiB, more than the "
            f"{MAX_IDLE_MEMORY_MIB} MiB an idle device may reserve")
    apps = _smi(("compute-apps", "pid,used_memory"), device)
    if apps:
        problems.append(
            f"device {device} has {len(apps)} compute process(es): {apps}")
    if problems:
        raise SystemExit(
            "refusing to capture: the L2-resident cells require an idle "
            "device.\n  " + "\n  ".join(problems))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--nvcc", default="/usr/local/cuda-12.2/bin/nvcc")
    args = parser.parse_args()

    require_idle_device(args.device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "per_thread_round_completion"
    subprocess.run(
        [args.nvcc, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
         "-o", str(binary), str(PROBE)], check=True)

    describe = subprocess.run(
        [str(binary), "--describe"], check=True, capture_output=True,
        text=True).stdout
    (output / "describe.txt").write_text(describe)

    raw = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True,
        env={"CUDA_VISIBLE_DEVICES": str(args.device), "PATH": "/usr/bin"})
    (output / "raw.csv").write_text(raw.stdout)

    (output / "capture.json").write_text(json.dumps({
        "owner": args.owner,
        "device": args.device,
        "probe_source": str(PROBE),
        "idle_precondition": "verified before launch",
    }, indent=2) + "\n")
    print(f"captured to {output}")


if __name__ == "__main__":
    main()
