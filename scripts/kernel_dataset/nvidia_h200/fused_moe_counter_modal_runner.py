"""Capture non-duration L2 and DRAM traffic for a synthetic vLLM 0.30 MoE call."""

import hashlib
import os
import json
from pathlib import Path
import shutil
import subprocess
import time
import zlib
from datetime import datetime, timezone

import modal


CASE = Path(__file__).with_name("fused_moe_counter_case.py")
IMAGE = modal.Image.from_registry(
    "vllm/vllm-openai:v0.30.0", add_python="3.12"
).entrypoint([]).apt_install(
    "cuda-nsight-compute-13-0", "cuda-cuobjdump-13-0", "cuda-nvdisasm-13-0"
).add_local_file(CASE, remote_path="/tmp/fused_moe_counter_case.py")
APP = modal.App("midlayer-h200-fused-moe-traffic-20261029")


@APP.function(gpu="H200", image=IMAGE, timeout=180, retries=0,
              max_containers=1, single_use_containers=True)
def capture_activation_static() -> dict:
    """Inspect the installed 0.30 activation binary without timing it."""
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    package = subprocess.run(
        ["/usr/bin/python3", "-c", "import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent)"],
        check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    libraries = sorted(Path(package).glob("_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"expected one vLLM _C library, found {libraries}")
    library = libraries[0]
    listed = subprocess.run(["cuobjdump", "--list-text", str(library)],
                            check=True, capture_output=True, text=True, timeout=100)
    matches = [line.strip() for line in listed.stdout.splitlines()
               if ".sm_90.elf.bin" in line and "vllm18act_and_mul_kernelIN3c108BFloat16E" in line
               and "packed_silu_kernel" in line
               and "Lb1ELb1ELb0ELb0EEEv" in line]
    if len(matches) != 1 or "x-" not in matches[0]:
        raise RuntimeError(f"expected one observed BF16 SiLU activation SM90 entry: {matches}")
    symbol = matches[0].split("x-", 1)[1].split(".sm_90.elf.bin", 1)[0]
    disassembled = subprocess.run(
        ["cuobjdump", "--dump-sass", "--gpu-architecture", "sm_90",
         "--function", symbol, str(library)],
        capture_output=True, timeout=100)
    if disassembled.returncode != 0:
        raise RuntimeError("cuobjdump activation disassembly failed: "
                           + disassembled.stderr.decode("utf-8", errors="replace")[:1000])
    sass = disassembled.stdout
    if not sass or len(sass) > 4_000_000 or b"MUFU" not in sass:
        raise RuntimeError("activation SASS is empty, unexpectedly large or lacks SFU instructions")
    artifacts = [{"symbol": symbol, "sass_sha256": hashlib.sha256(sass).hexdigest(),
                  "sass_zlib_hex": zlib.compress(sass, level=9).hex()}]
    return {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_wall_s": time.monotonic() - start, "gpu_identity": gpu,
            "exit_code": 0,
            "library_path": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "symbol_lines": matches, "artifacts": artifacts}


@APP.function(gpu="H200", image=IMAGE, timeout=180, retries=0,
              max_containers=1, single_use_containers=True)
def capture_alignment_static() -> dict:
    """Inspect installed 0.30 INT32 ordinary alignment without running it."""
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    package = subprocess.run(
        ["/usr/bin/python3", "-c", "import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent)"],
        check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    libraries = sorted(Path(package).glob("_moe_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"expected one vLLM MoE CUDA library, found {libraries}")
    library = libraries[0]
    listed = subprocess.run(["cuobjdump", "--list-text", str(library)],
                            check=True, capture_output=True, text=True, timeout=100)
    matches = [line.strip() for line in listed.stdout.splitlines()
               if ".sm_90.elf.bin" in line and "moe_align_block_size_kernelIi" in line]
    if len(matches) != 1 or "x-" not in matches[0]:
        return {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - start, "gpu_identity": gpu,
                "exit_code": 1, "library_path": str(library),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "symbol_lines": matches, "artifacts": []}
    symbol = matches[0].split("x-", 1)[1].split(".sm_90.elf.bin", 1)[0]
    disassembled = subprocess.run(
        ["cuobjdump", "--dump-sass", "--gpu-architecture", "sm_90",
         "--function", symbol, str(library)],
        capture_output=True, timeout=100)
    if disassembled.returncode != 0:
        raise RuntimeError("cuobjdump alignment disassembly failed: "
                           + disassembled.stderr.decode("utf-8", errors="replace")[:1000])
    sass = disassembled.stdout
    if not sass or len(sass) > 4_000_000 or b"ATOMS" not in sass:
        raise RuntimeError("alignment SASS is empty, unexpectedly large or lacks shared atomics")
    artifacts = [{"symbol": symbol, "sass_sha256": hashlib.sha256(sass).hexdigest(),
                  "sass_zlib_hex": zlib.compress(sass, level=9).hex()}]
    return {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_wall_s": time.monotonic() - start, "gpu_identity": gpu,
            "exit_code": 0, "library_path": str(library),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "symbol_lines": matches, "artifacts": artifacts}


@APP.function(gpu="H200", image=IMAGE, timeout=600, retries=0,
              max_containers=1, single_use_containers=True)
def capture(tokens: int, topk: int, stage: str, profile: str) -> dict:
    task_dir = Path("/tmp/midlayer-fused-moe-traffic-20261029")
    task_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
        ncu = subprocess.run(["ncu", "--version"], check=True,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        interpreter_checks = []
        selected_python = None
        for candidate in ("/usr/bin/python3", "/usr/bin/python3.12",
                          "/usr/local/bin/python"):
            if not Path(candidate).is_file():
                interpreter_checks.append({"path": candidate, "status": "missing"})
                continue
            check = subprocess.run(
                [candidate, "-c", "import torch, vllm; print(torch.__version__, vllm.__version__)"],
                capture_output=True, text=True, timeout=60)
            interpreter_checks.append({
                "path": candidate, "exit_code": check.returncode,
                "stdout": check.stdout, "stderr": check.stderr,
            })
            if check.returncode == 0:
                selected_python = candidate
                break
        if selected_python is None:
            return {
                "started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - start,
                "gpu_identity": gpu,
                "ncu_version": ncu,
                "interpreter_checks": interpreter_checks,
                "exit_code": 1,
                "stdout": "",
                "stderr": "No interpreter can import torch and vllm",
            }
        if stage == "sass":
            cache_dir = task_dir / "triton_cache"
            env = os.environ.copy()
            env["TRITON_CACHE_DIR"] = str(cache_dir)
            env["TRITON_KERNEL_DUMP"] = "1"
            completed = subprocess.run(
                [selected_python, "/tmp/fused_moe_counter_case.py", "--tokens", str(tokens),
                 "--topk", str(topk), "--profile", profile],
                capture_output=True, text=True, timeout=480, env=env)
            artifacts = []
            for cubin in sorted(cache_dir.rglob("fused_moe_kernel*.cubin")):
                ptx = cubin.with_suffix(".ptx")
                if not ptx.is_file():
                    raise RuntimeError("compiled fused MoE cubin has no sibling PTX: " + str(cubin))
                sass = subprocess.run(["cuobjdump", "--dump-sass", str(cubin)],
                                      capture_output=True, text=True, timeout=60)
                if sass.returncode != 0:
                    raise RuntimeError("cuobjdump failed for fused MoE cubin: " + sass.stderr)
                sass_bytes = sass.stdout.encode("utf-8")
                ptx_bytes = ptx.read_bytes()
                artifacts.append({
                    "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                    "sass_sha256": hashlib.sha256(sass_bytes).hexdigest(),
                    "sass_zlib_hex": zlib.compress(sass_bytes, level=9).hex(),
                    "ptx_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
                    "ptx_zlib_hex": zlib.compress(ptx_bytes, level=9).hex(),
                    "relative_cubin_path": str(cubin.relative_to(cache_dir)),
                })
            return {
                "started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - start,
                "gpu_identity": gpu,
                "ncu_version": ncu,
                "interpreter_checks": interpreter_checks,
                "command": [selected_python, "/tmp/fused_moe_counter_case.py", "--tokens", str(tokens),
                            "--topk", str(topk), "--profile", profile],
                "exit_code": completed.returncode if artifacts else 1,
                "stdout": completed.stdout,
                "stderr": completed.stderr if artifacts else completed.stderr + "\nNo fused_moe_kernel cubin in the isolated Triton cache",
                "artifacts": artifacts,
                "cached_cubin_count": len(tuple(cache_dir.rglob("*.cubin"))),
            }
        if stage in ("w1w2", "output_traffic", "service", "stall", "integer"):
            kernel_name, launch_skip, launch_count = "regex:fused_moe_kernel", "2", "2"
        elif stage in ("activation", "activation_issue"):
            kernel_name, launch_skip, launch_count = "regex:act_and_mul_kernel", "1", "1"
        else:
            raise ValueError("unsupported fused MoE counter stage")
        command = [
            "ncu", "--csv", "--page", "raw", "--cache-control", "none",
            "--clock-control", "none", "--kernel-name", kernel_name,
            "--launch-skip", launch_skip, "--launch-count", launch_count, "--metrics",
            ("smsp__warp_issue_stalled_barrier_per_warp_active.pct,"
             "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,"
             "smsp__warp_issue_stalled_wait_per_warp_active.pct,"
             "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,"
             "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
             if stage == "stall" else
             "smsp__warps_eligible.avg.per_cycle_active,smsp__warps_active.avg.per_cycle_active,"
             "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,"
             "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
             if stage == "service" else
             "smsp__sass_thread_inst_executed_op_integer_pred_on.sum"
             if stage == "integer" else
             "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,"
             "smsp__sass_thread_inst_executed_op_integer_pred_on.sum"
             if stage == "activation_issue" else
             "lts__t_sectors_op_read.sum,dram__bytes_read.sum"
             if stage == "w1w2" else
             "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum"),
            selected_python, "/tmp/fused_moe_counter_case.py", "--tokens",
            str(tokens), "--topk", str(topk), "--profile", profile,
        ]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=480)
        return {
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_wall_s": time.monotonic() - start,
            "gpu_identity": gpu,
            "ncu_version": ncu,
            "interpreter_checks": interpreter_checks,
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    finally:
        shutil.rmtree(task_dir)


@APP.local_entrypoint()
def main(output_dir: str, tokens: int, topk: int, stage: str = "w1w2",
         profile: str = "e128_h2048_i768"):
    if tokens not in (8192, 32768):
        raise ValueError("tokens must be 8192 or 32768")
    if topk not in (8, 16):
        raise ValueError("topk must be 8 or 16")
    if profile not in ("e128_h2048_i768", "e256_h6144_i256"):
        raise ValueError("unsupported synthetic FusedMoE profile")
    if profile == "e256_h6144_i256" and (tokens != 32768 or topk != 8):
        raise ValueError("E256/H6144/I256 probe requires tokens=32768 and topk=8")
    if stage not in ("w1w2", "output_traffic", "activation", "activation_issue", "service", "stall", "integer", "sass", "activation_sass", "alignment_sass"):
        raise ValueError("unsupported fused MoE capture stage")
    root = Path(__file__).resolve().parents[3]
    parent = root / "profiler_output/kernel_dataset/nvidia_h200/fused_moe_traffic_20261029"
    target = Path(output_dir).resolve()
    if target.parent != parent.resolve() or target.exists():
        raise ValueError("output must be a new child of the registered campaign")
    result = (capture_activation_static.remote() if stage == "activation_sass"
              else capture_alignment_static.remote() if stage == "alignment_sass"
              else capture.remote(tokens, topk, stage, profile))
    target.mkdir(parents=True)
    if stage in ("activation_sass", "alignment_sass"):
        for index, artifact in enumerate(result["artifacts"]):
            sass = zlib.decompress(bytes.fromhex(artifact["sass_zlib_hex"]))
            if hashlib.sha256(sass).hexdigest() != artifact["sass_sha256"]:
                raise ValueError("returned installed SASS hash mismatch")
            (target / f"{'activation' if stage == 'activation_sass' else 'alignment'}_{index}.sass").write_bytes(sass)
        result["artifacts"] = [{key: value for key, value in artifact.items()
            if key != "sass_zlib_hex"} for artifact in result["artifacts"]]
    result["case_sha256"] = hashlib.sha256(CASE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["baseline_commit"] = "ef73f6a476b8184d1cf5faad3c5c5a86dc9d8049"
    result["requested_tokens"] = tokens
    result["requested_topk"] = topk
    result["requested_profile"] = profile
    result["measurement_stage"] = stage
    result["measurement_boundary"] = ("installed_vllm030_bf16_activation_static_library_only"
        if stage == "activation_sass" else "installed_vllm030_int32_alignment_static_library_only"
        if stage == "alignment_sass" else "synthetic_route_complete_vllm030_bf16_fused_experts_static_cubin_only"
        if stage == "sass" else "synthetic_route_complete_vllm030_bf16_fused_experts_service_counters_only"
        if stage in ("service", "stall", "integer", "activation_issue") else "synthetic_route_complete_vllm030_bf16_fused_experts_traffic_only")
    if stage == "sass":
        for index, artifact in enumerate(result["artifacts"]):
            sass_bytes = zlib.decompress(bytes.fromhex(artifact["sass_zlib_hex"]))
            if hashlib.sha256(sass_bytes).hexdigest() != artifact["sass_sha256"]:
                raise ValueError("returned SASS content does not match its SHA256")
            (target / f"fused_moe_kernel_{index}.sass").write_bytes(sass_bytes)
            ptx_bytes = zlib.decompress(bytes.fromhex(artifact["ptx_zlib_hex"]))
            if hashlib.sha256(ptx_bytes).hexdigest() != artifact["ptx_sha256"]:
                raise ValueError("returned PTX content does not match its SHA256")
            (target / f"fused_moe_kernel_{index}.ptx").write_bytes(ptx_bytes)
        result["artifacts"] = [{key: value for key, value in artifact.items()
            if key not in ("sass_zlib_hex", "ptx_zlib_hex")} for artifact in result["artifacts"]]
    (target / "counter_attempt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(target), "gpu_wall_s": result["gpu_wall_s"],
        "exit_code": result["exit_code"],
    }, sort_keys=True))
