"""Gate a synthetic H200 routed-panel K-loop on Triton 3.7.1 code generation."""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import modal


HERE = Path(__file__).resolve().parent
TASK_DIR = Path("/tmp/midlayer-h200-routed-panel-triton-dependency-20261102")
SOURCE = (HERE / "routed_panel_triton_dependency.py" if HERE.name == "tensor_core"
          else TASK_DIR / "routed_panel_triton_dependency.py")
IMAGE = (modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("cuda-nsight-compute-12-9")
    .run_commands(
        "python -m pip install --no-cache-dir 'torch==2.13.0+cu129' "
        "'triton==3.7.1' "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "--extra-index-url https://wheels.vllm.ai/0.30.0/cu129/"
    )
    .add_local_file(SOURCE, remote_path=str(TASK_DIR / SOURCE.name)))
APP = modal.App("midlayer-h200-routed-panel-triton-dependency-20261102")
GATE_ENV = "MIDLAYER_H200_PAID_PROBE_CONTROLLER"
BOUNDARY = "synthetic_grouped_bf16_kloop_triton371_no_moe_output"
SOURCE_CONFIG = {"block_m": 128, "block_n": 256, "block_k": 64,
                 "group_size_m": 16, "num_warps": 8, "num_stages": 4,
                 "top_k_for_w2": 1, "swap_ab": False, "use_td": False,
                 "unquantized_bf16_branch": "accumulator += tl.dot(a, b)",
                 "router": False, "activation": False, "matrix_output": False,
                 "depths": [4, 8, 12], "conditions": ["small", "large"]}
METRICS = (
    "lts__t_sectors_op_read.sum,dram__bytes_read.sum,"
    "launch__registers_per_thread,launch__shared_mem_per_block,"
    "launch__occupancy_limit_shared_mem,launch__waves_per_multiprocessor,"
    "sm__warps_active.avg.pct_of_peak_sustained_active")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=TASK_DIR, capture_output=True,
                                text=True, timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "wall_s": time.monotonic() - started}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "returncode": None,
                "error": f"timeout after {timeout}s: {error}",
                "wall_s": time.monotonic() - started}


def _compile(module, signature: dict, constants: dict,
             *, rounds: int | None, condition: str) -> dict:
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    spec = dict(constants)
    if rounds is not None:
        spec["K_ROUNDS"] = rounds
    source = ASTSource(module.routed_panel_kloop, signature=signature,
                       constexprs=spec)
    options = {"num_warps": 8, "num_stages": 4}
    compiled = triton_compile(source, target=GPUTarget("cuda", 90, 32),
                              options=options)
    cubin = compiled.asm["cubin"]
    cubin_path = TASK_DIR / f"{condition}_{'runtime' if rounds is None else rounds}.cubin"
    cubin_path.write_bytes(cubin)
    sass_result = _run([shutil.which("cuobjdump"), "--dump-sass",
                        str(cubin_path)], 30)
    disasm_result = _run([shutil.which("nvdisasm"), str(cubin_path)], 30)
    if sass_result["returncode"] != 0 or disasm_result["returncode"] != 0:
        raise ValueError(f"SASS/resource inspection failed for {condition}/{rounds}")
    sass = sass_result["stdout"]
    reg_match = re.search(
        r"EIATTR_REGCOUNT.*?\.word\s+index@\(routed_panel_kloop\)"
        r".*?\.word\s+0x([0-9a-f]+)", disasm_result["stdout"], re.S)
    if reg_match is None:
        raise ValueError(f"register count missing for {condition}/{rounds}")
    counts = {"ldgsts": sass.count("LDGSTS"),
              "hgmma": sass.count("HGMMA."),
              "bar_sync": sass.count("BAR.SYNC"),
              "warpgroup_depbar": sass.count("WARPGROUP.DEPBAR"),
              "spill_load": len(re.findall(r"\bLDL\b", sass)),
              "spill_store": len(re.findall(r"\bSTL\b", sass))}
    return {"condition": condition, "rounds_specialization": rounds,
            "compiler": "triton.compile(ASTSource, GPUTarget(cuda,90,32))",
            "options": options, "metadata_shared_bytes": compiled.metadata.shared,
            "metadata_num_warps": compiled.metadata.num_warps,
            "metadata_num_stages": compiled.metadata.num_stages,
            "registers_per_thread": int(reg_match.group(1), 16),
            "counts": counts, "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "cubin_base64": base64.b64encode(cubin).decode("ascii"),
            "sass": sass,
            "ptx_sha256": hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest(),
            "ptx": compiled.asm["ptx"]}


def _source_equivalent(row: dict) -> bool:
    counts = row["counts"]
    return (row["metadata_shared_bytes"] == 196608
            and row["metadata_num_warps"] == 8
            and row["metadata_num_stages"] == 4
            and row["registers_per_thread"] >= 190
            and row["registers_per_thread"] <= 230
            and counts["ldgsts"] == 48 and counts["hgmma"] == 4
            and counts["bar_sync"] >= 6 and counts["bar_sync"] <= 10
            and counts["spill_load"] == 0 and counts["spill_store"] == 0)


@APP.function(gpu="H200", image=IMAGE, timeout=780, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    result = {"schema_version": 1, "measurement_boundary": BOUNDARY,
              "source_config": SOURCE_CONFIG, "status": "started"}
    try:
        import torch
        import triton

        result["source_sha256"] = _sha256(SOURCE)
        result["software"] = {"torch": str(torch.__version__),
                              "torch_cuda": str(torch.version.cuda),
                              "triton": triton.__version__}
        result["gpu_query"] = _run([
            "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm",
            "--format=csv,noheader"], 20)
        result["tool_versions"] = {
            name: _run([name, "--version"], 20)
            for name in ("cuobjdump", "nvdisasm", "ncu")}
        if (result["software"] != {"torch": "2.13.0+cu129",
                                   "torch_cuda": "12.9", "triton": "3.7.1"}
                or result["gpu_query"]["returncode"] != 0
                or not result["gpu_query"]["stdout"].startswith("NVIDIA H200,")
                or any(row["returncode"] != 0
                       for row in result["tool_versions"].values())):
            result["status"] = "environment_failed"
            return result
        spec = importlib.util.spec_from_file_location(
            "routed_panel_triton_dependency", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if (module.CONDITIONS["small"]["grid"] != 2112
                or module.CONDITIONS["large"]["grid"] != 33784
                or SOURCE_CONFIG["num_stages"] != 4
                or SOURCE_CONFIG["num_warps"] != 8
                or SOURCE_CONFIG["swap_ab"] is not False
                or SOURCE_CONFIG["use_td"] is not False
                or SOURCE_CONFIG["unquantized_bf16_branch"]
                != "accumulator += tl.dot(a, b)"):
            raise ValueError("source domain contract differs")
        signature = {"a_ptr": "*bf16", "b_ptr": "*bf16",
                     "sorted_token_ids_ptr": "*i32", "expert_ids_ptr": "*i32",
                     "num_tokens_post_padded_ptr": "*i32",
                     "checksum_ptr": "*fp32", "N": "i32", "K": "i32",
                     "EM": "i32", "NUM_VALID_TOKENS": "i32",
                     "K_ROUNDS": "i32"}
        constants = {"TOP_K": 1, "BLOCK_SIZE_M": 128,
                     "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
                     "GROUP_SIZE_M": 16}
        result["compiled"] = {"runtime": _compile(
            module, signature, constants, rounds=None,
            condition="source_domain")}
        for rounds in SOURCE_CONFIG["depths"]:
            result["compiled"][str(rounds)] = _compile(
                module, signature, constants, rounds=rounds,
                condition="source_domain")
        result["source_equivalent"] = _source_equivalent(
            result["compiled"]["runtime"])
        if result["source_equivalent"] is not True:
            result["status"] = "compile_mismatch"
            return result
        result["execution"] = []
        result["counters"] = []
        for condition in SOURCE_CONFIG["conditions"]:
            for rounds in SOURCE_CONFIG["depths"]:
                result["execution"].append(module.run_once(
                    condition, rounds, 5))
                command = [
                    shutil.which("ncu"), "--csv", "--page", "raw",
                    "--cache-control", "none", "--clock-control", "none",
                    "--kernel-name", "regex:routed_panel_kloop",
                    "--launch-skip", "2", "--launch-count", "1",
                    "--metrics", METRICS, sys.executable, str(SOURCE),
                    "--condition", condition, "--rounds", str(rounds),
                    "--samples", "1"]
                row = _run(command, 70)
                result["counters"].append({"condition": condition,
                                           "rounds": rounds, **row})
                if row["returncode"] != 0:
                    result["status"] = "counter_failed"
                    return result
        result["status"] = "complete"
        return result
    except Exception as error:
        result["status"] = "capture_failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK_DIR, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    if os.environ[GATE_ENV] != "1":
        raise PermissionError("paid H200 compiler probe requires spend controller")
    root = Path(__file__).resolve().parents[4]
    campaign = (root / "profiler_output/microbench/nvidia_h200/"
                "routed_panel_triton_dependency_20261102")
    target = Path(output_dir).resolve()
    if target.parent != campaign.resolve() or not target.name.startswith("attempt_"):
        raise ValueError("probe output must be a registered attempt")
    if target.exists():
        raise FileExistsError(target)
    raw = capture.remote()
    raw["runner_sha256"] = _sha256(Path(__file__))
    raw["local_source_sha256"] = _sha256(SOURCE)
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(raw, indent=2, sort_keys=True)
                                     + "\n")
    print(json.dumps({"output": str(target / "raw.json"),
                      "status": raw["status"],
                      "gpu_function_wall_s": raw["gpu_function_wall_s"]}))


if __name__ == "__main__":
    main()
