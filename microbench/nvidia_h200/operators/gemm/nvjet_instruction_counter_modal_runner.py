"""Non-duration instruction/PC observations; source passed without local mounts."""
import gzip
import hashlib
import json
from pathlib import Path

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12").apt_install("cuda-nsight-compute-12-9")
APP = modal.App("midlayer-h200-nvjet-instruction-counter-retry-20261107")


@APP.function(image=IMAGE, serialized=True, include_source=False, memory=1024,
              timeout=30, startup_timeout=60, retries=0, single_use_containers=True)
def precompile(source):
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="midlayer-nvjet-counter-cpu-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        cmd = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
               str(path), "-lcublas", "-o", directory + "/probe"]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        help_result = subprocess.run(["ncu", "--help"], capture_output=True, text=True, check=True, timeout=5)
        sections = subprocess.run(["ncu", "--list-sections"], capture_output=True, text=True, check=True, timeout=5)
        if any(name not in sections.stdout for name in ("LaunchStats", "InstructionStats", "SourceCounters")):
            raise ValueError("required NCU sections unavailable; GPU not called")
        return {"gpu": None, "source_sha256": hashlib.sha256(source).hexdigest(),
                "command": cmd, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                "ncu_help": help_result.stdout, "ncu_sections": sections.stdout}


@APP.function(gpu="H200", image=IMAGE, serialized=True, include_source=False, memory=4096,
              timeout=300, startup_timeout=60, retries=0, single_use_containers=True)
def capture(source, cases):
    import hashlib
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    env = dict(os.environ)
    for key in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if key in env:
            del env[key]
    result = {"gpu": gpu, "source_sha256": hashlib.sha256(source).hexdigest(), "cases": []}
    with tempfile.TemporaryDirectory(prefix="midlayer-nvjet-counter-") as directory:
        path = Path(directory) / "probe.cu"
        path.write_bytes(source)
        cmd = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
               str(path), "-lcublas", "-o", directory + "/probe"]
        build = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        result.update({"build_command": cmd, "build_exit_code": build.returncode,
                       "build_stdout": build.stdout, "build_stderr": build.stderr})
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            result["ncu_version"] = subprocess.run(["ncu", "--version"], capture_output=True, text=True, check=True, timeout=10).stdout
            for case in cases:
                report = Path(directory) / case["case_id"]
                csv_path = str(report) + ".csv"
                command = ["ncu", "--csv", "--page", "raw", "--cache-control", "none",
                           "--clock-control", "none", "--profile-from-start", "off",
                           "--kernel-name", "regex:nvjet_", "--launch-count", "1",
                           "--section", "LaunchStats", "--section", "InstructionStats",
                           "--section", "SourceCounters", "--export", str(report),
                           "--log-file", csv_path, directory + "/probe",
                           str(case["m"]), str(case["n"]), str(case["k"])]
                run = subprocess.run(command, capture_output=True, text=True, env=env, timeout=55)
                item = {"case_id": case["case_id"], "shape": case, "command": command,
                        "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
                if Path(csv_path).exists():
                    item["metrics_csv"] = Path(csv_path).read_text()
                report_path = Path(str(report) + ".ncu-rep")
                if report_path.exists():
                    item["report_bytes"] = report_path.read_bytes()
                result["cases"].append(item)
                if run.returncode != 0:
                    break
                dump = subprocess.run(["ncu", "--import", str(report_path), "--page", "source",
                                       "--print-source", "sass", "--csv"],
                                      capture_output=True, text=True, timeout=20)
                item["source_export_exit_code"] = dump.returncode
                item["source_export"] = dump.stdout
                item["source_export_stderr"] = dump.stderr
                if dump.returncode != 0:
                    break
        result["function_wall_s"] = time.monotonic() - started
        return result


@APP.local_entrypoint()
def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    target = root / "profiler_output/microbench/nvidia_h200/nvjet_instruction_counter_retry_20261107"
    if {p.name for p in target.iterdir()} != {"preflight.json", "local_build.log.gz"}:
        raise ValueError("refuse to overwrite captured evidence")
    source = (here / "nvjet_instruction_counter.cu").read_bytes()
    plan = json.loads((target / "preflight.json").read_text())["cases"]
    smoke = precompile.remote(source)
    (target / "cpu_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if smoke["exit_code"] != 0 or smoke["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("CPU source/compile failed; GPU not called")
    result = capture.remote(source, plan)
    if result["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("GPU source identity mismatch")
    for case in result["cases"]:
        for key, suffix in (("report_bytes", ".ncu-rep.gz"), ("metrics_csv", ".csv.gz"),
                            ("source_export", ".source.csv.gz")):
            if key in case:
                value = case[key]
                data = value if isinstance(value, bytes) else value.encode()
                name = case["case_id"] + suffix
                (target / name).write_bytes(gzip.compress(data, mtime=0))
                case[key + "_path"] = name
                case[key + "_sha256"] = hashlib.sha256(data).hexdigest()
                del case[key]
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    (snapshot / "nvjet_instruction_counter.cu").write_bytes(source)
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"build_exit_code": result["build_exit_code"], "case_exit_codes": [x["exit_code"] for x in result["cases"]], "function_wall_s": result["function_wall_s"]}))
