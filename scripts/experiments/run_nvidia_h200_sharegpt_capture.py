#!/usr/bin/env python3
"""Run the full H200 ShareGPT capture for one lane of cases (stdlib only).

One lane owns a fixed set of physical GPUs and captures its cases one after
another.  Every GPU stage runs as the only process on those GPUs: it is wrapped
in ``flock -x`` on each GPU's whole-GPU process lock (the kernel campaign uses the same ``gpu<N>.proc.lock`` files), taken in ascending GPU
order so lanes cannot deadlock.  Per case the raw artifacts are exactly those
behind an A800 canonical case:

``host_request_path_probe_before.json``  delivery-region probe (host only)
``trace_off/ground_truth.json``          no-instrumentation production run
``trace_on/ground_truth.json``           same run with the Scheduler trace
``trace_on/scheduler_steps.raw.jsonl``   (+ .runtime/.clock/.warmup/.bootstrap/.rank-*)
``trace_on/scheduler_trace_observer_effect.json``  trace-off/on 3% gate
``host_request_path_probe_after.json``   delivery-region probe (host only)
``raw/ShareGPT-<model>_<par>_n128_qps<q>/``  eager operator + Scheduler +
                                          Nsight capture (A800 serve_exp layout)
``clocks.jsonl``                          nvidia-smi clock/pstate around stages
``case_capture.json``                     commands, return codes, wall times

Stage failures are recorded and the lane continues with the next stage/case;
nothing already written is overwritten.  Usage on the selected capture host:

  python3 run_nvidia_h200_sharegpt_capture.py run --runtime-version 0.30.0 --python <runtime-python> --public-root <capture-root> \\
      --gpus 4 --case qwen3-4b/tp1_qps4 --case qwen3-8b/tp1_qps4 ...
  python3 run_nvidia_h200_sharegpt_capture.py plan --runtime-version 0.30.0 --python <runtime-python> --public-root <capture-root> \\
      --gpus 1,2,3,4,5,6,7   # prints one detached launch per lane
  python3 run_nvidia_h200_sharegpt_capture.py rerun-trace-on --runtime-version 0.30.0 --python <runtime-python> --public-root <capture-root> \\
      --gpus 4 --case-dir <case> --tag r1
      # repeats only trace_on (into trace_on_<tag>/) and its observer-effect
      # check against the case's existing trace_off; its stages are recorded
      # in case_capture.trace_on_<tag>.json, nothing existing is modified
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_RUNTIME_VERSIONS = ("0.30.0", "0.11.0")
DEFAULT_PATH_KEYS = (
    "output_root", "models_root", "dataset", "nsys", "cuda_home",
    "gpu_lock_dir", "cache_root", "home", "pydeps",
)


def _default_paths(public_root: Path) -> dict[str, Path]:
    return {
        "output_root": public_root / "serving_capture",
        "models_root": public_root / "models",
        "dataset": public_root / "datasets" / "ShareGPT_V3_unfiltered_cleaned_split.json",
        "nsys": public_root / "tools" / "nsys_2025.6.1" / "bin" / "nsys",
        "cuda_home": public_root / "tools" / "cuda-12.9",
        "gpu_lock_dir": public_root / "h200_microbench" / "kcampaign" / "runs",
        "cache_root": public_root / "cache",
        "home": public_root / "h200_microbench" / "home",
        # cp312 wheels the repository imports but vLLM does not ship (zstandard).
        "pydeps": public_root / "python_dependencies",
    }
PROBE = ROOT / "microbench" / "nvidia_a800" / "runtime" / "host_request_path" / (
    "vllm_delivery_region_runner.py")
PROBE_FANOUTS = "1,2,4,8,16,32,64"
PROBE_ROUNDS = "40"
PROBE_REPEATS = "3"

# model_id -> (model directory, raw model id, suite manifest by case_id)
_TP1_SUITE = "configs/evaluation/experiments/sharegpt_phase1.json"
CASES = {}
for _model_id, _directory, _raw in (
    ("qwen3-4b", "Qwen3-4B", "Qwen3-4B"),
    ("llama-3.1-8b", "Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("qwen3-8b", "Qwen3-8B", "Qwen3-8B"),
):
    for _qps in (4, 16, 64):
        CASES[f"{_model_id}/tp1_qps{_qps}"] = {
            "model_dir": _directory, "raw_model_id": _raw, "tp": 1, "ep": 1,
            "qps": _qps, "suite": _TP1_SUITE,
            "raw_dir": f"ShareGPT-{_raw}_TP1_n128_qps{_qps}",
        }
for _qps in (1, 4, 16):
    CASES[f"qwen3-32b/tp2_ep1_qps{_qps}"] = {
        "model_dir": "Qwen3-32B", "raw_model_id": "Qwen3-32B", "tp": 2, "ep": 1,
        "qps": _qps,
        "suite": (
            "configs/evaluation/experiments/qwen3_32b_qps1_suite.json" if _qps == 1
            else "configs/evaluation/experiments/qwen3_32b_qps4_qps16_suite.json"),
        "raw_dir": f"ShareGPT-Qwen3-32B_TP2_n128_qps{_qps}",
    }
    CASES[f"qwen3-30b-a3b/tp2_ep2_qps{_qps}"] = {
        "model_dir": "Qwen3-30B-A3B", "raw_model_id": "Qwen3-30B-A3B", "tp": 2,
        "ep": 2, "qps": _qps,
        "suite": "configs/evaluation/experiments/nvidia_a800_qwen3_30b_a3b_suite.json",
        "raw_dir": f"ShareGPT-Qwen3-30B-A3B_EP2_n128_qps{_qps}",
    }
if len(CASES) != 15:
    raise AssertionError("H200 ShareGPT capture matrix must contain 15 cases")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


class Lane:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = args.runtime_version
        self.python = Path(args.python)
        self.gpus = [int(value) for value in args.gpus.split(",")]
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("GPUs must be unique")
        # Device zero requires explicit assignment just like every other lane.
        if 0 in self.gpus and not args.allow_gpu0:
            raise ValueError("GPU 0 needs --allow-gpu0")
        self.output_root = Path(args.output_root)
        self.record_name = "case_capture.json"
        self.port = 18400 + 10 * min(self.gpus) + (0 if self.runtime == "0.30.0" else 5)

    def environment(self, gpus: list[int]) -> dict[str, str]:
        tag = f"sc{self.runtime.replace('.', '')}_gpu{'-'.join(map(str, gpus))}"
        cache = Path(self.args.cache_root)
        env = dict(os.environ)
        env.update({
            "HOME": str(self.args.home),
            "XDG_CACHE_HOME": str(cache),
            "TRITON_CACHE_DIR": str(cache / f"triton_{tag}"),
            "VLLM_CACHE_ROOT": str(cache / f"vllm_{tag}"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / f"inductor_{tag}"),
            "FLASHINFER_WORKSPACE_BASE": str(cache / f"flashinfer_{tag}"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_HOME": str(self.args.cuda_home),
            "PATH": f"{self.args.cuda_home}/bin:{os.environ['PATH']}",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "PYTHONPATH": f"{ROOT}:{self.args.pydeps}",
        })
        return env

    def locked(self, command: list[str]) -> list[str]:
        wrapped = list(command)
        for gpu in sorted(self.gpus, reverse=True):
            wrapped = ["flock", "-x", str(Path(self.args.gpu_lock_dir) / f"gpu{gpu}.proc.lock"),
                       *wrapped]
        return wrapped

    def clocks(self, case_dir: Path, label: str) -> None:
        completed = subprocess.run(
            ["nvidia-smi", "-i", ",".join(map(str, self.gpus)),
             "--query-gpu=index,pstate,clocks.sm,clocks.mem,clocks.max.sm,"
             "temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False)
        with (case_dir / "clocks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "label": label, "time": _now(), "returncode": completed.returncode,
                "fields": "index,pstate,clocks.sm,clocks.mem,clocks.max.sm,"
                          "temperature.gpu,power.draw,utilization.gpu,memory.used",
                "rows": completed.stdout.strip().splitlines(),
                "stderr": completed.stderr.strip(),
            }, sort_keys=True) + "\n")

    def stage(self, record: dict, case_dir: Path, name: str, command: list[str],
              env: dict[str, str], *, gpu: bool, log_dir: Path) -> int:
        log_dir.mkdir(parents=True, exist_ok=True)
        if gpu:
            command = self.locked(command)
            self.clocks(case_dir, f"{name}:before")
        started = time.monotonic()
        entry = {"name": name, "command": command, "started": _now(), "gpu": gpu,
                 "log": str(log_dir / f"{name}.log")}
        record["stages"].append(entry)
        self.write_record(case_dir, record)
        with (log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        entry.update({"returncode": completed.returncode, "ended": _now(),
                      "wall_s": time.monotonic() - started})
        if gpu:
            self.clocks(case_dir, f"{name}:after")
        self.write_record(case_dir, record)
        print(f"[{_now()}] {record['case']} {name} rc={completed.returncode} "
              f"wall={entry['wall_s']:.0f}s", flush=True)
        return completed.returncode

    def write_record(self, case_dir: Path, record: dict) -> None:
        temporary = case_dir / f".{self.record_name}.tmp"
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, case_dir / self.record_name)

    def probe(self, record: dict, case_dir: Path, model: Path, when: str) -> None:
        target = case_dir / f"host_request_path_probe_{when}.json"
        env = self.environment(self.gpus)
        env["CUDA_VISIBLE_DEVICES"] = ""
        returncode = self.stage(
            record, case_dir, f"probe_{when}",
            [str(self.python), str(PROBE), str(model), PROBE_FANOUTS, PROBE_ROUNDS,
             PROBE_REPEATS, str(target)],
            env, gpu=False, log_dir=case_dir / "logs")
        if returncode != 0 or not target.is_file():
            # The same explicit failure record run_sharegpt_ground_truth.py writes.
            target.write_text(json.dumps({
                "schema_version": 1, "status": "probe_failed",
                "returncode": returncode,
                "log": str(case_dir / "logs" / f"probe_{when}.log"),
            }, indent=2) + "\n")

    def ground_truth(self, record: dict, case_dir: Path, spec: dict, case_key: str,
                     model: Path, trace: bool, name: str | None = None) -> int:
        if name is None:
            name = "trace_on" if trace else "trace_off"
        run_dir = case_dir / name
        run_dir.mkdir()
        model_id, case_id = case_key.split("/")
        command = [
            str(self.python), "scripts/experiments/run_sharegpt_ground_truth.py",
            "--no-host-path-probe",
            "--model", str(model), "--dataset", str(self.args.dataset),
            "--case-manifest",
            f"serving_dataset/sharegpt/nvidia_a800/{model_id}/{case_id}/manifest.json",
            "--suite-manifest", spec["suite"],
            "--target-hardware-id", "nvidia_h200",
            "--runtime-version", self.runtime,
            "--devices", *map(str, self.gpus),
            "--port", str(self.port),
            "--kv-block-size-tokens", "16",
            "--output", str(run_dir / "ground_truth.json"),
        ]
        if trace:
            command += ["--scheduler-trace-output",
                        str(run_dir / "scheduler_steps.raw.jsonl")]
        env = self.environment(self.gpus)
        # The runner sets CUDA_VISIBLE_DEVICES for the server from --devices.
        del env["CUDA_VISIBLE_DEVICES"]
        return self.stage(record, case_dir, name, command, env, gpu=True,
                          log_dir=case_dir / "logs")

    def alignment(self, record: dict, case_dir: Path, spec: dict, model: Path,
                  requests: Path) -> None:
        raw_dir = case_dir / "raw" / spec["raw_dir"]
        raw_dir.mkdir(parents=True)
        env = self.environment(self.gpus)
        bootstrap = ROOT / "dataset_pipeline" / "serving_dataset" / "instrumentation_bootstrap"
        env["PYTHONPATH"] = f"{bootstrap}:{ROOT}:{self.args.pydeps}"
        env["MIDLAYER_SCHEDULER_SIDECAR_PATH"] = str(raw_dir / "plugin_trace.json_scheduler.jsonl")
        env["MIDLAYER_OPERATOR_TRACE_PATH"] = str(raw_dir / "plugin_trace.json_operators.jsonl")
        report = raw_dir / "nsys_profile"
        command = [
            str(self.args.nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
            "--cpuctxsw=none", "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node", "--force-overwrite=false",
            "--output", str(report),
            str(self.python), "scripts/experiments/nvidia_h200_serving_alignment_runner.py",
            "--case-dir", str(raw_dir), "--model", str(model),
            "--dataset", str(self.args.dataset), "--requests", str(requests),
            "--target-qps", str(spec["qps"]), "--seed", "42",
            "--tensor-parallel-size", str(spec["tp"]),
            "--expert-parallel-size", str(spec["ep"]),
            "--runtime-version", self.runtime,
        ]
        returncode = self.stage(record, case_dir, "alignment_nsys", command, env,
                                gpu=True, log_dir=case_dir / "logs")
        rep = raw_dir / "nsys_profile.nsys-rep"
        if rep.is_file():
            self.stage(record, case_dir, "nsys_export",
                       [str(self.args.nsys), "export", "--type", "sqlite",
                        "--output", str(raw_dir / "nsys_profile.sqlite"), str(rep)],
                       self.environment(self.gpus), gpu=False,
                       log_dir=case_dir / "logs")
        elif returncode == 0:
            raise RuntimeError(f"alignment capture succeeded without {rep}")

    def observer_effect(self, record: dict, case_dir: Path, name: str) -> int:
        stage = "observer_effect" if name == "trace_on" else f"observer_effect_{name}"
        return self.stage(record, case_dir, stage, [
            str(self.python),
            "scripts/evaluation/serving/scheduler/compare_scheduler_trace_ab.py",
            "--trace-off", str(case_dir / "trace_off" / "ground_truth.json"),
            "--trace-on", str(case_dir / name / "ground_truth.json"),
            "--output",
            str(case_dir / name / "scheduler_trace_observer_effect.json"),
        ], self.environment(self.gpus), gpu=False, log_dir=case_dir / "logs")

    def rerun_trace_on(self, case_dir: Path, tag: str) -> int:
        """Repeat only the trace-on run and its gate for a captured case."""

        original = json.loads((case_dir / "case_capture.json").read_text())
        case_key = original["case"]
        spec = CASES[case_key]
        if original["runtime_version"] != self.runtime:
            raise ValueError(
                f"{case_dir} was captured with vLLM {original['runtime_version']}, "
                f"not {self.runtime}")
        if spec["tp"] != len(self.gpus):
            raise ValueError(f"{case_key} needs {spec['tp']} GPUs, lane has {self.gpus}")
        if not (case_dir / "trace_off" / "ground_truth.json").is_file():
            raise FileNotFoundError(f"{case_dir} has no trace_off/ground_truth.json")
        name = f"trace_on_{tag}"
        self.record_name = f"case_capture.{name}.json"
        if (case_dir / self.record_name).exists():
            raise FileExistsError(case_dir / self.record_name)
        record = {
            "schema_version": 1, "case": case_key, "hardware_id": "nvidia_h200",
            "runtime_version": self.runtime, "python": str(self.python),
            "gpus": self.gpus, "port": self.port, "host": os.uname().nodename,
            "repository_root": str(ROOT),
            "source_commit": (ROOT / "SOURCE_COMMIT").read_text().strip(),
            "rerun_of": "trace_on", "rerun_dir": name,
            "original_source_commit": original["source_commit"],
            "model": original["model"], "dataset": str(self.args.dataset),
            "started": _now(), "stages": [],
            "gpu_process_rule": original["gpu_process_rule"],
        }
        self.write_record(case_dir, record)
        returncode = self.ground_truth(record, case_dir, spec, case_key,
                                       Path(original["model"]), trace=True, name=name)
        if returncode == 0:
            returncode = self.observer_effect(record, case_dir, name)
        record["ended"] = _now()
        self.write_record(case_dir, record)
        return returncode

    def run_case(self, case_key: str) -> None:
        spec = CASES[case_key]
        if spec["tp"] != len(self.gpus):
            raise ValueError(f"{case_key} needs {spec['tp']} GPUs, lane has {self.gpus}")
        model_id, case_id = case_key.split("/")
        case_dir = self.output_root / self.runtime / "nvidia_h200" / model_id / case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        model = Path(self.args.models_root) / spec["model_dir"]
        requests = ROOT / "serving_dataset" / "sharegpt" / "nvidia_a800" / model_id / case_id / "requests.csv"
        record = {
            "schema_version": 1, "case": case_key, "hardware_id": "nvidia_h200",
            "runtime_version": self.runtime, "python": str(self.python),
            "gpus": self.gpus, "port": self.port, "host": os.uname().nodename,
            "repository_root": str(ROOT),
            # Execution snapshots carry the commit they were exported from.
            "source_commit": (ROOT / "SOURCE_COMMIT").read_text().strip(),
            "request_template_case": f"nvidia_a800/{case_key}",
            "model": str(model), "dataset": str(self.args.dataset),
            "started": _now(), "stages": [],
            "gpu_process_rule": "one process per GPU; every GPU stage holds flock -x "
                                "on each owned gpu<N>.proc.lock",
            "skip_alignment": self.args.skip_alignment,
        }
        self.write_record(case_dir, record)
        self.probe(record, case_dir, model, "before")
        off = self.ground_truth(record, case_dir, spec, case_key, model, trace=False)
        on = self.ground_truth(record, case_dir, spec, case_key, model, trace=True)
        if off == 0 and on == 0:
            self.observer_effect(record, case_dir, "trace_on")
        self.probe(record, case_dir, model, "after")
        if not self.args.skip_alignment:
            self.alignment(record, case_dir, spec, model, requests)
        record["ended"] = _now()
        self.write_record(case_dir, record)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("run", "plan", "rerun-trace-on"))
    parser.add_argument("--runtime-version", choices=SUPPORTED_RUNTIME_VERSIONS, required=True)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=(Path(os.environ["MIDLAYER_H200_PUBLIC_ROOT"])
                 if "MIDLAYER_H200_PUBLIC_ROOT" in os.environ else None),
        help="H200 dependency/output root; may use MIDLAYER_H200_PUBLIC_ROOT",
    )
    parser.add_argument("--gpus", required=True,
                        help="run: this lane's GPUs (e.g. 4 or 4,5); plan: all usable GPUs")
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--python", required=True, help="Python executable for the selected vLLM runtime")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--allow-gpu0", action="store_true",
                        help="confirm that GPU 0 is assigned to this lane")
    parser.add_argument("--case-dir", help="rerun-trace-on: an existing case directory")
    parser.add_argument("--tag", help="rerun-trace-on: writes trace_on_<tag>/")
    parser.add_argument("--pairs", type=int, default=0,
                        help="plan: number of GPU pairs reserved for TP2/EP2 lanes")
    for key in DEFAULT_PATH_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}")
    return parser


def _plan(args: argparse.Namespace) -> None:
    """Greedy lanes: TP2 cases on GPU pairs, TP1 cases on single GPUs."""

    gpus = [int(value) for value in args.gpus.split(",")]
    if 0 in gpus and not args.allow_gpu0:
        raise ValueError("GPU 0 needs --allow-gpu0")
    cases = args.case if args.case else list(CASES)
    tp2 = [case for case in cases if CASES[case]["tp"] == 2]
    tp1 = [case for case in cases if CASES[case]["tp"] == 1]
    if 2 * args.pairs > len(gpus):
        raise ValueError("not enough GPUs for the requested TP2 pairs")
    pairs = [gpus[2 * index:2 * index + 2] for index in range(args.pairs)]
    singles = gpus[2 * args.pairs:]
    if tp2 and not pairs:
        raise ValueError("TP2 cases need at least two GPUs")
    if tp1 and not singles:
        raise ValueError("TP1 cases need a GPU outside the TP2 pairs")
    lanes = [(pair, tp2[index::len(pairs)]) for index, pair in enumerate(pairs)]
    lanes += [([gpu], tp1[index::len(singles)]) for index, gpu in enumerate(singles)]
    log_root = Path(args.output_root) / args.runtime_version / "lanes"
    print(f"mkdir -p {log_root}")
    for lane_gpus, lane_cases in lanes:
        if not lane_cases:
            continue
        name = "gpu" + "-".join(map(str, lane_gpus))
        command = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--runtime-version", args.runtime_version,
                   "--public-root", str(args.public_root),
                   "--gpus", ",".join(map(str, lane_gpus))]
        if args.allow_gpu0:
            command.append("--allow-gpu0")
        for key in DEFAULT_PATH_KEYS:
            command.extend((f"--{key.replace('_', '-')}", getattr(args, key)))
        if args.python is not None:
            command.extend(("--python", args.python))
        for case in lane_cases:
            command += ["--case", case]
        print(f"setsid nohup {shlex.join(command)} > {log_root}/{name}.log 2>&1 < /dev/null &")


def main() -> int:
    args = _parser().parse_args()
    if args.public_root is None:
        raise ValueError("--public-root or MIDLAYER_H200_PUBLIC_ROOT is required")
    defaults = _default_paths(args.public_root)
    for key in DEFAULT_PATH_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, str(defaults[key]))
    if args.action == "plan":
        _plan(args)
        return 0
    if args.action == "rerun-trace-on":
        if not args.case_dir or not args.tag:
            raise ValueError("rerun-trace-on requires --case-dir and --tag")
        if not args.tag.replace("_", "").isalnum():
            raise ValueError("--tag must be alphanumeric/underscore")
        return Lane(args).rerun_trace_on(Path(args.case_dir), args.tag)
    if not args.case:
        raise ValueError("run requires at least one --case")
    lane = Lane(args)
    failures = []
    for case in args.case:
        # A failed case must not strand the lane's remaining GPU time; the
        # failure is printed now and makes the lane exit non-zero at the end.
        try:
            lane.run_case(case)
        except Exception as error:
            failures.append(f"{case}: {type(error).__name__}: {error}")
            print(f"[{_now()}] {case} FAILED: {failures[-1]}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError("lane failures:\n" + "\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
