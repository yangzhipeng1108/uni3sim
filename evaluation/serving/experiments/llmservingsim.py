"""Explicit CPU subprocess boundary for the independent LLMServingSim baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY_ROOT / "configs/baselines/llmservingsim/source.json"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def target_hardware_statuses(root: Path) -> dict[str, str]:
    """Validate the explicit execution scope while retaining the historical matrix."""
    config = read_json(root / "configs/baselines/llmservingsim/targets.json")
    statuses = config["hardware_status"]
    ids = config["hardware_ids"]
    if (not isinstance(statuses, dict) or not isinstance(ids, list)
            or any(not isinstance(hardware, str) for hardware in ids)
            or len(set(ids)) != len(ids) or set(statuses) != set(ids)
            or any(status not in ("active", "deferred_by_user") for status in statuses.values())
            or "active" not in statuses.values()):
        raise ValueError("invalid target hardware scope")
    return statuses


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def baseline_environment(runtime: Path, python: Path) -> dict:
    command = """import hashlib, importlib.metadata, importlib.util, json, platform, sys
from pathlib import Path
converter = Path(importlib.util.find_spec('chakra.src.converter.llm_converter').origin)
schema = Path(importlib.util.find_spec('chakra.schema.protobuf').submodule_search_locations[0]) / 'et_def.proto'
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
 'packages': {name: importlib.metadata.version(name) for name in
 ['chakra', 'protobuf', 'grpcio-tools', 'numpy', 'pandas', 'PyYAML', 'pyinstrument', 'rich']},
 'converter_sha256': hashlib.sha256(converter.read_bytes()).hexdigest(),
 'schema_sha256': hashlib.sha256(schema.read_bytes()).hexdigest()}))
"""
    result = json.loads(subprocess.check_output([str(python.absolute()), "-c", command], text=True))
    chakra = runtime / "astra-sim/extern/graph_frontend/chakra"
    for field, relative in (("converter_sha256", "src/converter/llm_converter.py"),
                            ("schema_sha256", "schema/protobuf/et_def.proto")):
        if hashlib.sha256((chakra / relative).read_bytes()).hexdigest() != result[field]:
            raise ValueError("installed Chakra converter/schema differs from the source checkout")
    return result


def run_bundled_case(runtime: Path, python: Path, case: str, output: Path) -> dict:
    """Run the full recorded setting; never invoke profiler or download weights."""
    source = read_json(SOURCE)
    if case not in source["bundled_cases"]:
        raise ValueError(f"unapproved bundled case: {case}")
    runtime, python, output = runtime.resolve(), python.absolute(), output.resolve()
    artifact_root = (REPOSITORY_ROOT / "experiment_logs/evaluation/llmservingsim").resolve()
    if not runtime.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError("runtime and output must be isolated in this task's artifact directory")
    if git_revision(runtime) != source["commit"]:
        raise ValueError("frontend revision differs from the source manifest")
    if git_revision(runtime / "astra-sim") != source["astra_sim_commit"]:
        raise ValueError("ASTRA-Sim revision differs from the source manifest")
    chakra = runtime / "astra-sim/extern/graph_frontend/chakra"
    if git_revision(chakra) != source["chakra_commit"]:
        raise ValueError("Chakra revision differs from the source manifest")
    binary = runtime / "astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    example = Path("bench/examples") / case
    meta = read_json(runtime / example / "vllm/meta.json")
    dataset = runtime / meta["dataset_path"]
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != meta["dataset_hash"]:
        raise ValueError("bundled dataset hash differs from recorded vLLM metadata")
    count = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    if count != meta["num_requests"]:
        raise ValueError("bundled workload must contain exactly the recorded request count")
    output.mkdir(parents=True, exist_ok=False)
    engine = meta["engine_kwargs"]
    command = [str(python), "-m", "serving", "--cluster-config", str(example / "config.json"),
               "--dataset", meta["dataset_path"], "--output", str(output / "sim.csv"),
               "--num-reqs", str(count), "--max-num-seqs", str(engine["max_num_seqs"]),
               "--max-num-batched-tokens", str(engine["max_num_batched_tokens"]),
               "--block-size", str(meta["kv_cache"]["block_size"]),
               "--network-backend", "analytical", "--log-level", "WARNING"]
    manifest = {"case": case, "command": command, "cwd": str(runtime),
                "source_manifest": str(SOURCE), "config": read_json(runtime / example / "config.json"),
                "meta": meta, "request_count": count, "started_unix_s": time.time(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "environment": baseline_environment(runtime, python),
                "status": "running", "exit_code": None}
    write_json(output / "run.json", manifest)
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TERM": "dumb"})
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, cwd=runtime, env=environment, stdout=stdout, stderr=stderr, check=False)
    manifest.update({"exit_code": result.returncode, "finished_unix_s": time.time(),
                     "status": "executed" if result.returncode == 0 else "failed"})
    write_json(output / "run.json", manifest)
    if result.returncode:
        raise RuntimeError(f"LLMServingSim exited {result.returncode}; see {output}")
    from evaluation.serving.metrics.llmservingsim_comparison import (
        native_workload, normalize_native_csv, summarize,
    )
    model = manifest["config"]["nodes"][0]["instances"][0]["model_name"]
    instance_ids = tuple(range(sum(node["num_instances"] for node in manifest["config"]["nodes"])))
    try:
        requests = normalize_native_csv(output / "sim.csv", native_workload(dataset), model, instance_ids=instance_ids)
    except (ValueError, OSError) as exc:
        manifest.update({"status": "invalid_output", "output_error": str(exc)})
        write_json(output / "run.json", manifest)
        raise
    write_json(output / "normalized.json", {"requests": requests, "summary": summarize(requests)})
    manifest["status"] = "completed"
    write_json(output / "run.json", manifest)
    return manifest


def run_bundled_suite(runtime: Path, python: Path, output: Path) -> None:
    """Run the first setting before four independent settings, then write the report."""
    cases = read_json(SOURCE)["bundled_cases"]
    run_bundled_case(runtime, python, cases[0], output / "bundled" / cases[0].replace("/", "-"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_bundled_case, runtime, python, case,
                               output / "bundled" / case.replace("/", "-")) for case in cases[1:]]
        failures = []
        for future in futures:
            try:
                future.result()
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append(str(exc))
    from evaluation.serving.metrics.llmservingsim_comparison import generate_report
    generate_report(runtime.resolve(), output.resolve())
    if failures:
        raise RuntimeError("bundled suite failed: " + "; ".join(failures))


def main(argv: list[str] | None = None, *, load_case=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case")
    mode.add_argument("--standard-results", type=Path, help="run TP1 QPS4/64 with accepted LUTs")
    parser.add_argument("--accepted-results", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--prepare-targets", action="store_true")
    mode.add_argument("--bundled-suite", action="store_true")
    mode.add_argument("--midlayer-targets", action="store_true")
    mode.add_argument("--capacity-check", action="store_true")
    mode.add_argument("--target-case", help="one active case; user-deferred historical cases cannot execute")
    parser.add_argument("--profile-capture", type=Path)
    parser.add_argument("--hardware-capture", type=Path)
    mode.add_argument("--midlayer-command", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--midlayer-execute", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.standard_results is not None:
        from evaluation.serving.experiments.llmservingsim_standard import run_standard_suite
        if args.python is None or args.accepted_results is None or load_case is None:
            parser.error("standard results require --python, --accepted-results and canonical validator")
        run_standard_suite(args.runtime, args.python, args.accepted_results, args.standard_results,
                           args.output, args.jobs, load_case)
    elif args.target_case is not None:
        from evaluation.serving.experiments.llmservingsim_targets import run_native_target
        if args.python is None or args.profile_capture is None or args.hardware_capture is None or load_case is None:
            parser.error("--target-case requires --python, --profile-capture, --hardware-capture and the thin CLI canonical validator")
        run_native_target(args.runtime, args.python, args.output, args.target_case,
                          args.profile_capture, args.hardware_capture, load_case)
    elif args.capacity_check:
        from evaluation.serving.experiments.llmservingsim_targets import compare_target_capacity
        if args.python is None:
            parser.error("--python is required with --capacity-check")
        compare_target_capacity(args.runtime, args.python, args.output)
    elif args.midlayer_execute is not None:
        from evaluation.serving.experiments.llmservingsim_targets import execute_midlayer_target
        execute_midlayer_target(args.midlayer_execute, args.output)
    elif args.midlayer_command is not None:
        from evaluation.serving.experiments.llmservingsim_targets import midlayer_target_argv
        if load_case is None:
            parser.error("canonical dataset validator is required")
        write_json(args.output, {"argv": midlayer_target_argv(args.runtime, read_json(args.midlayer_command), load_case)})
    elif args.prepare_targets or args.midlayer_targets:
        from evaluation.serving.experiments.llmservingsim_targets import prepare_targets, run_midlayer_targets
        if load_case is None:
            parser.error("target preparation requires the thin CLI's canonical dataset validator")
        if args.prepare_targets:
            prepare_targets(args.runtime, args.output, load_case)
        else:
            if args.python is None:
                parser.error("--python is required with --midlayer-targets")
            result = run_midlayer_targets(args.output, args.python, load_case)
            if any(case["status"] != "executed" for case in result["cases"]):
                return 1
    elif args.report:
        from evaluation.serving.metrics.llmservingsim_comparison import generate_report
        generate_report(args.runtime.resolve(), args.output.resolve())
    elif args.bundled_suite:
        if args.python is None:
            parser.error("--python is required with --bundled-suite")
        run_bundled_suite(args.runtime, args.python, args.output)
    else:
        if args.python is None:
            parser.error("--python is required with --case")
        run_bundled_case(args.runtime, args.python, args.case, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
