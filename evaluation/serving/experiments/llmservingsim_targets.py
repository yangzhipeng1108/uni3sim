"""Preserve and validate the complete target denominator without invented profiles."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable

from evaluation.serving.experiments.llmservingsim import (
    REPOSITORY_ROOT, SOURCE, baseline_environment, git_revision, read_json, write_json,
    target_hardware_statuses,
)


def repository_payload(path: Path) -> bytes:
    """Read a local pinned LFS object without hydrating or mutating canonical data."""
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        match = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n", raw)
        if match is None:
            raise ValueError(f"invalid LFS pointer: {path}")
        oid = match[1].decode()
        common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=REPOSITORY_ROOT, text=True).strip())
        if not common.is_absolute():
            common = REPOSITORY_ROOT / common
        raw = (common / "lfs/objects" / oid[:2] / oid[2:4] / oid).read_bytes()
        if len(raw) != int(match[2]) or hashlib.sha256(raw).hexdigest() != oid:
            raise ValueError(f"LFS payload differs from pointer: {path}")
    return raw


def convert_requests(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert only arrivals/lengths, retaining original IDs and cache fields alongside."""
    native, mapping = [], []
    seen = set()
    for index, row in enumerate(rows):
        request_id = row["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("request IDs must be nonempty unique strings")
        seen.add(request_id)
        for field in ("input_tokens", "output_tokens", "cached_hit_tokens"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if row["input_tokens"] == 0 or row["output_tokens"] == 0:
            raise ValueError("native baseline requires positive input/output lengths")
        if row["cached_hit_tokens"] > row["input_tokens"]:
            raise ValueError("cached prefix exceeds input length")
        if type(row["arrival_time_ms"]) not in (int, float):
            raise ValueError("arrival must be numeric")
        milliseconds = Decimal(str(row["arrival_time_ms"]))
        nanos = milliseconds * 1_000_000
        if not nanos.is_finite() or nanos < 0:
            raise ValueError("arrival must be finite and nonnegative")
        rounded = nanos.to_integral_value(rounding=ROUND_HALF_EVEN)
        native.append({"input_toks": row["input_tokens"], "output_toks": row["output_tokens"],
                       "arrival_time_ns": int(rounded)})
        mapping.append({"native_request_id": index, **row,
                        "arrival_rounding_delta_ns": float(rounded - nanos)})
    if not rows:
        raise ValueError("request stream must not be empty")
    return native, mapping


def active_target_entries(entries: list[dict]) -> list[dict]:
    from evaluation.serving.metrics.llmservingsim_comparison import summarize_target_scope

    config = read_json(REPOSITORY_ROOT / 'configs/baselines/llmservingsim/targets.json')
    scope = target_hardware_statuses(REPOSITORY_ROOT)
    summarize_target_scope(entries, scope, [model['model_id'] for model in config['models']], config['case_id'])
    return [entry for entry in entries if scope[entry['case'].split('/')[0]] == 'active']


def prepare_targets(runtime: Path, output: Path, load_case: Callable) -> dict:
    runtime, output = runtime.resolve(), output.resolve()
    artifact_root = (REPOSITORY_ROOT / "experiment_logs/evaluation/llmservingsim").resolve()
    if not runtime.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError("runtime and output must be isolated in this task's artifact directory")
    targets = read_json(REPOSITORY_ROOT / "configs/baselines/llmservingsim/targets.json")
    scope = target_hardware_statuses(REPOSITORY_ROOT)
    settings = read_json(REPOSITORY_ROOT / targets["prediction_settings"])["prediction_contract"]
    cases = []
    for hardware in targets["hardware_ids"]:
        for model in targets["models"]:
            relative = Path("serving_dataset/sharegpt") / hardware / model["model_id"] / targets["case_id"]
            directory = output / "targets" / hardware / model["model_id"]
            canonical = directory / "canonical"
            canonical.mkdir(parents=True, exist_ok=True)
            for source in sorted((REPOSITORY_ROOT / relative).iterdir()):
                if not source.is_file():
                    raise ValueError(f"unexpected canonical subdirectory: {source}")
                (canonical / source.name).write_bytes(repository_payload(source))
            case = load_case(canonical)
            rows = [{"request_id": str(i), "arrival_time_ms": r.arrival_time_ms,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "cached_hit_tokens": r.cached_hit_tokens} for i, r in enumerate(case.requests)]
            native, mapping = convert_requests(rows)
            (directory / "workload.jsonl").write_text("".join(json.dumps(row) + "\n" for row in native))
            write_json(directory / "request_mapping.json", {"requests": mapping})
            model_config = read_json(REPOSITORY_ROOT / model["config"])
            native_config = runtime / "configs/model" / (model["native_model"] + ".json")
            native_config.parent.mkdir(parents=True, exist_ok=True)
            write_json(native_config, model_config)
            profile = runtime / "profiler/perf" / hardware / model["native_model"] / "bf16"
            if profile.exists():
                raise ValueError(f"new target profile requires explicit semantic/coverage validation: {profile}")
            entry = {"case": f"{hardware}/{model['model_id']}/{targets['case_id']}",
                     "scope_status": scope[hardware],
                     "request_count": len(native), "status": "missing_profile",
                     "canonical_case": str(relative), "preserved_inputs": str(canonical.resolve()),
                     "native_model": model["native_model"], "native_hardware": hardware,
                     "model_config_source": model["config"], "model_config": model_config,
                     "expected_profile": str(profile.resolve()),
                     "required_bundle": ["meta.yaml", "tp1/dense.csv", "tp1/attention.csv", "tp1/per_sequence.csv", "hardware.yaml at hardware root"],
                     "required_profile_domain": {"dtype": targets["dtype"], "tp": targets["tensor_parallel_size"],
                         "tokens": settings["max_batched_tokens"], "sequences": settings["max_requests"],
                         "max_context_tokens": settings["max_model_len"],
                         "prompt_tokens_range": [min(r["input_toks"] for r in native), max(r["input_toks"] for r in native)],
                         "output_tokens_range": [min(r["output_toks"] for r in native), max(r["output_toks"] for r in native)]},
                     "canonical_runtime": case.manifest["runtime"], "prediction_settings": settings,
                     "runtime_memory_policy": read_json(REPOSITORY_ROOT / f"configs/runtime_memory_policy/{hardware}/{model['model_id']}.json"),
                     "cache_evidence": case.manifest["requests"]["cached_hit_value_evidence"],
                     "truth_status": case.manifest["metrics"]["status"],
                     "comparability_gaps": ["No same-hardware/model/precision/TP layerwise performance bundle",
                         "Canonical CSV has lengths and cache representatives, not token IDs; exact native prefix sharing is unproven",
                         "Native weight-only KV budget omits Midlayer runtime policy non-KV reserves; equal memory utilization alone is insufficient"],
                     "llmservingsim_prediction": None, "midlayer_comparison": None}
            write_json(directory / "case.json", entry)
            cases.append(entry)
    result = {"expected_cases": 8, "prepared_cases": len(cases), "completed_simulations": 0,
              "missing_profiles": len(cases), "cases": cases}
    write_json(output / "targets.json", result)
    return result


def midlayer_target_argv(snapshot: Path, entry: dict, load_case: Callable) -> list[str]:
    hardware, _, _ = entry["case"].split("/")
    if target_hardware_statuses(REPOSITORY_ROOT)[hardware] == 'deferred_by_user':
        raise ValueError('target deferred by user; no Midlayer execution is permitted')
    from evaluation.serving.metrics import sharegpt_compact_evaluation as compact

    case = load_case(entry["preserved_inputs"])
    config_name = ("nvidia_h200_vllm030_sharegpt_tp1_suite.json" if hardware == "nvidia_h200"
                   else "sharegpt_phase1.json")
    config = compact.load_phase1_config(snapshot / "configs/evaluation/experiments" / config_name)
    cell, = compact.select_phase1_cells(config, [entry["case"]], all_cases=False)
    if hardware in compact.host_profile_variant_hardware():
        profiles = compact.case_host_profile_paths(snapshot, hardware, case.manifest["runtime"]["runtime_version"])
        launch, scheduler = profiles["host_launch"], profiles["scheduler_host"]
    else:
        launch = snapshot / "configs/host_timing/host_launch_profiles" / f"{hardware}.json"
        scheduler = snapshot / "configs/host_timing/scheduler_host_profiles" / f"{hardware}.json"
    return compact.build_simulation_argv(snapshot, config, cell, case,
                serving_timing_mode="device_plus_launch_plus_scheduler",
                host_launch_profile_path=launch, scheduler_host_profile_path=scheduler)


def execute_midlayer_target(argv_path: Path, output: Path) -> None:
    import yaml

    argv = read_json(argv_path)['argv']
    if argv.count('--hardware-config') != 1:
        raise ValueError('Midlayer target requires one explicit hardware configuration')
    hardware_config = Path(argv[argv.index('--hardware-config') + 1])
    hardware = yaml.safe_load(hardware_config.read_text())['hardware_id']
    if target_hardware_statuses(REPOSITORY_ROOT)[hardware] == 'deferred_by_user':
        raise ValueError('target deferred by user; no Midlayer execution is permitted')
    from evaluation.serving.simulation.simulation_runner import run_cli_args
    from evaluation.serving.simulation.simulation_artifacts import persistence_projection

    payload = run_cli_args(argv)
    write_json(output, {"aggregate": persistence_projection(payload),
                       "requests": payload["metrics"]["requests"]})


def compare_target_capacity(runtime: Path, python: Path, output: Path) -> dict:
    """Probe native static allocation on CPU, without reading any duration/profile."""
    runtime, output = runtime.resolve(), output.resolve()
    artifact_root = (REPOSITORY_ROOT / "experiment_logs/evaluation/llmservingsim").resolve()
    if not runtime.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError("capacity probe must stay in the isolated task directory")
    if git_revision(runtime) != read_json(SOURCE)["commit"]:
        raise ValueError("native source revision differs from the source manifest")
    inputs = []
    scope = target_hardware_statuses(REPOSITORY_ROOT)
    for entry in active_target_entries(read_json(output / "targets.json")["cases"]):
        hardware, model, _ = entry["case"].split("/")
        prediction_path = output / "targets" / hardware / model / "midlayer-request-metrics/prediction.json"
        pool, = read_json(prediction_path)["aggregate"]["kv_capacity"]["scheduler_pools"]
        rank, = pool["rank_budgets"]
        model_digest = hashlib.sha256((REPOSITORY_ROOT / entry["model_config_source"]).read_bytes()).hexdigest()
        if pool["hardware_id"] != hardware or pool["model_config_sha256"] != model_digest:
            raise ValueError("allocation manifest hardware/model differs from the target case")
        if pool["tensor_parallel_size"] != 1 or pool["pipeline_parallel_size"] != 1 or pool["expert_parallel_size"] != 1:
            raise ValueError("capacity probe supports the approved TP1 dense cases only")
        inputs.append({"case": entry["case"], "native_model": entry["native_model"],
                       "allocation_manifest": str(prediction_path),
                       "runtime_memory_policy": pool["runtime_memory_policy"],
                       "physical_hbm_bytes": rank["physical_hbm_bytes"],
                       "midlayer_weight_bytes": rank["model_weight_resident_bytes"],
                       "non_kv_reserve_bytes": rank["non_kv_reserve_bytes"],
                       "available_kv_bytes": rank["available_kv_bytes"],
                       "midlayer_blocks": pool["num_kv_blocks"],
                       "midlayer_block_bytes": pool["kv_block_bytes"],
                       "block_size_tokens": pool["kv_block_size_tokens"]})
    native_code = """from concurrent.futures import ThreadPoolExecutor
import json, sys
from serving.core.memory_model import MemoryModel
def check(row):
    utilization = row['runtime_memory_policy']['gpu_memory_utilization']
    configured = utilization['numerator'] / utilization['denominator']
    def memory(value):
        return MemoryModel(row['native_model'], 0, 0, 1, 1,
            row['physical_hbm_bytes'] / (1024 ** 3), 0, row['block_size_tokens'], 16,
            True, False, None, None, npu_memory_utilization=value)
    original = memory(configured)
    # Translate a specified KV byte budget into native weights-plus-KV units.
    # This uses only allocation fields, never times or error-dependent parameters.
    equivalent = (original.weight + row['available_kv_bytes']) / row['physical_hbm_bytes']
    if not 0 < equivalent <= 1:
        raise ValueError('specified capacity cannot be expressed by native utilization')
    matched = memory(equivalent)
    return {**row, 'configured_utilization': configured,
        'native_weight_bytes': original.weight,
        'native_blocks_at_same_utilization': original.npu_pool.num_blocks,
        'native_block_bytes': original.npu_pool.bytes_per_block,
        'native_utilization_for_same_kv_budget': equivalent,
        'native_matched_blocks': matched.npu_pool.num_blocks,
        'equal_block_bytes': original.npu_pool.bytes_per_block == row['midlayer_block_bytes'],
        'equal_block_count': matched.npu_pool.num_blocks == row['midlayer_blocks']}
rows = json.load(sys.stdin)
with ThreadPoolExecutor(max_workers=8) as executor:
    print(json.dumps(list(executor.map(check, rows)), allow_nan=False))
"""
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
    native = subprocess.run([str(python.absolute()), "-c", native_code], cwd=runtime,
                            input=json.dumps(inputs), text=True, capture_output=True,
                            env=environment, check=True)
    rows = json.loads(native.stdout)
    if len(rows) != len(inputs):
        raise ValueError("native capacity probe returned an incomplete denominator")
    if any(not row["equal_block_bytes"] or not row["equal_block_count"] for row in rows):
        raise ValueError("native allocation does not match the specified KV capacity")
    previous_path = output / 'capacity-comparison.json'
    historical_deferred = []
    if previous_path.is_file():
        historical_deferred = [row for row in read_json(previous_path)['cases']
                               if scope[row['case'].split('/')[0]] == 'deferred_by_user']
    result = {"kind": "CPU static allocation probe; no target end-to-end simulation",
              "source_manifest": str(SOURCE), "environment": baseline_environment(runtime, python),
              "basis": "configuration-only KV allocation fields; no durations accessed",
              "mapping": "native utilization = (native model weight bytes + specified available KV bytes) / physical HBM bytes",
              "limitations": "Only logical KV block capacity is checked; prefix sharing and complete runtime equivalence remain unproven. No target config or profile is changed.",
              "native_stderr": native.stderr, "current_case_count": len(rows),
              "deferred_history": [row['case'] for row in historical_deferred],
              "cases": rows + historical_deferred}
    write_json(output / "capacity-comparison.json", result)
    return result


def run_midlayer_targets(output: Path, python: Path, load_case: Callable) -> dict:
    """Run active cases only; deferred historical results are preserved."""
    output = output.resolve()
    artifact_root = (REPOSITORY_ROOT / "experiment_logs/evaluation/llmservingsim").resolve()
    if not output.is_relative_to(artifact_root):
        raise ValueError("Midlayer outputs must stay in the baseline artifact directory")
    targets = read_json(output / "targets.json")
    active_cases = active_target_entries(targets['cases'])
    scope = target_hardware_statuses(REPOSITORY_ROOT)
    snapshot = output / "midlayer-runtime"
    snapshot.mkdir(exist_ok=True)
    paths = subprocess.check_output(["git", "ls-files", "-z", "src", "evaluation", "dataset_pipeline",
                                     "scripts", "configs", "launch_data", "microbench"],
                                    cwd=REPOSITORY_ROOT, text=True).split("\0")
    for relative in paths:
        if relative:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(repository_payload(REPOSITORY_ROOT / relative))
    (snapshot / "build").mkdir(exist_ok=True)
    shutil.copyfile(REPOSITORY_ROOT / "build/lru_sim.so", snapshot / "build/lru_sim.so")
    canonical = REPOSITORY_ROOT.parent / "midlayer"
    for name in ("serving_dataset", "profiler_output"):
        link = snapshot / name
        if not link.exists():
            link.symlink_to(canonical / name, target_is_directory=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    write_json(output / "midlayer-snapshot.json", {"source_commit": source_commit,
               "code": "tracked worktree production code; canonical LFS payloads resolved read-only",
               "python": str(python.absolute()), "snapshot": str(snapshot)})
    def run_one(entry):
        hardware, model, _ = entry["case"].split("/")
        directory = output / "targets" / hardware / model / "midlayer-request-metrics"
        if (directory / "run.json").is_file():
            previous = read_json(directory / "run.json")
            if previous["status"] == "executed" and previous["exit_code"] == 0:
                return previous
        directory.mkdir(exist_ok=False)
        environment = dict(os.environ)
        environment.update({"PYTHONPATH": str(snapshot), "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})
        builder = [str(python.absolute()), "scripts/evaluation/serving/run_llmservingsim.py",
                   "--midlayer-command", str(directory.parent / "case.json"),
                   "--runtime", str(snapshot), "--output", str(directory / "argv.json")]
        built = subprocess.run(builder, cwd=snapshot, env=environment, capture_output=True, text=True, check=False)
        (directory / "preflight.log").write_text(built.stdout + built.stderr)
        if built.returncode:
            state = {"case": entry["case"], "status": "preflight_failed", "exit_code": built.returncode,
                     "command": builder, "error": built.stderr}
            write_json(directory / "run.json", state)
            return state
        argv = read_json(directory / "argv.json")["argv"]
        command = [str(python.absolute()), "scripts/evaluation/serving/run_llmservingsim.py",
                   "--midlayer-execute", str(directory / "argv.json"), "--runtime", str(snapshot),
                   "--output", str(directory / "prediction.json")]
        state = {"case": entry["case"], "status": "running", "command": command,
                 "cwd": str(snapshot), "source_manifest": str(output / "midlayer-snapshot.json"), "exit_code": None}
        write_json(directory / "run.json", state)
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, cwd=snapshot, env=environment, stdout=stdout, stderr=stderr, check=False)
        state.update({"exit_code": process.returncode, "status": "executed" if process.returncode == 0 else "failed"})
        write_json(directory / "run.json", state)
        return state
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_one, entry) for entry in active_cases]
        states = []
        for entry, future in zip(active_cases, futures):
            try:
                states.append(future.result())
            except (ValueError, OSError, RuntimeError) as exc:
                states.append({"case": entry["case"], "status": "preflight_failed", "error": str(exc)})
    result = {"expected_cases": len(active_cases), "cases": states,
              "deferred_cases": [entry['case'] for entry in targets['cases']
                                 if scope[entry['case'].split('/')[0]] == 'deferred_by_user']}
    write_json(output / "midlayer-targets.json", result)
    return result


def run_native_target(runtime: Path, python: Path, output: Path, case_id: str,
                      capture: Path, hardware_capture: Path, load_case: Callable) -> dict:
    """Validate a real capture, then run the original complete TP1 target stream."""
    import csv
    import math
    import yaml
    from evaluation.serving.metrics.llmservingsim_comparison import normalize_native_csv, summarize

    scope = target_hardware_statuses(REPOSITORY_ROOT)
    if scope[case_id.split('/')[0]] == 'deferred_by_user':
        raise ValueError('target deferred by user; no simulation or profile access is permitted')
    runtime, output = runtime.resolve(), output.resolve()
    artifact_root = (REPOSITORY_ROOT / 'experiment_logs/evaluation/llmservingsim').resolve()
    raw_root = (REPOSITORY_ROOT / 'profiler_output/serving/llmservingsim-20261016').resolve()
    capture, hardware_capture = capture.resolve(), hardware_capture.resolve()
    if not runtime.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError('target simulation must use the isolated execution copy')
    if not capture.is_relative_to(raw_root) or not hardware_capture.is_relative_to(raw_root):
        raise ValueError('target bundle must be bound to this task raw campaign')
    source = read_json(SOURCE)
    for checkout, key in ((runtime, 'commit'), (runtime / 'astra-sim', 'astra_sim_commit'),
                          (runtime / 'astra-sim/extern/graph_frontend/chakra', 'chakra_commit')):
        if git_revision(checkout) != source[key]:
            raise ValueError('native simulation source differs from the source binding')
    targets = read_json(output / 'targets.json')
    entries = [row for row in targets['cases'] if row['case'] == case_id]
    if len(entries) != 1:
        raise ValueError('target case must be uniquely present in the full denominator')
    entry, = entries
    hardware, model_id, _ = case_id.split('/')
    recorded = read_json(capture / 'run.json')
    if (recorded['status'] != 'completed' or recorded['exit_code'] != 0
            or recorded['hardware'] != hardware or entry['native_model'] not in recorded['command']):
        raise ValueError('bundle is not a successful same-hardware/model capture')
    if recorded['model_config'] != entry['model_config']:
        raise ValueError('profile and target model configurations differ')
    variant = capture / 'perf' / hardware / entry['native_model'] / 'bf16'
    meta = yaml.safe_load((variant / 'meta.yaml').read_text())
    for field, expected in (('hardware', hardware), ('model', entry['native_model']), ('variant', 'bf16')):
        if meta[field] != expected:
            raise ValueError('bundle identity differs from the target')
    if meta['tp_degrees'] != [1] or meta['engine_effective']['dtype'] != 'bfloat16':
        raise ValueError('target requires BF16 TP1 profiles')
    settings = entry['prediction_settings']
    resolved = meta['engine_resolved']['per_tp']['1']
    if (resolved['block_size'] != settings['kv_block_size_tokens']
            or resolved['max_model_len'] != settings['max_model_len']):
        validation = {'case': case_id, 'status': 'profile_geometry_mismatch',
                      'profile_capture': str(capture), 'simulation_executed': False,
                      'request_count': entry['request_count'],
                      'requested': {'block_size': settings['kv_block_size_tokens'],
                                    'max_model_len': settings['max_model_len']},
                      'resolved': resolved}
        entry.update({'status': 'profile_geometry_mismatch', 'llmservingsim_prediction': None,
                      'profile_validation': validation,
                      'comparability_gaps': ['Captured engine geometry differs from the unchanged target settings; no simulation executed']})
        directory = output / 'targets' / hardware / model_id
        write_json(directory / 'profile-validation.json', validation)
        write_json(directory / 'case.json', entry)
        raise ValueError('resolved profile cache/context geometry differs from the target')
    collection_adapter = None
    if hardware == 'ascend_910b' and recorded['block16_evidence'] is not None:
        binding = recorded['block16_evidence']
        capability_path = capture.parent / Path(binding['path']).name
        if hashlib.sha256(capability_path.read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('Ascend block capability evidence changed after capture')
        capability = read_json(capability_path)
        remote_raw_root = Path(binding['path']).parent
        for probe in capability['probes']:
            local = capture.parent / Path(probe['path']).relative_to(remote_raw_root)
            if hashlib.sha256(local.read_bytes()).hexdigest() != probe['sha256']:
                raise ValueError('Ascend operator control differs from the capture binding')
            result = read_json(local)
            if result['status'] != 'completed' or result['exit_code'] != 0:
                raise ValueError('Ascend operator control did not complete')
        collection_adapter = {
            'kind': 'validated-native-block16-capability-v1',
            'capability_evidence': str(capability_path), 'sha256': binding['sha256'],
            'scope': 'collector-private block sizing policy and supported-size declaration',
            'native_operations': 'unchanged FIA, paged-attention and cache device implementations',
            'actual_block_size': resolved['block_size'], 'vendor_default_block_size': 128}
    hw_path = hardware_capture / 'perf' / hardware / 'hardware.yaml'
    hw = yaml.safe_load(hw_path.read_text())
    if hw['hardware'] != hardware:
        raise ValueError('hardware metadata identity differs')
    coverage = {}
    for group in ('dense', 'per_sequence', 'attention'):
        table = variant / 'tp1' / f'{group}.csv'
        with table.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None or not {'time_us', 'measurement_protocol', 'measurement_sha256'}.issubset(fields):
                raise ValueError('profile table lacks acquisition identity')
            rows = list(reader)
        if not rows:
            raise ValueError('empty target profile table')
        coordinates = [name for name in fields if name not in ('time_us', 'measurement_protocol', 'measurement_sha256')]
        keys = set()
        identities = set()
        for row in rows:
            key = tuple(row[name] for name in coordinates)
            if key in keys or not math.isfinite(float(row['time_us'])) or float(row['time_us']) <= 0:
                raise ValueError('invalid or duplicate measured profile point')
            if not re.fullmatch('[0-9a-f]{64}',row['measurement_sha256']):
                raise ValueError('invalid acquisition fingerprint')
            if row['measurement_protocol'] not in ('cuda-active-union-dummy-kv-query-v4', 'npu-active-union-dummy-kv-query-v1'):
                raise ValueError('unsupported device-time acquisition protocol')
            keys.add(key)
            identities.add((row['measurement_protocol'],row['measurement_sha256']))
        if len(identities) != 1:
            raise ValueError('profile table mixes acquisition methods')
        coverage[group] = {'rows': len(rows), 'coordinates': coordinates,
                           'acquisition': sorted(identities),
                           'measured_ranges': {name: [min(float(row[name]) for row in rows),
                                                       max(float(row[name]) for row in rows)]
                                               for name in coordinates if name != 'layer'},
                           'sha256': hashlib.sha256(table.read_bytes()).hexdigest()}
    installed = runtime / 'profiler/perf' / hardware
    installed.mkdir(parents=True, exist_ok=True)
    destination = installed / entry['native_model'] / 'bf16'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == variant:
        pass
    elif destination.exists() or destination.is_symlink():
        raise ValueError('another target bundle is already installed')
    else:
        destination.symlink_to(variant, target_is_directory=True)
    hardware_destination = installed / 'hardware.yaml'
    if hardware_destination.is_symlink() and hardware_destination.resolve() == hw_path:
        pass
    elif hardware_destination.exists() or hardware_destination.is_symlink():
        raise ValueError('different target hardware metadata is already installed')
    else:
        hardware_destination.symlink_to(hw_path)
    directory = output / 'targets' / hardware / model_id
    runs = directory / 'llmservingsim'
    runs.mkdir(exist_ok=True)
    simulation = runs / f'run-{len(list(runs.glob("run-*"))) + 1:04d}'
    simulation.mkdir(exist_ok=False)
    capacities = [row for row in read_json(output / 'capacity-comparison.json')['cases'] if row['case'] == case_id]
    capacity, = capacities
    if not capacity['equal_block_bytes'] or not capacity['equal_block_count']:
        raise ValueError('target KV byte budget has not been matched')
    memory = {'mem_size': capacity['physical_hbm_bytes'] / 1024**3,
              'mem_bw': hw['spec']['memory_bw_gbps'], 'mem_latency': 0,
              'mem_util': capacity['native_utilization_for_same_kv_budget']}
    # Network values are explicit and unused for one node / one rank. They
    # do not claim measured interconnect performance.
    config = {'num_nodes':1, 'link_bw':1, 'link_latency':0, 'nodes':[
        {'num_instances':1, 'cpu_mem':{'mem_size':1,'mem_bw':1,'mem_latency':0}, 'instances':[
            {'model_name':entry['native_model'], 'hardware':hardware, 'npu_mem':memory,
             'num_npus':1,'tp_size':1,'pd_type':None,
             'enable_prefix_caching':settings['enable_prefix_caching'],
             'enable_chunked_prefill':settings['enable_chunked_prefill'],
             'async_scheduling':True}]}]}
    write_json(simulation / 'config.json',config)
    mapping = read_json(directory / 'request_mapping.json')['requests']
    canonical_case = load_case(Path(entry['preserved_inputs']))
    if len(mapping) != entry['request_count'] or len(mapping) != len(canonical_case.requests):
        raise ValueError('target mapping changes the original request denominator')
    for index, (mapped, canonical) in enumerate(zip(mapping, canonical_case.requests)):
        if (mapped['request_id'] != str(index)
                or mapped['arrival_time_ms'] != canonical.arrival_time_ms
                or mapped['input_tokens'] != canonical.input_tokens
                or mapped['output_tokens'] != canonical.output_tokens
                or mapped['cached_hit_tokens'] != canonical.cached_hit_tokens):
            raise ValueError('target mapping differs from validated canonical requests')
    native, _ = convert_requests(mapping)
    evidence = entry['cache_evidence']
    if (evidence['kind'] != 'scheduler_equivalent_representative'
            or evidence['representative_value'] != 0 or evidence['net_prefix_work_reduction_tokens'] != 0
            or any(r['cached_hit_tokens'] != 0 for r in mapping)):
        raise ValueError('this target bridge requires the evidenced zero-work-reduction representative')
    # Unique synthetic block-token identities instantiate the accepted
    # representative while leaving prefix allocation enabled. They are not
    # claimed to be captured prompt tokens or observed prefix hit counts.
    for index, row in enumerate(native):
        row['input_tok_ids'] = [index+1] * row['input_toks']
        row['output_tok_ids'] = [index+1] * row['output_toks']
    workload = simulation / 'workload.jsonl'
    workload.write_text(''.join(json.dumps(row)+'\n' for row in native))
    command = [str(python.absolute()),'-m','serving','--cluster-config',str(simulation/'config.json'),
               '--dataset',os.path.relpath(workload,runtime),'--output',str(simulation/'sim.csv'),
               '--num-reqs',str(entry['request_count']),'--max-num-seqs',str(settings['max_requests']),
               '--max-num-batched-tokens',str(settings['max_batched_tokens']),
               '--block-size',str(settings['kv_block_size_tokens']),
               '--network-backend','analytical','--log-level','WARNING']
    manifest = {'case':case_id,'source_manifest':str(SOURCE),'command':command,'cwd':str(runtime),
                'collection_adapter': collection_adapter,
                'adapter_baseline':git_revision(REPOSITORY_ROOT),
                'adapter_sources':{str(path.relative_to(REPOSITORY_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__), REPOSITORY_ROOT/'evaluation/serving/experiments/llmservingsim.py',
                                                REPOSITORY_ROOT/'evaluation/serving/metrics/llmservingsim_comparison.py')},
                'started_unix_s':time.time(),
                'binary_sha256':hashlib.sha256((runtime/'astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra').read_bytes()).hexdigest(),
                'request_count':len(native),'profile_capture':str(capture),'profile_metadata':meta,
                'profile_tables':coverage,'hardware_capture':str(hardware_capture),
                'cache_evidence':evidence,'cache_representation':'unique synthetic block tokens; scheduler-equivalent zero net work reduction, logical hits unobserved',
                'capacity':capacity,'compute_mode':'LUT','network_mode':'analytical',
                'unused_resources':'CPU memory (1 GiB, 1 GB/s) and interconnect (1 GiB/s, 0 ns) are inert positive config placeholders: no offloading, remote memory, network traffic or TP collectives in this TP1 case. They are not device measurements.',
                'environment':baseline_environment(runtime,python),'status':'running','exit_code':None}
    write_json(simulation/'run.json',manifest)
    environment = dict(os.environ)
    environment.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TERM':'dumb'})
    with (simulation/'stdout.log').open('w') as stdout,(simulation/'stderr.log').open('w') as stderr:
        result = subprocess.run(command,cwd=runtime,env=environment,stdout=stdout,stderr=stderr,check=False)
    manifest.update({'status':'executed' if result.returncode == 0 else 'failed',
                     'exit_code':result.returncode,'finished_unix_s':time.time()})
    write_json(simulation/'run.json',manifest)
    entry['status'] = manifest['status']
    entry['llmservingsim_prediction'] = str(simulation/'normalized.json') if result.returncode == 0 else None
    if result.returncode == 0:
        expected = [{'request_id':r['request_id'],**n} for r,n in zip(mapping,native)]
        try:
            normalized = normalize_native_csv(simulation/'sim.csv',expected,entry['native_model'])
        except (ValueError, OSError) as exc:
            manifest.update({'status': 'invalid_output', 'output_error': str(exc)})
            entry.update({'status': 'invalid_output', 'llmservingsim_prediction': None})
            write_json(simulation/'run.json',manifest)
            write_json(directory/'case.json',entry)
            raise
        write_json(simulation/'normalized.json',{'requests':normalized,'summary':summarize(normalized)})
        manifest['status'] = 'completed'
        entry['status'] = 'completed'
        entry['comparability_gaps'] = [
            'Native one-layer dummy-weight eager module timing; graph execution and capture software build may differ',
            'Uniform-only attention pilot; native interpolation/extrapolation retained, no skew measurements',
            'Logical prefix hits unobserved; unique synthetic token IDs instantiate the evidenced scheduler-equivalent zero-work-reduction representative',
            'KV byte budget matched; native and Midlayer differ in host timing and scheduler implementation']
        if collection_adapter is not None:
            entry['comparability_gaps'].append(
                'Collector-private block16 capability adapter; official vendor policy advertises128. '
                'Actual block16 caches and native device operations were validated independently.')
        log = (simulation/'stdout.log').read_text()
        hits = re.findall(r'^NPU prefix hit prompt tokens:\s+([0-9]+)\s*$',log,re.MULTILINE)
        if hits != ['0']:
            manifest.update({'status':'invalid_output','output_error':'native cache work does not match the evidenced representative'})
            entry.update({'status':'invalid_output','llmservingsim_prediction':None})
            write_json(simulation/'run.json',manifest)
            write_json(directory/'case.json',entry)
            raise ValueError('native cache work does not match the evidenced representative')
        manifest['native_prefix_hit_prompt_tokens'] = 0
        write_json(simulation/'run.json',manifest)
    write_json(directory/'case.json',entry)
    if result.returncode:
        raise RuntimeError(f'target simulator failed: {simulation}')
    return manifest
