"""Admit prepared H200 NCU batches and pinned partial-batch tail candidates."""

from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile


MISSING15_SOURCE_NAME = "missing15_ncu_review_only_20261102.json"
MISSING15_REFERENCE_KIND = "missing15_nsys_candidate_raw_only"
MISSING15_BATCH_KIND = "h200_ncu_missing15_raw_candidate"
QUALITY_RETRY_SOURCE_NAME = "ncu_quality_retry_source_20261102.json"
QUALITY_RETRY_BATCH_KIND = "h200_ncu_quality_retry_candidate"
CLOCK_SETTLE_RETRY_BATCH = "ncu_clock_settle_retry_fa_b64_20261102"
CLOCK_SETTLE_RETRY_CASE = "FLASHATTN_PAGED_PREFILL_B64_S2048_HQ32_HKV8_D128_C1"
DIRECTED7_DIAGNOSTIC = (
    "experiment_logs/h200_accuracy_20261028/"
    "ncu_remaining8_clock_reason_diagnostic_20261102.json")
DIRECTED7_PREFIX = "ncu_directed7_"
DIRECTED7_UNSTARTED_REPLACEMENT = "ncu_directed7_r2_07b_20261103"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_profiler_kernel_name(name: str) -> str:
    """Remove only known NSYS/NCU C++ demangler display differences.

    The full template and argument list remain.  Range, ordinal, geometry,
    shared memory and cluster checks are separate required identity gates.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("profiler kernel name must be non-empty")
    text = re.sub(r"\b(?:[A-Za-z_][A-Za-z_0-9]*::)+", "", name)
    text = re.sub(
        r"\((?:int|bool|unsigned long|long|short|size_t|"
        r"Fp8KVCacheDataType)\)", "", text)
    return re.sub(r"\s+", "", text)


def expected_launch_signature(case: dict) -> list:
    if "reference_kind" in case:
        if case["reference_kind"] != MISSING15_REFERENCE_KIND:
            raise ValueError("unknown NCU launch reference kind")
        return [[item["name"], item["grid"], item["block"]]
                for item in case["reference_launch_signature"]]
    return case["canonical_launch_signature"]


FA_INSTRUCTION_COUNTER_METRICS = (
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "sm__icc_requests.sum",
    "sm__icc_requests_lookup_hit.sum",
    "sm__icc_requests_lookup_miss.sum",
    "smsp__inst_executed.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
)


def extract_fa_online_phase_work(text: str, regions: list[dict], block_n: int) -> list[dict]:
    """Count coarse online work on each side of an existing wait0 boundary.

    Static work/order only. No instruction latency, duration, dynamic sample
    percentage or predictor formula is extracted from the profiler export.
    """
    if type(block_n) is not int or block_n <= 0 or block_n % 4 or not isinstance(regions, list):
        raise ValueError("FA online phases require an explicit register geometry and region list")
    expected = {"FMNMX": block_n // 2 + 4, "EX2": block_n // 2 + 2,
                "FFMA": block_n // 2 + 2, "FADD": block_n // 2}
    parsed = []
    for line in text.splitlines():
        match = re.match(r"^(0x[0-9a-f]+)\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if match is not None:
            parsed.append((match[1], match[2]))
    addresses = [pc for pc, _ in parsed]
    if len(addresses) != len(set(addresses)) or not regions:
        raise ValueError("FA online phase export has duplicate PCs or no regions")
    result = []
    for region in regions:
        for name in ("wait1_pc", "wait0_pc", "first_exp_pc"):
            if region[name] not in addresses:
                raise ValueError("FA online phase boundary is absent from its export")
        first = addresses.index(region["wait1_pc"])
        wait = addresses.index(region["wait0_pc"])
        if not first < wait or not parsed[first][1].startswith("WARPGROUP.DEPBAR") or not parsed[wait][1].startswith("WARPGROUP.DEPBAR"):
            raise ValueError("FA online phase wait boundaries differ")
        ends = [i for i in range(wait + 1, len(parsed)) if parsed[i][1].startswith("HGMMA.")]
        if not ends:
            raise ValueError("FA online phase lacks its next matrix boundary")
        end = ends[0]
        counts = []
        for a, b in ((first + 1, wait), (wait + 1, end)):
            work = {op: 0 for op in expected}
            for _, opcode in parsed[a:b]:
                family = "EX2" if opcode == "MUFU.EX2" else opcode.split(".")[0]
                if family in work:
                    work[family] += 1
            counts.append(work)
        if any(counts[0][op] + counts[1][op] != total for op, total in expected.items()):
            raise ValueError("FA online phase counts disagree with the source register geometry")
        first_exp = next(pc for pc, op in parsed[first + 1:end] if op == "MUFU.EX2")
        if first_exp != region["first_exp_pc"]:
            raise ValueError("FA online phase first exponent differs")
        result.append({**region, "online_before_wait0": counts[0], "online_totals": expected.copy(),
                       "next_mma_pc": parsed[end][0]})
    return result


def extract_fa_instruction_counters(text: str, case: dict) -> list[dict]:
    """Bind raw non-duration FA counters to the selected source launches.

    This is raw diagnostic evidence. It supplies no Forward price, cache
    policy, occupancy correction, canonical timing or publication decision.
    """
    rows = list(csv.reader(text.splitlines()))
    identity = ("ID", "Kernel Name", "Grid Size", "Block Size")
    headers = [i for i, row in enumerate(rows)
               if all(field in row for field in (*identity, *FA_INSTRUCTION_COUNTER_METRICS))]
    if len(headers) != 1:
        raise ValueError("FA instruction counter table lacks requested fields")
    header = rows[headers[0]]
    if len(header) != len(set(header)):
        raise ValueError("Duplicate FA instruction counter column")
    observations = [dict(zip(header, row)) for row in rows[headers[0] + 1:]
                    if len(row) == len(header) and row[header.index("ID")].isdigit()]
    expected = expected_launch_signature(case)
    if len(observations) != len(expected) or not observations:
        raise ValueError("FA instruction counter launch population differs")
    range_fields = [name for name in header if "Range" in name]
    if not range_fields:
        raise ValueError("FA instruction counters lack NVTX range identity")
    result = []
    for observed, prior in zip(observations, expected):
        dimensions = []
        for field in ("Grid Size", "Block Size"):
            if re.fullmatch(r"\(\s*[1-9]\d*\s*,\s*[1-9]\d*\s*,\s*[1-9]\d*\s*\)", observed[field]) is None:
                raise ValueError("Invalid FA instruction counter Dim3")
            dimensions.append([int(part.strip()) for part in observed[field].strip("()").split(",")])
        if (normalized_profiler_kernel_name(observed["Kernel Name"])
                != normalized_profiler_kernel_name(prior[0])
                or dimensions != [prior[1], prior[2]]):
            raise ValueError("FA instruction counter source launch differs")
        if not any(case["plan"]["marker"] in observed[field] for field in range_fields):
            raise ValueError("FA instruction counter NVTX range differs")
        values = {name: float(observed[name]) for name in FA_INSTRUCTION_COUNTER_METRICS}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("FA instruction counters must be finite and nonnegative")
        if any(value > 100 for name, value in values.items() if name.endswith(".pct") or ".pct_of_" in name):
            raise ValueError("FA instruction counter percentage exceeds100")
        if values["smsp__inst_executed.sum"] <= 0:
            raise ValueError("FA instruction counter contains no executed work")
        requests = values["sm__icc_requests.sum"]
        if (values["sm__icc_requests_lookup_hit.sum"] > requests
                or values["sm__icc_requests_lookup_miss.sum"] > requests):
            raise ValueError("FA instruction cache lookup count exceeds requests")
        result.append({"kernel_name": observed["Kernel Name"],
                       "grid": dimensions[0], "block": dimensions[1],
                       "non_duration_counters": values})
    return result


def _verify_pin(root: Path, pin: dict) -> Path:
    relative = Path(pin["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("missing15 evidence pin path escapes repository")
    path = root / relative
    if (not path.is_file() or path.stat().st_size != pin["size"]
            or _sha256(path) != pin["sha256"]):
        raise ValueError(f"missing15 evidence pin differs: {path}")
    return path


def preflight_cluster_facts(sqlite_path: Path, launch_path: Path) -> list[dict]:
    """Read same-window NSYS cluster dimensions for marker-filtered launches."""

    launches = json.loads(launch_path.read_text())["launches"]
    connection = sqlite3.connect(str(sqlite_path))
    try:
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        required = {"start", "deviceId", "gridX", "gridY", "gridZ",
                    "blockX", "blockY", "blockZ", "dynamicSharedMemory",
                    "clusterX", "clusterY", "clusterZ"}
        if not required <= columns:
            raise ValueError("NSYS SQLite lacks kernel cluster columns")
        facts = []
        for launch in launches:
            rows = connection.execute(
                "SELECT gridX, gridY, gridZ, blockX, blockY, blockZ, "
                "dynamicSharedMemory, clusterX, clusterY, clusterZ "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start=? AND deviceId=?",
                (launch["start_ns"], launch["device"])).fetchall()
            if len(rows) != 1:
                raise ValueError("NSYS launch has no unique SQLite cluster row")
            row = rows[0]
            if (list(row[:3]) != launch["grid"]
                    or list(row[3:6]) != launch["block"]
                    or row[6] != launch["dynamic_shared_memory_bytes"]
                    or any(type(value) is not int or value < 0
                           for value in row[7:])):
                raise ValueError("NSYS launch and SQLite cluster row differ")
            cluster = list(row[7:])
            if any(cluster) and any(value == 0 for value in cluster):
                raise ValueError("NSYS cluster dimensions are partially zero")
            facts.append({"name": launch["name"], "grid": launch["grid"],
                          "block": launch["block"],
                          "dynamic_shared_memory_bytes": row[6],
                          "cluster": cluster})
        return facts
    finally:
        connection.close()


def _missing15_source(campaign: Path, *, verify_external: bool) -> dict:
    plans = campaign / "plans"
    source = json.loads((plans / "candidates" / MISSING15_SOURCE_NAME).read_text())
    manifest_path = plans / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (source["schema_version"] != 1
            or source["kind"] != "h200_missing15_ncu_review_only"
            or source["status"] != "candidate_not_admitted"
            or source["publication_status"]
            != "raw_only_until_nsys_quality_and_dataset_owner_approve"
            or source["gpu"] != "H200" or source["gpu_count"] != 1
            or source["source_campaign_manifest_pin"]["sha256"]
            != _sha256(manifest_path)
            or manifest["hardware_id"] != "nvidia_h200"
            or manifest["a800_case_count"] != 1027):
        raise ValueError("missing15 source/campaign identity differs")
    batches = source["batches"]
    if len(batches) != 4 or sorted(len(batch["cases"]) for batch in batches) != [2, 4, 4, 5]:
        raise ValueError("missing15 source batch coverage differs")
    all_ids = [case["case_id"] for batch in batches for case in batch["cases"]]
    original_missing = {case_id for ids in manifest["missing_nsys_case_ids"].values()
                        for case_id in ids}
    if len(all_ids) != 15 or len(set(all_ids)) != 15 or set(all_ids) != original_missing:
        raise ValueError("missing15 source case set differs from campaign")
    if verify_external:
        root = campaign.parents[3]
        _verify_pin(root, source["source_campaign_manifest_pin"])
        _verify_pin(root, source["nsys_capture_review_pin"])
        for batch in batches:
            for case in batch["cases"]:
                plan_path = _verify_pin(root, case["runner_plan_pin"])
                old_launch_path = _verify_pin(root, case["nsys_launch_pin"])
                plans_document = json.loads(plan_path.read_text())
                matches = [item for item in plans_document["plans"]
                           if item["case_id"] == case["case_id"]]
                if len(matches) != 1 or matches[0] != case["runner_plan"]:
                    raise ValueError("missing15 pinned runner plan differs")
                old = json.loads(old_launch_path.read_text())
                old_signature = [{"name": item["name"], "grid": item["grid"],
                                  "block": item["block"],
                                  "dynamic_shared_memory_bytes":
                                  item["dynamic_shared_memory_bytes"],
                                  "cluster": expected["cluster"]}
                                 for item, expected in zip(
                                     old["launches"], case["expected_launch_signature"])]
                if (old["case_id"] != case["case_id"]
                        or old["marker"] != case["runner_plan"]["marker"]
                        or len(old["launches"]) != len(case["expected_launch_signature"])
                        or old_signature != case["expected_launch_signature"]):
                    raise ValueError("missing15 pinned old NSYS launch differs")
                for history in case["nsys_review_history"]:
                    _verify_pin(root, history["review"])
                if case["old_cluster_evidence_pin"] is not None:
                    sqlite_path = _verify_pin(root, case["old_cluster_evidence_pin"])
                    cluster_launch_path = sqlite_path.with_suffix(".launch.json")
                    if not cluster_launch_path.is_file():
                        raise FileNotFoundError(cluster_launch_path)
                    old_cluster = preflight_cluster_facts(
                        sqlite_path, cluster_launch_path)
                    if old_cluster != case["expected_launch_signature"]:
                        raise ValueError("missing15 old NSYS cluster evidence differs")
    return source


def prepare_missing15_batch_plans(campaign: Path) -> list[Path]:
    """Materialize raw-only plans after verifying every pinned old source."""

    source = _missing15_source(campaign, verify_external=True)
    source_path = campaign / "plans" / "candidates" / MISSING15_SOURCE_NAME
    source_hash = _sha256(source_path)
    metrics = json.loads((campaign / "plans" / "manifest.json").read_text())["metrics"]
    candidate_dir = source_path.parent
    names = [batch["batch_name"] for batch in source["batches"]]
    if len(set(names)) != len(names):
        raise ValueError("missing15 batch names are duplicated")
    for name in names:
        if ((candidate_dir / f"{name}.json").exists()
                or (candidate_dir / f"{name}.manifest.json").exists()):
            raise FileExistsError(f"missing15 plan already prepared: {name}")
    outputs = []
    for batch in source["batches"]:
        name = batch["batch_name"]
        plan_path = candidate_dir / f"{name}.json"
        manifest_path = candidate_dir / f"{name}.manifest.json"
        plan = {"schema_version": 1, "batch_name": name,
                "traffic_metrics": metrics["traffic_metrics"],
                "service_metrics": metrics["service_metrics"],
                "cases": [{"family": case["family"],
                           "case_id": case["case_id"],
                           "plan": case["runner_plan"],
                           "reference_kind": MISSING15_REFERENCE_KIND,
                           "reference_launch_signature":
                           case["expected_launch_signature"],
                           "source_candidate_sha256": source_hash}
                          for case in batch["cases"]]}
        plan_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        admission = {"schema_version": 1, "kind": MISSING15_BATCH_KIND,
                     "status": "admitted_for_raw_only_capture",
                     "batch_name": name, "case_ids": [case["case_id"]
                                                       for case in batch["cases"]],
                     "source_candidate_sha256": source_hash,
                     "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                     "publication_status": source["publication_status"]}
        with plan_path.open("xb") as stream:
            stream.write(plan_bytes)
        with manifest_path.open("x") as stream:
            json.dump(admission, stream, indent=2, sort_keys=True)
            stream.write("\n")
        outputs.append(plan_path)
    return outputs


def verify_missing15_local_evidence(campaign: Path, batch_name: str) -> None:
    source = _missing15_source(campaign, verify_external=True)
    if batch_name not in {batch["batch_name"] for batch in source["batches"]}:
        raise ValueError("unknown missing15 batch")
    if shutil.which("zstd") is None:
        raise FileNotFoundError("zstd is required before missing15 paid capture")


def validate_missing15_local_case(case: dict, target: Path,
                                  metrics: list[str]) -> None:
    """Recheck a remote complete result before counting it as raw candidate."""

    if case["reference_kind"] != MISSING15_REFERENCE_KIND:
        raise ValueError("not a missing15 raw-only case")
    case_id = case["case_id"]
    reference = case["reference_launch_signature"]
    validation = json.loads((target / "validation.json").read_text())
    if (validation["status"] != "complete"
            or validation["same_run_cluster_verified"] is not True
            or validation["kernel_count"] != len(reference)):
        raise ValueError("missing15 remote validation is incomplete")
    preflight = target / "preflight" / "cases" / case_id
    launch = json.loads((preflight / "window_01.launch.json").read_text())
    cluster = json.loads((target / "cluster_preflight.json").read_text())
    report_path = preflight / "window_01.nsys-rep"
    compressed_sqlite = preflight / "window_01.sqlite.zst"
    if (launch["case_id"] != case_id or launch["telemetry"]["clock_accepted"] is not True
            or cluster["schema_version"] != 1
            or cluster["kind"] != "same_run_nsys_cluster_facts"
            or cluster["case_id"] != case_id
            or cluster["nsys_report_sha256"] != _sha256(report_path)
            or cluster["sqlite_zst_sha256"] != _sha256(compressed_sqlite)
            or cluster["sqlite_zst_size"] != compressed_sqlite.stat().st_size
            or len(launch["launches"]) != len(reference)
            or len(cluster["launches"]) != len(reference)):
        raise ValueError("missing15 NSYS preflight evidence differs")
    zstd = shutil.which("zstd")
    if zstd is None:
        raise FileNotFoundError("zstd is required for missing15 review")
    with tempfile.TemporaryDirectory(prefix="h200-ncu-cluster-") as temporary:
        sqlite_path = Path(temporary) / "window_01.sqlite"
        with sqlite_path.open("wb") as dest:
            subprocess.run([zstd, "-d", "-q", "-c", str(compressed_sqlite)],
                           stdout=dest, stderr=subprocess.PIPE, check=True,
                           timeout=90)
        if preflight_cluster_facts(sqlite_path, preflight / "window_01.launch.json") \
                != cluster["launches"]:
            raise ValueError("missing15 cluster facts disagree with same-run SQLite")
    for actual, fact, prior in zip(launch["launches"], cluster["launches"],
                                   reference):
        for field in ("name", "grid", "block", "dynamic_shared_memory_bytes"):
            if actual[field] != prior[field] or fact[field] != prior[field]:
                raise ValueError("missing15 old/new NSYS launch differs")
        if prior["cluster"] is not None and fact["cluster"] != prior["cluster"]:
            raise ValueError("missing15 known old cluster differs")
    telemetry = json.loads((target / "ncu_telemetry.json").read_text())
    if telemetry["clock_accepted"] is not True:
        raise ValueError("missing15 NCU clock is rejected")
    with (target / "ncu_stdout.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    header_positions = [index for index, row in enumerate(rows)
                        if "Kernel Name" in row and all(metric in row for metric in metrics)]
    if len(header_positions) != 1:
        raise ValueError("missing15 NCU metric header is absent or duplicated")
    header_index = header_positions[0]
    header = rows[header_index]
    observations = [dict(zip(header, row)) for row in rows[header_index + 2:]
                    if len(row) == len(header)]
    if len(observations) != len(reference):
        raise ValueError("missing15 NCU kernel count differs")
    for observation, prior, fact in zip(observations, reference,
                                        cluster["launches"]):
        grid = [int(part.strip()) for part in observation["Grid Size"].strip("()").split(",")]
        block = [int(part.strip()) for part in observation["Block Size"].strip("()").split(",")]
        ncu_cluster = [int(observation[f"launch__cluster_dim_{axis}"])
                       for axis in "xyz"]
        dynamic_shared = int(observation["launch__shared_mem_per_block_dynamic"])
        if (grid != prior["grid"] or block != prior["block"]
                or normalized_profiler_kernel_name(observation["Kernel Name"])
                != normalized_profiler_kernel_name(prior["name"])
                or ncu_cluster != fact["cluster"]
                or dynamic_shared != prior["dynamic_shared_memory_bytes"]
                or not any(case["plan"]["marker"] in observation[field]
                           for field in header if "Range" in field)):
            raise ValueError("missing15 NCU launch, cluster or NVTX differs")
        values = [float(observation[metric]) for metric in metrics]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("missing15 NCU metric is invalid")
        active = float(observation["sm__cycles_active.avg"])
        elapsed = float(observation["sm__cycles_elapsed.avg"])
        if elapsed <= 0 or active > elapsed:
            raise ValueError("missing15 NCU cycles are invalid")


def _quality_retry_source(campaign: Path, *, verify_reviews: bool) -> dict:
    source_path = campaign / "plans" / "candidates" / QUALITY_RETRY_SOURCE_NAME
    source = json.loads(source_path.read_text())
    if (source["schema_version"] != 1
            or source["kind"] != "h200_ncu_quality_retry_source"
            or source["status"] != "candidate_not_executed"
            or source["gpu"] != "H200" or source["gpu_count"] != 1
            or source["campaign_manifest_sha256"]
            != _sha256(campaign / "plans" / "manifest.json")):
        raise ValueError("NCU quality retry source identity differs")
    entries = source["cases"]
    ids = [entry["case"]["case_id"] for entry in entries]
    if len(ids) != 19 or len(set(ids)) != 19:
        raise ValueError("NCU quality retry source must contain 19 unique cases")
    for entry in entries:
        plan_pin = entry["source_plan"]
        plan_path = _verify_pin(campaign, plan_pin)
        plan = json.loads(plan_path.read_text())
        matching = [case for case in plan["cases"]
                    if case["case_id"] == entry["case"]["case_id"]]
        if len(matching) != 1 or matching[0] != entry["case"]:
            raise ValueError("NCU quality retry source case differs from parent plan")
        if verify_reviews:
            review_path = _verify_pin(campaign, entry["source_review"])
            review = json.loads(review_path.read_text())
            case_id = entry["case"]["case_id"]
            if (review["plan_sha256"] != plan_pin["sha256"]
                    or review["statuses"][case_id]["status"]
                    != entry["failed_status"]
                    or entry["failed_status"] in ("complete", "not_attempted")):
                raise ValueError("NCU quality retry parent is not a failed attempt")
    return source


def prepare_quality_retry_batch_plans(campaign: Path) -> list[Path]:
    """Prepare four bounded raw retries from the reviewed 1027-case first pass."""

    plans = campaign / "plans"
    source_path = plans / "candidates" / QUALITY_RETRY_SOURCE_NAME
    if source_path.exists():
        raise FileExistsError(source_path)
    manifest_path = plans / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    plan_paths = [path for path in plans.rglob("*.json")
                  if "cases" in json.loads(path.read_text())]
    by_hash = {_sha256(path): path for path in plan_paths}
    case_history: dict[str, list[tuple[Path, Path, dict, str]]] = {}
    for review_path in sorted(campaign.glob("*_review.json")):
        review = json.loads(review_path.read_text())
        if review["plan_sha256"] not in by_hash:
            raise ValueError(f"NCU review has no pinned plan: {review_path}")
        plan_path = by_hash[review["plan_sha256"]]
        plan = json.loads(plan_path.read_text())
        cases = {case["case_id"]: case for case in plan["cases"]}
        for case_id, result in review["statuses"].items():
            if case_id not in cases:
                raise ValueError("NCU review contains a case absent from its plan")
            case_history.setdefault(case_id, []).append(
                (plan_path, review_path, cases[case_id], result["status"]))
    original = {case["case_id"] for path in plans.glob("*.json")
                if path.name != "manifest.json"
                for case in json.loads(path.read_text())["cases"]}
    missing = {case_id for ids in manifest["missing_nsys_case_ids"].values()
               for case_id in ids}
    if (len(original) != 1011 or len(missing) != 15
            or original & missing or set(case_history) != original | missing):
        raise ValueError("NCU quality retry source lacks complete 1027 coverage")
    entries = []
    for case_id, history in sorted(case_history.items()):
        if any(status == "complete" for _, _, _, status in history):
            continue
        failed = [(plan, review, case, status)
                  for plan, review, case, status in history
                  if status != "not_attempted"]
        if not failed:
            raise ValueError(f"NCU case has no attempted result: {case_id}")
        plan_path, review_path, case, status = failed[-1]
        entries.append({"case": case, "failed_status": status,
                        "source_plan": {"path": str(plan_path.relative_to(campaign)),
                                        "size": plan_path.stat().st_size,
                                        "sha256": _sha256(plan_path)},
                        "source_review": {"path": str(review_path.relative_to(campaign)),
                                          "size": review_path.stat().st_size,
                                          "sha256": _sha256(review_path)}})
    if len(entries) != 19:
        raise ValueError(f"expected 19 quality failures, found {len(entries)}")
    source = {"schema_version": 1, "kind": "h200_ncu_quality_retry_source",
              "status": "candidate_not_executed", "gpu": "H200", "gpu_count": 1,
              "campaign_manifest_sha256": _sha256(manifest_path),
              "cases": entries}
    candidate_dir = plans / "candidates"
    names = [f"ncu_quality_retry_{family}_20261102" for family in
             ("gemm", "fused_moe", "flash_attn", "vector")]
    for name in names:
        if ((candidate_dir / f"{name}.json").exists()
                or (candidate_dir / f"{name}.manifest.json").exists()):
            raise FileExistsError(f"NCU quality retry plan already exists: {name}")
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    _quality_retry_source(campaign, verify_reviews=True)
    outputs = []
    for family, name in zip(("gemm", "fused_moe", "flash_attn", "vector"), names):
        cases = [entry["case"] for entry in entries
                 if entry["case"]["family"] == family]
        if not cases or len(cases) > 6:
            raise ValueError(f"NCU quality retry family size is invalid: {family}")
        plan = {"schema_version": 1, "batch_name": name,
                "traffic_metrics": manifest["metrics"]["traffic_metrics"],
                "service_metrics": manifest["metrics"]["service_metrics"],
                "cases": cases}
        plan_path = candidate_dir / f"{name}.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        admission = {"schema_version": 1, "kind": QUALITY_RETRY_BATCH_KIND,
                     "batch_name": name, "case_ids": [case["case_id"] for case in cases],
                     "source_sha256": _sha256(source_path),
                     "plan_sha256": _sha256(plan_path)}
        (candidate_dir / f"{name}.manifest.json").write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n")
        outputs.append(plan_path)
    return outputs


def verify_quality_retry_local_evidence(campaign: Path, batch_name: str) -> None:
    source = _quality_retry_source(campaign, verify_reviews=True)
    if batch_name not in {f"ncu_quality_retry_{family}_20261102" for family in
                          ("gemm", "fused_moe", "flash_attn", "vector")}:
        raise ValueError("unknown NCU quality retry batch")
    family = batch_name.removeprefix("ncu_quality_retry_").removesuffix("_20261102")
    for entry in source["cases"]:
        if entry["case"]["family"] != family:
            continue
        case_id = entry["case"]["case_id"]
        for review_path in campaign.glob("*_review.json"):
            review = json.loads(review_path.read_text())
            if (case_id in review["statuses"]
                    and review["statuses"][case_id]["status"] == "complete"):
                raise ValueError(f"NCU retry case already accepted: {case_id}")


def _clock_settle_retry_inputs(campaign: Path) -> tuple[dict, dict, dict]:
    """Bind the one FA case whose preflight passed but NCU start clock fell."""

    source = _quality_retry_source(campaign, verify_reviews=True)
    matches = [entry for entry in source["cases"]
               if entry["case"]["case_id"] == CLOCK_SETTLE_RETRY_CASE]
    if len(matches) != 1:
        raise ValueError("clock settle retry source case is absent or duplicated")
    previous_path = campaign / "ncu_quality_retry_flash_attn_20261102_review.json"
    previous = json.loads(previous_path.read_text())
    previous_plan = campaign / "plans" / "candidates" / "ncu_quality_retry_flash_attn_20261102.json"
    if (previous["plan_sha256"] != _sha256(previous_plan)
            or previous["statuses"][CLOCK_SETTLE_RETRY_CASE]["status"]
            != "ncu_clock_rejected"):
        raise ValueError("clock settle retry lacks the failed NCU parent")
    for review_path in campaign.glob("*_review.json"):
        review = json.loads(review_path.read_text())
        if (CLOCK_SETTLE_RETRY_CASE in review["statuses"]
                and review["statuses"][CLOCK_SETTLE_RETRY_CASE]["status"]
                == "complete"):
            raise ValueError("clock settle retry case is already accepted")
    manifest = json.loads((campaign / "plans" / "manifest.json").read_text())
    plan = {"schema_version": 1, "batch_name": CLOCK_SETTLE_RETRY_BATCH,
            "traffic_metrics": manifest["metrics"]["traffic_metrics"],
            "service_metrics": manifest["metrics"]["service_metrics"],
            "cases": [matches[0]["case"]]}
    admission = {"schema_version": 1,
                 "kind": "h200_ncu_clock_settle_retry_candidate",
                 "batch_name": CLOCK_SETTLE_RETRY_BATCH,
                 "case_id": CLOCK_SETTLE_RETRY_CASE,
                 "source_sha256": _sha256(
                     campaign / "plans" / "candidates" / QUALITY_RETRY_SOURCE_NAME),
                 "previous_review_sha256": _sha256(previous_path)}
    return plan, admission, previous


def prepare_clock_settle_retry_plan(campaign: Path) -> Path:
    plan, admission, _ = _clock_settle_retry_inputs(campaign)
    directory = campaign / "plans" / "candidates"
    plan_path = directory / f"{CLOCK_SETTLE_RETRY_BATCH}.json"
    admission_path = directory / f"{CLOCK_SETTLE_RETRY_BATCH}.manifest.json"
    if plan_path.exists() or admission_path.exists():
        raise FileExistsError("clock settle retry plan already exists")
    with plan_path.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    admission["plan_sha256"] = _sha256(plan_path)
    with admission_path.open("x") as stream:
        json.dump(admission, stream, indent=2, sort_keys=True)
        stream.write("\n")
    admitted_batch_plan(campaign, CLOCK_SETTLE_RETRY_BATCH)
    return plan_path


def _directed7_inputs(campaign: Path) -> tuple[list[str], dict[str, dict], Path]:
    """Bind the user's two new attempts to the seven still missing cases."""

    diagnostic_path = campaign.parents[3] / DIRECTED7_DIAGNOSTIC
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic["kind"] != "h200_ncu_remaining8_clock_reason_diagnostic":
        raise ValueError("directed retry diagnostic identity differs")
    rows = diagnostic["rows"]
    ids = [row["case_id"] for row in rows
           if row["case_id"] != CLOCK_SETTLE_RETRY_CASE]
    if (len(rows) != 8 or len(ids) != 7 or len(set(ids)) != 7
            or sum(row["case_id"] == CLOCK_SETTLE_RETRY_CASE for row in rows) != 1):
        raise ValueError("directed retry must cover exactly seven missing cases")
    source = _quality_retry_source(campaign, verify_reviews=True)
    by_id = {entry["case"]["case_id"]: entry["case"]
             for entry in source["cases"]}
    if not set(ids) <= set(by_id):
        raise ValueError("directed retry case lacks pinned original workload")
    return ids, by_id, diagnostic_path


def prepare_directed7_retry_plans(campaign: Path) -> list[Path]:
    """Prepare two distinct single-case attempts, even if the first passes."""

    ids, by_id, diagnostic_path = _directed7_inputs(campaign)
    directory = campaign / "plans" / "candidates"
    source_path = directory / QUALITY_RETRY_SOURCE_NAME
    metrics = json.loads((campaign / "plans" / "manifest.json").read_text())["metrics"]
    names = [f"{DIRECTED7_PREFIX}r{round_index}_{index:02d}_20261102"
             for round_index in (1, 2) for index in range(1, 8)]
    for name in names:
        if (directory / f"{name}.json").exists() or (
                directory / f"{name}.manifest.json").exists():
            raise FileExistsError(f"directed retry plan already exists: {name}")
    outputs = []
    for round_index in (1, 2):
        for index, case_id in enumerate(ids, start=1):
            name = f"{DIRECTED7_PREFIX}r{round_index}_{index:02d}_20261102"
            plan = {"schema_version": 1, "batch_name": name,
                    "traffic_metrics": metrics["traffic_metrics"],
                    "service_metrics": metrics["service_metrics"],
                    "cases": [by_id[case_id]]}
            plan_path = directory / f"{name}.json"
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            admission = {"schema_version": 1,
                         "kind": "h200_ncu_directed7_two_attempt_candidate",
                         "batch_name": name, "round": round_index,
                         "case_ordinal": index, "case_id": case_id,
                         "source_sha256": _sha256(source_path),
                         "diagnostic_sha256": _sha256(diagnostic_path),
                         "plan_sha256": _sha256(plan_path)}
            (directory / f"{name}.manifest.json").write_text(
                json.dumps(admission, indent=2, sort_keys=True) + "\n")
            outputs.append(plan_path)
    return outputs


def prepare_directed7_unstarted_replacement(campaign: Path) -> Path:
    """Replace only the second GEMM app that stopped before case execution."""

    ids, by_id, diagnostic_path = _directed7_inputs(campaign)
    failed_name = "ncu_directed7_r2_07_20261102"
    failure_path = campaign / "batches" / failed_name / "failed_transfer_receipt.json"
    failure = json.loads(failure_path.read_text())
    if (failure["batch_name"] != failed_name
            or failure["case_ids"] != [ids[6]]
            or failure["failure_reason"] != "modal_process_nonzero"
            or failure["partial_raw_files"]
            or failure["modal_returncode"] == 0):
        raise ValueError("directed replacement parent reached the case")
    directory = campaign / "plans" / "candidates"
    name = DIRECTED7_UNSTARTED_REPLACEMENT
    plan_path = directory / f"{name}.json"
    admission_path = directory / f"{name}.manifest.json"
    if plan_path.exists() or admission_path.exists():
        raise FileExistsError("directed replacement already prepared")
    metrics = json.loads((campaign / "plans" / "manifest.json").read_text())["metrics"]
    plan = {"schema_version": 1, "batch_name": name,
            "traffic_metrics": metrics["traffic_metrics"],
            "service_metrics": metrics["service_metrics"],
            "cases": [by_id[ids[6]]]}
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    admission = {
        "schema_version": 1,
        "kind": "h200_ncu_directed7_unstarted_replacement_candidate",
        "batch_name": name, "case_id": ids[6],
        "failed_receipt_sha256": _sha256(failure_path),
        "diagnostic_sha256": _sha256(diagnostic_path),
        "source_sha256": _sha256(directory / QUALITY_RETRY_SOURCE_NAME),
        "plan_sha256": _sha256(plan_path),
    }
    admission_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    admitted_batch_plan(campaign, name)
    return plan_path


def admitted_batch_plan(campaign: Path, batch_name: str) -> Path:
    plans = campaign / "plans"
    campaign_manifest_path = plans / "manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text())
    if campaign_manifest["status"] != "prepared_not_executed":
        raise ValueError("NCU campaign is not prepared")

    if batch_name == DIRECTED7_UNSTARTED_REPLACEMENT:
        ids, by_id, diagnostic_path = _directed7_inputs(campaign)
        failed_name = "ncu_directed7_r2_07_20261102"
        failure_path = campaign / "batches" / failed_name / "failed_transfer_receipt.json"
        failure = json.loads(failure_path.read_text())
        if (failure["batch_name"] != failed_name
                or failure["case_ids"] != [ids[6]]
                or failure["failure_reason"] != "modal_process_nonzero"
                or failure["partial_raw_files"]
                or failure["modal_returncode"] == 0):
            raise ValueError("directed replacement parent reached the case")
        directory = plans / "candidates"
        plan_path = directory / f"{batch_name}.json"
        admission_path = directory / f"{batch_name}.manifest.json"
        metrics = campaign_manifest["metrics"]
        plan = {"schema_version": 1, "batch_name": batch_name,
                "traffic_metrics": metrics["traffic_metrics"],
                "service_metrics": metrics["service_metrics"],
                "cases": [by_id[ids[6]]]}
        admission = {
            "schema_version": 1,
            "kind": "h200_ncu_directed7_unstarted_replacement_candidate",
            "batch_name": batch_name, "case_id": ids[6],
            "failed_receipt_sha256": _sha256(failure_path),
            "diagnostic_sha256": _sha256(diagnostic_path),
            "source_sha256": _sha256(directory / QUALITY_RETRY_SOURCE_NAME),
            "plan_sha256": _sha256(plan_path),
        }
        if (json.loads(plan_path.read_text()) != plan
                or json.loads(admission_path.read_text()) != admission):
            raise ValueError("directed replacement admission or workload differs")
        return plan_path

    if batch_name.startswith(DIRECTED7_PREFIX):
        match = re.fullmatch(r"ncu_directed7_r([12])_(0[1-7])_20261102",
                             batch_name)
        if match is None:
            raise ValueError("invalid directed retry batch name")
        ids, by_id, diagnostic_path = _directed7_inputs(campaign)
        round_index, index = int(match[1]), int(match[2])
        case_id = ids[index - 1]
        directory = plans / "candidates"
        plan_path = directory / f"{batch_name}.json"
        admission_path = directory / f"{batch_name}.manifest.json"
        plan = json.loads(plan_path.read_text())
        admission = json.loads(admission_path.read_text())
        expected_plan = {"schema_version": 1, "batch_name": batch_name,
                         "traffic_metrics": campaign_manifest["metrics"]["traffic_metrics"],
                         "service_metrics": campaign_manifest["metrics"]["service_metrics"],
                         "cases": [by_id[case_id]]}
        if (plan != expected_plan or admission != {
                "schema_version": 1,
                "kind": "h200_ncu_directed7_two_attempt_candidate",
                "batch_name": batch_name, "round": round_index,
                "case_ordinal": index, "case_id": case_id,
                "source_sha256": _sha256(directory / QUALITY_RETRY_SOURCE_NAME),
                "diagnostic_sha256": _sha256(diagnostic_path),
                "plan_sha256": _sha256(plan_path)}):
            raise ValueError("directed retry admission or workload differs")
        return plan_path
    if batch_name in campaign_manifest["batches"]:
        plan_path = plans / f"{batch_name}.json"
        plan = json.loads(plan_path.read_text())
        if (plan["batch_name"] != batch_name
                or [case["case_id"] for case in plan["cases"]]
                != campaign_manifest["batches"][batch_name]):
            raise ValueError(f"NCU batch {batch_name} differs from manifest")
        return plan_path

    if batch_name == CLOCK_SETTLE_RETRY_BATCH:
        plan, admission, _ = _clock_settle_retry_inputs(campaign)
        directory = plans / "candidates"
        plan_path = directory / f"{batch_name}.json"
        admission_path = directory / f"{batch_name}.manifest.json"
        if (json.loads(plan_path.read_text()) != plan
                or json.loads(admission_path.read_text())
                != {**admission, "plan_sha256": _sha256(plan_path)}):
            raise ValueError("clock settle retry admission or plan differs")
        return plan_path

    if batch_name.startswith("ncu_missing15_"):
        source = _missing15_source(campaign, verify_external=False)
        matching = [batch for batch in source["batches"]
                    if batch["batch_name"] == batch_name]
        if len(matching) != 1:
            raise ValueError("missing15 batch name is absent or duplicated")
        candidate_dir = plans / "candidates"
        source_path = candidate_dir / MISSING15_SOURCE_NAME
        plan_path = candidate_dir / f"{batch_name}.json"
        admission_path = candidate_dir / f"{batch_name}.manifest.json"
        if not plan_path.is_file() or not admission_path.is_file():
            raise FileNotFoundError("missing15 batch lacks raw-only admission")
        admission = json.loads(admission_path.read_text())
        plan = json.loads(plan_path.read_text())
        batch = matching[0]
        source_hash = _sha256(source_path)
        expected_cases = [{"family": case["family"],
                           "case_id": case["case_id"],
                           "plan": case["runner_plan"],
                           "reference_kind": MISSING15_REFERENCE_KIND,
                           "reference_launch_signature":
                           case["expected_launch_signature"],
                           "source_candidate_sha256": source_hash}
                          for case in batch["cases"]]
        if (admission["schema_version"] != 1
                or admission["kind"] != MISSING15_BATCH_KIND
                or admission["status"] != "admitted_for_raw_only_capture"
                or admission["batch_name"] != batch_name
                or admission["case_ids"] != [case["case_id"] for case in expected_cases]
                or admission["source_candidate_sha256"] != source_hash
                or admission["plan_sha256"] != _sha256(plan_path)
                or admission["publication_status"] != source["publication_status"]
                or plan != {"schema_version": 1, "batch_name": batch_name,
                            "traffic_metrics": campaign_manifest["metrics"]["traffic_metrics"],
                            "service_metrics": campaign_manifest["metrics"]["service_metrics"],
                            "cases": expected_cases}):
            raise ValueError("missing15 batch admission or plan differs")
        return plan_path

    if batch_name.startswith("ncu_quality_retry_"):
        source = _quality_retry_source(campaign, verify_reviews=False)
        candidate_dir = plans / "candidates"
        plan_path = candidate_dir / f"{batch_name}.json"
        admission_path = candidate_dir / f"{batch_name}.manifest.json"
        if not plan_path.is_file() or not admission_path.is_file():
            raise FileNotFoundError("NCU quality retry lacks admission")
        family = batch_name.removeprefix("ncu_quality_retry_").removesuffix("_20261102")
        if batch_name != f"ncu_quality_retry_{family}_20261102" or family not in (
                "gemm", "fused_moe", "flash_attn", "vector"):
            raise ValueError("unknown NCU quality retry batch")
        cases = [entry["case"] for entry in source["cases"]
                 if entry["case"]["family"] == family]
        plan = json.loads(plan_path.read_text())
        admission = json.loads(admission_path.read_text())
        if (admission["schema_version"] != 1
                or admission["kind"] != QUALITY_RETRY_BATCH_KIND
                or admission["batch_name"] != batch_name
                or admission["case_ids"] != [case["case_id"] for case in cases]
                or admission["source_sha256"]
                != _sha256(candidate_dir / QUALITY_RETRY_SOURCE_NAME)
                or admission["plan_sha256"] != _sha256(plan_path)
                or plan != {"schema_version": 1, "batch_name": batch_name,
                            "traffic_metrics": campaign_manifest["metrics"]["traffic_metrics"],
                            "service_metrics": campaign_manifest["metrics"]["service_metrics"],
                            "cases": cases}):
            raise ValueError("NCU quality retry admission or plan differs")
        return plan_path

    candidate_dir = plans / "candidates"
    plan_path = candidate_dir / f"{batch_name}.json"
    candidate_manifest_path = candidate_dir / f"{batch_name}.manifest.json"
    if not plan_path.is_file() or not candidate_manifest_path.is_file():
        raise ValueError(f"unprepared NCU batch: {batch_name}")
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    if (candidate_manifest["schema_version"] != 1
            or candidate_manifest["kind"] != "h200_ncu_partial_tail_candidate"
            or candidate_manifest["batch_name"] != batch_name
            or candidate_manifest["campaign_manifest_sha256"]
            != _sha256(campaign_manifest_path)
            or candidate_manifest["plan_sha256"] != _sha256(plan_path)):
        raise ValueError(f"NCU candidate manifest differs: {batch_name}")
    parent_name = candidate_manifest["parent_batch_name"]
    if parent_name not in campaign_manifest["batches"]:
        raise ValueError(f"NCU candidate parent is not prepared: {parent_name}")
    parent_plan_path = plans / f"{parent_name}.json"
    parent_review_path = campaign / f"{parent_name}_review.json"
    parent_status_path = campaign / "batches" / parent_name / "batch_status.json"
    if (candidate_manifest["parent_plan_sha256"] != _sha256(parent_plan_path)
            or candidate_manifest["parent_review_sha256"] != _sha256(parent_review_path)
            or candidate_manifest["parent_status_sha256"] != _sha256(parent_status_path)):
        raise ValueError(f"NCU candidate parent evidence differs: {batch_name}")
    parent_plan = json.loads(parent_plan_path.read_text())
    parent_review = json.loads(parent_review_path.read_text())
    parent_status = json.loads(parent_status_path.read_text())
    if (parent_review["plan_sha256"] != _sha256(parent_plan_path)
            or parent_review["batch_kind"] != "partial"
            or parent_status["kind"] != "partial"
            or parent_status["reason"] != "batch_time_limit"):
        raise ValueError(f"NCU candidate parent is not a time-limit partial: {parent_name}")
    plan = json.loads(plan_path.read_text())
    case_ids = [case["case_id"] for case in plan["cases"]]
    if (plan["batch_name"] != batch_name or not case_ids
            or len(case_ids) != len(set(case_ids))
            or case_ids != candidate_manifest["case_ids"]):
        raise ValueError(f"NCU candidate case list differs: {batch_name}")
    parent_cases = {case["case_id"]: case for case in parent_plan["cases"]}
    for case in plan["cases"]:
        case_id = case["case_id"]
        if (case_id not in parent_cases or case != parent_cases[case_id]
                or parent_review["statuses"][case_id]["status"] != "not_attempted"
                or case_id in parent_status["case_statuses"]):
            raise ValueError(f"NCU candidate case was attempted or changed: {case_id}")
    expected_plan = {**parent_plan, "batch_name": batch_name,
                     "cases": [parent_cases[case_id] for case_id in case_ids]}
    if plan != expected_plan:
        raise ValueError(f"NCU candidate plan differs from the parent: {batch_name}")
    return plan_path
