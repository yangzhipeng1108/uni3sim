"""Run bounded consecutive V4 metadata forwards and retain source receipts."""

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess

from .symbolic_model import evaluate_symbolic_model
from .v4_state_source import compare_v4_logical_sequence
from src.serving.workload.runtime_inputs import ExpertRoutingInput


def _evaluate_step(arguments):
    options, declaration = arguments
    requests = declaration["requests"]
    return evaluate_symbolic_model(options["model"], tuple(r["query"] for r in requests),
        tuple(r["computed"] for r in requests), upstream_root=Path(options["source_root"]),
        source_profile="vllm_0_30_v4_sync_fp8_ds_mla", tensor_parallel_size=4,
        source_max_model_len=options["max_model_len"], max_batched_tokens=options["max_batched_tokens"],
        source_use_deep_gemm=options["use_deep_gemm"],
        expert_routing=ExpertRoutingInput.seeded_assumption("declared-sequence-source-validation", 0),
        torch_include=Path(options["torch_include"]), flashinfer_source_root=Path(options["flashinfer_source_root"]),
        workspace_initial_capacity_bytes=options["workspace_initial_capacity_bytes"],
        indexer_selection_source=options["indexer_selection_source"])


def write_v4_source_sequence(options, declarations, *, initial_prefixes, output, workers):
    """Each forecast is independent CPU work; sequence validation runs after it.

    Explicit contexts stand for prior logical cache contents. No model values,
    physical pages or request scheduling decisions are fabricated or executed.
    """
    if not declarations or type(workers) is not int or workers <= 0:
        raise ValueError("source sequence needs nonempty declarations and positive CPU worker count")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    snapshot = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip(), "uncommitted_changes": subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()}
    arguments = [(options, declaration) for declaration in declarations]
    # Concurrent forecasts do not mutate numerical/cache state. Their declared
    # logical ordering is checked separately below; GPU concurrency is absent.
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
        results = list(pool.map(_evaluate_step, arguments))
    comparison = compare_v4_logical_sequence(declarations, results, initial_prefixes=initial_prefixes)
    summary, source_failures = [], []
    for ordinal, (declaration, result) in enumerate(zip(declarations, results, strict=True)):
        path = output / f"step-{ordinal}.json.gz"
        with gzip.open(path, "wt", compresslevel=1) as stream:
            json.dump({"snapshot": snapshot, "declaration": declaration, "result": result}, stream, separators=(",", ":"))
        scopes = {name: {field: value[field] for field in ("total", "matched", "mismatched", "not_checked")}
            for name, value in result.items() if name.startswith("source_") and name.endswith("_comparison") and "total" in value}
        for name, counts in scopes.items():
            if counts["mismatched"] or counts["not_checked"]:
                source_failures.append({"step": declaration["step_id"], "scope": name, "counts": counts})
        inventory = result["source_v4_scope_inventory"]
        if not inventory["logical_math_role_inventory_complete"]:
            source_failures.append({"step": declaration["step_id"], "scope": "logical_math_role_inventory", "inventory": inventory})
        if result["unpriced_calls"]:
            source_failures.append({"step": declaration["step_id"], "scope": "unpriced_calls", "calls": result["unpriced_calls"]})
        summary.append({"step": declaration["step_id"], "result": str(path), "calls": result["kernel_calls"],
            "priced": result["priced_calls"], "source_counts": scopes,
            "source_role_families": inventory["source_role_families"],
            "primitive_role_families": inventory["primitive_role_families"],
            "source_storage_counts": inventory["source_storage_counts"],
            "unchecked_storage_by_role": inventory["unchecked_storage_by_role"],
            "logical_math_role_inventory_complete": inventory["logical_math_role_inventory_complete"]})
    report = {"snapshot": snapshot, "execution": "pure_python_metadata_no_real_tensor_or_device",
        "options": options, "declarations": declarations, "initial_prefixes": initial_prefixes,
        "steps": summary, "sequence_comparison": comparison, "source_failures": source_failures,
        "logical_sequence_passed": comparison["mismatched"] == 0 and not source_failures,
        "whole_model_source_alignment_accepted": False, "normal_capacity_accepted": False, "runtime_accepted": False,
        "remaining_required_facets": ["source internal allocation/alias/lifetime receipt join; see each step's scope inventory"],
        "limits": ["initialized prefixes and expert assignments are declared synthetic inputs",
            "source forecasts are not real Scheduler requests, numerical forwards or physical cache continuity",
            "private provider workspace and compiler lifetimes are not proved by logical ports"]}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# DeepSeek-V4 Flash consecutive source-graph checks", "", "Full source alignment remains incomplete. These checks cover consecutive logical requests, full-layer-stack Roofline, cache ports, and TP data volume.", "",
        "| Phase | Priced calls | Cache checks | TP reduction checks |", "|---|---:|---:|---:|"]
    for row in summary:
        state = row["source_counts"]["source_v4_logical_state_comparison"]
        reduction = row["source_counts"]["source_v4_reduction_comparison"]
        lines.append(f"| {row['step']} | {row['priced']}/{row['calls']} | {state['matched']}/{state['total']} | {reduction['matched']}/{reduction['total']} |")
    lines.extend(["", f"Consecutive logical states: {comparison['matched']}/{comparison['total']}; failed scopes: {len(source_failures)}.",
        "", "Unified comparisons of source-internal temporary allocation, aliases, and lifetimes remain incomplete. Actual provider, normal capacity, and MTP are independently unaccepted. No GPU or real Tensor compute is performed.",
        "", "[Full comparison and command arguments](comparison.json)", ""])
    (output / "report.md").write_text("\n".join(lines))
    return report
