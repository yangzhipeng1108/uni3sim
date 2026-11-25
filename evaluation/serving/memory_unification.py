"""Memory diagnostics for the executed, traced Serving KernelPlan.

The trace owns storage. This report consumes per-call receipts and does not
construct another graph, allocate replacement storage or calculate timings.
"""
import json
from pathlib import Path

from evaluation.reporting.memory_tiles import encode, program_diagnostics
from src.kernel.kernel_base import build_memory_access_program
from src.kernel.memory_access import Granularity
from src.kernel.memory_access_codec import dump_jsonl
from src.serving.kernel_lowering.kernel_plan import KernelPlan


def summarize_memory_plan(plan: KernelPlan, hardware_by_id, *, directory: Path | None = None):
    """Report exact caller bindings; incomplete families remain partial.

    Only NotImplementedError from an admitted export is recorded as unsupported.
    Malformed bindings and missing hardware remain failures, not zero-byte rows.
    """
    if not isinstance(plan, KernelPlan):
        raise TypeError("memory report requires the current KernelPlan")
    directory = None if directory is None else Path(directory)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, call in enumerate(plan.logical_calls()):
        source = call.memory_source
        row = {
            "call_id": call.call_id, "pool_name": call.pool_name, "rank": call.rank,
            "kind": call.kernel_spec.kind.value, "variant": call.kernel_spec.variant,
            "source_node_ids": list(call.source_node_ids), "dependencies": list(call.dependencies),
            "source_ports": None if source is None else encode(source),
            "export_path": None, "program": None,
        }
        if call.memory_bindings is None:
            row.update(status="partial", reason=(call.memory_unbound_reason
                if call.memory_unbound_reason is not None else "source_trace_not_supplied"))
        else:
            hardware = hardware_by_id[call.kernel_spec.hardware_id]
            try:
                program = build_memory_access_program(call.kernel_spec, hardware,
                    call.memory_bindings, Granularity.KERNEL, program_id=call.call_id + ":memory")
                diagnostics = program_diagnostics(program)
            except NotImplementedError as error:
                row.update(status="unsupported", reason=str(error))
            else:
                row.update(status="exported", reason=None, program=diagnostics)
                if directory is not None:
                    output = directory / f"call_{index:06d}.jsonl"
                    with output.open("w") as stream:
                        dump_jsonl(program, stream)
                    row["export_path"] = output.name
        rows.append(row)
    exported = sum(row["status"] == "exported" for row in rows)
    return {
        "logical_call_count": len(rows), "exported_call_count": exported,
        "binding_coverage": dict(plan.memory_binding_coverage), "calls": rows,
        "status": "exported" if rows and exported == len(rows) else "partial",
        "performance_acceptance": False, "normal_capacity_accepted": False,
        "limitations": [
            "Per-call source bytes are logical access evidence, not DRAM traffic or duration.",
            "Shared allocation identities are retained; per-call footprints must not be summed as a live peak.",
            "Bindings and successful exports do not certify native provider or normal capacity coverage.",
        ],
    }


def write_memory_plan_report(plan, hardware_by_id, directory: Path):
    directory = Path(directory)
    result = summarize_memory_plan(plan, hardware_by_id, directory=directory)
    (directory / "memory_report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Traced Serving memory", "", f"Status: {result['status']}. "
        f"Exported {result['exported_call_count']}/{result['logical_call_count']} calls.", "",
        "| Call | Pool / rank | Status | Reason |", "|---|---|---|---|"]
    for row in result["calls"]:
        lines.append(f"| {row['call_id']} | {row['pool_name']} / {row['rank']} | {row['status']} | {row['reason']} |")
    lines.extend(["", *result["limitations"]])
    (directory / "memory_report.md").write_text("\n".join(lines) + "\n")
    return result
