"""Current memory source directory; declarations do not certify execution."""
import ast
import json
from pathlib import Path

from src.kernel.kernel_base import registered_kernel_kinds


def functional_inventory():
    """List actual source predicates with locations, without predicting support.

    Historical per-shape certificates intentionally do not enter this result.
    A caller can export its exact KernelSpec to test the corresponding domain.
    """
    root = Path(__file__).resolve().parents[2]
    sources = []
    for path in sorted((root / "src/kernel").glob("memory_*.py")):
        tree = ast.parse(path.read_text())
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith(("has_", "build_", "standalone_")):
                continue
            predicates = [ast.unparse(node.test) for node in ast.walk(function) if isinstance(node, ast.If)]
            sources.append({"path": str(path.relative_to(root)), "function": function.name,
                "line": function.lineno, "predicates": predicates,
                "status": "declared_not_execution_verified"})
    return {
        "registered_kernel_kinds": sorted(kind.value for kind in registered_kernel_kinds()),
        "hardware_ids": sorted(path.stem for path in (root / "configs/hardware").glob("*.yaml")),
        "source_entries": sources,
        "entry_count": len(sources),
        "public_export": "src.kernel.kernel_base.build_memory_access_program",
        "current_whole_domain_acceptance": False,
        "limitations": [
            "Source predicates are code locations, not a supported shape/hardware cross product.",
            "Kernel registration does not imply a complete native memory source.",
            "Caller storage, dynamic values and source ABI must be supplied for actual export.",
            "This directory does not import historical completion counts or timing observations.",
        ],
    }


def write_functional_inventory(directory: Path):
    result = functional_inventory()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Current memory source directory", "",
        "This lists declarations, not validated support or performance coverage.", "",
        "| Source | Function | Status |", "|---|---|---|"]
    for row in result["source_entries"]:
        lines.append(f"| {row['path']}:{row['line']} | {row['function']} | {row['status']} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return result
