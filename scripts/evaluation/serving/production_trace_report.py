#!/usr/bin/env python3
"""Render a projected production-Trace arrival-rate sweep as Markdown."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.production_trace_report import (  # noqa: E402
    render_production_trace_sweep_report,
    write_production_trace_sweep_plots,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="sweep JSON path")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument(
        "--plot-dir",
        help="plot output directory; defaults beside the Markdown report",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("report input and output paths must differ")
    try:
        payload = json.loads(
            input_path.read_bytes(),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid production Trace sweep JSON: {input_path}") from error
    report = render_production_trace_sweep_report(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = (
        Path(args.plot_dir).resolve()
        if args.plot_dir is not None
        else output_path.with_name(f"{output_path.stem}_plots")
    )
    plot_paths = write_production_trace_sweep_plots(plot_dir, payload)
    report += "## Plots\n\n"
    for name, path in sorted(plot_paths.items()):
        relative_path = os.path.relpath(path, output_path.parent)
        report += f"- {name}: ![{name}]({relative_path})\n"
    report += "\n"
    output_path.write_text(report, encoding="utf-8")
    return 0


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(
                f"duplicate production Trace report JSON field: {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
