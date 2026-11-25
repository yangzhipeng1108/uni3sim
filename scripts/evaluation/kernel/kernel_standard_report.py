#!/usr/bin/env python3
"""CLI for the report-only single-kernel Forward standard report assembly."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.kernel.report_only import REPORT_HARDWARE, write_kernel_matrix_report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for hardware_id in REPORT_HARDWARE:
        parser.add_argument(f"--{hardware_id.replace('_', '-')}-semantic-json",
            type=Path, dest=f"{hardware_id}_semantic_json",
            help=f"saved semantic JSON for {hardware_id}")
    parser.add_argument("--matrix-output", type=Path, required=True,
        help="rebuilt kernel matrix JSON consumed by the standard report")
    parser.add_argument("--overall-output", type=Path,
        help="cross-hardware latency scatter overview PNG")
    parser.add_argument("--detailed-output", type=Path,
        help="per-bucket scatter / APE CDF detail PNG")
    parser.add_argument("--wall-time-output", type=Path,
        help="optional JSON record of the report-only rebuild wall times")
    args = parser.parse_args()

    semantic_json = {
        hardware_id: getattr(args, f"{hardware_id}_semantic_json")
        for hardware_id in REPORT_HARDWARE
    }
    result = write_kernel_matrix_report(
        semantic_json, args.matrix_output, args.overall_output,
        args.detailed_output, args.wall_time_output)
    print(result["matrix_path"])
    if result["overall_path"] is not None:
        print(result["overall_path"])
        print(result["detailed_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
