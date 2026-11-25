#!/usr/bin/env python3
"""Run aggregate-only ShareGPT compact-suite evaluation."""
import argparse, json, os
from pathlib import Path
import sys
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path: sys.path.insert(0, str(REPOSITORY_ROOT))
from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact  # noqa: E402
from evaluation.serving.metrics.compact_alignment_suite import run_compact_alignment_suite  # noqa: E402
from evaluation.serving.simulation import simulation_runner  # noqa: E402
DEFAULT_CONFIG = Path("configs/evaluation/experiments/sharegpt_phase1.json")
DEFAULT_JOBS = os.cpu_count() or 1
def _run_isolated_case(repository_root: str, config: compact.Phase1Config, cell: compact.Phase1Cell, source_commit: str, alignment_checked: bool | dict[str, object], execution_mode: str, debug_num_kv_blocks: int | None = None, debug_analytical_launch: bool = False, host_launch_profile_paths: dict[str, str] | None = None, scheduler_host_profile_paths: dict[str, str] | None = None) -> dict[str, object]:
    return compact.run_isolated_case(
        repository_root, config, cell, source_commit, alignment_checked, execution_mode,
        load_validated_case=load_compact_case, run_simulation=simulation_runner.run_cli_args,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch, host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths)


def _run_isolated_case_three_modes(repository_root: str, config: compact.Phase1Config, cell: compact.Phase1Cell, source_commit: str, alignment_checked: bool | dict[str, object], execution_mode: str, debug_num_kv_blocks: int | None = None, debug_analytical_launch: bool = False, host_launch_profile_paths: dict[str, str] | None = None, scheduler_host_profile_paths: dict[str, str] | None = None) -> dict[str, object]:
    return compact.run_isolated_case_three_modes(
        repository_root, config, cell, source_commit, alignment_checked, execution_mode,
        load_validated_case=load_compact_case, run_simulation=simulation_runner.run_cli_args, debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch, host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths)
def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append", default=[], help="hardware/model/case_id; repeat for multiple cells")
    selection.add_argument("--all-cases", action="store_true")
    selection.add_argument("--all-official-metrics", action="store_true", help="run the frozen 45-case official-metrics matrix; --virtual-forward also retains its 15 H2000.30 cases")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-mode", choices=("accelerated", "reference"), default="accelerated", help="run-local execution strategy")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, metavar="N", help="isolated case workers; parent writes ordered output")
    parser.add_argument("--metrics-only", action="store_true", help="debug-only: skip alignment and mark results non-formal")
    parser.add_argument("--debug-num-kv-blocks", type=int, help="debug-only explicit Scheduler capacity; requires one case and metrics-only")
    parser.add_argument("--debug-analytical-launch", action="store_true", help="debug-only: use analytical rather than injected launch facts")
    parser.add_argument("--host-launch-profile", action="append", default=[], help="override automatic Host launch profiles; repeat for selected hardware")
    parser.add_argument("--scheduler-host-profile", action="append", default=[], help="override automatic Scheduler Host profiles; repeat for selected hardware")
    parser.add_argument("--device-only", action="store_true", help="disable the default full Host timing run and report only Device timing")
    parser.add_argument("--three-modes", action="store_true", help="run independent device-only, launch, and scheduler simulations")
    parser.add_argument("--virtual-forward", action="store_true", help="evaluate virtual Tensor migration with each original prediction contract, using device-only timing; unsupported source bindings remain refusals, not Host or three-mode acceptance")
    parser.add_argument("--vllm-sdk-python", type=Path, help="optional genuine CPU vLLM0.30 SDK Python, needed only by an original0.30 Scheduler contract; a complete SDK/source pair selects virtual migration evaluation")
    parser.add_argument("--vllm-source-root", type=Path, help="optional local vLLM0.30 source paired with --vllm-sdk-python; original0.11 Scheduler cases use the current CPU Python without this pair")
    return parser.parse_args(argv)
def run_cli(argv=None) -> dict[str, object]:
    args = _parse_args(argv)
    declared_cpu_source = (args.vllm_sdk_python is not None, args.vllm_source_root is not None)
    if any(declared_cpu_source):
        if not all(declared_cpu_source):
            raise ValueError("virtual forward requires both --vllm-sdk-python and --vllm-source-root")
        args.virtual_forward = True
    if args.virtual_forward:
        if (Path(args.config) != DEFAULT_CONFIG or args.metrics_only
                or args.debug_num_kv_blocks is not None or args.debug_analytical_launch
                or args.host_launch_profile or args.scheduler_host_profile or args.three_modes
                or args.execution_mode != "accelerated"):
            raise ValueError("--virtual-forward preserves original prediction contracts with device-only timing; Host timing overrides, alternate execution, custom configs and debug capacity are not admitted")
        if all(declared_cpu_source):
            if not args.vllm_sdk_python.is_file() or not args.vllm_source_root.is_dir():
                raise ValueError("virtual-forward SDK Python or source root does not exist")
        if args.all_cases:
            config = compact.load_phase1_config(REPOSITORY_ROOT / DEFAULT_CONFIG)
            selected_ids = tuple(cell.cell_id for cell in compact.phase1_cells(config))
        else:
            selected_ids = tuple(args.case)
        return compact.run_virtual_forward_report(
            REPOSITORY_ROOT, args.run_id, selected_ids, jobs=args.jobs,
            python=Path(sys.executable),
            sdk_python=None if args.vllm_sdk_python is None else args.vllm_sdk_python.absolute(),
            source_root=None if args.vllm_source_root is None else args.vllm_source_root.resolve(),
            load_validated_case=load_compact_case)
    if args.device_only and (args.host_launch_profile or args.scheduler_host_profile or args.three_modes):
        raise ValueError("--device-only cannot be combined with Host timing options")
    if args.debug_num_kv_blocks is not None:
        if args.debug_num_kv_blocks < 1:
            raise ValueError("--debug-num-kv-blocks must be positive")
        if len(args.case) != 1 or not args.metrics_only:
            raise ValueError("--debug-num-kv-blocks requires exactly one --case and --metrics-only")
    source_commit = compact.repository_source_commit(REPOSITORY_ROOT)
    isolated_case_worker = (
        _run_isolated_case_three_modes if args.three_modes else _run_isolated_case)
    if args.all_official_metrics:
        if Path(args.config) != DEFAULT_CONFIG:
            raise ValueError("--all-official-metrics cannot override --config")
        alignment = None if args.metrics_only else run_compact_alignment_suite(REPOSITORY_ROOT, load_compact_case, source_commit=source_commit, jobs=args.jobs)
        host_launch_profile_paths = None if args.device_only else compact.resolve_host_launch_profile_paths(REPOSITORY_ROOT, args.host_launch_profile, None)
        scheduler_host_profile_paths = None if args.device_only else compact.resolve_scheduler_host_profile_paths(REPOSITORY_ROOT, args.scheduler_host_profile, None)
        return compact.run_official_metrics_evaluation(
            REPOSITORY_ROOT, args.run_id, source_commit, jobs=args.jobs, metrics_only=args.metrics_only, compact_alignment_summary=alignment, execution_mode=args.execution_mode,
            isolated_case_worker=isolated_case_worker, debug_analytical_launch=args.debug_analytical_launch,
            host_launch_profile_paths=host_launch_profile_paths,
            scheduler_host_profile_paths=scheduler_host_profile_paths)
    config_path = compact.resolve_repository_input(REPOSITORY_ROOT, args.config, "experiment config")
    config = compact.load_phase1_config(config_path)
    cells = compact.select_phase1_cells(config, args.case, all_cases=args.all_cases)
    host_launch_profile_paths = None if args.device_only else compact.resolve_host_launch_profile_paths(REPOSITORY_ROOT, args.host_launch_profile, {cell.hardware.hardware_id for cell in cells})
    scheduler_host_profile_paths = None if args.device_only else compact.resolve_scheduler_host_profile_paths(REPOSITORY_ROOT, args.scheduler_host_profile, {cell.hardware.hardware_id for cell in cells})
    return compact.run_phase1_evaluation(
        REPOSITORY_ROOT, config, cells, args.run_id, source_commit, jobs=args.jobs, metrics_only=args.metrics_only, execution_mode=args.execution_mode, isolated_case_worker=isolated_case_worker,
        debug_num_kv_blocks=args.debug_num_kv_blocks, debug_analytical_launch=args.debug_analytical_launch,
        host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths)
def main(argv=None) -> int:
    summary = run_cli(argv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["suite_id"] == "sharegpt_virtual_tensor":
        return int(summary["completed_case_count"] != summary["case_count"])
    failed_formal = (summary["suite_id"] == "sharegpt_official_metrics_45" and summary["metrics_only_debug"] is False and summary["formal_conclusion_allowed"] is False)
    return int(failed_formal)
if __name__ == "__main__":
    raise SystemExit(main())
