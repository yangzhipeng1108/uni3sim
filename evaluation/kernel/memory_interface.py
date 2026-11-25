"""Untimed memory export, independent small-case checks and diagnostic CLI."""
from pathlib import Path
import json

from src.kernel.memory_access import Operation, Span, Strided, statistics
from src.kernel.memory_access_codec import load_jsonl


def independent_counts(program):
    """Small-case reference enumerator, independent of production interval code."""
    from itertools import product
    requested = 0
    footprint = set()
    if program.tile_programs:
        raise NotImplementedError("expand compressed tiles before independent enumeration")
    enumerated = 0
    for access in program.accesses:
        if access.predicate is not None:
            raise NotImplementedError("conditional access needs explicit runtime values")
        region = access.region
        if isinstance(region, Span):
            starts = [region.offset_bytes]
            width = region.length_bytes
        elif isinstance(region, Strided):
            if region.element_span_bytes * product_count(region.counts) > 1_000_000:
                raise NotImplementedError("reference enumeration is limited to small acceptance cases")
            starts = [region.base_offset_bytes + sum(i*s for i, s in zip(indices, region.strides_bytes))
                      for indices in product(*(range(n) for n in region.counts))]
            width = region.element_span_bytes
        else:
            raise NotImplementedError("reference region not supported")
        enumerated += len(starts) * width
        if enumerated > 1_000_000:
            raise NotImplementedError("reference enumeration is limited to one million candidate bytes")
        requested += len(starts) * width * access.repeat
        for start in starts:
            footprint.update((access.allocation_id, byte) for byte in range(start, start + width))
    return requested, len(footprint)


def product_count(counts):
    result = 1
    for count in counts:
        result *= count
    return result



def write_report(program, export_path: Path):
    """Produce current evidence; a local export never completes a Goal."""
    from evaluation.reporting.memory_tiles import program_diagnostics
    with export_path.open() as stream:
        restored = load_jsonl(stream)
    if restored != program:
        raise ValueError("export roundtrip mismatch")
    coverage = program_diagnostics(program)
    try:
        requested, unique = independent_counts(program)
    except NotImplementedError as error:
        coverage["independent_count"] = "unsupported"
        coverage["independent_count_reason"] = str(error)
    else:
        if (requested, unique) != (coverage["requested_bytes"], coverage["unique_footprint_bytes"]):
            raise ValueError("independent counts differ from the production statistics")
        coverage["independent_count"] = "pass"
        coverage["independent_count_reason"] = None
    directory = export_path.parent
    (directory / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (directory / "report.md").write_text(
        "# Memory export evidence\n\n"
        f"Program `{program.program_id}`: {program.coverage.semantics.value}, "
        f"{program.coverage.granularity.value}. This is not physical traffic or kernel timing.\n\n"
        "| Check | Result |\n|---|---|\n"
        f"| Codec roundtrip | {coverage['roundtrip']} |\n"
        f"| Requested bytes (unknown if conditional) | {coverage['requested_bytes']} |\n"
        f"| Candidate requested bytes | {coverage['candidate_requested_bytes']} |\n"
        f"| Unique footprint bytes (unknown if unresolved) | {coverage['unique_footprint_bytes']} |\n"
        f"| Independent small-case count | {coverage['independent_count']} |\n\n"
        + "\n".join(f"- {limitation}" for limitation in program.coverage.limitations) + "\n")
    return coverage


def _output_path(path):
    root = Path(__file__).resolve().parents[2]
    path = path.resolve()
    if not any(path.is_relative_to(root / "experiment_logs/evaluation" / name)
               for name in ("memory_interface", "memory_unification")):
        raise ValueError("memory outputs must be under experiment_logs/evaluation/memory_interface or memory_unification")
    return path


def main(argv=None):
    """Argument handling belongs to evaluation, not the executable shell."""
    import argparse
    import sys
    from src.kernel.memory_access import Granularity
    from src.kernel.memory_access_codec import dump_jsonl
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and not argv[0].startswith("-") else "gemm"
    parser = argparse.ArgumentParser(description=__doc__)
    if command in ("inventory", "functional-inventory"):
        from evaluation.kernel.memory_inventory import write_functional_inventory
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args(argv)
        result = write_functional_inventory(_output_path(args.output_dir))
        print(f"{result['entry_count']} source declarations; no whole-domain acceptance claimed")
        return
    if command == "compare-exports":
        from evaluation.kernel.memory_frozen_evaluation import compare_exports
        parser.add_argument("--before", type=Path, required=True)
        parser.add_argument("--after", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv)
        output = _output_path(args.output)
        if output in (args.before.resolve(), args.after.resolve()):
            parser.error("comparison output cannot replace either input")
        result = compare_exports(args.before, args.after)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(output)
        return
    if command == "expand-tiles":
        from evaluation.reporting.memory_tiles import expand_export
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--max-accesses", type=int, default=10000)
        parser.add_argument("--max-dependencies", type=int, default=100000)
        args = parser.parse_args(argv)
        output = _output_path(args.output)
        expand_export(args.input, output, max_accesses=args.max_accesses, max_dependencies=args.max_dependencies)
        with output.open() as stream:
            write_report(load_jsonl(stream), output)
        print(output)
        return
    if command == "report":
        parser.add_argument("--input", type=Path, required=True)
        args = parser.parse_args(argv)
        source = _output_path(args.input)
        with source.open() as stream:
            write_report(load_jsonl(stream), source)
        print(source.parent / "report.md")
        return
    if command not in ("gemm", "transfer"):
        parser.error("unknown or historical campaign command; use --help for supported commands")
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", action="store_true")
    if command == "gemm":
        from src.kernel.memory_access_builder import GEMMWork, build_logical_gemm, standalone_bindings
        parser.add_argument("--m", type=int, required=True)
        parser.add_argument("--n", type=int, required=True)
        parser.add_argument("--k", type=int, required=True)
        parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), required=True)
        parser.add_argument("--granularity", choices=tuple(g.value for g in Granularity), required=True)
        parser.add_argument("--tile", nargs=3, type=int)
        parser.add_argument("--call-id", required=True)
        parser.add_argument("--binding-id", required=True)
        args = parser.parse_args(argv)
        output = _output_path(args.output)
        work = GEMMWork(args.m, args.n, args.k, args.dtype, 4 if args.dtype == "fp32" else 2)
        bindings = standalone_bindings(work, call_id=args.call_id, binding_id=args.binding_id,
            device_id="offline-device", visibility_domain="offline-visible")
        # This logical ABI makes no native launch claim. Actual KernelSpecs use
        # export_kernel_memory and the shared public kernel owner.
        program = build_logical_gemm(work, args.hardware_id, bindings, program_id=args.program_id,
            granularity=Granularity(args.granularity), tile=None if args.tile is None else tuple(args.tile))
    else:
        from src.kernel.hardware_spec import HWSpec
        from src.kernel.hardware_ids import canonical_hardware_id
        from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
        from src.kernel.kernel_base import build_memory_access_program
        parser.add_argument("--kind", choices=("MEMCPY", "KV_OFFLOAD"), required=True)
        parser.add_argument("--bytes", type=int, required=True)
        parser.add_argument("--direction", required=True)
        parser.add_argument("--algorithm", required=True)
        parser.add_argument("--link", required=True)
        parser.add_argument("--tier")
        args = parser.parse_args(argv)
        output = _output_path(args.output)
        root = Path(__file__).resolve().parents[2]
        hardware = HWSpec(root / "configs/hardware" / f"{canonical_hardware_id(args.hardware_id)}.yaml")
        kernel = KernelSpec(KernelKind.parse(args.kind), hardware.hardware_id,
            transfer=PhysicalTransferConfig(args.algorithm, args.bytes, args.direction, args.link, tier=args.tier))
        program = build_memory_access_program(kernel, hardware, None, Granularity.KERNEL, program_id=args.program_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        dump_jsonl(program, stream)
    if args.report:
        write_report(program, output)
    print(output)
