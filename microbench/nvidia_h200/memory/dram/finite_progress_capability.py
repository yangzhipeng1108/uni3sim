"""Emit byte-progress capabilities from a pinned independent capture only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def emit(manifest_path, reference_output, yaml_output, *, slots=None, append=False, coalesced=False):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["forward_eligible"] is not True:
        raise ValueError("capture is not eligible for capability publication")
    for kind in ("raw", "summary"):
        if hashlib.sha256(Path(manifest[kind+"_path"]).read_bytes()).hexdigest() != manifest[kind+"_sha256"]:
            raise ValueError("pinned independent evidence changed")
    summary = json.loads(Path(manifest["summary_path"]).read_text())
    if (summary["production_or_evaluation_duration_consumed"] is not False
            or not all(summary["instruction_counts_match"].values())
            or not all(summary["counter_dram_logical_within_1pct"].values())):
        raise ValueError("independent scope/unit checks failed")
    groups = defaultdict(list)
    captured_points = 0
    for point in summary["progress_points"]:
        if coalesced:
            panel, depth, threads, grid, rounds, row, stride, tier, amode, fanout = point["key"]
        else:
            panel, depth, threads, grid, rounds, row, stride, tier, amode = point["key"]
            fanout = 0
        if slots is not None and depth != slots:
            continue
        captured_points += 1
        if (point["eligible"] != all(point["checks"].values())
                or tier != ("cold_shared_requests" if coalesced else "cold_unique")
                or (amode,threads) not in ((2,128),(4,256),(5,256))
                or rounds != (stride+row-1)//row or panel % row):
            raise ValueError("byte-progress physical scope drift")
        logical = panel//row*stride*grid
        expected = logical/max(point["copy_body_ns_by_process"])*1e9
        if logical != point["logical_B_bytes"] or not math.isclose(
                point["logical_byte_capacity_per_s"], expected, rel_tol=1e-12):
            raise ValueError("logical-byte capacity does not conserve work")
        if point["eligible"]:
            groups[panel,depth,threads,row,stride,128 if amode in (4,5) else 16,fanout].append(
                (grid, point["logical_byte_capacity_per_s"]))
    programs, yaml = [], []
    for (panel,slots,threads,row,stride,tile_m,fanout), points in sorted(groups.items()):
        points.sort()
        name = (f"single_panel_async_copy_threads{threads}_resident8_b{panel}" if slots == 1
                else f"finite_async_copy_threads{threads}_slots{slots}_b{panel}")
        if coalesced:
            name = f"coalesced_async_copy_threads{threads}_slots{slots}_b{panel}_fanout{fanout}"
        pid = f"b{panel}_s{slots}_r{row}_stride{stride}" + (f"_fanout{fanout}" if coalesced else "")
        signature = {"access_class":name,"dtype":"bf16","tile_m":tile_m,
                     "tile_k":row//2,"logical_row_bytes":stride}
        programs.append({"program_id":pid,"service":name,"signature":signature,
            "panel_B_bytes":panel,"companion_A_bytes":tile_m*row,"companion_valid_rows":tile_m if threads == 256 and not coalesced else 1,
            "buffer_slots":slots,"companion_issued_rows":tile_m,"threads_per_cta":threads,"physical_row_stride_bytes":stride,
            "logical_byte_capacity_per_s_by_grid_ctas":points})
        if coalesced:
            programs[-1]["request_fanout"] = fanout
        yaml.extend([f"    - service: {name}","      access_kind: read",
            "      ingress_resource_role: l2_cache","      signature:",
            f"        access_class: {name}","        dtype: bf16",f"        tile_m: {tile_m}",
            f"        tile_k: {row//2}",f"        logical_row_bytes: {stride}",
            "      scope: full_chip",f"      logical_byte_capacity_per_s: {points[0][1]}",
            "      logical_byte_capacity_per_s_by_grid_ctas:"])
        yaml.extend(f"        {grid}: {capacity}" for grid,capacity in points)
        yaml.extend([f"      evidence_ref: microbench/nvidia_h200/references/finite_async_byte_progress_sm90.json#programs/{pid}",
                     "      replay_to_dram_fraction: null"])
    reference = {"hardware_id":"nvidia_h200","mechanism":"finite_async_byte_ring_progress",
        "production_or_evaluation_duration_consumed":False,
        "source_manifest_path":str(manifest_path),
        "source_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_rule":summary["selection"],
        "counter_dram_logical_within_1pct":summary["counter_dram_logical_within_1pct"],
        "instruction_counts_match":summary["instruction_counts_match"],
        "eligible_points":sum(len(points) for points in groups.values()),
        "captured_points":captured_points,
        "scope":"Complete independent byte-ring progress; only exact eligible grid/signature points bind. No matrix/routing/operator output.",
        "composition":"Competing execution floor beside unchanged generic work; no control subtraction or physical traffic inflation.",
        "approximations":["Broadcast warm one-row A and contiguous synthetic B; full original A/DRAM work stays in the normal pipeline.",
                          "Native shared layout, metadata and expert-address distribution remain analytical.",
                          "Mechanism progress reference, not a certified mathematical lower bound for every compiler."],
        "programs":programs}
    if append:
        existing = json.loads(reference_output.read_text())
        if {p["program_id"] for p in existing["programs"]} & {p["program_id"] for p in programs}:
            raise ValueError("supplement cannot replace an existing program")
        for program in programs:
            program["source_manifest_path"] = str(manifest_path)
            program["source_manifest_sha256"] = reference["source_manifest_sha256"]
        existing["programs"].extend(programs)
        existing["eligible_points"] += reference["eligible_points"]
        existing["captured_points"] += captured_points
        reference = existing
    reference_output.write_text(json.dumps(reference,indent=2)+"\n")
    yaml_output.write_text("\n".join(yaml)+"\n")
    print(f"programs={len(programs)} eligible_points={reference['eligible_points']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("reference_output",type=Path)
    parser.add_argument("yaml_output",type=Path)
    parser.add_argument("--slots",type=int)
    parser.add_argument("--append",action="store_true")
    parser.add_argument("--coalesced",action="store_true")
    args = parser.parse_args()
    emit(args.manifest,args.reference_output,args.yaml_output,slots=args.slots,append=args.append,coalesced=args.coalesced)
