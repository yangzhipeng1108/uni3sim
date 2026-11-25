"""Audit H200 GEMM publication provenance and emit duration-free key evidence.

The output is an unpublished review candidate. It does not change the H200
launch schema, select production launches, or price GEMM execution.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import zstandard

from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    gemm_launch_selection_key,
)


RUNNER_RELPATH = "scripts/kernel_dataset/nvidia_h200/gemm_runner.py"
PUBLICATION_RELPATH = "candidates/publication/publication.json"
PYTORCH_LINEAR_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/"
    "aten/src/ATen/native/Linear.cpp")
PYTORCH_MATMUL_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/"
    "aten/src/ATen/native/LinearAlgebra.cpp")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(path: Path) -> dict:
    return json.loads(zstandard.ZstdDecompressor().decompress(
        path.read_bytes()))


def _vector(text: str) -> list[int]:
    values = ast.literal_eval(text)
    if (not isinstance(values, tuple) or len(values) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 1 for value in values)):
        raise ValueError(f"invalid launch vector: {text!r}")
    return list(values)


def _launches(entry: dict) -> list[dict]:
    launches = []
    for kernel in entry["kernels"]:
        metadata = kernel["Metadata"]
        stats = kernel["Sections"]["Launch Statistics"]["Header"]
        launches.append({
            "name": metadata["Kernel Name"],
            "grid": _vector(metadata["Grid Size"]),
            "block": _vector(metadata["Block Size"]),
            "dynamic_shared_bytes": int(
                stats["Dynamic Shared Memory Per Block [byte/block]"]),
            "registers_per_thread": int(
                stats["Registers Per Thread [register/thread]"]),
        })
    if not launches:
        raise ValueError("GEMM case has no captured launch")
    return launches


def _runner_options(arguments: list[str]) -> dict[str, str]:
    if len(arguments) % 2:
        raise ValueError("GEMM runner arguments are not option/value pairs")
    options = dict(zip(arguments[::2], arguments[1::2], strict=True))
    if len(options) != len(arguments) // 2:
        raise ValueError("GEMM runner arguments repeat an option")
    required = {"--bs", "--sl", "--in_feat", "--out_feat", "--dtype",
                "--nvtx_range_name"}
    if set(options) != required:
        raise ValueError("GEMM runner arguments differ from fixed contract")
    return options


def _key(attributes: dict) -> tuple[object, ...]:
    # PyTorch v2.13.0 Linear.cpp calls matmul(input, weight.t()) without bias.
    # LinearAlgebra.cpp should_fold folds contiguous [B,M,K] to [B*M,K] mm.
    shape = {
        "b": 1,
        "m": attributes["B"] * attributes["M"],
        "n": attributes["N"],
        "k": attributes["K"],
    }
    return gemm_launch_selection_key(
        shape, attributes["dtype"], "gemm", {
            "a_layout": "row_major",
            "b_layout": "row_major",
            "c_layout": "row_major",
            "trans_a": False,
            "trans_b": True,
            "epilogue": "none",
            "has_bias": False,
        })


def build_candidate(*, repo_root: Path, campaign_root: Path,
                    output_dir: Path) -> dict:
    """Check one pinned publication chain, then write a review candidate."""

    canonical_path = repo_root / "kernel_dataset/nvidia_h200/gemm.json.zst"
    runner_path = repo_root / RUNNER_RELPATH
    source_root = output_dir
    torch_linear = source_root / "torch_v2.13.0_Linear.cpp"
    torch_matmul = source_root / "torch_v2.13.0_LinearAlgebra.cpp"
    for source in (torch_linear, torch_matmul):
        if not source.is_file():
            raise ValueError(f"missing pinned public source: {source}")
    if "auto output = at::matmul(input, weight.t());" not in torch_linear.read_text():
        raise ValueError("pinned Linear.cpp lacks the expected no-bias path")
    matmul_text = torch_matmul.read_text()
    if ("static bool should_fold" not in matmul_text
            or "const auto t1_folded = at::reshape_symint" not in matmul_text
            or "t1_folded.mm(*t2)" not in matmul_text):
        raise ValueError("pinned LinearAlgebra.cpp lacks the fold-to-mm path")

    publication_path = campaign_root / PUBLICATION_RELPATH
    publication = json.loads(publication_path.read_text())
    if (publication["artifact_kind"]
            != "nvidia_h200_kernel_dataset_publication"
            or publication["hardware_id"] != "nvidia_h200"
            or publication["vllm_stack"] != "vllm030"
            or _sha(canonical_path)
            != publication["canonical_sha256"]["gemm"]):
        raise ValueError("publication does not bind this H200 GEMM canonical")
    canonical = _document(canonical_path)
    runner_sha = _sha(runner_path)
    sources = [source for source in publication["sources"]
               if source["family"] == "gemm"]
    if not sources:
        raise ValueError("publication has no GEMM sources")
    seen: dict[str, list[tuple[dict, list[dict]]]] = defaultdict(list)
    provenance_records = 0
    candidate_root = campaign_root / "candidates"
    for source in sources:
        campaign = source["campaign"]
        manifest_path = candidate_root / campaign / "gemm.manifest.json"
        candidate_path = candidate_root / campaign / "gemm.json.zst"
        plan_path = campaign_root / "plans" / f"{campaign}.json"
        if (_sha(manifest_path) != source["manifest_sha256"]
                or _sha(candidate_path) != source["candidate_sha256"]):
            raise ValueError(f"publication source hash mismatch: {campaign}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["campaign"] != campaign
                or manifest["family"] != "gemm"
                or manifest["vllm_stack"] != "vllm030"
                or _sha(plan_path) != manifest["plan_sha256"]):
            raise ValueError(f"GEMM manifest/plan mismatch: {campaign}")
        plan = json.loads(plan_path.read_text())
        plan_index = {item["case_id"]: item for item in plan["plans"]}
        if len(plan_index) != len(plan["plans"]):
            raise ValueError(f"duplicate plan case: {campaign}")
        for provenance in manifest["run_provenance"]:
            if (provenance["plan_sha256"] != _sha(plan_path)
                    or provenance["files_sha256"][RUNNER_RELPATH]
                    != runner_sha):
                raise ValueError(f"GEMM runner provenance mismatch: {campaign}")
            provenance_records += 1
        candidate = _document(candidate_path)
        if len(candidate) != source["measured_case_count"]:
            raise ValueError(f"GEMM candidate case count mismatch: {campaign}")
        for case_id, entry in candidate.items():
            if case_id not in plan_index:
                raise ValueError(f"unplanned GEMM case: {case_id}")
            plan_item = plan_index[case_id]
            if plan_item["runner_relpath"] != RUNNER_RELPATH:
                raise ValueError(f"GEMM case changed runner: {case_id}")
            options = _runner_options(plan_item["runner_args"])
            attributes = entry["op_info"]["Attr"]
            if (int(options["--bs"]), int(options["--sl"]),
                    int(options["--in_feat"]), int(options["--out_feat"]),
                    options["--dtype"]) != (
                    attributes["B"], attributes["M"], attributes["K"],
                    attributes["N"], attributes["dtype"]):
                raise ValueError(f"GEMM plan shape mismatch: {case_id}")
            if entry["h200_capture"]["campaign"] != campaign:
                raise ValueError(f"GEMM candidate origin mismatch: {case_id}")
            environment = entry["h200_capture"]["environment"]
            if (environment["torch_version"] != "2.13.0+cu129"
                    or environment["torch_cuda_version"] != "12.9"):
                raise ValueError(f"GEMM runtime version mismatch: {case_id}")
            seen[case_id].append((attributes, _launches(entry)))
    if set(seen) != set(canonical):
        raise ValueError("candidate cases do not cover published GEMM cases")

    rows: dict[tuple[object, ...], dict] = {}
    for case_id, entry in canonical.items():
        attributes = entry["op_info"]["Attr"]
        launches = _launches(entry)
        if (attributes, launches) not in seen[case_id]:
            raise ValueError(f"published GEMM launch has no source: {case_id}")
        key = _key(attributes)
        if key in rows:
            if rows[key]["launches"] != launches:
                raise ValueError(f"same physical GEMM key has conflicting launch: {case_id}")
            rows[key]["case_ids"].append(case_id)
        else:
            rows[key] = {
                "selection_key_values": list(key),
                "case_ids": [case_id],
                "launches": launches,
            }
    candidate_payload = {
        "kind": "h200_dense_gemm_physical_key_review_candidate",
        "status": "unpublished_not_for_forward",
        "selection_key_fields": list(GEMM_LAUNCH_SELECTION_KEY_FIELDS),
        "source_canonical_sha256": _sha(canonical_path),
        "source_publication_sha256": _sha(publication_path),
        "source_runner_sha256": runner_sha,
        "source_torch_linear_sha256": _sha(torch_linear),
        "source_torch_matmul_sha256": _sha(torch_matmul),
        "case_count": len(canonical),
        "unique_physical_key_count": len(rows),
        "gemm_source_campaign_count": len(sources),
        "provenance_record_count": provenance_records,
        "rows": [rows[key] for key in sorted(rows)],
        "limitations": [
            "Logical row-major key is derived from the source operation; the exact cuBLASLt descriptor and workspace are not captured for all cases.",
            "Kernel names and launch metadata are observations, not tile execution semantics or time prices.",
            "Current H200 launch schema cannot encode cluster/cooperative fields; main agent must review any canonical publication.",
        ],
    }
    candidate_path = output_dir / "physical_keys.json"
    manifest_path = output_dir / "manifest.json"
    if candidate_path.exists() or manifest_path.exists():
        raise ValueError("candidate output already exists")
    candidate_path.write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "kind": "h200_dense_gemm_physical_key_review_manifest",
        "status": "candidate_only_unpublished",
        "canonical_sha256": _sha(canonical_path),
        "publication_sha256": _sha(publication_path),
        "runner_sha256": runner_sha,
        "torch_sources": [
            {"path": torch_linear.name, "url": PYTORCH_LINEAR_URL,
             "sha256": _sha(torch_linear)},
            {"path": torch_matmul.name, "url": PYTORCH_MATMUL_URL,
             "sha256": _sha(torch_matmul)},
        ],
        "candidate_sha256": _sha(candidate_path),
        "case_count": len(canonical),
        "unique_physical_key_count": len(rows),
        "duration_consumed": False,
        "hardware_execution": False,
        "retention": "pinned-raw; compress after closure, do not delete",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_candidate(
        repo_root=args.repo_root, campaign_root=args.campaign_root,
        output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
