"""Audit one pinned H200 FA3 GQA D128 main kernel's static dependency order.

The NCU SASS view is a static listing, not a dynamic CTA trace or a duration
source.  This script does not price Forward service or inspect other kernels.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CASE = "FA_SERVING_HQ32_HKV4_D128_PREFILL_B1_QSUM729_QMAX729_KVMAX729_a469d3f812b6"
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
RAW = CAMPAIGN / "batches/ncu_flash_attn_004/ncu_raw" / CASE
REPORT = RAW / "profile.ncu-rep.zst"
PREFLIGHT = RAW / "preflight/cases" / CASE / "window_01.launch.json"
REVIEW = CAMPAIGN / "ncu_flash_attn_004_review.json"
AUDIT = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_source_audit.json"
PARTIAL = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_gqa_prefill_partial_6case.json"
FA2_FINITE = ROOT / "microbench/nvidia_h200/references/fa2_resident_ldgsts_finite_sm90.json"
FA2_QUEUED = ROOT / "microbench/nvidia_h200/references/fa2_queued_ldgsts_steady_frontend_sm90.json"
OUTPUT = ROOT / "experiment_logs/h200_accuracy_20261028/fa3_q729_binary_dependency_20261102.json"
NCU = Path("/opt/nvidia/nsight-compute/2023.2.2/ncu")


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _report_sass() -> tuple[bytes, str, list[tuple[int, str]]]:
    with tempfile.TemporaryDirectory(prefix="h200-fa3-sass-") as directory:
        rep = Path(directory) / "profile.ncu-rep"
        with rep.open("wb") as target:
            subprocess.run(["zstd", "-d", "-q", "-c", str(REPORT)],
                           stdout=target, check=True)
        result = subprocess.run(
            [str(NCU), "--import", str(rep), "--page", "source",
             "--print-source", "sass", "--csv"],
            capture_output=True, check=True)
    listing = list(csv.reader(io.StringIO(result.stdout.decode())))
    if (len(listing) < 3 or listing[0][0] != "Kernel Name"
            or "FlashAttnFwdSm90" not in listing[0][1]
            or listing[1] != ["Address", "Source"]):
        raise ValueError("NCU report lacks the expected FA3 SASS view")
    rows = []
    for address, source in listing[2:]:
        if address == "Kernel Name":
            break
        rows.append((int(address, 16), source.strip()))
    if not rows:
        raise ValueError("FA3 main SASS view is empty")
    return result.stdout, listing[0][1], rows


def main() -> None:
    review = json.loads(REVIEW.read_text())
    if (review["statuses"][CASE]["status"] != "complete"
            or review["statuses"][CASE]["kernel_count"] != 2):
        raise ValueError("FA3 source case is not quality accepted")
    campaign_paths = {item["path"]: item for item in review["raw_files"]}
    for path in (REPORT, PREFLIGHT):
        relative = str(path.relative_to(CAMPAIGN))
        local = _pin(path)
        if (relative not in campaign_paths
                or local["sha256"] != campaign_paths[relative]["sha256"]
                or local["size"] != campaign_paths[relative]["size"]):
            raise ValueError(f"FA3 raw file is not pinned: {relative}")
    launch = json.loads(PREFLIGHT.read_text())["launches"][0]
    name = launch["name"]
    if ("FlashAttnFwdSm90" not in name
            or launch["grid"] != [132, 1, 1]
            or launch["block"] != [384, 1, 1]
            or launch["registers_per_thread"] != 168
            or launch["dynamic_shared_memory_bytes"] != 166912):
        raise ValueError("FA3 launch/resources differ from source-bound case")
    mainloop = name.split("flash::CollectiveEpilogueFwd", 1)[0]
    booleans = tuple(int(value) for value in re.findall(r"\(bool\)([01])", mainloop))
    if ("CollectiveMainloopFwdSm90<(int)2," not in mainloop
            or booleans != (1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0)
            or "cutlass::bfloat16_t, (int)8>" not in mainloop):
        raise ValueError("FA3 template no longer selects staged paged GQA D128")
    audit = json.loads(AUDIT.read_text())
    if (audit["source_commit"] != "506341a143fcabd4bb79052a7605ada727d6b3f5"
            or audit["accepted_static_domain"]["block_m"] != 128
            or audit["accepted_static_domain"]["block_n"] != 128):
        raise ValueError("FA3 source audit changed")
    partial = json.loads(PARTIAL.read_text())
    candidate = next(row for row in partial["results"] if row["case_id"] == CASE)
    if candidate["main_support_level"] != "partial":
        raise ValueError("FA3 saved candidate support changed")
    source_csv, report_kernel_name, sass = _report_sass()
    if report_kernel_name != name:
        raise ValueError("NCU main SASS name differs from pinned preflight launch")
    base = sass[0][0]
    relative = {address - base: instruction for address, instruction in sass}
    # One observed steady-loop *static* path.  These offsets intentionally
    # fail closed if the report's program changes.  They are not a trace.
    expected = {
        0x6ad0: "BAR.SYNC.DEFER_BLOCKING R18, 0x100",
        0x6b70: "HGMMA.64x128x16.F32.BF16 R24",
        0x6d80: "HGMMA.64x128x16.F32.BF16 R24",
        0x6dd0: "SYNCS.PHASECHK.TRANS64.TRYWAIT",
        0x6e20: "WARPGROUP.ARRIVE",
        0x6ee0: "HGMMA.64x128x16.F32.BF16 R88",
        0x70d0: "HGMMA.64x128x16.F32.BF16 R88",
        0x70e0: "BAR.ARV R18, 0x100",
        0x70f0: "WARPGROUP.DEPBAR.LE gsb0, 0x1",
        0x1e30: "WARPGROUP.DEPBAR.LE gsb0, 0x0",
        0x12ac0: "UTMALDG.5D",
        0x12af0: "UTMALDG.5D",
    }
    for offset, token in expected.items():
        if token not in relative[offset]:
            raise ValueError(f"FA3 static instruction changed: {hex(offset)}")
    qk = [instruction for offset, instruction in relative.items()
          if 0x6b70 <= offset <= 0x6d80 and "HGMMA." in instruction]
    pv = [instruction for offset, instruction in relative.items()
          if 0x6ee0 <= offset <= 0x70d0 and "HGMMA." in instruction]
    if len(qk) != 8 or len(pv) != 8:
        raise ValueError("FA3 QK/PV static instruction groups changed")
    counts = {token: sum(token in instruction for _, instruction in sass)
              for token in ("UTMALDG.5D", "LDGSTS", "HGMMA.64x128x16",
                            "WARPGROUP.DEPBAR", "BAR.SYNC", "BAR.ARV")}
    if counts != {"UTMALDG.5D": 2, "LDGSTS": 68,
                  "HGMMA.64x128x16": 48, "WARPGROUP.DEPBAR": 6,
                  "BAR.SYNC": 11, "BAR.ARV": 11}:
        raise ValueError("FA3 static instruction census changed")
    finite = json.loads(FA2_FINITE.read_text())
    queued = json.loads(FA2_QUEUED.read_text())
    if (finite["resource_contract"]["resident_ctas_per_sm"] != 2
            or finite["resource_contract"]["threads_per_cta"] != 128
            or queued["resource_contract"]["resident_ctas_per_sm"] != 2
            or queued["resource_contract"]["static_ldgsts_bypass_128_instructions"] != 32):
        raise ValueError("FA2 service applicability changed")
    gap_ns = candidate["op_actual_ns"] - candidate["op_candidate_ns"]
    output = {
        "kind": "h200_fa3_q729_binary_dependency_audit",
        "status": "static_source_bound_no_forward_price",
        "case_id": CASE,
        "pins": {"report": _pin(REPORT), "preflight": _pin(PREFLIGHT),
                 "review": _pin(REVIEW), "source_audit": _pin(AUDIT),
                 "saved_partial": _pin(PARTIAL), "fa2_finite": _pin(FA2_FINITE),
                 "fa2_queued": _pin(FA2_QUEUED),
                 "ncu_import_sass_csv": {"size": len(source_csv),
                                         "sha256": hashlib.sha256(source_csv).hexdigest()}},
        "pinned_source_binding": {
            "revision": audit["source_commit"],
            "path": "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp",
            "sha256": audit["files"]["hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"]["sha256"],
            "source_spans": {
                "template_and_transport": "lines 30-59",
                "paged_kv_producer": "lines 786-811",
                "named_scheduler_barrier": "lines 953-968",
                "intra_wg_overlap": "lines 1212-1430",
            },
        },
        "tool": {"ncu": str(NCU), "page": "source", "view": "sass",
                 "import_is_cpu_only": True},
        "launch": {"name": name, "grid": launch["grid"],
                   "block": launch["block"], "registers_per_thread": 168,
                   "dynamic_shared_bytes": 166912},
        "template": {"stages": 2, "is_causal": True,
                     "varlen": True, "paged_kv_non_tma": True,
                     "mma_pv_rs": True, "intra_wg_overlap": True,
                     "pack_gqa": True, "split": False, "block_h": 8,
                     "source_derived_use_tma_q": True,
                     "source_derived_use_tma_kv": False},
        "static_sass": {"main_instruction_rows": len(sass),
                        "instruction_counts": counts,
                        "selected_steady_path_offsets": [
                            {"offset_hex": hex(offset), "instruction": relative[offset]}
                            for offset in expected]},
        "inference": [
            "Pinned source sets Use_TMA_Q true for PackGQA with block_h=8 and Use_TMA_KV false for PagedKVNonTMA. The two static UTMALDG.5D sites and 68 LDGSTS sites agree with this mixed path, but total static counts do not allocate every LDGSTS to K or V.",
            "A steady static path places a named scheduler sync before eight QK SS HGMMA, then eight PV RS HGMMA, a named arrive and wait_group 1. Its instruction order is source-bound, not a measured dynamic timeline.",
            "The previous three-operand 2D TMA probe selects the wrong K/V transport and serializes its HGMMA; its timing cannot price this kernel.",
        ],
        "fa2_reference_admissibility": {
            "finite_threads": 128, "finite_dynamic_shared_bytes":
                finite["resource_contract"]["dynamic_shared_bytes_per_cta"],
            "finite_resident_ctas_per_sm": 2,
            "queued_static_ldgsts": queued["resource_contract"][
                "static_ldgsts_bypass_128_instructions"],
            "actual_threads": 384, "actual_dynamic_shared_bytes": 166912,
            "actual_resident_ctas_per_sm_upper_bound": 1,
            "actual_all_path_static_ldgsts": counts["LDGSTS"],
            "decision": "FA2 finite/queued LDGSTS service is outside its measured residency, launch and static issue domain; no rate is imported into FA3 Forward.",
        },
        "conditional_benefit_upper_bound": {
            "saved_partial_q729_operator_abs_error_ns": abs(gap_ns),
            "if_36_identical_layers_only_ms": 36 * abs(gap_ns) / 1e6,
            "scope": "The 36-layer multiplication applies only to the old eager 729-token diagnostic, not the standard graph/async Serving QPS16 trajectory.",
        },
        "reopen_condition": (
            "An independent one-CTA/384-thread/166912-B, two-stage probe must "
            "retain Q TMA, paged K/V cp.async, eight-instruction QK/PV groups, "
            "wait_group 1 and named barrier order, with comparable compiled "
            "resources and non-duration traffic before timing can supply a "
            "portable progress service. It must omit the full Attention output."
        ),
        "limits": ["NCU SASS is static and includes conditional paths; it does not measure dynamic frequency, occupancy timeline or CTA stalls.",
                   "The source-bound partial predictor charges aggregate logical tile reads, not separate Q TMA or paged K/V cp.async services.",
                   "The saved partial predictor is unpublished; this audit does not claim a before/after H2 improvement."],
        "modal_gpu_seconds": 0, "forward_changed": False,
        "reproduce": "python -m evaluation.kernel.h200_fa3_q729_binary_dependency",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sass_counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
