#!/usr/bin/env python3
"""Derive the WP1 H200 ``*_sm90.json`` Triton fused-MoE references.

Input: the local copy of the H200 campaign raw directory written by
``fused_moe_triton_wp1_campaign.py run`` (per capture ``result.json`` with
``DONE``/``ACCEPTED`` markers; for the hard-coded BM16 W1 source runner a
``raw.csv``) and the capture plan rebuilt by
``fused_moe_triton_wp1_campaign.build_plan``.

Derivation per A800 reference (the A800 selection rule is applied unchanged):
  - process value: the runner's ``logical_operand_capacity_bytes_per_s``
    (logical bytes / median of the process's CUDA-event trials); for the
    hard-coded BM16 W1 source runner the same W2R logical-byte formula over
    the per-case median of 9 trials;
  - cell value: median over the fresh processes of one (role, geometry,
    depth, GPU) cell; cells with one process use that process;
  - service value: minimum cell value over every cell of the service
    ("minimum fresh-process cell median across ... devices 2 and 6" becomes
    "... H200 GPUs 0 and 3").
Each H200 reference keeps the A800 reference's keys (values replaced, device
keys 2/6 renamed 0/3) and adds ``h200_capture_provenance``.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fused_moe_triton_wp1_campaign as campaign  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
A800_REFERENCES = ROOT / "microbench/nvidia_a800/references"
H200_REFERENCES = ROOT / "microbench/nvidia_h200/references"
FUSED_MOE = ROOT / "microbench/nvidia_h200/operators/fused_moe"
RUNNER_PATH = {
    "tri": campaign.TRI, "w2r": campaign.W2R, "w1r": campaign.W1R}
KERNEL_MODULE = "microbench/nvidia_h200/operators/fused_moe/vllm_0_11_fused_moe_kernel.py"
VLLM_SOURCE = str(
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
VLLM_SOURCE_SHA256 = (
    "8be8477a653894194ef5c5095afc287b0409a51d980590a36dd43c8471f6740e")
REMOTE_ROOT = os.path.join(
    os.environ["MIDLAYER_H200_PUBLIC_ROOT"], "h200_microbench/wp1_moe_triton")
DEVICE_RENAME = {2: 0, 6: 3}
TEXT_RENAMES = (
    ("idle devices 2 and 6", "idle H200 GPUs 0 and 3"),
    ("devices 2 and 6", "H200 GPUs 0 and 3"),
    ("devices 2/6", "H200 GPUs 0/3"),
    ("2 complete source waves * 108 SM * group_size_m 8",
     "2 complete source waves * 132 SM * group_size_m 8"),
    ("108-SM", "132-SM"),
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value):
    for old, new in TEXT_RENAMES:
        value = value.replace(old, new)
    return value


def _rename_text(node):
    if isinstance(node, dict):
        return {key: _rename_text(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_rename_text(value) for value in node]
    if isinstance(node, str):
        return _text(node)
    return node


class Capture:
    """One accepted fresh process of the H200 campaign."""

    def __init__(self, item, raw_dir, run_id):
        self.item = item
        base = raw_dir / item["reference"] / item["name"]
        done = (base / "DONE").read_text().strip()
        accepted_dir = raw_dir / item["reference"] / Path(done).name
        if not (accepted_dir / "ACCEPTED").exists():
            raise RuntimeError(f"{accepted_dir} is not an accepted capture")
        self.directory = accepted_dir
        self.attempts = sorted(
            path for path in (raw_dir / item["reference"]).iterdir()
            if path.is_dir() and (path.name == item["name"]
                                  or path.name.startswith(
                                      f"{item['name']}_attempt")))
        self.rejected = sum(
            1 for path in self.attempts
            if not (path / "ACCEPTED").exists() and path != accepted_dir
            and ((path / "result.json").exists()
                 or (path / "raw.csv").exists()))
        self.relative = (Path("profiler_output/microbench/nvidia_h200")
                         / run_id / item["reference"] / accepted_dir.name)
        self.gpu = item["gpu"]
        self.cell = item["cell"]
        if item["runner"] == "w1r":
            self.path = accepted_dir / "raw.csv"
            self.result = None
            return
        self.path = accepted_dir / "result.json"
        payload = json.loads(self.path.read_text())
        self.payload = payload
        self.result = payload["result"]
        if not self.result["clock_gate"]["accepted"]:
            raise RuntimeError(f"{self.path} failed its clock gate")
        if payload["vllm_source_sha256"] != VLLM_SOURCE_SHA256:
            raise RuntimeError(f"{self.path} ran another vLLM source")

    @property
    def capacity(self):
        return self.result["logical_operand_capacity_bytes_per_s"]

    @property
    def trial_capacity_range_over_median(self):
        if "duration_ms" in self.result:
            values = [self.result["logical_operand_bytes"] / (ms * 1e-3)
                      for ms in self.result["duration_ms"]]
        else:
            values = [self.result["logical_operand_bytes"] / (ns * 1e-9)
                      for ns in self.result["duration_ns"]]
        return (max(values) - min(values)) / statistics.median(values)

    @property
    def role(self):
        name = self.item["name"]
        for role in ("w1", "w2"):
            if f"_{role}_" in f"_{name}_" or name.startswith(f"{role}_"):
                return role
        raise ValueError(f"capture {name} has no role")

    def reference_path(self):
        return str(self.relative / self.path.name)

    def sha256(self):
        return _sha256(self.path)


def _median_duration_ns(capture):
    if "median_duration_ns" in capture.result:
        return capture.result["median_duration_ns"]
    return capture.result["median_duration_ms"] * 1e6


class Cell:
    def __init__(self, captures):
        self.captures = captures
        values = [capture.capacity for capture in captures]
        self.value = statistics.median(values)
        self.first = captures[0]
        if len(captures) > 1:
            self.range_over_median = (max(values) - min(values)) / self.value
        else:
            self.range_over_median = (
                captures[0].trial_capacity_range_over_median)
        self.median_duration_ns = statistics.median(
            _median_duration_ns(capture) for capture in captures)


def _cells(captures):
    return {key: Cell([capture for capture in captures
                       if capture.cell == key])
            for key in dict.fromkeys(capture.cell for capture in captures)}


def _group_by_service(reference, captures):
    services = dict.fromkeys(
        _service_of(reference, capture) for capture in captures)
    return {service: [capture for capture in captures
                      if _service_of(reference, capture) == service]
            for service in services}




def _clock_stats(captures):
    windows = [capture.result["clock_gate"] for capture in captures
               if capture.result is not None]
    minima = [window["window_clock_mhz_min"] for window in windows]
    return {
        "gate": ("every nvidia-smi 50-ms sample inside [first timed launch - "
                 "0.1 s, last timed launch + 0.1 s] reads >= 1980 MHz; a "
                 "rejected process is repeated in a fresh process"),
        "accepted_captures": len(captures),
        "rejected_captures": sum(capture.rejected for capture in captures),
        "window_samples_total": sum(
            window["window_samples"] for window in windows),
        "window_clock_mhz_min": min(minima) if minima else None,
        "window_clock_mhz_max": max(
            window["window_clock_mhz_max"] for window in windows)
        if windows else None,
        "measured_sm_clock_mhz": 1980,
    }


def _raw_tree_digest(captures):
    records = sorted(
        f"{capture.sha256()}  {capture.reference_path()}"
        for capture in captures)
    return hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest()


def _common(reference, a800, captures, context):
    runner = RUNNER_PATH[captures[0].item["runner"]]
    runner_sha = _sha256(ROOT / runner)
    out = _rename_text(copy.deepcopy(a800))
    replacements = {
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "source_baseline_commit": context["baseline_commit"],
        "runner": runner, "source": runner,
        "runner_sha256": runner_sha, "source_sha256": runner_sha,
        "capture_runner_sha256": runner_sha,
        "reproduction_runner_sha256": runner_sha,
        "reproduction_change": (
            "The H200 capture and reproduction runner is the single H200 "
            "owner copy named in h200_capture_provenance."),
        "vllm_source": VLLM_SOURCE,
        "vllm_source_sha256": VLLM_SOURCE_SHA256,
        "software_revision": "vllm_0.11.0_source_triton_3.6_sm90",
        "production_or_evaluation_duration_consumed": False,
    }
    for key, value in replacements.items():
        if key in out:
            out[key] = value
    for key in ("schema", "schema_version"):
        if key in out and isinstance(out[key], str):
            out[key] = out[key].replace("nvidia_a800", "nvidia_h200").replace(
                "a800", "h200")
    if "static_selection" in out:
        selection = out["static_selection"]
        if "minimum_exclusive_maximum_m_tiles" in selection:
            selection["minimum_exclusive_maximum_m_tiles"] = (
                campaign.DEEP_GATE_H200)
    if "capture_root" in out:
        out["capture_root"] = str(captures[0].relative.parent)
    kernel_sha = _sha256(ROOT / KERNEL_MODULE)
    out["h200_capture_provenance"] = {
        "a800_reference": f"microbench/nvidia_a800/references/{reference}_sm80.json",
        "runner": runner,
        "runner_sha256": runner_sha,
        "kernel_module": KERNEL_MODULE,
        "kernel_module_sha256": kernel_sha,
        "kernel_source": (
            "byte-identical vLLM 0.11 fused_moe_kernel / "
            "invoke_fused_moe_kernel / moe_align_block_size segments "
            f"(vLLM 0.11 fused_moe.py sha256 {VLLM_SOURCE_SHA256}) compiled "
            "by Triton 3.6.0; sort op = vLLM 0.24.0 _moe_C"),
        "software": captures[0].payload["software"]
        if captures[0].result is not None else context["w1r_software"],
        "campaign": "microbench/nvidia_h200/operators/fused_moe/fused_moe_triton_wp1_campaign.py",
        "campaign_sha256": _sha256(FUSED_MOE / "fused_moe_triton_wp1_campaign.py"),
        "derivation": "microbench/nvidia_h200/operators/fused_moe/derive_fused_moe_triton_wp1_references.py",
        "derivation_sha256": _sha256(Path(__file__)),
        "baseline_commit": context["baseline_commit"],
        "raw_dir": str(captures[0].relative.parent),
        "remote_raw_dir": (f"{REMOTE_ROOT}/repo/profiler_output/microbench/"
                           f"nvidia_h200/{context['run_id']}/{reference}"),
        "raw_tree_sha256": _raw_tree_digest(captures),
        "device_mapping": "A800 device 2 -> H200 GPU 0, A800 device 6 -> H200 GPU 3",
        "clock": _clock_stats(captures),
        "deep_gate_m_tiles": {"a800": campaign.DEEP_GATE_A800,
                              "h200": campaign.DEEP_GATE_H200},
    }
    if ("static_selection" in a800
            and "minimum_exclusive_maximum_m_tiles" in a800["static_selection"]):
        covered = a800["static_selection"]["covered_canonical_maximum_m_tiles"]
        below = [depth for depth in covered
                 if depth <= campaign.DEEP_GATE_H200]
        out["h200_capture_provenance"]["deep_gate_covered_depths"] = (
            f"measured depths all exceed {campaign.DEEP_GATE_H200}; covered "
            f"canonical depths {below} lie in ({campaign.DEEP_GATE_A800}, "
            f"{campaign.DEEP_GATE_H200}] and no longer pass the H200 gate, "
            "so the selector must not route them to this service on H200"
            if below else
            f"every covered canonical depth exceeds {campaign.DEEP_GATE_H200}")
    if reference in campaign.DEEP_GATE_RECHOSEN:
        out["h200_capture_provenance"]["deep_gate_rechosen"] = (
            campaign.DEEP_GATE_RECHOSEN[reference])
    if reference in campaign.INFERRED:
        out["h200_capture_provenance"]["inferred_probe_arguments"] = (
            campaign.INFERRED[reference])
    return out


EXTRA_FIELDS = {
    "device_id": lambda c: c.gpu,
    "role": lambda c: c.role,
    "n": lambda c: c.result["n"],
    "tokens": lambda c: c.result["tokens"],
    "route_seed": lambda c: c.result["route_seed"],
    "k_tiles": lambda c: c.result["k"] // 32,
    "n_tiles": lambda c: c.result["n"] // 64,
    "active_m_tiles": lambda c: c.result["active_m_tiles"],
    "maximum_m_tiles": lambda c: c.result["maximum_m_tiles"],
    "logical_operand_capacity_bytes_per_s": lambda c: c.capacity,
    "logical_byte_capacity_per_s": lambda c: c.capacity,
    "capacity_bytes_per_s": lambda c: c.capacity,
}


def _capture_entries(template, captures):
    entries = []
    for capture in captures:
        entry = {}
        for key in template:
            if key == "path":
                entry[key] = capture.reference_path()
            elif key.endswith("sha256"):
                entry[key] = capture.sha256()
            else:
                entry[key] = EXTRA_FIELDS[key](capture)
        entries.append(entry)
    return entries


def _spread_entries(template, cells):
    entries = []
    for cell in cells.values():
        entry = {}
        for key in template:
            if key == "median_logical_byte_capacity_per_s":
                entry[key] = cell.value
            elif key == "range_over_median":
                entry[key] = cell.range_over_median
            else:
                entry[key] = EXTRA_FIELDS[key](cell.first)
        entries.append(entry)
    return entries


def _union_keys(entries):
    keys = []
    for entry in entries:
        for key in entry:
            if key not in keys:
                keys.append(key)
    return keys


def _service_of(reference, capture):
    """Service key of a capture, following the A800 reference layout."""
    result = capture.result
    if reference == "bm64_triton_source_depth":
        return {2048: "68_active_72_max", 4096: "132_active_136_max"}[
            result["tokens"]]
    if reference == "bm64_triton_e8_h4096_exact_w2_topk1_source_order":
        return f"tokens{result['tokens']}"
    if reference == "bm16_sparse_w2_source_depth":
        return capture.item["name"].split("_d")[0]
    if reference in ("bm64_triton_e256_h7168_deep_source_depth",
                     "bm64_triton_e256_h7168_exact_source_order",
                     "bm64_triton_e8_h6144_8_active_9_max_source_depth",
                     "bm64_triton_e128_h2880_1084_active_1150_exact_source_order",
                     "bm64_e32_h2880_32_active_36_max_source_depth",
                     "bm64_e32_h2880_143_active_160_max_source_depth",
                     "bm64_e512_1001_active_1144_max_source_depth",
                     "bm64_e128_574_active_638_max_source_depth",
                     "bm64_triton_e256_h7168_mid_workspace_source_depth",
                     "bm16_e128_177_active_248_max_source_depth",
                     "bm16_e128_88_active_128_max_source_depth",
                     "bm16_e8_7_active_8_max_source_depth",
                     "bm16_e8_h6144_7_active_8_max_source_depth"):
        return capture.role
    return None


def _minimum(cells):
    return min(cell.value for cell in cells.values())


def _family_a(reference, a800, captures, context):
    out = _common(reference, a800, captures, context)
    if "captures" in a800:
        out["captures"] = _capture_entries(
            _union_keys(a800["captures"]), captures)
    by_service = _group_by_service(reference, captures)
    cells = {service: _cells(members)
             for service, members in by_service.items()}
    all_cells = _cells(captures)
    if "fresh_process_cell_spread" in a800:
        out["fresh_process_cell_spread"] = _spread_entries(
            _union_keys(a800["fresh_process_cell_spread"]), all_cells)
    if "selected_service" in a800:
        if set(by_service) != {None}:
            raise AssertionError(f"{reference} has one service")
        value = _minimum(cells[None])
        if isinstance(a800["selected_service"], dict):
            out["selected_service"]["logical_byte_capacity_per_s"] = value
        else:
            out["selected_service"] = value
    if "selected_services" in a800:
        selected = out["selected_services"]
        if reference == "bm64_triton_e256_h7168_mid_workspace_source_depth":
            for service in ("w1", "w2"):
                selected[service] = _minimum(cells[service])
        else:
            if set(selected) != set(cells):
                raise AssertionError(
                    f"{reference} services {set(selected)} != {set(cells)}")
            for service, service_cells in cells.items():
                selected[service]["logical_byte_capacity_per_s"] = _minimum(
                    service_cells)
                if "relative_range" in selected[service]:
                    values = [cell.value for cell in service_cells.values()]
                    selected[service]["relative_range"] = (
                        (max(values) - min(values)) / statistics.median(values))
    if "fresh_process_cell_medians" in a800:
        medians = {}
        for service in ("w1", "w2"):
            rows = []
            for cell in cells[service].values():
                row = {"device_id": cell.first.gpu,
                       "n": cell.first.result["n"],
                       "logical_byte_capacity_per_s": cell.value}
                if cell.first.result["route_seed"] is not None:
                    row = {"device_id": cell.first.gpu,
                           "route_seed": cell.first.result["route_seed"],
                           "n": cell.first.result["n"],
                           "logical_byte_capacity_per_s": cell.value}
                rows.append(row)
            medians[service] = rows
        out["fresh_process_cell_medians"] = medians
    if reference == "bm64_e256_h7168_5_3_source_group":
        ranges = []
        for n in (4032, 4160):
            values = [capture.capacity for capture in captures
                      if capture.result["n"] == n]
            ranges.append((max(values) - min(values))
                          / statistics.median(values) * 100)
        out["selection_rule"] = (
            "minimum capacity across two adjacent non-production N endpoints "
            "and three fresh processes per endpoint; endpoint ranges divided "
            f"by their medians are {ranges[0]:.6f}% and {ranges[1]:.6f}%")
    return out


def _family_b(reference, a800, captures, context):
    out = _common(reference, a800, captures, context)
    runner = RUNNER_PATH[captures[0].item["runner"]]
    runner_sha = _sha256(ROOT / runner)
    by_service = _group_by_service(reference, captures)
    selected = out["selected_services"]
    if set(selected) != set(by_service):
        raise AssertionError(
            f"{reference} services {set(selected)} != {set(by_service)}")
    maximum_range_pct = 0.0
    for service, members in by_service.items():
        cells = _cells(members)
        selected[service]["logical_byte_capacity_per_s"] = _minimum(cells)
        for key in ("capture_medians_ns", "capture_medians_bytes_per_s"):
            if key in selected[service]:
                selected[service][key] = {
                    f"device{cell.first.gpu}_n{cell.first.result['n']}": (
                        cell.median_duration_ns if key.endswith("_ns")
                        else cell.value)
                    for cell in cells.values()}
        for cell in cells.values():
            values = [capture.capacity for capture in cell.captures]
            maximum_range_pct = max(
                maximum_range_pct,
                (max(values) - min(values)) / statistics.mean(values) * 100)
    capture_block = out["capture"]
    for key in list(capture_block):
        if key in ("baseline_commit",):
            capture_block[key] = context["baseline_commit"]
        elif key == "baseline_commits":
            capture_block[key] = [context["baseline_commit"]]
        elif key == "runner_path":
            capture_block[key] = runner
        elif key in ("runner_sha256", "runner_sha256_at_capture"):
            capture_block[key] = runner_sha
        elif key == "physical_devices":
            capture_block[key] = [0, 3]
        elif key in ("maximum_capture_capacity_range_over_mean_pct",
                     "maximum_cell_range_over_mean_pct"):
            capture_block[key] = maximum_range_pct
        elif key == "raw_root":
            capture_block[key] = {
                "path": str(captures[0].relative.parent),
                "sha256": _raw_tree_digest(captures)}
        elif key == "raw_roots":
            capture_block[key] = [{
                "path": str(captures[0].relative.parent),
                "sha256": _raw_tree_digest(captures)}]
        elif key == "raw_capture_records":
            capture_block[key] = sorted(
                str(Path(capture.reference_path()).relative_to(
                    captures[0].relative.parent))
                for capture in captures
                if reference != "bm16_e8_h6144_7_active_8_max_source_depth"
                or capture.role == "w1")
        elif key == "results":
            capture_block[key] = _capture_entries(
                _union_keys(a800["capture"]["results"]), captures)
    if "w2_source_topk_correction_capture" in out:
        block = out["w2_source_topk_correction_capture"]
        w2 = [capture for capture in captures if capture.role == "w2"]
        block["baseline_commit"] = context["baseline_commit"]
        block["runner_path"] = runner
        block["runner_sha256_at_capture"] = runner_sha
        block["physical_devices"] = [0, 3]
        block["raw_root"] = {"path": str(captures[0].relative.parent),
                             "sha256": _raw_tree_digest(w2)}
        block["raw_capture_records"] = sorted(
            str(Path(capture.reference_path()).relative_to(
                captures[0].relative.parent)) for capture in w2)
        out["h200_capture_provenance"]["w2_source_topk"] = (
            "H200 measured W2 with source_top_k 1 directly (the A800 "
            "correction capture's semantics); capture and correction blocks "
            "therefore share one H200 raw root.")
    return out


def _w1r_rows(capture):
    """Per-case medians from the hard-coded BM16 W1 source runner."""
    summaries = []
    clocks = []
    for line in capture.path.read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "summary":
            summaries.append(fields[1:])
        elif fields[0] == "clock" and fields[1] != "tokens":
            clocks.append(fields[1:])
    rows = []
    for fields in summaries:
        (tokens, experts, topk, active_experts, k, n, active_m_tiles,
         maximum_m_tiles, grid) = (int(value) for value in fields[:9])
        median_ns = float(fields[9])
        n_tiles = math.ceil(n / 32)
        logical_bytes = (tokens * topk * k * 2 * n_tiles
                         + active_m_tiles * n_tiles * k * 32 * 2)
        rows.append({
            "tokens": tokens, "experts": experts, "topk": topk,
            "active_experts": active_experts, "k": k, "n": n,
            "active_m_tiles": active_m_tiles,
            "maximum_m_tiles": maximum_m_tiles, "grid": grid,
            "median_duration_ns": median_ns,
            "logical_operand_bytes": logical_bytes,
            "logical_operand_capacity_bytes_per_s": (
                logical_bytes / (median_ns * 1e-9))})
    return rows, clocks


def _w1r_reference(reference, a800, captures, context):
    out = _common(reference, a800, captures, context)
    per_class = {"31_active": [], "53_active": []}
    clock_rows = []
    for capture in captures:
        rows, clocks = _w1r_rows(capture)
        clock_rows.extend(clocks)
        if not clocks or any(row[-1] != "1" for row in clocks):
            raise RuntimeError(f"{capture.path} has a rejected case")
        for row in rows:
            if (row["active_experts"] in (31, 53)
                    and row["k"] in (2816, 2944)):
                per_class[f"{row['active_experts']}_active"].append(
                    dict(row, gpu=capture.gpu))
    for service, rows in per_class.items():
        if len(rows) != 8:
            raise AssertionError(f"{service} has {len(rows)} case rows")
        out["selected_services"][service]["logical_byte_capacity_per_s"] = min(
            row["logical_operand_capacity_bytes_per_s"] for row in rows)
    out["capture"]["baseline_commit"] = context["baseline_commit"]
    out["capture"]["runner_path"] = RUNNER_PATH["w1r"]
    out["capture"]["runner_sha256"] = _sha256(ROOT / RUNNER_PATH["w1r"])
    out["capture"]["physical_devices"] = [0, 3]
    out["capture"]["raw_paths"] = [capture.reference_path()
                                   for capture in captures]
    out["capture"]["raw_sha256"] = [capture.sha256() for capture in captures]
    provenance = out["h200_capture_provenance"]
    provenance["logical_bytes"] = (
        "routed_rows*K*2*ceil(N/32) + active_m_tiles*ceil(N/32)*K*32*2 over "
        "the per-case median of 9 CUDA-event trials (the BM16 W2R formula)")
    provenance["source_depth_case_rows"] = per_class
    provenance["clock"] = {
        "gate": ("per case: every nvidia-smi 50-ms sample inside [first timed "
                 "launch - 0.1 s, last timed launch + 0.1 s] reads >= 1980 "
                 "MHz"),
        "accepted_captures": len(captures),
        "rejected_captures": sum(capture.rejected for capture in captures),
        "accepted_case_windows": len(clock_rows),
        "window_clock_mhz_min": min(int(row[-2]) for row in clock_rows),
        "measured_sm_clock_mhz": 1980,
    }
    return out


def _device_neutral(node):
    """Map device-indexed keys (A800 device2/6, H200 device0/3) to device*."""
    return {re.sub(r"^device\d+_", "device*_", key): value
            for key, value in node.items()}


def _key_diff(a800, h200, path=""):
    """Recursive dict-key differences (lists compared by first element).

    Device-indexed map keys (``device2_n1504`` on A800, ``device0_n1504`` on
    H200) are compared with the device index masked.
    """
    diffs = []
    if isinstance(a800, dict) and isinstance(h200, dict):
        a800 = _device_neutral(a800)
        h200 = _device_neutral(h200)
        for key in set(a800) | set(h200):
            if key not in h200:
                diffs.append(f"missing {path}/{key}")
            elif key not in a800:
                diffs.append(f"added {path}/{key}")
            else:
                diffs.extend(_key_diff(a800[key], h200[key], f"{path}/{key}"))
    elif (isinstance(a800, list) and isinstance(h200, list) and a800 and h200
          and isinstance(a800[0], dict)):
        diffs.extend(_key_diff(a800[0], h200[0], f"{path}[0]"))
    return diffs


def _operand_service_reference(directory, baseline_commit):
    """H200 reference for ``grouped_fused_moe_operand_read``.

    The A800 row cites the raw summary
    ``profiler_output/microbench/nvidia_a800/fused_moe_operand_820defa_20260830/summary.json``
    (no A800 reference JSON exists); the H200 value is the same statistic,
    the median of 15 ``logical_bandwidth_gb_s`` samples of case
    ``operand_bm16_bn32_bk64``, converted to bytes/s.
    """
    summary_path = directory / "summary.json"
    manifest_path = directory / "manifest.json"
    raw_path = directory / "raw.csv"
    summaries = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    relative = Path("profiler_output/microbench/nvidia_h200") / directory.name
    selected = {
        item["case_name"]: {
            "logical_byte_capacity_per_s": (
                item["median_logical_bandwidth_gb_s"] * 1e9),
            "selection_rule": (
                "median of 3 fresh-process trials x 5 CUDA-event samples of "
                "logical_bandwidth_gb_s (the A800 summary statistic), x 1e9"),
            "tile_m": item["tile_m"], "tile_n": item["tile_n"],
            "tile_k": item["tile_k"], "group_size_m": item["group_size_m"],
            "relative_range_pct": item["relative_range_pct"]}
        for item in summaries}
    return {
        "schema": "nvidia_h200_grouped_fused_moe_operand_read_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "a800_evidence": (
            "profiler_output/microbench/nvidia_a800/"
            "fused_moe_operand_820defa_20260830/summary.json"
            "#operand_bm16_bn32_bk64 (2427259310000.0 B/s; raw not retained "
            "locally)"),
        "source": manifest["source_path"],
        "source_sha256": manifest["source_sha256"],
        "runner": manifest["runner_path"],
        "runner_sha256": manifest["runner_sha256"],
        "runner_procedure_origin": (
            "A800 fused_moe_operand_service_runner.py at commit f4476b33b, "
            "the version that produced the A800 summary; the current A800 "
            "runner is fixed to CASE_ID=1 with an active-row sweep"),
        "source_baseline_commit": baseline_commit,
        "build": manifest["build"],
        "device": manifest["device_before"],
        "summaries": summaries,
        "selected_services": selected,
        "clock": manifest["clock"],
        "captures": {"raw": {"path": str(relative / "raw.csv"),
                             "sha256": _sha256(raw_path)},
                     "summary": {"path": str(relative / "summary.json"),
                                 "sha256": _sha256(summary_path)},
                     "manifest": {"path": str(relative / "manifest.json"),
                                  "sha256": _sha256(manifest_path)}},
        "remote_raw_dir": (f"{REMOTE_ROOT}/repo/profiler_output/microbench/"
                           f"nvidia_h200/{directory.name}"),
        "mechanism": (
            "exact BF16 A/B grouped-MoE tile transfer into shared memory with "
            "a checksum consumer; no tensor-core operation, output matrix, "
            "routing pipeline, or production operator timing"),
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--operand-service-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--reference", nargs="*")
    arguments = parser.parse_args()
    raw_dir = arguments.raw_dir.resolve()
    run_id = raw_dir.name
    plan = campaign.build_plan(ROOT)
    references = sorted({item["reference"] for item in plan})
    if arguments.reference is not None:
        references = [name for name in references
                      if name in arguments.reference]
    context = {"baseline_commit": arguments.baseline_commit, "run_id": run_id,
               "w1r_software": {"torch": "2.11.0+cu129", "triton": "3.6.0",
                                "vllm_package_for_moe_align_op": "0.24.0"}}
    summary = {}
    for reference in references:
        items = [item for item in plan if item["reference"] == reference]
        missing = [item["name"] for item in items if not (
            raw_dir / reference / item["name"] / "DONE").exists()]
        if missing:
            print(f"SKIP {reference}: {len(missing)} captures missing")
            continue
        captures = [Capture(item, raw_dir, run_id) for item in items]
        a800 = json.loads(
            (A800_REFERENCES / f"{reference}_sm80.json").read_text())
        if captures[0].item["runner"] == "w1r":
            h200 = _w1r_reference(reference, a800, captures, context)
        elif "schema_version" in a800:
            h200 = _family_b(reference, a800, captures, context)
        else:
            h200 = _family_a(reference, a800, captures, context)
        h200["h200_capture_provenance"]["derived_at_utc"] = datetime.now(
            timezone.utc).isoformat()
        diffs = [diff for diff in _key_diff(a800, h200)
                 if diff != "added /h200_capture_provenance"]
        if diffs:
            raise AssertionError(f"{reference} key drift: {diffs}")
        target = H200_REFERENCES / f"{reference}_sm90.json"
        target.write_text(json.dumps(h200, indent=2) + "\n")
        summary[reference] = (
            h200["selected_service"] if "selected_service" in h200
            else h200["selected_services"])
        print(f"WROTE {target.relative_to(ROOT)}")
    operand = _operand_service_reference(
        arguments.operand_service_dir.resolve(), arguments.baseline_commit)
    target = H200_REFERENCES / "grouped_fused_moe_operand_read_sm90.json"
    target.write_text(json.dumps(operand, indent=2) + "\n")
    summary["grouped_fused_moe_operand_read"] = operand["selected_services"]
    print(f"WROTE {target.relative_to(ROOT)}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
