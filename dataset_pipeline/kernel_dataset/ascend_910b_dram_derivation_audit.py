"""Audit the 910B DRAM columns a capture implies against the published ones.

The canonical 910B FlashAttention entries carry DRAM read and write bytes that
this package's own aggregation no longer reproduces.  This audit re-reads a
``--profile-run`` capture, reports the per-domain msprof counters it rests on,
and evaluates two derivations side by side:

``per_domain``
    what :func:`_aggregate_msprof_memory_rows` computes today -- every core
    row's traffic scaled by that row's own hit rate, summed over AIC and AIV.

``single_domain``
    the whole counted stream scaled by one domain's hit rate, which is what the
    published values reproduce.

Nothing here writes a dataset.  It reads a staging capture and the canonical
file and prints the comparison, so a reviewer can redo the arithmetic by hand
from the same four raw numbers per domain.
"""

from __future__ import annotations

import glob
import io
import json
import os

import zstandard

from dataset_pipeline.kernel_dataset.ascend_910b import (
    _aggregate_msprof_memory_rows,
    _read_csv_rows,
)

MIB = 2 ** 20
_DOMAIN_SUBBLOCK = "cube0"


def _domain_of(l2_row):
    """Return the execution domain a joined Memory/L2Cache row belongs to."""

    if "sub_block_id" not in l2_row:
        raise KeyError("msprof L2Cache row has no sub_block_id")
    return "aic" if l2_row["sub_block_id"] == _DOMAIN_SUBBLOCK else "aiv"


def _number(row, column):
    if column not in row:
        raise KeyError(f"msprof row has no column {column!r}")
    text = row[column].strip()
    if text in ("", "N/A", "NA"):
        return 0.0
    return float(text)


def domain_counters(memory_rows, l2_rows):
    """Return per-domain read/write KB and volume-weighted hit rates."""

    totals = {
        domain: {"read_kb": 0.0, "write_kb": 0.0,
                 "read_miss_kb": 0.0, "write_miss_kb": 0.0, "rows": 0}
        for domain in ("aic", "aiv")
    }
    for l2_row, memory_row in zip(l2_rows, memory_rows, strict=True):
        domain = _domain_of(l2_row)
        entry = totals[domain]
        read_kb = _number(memory_row, "read_main_memory_datas(KB)")
        write_kb = _number(memory_row, "write_main_memory_datas(KB)")
        read_hit = _number(l2_row, f"{domain}_read_hit_rate(%)")
        write_hit = _number(l2_row, f"{domain}_write_hit_rate(%)")
        entry["read_kb"] += read_kb
        entry["write_kb"] += write_kb
        entry["read_miss_kb"] += read_kb * (1.0 - read_hit / 100.0)
        entry["write_miss_kb"] += write_kb * (1.0 - write_hit / 100.0)
        entry["rows"] += 1
    for entry in totals.values():
        entry["read_hit_pct"] = (
            100.0 * (1.0 - entry["read_miss_kb"] / entry["read_kb"])
            if entry["read_kb"] > 0.0 else 0.0)
        entry["write_hit_pct"] = (
            100.0 * (1.0 - entry["write_miss_kb"] / entry["write_kb"])
            if entry["write_kb"] > 0.0 else 0.0)
    return totals


def derivations(totals):
    """Return the two candidate DRAM derivations, in bytes."""

    per_domain_read = sum(t["read_miss_kb"] for t in totals.values()) * 1024.0
    per_domain_write = sum(t["write_miss_kb"] for t in totals.values()) * 1024.0
    read_kb = sum(t["read_kb"] for t in totals.values())
    write_kb = sum(t["write_kb"] for t in totals.values())
    aic = totals["aic"]
    single_read = read_kb * (1.0 - aic["read_hit_pct"] / 100.0) * 1024.0
    single_write = write_kb * (1.0 - aic["write_hit_pct"] / 100.0) * 1024.0
    return {
        "per_domain_read_bytes": per_domain_read,
        "per_domain_write_bytes": per_domain_write,
        "single_domain_read_bytes": single_read,
        "single_domain_write_bytes": single_write,
    }


def load_canonical(dataset_path):
    """Return the canonical FlashAttention entries keyed by case id."""

    with open(dataset_path, "rb") as handle:
        stream = zstandard.ZstdDecompressor().stream_reader(handle)
        return json.load(io.TextIOWrapper(stream, "utf-8"))


def capture_cases(profile_run_dir):
    """Yield (case_id, memory_rows, l2_rows, op_type, block_dim) per capture."""

    traces = os.path.join(profile_run_dir, "flash_attn_traces")
    if not os.path.isdir(traces):
        raise FileNotFoundError(f"no flash_attn_traces under {profile_run_dir}")
    for case_id in sorted(os.listdir(traces)):
        pattern = os.path.join(traces, case_id, "OPPROF_*", "*mix_aic", "0")
        matches = sorted(glob.glob(pattern))
        if not matches:
            continue
        memory = glob.glob(os.path.join(matches[0], "Memory*.csv"))
        l2 = glob.glob(os.path.join(matches[0], "L2Cache*.csv"))
        if not memory or not l2:
            continue
        yield case_id, _read_csv_rows(memory[0]), _read_csv_rows(l2[0])


def audit(profile_run_dir, dataset_path):
    """Return one audit record per captured case.

    The launch geometry is taken from the capture's own row counts, so a run
    that mixes 22-core and 24-core launches audits in one pass.
    """

    canonical = load_canonical(dataset_path)
    records = []
    for case_id, memory_rows, l2_rows in capture_cases(profile_run_dir):
        totals = domain_counters(memory_rows, l2_rows)
        candidates = derivations(totals)
        aggregated = _aggregate_msprof_memory_rows(
            memory_rows, l2_rows, "mix",
            totals["aic"]["rows"], totals["aiv"]["rows"])
        if case_id in canonical and "kernels" in canonical[case_id]:
            kernels = canonical[case_id]["kernels"]
            published_read = sum(
                float(kernel["Ascend"]["dram_read_bytes"])
                for kernel in kernels)
            # The L2 columns are the counted request streams.  A capture that
            # reproduces the published ones is the same measurement, which is
            # what lets the DRAM difference be attributed to the derivation.
            published_l2_read = sum(
                float(kernel["Ascend"]["raw"]["read_main_memory_datas(KB)"])
                for kernel in kernels) * 1024.0
            published_l2_write = sum(
                float(kernel["Ascend"]["raw"]["write_main_memory_datas(KB)"])
                for kernel in kernels) * 1024.0
        else:
            published_read = None
            published_l2_read = None
            published_l2_write = None
        records.append({
            "case_id": case_id,
            "domains": totals,
            "candidates": candidates,
            "pipeline_read_bytes": aggregated["read_bytes"],
            "pipeline_write_bytes": aggregated["write_bytes"],
            "pipeline_l2_read_bytes": aggregated["l2_read_bytes"],
            "pipeline_l2_write_bytes": aggregated["l2_write_bytes"],
            "published_dram_read_bytes": published_read,
            "published_l2_read_bytes": published_l2_read,
            "published_l2_write_bytes": published_l2_write,
        })
    return records


def render(records):
    """Return the reviewer-facing table as text."""

    lines = []
    header = (
        f"{'case':44s} {'dom':>4s} "
        f"{'read_KB':>12s} {'write_KB':>11s} {'read_hit%':>9s} "
        f"{'write_hit%':>10s}")
    lines.append(header)
    lines.append("-" * len(header))
    for record in records:
        for domain in ("aic", "aiv"):
            entry = record["domains"][domain]
            lines.append(
                f"{record['case_id'][:44]:44s} {domain:>4s} "
                f"{entry['read_kb']:12.1f} {entry['write_kb']:11.1f} "
                f"{entry['read_hit_pct']:9.2f} {entry['write_hit_pct']:10.2f}")
    lines.append("")
    l2head = (
        f"{'case':44s} {'l2_rd_new':>12s} {'l2_rd_pub':>12s} {'rd_delta':>10s} "
        f"{'l2_wr_new':>12s} {'l2_wr_pub':>12s} {'wr_delta':>10s}")
    lines.append(l2head)
    lines.append("-" * len(l2head))
    for record in records:
        pub_r = record["published_l2_read_bytes"]
        pub_w = record["published_l2_write_bytes"]
        if pub_r is None:
            continue
        new_r = record["pipeline_l2_read_bytes"]
        new_w = record["pipeline_l2_write_bytes"]
        lines.append(
            f"{record['case_id'][:44]:44s} {new_r / MIB:12.2f} "
            f"{pub_r / MIB:12.2f} {(new_r - pub_r) / MIB:10.4f} "
            f"{new_w / MIB:12.2f} {pub_w / MIB:12.2f} "
            f"{(new_w - pub_w) / MIB:10.4f}")
    lines.append("")
    head = (
        f"{'case':44s} {'per_dom_rd':>11s} {'per_dom_wr':>11s} "
        f"{'1dom_rd':>11s} {'1dom_wr':>11s} {'canon_rd':>11s}")
    lines.append(head)
    lines.append("-" * len(head))
    for record in records:
        candidates = record["candidates"]
        published = record["published_dram_read_bytes"]
        published_text = (
            "         --" if published is None
            else f"{published / MIB:11.2f}")
        lines.append(
            f"{record['case_id'][:44]:44s} "
            f"{candidates['per_domain_read_bytes'] / MIB:11.2f} "
            f"{candidates['per_domain_write_bytes'] / MIB:11.2f} "
            f"{candidates['single_domain_read_bytes'] / MIB:11.2f} "
            f"{candidates['single_domain_write_bytes'] / MIB:11.2f} "
            f"{published_text}")
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run-dir", required=True)
    parser.add_argument("--canonical-dataset", required=True)
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)
    records = audit(args.profile_run_dir, args.canonical_dataset)
    print(render(records))
    if args.json_output is not None:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=1, sort_keys=True)
    return 0
