"""Summarize independent request samples; never read evaluation timings."""
import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def median_confidence_interval(samples):
    """Distribution-free two-sided95% interval for a continuous median."""
    ordered = sorted(samples)
    n = len(ordered)
    denominator = 2**n
    eligible = [(k, sum(math.comb(n, i) for i in range(k)))
                for k in range(1, n//2+1)
                if 40*sum(math.comb(n, i) for i in range(k)) <= denominator]
    if not eligible:
        raise ValueError("insufficient samples for median confidence")
    rank, tail = eligible[-1]
    return {"lower_ns":ordered[rank-1], "upper_ns":ordered[n-rank],
            "coverage":1-2*tail/denominator, "order_rank":rank,
            "sample_count":n}


def analyze(raw, profile):
    fanout = profile == "coalesced_progress"
    if fanout:
        profile = "direct_precision"
    bm128 = profile == "bm128_progress"
    if bm128:
        profile = "direct_precision"
    warm_only = profile == "short_warm"
    if warm_only:
        profile = "cohort_short"
    if raw["status"] != "complete":
        raise ValueError("incomplete capture")
    if profile == "private32":
        expected_rows, panel, slots, grid, counter_code = 8940, 32768, 3, 264, 3
    elif profile == "one_resident48":
        expected_rows, panel, slots, grid, counter_code = 60, 49152, 4, 132, 8
    elif profile == "finite_rows":
        expected_rows, panel, slots, grid, counter_code = 5544, None, None, None, None
    elif profile == "companion":
        expected_rows, panel, slots, grid, counter_code = 504, None, None, None, None
    elif profile == "direct_progress":
        expected_rows, panel, slots, grid, counter_code = 12768, None, None, None, None
    elif profile == "direct_precision":
        expected_rows, panel, slots, grid, counter_code = 38304, None, None, None, None
    elif profile == "cohort_short":
        expected_rows, panel, slots, grid, counter_code = None, None, None, None, None
    else:
        raise ValueError("unknown independent profile")
    cells = defaultdict(lambda: defaultdict(list))
    for rotation in raw["rotations"]:
        rows = list(csv.DictReader(io.StringIO(rotation["measurement"]["stdout"])))
        if profile == "cohort_short":
            expected_rows = {24:2268, 25:2268, 30:2772}[rotation["profile_code"]]
        if bm128:
            expected_rows = 4158
        if fanout:
            expected_rows = 9450
        if len(rows) != expected_rows:
            raise ValueError("incomplete sweep")
        for row in rows:
            key = tuple(int(row[k]) for k in (
                "panel_bytes", "slots", "threads", "grid", "rounds",
                "row_bytes", "row_stride")) + (row["tier"],)
            if profile in ("companion", "direct_progress", "direct_precision", "cohort_short"):
                key += (int(row["companion_mode"]),)
            if fanout:
                key += (int(row["request_fanout"]),)
            key += (row["mode"],)
            cells[key][rotation["rotation"]].append(row)
    summaries = []
    for key, processes in sorted(cells.items()):
        expected_trials = (63 if profile in ("direct_precision", "cohort_short") else
                           21 if profile in ("finite_rows", "companion", "direct_progress") else 5)
        if set(processes) != {0, 1} or any(len(r) != expected_trials for r in processes.values()):
            raise ValueError("incomplete cell")
        medians, iqrs, clocks, confidence = [], [], [], []
        for process in (0, 1):
            spans = [float(r["body_ns"]) for r in processes[process]]
            median = statistics.median(spans)
            if profile in ("direct_precision", "cohort_short"):
                confidence.append(median_confidence_interval(spans))
            quartiles = statistics.quantiles(spans, n=4, method="inclusive")
            medians.append(median)
            iqrs.append((quartiles[2] - quartiles[0]) / median)
            clocks.append(statistics.median(float(r["weighted_mhz"])
                                            for r in processes[process]))
        summaries.append({"key": key, "median_body_ns": medians,
                          "relative_iqr": iqrs, "weighted_mhz": clocks,
                          "relative_process_spread": max(medians) / min(medians) - 1})
        if profile in ("direct_precision", "cohort_short"):
            summaries[-1]["median_confidence95_by_process"] = confidence
        if profile == "cohort_short":
            for field in ("shared_reservation_bytes", "resident_ctas_per_sm"):
                values = {int(r[field]) for rows in processes.values() for r in rows}
                if len(values) != 1:
                    raise ValueError("resource identity changed within cell")
                summaries[-1][field] = values.pop()
    counters = []
    logical = {1: 16384*128*16, 2: 8192*192*16,
               3: 32768*264*1024, 4: 32768*264*256,
               5: 32768*264*256, 6: 32768*264*256, 8: 49152*132*1024, 10: 4096*320*32, 11: 8192*192*32,
               12: 16384*160*32, 13: 16384*128*16,
               15: 4096*320*32, 16: 4096*320*32+4096,
               17: 4096*320*32+65536,
               19:64*5760*360+5760, 20:64*1536*256+1536,
               21:128*4096*160+4096, 22:64*8192*320+8192,
               26:16384*1272*16+4096,27:16384*1696*6+1536,
               28:49152*132*8,29:49152*132*32,
               31:16384*320*32+4096,32:16384*640*8+1024}
    logical.update({34:32768*1530*32+128*4096,
                    35:32768*2040*12+128*1536,
                    36:32768*4592*32+128*4096})
    logical.update({38:32768*765*32+4096,39:32768*510*12+1536,
                    40:32768*2296*32+4096,41:32768*4592*8+1024,
                    42:32768*9*32+4096,43:32768*4592*32+4096})
    for counter in raw["counters"]:
        lines = counter["measurement"]["stdout"].splitlines()
        begin = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
        records = [r for r in csv.DictReader(lines[begin:]) if r["ID"] == "0"]
        if len(records) != 1:
            raise ValueError("counter launch is not unique")
        row = records[0]
        code = counter["cell_code"]
        counters.append({"cell_code": code, "logical_bytes": logical[code],
                         "dram_read_bytes": int(row["dram__bytes_read.sum"]),
                         "l2_read_sector_bytes": 32*int(row["lts__t_sectors_op_read.sum"]),
                         "registers_per_thread": int(row["launch__registers_per_thread"]),
                         "static_shared_bytes": int(row["launch__shared_mem_per_block_static"]),
                         "dynamic_shared_bytes": int(row["launch__shared_mem_per_block_dynamic"]),
                         "ldgsts_warp_instructions": (int(row["sm__sass_inst_executed_op_ldgsts_cache_bypass.sum"])
                                                       if profile in ("companion", "direct_progress", "direct_precision", "cohort_short") else None)})
    if fanout:
        expected={38:96*1530*32,39:96*2040*12,40:96*4592*32,
                  41:96*9184*8,42:96*132*32,43:96*4592*32}
        if {c["cell_code"] for c in counters} != set(expected):
            raise ValueError("incomplete fanout counters")
        points=[]
        for c in summaries:
            if c["key"][-1] != "copy":
                continue
            p,s,t,g,r,row,stride,tier,a,f,_=c["key"]
            if (p,s,t,row,tier,a)!=(32768,4,256,128,"cold_shared_requests",5) or r!=stride//row or f not in (1,2,4,8,16):
                raise ValueError("coalesced physical scope drift")
            checks={"process_spread_le_3pct":c["relative_process_spread"]<=.03,
                    "sm_clock_within_3pct_nominal":all(abs(v/1980-1)<=.03 for v in c["weighted_mhz"]),
                    "median_confidence95_within_3pct":all(max(m-ci["lower_ns"],ci["upper_ns"]-m)/m<=.03 for m,ci in zip(c["median_body_ns"],c["median_confidence95_by_process"]))}
            logical=p*r*g
            points.append({"key":c["key"][:-1],"checks":checks,"eligible":all(checks.values()),
                "logical_B_bytes":logical,"unique_B_bytes":p*r*((g+f-1)//f),
                "copy_body_ns_by_process":c["median_body_ns"],
                "logical_byte_capacity_per_s":logical/max(c["median_body_ns"])*1e9,
                "median_confidence95_by_process":c["median_confidence95_by_process"]})
        return {"production_or_evaluation_duration_consumed":False,
                "selection":"Slower process complete body median; every published point passes95%medianCI/process/nominalclock3%; no production durations or failed endpoint borrowing",
                "instruction_counts_match":{str(c["cell_code"]):c["ldgsts_warp_instructions"]==expected[c["cell_code"]] for c in counters},
                "counter_dram_logical_within_1pct":{str(c["cell_code"]):abs(c["dram_read_bytes"]/c["logical_bytes"]-1)<=.01 for c in counters},
                "progress_points":points,"counters":counters,"cells":summaries}
    if profile == "cohort_short":
        expected = {26:814080,27:407040,28:101376,29:405504,31:368640,32:184320}
        if warm_only:
            expected = {28:101376,29:405504}
        if {c["cell_code"] for c in counters} != set(expected):
            raise ValueError("incomplete cohort counters")
        points = []
        for c in summaries:
            if c["key"][-1] != "copy":
                continue
            p,s,t,g,r,row,stride,tier,a,_ = c["key"]
            checks = {
                "process_spread_le_3pct":c["relative_process_spread"] <= .03,
                "sm_clock_within_3pct_nominal":all(abs(v/1980-1)<=.03 for v in c["weighted_mhz"]),
                "median_confidence95_within_3pct":all(max(m-ci["lower_ns"],ci["upper_ns"]-m)/m<=.03 for m,ci in zip(c["median_body_ns"],c["median_confidence95_by_process"]))}
            logical = p*r*g
            if s == 1:
                checks["resident_ctas_equal8"] = c["resident_ctas_per_sm"] == 8
            points.append({"key":c["key"][:-1],"checks":checks,"eligible":all(checks.values()),
                           "logical_B_bytes":logical,"copy_body_ns_by_process":c["median_body_ns"],
                           "logical_byte_capacity_per_s":logical/max(c["median_body_ns"])*1e9,
                           "median_confidence95_by_process":c["median_confidence95_by_process"]})
        return {"production_or_evaluation_duration_consumed":False,
                "selection":"Fresh63-sample slower process median complete body; confidence/process/nominal-clock3%; warm publication grid132 only; no control subtraction or production error selection",
                "instruction_counts_match":{str(c["cell_code"]):c["ldgsts_warp_instructions"]==expected[c["cell_code"]] for c in counters},
                "counter_dram_logical_within_1pct":{str(c["cell_code"]):abs(c["dram_read_bytes"]/c["logical_bytes"]-1)<=.01 for c in counters if c["cell_code"] not in (28,29)},
                "warm_counter_checks":{str(c["cell_code"]):c["dram_read_bytes"]/c["logical_bytes"]<=.01 and abs(c["l2_read_sector_bytes"]/c["logical_bytes"]-1)<=.01 for c in counters if c["cell_code"] in (28,29)},
                "progress_points":points,"counters":counters,"cells":summaries}
    if profile in ("direct_progress", "direct_precision"):
        points = []
        for c in summaries:
            if c["key"][-1] != "copy":
                continue
            p, slots, threads, grid, rounds, row, stride, tier, a_mode, _ = c["key"]
            if (rounds != (stride+row-1)//row or tier != "cold_unique"
                    or a_mode != (4 if bm128 else 2) or threads != (256 if bm128 else 128)):
                raise ValueError("direct progress physical program drift")
            checks={"copy_iqr_le_3pct":max(c["relative_iqr"]) <= .03,
                    "process_spread_le_3pct":c["relative_process_spread"] <= .03,
                    "sm_clock_within_3pct_nominal":all(abs(clock/1980-1)<=.03 for clock in c["weighted_mhz"])}
            if profile == "direct_precision":
                del checks["copy_iqr_le_3pct"]
                checks["median_confidence95_within_3pct"] = all(
                    max(m-ci["lower_ns"], ci["upper_ns"]-m)/m <= .03
                    for m,ci in zip(c["median_body_ns"],c["median_confidence95_by_process"]))
            logical=p//row*stride*grid
            points.append({"key":c["key"][:-1], "logical_B_bytes":logical,
                           "copy_body_ns_by_process":c["median_body_ns"],
                           "checks":checks,"eligible":all(checks.values()),
                           "logical_byte_capacity_per_s":logical/max(c["median_body_ns"])*1e9})
            if profile == "direct_precision":
                points[-1]["median_confidence95_by_process"] = c["median_confidence95_by_process"]
                points[-1]["individual_body_relative_iqr"] = c["relative_iqr"]
        expected={19:331200,20:61440,21:184320,22:409600}
        if bm128:
            expected={34:96*1530*32,35:96*2040*12,36:96*4592*32}
        instructions={str(c["cell_code"]):c["ldgsts_warp_instructions"]==expected[c["cell_code"]] for c in counters}
        units={str(c["cell_code"]):abs(c["dram_read_bytes"]/c["logical_bytes"]-1)<=.01 for c in counters}
        return {"production_or_evaluation_duration_consumed":False,
                "selection":("Fresh63-sample slower process median complete body, no control subtraction; distribution-free95% median CI endpoints within3%, process medians/SM clock within3%; no cross-capture best selection"
                             if profile == "direct_precision" else
                             "Predeclared slower process median complete-copy body; no control subtraction; each copy IQR/process spread/clock within3%; failed cells unbound"),
                "instruction_counts_match":instructions,
                "counter_dram_logical_within_1pct":units,
                "progress_points":points,"counters":counters,"cells":summaries}
    if profile == "companion":
        by_key = {tuple(s["key"]):s for s in summaries}
        contrasts = []
        for key, copy in by_key.items():
            if key[-1] != "copy":
                continue
            control = by_key[key[:-1]+("control",)]
            bare = by_key[key[:-2]+(0,"copy")]
            contrasts.append({"key":key[:-1],
                "copy_body_ns":copy["median_body_ns"],
                "control_body_ns":control["median_body_ns"],
                "net_copy_minus_control_ns":[a-b for a,b in zip(copy["median_body_ns"],control["median_body_ns"])],
                "body_ratio_to_b_only":[a/b for a,b in zip(copy["median_body_ns"],bare["median_body_ns"])],
                "copy_iqr":copy["relative_iqr"],
                "copy_process_spread":copy["relative_process_spread"],
                "weighted_mhz":copy["weighted_mhz"]})
        by_code = {c["cell_code"]:c for c in counters}
        expected_instructions={10:81920,15:122880,16:122880,17:122880}
        checks={str(code):by_code[code]["ldgsts_warp_instructions"]==count
                for code,count in expected_instructions.items()}
        return {"production_or_evaluation_duration_consumed":False,
                "status":"causal_screen_only_no_Forward_rate",
                "instruction_counts_match":checks,
                "counter_dram_logical_within_1pct":{str(c["cell_code"]):abs(c["dram_read_bytes"]/c["logical_bytes"]-1)<=.01 for c in counters},
                "contrasts":contrasts,"counters":counters,"cells":summaries}
    if profile == "finite_rows":
        by_key = {tuple(s["key"]): s for s in summaries}
        finite = []
        for key, copy in by_key.items():
            if key[-1] != "copy":
                continue
            control = by_key[key[:-1] + ("control",)]
            net = [a-b for a,b in zip(copy["median_body_ns"], control["median_body_ns"])]
            if min(net) <= 0:
                raise ValueError("nonpositive copy-minus-control service")
            uncertainty = [(a*ia+b*ib)/(a-b) for a,b,ia,ib in zip(
                copy["median_body_ns"], control["median_body_ns"],
                copy["relative_iqr"], control["relative_iqr"])]
            checks = {"process_net_spread_le_3pct": max(net)/min(net)-1 <= .03,
                      "net_iqr_uncertainty_le_3pct": max(uncertainty) <= .03,
                      "sm_clock_within_3pct_nominal": all(abs(c/1980-1) <= .03 for c in copy["weighted_mhz"])}
            p, slots, threads, grid, rounds, row, stride, tier, _ = key
            finite.append({"key":key[:-1], "net_median_ns_by_process":net,
                           "net_iqr_relative_uncertainty":uncertainty,
                           "checks":checks, "eligible":all(checks.values()),
                           "logical_byte_capacity_per_s":p*grid*rounds/max(net)*1e9})
        unit_checks = {str(c["cell_code"]): abs(c["dram_read_bytes"]/c["logical_bytes"]-1) <= .01 for c in counters}
        return {"production_or_evaluation_duration_consumed":False,
                "scope":"independent cold finite row requests; no production program duration",
                "selection_rule":"Slower process median copy-minus-control; per-cell3% net spread/IQR and clock checks; no production error selection",
                "counter_dram_logical_within_1pct":unit_checks,
                "finite_cells":finite, "counters":counters, "cells":summaries}
    selected = next(s for s in summaries if s["key"] == (
        panel, slots, 256, grid, 1024, 128, 128, "warm_private", "copy"))
    warm_counter = next(c for c in counters if c["cell_code"] == counter_code)
    warm_clocks = [clock for s in summaries if s["key"][0] == panel
                   and s["key"][4] == 1024 and s["key"][-2:] == ("warm_private", "copy")
                   for clock in s["weighted_mhz"]]
    checks = {
        "process_spread_le_3pct": selected["relative_process_spread"] <= .03,
        "iqr_le_3pct": max(selected["relative_iqr"]) <= .03,
        "warm_dram_le_1pct": warm_counter["dram_read_bytes"] / warm_counter["logical_bytes"] <= .01,
        "warm_l2_within_1pct": abs(warm_counter["l2_read_sector_bytes"] / warm_counter["logical_bytes"] - 1) <= .01,
        "long_warm_clock_spread_le_3pct": max(warm_clocks) / min(warm_clocks) - 1 <= .03,
    }
    return {"production_or_evaluation_duration_consumed": False,
            "warm_frozen_checks": checks,
            "warm_candidate_bytes_per_s": warm_counter["logical_bytes"] / max(selected["median_body_ns"]) * 1e9,
            "warm_candidate_clock_mhz": statistics.median(selected["weighted_mhz"]),
            "transfer_status": "independent mechanism candidate; source progress and clock transfer remain explicit",
            "counters": counters, "cells": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("private32", "one_resident48", "finite_rows", "companion", "direct_progress", "direct_precision", "cohort_short", "short_warm", "bm128_progress", "coalesced_progress"), default="private32")
    args = parser.parse_args()
    result = analyze(json.loads(args.raw.read_text()), args.profile)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "counters", "finite_cells", "contrasts", "progress_points")}))
