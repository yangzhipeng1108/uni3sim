"""Decide whether a second kernel's text evicts the first from the AI vector
instruction cache.

The footprint sweep can only contrast a process's first launch against later
launches of the same binary, which conflates a cold instruction cache with
one-time runtime setup.  This reduction contrasts two processes that pay that
setup identically and differ only in what ran between two launches of the same
kernel:

    repeat     A A A A ...
    alternate  A B A B ...

Both modes are read at the same task positions, skipping the warm-up pair, so
the comparison is between steady-state A-after-A and steady-state A-after-B.
The PMU counter aiv_icache_miss_rate arbitrates; Task Duration(us) states what
the miss costs.
"""

import argparse
import csv
import json
import pathlib
import statistics


def _task_rows(capture_dir):
    summaries = sorted(capture_dir.rglob("op_summary*.csv"))
    if not summaries:
        raise FileNotFoundError(f"no op_summary under {capture_dir}")
    rows = []
    for summary in summaries:
        with summary.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _kernel_tasks(rows, name):
    selected = [row for row in rows if row["Op Name"].startswith(name)]
    selected.sort(key=lambda row: float(row["Task Start Time(us)"]))
    return selected


def _mode_summary(captures, variant, mode, kernel_name, skip):
    durations = []
    miss_rates = []
    first_durations = []
    for capture in sorted(
            path for path in captures.glob(f"n{variant}_{mode}_r*")
            if path.is_dir()):
        tasks = _kernel_tasks(_task_rows(capture), kernel_name)
        if len(tasks) <= skip:
            raise ValueError(
                f"{capture} has {len(tasks)} {kernel_name} tasks, "
                f"need more than {skip}")
        first_durations.append(float(tasks[0]["Task Duration(us)"]))
        durations.append(statistics.median(
            float(task["Task Duration(us)"]) for task in tasks[skip:]))
        miss_rates.append(statistics.median(
            float(task["aiv_icache_miss_rate"]) for task in tasks[skip:]))
    if not durations:
        raise FileNotFoundError(
            f"no {mode} captures for variant {variant} under {captures}")
    return {
        "capture_count": len(durations),
        "first_task_duration_us_median": statistics.median(first_durations),
        "steady_duration_us_median": statistics.median(durations),
        "steady_icache_miss_rate_median": statistics.median(miss_rates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=pathlib.Path)
    parser.add_argument("--text-sizes", required=True, type=pathlib.Path,
                        help="whitespace table of `variant text_bytes`")
    parser.add_argument("--kernel-name", default="aiv_text_footprint")
    parser.add_argument("--skip", type=int, default=2,
                        help="leading tasks excluded as process warm-up")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    text_bytes = {}
    for line in arguments.text_sizes.read_text().split("\n"):
        if not line.strip():
            continue
        variant, size = line.split()
        text_bytes[int(variant)] = int(size)

    variants = []
    for variant in sorted(text_bytes):
        if not list(arguments.captures.glob(f"n{variant}_repeat_r*")):
            continue
        repeat = _mode_summary(
            arguments.captures, variant, "repeat", arguments.kernel_name,
            arguments.skip)
        alternate = _mode_summary(
            arguments.captures, variant, "alternate", arguments.kernel_name,
            arguments.skip)
        variants.append({
            "text_body": variant,
            "text_bytes": text_bytes[variant],
            "resident_pair_bytes": 2 * text_bytes[variant],
            "repeat": repeat,
            "alternate": alternate,
            "alternate_minus_repeat_us": (
                alternate["steady_duration_us_median"]
                - repeat["steady_duration_us_median"]),
            "alternate_minus_repeat_miss_rate": (
                alternate["steady_icache_miss_rate_median"]
                - repeat["steady_icache_miss_rate_median"]),
        })

    arguments.output.write_text(json.dumps({
        "mechanism": "aiv_instruction_cache_eviction_control",
        "kernel_name": arguments.kernel_name,
        "warmup_tasks_skipped": arguments.skip,
        "variants": variants,
    }, indent=2) + "\n")
    for entry in variants:
        print(
            f"n={entry['text_body']:>4} text={entry['text_bytes']:>6} B "
            f"repeat {entry['repeat']['steady_duration_us_median']:8.3f} us "
            f"miss {entry['repeat']['steady_icache_miss_rate_median']:.5f} | "
            f"alternate {entry['alternate']['steady_duration_us_median']:8.3f} us "
            f"miss {entry['alternate']['steady_icache_miss_rate_median']:.5f} | "
            f"delta {entry['alternate_minus_repeat_us']:+7.3f} us")


if __name__ == "__main__":
    main()
