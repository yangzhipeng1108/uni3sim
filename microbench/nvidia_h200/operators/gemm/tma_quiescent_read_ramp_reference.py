"""Derive a finite TMA byte-service envelope from the complete independent probe."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.kernel.hardware_spec import HWSpec, _derive_tma_quiescent_read_ramp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-name", choices=("tma_activity_transition_full_20261108",
        "tma_activity_transition_wide_20261108"), default="tma_activity_transition_wide_20261108")
    arguments = parser.parse_args()
    campaign = ROOT / "profiler_output/microbench/nvidia_h200" / arguments.campaign_name
    request_b_sizes = ((4096, 8192, 16384, 32768)
        if arguments.campaign_name == "tma_activity_transition_wide_20261108"
        else (4096, 8192, 16384))
    raw_bytes = (campaign / "raw.json").read_bytes()
    raw = json.loads(raw_bytes)
    measurements = []
    for item in raw["captures"]:
        if item["build_exit_code"] != 0 or item["run_exit_code"] != 0:
            raise ValueError("all independent controls must pass")
        if item["activity_prelude"] != 2:
            continue
        for line in item["stdout"].splitlines():
            row = json.loads(line)
            if row["resident"] == 0:
                measurements.append({"grid_ctas": row["grid"], "a_bytes": row["a_bytes"],
                    "b_bytes": row["b_bytes"], "depth": row["depth"],
                    "rounds": row["rounds"], "span_ns": row["span_ns_p50"]})
    hw = HWSpec(str(ROOT / "configs/hardware/nvidia_h200.yaml"))
    capacity = hw.dram_capacity_for_access("read", active_units=hw.num_compute_units,
        sm_freq_hz=hw.nominal_sm_freq_hz, dram_freq_hz=hw.nominal_dram_freq_hz)
    grids, lengths, ramp = _derive_tma_quiescent_read_ramp(measurements, capacity,
        request_b_sizes=request_b_sizes)
    reference = {"schema": "dram_finite_read_stream_ramp_reference_v3_tma_quiescent",
        "hardware_id": "nvidia_h200", "mechanism": "dram_finite_read_stream_ramp",
        "production_or_evaluation_duration_consumed": False,
        "selection_rule": "quiescent_advancing_cumulative_grid_length_capacity_upper_envelope",
        "capture": {"path": str((campaign / "raw.json").relative_to(ROOT)),
            "sha256": hashlib.sha256(raw_bytes).hexdigest()},
        "scope": {"transport": "tma", "threads_per_cta": 384, "tile_k": 64,
            "operand_element_bytes": 2, "memory_start_state": "quiescent_after_3ms_ALU",
            "request_total_bytes_min": 5120, "request_total_bytes_max": 1024 + max(request_b_sizes)},
        "source_state_basis": "H200 capture helper before_nvtx waits GPU-idle for two clock samples with0.25s interval; ALU controls isolate a memory-quiescent high-SM-clock domain. This is an aggregate start-state assumption, not an observation of every Serving request.",
        "measurements": measurements,
        "derived_model": {"dram_read_capacity_bytes_per_s": capacity,
            "grids": list(grids), "lengths_bytes_per_cta": list(lengths),
            "ramp": [{"grid_ctas": grid, "bytes_per_cta": length,
                "achieved_fraction_of_dram_read_capacity": ramp[grid, length]}
                for grid in grids for length in lengths]},
        "allowed_use": "Ceiling grid/length lookup inside measured domain; maximum over every legal payload/depth and cumulative smaller grid/length. Use as an optimistic finite-copy completion bound. Compose by max after subtracting separately priced overlapping transport prefix; never add a wait residual or reuse LSU ramp prices.",
        "limits": ["Private ordinary3D requests; native descriptor, multicast fabric and stage-specific miss phase remain aggregate approximations.",
            "Forward quiescent start is an explicit assumption; hot predecessor state may make this bound inapplicable.",
            "Advancing panels plus eviction do not assert observed zero hits; logical rates are optimistic capacity envelopes."]}
    target = ROOT / "microbench/nvidia_h200/references/dram_finite_tma_read_stream_ramp_sm90.json"
    data = (json.dumps(reference, indent=2) + "\n").encode()
    target.write_bytes(data)
    print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(),
        "measurements": len(measurements), "grids": grids, "lengths": lengths}))


if __name__ == "__main__":
    main()
