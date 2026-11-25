"""NCU metric extraction: parse raw profiling data into structured values.

Pure data extraction — no kernel classification, no demand assembly.
Those responsibilities belong to the per-op-type predictors.
"""

import re

from evaluation.kernel.dataset import to_float
from evaluation.kernel.observation_parser import parse_complete_gemm_tile


# Kernel-record kind per hardware.  ``ncu_sections``: Nsight-Compute-shaped
# counter sections (NCU itself, or the MetaX/Ascend/synthetic adapters that
# emit the same schema).  ``nsys_launch_only``: the device has no kernel
# counter profiler in the original campaign. H200 records carry NSYS/CUPTI
# launch facts; selected records may additionally carry duration-free NCU
# counters. Counter-derived prediction methods remain unsupported.
KERNEL_RECORD_KIND = {
    "nvidia_a800": "ncu_sections",
    "nvidia_h200": "nsys_launch_only",
    "metax_c500": "ncu_sections",
    "ascend_910b": "ncu_sections",
    "samsung_hbm_pim": "ncu_sections",
    "groq_lpu": "ncu_sections",
    "ucb_gemmini": "ncu_sections",
}
NSYS_LAUNCH_ONLY_SOURCE = "nsys_cupti_kernel_activity"
_NSYS_LAUNCH_STATISTICS_FIELDS = {
    "Block Size",
    "Dynamic Shared Memory Per Block [byte/block]",
    "Grid Size",
    "Registers Per Thread [register/thread]",
    "Shared Memory Configuration Size [byte]",
    "Static Shared Memory Per Block [byte/block]",
    "Threads [thread]",
}
_H200_NCU_METRIC_UNITS = {
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "smsp__warps_eligible.avg.per_cycle_active": "warp",
    "smsp__warps_active.avg.per_cycle_active": "warp",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "%",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "%",
    "sm__cycles_active.avg": "cycle",
    "sm__cycles_elapsed.avg": "cycle",
}
_H200_NCU_BASE_METRIC_UNITS = {
    name: unit for name, unit in _H200_NCU_METRIC_UNITS.items()
    if name not in ("sm__cycles_active.avg", "sm__cycles_elapsed.avg")}


def _validate_h200_ncu_counters(counters):
    if not isinstance(counters, dict) or set(counters) != {
            "Schema Version", "Source", "Metrics", "Units", "Provenance"}:
        raise ValueError("H200 NCU counter section has invalid fields")
    if counters["Schema Version"] != 1 or counters["Source"] != "ncu_raw_page":
        raise ValueError("H200 NCU counter source/version changed")
    if counters["Units"] not in (
            _H200_NCU_BASE_METRIC_UNITS, _H200_NCU_METRIC_UNITS):
        raise ValueError("H200 NCU counter units changed")
    metrics = counters["Metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(counters["Units"]):
        raise ValueError("H200 NCU metrics are incomplete")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"H200 NCU metric {name} must be numeric")
        if not 0 <= value < float("inf"):
            raise ValueError(f"H200 NCU metric {name} is invalid")
    if "sm__cycles_active.avg" in metrics:
        if not 0 < metrics["sm__cycles_active.avg"] <= (
                metrics["sm__cycles_elapsed.avg"]):
            raise ValueError("H200 NCU active/elapsed cycles are invalid")
    provenance = counters["Provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
            "campaign", "case_id", "kernel_index", "ncu_kernel_name",
            "csv_sha256", "report_sha256", "report_name", "app_id",
            "ncu_version"}:
        raise ValueError("H200 NCU provenance fields are invalid")
    for name in ("campaign", "case_id", "ncu_kernel_name", "report_name",
                 "app_id", "ncu_version"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(f"H200 NCU provenance {name} is empty")
    if type(provenance["kernel_index"]) is not int or provenance["kernel_index"] < 0:
        raise ValueError("H200 NCU kernel index is invalid")
    for name in ("csv_sha256", "report_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", provenance[name]) is None:
            raise ValueError(f"H200 NCU provenance {name} is invalid")


def require_hardware_kernel_record_kind(metrics, hardware_id):
    """Fail unless a record's kind is the one declared for its hardware."""
    if hardware_id not in KERNEL_RECORD_KIND:
        raise ValueError(
            f"no kernel-record kind declared for hardware {hardware_id!r}")
    expected = KERNEL_RECORD_KIND[hardware_id]
    if metrics["record_kind"] != expected:
        raise ValueError(
            f"{hardware_id} kernel records must be {expected!r}, got "
            f"{metrics['record_kind']!r}")


def _dimensions(text, context):
    if not isinstance(text, str):
        raise TypeError(f"{context} must be a string")
    match = re.fullmatch(
        r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*", text)
    if match is None:
        raise ValueError(
            f"{context} must contain exactly three dimensions: {text!r}")
    values = [int(value) for value in match.groups()]
    if any(value <= 0 for value in values):
        raise ValueError(f"{context} dimensions must be positive: {values}")
    return values


def _extract_nsys_launch_metrics(kdata):
    """Launch facts of one NSYS/CUPTI record, with an optional NCU sidecar."""
    meta = kdata["Metadata"]
    sections = kdata["Sections"]
    if set(sections) not in ({"Launch Statistics"},
                            {"Launch Statistics", "NCU Counters"}):
        raise ValueError(
            "NSYS launch record has invalid sections: "
            f"{sorted(sections)}")
    if "NCU Counters" in sections:
        _validate_h200_ncu_counters(sections["NCU Counters"])
    header = sections["Launch Statistics"]["Header"]
    if set(header) != _NSYS_LAUNCH_STATISTICS_FIELDS:
        raise ValueError(
            "NSYS Launch Statistics fields are invalid: "
            f"{sorted(header)}")
    kernel_name = meta["Kernel Name"]
    if not isinstance(kernel_name, str) or not kernel_name:
        raise TypeError("Metadata.Kernel Name must be a non-empty string")
    grid_x, grid_y, grid_z = _dimensions(
        meta["Grid Size"], "Metadata.Grid Size")
    block_x, block_y, block_z = _dimensions(
        meta["Block Size"], "Metadata.Block Size")
    grid_size = grid_x * grid_y * grid_z
    block_size = block_x * block_y * block_z
    if _integer_measurement(header["Grid Size"], "Launch Statistics.Grid Size",
                            positive=True) != grid_size:
        raise ValueError("NSYS grid dimensions disagree with the grid size")
    if _integer_measurement(header["Block Size"],
                            "Launch Statistics.Block Size",
                            positive=True) != block_size:
        raise ValueError("NSYS block dimensions disagree with the block size")
    return {
        "record_kind": "nsys_launch_only",
        "kernel_name": kernel_name,
        "tile": parse_complete_gemm_tile(kernel_name),
        "grid_size": grid_size,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "block_size": block_size,
        "block_x": block_x,
        "block_y": block_y,
        "block_z": block_z,
        "regs_per_thread": _integer_measurement(
            header["Registers Per Thread [register/thread]"],
            "Launch Statistics.Registers Per Thread"),
        "static_smem_per_block": _integer_measurement(
            header["Static Shared Memory Per Block [byte/block]"],
            "Launch Statistics.Static Shared Memory Per Block"),
        "dynamic_smem_per_block": _integer_measurement(
            header["Dynamic Shared Memory Per Block [byte/block]"],
            "Launch Statistics.Dynamic Shared Memory Per Block"),
        "op_type": None,
        "cube_fops": None,
        "ascend": None,
    }


def _integer_measurement(value, context, *, positive=False):
    number = to_float(value)
    if not number.is_integer():
        raise ValueError(f"{context} must be an integer, got {number}")
    integer = int(number)
    if positive and integer <= 0:
        raise ValueError(f"{context} must be positive, got {integer}")
    if not positive and integer < 0:
        raise ValueError(f"{context} cannot be negative, got {integer}")
    return integer


def extract_ncu_metrics(kdata):
    """Extract raw NCU metrics from one kernel's profiling data.

    Returns dict with measured quantities — no classification, no demand dict.
    Callers (predictors) assemble demands from these raw values.

    Keys returned:
        duration_ns, sm_freq_hz, dram_freq_hz, elapsed_cycles, active_cycles,
        sm_active_frac, tensor_pct,
        l2_total, dram_bytes, l2_hit_rate, l1_hit_rate, ldgsts_bypass,
        grid_size, grid_x, grid_y, grid_z,
        block_size, block_x, block_y, block_z, regs_per_thread, smem_per_block,
        kernel_name,
        achieved_occupancy_pct, theoretical_occupancy_pct, block_limit_registers
    """
    if not isinstance(kdata, dict):
        raise TypeError("profiler kernel record must be a mapping")
    allowed_record_fields = {"Metadata", "Sections", "Ascend", "MetaX"}
    unknown_record_fields = set(kdata) - allowed_record_fields
    if unknown_record_fields:
        raise ValueError(
            "profiler kernel record has unknown fields: "
            f"{sorted(unknown_record_fields)}")
    required_record_fields = {"Metadata", "Sections"}
    missing_record_fields = required_record_fields - set(kdata)
    if missing_record_fields:
        raise ValueError(
            "profiler kernel record is missing fields: "
            f"{sorted(missing_record_fields)}")
    meta = kdata["Metadata"]
    if "Source" in meta:
        # Only NSYS launch-only records declare a Metadata.Source.
        if meta["Source"] != NSYS_LAUNCH_ONLY_SOURCE:
            raise ValueError(
                f"unknown profiler record source: {meta['Source']!r}")
        if set(kdata) != {"Metadata", "Sections"}:
            raise ValueError(
                "NSYS launch-only record has vendor payload fields: "
                f"{sorted(kdata)}")
        return _extract_nsys_launch_metrics(kdata)
    sec = kdata["Sections"]
    if "MetaX" in kdata:
        metax = kdata["MetaX"]
        if not isinstance(metax, dict):
            raise TypeError("MetaX profiler payload must be a mapping")
        required_metax_fields = {
            "all_scalars", "charts", "dur_source", "during",
            "n_invocations", "roofline", "source",
        }
        optional_metax_fields = {"tool_limitation"}
        missing_metax_fields = required_metax_fields - set(metax)
        unknown_metax_fields = (
            set(metax) - required_metax_fields - optional_metax_fields)
        if missing_metax_fields or unknown_metax_fields:
            raise ValueError(
                "MetaX profiler payload fields are invalid: "
                f"missing={sorted(missing_metax_fields)}, "
                f"unknown={sorted(unknown_metax_fields)}")
        if "tool_limitation" in metax:
            tool_limitation = metax["tool_limitation"]
            if not isinstance(tool_limitation, str) or not tool_limitation:
                raise TypeError(
                    "MetaX.tool_limitation must be a non-empty string")
    # ─── GPU Speed Of Light ───
    sol = sec["GPU Speed Of Light Throughput"]["Header"]
    duration_ns = to_float(sol["Duration [nsecond]"])
    sm_freq_hz = to_float(sol["SM Frequency [cycle/second]"])
    dram_freq_hz = to_float(sol["DRAM Frequency [cycle/second]"])
    elapsed_cycles = to_float(sol["Elapsed Cycles [cycle]"])
    active_cycles = to_float(sol["SM Active Cycles [cycle]"])
    if duration_ns <= 0 or sm_freq_hz <= 0 or dram_freq_hz <= 0:
        raise ValueError("profiler duration and clock frequencies must be positive")
    if elapsed_cycles <= 0 or active_cycles < 0:
        raise ValueError("profiler elapsed cycles must be positive and active cycles non-negative")
    sm_active_frac = active_cycles / elapsed_cycles

    # ─── Pipe Utilization ───
    pa = sec["Compute Workload Analysis"]["Pipe Utilization (% of active cycles)"]
    if "Tensor (All) [%]" in pa:
        tensor_pct = to_float(pa["Tensor (All) [%]"])
    elif "Shared (FP64+Tensor) [%]" in pa:
        tensor_pct = to_float(pa["Shared (FP64+Tensor) [%]"])
    else:
        raise KeyError(f"No tensor pipe utilization metric found in: {list(pa.keys())}")

    fma_pct = to_float(pa["FMA [%]"])
    alu_pct = to_float(pa["ALU [%]"])

    pa_peak = sec["Compute Workload Analysis"]["Pipe Utilization (% of peak instructions executed)"]
    sfu_pct = to_float(pa_peak["XU [%]"])
    if "LSU [%]" in pa_peak:
        lsu_pct = to_float(pa_peak["LSU [%]"])
    else:
        # The provider schema explicitly omits LSU for synthetic non-GPU
        # observations.  Normalize that declared optional field once here.
        lsu_pct = None

    # ─── Memory Chart ───
    mc = sec["Memory Workload Analysis Chart"]["Memory Chart"]
    xbar_read = to_float(mc["l1tex__m_xbar2l1tex_read_bytes.sum [byte]"])
    xbar_write = to_float(mc["l1tex__m_l1tex2xbar_write_bytes.sum [byte]"])
    l2_total = xbar_read + xbar_write

    dram_read = to_float(mc["dram__bytes_read.sum [byte]"])
    dram_write = to_float(mc["dram__bytes_write.sum [byte]"])
    dram_bytes = dram_read + dram_write

    l1_hit_rate = to_float(mc["l1tex__t_sector_hit_rate.pct [%]"]) / 100.0

    l2_hit_rate = to_float(mc["lts__t_sector_hit_rate.pct [%]"]) / 100.0

    ldgsts_bypass = to_float(mc["sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]"])

    # ─── Launch Statistics ───
    lh = sec["Launch Statistics"]["Header"]
    block_size = _integer_measurement(
        lh["Block Size"], "Launch Statistics.Block Size", positive=True)
    block_text = meta["Block Size"]
    if not isinstance(block_text, str):
        raise TypeError("Metadata.Block Size must be a string")
    block_match = re.fullmatch(
        r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*",
        block_text,
    )
    if block_match is None:
        raise ValueError(
            "Metadata.Block Size must contain exactly three positive "
            f"dimensions: {block_text!r}")
    bd = [int(value) for value in block_match.groups()]
    if any(dimension <= 0 for dimension in bd):
        raise ValueError(
            f"Metadata.Block Size must contain three positive dimensions: {bd}")
    block_product = 1
    for dimension in bd:
        block_product *= dimension
    if block_product != block_size:
        raise ValueError(
            "Metadata.Block Size dimensions disagree with Launch Statistics "
            f"block size: dimensions={bd}, scalar={block_size}")
    block_x, block_y, block_z = bd
    regs_per_thread = _integer_measurement(
        lh["Registers Per Thread [register/thread]"],
        "Launch Statistics.Registers Per Thread")
    smem_dynamic = to_float(lh["Dynamic Shared Memory Per Block [byte/block]"])
    smem_static = to_float(lh["Static Shared Memory Per Block [byte/block]"])
    smem_driver = to_float(lh["Driver Shared Memory Per Block [byte/block]"])
    smem_total = smem_dynamic + smem_static + smem_driver
    if not smem_total.is_integer():
        raise ValueError(
            "Launch Statistics shared-memory bytes must sum to an integer, "
            f"got {smem_total}")
    smem_per_block = int(smem_total)

    # ─── Occupancy (from Launch Statistics Header and Occupancy section) ───
    # Achieved Occupancy [%] and Theoretical Occupancy [%] appear in both
    # Launch Statistics Header and Occupancy Header — values are identical;
    # read from Launch Statistics Header (always present).
    # These are expressed as percentages (0-100); store as raw percentages.
    achieved_occupancy_pct = to_float(lh["Achieved Occupancy [%]"])
    theoretical_occupancy_pct = to_float(lh["Theoretical Occupancy [%]"])

    # Block Limit Registers from Occupancy section (hardware register constraint).
    occ_h = sec["Occupancy"]["Header"]
    block_limit_registers = _integer_measurement(
        occ_h["Block Limit Registers [block]"],
        "Occupancy.Block Limit Registers")

    # ─── Grid Size ───
    kernel_name = meta["Kernel Name"]
    if not isinstance(kernel_name, str) or not kernel_name:
        raise TypeError("Metadata.Kernel Name must be a non-empty string")
    grid_text = meta["Grid Size"]
    if not isinstance(grid_text, str):
        raise TypeError("Metadata.Grid Size must be a string")
    grid_match = re.fullmatch(
        r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*",
        grid_text,
    )
    if grid_match is None:
        raise ValueError(
            "Metadata.Grid Size must contain exactly three positive "
            f"dimensions: {grid_text!r}")
    gd = [int(value) for value in grid_match.groups()]
    if any(dimension <= 0 for dimension in gd):
        raise ValueError(
            f"Metadata.Grid Size must contain three positive dimensions: {gd}")
    grid_size = 1
    for d in gd:
        grid_size *= d
    grid_x, grid_y, grid_z = gd

    # ─── Ascend-native fields (present only in 910B dataset) ───
    # Metadata."Op Type" (cube|vector|mix) is the classification discriminator on
    # Ascend; the "Ascend" sub-dict carries measured cube_fops / block dims.
    # GPU records have neither and are normalized to explicit None values.
    has_op_type = "Op Type" in meta
    has_ascend_payload = "Ascend" in kdata
    if has_op_type != has_ascend_payload:
        raise ValueError(
            "profiler record must contain Metadata.Op Type and Ascend "
            "together")
    if has_op_type:
        op_type_value = meta["Op Type"]
    else:
        op_type_value = None
    if op_type_value is None:
        op_type = None
    elif isinstance(op_type_value, str) and op_type_value:
        op_type = op_type_value.lower()
    else:
        raise TypeError("Metadata.Op Type must be a non-empty string or null")
    if op_type not in (None, "cube", "mix", "vector"):
        raise ValueError(f"unknown profiler Op Type: {op_type!r}")
    if has_ascend_payload:
        ascend = kdata["Ascend"]
    else:
        ascend = None
    if ascend is not None and not isinstance(ascend, dict):
        raise ValueError("Ascend profiler payload must be a mapping or null")
    cube_fops = None if ascend is None else to_float(ascend["cube_fops"])
    if cube_fops is not None and cube_fops < 0:
        raise ValueError("Ascend cube_fops cannot be negative")

    # ─── DRAM throughput evidence for Oracle resource normalization ───
    dm = sec["Memory Workload Analysis Tables"]["Device Memory"]
    dram_read_bw_pct = to_float(dm["dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]"])
    dram_write_bw_pct = to_float(dm["dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]"])
    # Total DRAM BW as fraction of peak sustained
    dram_bw_pct = (dram_read_bw_pct + dram_write_bw_pct) / 100.0

    return {
        "record_kind": "ncu_sections",
        "duration_ns": duration_ns,
        "sm_freq_hz": sm_freq_hz,
        "dram_freq_hz": dram_freq_hz,
        "elapsed_cycles": elapsed_cycles,
        "active_cycles": active_cycles,
        "sm_active_frac": sm_active_frac,
        "tensor_pct": tensor_pct,
        "fma_pct": fma_pct,
        "alu_pct": alu_pct,
        "sfu_pct": sfu_pct,
        "lsu_pct": lsu_pct,
        "l2_total": l2_total,
        "dram_bytes": dram_bytes,
        "dram_bw_pct": dram_bw_pct,
        "l2_hit_rate": l2_hit_rate,
        "l1_hit_rate": l1_hit_rate,
        "ldgsts_bypass": ldgsts_bypass,
        "grid_size": grid_size,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "block_size": block_size,
        "block_x": block_x,
        "block_y": block_y,
        "block_z": block_z,
        "regs_per_thread": regs_per_thread,
        "smem_per_block": smem_per_block,
        "kernel_name": kernel_name,
        "tile": parse_complete_gemm_tile(kernel_name),
        "achieved_occupancy_pct": achieved_occupancy_pct,        "theoretical_occupancy_pct": theoretical_occupancy_pct,
        "block_limit_registers": block_limit_registers,
        "op_type": op_type,
        "cube_fops": cube_fops,
        "ascend": ascend,
    }
