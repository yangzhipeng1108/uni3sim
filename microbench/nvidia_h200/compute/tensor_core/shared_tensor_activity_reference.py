"""Publish independent activity statistics without reading evaluation data."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0,str(ROOT))
from src.kernel.hardware_spec import _gemm_accumulator_activity_statistics


def derive():
    captures,measurements=[],[]
    source_identity = None
    campaigns = tuple(ROOT / "profiler_output/microbench/nvidia_h200" / name
        for name in ("shared_tensor_sustained_20261107", "shared_tensor_sustained_replica_20261107"))
    for repetition, campaign in enumerate(campaigns):
        for index,(name,expected_points) in enumerate(zip(("base_raw.json","extended_raw.json"),(96,128))):
            path=campaign/name
            data=path.read_bytes()
            raw=json.loads(data)
            if (raw["build_exit_code"]!=0 or raw["run_exit_code"]!=0
                    or raw["production_duration_consumed"] is not False
                    or raw["sustained_device_warmup"] is not True
                    or "-DSUSTAINED_DEVICE_ACTIVITY" not in raw["command"]):
                raise ValueError("only complete, numerically checked independent captures can publish")
            rows=[json.loads(line) for line in raw["stdout"].splitlines()]
            if len(rows)!=expected_points or any(row["numerical_check_passed"] is not True for row in rows):
                raise ValueError("activity control sweep is incomplete or numerically invalid")
            geometries={(row["instruction_n"],row["accumulator_chains"]) for row in rows}
            expected={(n,chains,pattern,seed,rounds) for (n,chains),pattern,seed,rounds in itertools.product(
                geometries,("signed_power_of_two","random_sign_exponent_and_mantissa"),(0,1,2,3),(128,512))}
            keys=[tuple(row[field] for field in ("instruction_n","accumulator_chains","operand_pattern","seed","rounds")) for row in rows]
            if len(keys)!=len(expected) or set(keys)!=expected or raw["extended_controls"] is not bool(index):
                raise ValueError("paired fraction controls are absent or duplicated")
            if source_identity is None:
                source_identity = raw["sources"]
            elif raw["sources"] != source_identity:
                raise ValueError("matched activity repetitions changed source/generator/helper")
            runner_source = (campaign/"source/shared_tensor_mantissa_modal_runner.py").read_text()
            if "nvidia/cuda:12.9.1-devel-ubuntu24.04" not in runner_source:
                raise ValueError("matched activity repetition changed the declared CUDA image")
            for source_name,sha in raw["sources"].items():
                snapshot=campaign/"source"/Path(source_name).name
                if snapshot.name=="fa3_wgmma_service.cu":
                    payload=gzip.decompress(snapshot.with_suffix(".cu.gz").read_bytes())
                else:
                    payload=snapshot.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=sha:
                    raise ValueError("fraction probe source pin drifted")
            if hashlib.sha256((campaign/"source/shared_tensor_mantissa_modal_runner.py").read_bytes()).hexdigest()!=raw["runner_sha256"]:
                raise ValueError("fraction runner pin drifted")
            captures.append({"path":str(path.relative_to(ROOT)),
                "sha256":hashlib.sha256(data).hexdigest(),
                "source_sha256":raw["sources"]["microbench/nvidia_h200/compute/tensor_core/shared_tensor_activity.cu"],
                "runner_sha256":raw["runner_sha256"]})
            for row in rows:
                if row["seed"]==0 or row["operand_pattern"]!="random_sign_exponent_and_mantissa":
                    continue
                if row["grid_ctas"]!=132 or row["threads_per_cta"]!=256:
                    raise ValueError("unexpected independent activity geometry")
                measurements.append({"capture_index":2 * repetition + index,"instruction_n":row["instruction_n"],
                    "accumulator_chains":row["accumulator_chains"],"operand_seed":row["seed"],
                    "rounds":row["rounds"],"math_cycles":row["cycles_p50"],"clock_mhz":row["clock_mhz"]})
    if len({capture["source_sha256"] for capture in captures}) != 1:
        raise ValueError("matched activity repetitions changed the probe source")
    rate,points=_gemm_accumulator_activity_statistics(measurements,"independent Tensor activity",
        measurement_repetitions=len(campaigns))
    reference={"schema_version":7,"measurement_repetitions":len(campaigns),
        "measurement_protocol":{"kind":"host_gap_free_device_warmup",
            "device_warmup_reference_cycles":96*1024*1024,"accumulator_reset_each_window":True},"hardware_id":"nvidia_h200",
        "mechanism":"shared_operand_fp16_wgmma_accumulator_activity_service",
        "selection":"median_per_capture_then_geometry_then_accumulator_footprint",
        "operand_activity_pattern":"random_sign_exponent_and_mantissa",
        "grid_ctas":132,"threads_per_cta":384,"measurement_threads_per_cta":256,
        "math_warpgroups":2,"captures":captures,"measurements":measurements,
        "selected_flops_per_sm_cycle":rate,"activity_points":points,
        "scope_limitations":[
            "independent on-chip FP16 WGMMA with two math warpgroups and one CTA per SM",
            "measurement uses 256 math threads and 196608-B residency; the aggregate TMA consumer has 384 total threads including a producer warpgroup",
            "per-math-thread accumulator footprint is dense output-tile elements divided by 256, checked statically on four retained matrix families without duration",
            "all three fixed random fraction seeds and both lengths contribute; all-one and signed-power-of-two controls do not price activity",
            "original complete v6 capture and exactly one same-source complete repetition are equally included before production comparison; no app selected by higher clock or lower production error",
            "capture medians, geometry medians, then equal-weight footprint medians; all paired controls must be complete and checked",
            "only measured footprints; no extrapolation or interpolation, no production case or kernel-name price",
            "random sign, exponent and ten fraction bits form an activity envelope, not exact Gaussian input/weight power or native HGMMA schedule certification",
            "sustained isolated math activity does not bind native producer/memory pauses, power policy or actual input/weight amplitude distributions",
            "all-one arithmetic is exact; old power-of-two long/short arithmetic is exact; random fraction outputs are checked for finiteness and the analytical magnitude bound, not bitwise FP32 equivalence",
            "memory byte capacities and latency ns are unchanged"],
        "production_or_evaluation_duration_consumed":False}
    target=ROOT/"microbench/nvidia_h200/references/dense_shared_tensor_service_sm90.json"
    target.write_text(json.dumps(reference,indent=2)+"\n")
    return reference


if __name__=="__main__":
    result=derive()
    print(json.dumps({"flops_per_sm_cycle":result["selected_flops_per_sm_cycle"],
        "activity_points":result["activity_points"]},indent=2))
