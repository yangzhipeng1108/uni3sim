"""Publish the protocol median of independent Tensor-activity clocks."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[4]
RAW=ROOT/'profiler_output/microbench/nvidia_h200/dense_register_tensor_clock_20261107/raw.json'
REFERENCE=ROOT/'microbench/nvidia_h200/references/dense_register_tensor_clock_sm90.json'

if __name__=='__main__':
    raw_bytes=RAW.read_bytes();raw=json.loads(raw_bytes)
    if raw['run_exit_code']!=0 or raw['production_duration_consumed'] is not False:
        raise ValueError('independent register Tensor clock probe failed')
    controls=[json.loads(line) for line in raw['stdout'].splitlines()]
    if len(controls)!=8 or any(row['patterned']==0 and row['ones_check_passed']!=1 for row in controls):
        raise ValueError('clock controls or numerical checks are incomplete')
    rows=[{'grid_ctas':row['grid'],'rounds':row['rounds'],'patterned':row['patterned'],
        'clock_mhz':row['median_sm_mhz']} for row in controls]
    source=RAW.parent/'dense_register_tensor_clock.cu'
    runner=RAW.parent/'dense_register_tensor_clock_modal_runner.py'
    reference={'schema_version':1,'hardware_id':'nvidia_h200',
        'mechanism':'register_tensor_activity_clock',
        'selection':'median_of_all_eight_control_point_clock_medians','measurements':rows,
        'selected_sm_clock_hz':statistics.median(row['clock_mhz']*1e6 for row in rows),
        'raw':{'path':str(RAW.relative_to(ROOT)),'sha256':hashlib.sha256(raw_bytes).hexdigest()},
        'source_sha256':raw['sources']['microbench/nvidia_h200/compute/tensor_core/dense_register_tensor_clock.cu'],
        'runner_sha256':raw['runner_sha256'],
        'scope_limitations':['provisional single-device register-only Tensor activity clock',
            'not native HGMMA instruction schedule, numerical/provider certification, or a duration-derived efficiency',
            'applies only to declared two-byte TMA GEMM Tensor cycle service inside measured active-unit domain',
            'independent memory byte capacities and latency ns are not rescaled'],
        'production_or_evaluation_duration_consumed':False}
    REFERENCE.write_text(json.dumps(reference,indent=2)+'\n')
    print(reference['selected_sm_clock_hz'])
