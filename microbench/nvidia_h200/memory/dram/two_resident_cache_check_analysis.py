"""Independent traffic-only check of unchanged cache-replay forecasts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from microbench.nvidia_h200.memory.dram.two_resident_cold_copy_analysis import fixture

def analyze(raw,plan_bytes):
    plan=json.loads(plan_bytes)
    if raw['status']!='complete' or raw['plan']!=plan or raw['plan_sha256']!=hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError('complete frozen independent capture required')
    expected={(c['stride'],c['grid'],c['fanout']):c for c in plan['physical_cells']}
    if len(expected)!=8:raise ValueError('all eight heldout cells required')
    cells={}
    for item in raw['counters']:
        key=item['stride'],item['grid'],item['fanout']
        if key not in expected or key in cells:raise ValueError('unknown or duplicate cell')
        fixture(item,1)
        lines=item['measurement']['stdout'].splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def metric(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        s,g,f=key;rounds=s//128;logical=g*rounds*(16384+128);forecast=expected[key]
        upstream=32*metric('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*metric('lts__t_sectors_srcunit_tex_op_read.sum');fabric=32*metric('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*metric('lts__t_sectors_op_read.sum');dram=metric('dram__bytes_read.sum')
        sm=metric('sm__cycles_elapsed.avg.per_second');lts=metric('lts__cycles_elapsed.avg.per_second')
        predicted=forecast['predicted_read_bytes'];unique=forecast['unique_read_bytes']
        cells[key]=dict(identity=key,observed_dram_bytes=dram,predicted_read_bytes=predicted,
            unique_read_bytes=unique,relative_read_error=abs(dram/predicted-1),sm_clock_hz=sm,lts_clock_hz=lts,
            checks=dict(exact_ctas=metric('sm__ctas_launched.sum')==g,
                exact_ldgsts=metric('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==g*rounds*64,
                upstream1pct=abs(upstream/logical-1)<=.01,
                downstream_conservation=unique*.99<=tex<=logical*1.01,
                source_units1pct=abs(total-tex-fabric)/logical<=.01,
                cache_forecast_read1pct=abs(dram/predicted-1)<=.01,
                expected_replay_regime=(dram>unique*1.01 if predicted>unique*1.01 else abs(dram/unique-1)<=.01),
                sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(cells)!=set(expected):raise ValueError('missing heldout counter')
    checks={'all_cell_checks':all(all(c['checks'].values()) for c in cells.values())}
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[c[name] for c in cells.values()];checks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    return dict(qualified=all(checks.values()),checks=checks,cells=list(cells.values()),
        production_or_evaluation_duration_consumed=False,original_unique_gate_still_failed=True,
        new_rate_or_parameter_published=False,selection=plan['selection'])

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    result=analyze(json.loads(a.raw.read_text()),a.plan.read_bytes());a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
