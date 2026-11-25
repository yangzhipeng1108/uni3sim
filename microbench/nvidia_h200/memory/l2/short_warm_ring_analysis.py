"""Check the frozen short-ring scope without selecting a new rate."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def analyze(raw, existing_reference):
    if raw['status']!='complete':raise ValueError('short-ring capture incomplete')
    capability=existing_reference['selected_service']
    if (capability['panel_bytes'],capability['buffer_slots'],capability['threads_per_cta'],capability['resident_ctas_per_sm'])!=(49152,4,256,1):raise ValueError('existing mechanism does not match')
    capacity=capability['logical_byte_capacity_per_s']
    cells={}
    for item in raw['cells']:
        key=item['process'],item['rounds'];process,rounds=key
        if key in cells or item['measurement']['exit_code']:raise ValueError('duplicate or failed body cell')
        rows=[json.loads(line) for line in item['measurement']['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):raise ValueError('incomplete body sample sequence')
        values=[];clocks=[]
        for row in rows:
            if (row['process'],row['rounds'],row['resident'])!=(process,rounds,1) or len(row['ctas'])!=132:raise ValueError('changed body geometry')
            if sorted(c[0] for c in row['ctas'])!=list(range(132)):raise ValueError('not one CTA per SM')
            if any(len(c)!=4 or c[2]<=c[1] or c[3]<=0 for c in row['ctas']):raise ValueError('invalid CTA stamps')
            values.append(max(c[2] for c in row['ctas'])-min(c[1] for c in row['ctas']))
            clocks.append(1e9*sum(c[3] for c in row['ctas'])/sum(c[2]-c[1] for c in row['ctas']))
        median=statistics.median(values);ci=median_confidence_interval(values);clock=statistics.median(clocks)
        floor=132*rounds*49152/capacity*1e9
        cells[key]=dict(process=process,rounds=rounds,median_body_ns=median,confidence95=ci,clock64_hz=clock,existing_capacity_floor_ns=floor,checks=dict(
            median_precision3pct=max(median-ci['lower_ns'],ci['upper_ns']-median)/median<=.03,
            nominal_clock3pct=abs(clock/1.98e9-1)<=.03,
            old_capacity_not_exceeded=ci['lower_ns']>=.97*floor))
    if set(cells)!={(p,r) for p in range(2) for r in (8,12,16,32,64)}:raise ValueError('incomplete physical body sweep')
    for (p,r),cell in cells.items():
        a,b=cell['median_body_ns'],cells[1-p,r]['median_body_ns']
        cell['checks']['process_spread3pct']=max(a,b)/min(a,b)-1<=.03
    counters={}
    for item in raw['counters']:
        rounds=item['rounds'];measurement=item['measurement']
        if rounds in counters or measurement['exit_code']:raise ValueError('duplicate or failed counter')
        lines=measurement['stdout'].splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            v=float(row[name].replace(',',''))
            if not math.isfinite(v) or v<0:raise ValueError('invalid hardware counter')
            return v
        counters[rounds]=dict(rounds=rounds,tex_bytes=32*value('lts__t_sectors_srcunit_tex_op_read.sum'),fabric_bytes=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum'),l2_bytes=32*value('lts__t_sectors_op_read.sum'),dram_bytes=value('dram__bytes_read.sum'),sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),checks=dict(
            exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==132*(4+rounds)*49152/512,
            exact_ctas=value('sm__ctas_launched.sum')==132))
    if set(counters)!={0,8,12,16,32,64}:raise ValueError('incomplete counter sweep')
    zero=counters[0]
    for rounds,c in counters.items():
        if not rounds:continue
        logical=132*rounds*49152
        c['body_logical_bytes']=logical;c['net_tex_bytes']=c['tex_bytes']-zero['tex_bytes'];c['net_dram_bytes']=c['dram_bytes']-zero['dram_bytes']
        c['checks'].update(net_tex1pct=abs(c['net_tex_bytes']/logical-1)<=.01,net_dram1pct=abs(c['net_dram_bytes'])/logical<=.01,
            source_units1pct=abs(c['l2_bytes']-c['tex_bytes']-c['fabric_bytes'])/logical<=.01)
    checks=dict(all_body_cells=all(all(c['checks'].values()) for c in cells.values()),all_counter_cells=all(all(c['checks'].values()) for c in counters.values()))
    for field in ('sm_clock_hz','lts_clock_hz'):
        values=[c[field] for c in counters.values()];checks[field+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    checks['direct_sm_nominal3pct']=all(abs(c['sm_clock_hz']/1.98e9-1)<=.03 for c in counters.values())
    return dict(short_scope_extension_qualified=all(checks.values()),checks=checks,cells=list(cells.values()),counters=list(counters.values()),existing_capacity_per_s=capacity,new_rate_selected=False,production_duration_consumed=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    reference=Path('microbench/nvidia_h200/references/async_copy_ring_one_resident_sm90.json')
    result=analyze(json.loads(args.raw.read_text()),json.loads(reference.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('short_scope_extension_qualified','checks')}))
