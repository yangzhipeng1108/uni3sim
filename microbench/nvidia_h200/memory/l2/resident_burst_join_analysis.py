"""Test the frozen all-return hypotheses without any production durations."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def precision(values):
    m=statistics.median(values);ci=median_confidence_interval(values)
    return dict(median=m,confidence95=ci,eligible=max(m-ci['lower_ns'],ci['upper_ns']-m)/m<=.03)


def analyze(raw):
    if raw['status']!='complete':raise ValueError('burst capture incomplete')
    protocol=raw['counter_protocol'] if 'counter_protocol' in raw else 1
    if protocol not in (1,2):raise ValueError('unknown counter protocol')
    cells={}
    for item in raw['cells']:
        mode,hot,process=item['mode'],item['hot_bytes'],item['process'];key=mode,hot,process
        if key in cells or item['measurement']['exit_code']:raise ValueError('duplicate or failed burst cell')
        rows=[json.loads(line) for line in item['measurement']['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):raise ValueError('incomplete burst samples')
        values=[];clocks=[]
        for row in rows:
            if (row['mode'],row['hot_bytes'],row['process'])!=key or row['rounds']!=256 or row['occupancy']!=2 or len(row['ctas'])!=264:raise ValueError('changed physical launch')
            bysm={sm:[] for sm in range(132)}
            for c in row['ctas']:
                if len(c)!=12:raise ValueError('changed CTA schema')
                sm,entry,total,lo,hi,begin,end,cycles,first_begin,first_end,last_begin,last_end=c
                if sm not in bysm or entry not in (0,1) or not 0<lo<=total/256<=hi or not begin<=first_begin<first_end<=last_begin<last_end<=end or cycles<=0:raise ValueError('invalid CTA timeline')
                bysm[sm].append(entry)
            if any(sorted(v)!=[0,1] for v in bysm.values()):raise ValueError('incomplete per-SM residency')
            values.append(statistics.median(c[2]/256 for c in row['ctas']))
            clocks.append(1e9*sum(c[7] for c in row['ctas'])/sum(c[6]-c[5] for c in row['ctas']))
        cells[key]=dict(key=key,latency=precision(values),outer_clock64_hz=statistics.median(clocks))
    if set(cells)!={(m,h,p) for m in range(4) for h in (4096,16384) for p in range(2)}:raise ValueError('incomplete physical sweep')
    for (mode,hot,process),cell in cells.items():
        peer=cells[mode,hot,1-process]['latency']['median'];m=cell['latency']['median']
        cell['checks']=dict(precision=cell['latency']['eligible'],process_medians=max(m,peer)/min(m,peer)-1<=.03)
    counters={}
    for item in raw['counters']:
        key=item['mode'],item['hot_bytes'];mode,hot=key;measurement=item['measurement']
        if key in counters or measurement['exit_code']:raise ValueError('duplicate or failed counter')
        lines=measurement['stdout'].splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            v=float(row[name].replace(',',''))
            if not math.isfinite(v) or v<0:raise ValueError('invalid hardware counter')
            return v
        logical=264*256*(hot+1024 if mode in (0,1,2) else 1024 if mode==3 else 0)
        cold=264*256*(hot+1024 if mode==2 else 1024 if mode in (1,3) else 0)
        instructions=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')
        counters[key]=dict(key=key,logical_bytes=logical,cold_bytes=cold,l2_read_bytes=32*value('lts__t_sectors_op_read.sum'),dram_read_bytes=value('dram__bytes_read.sum'),ldgsts_instructions=instructions,sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),checks=dict(exact_ldgsts=instructions==logical/512,exact_ctas=value('sm__ctas_launched.sum')==264))
        if protocol==2:
            counters[key]['tex_read_bytes']=32*value('lts__t_sectors_srcunit_tex_op_read.sum')
            counters[key]['fabric_read_bytes']=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
    if set(counters)!={(m,h) for m in range(4) for h in (4096,16384)}|{(4,4096)}:raise ValueError('incomplete counter sweep')
    payload_field='tex_read_bytes' if protocol==2 else 'l2_read_bytes'
    metadata=counters[4,4096][payload_field]
    for (mode,hot),c in counters.items():
        if mode==4:continue
        c['net_l2_read_bytes']=c[payload_field]-metadata
        c['checks']['l2_payload1pct']=abs(c['net_l2_read_bytes']/c['logical_bytes']-1)<=.01
        if protocol==2:
            c['checks']['source_units_cover_total1pct']=abs(c['l2_read_bytes']-c['tex_read_bytes']-c['fabric_read_bytes'])/c['logical_bytes']<=.01
        c['checks']['dram_payload1pct']=(abs(c['dram_read_bytes']/c['cold_bytes']-1)<=.01 if c['cold_bytes'] else c['dram_read_bytes']/c['logical_bytes']<=.01)
    checks=dict(all_precision_and_process=all(all(c['checks'].values()) for c in cells.values()),all_counters=all(all(c['checks'].values()) for c in counters.values()))
    for name in ('sm_clock_hz','lts_clock_hz'):
        clocks=[c[name] for c in counters.values()];checks[name+'_stable3pct']=min(clocks)>0 and max(clocks)/min(clocks)-1<=.03
    contrasts=[]
    for hot in (4096,16384):
        for process in range(2):
            lat=[cells[m,hot,process]['latency'] for m in range(4)];h,m,c,only=lat
            denominator=c['median']-h['median']
            if denominator<=0:raise ValueError('nonpositive cold contrast')
            q=(m['median']-h['median'])/denominator
            ci=[x['confidence95'] for x in lat];lower=(ci[1]['lower_ns']-ci[0]['upper_ns'])/(ci[2]['upper_ns']-ci[0]['lower_ns'])
            join=ci[1]['lower_ns']>=.97*max(ci[0]['lower_ns'],ci[3]['lower_ns'])
            contrasts.append(dict(hot_bytes=hot,process=process,cold_fraction=1024/(hot+1024),q_mixed=q,conservative_q_lower95=lower,all_return_lower_bound_passed=join,distinguishes_mean_only=(lower>=2*1024/(hot+1024)) if hot==4096 else None))
    checks['all_join_bounds']=all(c['all_return_lower_bound_passed'] for c in contrasts)
    checks['both_designated_mean_only_distinctions']=all(c['distinguishes_mean_only'] for c in contrasts if c['hot_bytes']==4096)
    return dict(all_return_join_evidence_qualified=all(checks.values()),checks=checks,cells=list(cells.values()),counters=list(counters.values()),contrasts=contrasts,production_duration_consumed=False,forward_parameter_published=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('all_return_join_evidence_qualified','checks','contrasts')}))
