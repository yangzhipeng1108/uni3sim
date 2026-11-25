"""Qualify exact new cold-request points without production timings."""
import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def analyze(raw):
    if raw['status']!='complete':raise ValueError('coalesced capture incomplete')
    physical={tuple(c) for c in raw['physical_cells']}
    if len(physical)!=28:raise ValueError('changed frozen physical point set')
    cells={}
    for rotation in raw['rotations']:
        process=rotation['process'];measurement=rotation['measurement']
        if process not in (0,1) or measurement['exit_code']:raise ValueError('failed process')
        rows=list(csv.DictReader(io.StringIO(measurement['stdout'])))
        if len(rows)!=28*126:raise ValueError('incomplete sweep')
        for row in rows:
            stride,grid,fanout=(int(row[n]) for n in ('row_stride','grid','request_fanout'))
            if (stride,grid,fanout) not in physical:raise ValueError('unknown physical point')
            if (int(row['panel_bytes']),int(row['slots']),int(row['threads']),int(row['row_bytes']),int(row['rounds']),int(row['companion_mode']),int(row['shared_reservation_bytes']),int(row['resident_ctas_per_sm']))!=(32768,4,256,128,stride//128,5,196608,1):raise ValueError('changed copy program')
            if row['tier']!='cold_shared_requests' or row['mode'] not in ('copy','control') or int(row['distinct_sms'])!=132:raise ValueError('changed traffic or SM coverage')
            key=stride,grid,fanout,process,row['mode']
            if key not in cells:cells[key]=[]
            cells[key].append(row)
    if set(cells)!={(s,g,f,p,m) for s,g,f in physical for p in (0,1) for m in ('copy','control')}:raise ValueError('incomplete physical/process coverage')
    summaries={}
    for key,rows in cells.items():
        if len(rows)!=63 or {int(r['trial']) for r in rows}!=set(range(63)):raise ValueError('incomplete sample sequence')
        s,g,f,p,mode=key
        if mode!='copy':continue
        values=[float(r['body_ns']) for r in rows]
        if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid body time')
        median=statistics.median(values);ci=median_confidence_interval(values);clock=statistics.median(float(r['weighted_mhz']) for r in rows)
        summaries[s,g,f,p]=dict(stride=s,grid=g,fanout=f,process=p,median_body_ns=median,confidence95=ci,clock64_mhz=clock,checks=dict(
            median_precision3pct=max(median-ci['lower_ns'],ci['upper_ns']-median)/median<=.03,
            nominal_clock3pct=abs(clock/1980-1)<=.03))
    for (s,g,f,p),c in summaries.items():
        a,b=c['median_body_ns'],summaries[s,g,f,1-p]['median_body_ns'];c['checks']['process3pct']=max(a,b)/min(a,b)-1<=.03
    counters={}
    for item in raw['counters']:
        key=item['stride'],item['grid'],item['fanout'];s,g,f=key;measurement=item['measurement']
        if key in counters or measurement['exit_code']:raise ValueError('duplicate or failed counter')
        lines=measurement['stdout'].splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        record,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            v=float(record[name].replace(',',''))
            if not math.isfinite(v) or v<0:raise ValueError('invalid hardware counter')
            return v
        rounds=s//128;logical=g*rounds*(32768+128);unique=math.ceil(g/f)*rounds*32768+s
        tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum');fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum');total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        counters[key]=dict(stride=s,grid=g,fanout=f,logical_bytes=logical,unique_bytes=unique,tex_bytes=tex,l2_bytes=total,dram_bytes=dram,sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),checks=dict(
            exact_ctas=value('sm__ctas_launched.sum')==g,
            exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==g*rounds*96,
            tex1pct=abs(tex/logical-1)<=.01,dram1pct=abs(dram/unique-1)<=.01,
            source_units1pct=abs(total-tex-fabric)/logical<=.01))
    if set(counters)!={tuple(c) for c in raw['counter_cells']}:raise ValueError('incomplete frozen counter set')
    clocks={}
    for field in ('sm_clock_hz','lts_clock_hz'):
        values=[c[field] for c in counters.values()];clocks[field+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    clocks['direct_sm_nominal3pct']=all(abs(c['sm_clock_hz']/1.98e9-1)<=.03 for c in counters.values())
    points=[]
    for s,g,f in sorted(physical):
        pair=[summaries[s,g,f,p] for p in (0,1)]
        class_ok=all(clocks.values()) and all(all(c['checks'].values()) for (stride,grid,fanout),c in counters.items() if stride==s)
        eligible=class_ok and all(all(c['checks'].values()) for c in pair)
        points.append(dict(stride=s,grid=g,fanout=f,class_qualified=class_ok,eligible=eligible,
            logical_B_capacity_per_s=32768*g*(s//128)*1e9/max(c['median_body_ns'] for c in pair)))
    contrasts=[]
    for s in sorted({c[0] for c in physical}):
        g=max(c[1] for c in physical if c[0]==s)
        for p in (0,1):contrasts.append(dict(stride=s,grid=g,process=p,fanout2_over_1=summaries[s,g,2,p]['median_body_ns']/summaries[s,g,1,p]['median_body_ns']))
    return dict(all_points_qualified=all(p['eligible'] for p in points),clock_checks=clocks,points=points,cells=list(summaries.values()),counters=list(counters.values()),contrasts=contrasts,production_duration_consumed=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_points_qualified','clock_checks','points','contrasts')}))
