"""Frozen byte-only progress qualification; no production duration consumed."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

def confidence(values):
    a=sorted(values);n=len(a)
    ranks=[k for k in range(1,n//2+1) if 40*sum(math.comb(n,i) for i in range(k))<=2**n]
    if not ranks:raise ValueError('insufficient median confidence samples')
    k=ranks[-1];return a[k-1],a[n-k]

def fixture(item,trials):
    m=item['measurement']
    if m['exit_code']!=0:raise ValueError('failed measurement')
    candidates=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"stride":')]
    if not candidates or (trials==63 and len(candidates)!=1):raise ValueError('complete numerical fixture required')
    s,g,f=(item[k] for k in ('stride','grid','fanout'))
    for c in candidates:
        if (c['stride'],c['grid'],c['fanout'],c['panel_bytes'],c['a_destination_bytes'],c['valid_a_bytes_per_round'],
            c['threads'],c['slots'],c['shared_bytes'],c['resident_ctas'],c['rounds'],c['unique_b_bytes'],
            c['all_lane_checksum_pass'],c['distinct_sms'])!=(s,g,f,16384,16384,128,256,3,98304,2,s//128,
                math.ceil(g/f)*128*s,True,132):raise ValueError('changed physical/numerical identity')
        if len(c['samples'])!=trials:raise ValueError('missing samples')
        for r in c['samples']:
            if any(not isinstance(r[n],(int,float)) or not math.isfinite(r[n]) or r[n]<=0
                   for n in ('body_ns','median_cta_ns','weighted_mhz')):raise ValueError('invalid sample')
    return candidates[0]

def analyze(raw,plan_bytes,*,control_only=False,cache_accounted=False):
    plan=json.loads(plan_bytes)
    expected_kind=('independent_two_resident_cache_accounted_progress_plan' if cache_accounted
                   else 'independent_two_resident_cold_copy_plan')
    if plan['kind']!=expected_kind:raise ValueError('explicit matching accounting protocol required')
    if raw['plan']!=plan or raw['plan_sha256']!=hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError('changed frozen plan identity')
    valid_status=('screen_complete','mechanism_screen_failed') if control_only else ('complete',)
    if raw['status'] not in valid_status:
        raise ValueError('incomplete requested capture stage')
    physical={tuple(c) for c in plan['counter_cells' if control_only else 'physical_cells']}
    forecasts={}
    if cache_accounted:
        forecasts={(c['stride'],c['grid'],c['fanout']):c['predicted_read_bytes'] for c in plan['counter_read_forecasts']}
        if set(forecasts)!={tuple(c) for c in plan['counter_cells']}:raise ValueError('complete source-cache forecasts required')
    timing={};counters={}
    for item in raw['timing']:
        key=item['stride'],item['grid'],item['fanout'],item['process']
        if key[:3] not in physical or key[3] not in (0,1) or key in timing:raise ValueError('unknown/duplicate timing cell')
        c=fixture(item,63);values=[x['body_ns'] for x in c['samples']]
        med=statistics.median(values);lo,hi=confidence(values)
        clock=statistics.median(x['weighted_mhz'] for x in c['samples'])
        timing[key]=dict(identity=key,median_body_ns=med,confidence95=[lo,hi],weighted_mhz=clock,
            checks=dict(precision3pct=max(med-lo,hi-med)/med<=.03,nominal_clock3pct=abs(clock/1980-1)<=.03))
    if set(timing)!={(s,g,f,p) for s,g,f in physical for p in (0,1)}:raise ValueError('missing timing identity')
    for (s,g,f,p),c in timing.items():
        a,b=c['median_body_ns'],timing[s,g,f,1-p]['median_body_ns']
        c['checks']['process3pct']=max(a,b)/min(a,b)-1<=.03
    for item in raw['counters']:
        key=item['stride'],item['grid'],item['fanout'];s,g,f=key
        if key not in {tuple(c) for c in plan['counter_cells']} or key in counters:raise ValueError('unknown/duplicate counter')
        fixture(item,1);lines=item['measurement']['stdout'].splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        rounds=s//128;logical=g*rounds*(16384+128);unique=math.ceil(g/f)*128*s+s
        upstream=32*value('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum');fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        sm=value('sm__cycles_elapsed.avg.per_second');lts=value('lts__cycles_elapsed.avg.per_second')
        checks=dict(
                exact_ctas=value('sm__ctas_launched.sum')==g,
                exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==g*rounds*64,
                upstream1pct=abs(upstream/logical-1)<=.01,
                downstream_conservation=unique*.99<=tex<=logical*1.01,
                source_units1pct=abs(total-tex-fabric)/logical<=.01,sm_nominal3pct=abs(sm/1.98e9-1)<=.03)
        if cache_accounted:
            checks['source_cache_read1pct']=abs(dram/forecasts[key]-1)<=.01
        else:
            checks['unique_dram1pct']=abs(dram/unique-1)<=.01
        counters[key]=dict(identity=key,logical_bytes=logical,unique_bytes=unique,upstream_bytes=upstream,
            downstream_tex_bytes=tex,dram_bytes=dram,sm_clock_hz=sm,lts_clock_hz=lts,
            checks=checks,legacy_unique_read_check=abs(dram/unique-1)<=.01)
        if cache_accounted:counters[key]['source_cache_read_forecast_bytes']=forecasts[key]
    if set(counters)!={tuple(c) for c in plan['counter_cells']}:raise ValueError('missing counter identity')
    checks=dict(all_timing=all(all(x['checks'].values()) for x in timing.values()),all_counters=all(all(x['checks'].values()) for x in counters.values()))
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[x[name] for x in counters.values()];checks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    contrasts=[]
    for s in sorted({c[0] for c in plan['counter_cells']}):
        g=min(c[1] for c in plan['counter_cells'] if c[0]==s)
        for p in (0,1):
            a=timing[s,g,1,p];b=timing[s,g,2,p]
            lower=b['confidence95'][0]/a['confidence95'][1]
            contrasts.append(dict(stride=s,grid=g,process=p,fanout2_over_1=b['median_body_ns']/a['median_body_ns'],
                conservative_lower95=lower,pending_request_distinction=lower>plan['mechanism_min_ratio']))
    checks['all_pending_request_distinctions']=all(c['pending_request_distinction'] for c in contrasts)
    qualified=all(checks.values());points=[]
    if qualified and not control_only:
        for s,g,f in sorted(physical):
            elapsed=max(timing[s,g,f,p]['median_body_ns'] for p in (0,1))
            points.append(dict(stride=s,grid=g,fanout=f,logical_B_capacity_per_s=g*16384*(s//128)*1e9/elapsed))
    return dict(mechanism_screen_qualified=qualified,full_capture_qualified=qualified and not control_only,
        checks=checks,cells=list(timing.values()),counters=list(counters.values()),contrasts=contrasts,
        selected_points=points,production_or_evaluation_duration_consumed=False,selection=plan['selection'],
        accounting_protocol=plan['kind'],legacy_unique_read_all_pass=all(c['legacy_unique_read_check'] for c in counters.values()))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--control-only',action='store_true');p.add_argument('--cache-accounted',action='store_true');a=p.parse_args()
    out=analyze(json.loads(a.raw.read_text()),a.plan.read_bytes(),control_only=a.control_only,cache_accounted=a.cache_accounted)
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('mechanism_screen_qualified','full_capture_qualified','checks','contrasts')}))
