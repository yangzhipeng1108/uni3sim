"""Independent CTA completion joins; no timing price or Forward parameter."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

PHYSICAL=((16384,132),(16384,264),(32768,132))
EXPECTED={(b,g,l,p) for b,g in PHYSICAL for l in (1,2,4) for p in (0,1)}

def fixture(item,trials):
    m=item['measurement']
    if m['exit_code']!=0:
        raise ValueError('failed capture is ineligible')
    candidates=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"B_bytes":')]
    if not candidates or (trials==63 and len(candidates)!=1):
        raise ValueError('complete fixture required')
    b,g,l=item['B_bytes'],item['grid'],item['lookahead']
    for c in candidates:
        if (c['B_bytes'],c['grid'],c['lookahead'],c['rounds'],c['threads'],c['producer_threads'],
            c['shared_bytes'],c['resident_ctas'],c['distinct_sms'],c['all_lane_checksum_pass'])!=(
            b,g,l,127,288,256,6*b,2 if b==16384 else 1,132,True):
            raise ValueError('changed numerical/physical identity')
        if len(c['samples'])!=trials:
            raise ValueError('missing frozen samples')
        for s in c['samples']:
            f,a,e,poll=(s[n] for n in ('first_ns','all_ns','enqueue_ns','max_poll_ns'))
            if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in (f,a,e,poll)) or f>a:
                raise ValueError('invalid completion observation')
    return candidates[0]

def interval(values,rng):
    a=np.asarray(values,dtype=float)
    draws=np.median(rng.choice(a,size=(1200,len(a)),replace=True),axis=1)
    return float(np.median(a)),[float(x) for x in np.quantile(draws,[.025,.975])],draws

def analyze(raw):
    if raw['status']!='complete' or len(raw['timing'])!=18 or len(raw['counters'])!=18:
        raise ValueError('all 18 timing/counter cells required')
    if {tuple(x) for x in raw['physical_cells']}!={x[:3] for x in EXPECTED}:
        raise ValueError('changed frozen cells')
    rng=np.random.default_rng(20261108);timing={};counters={};boot={}
    for item in raw['timing']:
        key=item['B_bytes'],item['grid'],item['lookahead'],item['process']
        if key not in EXPECTED or key in timing:
            raise ValueError('unknown or duplicate timing identity')
        c=fixture(item,63);data={};checks={}
        for name in ('first_ns','all_ns','enqueue_ns','max_poll_ns'):
            median,ci,draws=interval([s[name] for s in c['samples']],rng)
            data[name]=dict(median=median,confidence95=ci);boot[key,name]=draws
            if name in ('first_ns','all_ns'):
                checks[name+'_median_ci3pct']=(ci[1]-ci[0])/(2*median)<=.03
        # Every CTA/round's longest polling interval bounds a missed event's
        # timestamp resolution. No enqueue-gap subtraction or fitted offset.
        checks['observer_poll10pct']=data['max_poll_ns']['confidence95'][1]<=.10*data['first_ns']['confidence95'][0]
        timing[key]=dict(identity=key,summaries=data,checks=checks)
    for item in raw['counters']:
        key=item['B_bytes'],item['grid'],item['lookahead'],item['process']
        if key not in EXPECTED or key in counters:
            raise ValueError('unknown or duplicate counter identity')
        fixture(item,1);lines=item['measurement']['stdout'].splitlines()
        start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def metric(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        b,g,l,p=key;logical=g*b*(4+127*l)
        up=32*metric('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*metric('lts__t_sectors_srcunit_tex_op_read.sum')
        fabric=32*metric('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*metric('lts__t_sectors_op_read.sum');dram=metric('dram__bytes_read.sum')
        sm=metric('sm__cycles_elapsed.avg.per_second');lts=metric('lts__cycles_elapsed.avg.per_second')
        counters[key]=dict(identity=key,logical_bytes=logical,upstream_bytes=up,tex_bytes=tex,dram_bytes=dram,
            sm_clock_hz=sm,lts_clock_hz=lts,checks=dict(exact_ctas=metric('sm__ctas_launched.sum')==g,
                exact_ldgsts=metric('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==logical/512,
                upstream1pct=abs(up/logical-1)<=.01,tex1pct=abs(tex/logical-1)<=.01,
                dram1pct=dram/logical<=.01,source_units1pct=abs(total-tex-fabric)/logical<=.01,
                sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(timing)!=EXPECTED or set(counters)!=EXPECTED:
        raise ValueError('missing frozen identity')
    checks={'all_timing_and_observer':all(all(x['checks'].values()) for x in timing.values()),
            'all_counters':all(all(x['checks'].values()) for x in counters.values())}
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[x[name] for x in counters.values()]
        checks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    for b,g in PHYSICAL:
        for l in (1,2,4):
            for name in ('first_ns','all_ns'):
                values=[timing[b,g,l,p]['summaries'][name]['median'] for p in (0,1)]
                checks[f'process_{b}_{g}_{l}_{name}_3pct']=max(values)/min(values)-1<=.03
    contrasts=[]
    for b,g in PHYSICAL:
        for l in (2,4):
            for p in (0,1):
                first=boot[(b,g,l,p),'first_ns'];base=boot[(b,g,1,p),'first_ns'];end=boot[(b,g,l,p),'all_ns']
                den=end-base
                if (den<=0).any():raise ValueError('undistinguished all-completion contrast')
                ci=[float(x) for x in np.quantile((first-base)/den,[.025,.975])]
                contrasts.append(dict(B_bytes=b,grid=g,lookahead=l,process=p,Q95=ci,
                    first_priority=ci[1]<=.25,equal_burst_slow_response=ci[0]>=.75))
    qualified=all(checks.values())
    verdict=('first_priority' if all(x['first_priority'] for x in contrasts) else
             'equal_burst_slow_response' if all(x['equal_burst_slow_response'] for x in contrasts) else
             'inconclusive') if qualified else 'ineligible'
    return dict(measurement_qualified=qualified,verdict=verdict,checks=checks,timing=list(timing.values()),
        counters=list(counters.values()),contrasts=contrasts,production_or_evaluation_duration_consumed=False,
        forward_parameter_or_policy_adopted=False,selection=raw['selection'],
        scope='mbarrier-tagged private copy panels with an independent ninth warp; no native matrix phase or rate')

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    out=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('measurement_qualified','verdict','checks','contrasts')}))
