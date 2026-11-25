"""Validate independent group returns; never publish a measured timing price."""
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
        raise ValueError('failed measurement is ineligible')
    candidates=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"B_bytes":')]
    if not candidates or (trials==63 and len(candidates)!=1):
        raise ValueError('one consistent numerical fixture required')
    b,g,l=item['B_bytes'],item['grid'],item['lookahead']
    # Application replay prints one fixture per pass. Every pass must retain
    # the exact program and pass its numerical checks; profiler-perturbed
    # timings need not agree and are never used for the response contrasts.
    for c in candidates:
        if (c['B_bytes'],c['grid'],c['lookahead'],c['rounds'],c['threads'],c['shared_bytes'],c['resident_ctas'],
            c['warm_kernels'],c['warm_groups_per_kernel'],c['distinct_sms'],c['all_lane_checksum_pass'])!=(b,g,l,127,256,6*b,2 if b==16384 else 1,6,4,132,True):
            raise ValueError('changed numerical/resource/source program identity')
        if len(c['samples'])!=trials:
            raise ValueError('missing frozen samples')
        for s in c['samples']:
            f,a,gap=s['first_ns'],s['all_ns'],s['issue_gap_ns']
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (f,a,gap)) or not 0<=gap<=f<=a or f<=0:
                raise ValueError('nonfinite or impossible group-return sample')
    return candidates[0]

def interval(values,rng):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty samples required')
    bootstrap=np.median(rng.choice(a,size=(1200,len(a)),replace=True),axis=1)
    return float(np.median(a)),[float(x) for x in np.quantile(bootstrap,[.025,.975])],bootstrap

def analyze(raw):
    if raw['status']!='complete' or len(raw['timing'])!=18 or len(raw['counters'])!=18:
        raise ValueError('all frozen timing/counter cells required; partial captures remain ineligible')
    if {tuple(x) for x in raw['physical_cells']}!={x[:3] for x in EXPECTED}:
        raise ValueError('changed frozen cell declaration')
    rng=np.random.default_rng(20261108);timing={};counter={};boot={}
    for item in raw['timing']:
        key=item['B_bytes'],item['grid'],item['lookahead'],item['process']
        if key not in EXPECTED or key in timing:
            raise ValueError('unknown or duplicate timing identity')
        c=fixture(item,63);data={};checks={}
        for name in ('first_ns','all_ns','issue_gap_ns'):
            median,ci,draws=interval([s[name] for s in c['samples']],rng)
            data[name]=dict(median=median,confidence95=ci)
            boot[key,name]=draws
            if name!='issue_gap_ns':checks[name+'_median_ci3pct']=(ci[1]-ci[0])/(2*median)<=.03
        timing[key]=dict(identity=key,summaries=data,checks=checks)
    for item in raw['counters']:
        key=item['B_bytes'],item['grid'],item['lookahead'],item['process']
        if key not in EXPECTED or key in counter:
            raise ValueError('unknown or duplicate counter identity')
        fixture(item,1);lines=item['measurement']['stdout'].splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def metric(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        b,g,l,p=key;logical=g*b*(4+127*l)
        upstream=32*metric('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*metric('lts__t_sectors_srcunit_tex_op_read.sum')
        fabric=32*metric('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*metric('lts__t_sectors_op_read.sum');dram=metric('dram__bytes_read.sum')
        sm=metric('sm__cycles_elapsed.avg.per_second');lts=metric('lts__cycles_elapsed.avg.per_second')
        counter[key]=dict(identity=key,logical_bytes=logical,upstream_bytes=upstream,tex_bytes=tex,dram_bytes=dram,
          sm_clock_hz=sm,lts_clock_hz=lts,checks=dict(exact_ctas=metric('sm__ctas_launched.sum')==g,
          exact_ldgsts=metric('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==logical/512,
          upstream1pct=abs(upstream/logical-1)<=.01,tex1pct=abs(tex/logical-1)<=.01,
          dram1pct=dram/logical<=.01,source_units1pct=abs(total-tex-fabric)/logical<=.01,
          sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(timing)!=EXPECTED or set(counter)!=EXPECTED:
        raise ValueError('missing frozen identity')
    checks={}
    for clock in ('sm_clock_hz','lts_clock_hz'):
        values=[x[clock] for x in counter.values()];checks[clock+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    checks['all_timing_precision']=all(all(x['checks'].values()) for x in timing.values())
    checks['all_counter_checks']=all(all(x['checks'].values()) for x in counter.values())
    for b,g in PHYSICAL:
        for l in (1,2,4):
            for name in ('first_ns','all_ns'):
                med=[timing[b,g,l,p]['summaries'][name]['median'] for p in (0,1)]
                checks[f'process_{b}_{g}_{l}_{name}_3pct']=max(med)/min(med)-1<=.03
    contrasts=[]
    for b,g in PHYSICAL:
        for l in (2,4):
            for p in (0,1):
                base=boot[(b,g,1,p),'first_ns'];first=boot[(b,g,l,p),'first_ns'];end=boot[(b,g,l,p),'all_ns'];gap=boot[(b,g,l,p),'issue_gap_ns']
                denominator=end-base
                if (denominator<=0).any():raise ValueError('all-return contrast does not distinguish the lookahead')
                q=(first-base)/denominator;relative_gap=gap/denominator
                qci=[float(x) for x in np.quantile(q,[.025,.975])];gapci=[float(x) for x in np.quantile(relative_gap,[.025,.975])]
                contrasts.append(dict(B_bytes=b,grid=g,lookahead=l,process=p,Q95=qci,relative_issue_gap95=gapci,
                    supports_first_group_priority=qci[1]<=.25,supports_equal_burst_slow_response=qci[0]>=.75,observer_gap_pass=gapci[1]<=.20))
    checks['all_observer_gap_checks']=all(x['observer_gap_pass'] for x in contrasts)
    qualified=all(checks.values())
    verdict=('first_group_priority' if all(x['supports_first_group_priority'] for x in contrasts) else
             'equal_burst_slow_response' if all(x['supports_equal_burst_slow_response'] for x in contrasts) else 'inconclusive') if qualified else 'ineligible'
    return dict(production_or_evaluation_duration_consumed=False,new_timing_price_published=False,checks=checks,
        measurement_qualified=qualified,verdict=verdict,timing=list(timing.values()),counters=list(counter.values()),contrasts=contrasts,
        scope='Independent private steady-loop warp group returns only; no native scheduling,rate,Qparameter or Forward policy published',selection=raw['selection'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('measurement_qualified','verdict','checks','contrasts')}))
