"""Frozen register-observer qualification; no operator time or Forward price."""
import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
import statistics


def poll_loop_proof(sass):
    proofs=[]
    for chunk in sass.split('Function :')[1:]:
        name=chunk.splitlines()[0].strip();instructions=[]
        for line in chunk.splitlines():
            m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+([^;]+);',line)
            if m:instructions.append((int(m[1],16),' '.join(m[2].split())))
        phase=[i for i,(a,t) in enumerate(instructions) if 'SYNCS.PHASECHK' in t]
        if not phase:continue
        if len(phase)!=2:raise ValueError('initial arming check and one repeated check required')
        arming,check=phase;loops=[]
        for j,(address,text) in enumerate(instructions):
            m=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA(?:\.U)? (0x[0-9a-f]+)',text)
            if m and int(m[1],16)<=instructions[check][0]<=address and int(m[1],16)<address:
                begin=[i for i,(a,t) in enumerate(instructions) if a==int(m[1],16)]
                if len(begin)==1:loops.append((begin[0],j))
        if not loops:raise ValueError('closed polling backedge required')
        begin,end=min(loops,key=lambda x:x[1]-x[0]);body=instructions[begin:end+1]
        if arming>=begin or not any('STS' in t for a,t in instructions[arming:begin]):
            raise ValueError('initial pending check must precede shared producer release')
        for index,(address,text) in enumerate(body):
            opcode=text.split()[1] if text.startswith('@') else text.split()[0]
            if (opcode.startswith(('LD','ST','ATOM','CALL','RET','BRX','BRA.DIV'))
                    or 'SR_GLOBALTIMER' in text
                    or (opcode.startswith('BRA') and index!=len(body)-1)):
                raise ValueError('polling must remain register-only with one uniform backedge')
        if not any('SR_CLOCKLO' in t for a,t in body):raise ValueError('polling clock required')
        guards=[]
        by_address={a:i for i,(a,t) in enumerate(instructions)}
        for j,(address,text) in enumerate(instructions):
            m=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA(?:\.U)? (0x[0-9a-f]+)',text)
            if m is None or int(m[1],16)>=address or int(m[1],16) not in by_address:
                continue
            i=by_address[int(m[1],16)];candidate=instructions[i:j+1]
            if (len(candidate)<=8 and any(re.search(r'\bLDS(?:\.|\s)',t) for a,t in candidate)
                    and not any('LDGSTS' in t or 'PHASECHK' in t for a,t in candidate)
                    and any('SR_CLOCKLO' in t for a,t in instructions[j+1:j+9])):
                guards.append([dict(address=hex(a),instruction=t) for a,t in candidate])
        if len(guards)!=1:raise ValueError('producer shared-flag guard before start clock required')
        proofs.append(dict(function=name,instructions=len(body),begin=hex(body[0][0]),
            end=hex(body[-1][0]),producer_start_guard=guards[0],
            body=[dict(address=hex(a),instruction=t) for a,t in body]))
    if len(proofs)!=6:raise ValueError('all six physical observer bodies must close')
    return proofs


def confidence(values, tail_denominator=40):
    a=sorted(values);n=len(a)
    ranks=[k for k in range(1,n//2+1)
           if tail_denominator*sum(math.comb(n,i) for i in range(k))<=2**n]
    if not ranks:raise ValueError('insufficient median confidence samples')
    k=ranks[-1];return a[k-1],a[n-k]


def fixture(item,trials):
    m=item['measurement']
    if m['exit_code']:raise ValueError('failed measurement')
    rows=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"B_bytes":')]
    if not rows or (trials==63 and len(rows)!=1):raise ValueError('complete numerical fixture required')
    b,g,l=(item[n] for n in ('B_bytes','grid','lookahead'))
    for c in rows:
        if (c['B_bytes'],c['grid'],c['lookahead'],c['rounds'],c['threads'],c['producer_threads'],
            c['shared_bytes'],c['resident_ctas'],c['distinct_sms'],c['all_lane_checksum_pass'],
            c['first_poll_pending_all'],c['start_policy'])!=(b,g,l,127,288,256,6*b,2 if b==16384 else 1,132,True,True,'observer_pending_before_producer_release'):
            raise ValueError('changed physical/numerical/initial-observer identity')
        if len(c['samples'])!=trials:raise ValueError('missing samples')
        for s in c['samples']:
            if any(not isinstance(s[n],(int,float)) or not math.isfinite(s[n]) or s[n]<=0
                   for n in ('first_ns','all_ns','enqueue_ns','max_poll_ns','calibration_mhz',
                             'min_cta_calibration_mhz','max_cta_calibration_mhz')):
                raise ValueError('invalid sample')
            if s['first_ns']>s['all_ns']:raise ValueError('invalid completion order')
    return rows[0]


def analyze(raw,plan,*,control_only=False):
    if raw['plan']!=plan:raise ValueError('changed frozen protocol')
    if raw['status'] not in (('screen_complete','observer_screen_failed') if control_only else ('complete',)):
        raise ValueError('incomplete requested capture stage')
    physical={tuple(c) for c in plan['control_cells' if control_only else 'physical_cells']}
    expected={(b,g,l,p) for b,g,l in physical for p in (0,1)}
    timing={};samples={};counters={}
    for item in raw['timing']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','process'))
        if key not in expected or key in timing:raise ValueError('unknown/duplicate timing')
        c=fixture(item,63);data={};checks={}
        for name in ('first_ns','all_ns','enqueue_ns','max_poll_ns'):
            values=[s[name] for s in c['samples']];samples[key,name]=values
            median=statistics.median(values);lo,hi=confidence(values)
            data[name]=dict(median=median,confidence95=[lo,hi])
            if name in ('first_ns','all_ns'):
                checks[name+'_precision3pct']=max(median-lo,hi-median)/median<=.03
        checks['observer_poll10pct']=data['max_poll_ns']['confidence95'][1]<=.1*data['first_ns']['confidence95'][0]
        checks['all_cta_clock_calibration3pct']=all(
            abs(s[n]/1980-1)<=.03 for s in c['samples']
            for n in ('min_cta_calibration_mhz','max_cta_calibration_mhz'))
        timing[key]=dict(identity=key,summaries=data,checks=checks)
    if set(timing)!=expected:raise ValueError('missing timing identity')
    for (b,g,l,p),c in timing.items():
        for name in ('first_ns','all_ns'):
            a=c['summaries'][name]['median'];other=timing[b,g,l,1-p]['summaries'][name]['median']
            c['checks'][name+'_process3pct']=max(a,other)/min(a,other)-1<=.03
    for item in raw['counters']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','process'))
        if key not in expected or key in counters:raise ValueError('unknown/duplicate counter')
        fixture(item,1);lines=item['measurement']['stdout'].splitlines()
        start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        b,g,l,p=key;logical=g*b*(4+127*l)
        up=32*value('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum')
        fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        sm=value('sm__cycles_elapsed.avg.per_second');lts=value('lts__cycles_elapsed.avg.per_second')
        counters[key]=dict(identity=key,logical_bytes=logical,upstream_bytes=up,tex_bytes=tex,
            dram_bytes=dram,sm_clock_hz=sm,lts_clock_hz=lts,checks=dict(
                exact_ctas=value('sm__ctas_launched.sum')==g,
                exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==logical/512,
                upstream1pct=abs(up/logical-1)<=.01,tex1pct=abs(tex/logical-1)<=.01,
                source_units1pct=abs(total-tex-fabric)/logical<=.01,
                dram1pct=dram/logical<=.01,sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(counters)!=expected:raise ValueError('missing counter identity')
    checks=dict(all_timing_and_observer=all(all(c['checks'].values()) for c in timing.values()),
                all_counters=all(all(c['checks'].values()) for c in counters.values()))
    for name in ('sm_clock_hz','lts_clock_hz'):
        v=[c[name] for c in counters.values()];checks[name+'_stable3pct']=min(v)>0 and max(v)/min(v)-1<=.03
    contrasts=[]
    if not control_only:
        for b,g in sorted({(c[0],c[1]) for c in physical}):
            for l,p in itertools.product((2,4),(0,1)):
                # Three 98 1/3% exact median intervals give a conservative
                # >=95% joint interval by the union bound. No bootstrap seed,
                # source timing subtraction or fitted Q is introduced.
                intervals=[confidence(samples[k,n],120) for k,n in (
                    ((b,g,l,p),'first_ns'),((b,g,1,p),'first_ns'),((b,g,l,p),'all_ns'))]
                if intervals[2][0]<=intervals[1][1]:
                    contrasts.append(dict(identity=[b,g,l,p],distinguished=False,first_priority=False,equal_burst_slow_response=False))
                    continue
                q=[(f-base)/(end-base) for f,base,end in itertools.product(*intervals)]
                contrasts.append(dict(identity=[b,g,l,p],distinguished=True,Q95=[min(q),max(q)],
                    first_priority=max(q)<=.25,equal_burst_slow_response=min(q)>=.75))
    qualified=all(checks.values())
    verdict=('first_priority' if all(c['first_priority'] for c in contrasts) else
             'equal_burst_slow_response' if all(c['equal_burst_slow_response'] for c in contrasts) else
             'inconclusive') if qualified and not control_only else 'screen_pass' if qualified else 'ineligible'
    return dict(measurement_qualified=qualified,full_capture_qualified=qualified and not control_only,
        verdict=verdict,checks=checks,timing=list(timing.values()),counters=list(counters.values()),
        contrasts=contrasts,production_or_evaluation_duration_consumed=False,
        forward_parameter_or_policy_adopted=False,selection=plan['selection'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--control-only',action='store_true');a=p.parse_args()
    result=analyze(json.loads(a.raw.read_text()),json.loads(a.plan.read_text()),control_only=a.control_only)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('measurement_qualified','full_capture_qualified','verdict','checks','contrasts')}))
