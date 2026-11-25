"""One-shot interval-censored readiness census; no Forward rate or price."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re


def query_proof(sass):
    proofs=[]
    for chunk in sass.split('Function :')[1:]:
        instructions=[]
        for line in chunk.splitlines():
            m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+([^;]+);',line)
            if m:instructions.append((int(m[1],16),' '.join(m[2].split())))
        queries=[i for i,(_,t) in enumerate(instructions) if 'SYNCS.PHASECHK' in t]
        if not queries:continue
        if len(queries)!=2:raise ValueError('one initial and one timed query required')
        initial,query=queries
        clocks=[i for i,(_,t) in enumerate(instructions) if 'SR_CLOCKLO' in t]
        before=max(i for i in clocks if i<query);after=min(i for i in clocks if i>query)
        body=instructions[before:after+1]
        if len(body)>8:raise ValueError('query clock bracket expanded')
        for _,text in body:
            op=text.split()[1] if text.startswith('@') else text.split()[0]
            if op.startswith(('LD','ST','ATOM','BRA','CALL','RET','YIELD','WARPSYNC')):
                raise ValueError('query bracket must have no memory, branch or yield')
        by_address={a:i for i,(a,_) in enumerate(instructions)}
        delays=[]
        for j in range(initial+1,before):
            address,text=instructions[j]
            m=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA(?:\.U)? (0x[0-9a-f]+)',text)
            if m is None:continue
            target=int(m[1],16)
            if target not in by_address or target>=address:continue
            i=by_address[target]
            if i<=initial:continue
            loop=instructions[i:j+1]
            if not any('IADD3' in t for _,t in loop):continue
            if any(re.search(r'\b(?:YIELD|SYNCS|LDS|STS|LDG|STG|CALL|WARPSYNC)\b',t) for _,t in loop):
                raise ValueError('finite delay must have only arithmetic and loop control')
            delays.append(loop)
        if len(delays)!=1:raise ValueError('exactly one finite arithmetic delay required')
        between=instructions[initial:before]
        if not any('VOTE' in t for _,t in between) or not any('STS' in t for _,t in between):
            raise ValueError('initial full-warp check must precede producer release')
        proofs.append(dict(function=chunk.splitlines()[0].strip(),
            query_bracket=[dict(address=hex(a),instruction=t) for a,t in body],
            finite_delay=[dict(address=hex(a),instruction=t) for a,t in delays[0]],
            readiness_queries_per_round=1,readiness_polling_backedge=False))
    if len(proofs)!=6:raise ValueError('all six physical bodies required')
    return proofs


def readiness_intervals(samples, *, grid, lookahead):
    """Conservative median-launch CDF crossing, under stationary repeats.

    Query times may vary: every tier uses the minimum before and maximum
    after over ALL CTAs/rounds/trials. No late query or CTA is filtered.
    Min/max of17 independent launch fractions bounds their median with
    failure <=2/2**17. The full9cell x2process x65tier x2event family has
    union failure <=0.035706. This measures median launch spatial readiness,
    not the old observer's mean per-CTA response.
    """
    if len(samples)!=17:raise ValueError('17 independent launch samples required')
    low_clock=min(s['min_cta_calibration_mhz'] for s in samples)/1000
    high_clock=max(s['max_cta_calibration_mhz'] for s in samples)/1000
    if not 0<low_clock<=high_clock:raise ValueError('invalid calibrated clocks')
    tiers=[]
    for t in range(65):
        rows=[s['rows'][t] for s in samples]
        for row in rows:
            if (len(row)!=7 or any(isinstance(v,bool) or not isinstance(v,int) for v in row)
                    or row[0]!=t or row[1]<0 or row[2]<row[1]
                    or not 0<=row[4]<=row[3]<=row[5] or row[5]!=4*grid
                    or row[6]<0 or row[6]>row[2]-row[1]):
                raise ValueError('invalid complete query tier')
            if lookahead==1 and row[3]!=row[4]:raise ValueError('one group must share first/all readiness')
        tiers.append(dict(tier=t,before_ns=min(r[1] for r in rows)/high_clock,
            after_ns=max(r[2] for r in rows)/low_clock,
            max_query_ns=max(r[6] for r in rows)/low_clock,
            first_fraction95=[min(r[3]/r[5] for r in rows),max(r[3]/r[5] for r in rows)],
            all_fraction95=[min(r[4]/r[5] for r in rows),max(r[4]/r[5] for r in rows)]))
    out={}
    for name in ('first','all'):
        lower=[q['before_ns'] for q in tiers if q[name+'_fraction95'][1]<.5]
        upper=[q['after_ns'] for q in tiers if q[name+'_fraction95'][0]>.5]
        interval=[max(lower),min(upper)] if lower and upper else None
        monotone=all(not(a[name+'_fraction95'][0]>.5 and b[name+'_fraction95'][1]<.5)
            for a,b in itertools.permutations(tiers,2) if a['after_ns']<b['before_ns'])
        valid=interval is not None and 0<interval[0]<=interval[1]
        out[name]=dict(response95_ns=interval,checks=dict(stationary_monotonicity=monotone,
            bracket_present_and_ordered=valid,
            bracket_resolution10pct=valid and interval[1]/interval[0]-1<=.1,
            one_query_width10pct=valid and max(q['max_query_ns'] for q in tiers)<=.1*interval[0]))
    return dict(events=out,tiers=tiers,checks=dict(
        all_cta_calibration_nominal3pct=low_clock>=1.98*.97 and high_clock<=1.98*1.03,
        all_event_brackets=all(all(e['checks'].values()) for e in out.values())))


def measurement_rows(item,trials):
    m=item['measurement']
    if m['exit_code']:raise ValueError('failed measurement')
    rows=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"B_bytes":')]
    if not rows or (trials!=1 and len(rows)!=trials):
        raise ValueError('complete launch samples required')
    b,g,l,p=(item[n] for n in ('B_bytes','grid','lookahead','process'))
    for i,s in enumerate(rows):
        if ((s['B_bytes'],s['grid'],s['lookahead'],s['threads'],s['shared_bytes'],s['resident_ctas'],
             s['rounds'],s['tiers'],s['trial'],s['reverse'],s['all_lane_checksum_pass'],
             s['first_query_pending_all'],s['distinct_sms'])!=
            (b,g,l,288,6*b,2 if b==16384 else 1,260,65,0 if trials==1 else i,p,True,True,132)):
            raise ValueError('changed query/source/numerical identity')
        if len(s['rows'])!=65:raise ValueError('all deadline tiers required')
        for n in ('min_cta_calibration_mhz','max_cta_calibration_mhz'):
            if not isinstance(s[n],(float,int)) or not math.isfinite(s[n]) or s[n]<=0:
                raise ValueError('invalid calibration')
    return rows


def analyze(raw,plan,*,control_only=False):
    if raw['plan']!=plan:raise ValueError('changed frozen protocol')
    if raw['status'] not in (('screen_complete','deadline_screen_failed','screen_analysis_failed') if control_only else ('complete',)):
        raise ValueError('incomplete capture stage')
    physical={tuple(c) for c in plan['control_cells' if control_only else 'physical_cells']}
    expected={(b,g,l,p) for b,g,l in physical for p in (0,1)}
    cells={};counters={}
    for item in raw['timing']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','process'))
        if key not in expected or key in cells:raise ValueError('unknown/duplicate timing cell')
        rows=measurement_rows(item,17)
        cells[key]=readiness_intervals(rows,grid=key[1],lookahead=key[2])
        cells[key]['identity']=key
    if set(cells)!=expected:raise ValueError('missing timing cell')
    for item in raw['counters']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','process'))
        if key not in expected or key in counters:raise ValueError('unknown/duplicate counter cell')
        replay_fixtures=measurement_rows(item,1);lines=item['measurement']['stdout'].splitlines()
        start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        b,g,l,p=key;logical=g*b*(4+260*l)
        up=32*value('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum')
        fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        sm=value('sm__cycles_elapsed.avg.per_second');lts=value('lts__cycles_elapsed.avg.per_second')
        counters[key]=dict(identity=key,application_replay_fixture_count=len(replay_fixtures),logical_bytes=logical,upstream_bytes=up,tex_bytes=tex,
            dram_bytes=dram,sm_clock_hz=sm,lts_clock_hz=lts,checks=dict(
                exact_ctas=value('sm__ctas_launched.sum')==g,
                exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==logical/512,
                upstream1pct=abs(up/logical-1)<=.01,tex1pct=abs(tex/logical-1)<=.01,
                source_units1pct=abs(total-tex-fabric)/logical<=.01,dram1pct=dram/logical<=.01,
                sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(counters)!=expected:raise ValueError('missing counter cell')
    checks=dict(all_query_brackets=all(all(c['checks'].values()) for c in cells.values()),
        all_counters=all(all(c['checks'].values()) for c in counters.values()))
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[c[name] for c in counters.values()]
        checks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    for name in ('first','all'):
        comparisons=[]
        for b,g,l in physical:
            intervals=[cells[b,g,l,p]['events'][name]['response95_ns'] for p in (0,1)]
            comparisons.append(all(q is not None and q[0]>0 for q in intervals) and
                max(sum(q)/2 for q in intervals)/min(sum(q)/2 for q in intervals)-1<=.03)
        checks[name+'_process3pct']=all(comparisons)
    qualified=all(checks.values());contrasts=[]
    if qualified and not control_only:
        for b,g in sorted({(c[0],c[1]) for c in physical}):
            for l,p in itertools.product((2,4),(0,1)):
                f=cells[b,g,l,p]['events']['first']['response95_ns']
                base=cells[b,g,1,p]['events']['first']['response95_ns']
                end=cells[b,g,l,p]['events']['all']['response95_ns']
                if end[0]<=base[1]:contrasts.append(dict(identity=[b,g,l,p],distinguished=False));continue
                q=[(v-u)/(w-u) for v,u,w in itertools.product(f,base,end)]
                contrasts.append(dict(identity=[b,g,l,p],distinguished=True,Q95=[min(q),max(q)]))
    return dict(measurement_qualified=qualified,full_capture_qualified=qualified and not control_only,
        verdict='screen_pass' if qualified and control_only else 'qualified_census' if qualified else 'ineligible',
        checks=checks,timing=list(cells.values()),counters=list(counters.values()),contrasts=contrasts,
        estimand='50pct crossing of median-launch spatial readiness CDF under stationary repeats; not prior mean-response Q',
        production_or_evaluation_duration_consumed=False,forward_parameter_or_policy_adopted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--control-only',action='store_true');a=p.parse_args()
    result=analyze(json.loads(a.raw.read_text()),json.loads(a.plan.read_text()),control_only=a.control_only)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('measurement_qualified','verdict','checks','contrasts')}))
