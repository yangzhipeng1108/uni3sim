"""Fixed-state PHASECHK address multiplicity, never a DMA response price."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import statistics


def query_proof(sass):
    proofs=[]
    for chunk in sass.split('Function :')[1:]:
        name=chunk.splitlines()[0].strip();m=re.search(r'query_faninILi(1|128)EE',name)
        if m is None:continue
        count=int(m[1]);instructions=[]
        for line in chunk.splitlines():
            match=re.match(r'\s*/\*([0-9a-f]+)\*/\s+([^;]+);',line)
            if match:instructions.append((int(match[1],16),' '.join(match[2].split())))
        queries=[i for i,(_,t) in enumerate(instructions) if 'PHASECHK' in t]
        if len(queries)!=count:raise ValueError('fixed unrolled query count changed')
        first,last=queries[0],queries[-1]
        before=max(i for i,(_,t) in enumerate(instructions[:first]) if 'SR_CLOCKLO' in t)
        after=next(i for i in range(last+1,len(instructions)) if 'SR_CLOCKLO' in instructions[i][1])
        body=instructions[before:after+1]
        for _,text in body:
            op=text.split()[1] if text.startswith('@') else text.split()[0]
            if op.startswith(('BRA','CALL','RET','YIELD','LDS','STS','LDG','STG','ATOM','WARPSYNC')):
                raise ValueError('query bracket includes control/helper/memory work')
        tail=instructions[last+1:after]
        if not any('ISETP.' in t for _,t in tail):raise ValueError('query result must be compared before end clock')
        if not any('SEL ' in t or t.startswith('@') for _,t in tail):
            raise ValueError('query predicate must be consumed before end clock')
        proofs.append(dict(function=name,queries=count,instructions=len(body),
            begin=hex(instructions[before][0]),end=hex(instructions[after][0]),
            completion_consumption=[dict(address=hex(a),instruction=t) for a,t in tail]))
    if sorted(p['queries'] for p in proofs)!=[1,128]:raise ValueError('both query bodies required')
    return proofs


def fixture(item,trials):
    m=item['measurement']
    if m['exit_code']:raise ValueError('failed instruction measurement')
    rows=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"grid":')]
    if not rows or (trials!=1 and len(rows)!=1):raise ValueError('complete fixture required')
    key=tuple(item[n] for n in ('grid','distinct_addresses','ready','queries'))
    for r in rows:
        if ((r['grid'],r['distinct_addresses'],r['ready'],r['queries_per_bracket'],r['rounds'],
             r['threads'],r['shared_bytes'],r['resident_ctas'])!=(*key,257,288,98304,2)):
            raise ValueError('changed physical/query/state identity')
        if len(r['samples'])!=trials:raise ValueError('all independent launch samples required')
        for s in r['samples']:
            if s['all_lane_numeric_pass'] is not True or s['distinct_sms']!=132:raise ValueError('numerical/placement failure')
            for name in ('mean_query_ns','max_bracket_ns','min_cta_clock_mhz','max_cta_clock_mhz'):
                if isinstance(s[name],bool) or not isinstance(s[name],(int,float)) or not math.isfinite(s[name]) or s[name]<=0:
                    raise ValueError('invalid instruction sample')
    return rows


def analyze(raw,plan):
    if raw['plan']!=plan or raw['status']!='complete':raise ValueError('incomplete frozen capture')
    expected={(*c,p) for c in plan['cells'] for p in (0,1)};cells={};counters={}
    for item in raw['timing']:
        key=tuple(item[n] for n in ('grid','distinct_addresses','ready','queries','process'))
        if key not in expected or key in cells:raise ValueError('unknown/duplicate instruction cell')
        r,=fixture(item,17);metrics={}
        for name in ('mean_query_ns','max_bracket_ns'):
            values=[s[name] for s in r['samples']];med=statistics.median(values)
            metrics[name]=dict(median=med,confidence95=[min(values),max(values)])
        lo,hi=metrics['mean_query_ns']['confidence95'];med=metrics['mean_query_ns']['median']
        cells[key]=dict(identity=key,metrics=metrics,checks=dict(
            mean_precision3pct=max(med-lo,hi-med)/med<=.03,
            all_cta_nominal_clock3pct=all(1.98e3*.97<=s['min_cta_clock_mhz']<=s['max_cta_clock_mhz']<=1.98e3*1.03 for s in r['samples'])))
    if set(cells)!=expected:raise ValueError('missing instruction cells')
    expected_counters={tuple(c) for c in plan['counter_cells']}
    for item in raw['counters']:
        key=tuple(item[n] for n in ('grid','distinct_addresses','ready','queries'))
        if key not in expected_counters or key in counters:raise ValueError('unknown/duplicate clock counter')
        fixtures=fixture(item,1);lines=item['measurement']['stdout'].splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        clock=float(row['sm__cycles_elapsed.avg.per_second'].replace(',',''));ctas=float(row['sm__ctas_launched.sum'].replace(',',''))
        if not math.isfinite(clock) or clock<=0:raise ValueError('invalid direct clock')
        counters[key]=dict(identity=key,clock_hz=clock,replay_fixture_count=len(fixtures),checks=dict(exact_ctas=ctas==key[0],nominal_clock3pct=abs(clock/1.98e9-1)<=.03))
    if set(counters)!=expected_counters:raise ValueError('missing clock counters')
    checks=dict(all_instruction_cells=all(all(c['checks'].values()) for c in cells.values()),all_clock_counters=all(all(c['checks'].values()) for c in counters.values()))
    checks['process3pct']=all(max(cells[(*c,p)]['metrics']['mean_query_ns']['median'] for p in (0,1))/min(cells[(*c,p)]['metrics']['mean_query_ns']['median'] for p in (0,1))-1<=.03 for c in plan['cells'])
    clocks=[c['clock_hz'] for c in counters.values()];checks['direct_clock_stability3pct']=max(clocks)/min(clocks)-1<=.03
    contrasts=[]
    for g,r,q,p in itertools.product((132,264),(0,1),(1,128),(0,1)):
        one=cells[g,1,r,q,p]['metrics']['mean_query_ns']['confidence95'];many=cells[g,32,r,q,p]['metrics']['mean_query_ns']['confidence95']
        contrasts.append(dict(identity=[g,r,q,p],ratio32_over1_95=[many[0]/one[1],many[1]/one[0]]))
    qualified=all(checks.values())
    return dict(measurement_qualified=qualified,checks=checks,cells=list(cells.values()),counters=list(counters.values()),contrasts=contrasts,
        multiplicity_explains_query_width=qualified and all(c['ratio32_over1_95'][0]>=8 for c in contrasts),
        forward_parameter_or_policy_adopted=False,old_observer_requalified=False,
        limitations=['N=128 is amortized throughput, not unloaded single-query latency','fixed pending/ready event states contain no async DMA; source observer still needs independent full-activity resolution proof'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();result=analyze(json.loads(a.raw.read_text()),json.loads(a.plan.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('measurement_qualified','checks','multiplicity_explains_query_width')}))
