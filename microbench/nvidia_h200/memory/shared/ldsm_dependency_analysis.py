"""Qualify complete independent LDSM bundles; never subtract a control or price an operator."""
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys


def resource_proof(text):
    rows=re.findall(r'Function _Z15ldsm_dependencyILi([14])EEvP6Recordj:\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',text)
    if sorted(int(r[0]) for r in rows)!=[1,4]:raise ValueError('both compiled resource identities required')
    if any(int(r[2])!=0 or int(r[4])!=0 or int(r[3])>1024 or int(r[1])>64 for r in rows):
        raise ValueError('no spills/stack and at most1024B cuobjdump-reported shared required')
    # This binary tool field must not be relabeled as NCU user-static. The
    # retained NCU launch facts distinguish driver1024,static0,dynamic196608.
    return [dict(chains=int(c),registers=int(reg),stack_bytes=int(stack),cuobjdump_reported_shared_bytes=int(shared),local_bytes=int(local)) for c,reg,stack,shared,local in rows]


def static_proof(sass):
    result=[]
    for section in re.split(r'Function\s*:\s*',sass)[1:]:
        name=section.splitlines()[0].strip()
        match=re.fullmatch(r'_Z15ldsm_dependencyILi([14])EEvP6Recordj',name)
        if match is None:continue
        chains=int(match[1]);ins=[]
        for line in section.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.+?)\s*;',line)
            if m:ins.append((int(m[1],16),' '.join(m[2].split())))
        clocks=[i for i,(_,t) in enumerate(ins) if 'SR_CLOCKLO' in t]
        if len(clocks)!=2:raise ValueError('exactly two full clock instructions required')
        timed=ins[clocks[0]+1:clocks[1]]
        loads=[(a,t) for a,t in timed if t.startswith('LDSM.16.M88.4 ')]
        stores=[(a,t) for a,t in timed if t.startswith('STS ')]
        if len(loads)!=chains or len(stores)!=chains or max(a for a,_ in loads)>=min(a for a,_ in stores):
            raise ValueError('one matrix load per chain and tail consumption stores required')
        if any(re.search(r'\b(?:LDG|LDL|STL|STG|ATOM|RED|LDGSTS|HMMA|HGMMA)(?:\.|\s)',t) for _,t in timed):
            raise ValueError('foreign memory/matrix/spill operation in timed primitive')
        if any('HMMA.' in t or 'HGMMA.' in t or 'LDGSTS.' in t or re.search(r'\b(?:LDL|STL)(?:\.|\s)',t) for _,t in ins):
            raise ValueError('matrix/async/spill outside primitive boundary')
        branches=[]
        for a,t in timed:
            m=re.fullmatch(r'@!?P\d+ BRA (0x[0-9a-f]+)',t)
            if m and int(m[1],16)<a:branches.append((a,int(m[1],16)))
        if len(branches)!=1 or not branches[0][1]<=loads[0][0]<=loads[-1][0]<branches[0][0]<stores[0][0]:
            raise ValueError('one finite load/use backedge required')
        result.append(dict(chains=chains,symbol=name,clock_addresses=[hex(ins[i][0]) for i in clocks],
            static_ldsm_count=len(loads),tail_store_count=len(stores),timed_instructions=[dict(address=hex(a),instruction=t) for a,t in timed],
            interpretation='Full LDSM+four-register XOR+address/checksum+loop+tail-store elapsed bound; not isolated load latency'))
    if sorted(r['chains'] for r in result)!=[1,4]:raise ValueError('both independent chain variants required')
    return result


def quantile(values,q):
    values=sorted(values);x=(len(values)-1)*q;lo=math.floor(x);hi=math.ceil(x)
    return values[lo]*(hi-x)+values[hi]*(x-lo) if hi!=lo else values[lo]


def timing_summary(item):
    if item['measurement']['exit_code']!=0:raise ValueError('primitive process failed')
    raw=json.loads(item['measurement']['stdout']);chains,warps,n=item['chains'],item['warps'],item['inner']
    if (raw['chains'],raw['warps'],raw['inner'],raw['bundles_per_chain'])!=(chains,warps,n,n) or len(raw['samples'])!=17:
        raise ValueError('sample identity/count mismatch')
    samples=[];all_max=[];frequencies=[]
    for sample in raw['samples']:
        if len(sample)!=132 or sorted(row[0] for row in sample)!=list(range(132)):raise ValueError('all132SM samples required')
        maxima=[]
        for row in sample:
            if len(row)!=9 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in row):raise ValueError('invalid integer raw data')
            sm,minimum,maximum,min_ns,max_ns,sum_c,sum_n,start_lo,start_hi=row
            if minimum<=0 or minimum>maximum or min_ns<=0 or min_ns>max_ns or sum_c<=0 or sum_n<=0 or start_hi<start_lo:
                raise ValueError('invalid perCTA primitive interval')
            if not minimum*warps<=sum_c<=maximum*warps or not min_ns*warps<=sum_n<=max_ns*warps:raise ValueError('warp aggregate does not conserve intervals')
            maxima.append(maximum/n);frequencies.append(sum_c/sum_n*1e9)
        samples.append(statistics.median(maxima));all_max.extend(maxima)
    # Fixed nonparametric central interval; no sample removal or cycle fit.
    ordered=sorted(samples);lower,upper=ordered[4],ordered[12];median=statistics.median(samples)
    return dict(chains=chains,warps=warps,inner=n,process=item['process'],samples=17,
        median_cta_max_bundle_cycles=median,central_interval_cycles=[lower,upper],relative_interval_width=(upper-lower)/median,
        p99_cta_max_bundle_cycles=quantile(all_max,.99),maximum_observed_bundle_cycles=max(all_max),
        median_bundle_cycles_per_independent_chain=median/chains,
        minimum_measured_sm_hz=min(frequencies),maximum_measured_sm_hz=max(frequencies),median_measured_sm_hz=statistics.median(frequencies),
        exact_gpu_checksum_and_placement_passed=True)


def counter_values(stdout):
    lines=stdout.splitlines();headers=[i for i,l in enumerate(lines) if '"Metric Name"' in l and '"Metric Value"' in l]
    wide=[i for i,l in enumerate(lines) if l.startswith('"ID","Process ID",')]
    if len(headers)+len(wide)!=1:raise ValueError('one supported raw NCU table required')
    if wide:
        table=list(csv.reader(lines[wide[0]:]));table=[r for r in table if r]
        if len(table)!=3 or len(set(table[0]))!=len(table[0]) or any(len(r)!=len(table[0]) for r in table):
            raise ValueError('wide NCU requires one units row and one kernel row')
        names,units,values=table
        if units[0]!='' or values[0]!='0':raise ValueError('wide counter record identity failed')
        required=('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
            'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
            'sm__ctas_launched.sum','sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second')
        return {name:float(values[names.index(name)].replace(',','')) for name in required}
    reader=csv.DictReader(io.StringIO('\n'.join(lines[headers[0]:])));result={}
    for row in reader:
        name=row['Metric Name']
        if name in result:raise ValueError('duplicate counter')
        value=row['Metric Value'].replace(',','')
        result[name]=float(value)
    return result


def analyze(raw,plan):
    proofs=static_proof(raw['sass']['stdout'])
    resources=resource_proof(raw['resources']['stdout'])
    if raw['status']!='complete':raise ValueError('full frozen capture required')
    rows=[timing_summary(i) for i in raw['timing']]
    expected={(p,c,w,n) for p in (0,1) for c,w,n in plan['cells']}
    if {(r['process'],r['chains'],r['warps'],r['inner']) for r in rows}!=expected or len(rows)!=len(expected):raise ValueError('physical timing scope incomplete')
    checks=[]
    for r in rows:
        checks.append(dict(name='median_precision',cell=[r['process'],r['chains'],r['warps'],r['inner']],passed=r['relative_interval_width']<=plan['relative_precision']))
        checks.append(dict(name='all_warp_clock_support',cell=[r['process'],r['chains'],r['warps'],r['inner']],passed=
            r['minimum_measured_sm_hz']>=plan['nominal_sm_hz']*(1-plan['clock_tolerance']) and r['maximum_measured_sm_hz']<=plan['nominal_sm_hz']*(1+plan['clock_tolerance'])))
    lookup={(r['process'],r['chains'],r['warps'],r['inner']):r for r in rows}
    for c,w,n in plan['cells']:
        a,b=[lookup[p,c,w,n]['median_cta_max_bundle_cycles'] for p in (0,1)]
        checks.append(dict(name='process_repeat',cell=[c,w,n],passed=abs(a-b)/min(a,b)<=plan['relative_precision']))
    for p in (0,1):
        for c,w in {(c,w) for c,w,n in plan['cells']}:
            a,b=[lookup[p,c,w,n]['median_cta_max_bundle_cycles'] for n in (256,1024)]
            checks.append(dict(name='length_repeat',cell=[p,c,w],passed=abs(a-b)/min(a,b)<=plan['relative_precision']))
    counters=[]
    for item in raw['counters']:
        if item['measurement']['exit_code']!=0:raise ValueError('counter collection failed')
        values=counter_values(item['measurement']['stdout']);c,w=item['chains'],item['warps']
        expected_waves=132*w*1024*c*4
        waves=values['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'];conflicts=values['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']
        passed=(values['sm__ctas_launched.sum']==132 and abs(waves/expected_waves-1)<=plan['counter_tolerance'] and conflicts==0 and abs(values['sm__cycles_elapsed.avg.per_second']/plan['nominal_sm_hz']-1)<=plan['clock_tolerance'])
        counters.append(dict(chains=c,warps=w,expected_shared_load_wavefronts=expected_waves,values=values,passed=passed))
    if {(r['chains'],r['warps']) for r in counters}!={tuple(c) for c in plan['counter_cells']} or len(counters)!=len(plan['counter_cells']):raise ValueError('counter scope incomplete')
    qualified=all(c['passed'] for c in checks+counters)
    return dict(measurement_qualified=qualified,checks=checks,counters=counters,timing=rows,source_proofs=proofs,resources=resources,
        production_price_supplied=False,scope='Independent complete LDSM x4 use/address bundle elapsed bounds; no control subtraction or operator pricing',
        limitations=['One resident CTA with1/4/8/16/32warps;crossCTA interleaving and concurrent LDGSTS are unmeasured.',
         'Elapsed bundle includes arithmetic,finite-loop issue,clock brackets and final shared consumption stores.',
         'No measured value is an architectural worst-case bound or directly interchangeable with a HMMA consumer.',
         'All raw cells/checks remain;failed scope cannot be filtered into a qualified price.'])


if __name__=='__main__':
    raw=json.loads(Path(sys.argv[1]).read_text());plan=json.loads(Path(sys.argv[3]).read_text())
    result=analyze(raw,plan);Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(measurement_qualified=result['measurement_qualified'],timing_cells=len(result['timing']),counter_cells=len(result['counters']))))
