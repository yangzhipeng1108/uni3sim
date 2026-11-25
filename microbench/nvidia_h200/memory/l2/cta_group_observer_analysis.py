"""CTA hardware joins and uniform queries; no operator timing price."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import statistics


def poll_proof(sass):
    proofs=[]
    for chunk in sass.split('Function :')[1:]:
        name=chunk.splitlines()[0].strip()
        if 'cta_group_observer' not in name:continue
        ins=[]
        for line in chunk.splitlines():
            m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+([^;]+);',line)
            if m:ins.append((int(m[1],16),' '.join(m[2].split())))
        by={a:i for i,(a,_) in enumerate(ins)};loops=[]
        if sum('ARRIVES.LDGSTSBAR' in t for _,t in ins)!=2:
            raise ValueError('first/all deferred-arrival sites must be exactly two')
        for j,(a,t) in enumerate(ins):
            m=re.fullmatch(r'(?:@!?P[0-9]+ )?BRA(?:\.U)? (0x[0-9a-f]+)',t)
            if m is None or int(m[1],16)>=a or int(m[1],16) not in by:continue
            i=by[int(m[1],16)];body=ins[i:j+1]
            if any('PHASECHK' in t for _,t in body) and any('SR_CLOCKLO' in t for _,t in body):loops.append((i,j,body))
        if not loops:raise ValueError('timed query backedge required')
        i,j,body=min(loops,key=lambda r:len(r[2]))
        if sum('PHASECHK' in t for _,t in body)!=1 or not any('VOTE.ALL' in t for _,t in body):
            raise ValueError('one uniform-address query and all-lane ready vote required')
        for index,(_,text) in enumerate(body):
            op=text.split()[1] if text.startswith('@') else text.split()[0]
            if (op.startswith(('LD','ST','ATOM','CALL','RET','BAR.','BRX','BRA.DIV'))
                    or 'SR_GLOBALTIMER' in text
                    or (op.startswith('BRA') and index!=len(body)-1)):
                raise ValueError('query loop has unsupported memory/helper/CTA control')
        initial=[k for k in range(i) if 'PHASECHK' in ins[k][1]]
        if len(initial)!=2 or not any('STS' in t for _,t in ins[initial[-1]+1:i]):
            raise ValueError('both initial event checks must precede producer release')
        proofs.append(dict(function=name,instructions=len(body),begin=hex(ins[i][0]),end=hex(ins[j][0]),
            deferred_arrival_sites=2,query_is_uniform_by_source=True,
            body=[dict(address=hex(a),instruction=t) for a,t in body]))
    if len(proofs)!=12:raise ValueError('all twelve B/lookahead/endpoint bodies required')
    return proofs


def arrival_proof(ptx):
    proofs=[]
    for chunk in ptx.split('.visible .entry ')[1:]:
        name=chunk.split('(')[0].strip()
        if 'cta_group_observer' not in name:continue
        count=re.findall(r'mbarrier\.init\.shared\.b64\s+\[[^\]]+\],\s*(\d+)\s*;',chunk)
        if count!=['257']:raise ValueError('PTX must retain one257-count event initializer')
        if len(re.findall(r'cp\.async\.mbarrier\.arrive\.noinc\.shared\.b64',chunk))!=2:
            raise ValueError('PTX needs first/all deferred arrival sites')
        if len(re.findall(r'(?<![.\w])mbarrier\.arrive\.shared\.b64',chunk))!=2:
            raise ValueError('PTX needs first/all ordinary arrival sites')
        proofs.append(dict(function=name,initial_count=257,deferred_per_endpoint_by_source=256,
            ordinary_per_endpoint_by_source=1,count_conserved=True))
    if len(proofs)!=12:raise ValueError('all12PTXsource bodies required')
    return proofs


def static_proof(sass,ptx):
    loops=poll_proof(sass);arrivals=arrival_proof(ptx)
    if {v['function'] for v in loops}!={v['function'] for v in arrivals}:raise ValueError('PTX/SASS source bodies disagree')
    return dict(poll_loops=loops,hardware_join_arrivals=arrivals)


def compilation_proof(build_stderr,sass):
    bodies={}
    for chunk in sass.split('Function :')[1:]:
        name=chunk.splitlines()[0].strip()
        if 'cta_group_observer' in name:
            if name in bodies:raise ValueError('duplicate SASS body')
            bodies[name]=chunk
    proofs=[]
    for chunk in build_stderr.split('Compiling entry function ')[1:]:
        match=re.match(r"'([^']+)'",chunk)
        if match is None:raise ValueError('unrecognized compiler entry')
        name=match[1]
        if 'cta_group_observer' not in name:continue
        counts=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',chunk)
        if len(counts)!=1:raise ValueError('explicit compiler stack/spill accounting required')
        stack,stores,loads=map(int,counts[0])
        local=bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)',bodies[name]))
        proofs.append(dict(function=name,stack_bytes=stack,spill_store_bytes=stores,
            spill_load_bytes=loads,local_memory_instructions=local,
            no_stack_spills_or_local_memory=not(stack or stores or loads or local)))
    if len(proofs)!=12 or {p['function'] for p in proofs}!=set(bodies):
        raise ValueError('all twelve compiled/SASS bodies required')
    return proofs


def confidence(values,tail_denominator=40):
    a=sorted(values);n=len(a);ranks=[k for k in range(1,n//2+1) if tail_denominator*sum(math.comb(n,i) for i in range(k))<=2**n]
    if not ranks:raise ValueError('insufficient exact median confidence samples')
    k=ranks[-1];return a[k-1],a[n-k]


def fixture(item,trials):
    m=item['measurement']
    if m['exit_code']:raise ValueError('failed CTA-join measurement')
    rows=[json.loads(s) for s in m['stdout'].splitlines() if s.startswith('{"B_bytes":')]
    if not rows or (trials!=1 and len(rows)!=1):raise ValueError('complete numerical fixture required')
    b,g,l,e=(item[n] for n in ('B_bytes','grid','lookahead','endpoint'))
    for c in rows:
        if ((c['B_bytes'],c['grid'],c['lookahead'],c['endpoint'],c['barrier_arrivals'],c['rounds'],
             c['threads'],c['producer_threads'],c['shared_bytes'],c['resident_ctas'],c['distinct_sms'],
             c['all_lane_checksum_pass'],c['first_poll_pending_all'],c['start_policy'])!=
            (b,g,l,e,257,127,288,256,6*b,2 if b==16384 else 1,132,True,True,'observer_pending_before_producer_release')):
            raise ValueError('changed CTA-join physical/numeric/source identity')
        if len(c['samples'])!=trials:raise ValueError('all launch samples required')
        for s in c['samples']:
            for name in ('response_ns','enqueue_ns','max_poll_ns','calibration_mhz','min_cta_calibration_mhz','max_cta_calibration_mhz'):
                if isinstance(s[name],bool) or not isinstance(s[name],(int,float)) or not math.isfinite(s[name]) or s[name]<=0:
                    raise ValueError('invalid CTA-join sample')
    return rows


def analyze(raw,plan,*,control_only=False):
    if raw['plan']!=plan:raise ValueError('changed frozen protocol')
    if raw['status'] not in (('screen_complete','observer_screen_failed') if control_only else ('complete',)):
        raise ValueError('incomplete requested capture')
    source_proof=static_proof(raw['sass']['stdout'],raw['ptx']['stdout'])
    compiled=compilation_proof(raw['build']['stderr'],raw['sass']['stdout'])
    physical={tuple(c) for c in plan['control_cells' if control_only else 'physical_cells']};expected={(*c,p) for c in physical for p in (0,1)}
    timing={};samples={};counters={}
    for item in raw['timing']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','endpoint','process'))
        if key not in expected or key in timing:raise ValueError('unknown/duplicate endpoint timing')
        c,=fixture(item,63);data={};checks={}
        for name in ('response_ns','enqueue_ns','max_poll_ns'):
            values=[s[name] for s in c['samples']];samples[key,name]=values;median=statistics.median(values);lo,hi=confidence(values)
            data[name]=dict(median=median,confidence95=[lo,hi])
        lo,hi=data['response_ns']['confidence95'];median=data['response_ns']['median']
        checks['response_precision3pct']=max(median-lo,hi-median)/median<=.03
        checks['all_cta_nominal_calibration3pct']=all(1980*.97<=s['min_cta_calibration_mhz']<=s['max_cta_calibration_mhz']<=1980*1.03 for s in c['samples'])
        timing[key]=dict(identity=key,metrics=data,checks=checks)
    if set(timing)!=expected:raise ValueError('missing first/all endpoint timing')
    for key,c in timing.items():
        b,g,l,e,p=key;first_key=(b,g,l,0,p)
        # Same original10%first-response reference for BOTH endpoint loops.
        # Two97.5%intervals make this comparison jointly at least95%.
        first_lo=confidence(samples[first_key,'response_ns'],80)[0]
        poll_hi=confidence(samples[key,'max_poll_ns'],80)[1]
        c['checks']['poll_resolution10pct_first_response']=poll_hi<=.1*first_lo
        other=timing[b,g,l,e,1-p]['metrics']['response_ns']['median'];value=c['metrics']['response_ns']['median']
        c['checks']['response_process3pct']=max(value,other)/min(value,other)-1<=.03
        first=timing[b,g,l,0,p]['metrics']['response_ns']['median'];all_=timing[b,g,l,1,p]['metrics']['response_ns']['median']
        c['checks']['median_endpoint_order']=all_>=first
    for item in raw['counters']:
        key=tuple(item[n] for n in ('B_bytes','grid','lookahead','endpoint','process'))
        if key not in expected or key in counters:raise ValueError('unknown/duplicate endpoint counter')
        rows=fixture(item,1);lines=item['measurement']['stdout'].splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            x=float(row[name].replace(',',''))
            if not math.isfinite(x) or x<0:raise ValueError('invalid counter')
            return x
        b,g,l,e,p=key;logical=g*b*(4+127*l);up=32*value('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum');tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum');fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum');total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum');sm=value('sm__cycles_elapsed.avg.per_second');lts=value('lts__cycles_elapsed.avg.per_second')
        counters[key]=dict(identity=key,replay_fixture_count=len(rows),logical_bytes=logical,upstream_bytes=up,tex_bytes=tex,dram_bytes=dram,sm_clock_hz=sm,lts_clock_hz=lts,checks=dict(exact_ctas=value('sm__ctas_launched.sum')==g,exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==logical/512,upstream1pct=abs(up/logical-1)<=.01,tex1pct=abs(tex/logical-1)<=.01,source_units1pct=abs(total-tex-fabric)/logical<=.01,dram1pct=dram/logical<=.01,sm_nominal3pct=abs(sm/1.98e9-1)<=.03))
    if set(counters)!=expected:raise ValueError('missing endpoint counter')
    checks=dict(compiled_no_stack_spills_or_local_memory=all(c['no_stack_spills_or_local_memory'] for c in compiled),
        compiled_no_matrix_operations=re.search(r'\b(?:HMMA|HGMMA)\.',raw['sass']['stdout']) is None,
        all_timing_and_resolution=all(all(c['checks'].values()) for c in timing.values()),all_counters=all(all(c['checks'].values()) for c in counters.values()))
    for name in ('sm_clock_hz','lts_clock_hz'):
        v=[c[name] for c in counters.values()];checks[name+'_stable3pct']=min(v)>0 and max(v)/min(v)-1<=.03
    qualified=all(checks.values());contrasts=[]
    if qualified and not control_only:
        for b,g in sorted({c[:2] for c in physical}):
            for l,p in itertools.product((2,4),(0,1)):
                intervals=[confidence(samples[k,'response_ns'],120) for k in ((b,g,l,0,p),(b,g,1,0,p),(b,g,l,1,p))]
                if intervals[2][0]<=intervals[1][1]:contrasts.append(dict(identity=[b,g,l,p],distinguished=False,first_priority=False,equal_burst_slow_response=False));continue
                q=[(f-base)/(end-base) for f,base,end in itertools.product(*intervals)]
                contrasts.append(dict(identity=[b,g,l,p],distinguished=True,Q95=[min(q),max(q)],first_priority=max(q)<=.25,equal_burst_slow_response=min(q)>=.75))
    verdict=('first_priority' if all(c['first_priority'] for c in contrasts) else 'equal_burst_slow_response' if all(c['equal_burst_slow_response'] for c in contrasts) else 'inconclusive') if qualified and not control_only else 'screen_pass' if qualified else 'ineligible'
    return dict(measurement_qualified=qualified,full_capture_qualified=qualified and not control_only,verdict=verdict,checks=checks,static_proof=source_proof,compilation_proof=compiled,timing=list(timing.values()),counters=list(counters.values()),contrasts=contrasts,production_or_evaluation_duration_consumed=False,forward_parameter_or_policy_adopted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--control-only',action='store_true');a=p.parse_args();result=analyze(json.loads(a.raw.read_text()),json.loads(a.plan.read_text()),control_only=a.control_only);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('measurement_qualified','full_capture_qualified','verdict','checks','contrasts')}))
