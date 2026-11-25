"""Analyze predeclared byte-only arbitration hypotheses, without evaluation data."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def precision(values):
    m=statistics.median(values);ci=median_confidence_interval(values)
    return dict(median=m,confidence95=ci,eligible=max(m-ci['lower_ns'],ci['upper_ns']-m)/m<=.03)


def analyze(raw):
    if raw['status']!='complete':raise ValueError('capture incomplete')
    version=raw['protocol_version'] if 'protocol_version' in raw else 1
    if version not in (1,2):raise ValueError('unknown protocol')
    cells={}
    for item in raw['cells']:
        mode,rounds,rotation=item['mode'],item['rounds'],item['rotation'];process=item['process'] if version==2 else 0;key=mode,rounds,rotation,process
        if key in cells or item['measurement']['exit_code']:raise ValueError('duplicate or failed physical cell')
        rows=[json.loads(line) for line in item['measurement']['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):raise ValueError('incomplete samples')
        values={size:[] for size in ((32768,98304) if mode==0 else (32768,) if mode==1 else (98304,) if mode==2 else (65536,))}
        starts=[];finishes=[];clocks=[]
        for row in rows:
            if (row['mode'],row['rounds'],row['rotation'])!=key[:3] or row['occupancy']!=2 or len(row['ctas'])!=264:raise ValueError('changed physical launch')
            bysm={sm:[] for sm in range(132)}
            for c in row['ctas']:
                if len(c)!=15:raise ValueError('changed CTA schema')
                sm,role,size,total,lo,hi,start,finish,begin,end,cycles,first,last,last_begin,last_end=c
                active=(role+rotation)%2
                expected=(32768 if active==0 else 98304) if mode==0 else (32768 if active==0 else 0) if mode==1 else (98304 if active==1 else 0) if mode==2 else 65536 if mode==3 else (65536 if active==0 else 0)
                if sm not in bysm or role not in (0,1) or size!=expected or not 0<lo<=total/rounds<=hi or not begin<=first<last<=last_begin<last_end<=end or cycles<=0:raise ValueError('invalid CTA work/timeline')
                bysm[sm].append(c)
            if any(len(v)!=2 or {c[1] for c in v}!={0,1} for v in bysm.values()):raise ValueError('incomplete per-SM residency')
            for size in values:values[size].append(statistics.median(c[3]/rounds for c in row['ctas'] if c[2]==size))
            starts.append(statistics.median(c[6]/rounds for c in row['ctas']))
            finishes.append(statistics.median(c[7]/rounds for c in row['ctas']))
            clocks.append(1e9*sum(c[10] for c in row['ctas'])/sum(c[9]-c[8] for c in row['ctas']))
        cells[key]=dict(key=key,roles={str(size):precision(v) for size,v in values.items()},start_skew_ns=statistics.median(starts),finish_skew_ns=statistics.median(finishes),outer_clock64_hz=statistics.median(clocks))
    if set(cells)!={(m,r,t,p) for m in range(5) for r in (128,256) for t in range(2) for p in (range(2) if version==2 else (0,))}:raise ValueError('physical sweep incomplete')
    for (mode,rounds,rotation,process),cell in cells.items():
        cell['checks']={f'role{size}_precision':v['eligible'] for size,v in cell['roles'].items()}
        for size,v in cell['roles'].items():
            other=(mode,rounds,rotation,1-process) if version==2 else (mode,rounds,1-rotation,0)
            peer=cells[other]['roles'][size]['median'];m=v['median']
            cell['checks'][f'role{size}_process']=max(m,peer)/min(m,peer)-1<=.03
    counters={}
    for item in raw['counters']:
        mode=item['mode'];measurement=item['measurement']
        if mode in counters or measurement['exit_code']:raise ValueError('duplicate or failed counter')
        lines=measurement['stdout'].splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        counter,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):return float(counter[name].replace(',',''))
        logical=132*256*(131072 if mode in (0,3) else 32768 if mode==1 else 98304 if mode==2 else 65536 if mode==4 else 0)
        l2=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum');instructions=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')
        checks=dict(l2_payload1pct=(abs(l2/logical-1)<=.01 if logical else True),dram_le1pct=(dram/logical<=.01 if logical else True),exact_ldgsts=instructions==logical/512,exact_ctas=value('sm__ctas_launched.sum')==264)
        counters[mode]=dict(mode=mode,logical_bytes=logical,l2_read_bytes=l2,dram_read_bytes=dram,instructions=instructions,lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),checks=checks)
    if set(counters)!=set(range(6 if version==2 else 5)):raise ValueError('counter sweep incomplete')
    if version==2:
        metadata=counters[5]['l2_read_bytes']
        for mode in range(5):
            c=counters[mode];c['metadata_control_read_bytes']=metadata;c['net_copy_read_bytes']=c['l2_read_bytes']-metadata
            c['checks']['l2_payload1pct']=abs(c['net_copy_read_bytes']/c['logical_bytes']-1)<=.01
    clocks=[c['lts_clock_hz'] for c in counters.values()]
    checks=dict(all_precision_and_process=all(all(c['checks'].values()) for c in cells.values()),all_counters=all(all(c['checks'].values()) for c in counters.values()),lts_stable3pct=min(clocks)>0 and max(clocks)/min(clocks)-1<=.03)
    contrasts=[]
    for rounds in (128,256):
        for index in range(4 if version==2 else 2):
            rotation=index%2;process=index//2 if version==2 else 0
            def med(mode,size):return cells[mode,rounds,rotation,process]['roles'][str(size)]['median']
            denominator=med(2,98304)-med(1,32768)
            if denominator<=0:raise ValueError('nonpositive size contrast')
            q=(med(0,32768)-med(1,32768))/denominator
            skew=cells[3,rounds,rotation,process]['finish_skew_ns']/med(4,65536)
            aligned=max(cells[m,rounds,rotation,process]['start_skew_ns'] for m in range(5))/denominator<=.1
            contrasts.append(dict(rounds=rounds,rotation=rotation,process=process,q_short=q,equal_finish_skew_ratio=skew,start_alignment_passed=aligned,fluid=.25<=q<=.75 and skew<=.25,fifo=(-.25<=q<=.25 or 1.25<=q<=1.75) and skew>=.75))
    checks['all_start_alignment']=all(c['start_alignment_passed'] for c in contrasts)
    qualified=all(checks.values());policy='fluid' if qualified and all(c['fluid'] for c in contrasts) else 'fifo' if qualified and all(c['fifo'] for c in contrasts) else 'unbound'
    return dict(policy=policy,checks=checks,contrasts=contrasts,cells=list(cells.values()),counters=list(counters.values()),production_duration_consumed=False,forward_price_published=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('policy','checks','contrasts')}))
