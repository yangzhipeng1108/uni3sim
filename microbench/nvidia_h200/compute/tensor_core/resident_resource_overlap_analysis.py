"""Analyze independent co-resident CTA timelines; no evaluation imports."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def precision(samples):
    median=statistics.median(samples);interval=median_confidence_interval(samples)
    return dict(median=median,confidence95=interval,
        eligible=max(median-interval['lower_ns'],interval['upper_ns']-median)/median<=.03)


def analyze(raw):
    if raw['status']!='complete':raise ValueError('resource coexistence capture incomplete')
    cells={}
    for item in raw['cells']:
        measurement=item['measurement']
        if measurement['exit_code']:raise ValueError('failed coexistence measurement')
        rows=[json.loads(line) for line in measurement['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):
            raise ValueError('incomplete coexistence samples')
        key=item['mode'],item['rounds'],item['rotation']
        if key in cells:raise ValueError('duplicate physical cell')
        quantities={'body':[],'copy':[],'matrix':[],'copy_mhz':[],'matrix_mhz':[],'overlap':[]}
        for row in rows:
            if (row['mode'],row['rounds'],row['rotation'])!=key or row['resident_ctas']!=2:
                raise ValueError('cell metadata changed')
            mode=key[0];ctas=row['ctas'];grid=132 if mode>=3 else 264
            if row['grid']!=grid or len(ctas)!=grid:raise ValueError('CTA coverage changed')
            bysm={sm:[] for sm in range(132)}
            for sm,role,entry,begin,end,cycles in ctas:
                if sm not in bysm or role not in (0,1) or end<=begin or cycles<=0:
                    raise ValueError('malformed CTA timeline')
                bysm[sm].append((role,entry,begin,end,cycles))
            for members in bysm.values():
                if len(members)!=(1 if mode>=3 else 2):raise ValueError('not the declared per-SM residency')
                if {m[1] for m in members}!=set(range(len(members))):raise ValueError('nonresident entry history')
                if mode==2 and {m[0] for m in members}!={0,1}:raise ValueError('mixed roles did not colocate')
            quantities['body'].append(row['body_ns'])
            for role,name in ((0,'copy'),(1,'matrix')):
                members=[r for r in ctas if r[1]==role]
                if not members:continue
                quantities[name].append(statistics.median(r[4]-r[3] for r in members))
                quantities[name+'_mhz'].append(1000*sum(r[5] for r in members)/sum(r[4]-r[3] for r in members))
            if mode==2:
                overlaps=[]
                for members in bysm.values():
                    a,b=members;overlap=max(0,min(a[3],b[3])-max(a[2],b[2]))
                    overlaps.append(overlap/min(a[3]-a[2],b[3]-b[2]))
                quantities['overlap'].append(statistics.median(overlaps))
        summarized={name:precision(values) for name,values in quantities.items() if values and name in ('body','copy','matrix')}
        cells[key]=dict(key=key,statistics=summarized,
            copy_mhz=statistics.median(quantities['copy_mhz']) if quantities['copy_mhz'] else None,
            matrix_mhz=statistics.median(quantities['matrix_mhz']) if quantities['matrix_mhz'] else None,
            minimum_sample_median_overlap=min(quantities['overlap']) if quantities['overlap'] else None)
    if set(cells)!={(m,r,p) for m in range(5) for r in (256,1024) for p in range(2)}:
        raise ValueError('incomplete physical coexistence sweep')
    for key,cell in cells.items():
        mode,rounds,rotation=key;other=cells[mode,rounds,1-rotation]
        cell['checks']={name+'_precision':value['eligible'] for name,value in cell['statistics'].items()}
        for name,value in cell['statistics'].items():
            medians=value['median'],other['statistics'][name]['median']
            cell['checks'][name+'_process_medians']=max(medians)/min(medians)-1<=.03
        cell['eligible']=all(cell['checks'].values())
    counter,=raw['counters']
    if counter['exit_code']:raise ValueError('failed coexistence counter')
    lines=counter['stdout'].splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
    records=[r for r in csv.DictReader(lines[start:]) if r['ID']=='0']
    row,=records
    def count(name):return int(row[name].replace(',',''))
    logical=132*1024*32768;l2=32*count('lts__t_sectors_op_read.sum');dram=count('dram__bytes_read.sum')
    instructions=count('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')
    counter_checks=dict(dram_le1pct=dram/logical<=.01,l2_logical_within1pct=abs(l2/logical-1)<=.01,
        exact_ldgsts_count=instructions==64*132*1024)
    contrasts=[]
    for rounds in (256,1024):
        checks=dict(counter_checks);ratios=[];clock_ratios=[]
        for process in range(2):
            mixed=cells[2,rounds,process];single=cells[3,rounds,process]
            ratio=mixed['statistics']['copy']['median']/single['statistics']['copy']['median'];ratios.append(ratio)
            clock_ratio=mixed['copy_mhz']/single['copy_mhz'];clock_ratios.append(clock_ratio)
            checks[f'process{process}_precision']=mixed['eligible'] and single['eligible']
            checks[f'process{process}_mixed_copy_ratio_lt1_5']=ratio<1.5
            checks[f'process{process}_copy_clocks_agree3pct']=max(clock_ratio,1/clock_ratio)-1<=.03
            checks[f'process{process}_copy_clocks_nominal3pct']=all(abs(c/1980-1)<=.03 for c in (mixed['copy_mhz'],single['copy_mhz']))
            checks[f'process{process}_mixed_overlap_ge75pct']=mixed['minimum_sample_median_overlap']>=.75
        contrasts.append(dict(rounds=rounds,mixed_single_copy_span_ratios=ratios,
            mixed_single_copy_clock_ratios=clock_ratios,checks=checks,eligible=all(checks.values())))
    return dict(production_or_evaluation_duration_consumed=False,forward_parameter_published=False,
        selection=raw['selection'],cells=list(cells.values()),contrasts=contrasts,
        counters=dict(logical_copy_bytes=logical,l2_read_sector_bytes=l2,dram_read_bytes=dram,
            ldgsts_warp_instructions=instructions,checks=counter_checks),
        coupled_evidence_gate_passed=all(c['eligible'] for c in contrasts))


def analyze_clock_domains(raw):
    if raw['status']!='complete':raise ValueError('hardware clock capture incomplete')
    cells={}
    for cell in raw['clock_counter_cells']:
        measurement=cell['measurement']
        if measurement['exit_code']:raise ValueError('failed hardware clock counter')
        key=cell['mode'],cell['rounds'],cell['matrix_pause_ns'],cell['rotation']
        if key in cells:raise ValueError('duplicate hardware clock cell')
        lines=measurement['stdout'].splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        records=list(csv.DictReader(lines[start:]));units,=tuple(r for r in records if r['ID']=='')
        counter,=tuple(r for r in records if r['ID']=='0')
        for metric in ('sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second'):
            if units[metric] not in ('hz','cycle/second'):
                raise ValueError('hardware clock is not in base cycles per second')
        def value(name):
            result=float(counter[name].replace(',',''))
            if not math.isfinite(result) or result<0:
                raise ValueError('hardware counter must be finite and nonnegative')
            return result
        samples=[json.loads(line) for line in lines if line.startswith('{')]
        profiled,=tuple(s for s in samples if s['trial']==0)
        if (profiled['mode'],profiled['rounds'],profiled['rotation'])!=(key[0],key[1],key[3]):
            raise ValueError('profiled clock launch identity changed')
        if measurement['command'][-5:]!=[str(key[0]),str(key[1]),'3',str(key[3]),str(key[2])]:
            raise ValueError('profiled clock executable arguments changed')
        bysm={sm:[] for sm in range(132)}
        for sm,role,entry,begin,end,cycles in profiled['ctas']:
            if sm not in bysm or role not in (0,1) or end<=begin or cycles<=0:
                raise ValueError('invalid profiled clock CTA timeline')
            bysm[sm].append((role,entry))
        expected={(0,0),(1,1)} if key[0]==2 else {(0,0)}
        if any(set(members)!=expected or len(members)!=len(expected) for members in bysm.values()):
            raise ValueError('profiled clock per-SM role placement changed')
        if any(value(metric)<=0 for metric in ('sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second')):
            raise ValueError('hardware clock must be positive')
        copy=[r for r in profiled['ctas'] if r[1]==0]
        if len(copy)!=132:raise ValueError('hardware clock copy census changed')
        logical=132*key[1]*32768;l2=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        instructions=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')
        checks=dict(l2_logical_within1pct=abs(l2/logical-1)<=.01,dram_le1pct=dram/logical<=.01,
            exact_ldgsts_count=instructions==64*132*key[1])
        cells[key]=dict(key=key,sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),
            lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),
            clock64_copy_hz=1e9*sum(r[5] for r in copy)/sum(r[4]-r[3] for r in copy),
            l2_read_sector_bytes=l2,dram_read_bytes=dram,ldgsts_warp_instructions=instructions,checks=checks)
    if set(cells)!={(m,r,p,s) for m in (2,3) for r in (256,1024) for p in (0,1024) for s in range(2)}:
        raise ValueError('hardware clock physical sweep incomplete')
    clocks=[c['lts_clock_hz'] for c in cells.values()];reference=statistics.median(clocks)
    pairs=[]
    for rounds in (256,1024):
        for pause in (0,1024):
            for rotation in range(2):
                mixed=cells[2,rounds,pause,rotation];single=cells[3,rounds,pause,rotation]
                ratios={name:mixed[name]/single[name] for name in ('sm_clock_hz','lts_clock_hz','clock64_copy_hz')}
                changed=any(max(ratios[k],1/ratios[k])-1>=.05 for k in ('sm_clock_hz','clock64_copy_hz'))
                stable=max(ratios['lts_clock_hz'],1/ratios['lts_clock_hz'])-1<=.03
                pairs.append(dict(rounds=rounds,matrix_pause_ns=pause,rotation=rotation,ratios=ratios,
                    independent_lts_binding_distinguished=changed and stable))
    checks=dict(all_traffic_and_instructions=all(all(c['checks'].values()) for c in cells.values()),
        all_lts_clocks_within3pct=max(clocks)/min(clocks)-1<=.03,
        every_pair_distinguishes_lts_binding=all(p['independent_lts_binding_distinguished'] for p in pairs))
    return dict(production_or_evaluation_duration_consumed=False,selection=raw['selection'],checks=checks,
        clock_binding_gate_passed=all(checks.values()),reference_lts_clock_hz=reference,
        reference_clock_is_candidate_only=not all(checks.values()),cells=list(cells.values()),pairs=pairs,
        memory_bandwidth_or_sm_clock_published=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--clock-domains',action='store_true');args=parser.parse_args()
    source=json.loads(args.raw.read_text());result=analyze_clock_domains(source) if args.clock_domains else analyze(source)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.clock_domains:print(json.dumps(dict(checks=result['checks'],reference_lts_clock_hz=result['reference_lts_clock_hz'],pairs=result['pairs'])))
    else:print(json.dumps(dict(coupled_evidence_gate_passed=result['coupled_evidence_gate_passed'],contrasts=result['contrasts'],counters=result['counters'])))
