"""Analyze independent byte ingress; never read production/evaluation data."""
import argparse
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval

STRIDES=(512,4096,5760,12288,32768)

def analyze(raw, *, entropy=False):
    if raw['status']!='complete':
        raise ValueError('incomplete ingress capture')
    strides=(512,12288) if entropy else STRIDES
    fields=('grid','row_stride_bytes','rounds','consumer')+ (('pattern',) if entropy else ())
    cells=defaultdict(dict)
    for cell in raw['cells']:
        if cell['measurement']['exit_code']!=0:
            raise ValueError('failed measurement')
        rows=[json.loads(line) for line in cell['measurement']['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):
            raise ValueError('incomplete cell samples')
        key=tuple(rows[0][k] for k in fields)
        if any(tuple(r[k] for k in fields)!=key for r in rows):
            raise ValueError('cell identity changed')
        if cell['rotation'] in cells[key]:
            raise ValueError('duplicate process cell')
        cells[key][cell['rotation']]=rows
    expected={(g,s,r,c) for g in (132,264,528) for s in strides for r in (256,1024) for c in (0,1)}
    if entropy:
        expected={(g,s,r,0,pattern) for g in (132,264,528) for s in strides for r in (256,1024) for pattern in (0,1,2)}
    if set(cells)!=expected or any(set(p)!={0,1} for p in cells.values()):
        raise ValueError('incomplete physical sweep')
    summary={}
    for key,processes in sorted(cells.items()):
        medians=[];confidence=[];clocks=[]
        for process in (0,1):
            rows=processes[process];samples=[r['body_ns'] for r in rows]
            medians.append(statistics.median(samples));confidence.append(median_confidence_interval(samples))
            clocks.append(statistics.median(r['weighted_mhz'] for r in rows))
        g,s,r,c=key[:4];logical=g*32768*r
        checks={'median_confidence95_within3pct':all(max(m-ci['lower_ns'],ci['upper_ns']-m)/m<=.03 for m,ci in zip(medians,confidence)),
                'process_medians_within3pct':max(medians)/min(medians)-1<=.03,
                'nominal_sm_clock_within3pct':all(abs(v/1980-1)<=.03 for v in clocks),
                'resident_ctas_equal2':all(row['resident_ctas_per_sm']==2 for rows in processes.values() for row in rows),
                'logical_bytes_conserved':all(row['logical_bytes']==logical for rows in processes.values() for row in rows)}
        summary[key]=dict(key=key,median_body_ns=medians,median_confidence95=confidence,
            weighted_mhz=clocks,checks=checks,eligible=all(checks.values()),logical_bytes=logical,
            logical_byte_capacity_per_s=logical/max(medians)*1e9)
    counters={}
    for item in raw['counters']:
        if item['measurement']['exit_code']!=0:raise ValueError('failed counter measurement')
        lines=item['measurement']['stdout'].splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
        rows=[r for r in csv.DictReader(lines[start:]) if r['ID']=='0']
        if len(rows)!=1:raise ValueError('counter launch is not unique')
        row=rows[0];stride=item['row_stride_bytes'];identity=(stride,item['pattern']) if entropy else stride;logical=264*32768*1024
        if identity in counters:raise ValueError('duplicate stride counter')
        def count(name):return int(row[name].replace(',',''))
        dram=count('dram__bytes_read.sum');l2=32*count('lts__t_sectors_op_read.sum');instructions=count('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')
        counters[identity]=dict(row_stride_bytes=stride,logical_bytes=logical,dram_read_bytes=dram,l2_read_sector_bytes=l2,
            ldgsts_warp_instructions=instructions,checks={'dram_le1pct':dram/logical<=.01,'l2_logical_within1pct':abs(l2/logical-1)<=.01,'exact_ldgsts_count':instructions==64*264*1024})
    expected_counters={(s,p) for s in strides for p in (0,1,2)} if entropy else set(strides)
    if set(counters)!=expected_counters:raise ValueError('incomplete stride counters')
    services=[]
    for identity in sorted(expected_counters):
        stride,pattern=identity if entropy else (identity,None)
        suffix=(pattern,) if entropy else ()
        selected=summary[(264,stride,1024,0)+suffix];plateau=summary[(528,stride,1024,0)+suffix]
        rates=(selected['logical_byte_capacity_per_s'],plateau['logical_byte_capacity_per_s'])
        checks={'selected_point_valid':selected['eligible'],'plateau_point_valid':plateau['eligible'],
            'plateau_capacity_within3pct':max(rates)/min(rates)-1<=.03,**counters[identity]['checks']}
        service=dict(row_stride_bytes=stride,logical_byte_capacity_per_s=selected['logical_byte_capacity_per_s'],
            checks=checks,eligible=all(checks.values()),selected_key=selected['key'])
        if entropy:service['pattern']=pattern;counters[identity]['pattern']=pattern
        services.append(service)
    result=dict(production_or_evaluation_duration_consumed=False,selection=raw['selection'],services=services,
        eligible_strides=[s['row_stride_bytes'] for s in services if s['eligible']],counters=list(counters.values()),cells=list(summary.values()))
    if entropy:
        result['forward_parameter_published']=False
        result['qualified_contrasts']=[dict(row_stride_bytes=s,pattern=p) for s,p in expected_counters if next(v for v in services if v['row_stride_bytes']==s and v['pattern']==p)['eligible']]
        del result['eligible_strides']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--entropy',action='store_true');a=p.parse_args()
    out=analyze(json.loads(a.raw.read_text()),entropy=a.entropy);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(services=out['services'])))
