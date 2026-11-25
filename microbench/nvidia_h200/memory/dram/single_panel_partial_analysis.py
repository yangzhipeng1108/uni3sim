"""Validate partial-cohort scope; measured bodies never price an operator."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from microbench.nvidia_h200.memory.dram.finite_strided_panel_analysis import median_confidence_interval


def analyze(raw):
    if raw['status']!='complete':raise ValueError('partial-cohort capture incomplete')
    configs={c[0]:tuple(c) for c in raw['configurations']}
    if len(configs)!=5 or set(configs)!=set(range(5)):raise ValueError('incomplete physical configurations')
    programs={(p['config'],p['stride']):p for p in raw['programs']}
    if len(programs)!=11:raise ValueError('incomplete programs')
    grids={key:sorted({132*configs[key[0]][-1],165*configs[key[0]][-1],*p['target_grids']}) for key,p in programs.items()}
    cells={}
    for item in raw['cells']:
        key=item['config'],item['stride'],item['grid'],item['process'];config,stride,grid,process=key
        if key in cells or item['measurement']['exit_code']:raise ValueError('duplicate or failed body cell')
        _,panel,rows_a,row,threads,shared,resident=configs[config]
        rows=[json.loads(line) for line in item['measurement']['stdout'].splitlines()]
        if len(rows)!=63 or {r['trial'] for r in rows}!=set(range(63)):raise ValueError('incomplete sample sequence')
        for r in rows:
            if (r['config'],r['stride'],r['grid'],r['process'])!=key:raise ValueError('changed condition')
            if (r['panel'],r['rows_a'],r['row'],r['threads'],r['shared'],r['resident'],r['rounds'])!=(panel,rows_a,row,threads,shared,resident,stride//row):raise ValueError('changed physical configuration')
            if r['distinct_sms']!=132 or r['body_ns']<=0 or not 0<r['registers']<=255 or r['max_ctas_per_sm']<1:raise ValueError('invalid placement or body')
        values=[r['body_ns'] for r in rows];median=statistics.median(values);ci=median_confidence_interval(values);clock=statistics.median(r['clock64_hz'] for r in rows)
        cells[key]=dict(config=config,stride=stride,grid=grid,process=process,median_body_ns=median,confidence95=ci,clock64_hz=clock,checks=dict(
            median_precision3pct=max(median-ci['lower_ns'],ci['upper_ns']-median)/median<=.03,
            nominal_clock3pct=abs(clock/1.98e9-1)<=.03))
    if set(cells)!={(c,s,g,p) for (c,s),gs in grids.items() for g in gs for p in range(2)}:raise ValueError('incomplete physical sweep')
    for (c,s,g,p),cell in cells.items():
        a,b=cell['median_body_ns'],cells[c,s,g,1-p]['median_body_ns'];cell['checks']['process3pct']=max(a,b)/min(a,b)-1<=.03
    counters={}
    for item in raw['counters']:
        key=item['config'],item['stride'];config,stride=key;measurement=item['measurement']
        if key in counters or measurement['exit_code']:raise ValueError('duplicate or failed counter')
        _,panel,rows_a,row,threads,shared,resident=configs[config];grid=165*resident;rounds=stride//row
        if item['grid']!=grid:raise ValueError('changed counter grid')
        lines=measurement['stdout'].splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        record,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            v=float(record[name].replace(',',''))
            if not math.isfinite(v) or v<0:raise ValueError('invalid hardware counter')
            return v
        logical=grid*rounds*(panel+row);cold=grid*rounds*panel+stride
        tex=32*value('lts__t_sectors_srcunit_tex_op_read.sum');fabric=32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum');total=32*value('lts__t_sectors_op_read.sum');dram=value('dram__bytes_read.sum')
        counters[key]=dict(config=config,stride=stride,grid=grid,logical_bytes=logical,unique_bytes=cold,tex_bytes=tex,fabric_bytes=fabric,l2_bytes=total,dram_bytes=dram,sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),checks=dict(
            exact_ctas=value('sm__ctas_launched.sum')==grid,
            exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==grid*rounds*(panel+rows_a*row)/512,
            tex1pct=abs(tex/logical-1)<=.01,dram1pct=abs(dram/cold-1)<=.01,
            source_units1pct=abs(total-tex-fabric)/logical<=.01))
    if set(counters)!=set(programs):raise ValueError('incomplete counter programs')
    clocks={}
    for field in ('sm_clock_hz','lts_clock_hz'):
        values=[c[field] for c in counters.values()];clocks[field+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    clocks['direct_sm_nominal3pct']=all(abs(c['sm_clock_hz']/1.98e9-1)<=.03 for c in counters.values())
    classes=[]
    for (config,stride),program in programs.items():
        resident=configs[config][-1];full=132*resident;partial=165*resident;contrasts=[]
        for process in range(2):
            base=cells[config,stride,full,process];tail=cells[config,stride,partial,process]
            ratio=tail['confidence95']['upper_ns']/base['confidence95']['lower_ns']
            contrasts.append(dict(process=process,upper95_partial_to_full_ratio=ratio,pass_lt1_8=ratio<1.8))
        class_ok=all(clocks.values()) and all(counters[config,stride]['checks'].values()) and all(c['pass_lt1_8'] for c in contrasts) and all(all(cells[config,stride,g,p]['checks'].values()) for g in (full,partial) for p in range(2))
        eligible=[g for g in grids[config,stride] if class_ok and all(all(cells[config,stride,g,p]['checks'].values()) for p in range(2))]
        classes.append(dict(config=config,stride=stride,physical_configuration=configs[config],contrasts=contrasts,class_qualified=class_ok,eligible_grids=eligible,unbound_grids=[g for g in grids[config,stride] if g not in eligible]))
    return dict(all_classes_qualified=all(c['class_qualified'] for c in classes),all_points_qualified=all(not c['unbound_grids'] for c in classes),clock_checks=clocks,classes=classes,cells=list(cells.values()),counters=list(counters.values()),production_duration_consumed=False,forward_rate_published=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()));args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_classes_qualified','all_points_qualified','clock_checks','classes')}))
