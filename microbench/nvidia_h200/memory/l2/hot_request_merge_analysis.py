"""Qualify physical hot-request coalescing without any timing parameters."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

PROGRAMS=((16384,3,12288,2),(16384,3,32768,2),(32768,4,1536,1),(32768,4,4096,1))


def _parse_cells(raw_cells,expected):
    cells={}
    for c in raw_cells:
        key=c['B_bytes'],c['slots'],c['stride'],c['grid'],c['fanout'],c['skew_cycles'],c['process']
        if key not in expected or key in cells or c['measurement']['exit_code']:
            raise ValueError('unknown,duplicate or failed counter')
        b,slots,s,g,f,skew,p=key
        lines=c['measurement']['stdout'].splitlines()
        start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        row,=tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def metric(name):
            v=float(row[name].replace(',',''))
            if not math.isfinite(v) or v<0:
                raise ValueError('invalid hardware counter')
            return v
        fixtures=[json.loads(line) for line in lines if line.startswith('{"B_bytes":')]
        if not fixtures or len({json.dumps(x,sort_keys=True) for x in fixtures})!=1:
            raise ValueError('missing or changed all-lane numerical evidence')
        fixture=fixtures[0]
        resident=2 if b==16384 else 1
        if (fixture['B_bytes'],fixture['slots'],fixture['stride'],fixture['grid'],fixture['fanout'],fixture['skew_cycles'],fixture['rounds'],fixture['threads'],fixture['shared_bytes'],fixture['resident_ctas'],fixture['distinct_sms'],fixture['all_lane_checksum_pass'])!=(b,slots,s,g,f,skew,257,256,slots*(16384+b),resident,132,True):
            raise ValueError('changed copy program or numeric/resource identity')
        logical=b*g*257
        upstream=32*metric('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex=32*metric('lts__t_sectors_srcunit_tex_op_read.sum')
        fabric=32*metric('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        total=32*metric('lts__t_sectors_op_read.sum')
        dram=metric('dram__bytes_read.sum')
        cells[key]=dict(B_bytes=b,slots=slots,stride=s,grid=g,fanout=f,skew_cycles=skew,process=p,
            logical_bytes=logical,upstream_bytes=upstream,tex_bytes=tex,fabric_bytes=fabric,dram_bytes=dram,
            remaining_request_fraction=tex/upstream,sm_clock_hz=metric('sm__cycles_elapsed.avg.per_second'),
            lts_clock_hz=metric('lts__cycles_elapsed.avg.per_second'),checks=dict(
                exact_ctas=metric('sm__ctas_launched.sum')==g,
                exact_ldgsts=metric('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==g*257*(b+16384)//512,
                upstream_logical1pct=abs(upstream/logical-1)<=.01,
                dram_le1pct=dram/logical<=.01,
                source_units1pct=abs(total-tex-fabric)/logical<=.01,
                downstream_conservation1pct=1/f-.01<=tex/upstream<=1.01,
                sm_nominal3pct=abs(metric('sm__cycles_elapsed.avg.per_second')/1.98e9-1)<=.03))
    return cells


def analyze(raw):
    expected={(b,slots,s,132*resident*w,f,skew,p) for b,slots,s,resident in PROGRAMS
              for w in (1,2) for f in (1,2,4,8,16) for skew in (0,512) for p in (0,1)}
    if raw['status']!='complete' or len(raw['cells'])!=160:
        raise ValueError('requires all frozen counter cells')
    declared={tuple(c) for c in raw['physical_cells']}
    if declared!={key[:-1] for key in expected}:
        raise ValueError('frozen physical cell set changed')
    cells=_parse_cells(raw['cells'],expected)
    if set(cells)!=expected:
        raise ValueError('missing frozen counter identity')
    clocks={}
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[c[name] for c in cells.values()]
        clocks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    programs=[]
    for b,slots,s,resident in PROGRAMS:
        group=[c for c in cells.values() if (c['B_bytes'],c['slots'],c['stride'])==(b,slots,s)]
        checks=dict(global_clocks=all(clocks.values()),all_cell_checks=all(all(c['checks'].values()) for c in group),
                    fanout1_identity1pct=all(abs(c['remaining_request_fraction']-1)<=.01 for c in group if c['fanout']==1))
        points=[]
        for f in (1,2,4,8,16):
            conditions=[c for c in group if c['fanout']==f]
            if len(conditions)!=8:
                raise ValueError('missing wave/phase/process contrast')
            values=[c['remaining_request_fraction'] for c in conditions]
            stable=min(values)>0 and max(values)/min(values)-1<=.03
            checks[f'fanout{f}_phase_wave_process3pct']=stable
            points.append(dict(fanout=f,minimum_fraction=min(values),maximum_fraction=max(values),stable=stable))
        programs.append(dict(B_bytes=b,slots=slots,stride=s,resident_ctas=resident,checks=checks,
                             qualified=all(checks.values()),fanouts=points))
    return dict(production_or_evaluation_duration_consumed=False,new_timing_price_published=False,
                all_programs_qualified=all(p['qualified'] for p in programs),clock_checks=clocks,
                programs=programs,cells=list(cells.values()),selection=raw['selection'])


def analyze_partial(raw):
    """Validate available counters for rejection only; never publish a datum."""
    if raw['status'] not in ('counter_failed','budget_stop_partial'):
        raise ValueError('partial screen requires explicit incomplete capture status')
    expected={(b,slots,s,132*resident*w,f,skew,p) for b,slots,s,resident in PROGRAMS
              for w in (1,2) for f in (1,2,4,8,16) for skew in (0,512) for p in (0,1)}
    available=[c for c in raw['cells'] if c['measurement']['exit_code']==0]
    cells=_parse_cells(available,expected)
    if not cells or len(cells)>=len(expected):
        raise ValueError('partial screen requires a nonempty incomplete cell set')
    programs=[]
    for b,slots,s,resident in PROGRAMS:
        group=[c for c in cells.values() if (c['B_bytes'],c['slots'],c['stride'])==(b,slots,s)]
        points=[]
        for f in (1,2,4,8,16):
            values=[c['remaining_request_fraction'] for c in group if c['fanout']==f]
            if not values:
                continue
            spread=max(values)/min(values)-1
            points.append(dict(fanout=f,observed_cells=len(values),minimum_fraction=min(values),
                               maximum_fraction=max(values),relative_spread=spread,
                               frozen_stability_gate_rejected=spread>.03))
        programs.append(dict(B_bytes=b,slots=slots,stride=s,qualified=False,
                             rejected_by_available_stability_evidence=any(p['frozen_stability_gate_rejected'] for p in points),
                             fanouts=points))
    clocks={}
    for name in ('sm_clock_hz','lts_clock_hz'):
        values=[c[name] for c in cells.values()]
        clocks[name+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    contrasts={}
    for dimension in ('skew_cycles','grid','process'):
        groups=defaultdict(list)
        fields=tuple(f for f in ('B_bytes','stride','grid','fanout','skew_cycles','process') if f!=dimension)
        for c in cells.values():
            groups[tuple(c[f] for f in fields)].append(c)
        pairs=[]
        for key,rows in groups.items():
            if len({r[dimension] for r in rows})!=2:
                continue
            values=[r['remaining_request_fraction'] for r in rows]
            pairs.append(dict(identity=dict(zip(fields,key)),dimension=dimension,
                relative_spread=max(values)/min(values)-1,
                conditions=[dict(value=r[dimension],fraction=r['remaining_request_fraction']) for r in rows]))
        contrasts[dimension]=pairs
    return dict(status='partial_bounded_stop',eligible=False,full160_gate_passed=False,
                all_programs_qualified=False,successful_cells=len(cells),expected_cells=len(expected),
                failed_cell_checks=[{k:c[k] for k in ('B_bytes','stride','grid','fanout','skew_cycles','process','checks')}
                                    for c in cells.values() if not all(c['checks'].values())],
                clock_checks=clocks,programs=programs,cells=list(cells.values()),paired_contrasts=contrasts,
                production_or_evaluation_duration_consumed=False,new_timing_price_published=False,
                purpose='Partial rejection screen only; missing cells stay missing and no parameter is published')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--partial-screen',action='store_true')
    args=parser.parse_args()
    raw=json.loads(args.raw.read_text())
    result=analyze_partial(raw) if args.partial_screen else analyze(raw)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_programs_qualified','clock_checks','programs')}))
