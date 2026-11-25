"""Check frozen upstream/downstream request-conservation evidence."""
import argparse
import csv
import json
import math
from pathlib import Path


def analyze(raw):
    expected = {(s,g,f) for s,g in ((1024,24544),(1536,3064),(4096,25338)) for f in (1,2)}
    if raw['status'] != 'complete' or {tuple(c) for c in raw['physical_cells']} != expected:
        raise ValueError('incomplete frozen accounting capture')
    cells = {}
    for item in raw['counters']:
        key = item['stride'],item['grid'],item['fanout']
        s,g,f = key
        if key not in expected or key in cells or item['measurement']['exit_code']:
            raise ValueError('unknown,duplicate or failed counter cell')
        lines = item['measurement']['stdout'].splitlines()
        start = next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
        row, = tuple(r for r in csv.DictReader(lines[start:]) if r['ID']=='0')
        def value(name):
            x = float(row[name].replace(',',''))
            if not math.isfinite(x) or x < 0:
                raise ValueError('invalid counter')
            return x
        rounds = s//128
        logical = g*rounds*(32768+128)
        unique = math.ceil(g/f)*rounds*32768+s
        l1 = 32*value('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        tex = 32*value('lts__t_sectors_srcunit_tex_op_read.sum')
        total = 32*value('lts__t_sectors_op_read.sum')
        fabric = 32*value('lts__t_sectors_srcunit_ltcfabric_op_read.sum')
        dram = value('dram__bytes_read.sum')
        cells[key] = dict(stride=s,grid=g,fanout=f,logical_bytes=logical,unique_bytes=unique,
            upstream_l1tex_bytes=l1,downstream_tex_bytes=tex,dram_bytes=dram,
            upstream_l1tex_requests=value('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'),
            sm_clock_hz=value('sm__cycles_elapsed.avg.per_second'),lts_clock_hz=value('lts__cycles_elapsed.avg.per_second'),
            checks=dict(upstream_logical1pct=abs(l1/logical-1)<=.01,unique_dram1pct=abs(dram/unique-1)<=.01,
                downstream_conservation=unique*.99<=tex<=logical*1.01,source_units1pct=abs(total-tex-fabric)/logical<=.01,
                exact_ctas=value('sm__ctas_launched.sum')==g,
                exact_ldgsts=value('sm__sass_inst_executed_op_ldgsts_cache_bypass.sum')==g*rounds*96))
    if set(cells)!=expected:
        raise ValueError('missing counter cell')
    checks = {'all_cell_checks':all(all(c['checks'].values()) for c in cells.values()),
              'all_duplicate_downstream_distinct':all(c['downstream_tex_bytes']/c['logical_bytes']<.97 for c in cells.values() if c['fanout']==2),
              'sm_nominal3pct':all(abs(c['sm_clock_hz']/1.98e9-1)<=.03 for c in cells.values())}
    for field in ('sm_clock_hz','lts_clock_hz'):
        values=[c[field] for c in cells.values()]
        checks[field+'_stable3pct']=min(values)>0 and max(values)/min(values)-1<=.03
    return dict(qualified=all(checks.values()),checks=checks,cells=list(cells.values()),
                production_or_evaluation_duration_consumed=False,new_timing_price_published=False,selection=raw['selection'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=analyze(json.loads(args.raw.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
