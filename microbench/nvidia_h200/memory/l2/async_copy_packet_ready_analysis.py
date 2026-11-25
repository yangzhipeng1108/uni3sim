"""Fail-closed finite-packet diagnostics. Never emits a Forward parameter."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median

METRICS = (
    'sm__ctas_launched.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'lts__t_sectors_srcunit_tex_op_read.sum',
    'lts__t_sectors_srcunit_ltcfabric_op_read.sum',
    'lts__t_sectors_op_read.sum',
    'dram__bytes_read.sum',
    'sm__cycles_elapsed.avg.per_second',
)


def code_proof(sass, build_stderr):
    properties = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', build_stderr)
    if len(properties) != 10 or any(set(p) != {'0'} for p in properties):
        raise ValueError('all ten kernels must have no stack or spills')
    result = []
    for words in (1, 4):
        for copy in (0, 1):
            name = f'_Z12packet_probeILi{words}ELb{copy}EE'
            blocks = [b for b in sass.split('Function : ') if b.startswith(name)]
            if len(blocks) != 1:
                raise ValueError('one packet variant per physical program required')
            ops = re.findall(r'/\*([0-9a-f]+)\*/\s+([^;]+);', blocks[0])
            clocks = [i for i, (_, op) in enumerate(ops) if 'SR_CLOCKLO' in op]
            if len(clocks) != 4:
                raise ValueError('outer and inner clock endpoints required')
            timed = ops[clocks[1]:clocks[2]+1]
            text = '\n'.join(op for _, op in timed)
            active = [op for _, op in timed if not op.startswith('@!PT ')]
            copies = [op for op in active if 'LDGSTS.' in op]
            if len(copies) != copy:
                raise ValueError('copy/control instruction count mismatch')
            if copy and (('BYPASS.128' in copies[0]) != (words == 4)):
                raise ValueError('ca4B versus cg16B mismatch')
            if not any('DEPBAR.LE SB0, 0x0' in op for op in active):
                raise ValueError('missing wait_group0')
            loads = [i for i, op in enumerate(active) if re.match(r'LDS\s', op)]
            if len(loads) != 1:
                raise ValueError('one active dependent shared consumption required')
            load_index = loads[0]
            loaded = re.search(r'LDS (R\d+),', active[load_index])[1]
            after = active[load_index+1:]
            if (not after or not after[0].startswith('ISETP.') or loaded not in after[0]
                    or not any('BRA ' in op for op in after)
                    or not any('BPT.TRAP' in op for op in after)):
                raise ValueError('shared data must control a real branch before end clock')
            before = active[:load_index]
            if not any('BRA ' in op for op in before) or not any('BPT.TRAP' in op for op in before):
                raise ValueError('start clock dependency guard missing')
            forbidden = ('HMMA', 'HGMMA', 'UTMALDG', 'STG.', 'LDG.', 'STS ', 'LDL', 'STL')
            if any(token in text for token in forbidden):
                raise ValueError('foreign timed payload or local traffic')
            # Compiler descriptor loads remain part of complete issue response.
            # Retain them explicitly rather than claiming an intrinsic latency.
            constants = [op for op in active if 'LDC' in op]
            result.append(dict(words=words, copy=bool(copy), timed_operations=active,
                               timed_constant_reads=constants, inner_clock_count=2))
    return result


def measurement(text, words, sparse, grid, trial, samples=21, counter=False):
    rows = list(csv.reader(line for line in text.splitlines() if line.startswith(('I,', 'P,', 'R,', 'END,'))))
    identity = [r for r in rows if r[0] == 'I']
    if len(identity) != 1 or 'H200' not in identity[0][1] or list(map(int, identity[0][2:])) != [132,256,196608,words,sparse,grid,trial]:
        raise ValueError('physical identity mismatch')
    if [r for r in rows if r[0] == 'END'] != [['END','0']]:
        raise ValueError('incomplete or failed numerical run')
    preludes = [list(map(int, r[1:])) for r in rows if r[0] == 'P']
    if len(preludes) != 132 or {r[4] for r in preludes} != set(range(132)) or len({r[5] for r in preludes}) != 132:
        raise ValueError('missing prelude or SM placement')
    if any(r[:4] != [words,sparse,grid,trial] or r[6] <= 0 or r[7] < 3000000 for r in preludes):
        raise ValueError('invalid prelude clock frame')
    parsed = [list(map(int, r[1:])) for r in rows if r[0] == 'R']
    copies = (1,) if counter else (0,1)
    expected_count = samples*len(copies)*grid*8
    if len(parsed) != expected_count or any(len(r) != 16 for r in parsed):
        raise ValueError('incomplete all-warp sample population')
    keyed = {}
    for r in parsed:
        if r[:4] != [words,sparse,grid,trial]:
            raise ValueError('foreign cell in run')
        sample,copy,cta,warp,sm,c0,c1,g0,g1,outer_c,outer_g,error = r[4:]
        key = (sample,copy,cta,warp)
        if (key in keyed or not 0 <= sample < samples or copy not in copies or not 0 <= cta < grid
                or not 0 <= warp < 8 or not 0 <= sm < 132 or c1 <= c0 or g1 < g0
                or outer_c <= 0 or outer_g <= 0 or error != 0):
            raise ValueError('invalid, duplicate or numerically failed warp')
        keyed[key] = (sm,c0,c1,g0,g1,outer_c,outer_g)
    responses = {copy: [] for copy in copies}
    warp_responses = {copy: [] for copy in copies}
    gt_responses = {copy: [] for copy in copies}
    outer_clocks = []
    for sample in range(samples):
        for copy in copies:
            sms = []
            for cta in range(grid):
                values = [keyed[(sample,copy,cta,warp)] for warp in range(8)]
                if len({v[0] for v in values}) != 1:
                    raise ValueError('CTA warp SM mismatch')
                sms.append(values[0][0])
                responses[copy].append(max(v[2] for v in values)-min(v[1] for v in values))
                gt_responses[copy].append(max(v[4] for v in values)-min(v[3] for v in values))
                warp_responses[copy].extend(v[2]-v[1] for v in values)
                outer_clocks.extend(v[5]/v[6]*1000 for v in values)
            if len(set(sms)) != grid:
                raise ValueError('grid does not occupy distinct SMs')
    def stats(values):
        ordered = sorted(values)
        return dict(count=len(values), median=median(values), p10=ordered[len(ordered)//10],
                    p90=ordered[len(ordered)*9//10], minimum=ordered[0], maximum=ordered[-1])
    return dict(words=words,sparse=sparse,grid=grid,trial=trial,warps=len(parsed),
        all_lane_numeric_pass=True,distinct_sms_pass=True,
        prelude_mhz_range=[min(r[6]/r[7]*1000 for r in preludes), max(r[6]/r[7]*1000 for r in preludes)],
        outer_mhz_range=[min(outer_clocks),max(outer_clocks)],
        responses={str(copy):dict(cta_cycles=stats(responses[copy]),warp_cycles=stats(warp_responses[copy]),
                       cta_globaltimer_ns=stats(gt_responses[copy])) for copy in copies},
        descriptive_copy_minus_control_cycles=None if counter else median(responses[1])-median(responses[0]))


def counter_proof(text, words):
    lines = text.splitlines()
    heads = [i for i,line in enumerate(lines) if line.startswith('"ID","Process ID"')]
    if len(heads) != 1:
        raise ValueError('one wide NCU table required')
    rows = [r for r in csv.DictReader(lines[heads[0]:]) if r['ID'] != '']
    if len(rows) != 1 or re.search(rf'packet_probe<{words},(?:true|1)>',rows[0]['Kernel Name'].replace(' ','')) is None:
        raise ValueError('counter must select only the requested copy kernel')
    row = rows[0]
    values = {metric:float(row[metric].replace(',','')) for metric in METRICS}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('missing/nonfinite counter is unknown, never zero')
    payload = 132*256*words*4
    l1 = 32*values['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
    tex = 32*values['lts__t_sectors_srcunit_tex_op_read.sum']
    dram = values['dram__bytes_read.sum']
    checks = dict(exact_ctas=values['sm__ctas_launched.sum']==132,
        l1_requested_payload=abs(l1/payload-1)<=0.01,
        l1_cold_sector_support=tex >= 0.99*payload,
        l2_warm_dram_support=tex>0 and dram<=0.01*tex)
    return dict(counters=values,payload_bytes=payload,l1tex_bytes=l1,lts_tex_bytes=tex,
        dram_read_bytes=dram,checks=checks,cache_condition_supported=all(checks.values()),
        lts_total_equals_tex_assumed=False,profiled_durations_consumed=False,
        cache_support_scope='Aggregate counters in this fullgrid target copy only; not per-warp hit classification or grid1 evidence')


def counter_measurements(text, words, sparse):
    """Validate every application-replay process, never pick one passing run."""
    frames=[]
    current=[]
    for line in text.splitlines():
        if line.startswith('I,'):
            if current:raise ValueError('unfinished application-replay frame')
            current=[line]
        elif line.startswith(('P,','R,','END,')):
            if not current:raise ValueError('application-replay data outside identity frame')
            current.append(line)
            if line.startswith('END,'):
                frame=measurement('\n'.join(current),words,sparse,132,0,samples=1,counter=True)
                frames.append({k:frame[k] for k in ('warps','all_lane_numeric_pass','distinct_sms_pass')})
                current=[]
    if current or not frames:raise ValueError('incomplete application-replay population')
    return frames


def analyze(raw):
    code = code_proof(raw['sass']['stdout'],raw['build']['stderr'])
    timing = []
    keys = set()
    for item in raw['timing']:
        key = tuple(item[k] for k in ('words','sparse','grid','trial'))
        if key in keys or item['measurement']['exit_code'] != 0:
            raise ValueError('duplicate or failed timing command')
        keys.add(key)
        timing.append(measurement(item['measurement']['stdout'],*key))
    required = {(w,s,g,t) for w in (1,4) for s in (0,1) for g in (1,132) for t in range(3)}
    if keys != required:
        raise ValueError('all 24 trial cells required')
    counters = []
    keys = set()
    for item in raw['counters']:
        key = (item['words'],item['sparse'])
        if key in keys or item['measurement']['exit_code'] != 0:
            raise ValueError('duplicate or failed counter command')
        keys.add(key)
        # App replay also has complete numeric output. Do not consume its times.
        frames=counter_measurements(item['measurement']['stdout'],*key)
        counters.append(dict(words=key[0],sparse=key[1],replay_numeric_frames=frames,
                             **counter_proof(item['measurement']['stdout'],key[0])))
    required_counters={(w,s) for w in (1,4) for s in (0,1)}
    if not keys <= required_counters:raise ValueError('foreign target-copy counter cell')
    missing_counters=sorted(required_counters-keys)
    cells = []
    for words in (1,4):
        for sparse in (0,1):
            for grid in (1,132):
                group = [v for v in timing if (v['words'],v['sparse'],v['grid'])==(words,sparse,grid)]
                medians = [v['responses']['1']['cta_cycles']['median'] for v in group]
                cells.append(dict(words=words,sparse=sparse,grid=grid,
                    copy_median_cycles_by_trial=medians,
                    control_median_cycles_by_trial=[v['responses']['0']['cta_cycles']['median'] for v in group],
                    descriptive_delta_cycles_by_trial=[v['descriptive_copy_minus_control_cycles'] for v in group],
                    trial_median_spread_fraction=(max(medians)-min(medians))/median(medians)))
    return dict(status='diagnostic_incomplete' if missing_counters else 'diagnostic_complete',
        raw_capture_status=raw['status'],code=code,timing=timing,counters=counters,cells=cells,
        counter_population_complete=not missing_counters,missing_counter_cells=missing_counters,
        cache_condition_supported=not missing_counters and all(v['cache_condition_supported'] for v in counters),
        cache_counter_grids=[132],grid1_cache_condition='unprofiled',
        timed_warp_records=sum(v['warps'] for v in timing),
        Forward_admission=False,parameter_emitted=False,
        interpretation='Complete finite copy/commit/wait/dependent shared-use response plus frame; control differences descriptive only. No intrinsic L2 latency or source-residency transfer established.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    raw_text=gzip.decompress(args.raw.read_bytes()).decode() if args.raw.suffix=='.gz' else args.raw.read_text()
    result=analyze(json.loads(raw_text))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cache_condition_supported','timed_warp_records','Forward_admission')}))
