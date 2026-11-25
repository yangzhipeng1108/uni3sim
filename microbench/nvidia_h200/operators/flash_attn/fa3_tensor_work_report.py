"""Cross-check N176 physical matrix work using retained non-duration counts."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--campaign', type=Path, required=True)
parser.add_argument('--peak-checks', type=Path, required=True)
parser.add_argument('--scalar-work-only', action='store_true',
                    help='Write a separate coarse arithmetic census; preserve the older matrix report')
options = parser.parse_args()
P = options.campaign.resolve()
raw_path = P / 'attempt_01/raw.json'
raw = json.loads(raw_path.read_text())
plan_path = P / 'plan.json'
plan = json.loads(plan_path.read_text())
assert raw['plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert [c['case_id'] for c in raw['cases']] == [c['case_id'] for c in plan['cases']]
cases = []
scalar_cases = []
for case, planned in zip(raw['cases'], plan['cases']):
    args = planned['plan']['runner_args']
    if args[args.index('--kv_layout')+1] != 'contiguous':
        continue
    assert case['status'] == 'raw_diagnostic_complete'
    assert case['telemetry']['clock_accepted'] and case['source_export']['exit_code'] == 0
    batch = int(args[args.index('--batch')+1])
    heads = int(args[args.index('--num_heads_q')+1])
    assert heads == int(args[args.index('--num_heads_kv')+1])
    assert int(args[args.index('--head_dim')+1]) == 128
    assert int(args[args.index('--seq_len_q')+1]) == 1
    assert int(args[args.index('--num_splits')+1]) == 1
    assert args[args.index('--causal')+1] == '0'
    assert json.loads(args[args.index('--q-lens-json')+1]) == [1]*batch
    kv_lens = json.loads(args[args.index('--kv-lens-json')+1])
    assert len(kv_lens) == batch and all(kv > 0 for kv in kv_lens)
    for signature in planned['canonical_launch_signature']:
        assert 'cute::C<(int)64>, cute::C<(int)176>, cute::C<(int)128>' in signature[0]
        assert signature[1:] == [[132, 1, 1], [256, 1, 1]]
    panels = heads * sum(math.ceil(kv/176) for kv in kv_lens)
    # A warpgroup has four warps. Each source panel has eight K16 QK
    # instructions and eleven K16 PV instructions; padded M64 is retained.
    expected = {'HGMMA.64x176x16.F32.BF16': panels*8*4,
                'HGMMA.64x128x16.F32.BF16': panels*11*4}
    lines = case['source_export']['stdout'].splitlines()
    spans = [(m.start(), m.end()) for m in re.finditer('-+', lines[0])]
    header = next(i for i, line in enumerate(lines) if line.startswith('Address '))
    start = next(i for i in range(header, len(lines)) if lines[i].startswith('0x'))
    names = [''.join(lines[i][a:b].strip() for i in range(header, start)).replace(' ', '')
             for a, b in spans]
    counts = Counter()
    scalar_counts = Counter()
    for line in lines:
        if not line.startswith('0x'):
            continue
        row = [line[a:b].strip() for a, b in spans]
        op = re.sub(r'^@!?\w+\s+', '', row[1]).split()[0]
        category = op if op.startswith('MUFU.') else op.split('.')[0]
        scalar_counts[category] += int(row[names.index('InstructionsExecuted')].replace(',', ''))
        if op.startswith('HGMMA'):
            counts[op] += int(row[names.index('InstructionsExecuted')].replace(',', ''))
    assert dict(counts) == expected
    qk_flops = counts['HGMMA.64x176x16.F32.BF16']/4 * 2*64*176*16
    pv_flops = counts['HGMMA.64x128x16.F32.BF16']/4 * 2*64*128*16
    assert qk_flops == pv_flops == panels*2*64*176*128
    cases.append(dict(case_id=case['case_id'], panel_tasks=panels,
        expected_warp_instructions=expected, observed_warp_instructions=dict(counts),
        qk_flops=qk_flops, pv_flops=pv_flops, padded_M_retained=True))
    if options.scalar_work_only:
        tasks = batch * heads
        steady = panels - tasks
        # Two row states/thread, N/4=44 score values/row, four warps/WG.
        # First max/sum assign a seed; later panels also update old row
        # state. Final quad sum has two shuffle/add levels per row.
        expected_scalar = {
            'FMNMX': 4 * (2 * (43 + 2) * tasks + 2 * (44 + 2) * steady),
            'FADD': 4 * (2 * 43 * tasks + 2 * 44 * steady + 4 * tasks),
            'MUFU.EX2': 4 * (88 * panels + 2 * steady),
            'FFMA': 4 * 90 * panels,
            'F2FP': 4 * (44 * panels + 32 * tasks),
            # O is scaled after each steady update and final normalization,
            # never before the first PV. Remaining FMULs update row state.
            'FMUL': 4 * (64 * (steady + tasks)
                         + 2 * panels + 2 * steady + 4 * tasks),
        }
        observed_scalar = {key: scalar_counts[key] for key in expected_scalar}
        assert observed_scalar == expected_scalar
        scalar_cases.append(dict(case_id=case['case_id'], tasks=tasks, panels=panels,
            steady_panels=steady, expected_warp_instructions=expected_scalar,
            observed_warp_instructions=observed_scalar,
            extra_first_O_scale_warp_instructions_in_old_schedule=4 * 64 * tasks))
assert len(cases) == 2
if options.scalar_work_only:
    result = dict(cases=scalar_cases,
        source_raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_duration_consumed=False, rates_selected=False,
        source_geometry='N176/D128 one WG, two rows/thread, quad4, 44 score values/row',
        decision='Six coarse arithmetic categories close for both complete launches; remove the prologue O rescale absent from source, retain other services.',
        limits=['Two retained unsplit BF16 N176 programs only; no complete native schedule reconstruction.',
                'Work counts validate source semantics, not latency, capacity or RF causality.',
                'No sampled stall count or production duration is consumed.'])
    (P/'scalar_work_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0)
peaks_path = options.peak_checks.resolve()
peaks = json.loads(peaks_path.read_text())
n176 = [c for c in peaks if c['columns'] == 176]
assert len(n176) == 8 and all(c['passed'] for c in n176)
assert all(c['expected'] == 88 for c in n176)
result = dict(cases=cases, source_raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    peak_checks_path=str(peaks_path), peak_checks_sha256=hashlib.sha256(peaks_path.read_bytes()).hexdigest(),
    independent_n176_peak_checks=n176,
    maximum_independent_n176_peak_error=max(abs(c['cycles_per_instruction']/88-1) for c in n176),
    production_duration_consumed=False, Forward_parameter_admitted=False,
    limits=['Two retained unsplit N176 BF16 native cases only; no other source class is certified.',
            'Matrix instruction work conservation does not establish execution overlap or local retirement.',
            'SourceCounters replay is not the canonical timing run; no duration or stall fraction is priced.'],
    decision='No evidence for omitted padded matrix work or an N176 matrix-capacity surcharge; keep existing services.')
(P/'tensor_work_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
