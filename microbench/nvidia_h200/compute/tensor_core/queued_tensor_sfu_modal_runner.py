"""Independent continuous SS/RS queue and bounded EX2 streams."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import modal

TASK = Path('/tmp/midlayer-queued-tensor-sfu-20261107')
if (TASK / 'queued_tensor_sfu.cu').is_file():
    SOURCE = TASK / 'queued_tensor_sfu.cu'
    REFERENCE = TASK / 'fa3_wgmma_service.cu'
    GENERATOR = TASK / 'fa3_wgmma_service_gen.py'
    WARMUP = TASK / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.debian_slim()
else:
    HERE = Path(__file__).resolve().parent
    SOURCE = HERE / 'queued_tensor_sfu.cu'
    REFERENCE = HERE.parents[1] / 'operators/flash_attn/fa3_wgmma_service.cu'
    GENERATOR = HERE.parents[1] / 'operators/flash_attn/fa3_wgmma_service_gen.py'
    WARMUP = HERE.parent / 'gpu_clock_warmup.cu'
    IMAGE = modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04', add_python='3.11')
    for path in (SOURCE, REFERENCE, GENERATOR, WARMUP):
        IMAGE = IMAGE.add_local_file(path, remote_path=str(TASK / path.name))
APP = modal.App('midlayer-queued-tensor-sfu-20261107')


def helper(reference, generator):
    marker = '__device__ __forceinline__ void wgmma_ss_n64_kmajor'
    if reference.count(marker) != 1:
        raise ValueError('Independent helper preamble changed')
    functions = [n for n in ast.parse(generator).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'wgmma_function']
    if len(functions) != 1:
        raise ValueError('Independent PTX generator changed')
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<primitive-generator>', 'exec'), namespace)
    return (reference[:reference.index(marker)]
            + namespace['wgmma_function']('ss_n128_kmajor', 128, False, 0)
            + namespace['wgmma_function']('rs_n128_mnmajor', 128, True, 1))


def code_gate(sass, k_steps=4, accumulator_banks=2, post_join=False, bracket_clock=False, cover_sample=None):
    if (k_steps, accumulator_banks) not in ((4, 2), (8, 1)):
        raise ValueError('Queued WGMMA scope requires K4/two-bank or full-A K8/one-bank')
    if bracket_clock and (k_steps != 8 or not post_join):
        raise ValueError('Bracket clock requires the full-A post-join scope')
    if cover_sample is not None and (type(cover_sample) is not bool
            or k_steps != 8 or not post_join or bracket_clock):
        raise ValueError('Cover sample requires the full-A non-bracket scope')
    from collections import Counter
    bodies = sass.split('Function : ')[1:]
    if len(bodies) != 12 or 'WG.DP' in sass or re.search(r'\b(?:LDL|STL)\b', sass):
        return False
    seen = set()
    for body in bodies:
        match = re.match(r'_Z6queuedILb([01])ELi([14])ELi(32|64|128)EE', body)
        if match is None:
            return False
        register_a, depth, width = map(int, match.groups())
        seen.add((register_a, depth, width))
        matrix = re.findall(r'HGMMA\.64x128x16\.F32\.BF16 (R\d+),', body)
        if len(matrix) != accumulator_banks * k_steps or sorted(Counter(matrix).values()) != [k_steps] * accumulator_banks:
            return False
        batches = [p.count('HGMMA.') for p in body.split('WARPGROUP.DEPBAR') if 'HGMMA.' in p]
        waits = [int(v, 16) for v in re.findall(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x([0-9a-f]+)', body)]
        expected_waits = [depth - 1] * accumulator_banks + [0]
        if (batches != [k_steps] * accumulator_banks
                or waits not in ([expected_waits, expected_waits + [0]] if bracket_clock else [expected_waits])):
            return False
        ex2 = re.findall(r'MUFU\.EX2 (R\d+), -(R\d+)', body)
        if len(ex2) != width or any(a != b for a, b in ex2):
            return False
        if body.count('WARPGROUP.ARRIVE') != 2:
            return False
        if cover_sample is not None:
            stamps = 8 if cover_sample else 6
            if body.count('SR_CLOCKLO') != stamps or body.count('SR_GLOBALTIMERLO') != stamps:
                return False
        if post_join:
            instructions = [(int(pc, 16), text.strip()) for pc, text in
                re.findall(r'/\*([0-9a-f]+)\*/\s+([^;]+);', body)]
            if bracket_clock:
                # A predicated PTX mov can become an unordered CS2R+SEL.
                # Require actual data-dependent control flow around a trap.
                transitions = []
                instruction_at = dict(instructions)
                for index, (pc, text) in enumerate(instructions):
                    read = re.fullmatch(r'CS2R R(\d+), (SR_GLOBALTIMERLO|SR_CLOCKLO)', text)
                    if read is None or index + 5 >= len(instructions):
                        continue
                    register, counter = int(read[1]), read[2]
                    low, high, branch = instructions[index + 1:index + 4]
                    compare = re.fullmatch(r'ISETP\.NE\.U32\.AND (P\d+), PT, R' + str(register) + r', RZ, PT', low[1])
                    if compare is None:
                        continue
                    predicate = compare[1]
                    # CUDA12.2 emits an inline trap;12.6 can outline the
                    # terminal error block. Both branch on the actual read.
                    jump = re.fullmatch(r'@(!?)' + predicate + r' BRA (0x[0-9a-f]+)', branch[1])
                    if jump is None:
                        return False
                    target = int(jump[2], 16)
                    if jump[1] == '!':
                        following = instructions[index + 4]
                        if target not in instruction_at:
                            return False
                        if instruction_at[target] == 'WARPGROUP.DEPBAR.LE gsb0, 0x0':
                            target += 16
                        if target not in instruction_at or instruction_at[target] != 'BPT.TRAP 0x1':
                            return False
                    else:
                        trap, following = instructions[index + 4:index + 6]
                        if trap[1] != 'BPT.TRAP 0x1' or target != following[0]:
                            return False
                    next_read = re.fullmatch(r'CS2R R\d+, (SR_GLOBALTIMERLO|SR_CLOCKLO)', following[1])
                    if (high[1] != f'ISETP.NE.U32.AND.EX {predicate}, PT, R{register + 1}, RZ, PT, {predicate}'
                            or next_read is None):
                        return False
                    transitions.append((pc, following[0], counter, next_read[1]))
                starts = [item for item in transitions if item[2:] == ('SR_GLOBALTIMERLO', 'SR_CLOCKLO')]
                ends = [item for item in transitions if item[2:] == ('SR_CLOCKLO', 'SR_GLOBALTIMERLO')]
                if (len(starts) != 4 or len(ends) != 4
                        or {item[1] for item in starts} != {item[0] for item in ends}
                        or sum(text == 'BPT.TRAP 0x1' for _, text in instructions) != 8):
                    return False
            matrix_pcs = [pc for pc, text in instructions if text.startswith('HGMMA.')]
            scalar_pcs = [pc for pc, text in instructions if text.startswith('MUFU.EX2')]
            if cover_sample is not None:
                # Timestamp common-subexpression motion across the release
                # barrier invalidates full-window coverage even if counts match.
                releases = [pc for pc, text in instructions
                    if pc < min(scalar_pcs)
                    and re.fullmatch(r'BAR\.SYNC(?:\.DEFER_BLOCKING)? 0x4, 0x100', text)]
                if not releases:
                    return False
                after_release = [text for pc, text in instructions
                    if max(releases) < pc < min(scalar_pcs)]
                if (not any('SR_CLOCKLO' in text for text in after_release)
                        or not any('SR_GLOBALTIMERLO' in text for text in after_release)):
                    return False
            joins = [pc for pc, text in instructions
                if re.fullmatch(r'BAR\.SYNC(?:\.DEFER_BLOCKING)? 0x5, 0x100', text)]
            if not joins:
                return False
            zero_waits = [pc for pc, text in instructions
                if pc > max(matrix_pcs) and text == 'WARPGROUP.DEPBAR.LE gsb0, 0x0']
            if not zero_waits:
                return False
            # Exclude queue1's in-loop wait0 and outlined terminal trap cleanup.
            recurrence_backs = [pc for pc, text in instructions
                for target in re.findall(r'BRA (0x[0-9a-f]+)', text)
                if max(matrix_pcs) < pc < min(joins) and int(target, 16) <= min(matrix_pcs)]
            terminal_waits = [pc for pc in zero_waits if recurrence_backs and pc > max(recurrence_backs)]
            if not terminal_waits:
                return False
            final_wait = min(terminal_waits)
            if bracket_clock:
                extra_waits = [pc for pc in zero_waits if pc > final_wait]
                if any(pc < min(joins) or pc + 16 not in instruction_at
                        or instruction_at[pc + 16] != 'BPT.TRAP 0x1' for pc in extra_waits):
                    return False
            for end in (final_wait, max(scalar_pcs)):
                following = [pc for pc in joins if pc > end]
                if not following:
                    return False
                between = [text for pc, text in instructions if end < pc < min(following)]
                if (not any('SR_GLOBALTIMERLO' in text for text in between)
                        or not any('SR_CLOCKLO' in text for text in between)
                        or any(re.match(r'(?:@!?P\d+\s+)?FSETP\.', text) for text in between)):
                    return False
            if register_a:
                fragments = [int(value) for value in re.findall(
                    r'HGMMA\.64x128x16\.F32\.BF16 R\d+, R(\d+),', body)]
                words = {value + lane for value in fragments for lane in range(4)}
                if len(fragments) != 8 or len(words) != 32:
                    return False
                backedges = [(pc, int(target, 16)) for pc, text in instructions
                    for target in re.findall(r'BRA (0x[0-9a-f]+)', text)
                    if max(matrix_pcs) < pc < final_wait and int(target, 16) <= min(matrix_pcs)]
                if len(backedges) != 1:
                    return False
                back, start = backedges[0]
                for pc, text in instructions:
                    if start <= pc <= back:
                        if re.search(r'\b(?:LDS|LDG|LDC|LDL|STL)\b', text):
                            return False
                        destination = re.match(r'(?:@!?P\d+\s+)?(\S+) R(\d+),', text)
                        if destination is not None and int(destination[2]) in words:
                            return False
    return seen == {(r, d, w) for r in (0, 1) for d in (1, 4) for w in (32, 64, 128)}


def run(command, environment, timeout):
    process = subprocess.Popen(command, cwd=TASK, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    return {'command': command, 'exit_code': process.returncode, 'timed_out': timed_out, 'stdout': out, 'stderr': err}


@APP.function(gpu='H200', image=IMAGE, timeout=120, retries=0, single_use_containers=True)
def capture(k_steps: int = 4, post_join: bool = False, accumulator_banks: int = 2, bracket_clock: bool = False, coactive_cover: bool = False):
    started = time.monotonic()
    result = {}
    try:
        environment = dict(os.environ)
        for key, folder in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cuda'), ('XDG_CACHE_HOME', 'xdg')):
            path = TASK / folder
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment['CUDA_VISIBLE_DEVICES'] = '0'
        header = helper(REFERENCE.read_text(), GENERATOR.read_text())
        (TASK / 'queued_wgmma_helper.cuh').write_text(header)
        result['helper_source'] = header
        if ((k_steps, accumulator_banks) not in ((4, 2), (8, 1))
                or type(post_join) is not bool or type(bracket_clock) is not bool or type(coactive_cover) is not bool
                or k_steps == 8 and not post_join
                or bracket_clock and (k_steps != 8 or not post_join)
                or coactive_cover and (k_steps != 8 or not post_join or bracket_clock)):
            raise ValueError('Invalid queued mechanism scope')
        result['mechanism_scope'] = {'k_steps': k_steps,
            'register_A_words_per_thread': 4 * k_steps,
            'accumulator_banks': accumulator_banks,
            'post_timing_CTA_validation_join': post_join}
        if bracket_clock:
            result['mechanism_scope']['bracket_clock'] = True
        if coactive_cover:
            result['mechanism_scope']['coactive_cover'] = dict(tensor_repeat_multiplier=4, scalar_repeats=[256,512,1024], matrix_markers=[32,128], scalar_release_after_matrix_packets=8, matched_control=True)
        commands = {
            'identity': ['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm', '--format=csv,noheader'],
            'compiler': ['nvcc', '--version'],
            'build': ['nvcc', '-O3', '-arch=sm_90a', '-Xptxas=-v',
                      f'-DQUEUED_WGMMA_K_STEPS={k_steps}', f'-DQUEUED_POST_JOIN={int(post_join)}',
                      f'-DQUEUED_ACCUMULATOR_BANKS={accumulator_banks}',
                      f'-DQUEUED_BRACKET_CLOCK={int(bracket_clock)}',
                      f'-DQUEUED_COVER_SCALAR={int(coactive_cover)}', f'-DQUEUED_COVER_SAMPLE={int(coactive_cover)}',
                      str(TASK / SOURCE.name), '-o', str(TASK / 'probe')],
            'sass': ['cuobjdump', '--dump-sass', str(TASK / 'probe')],
            'build_warmup': ['nvcc', '-O3', '-arch=sm_90', str(TASK / WARMUP.name), '-o', str(TASK / 'warmup')],
            'warmup': ['./warmup', '600'],
            'before': ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader'],
            'plain': ['./probe'],
            'after': ['nvidia-smi', '--query-gpu=clocks.sm', '--format=csv,noheader'],
        }
        if coactive_cover:
            build_control = [arg.replace('-DQUEUED_COVER_SAMPLE=1','-DQUEUED_COVER_SAMPLE=0')
                if arg != str(TASK / 'probe') else str(TASK / 'control') for arg in commands['build']]
            commands = dict(list(commands.items())[:4] + [
                ('build_control',build_control),('sass_control',['cuobjdump','--dump-sass',str(TASK / 'control')])]
                + list(commands.items())[4:])
            # Complete the paired uninstrumented protocol before the sample.
            names = list(commands)
            commands = dict(list(commands.items())[:names.index('plain')]
                + [('control',['./control'])] + list(commands.items())[names.index('plain'):])
        result['required_commands'] = list(commands)
        for name, command in commands.items():
            remaining = 110 - (time.monotonic() - started)
            if remaining <= 0:
                result['deadline_exceeded'] = True
                break
            item = run(command, environment, min(75, remaining))
            result[name] = item
            if item['exit_code'] != 0:
                break
            if name == 'identity' and (len(item['stdout'].strip().splitlines()) != 1 or not item['stdout'].startswith('NVIDIA H200,')):
                result['identity_gate'] = False
                break
            if name in ('build','build_control'):
                spills = re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', item['stderr'])
                stacks = re.findall(r'(\d+) bytes stack frame', item['stderr'])
                spill_key = 'zero_spill_gate' if name == 'build' else 'control_zero_spill_gate'
                result[spill_key] = (len(spills) == len(stacks) == 12
                    and all(a == b == '0' for a, b in spills) and all(size == '0' for size in stacks))
                if not result[spill_key]:
                    break
            if name in ('sass','sass_control'):
                gate_key = 'code_gate' if name == 'sass' else 'control_code_gate'
                result[gate_key] = code_gate(item['stdout'], k_steps, accumulator_banks, post_join, bracket_clock,
                    (name == 'sass') if coactive_cover else None)
                if not result[gate_key]:
                    break
        return result
    finally:
        result['gpu_function_wall_s'] = time.monotonic() - started
        if TASK.exists():
            shutil.rmtree(TASK)


@APP.local_entrypoint()
def main(output_dir: str, k_steps: int = 4, post_join: bool = False, accumulator_banks: int = 2, bracket_clock: bool = False, coactive_cover: bool = False):
    out = Path(output_dir)
    if out.exists():
        raise ValueError('Queued Tensor/SFU capture requires a new attempt directory')
    if ((k_steps, accumulator_banks) not in ((4, 2), (8, 1))
            or k_steps == 8 and not post_join
            or bracket_clock and (k_steps != 8 or not post_join)
                or coactive_cover and (k_steps != 8 or not post_join or bracket_clock)):
        raise ValueError('Queued WGMMA scope requires K4/two-bank or K8/one-bank/post-join')
    result = capture.remote(k_steps, post_join, accumulator_banks, bracket_clock, coactive_cover)
    out.mkdir(parents=True)
    files = {}
    for path in (SOURCE, REFERENCE, GENERATOR, WARMUP, Path(__file__)):
        content = path.read_bytes()
        (out / path.name).write_bytes(content)
        files[path.name] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
    result['source_files'] = files
    (out / 'raw.json').write_text(json.dumps(result, indent=2) + '\n')
    if not all(k in result and result[k]['exit_code'] == 0 for k in result['required_commands']):
        raise RuntimeError('Queued Tensor/SFU capture incomplete; raw retained')
    print({'status': 'raw_independent_queued_capacity', 'wall_s': result['gpu_function_wall_s']})
