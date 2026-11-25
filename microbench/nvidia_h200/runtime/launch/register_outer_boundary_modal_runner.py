"""Independent CTA register-pool outer boundary with matched store-drain control."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3
from statistics import median
import subprocess
import time
import zlib
import modal

TASK=Path('/tmp/midlayer-register-outer-boundary-20261108')
if (TASK/'register_outer_boundary.cu').is_file():
    SOURCE=TASK/'register_outer_boundary.cu'
    IMAGE=modal.Image.debian_slim()
else:
    SOURCE=Path(__file__).with_name('register_outer_boundary.cu')
    IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11').apt_install('cuda-nsight-systems-12-6').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name))
APP=modal.App('midlayer-register-outer-boundary-20261108')

def pressure_helper():
    lines=['template<int Threads> __device__ __forceinline__ float pressure(float seed);']
    for threads,count in ((256,224),(384,144),(512,128)):
        lines += [f'template<> __device__ __forceinline__ float pressure<{threads}>(float seed) {{',f'float v[{count}];']
        lines += [f'v[{i}] = seed + {float(i)}f;' for i in range(count)]
        operands=', '.join(f'"+f"(v[{i}])' for i in range(count))
        lines += ['asm volatile("bar.warp.sync 0xffffffff;" : '+operands+');','float sum=0;']
        lines += [f'sum += v[{i}];' for i in range(count)]
        lines += ['return sum;','}']
    return '\n'.join(lines)+'\n'

def build_gate(log):
    spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',log)
    entries=re.findall(r"Compiling entry function '[^']*boundary_envelopeILi(256|384)ELb([01])ELb([01])EE.*?Used (\d+) registers",log,re.S)
    if len(spills)!=8 or not all(a==b=='0' for a,b in spills) or len(entries)!=8:
        return False
    for threads,high,drain,registers in entries:
        expected=255 if threads=='256' else 168
        if (high=='1' and int(registers)!=expected) or (high=='0' and int(registers)>32):
            return False
    return True

def code_gate(sass):
    bodies=sass.split('Function : ')[1:]
    if len(bodies)!=8:
        return False
    keys=set()
    for body in bodies:
        match=re.search(r'boundary_envelopeILi(256|384)ELb([01])ELb([01])EE',body)
        if match is None or 'LDL' in body or 'STL' in body:
            return False
        threads,high,drain=map(int,match.groups());keys.add((threads,high,drain))
        if body.count('USETMAXREG')!=2*high or ('MEMBAR.SC.GPU' in body)!=bool(drain):
            return False
        markers=[m.start() for m in re.finditer('SR_GLOBALTIMERLO',body)]
        if len(markers)!=2:
            return False
        if drain and not markers[0]<body.index('MEMBAR.SC.GPU')<markers[1]:
            return False
    return keys=={(t,h,d) for t in (256,384) for h in (0,1) for d in (0,1)}

def analyze(result):
    header='launch_id,grid,threads,high,drain,extended,hold_cycles,trial,registers,begin_ns,end_ns,body_cycles'
    lines=result['profile']['stdout'].splitlines()
    if lines.count(header)!=1:
        raise ValueError('Expected one complete profile header')
    rows=[{k:int(v) for k,v in r.items()} for r in csv.DictReader([header]+[l for l in lines if re.fullmatch(r'[0-9]+(?:,-?[0-9]+){11}',l)])]
    groups={}
    for r in rows:
        if r['launch_id'] not in groups:groups[r['launch_id']]=[]
        groups[r['launch_id']].append(r)
    if len(rows)!=31920 or set(groups)!=set(range(480)) or len(result['kernel_timestamps'])!=480:
        raise ValueError('Incomplete device/CUPTI launch population')
    fields=header.split(',')[:9];launches=[]
    for index,kernel in enumerate(result['kernel_timestamps']):
        group=groups[index];r=group[0];start,end,grid,threads,shared,symbol=kernel
        if (len(group)!=grid or r['grid']!=grid or r['threads']!=threads or shared!=131072
            or any(any(x[f]!=r[f] for f in fields) for x in group)
            or f'<(int){threads}, (bool){r["high"]}, (bool){r["drain"]}>' not in symbol):
            raise ValueError('CUPTI/device geometry or source-key mismatch')
        body=max(x['end_ns'] for x in group)-min(x['begin_ns'] for x in group)
        if not 0<body<=end-start or any(x['body_cycles']<=0 for x in group):
            raise ValueError('Invalid outer/body interval')
        launches.append({**{f:r[f] for f in fields},'kernel_ns':end-start,'body_span_ns':body,'outer_ns':end-start-body})
    cells={}
    for r in launches:
        if r['trial']<0:continue
        key=tuple(r[f] for f in fields[1:7])
        if key not in cells:cells[key]=[]
        cells[key].append(r['outer_ns'])
    if len(cells)!=96 or any(len(v)!=3 for v in cells.values()):
        raise ValueError('Incomplete outer cell population')
    med={k:median(v) for k,v in cells.items()};differences=[]
    for t in (256,384):
        for g in (1,132):
            for drain in (0,1):
                for extended in (0,1):
                    for hold in (0,4096,32768):
                        low=med[g,t,0,drain,extended,hold];high=med[g,t,1,drain,extended,hold]
                        differences.append({'threads':t,'grid':g,'drain':drain,'extended':extended,'hold_cycles':hold,'low_outer_ns':low,'high_outer_ns':high,'increment_ns':high-low})
    calibration=med[132,384,1,1,1,32768]-med[132,384,0,1,1,32768]
    heldout=[r for r in differences if r['drain']==1 and (r['threads'],r['grid'],r['extended'],r['hold_cycles'])!=(384,132,1,32768)]
    tolerance=max(64.,.2*abs(calibration))
    maximum=max(abs(r['increment_ns']-calibration) for r in heldout)
    summary={'kernel_count':480,'timed_launches':288,'cta_trace_rows':len(rows),'complete_numeric_source_geometry_gate':True,
        'calibration_increment_ns':calibration,'heldout_controls':len(heldout),'predeclared_tolerance_ns':tolerance,
        'maximum_heldout_deviation_ns':maximum,'independent_heldout_gate':calibration>0 and maximum<=tolerance,
        'differences':differences,'forward_parameter_admitted':False,
        'scope':'Independent drained-store/elected-marker outer increment. Native register geometry and duplicate-cost exclusion still required.'}
    return launches,summary

@APP.function(gpu='H200',image=IMAGE,timeout=120,retries=0,single_use_containers=True)
def capture():
    started=time.monotonic();result={'reports':{}}
    try:
        helper=pressure_helper();(TASK/'outer_pressure.cuh').write_text(helper);result['generated_helper']=helper;stem=TASK/'trace'
        commands={
            'identity':['nvidia-smi','--query-gpu=name,driver_version,clocks.sm','--format=csv,noheader'],
            'compiler':['nvcc','--version'],
            'build':['nvcc','-O3','-arch=sm_90a','-Xptxas=-v',str(TASK/SOURCE.name),'-o',str(TASK/'probe')],
            'sass':['cuobjdump','--dump-sass',str(TASK/'probe')],
            'profiler_version':['nsys','--version'],
            'plain':[str(TASK/'probe')],
            'profile':['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','--output',str(stem),str(TASK/'probe')],
            'export':['nsys','export','--type=sqlite','--force-overwrite=true','--output',str(stem)+'.sqlite',str(stem)+'.nsys-rep']}
        result['required_commands']=list(commands)
        for name,command in commands.items():
            p=subprocess.run(command,capture_output=True,text=True,timeout=75,cwd=TASK)
            result[name]={'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
            if p.returncode:break
            if name=='identity' and not p.stdout.startswith('NVIDIA H200,'):raise ValueError('Authorized H200 required')
            if name=='build':
                result['build_gate']=build_gate(p.stderr)
                if not result['build_gate']:break
            if name=='sass':
                result['code_gate']=code_gate(p.stdout)
                if not result['code_gate']:break
            if name=='export':
                with sqlite3.connect(str(stem)+'.sqlite') as db:
                    result['kernel_timestamps']=db.execute("SELECT k.start,k.end,k.gridX,k.blockX,k.dynamicSharedMemory,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE '%boundary_envelope%' ORDER BY k.start").fetchall()
                result['reports']['trace.nsys-rep.zlib']=zlib.compress(Path(str(stem)+'.nsys-rep').read_bytes())
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-started
        if TASK.exists():shutil.rmtree(TASK)

@APP.local_entrypoint()
def main(output_dir:str):
    out=Path(output_dir)
    if out.exists():raise ValueError('New attempt directory required')
    result=capture.remote();out.mkdir(parents=True)
    for name,payload in result['reports'].items():(out/name).write_bytes(payload)
    result['reports']={name:{'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),'size':(out/name).stat().st_size,'compression':'zlib'} for name in result['reports']}
    result['source_files']={}
    for path in (SOURCE,Path(__file__)):
        data=path.read_bytes();(out/path.name).write_bytes(data);result['source_files'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data)}
    (out/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(k in result and result[k]['exit_code']==0 for k in result['required_commands']):raise RuntimeError('Capture incomplete;raw retained')
    launches,summary=analyze(result);(out/'launch_spans.json').write_text(json.dumps(launches,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({'function_wall_s_not_cost':result['gpu_function_wall_s'],'independent_heldout_gate':summary['independent_heldout_gate'],'calibration_increment_ns':summary['calibration_increment_ns']})
