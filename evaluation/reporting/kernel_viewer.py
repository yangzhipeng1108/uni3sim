"""Offline UI over saved Kernel comparisons; no prediction or data collection."""
import html
import json
import math
from collections import Counter
from pathlib import Path

from evaluation.reporting.serving_viewer_ui import _configuration_metadata


METHODS = ("forward", "oracle", "roofline", "pipeweave")


def kernel_display_projection(artifact, metrics):
    """Keep all cases and recorded diagnostics, with case-level subset statistics."""
    from evaluation.kernel.evaluation_runner import compute_metrics

    if artifact["case_count"] != len(artifact["cases"]):
        raise ValueError("Kernel case count does not match saved cases")
    identifiers = [case["case_id"] for case in artifact["cases"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Kernel case IDs must be unique")
    cases = _configuration_metadata(artifact["cases"])
    operators = sorted({case["report_meta"]["op_type"] for case in cases})
    coverage, by_operator = {}, {}
    for method in METHODS:
        pairs = []
        reasons = Counter()
        for case in cases:
            pair = case[method]
            if isinstance(pair, list):
                if len(pair) != 2 or any(type(value) not in (int, float)
                        or not math.isfinite(value) or value <= 0 for value in pair):
                    raise ValueError("Kernel comparison must contain positive finite nanoseconds")
                pairs.append(tuple(pair))
            elif pair in ("ERR", "OUT_OF_DOMAIN"):
                pairs.append(pair)
                recorded = {row[method + "_unavailable_reason"] for row in case["kernel_results"]
                    if method + "_unavailable_reason" in row}
                if not recorded:
                    recorded = {"Original result did not record a specific reason · " + pair}
                reasons.update(recorded)
            elif pair is not None:
                raise ValueError("Unknown Kernel comparison state")
        selected = metrics[method] is not None
        if not selected and pairs:
            raise ValueError("Unselected method has saved comparisons")
        coverage[method] = {
            "selected": selected, "paired": sum(isinstance(pair, tuple) for pair in pairs),
            "unavailable": sum(isinstance(pair, str) for pair in pairs),
            "unrecorded": len(cases) - len(pairs) if selected else 0,
            "subset_metrics": compute_metrics(pairs, exclude_unsupported=True),
            "unavailable_reasons": dict(reasons),
        }
        by_operator[method] = {}
        for operator in operators:
            subset = [tuple(case[method]) if isinstance(case[method], list) else case[method]
                for case in cases if case["report_meta"]["op_type"] == operator
                and case[method] is not None]
            by_operator[method][operator] = compute_metrics(subset, exclude_unsupported=True)
    support = Counter(row["forward_support_level"] if row["forward_support_level"] is not None
        else "unrecorded" for case in cases
        for row in case["kernel_results"] if "forward_support_level" in row)
    limitations = Counter()
    for case in cases:
        limitations.update({item for row in case["kernel_results"]
            if "forward_limitations" in row and row["forward_limitations"] is not None
            for item in row["forward_limitations"]})
    result = {"cases": cases, "case_count": len(cases), "metrics": metrics,
        "coverage": coverage, "by_operator": by_operator,
        "forward_kernel_support": dict(support), "limitation_case_counts": dict(limitations)}
    # The metric owner returns NumPy scalar values; freeze its JSON result
    # before passing through the shared metadata/privacy boundary.
    return json.loads(json.dumps(result, allow_nan=False))


def kernel_page(payload):
    """Render a single file, including searchable configuration and SVG plots."""
    encoded = json.dumps(_configuration_metadata(payload), ensure_ascii=True,
        allow_nan=False).replace("<", "\\u003c")
    links = "".join('<a href="' + html.escape(link["path"] if
        link["path"].startswith("file:") else Path(link["path"]).resolve().as_uri(),
        quote=True) + '">' + html.escape(link["label"]) + '</a>'
        for link in payload["report_links"])
    return _PAGE.replace("__DATA__", encoded).replace("__LINKS__", links)


_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Kernel evaluation report</title>
<style>
:root{color-scheme:light;--line:#dce4ed;--muted:#52667d}*{box-sizing:border-box}
body{margin:0;background:#f3f6fa;color:#23364b;font:14px/1.65 system-ui,"PingFang SC","Heiti SC","Microsoft YaHei",sans-serif}
main{max-width:1240px;padding:24px;margin:auto}h1{font-size:26px;margin:0}h2{font-size:19px;margin:0 0 10px}h3{font-size:16px}
a{color:#185ca2}nav,.controls,.pager{display:flex;flex-wrap:wrap;align-items:center;gap:12px}nav{margin:12px 0}
.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0;min-width:0;overflow-wrap:anywhere}
.muted{color:var(--muted)}.facts{display:flex;flex-wrap:wrap;gap:10px}.badge{background:#eaf1f8;border-radius:6px;padding:5px 10px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 10px;vertical-align:top}
th{color:var(--muted)}.table-wrap{overflow-x:auto}#case-table{table-layout:fixed}#case-table th:first-child{width:38%}
td,pre,.badge{overflow-wrap:anywhere}input,select,button{font:inherit;padding:7px 10px;border:1px solid #bdcbd9;border-radius:6px;background:white;color:inherit}
input{width:min(440px,100%)}label{display:flex;align-items:center;gap:8px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}
summary{cursor:pointer;font-weight:600}details>summary{padding:4px 0}pre{white-space:pre-wrap;background:#f4f7fa;padding:12px;border-radius:6px;font:12px/1.6 ui-monospace,monospace;max-height:420px;overflow:auto}
.columns{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:20px}
svg{display:block;width:100%;height:auto}svg .point{cursor:pointer}svg .point:hover,svg .point:focus{stroke:#101d30;stroke-width:3}
#scatter-info{min-height:48px}.metric{font-variant-numeric:tabular-nums}.diagnostic{background:#f7faff;border:1px solid var(--line);padding:14px;border-radius:8px;margin-top:10px}
#case-count{margin:12px 0}.pager{margin-top:14px}#case-summary{font-size:19px}#cost-table th{width:32%}#config-table{table-layout:fixed}#config-table th{width:48%}
@media(max-width:850px){main{padding:12px}.card{padding:14px}.columns{grid-template-columns:1fr}#case-table th:first-child{width:32%}th,td{padding:7px 5px}label{flex-wrap:wrap}.controls{gap:8px}h1{font-size:23px}}
@media(max-width:560px){#case-table,#case-table tbody,#case-table tr,#case-table td{display:block}#case-table thead{display:none}#case-table td:before{content:attr(data-label);display:block;color:var(--muted);font-size:12px}#case-table tr{border-bottom:2px solid var(--line);padding:6px 0}#case-table td{border:0}}
</style></head><body><main>
<h1>Kernel evaluation report</h1><nav aria-label="Report navigation"><a href="#overview">Evaluation overview</a><a href="#coverage">Coverage and support</a><a href="#operators">Operator statistics</a><a href="#cases">All cases</a><a href="#configuration">Experiment configuration</a><a href="#costs">Run costs</a></nav>
<section class="card" id="overview"><h2 id="status"></h2><div class="facts" id="run-facts"></div><p id="reason" class="muted"></p>
<div class="table-wrap"><table id="method-metrics"><thead><tr><th>Method</th><th>Original summary status</th><th>Paired / total</th><th>MAPE</th><th>Median APE</th><th>P90 APE</th><th>Max APE</th></tr></thead><tbody></tbody></table></div>
<p class="muted">APE = abs(predicted - measured) / measured × 100%; MAPE is the arithmetic mean of case APEs. Latencies are displayed in µs (stored in ns). Methods may use different paired subsets; assess accuracy together with coverage.</p>
<div class="columns"><div><h3>Predicted versus measured latency</h3><label>Scatter method <select id="plot-method" aria-label="Scatter method"></select></label><svg id="scatter" viewBox="0 0 560 470" role="img" aria-label="Kernel predicted versus measured latency scatter plot"></svg><div id="scatter-info" class="muted" aria-live="polite">Hover or focus to inspect a case; select a point to open its details.</div></div><div><h3>Cases with larger errors</h3><p class="muted">Cases ordered by descending APE for the selected method; select one to inspect recorded diagnostics.</p><div id="outliers"></div><p>Diagnostics show bottlenecks and limitations recorded in the original results; they do not establish the cause of prediction errors.</p></div></div></section>
<section class="card" id="coverage"><h2>Coverage and support</h2><div id="coverage-content"></div></section>
<section class="card" id="operators"><h2>Per-operator error statistics</h2><label>Statistics method <select id="operator-method" aria-label="Statistics method"></select></label><p class="muted">Statistics cover all cases independently of the detail filters and include only paired cases for the selected method.</p><div class="table-wrap"><table id="operator-table"><thead><tr><th>Operator</th><th>Paired / total</th><th>MAPE</th><th>Median APE</th><th>P90 APE</th><th>Max APE</th></tr></thead><tbody></tbody></table></div></section>
<details class="card" id="cases"><summary id="case-summary">All case details (expand)</summary><p class="muted">Expand to search all records. Filters combine; page size limits displayed rows while search covers every case.</p>
<div class="controls"><input id="case-search" type="search" placeholder="Search case ID, operator, shape or limitations" aria-label="Search all cases">
<label>Method <select id="case-method" aria-label="Detail method"></select></label><label>Operator <select id="case-operator" aria-label="Operator filter"><option value="">All operators</option></select></label>
<label>Support <select id="case-support" aria-label="Support filter"><option value="">All statuses</option></select></label><label>Sort <select id="case-sort" aria-label="Detail sort order"><option value="error-desc">APE descending</option><option value="error-asc">APE ascending</option><option value="id">Case ID</option></select></label>
<label>Rows per page <select id="page-size" aria-label="Rows per page"><option>25</option><option>50</option><option>100</option></select></label><button id="reset" type="button">Reset filters</button></div>
<p id="case-count" aria-live="polite"></p><table id="case-table"><thead><tr><th>Case / operator / diagnostics</th><th>Predicted µs</th><th>Measured µs</th><th>APE</th><th>Forward support</th></tr></thead><tbody></tbody></table>
<div class="pager"><button id="prev" type="button">Previous</button><span id="page-label"></span><button id="next" type="button">Next</button></div></details>
<details class="card" id="configuration"><summary>Experiment configuration (search all saved parameters)</summary><p class="muted">Parameters and configuration snapshots saved with this run. Missing values are not inferred from filenames. Kernel evaluation has no Serving model, scheduler or request-stream settings.</p>
<input id="config-search" type="search" placeholder="Search hardware, mode, jobs, path, SHA256…" aria-label="Search experiment configuration"><p id="config-count"></p>
<table id="config-table"><thead><tr><th>Parameter / source path</th><th>Saved value</th></tr></thead><tbody></tbody></table><button id="download-config" type="button">Download configuration JSON</button>
<details><summary>Validated hardware configuration snapshots</summary><div id="config-files"></div></details></details>
<section class="card" id="costs"><h2>Run status and costs</h2><table id="cost-table"><thead><tr><th>Phase</th><th>Wall time (s)</th><th>Measurement boundary</th></tr></thead><tbody></tbody></table><p class="muted">Some phases contain others; do not sum every row. Original evaluation costs and this page rebuild are listed separately. Missing phases show “Not separately recorded”. Predicted latency and evaluation wall time are distinct metrics.</p><details><summary>Complete cost record JSON</summary><pre id="cost-json"></pre></details></section>
<section class="card"><h2>Saved results</h2><nav>__LINKS__</nav><p class="muted">This HTML embeds all case details and plots for offline use without external scripts, network access or chart libraries. Result links refer to local sidecar files.</p></section>
</main><script type="application/json" id="kernel-data">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.querySelector('#kernel-data').textContent),K=data.kernel_overview,C=K.cases,config=data.visualization_experiment_configuration;
const $=id=>document.getElementById(id),names={forward:'Forward',oracle:'Oracle',roofline:'Roofline',pipeweave:'PipeWeave'},selected=Object.keys(names).filter(m=>K.coverage[m].selected);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=v=>v==null||!Number.isFinite(v)?'Unavailable':v.toFixed(3),percent=v=>v==null||!Number.isFinite(v)?'Unavailable':v.toFixed(2)+'%',paired=p=>Array.isArray(p);
const error=(c,m)=>paired(c[m])?Math.abs(c[m][0]-c[m][1])/c[m][1]*100:null;
function supports(c){const levels=[...new Set(c.kernel_results.filter(r=>'forward_support_level'in r).map(r=>r.forward_support_level===null?'unrecorded':r.forward_support_level))];if(c.forward==='ERR')levels.push('unsupported');return levels.length?levels:['unrecorded'];}
function metricCells(v){return v?.status==='ok'?['MAPE','MedAPE','P90APE','MaxAPE'].map(k=>'<td class="metric">'+percent(v[k])+'</td>').join(''):'<td colspan="4">Unavailable · no paired cases</td>';}
const args=config.provenance.kernel_evaluation.arguments;
$('status').textContent=({complete:'Evaluation complete',partial:'Partially complete',unsupported:'Unsupported',failed:'Evaluation failed'})[data.main_chain_status.status];
$('reason').textContent=data.main_chain_status.reason;
$('run-facts').innerHTML=[['Hardware',args.hardware_id],['Evaluation mode',args.mode],['Parallel jobs',args.effective_jobs??'Not recorded'],['Total cases',K.case_count]].map(([k,v])=>'<span class="badge">'+esc(k)+': '+esc(v)+'</span>').join('');
for(const m of Object.keys(names)){const cov=K.coverage[m],raw=K.metrics[m];$('method-metrics').tBodies[0].insertAdjacentHTML('beforeend','<tr><th>'+names[m]+'</th><td>'+esc(raw?.status??'Not selected')+'</td><td>'+cov.paired+' / '+K.case_count+'</td>'+metricCells(cov.subset_metrics)+'</tr>');}
$('coverage-content').innerHTML=selected.map(m=>{const v=K.coverage[m];return '<h3>'+names[m]+': paired '+v.paired+' / '+K.case_count+', Unavailable '+v.unavailable+', Not recorded '+v.unrecorded+'</h3>'+((m==='oracle'&&K.metrics[m]?.status==='unsupported')?'<p>The original Oracle summary requires complete coverage: any ERR marks it unsupported, even when some cases are paired successfully. Error statistics above use the paired subset while preserving the original summary status.</p>':'')+Object.entries(v.unavailable_reasons).map(([reason,count])=>'<p><b>'+count+' case</b> · '+esc(reason)+'</p>').join('');}).join('')+'<h3>Forward kernel support levels (kernel counts)</h3><div class="facts">'+Object.entries(K.forward_kernel_support).map(([s,n])=>'<span class="badge">'+esc(s)+': '+n+'</span>').join('')+'</div><p>partial marks evidence or modeling gaps in key mechanisms; aggregate_only provides aggregate predictions with incomplete detail. These levels can have predictions and errors while full mechanism acceptance remains open. A case may include several levels.</p><details><summary>Forward limitations and affected case counts</summary>'+Object.entries(K.limitation_case_counts).map(([s,n])=>'<p>'+n+' case · '+esc(s)+'</p>').join('')+'</details>';
for(const id of ['plot-method','operator-method','case-method']){for(const m of selected.length?selected:['forward'])$(id).insertAdjacentHTML('beforeend','<option value="'+m+'">'+names[m]+(selected.length?'':' · Not selected')+'</option>');$(id).disabled=!selected.length;}
for(const op of Object.keys(K.by_operator.forward))$('case-operator').insertAdjacentHTML('beforeend','<option>'+esc(op)+'</option>');
for(const s of [...new Set(C.flatMap(supports))].sort())$('case-support').insertAdjacentHTML('beforeend','<option>'+esc(s)+'</option>');
function renderOperators(){const m=$('operator-method').value;$('operator-table').tBodies[0].innerHTML=Object.entries(K.by_operator[m]).map(([op,v])=>'<tr><th>'+esc(op)+'</th><td>'+(v?.scored??0)+' / '+C.filter(c=>c.report_meta.op_type===op).length+'</td>'+metricCells(v)+'</tr>').join('');}
let page=0;const searchTexts=C.map(c=>JSON.stringify(c).toLowerCase());
function filtered(){const q=$('case-search').value.trim().toLowerCase(),m=$('case-method').value,op=$('case-operator').value,s=$('case-support').value,sort=$('case-sort').value;return C.filter((c,i)=>(!q||searchTexts[i].includes(q))&&(!op||c.report_meta.op_type===op)&&(!s||supports(c).includes(s))).slice().sort((a,b)=>{if(sort==='id')return a.case_id.localeCompare(b.case_id);const x=error(a,m),y=error(b,m);if(x===null)return y===null?a.case_id.localeCompare(b.case_id):1;if(y===null)return -1;return (sort==='error-desc'?y-x:x-y)||a.case_id.localeCompare(b.case_id);});}
function unavailable(c,m){const rows=c.kernel_results.filter(r=>m+'_unavailable_reason'in r);return rows.length?[...new Set(rows.map(r=>r[m+'_unavailable_reason']))].join('; '):c[m]===null?'Method not selected or not recorded':String(c[m]);}
function diagnostics(c){return '<div class="diagnostic"><b>Case metadata</b><pre>'+esc(JSON.stringify(c.report_meta,null,2))+'</pre><b>Methods</b>'+Object.keys(names).map(m=>'<p>'+names[m]+': '+(paired(c[m])?'Predicted '+number(c[m][0]/1000)+' µs / Measured '+number(c[m][1]/1000)+' µs · APE '+percent(error(c,m)):esc(unavailable(c,m)))+'</p>').join('')+'<b>Original kernel diagnostics (support, limitations, bottlenecks)</b><pre>'+esc(JSON.stringify(c.kernel_results,null,2))+'</pre></div>';}
function renderCases(){const list=filtered(),size=Number($('page-size').value),pages=Math.max(1,Math.ceil(list.length/size));page=Math.min(page,pages-1);const m=$('case-method').value;$('case-count').textContent='Matched '+list.length+' / all '+K.case_count+' case · current '+names[m]+' paired '+list.filter(c=>paired(c[m])).length;
$('case-table').tBodies[0].innerHTML=list.slice(page*size,(page+1)*size).map(c=>'<tr data-case-id="'+esc(c.case_id)+'"><td data-label="Case / operator"><b>'+esc(c.case_id)+'</b><div class="muted">'+esc(c.report_meta.op_type)+'</div><details class="case-diagnostic"><summary>View diagnostics</summary></details></td><td data-label="Predicted µs">'+(paired(c[m])?number(c[m][0]/1000):esc(c[m]??'Not selected'))+'</td><td data-label="Measured µs">'+number((paired(c[m])?c[m][1]:c.report_meta.actual_ns)/1000)+'</td><td data-label="APE">'+percent(error(c,m))+'</td><td data-label="Forward support">'+esc(supports(c).join(' / '))+'</td></tr>').join('');
for(const detail of $('case-table').querySelectorAll('.case-diagnostic'))detail.addEventListener('toggle',()=>{if(detail.open&&!detail.querySelector('.diagnostic')){const c=C.find(c=>c.case_id===detail.closest('tr').dataset.caseId);detail.insertAdjacentHTML('beforeend',diagnostics(c));}});
$('page-label').textContent='Page '+(page+1)+' / '+pages+'';$('prev').disabled=page===0;$('next').disabled=page===pages-1;}
function jump(id){$('cases').open=true;$('case-search').value=id;$('case-operator').value='';$('case-support').value='';$('case-method').value=$('plot-method').value;page=0;renderCases();$('cases').scrollIntoView({behavior:'smooth',block:'start'});const d=$('case-table').querySelector('.case-diagnostic');if(d)d.open=true;}
function renderPlot(){const m=$('plot-method').value,points=C.filter(c=>paired(c[m]));if(!points.length){$('scatter').innerHTML='<text x="80" y="180">No paired cases</text>';$('outliers').textContent='Unavailable';return;}
const vals=points.flatMap(c=>c[m].map(v=>v/1000));let lo=Math.floor(Math.log10(Math.min(...vals))),hi=Math.ceil(Math.log10(Math.max(...vals)));if(hi===lo)hi=lo+1;
const x=v=>70+(Math.log10(v)-lo)/(hi-lo)*445,y=v=>405-(Math.log10(v)-lo)/(hi-lo)*365,ops=Object.keys(K.by_operator[m]),colors=['#2763ad','#d05a1b','#379069','#a03d7c','#8272bc','#806848','#247f8c','#bd7c19'];
let svg='<rect x="70" y="40" width="445" height="365" fill="#fcfdff" stroke="#c5d2df"/><line x1="70" y1="405" x2="515" y2="40" stroke="#6c7f90" stroke-dasharray="5 5"/>';
for(let p=lo;p<=hi;p++){const v=10**p;svg+='<line x1="'+x(v)+'" y1="40" x2="'+x(v)+'" y2="405" stroke="#e3eaf1"/><line x1="70" y1="'+y(v)+'" x2="515" y2="'+y(v)+'" stroke="#e3eaf1"/><text x="'+x(v)+'" y="430" text-anchor="middle" font-size="12">10^'+p+'</text><text x="61" y="'+(y(v)+4)+'" text-anchor="end" font-size="12">10^'+p+'</text>';}
svg+='<text x="290" y="458" text-anchor="middle">Measured latency (µs, log scale)</text><text transform="translate(19 230) rotate(-90)" text-anchor="middle">Predicted latency (µs, log scale)</text><text x="75" y="22" font-size="13">Dashed: predicted = measured · '+names[m]+' · '+points.length+' case</text>';
for(const c of points)svg+='<circle class="point" tabindex="0" role="button" aria-label="'+esc(c.case_id)+'" data-case-id="'+esc(c.case_id)+'" cx="'+x(c[m][1]/1000)+'" cy="'+y(c[m][0]/1000)+'" r="3.5" fill="'+colors[ops.indexOf(c.report_meta.op_type)%colors.length]+'" opacity=".7"><title>'+esc(c.case_id)+' · '+percent(error(c,m))+'</title></circle>';
$('scatter').innerHTML=svg;for(const p of $('scatter').querySelectorAll('.point')){const c=C.find(c=>c.case_id===p.dataset.caseId);const show=()=>$('scatter-info').textContent=c.case_id+' · '+c.report_meta.op_type+' · Predicted '+number(c[m][0]/1000)+' / Measured '+number(c[m][1]/1000)+' µs · APE '+percent(error(c,m));p.onmouseenter=show;p.onfocus=show;p.onclick=()=>jump(c.case_id);p.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();jump(c.case_id);}};}
const worst=points.slice().sort((a,b)=>error(b,m)-error(a,m)).slice(0,5);$('outliers').innerHTML=worst.map(c=>'<p><button type="button" data-case-id="'+esc(c.case_id)+'">'+percent(error(c,m))+'</button> · '+esc(c.case_id)+'</p>').join('')+'<div class="facts">'+ops.map((op,i)=>'<span class="badge" style="border-left:4px solid '+colors[i%colors.length]+'">'+esc(op)+'</span>').join('')+'</div>';for(const b of $('outliers').querySelectorAll('button'))b.onclick=()=>jump(b.dataset.caseId);}
function flatten(value,path,out){if(value!==null&&typeof value==='object'&&!Array.isArray(value)){for(const [k,v] of Object.entries(value))flatten(v,path?path+'.'+k:k,out);}else out.push([path,value]);}
const fields=[];flatten(config.provenance.kernel_evaluation,'kernel_evaluation',fields);fields.push(['hardware_config_snapshot_count',config.files.length]);function renderConfig(){const q=$('config-search').value.trim().toLowerCase(),rows=fields.filter(([k,v])=>(k+' '+JSON.stringify(v)).toLowerCase().includes(q));$('config-table').tBodies[0].innerHTML=rows.map(([k,v])=>'<tr><th>'+esc(k)+'</th><td>'+(v===null?'Not recorded / not applicable':esc(typeof v==='object'?JSON.stringify(v):v))+'</td></tr>').join('');$('config-count').textContent='Matched '+rows.length+' / '+fields.length+' saved fields';}
for(const f of config.files)$('config-files').insertAdjacentHTML('beforeend','<p>'+esc(f.path)+'</p><pre>'+esc(f.content)+'</pre>');
$('download-config').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(config,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='kernel-experiment-configuration.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
function renderCosts(){const recorded=$('recorded-report-costs'),cost=recorded?JSON.parse(recorded.textContent):data.visualization_run_costs;const phases=[['evaluation_before_report_s','Original evaluation body','From evaluation timing start to report generation; includes input reading and evaluation, not just prediction'],['report_generation_s','Original report generation','Includes plots, projection, HTML and result writes; HTML is a subphase'],['html_generation_s','Original HTML / projection write','Subphase of report generation; missing historical values remain unknown'],['total_wall_through_result_write_s','Original elapsed time through result write','Evaluation body + report generation; excludes later cost attachment and full CLI startup'],['cost_metadata_attachment_s','Original cost metadata attachment','Measured separately after the elapsed-time boundary above'],['regeneration_wall_s','This page rebuild','Reads saved results and generates the updated page without rerunning simulation']];
$('cost-table').tBodies[0].innerHTML=phases.map(([k,label,boundary])=>'<tr><th>'+label+'</th><td>'+(cost?.[k]==null?'Not separately recorded':number(cost[k]))+'</td><td>'+boundary+'</td></tr>').join('');$('cost-json').textContent=JSON.stringify(cost,null,2);}
for(const id of ['case-search','case-method','case-operator','case-support','case-sort','page-size'])$(id).addEventListener(id==='case-search'?'input':'change',()=>{page=0;renderCases();});
$('prev').onclick=()=>{page--;renderCases();};$('next').onclick=()=>{page++;renderCases();};$('reset').onclick=()=>{$('case-search').value='';$('case-operator').value='';$('case-support').value='';$('case-sort').value='error-desc';page=0;renderCases();};
$('plot-method').onchange=renderPlot;$('operator-method').onchange=renderOperators;$('config-search').oninput=renderConfig;
for(const a of document.querySelectorAll('nav a[href="#cases"]'))a.addEventListener('click',()=>$('cases').open=true);
renderPlot();renderOperators();renderCases();renderConfig();window.addEventListener('DOMContentLoaded',renderCosts);
</script></body></html>'''
