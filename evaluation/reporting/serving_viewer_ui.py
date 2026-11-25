"""Shared English presentation for the standalone serving viewers.

Input values, identifiers, raw evidence and embedded data remain unchanged.
"""
import json
import hashlib
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def report_run_context(payload, coverage):
    """Keep cost provenance optional and unknown costs distinct from zero."""
    execution = payload["main_chain_status"] if "main_chain_status" in payload else None
    if execution is not None:
        execution = {key: execution[key] for key in ("status", "chain", "execution_artifact_sha256")
            if key in execution}
    recording = coverage["resource_recording_s"] if "resource_recording_s" in coverage else None
    source = payload["visualization_run_costs"] if "visualization_run_costs" in payload else None
    costs = ({"html_generation_s": source["html_generation_s"] if "html_generation_s" in source else None,
        "scope": source["scope"] if "scope" in source else None} if source is not None else None)
    if source is not None:
        for key in ("total_wall_through_result_write_s", "simulation_wall_s", "recording_projection_s",
                "finalization_wall_s", "result_json_projection_write_s",
                "evaluation_before_report_s", "report_generation_s",
                "cost_metadata_attachment_s", "regeneration_wall_s"):
            if key in source:
                costs[key] = source[key]
    values = [recording] + ([value for key, value in costs.items() if key != "scope"] if costs is not None else [])
    if any(value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or value < 0) for value in values):
        raise ValueError("recorded report costs must be finite nonnegative seconds")
    result = {"execution": execution, "resource_recording_s": recording, "report_costs": costs}
    if "performance_timing_status" in payload:
        from evaluation.serving.simulation.execution_artifact import _require_performance_timing_status
        result["performance_timing_status"] = _require_performance_timing_status(payload["performance_timing_status"])
    return result


def _configuration_metadata(value):
    private = {"prompt", "prompt_text", "request_text", "prompt_token_ids",
        "output_token_ids", "embedding_token_ids", "token_ids", "access_token", "api_key",
        "prompt_texts", "prompts", "request_texts", "input_text", "input_texts",
        "sglang_request_token_transcripts", "request_token_transcripts", "mtp_draft_token_ids_by_step",
        "prompt_token_ids_by_request", "output_token_ids_by_request", "token_ids_by_request"}
    if isinstance(value, Enum):
        return _configuration_metadata(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration metadata keys must be strings")
        return {key: _configuration_metadata(item) for key, item in value.items()
            if key not in private and not key.endswith("_token_ids") and not key.endswith("_token_transcripts")}
    if isinstance(value, (list, tuple)):
        return [_configuration_metadata(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("experiment configuration metadata must be JSON-compatible")


def report_experiment_configuration(payload):
    """Project recorded configuration metadata; never export request contents."""
    if "visualization_experiment_configuration" in payload:
        result = _configuration_metadata(payload["visualization_experiment_configuration"])
        _validate_configuration_files(result["files"])
        return result
    groups = {
        "model": ("model_config_provenance", "model_compute_dtype"),
        "deployment": ("deployment", "deployment_plan", "hardware_ids", "routing_policy",
            "execution_mode", "serving_timing_mode", "kv_handoff_mode", "prediction_method", "launch_policy"),
        "scheduler": ("scheduler_runtime",),
        "inputs": ("request_source", "arrival_process", "workload_hash", "workload_assumption_binding"),
        "capacity": ("kv_capacity", "kv_lifecycle_replay"),
        "speculation": ("mtp_scenario", "mtp_pd_handoff_contract"),
        "provenance": ("execution_provenance",),
    }
    result = {group: {key: payload[key] for key in keys if key in payload}
        for group, keys in groups.items()}
    result["host_timing"] = ({key: payload["host_timing"][key] for key in (
        "schema_version", "timing_mode", "launch_measurement_boundary",
        "scheduler_measurement_boundary", "profiles", "runtime_prepare_profiles")
        if key in payload["host_timing"]} if "host_timing" in payload else {})
    result["hardware_display"] = (payload["visualization_hardware"]
        if "visualization_hardware" in payload and payload["visualization_hardware"] is not None else {})
    result["bandwidth_peaks"] = payload["visualization_memory_bandwidth_reference"] if "visualization_memory_bandwidth_reference" in payload else None
    files = payload["visualization_configuration_files"] if "visualization_configuration_files" in payload else []
    _validate_configuration_files(files)
    result["files"] = files
    result["unrecorded"] = [name for name in ("model_config_provenance", "deployment_plan",
        "scheduler_runtime", "request_source", "execution_provenance") if name not in payload]
    if "visualization_configuration_snapshot_missing" in payload:
        result["unrecorded"].extend(payload["visualization_configuration_snapshot_missing"])
    return _configuration_metadata(result)


def _validate_configuration_files(files):
    for record in files:
        if record["kind"] not in ("model_config", "hardware_config"):
            raise ValueError("unsupported experiment configuration file kind")
        if hashlib.sha256(record["content"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("configuration snapshot content does not match recorded hash")


_SCRIPT = r'''<script>
(()=>{'use strict';
const embedded=document.documentElement.dataset.embeddedDetail==='true';
window.renderExperimentConfiguration=function(config){if(embedded)return;let panel=document.querySelector('#experiment-configuration');if(panel)return;panel=document.createElement('details');panel.id='experiment-configuration';panel.className='card';panel.style.margin='16px 0';const heading=document.createElement('summary');heading.style.fontSize='17px';heading.style.fontWeight='650';heading.textContent='Experiment configuration';panel.append(heading);const note=document.createElement('p');note.textContent='Recorded configuration and provenance for this run. Requested → effective distinguishes requested settings from settings actually used. Missing fields are listed explicitly; nothing is inferred from filenames.';panel.append(note);
const table=document.createElement('table');table.style.cssText='width:100%;border-collapse:collapse;font-size:13px';function row(name,value){const tr=document.createElement('tr'),th=document.createElement('th'),td=document.createElement('td');th.textContent=name;th.style.cssText='text-align:left;padding:7px 12px;width:36%;border-bottom:1px solid #e2e9f0;color:#526c82';td.style.cssText='padding:7px 12px;border-bottom:1px solid #e2e9f0;overflow-wrap:anywhere';if(value!==undefined&&value!==null)td.setAttribute('data-no-translate','');td.textContent=value===undefined||value===null?'Not recorded':typeof value==='object'?JSON.stringify(value):String(value);tr.dataset.name=name;tr.dataset.search=(name+' '+String(value)).toLowerCase();tr.append(th,td);table.append(tr);}
const model=config.model.model_config_provenance,dep=config.deployment,inputs=config.inputs,provenance=config.provenance.execution_provenance;
row('Model',model?(model.model_id||model.architecture||model.model_type):null);row('Compute precision',config.model.model_compute_dtype||model?.compute_dtype);row('Hardware',Object.keys(config.hardware_display).join(', ')||null);row('Deployment',dep.deployment||dep.deployment_plan?.mode);row('Prediction method',dep.prediction_method);row('Launch policy',dep.launch_policy);row('Serving timing mode',dep.serving_timing_mode||config.host_timing.timing_mode);row('Request count',inputs.request_source?.row_count);row('Input dataset path',inputs.request_source?.input_path);row('Input dataset SHA256',inputs.request_source?.source_sha256);row('Simulation code commit',provenance?.repository?.git_commit);
const runtime=provenance?.runtime_inputs;row('Model execution source',runtime?.model_execution_version);row('CUDA Graph mode',runtime?.graph_execution?.configured_mode);
for(const [pool,v] of Object.entries(config.scheduler.scheduler_runtime||{})){row(pool+' · Scheduler runtime',v.version);row(pool+' · TP / PP',String(v.effective_tensor_parallel_size??'Not recorded')+' / '+String(v.effective_pipeline_parallel_size??'Not recorded'));row(pool+' · Max batch tokens',v.effective_max_num_batched_tokens??v.max_num_batched_tokens);row(pool+' · Max requests',v.max_num_seqs);row(pool+' · Max context length',v.max_model_len);row(pool+' · KV block tokens',v.effective_cache_block_size??v.kv_block_size_tokens);for(const [key,label] of [['prefix_caching','Prefix caching'],['chunked_prefill','Chunked prefill'],['async_scheduling','Async scheduling']])row(pool+' · '+label+' (requested → effective)',String(v['requested_'+key]??'Not recorded')+' → '+String(v['effective_'+key]??'Not recorded'));}
const extra=document.createElement('details'),extraHeading=document.createElement('summary'),extraTable=document.createElement('table');extra.className='config-extra';extraHeading.textContent='All parameters and source paths';extraTable.style.cssText=table.style.cssText;for(const tr of [...table.rows]){const name=tr.dataset.name;if(!['Model','Compute precision','Hardware','Deployment','Prediction method'].includes(name)&&!['Max batch tokens','Max requests','Prefix caching'].some(key=>name.includes(key)))extraTable.append(tr);}extra.append(extraHeading,extraTable);const search=document.createElement('input');search.type='search';search.className='config-search';search.setAttribute('aria-label','Search configuration settings');search.placeholder='Filter settings: TP, cache, SHA256…';let priorOpen=false,lastQuery='';search.oninput=()=>{const query=search.value.trim().toLowerCase();if(!lastQuery&&query)priorOpen=extra.open;for(const tr of [...table.rows,...extraTable.rows])tr.hidden=!!query&&!tr.dataset.search.includes(query);extra.open=query?true:priorOpen;lastQuery=query;};panel.append(search,table,extra);const missing=document.createElement('p');missing.textContent=config.unrecorded.length?'Configuration fields not recorded: '+config.unrecorded.join(', '):'Recorded run configuration is shown below.';panel.append(missing);
const download=document.createElement('button');download.type='button';download.textContent='Download configuration JSON';download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(config,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='midlayer-experiment-configuration.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};panel.append(download);
const names={model:'Model and precision',deployment:'Deployment and parallelism',scheduler:'Full scheduler configuration',inputs:'Requests and input identity',capacity:'KV capacity and cache',speculation:'Speculative decoding',host_timing:'Host timing model',provenance:'Versions and provenance',hardware_display:'Hardware summary',bandwidth_peaks:'Display bandwidth peaks',files:'Verified configuration file snapshots'};
for(const [key,value] of Object.entries(config)){if(key==='unrecorded')continue;const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=names[key]||key;pre.textContent=JSON.stringify(value,null,2);pre.style.cssText='white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;font:12px/1.6 ui-monospace,monospace;background:#f4f7fa;padding:12px;border-radius:5px';details.append(summary,pre);panel.append(details);}
const diagnostics=document.querySelector('#bottleneck-diagnostics'),anchor=document.querySelector('.page > .card');if(diagnostics)diagnostics.append(panel);else if(anchor?.querySelector('#metrics'))anchor.append(panel);else if(anchor)anchor.after(panel);else document.body.append(panel);};

})();
</script>'''


def with_serving_viewer_ui(html):
    """Embed shared configuration controls before viewer scripts execute."""
    return html.replace('<meta charset="utf-8">', '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + _SCRIPT, 1)


def attach_report_costs(report, costs):
    """Attach producer-measured costs, keeping replay inputs self-contained.

    The producer must measure this metadata write separately in its final receipt;
    the embedded total is explicitly the earlier result-write boundary.
    """
    import time
    started = time.perf_counter()
    normalized = report_run_context({"visualization_run_costs": costs}, {})["report_costs"]
    path, data_path = Path(report["path"]), Path(report["data_path"])
    payload = json.loads(data_path.read_text())
    payload["visualization_run_costs"] = normalized
    data_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    encoded = json.dumps(normalized, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    with path.open("a") as stream:
        stream.write('<script type="application/json" id="recorded-report-costs">' + encoded + '</script>')
    report["html_bytes"], report["data_bytes"] = path.stat().st_size, data_path.stat().st_size
    return time.perf_counter() - started


def configuration_file_snapshots(model_path, hardware_paths):
    """Snapshot readable config sources before execution, retaining missingness."""
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "config.json"
    inputs = [("model_config", model_path)] + [("hardware_config", Path(path)) for path in hardware_paths]
    records, missing = [], []
    for kind, path in inputs:
        if not path.is_file():
            missing.append(kind + " snapshot: " + str(path))
            continue
        content = path.read_text()
        records.append({"kind": kind, "path": str(path.resolve()), "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest()})
    return records, missing
