"""Independent FlashMLA caller output views, not unknown callee internals."""


def attach_attention_output_regions(raw, cases, operators):
    by_phase = {case["phase"]: case for case in cases}
    local_heads, dim = raw["num_attention_heads"] // 4, raw["head_dim"]
    padded = 64 if local_heads <= 64 else 128
    for record in operators:
        if record["variant"] not in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
            continue
        case = by_phase[record["node_id"].split(":", 1)[0]]
        requests, modes = case["requests"], case["attention_phases"]
        rows = sum(request["query"] for request in requests)
        decode_rows = sum(request["query"] for request, mode in zip(requests, modes, strict=True) if mode == "decode")
        decode = record["variant"] == "dsv4_sparse_decode"
        count = sum(request["query"] for request, mode in zip(requests, modes, strict=True)
                    if mode == ("decode" if decode else "prefill"))
        offset = 0 if decode else decode_rows * padded * dim
        query = record["attributes"]["source_query_input"]["producer"]
        consumers = [other for other in operators if other["variant"] == "dsv4_inv_rope_fp8_quant"
                     and record["node_id"] in other["required_ancestors"]]
        if len(consumers) != 1 or consumers[0]["rank"] != record["rank"]:
            raise ValueError("independent attention output requires its inverse-RoPE consumer")
        consumer, = consumers
        output = {"output_role": "attention_output_region", "dtype": "bf16", "base": "attention.o_padded",
            "base_shape": [rows, padded, dim], "shape": [count, padded, dim],
            "strides_elements": [padded * dim, dim, 1], "offset_elements": offset}
        port = {"producer": record["node_id"], **output, "shape": [count, local_heads, dim]}
        if (record["attributes"]["source_tensor_outputs"] != [output]
                or port not in consumer["attributes"]["source_tensor_inputs"]):
            raise ValueError("independent attention output/input view accounts disagree")
        flow = {"program": "vllm_0_30_flashmla_supplied_output_regions",
            "allocation_scope": {"query_producer": query, "rank": record["rank"]},
            "allocation": {"base": "attention.o_padded", "dtype": "bf16", "shape": [rows, padded, dim],
                "empty_allocations_per_forward": 1, "initialization_write_bytes": 0, "aliases_padded_q": False},
            "supplied_output": output, "phase_slice_allocates": False,
            "callee_out_shape": [count, 1, padded, dim] if decode else [count, padded, dim],
            "callee_out_strides_elements": [padded * dim, padded * dim, dim, 1] if decode else [padded * dim, dim, 1],
            "callee_out_unsqueeze_allocates": False,
            "prefill_subchunk_out": None if decode else "output[query_start:query_end] views the same phase slice",
            "callee_return_replaces_supplied_output": False, "caller_output_copy_bytes": 0,
            "consumer": consumer["node_id"], "consumer_live_head_view": port,
            "callee_internal_temporaries_and_write_coverage_proved": False}
        group = [other for other in operators if other["variant"] in ("dsv4_sparse_prefill", "dsv4_sparse_decode")
                 and other["attributes"]["source_query_input"]["producer"] == query]
        prefill = [other["node_id"] for other in group if other["variant"] == "dsv4_sparse_prefill"]
        record["attributes"]["source_attention_output_region"] = flow
        record["attention_output_region_account"] = {"declaration": flow,
            "query_input_view": record["attributes"]["source_query_input"],
            "output_tensor_ports": [output], "consumer_input_ports": [port],
            "group_output_tensor_ports": {other["node_id"]: other["attributes"]["source_tensor_outputs"] for other in group},
            "requested_output_shape_dtype_compatible": True,
            "lowered_consumer_reads_after_output": True,
            "prefill_before_decode": True,
            "prefill_predecessors": prefill if decode else []}
