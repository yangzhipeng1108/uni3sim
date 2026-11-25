"""vLLM0.30 prefill-context parallel rank topology (MLA only at source)."""

from dataclasses import replace


def apply(plan, config):
    if config.context_parallel_size == 1:
        return plan
    if config.pipeline_parallel_size != 1:
        raise ValueError("vLLM0.30 MLA PCP runtime requires PP=1")
    return replace(plan, enabled_strategies=plan.enabled_strategies + ("prefill_context_parallel",),
        communication_rules=plan.communication_rules + ("context_kv_all_gather", "context_hidden_restore"))
