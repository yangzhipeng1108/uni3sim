"""Opt-in bootstrap for Midlayer vLLM Scheduler instrumentation."""

from dataset_pipeline.serving_dataset.pd_mooncake_adapter import (
    install_from_environment as install_pd_mooncake_adapter,
)
from dataset_pipeline.serving_dataset.vllm_scheduler_trace import (
    install_from_environment,
)
from dataset_pipeline.serving_dataset.vllm_capture_plugin import (
    install_from_environment as install_vllm_capture_plugin,
)
from dataset_pipeline.serving_dataset.vllm_operator_trace import (
    install_from_environment as install_vllm_operator_trace,
)


install_pd_mooncake_adapter()
install_from_environment()
install_vllm_capture_plugin()
install_vllm_operator_trace()
