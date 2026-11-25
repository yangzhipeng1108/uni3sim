"""Mechanism report rows; extracted without changing the established contracts.

This module is the compatibility re-export layer for the historical
import surface; the implementation lives in the sibling modules of
this package.
"""

from evaluation.kernel.mechanism_report_primitives import (
    metric_summary,
    format_percentage,
    result_scope_title,
    classify_comparison_status,
    observed_duration_ns,
    kernel_ordinal_for_scope,
    kernel_ordinal_sort_value,
    kernel_ordinal_suffix,
    plot_case_id,
    support_level,
    kernel_launch_class,
    task_execution_signature,
    physical_bucket,
    task_axis_summary,
    task_rows,
    error_direction,
    memory_path_bytes,
    write_csv,
    read_result_csv,
    verify_manifest_artifact,
    result_join_key,
    index_result_rows,
    require_same_result_identity,
    comparison_model_identity,
    positive_csv_float,
    physical_bucket_definitions,
    component_ratio_summary,
)

from evaluation.kernel.mechanism_traffic_rows import (
    directional_kernel_traffic_fields,
    directional_traffic_row,
    nvidia_directional_traffic_rows,
    ascend_directional_traffic_rows,
    metax_directional_traffic_rows,
    MechanismHardwareContract,
    MECHANISM_HARDWARE,
    comparison_rows,
    _MechanismHardwareContract,
)

from evaluation.kernel.mechanism_comparison_rows import (
    counterfactual_rows,
    counterfactual_summary_rows,
    kernel_rows,
    baseline_rows,
    before_after_case_rows,
    before_after_bucket_rows,
    before_after_summary_rows,
    bucket_rows,
    severe_duration_rows,
    case_impact_rows,
    accuracy_gate,
    comparison_summary,
    traffic_outlier_rows,
    coverage_rows,
)

from evaluation.kernel.mechanism_case_study_rows import (
    ascend_moe_tile_replay_rows,
    c500_silu_mechanism_rows,
    c500_silu_mechanism_summary,
)
