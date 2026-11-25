"""DeepSeek-V4 C128 physical stage launches from the aggregate launch contract."""

from dataclasses import replace


def dsv4_c128_stage_launches(launch):
    if launch.algorithm != "vllm_dsv4_c128_two_stage_aggregate_projection":
        raise ValueError("C128 stage launches require the aggregate source launch")
    parameters = launch.parameters
    store_grid = parameters["dsv4_finish_store_grid"]
    store_block = parameters["dsv4_finish_store_block"]
    if not store_grid.is_integer() or not store_block.is_integer():
        raise ValueError("C128 aggregate store launch geometry must be integral")
    return (
        ("compress", replace(launch,
            algorithm="vllm_dsv4_c128_compress_block8_cutedsl", parameters={})),
        ("store", replace(launch,
            algorithm="vllm_dsv4_c128_norm_rope_store_cutedsl",
            grid=(int(store_grid), 1, 1), block=(int(store_block), 1, 1), parameters={})),
    )
