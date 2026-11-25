#!/usr/bin/env python3
"""Generate a logical startup weight inventory for colocated instances."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import DeploymentConfig, WorkerPoolConfig
from src.serving.kv_cache.model_residency import deployment_weight_placements
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import load_model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--v4-weight-storage-profile", choices=("sm90_deepgemm_marlin",))
    parser.add_argument("--instance", action="append", required=True,
                        help="instance name and ordered physical ranks, e.g. main:2,7")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, help="optional per-shard TSV table")
    args = parser.parse_args()
    if args.table is not None and args.table.resolve() == args.output.resolve():
        parser.error("--table and --output must be different files")
    model = load_model_config(args.model)
    parallel = ParallelConfig(tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
                              enable_expert_parallel=args.ep)
    pools = []
    for instance in args.instance:
        name, separator, ranks = instance.partition(":")
        if not separator or not name or not ranks:
            parser.error("--instance must be NAME:RANK[,RANK...]")
        try:
            physical_ranks = tuple(int(rank) for rank in ranks.split(","))
        except ValueError:
            parser.error("--instance ranks must be integers")
        pools.append(WorkerPoolConfig(name, "colocated", args.hardware, parallel, physical_ranks))
    deployment = build_colocated_plan(DeploymentConfig("colocated", tuple(pools)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    rows = deployment_weight_placements(model, deployment, v4_weight_storage_profile=args.v4_weight_storage_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        columns = ("instance", "physical_rank", "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                   "expert_parallel_rank", "hardware_id", "parameter", "logical_shape", "local_shape",
                   "dtype", "storage_kind", "metadata_for", "placement_status", "shard_axis", "shard_interval", "shard_intervals", "represented_weight_shape", "represented_weight_shard_intervals", "storage_bytes", "object", "object_view", "alias_of")
        with args.table.open("w", newline="") as table:
            writer = csv.writer(table, delimiter="\t")
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[column] if column in row else None for column in columns])
    print("instance\tlogical rank\tphysical rank\thardware\tobjects\tstorage bytes")
    for pool in deployment.worker_pools.values():
        for logical_rank in range(pool.parallel_plan.rank_groups.world_size):
            local = [row for row in rows if row["pool"] == pool.name and row["rank"] == logical_rank]
            print(f"{pool.name}\t{logical_rank}\t{pool.physical_rank_for_logical_rank(logical_rank)}\t"
                  f"{pool.hardware_id}\t{len({row['object'] for row in local})}\t{sum(row['storage_bytes'] for row in local)}")
    print(f"inventory: {args.output}")


if __name__ == "__main__":
    main()
