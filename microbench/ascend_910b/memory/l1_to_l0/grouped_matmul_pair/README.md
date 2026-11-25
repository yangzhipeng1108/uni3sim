# GroupedMatmul MTE1 pair probe

This standalone CANN 8.5.1 probe isolates the BF16 `CFG_MDL` `SplitA` then `SplitB` L1-to-L0 transfer pair for one GroupedMatmul base-K tile. It covers every 16-row-aligned M extent from 16 through 128 used by production base and tail tiles, supports 16-aligned base-N extents through 256 and 64-aligned B1 K extents through 256, includes the MTE1-to-M completion dependency, and executes no Cube, output, routing, activation, reduction, or complete production operator work.

Configure and build inside the agreed 910B container with `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2`, then `cmake --build build -j 8`. Run the executable as `grouped_matmul_mte1_pair --batch base_m block_dim iterations [cell ...]`; use `ASCEND_RT_VISIBLE_DEVICES` to bind one idle physical device before launch.

The compatibility `--batch` form fixes base-N and the B1 K extent to 256. Use `--batch-geometry base_m base_n b_step_k block_dim iterations [cell ...]` to validate other source-defined L1 geometries.
