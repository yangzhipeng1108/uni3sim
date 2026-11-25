# Short GroupedMatmul B-ingress source-ring probe

This independent Ascend 910B mechanism probe isolates the dominant B MTE2 ingress of CANN 8.5.1 BF16 GroupedMatmul's short `K=512`, `M=64`, `N=256` program. It replays a supplied non-negative group-row census through the source AIC ring and each populated group's diagonal `M/N` assignment. Each assigned output tile issues the two source-defined `K=256` B1 fills through the real `CopyND2NZOnTheFly`-equivalent sixteen-C0-copy pattern, B1 ping-pong, MTE2 completion and cache-bypass hint. It measures the maximum 24-AIC ingress completion.

It intentionally contains no routing kernel, activation, A ingress, MTE1, Cube, FixPipe, output, or complete FusedMoE call. The B backing allocation packs only populated groups because the source hint disables L2 caching; physical-address sparsity is not a tested variable. The result can establish how the source-selected active-group/AIC topology changes B-ingress concurrency, but is not a production duration or a complete Matmul service.

Build in the agreed 910B container with `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2`, then `cmake --build build -j 8`. The runner accepts a JSON row census and requires three fresh processes on an idle device.
