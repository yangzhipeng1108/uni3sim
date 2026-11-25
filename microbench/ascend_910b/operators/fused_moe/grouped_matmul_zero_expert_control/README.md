# GroupedMatmul zero-expert controller probe

This standalone CANN 8.5.1 probe isolates the zero-row branch executed by every AIC in `GMMProcess::Process`: `UpdateMnConfig`, one cumulative `GlobalTensor<int64_t>::GetValue`, `SetMNConfig`, and the early `continue`. It executes no matrix, vector, routing, activation, reduction, or complete production operator work and consumes no production/evaluation duration.

Configure and build inside the agreed 910B container with `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2`, then `cmake --build build -j 8`. The runner requires physical device 6 or 7, remaps it to logical device zero, and executes the frozen group-count sweep `{8,32,64,128,256,512}` in alternating order.
