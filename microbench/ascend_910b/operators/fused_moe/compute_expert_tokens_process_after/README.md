# MoeComputeExpertTokens key-1003 ProcessAfter probe

This standalone CANN 8.5.1 probe isolates the source-defined key-1003 `ProcessAfter` controller: a workspace `DataCopyPad`, int32-to-float `Cast`, `WholeReduceMax`, float-to-int32 `Cast`, and output `DataCopyPad`. It excludes the production search body, both `SyncAll` calls, task launch, and all production/evaluation duration. The device timer records the longest physical-AIV controller interval after an explicit final MTE3 completion edge.

Configure and build inside the agreed 910B container with `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1 cmake -S . -B build -DSOC_VERSION=Ascend910B2`, then `cmake --build build -j 8`. Run with `ASCEND_RT_VISIBLE_DEVICES=<physical-device> ASCEND_DEVICE_ID=0 ./build/compute_expert_tokens_reduce --experts E --used-before U --samples N`.
