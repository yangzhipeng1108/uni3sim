#!/usr/bin/env bash
# Compile one AscendC probe at one NREP with the flags CMake uses for AIC.
# usage: build.sh <NREP> <source.cpp> <output.o>
set -euo pipefail
A=/usr/local/Ascend/cann-8.5.1/aarch64-linux
CCEC=$A/ccec_compiler/bin/ccec
INC="-I$A/asc/impl/adv_api -I$A/asc/impl/basic_api -I$A/asc/impl/utils \
-I$A/asc/include -I$A/asc/include/adv_api -I$A/asc/include/basic_api \
-I$A/asc/include/aicpu_api -I$A/asc/include/utils \
-I$A/tikcpp/tikcfw -I$A/tikcpp/tikcfw/interface -I$A/tikcpp/tikcfw/impl"
FLAGS="-O3 -DNDEBUG --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
--cce-auto-sync -mllvm -cce-aicore-stack-size=0x8000 \
-mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-addr-transform \
-mllvm -cce-aicore-dcci-insert-for-scalar=false -std=c++17 --cce-aicore-lang \
-DTILING_KEY_VAR=0 -include $A/../include/ascendc/asc_devkit_version.h"
$CCEC -c $FLAGS $INC -DNREP="$1" -o "$3" "$2"
