// Empty 1 AIC : 2 AIV kernel for the CANN mixed-task scheduling boundary.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

extern "C" __global__ __aicore__ void empty_mixed() {}
