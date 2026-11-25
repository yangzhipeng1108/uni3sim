// Empty 1 AIC : 0 AIV task for the CANN MIX_AIC task-boundary mechanism.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);

extern "C" __global__ __aicore__ void empty_mix_aic() {}
