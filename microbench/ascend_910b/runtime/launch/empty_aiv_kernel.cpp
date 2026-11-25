// Empty AIV kernel for measuring the CANN device-stream dispatch boundary.
#include "kernel_operator.h"

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

extern "C" __global__ __aicore__ void empty_aiv() {}
