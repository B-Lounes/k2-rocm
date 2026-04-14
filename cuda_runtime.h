#pragma once

#ifdef K2_WITH_ROCM
#include <hip/hip_runtime.h>
#else
#include_next <cuda_runtime.h>
#endif
