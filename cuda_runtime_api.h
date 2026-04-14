#pragma once

#ifdef K2_WITH_ROCM
#include <hip/hip_runtime_api.h>
#else
#include_next <cuda_runtime_api.h>
#endif
