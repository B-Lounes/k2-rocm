#ifndef K2_CUDA_COMPAT_HEADER_
#define K2_CUDA_COMPAT_HEADER_

#ifdef K2_WITH_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include_next <cuda.h>
#endif

#endif  // K2_CUDA_COMPAT_HEADER_
