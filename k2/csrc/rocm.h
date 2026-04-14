/**
 * ROCm compatibility header for compiling the CUDA-oriented k2 codebase
 * with HIP. It is force-included for HIP builds from the top-level CMake.
 */

#ifndef K2_CSRC_ROCM_H_
#define K2_CSRC_ROCM_H_

#ifdef K2_WITH_ROCM

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#define __CUDACC_VER_MAJOR__ HIP_VERSION_MAJOR
#define __CUDACC_VER_MINOR__ HIP_VERSION_MINOR
#define __CUDACC_VER_BUILD__ HIP_VERSION_PATCH

#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyKind hipMemcpyKind

#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorInitializationError hipErrorInitializationError
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorAssert hipErrorAssert

#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaEventDisableTiming hipEventDisableTiming

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags \
  hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamPerThread hipStreamPerThread
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes(attr, func) \
  hipFuncGetAttributes((attr), reinterpret_cast<const void *>(func))

enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = hipStreamCaptureModeGlobal,
  cudaStreamCaptureModeThreadLocal = hipStreamCaptureModeThreadLocal,
  cudaStreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed
};

enum cudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = hipStreamCaptureStatusNone,
  cudaStreamCaptureStatusActive = hipStreamCaptureStatusActive,
  cudaStreamCaptureStatusInvalidated = hipStreamCaptureStatusInvalidated
};

inline hipError_t cudaThreadExchangeStreamCaptureMode(
    cudaStreamCaptureMode *mode) {
  auto hip_mode = static_cast<hipStreamCaptureMode>(*mode);
  hipError_t ans = hipThreadExchangeStreamCaptureMode(&hip_mode);
  *mode = static_cast<cudaStreamCaptureMode>(hip_mode);
  return ans;
}

inline hipError_t cudaStreamIsCapturing(cudaStream_t stream,
                                        cudaStreamCaptureStatus *status) {
  hipStreamCaptureStatus hip_status = hipStreamCaptureStatusNone;
  hipError_t ans = hipStreamIsCapturing(stream, &hip_status);
  *status = static_cast<cudaStreamCaptureStatus>(hip_status);
  return ans;
}

namespace cuda {
namespace std {
using ::std::plus;
}  // namespace std
}  // namespace cuda

#endif

#endif  // K2_CSRC_ROCM_H_
