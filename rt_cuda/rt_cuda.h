#pragma once

#include "types_cuda.h"

#include "ray_tracing/types.hpp"

// CUDA headers.
#include "driver_types.h"

#include <cstdint>
#include <functional>

namespace ray_tracing_cuda
{
// CUDA runtime initialization.
extern bool Initialize();

// Ray trace scene. It returns completion event.
extern cudaEvent_t RayTrace(CudaSphere * spheres, uint32_t spheresCount, CudaMaterial * materials,
                            uint32_t materialsCount, CudaLight * lightSources, uint32_t lightSourcesCount,
                            uint32_t samplesInRowCount, float3 backgroundColor, float3 cameraPosition,
                            float3 cameraDirection, float fov, float znear, float zfar, 
                            uint32_t width, uint32_t height, std::function<bool()> && realtimeHandler);

// Check if ray tracing routine is in progress.
extern bool InProgress(cudaEvent_t completion);

// Copy output image to the buffer.
extern void CopyOutputToBuffer(float * buffer);

// Finish current ray tracing process. It also destroys completion event object.
extern void FinishRayTrace(float * output, cudaEvent_t completion);
}  // namespace ray_tracing_cuda
