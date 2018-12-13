#pragma once

// CUDA headers.
#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cutil_math.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#include <cstdint>

namespace ray_tracing_cuda
{
__device__ float3 BallRand(curandState * randState, float radius)
{
  float sqRad = radius * radius;
  for (int i = 0; i < 1000; i++)
  {
    float x = radius * (2.0f * curand_uniform(randState) - 1.0f);
    float y = radius * (2.0f * curand_uniform(randState) - 1.0f);
    float z = radius * (2.0f * curand_uniform(randState) - 1.0f);
    if (x * x + y * y + z * z < sqRad)
      return make_float3(x, y, z);
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float LinearRand(curandState * randState, float minVal, float maxVal)
{
  return lerp(minVal, maxVal, 1.0f - curand_uniform(randState));
}
}  // namespace ray_tracing_cuda
