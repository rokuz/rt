#pragma once

#include "rt_hits.h"
#include "rt_random.h"
#include "types_cuda.h"

#include "ray_tracing/types.hpp"

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
uint32_t constexpr kLightSamplesCount = 3;

__device__ float3 TraceDiffuseDirectionalLight(CudaHit * hit, CudaSphere * spheres,
                                               uint32_t spheresCount, CudaMaterial * materials,
                                               CudaLight * lightSource, curandState * randState)
{
  float3 color = make_float3(0.0f, 0.0f, 0.0f);
  CudaHit h;
  CudaRay ray;
  for (uint32_t i = 0; i < kLightSamplesCount; ++i)
  {
    ray.m_origin = hit->m_position;
    ray.m_direction = normalize(-lightSource->m_direction + BallRand(randState, 0.25f));

    if (!HitObjects(&ray, spheres, spheresCount, 0.001f, 1000.0f, &h))
      color += (lightSource->m_color * max(0.0f, dot(ray.m_direction, hit->m_normal)));
  }
  return color / static_cast<float>(kLightSamplesCount);
}

__device__ float3 TraceDiffuseLight(CudaHit * hit, CudaSphere * spheres, uint32_t spheresCount,
                                    CudaMaterial * materials, CudaLight * lightSources,
                                    uint32_t lightSourcesCount, curandState * randState)
{
  float3 lightColor = make_float3(0.0f, 0.0f, 0.0f);
  for (uint32_t i = 0; i < lightSourcesCount; ++i)
  {
    if (lightSources[i].m_lightType == ray_tracing::kLightSourceDirectionalType)
    {
      lightColor += TraceDiffuseDirectionalLight(hit, spheres, spheresCount, materials,
                                                 &lightSources[i], randState);
    }
  }
  return lightColor;
}
}  // namespace ray_tracing_cuda
