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
uint32_t constexpr kLightSamplesCount = 5;

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
    {
      color += (lightSource->m_color * max(0.0f, dot(ray.m_direction, hit->m_normal)));
    }
    else
    {
      CudaMaterial * mat = &materials[h.m_materialIndex];
      float3 c = lerp(mat->m_albedo, lightSource->m_color, 0.75f);
      color += (mat->m_refraction * 0.75f * c * max(0.0f, dot(ray.m_direction, hit->m_normal)));
    }
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

__device__ float3 GetSpecularDirectionalLight(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                                              CudaLight * lightSource, curandState * randState)
{
  // Cook-Torrance specular.
  float vdn = clamp(dot(-ray->m_direction, hit->m_normal), 0.0f, 1.0f);
  float ndl = clamp(dot(hit->m_normal, -lightSource->m_direction), 0.0f, 1.0f);
  float3 h = -normalize(lightSource->m_direction + ray->m_direction);
  float ndh = clamp(dot(hit->m_normal, h), ray_tracing::kEps, 1.0f);
  float vdh = clamp(dot(-ray->m_direction, h), 0.0f, 1.0f);

  float G = (fabs(vdh) >= ray_tracing::kEps) ? min(1.0f, 2.0f * ndh * min(vdn, ndl) / vdh) : 1.0f;
  float ndh2 = ndh * ndh;

  CudaMaterial * mat = &materials[hit->m_materialIndex];

  float const roughness = max(mat->m_roughness, 0.03f);
  float const r2 = roughness * roughness;
  float D = exp((ndh2 - 1.0f) / (r2 * ndh2)) / (4.0f * r2 * ndh2 * ndh2);

  float const f0 = (1.0f - mat->m_refraction) / (1.0f + mat->m_refraction);
  float const f2 = f0 * f0;
  float const F = f2 + (1.0f - f2) * pow(1.0f - vdn, 5.0f);

  float spec = clamp(G * D * F, 0.0f, 1.0f);
  return lerp(lightSource->m_color, mat->m_albedo, 0.5f) * spec;
}

__device__ float3 GetSpecularLight(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                                   CudaLight * lightSources, uint32_t lightSourcesCount,
                                   curandState * randState)
{
  float3 lightColor = make_float3(0.0f, 0.0f, 0.0f);
  for (uint32_t i = 0; i < lightSourcesCount; ++i)
  {
    if (lightSources[i].m_lightType == ray_tracing::kLightSourceDirectionalType)
      lightColor += GetSpecularDirectionalLight(ray, hit, materials, &lightSources[i], randState);
  }
  return lightColor;
}
}  // namespace ray_tracing_cuda
