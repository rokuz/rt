#pragma once

// CUDA headers.
#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#include <cstdint>

namespace ray_tracing_cuda
{
__device__ bool HitSphere(CudaSphere * sphere, CudaRay * ray, float tmin, float tmax, CudaHit * hit)
{
  float3 const d = ray->m_origin - sphere->m_center;
  float const a = dot(ray->m_direction, ray->m_direction);
  float const b = 2.0f * dot(d, ray->m_direction);
  float const c = dot(d, d) - sphere->m_radius * sphere->m_radius;
  float const discriminant = b * b - 4 * a * c;
  if (discriminant < 0.0f)
    return false;

  auto const sqrtD = sqrt(discriminant);
  auto const t = min((-b - sqrtD) / (2.0f * a), (-b + sqrtD) / (2.0f * a));
  if (t < tmin || t > tmax)
    return false;

  hit->m_parameterT = t;
  hit->m_position = ray->m_origin + ray->m_direction * t;
  hit->m_normal = normalize(hit->m_position - sphere->m_center);
  hit->m_materialIndex = sphere->m_materialIndex;
  return true;
}

__device__ bool HitObjects(CudaRay * ray, CudaSphere * spheres, uint32_t spheresCount,
                           float znear, float zfar, CudaHit * hit)
{
  hit->m_parameterT = zfar + 1.0f;
  bool hitFound = false;
  for (uint32_t i = 0; i < spheresCount; ++i)
  {
    CudaHit h;
    if (HitSphere(&spheres[i], ray, znear, zfar, &h))
    {
      hitFound = true;
      if (h.m_parameterT < hit->m_parameterT)
        *hit = h;
    }
  }
  return hitFound;
}
}  // namespace ray_tracing_cuda
