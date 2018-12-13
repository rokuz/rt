#pragma once

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
struct ScatterResult
{
  float3 m_attenuation;
  CudaRay m_scatteredRay;
  float m_energyEmissivity;
};

__device__ void ScatterMatte(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                             curandState * randState, ScatterResult * result)
{
  result->m_attenuation = materials[hit->m_materialIndex].m_albedo;
  result->m_scatteredRay.m_origin = hit->m_position;
  result->m_scatteredRay.m_direction = normalize(hit->m_normal + BallRand(randState, 1.0f));
  result->m_energyEmissivity = 0.7f;
}

__device__ void ScatterMetal(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                             curandState * randState, ScatterResult * result)
{
  CudaMaterial * mat = &materials[hit->m_materialIndex];
  float3 reflectVector = reflect(ray->m_direction, hit->m_normal);
  reflectVector = normalize(reflectVector + BallRand(randState, 1.0f) * mat->m_roughness);
  result->m_attenuation = mat->m_albedo;
  result->m_scatteredRay.m_origin = hit->m_position;
  result->m_scatteredRay.m_direction = reflectVector;

  if (dot(hit->m_normal, reflectVector) < 0.0f)
  {
    result->m_attenuation = make_float3(0.0f, 0.0f, 0.0f);
    result->m_energyEmissivity = 0.0f;
  }
  else
  {
    result->m_energyEmissivity = 0.9f * (1.0f - mat->m_roughness);
  }
}

__device__ void ScatterGlass(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                             curandState * randState, ScatterResult * result)
{

}

__device__ void Scatter(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                        curandState * randState, ScatterResult * result)
{
  uint8_t materialType = materials[hit->m_materialIndex].m_materialType;
  if (materialType == ray_tracing::kMaterialMatteType)
    ScatterMatte(ray, hit, materials, randState, result);
  else if (materialType == ray_tracing::kMaterialMetalType)
    ScatterMetal(ray, hit, materials, randState, result);
  else if (materialType == ray_tracing::kMaterialGlassType)
    ScatterGlass(ray, hit, materials, randState, result);
}
}  // namespace ray_tracing_cuda
