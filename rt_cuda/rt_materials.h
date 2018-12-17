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

  if (dot(hit->m_normal, reflectVector) <= 0.0f)
    result->m_energyEmissivity = 0.0f;
  else
    result->m_energyEmissivity = 0.9f * (1.0f - mat->m_roughness);
}

__device__ bool refract(float3 I, float3 N, float eta, float3 * refractVector)
{
  float dotValue = dot(N, I);
  float k = (1.0f - eta * eta * (1.0f - dotValue * dotValue));
  if (k < 0.0f)
    return false;
  *refractVector = (eta * I - (eta * dotValue + sqrt(k)) * N);
  return true;
}

__device__ void ScatterGlass(CudaRay * ray, CudaHit * hit, CudaMaterial * materials,
                             curandState * randState, ScatterResult * result)
{
  CudaMaterial * mat = &materials[hit->m_materialIndex];
  float3 reflectVector = reflect(ray->m_direction, hit->m_normal);

  result->m_attenuation = mat->m_albedo;
  float3 outwardNormal;
  float eta;
  float vdn = dot(ray->m_direction, hit->m_normal);
  if (vdn > 0.0f)
  {
    outwardNormal = -hit->m_normal;
    eta = mat->m_refraction;
    vdn *= eta;
  }
  else
  {
    outwardNormal = hit->m_normal;
    eta = 1.0f / mat->m_refraction;
    vdn = -vdn;
  }

  float3 refractVector;
  float ref = 1.0f;
  if (refract(ray->m_direction, outwardNormal, eta, &refractVector))
  {
    float f0 = (1.0f - mat->m_refraction) / (1.0f + mat->m_refraction);
    float f2 = f0 * f0;
    ref = f2 + (1.0f - f2) * pow(1.0f - vdn, 5.0f);
  }
  result->m_scatteredRay.m_origin = hit->m_position;
  float rnd = 1.0f - curand_uniform(randState);
  result->m_scatteredRay.m_direction = (rnd < ref) ? reflectVector : refractVector;
  result->m_energyEmissivity = 1.0f;
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
