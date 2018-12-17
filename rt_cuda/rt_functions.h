#pragma once

#include "rt_hits.h"
#include "rt_light.h"
#include "rt_materials.h"
#include "rt_random.h"

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
int constexpr kMaxDepth = 10;

__device__ void TraceRayGPU(CudaRay * ray, CudaSphere * spheres, uint32_t spheresCount,
                            CudaMaterial * materials, CudaLight * lightSources,
                            uint32_t lightSourcesCount, float3 backgroundColor, float znear,
                            float zfar, curandState * randState, float3 * output)
{
  ScatterResult scatterResults[kMaxDepth];
  float3 diffuseLight[kMaxDepth];
  float3 specularLight[kMaxDepth];
  int scatterResultsCount = 0;

  // Back tracing.
  CudaHit hit;
  CudaRay * scatteredRay = ray;
  for (; scatterResultsCount < kMaxDepth; ++scatterResultsCount)
  {
    diffuseLight[scatterResultsCount] = make_float3(0.0f, 0.0f, 0.0f);
    specularLight[scatterResultsCount] = make_float3(0.0f, 0.0f, 0.0f);

    bool hitFound = HitObjects(scatteredRay, spheres, spheresCount, znear, zfar, &hit);
    if (!hitFound)
      break;

    Scatter(scatteredRay, &hit, materials, randState, &scatterResults[scatterResultsCount]);
    if (fabs(scatterResults[scatterResultsCount].m_energyEmissivity) < ray_tracing::kEps)
    {
      scatterResults[scatterResultsCount].m_attenuation = backgroundColor;
      scatterResults[scatterResultsCount].m_energyEmissivity = 1.0f;
      ++scatterResultsCount;
      break;
    }

    diffuseLight[scatterResultsCount] = TraceDiffuseLight(&hit, spheres, spheresCount, materials,
                                                          lightSources, lightSourcesCount, randState);
    specularLight[scatterResultsCount] = GetSpecularLight(scatteredRay, &hit, materials, lightSources,
                                                          lightSourcesCount, randState);

    scatteredRay = &scatterResults[scatterResultsCount].m_scatteredRay;

    // Secondary hits start with a slight offset.
    znear = 0.001f;
  }

  if (scatterResultsCount == 0)
  {
    // Get we use environment color.
    *output = backgroundColor;
  }
  else
  {
    // Accumulate color with energy saving influence.
    int i = static_cast<int>(scatterResultsCount) - 1;
    float3 att = make_float3(1.0f, 1.0f, 1.0f);
    float restEnergy = 1.0f;
    while (i >= 0)
    {
      float3 l = scatterResults[i].m_attenuation + diffuseLight[i] + specularLight[i];
      att *= (l * restEnergy);
      restEnergy *= scatterResults[i].m_energyEmissivity;
      i--;
      if (fabs(restEnergy) < ray_tracing::kEps)
        break;
    }

    // Brigthness correction.
    *output = 0.5f * att;
  }
}
}  // namespace ray_tracing_cuda
