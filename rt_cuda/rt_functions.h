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
  CudaHit hit;
  bool hitFound = HitObjects(ray, spheres, spheresCount, znear, zfar, &hit);

  if (hitFound)
  {
    ScatterResult scatterResults[kMaxDepth];
    float3 diffuseLight[kMaxDepth];
    int scatterResultsCount = 0;

    CudaRay * scatteredRay = ray;
    CudaHit h = hit;
    for (; scatterResultsCount < kMaxDepth; ++scatterResultsCount)
    {
      diffuseLight[scatterResultsCount] = make_float3(0.0f, 0.0f, 0.0f);
      Scatter(scatteredRay, &h, materials, randState, &scatterResults[scatterResultsCount]);
      if (fabs(scatterResults[scatterResultsCount].m_energyEmissivity) < ray_tracing::kEps)
        break;

      diffuseLight[scatterResultsCount] = TraceDiffuseLight(&h, spheres, spheresCount, materials,
                                                            lightSources, lightSourcesCount, randState);

      scatteredRay = &scatterResults[scatterResultsCount].m_scatteredRay;
      if (!HitObjects(scatteredRay, spheres, spheresCount, 0.001f, zfar, &h))
        break;
    }

    float3 outputColor;
    if (scatterResultsCount < 2)
    {
      outputColor = (scatterResults[0].m_attenuation + diffuseLight[0]);
    }
    else
    {
      int i = scatterResultsCount - 1;
      float3 att = (scatterResults[i].m_attenuation + diffuseLight[i]);
      float restEnergy = scatterResults[i].m_energyEmissivity;
      while (i > 0)
      {
        att *= ((scatterResults[i - 1].m_attenuation + diffuseLight[i - 1]) * restEnergy);
        restEnergy *= scatterResults[i - 1].m_energyEmissivity;
        i--;
        if (fabs(restEnergy) < ray_tracing::kEps)
          break;
      }
      outputColor = att;
    }

    // Brigthness correction.
    *output = 0.5f * outputColor;
  }
  else
  {
    *output = backgroundColor;
  }
}
}  // namespace ray_tracing_cuda
