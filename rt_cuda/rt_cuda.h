#pragma once

#include "driver_types.h"
#include "vector_types.h"

#include "ray_tracing/types.hpp"

#include <cstdint>
#include <functional>

namespace ray_tracing_cuda
{
extern bool Initialize();

struct CudaSphere
{
  uint8_t m_materialIndex;
  float3 m_center;
  float m_radius;
};

struct CudaMaterial
{
  uint8_t m_materialType;
  float3 m_albedo;
  float m_roughness;
  float m_refraction;
};

struct CudaLight
{
  uint8_t m_lightType;
  float3 m_direction;
  float3 m_color;
};

struct CudaRay
{
  float3 m_origin;
  float3 m_direction;
};

struct CudaHit
{
  float m_parameterT;
  float3 m_position;
  float3 m_normal;
  uint8_t m_materialIndex;
};

extern cudaEvent_t RayTrace(CudaSphere * spheres, uint32_t spheresCount, CudaMaterial * materials,
                            uint32_t materialsCount, CudaLight * lightSources, uint32_t lightSourcesCount,
                            uint32_t samplesInRowCount, float3 backgroundColor, float3 cameraPosition,
                            float3 cameraDirection, float fov, float znear, float zfar, 
                            uint32_t width, uint32_t height, std::function<bool()> && realtimeHandler);
extern bool InProgress(cudaEvent_t completion);
extern void CopyOutputToBuffer(float * buffer);
extern void FinishRayTrace(float * output, cudaEvent_t completion);
}  // namespace ray_tracing_cuda
