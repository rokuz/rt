#pragma once

#include "vector_types.h"

#include <cstdint>

namespace ray_tracing_cuda
{
struct CudaSphere
{
  uint32_t m_materialIndex;
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
  uint32_t m_materialIndex;
};
}  // namespace ray_tracing_cuda
