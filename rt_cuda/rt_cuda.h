#pragma once

#include "ray_tracing/types.hpp"

#include <cstdint>

namespace ray_tracing_cuda
{
extern bool Initialize();

struct CudaSphere
{
  __device__ __host__ uint1 m_materialIndex;
  __device__ __host__ float3 m_center;
  __device__ __host__ float m_radius;
};

struct CudaMaterial
{
  __device__ __host__ uchar1 m_materialType;
  __device__ __host__ float3 m_albedo;
  __device__ __host__ float m_roughness;
  __device__ __host__ float m_refraction;
};

struct CudaLight
{
  __device__ __host__ uchar1 m_lightType;
  __device__ __host__ float3 m_direction;
  __device__ __host__ float3 m_color;
};

extern void RayTrace(CudaSphere * spheres, uint32_t spheresCount,
                     CudaMaterial * materials, uint32_t materialsCount,
                     CudaLight * lightSources, uint32_t lightSourcesCount,
                     uint32_t samplesInRowCount, float3 backgroundColor,
                     float3 cameraPosition, float3 cameraDirection,
                     float fov, float znear, float zfar,
                     uint32_t width, uint32_t height, float3 & output);
}  // namespace ray_tracing_cuda