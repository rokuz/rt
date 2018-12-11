#include "demo_frame_cuda.hpp"

#include "ray_tracing/default_materials.hpp"
#include "ray_tracing/directional_light.hpp"
#include "ray_tracing/sphere.hpp"
#include "ray_tracing/types.hpp"

#include "vector_functions.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace ray_tracing_cuda
{
bool DemoFrameCUDA::Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                               uint32_t width, uint32_t height,
                               uint32_t samplesInRowCount)
{
  if (!Frame::Initialize(buffer, width, height, samplesInRowCount))
    return false;

  m_output.resize(m_width * m_height);
  return true;
}

void DemoFrameCUDA::TraceAllRays()
{
  auto bgColor = make_float3(m_backgroundColor.x, m_backgroundColor.y, m_backgroundColor.z);
  auto cp = make_float3(m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
  auto cd = make_float3(m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);

  float3 * o = m_output.data();
  ray_tracing_cuda::RayTrace(m_spheres.data(), static_cast<uint32_t>(m_spheres.size()),
                             m_materials.data(), static_cast<uint32_t>(m_materials.size()),
                             m_lightSources.data(), static_cast<uint32_t>(m_lightSources.size()),
                             m_samplesInRowCount, bgColor, cp, cd, m_fov, m_znear, m_zfar,
                             m_width, m_height, o);

  memcpy(m_buffer->data(), m_output.data(), m_output.size() * sizeof(float3));
}

void DemoFrameCUDA::AddObject(std::unique_ptr<ray_tracing::HitableObject> && object)
{
  if (dynamic_cast<ray_tracing::Sphere *>(object.get()) != nullptr)
  {
    ray_tracing::Sphere * ptr = static_cast<ray_tracing::Sphere *>(object.get());

    auto const c = ptr->GetCenter();
    auto const r = ptr->GetRadius();

    ray_tracing_cuda::CudaSphere s;
    s.m_materialIndex = make_uint1(FindMaterial(ptr->GetMaterial()));
    s.m_center = make_float3(c.x, c.y, c.z);
    s.m_radius = r;

    m_spheres.push_back(std::move(s));
  }
}

uint32_t DemoFrameCUDA::FindMaterial(std::shared_ptr<ray_tracing::Material> mat)
{
  auto const aldebo = mat->GetAlbedo();
  auto const roughness = mat->GetRoughness();
  auto const refraction = mat->GetRefraction();

  auto it = std::find_if(m_materials.begin(), m_materials.end(),
    [&aldebo, &roughness, &refraction](ray_tracing_cuda::CudaMaterial const & m)
  {
    return fabs(m.m_albedo.x - aldebo.x) < ray_tracing::kEps &&
           fabs(m.m_albedo.y - aldebo.y) < ray_tracing::kEps &&
           fabs(m.m_albedo.z - aldebo.z) < ray_tracing::kEps &&
           fabs(m.m_roughness - roughness) < ray_tracing::kEps &&
           fabs(m.m_refraction - refraction) < ray_tracing::kEps;
  });
  if (it != m_materials.end())
    return static_cast<uint32_t>(std::distance(m_materials.begin(), it));

  ray_tracing_cuda::CudaMaterial cm;
  cm.m_materialType = make_uchar1(mat->GetType());
  cm.m_albedo = make_float3(aldebo.x, aldebo.y, aldebo.z);
  cm.m_roughness = roughness;
  cm.m_refraction = refraction;
  m_materials.push_back(std::move(cm));
  return static_cast<uint32_t>(m_materials.size() - 1);
}

void DemoFrameCUDA::AddLightSource(std::unique_ptr<ray_tracing::Light> && light)
{
  if (dynamic_cast<ray_tracing::DirectionalLight *>(light.get()) != nullptr)
  {
    ray_tracing::DirectionalLight * ptr = static_cast<ray_tracing::DirectionalLight *>(light.get());

    auto const c = ptr->GetColor();
    auto const d = ptr->GetDirection();

    ray_tracing_cuda::CudaLight l;
    l.m_lightType = make_uchar1(light->GetType());
    l.m_direction = make_float3(d.x, d.y, d.z);
    l.m_color = make_float3(c.x, c.y, c.z);
    m_lightSources.push_back(std::move(l));
  }
}
}  // namespace ray_tracing_cuda
