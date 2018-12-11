#include "demo_frame_cpu.hpp"

#include "ray_tracing/default_materials.hpp"

#include <cassert>

namespace demo
{
DemoFrameCPU::DemoFrameCPU(uint32_t rayTracingThreadsCount)
  : ray_tracing::MultiThreadedFrame(rayTracingThreadsCount)
{}

std::optional<ray_tracing::Hit> DemoFrameCPU::HitObjects(ray_tracing::Ray const & ray,
                                                         float znear, float zfar) const
{
  return ray_tracing::TraceNearestInHitableCollection(m_objects, ray, znear, zfar);
}

glm::vec3 DemoFrameCPU::RayTrace(ray_tracing::Ray const & ray, float znear, float zfar)
{
  using namespace std::placeholders;

  auto const hit = HitObjects(ray, znear, zfar);
  if (!hit)
    return m_backgroundColor;

  auto const diffuseColor = RayTraceObjects(ray, hit.value(), 0.001f, zfar, 1);

  glm::vec3 specularColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    specularColor += source->GetSpecular(ray, hit.value());
  if (!m_lightSources.empty())
    specularColor /= m_lightSources.size();

  return diffuseColor + specularColor;
}

glm::vec3 DemoFrameCPU::RayTraceObjects(ray_tracing::Ray const & ray,
                                        ray_tracing::Hit const & hit,
                                        float znear, float zfar, int depth)
{
  using namespace std::placeholders;

  auto const scatterResult = hit.m_material->Scatter(ray, hit);

  glm::vec3 lightColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
  {
    lightColor += source->TraceLight(ray, hit,
      std::bind(&DemoFrameCPU::HitObjects, this, _1, _2, _3));
  }
  if (!m_lightSources.empty())
    lightColor /= m_lightSources.size();

  auto c = lightColor * scatterResult.m_attenuation;
  if (depth >= 5 || fabs(scatterResult.m_energyImpact) < ray_tracing::kEps)
    return c;

  auto const h = HitObjects(scatterResult.m_scatteredRay, znear, zfar);
  if (h)
  {
    auto const sc = RayTraceObjects(scatterResult.m_scatteredRay, h.value(),
                                    znear, zfar, depth + 1);
    c = glm::mix(lightColor * sc, c, scatterResult.m_energyImpact);
  }
  return c;
}

void DemoFrameCPU::AddObject(std::unique_ptr<ray_tracing::HitableObject> && object)
{
  assert(!InProgress());
  m_objects.push_back(std::move(object));
}

void DemoFrameCPU::AddLightSource(std::unique_ptr<ray_tracing::Light> && light)
{
  assert(!InProgress());
  m_lightSources.push_back(std::move(light));
}
}  // namespace demo
