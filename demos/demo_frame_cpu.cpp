#include "demo_frame_cpu.hpp"

#include "ray_tracing/default_materials.hpp"

#include <array>
#include <cassert>

namespace demo
{
uint32_t constexpr kMaxDepth = 10;

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
  std::array<ray_tracing::Material::ScatterResult, kMaxDepth> scatterResults {};
  std::array<glm::vec3, kMaxDepth> diffuseLight {};
  std::array<glm::vec3, kMaxDepth> specularLight {};
  uint32_t scatterResultsCount = 0;

  // Back tracing.
  ray_tracing::Ray scatteredRay = ray;
  for (; scatterResultsCount < kMaxDepth; ++scatterResultsCount)
  {
    auto const hit = HitObjects(scatteredRay, znear, zfar);
    if (!hit)
      break;

    auto h = hit.value();
    scatterResults[scatterResultsCount] = h.m_material->Scatter(scatteredRay, h);
    if (fabs(scatterResults[scatterResultsCount].m_energyEmissivity) < ray_tracing::kEps)
    {
      scatterResults[scatterResultsCount].m_attenuation = m_backgroundColor;
      scatterResults[scatterResultsCount].m_energyEmissivity = 1.0f;
      ++scatterResultsCount;
      break;
    }

    diffuseLight[scatterResultsCount] = TraceDiffuseLight(scatteredRay, h);
    specularLight[scatterResultsCount] = GetSpecularLight(scatteredRay, h);

    scatteredRay = scatterResults[scatterResultsCount].m_scatteredRay;

    // Secondary hits start with a slight offset.
    znear = 0.001f;
  }

  // Get we use environment color.
  if (scatterResultsCount == 0)
    return m_backgroundColor;

  // Accumulate color with energy saving influence.
  int i = static_cast<int>(scatterResultsCount) - 1;
  glm::vec3 att = glm::vec3(1.0f, 1.0f, 1.0f);
  float restEnergy = 1.0f;
  while (i >= 0)
  {
    auto const l = scatterResults[i].m_attenuation + diffuseLight[i] + specularLight[i];
    att *= (l * restEnergy);
    restEnergy *= scatterResults[i].m_energyEmissivity;
    i--;
    if (fabs(restEnergy) < ray_tracing::kEps)
      break;
  }

  // Brightness correction.
  return 0.5f * att;
}

glm::vec3 DemoFrameCPU::TraceDiffuseLight(ray_tracing::Ray const & ray,
                                          ray_tracing::Hit const & hit)
{
  using namespace std::placeholders;
  glm::vec3 lightColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
  {
    lightColor += source->TraceLight(ray, hit,
      std::bind(&DemoFrameCPU::HitObjects, this, _1, _2, _3));
  }
  return lightColor;
}

glm::vec3 DemoFrameCPU::GetSpecularLight(ray_tracing::Ray const & ray,
                                         ray_tracing::Hit const & hit)
{
  glm::vec3 specularColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    specularColor += source->GetSpecular(ray, hit);
  return specularColor;
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
