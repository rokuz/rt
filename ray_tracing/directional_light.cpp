#include "directional_light.hpp"
#include "default_materials.hpp"
#include "global.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>

#include <algorithm>
#include <cmath>

namespace ray_tracing
{
glm::vec3 DirectionalLight::TraceLight(Ray const & ray, Hit const & hit, Tracer && tracer)
{
  return TraceLightWithDepth(hit, std::move(tracer), 1);
}

glm::vec3 DirectionalLight::GetSpecular(Ray const & ray, Hit const & hit)
{
  // Cook-Torrance specular.
  float const vdn = glm::clamp(glm::dot(-ray.Direction(), hit.m_normal), 0.0f, 1.0f);
  float const ndl = glm::clamp(glm::dot(hit.m_normal, -m_direction), 0.0f, 1.0f);
  auto const h = -glm::normalize(m_direction + ray.Direction());
  float const ndh = glm::clamp(glm::dot(hit.m_normal, h), 0.0f, 1.0f);
  float const vdh = glm::clamp(glm::dot(-ray.Direction(), h), 0.0f, 1.0f);

  float const G = (fabs(vdh) >= kEps) ? std::min(1.0f, 2.0f * ndh * std::min(vdn, ndl) / vdh) : 1.0f;
  float const ndh2 = ndh * ndh;

  float D = 1.0f;
  float const roughness = std::max(hit.m_material->GetRoughness(), 0.03f);
  float const r2 = roughness * roughness;
  if (fabs(roughness) >= kEps && fabs(ndh2) >= kEps)
    D = exp((ndh2 - 1.0f) / (r2 * ndh2)) / (4.0f * r2 * ndh2 * ndh2);

  float const ref = hit.m_material->GetRefraction();
  float const f0 = (1.0f - ref) / (1.0f + ref);
  float const f2 = f0 * f0;
  float const F = f2 + (1.0f - f2) * pow(1.0f - vdn, 5.0f);

  float spec = 1.0f;
  spec = glm::clamp(G * D * F, 0.0f, 1.0f);
  return m_color * spec;
}

glm::vec3 DirectionalLight::TraceLightWithDepth(Hit const & hit, Tracer && tracer, int depth)
{
  if (!tracer || depth >= 3)
    return m_color;

  auto const lightDir = glm::normalize(-m_direction + glm::ballRand(0.1f));
  auto const hits = tracer(Ray(hit.m_position, lightDir), 0.01f, 1000.0f);
  if (hits.empty())
    return m_color;

  glm::vec3 c = glm::vec3(0.0f, 0.0f, 0.0f);
  uint32_t kCount = 5;
  for (uint32_t i = 0; i < kCount; ++i)
  {
    auto const dir = glm::normalize(hits[0].m_normal + glm::ballRand(1.0f));
    auto const hitsSecondary = tracer(Ray(hits[0].m_position, dir), 0.001f, 1000.0f);
    if (hitsSecondary.empty())
      c += 0.5f * m_color;
    else
      c += 0.5f * TraceLightWithDepth(hitsSecondary[0], std::move(tracer), depth + 1);
  }
  return c / static_cast<float>(kCount);
}
}  // namespace ray_tracing
