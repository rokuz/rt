#include "directional_light.hpp"
#include "default_materials.hpp"
#include "types.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>

#include <algorithm>
#include <cmath>

namespace ray_tracing
{
uint32_t constexpr kLightSamplesCount = 5;

glm::vec3 DirectionalLight::TraceLight(Ray const & ray, Hit const & hit, Tracer && tracer)
{
  glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
  for (uint32_t i = 0; i < kLightSamplesCount; ++i)
  {
    Ray const r(hit.m_position, glm::normalize(-m_direction + glm::ballRand(0.25f)));

    if (!tracer(r, 0.001f, 1000.0f))
    {
      color += (m_color * std::max(0.0f, glm::dot(r.Direction(), hit.m_normal)));
    }
    else
    {
      // Highly refracted surfaces can color shadows.
      auto const c = glm::mix(hit.m_material->GetAlbedo(), m_color, 0.75f);
      color += (hit.m_material->GetRefraction() * 0.75f * c *
                std::max(0.0f, glm::dot(r.Direction(), hit.m_normal)));
    } 
  }
  return color / static_cast<float>(kLightSamplesCount);
}

glm::vec3 DirectionalLight::GetSpecular(Ray const & ray, Hit const & hit)
{
  // Cook-Torrance specular.
  float const vdn = glm::clamp(glm::dot(-ray.Direction(), hit.m_normal), 0.0f, 1.0f);
  float const ndl = glm::clamp(glm::dot(hit.m_normal, -m_direction), 0.0f, 1.0f);
  auto const h = -glm::normalize(m_direction + ray.Direction());
  float const ndh = glm::clamp(glm::dot(hit.m_normal, h), kEps, 1.0f);
  float const vdh = glm::clamp(glm::dot(-ray.Direction(), h), 0.0f, 1.0f);

  float const G = (fabs(vdh) >= kEps) ? std::min(1.0f, 2.0f * ndh * std::min(vdn, ndl) / vdh) : 1.0f;
  float const ndh2 = ndh * ndh;

  float const roughness = std::max(hit.m_material->GetRoughness(), 0.03f);
  float const r2 = roughness * roughness;
  float D = exp((ndh2 - 1.0f) / (r2 * ndh2)) / (4.0f * r2 * ndh2 * ndh2);

  float const ref = hit.m_material->GetRefraction();
  float const f0 = (1.0f - ref) / (1.0f + ref);
  float const f2 = f0 * f0;
  float const F = f2 + (1.0f - f2) * pow(1.0f - vdn, 5.0f);

  float const spec = glm::clamp(G * D * F, 0.0f, 1.0f);
  return glm::mix(m_color, hit.m_material->GetAlbedo(), 0.5f) * spec;
}
}  // namespace ray_tracing
