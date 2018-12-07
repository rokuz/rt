#include "directional_light.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>

namespace ray_tracing
{
glm::vec3 DirectionalLight::TraceLight(Hit const & hit, Tracer && tracer)
{
  return TraceLightWithDepth(hit, std::move(tracer), 1);
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
