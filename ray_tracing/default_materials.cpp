#include "default_materials.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/random.hpp>

namespace ray_tracing
{
namespace material
{
Material::ScatterResult Matte::Scatter(Ray const & ray, Hit const & hit)
{
  ScatterResult result;
  result.m_attenuation = m_albedo;
  auto const dir = glm::normalize(hit.m_normal + glm::ballRand(1.0f));
  result.m_scatteredRay = Ray(hit.m_position, dir);
  return result;
}

Material::ScatterResult Metal::Scatter(Ray const & ray, Hit const & hit)
{
  auto reflectVector = glm::reflect(ray.Direction(), hit.m_normal);
  reflectVector = glm::normalize(reflectVector + glm::ballRand(1.0f) * m_roughness);
  ScatterResult result;
  result.m_attenuation = m_albedo;
  result.m_scatteredRay = Ray(hit.m_position, reflectVector);
  if (glm::dot(hit.m_normal, reflectVector) <= 0.0f)
    result.m_energyImpact = 0.0f;
  return result;
}
}  // namespace material
}  // namespace ray_tracing
