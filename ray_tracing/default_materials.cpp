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
  result.m_radiance = 0.2126f * m_albedo.x + 0.7152f * m_albedo.y + 0.0722f * m_albedo.z;
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
  auto const lum = 0.2126f * m_albedo.x + 0.7152f * m_albedo.y + 0.0722f * m_albedo.z;
  result.m_radiance = (glm::dot(reflectVector, hit.m_normal)) > 0 ? lum : 0.0f;
  result.m_attenuation = m_albedo;
  result.m_scatteredRay = Ray(hit.m_position, reflectVector);
  return result;
}
}  // namespace material
}  // namespace ray_tracing
