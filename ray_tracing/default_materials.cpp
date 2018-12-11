#include "default_materials.hpp"
#include "types.hpp"

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
  result.m_energyImpact = 0.7f;
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
  else
    result.m_energyImpact = 0.3f;
  return result;
}

Material::ScatterResult Glass::Scatter(Ray const & ray, Hit const & hit)
{
  auto const reflectVector = glm::reflect(ray.Direction(), hit.m_normal);
  auto const refraction = hit.m_material->GetRefraction();
  ScatterResult result;
  result.m_attenuation = m_albedo;
  glm::vec3 outwardNormal;
  float eta;
  auto vdn = glm::dot(ray.Direction(), hit.m_normal);
  if (vdn > 0.0f)
  {
    outwardNormal = -hit.m_normal;
    eta = refraction;
    vdn *= eta;
  }
  else
  {
    outwardNormal = hit.m_normal;
    eta = 1.0f / refraction;
    vdn = -vdn;
  }
  auto const refractVector = glm::refract(ray.Direction(), outwardNormal, eta);
  float ref = 1.0f;
  if (glm::dot(refractVector, refractVector) > kEps)
  {
    float const f0 = (1.0f - refraction) / (1.0f + refraction);
    float const f2 = f0 * f0;
    ref = f2 + (1.0f - f2) * pow(1.0f - vdn, 5.0f);
  }
  result.m_scatteredRay = Ray(hit.m_position, (glm::linearRand(0.0f, 1.0f) < ref) ? reflectVector : refractVector);
  result.m_energyImpact = 0.2f;
  return result;
}
}  // namespace material
}  // namespace ray_tracing
