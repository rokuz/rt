#pragma once

#include "material.hpp"

#include <glm/vec3.hpp>

namespace ray_tracing
{
namespace material
{
class Matte : public Material
{
public:
  explicit Matte(glm::vec3 const & albedo)
    : m_albedo(albedo)
  {}

  ScatterResult Scatter(Ray const & ray, Hit const & hit) override;

private:
  glm::vec3 m_albedo;
};

class Metal : public Material
{
public:
  Metal(glm::vec3 const & albedo, float roughness = 0.0f)
    : m_albedo(albedo)
    , m_roughness(roughness)
  {}

  ScatterResult Scatter(Ray const & ray, Hit const & hit) override;

private:
  glm::vec3 m_albedo;
  float m_roughness;
};
}  // namespace material
}  // namespace ray_tracing
