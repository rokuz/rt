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
  glm::vec3 GetAlbedo() const override { return m_albedo; }

private:
  glm::vec3 m_albedo;
};

class Metal : public Material
{
public:
  Metal(glm::vec3 const & albedo, float roughness, float refraction)
    : m_albedo(albedo)
    , m_roughness(roughness)
    , m_refraction(refraction)
  {}

  ScatterResult Scatter(Ray const & ray, Hit const & hit) override;

  glm::vec3 GetAlbedo() const override { return m_albedo; }
  float GetRoughness() const override { return m_roughness; }
  float GetRefraction() const override { return m_refraction; }

private:
  glm::vec3 m_albedo;
  float m_roughness;
  float m_refraction;
};
}  // namespace material
}  // namespace ray_tracing
