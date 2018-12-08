#pragma once

#include "hitable_object.hpp"
#include "ray.hpp"

namespace ray_tracing
{
class Material
{
public:
  virtual ~Material() = default;

  struct ScatterResult
  {
    glm::vec3 m_attenuation = {};
    Ray m_scatteredRay = {};
    float m_energyImpact = 0.5f;
  };

  virtual ScatterResult Scatter(Ray const & ray, Hit const & hit) = 0;

  virtual glm::vec3 GetAlbedo() const { return glm::vec3(0.0f, 0.0f, 0.0f); }
  virtual float GetRoughness() const { return 0.7f; }
  virtual float GetRefraction() const { return 0.0f; }
};
}  // namespace ray_tracing
