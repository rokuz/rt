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

    ScatterResult() = default;
    ScatterResult(glm::vec3 const & attenuation, Ray const & scatteredRay)
      : m_attenuation(attenuation)
      , m_scatteredRay(scatteredRay)
    {}
  };

  virtual ScatterResult Scatter(Ray const & ray, Hit const & hit) = 0;
  virtual float GetRoughness() const { return 1.0f; }
  virtual float GetRefraction() const { return 0.0f; }
};
}  // namespace ray_tracing
