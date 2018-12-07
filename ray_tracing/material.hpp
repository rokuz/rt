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
    float m_radiance = 0.0f; // 0.0 - absorb all, 1.0 - reflect all.
    glm::vec3 m_attenuation = {};
    Ray m_scatteredRay = {};

    ScatterResult() = default;
    ScatterResult(float radiance, glm::vec3 const & attenuation,
                  Ray const & scatteredRay)
      : m_radiance(radiance)
      , m_attenuation(attenuation)
      , m_scatteredRay(scatteredRay)
    {}
  };

  virtual ScatterResult Scatter(Ray const & ray, Hit const & hit) = 0;
};
}  // namespace ray_tracing
