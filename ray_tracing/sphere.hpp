#pragma once

#include "hitable_object.hpp"

#include <glm/vec3.hpp>

#include <vector>

namespace ray_tracing
{
class Sphere : public HitableObject
{
public:
  Sphere() = default;
  Sphere(glm::vec3 const & center, float radius)
    : m_center(center)
    , m_radius(radius)
  {}

  glm::vec3 const GetCenter() const { return m_center; }
  float GetRadius() const { return m_radius; }

  std::vector<Hit> Trace(Ray const & ray, float tmin, float tmax) const override;

private:
  glm::vec3 m_center;
  float m_radius = 1.0f;
};
}  // namespace ray_tracing