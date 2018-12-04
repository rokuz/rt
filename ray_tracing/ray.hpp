#pragma once

#include <glm/vec3.hpp>

namespace ray_tracing
{
class Ray
{
public:
  Ray() = default;

  Ray(glm::vec3 const & origin, glm::vec3 const & direction)
    : m_origin(origin)
    , m_direction(direction)
  {}

  glm::vec3 const & Origin() const { return m_origin; }
  glm::vec3 const & Direction() const { return m_direction; }
  glm::vec3 PointAt(float t) const { return m_origin + m_direction * t; }

private:
  glm::vec3 m_origin;
  glm::vec3 m_direction;
};
}  // namespace ray_tracing
