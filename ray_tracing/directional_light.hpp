#pragma once

#include "light.hpp"

#include <glm/vec3.hpp>

namespace ray_tracing
{
class DirectionalLight : public Light
{
public:
  DirectionalLight(glm::vec3 const & direction, glm::vec3 const & color)
    : m_direction(direction)
    , m_color(color)
  {}

  glm::vec3 TraceLight(Hit const & hit, Tracer && tracer) override;

private:
  glm::vec3 TraceLightWithDepth(Hit const & hit, Tracer && tracer, int depth);

  glm::vec3 const m_direction;
  glm::vec3 const m_color;
};
}  // namespace ray_tracing
