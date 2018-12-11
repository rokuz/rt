#pragma once

#include "light.hpp"
#include "types.hpp"

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

  uint32_t GetType() const override { return kLightSourceDirectionalType; }

  glm::vec3 TraceLight(Ray const & ray, Hit const & hit, Tracer && tracer) override;
  glm::vec3 GetSpecular(Ray const & ray, Hit const & hit) override;

private:
  glm::vec3 TraceLightWithDepth(Hit const & hit, Tracer && tracer, int depth);

  glm::vec3 const m_direction;
  glm::vec3 const m_color;
};
}  // namespace ray_tracing
