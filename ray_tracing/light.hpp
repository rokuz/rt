#pragma once

#include "hitable_object.hpp"

#include <glm/vec3.hpp>

#include <functional>
#include <vector>

namespace ray_tracing
{
class Light
{
public:
  using Tracer = std::function<std::vector<Hit>(Ray const & ray,float tmin, float tmax)>;

  virtual ~Light() = default;

  virtual glm::vec3 TraceLight(Hit const & hit, Tracer && tracer) = 0;
};
}  // namespace ray_tracing
