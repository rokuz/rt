#pragma once

#include "hitable_object.hpp"

#include <glm/vec3.hpp>

#include <functional>
#include <optional>
#include <vector>

namespace ray_tracing
{
class Light
{
public:
  using Tracer = std::function<std::optional<Hit>(Ray const & ray, float tmin, float tmax)>;

  virtual ~Light() = default;

  virtual uint32_t GetType() const = 0;

  virtual glm::vec3 TraceLight(Ray const & ray, Hit const & hit, Tracer && tracer) = 0;
  virtual glm::vec3 GetSpecular(Ray const & ray, Hit const & hit) = 0;
};
}  // namespace ray_tracing
