#pragma once

#include <glm/vec3.hpp>

#include <cassert>
#include <cstdint>
#include <memory>

namespace ray_tracing
{
class Frame
{
public:
  Frame() = default;
  virtual ~Frame() = default;

  virtual bool Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                          uint32_t width, uint32_t height) = 0;
  virtual void Trace() = 0;
};
}  // namespace ray_tracing
