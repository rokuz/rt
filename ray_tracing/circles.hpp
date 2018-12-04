#pragma once

#include "frame.hpp"

namespace ray_tracing
{
class Circles : public Frame
{
public:
  bool Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                  uint32_t width, uint32_t height) override;
  void Trace(double timeSinceStart, double elapsedTime) override;
};
}  // namespace ray_tracing
