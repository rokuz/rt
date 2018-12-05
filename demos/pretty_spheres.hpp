#pragma once

#include "ray_tracing/frame.hpp"
#include "ray_tracing/sphere.hpp"

#include <glm/vec3.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace demo
{
class PrettySpheres : public ray_tracing::Frame
{
public:
  bool Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t rayTracingThreadsCount) override;
  void Trace(double timeSinceStart, double elapsedTime) override;

private:
  // Must be immutable during ray tracing.
  std::vector<std::unique_ptr<ray_tracing::Sphere>> m_spheres;
};
}  // namespace demo
