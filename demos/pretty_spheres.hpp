#pragma once

#include "ray_tracing/multithreaded_frame.hpp"
#include "ray_tracing/sphere.hpp"

#include <glm/vec3.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace demo
{
class PrettySpheres : public ray_tracing::MultiThreadedFrame
{
public:
  explicit PrettySpheres(uint32_t rayTracingThreadsCount);

  bool Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override;
  glm::vec3 RayTrace(ray_tracing::Ray const & ray, float near, float far) override;

private:
  // Must be immutable during ray tracing.
  std::vector<std::unique_ptr<ray_tracing::Sphere>> m_spheres;
};
}  // namespace demo
