#pragma once

#include "ray_tracing/hitable_object.hpp"
#include "ray_tracing/light.hpp"
#include "ray_tracing/multithreaded_frame.hpp"
#include "ray_tracing/sphere.hpp"

#include <glm/vec3.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace demo
{
class GlassSpheres : public ray_tracing::MultiThreadedFrame
{
public:
  explicit GlassSpheres(uint32_t rayTracingThreadsCount);

  bool Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override;
  glm::vec3 RayTrace(ray_tracing::Ray const & ray, float near, float far) override;

private:
  glm::vec3 RayTraceObjects(ray_tracing::Ray const & ray, ray_tracing::Hit const & hit,
                            float near, float far, int depth);

  std::vector<ray_tracing::Hit> HitObjects(ray_tracing::Ray const & ray,
                                           float near, float far) const;
  // These collections must be immutable during ray tracing.
  std::vector<std::unique_ptr<ray_tracing::Sphere>> m_spheres;
  std::vector<std::unique_ptr<ray_tracing::Light>> m_lightSources;
};
}  // namespace demo
