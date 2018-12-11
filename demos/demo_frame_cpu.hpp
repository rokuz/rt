#pragma once

#include "ray_tracing/multithreaded_frame.hpp"

#include <glm/vec3.hpp>

#include <vector>

namespace demo
{
class DemoFrameCPU : public ray_tracing::MultiThreadedFrame
{
public:
  explicit DemoFrameCPU(uint32_t rayTracingThreadsCount);

  glm::vec3 RayTrace(ray_tracing::Ray const & ray, float znear, float zfar) override;

  void AddObject(std::unique_ptr<ray_tracing::HitableObject> && object) override;
  void AddLightSource(std::unique_ptr<ray_tracing::Light> && light) override;

private:
  glm::vec3 RayTraceObjects(ray_tracing::Ray const & ray, ray_tracing::Hit const & hit,
                            float znear, float zfar, int depth);

  std::optional<ray_tracing::Hit> HitObjects(ray_tracing::Ray const & ray,
                                             float znear, float zfar) const;

  // These collections must be immutable during ray tracing.
  std::vector<std::unique_ptr<ray_tracing::HitableObject>> m_objects;
  std::vector<std::unique_ptr<ray_tracing::Light>> m_lightSources;
};
}  // namespace demo
