#pragma once

#include "rt_cuda.h"

#include "ray_tracing/frame.hpp"

#include <glm/vec3.hpp>

#include <future>
#include <mutex>
#include <vector>

namespace ray_tracing_cuda
{
class DemoFrameCUDA : public ray_tracing::Frame
{
public:
  DemoFrameCUDA() = default;

  bool Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override;
  void Uninitialize() override;

  void TraceAllRays() override;

  bool HasFinished() override;
  bool InProgress() override;
  void CopyToBuffer(ray_tracing::ColorBuffer & buffer) override;

  void AddObject(std::unique_ptr<ray_tracing::HitableObject> && object) override;
  void AddLightSource(std::unique_ptr<ray_tracing::Light> && light) override;

private:
  uint32_t FindMaterial(std::shared_ptr<ray_tracing::Material> mat);

  bool m_rayTracingStarted = false;
  bool m_needInterrupt = false;
  std::mutex m_mutex;
  std::future<cudaEvent_t> m_future;
  cudaEvent_t m_completionEvent = nullptr;
  ray_tracing::ColorBuffer m_realtimeBuffer;

  // These collections must be immutable during ray tracing.
  std::vector<CudaSphere> m_spheres;
  std::vector<CudaMaterial> m_materials;
  std::vector<CudaLight> m_lightSources;
};
}  // namespace ray_tracing_cuda
