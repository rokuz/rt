#pragma once

#include "rt_cuda.h"

#include "ray_tracing/frame.hpp"

#include <glm/vec3.hpp>

#include <vector>

namespace demo
{
class DemoFrameCUDA : public ray_tracing::Frame
{
public:
  DemoFrameCUDA() = default;

  bool Initialize(std::shared_ptr<ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override;

  void TraceAllRays() override;

  void AddObject(std::unique_ptr<ray_tracing::HitableObject> && object) override;
  void AddLightSource(std::unique_ptr<ray_tracing::Light> && light) override;

private:
  uint32_t FindMaterial(std::shared_ptr<Material> mat);

  std::vector<float3> m_output;

  // These collections must be immutable during ray tracing.
  std::vector<ray_tracing_cuda::CudaSphere> m_spheres;
  std::vector<ray_tracing_cuda::CudaMaterial> m_materials;
  std::vector<ray_tracing_cuda::CudaLight> m_lightSources;
};
}  // namespace demo
