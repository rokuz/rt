#pragma once

#include "ray.hpp"

#include "global.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace ray_tracing
{
using ColorBuffer = std::vector<glm::vec3>;
using RayHandler = std::function<glm::vec3(Ray const & ray)>;

class Frame
{
public:
  Frame();
  virtual ~Frame() = default;

  virtual bool Initialize(std::shared_ptr<ColorBuffer> buffer,
                          uint32_t width, uint32_t height,
                          uint32_t samplesInRowCount);
  virtual void Uninitialize() {}

  virtual void TraceAllRays();

  virtual bool HasFinished() { return true; }
  virtual bool InProgress() { return false; }
  virtual void CopyToBuffer(ColorBuffer & buffer) {}

protected:
  virtual glm::vec3 RayTrace(Ray const & ray, float near, float far) { return {}; }

  virtual void ForEachRay(RayHandler && func);
  virtual bool OnStartRow(uint32_t row) { return true; }
  virtual void OnEndRow(uint32_t row) {}

  void ForEachRayImpl(RayHandler && func, uint32_t startRow, uint32_t pitch);

  std::shared_ptr<ColorBuffer> m_buffer;
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  uint32_t m_samplesInRowCount = 1;

  std::random_device m_random;
  std::mt19937 m_generator;

  glm::vec3 m_cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_cameraDirection = glm::vec3(0.0f, 0.0f, 1.0f);
  float m_fov = 2.0f * kPi / 3.0f;
  float m_znear = 0.001f;
  float m_zfar = 1000.0f;
};
}  // namespace ray_tracing
