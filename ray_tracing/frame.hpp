#pragma once

#include "ray.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <condition_variable>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace ray_tracing
{
class Frame
{
public:
  Frame();
  virtual ~Frame() = default;

  virtual bool Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                          uint32_t width, uint32_t height,
                          uint32_t rayTracingThreadsCount);
  virtual void Uninitialize();

  virtual void Trace(double timeSinceStart, double elapsedTime) = 0;

  bool HasFinished();
  bool InProgress();
  void CopyToBuffer(std::vector<glm::vec3> & buffer);

protected:
  void ForEachRay(std::function<glm::vec3(Ray ray)> && func);
  void ForEachRayImpl(std::function<glm::vec3(Ray ray)> && func,
                      size_t startRow, size_t pitch);

  std::shared_ptr<std::vector<glm::vec3>> m_buffer;
  uint32_t m_width = 0;
  uint32_t m_height = 0;

  uint32_t m_samplesInRowCount = 1;
  std::random_device m_random;
  std::mt19937 m_generator;

  std::vector<std::unique_ptr<std::thread>> m_threads;
  std::vector<std::function<glm::vec3(Ray ray)>> m_rayFunctions;
  bool m_areThreadRunning = false;
  bool m_needTrace = false;
  size_t m_traceFinishingCounter = 0;
  std::vector<glm::vec3> m_realtimeBuffer;
  std::condition_variable m_condition;
  std::mutex m_mutex;

  glm::vec3 m_cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_cameraDirection = glm::vec3(0.0f, 0.0f, 1.0f);
  float m_fov = static_cast<float>(2.0f * M_PI / 3.0f);
  float m_znear = 0.001f;
  float m_zfar = 1000.0f;
};
}  // namespace ray_tracing
