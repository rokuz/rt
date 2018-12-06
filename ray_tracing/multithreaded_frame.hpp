#pragma once

#include "frame.hpp"

#include "global.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <condition_variable>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ray_tracing
{
using ColorBuffer = std::vector<glm::vec3>;
using RayHandler = std::function<glm::vec3(Ray const & ray)>;

class MultiThreadedFrame : public Frame
{
public:
  explicit MultiThreadedFrame(uint32_t rayTracingThreadsCount);

  bool Initialize(std::shared_ptr<ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override;
  void Uninitialize() override;

  bool HasFinished() override;
  bool InProgress() override;
  void CopyToBuffer(ColorBuffer & buffer) override;

protected:
  void ThreadRoutine(uint32_t index);

  bool OnStartRow(uint32_t row) override;
  void OnEndRow(uint32_t row) override;
  void ForEachRay(RayHandler && func) override;

  uint32_t const m_rayTracingThreadsCount;

  std::vector<RayHandler> m_rayFunctions;
  bool m_areThreadRunning = false;
  bool m_needTrace = false;
  size_t m_traceFinishingCounter = 0;
  ColorBuffer m_realtimeBuffer;

  std::vector<std::unique_ptr<std::thread>> m_threads;
  std::condition_variable m_condition;
  std::mutex m_mutex;
};
}  // namespace ray_tracing
