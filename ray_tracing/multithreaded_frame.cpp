#include "multithreaded_frame.hpp"

#include <cstring>

namespace ray_tracing
{
MultiThreadedFrame::MultiThreadedFrame(uint32_t rayTracingThreadsCount)
  : m_rayTracingThreadsCount(rayTracingThreadsCount)
{}

bool MultiThreadedFrame::Initialize(std::shared_ptr<ColorBuffer> buffer,
                                    uint32_t width, uint32_t height,
                                    uint32_t samplesInRowCount)
{
  if (!Frame::Initialize(buffer, width, height, samplesInRowCount))
    return false;

  // Simple multithreading workers.
  m_areThreadRunning = true;
  m_traceFinishingCounter = 0;
  m_rayFunctions.resize(m_rayTracingThreadsCount);
  m_realtimeBuffer.resize(m_buffer->size());
  for (uint32_t i = 0; i < m_rayTracingThreadsCount; ++i)
  {
    m_threads.emplace_back(std::make_unique<std::thread>([this, i]()
    {
      ThreadRoutine(i);
    }));
  }
  return true;
}

void MultiThreadedFrame::ThreadRoutine(uint32_t index)
{
  while (m_areThreadRunning)
  {
    bool needTrace;
    RayHandler func;
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_condition.wait(lock, [this](){ return !m_areThreadRunning || m_needTrace; });
      needTrace = m_needTrace;
      func = m_rayFunctions[index];
      if (!m_areThreadRunning)
        break;
    }

    if (needTrace && func)
      ForEachRayImpl(std::move(func), index, m_rayTracingThreadsCount);

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_traceFinishingCounter++;
      m_rayFunctions[index] = nullptr;
      if (m_traceFinishingCounter == m_rayTracingThreadsCount)
        m_needTrace = false;
    }
  }
}

void MultiThreadedFrame::Uninitialize()
{
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_areThreadRunning = false;
    m_needTrace = false;
  }
  m_condition.notify_all();
  for (auto & threadPtr : m_threads)
    threadPtr->join();
}

bool MultiThreadedFrame::HasFinished()
{
  if (m_threads.empty())
    return true;

  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_needTrace)
    return true;
  return m_traceFinishingCounter == m_threads.size();
}

bool MultiThreadedFrame::InProgress()
{
  if (m_threads.empty())
    return false;

  std::lock_guard<std::mutex> lock(m_mutex);
  return m_needTrace && (m_traceFinishingCounter < m_threads.size());
}

void MultiThreadedFrame::CopyToBuffer(ColorBuffer & buffer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  memcpy(buffer.data(), m_realtimeBuffer.data(), buffer.size() * sizeof(glm::vec3));
}

void MultiThreadedFrame::ForEachRay(RayHandler && func)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_traceFinishingCounter = 0;
  m_needTrace = true;
  memset(m_realtimeBuffer.data(), 0, m_realtimeBuffer.size() * sizeof(glm::vec3));
  for (auto & rayFunction : m_rayFunctions)
    rayFunction = func;
  m_condition.notify_all();
}

bool MultiThreadedFrame::OnStartRow(uint32_t row)
{
  if (m_threads.empty())
    return true;

  std::lock_guard<std::mutex> lock(m_mutex);
  return m_areThreadRunning;
}

void MultiThreadedFrame::OnEndRow(uint32_t row)
{
  if (m_threads.empty())
    return;

  std::lock_guard<std::mutex> lock(m_mutex);
  memcpy(m_realtimeBuffer.data() + row * m_width, m_buffer->data() + row * m_width,
         m_width * sizeof(glm::vec3));
}
}  // namespace ray_tracing
