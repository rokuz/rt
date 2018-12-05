#include "frame.hpp"

#include <cstring>

namespace ray_tracing
{
Frame::Frame()
  : m_generator(m_random())
{}

bool Frame::Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                       uint32_t width, uint32_t height,
                       uint32_t rayTracingThreadsCount)
{
  m_buffer = std::move(buffer);
  m_width = width;
  m_height = height;

  // Simple multithreading workers.
  m_areThreadRunning = true;
  m_traceFinishingCounter = 0;
  m_rayFunctions.resize(rayTracingThreadsCount);
  m_realtimeBuffer.resize(m_buffer->size());
  for (uint32_t i = 0; i < rayTracingThreadsCount; ++i)
  {
    m_threads.emplace_back(std::make_unique<std::thread>(
      [this, index = i, rayTracingThreadsCount]()
    {
      while (true)
      {
        bool needTrace;
        std::function<glm::vec3(Ray ray)> func;
        {
          std::unique_lock<std::mutex> lock(m_mutex);
          m_condition.wait(lock, [this](){return !m_areThreadRunning || m_needTrace;});
          needTrace = m_needTrace;
          func = m_rayFunctions[index];
          if (!m_areThreadRunning)
            break;
        }

        if (needTrace && func)
          ForEachRayImpl(std::move(func), index, rayTracingThreadsCount);

        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_traceFinishingCounter++;
          m_rayFunctions[index] = nullptr;
          if (m_traceFinishingCounter == rayTracingThreadsCount)
            m_needTrace = false;
        }
      }
    }));
  }
  return true;
}

void Frame::Uninitialize()
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

bool Frame::HasFinished()
{
  if (m_threads.empty())
    return true;

  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_needTrace)
    return true;
  return m_traceFinishingCounter == m_threads.size();
}

bool Frame::InProgress()
{
  if (m_threads.empty())
    return false;

  std::lock_guard<std::mutex> lock(m_mutex);
  return m_needTrace && (m_traceFinishingCounter < m_threads.size());
}

void Frame::CopyToBuffer(std::vector<glm::vec3> & buffer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  memcpy(buffer.data(), m_realtimeBuffer.data(), buffer.size() * sizeof(glm::vec3));
}

void Frame::ForEachRay(std::function<glm::vec3(Ray ray)> && func)
{
  m_cameraDirection = glm::normalize(m_cameraDirection);
  if (m_threads.empty())
  {
    ForEachRayImpl(std::move(func), 0, 1);
  }
  else
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_traceFinishingCounter = 0;
    m_needTrace = true;
    memset(m_realtimeBuffer.data(), 0, m_realtimeBuffer.size() * sizeof(glm::vec3));
    for (size_t i = 0; i < m_rayFunctions.size(); ++i)
      m_rayFunctions[i] = func;
    m_condition.notify_all();
  }
}

void Frame::ForEachRayImpl(std::function<glm::vec3(Ray ray)> && func,
                           size_t startRow, size_t pitch)
{
  static glm::vec3 kUp = glm::vec3(0.0f, 1.0f, 0.0f);

  auto const right = glm::cross(kUp, m_cameraDirection);
  auto const up = glm::cross(m_cameraDirection, right);

  auto const dw = m_znear / tan(0.5f * m_fov);
  auto const aspect = static_cast<float>(m_height) / m_width;
  auto const dh = dw * aspect;

  auto const cellX = 2.0f * dw / m_width;
  auto const cellY = 2.0f * dh / m_height;

  auto const invSampleCount = 1.0f / (m_samplesInRowCount * m_samplesInRowCount);
  for (size_t y = startRow; y < m_height; y += pitch)
  {
    if (!m_threads.empty())
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      if (!m_areThreadRunning)
        break;
    }

    auto const dy = (-2.0f * static_cast<float>(y) / m_height + 1.0f) * dh;
    for (size_t x = 0; x < m_width; ++x)
    {
      auto const dx = (2.0f * static_cast<float>(x) / m_width - 1.0f) * dw;

      glm::vec3 color = {};
      for (uint32_t sampleX = 0; sampleX < m_samplesInRowCount; ++sampleX)
      {
        auto const sdx = dx + cellX * sampleX / m_samplesInRowCount;
        for (uint32_t sampleY = 0; sampleY < m_samplesInRowCount; ++sampleY)
        {
          auto const sdy = dy - cellY * sampleY / m_samplesInRowCount;
          auto const d = glm::normalize(m_cameraDirection * m_znear + up * sdy + right * sdx);
          if (func)
            color += func(Ray(m_cameraPosition, d));
        }
      }
      (*m_buffer)[y * m_width + x] = (color * invSampleCount);
    }

    if (!m_threads.empty())
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      memcpy(m_realtimeBuffer.data() + y * m_width, m_buffer->data() + y * m_width,
             m_width * sizeof(glm::vec3));
    }
  }
}
}  // namespace ray_tracing
