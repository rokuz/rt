#include "frame.hpp"

namespace ray_tracing
{
Frame::Frame()
  : m_generator(m_random())
{}

bool Frame::Initialize(std::shared_ptr<ColorBuffer> buffer,
                       uint32_t width, uint32_t height,
                       uint32_t samplesInRowCount)
{
  m_buffer = std::move(buffer);
  m_width = width;
  m_height = height;
  m_samplesInRowCount = samplesInRowCount;
  return true;
}

void Frame::TraceAllRays()
{
  ForEachRay([this](ray_tracing::Ray const & ray)
  {
    return RayTrace(ray, m_znear, m_zfar);
  });
}

void Frame::ForEachRay(RayHandler && func)
{
  m_cameraDirection = glm::normalize(m_cameraDirection);
  ForEachRayImpl(std::move(func), 0, 1);
}

void Frame::ForEachRayImpl(RayHandler && func, uint32_t startRow, uint32_t pitch)
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
  for (uint32_t y = startRow; y < m_height; y += pitch)
  {
    if (!OnStartRow(y))
      break;

    auto const dy = (-2.0f * static_cast<float>(y) / m_height + 1.0f) * dh;
    for (uint32_t x = 0; x < m_width; ++x)
    {
      auto const dx = (2.0f * static_cast<float>(x) / m_width - 1.0f) * dw;

      // Trace with supersampling.
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

    OnEndRow(y);
  }
}
}  // namespace ray_tracing
