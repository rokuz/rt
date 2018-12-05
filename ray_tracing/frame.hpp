#pragma once

#include "ray.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <memory>
#include <vector>

namespace ray_tracing
{
class Frame
{
public:
  Frame()
    : m_generator(m_random())
  {}

  virtual ~Frame() = default;

  virtual bool Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                          uint32_t width, uint32_t height)
  {
    m_buffer = std::move(buffer);
    m_width = width;
    m_height = height;
    return true;
  }

  virtual void Trace(double timeSinceStart, double elapsedTime) = 0;

protected:
  void ForEachRay(std::function<glm::vec3(Ray ray)> && func)
  {
    static glm::vec3 kUp = glm::vec3(0.0f, 1.0f, 0.0f);

    std::uniform_real_distribution<> distribution(0.0, 1.0);

    m_cameraDirection = glm::normalize(m_cameraDirection);

    auto const right = glm::cross(kUp, m_cameraDirection);
    auto const up = glm::cross(m_cameraDirection, right);

    auto const dw = m_znear / tan(0.5f * m_fov);
    auto const aspect = static_cast<float>(m_height) / m_width;
    auto const dh = dw * aspect;

    auto const cellX = 2.0f * dw / m_width;
    auto const cellY = 2.0f * dh / m_height;

    auto const invSampleCount = 1.0f / (m_samplesInRowCount * m_samplesInRowCount);
    for (size_t y = 0; y < m_height; ++y)
    {
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
    }
  }

  std::shared_ptr<std::vector<glm::vec3>> m_buffer;
  uint32_t m_width = 0;
  uint32_t m_height = 0;

  uint32_t m_samplesInRowCount = 1;
  std::random_device m_random;
  std::mt19937 m_generator;

  glm::vec3 m_cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_cameraDirection = glm::vec3(0.0f, 0.0f, 1.0f);
  float m_fov = static_cast<float>(2.0f * M_PI / 3.0f);
  float m_znear = 0.001f;
  float m_zfar = 1000.0f;
};
}  // namespace ray_tracing
