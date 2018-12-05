#include "pretty_spheres.hpp"

#include <random>

namespace demo
{
bool PrettySpheres::Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                               uint32_t width, uint32_t height,
                               uint32_t rayTracingThreadsCount)
{
  if (!Frame::Initialize(buffer, width, height, rayTracingThreadsCount))
    return false;

  m_samplesInRowCount = 3;

  m_cameraPosition = glm::vec3(0.0f, 8.0f, -15.0f);
  m_cameraDirection = glm::vec3(0.0f, -1.0f, 2.0f);

  std::uniform_int_distribution<> distribution(-10, 10);
  for (size_t i = 0; i < 10; ++i)
  {
    auto const x = static_cast<float>(distribution(m_generator));
    auto const z = static_cast<float>(distribution(m_generator));

    m_spheres.emplace_back(std::make_unique<ray_tracing::Sphere>(glm::vec3(x, 0.0f, z), 1.0f));
  }
  m_spheres.emplace_back(std::make_unique<ray_tracing::Sphere>(glm::vec3(0.0, -1001.0f, 0.0), 1000.0f));

  return true;
}

void PrettySpheres::Trace(double timeSinceStart, double elapsedTime)
{
  ForEachRay([this](ray_tracing::Ray ray)
  {
    auto const hits = ray_tracing::TraceHitableCollection(m_spheres, ray, m_znear, m_zfar);
    if (hits.empty())
      return glm::vec3(1.0f, 1.0f, 1.0f);

    auto const & nearestHit = hits[0];
    return nearestHit.m_normal * 0.5f + 0.5f;
  });
}
}  // namespace demo