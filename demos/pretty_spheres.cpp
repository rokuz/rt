#include "pretty_spheres.hpp"

#include "ray_tracing/default_materials.hpp"

namespace demo
{
PrettySpheres::PrettySpheres(uint32_t rayTracingThreadsCount)
  : ray_tracing::MultiThreadedFrame(rayTracingThreadsCount)
{}

bool PrettySpheres::Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                               uint32_t width, uint32_t height,
                               uint32_t samplesInRowCount)
{
  if (!MultiThreadedFrame::Initialize(buffer, width, height, samplesInRowCount))
    return false;

  using namespace ray_tracing;

  m_cameraPosition = glm::vec3(0.0f, 8.0f, -15.0f);
  m_cameraDirection = glm::vec3(0.0f, -1.0f, 2.0f);

  m_samplesInRowCount = 3;

  std::uniform_int_distribution<> randomPos(-10, 10);
  std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);
  for (size_t i = 0; i < 10; ++i)
  {
    auto const x = static_cast<float>(randomPos(m_generator));
    auto const z = static_cast<float>(randomPos(m_generator));
    auto const c = glm::vec3(randomFloat(m_generator), randomFloat(m_generator), randomFloat(m_generator));

    std::shared_ptr<Material> mat;
    if (i >= 5)
      mat = std::make_shared<material::Metal>(c, 0.3f * randomFloat(m_generator));
    else
      mat = std::make_shared<material::Matte>(c);

    m_spheres.emplace_back(std::make_unique<Sphere>(glm::vec3(x, 0.0f, z), 1.0f, mat));
  }
  m_spheres.emplace_back(std::make_unique<Sphere>(
    glm::vec3(0.0, -1001.0f, 0.0), 1000.0f,
    std::make_shared<material::Matte>(glm::vec3(0.75f, 0.75f, 0.75f))));

  return true;
}

glm::vec3 PrettySpheres::RayTrace(ray_tracing::Ray const & ray, float near, float far)
{
  auto const hits = ray_tracing::TraceHitableCollection(m_spheres, ray, near, far);
  if (hits.empty())
    return glm::vec3(1.0f, 1.0f, 1.0f);

  auto const & nearestHit = hits[0];

  auto const scatterResult = nearestHit.m_material->Scatter(ray, nearestHit);
  if (scatterResult.m_radiance < 0.1f)
    return scatterResult.m_attenuation;

  return scatterResult.m_attenuation * RayTrace(scatterResult.m_scatteredRay, 0.001f, far);
}
}  // namespace demo