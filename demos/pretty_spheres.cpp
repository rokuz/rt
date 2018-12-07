#include "pretty_spheres.hpp"

#include "ray_tracing/default_materials.hpp"
#include "ray_tracing/directional_light.hpp"

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

  m_cameraPosition = glm::vec3(0.0f, 8.0f, -20.0f);
  m_cameraDirection = glm::vec3(0.0f, -1.0f, 2.0f);

  //m_samplesInRowCount = 10;

  std::uniform_int_distribution<> randomPos(-10, 10);
  std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);
  for (size_t i = 0; i < 10; ++i)
  {
    auto const x = static_cast<float>(randomPos(m_generator));
    auto const z = static_cast<float>(randomPos(m_generator));
    auto const c = glm::vec3(randomFloat(m_generator), randomFloat(m_generator), randomFloat(m_generator));

    std::shared_ptr<Material> mat;
    if (i >= 5)
      mat = std::make_shared<material::Metal>(c, glm::mix(0.1f, 0.2f, randomFloat(m_generator)));
    else if (i >= 2)
      mat = std::make_shared<material::Metal>(c);
    else
      mat = std::make_shared<material::Matte>(c);

    m_spheres.emplace_back(std::make_unique<Sphere>(glm::vec3(x, 0.0f, z), 1.0f, mat));
  }
  m_spheres.emplace_back(std::make_unique<Sphere>(
    glm::vec3(0.0, -1001.0f, 0.0), 1000.0f,
    std::make_shared<material::Matte>(glm::vec3(0.75f, 0.75f, 0.75f))));

  m_lightSources.emplace_back(std::make_unique<DirectionalLight>(
    glm::normalize(glm::vec3(-0.4f, -1.0f, 0.6f)),
    glm::vec3(1.0f, 1.0f, 1.0f)));

  return true;
}

std::vector<ray_tracing::Hit> PrettySpheres::HitObjects(ray_tracing::Ray const & ray,
                                                        float near, float far) const
{
  return ray_tracing::TraceHitableCollection(m_spheres, ray, near, far);
}

glm::vec3 PrettySpheres::RayTrace(ray_tracing::Ray const & ray, float near, float far)
{
  using namespace std::placeholders;

  auto const hits = HitObjects(ray, near, far);
  if (hits.empty())
    return glm::vec3(1.0f, 1.0f, 1.0f);

  auto const diffuseColor = RayTraceObjects(ray, hits[0], 0.001f, far, 1);

  glm::vec3 specularColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    specularColor += source->GetSpecular(ray, hits[0]);
  if (!m_lightSources.empty())
    specularColor /= m_lightSources.size();

  return diffuseColor + specularColor;
}

glm::vec3 PrettySpheres::RayTraceObjects(ray_tracing::Ray const & ray, ray_tracing::Hit const & hit,
                                         float near, float far, int depth)
{
  using namespace std::placeholders;

  auto const scatterResult = hit.m_material->Scatter(ray, hit);

  glm::vec3 lightColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    lightColor += source->TraceLight(ray, hit, std::bind(&PrettySpheres::HitObjects, this, _1, _2, _3));
  if (!m_lightSources.empty())
    lightColor /= m_lightSources.size();

  auto c = lightColor * scatterResult.m_attenuation;
  if (depth >= 5)
    return c;

  auto const hits = HitObjects(scatterResult.m_scatteredRay, near, far);
  if (!hits.empty())
  {
    auto const sc = c * RayTraceObjects(scatterResult.m_scatteredRay, hits[0], near, far, depth + 1);
    c = glm::mix(sc, c, 0.5f);
  }
  return c;
}
}  // namespace demo
