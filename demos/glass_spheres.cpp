#include "glass_spheres.hpp"
#include "palette.hpp"

#include "ray_tracing/default_materials.hpp"
#include "ray_tracing/directional_light.hpp"

#include <set>

namespace demo
{
GlassSpheres::GlassSpheres(uint32_t rayTracingThreadsCount)
  : ray_tracing::MultiThreadedFrame(rayTracingThreadsCount)
{}

bool GlassSpheres::Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                              uint32_t width, uint32_t height,
                              uint32_t samplesInRowCount)
{
  if (!MultiThreadedFrame::Initialize(buffer, width, height, samplesInRowCount))
    return false;

  using namespace ray_tracing;

  m_cameraPosition = glm::vec3(0.0f, 2.0f, -10.0f);
  m_cameraDirection = glm::vec3(0.0f, -1.0f, 3.0f);

  std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);

  std::set<std::pair<int, int>> s;
  std::uniform_int_distribution<> randomPos(-5, 5);
  auto geneneratePosition = [this, &s, &randomPos]()
  {
    std::pair<int, int> p;
    int i = 0;
    do
    {
      p = std::make_pair(randomPos(m_generator), randomPos(m_generator));
      ++i;
    }
    while (s.find(p) != s.end() && i < 5000);
    s.insert(p);
    return p;
  };

  for (size_t i = 0; i < 40; ++i)
  {
    auto const p = geneneratePosition();
    auto const c = Palette::RandomFromAll(m_generator);

    auto mat = std::make_shared<material::Glass>(c, glm::mix(0.7f, 0.9f, randomFloat(m_generator)));
    float const radius = glm::mix(0.25f, 0.5f, randomFloat(m_generator));
    m_spheres.emplace_back(std::make_unique<Sphere>(glm::vec3(p.first, radius - 1.0f, p.second), radius, mat));
  }
  m_spheres.emplace_back(std::make_unique<Sphere>(
    glm::vec3(0.0, -1001.0f, 0.0), 1000.0f,
    std::make_shared<material::Matte>(glm::vec3(0.15f, 0.15f, 0.15f))));

  m_lightSources.emplace_back(std::make_unique<DirectionalLight>(
    glm::normalize(glm::vec3(-0.4f, -1.0f, 0.6f)),
    glm::vec3(1.0f, 1.0f, 1.0f)));

  return true;
}

std::vector<ray_tracing::Hit> GlassSpheres::HitObjects(ray_tracing::Ray const & ray,
                                                       float near, float far) const
{
  return ray_tracing::TraceHitableCollection(m_spheres, ray, near, far);
}

glm::vec3 GlassSpheres::RayTrace(ray_tracing::Ray const & ray, float near, float far)
{
  using namespace std::placeholders;

  auto const hits = HitObjects(ray, near, far);
  if (hits.empty())
    return glm::vec3(0.1f, 0.1f, 0.1f);

  auto const diffuseColor = RayTraceObjects(ray, hits[0], 0.001f, far, 1);

  glm::vec3 specularColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    specularColor += source->GetSpecular(ray, hits[0]);
  if (!m_lightSources.empty())
    specularColor /= m_lightSources.size();

  return diffuseColor + specularColor;
}

glm::vec3 GlassSpheres::RayTraceObjects(ray_tracing::Ray const & ray, ray_tracing::Hit const & hit,
                                        float near, float far, int depth)
{
  using namespace std::placeholders;

  auto const scatterResult = hit.m_material->Scatter(ray, hit);

  glm::vec3 lightColor = glm::vec3(0.0f, 0.0f, 0.0f);
  for (auto const & source : m_lightSources)
    lightColor += source->TraceLight(ray, hit, std::bind(&GlassSpheres::HitObjects, this, _1, _2, _3));
  if (!m_lightSources.empty())
    lightColor /= m_lightSources.size();

  auto c = lightColor * scatterResult.m_attenuation;
  if (depth >= 5 || fabs(scatterResult.m_energyImpact) < kEps)
    return c;

  auto const hits = HitObjects(scatterResult.m_scatteredRay, near, far);
  if (!hits.empty())
  {
    auto const sc = RayTraceObjects(scatterResult.m_scatteredRay, hits[0], near, far, depth + 1);
    c = glm::mix(lightColor * sc, c, scatterResult.m_energyImpact);
  }
  return c;
}
}  // namespace demo
