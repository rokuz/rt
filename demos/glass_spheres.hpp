#pragma once

#include "demo.hpp"
#include "palette.hpp"

#include "ray_tracing/default_materials.hpp"
#include "ray_tracing/directional_light.hpp"
#include "ray_tracing/frame.hpp"
#include "ray_tracing/sphere.hpp"

#include <cstdint>
#include <memory>
#include <set>

namespace demo
{
class GlassSpheres : public Demo
{
public:
  explicit GlassSpheres(std::unique_ptr<ray_tracing::Frame> && frame)
    : m_frame(std::move(frame))
  {}

  bool Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                  uint32_t width, uint32_t height,
                  uint32_t samplesInRowCount) override
  {
    if (!m_frame || !m_frame->Initialize(buffer, width, height, samplesInRowCount))
      return false;

    using namespace ray_tracing;

    m_frame->SetBackgroundColor(glm::vec3(0.3f, 0.3f, 0.3f));

    m_frame->SetCameraPosition(glm::vec3(0.0f, 2.0f, -10.0f));
    m_frame->SetCameraDirection(glm::vec3(0.0f, -1.0f, 3.0f));

    std::random_device random;
    std::mt19937 generator(random());
    std::uniform_real_distribution<float> randomFloat(0.0f, 1.0f);

    std::set<std::pair<int, int>> s;
    std::uniform_int_distribution<> randomPos(-5, 5);
    auto generatePosition = [this, &s, &randomPos, &generator]()
    {
      std::pair<int, int> p;
      int i = 0;
      do
      {
        p = std::make_pair(randomPos(generator), randomPos(generator));
        ++i;
      }
      while (s.find(p) != s.end() && i < 5000);
      s.insert(p);
      return p;
    };

    for (size_t i = 0; i < 40; ++i)
    {
      auto const p = generatePosition();
      auto const c = Palette::RandomFromAll(generator);

      auto mat = std::make_shared<material::Glass>(c, glm::mix(0.7f, 0.9f, randomFloat(generator)));
      float const radius = glm::mix(0.25f, 0.5f, randomFloat(generator));
      m_frame->AddObject(std::make_unique<Sphere>(glm::vec3(p.first, radius - 1.0f, p.second), radius, mat));
    }
    m_frame->AddObject(std::make_unique<Sphere>(
      glm::vec3(0.0, -1001.0f, 0.0), 1000.0f,
      std::make_shared<material::Matte>(glm::vec3(0.1f, 0.1f, 0.1f))));

    m_frame->AddLightSource(std::make_unique<DirectionalLight>(
      glm::normalize(glm::vec3(-0.4f, -1.0f, 0.6f)),
      glm::vec3(1.0f, 1.0f, 1.0f)));

    return true;
  }

  std::unique_ptr<ray_tracing::Frame> & GetFrame() override { return m_frame; }

private:
  std::unique_ptr<ray_tracing::Frame> m_frame;
};
}  // namespace demo
