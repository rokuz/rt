#include "circles.hpp"

namespace ray_tracing
{
namespace
{
bool HitSphere(glm::vec3 const &center, float radius, Ray const &ray)
{
  auto const d = ray.Origin() - center;
  auto const a = glm::dot(ray.Direction(), ray.Direction());
  auto const b = 2.0f * glm::dot(d, ray.Direction());
  auto const c = glm::dot(d, d) - radius * radius;
  float discriminant = b * b - 4 * a * c;
  return discriminant > 0;
}
}  // namespace

bool Circles::Initialize(std::shared_ptr<std::vector<glm::vec3>> buffer,
                         uint32_t width, uint32_t height)
{
  if (!Frame::Initialize(buffer, width, height))
    return false;

  return true;
}

void Circles::Trace(double timeSinceStart, double elapsedTime)
{
  ForEachRay([](Ray ray)
  {
    if (HitSphere(glm::vec3(0, 0, 5.0f), 1.0f, ray))
      return glm::vec3(1.0f, 0.0f, 0.0f);

    return glm::vec3(1.0f, 1.0f, 1.0f);
  });
}
}  // namespace ray_tracing