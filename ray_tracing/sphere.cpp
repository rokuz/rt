#include "sphere.hpp"

#include "global.hpp"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace ray_tracing
{
std::vector<Hit> Sphere::Trace(Ray const & ray, float tmin, float tmax) const
{
  auto const d = ray.Origin() - m_center;
  auto const a = glm::dot(ray.Direction(), ray.Direction());
  auto const b = 2.0f * glm::dot(d, ray.Direction());
  auto const c = glm::dot(d, d) - m_radius * m_radius;
  auto const discriminant = b * b - 4 * a * c;
  if (discriminant < 0.0f)
    return {};

  if (fabs(discriminant) < kEps)
  {
    auto const t = -b / (2.0f * a);
    if (t < tmin || t > tmax)
      return {};

    auto const p = ray.PointAt(t);
    auto const n = glm::normalize(p - m_center);
    return {Hit(t, p, n)};
  }

  auto const sqrtD = sqrt(discriminant);
  auto const t1 = (-b - sqrtD) / (2.0f * a);
  auto const t2 = (-b + sqrtD) / (2.0f * a);
  bool const hasT1 = (t1 >= tmin && t1 <= tmax);
  bool const hasT2 = (t2 >= tmin && t2 <= tmax);
  if (!hasT1 && !hasT2)
    return {};

  if (hasT1 && !hasT2)
  {
    auto const p = ray.PointAt(t1);
    auto const n = glm::normalize(p - m_center);
    return {Hit(t1, p, n)};
  }

  if (hasT2 && !hasT1)
  {
    auto const p = ray.PointAt(t2);
    auto const n = glm::normalize(p - m_center);
    return {Hit(t2, p, n)};
  }

  auto const nt1 = std::min(t1, t2);
  auto const nt2 = std::max(t1, t2);
  auto const p1 = ray.PointAt(nt1);
  auto const n1 = glm::normalize(p1 - m_center);
  auto const p2 = ray.PointAt(nt2);
  auto const n2 = glm::normalize(p2 - m_center);
  return {Hit(nt1, p1, n1), Hit(nt2, p2, n2)};
}
}  // namespace ray_tracing
