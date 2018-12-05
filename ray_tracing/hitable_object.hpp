#pragma once

#include "ray.hpp"

#include <glm/vec3.hpp>

#include <vector>

namespace ray_tracing
{
struct Hit
{
  float m_parameterT = 0.0f;
  glm::vec3 m_position;
  glm::vec3 m_normal;

  Hit() = default;
  Hit(float t, glm::vec3 const & position, glm::vec3 const & normal)
    : m_parameterT(t)
    , m_position(position)
    , m_normal(normal)
  {}
};

class HitableObject
{
public:
  HitableObject() = default;
  virtual ~HitableObject() = default;

  // Returning vector must be sorted by m_parameterT.
  virtual std::vector<Hit> Trace(Ray const & ray, float tmin, float tmax) const = 0;
};

template <typename T>
std::vector<Hit> TraceHitableCollection(T const & collection, Ray const & ray,
                                        float tmin, float tmax)
{
  std::vector<Hit> result;
  result.reserve(collection.size() * 2);
  for (auto const & c : collection)
  {
    auto hits = c->Trace(ray, tmin, tmax);
    result.insert(result.end(), hits.begin(), hits.end());
  }
  std::sort(result.begin(), result.end(), [](Hit const & h1, Hit const & h2)
  {
    return h1.m_parameterT < h2.m_parameterT;
  });
  return result;
}
}  // namespace ray_tracing
