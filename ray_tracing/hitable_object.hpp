#pragma once

#include "ray.hpp"

#include <glm/vec3.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace ray_tracing
{
class Material;

struct Hit
{
  float m_parameterT = 0.0f;
  glm::vec3 m_position = {};
  glm::vec3 m_normal = {};
  std::shared_ptr<Material> m_material;

  Hit() = default;
  Hit(float t, glm::vec3 const & position, glm::vec3 const & normal,
      std::shared_ptr<Material> material)
    : m_parameterT(t)
    , m_position(position)
    , m_normal(normal)
    , m_material(std::move(material))
  {}
};

class HitableObject
{
public:
  explicit HitableObject(std::shared_ptr<Material> material)
    : m_material(std::move(material))
  {}

  virtual ~HitableObject() = default;

  virtual uint32_t GetType() const = 0;

  // Returning vector must be sorted by m_parameterT.
  virtual std::vector<Hit> Trace(Ray const & ray, float tmin, float tmax) const = 0;
  virtual std::optional<Hit> TraceNearest(Ray const & ray, float tmin, float tmax) const = 0;

protected:
  std::shared_ptr<Material> m_material;
};

template <typename T>
std::vector<Hit> TraceInHitableCollection(T const & collection, Ray const & ray,
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

template <typename T>
std::optional<Hit> TraceNearestInHitableCollection(T const & collection,
                                                   Ray const & ray,
                                                   float tmin, float tmax)
{
  std::optional<Hit> nearest;
  for (auto const & c : collection)
  {
    auto n = c->TraceNearest(ray, tmin, tmax);
    if (!n)
      continue;

    if (!nearest || n->m_parameterT < nearest->m_parameterT)
      nearest = n;
  }
  return nearest;
}
}  // namespace ray_tracing
