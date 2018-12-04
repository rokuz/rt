#pragma once

#include <string>

namespace rendering
{
enum MeshVertexAttribute
{
  Position = 1 << 0,
  UV0 = 1 << 1
};

std::string const kAttributesNames[] = {"aPosition", "aUV0"};

MeshVertexAttribute const kAllAttributes[] = {Position, UV0};

uint32_t constexpr kAttributesCount = sizeof(kAllAttributes) / sizeof(kAllAttributes[0]);

class VertexArray
{
public:
  VertexArray(uint32_t componentsMask);
  ~VertexArray();
  void Bind();

private:
  uint32_t m_vertexArray;
};

class Mesh
{
public:
  Mesh() = default;
  ~Mesh();
  void Init(uint32_t componentsMask, uint32_t verticesCount, void const * data = nullptr);
  void Render();

private:
  void Destroy();

  std::unique_ptr<VertexArray> m_vertexArray;
  uint32_t m_vertexBuffer = 0;
  uint32_t m_verticesCount = 0;
};

extern std::unique_ptr<Mesh> BuildQuad();
}  // namespace rendering
