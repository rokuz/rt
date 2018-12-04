#include "mesh.hpp"

#include <OpenGL/gl3.h>

#include <cassert>
#include <functional>

namespace rendering
{
namespace
{
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

void ForEachAttribute(uint32_t componentsMask,
                      std::function<void(MeshVertexAttribute)> const & func)
{
  for (uint32_t i = 0; i < kAttributesCount; i++)
  {
    if ((componentsMask & kAllAttributes[i]) != 0)
      func(kAllAttributes[i]);
  }
}

void ForEachAttribute(uint32_t componentsMask,
                      std::function<bool(MeshVertexAttribute)> const & func)
{
  for (uint32_t i = 0; i < kAttributesCount; i++)
  {
    if ((componentsMask & kAllAttributes[i]) != 0)
      if (!func(kAllAttributes[i]))
        return;
  }
}

uint32_t GetAttributeElements(MeshVertexAttribute attr)
{
  switch (attr)
  {
  case Position:
    return 3;
  case UV0:
    return 2;
  default:
    assert(0 && "Unknown attribute");
  }
  return 0;
}

uint32_t GetAttributeSize(MeshVertexAttribute attr)
{
  return GetAttributeElements(attr) * sizeof(float);
}

uint32_t GetVertexSize(uint32_t componentsMask)
{
  uint32_t size = 0;
  ForEachAttribute(componentsMask,
                   [&size](MeshVertexAttribute attr) { size += GetAttributeSize(attr); });
  return size;
}

uint32_t GetAttributeOffset(uint32_t componentsMask, MeshVertexAttribute attr)
{
  uint32_t offset = 0;
  ForEachAttribute(
      componentsMask,
      (std::function<bool(MeshVertexAttribute)>)[&offset, attr ](MeshVertexAttribute a)->bool {
        if (attr == a)
          return false;
        offset += GetAttributeSize(a);
        return true;
      });
  return offset;
}

void CreateVertexArray(uint32_t componentsMask)
{
  const uint32_t vertexSize = GetVertexSize(componentsMask);
  uint32_t offset = 0;
  uint32_t index = 0;
  ForEachAttribute(componentsMask, [&offset, &index, vertexSize](MeshVertexAttribute attr) {
    glVertexAttribPointer(index, GetAttributeElements(attr), GL_FLOAT, GL_FALSE, vertexSize,
                          BUFFER_OFFSET(offset));
    glEnableVertexAttribArray(index);
    offset += GetAttributeSize(attr);
    index++;
  });
}
}  // namespace

VertexArray::VertexArray(uint32_t componentsMask)
{
  glGenVertexArrays(1, &m_vertexArray);
  glBindVertexArray(m_vertexArray);
  CreateVertexArray(componentsMask);
  glBindVertexArray(0);
}

VertexArray::~VertexArray()
{
  if (m_vertexArray != 0)
  {
    glDeleteVertexArrays(1, &m_vertexArray);
    m_vertexArray = 0;
  }
}

void VertexArray::Bind() { glBindVertexArray(m_vertexArray); }

Mesh::~Mesh() { Destroy(); }

void Mesh::Init(uint32_t componentsMask, uint32_t verticesCount, void const * data)
{
  m_verticesCount = verticesCount;
  glGenBuffers(1, &m_vertexBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
  m_vertexArray = std::make_unique<VertexArray>(componentsMask);
  uint32_t const vertexSizeInBytes = GetVertexSize(componentsMask);
  glBufferData(GL_ARRAY_BUFFER, vertexSizeInBytes * verticesCount, data, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Mesh::Render()
{
  m_vertexArray->Bind();
  glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
  glDrawArrays(GL_TRIANGLES, 0, m_verticesCount);
}

void Mesh::Destroy()
{
  if (m_vertexBuffer != 0)
  {
    glDeleteBuffers(1, &m_vertexBuffer);
    m_vertexBuffer = 0;
  }
  m_vertexArray.reset();
}

std::unique_ptr<Mesh> BuildQuad()
{
  static float vertices[] = {
    -1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
  };
  auto mesh = std::make_unique<Mesh>();
  mesh->Init(MeshVertexAttribute::Position | MeshVertexAttribute::UV0, 6, vertices);
  return mesh;
}
}  // namespace rendering
