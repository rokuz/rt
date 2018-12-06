#include "texture.hpp"

#include "global.hpp"

#include <algorithm>
#include <cmath>

namespace rendering
{
namespace
{
int FindPixelFormat(int textureFormat)
{
  switch (textureFormat)
  {
  case GL_R8:
    return GL_RED;
  case GL_R8_SNORM:
    return GL_RED;
  case GL_R16:
    return GL_RED;
  case GL_R16_SNORM:
    return GL_RED;
  case GL_RG8:
    return GL_RG;
  case GL_RG8_SNORM:
    return GL_RG;
  case GL_RG16:
    return GL_RG;
  case GL_RG16_SNORM:
    return GL_RG;
  case GL_R3_G3_B2:
    return GL_RGB;
  case GL_RGB4:
    return GL_RGB;
  case GL_RGB5:
    return GL_RGB;
  case GL_RGB8:
    return GL_RGB;
  case GL_RGB8_SNORM:
    return GL_RGB;
  case GL_RGB10:
    return GL_RGB;
  case GL_RGB12:
    return GL_RGB;
  case GL_RGB16_SNORM:
    return GL_RGB;
  case GL_RGBA2:
    return GL_RGB;
  case GL_RGBA4:
    return GL_RGB;
  case GL_RGB5_A1:
    return GL_RGBA;
  case GL_RGBA8:
    return GL_RGBA;
  case GL_RGBA8_SNORM:
    return GL_RGBA;
  case GL_RGB10_A2:
    return GL_RGBA;
  case GL_RGB10_A2UI:
    return GL_RGBA;
  case GL_RGBA12:
    return GL_RGBA;
  case GL_RGBA16:
    return GL_RGBA;
  case GL_SRGB8:
    return GL_RGB;
  case GL_SRGB8_ALPHA8:
    return GL_RGBA;
  case GL_R16F:
    return GL_RED;
  case GL_RG16F:
    return GL_RG;
  case GL_RGB16F:
    return GL_RGB;
  case GL_RGBA16F:
    return GL_RGBA;
  case GL_R32F:
    return GL_RED;
  case GL_RG32F:
    return GL_RG;
  case GL_RGB32F:
    return GL_RGB;
  case GL_RGBA32F:
    return GL_RGBA;
  case GL_R11F_G11F_B10F:
    return GL_RGB;
  case GL_RGB9_E5:
    return GL_RGB;
  case GL_R8I:
    return GL_RED;
  case GL_R8UI:
    return GL_RED;
  case GL_R16I:
    return GL_RED;
  case GL_R16UI:
    return GL_RED;
  case GL_R32I:
    return GL_RED;
  case GL_R32UI:
    return GL_RED;
  case GL_RG8I:
    return GL_RG;
  case GL_RG8UI:
    return GL_RG;
  case GL_RG16I:
    return GL_RG;
  case GL_RG16UI:
    return GL_RG;
  case GL_RG32I:
    return GL_RG;
  case GL_RG32UI:
    return GL_RG;
  case GL_RGB8I:
    return GL_RGB;
  case GL_RGB8UI:
    return GL_RGB;
  case GL_RGB16I:
    return GL_RGB;
  case GL_RGB16UI:
    return GL_RGB;
  case GL_RGB32I:
    return GL_RGB;
  case GL_RGB32UI:
    return GL_RGB;
  case GL_RGBA8I:
    return GL_RGBA;
  case GL_RGBA8UI:
    return GL_RGBA;
  case GL_RGBA16I:
    return GL_RGBA;
  case GL_RGBA16UI:
    return GL_RGBA;
  case GL_RGBA32I:
    return GL_RGBA;
  case GL_RGBA32UI:
    return GL_RGBA;
  }
  return -1;
}
}

Texture::Texture()
  : m_texture(0), m_pbo(0), m_target(0), m_width(0), m_height(0), m_format(-1), m_pixelFormat(-1)
{
}

Texture::~Texture() { Destroy(); }
void Texture::Init(int format, uint8_t const * buffer, uint32_t bufferSize, uint32_t width,
                   uint32_t height, int pixelFormat)
{
  Destroy();

  m_target = GL_TEXTURE_2D;
  m_format = format;
  m_pixelFormat = pixelFormat < 0 ? FindPixelFormat(format) : pixelFormat;
  m_width = width;
  m_height = height;
  glGenTextures(1, &m_texture);
  glBindTexture(m_target, m_texture);
  glGenBuffers(1, &m_pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, buffer, GL_STREAM_DRAW);
  glTexStorage2D(m_target, 1, m_format, m_width, m_height);
  glTexSubImage2D(m_target, 0, 0, 0, m_width, m_height, m_pixelFormat, GL_UNSIGNED_BYTE, 0);
  SetSampling();
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(m_target, 0);
}

void Texture::Update(uint8_t const * buffer, uint32_t bufferSize)
{
  glBindTexture(m_target, m_texture);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
  auto ptr = static_cast<uint8_t *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
  if (ptr != nullptr)
  {
    memcpy(ptr, buffer, bufferSize);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
  }
  glTexSubImage2D(m_target, 0, 0, 0, m_width, m_height, m_pixelFormat, GL_UNSIGNED_BYTE, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(m_target, 0);
}

void Texture::SetSampling()
{
  if (m_target != 0 && m_texture != 0)
  {
    if (m_target == GL_TEXTURE_CUBE_MAP)
    {
      glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(m_target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
    else if (m_target == GL_TEXTURE_2D_ARRAY)
    {
      glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(m_target, GL_TEXTURE_WRAP_R, GL_REPEAT);
    }
    else
    {
      glTexParameteri(m_target, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(m_target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    glTexParameteri(m_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(m_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  }
}

void Texture::Destroy()
{
  if (m_pbo != 0)
    glDeleteBuffers(1, &m_pbo);
  if (m_texture != 0)
    glDeleteTextures(1, &m_texture);
}

void Texture::Bind() { glBindTexture(m_target, m_texture); }
}  // namespace rendering
