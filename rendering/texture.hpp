#pragma once

#include <cstdint>

namespace rendering
{
class Texture
{
public:
  Texture();
  ~Texture();

  void Init(int format, uint8_t const * buffer, uint32_t bufferSize, uint32_t width,
            uint32_t height, int pixelFormat = -1);
  void Update(uint8_t const * buffer, uint32_t bufferSize);
  void Bind();

  uint32_t GetWidth() const { return m_width; }
  uint32_t GetHeight() const { return m_height; }
  int GetFormat() const { return m_format; }
  int GetPixelFormat() const { return m_pixelFormat; }
private:
  uint32_t m_texture;
  uint32_t m_pbo;
  uint32_t m_target;
  uint32_t m_width;
  uint32_t m_height;
  int m_format;
  int m_pixelFormat;

  void SetSampling();
  void Destroy();
};
}  // namespace rendering
