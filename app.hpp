#pragma once

#include "ray_tracing/frame.hpp"

#include "rendering/gpuprogram.hpp"
#include "rendering/mesh.hpp"
#include "rendering/texture.hpp"

#include <glm/vec3.hpp>

#include <cassert>
#include <cstdint>
#include <memory>

class App
{
public:
  App() = default;

  bool Initialize(std::unique_ptr<ray_tracing::Frame> && frame,
                  uint32_t screenWidth, uint32_t screenHeight,
                  uint32_t rayTracingThreadsCount);
  void Uninitialize();
  void Render(double timeSinceStart, double elapsedTime, bool trace);

  void OnKeyButton(int key, int scancode, bool pressed);
  void OnMouseButton(double xpos, double ypos, int button, bool pressed);
  void OnMouseMove(double xpos, double ypos);

private:
  void UpdateTexture(std::vector<glm::vec3> const & buffer);
  void CheckOpenGLErrors();

  uint32_t m_screenWidth = 0;
  uint32_t m_screenHeight = 0;
  std::shared_ptr<std::vector<glm::vec3>> m_buffer;
  std::vector<glm::vec3> m_secondBuffer;
  std::shared_ptr<std::vector<uint8_t>> m_byteBuffer;
  std::unique_ptr<ray_tracing::Frame> m_frame;

  std::unique_ptr<rendering::Mesh> m_quad;
  std::unique_ptr<rendering::GpuProgram> m_quadGpuProgram;
  std::shared_ptr<rendering::Texture> m_texture;
};
