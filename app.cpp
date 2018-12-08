#include "app.hpp"

#include "rendering/pipelinestate.hpp"

#include <glm/geometric.hpp>

#include <glfw/glfw3.h>

#include <algorithm>

bool App::Initialize(std::unique_ptr<ray_tracing::Frame> && frame,
                     uint32_t screenWidth, uint32_t screenHeight,
                     uint32_t samplesInRowCount)
{
  m_frame = std::move(frame);
  m_originalSamplesCount = samplesInRowCount;

  m_quad = rendering::BuildQuad();
  m_quadGpuProgram = rendering::BuildQuadGpuProgram();
  if (m_quadGpuProgram == nullptr)
    return false;

  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;
  m_buffer = std::make_shared<ray_tracing::ColorBuffer>(m_screenWidth * m_screenHeight);
  m_realtimeBuffer.resize(m_screenWidth * m_screenHeight);
  m_byteBuffer = std::make_shared<std::vector<uint8_t>>(m_screenWidth * m_screenHeight * 4, 0);

  if (m_frame)
    m_frame->Initialize(m_buffer, m_screenWidth, m_screenHeight, samplesInRowCount);

  m_texture = std::make_shared<rendering::Texture>();
  m_texture->Init(GL_RGBA8, m_byteBuffer->data(), static_cast<uint32_t>(m_byteBuffer->size()),
                  screenWidth, screenHeight);

  glViewport(0, 0, screenWidth, screenHeight);
  rendering::DepthState(false /* enabled */, false /* writeEnable */).Apply();
  rendering::PipelineState(GL_CULL_FACE, true).Apply();

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDrawBuffer(GL_BACK);

  CheckOpenGLErrors();

  return true;
}

void App::Uninitialize()
{
  if (m_frame)
    m_frame->Uninitialize();

  assert(m_texture.use_count() == 1);
  m_texture.reset();
  m_quad.reset();
  m_quadGpuProgram.reset();
}

void App::Render(double timeSinceStart, double elapsedTime)
{
  if (m_frame)
  {
    // Update realtime buffer to show progress.
    float constexpr kUpdatePeriodSec = 0.25;
    m_lastUpdateRealtimeBuffer += elapsedTime;
    if (m_frame->InProgress() && m_lastUpdateRealtimeBuffer >= kUpdatePeriodSec)
    {
      m_lastUpdateRealtimeBuffer -= kUpdatePeriodSec;
      m_frame->CopyToBuffer(m_realtimeBuffer);
      UpdateTexture(m_realtimeBuffer);
    }

    if (m_frame->HasFinished())
    {
      UpdateTexture(*m_buffer);
      m_frame->SetSamplesInRowCount(m_originalSamplesCount);
    }
  }

  if (m_quadGpuProgram->Use())
  {
    m_quadGpuProgram->SetTexture<rendering::QuadUniforms>(Quad::uTexture, m_texture);
    m_quad->Render();
  }
  CheckOpenGLErrors();
}

void App::RayTrace(bool highQuality)
{
  if (!m_frame || !m_frame->HasFinished())
    return;

  m_originalSamplesCount = m_frame->GetSamplesInRowCount();
  if (highQuality)
    m_frame->SetSamplesInRowCount(std::max(m_originalSamplesCount, static_cast<uint32_t>(10)));

  for (auto & p : *m_buffer)
    p = glm::vec3(0.0, 0.0, 0.0);

  UpdateTexture(*m_buffer);
  m_lastUpdateRealtimeBuffer = 0.0;

  m_frame->TraceAllRays();
}

void App::OnKeyButton(int key, int scancode, bool pressed)
{
  switch (key)
  {
  case GLFW_KEY_T:
    RayTrace(false /* highQuality */);
    break;
  case GLFW_KEY_Q:
    RayTrace(true /* highQuality */);
    break;
  }
}

void App::OnMouseButton(double xpos, double ypos, int button, bool pressed) {}

void App::OnMouseMove(double xpos, double ypos) {}

void App::UpdateTexture(std::vector<glm::vec3> const & buffer)
{
  for (size_t i = 0; i < buffer.size(); ++i)
  {
    (*m_byteBuffer)[i * 4 + 0] = static_cast<uint8_t>(255 * glm::clamp(buffer[i].r, 0.0f, 1.0f));
    (*m_byteBuffer)[i * 4 + 1] = static_cast<uint8_t>(255 * glm::clamp(buffer[i].g, 0.0f, 1.0f));
    (*m_byteBuffer)[i * 4 + 2] = static_cast<uint8_t>(255 * glm::clamp(buffer[i].b, 0.0f, 1.0f));
    (*m_byteBuffer)[i * 4 + 3] = 255;
  }
  m_texture->Update(m_byteBuffer->data(), static_cast<uint32_t>(m_byteBuffer->size()));
}

void App::CheckOpenGLErrors()
{
  GLenum err(glGetError());

  while (err != GL_NO_ERROR)
  {
    std::string error;

    switch (err)
    {
    case GL_INVALID_OPERATION:
      error = "GL_INVALID_OPERATION";
      break;
    case GL_INVALID_ENUM:
      error = "GL_INVALID_ENUM";
      break;
    case GL_INVALID_VALUE:
      error = "GL_INVALID_VALUE";
      break;
    case GL_OUT_OF_MEMORY:
      error = "GL_OUT_OF_MEMORY";
      break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      error = "GL_INVALID_FRAMEBUFFER_OPERATION";
      break;
    default:
      error = "Unknown";
    }

    std::cout << "OpenGL error: " << error << std::endl;
    err = glGetError();
  }
}
