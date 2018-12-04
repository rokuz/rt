#include "app.hpp"

#include "rendering/pipelinestate.hpp"

bool App::Initialize(std::unique_ptr<ray_tracing::Frame> && frame,
                     uint32_t screenWidth, uint32_t screenHeight)
{
  m_frame = std::move(frame);

  m_quad = rendering::BuildQuad();
  m_quadGpuProgram = rendering::BuildQuadGpuProgram();
  if (m_quadGpuProgram == nullptr)
    return false;

  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;
  m_buffer = std::make_shared<std::vector<glm::vec3>>(m_screenWidth * m_screenHeight);
  m_byteBuffer = std::make_shared<std::vector<uint8_t>>(m_screenWidth * m_screenHeight * 4, 0);

  if (m_frame)
    m_frame->Initialize(m_buffer, m_screenWidth, m_screenHeight);

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
  assert(m_texture.use_count() == 1);
  m_texture.reset();
  m_quad.reset();
  m_quadGpuProgram.reset();
}

void App::Render(double timeSinceStart, double elapsedTime, bool trace)
{
  if (trace)
  {
    for (auto & p : *m_buffer)
      p = glm::vec3(0.0, 0.0, 0.0);

    if (m_frame)
      m_frame->Trace();

    for (size_t i = 0; i < m_buffer->size(); ++i)
    {
      (*m_byteBuffer)[i * 4 + 0] = static_cast<uint8_t>(255 * (*m_buffer)[i].r);
      (*m_byteBuffer)[i * 4 + 1] = static_cast<uint8_t>(255 * (*m_buffer)[i].g);
      (*m_byteBuffer)[i * 4 + 2] = static_cast<uint8_t>(255 * (*m_buffer)[i].b);
      (*m_byteBuffer)[i * 4 + 3] = 255;
    }
    m_texture->Update(m_byteBuffer->data(), static_cast<uint32_t>(m_byteBuffer->size()));
  }

  if (m_quadGpuProgram->Use())
  {
    m_quadGpuProgram->SetTexture<rendering::QuadUniforms>(Quad::uTexture, m_texture);
    m_quad->Render();
  }
  CheckOpenGLErrors();
}

void App::OnKeyButton(int key, int scancode, bool pressed) {}

void App::OnMouseButton(double xpos, double ypos, int button, bool pressed) {}

void App::OnMouseMove(double xpos, double ypos) {}

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
