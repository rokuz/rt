#include "gpuprogram.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <list>

namespace rendering
{
namespace
{
int GetOGLShaderType(ShaderType type)
{
  switch (type)
  {
  case rendering::VertexShader:
    return GL_VERTEX_SHADER;

  case rendering::FragmentShader:
    return GL_FRAGMENT_SHADER;

  default:
    return -1;
  }
  return -1;
}

std::string ShaderTypeToString(ShaderType type)
{
  switch (type)
  {
  case rendering::VertexShader:
    return "Vertex shader";

  case rendering::FragmentShader:
    return "Fragment shader";

  default:
    return "";
  }
  return "";
}
}

GpuProgram::GpuProgram() : m_program(0), m_freeTextureSlot(0)
{
  for (size_t i = 0; i < kMaxUniforms; i++)
    m_uniforms[i] = -1;

  m_shaders.resize(ShadersCount);
}

GpuProgram::~GpuProgram() { Destroy(); }
void GpuProgram::AddShader(ShaderType shaderType, std::string const & shaderCode)
{
  m_shaders[shaderType] = shaderCode;
}

bool GpuProgram::Init()
{
  Destroy();
  m_program = glCreateProgram();

  std::list<uint32_t> compiledShaders;
  for (size_t shaderIndex = 0; shaderIndex < ShadersCount; shaderIndex++)
  {
    if (m_shaders[shaderIndex].empty())
      continue;

    uint32_t shader;
    if (!CompileShader(&shader, GetOGLShaderType(static_cast<ShaderType>(shaderIndex)),
                       m_shaders[shaderIndex]))
    {
      Destroy();
      std::cout << "Error: failed to compile "
                << ShaderTypeToString(static_cast<ShaderType>(shaderIndex)) << std::endl;
      return false;
    }

    glAttachShader(m_program, shader);
    compiledShaders.push_back(shader);
  }

  if (compiledShaders.empty())
  {
    Destroy();
    return false;
  }

  if (!LinkProgram(m_program))
  {
    std::cout << "Error: failed to link program" << std::endl;
    for (auto it = compiledShaders.begin(); it != compiledShaders.end(); ++it)
      glDeleteShader(*it);
    Destroy();
    return false;
  }

  for (auto it = compiledShaders.begin(); it != compiledShaders.end(); ++it)
  {
    glDetachShader(m_program, *it);
    glDeleteShader(*it);
  }

  return true;
}

bool GpuProgram::IsValid() const { return m_program != 0; }
void GpuProgram::Destroy()
{
  if (m_program)
  {
    glDeleteProgram(m_program);
    m_program = 0;
  }

  m_attributes.clear();
}

bool GpuProgram::CompileShader(uint32_t * shader, GLenum type, std::string const & shaderCode)
{
  char const * source = shaderCode.c_str();

  *shader = glCreateShader(type);
  glShaderSource(*shader, 1, &source, NULL);
  glCompileShader(*shader);

  GLint logLength;
  glGetShaderiv(*shader, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0)
  {
    char * log = new char[logLength];
    glGetShaderInfoLog(*shader, logLength, &logLength, log);
    std::cout << "Shader compilation log:\n" << log << std::endl;
    delete[] log;
  }

  int status = 0;
  glGetShaderiv(*shader, GL_COMPILE_STATUS, &status);
  if (status == 0)
  {
    glDeleteShader(*shader);
    return false;
  }

  return true;
}

bool GpuProgram::LinkProgram(uint32_t prog)
{
  glLinkProgram(prog);

  int logLength;
  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0)
  {
    char * log = new char[logLength];
    glGetProgramInfoLog(prog, logLength, &logLength, log);
    std::cout << "Shader linkage log:\n" << log << std::endl;
    delete[] log;
  }

  int status = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &status);
  if (status == 0)
    return false;

  int attributesCount = 0;
  glGetProgramiv(prog, GL_ACTIVE_ATTRIBUTES, &attributesCount);
  char uname[kMaxUniforms] = "";
  for (int i = 0; i < attributesCount; i++)
  {
    GLsizei unamesize = 0;
    GLint size = 0;
    GLenum type;
    glGetActiveAttrib(prog, i, sizeof(uname), &unamesize, &size, &type, uname);
    if (unamesize > 0)
    {
      int const location = glGetAttribLocation(prog, uname);
      m_attributes.insert(std::make_pair(location, std::string(uname)));
    }
  }

  if (m_attributes.empty())
    return true;

  bool failed = false;
  std::vector<int> indices;
  indices.reserve(m_attributes.size());
  for (auto it = m_attributes.begin(); it != m_attributes.end(); ++it)
  {
    int index = -1;
    for (int i = 0; i < (int)kAttributesCount; i++)
    {
      if (kAttributesNames[i] == it->second)
      {
        index = i;
        break;
      }
    }
    if (index < 0)
    {
      std::cout << "Gpu program contains unknown attribute " << it->second.c_str() << std::endl;
      failed = true;
    }
    else
    {
      indices.push_back(index);
    }
  }

  if (!failed && !std::is_sorted(indices.begin(), indices.end()))
  {
    std::cout << "Gpu program contains attributes in incorrect order" << std::endl;
    failed = true;
  }

  return !failed;
}

bool GpuProgram::Use()
{
  if (!IsValid())
    return false;

  glUseProgram(m_program);
  m_freeTextureSlot = 0;

  return true;
}

void GpuProgram::SetTextureInternal(int uniformIndex, std::shared_ptr<Texture> texture, int slot)
{
  int slotIndex = slot < 0 ? m_freeTextureSlot : slot;
  if (slotIndex >= kMaxBoundTextures)
    return;

  glActiveTexture(GL_TEXTURE0 + slotIndex);
  texture->Bind();
  glUniform1i(uniformIndex, slotIndex);

  if (slot < 0)
    m_freeTextureSlot++;
}

std::unique_ptr<GpuProgram> BuildQuadGpuProgram()
{
  static char const * kVertexShaderCode = {
      "#version 410 core\n"
      "layout(location = 0) in vec3 aPosition;\n"
      "layout(location = 1) in vec2 aUV0;\n"
      "out vec2 vUV0;\n"
      "void main()\n"
      "{\n"
      "  gl_Position = vec4(aPosition, 1.0);\n"
      "  vUV0 = aUV0;\n"
      "}"};
  static char const * kFragmentShaderCode = {
      "#version 410 core\n"
      "in vec2 vUV0\n;"
      "out vec4 oColor;\n"
      "uniform sampler2D uTexture;\n"
      "void main()\n"
      "{\n"
      "  oColor = texture(uTexture, vUV0);\n"
      "}"};
  auto program = std::unique_ptr<GpuProgram>(new GpuProgram());
  program->AddShader(ShaderType::VertexShader, kVertexShaderCode);
  program->AddShader(ShaderType::FragmentShader, kFragmentShaderCode);
  if (!program->Init())
    return nullptr;
  program->BindUniform<QuadUniforms>(Quad::uTexture, "uTexture");
  return program;
}
}  // namespace rendering
