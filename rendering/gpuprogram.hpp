#pragma once

#include "texture.hpp"

#include <OpenGL/gl3.h>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace rendering
{
int const kMaxUniforms = 256;
int const kMaxBoundTextures = 32;

template <typename UniformType>
class UniformBase
{
public:
  enum Uniform
  {
    uFake = 0
  };
};

#define DECLARE_UNIFORMS_BEGIN(UniformType) class UniformType{}; \
template<> class rendering::UniformBase<UniformType> { public: enum Uniform {
#define DECLARE_UNIFORMS_END() }; }; 

enum ShaderType
{
  VertexShader = 0,
  FragmentShader,

  ShadersCount
};

class GpuProgram
{
public:
  GpuProgram();
  ~GpuProgram();

  void AddShader(ShaderType shaderType, std::string const & shaderCode);
  bool Init();
  bool IsValid() const;

  template <typename UniformType>
  void BindUniform(typename UniformBase<UniformType>::Uniform uniform, std::string const & name);

  template <typename UniformType>
  int GetUniform(typename UniformBase<UniformType>::Uniform uniform);

  template <typename UniformType>
  void SetFloat(typename UniformBase<UniformType>::Uniform uniform, float v);

  template <typename UniformType>
  void SetUint(typename UniformBase<UniformType>::Uniform uniform, unsigned int v);

  template <typename UniformType>
  void SetInt(typename UniformBase<UniformType>::Uniform uniform, int v);

  template <typename UniformType>
  void SetIntArray(typename UniformBase<UniformType>::Uniform uniform, int const * v, int count);

  template <typename UniformType>
  void SetVector(typename UniformBase<UniformType>::Uniform uniform, glm::vec2 const & vec);

  template <typename UniformType>
  void SetVector(typename UniformBase<UniformType>::Uniform uniform, glm::vec3 const & vec);

  template <typename UniformType>
  void SetVector(typename UniformBase<UniformType>::Uniform uniform, glm::vec4 const & vec);

  template <typename UniformType>
  void SetMatrix(typename UniformBase<UniformType>::Uniform uniform, glm::mat4x4 const & mat);

  template <typename UniformType>
  void SetMatrixArray(typename UniformBase<UniformType>::Uniform uniform, glm::mat4x4 const * mat,
                      int count);

  template <typename UniformType>
  void SetTexture(typename UniformBase<UniformType>::Uniform uniform,
                  std::shared_ptr<Texture> texture, int slot = -1);

  bool Use();

private:
  uint32_t m_program;
  int m_uniforms[kMaxUniforms];
  std::vector<std::string> m_shaders;
  int m_freeTextureSlot;
  std::map<int, std::string> m_attributes;

  bool CompileShader(uint32_t * shader, GLenum type, std::string const & fileName);
  bool LinkProgram(uint32_t prog);

  void SetTextureInternal(int uniformIndex, std::shared_ptr<Texture> texture, int slot);

  void Destroy();
};

template <typename UniformType>
void GpuProgram::BindUniform(typename UniformBase<UniformType>::Uniform uniform,
                             std::string const & name)
{
  if (!IsValid())
    return;
  if (uniform >= kMaxUniforms)
    return;

  m_uniforms[uniform] = glGetUniformLocation(m_program, name.c_str());
  if (m_uniforms[uniform] < 0)
  {
    std::cout << "Error: Uniform '" << name << "' has not been found to bind." << std::endl;
  }
}

template <typename UniformType>
int GpuProgram::GetUniform(typename UniformBase<UniformType>::Uniform uniform)
{
  return m_uniforms[uniform];
}

template <typename UniformType>
void GpuProgram::SetFloat(typename UniformBase<UniformType>::Uniform uniform, float v)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform1fv(uf, 1, &v);
}

template <typename UniformType>
void GpuProgram::SetUint(typename UniformBase<UniformType>::Uniform uniform, unsigned int v)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform1uiv(uf, 1, &v);
}

template <typename UniformType>
void GpuProgram::SetInt(typename UniformBase<UniformType>::Uniform uniform, int v)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform1iv(uf, 1, &v);
}

template <typename UniformType>
void GpuProgram::SetIntArray(typename UniformBase<UniformType>::Uniform uniform, int const * v,
                             int count)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform1iv(uf, count, v);
}

template <typename UniformType>
void GpuProgram::SetVector(typename UniformBase<UniformType>::Uniform uniform,
                           glm::vec2 const & vec)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform2fv(uf, 1, reinterpret_cast<float const *>(&vec));
}

template <typename UniformType>
void GpuProgram::SetVector(typename UniformBase<UniformType>::Uniform uniform,
                           glm::vec3 const & vec)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform3fv(uf, 1, reinterpret_cast<float const *>(&vec));
}

template <typename UniformType>
void GpuProgram::SetVector(typename UniformBase<UniformType>::Uniform uniform,
                           glm::vec4 const & vec)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniform4fv(uf, 1, reinterpret_cast<float const *>(&vec));
}

template <typename UniformType>
void GpuProgram::SetMatrix(typename UniformBase<UniformType>::Uniform uniform,
                           glm::mat4x4 const & mat)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniformMatrix4fv(uf, 1, false, reinterpret_cast<float const *>(&mat));
}

template <typename UniformType>
void GpuProgram::SetMatrixArray(typename UniformBase<UniformType>::Uniform uniform,
                                glm::mat4x4 const * mat, int count)
{
  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  glUniformMatrix4fv(uf, count, false, reinterpret_cast<const float *>(mat));
}

template <typename UniformType>
void GpuProgram::SetTexture(typename UniformBase<UniformType>::Uniform uniform,
                            std::shared_ptr<Texture> texture, int slot)
{
  if (!texture)
    return;

  int uf = GetUniform<UniformType>(uniform);
  if (uf < 0)
    return;
  SetTextureInternal(uf, std::move(texture), slot);
}

DECLARE_UNIFORMS_BEGIN(QuadUniforms)
  uTexture,
DECLARE_UNIFORMS_END()
#define Quad rendering::UniformBase<rendering::QuadUniforms>::Uniform

extern std::unique_ptr<GpuProgram> BuildQuadGpuProgram();
}  // namespace rendering
