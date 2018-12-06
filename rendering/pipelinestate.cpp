#include "pipelinestate.hpp"

#include "global.hpp"

#include <iostream>

namespace rendering
{
PipelineStateManager & PipelineStateManager::Instance()
{
  static PipelineStateManager inst;
  return inst;
}

void PipelineStateManager::Initialize()
{
  auto checkState = [this](uint32_t state) {
    const bool enabled = (glIsEnabled(state) != 0);
    m_currentStates.insert(std::make_pair(state, enabled));
  };
  checkState(GL_CULL_FACE);
  checkState(GL_MULTISAMPLE);
  checkState(GL_BLEND);
  checkState(GL_DEPTH_TEST);

  glGetIntegerv(GL_BLEND_SRC_ALPHA, &m_currentBlendSrc);
  glGetIntegerv(GL_BLEND_DST_ALPHA, &m_currentBlendDst);
  glGetBooleanv(GL_DEPTH_WRITEMASK, &m_currentDepthWriteMask);
  glGetBooleanv(GL_COLOR_WRITEMASK, m_currentColorMask);
}

PipelineState::PipelineState(uint32_t state, bool enabled)
  : m_state(state), m_isEnabled(enabled), m_wasEnabled(false)
{
}

void PipelineState::Apply()
{
  auto & curStates = PipelineStateManager::Instance().m_currentStates;
  auto it = curStates.find(m_state);
  if (it == curStates.end())
  {
    std::cout << "Error: unregistered pipeline state " << m_state << std::endl;
    return;
  }

  m_wasEnabled = it->second;
  if (m_wasEnabled != m_isEnabled)
  {
    it->second = m_isEnabled;
    if (m_isEnabled)
      glEnable(m_state);
    else
      glDisable(m_state);
  }
}

void PipelineState::Cancel()
{
  auto & curStates = PipelineStateManager::Instance().m_currentStates;
  auto it = curStates.find(m_state);
  if (it == curStates.end())
  {
    std::cout << "Error: unregistered pipeline state " << m_state << std::endl;
    return;
  }

  if (m_wasEnabled != m_isEnabled)
  {
    it->second = m_wasEnabled;
    if (m_wasEnabled)
      glEnable(m_state);
    else
      glDisable(m_state);
  }
}

BlendState::BlendState(bool enabled)
  : m_pipelineState(GL_BLEND, enabled)
  , m_oldBlendSrc(-1)
  , m_oldBlendDst(-1)
  , m_blendSrc(-1)
  , m_blendDst(-1)
{
}

void BlendState::SetBlending(int src, int dest)
{
  m_oldBlendSrc = PipelineStateManager::Instance().m_currentBlendSrc;
  m_oldBlendDst = PipelineStateManager::Instance().m_currentBlendDst;
  m_blendSrc = src;
  m_blendDst = dest;
}

void BlendState::Apply()
{
  m_pipelineState.Apply();

  if (m_blendSrc != m_oldBlendSrc || m_blendDst != m_oldBlendDst)
  {
    PipelineStateManager::Instance().m_currentBlendSrc = m_blendSrc;
    PipelineStateManager::Instance().m_currentBlendDst = m_blendDst;
    glBlendFunc(m_blendSrc, m_blendDst);
  }
}

void BlendState::Cancel()
{
  m_pipelineState.Cancel();

  if (m_blendSrc != m_oldBlendSrc || m_blendDst != m_oldBlendDst)
  {
    PipelineStateManager::Instance().m_currentBlendSrc = m_oldBlendSrc;
    PipelineStateManager::Instance().m_currentBlendDst = m_oldBlendDst;
    glBlendFunc(m_oldBlendSrc, m_oldBlendDst);
  }
}

DepthState::DepthState(bool enabled)
  : m_pipelineState(GL_DEPTH_TEST, enabled), m_oldDepthWriteMask(-1), m_depthWriteMask(-1)
{
}

DepthState::DepthState(bool enabled, bool writeEnable) : DepthState(enabled)
{
  SetWriteEnable(writeEnable);
}

void DepthState::SetWriteEnable(bool enable)
{
  m_oldDepthWriteMask = PipelineStateManager::Instance().m_currentDepthWriteMask;
  m_depthWriteMask = enable ? GL_TRUE : GL_FALSE;
}

void DepthState::Apply()
{
  m_pipelineState.Apply();

  if (m_oldDepthWriteMask != m_depthWriteMask)
  {
    PipelineStateManager::Instance().m_currentDepthWriteMask = m_depthWriteMask;
    glDepthMask(m_depthWriteMask);
  }
}

void DepthState::Cancel()
{
  m_pipelineState.Cancel();

  if (m_oldDepthWriteMask != m_depthWriteMask)
  {
    PipelineStateManager::Instance().m_currentDepthWriteMask = m_oldDepthWriteMask;
    glDepthMask(m_oldDepthWriteMask);
  }
}

ColorOutputState::ColorOutputState(bool enabled) : m_isEnabled(enabled) {}
void ColorOutputState::Apply()
{
  memcpy(m_oldColorMask, PipelineStateManager::Instance().m_currentColorMask,
         sizeof(m_oldColorMask));
  uint8_t enabled = m_isEnabled ? GL_TRUE : GL_FALSE;
  if (m_oldColorMask[0] != enabled || m_oldColorMask[1] != enabled ||
      m_oldColorMask[2] != enabled || m_oldColorMask[3] != enabled)
  {
    uint8_t mask[] = {enabled, enabled, enabled, enabled};
    memcpy(PipelineStateManager::Instance().m_currentColorMask, mask, sizeof(mask));
    glColorMask(enabled, enabled, enabled, enabled);
  }
}

void ColorOutputState::Cancel()
{
  uint8_t enabled = m_isEnabled ? GL_TRUE : GL_FALSE;
  if (m_oldColorMask[0] != enabled || m_oldColorMask[1] != enabled ||
      m_oldColorMask[2] != enabled || m_oldColorMask[3] != enabled)
  {
    memcpy(PipelineStateManager::Instance().m_currentColorMask, m_oldColorMask,
           sizeof(m_oldColorMask));
    glColorMask(m_oldColorMask[0], m_oldColorMask[1], m_oldColorMask[2], m_oldColorMask[3]);
  }
}
}  // namespace rendering
