#pragma once

#include <unordered_map>

namespace rendering
{
class PipelineStateManager
{
  friend class PipelineState;
  friend class BlendState;
  friend class DepthState;
  friend class ColorOutputState;

public:
  static PipelineStateManager & Instance();
  void Initialize();

private:
  PipelineStateManager() = default;
  std::unordered_map<uint32_t, bool> m_currentStates;
  int m_currentBlendSrc;
  int m_currentBlendDst;
  uint8_t m_currentDepthWriteMask;
  uint8_t m_currentColorMask[4];
};

class PipelineState
{
public:
  PipelineState(uint32_t state, bool enabled);

  void Apply();
  void Cancel();

private:
  uint32_t m_state;
  bool m_isEnabled;
  bool m_wasEnabled;
};

class BlendState
{
public:
  explicit BlendState(bool enabled);

  void SetBlending(int src, int dest);

  void Apply();
  void Cancel();

private:
  PipelineState m_pipelineState;
  int m_oldBlendSrc;
  int m_oldBlendDst;
  int m_blendSrc;
  int m_blendDst;
};

class DepthState
{
public:
  explicit DepthState(bool enabled);
  DepthState(bool enabled, bool writeEnable);

  void SetWriteEnable(bool enable);

  void Apply();
  void Cancel();

private:
  PipelineState m_pipelineState;
  int m_oldDepthWriteMask;
  int m_depthWriteMask;
};

class ColorOutputState
{
public:
  explicit ColorOutputState(bool enabled);

  void Apply();
  void Cancel();

private:
  bool m_isEnabled;
  uint8_t m_oldColorMask[4];
};
}  // namespace rendering
