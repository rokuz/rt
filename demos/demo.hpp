#pragma once

#include "ray_tracing/frame.hpp"

#include <memory>

namespace demo
{
class Demo
{
public:
  virtual ~Demo() = default;

  virtual bool Initialize(std::shared_ptr<ray_tracing::ColorBuffer> buffer,
                          uint32_t width, uint32_t height,
                          uint32_t samplesInRowCount) = 0;

  virtual std::unique_ptr<ray_tracing::Frame> & GetFrame() = 0;
};
}  // namespace demo
