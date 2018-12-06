#pragma once

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
#include "gl3w.h"
#else
#include <OpenGL/gl3.h>
#include <OpenGl/gl3ext.h>
#endif

float constexpr kPi = static_cast<float>(3.14159265358979323846);
float constexpr kEps = 1e-5f;