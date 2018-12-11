#pragma once

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
#include "gl3w.h"
#undef min
#undef max
#else
#include <OpenGL/gl3.h>
#include <OpenGl/gl3ext.h>
#endif
