#include "app.hpp"
#include "global.hpp"

#include "demos/glass_spheres.hpp"
#include "demos/pretty_spheres.hpp"

#include "rendering/pipelinestate.hpp"

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
#include "rt_cuda/rt_cuda.h"
#endif

#include <glfw/glfw3.h>

#include <cxxopts.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

App app;

void KeyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
  switch (key)
  {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(window, GL_TRUE);
    break;
  default:
    app.OnKeyButton(key, scancode, action != GLFW_RELEASE);
  }
}

void MouseCallback(GLFWwindow * window, int button, int action, int mods)
{
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);
  app.OnMouseButton(xpos, ypos, button, action != GLFW_RELEASE);
}

void MouseMoveCallback(GLFWwindow * window, double xpos, double ypos)
{
  app.OnMouseMove(xpos, ypos);
}

void ErrorCallback(int error, const char * description)
{
  std::cout << "Error(" << error << "): " << description << std::endl;
}

int main(int argc, char * argv[])
{
  uint32_t width = 1024;
  uint32_t height = 768;
  uint32_t rtThreadsCount = 4;
  uint32_t samplesInRow = 2;
  uint32_t demoIndex = 1;
  try
  {
    cxxopts::Options options(argv[0], " - simple C++ ray tracer");
    options.allow_unrecognised_options().add_options()(
        "w,width", "Window width", cxxopts::value<uint32_t>(width)->default_value("1024"))(
        "h,height", "Window height", cxxopts::value<uint32_t>(height)->default_value("768"))(
        "s,samples", "Supersampling samples in row count",
        cxxopts::value<uint32_t>(samplesInRow)->default_value("2"))(
        "t,threads", "Ray tracing threads count",
        cxxopts::value<uint32_t>(rtThreadsCount)->default_value("4"))(
        "d,demo", "Demo index", cxxopts::value<uint32_t>(demoIndex)->default_value("1"));
    options.parse(argc, argv);
  }
  catch (cxxopts::OptionException const & e)
  {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    return 1;
  }

  std::vector<std::unique_ptr<ray_tracing::Frame>> frames;
  frames.emplace_back(std::make_unique<demo::PrettySpheres>(rtThreadsCount));
  frames.emplace_back(std::make_unique<demo::GlassSpheres>(rtThreadsCount));

  demoIndex = std::max(static_cast<uint32_t>(1), demoIndex);
  if (demoIndex > frames.size())
  {
    std::cout << "Incorrect demo index." << std::endl;
    return 1;
  }

  glfwSetErrorCallback(ErrorCallback);

  if (!glfwInit())
    return 1;

  uint8_t constexpr kOpenGLMajor = 4;
  uint8_t constexpr kOpenGLMinor = 1;
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, kOpenGLMajor);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, kOpenGLMinor);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 0);

  GLFWwindow * window =
      glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "rt", nullptr, nullptr);
  if (!window)
  {
    glfwTerminate();
    return 1;
  }

  glfwSetKeyCallback(window, KeyCallback);
  glfwSetMouseButtonCallback(window, MouseCallback);
  glfwSetCursorPosCallback(window, MouseMoveCallback);

  glfwMakeContextCurrent(window);

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER)
  if (gl3wInit() < 0)
  {
    std::cout << "Error initialization OpenGL." << std::endl;
    glfwTerminate();
    return 1;
  }

  if (!gl3wIsSupported(kOpenGLMajor, kOpenGLMinor))
  {
    std::cout << "OpenGL version " << kOpenGLMajor << "." << kOpenGLMinor << " is not supported."
              << std::endl;
    glfwTerminate();
    return 1;
  }

  if (!ray_tracing_cuda::Initialize())
    return 1;
#endif

  rendering::PipelineStateManager::Instance().Initialize();

  if (!app.Initialize(std::move(frames[demoIndex - 1]), width, height, samplesInRow))
  {
    app.Uninitialize();
    glfwTerminate();
    return 1;
  }

  double lastTime = glfwGetTime();
  bool firstFrame = true;
  while (!glfwWindowShouldClose(window))
  {
    double const currentTime = glfwGetTime();
    double const elapsedTime = currentTime - lastTime;
    lastTime = currentTime;

    app.Render(currentTime, elapsedTime);

    if (firstFrame)
    {
      firstFrame = false;
      app.RayTrace(false /* highQuality */);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  app.Uninitialize();
  glfwTerminate();
  return 0;
}
