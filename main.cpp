#include "app.hpp"

#include "ray_tracing/circles.hpp"

#include "rendering/pipelinestate.hpp"

#include <glfw/glfw3.h>

#include <cxxopts.hpp>

#include <iostream>
#include <limits>
#include <string>

App app;
bool needTrace = true;

void KeyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
  switch (key)
  {
  case GLFW_KEY_ESCAPE:
    glfwSetWindowShouldClose(window, GL_TRUE);
    break;
  case GLFW_KEY_T:
    needTrace = true;
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
  bool renderSingleFrame = true;

  try
  {
    cxxopts::Options options(argv[0], " - simple C++ ray tracer");
    options
      .allow_unrecognised_options()
      .add_options()
        ("w,width", "window width", cxxopts::value<uint32_t>(width)->default_value("1024"))
        ("h,height", "window height", cxxopts::value<uint32_t>(height)->default_value("768"))
        ("s,singleFrame", "render single frame", cxxopts::value<bool>()->default_value("true"));
    options.parse(argc, argv);
  }
  catch (const cxxopts::OptionException& e)
  {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    return 1;
  }

  glfwSetErrorCallback(ErrorCallback);

  if (!glfwInit())
    return 1;

  uint8_t const kOpenGLMajor = 4;
  uint8_t const kOpenGLMinor = 1;
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, kOpenGLMajor);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, kOpenGLMinor);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 0);

  GLFWwindow * window = glfwCreateWindow(width, height, "rt", nullptr, nullptr);
  if (!window)
  {
    glfwTerminate();
    return 1;
  }

  glfwSetKeyCallback(window, KeyCallback);
  glfwSetMouseButtonCallback(window, MouseCallback);
  glfwSetCursorPosCallback(window, MouseMoveCallback);

  glfwMakeContextCurrent(window);

  rendering::PipelineStateManager::Instance().Initialize();

  std::unique_ptr<ray_tracing::Frame> frame = nullptr;
  frame = std::make_unique<ray_tracing::Circles>();

  if (!app.Initialize(std::move(frame), width, height))
  {
    app.Uninitialize();
    glfwTerminate();
    return 1;
  }

  double lastTime = glfwGetTime();
  while (!glfwWindowShouldClose(window))
  {
    double const currentTime = glfwGetTime();
    double const elapsedTime = currentTime - lastTime;
    lastTime = currentTime;

    if (!renderSingleFrame)
      needTrace = true;
    app.Render(currentTime, elapsedTime, needTrace);
    needTrace = false;

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  app.Uninitialize();
  glfwTerminate();
  return 0;
}
