cmake_minimum_required(VERSION 3.12)

set(CUDA_ENABLE ON)

if (WIN32)
  if (CUDA_ENABLE)
    project(rt LANGUAGES CXX CUDA)
  else()
    project(rt)
  endif()
else(WIN32)
  project(rt)
endif(WIN32)

set(CMAKE_CXX_STANDARD 17)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")
add_subdirectory(3party/glfw)

set(GLM_STATIC_LIBRARY_ENABLE OFF CACHE BOOL "")
set(GLM_INSTALL_ENABLE OFF CACHE BOOL "")
set(GLM_TEST_ENABLE OFF CACHE BOOL "")
add_subdirectory(3party/glm)

if (WIN32)
add_subdirectory(3party/gl3w)
if (CUDA_ENABLE)
  add_definitions(-DENABLE_CUDA)
  add_subdirectory(rt_cuda)
endif()
endif (WIN32)

find_package(OpenGL REQUIRED)

include_directories(.)

include_directories(3party/glfw/include)

include_directories(3party/glm)

include_directories(3party/cxxopts/include)

if (WIN32)
if (CUDA_ENABLE)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()
  include_directories(3party/gl3w)
endif()

if (APPLE)
add_definitions(-DGL_SILENCE_DEPRECATION)
endif (APPLE)

set(SOURCE_ROOT .)
aux_source_directory(${SOURCE_ROOT} SRC_LIST)
aux_source_directory(${SOURCE_ROOT}/rendering SRC_LIST_RENDERING)
aux_source_directory(${SOURCE_ROOT}/demos SRC_LIST_DEMOS)

add_subdirectory(ray_tracing)

add_executable(${PROJECT_NAME}
               ${SRC_LIST}
               ${SRC_LIST_RENDERING}
               ${SRC_LIST_DEMOS})

if (WIN32 AND CUDA_ENABLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(${PROJECT_NAME} ray_tracing)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY} glfw)
if (WIN32)
  target_link_libraries(${PROJECT_NAME} gl3w)
  if (CUDA_ENABLE)
    target_link_libraries(${PROJECT_NAME} rt_cuda)
  endif()
endif()
