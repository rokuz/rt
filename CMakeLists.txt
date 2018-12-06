cmake_minimum_required(VERSION 3.12)
project(rt)

set(CMAKE_CXX_STANDARD 14)

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
endif (WIN32)

find_package(OpenGL REQUIRED)

include_directories(.)

include_directories(3party/glfw/include)

include_directories(3party/glm)

include_directories(3party/cxxopts/include)

if (WIN32)
include_directories(3party/gl3w)
endif (WIN32)

if (APPLE)
add_definitions(-DGL_SILENCE_DEPRECATION)
endif (APPLE)

set(SOURCE_ROOT .)
aux_source_directory(${SOURCE_ROOT} SRC_LIST)
aux_source_directory(${SOURCE_ROOT}/rendering SRC_LIST_RENDERING)
aux_source_directory(${SOURCE_ROOT}/ray_tracing SRC_LIST_RAY_TRACING)
aux_source_directory(${SOURCE_ROOT}/demos SRC_LIST_DEMOS)

add_executable(${PROJECT_NAME}
               ${SRC_LIST}
               ${SRC_LIST_RENDERING}
               ${SRC_LIST_RAY_TRACING}
               ${SRC_LIST_DEMOS})

target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY} glfw)
if (WIN32)
target_link_libraries(${PROJECT_NAME} gl3w)
endif (WIN32)
