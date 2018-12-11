cmake_minimum_required(VERSION 3.12)
if (WIN32)
project(ray_tracing LANGUAGES CXX CUDA)
else()
project(ray_tracing)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(../3party/glm)

set(SRC_LIST 
	default_materials.cpp
	default_materials.hpp
	directional_light.cpp
	directional_light.hpp
	frame.cpp
	frame.hpp
	hitable_object.hpp
	light.hpp
	material.hpp
	multithreaded_frame.cpp
	multithreaded_frame.hpp
	ray.hpp
	sphere.cpp
	sphere.hpp
	types.hpp
)

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})

if (WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
