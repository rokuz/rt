cmake_minimum_required(VERSION 3.12)
project(rt_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
set(CUDA_COMPILE_TIME_EXTRA_FLAGS -g -G)

include_directories(..)
include_directories(../3party/glm)
include_directories(../3party/cuda-workshop/cutil/inc)
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_LIB
  demo_frame_cuda.cpp
  demo_frame_cuda.hpp
  rt_cuda.cu
  rt_cuda.h
  rt_functions.h
  rt_hits.h
  rt_random.h
  rt_light.h
  rt_materials.h
  types_cuda.h
)

add_library(${PROJECT_NAME} STATIC ${SOURCE_LIB})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)