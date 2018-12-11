#include "rt_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

namespace ray_tracing_cuda
{
bool Initialize()
{
  int count;
  cudaError_t cudaStatus = cudaGetDeviceCount(&count);
  if (cudaStatus != cudaSuccess || count == 0)
  {
    std::cout << "Error call cudaGetDeviceCount." << std::endl;
    return false;
  }

  if (cudaSetDevice(0) != cudaSuccess)
  {
    std::cout << "Error call cudaSetDevice." << std::endl;
    return false;
  }

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
  {
    std::cout << "Error call cudaSetDeviceProperties." << std::endl;
    return false;
  }

  std::cout << "CUDA device: " << prop.name << std::endl;

  return true;
}

void RayTrace(CudaSphere * spheres, uint32_t spheresCount,
              CudaMaterial * materials, uint32_t materialsCount,
              CudaLight * lightSources, uint32_t lightSourcesCount,
              uint32_t samplesInRowCount, float3 backgroundColor,
              float3 cameraPosition, float3 cameraDirection,
              float fov, float znear, float zfar,
              uint32_t width, uint32_t height, float3 *& output)
{

}
}  // namespace ray_tracing_cuda