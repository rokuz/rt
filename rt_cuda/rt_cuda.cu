#include "rt_cuda.h"
#include "rt_functions.h"

// CUDA headers.
#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cutil_math.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <ctime>
#include <iostream>
#include <vector>

namespace ray_tracing_cuda
{
namespace
{
uint32_t constexpr kThreadsInRow = 8;

std::vector<void *> delayedFreeMemory;

template<typename T>
class GPUPtr
{
public:
  explicit GPUPtr(uint32_t size, bool delayedFree = false)
    : m_size(size * sizeof(T)), m_delayedFree(delayedFree)
  {
    if (cudaMalloc(&m_ptr, m_size) != cudaSuccess)
      m_ptr = nullptr;

    if (m_delayedFree && m_ptr != nullptr)
      delayedFreeMemory.push_back(m_ptr);
  }

  ~GPUPtr()
  {
    if (!m_delayedFree && m_ptr != nullptr)
      cudaFree(m_ptr);
  }

  operator bool() { return m_ptr != nullptr; }

  T * m_ptr = nullptr;
  uint32_t m_size = 0;
  bool m_delayedFree = false;
};

struct TransferredGPUPtr
{
  void * m_ptr = nullptr;
  uint32_t m_size = 0;

  TransferredGPUPtr() = default;
  TransferredGPUPtr(void * ptr, uint32_t size)
    : m_ptr(ptr), m_size(size) {}

  template<typename T>
  void Set(GPUPtr<T> const &ptr)
  {
    m_ptr = ptr.m_ptr;
    m_size = ptr.m_size;
  }
};
TransferredGPUPtr transferredOutputPtr;
}  // namespace

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

__global__ void InitRandom(curandState * randStates, unsigned long long seed) 
{
  int rndx = blockIdx.x * blockDim.x + threadIdx.x;
  int rndy = blockIdx.y * blockDim.y + threadIdx.y;
  int rndIndex = rndy * gridDim.x * blockDim.x + rndx;
  curand_init(seed, rndIndex, 0, &randStates[rndIndex]);
}

__global__ void TraceAllRaysGPU(CudaSphere * spheres, uint32_t spheresCount,
                                CudaMaterial * materials, CudaLight * lightSources,
                                uint32_t lightSourcesCount, float3 backgroundColor, 
                                float3 origin, float3 forward, float3 up,
                                float3 right, float2 halfScreenSize, float2 cellSize,
                                uint32_t samplesInRowCount, float invSampleCount,
                                float znear, float zfar, uint32_t offsetX, uint32_t offsetY,
                                uint32_t width, uint32_t height, curandState * randStates,
                                float3 * output)
{
  __shared__ float3 samples[kThreadsInRow][kThreadsInRow];
  int x = blockIdx.x + offsetX;
  int y = blockIdx.y + offsetY;

  int rndx = blockIdx.x * blockDim.x + threadIdx.x;
  int rndy = blockIdx.y * blockDim.y + threadIdx.y;
  int rndIndex = rndy * gridDim.x * blockDim.x + rndx;

  samples[threadIdx.x][threadIdx.y] = make_float3(0.0f, 0.0f, 0.0f);
  if (x < width && y < height)
  {
    int tx = threadIdx.x;
    while (tx < samplesInRowCount)
    {
      int ty = threadIdx.y;
      while (ty < samplesInRowCount)
      {
        float const dx = (2.0f * x / width - 1.0f) * halfScreenSize.x;
        float const sdx = dx + cellSize.x * tx / samplesInRowCount;
        float const dy = (-2.0f * y / height + 1.0f) * halfScreenSize.y;
        float const sdy = dy - cellSize.y * ty / samplesInRowCount;

        CudaRay ray;
        ray.m_origin = origin;
        ray.m_direction = normalize(forward * znear + up * sdy + right * sdx);

        float3 outputColor;
        TraceRayGPU(&ray, spheres, spheresCount, materials,
                    lightSources, lightSourcesCount, backgroundColor,
                    znear, zfar, &randStates[rndIndex], &outputColor);
        samples[threadIdx.x][threadIdx.y] += outputColor;

        ty += blockDim.y;
      }
      tx += blockDim.x;
    }
  }

  // Samples reduction.
  __syncthreads();
  int j = kThreadsInRow / 2;
  while (j != 0)
  {
    if (threadIdx.x < j && threadIdx.x + j < blockDim.x)
      samples[threadIdx.x][threadIdx.y] += samples[threadIdx.x + j][threadIdx.y];
    __syncthreads();

    if (threadIdx.y < j && threadIdx.y + j < blockDim.y)
      samples[threadIdx.x][threadIdx.y] += samples[threadIdx.x][threadIdx.y + j];
    __syncthreads();
    j /= 2;
  }

  if (threadIdx.x == 0 && threadIdx.y == 0 && x < width && y < height)
    output[x + y * width] = samples[0][0] * invSampleCount;
}

cudaEvent_t RayTrace(CudaSphere * spheres, uint32_t spheresCount, CudaMaterial * materials,
                     uint32_t materialsCount, CudaLight * lightSources, uint32_t lightSourcesCount,
                     uint32_t samplesInRowCount, float3 backgroundColor, float3 cameraPosition,
                     float3 cameraDirection, float fov, float znear, float zfar, uint32_t width,
                     uint32_t height, std::function<bool()> && realtimeHandler)
{
  cudaEvent_t completion;
  if (cudaEventCreate(&completion) != cudaSuccess)
  {
    std::cout << "Error call cudaEventCreate." << std::endl;
    return nullptr;
  }

  GPUPtr<CudaSphere> spheresGPU(spheresCount);
  if (!spheresGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpy(spheresGPU.m_ptr, spheres, spheresGPU.m_size, cudaMemcpyHostToDevice) !=
      cudaSuccess)
  {
    std::cout << "Error call cudaMemcpy (spheresGPU)." << std::endl;
    return completion;
  }

  GPUPtr<CudaMaterial> materialsGPU(materialsCount);
  if (!materialsGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpy(materialsGPU.m_ptr, materials, materialsGPU.m_size, cudaMemcpyHostToDevice) !=
      cudaSuccess)
  {
    std::cout << "Error call cudaMemcpy (materialsGPU)." << std::endl;
    return completion;
  }

  GPUPtr<CudaLight> lightSourcesGPU(lightSourcesCount);
  if (!lightSourcesGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpy(lightSourcesGPU.m_ptr, lightSources, lightSourcesGPU.m_size,
                 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error call cudaMemcpy (lightSourcesGPU)." << std::endl;
    return completion;
  }

  uint32_t constexpr kPartsCount = 16;
  dim3 grids((width + kPartsCount - 1) / kPartsCount, (height + kPartsCount - 1) / kPartsCount);
  dim3 threads(kThreadsInRow, kThreadsInRow);

  GPUPtr<curandState> randStatesGPU(grids.x * grids.y * threads.x * threads.y);
  if (!randStatesGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }

  GPUPtr<float3> outputGPU(width * height, true /* delayedFree */);
  if (!outputGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  transferredOutputPtr.Set(outputGPU);

  InitRandom<<<grids, threads>>>(randStatesGPU.m_ptr, 
                                 static_cast<unsigned long long>(time(nullptr)));

  static float3 kUp = make_float3(0.0f, 1.0f, 0.0f);
  auto const aspect = static_cast<float>(height) / width;

  float3 const right = cross(kUp, cameraDirection);
  float3 const up = cross(cameraDirection, right);
  float const dw = znear / tan(0.5f * fov);
  float2 const halfScreenSize = make_float2(dw, dw * aspect);
  float2 const cellSize =
      make_float2(2.0f * halfScreenSize.x / width, 2.0f * halfScreenSize.y / height);
  float const invSampleCount = 1.0f / (samplesInRowCount * samplesInRowCount);

  for (uint32_t i = 0; i < kPartsCount; ++i)
  {
    bool needInterrupt = false;
    for (uint32_t j = 0; j < kPartsCount; ++j)
    {
      TraceAllRaysGPU<<<grids, threads>>>(
        spheresGPU.m_ptr, spheresCount, materialsGPU.m_ptr,
        lightSourcesGPU.m_ptr, lightSourcesCount, backgroundColor,
        cameraPosition, cameraDirection, up, right, halfScreenSize,
        cellSize, samplesInRowCount, invSampleCount, znear, zfar,
        i * grids.x, j * grids.y, width, height, randStatesGPU.m_ptr,
        outputGPU.m_ptr);

      if (realtimeHandler)
        needInterrupt = realtimeHandler();
    }
    if (needInterrupt)
      break;
  }

  if (cudaEventRecord(completion, 0) != cudaSuccess)
    std::cout << "Error call cudaEventRecord." << std::endl;

  return completion;
}

bool InProgress(cudaEvent_t completion)
{
  if (cudaEventQuery(completion) != cudaErrorNotReady)
  {
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      std::cout << "Error CUDA: " << cudaGetErrorString(err) << std::endl;
      return true;
    }
    return false;
  }
  return true;
}

void CopyOutputToBuffer(float * buffer)
{
  if (cudaMemcpy(buffer, transferredOutputPtr.m_ptr, transferredOutputPtr.m_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::cout << "Error call cudaMemcpy (realtimeBuffer)." << std::endl;
  }
}

void FinishRayTrace(float * output, cudaEvent_t completion)
{
  if (cudaMemcpy(output, transferredOutputPtr.m_ptr, transferredOutputPtr.m_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::cout << "Error call cudaMemcpy (output)." << std::endl;
  }
    
  if (cudaDeviceSynchronize() != cudaSuccess)
    std::cout << "Error call cudaDeviceSynchronize." << std::endl;

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    std::cout << "Error CUDA: " << cudaGetErrorString(err) << std::endl;

  for (size_t i = 0; i < delayedFreeMemory.size(); ++i)
    cudaFree(delayedFreeMemory[i]);
  delayedFreeMemory.clear();

  if (cudaEventDestroy(completion) != cudaSuccess)
    std::cout << "Error call cudaEventDestroy." << std::endl;
}
}  // namespace ray_tracing_cuda
