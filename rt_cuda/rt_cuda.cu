#include "rt_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cutil_math.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <iostream>
#include <vector>

namespace ray_tracing_cuda
{
uint32_t constexpr kThreadsInRow = 8;

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

std::vector<void *> asyncFreeMemory;

template <typename T>
class GPUPtr
{
public:
  explicit GPUPtr(uint32_t size, bool asyncFree = true) : m_size(size), m_asyncFree(asyncFree)
  {
    if (cudaMalloc(&m_ptr, m_size) != cudaSuccess)
      m_ptr = nullptr;

    if (m_asyncFree && m_ptr != nullptr)
      asyncFreeMemory.push_back(m_ptr);
  }

  ~GPUPtr()
  {
    if (!m_asyncFree && m_ptr != nullptr)
      cudaFree(m_ptr);
  }

  operator bool() { return m_ptr != nullptr; }

  T * m_ptr = nullptr;
  uint32_t m_size;
  bool m_asyncFree;
};

struct TransferredGPUPtr
{
  void * m_ptr = nullptr;
  uint32_t m_size = 0;

  TransferredGPUPtr() = default;
  TransferredGPUPtr(void * ptr, uint32_t size) : m_ptr(ptr), m_size(size) {}

  template<typename T>
  void Set(GPUPtr<T> const & ptr)
  {
    m_ptr = ptr.m_ptr;
    m_size = ptr.m_size;
  }
};
TransferredGPUPtr transferredOutputPtr;

__device__ bool hitSphere(CudaSphere * sphere, CudaRay * ray, float tmin, float tmax, CudaHit * hit)
{
  float3 const d = ray->m_origin - sphere->m_center;
  float const a = dot(ray->m_direction, ray->m_direction);
  float const b = 2.0f * dot(d, ray->m_direction);
  float const c = dot(d, d) - sphere->m_radius * sphere->m_radius;
  float const discriminant = b * b - 4 * a * c;
  if (discriminant < 0.0f)
    return false;

  auto const sqrtD = sqrt(discriminant);
  auto const t = min((-b - sqrtD) / (2.0f * a), (-b + sqrtD) / (2.0f * a));
  if (t < tmin || t > tmax)
    return false;

  hit->m_parameterT = t;
  hit->m_position = ray->m_origin + ray->m_direction * t;
  hit->m_normal = normalize(hit->m_position - sphere->m_center);
  hit->m_materialIndex = sphere->m_materialIndex;
  return true;
}

__device__ void TraceRayGPU(CudaRay * ray, CudaSphere * spheres, uint32_t spheresCount,
                            CudaMaterial * materials, uint32_t materialsCount,
                            CudaLight * lightSources, uint32_t lightSourcesCount,
                            float3 backgroundColor, float znear, float zfar, float3 * output)
{
  CudaHit hit;
  hit.m_parameterT = zfar + 1.0f;
  bool hitFound = false;
  for (uint32_t i = 0; i < spheresCount; ++i)
  {
    CudaHit h;
    if (hitSphere(&spheres[i], ray, znear, zfar, &h))
    {
      hitFound = true;
      if (h.m_parameterT < hit.m_parameterT)
        hit = h;
    }
  }

  if (hitFound)
    *output = materials[hit.m_materialIndex].m_albedo;
  else
    *output = backgroundColor;
}

__global__ void TraceAllRaysGPU(CudaSphere * spheres, uint32_t spheresCount,
                                CudaMaterial * materials, uint32_t materialsCount,
                                CudaLight * lightSources, uint32_t lightSourcesCount,
                                float3 backgroundColor, float3 origin, float3 forward, float3 up,
                                float3 right, float2 halfScreenSize, float2 cellSize,
                                uint32_t samplesInRowCount, float invSampleCount, float znear,
                                float zfar, float3 * output)
{
  __shared__ float3 samples[kThreadsInRow][kThreadsInRow];
  int x = blockIdx.x;
  int y = blockIdx.y;

  samples[threadIdx.x][threadIdx.y] = make_float3(0.0f, 0.0f, 0.0f);
  int tx = threadIdx.x;
  while (tx < samplesInRowCount)
  {
    int ty = threadIdx.y;
    while (ty < samplesInRowCount)
    {
      float const dx = (2.0f * x / gridDim.x - 1.0f) * halfScreenSize.x;
      float const sdx = dx + cellSize.x * tx / samplesInRowCount;
      float const dy = (-2.0f * y / gridDim.y + 1.0f) * halfScreenSize.y;
      float const sdy = dy - cellSize.y * ty / samplesInRowCount;

      CudaRay ray;
      ray.m_origin = origin;
      ray.m_direction = normalize(forward * znear + up * sdy + right * sdx);

      float3 outputColor;
      TraceRayGPU(&ray, spheres, spheresCount, materials, materialsCount, lightSources,
                  lightSourcesCount, backgroundColor, znear, zfar, &outputColor);
      samples[threadIdx.x][threadIdx.y] += outputColor;

      ty += blockDim.y;
    }
    tx += blockDim.x;
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

  int offset = x + y * gridDim.x;
  if (threadIdx.x == 0 && threadIdx.y == 0)
    output[offset] = samples[0][0] * invSampleCount;
}

cudaEvent_t RayTrace(CudaSphere * spheres, uint32_t spheresCount, CudaMaterial * materials,
                     uint32_t materialsCount, CudaLight * lightSources, uint32_t lightSourcesCount,
                     uint32_t samplesInRowCount, float3 backgroundColor, float3 cameraPosition,
                     float3 cameraDirection, float fov, float znear, float zfar, uint32_t width,
                     uint32_t height)
{
  cudaEvent_t completion;
  if (cudaEventCreate(&completion) != cudaSuccess)
  {
    std::cout << "Error call cudaEventCreate." << std::endl;
    return nullptr;
  }

  GPUPtr<CudaSphere> spheresGPU(spheresCount * sizeof(CudaSphere));
  if (!spheresGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpyAsync(spheresGPU.m_ptr, spheres, spheresGPU.m_size, cudaMemcpyHostToDevice) !=
      cudaSuccess)
  {
    std::cout << "Error call cudaMemcpyAsync (spheresGPU)." << std::endl;
    return completion;
  }

  GPUPtr<CudaMaterial> materialsGPU(materialsCount * sizeof(CudaMaterial));
  if (!materialsGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpyAsync(materialsGPU.m_ptr, materials, materialsGPU.m_size, cudaMemcpyHostToDevice) !=
      cudaSuccess)
  {
    std::cout << "Error call cudaMemcpyAsync (materialsGPU)." << std::endl;
    return completion;
  }

  GPUPtr<CudaLight> lightSourcesGPU(lightSourcesCount * sizeof(CudaLight));
  if (!lightSourcesGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  if (cudaMemcpyAsync(lightSourcesGPU.m_ptr, lightSources, lightSourcesGPU.m_size,
                      cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cout << "Error call cudaMemcpyAsync (lightSourcesGPU)." << std::endl;
    return completion;
  }

  GPUPtr<float3> outputGPU(width * height * sizeof(float3));
  if (!outputGPU)
  {
    std::cout << "Error allocate GPU memory." << std::endl;
    return completion;
  }
  transferredOutputPtr.Set(outputGPU);

  static float3 kUp = make_float3(0.0f, 1.0f, 0.0f);
  auto const aspect = static_cast<float>(height) / width;

  float3 const right = cross(kUp, cameraDirection);
  float3 const up = cross(cameraDirection, right);
  float const dw = znear / tan(0.5f * fov);
  float2 const halfScreenSize = make_float2(dw, dw * aspect);
  float2 const cellSize =
      make_float2(2.0f * halfScreenSize.x / width, 2.0f * halfScreenSize.y / height);
  float const invSampleCount = 1.0f / (samplesInRowCount * samplesInRowCount);

  dim3 grids(width, height);
  dim3 threads(samplesInRowCount, samplesInRowCount);
  TraceAllRaysGPU<<<grids, kThreadsInRow>>>(
      spheresGPU.m_ptr, spheresCount, materialsGPU.m_ptr, materialsCount, lightSourcesGPU.m_ptr,
      lightSourcesCount, backgroundColor, cameraPosition, cameraDirection, up, right,
      halfScreenSize, cellSize, samplesInRowCount, invSampleCount, znear, zfar, outputGPU.m_ptr);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cout << "Error CUDA: " << cudaGetErrorString(err) << std::endl;
    return completion;
  }

  if (cudaEventRecord(completion, 0) != cudaSuccess)
  {
    std::cout << "Error call cudaEventRecord." << std::endl;
    return completion;
  }

  return completion;
}

bool IsInProgress(cudaEvent_t completion)
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

  for (size_t i = 0; i < asyncFreeMemory.size(); ++i)
    cudaFree(asyncFreeMemory[i]);
  asyncFreeMemory.clear();

  if (cudaEventDestroy(completion) != cudaSuccess)
    std::cout << "Error call cudaEventDestroy." << std::endl;
}
}  // namespace ray_tracing_cuda