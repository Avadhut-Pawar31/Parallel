#include "parallel/backend.hpp"
#include "parallel/benchmark.hpp"

#if defined(PARALLEL_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <iostream>

namespace parallel {

bool isCudaSupported() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount <= 0) {
        return false;
    }
    return true;
}

std::string getGpuModelName() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        return "None (CUDA unavailable)";
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        return std::string(prop.name);
    }
    return "Unknown NVIDIA GPU";
}

std::string getCudaDriverVersion() {
    int driverVersion = 0;
    if (cudaDriverGetVersion(&driverVersion) == cudaSuccess) {
        int major = driverVersion / 1000;
        int minor = (driverVersion % 100) / 10;
        return std::to_string(major) + "." + std::to_string(minor);
    }
    return "Unknown";
}

} // namespace parallel

#else

namespace parallel {

bool isCudaSupported() {
    return false;
}

std::string getGpuModelName() {
    return "N/A (CPU-only build)";
}

std::string getCudaDriverVersion() {
    return "N/A (CPU-only build)";
}

} // namespace parallel

#endif
