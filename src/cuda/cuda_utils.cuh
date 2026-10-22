#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::string msg = "CUDA error at " + std::string(__FILE__) + ":" +   \
                              std::to_string(__LINE__) + " code=" +              \
                              std::to_string(err) + " \"" +                      \
                              cudaGetErrorString(err) + "\"";                    \
            throw std::runtime_error(msg);                                       \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                      \
    do {                                                                         \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            std::string msg = "CUDA kernel launch error at " +                   \
                              std::string(__FILE__) + ":" +                      \
                              std::to_string(__LINE__) + " code=" +              \
                              std::to_string(err) + " \"" +                      \
                              cudaGetErrorString(err) + "\"";                    \
            throw std::runtime_error(msg);                                       \
        }                                                                        \
    } while (0)

namespace parallel {

inline dim3 calculateGridDim(int width, int height, dim3 blockDim) {
    return dim3(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
}

} // namespace parallel
