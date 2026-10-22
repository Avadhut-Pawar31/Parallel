#pragma once

#include "parallel/image_io.hpp"
#include "parallel/backend.hpp"
#include "parallel/result.hpp"
#include <string>

namespace parallel {

// CPU implementations
Image cpuGrayscale(const Image& input);
Image cpuGaussianBlur(const Image& input, int kernelSize = 3, float sigma = 1.0f);
Image cpuSharpen(const Image& input);
Image cpuSobel(const Image& input);

// CUDA implementations
#if defined(PARALLEL_ENABLE_CUDA)
Image cudaGrayscale(const Image& input, OperationResult* timing = nullptr);
Image cudaGaussianBlur(const Image& input, int kernelSize = 3, float sigma = 1.0f, OperationResult* timing = nullptr);
Image cudaSharpen(const Image& input, OperationResult* timing = nullptr);
Image cudaSobel(const Image& input, OperationResult* timing = nullptr);
#else
inline Image cudaGrayscale(const Image&, OperationResult*) {
    throw std::runtime_error("CUDA backend was not compiled in this build.");
}
inline Image cudaGaussianBlur(const Image&, int, float, OperationResult*) {
    throw std::runtime_error("CUDA backend was not compiled in this build.");
}
inline Image cudaSharpen(const Image&, OperationResult*) {
    throw std::runtime_error("CUDA backend was not compiled in this build.");
}
inline Image cudaSobel(const Image&, OperationResult*) {
    throw std::runtime_error("CUDA backend was not compiled in this build.");
}
#endif

// General dispatcher
Image executeOperation(
    const Image& input,
    const std::string& opName,
    Backend backend,
    OperationResult* result = nullptr,
    int kernelSize = 3,
    float sigma = 1.0f
);

// Verification and comparison utilities
struct ImageDifference {
    double meanAbsoluteError{0.0};
    double maxAbsoluteError{0.0};
    size_t differingPixels{0};
    size_t totalPixels{0};
    bool passed{false};
};

ImageDifference compareImages(const Image& imgA, const Image& imgB, double tolerance = 2.0);

} // namespace parallel
