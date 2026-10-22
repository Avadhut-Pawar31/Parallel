#include "parallel/image_ops.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

namespace parallel {

ImageDifference compareImages(const Image& imgA, const Image& imgB, double tolerance) {
    ImageDifference diff;
    if (imgA.width != imgB.width || imgA.height != imgB.height || imgA.channels != imgB.channels) {
        diff.passed = false;
        return diff;
    }

    const size_t totalBytes = imgA.size();
    diff.totalPixels = static_cast<size_t>(imgA.width * imgA.height);

    double totalError = 0.0;
    double maxError = 0.0;
    size_t differing = 0;

    const uint8_t* a = imgA.data.data();
    const uint8_t* b = imgB.data.data();

    for (size_t i = 0; i < totalBytes; ++i) {
        const double err = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        totalError += err;
        if (err > maxError) {
            maxError = err;
        }
        if (err > tolerance) {
            differing++;
        }
    }

    diff.meanAbsoluteError = totalBytes > 0 ? (totalError / totalBytes) : 0.0;
    diff.maxAbsoluteError = maxError;
    diff.differingPixels = differing / imgA.channels;
    diff.passed = (maxError <= tolerance);

    return diff;
}

Image executeOperation(
    const Image& input,
    const std::string& opName,
    Backend backend,
    OperationResult* result,
    int kernelSize,
    float sigma
) {
    if (!input.isValid()) {
        throw std::invalid_argument("Input image is invalid or empty");
    }

    auto start = std::chrono::high_resolution_clock::now();
    Image output;

    if (backend == Backend::CPU) {
        if (opName == "grayscale" || opName == "gray") {
            output = cpuGrayscale(input);
        } else if (opName == "gaussian" || opName == "blur" || opName == "gaussian_blur") {
            output = cpuGaussianBlur(input, kernelSize, sigma);
        } else if (opName == "sharpen") {
            output = cpuSharpen(input);
        } else if (opName == "sobel" || opName == "edge") {
            output = cpuSobel(input);
        } else {
            throw std::invalid_argument("Unknown operation: " + opName + ". Supported: grayscale, blur, sharpen, sobel");
        }
    } else if (backend == Backend::CUDA) {
#if defined(PARALLEL_ENABLE_CUDA)
        if (opName == "grayscale" || opName == "gray") {
            output = cudaGrayscale(input, result);
        } else if (opName == "gaussian" || opName == "blur" || opName == "gaussian_blur") {
            output = cudaGaussianBlur(input, kernelSize, sigma, result);
        } else if (opName == "sharpen") {
            output = cudaSharpen(input, result);
        } else if (opName == "sobel" || opName == "edge") {
            output = cudaSobel(input, result);
        } else {
            throw std::invalid_argument("Unknown operation: " + opName);
        }
#else
        throw std::runtime_error("CUDA backend is not available in this build. Compile with -DPARALLEL_ENABLE_CUDA=ON on a system with CUDA toolkit.");
#endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (result) {
        result->success = output.isValid();
        result->operation = opName;
        result->backend = backendToString(backend);
        result->width = output.width;
        result->height = output.height;
        result->channels = output.channels;
        result->executionTimeMs = elapsedMs;
    }

    return output;
}

} // namespace parallel
