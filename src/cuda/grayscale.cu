#include "parallel/image_ops.hpp"
#include "cuda_utils.cuh"

#if defined(PARALLEL_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cmath>

namespace parallel {

__global__ void rgbToGrayscaleKernel(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIdx = y * width + x;
    int inputOffset = pixelIdx * 3;

    float r = static_cast<float>(d_input[inputOffset + 0]);
    float g = static_cast<float>(d_input[inputOffset + 1]);
    float b = static_cast<float>(d_input[inputOffset + 2]);

    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    d_output[pixelIdx] = static_cast<uint8_t>(rintf(gray));
}

Image cudaGrayscale(const Image& input, OperationResult* timing) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid image passed to cudaGrayscale");
    }
    if (input.channels == 1) {
        return input;
    }
    if (input.channels != 3) {
        throw std::invalid_argument("cudaGrayscale expects 3 channels (RGB)");
    }

    const int width = input.width;
    const int height = input.height;
    const size_t inputBytes = static_cast<size_t>(width * height * 3);
    const size_t outputBytes = static_cast<size_t>(width * height * 1);

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    CUDA_CHECK(cudaEventCreate(&startH2D));
    CUDA_CHECK(cudaEventCreate(&stopH2D));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startD2H));
    CUDA_CHECK(cudaEventCreate(&stopD2H));

    // 1. Host to Device
    CUDA_CHECK(cudaEventRecord(startH2D));
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), inputBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // 2. Kernel execution
    dim3 blockDim(16, 16);
    dim3 gridDim = calculateGridDim(width, height, blockDim);

    CUDA_CHECK(cudaEventRecord(startKernel));
    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stopKernel));

    // 3. Device to Host
    Image output(width, height, 1);
    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, outputBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopD2H));

    CUDA_CHECK(cudaEventSynchronize(stopD2H));

    if (timing) {
        float h2dMs = 0.0f, kernelMs = 0.0f, d2hMs = 0.0f;
        cudaEventElapsedTime(&h2dMs, startH2D, stopH2D);
        cudaEventElapsedTime(&kernelMs, startKernel, stopKernel);
        cudaEventElapsedTime(&d2hMs, startD2H, stopD2H);

        timing->h2dTimeMs = static_cast<double>(h2dMs);
        timing->kernelTimeMs = static_cast<double>(kernelMs);
        timing->d2hTimeMs = static_cast<double>(d2hMs);
        timing->pipelineTimeMs = static_cast<double>(h2dMs + kernelMs + d2hMs);
    }

    CUDA_CHECK(cudaEventDestroy(startH2D));
    CUDA_CHECK(cudaEventDestroy(stopH2D));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(startD2H));
    CUDA_CHECK(cudaEventDestroy(stopD2H));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}

} // namespace parallel

#endif
