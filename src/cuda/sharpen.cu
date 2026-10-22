#include "parallel/image_ops.hpp"
#include "cuda_utils.cuh"

#if defined(PARALLEL_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cmath>

namespace parallel {

__global__ void sharpenKernel(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Laplacian sharpen filter
    // [  0, -1,  0 ]
    // [ -1,  5, -1 ]
    // [  0, -1,  0 ]
    for (int ch = 0; ch < channels; ++ch) {
        int centerVal = static_cast<int>(d_input[(y * width + x) * channels + ch]);

        int topY    = max(y - 1, 0);
        int bottomY = min(y + 1, height - 1);
        int leftX   = max(x - 1, 0);
        int rightX  = min(x + 1, width - 1);

        int topVal    = static_cast<int>(d_input[(topY * width + x) * channels + ch]);
        int bottomVal = static_cast<int>(d_input[(bottomY * width + x) * channels + ch]);
        int leftVal   = static_cast<int>(d_input[(y * width + leftX) * channels + ch]);
        int rightVal  = static_cast<int>(d_input[(y * width + rightX) * channels + ch]);

        int result = (5 * centerVal) - topVal - bottomVal - leftVal - rightVal;

        int outIdx = (y * width + x) * channels + ch;
        d_output[outIdx] = static_cast<uint8_t>(min(max(result, 0), 255));
    }
}

Image cudaSharpen(const Image& input, OperationResult* timing) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid image passed to cudaSharpen");
    }

    const int width = input.width;
    const int height = input.height;
    const int channels = input.channels;
    const size_t imageBytes = input.size();

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, imageBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imageBytes));

    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    CUDA_CHECK(cudaEventCreate(&startH2D));
    CUDA_CHECK(cudaEventCreate(&stopH2D));
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startD2H));
    CUDA_CHECK(cudaEventCreate(&stopD2H));

    // 1. Host to Device
    CUDA_CHECK(cudaEventRecord(startH2D));
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), imageBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // 2. Kernel execution
    dim3 blockDim(16, 16);
    dim3 gridDim = calculateGridDim(width, height, blockDim);

    CUDA_CHECK(cudaEventRecord(startKernel));
    sharpenKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stopKernel));

    // 3. Device to Host
    Image output(width, height, channels);
    CUDA_CHECK(cudaEventRecord(startD2H));
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, imageBytes, cudaMemcpyDeviceToHost));
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
