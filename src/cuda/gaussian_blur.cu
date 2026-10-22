#include "parallel/image_ops.hpp"
#include "cuda_utils.cuh"

#if defined(PARALLEL_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace parallel {

__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    const float* __restrict__ d_weights,
    int width,
    int height,
    int channels,
    int radius,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int ch = 0; ch < channels; ++ch) {
        float acc = 0.0f;

        for (int ky = -radius; ky <= radius; ++ky) {
            int py = min(max(y + ky, 0), height - 1);
            for (int kx = -radius; kx <= radius; ++kx) {
                int px = min(max(x + kx, 0), width - 1);

                int weightIdx = (ky + radius) * kernelSize + (kx + radius);
                float weight = d_weights[weightIdx];

                int pixelIdx = (py * width + px) * channels + ch;
                acc += static_cast<float>(d_input[pixelIdx]) * weight;
            }
        }

        int outIdx = (y * width + x) * channels + ch;
        d_output[outIdx] = static_cast<uint8_t>(min(max(rintf(acc), 0.0f), 255.0f));
    }
}

Image cudaGaussianBlur(const Image& input, int kernelSize, float sigma, OperationResult* timing) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid image passed to cudaGaussianBlur");
    }

    if (kernelSize % 2 == 0 || kernelSize < 3) {
        kernelSize = 3;
    }
    if (sigma <= 0.0f) {
        sigma = 1.0f;
    }

    const int radius = kernelSize / 2;
    std::vector<float> h_weights(kernelSize * kernelSize);
    float sum = 0.0f;
    const float twoSigmaSq = 2.0f * sigma * sigma;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float val = std::exp(-static_cast<float>(x * x + y * y) / twoSigmaSq);
            h_weights[(y + radius) * kernelSize + (x + radius)] = val;
            sum += val;
        }
    }
    for (auto& w : h_weights) {
        w /= sum;
    }

    const int width = input.width;
    const int height = input.height;
    const int channels = input.channels;
    const size_t imageBytes = input.size();
    const size_t weightsBytes = h_weights.size() * sizeof(float);

    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    float* d_weights = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, imageBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imageBytes));
    CUDA_CHECK(cudaMalloc(&d_weights, weightsBytes));

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
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weightsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // 2. Kernel execution
    dim3 blockDim(16, 16);
    dim3 gridDim = calculateGridDim(width, height, blockDim);

    CUDA_CHECK(cudaEventRecord(startKernel));
    gaussianBlurKernel<<<gridDim, blockDim>>>(d_input, d_output, d_weights, width, height, channels, radius, kernelSize);
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
    CUDA_CHECK(cudaFree(d_weights));

    return output;
}

} // namespace parallel

#endif
