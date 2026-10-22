#include "parallel/image_ops.hpp"
#include "cuda_utils.cuh"

#if defined(PARALLEL_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cmath>

namespace parallel {

__global__ void sobelKernel(
    const uint8_t* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sobel Gx and Gy kernels
    // Gx: [-1, 0, 1]      Gy: [-1, -2, -1]
    //     [-2, 0, 2]          [ 0,  0,  0]
    //     [-1, 0, 1]          [ 1,  2,  1]

    int x0 = max(x - 1, 0);
    int x1 = min(x + 1, width - 1);
    int y0 = max(y - 1, 0);
    int y1 = min(y + 1, height - 1);

    int p00 = static_cast<int>(d_input[y0 * width + x0]);
    int p01 = static_cast<int>(d_input[y0 * width + x ]);
    int p02 = static_cast<int>(d_input[y0 * width + x1]);

    int p10 = static_cast<int>(d_input[y  * width + x0]);
    int p12 = static_cast<int>(d_input[y  * width + x1]);

    int p20 = static_cast<int>(d_input[y1 * width + x0]);
    int p21 = static_cast<int>(d_input[y1 * width + x ]);
    int p22 = static_cast<int>(d_input[y1 * width + x1]);

    int gx = (-1 * p00) + (1 * p02) +
             (-2 * p10) + (2 * p12) +
             (-1 * p20) + (1 * p22);

    int gy = (-1 * p00) + (-2 * p01) + (-1 * p02) +
             ( 1 * p20) + ( 2 * p21) + ( 1 * p22);

    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    d_output[y * width + x] = static_cast<uint8_t>(min(max(rintf(mag), 0.0f), 255.0f));
}

Image cudaSobel(const Image& input, OperationResult* timing) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid image passed to cudaSobel");
    }

    Image gray = (input.channels == 1) ? input : cudaGrayscale(input, nullptr);

    const int width = gray.width;
    const int height = gray.height;
    const size_t imageBytes = static_cast<size_t>(width * height);

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
    CUDA_CHECK(cudaMemcpy(d_input, gray.data.data(), imageBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stopH2D));

    // 2. Kernel execution
    dim3 blockDim(16, 16);
    dim3 gridDim = calculateGridDim(width, height, blockDim);

    CUDA_CHECK(cudaEventRecord(startKernel));
    sobelKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stopKernel));

    // 3. Device to Host
    Image output(width, height, 1);
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
