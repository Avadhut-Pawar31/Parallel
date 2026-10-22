#include "parallel/image_ops.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace parallel {

Image cpuGaussianBlur(const Image& input, int kernelSize, float sigma) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid input image passed to cpuGaussianBlur");
    }

    if (kernelSize % 2 == 0 || kernelSize < 3) {
        kernelSize = 3; // Ensure odd and at least 3
    }
    if (sigma <= 0.0f) {
        sigma = 1.0f;
    }

    const int radius = kernelSize / 2;
    std::vector<float> kernel(static_cast<size_t>(kernelSize * kernelSize));
    float sum = 0.0f;

    // Generate normalized 2D Gaussian kernel
    const float twoSigmaSq = 2.0f * sigma * sigma;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const float val = std::exp(-static_cast<float>(x * x + y * y) / twoSigmaSq);
            kernel[(y + radius) * kernelSize + (x + radius)] = val;
            sum += val;
        }
    }
    for (auto& w : kernel) {
        w /= sum;
    }

    Image output(input.width, input.height, input.channels);
    const int w = input.width;
    const int h = input.height;
    const int c = input.channels;

    // 2D convolution with clamp-to-edge boundary policy
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky) {
                    const int py = std::clamp(y + ky, 0, h - 1);
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const int px = std::clamp(x + kx, 0, w - 1);
                        const float weight = kernel[(ky + radius) * kernelSize + (kx + radius)];
                        acc += static_cast<float>(input.at(px, py, ch)) * weight;
                    }
                }
                output.at(x, y, ch) = static_cast<uint8_t>(std::clamp(std::round(acc), 0.0f, 255.0f));
            }
        }
    }

    return output;
}

} // namespace parallel
