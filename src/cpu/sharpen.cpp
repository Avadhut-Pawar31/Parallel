#include "parallel/image_ops.hpp"
#include <algorithm>

namespace parallel {

Image cpuSharpen(const Image& input) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid input image passed to cpuSharpen");
    }

    // Standard 3x3 Laplacian sharpening kernel (sum of weights = 1)
    // [  0, -1,  0 ]
    // [ -1,  5, -1 ]
    // [  0, -1,  0 ]
    constexpr int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    Image output(input.width, input.height, input.channels);
    const int w = input.width;
    const int h = input.height;
    const int c = input.channels;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    const int py = std::clamp(y + ky, 0, h - 1);
                    for (int kx = -1; kx <= 1; ++kx) {
                        const int px = std::clamp(x + kx, 0, w - 1);
                        sum += static_cast<int>(input.at(px, py, ch)) * kernel[ky + 1][kx + 1];
                    }
                }
                output.at(x, y, ch) = static_cast<uint8_t>(std::clamp(sum, 0, 255));
            }
        }
    }

    return output;
}

} // namespace parallel
