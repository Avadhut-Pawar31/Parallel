#include "parallel/image_ops.hpp"
#include <cmath>
#include <algorithm>

namespace parallel {

Image cpuSobel(const Image& input) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid input image passed to cpuSobel");
    }

    // Sobel requires single-channel luminance input
    Image gray = (input.channels == 1) ? input : cpuGrayscale(input);

    const int w = gray.width;
    const int h = gray.height;
    Image output(w, h, 1);

    // Sobel kernels
    constexpr int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    constexpr int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int gx = 0;
            int gy = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                const int py = std::clamp(y + ky, 0, h - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int px = std::clamp(x + kx, 0, w - 1);
                    const int pixel = static_cast<int>(gray.at(px, py, 0));
                    gx += pixel * Gx[ky + 1][kx + 1];
                    gy += pixel * Gy[ky + 1][kx + 1];
                }
            }

            const float magnitude = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            output.at(x, y, 0) = static_cast<uint8_t>(std::clamp(std::round(magnitude), 0.0f, 255.0f));
        }
    }

    return output;
}

} // namespace parallel
