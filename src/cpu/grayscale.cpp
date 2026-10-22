#include "parallel/image_ops.hpp"
#include <cmath>

namespace parallel {

Image cpuGrayscale(const Image& input) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid input image passed to cpuGrayscale");
    }

    if (input.channels == 1) {
        return input; // Already grayscale
    }

    if (input.channels != 3) {
        throw std::invalid_argument("cpuGrayscale expects 3 channels (RGB) or 1 channel, got: " + std::to_string(input.channels));
    }

    Image output(input.width, input.height, 1);

    const int totalPixels = input.width * input.height;
    const uint8_t* inPtr = input.data.data();
    uint8_t* outPtr = output.data.data();

    // Standard ITU-R BT.601 luminance coefficients:
    // Y = 0.299 * R + 0.587 * G + 0.114 * B
    for (int i = 0; i < totalPixels; ++i) {
        const float r = static_cast<float>(inPtr[i * 3 + 0]);
        const float g = static_cast<float>(inPtr[i * 3 + 1]);
        const float b = static_cast<float>(inPtr[i * 3 + 2]);

        const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        outPtr[i] = static_cast<uint8_t>(std::round(gray));
    }

    return output;
}

} // namespace parallel
