#include "parallel/image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "parallel/third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "parallel/third_party/stb_image_write.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace parallel {

Image loadImage(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + path);
    }
    file.close();

    int w = 0, h = 0, c = 0;
    // Force 1 or 3 channels depending on input
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!raw) {
        throw std::runtime_error("Failed to decode image: " + path + " (Reason: " + stbi_failure_reason() + ")");
    }

    // If image has 4 channels (RGBA), convert to 3 channels (RGB) for consistent processing
    if (c == 4) {
        std::vector<uint8_t> rgbData(static_cast<size_t>(w * h * 3));
        for (int i = 0; i < w * h; ++i) {
            rgbData[i * 3 + 0] = raw[i * 4 + 0];
            rgbData[i * 3 + 1] = raw[i * 4 + 1];
            rgbData[i * 3 + 2] = raw[i * 4 + 2];
        }
        stbi_image_free(raw);
        return Image(w, h, 3, std::move(rgbData));
    }

    std::vector<uint8_t> data(raw, raw + (w * h * c));
    stbi_image_free(raw);
    return Image(w, h, c, std::move(data));
}

bool saveImage(const std::string& path, const Image& image) {
    if (!image.isValid()) {
        std::cerr << "Cannot save invalid image to " << path << std::endl;
        return false;
    }

    // Determine extension
    std::string ext;
    size_t dotPos = path.rfind('.');
    if (dotPos != std::string::npos) {
        ext = path.substr(dotPos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    int success = 0;
    if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(path.c_str(), image.width, image.height, image.channels, image.data.data(), 95);
    } else if (ext == ".bmp") {
        success = stbi_write_bmp(path.c_str(), image.width, image.height, image.channels, image.data.data());
    } else {
        // Default to PNG
        success = stbi_write_png(path.c_str(), image.width, image.height, image.channels, image.data.data(), image.width * image.channels);
    }

    return success != 0;
}

Image generateSyntheticImage(int width, int height, int channels) {
    Image img(width, height, channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = static_cast<float>(x) / static_cast<float>(width);
            float fy = static_cast<float>(y) / static_cast<float>(height);

            // Create circular ripple pattern combined with linear gradient
            float cx = fx - 0.5f;
            float cy = fy - 0.5f;
            float dist = std::sqrt(cx * cx + cy * cy);
            float ripple = 0.5f + 0.5f * std::sin(dist * 30.0f);

            uint8_t r = static_cast<uint8_t>(std::clamp(ripple * 255.0f, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(fx * 255.0f, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(fy * 255.0f, 0.0f, 255.0f));

            if (channels == 1) {
                img.at(x, y, 0) = r;
            } else {
                img.at(x, y, 0) = r;
                img.at(x, y, 1) = g;
                img.at(x, y, 2) = b;
            }
        }
    }
    return img;
}

} // namespace parallel
