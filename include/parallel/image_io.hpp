#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace parallel {

struct Image {
    int width{0};
    int height{0};
    int channels{0};
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c), data(static_cast<size_t>(w * h * c), 0) {}
    Image(int w, int h, int c, std::vector<uint8_t> d)
        : width(w), height(h), channels(c), data(std::move(d)) {}

    bool empty() const {
        return width <= 0 || height <= 0 || channels <= 0 || data.empty();
    }

    size_t size() const {
        return static_cast<size_t>(width * height * channels);
    }

    bool isValid() const {
        return !empty() && data.size() == size();
    }

    inline int clampX(int x) const {
        return std::max(0, std::min(x, width - 1));
    }

    inline int clampY(int y) const {
        return std::max(0, std::min(y, height - 1));
    }

    inline size_t index(int x, int y, int c = 0) const {
        return static_cast<size_t>((y * width + x) * channels + c);
    }

    inline uint8_t at(int x, int y, int c = 0) const {
        return data[index(x, y, c)];
    }

    inline uint8_t& at(int x, int y, int c = 0) {
        return data[index(x, y, c)];
    }

    inline uint8_t clampedAt(int x, int y, int c = 0) const {
        return data[index(clampX(x), clampY(y), c)];
    }
};

Image loadImage(const std::string& path);
bool saveImage(const std::string& path, const Image& image);

// Procedural image generator for tests and benchmarks without disk images
Image generateSyntheticImage(int width, int height, int channels = 3);

} // namespace parallel
