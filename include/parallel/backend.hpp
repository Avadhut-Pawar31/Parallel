#pragma once

#include <string>
#include <stdexcept>

namespace parallel {

enum class Backend {
    CPU,
    CUDA
};

inline std::string backendToString(Backend backend) {
    switch (backend) {
        case Backend::CPU:
            return "cpu";
        case Backend::CUDA:
            return "cuda";
        default:
            return "unknown";
    }
}

inline Backend stringToBackend(const std::string& str) {
    if (str == "cpu" || str == "CPU") {
        return Backend::CPU;
    } else if (str == "cuda" || str == "CUDA" || str == "gpu" || str == "GPU") {
        return Backend::CUDA;
    }
    throw std::invalid_argument("Unknown backend: " + str + ". Supported backends: 'cpu', 'cuda'");
}

bool isCudaSupported();

} // namespace parallel
