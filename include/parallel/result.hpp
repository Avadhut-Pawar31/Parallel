#pragma once

#include <string>
#include <sstream>
#include <iomanip>

namespace parallel {

struct OperationResult {
    bool success{false};
    std::string errorMessage;
    std::string operation;
    std::string backend;
    int width{0};
    int height{0};
    int channels{0};

    // Timings in milliseconds (ms)
    double executionTimeMs{0.0};  // Wall-clock processing time
    double kernelTimeMs{0.0};     // GPU Kernel execution only
    double h2dTimeMs{0.0};        // Host to Device transfer time
    double d2hTimeMs{0.0};        // Device to Host transfer time
    double pipelineTimeMs{0.0};   // Full GPU pipeline: H2D + Kernel + D2H

    std::string toJson() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        oss << "{\n"
            << "  \"success\": " << (success ? "true" : "false") << ",\n"
            << "  \"error\": \"" << errorMessage << "\",\n"
            << "  \"operation\": \"" << operation << "\",\n"
            << "  \"backend\": \"" << backend << "\",\n"
            << "  \"width\": " << width << ",\n"
            << "  \"height\": " << height << ",\n"
            << "  \"channels\": " << channels << ",\n"
            << "  \"execution_time_ms\": " << executionTimeMs << ",\n"
            << "  \"gpu_kernel_time_ms\": " << kernelTimeMs << ",\n"
            << "  \"gpu_h2d_time_ms\": " << h2dTimeMs << ",\n"
            << "  \"gpu_d2h_time_ms\": " << d2hTimeMs << ",\n"
            << "  \"gpu_pipeline_time_ms\": " << pipelineTimeMs << "\n"
            << "}";
        return oss.str();
    }
};

} // namespace parallel
