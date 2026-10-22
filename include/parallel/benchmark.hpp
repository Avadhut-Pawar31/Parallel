#pragma once

#include "parallel/backend.hpp"
#include "parallel/image_io.hpp"
#include <string>
#include <vector>
#include <chrono>

namespace parallel {

struct BenchmarkRecord {
    std::string timestamp;
    std::string operation;
    std::string backend;
    int width{0};
    int height{0};
    int channels{0};
    int pixelCount{0};
    int iterations{0};

    // Measurements in milliseconds
    double cpuMedianMs{0.0};
    double cpuMinMs{0.0};
    double cpuMaxMs{0.0};

    double gpuKernelMedianMs{0.0};
    double gpuKernelMinMs{0.0};
    double gpuKernelMaxMs{0.0};

    double gpuH2dMs{0.0};
    double gpuD2hMs{0.0};
    double gpuPipelineMedianMs{0.0};

    // Speedup ratios: cpu / gpu
    double speedupKernel{0.0};
    double speedupPipeline{0.0};

    // Environmental metadata
    std::string cpuModel;
    std::string gpuModel;
    std::string cudaVersion;
};

std::string getCpuModelName();
std::string getGpuModelName();
std::string getCudaDriverVersion();

class BenchmarkRunner {
public:
    static BenchmarkRecord run(
        const Image& input,
        const std::string& opName,
        Backend backend,
        int iterations = 20,
        int warmup = 3
    );

    static bool exportCsv(
        const std::string& csvPath,
        const std::vector<BenchmarkRecord>& records,
        bool append = true
    );
};

} // namespace parallel
