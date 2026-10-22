#include "parallel/benchmark.hpp"
#include "parallel/image_ops.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#endif

namespace parallel {

static std::string getIsoTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now_c);
#else
    localtime_r(&now_c, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

std::string getCpuModelName() {
#if defined(_WIN32)
    HKEY hKey;
    char cpuName[256] = "Unknown CPU";
    DWORD bufferSize = sizeof(cpuName);
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(cpuName), &bufferSize);
        RegCloseKey(hKey);
        // Trim leading spaces
        std::string name(cpuName);
        size_t first = name.find_first_not_of(' ');
        if (first != std::string::npos) {
            return name.substr(first);
        }
        return name;
    }
    return "x86_64 Processor";
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                return line.substr(colon + 2);
            }
        }
    }
    return "Linux CPU";
#else
    return "Generic CPU";
#endif
}

BenchmarkRecord BenchmarkRunner::run(
    const Image& input,
    const std::string& opName,
    Backend backend,
    int iterations,
    int warmup
) {
    if (!input.isValid()) {
        throw std::invalid_argument("Invalid image passed to BenchmarkRunner");
    }
    if (iterations <= 0) iterations = 10;
    if (warmup < 0) warmup = 2;

    BenchmarkRecord record;
    record.timestamp = getIsoTimestamp();
    record.operation = opName;
    record.backend = backendToString(backend);
    record.width = input.width;
    record.height = input.height;
    record.channels = input.channels;
    record.pixelCount = input.width * input.height;
    record.iterations = iterations;
    record.cpuModel = getCpuModelName();
    record.gpuModel = getGpuModelName();
    record.cudaVersion = getCudaDriverVersion();

    // 1. Warm-up iterations
    for (int i = 0; i < warmup; ++i) {
        OperationResult discardResult;
        executeOperation(input, opName, backend, &discardResult);
    }

    // 2. Timed iterations
    std::vector<double> cpuTimes;
    std::vector<double> gpuKernelTimes;
    std::vector<double> gpuPipelineTimes;
    std::vector<double> gpuH2dTimes;
    std::vector<double> gpuD2hTimes;

    for (int i = 0; i < iterations; ++i) {
        OperationResult timing;
        auto start = std::chrono::high_resolution_clock::now();
        Image out = executeOperation(input, opName, backend, &timing);
        auto end = std::chrono::high_resolution_clock::now();
        double wallClockMs = std::chrono::duration<double, std::milli>(end - start).count();

        if (backend == Backend::CPU) {
            cpuTimes.push_back(wallClockMs);
        } else {
            gpuKernelTimes.push_back(timing.kernelTimeMs);
            gpuH2dTimes.push_back(timing.h2dTimeMs);
            gpuD2hTimes.push_back(timing.d2hTimeMs);
            gpuPipelineTimes.push_back(timing.pipelineTimeMs > 0 ? timing.pipelineTimeMs : wallClockMs);
        }
    }

    if (backend == Backend::CPU && !cpuTimes.empty()) {
        std::sort(cpuTimes.begin(), cpuTimes.end());
        record.cpuMinMs = cpuTimes.front();
        record.cpuMaxMs = cpuTimes.back();
        record.cpuMedianMs = cpuTimes[cpuTimes.size() / 2];
    } else if (!gpuKernelTimes.empty()) {
        std::sort(gpuKernelTimes.begin(), gpuKernelTimes.end());
        std::sort(gpuPipelineTimes.begin(), gpuPipelineTimes.end());

        record.gpuKernelMinMs = gpuKernelTimes.front();
        record.gpuKernelMaxMs = gpuKernelTimes.back();
        record.gpuKernelMedianMs = gpuKernelTimes[gpuKernelTimes.size() / 2];

        record.gpuPipelineMedianMs = gpuPipelineTimes[gpuPipelineTimes.size() / 2];
        record.gpuH2dMs = gpuH2dTimes.empty() ? 0.0 : gpuH2dTimes[gpuH2dTimes.size() / 2];
        record.gpuD2hMs = gpuD2hTimes.empty() ? 0.0 : gpuD2hTimes[gpuD2hTimes.size() / 2];
    }

    return record;
}

bool BenchmarkRunner::exportCsv(
    const std::string& csvPath,
    const std::vector<BenchmarkRecord>& records,
    bool append
) {
    if (records.empty()) return true;

    bool fileExists = false;
    {
        std::ifstream check(csvPath);
        fileExists = check.good();
    }

    std::ofstream out(csvPath, (append && fileExists) ? std::ios::app : std::ios::out);
    if (!out.is_open()) {
        std::cerr << "Failed to open CSV for writing: " << csvPath << std::endl;
        return false;
    }

    if (!fileExists || !append) {
        out << "timestamp,operation,backend,width,height,channels,pixel_count,iterations,"
            << "cpu_median_ms,cpu_min_ms,cpu_max_ms,"
            << "gpu_kernel_median_ms,gpu_kernel_min_ms,gpu_kernel_max_ms,"
            << "gpu_h2d_ms,gpu_d2h_ms,gpu_pipeline_median_ms,"
            << "speedup_kernel,speedup_pipeline,"
            << "cpu_model,gpu_model,cuda_version\n";
    }

    out << std::fixed << std::setprecision(4);
    for (const auto& r : records) {
        out << r.timestamp << ","
            << r.operation << ","
            << r.backend << ","
            << r.width << ","
            << r.height << ","
            << r.channels << ","
            << r.pixelCount << ","
            << r.iterations << ","
            << r.cpuMedianMs << ","
            << r.cpuMinMs << ","
            << r.cpuMaxMs << ","
            << r.gpuKernelMedianMs << ","
            << r.gpuKernelMinMs << ","
            << r.gpuKernelMaxMs << ","
            << r.gpuH2dMs << ","
            << r.gpuD2hMs << ","
            << r.gpuPipelineMedianMs << ","
            << r.speedupKernel << ","
            << r.speedupPipeline << ",\""
            << r.cpuModel << "\",\""
            << r.gpuModel << "\",\""
            << r.cudaVersion << "\"\n";
    }

    return true;
}

} // namespace parallel
