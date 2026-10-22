#pragma once

#include <string>
#include <vector>

namespace parallel {

struct CliOptions {
    std::string command; // "process", "compare", "benchmark", "info", "help"
    std::string inputPath;
    std::string outputPath;
    std::string outputDir{"results"};
    std::string operation{"grayscale"};
    std::string backend{"cpu"};
    int kernelSize{3};
    float sigma{1.0f};
    int iterations{20};
    int warmup{3};
    bool jsonOutput{false};
    bool synthetic{false};
    int syntheticWidth{1920};
    int syntheticHeight{1080};
    double tolerance{2.0};
};

void printHelp();
void printVersion();
CliOptions parseArguments(int argc, char* argv[]);
int runCli(const CliOptions& options);

} // namespace parallel
