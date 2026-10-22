#include "parallel/cli.hpp"
#include "parallel/backend.hpp"
#include "parallel/image_io.hpp"
#include "parallel/image_ops.hpp"
#include "parallel/benchmark.hpp"
#include "parallel/result.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace parallel {

void printVersion() {
    std::cout << "Parallel v1.0.0 (High-Performance GPU & CPU Image Processing Engine)\n";
#if defined(PARALLEL_ENABLE_CUDA)
    std::cout << "Built with CUDA support (CUDA Backend: ENABLED)\n";
#else
    std::cout << "Built in CPU-only mode (CUDA Backend: DISABLED)\n";
#endif
}

void printHelp() {
    std::cout << "Usage: parallel <command> [options]\n\n"
              << "Commands:\n"
              << "  process    Process an image using CPU or CUDA backend\n"
              << "  compare    Process an image on both CPU and GPU and verify correctness\n"
              << "  benchmark  Run performance benchmarks and export metrics to CSV\n"
              << "  info       Display detected hardware and backend availability\n"
              << "  help       Show this help message\n\n"
              << "Process Options:\n"
              << "  -i, --input <path>       Input image path\n"
              << "  -o, --output <path>      Output image destination path\n"
              << "  --op, --operation <name> Operation: grayscale, blur, sharpen, sobel (default: grayscale)\n"
              << "  -b, --backend <backend>  Backend: cpu, cuda (default: cpu)\n"
              << "  --kernel-size <int>      Filter kernel size for blur (default: 3)\n"
              << "  --sigma <float>          Gaussian sigma for blur (default: 1.0)\n"
              << "  --synthetic <WxH>        Generate synthetic image instead of file (e.g. 1920x1080)\n"
              << "  --json                   Output result in machine-readable JSON format\n\n"
              << "Compare Options:\n"
              << "  -i, --input <path>       Input image path\n"
              << "  --op, --operation <name> Operation: grayscale, blur, sharpen, sobel\n"
              << "  -d, --output-dir <dir>   Output directory for compared images (default: results/comparison)\n"
              << "  --tolerance <float>      Max allowable pixel difference (default: 2.0)\n\n"
              << "Benchmark Options:\n"
              << "  -i, --input <path>       Input image path (or omit to benchmark across resolutions)\n"
              << "  --op, --operation <name> Operation: grayscale, blur, sharpen, sobel, all (default: grayscale)\n"
              << "  -b, --backend <backend>  Backend: cpu, cuda, both (default: both)\n"
              << "  -n, --iterations <int>   Number of timed benchmark repetitions (default: 20)\n"
              << "  -w, --warmup <int>       Number of warm-up iterations (default: 3)\n"
              << "  -o, --output <path>      Output CSV filepath (default: results/benchmark.csv)\n\n"
              << "Examples:\n"
              << "  parallel process -i samples/sample.png --op grayscale -b cpu -o results/gray_cpu.png\n"
              << "  parallel compare -i samples/sample.png --op sobel\n"
              << "  parallel benchmark -i samples/sample.png --op all -b both -o results/benchmark.csv\n"
              << "  parallel info --json\n";
}

CliOptions parseArguments(int argc, char* argv[]) {
    CliOptions opt;
    if (argc < 2) {
        opt.command = "help";
        return opt;
    }

    opt.command = argv[1];
    if (opt.command == "-h" || opt.command == "--help" || opt.command == "help") {
        opt.command = "help";
        return opt;
    }
    if (opt.command == "-v" || opt.command == "--version" || opt.command == "version") {
        opt.command = "version";
        return opt;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i" || arg == "--input") {
            if (++i < argc) opt.inputPath = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) opt.outputPath = argv[i];
        } else if (arg == "-d" || arg == "--output-dir") {
            if (++i < argc) opt.outputDir = argv[i];
        } else if (arg == "--op" || arg == "--operation") {
            if (++i < argc) opt.operation = argv[i];
        } else if (arg == "-b" || arg == "--backend") {
            if (++i < argc) opt.backend = argv[i];
        } else if (arg == "--kernel-size") {
            if (++i < argc) opt.kernelSize = std::stoi(argv[i]);
        } else if (arg == "--sigma") {
            if (++i < argc) opt.sigma = std::stof(argv[i]);
        } else if (arg == "-n" || arg == "--iterations") {
            if (++i < argc) opt.iterations = std::stoi(argv[i]);
        } else if (arg == "-w" || arg == "--warmup") {
            if (++i < argc) opt.warmup = std::stoi(argv[i]);
        } else if (arg == "--tolerance") {
            if (++i < argc) opt.tolerance = std::stod(argv[i]);
        } else if (arg == "--synthetic") {
            opt.synthetic = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string res = argv[++i];
                size_t xPos = res.find('x');
                if (xPos != std::string::npos) {
                    opt.syntheticWidth = std::stoi(res.substr(0, xPos));
                    opt.syntheticHeight = std::stoi(res.substr(xPos + 1));
                }
            }
        } else if (arg == "--json") {
            opt.jsonOutput = true;
        } else if (arg == "-h" || arg == "--help") {
            opt.command = "help";
            return opt;
        }
    }

    return opt;
}

static int handleInfoCommand(const CliOptions& opt) {
    std::string cpuModel = getCpuModelName();
    std::string gpuModel = getGpuModelName();
    std::string cudaVer = getCudaDriverVersion();
    bool cudaAvailable = isCudaSupported();

    if (opt.jsonOutput) {
        std::cout << "{\n"
                  << "  \"application\": \"Parallel\",\n"
                  << "  \"version\": \"1.0.0\",\n"
                  << "  \"cpu_model\": \"" << cpuModel << "\",\n"
                  << "  \"gpu_model\": \"" << gpuModel << "\",\n"
                  << "  \"cuda_driver_version\": \"" << cudaVer << "\",\n"
                  << "  \"cuda_available\": " << (cudaAvailable ? "true" : "false") << ",\n"
#if defined(PARALLEL_ENABLE_CUDA)
                  << "  \"cuda_compiled\": true\n"
#else
                  << "  \"cuda_compiled\": false\n"
#endif
                  << "}\n";
    } else {
        std::cout << "========================================================\n"
                  << "  Parallel: System Environment & Hardware Configuration \n"
                  << "========================================================\n"
                  << "  CPU Model:             " << cpuModel << "\n"
                  << "  GPU Model:             " << gpuModel << "\n"
                  << "  CUDA Driver Version:   " << cudaVer << "\n"
                  << "  CUDA Hardware Active:  " << (cudaAvailable ? "YES" : "NO") << "\n"
#if defined(PARALLEL_ENABLE_CUDA)
                  << "  CUDA Engine Compiled:  YES\n"
#else
                  << "  CUDA Engine Compiled:  NO (CPU-only build)\n"
#endif
                  << "  Supported Operations:  grayscale, blur, sharpen, sobel\n"
                  << "========================================================\n";
    }
    return 0;
}

static int handleProcessCommand(const CliOptions& opt) {
    Image input;
    if (opt.synthetic) {
        input = generateSyntheticImage(opt.syntheticWidth, opt.syntheticHeight, 3);
    } else {
        if (opt.inputPath.empty()) {
            std::cerr << "Error: --input <path> is required for process command\n";
            return 1;
        }
        try {
            input = loadImage(opt.inputPath);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }

    Backend backend;
    try {
        backend = stringToBackend(opt.backend);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (backend == Backend::CUDA && !isCudaSupported()) {
        std::cerr << "Error: CUDA backend was requested, but no CUDA-capable GPU or driver is active.\n";
        return 2;
    }

    OperationResult result;
    Image output;
    try {
        output = executeOperation(input, opt.operation, backend, &result, opt.kernelSize, opt.sigma);
    } catch (const std::exception& e) {
        std::cerr << "Processing failure: " << e.what() << "\n";
        return 3;
    }

    if (!opt.outputPath.empty()) {
        // Ensure parent directory exists
        fs::path p(opt.outputPath);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        if (!saveImage(opt.outputPath, output)) {
            std::cerr << "Error: Failed to save output image to: " << opt.outputPath << "\n";
            return 4;
        }
    }

    if (opt.jsonOutput) {
        std::cout << result.toJson() << "\n";
    } else {
        std::cout << "[Parallel] Process complete:\n"
                  << "  Operation:        " << opt.operation << "\n"
                  << "  Backend:          " << opt.backend << "\n"
                  << "  Resolution:       " << output.width << "x" << output.height << " (" << output.channels << " channels)\n"
                  << "  Execution Time:   " << std::fixed << std::setprecision(3) << result.executionTimeMs << " ms\n";
        if (backend == Backend::CUDA) {
            std::cout << "    - H2D Transfer: " << result.h2dTimeMs << " ms\n"
                      << "    - Kernel Only:  " << result.kernelTimeMs << " ms\n"
                      << "    - D2H Transfer: " << result.d2hTimeMs << " ms\n"
                      << "    - GPU Pipeline: " << result.pipelineTimeMs << " ms\n";
        }
        if (!opt.outputPath.empty()) {
            std::cout << "  Output saved to:  " << opt.outputPath << "\n";
        }
    }

    return 0;
}

static int handleCompareCommand(const CliOptions& opt) {
    if (opt.inputPath.empty()) {
        std::cerr << "Error: --input <path> is required for compare command\n";
        return 1;
    }

    Image input;
    try {
        input = loadImage(opt.inputPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (!isCudaSupported()) {
        std::cout << "Notice: CUDA is not active on this system. Comparing CPU output with CPU ground truth.\n";
    }

    OperationResult cpuTiming;
    Image cpuOutput = executeOperation(input, opt.operation, Backend::CPU, &cpuTiming, opt.kernelSize, opt.sigma);

    OperationResult gpuTiming;
    Image gpuOutput;
    bool gpuRan = false;

    if (isCudaSupported()) {
        try {
            gpuOutput = executeOperation(input, opt.operation, Backend::CUDA, &gpuTiming, opt.kernelSize, opt.sigma);
            gpuRan = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: GPU execution failed during comparison: " << e.what() << "\n";
        }
    }

    fs::path dir(opt.outputDir.empty() ? "results/comparison" : opt.outputDir);
    fs::create_directories(dir);

    std::string cpuOutPath = (dir / (opt.operation + "_cpu.png")).string();
    saveImage(cpuOutPath, cpuOutput);

    if (gpuRan) {
        std::string gpuOutPath = (dir / (opt.operation + "_cuda.png")).string();
        saveImage(gpuOutPath, gpuOutput);

        ImageDifference diff = compareImages(cpuOutput, gpuOutput, opt.tolerance);

        if (opt.jsonOutput) {
            std::cout << "{\n"
                      << "  \"operation\": \"" << opt.operation << "\",\n"
                      << "  \"verified\": " << (diff.passed ? "true" : "false") << ",\n"
                      << "  \"mean_absolute_error\": " << diff.meanAbsoluteError << ",\n"
                      << "  \"max_absolute_error\": " << diff.maxAbsoluteError << ",\n"
                      << "  \"differing_pixels\": " << diff.differingPixels << ",\n"
                      << "  \"total_pixels\": " << diff.totalPixels << ",\n"
                      << "  \"cpu_time_ms\": " << cpuTiming.executionTimeMs << ",\n"
                      << "  \"gpu_pipeline_time_ms\": " << gpuTiming.pipelineTimeMs << ",\n"
                      << "  \"gpu_kernel_time_ms\": " << gpuTiming.kernelTimeMs << "\n"
                      << "}\n";
        } else {
            std::cout << "========================================================\n"
                      << "  Parallel: CPU vs. GPU Correctness Verification        \n"
                      << "========================================================\n"
                      << "  Operation:             " << opt.operation << "\n"
                      << "  Dimensions:            " << input.width << "x" << input.height << "\n"
                      << "  Mean Absolute Error:   " << std::fixed << std::setprecision(4) << diff.meanAbsoluteError << "\n"
                      << "  Max Absolute Error:    " << diff.maxAbsoluteError << " (Tolerance: " << opt.tolerance << ")\n"
                      << "  Differing Pixels:      " << diff.differingPixels << " / " << diff.totalPixels << "\n"
                      << "  Status:                " << (diff.passed ? "[PASSED - Outputs Match]" : "[FAILED - Tolerance Exceeded]") << "\n"
                      << "--------------------------------------------------------\n"
                      << "  CPU Time:              " << std::setprecision(3) << cpuTiming.executionTimeMs << " ms\n"
                      << "  GPU Pipeline Time:     " << gpuTiming.pipelineTimeMs << " ms\n"
                      << "  GPU Kernel Only:       " << gpuTiming.kernelTimeMs << " ms\n"
                      << "  Kernel Speedup:        " << (gpuTiming.kernelTimeMs > 0 ? (cpuTiming.executionTimeMs / gpuTiming.kernelTimeMs) : 0.0) << "x\n"
                      << "  Pipeline Speedup:      " << (gpuTiming.pipelineTimeMs > 0 ? (cpuTiming.executionTimeMs / gpuTiming.pipelineTimeMs) : 0.0) << "x\n"
                      << "========================================================\n";
        }
    } else {
        std::cout << "[Parallel] CPU processed successfully. GPU comparison skipped because CUDA is unavailable.\n"
                  << "CPU output saved to: " << cpuOutPath << "\n";
    }

    return 0;
}

static int handleBenchmarkCommand(const CliOptions& opt) {
    std::vector<Image> benchmarkImages;
    if (opt.synthetic) {
        benchmarkImages.push_back(generateSyntheticImage(opt.syntheticWidth, opt.syntheticHeight, 3));
    } else if (!opt.inputPath.empty()) {
        try {
            benchmarkImages.push_back(loadImage(opt.inputPath));
        } catch (const std::exception& e) {
            std::cerr << "Error loading benchmark image: " << e.what() << "\n";
            return 1;
        }
    } else {
        // Multi-resolution test suite (256x256, 512x512, 1080p, 4K)
        std::cout << "[Parallel] Generating standard multi-resolution benchmark suite...\n";
        benchmarkImages.push_back(generateSyntheticImage(256, 256, 3));
        benchmarkImages.push_back(generateSyntheticImage(512, 512, 3));
        benchmarkImages.push_back(generateSyntheticImage(1280, 720, 3));
        benchmarkImages.push_back(generateSyntheticImage(1920, 1080, 3));
        benchmarkImages.push_back(generateSyntheticImage(3840, 2160, 3));
    }

    std::vector<std::string> opsToRun;
    if (opt.operation == "all") {
        opsToRun = {"grayscale", "blur", "sharpen", "sobel"};
    } else {
        opsToRun = {opt.operation};
    }

    std::vector<BenchmarkRecord> allRecords;
    bool runCpu = (opt.backend == "cpu" || opt.backend == "both");
    bool runCuda = (opt.backend == "cuda" || opt.backend == "both") && isCudaSupported();

    for (const auto& img : benchmarkImages) {
        for (const auto& op : opsToRun) {
            std::cout << "[Benchmarking] Operation: " << std::left << std::setw(10) << op
                      << " Resolution: " << img.width << "x" << img.height << " ... " << std::flush;

            BenchmarkRecord cpuRec;
            if (runCpu) {
                cpuRec = BenchmarkRunner::run(img, op, Backend::CPU, opt.iterations, opt.warmup);
            }

            BenchmarkRecord gpuRec;
            if (runCuda) {
                gpuRec = BenchmarkRunner::run(img, op, Backend::CUDA, opt.iterations, opt.warmup);
            }

            if (runCpu && runCuda && gpuRec.gpuKernelMedianMs > 0.0) {
                gpuRec.cpuMedianMs = cpuRec.cpuMedianMs;
                gpuRec.cpuMinMs = cpuRec.cpuMinMs;
                gpuRec.cpuMaxMs = cpuRec.cpuMaxMs;
                gpuRec.speedupKernel = cpuRec.cpuMedianMs / gpuRec.gpuKernelMedianMs;
                gpuRec.speedupPipeline = cpuRec.cpuMedianMs / (gpuRec.gpuPipelineMedianMs > 0 ? gpuRec.gpuPipelineMedianMs : 1.0);
            }

            if (runCpu) allRecords.push_back(cpuRec);
            if (runCuda) allRecords.push_back(gpuRec);

            std::cout << "Done (CPU: " << std::fixed << std::setprecision(2) << cpuRec.cpuMedianMs << " ms";
            if (runCuda) {
                std::cout << ", GPU Kernel: " << gpuRec.gpuKernelMedianMs << " ms, Speedup: "
                          << gpuRec.speedupKernel << "x";
            }
            std::cout << ")\n";
        }
    }

    std::string csvOut = opt.outputPath.empty() ? "results/benchmark.csv" : opt.outputPath;
    fs::path p(csvOut);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    if (BenchmarkRunner::exportCsv(csvOut, allRecords, false)) {
        std::cout << "\n[Parallel] Benchmark completed successfully!\n"
                  << "  Total measurements: " << allRecords.size() << "\n"
                  << "  CSV exported to:    " << csvOut << "\n";
    }

    return 0;
}

int runCli(const CliOptions& opt) {
    if (opt.command == "help") {
        printHelp();
        return 0;
    }
    if (opt.command == "version") {
        printVersion();
        return 0;
    }
    if (opt.command == "info") {
        return handleInfoCommand(opt);
    }
    if (opt.command == "process") {
        return handleProcessCommand(opt);
    }
    if (opt.command == "compare") {
        return handleCompareCommand(opt);
    }
    if (opt.command == "benchmark") {
        return handleBenchmarkCommand(opt);
    }

    std::cerr << "Unknown command: '" << opt.command << "'. Run 'parallel help' for usage.\n";
    return 1;
}

} // namespace parallel
