# Parallel ⚡

> **Harness parallel computing. Measure real performance.**

[![Parallel CI](https://github.com/Avadhut-Pawar31/Parallel/actions/workflows/ci.yml/badge.svg)](https://github.com/Avadhut-Pawar31/Parallel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

**Parallel** is a high-performance image-processing engine, benchmarking suite, and interactive visualization dashboard built from scratch in **C++17** and **NVIDIA CUDA C++**.

It empirically demonstrates the architectural, algorithmic, and throughput differences between sequential CPU computation and massively parallel GPU execution on real hardware.

---

## 🎯 Key Features

- **Dual-Engine Architecture:** Equivalent C++17 CPU baseline and NVIDIA CUDA C++ accelerated backends.
- **Four Core Image Algorithms:**
  - **Grayscale Conversion:** ITU-R BT.601 luminance weighting ($Y = 0.299R + 0.587G + 0.114B$).
  - **Gaussian Blur:** Parameterized 2D spatial convolution with dynamic normalization and clamp-to-edge boundary replication.
  - **Image Sharpening:** Laplacian high-pass filter ($3 \times 3$) emphasizing high-frequency edges.
  - **Sobel Edge Detection:** Orthogonal gradient convolution ($G_x, G_y$) calculating Euclidean edge magnitude.
- **Hardware-Aware CUDA Kernels:**
  - 2D grid/block thread decomposition ($16 \times 16 = 256$ threads per block for optimal warp occupancy).
  - Robust boundary guards supporting arbitrary, non-square, and odd image resolutions.
- **Empirical Benchmarking & Timing Rigor:**
  - Distinguishes CPU wall-clock, GPU kernel execution, and Host-to-Device (H2D) / Device-to-Host (D2H) memory transfer overhead using hardware CUDA events.
  - Generates structured CSV datasets and automated markdown benchmark reports.
- **Interactive Streamlit Dashboard:**
  - Dark charcoal technical UI inspired by GPU profiling tools.
  - Real-time image processing lab with side-by-side comparison.
  - Dynamic Plotly performance charts and dataset explorer.
- **Zero-Dependency CPU Fallback:**
  - Compiles and runs out-of-the-box in CPU-only mode without requiring external GPU hardware or heavy third-party DLLs.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / CLI / Dashboard]) --> CLI[Command-Line Interface / cli.cpp]
    CLI --> Dispatcher[Operation Dispatcher / image_ops.cpp]

    subgraph CPU_Engine [CPU Engine (C++17)]
        Dispatcher -->|Backend::CPU| CPU_Gray[cpuGrayscale]
        Dispatcher -->|Backend::CPU| CPU_Blur[cpuGaussianBlur]
        Dispatcher -->|Backend::CPU| CPU_Sharp[cpuSharpen]
        Dispatcher -->|Backend::CPU| CPU_Sobel[cpuSobel]
    end

    subgraph CUDA_Engine [CUDA GPU Engine (NVIDIA CUDA C++)]
        Dispatcher -->|Backend::CUDA| CUDA_H2D[Host-to-Device Memory Transfer]
        CUDA_H2D --> CUDA_Kernels[2D CUDA Kernels: 16x16 Blocks]
        CUDA_Kernels --> CUDA_D2H[Device-to-Host Memory Transfer]
    end

    CPU_Engine --> Verification[Pixel-by-Pixel Verification & Output]
    CUDA_Engine --> Verification
    Verification --> Benchmarks[Benchmark Engine & CSV Export]
    Benchmarks --> Dashboard[Streamlit Interactive Dashboard]
```

---

## 📊 Measured Benchmark Results

*Measured on an **Intel(R) Core(TM) 5 210H** (10 repetitions, median execution time):*

| Operation | Resolution | Pixel Count | CPU Median (ms) | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Grayscale** | $1920 \times 1080$ | 2,073,600 | **9.987 ms** | `results/sample_gray.png` |
| **Gaussian Blur** | $1920 \times 1080$ | 2,073,600 | **100.816 ms** | `results/sample_blur.png` |
| **Sharpen** | $1920 \times 1080$ | 2,073,600 | **26.911 ms** | `results/sample_sharp.png` |
| **Sobel Edge** | $1920 \times 1080$ | 2,073,600 | **22.399 ms** | `results/sample_sobel.png` |

> [!NOTE]
> GPU kernel speedup increases proportionally with image resolution. For small images ($< 512 \times 512$), memory transfer across the PCIe bus can exceed CPU computation time. On 1080p and 4K images, parallel thread saturation produces massive speedups.

---

## 🚀 Quick Start & Build Instructions

### Prerequisites
- **C++ Compiler:** GCC $\ge 12$, Clang $\ge 15$, or MSVC $\ge 2019$
- **Build System:** CMake $\ge 3.18$ and Ninja / Make
- **Python:** Python 3.10+ (for dashboard and visualization)
- **NVIDIA GPU:** (Optional) Compute Capability $\ge 5.0$ with CUDA Toolkit 12.x / 13.x

### 1. Build the C++ Engine (CPU Mode)
```bash
# Configure CMake
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# Compile binaries
cmake --build build

# Run automated C++ unit tests
./build/bin/test_cpu_ops
```

### 2. Build with CUDA GPU Acceleration
```bash
# Configure with CUDA enabled
cmake -B build -G "Ninja" -DPARALLEL_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## 💻 CLI Usage Examples

Parallel provides a clean command-line interface:

```bash
# 1. Inspect detected hardware and backend availability
./build/bin/parallel info
./build/bin/parallel info --json

# 2. Process an image using the CPU engine
./build/bin/parallel process -i samples/sample.png --op grayscale -b cpu -o results/gray_cpu.png

# 3. Process an image using CUDA GPU acceleration
./build/bin/parallel process -i samples/sample.png --op sobel -b cuda -o results/sobel_cuda.png

# 4. Compare CPU vs. GPU outputs and verify pixel-by-pixel correctness
./build/bin/parallel compare -i samples/sample.png --op blur --tolerance 2.0

# 5. Run performance benchmarks and export to CSV
./build/bin/parallel benchmark -i samples/sample.png --op all -b both -n 20 -o results/benchmark.csv
```

---

## 🌐 Launching the Interactive Dashboard

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501` to access:
- **System Overview:** Hardware specifications and backend availability.
- **Image Lab:** Interactive before-and-after image processing and download.
- **Performance Lab:** Live multi-resolution benchmark execution and Plotly latency curves.
- **Results Explorer:** CSV dataset filtering and statistics.
- **Documentation:** Engineering principles, CUDA warp theory, and mathematical proofs.

---

## 🧪 Testing & Verification

Run the full automated test suite:

```bash
# Run C++ test suite (8 tests covering dimensions, uniform pixels, odd aspect ratios, edge detection)
./build/bin/test_cpu_ops

# Run Python automated test suite (schema validation, CLI JSON parsing, plot generation)
pytest tests/python/ -v
```

---

## 📁 Repository Structure

```
parallel/
├── CMakeLists.txt          # CMake configuration (CPU & CUDA dual support)
├── CMakePresets.json       # Build presets
├── requirements.txt        # Runtime dependencies (Streamlit, Plotly, Pandas)
├── requirements-dev.txt    # Testing dependencies (pytest)
├── include/parallel/       # Public C++ headers
│   ├── backend.hpp         # Backend definitions (CPU vs CUDA)
│   ├── image_io.hpp        # Image buffer representations & I/O
│   ├── image_ops.hpp       # Algorithm declarations & verification utilities
│   ├── benchmark.hpp       # Benchmarking engine interface
│   └── cli.hpp             # Command-line interface definitions
├── src/
│   ├── core/               # Shared engine implementations
│   ├── cpu/                # C++17 CPU algorithm reference implementations
│   ├── cuda/               # NVIDIA CUDA C++ kernels & device pipelines
│   └── main.cpp            # Main executable entry point
├── dashboard/              # Streamlit interactive application
│   ├── app.py              # Main dashboard
│   └── components/         # Reusable dashboard components & runners
├── scripts/                # Visualization & reporting utilities
│   ├── generate_samples.py # Sample image generator
│   ├── plot_results.py     # Matplotlib benchmark visualizer
│   └── generate_report.py  # Markdown report generator
├── tests/
│   ├── cpp/                # C++ correctness unit tests
│   └── python/             # Pytest automated test suite
├── samples/                # Sample test images
├── results/                # Output images, CSVs, and charts
└── docs/                   # Technical documentation
    ├── architecture.md     # Memory layout & CUDA thread hierarchy
    ├── benchmarking.md     # Benchmarking methodology
    └── troubleshooting.md  # Toolchain & WSL2 setup guide
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
