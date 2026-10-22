# Benchmarking Methodology in Parallel

Benchmarking parallel GPU compute kernels versus serial CPU implementations requires strict timing boundaries, clear definitions, and statistical rigor.

## 1. Timing Boundaries

Parallel isolates three distinct timing zones:

| Metric | Measurement Tool | Description |
| :--- | :--- | :--- |
| **CPU Time** | `std::chrono::high_resolution_clock` | Total wall-clock time for the CPU algorithm |
| **GPU Kernel Time** | `cudaEventElapsedTime(startKernel, stopKernel)` | Pure device execution time on CUDA cores |
| **GPU Memory Transfer** | `cudaEventElapsedTime` | Host-to-Device (H2D) and Device-to-Host (D2H) copy times |
| **GPU Pipeline Time** | Sum $(T_{\text{H2D}} + T_{\text{Kernel}} + T_{\text{D2H}})$ | Complete end-to-end device lifecycle |

## 2. Speedup Calculations

Speedup must specify which GPU denominator is being compared against CPU time:

1. **Kernel Speedup ($S_{\text{kernel}}$):**
   $$S_{\text{kernel}} = \frac{T_{\text{CPU}}}{T_{\text{GPU\_kernel}}}$$
   *Represents the raw computational throughput ratio of GPU hardware execution units vs CPU ALU.*

2. **Pipeline Speedup ($S_{\text{pipeline}}$):**
   $$S_{\text{pipeline}} = \frac{T_{\text{CPU}}}{T_{\text{GPU\_pipeline}}}$$
   *Represents real-world practical speedup including the PCIe bus transfer latency.*

## 3. Why GPUs Can Be Slower on Small Images

A common misconception among beginner engineers is that a GPU is "always faster" than a CPU. In practice:
- **PCIe Latency:** Copying memory over PCIe Gen3/Gen4 takes a base latency (~5-15 microseconds) plus bandwidth limits.
- **Kernel Launch Overhead:** Initiating a CUDA driver launch call introduces ~2-5 microseconds of host overhead.
- For small images ($< 512 \times 512$), a multi-core CPU can process the image within cache before the GPU even finishes transferring the input buffer.
- GPUs demonstrate massive orders-of-magnitude speedup as the problem size scales (1080p, 4K, 8K) where computational density amortizes the fixed transfer cost.
