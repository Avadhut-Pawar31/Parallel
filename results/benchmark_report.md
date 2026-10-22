# Parallel: Performance Benchmark Report

**Generated:** 2026-10-22 15:30:51  

**CPU Architecture:** `Intel(R) Core(TM) 5 210H`  

**GPU Architecture:** `N/A (CPU-only build)`  

**CUDA Runtime/Driver:** `N/A (CPU-only build)`  

**Active Mode:** `CPU Reference Mode`  


## 1. Summary of Measured Operations

| Operation | Resolution | Pixels | Repetitions | CPU Median (ms) | CPU Min (ms) | CPU Max (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Grayscale | 512x512 | 262,144 | 10 | 0.004 | 0.004 | 0.005 |
| Blur | 512x512 | 262,144 | 10 | 3.318 | 3.092 | 4.119 |
| Sharpen | 512x512 | 262,144 | 10 | 0.860 | 0.841 | 1.186 |
| Sobel | 512x512 | 262,144 | 10 | 1.798 | 1.715 | 2.267 |

## 2. Measurement Methodology

- **CPU Monotonic Clock:** High-resolution steady clock (`std::chrono::high_resolution_clock`).
- **Warmup Phase:** Unrecorded iterations executed prior to measurement to prime caches and instruction pipelines.
- **Statistical Metric:** Median execution time across configured repetitions to minimize transient OS scheduling noise.
- **GPU Metrics (when active):** Explicitly isolates device global memory transfers (H2D, D2H) from pure compute kernel execution using CUDA hardware events.
