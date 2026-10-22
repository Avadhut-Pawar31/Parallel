# Architecture of Parallel

Parallel is structured as a dual-backend image processing and performance benchmarking engine, built in C++17 and NVIDIA CUDA C++.

```mermaid
graph TD
    User([User / CLI / Dashboard]) --> CLI[Command-Line Interface / cli.cpp]
    CLI --> Dispatcher[Operation Dispatcher / image_ops.cpp]

    subgraph CPU_Backend [CPU Engine (C++17)]
        Dispatcher -->|Backend::CPU| CPU_Gray[cpuGrayscale]
        Dispatcher -->|Backend::CPU| CPU_Blur[cpuGaussianBlur]
        Dispatcher -->|Backend::CPU| CPU_Sharp[cpuSharpen]
        Dispatcher -->|Backend::CPU| CPU_Sobel[cpuSobel]
    end

    subgraph CUDA_Backend [CUDA Engine (NVIDIA CUDA C++)]
        Dispatcher -->|Backend::CUDA| CUDA_H2D[Host-to-Device Transfer]
        CUDA_H2D --> CUDA_Kernels[2D CUDA Kernels (16x16 Blocks)]
        CUDA_Kernels --> CUDA_D2H[Device-to-Host Transfer]
    end

    CPU_Backend --> Output[Image Output & Verification]
    CUDA_Backend --> Output
    Output --> Metrics[Benchmark & Timing Analysis]
    Metrics --> CSV[(results/benchmark.csv)]
    CSV --> Dashboard[Streamlit Dashboard / app.py]
```

## 1. Memory Layout and Thread Decompositions

Images in Parallel are represented as contiguous 1D byte arrays (`std::vector<uint8_t>`):
- Interleaved pixel layout: `(y * width + x) * channels + c`
- Channels: 1 (Grayscale/Edge magnitude) or 3 (RGB color)

### 2D CUDA Grid Decomposition
For GPU execution, the image space is partitioned into a two-dimensional grid of thread blocks:
- **Block Dimensions:** $16 \times 16 = 256$ threads per block.
- **Warp Alignment:** 256 is an exact multiple of the 32-thread hardware warp size, ensuring maximum warp occupancy and zero inactive lane divergence.
- **Grid Dimensions:**
  $$\text{gridDim.x} = \left\lceil \frac{\text{width}}{16} \right\rceil, \quad \text{gridDim.y} = \left\lceil \frac{\text{height}}{16} \right\rceil$$
- **Boundary Guards:** Every kernel enforces `if (x >= width || y >= height) return;` to support non-square, odd, and non-multiple image dimensions safely.

## 2. Algorithmic Equivalence & Precision

1. **Grayscale Conversion:**
   Uses standard ITU-R BT.601 luminance coefficients:
   $$Y = \text{round}(0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B)$$
   Both CPU and GPU apply floating-point multiplication with standard IEEE-754 half-to-even rounding (`std::round` / `rintf`).

2. **2D Spatial Convolutions:**
   - **Boundary Handling:** Clamp-to-edge policy replicate boundary values:
     $$x' = \max(0, \min(x, W-1)), \quad y' = \max(0, \min(y, H-1))$$
   - **Gaussian Blur:** Dynamically normalized 2D Gaussian kernel with configurable radius and $\sigma$.
   - **Sharpen:** Laplacian 4-neighbor high-pass filter:
     $$\begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
   - **Sobel Edge Detection:** Horizontal ($G_x$) and vertical ($G_y$) gradients combined via Euclidean norm $\min(255, \sqrt{G_x^2 + G_y^2})$.
