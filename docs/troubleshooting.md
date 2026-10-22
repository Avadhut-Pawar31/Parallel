# Troubleshooting & Environment Setup

## 1. Building on Windows (Native)

### Requirements
- **CMake** $\ge 3.18$
- **C++17 Compiler** (GCC 12+, Clang 15+, or MSVC 2019/2022)
- **NVIDIA CUDA Toolkit** (Optional for GPU mode, 11.x - 13.x)

### Compiling in CPU-Only Mode
```powershell
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Compiling with CUDA Acceleration
To enable the CUDA backend on Windows, ensure `nvcc` and the MSVC C++ toolchain are installed:
```powershell
cmake -B build -G "Ninja" -DPARALLEL_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 2. Fast Setup on Windows Subsystem for Linux (WSL2)

If you have an NVIDIA RTX GPU on Windows, WSL2 offers an instant, zero-friction POSIX development environment:
1. In Windows PowerShell:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
2. Open Ubuntu terminal:
   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake ninja-build nvidia-cuda-toolkit
   ```
3. Build Parallel with full CUDA support:
   ```bash
   git clone https://github.com/Avadhut-Pawar31/Parallel.git
   cd Parallel
   cmake -B build -DPARALLEL_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ```

## 3. Common Issues

### Issue: "CUDA backend is not available in this build"
- **Cause:** Binary was compiled with `PARALLEL_ENABLE_CUDA=OFF`.
- **Solution:** Reconfigure CMake on a system with the NVIDIA CUDA Toolkit and set `-DPARALLEL_ENABLE_CUDA=ON`.

### Issue: Missing DLLs when running `parallel.exe`
- **Solution:** Parallel's CMakeLists uses `-static -static-libgcc -static-libstdc++` on MinGW to bundle runtime requirements directly into the binary.
