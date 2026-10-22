cmake_minimum_required(VERSION 3.18)

project(Parallel
    VERSION 1.0.0
    DESCRIPTION "High-Performance GPU & CPU Image Processing Engine and Benchmarking Suite"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configurable options
option(PARALLEL_ENABLE_CUDA "Build Parallel with NVIDIA CUDA acceleration" OFF)
option(PARALLEL_BUILD_TESTS "Build Parallel C++ test suite" ON)

# Configure output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
set(PARALLEL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Core C++ sources
set(PARALLEL_CORE_SOURCES
    src/core/image_io.cpp
    src/core/image_ops.cpp
    src/core/benchmark.cpp
    src/core/cli.cpp
    src/cpu/grayscale.cpp
    src/cpu/gaussian_blur.cpp
    src/cpu/sharpen.cpp
    src/cpu/sobel.cpp
)

# Check and configure CUDA
if(PARALLEL_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(PARALLEL_ENABLE_CUDA)

        list(APPEND PARALLEL_CORE_SOURCES
            src/cuda/cuda_utils.cu
            src/cuda/grayscale.cu
            src/cuda/gaussian_blur.cu
            src/cuda/sharpen.cu
            src/cuda/sobel.cu
        )
        message(STATUS "CUDA backend ENABLED: Using ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "PARALLEL_ENABLE_CUDA requested, but CUDA compiler was not found. Falling back to CPU-only build.")
        list(APPEND PARALLEL_CORE_SOURCES src/core/cuda_stubs.cpp)
    endif()
else()
    message(STATUS "Building Parallel in CPU-only mode (PARALLEL_ENABLE_CUDA=OFF)")
    list(APPEND PARALLEL_CORE_SOURCES src/core/cuda_stubs.cpp)
endif()

# Create core static library
add_library(parallel_core STATIC ${PARALLEL_CORE_SOURCES})
target_include_directories(parallel_core PUBLIC ${PARALLEL_INCLUDE_DIRS})

# Target compiler options
if(MSVC)
    target_compile_options(parallel_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(parallel_core PRIVATE -Wall -Wextra -Wno-missing-field-initializers -O3)
endif()

# Create main executable
add_executable(parallel src/main.cpp)
target_link_libraries(parallel PRIVATE parallel_core)

# CTest and automated testing
if(PARALLEL_BUILD_TESTS)
    enable_testing()
    add_executable(test_cpu_ops tests/cpp/test_cpu_ops.cpp)
    target_link_libraries(test_cpu_ops PRIVATE parallel_core)
    add_test(NAME CpuOpsCorrectness COMMAND test_cpu_ops)
endif()

# Portable static linking on MinGW so executables run independently
if(MINGW)
    target_link_options(parallel PRIVATE -static -static-libgcc -static-libstdc++)
    if(PARALLEL_BUILD_TESTS)
        target_link_options(test_cpu_ops PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endif()
