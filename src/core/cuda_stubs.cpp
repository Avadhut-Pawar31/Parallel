#include "parallel/backend.hpp"
#include "parallel/benchmark.hpp"

#if !defined(PARALLEL_ENABLE_CUDA)

namespace parallel {

bool isCudaSupported() {
    return false;
}

std::string getGpuModelName() {
    return "N/A (CPU-only build)";
}

std::string getCudaDriverVersion() {
    return "N/A (CPU-only build)";
}

} // namespace parallel

#endif
