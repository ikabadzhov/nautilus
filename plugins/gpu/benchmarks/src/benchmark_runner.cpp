#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using VecAddSyncFn = void (*)(uint32_t*, uint32_t*, uint32_t*, uint32_t);
using VecAddPipeFn = void (*)(uint32_t*, uint32_t*, uint32_t*, uint32_t, int);

using SaxpySyncFn = void (*)(float*, float*, float, uint32_t);
using SaxpyPipeFn = void (*)(float*, float*, float, uint32_t, int);

using ClassifySyncFn = void (*)(uint32_t*, uint32_t*, uint32_t);
using ClassifyPipeFn = void (*)(uint32_t*, uint32_t*, uint32_t, int);

struct ToolchainLib {
    std::string name;
    void* handle;
    VecAddSyncFn vecAddSync;
    VecAddPipeFn vecAddPipe;
    SaxpySyncFn saxpySync;
    SaxpyPipeFn saxpyPipe;
    ClassifySyncFn classifySync;
    ClassifyPipeFn classifyPipe;
};

struct Stats {
    double mean;
    double stdDev;
};

Stats computeStats(const std::vector<double>& vals) {
    if (vals.empty()) return {0.0, 0.0};
    double sum = 0.0;
    for (double v : vals) sum += v;
    double mean = sum / vals.size();
    double varSum = 0.0;
    for (double v : vals) {
        varSum += (v - mean) * (v - mean);
    }
    double stdDev = std::sqrt(varSum / vals.size());
    return {mean, stdDev};
}

int main(int argc, char** argv) {
    // Determine output data directory
    fs::path baseDir = fs::current_path();
    fs::path outJsonPath = baseDir / "data" / "exec_bench_stats.json";
    if (argc > 1) {
        outJsonPath = argv[1];
    }
    fs::create_directories(outJsonPath.parent_path());

    std::vector<std::pair<std::string, std::string>> libConfigs = {
        {"Native NVCC", (baseDir / "bin" / "libbench_nvcc.so").string()},
        {"ACPP PCUDA",  (baseDir / "bin" / "libbench_pcuda.so").string()},
        {"ACPP Native", (baseDir / "bin" / "libbench_native.so").string()}
    };

    std::vector<ToolchainLib> toolchains;
    for (const auto& [name, path] : libConfigs) {
        if (!fs::exists(path)) {
            std::cout << "[INFO] " << path << " not found, skipping " << name << "\n";
            continue;
        }
        void* h = dlopen(path.c_str(), RTLD_NOW);
        if (!h) {
            std::cerr << "Failed to dlopen " << path << ": " << dlerror() << "\n";
            continue;
        }
        ToolchainLib tl;
        tl.name = name;
        tl.handle = h;
        tl.vecAddSync = (VecAddSyncFn)dlsym(h, "runVecAddSync");
        tl.vecAddPipe = (VecAddPipeFn)dlsym(h, "runVecAddPipelined");
        tl.saxpySync = (SaxpySyncFn)dlsym(h, "runSaxpySync");
        tl.saxpyPipe = (SaxpyPipeFn)dlsym(h, "runSaxpyPipelined");
        tl.classifySync = (ClassifySyncFn)dlsym(h, "runClassifySync");
        tl.classifyPipe = (ClassifyPipeFn)dlsym(h, "runClassifyPipelined");
        toolchains.push_back(tl);
    }

    if (toolchains.empty()) {
        std::cerr << "[ERROR] No toolchain libraries could be loaded.\n";
        return 1;
    }

    std::vector<std::string> benchmarks = {"VecAdd", "SAXPY", "Classify"};
    std::vector<std::pair<std::string, uint32_t>> scales = {
        {"1M", 1000000},
        {"10M", 10000000},
        {"100M", 100000000}
    };

    constexpr int NUM_REPS = 40;
    constexpr int PIPE_ITERS = 20;

    std::cout << "Starting Execution Benchmarks (" << NUM_REPS << " reps per configuration)...\n";

    std::ofstream outJson(outJsonPath);
    outJson << "{\n";

    for (size_t bIdx = 0; bIdx < benchmarks.size(); ++bIdx) {
        const auto& bName = benchmarks[bIdx];
        outJson << "  \"" << bName << "\": {\n";

        for (size_t sIdx = 0; sIdx < scales.size(); ++sIdx) {
            const auto& [sName, N] = scales[sIdx];
            outJson << "    \"" << sName << "\": {\n";

            // Allocate Unified Shared Memory
            uint32_t *u_a = nullptr, *u_b = nullptr, *u_c = nullptr;
            float *f_x = nullptr, *f_y = nullptr;
            cudaMallocManaged(&u_a, N * sizeof(uint32_t));
            cudaMallocManaged(&u_b, N * sizeof(uint32_t));
            cudaMallocManaged(&u_c, N * sizeof(uint32_t));
            cudaMallocManaged(&f_x, N * sizeof(float));
            cudaMallocManaged(&f_y, N * sizeof(float));

            for (uint32_t i = 0; i < N; ++i) {
                u_a[i] = i % 255;
                u_b[i] = (i * 2) % 255;
                u_c[i] = 0;
                f_x[i] = static_cast<float>(i % 100);
                f_y[i] = static_cast<float>((i % 100) * 2);
            }
            cudaDeviceSynchronize();

            for (size_t tIdx = 0; tIdx < toolchains.size(); ++tIdx) {
                auto& tc = toolchains[tIdx];

                // Warmup
                if (bName == "VecAdd") tc.vecAddSync(u_a, u_b, u_c, N);
                else if (bName == "SAXPY") tc.saxpySync(f_x, f_y, 2.0f, N);
                else if (bName == "Classify") tc.classifySync(u_a, u_c, N);

                // Benchmark 1: Synchronous Latency (Pure Kernel + Launch Overhead)
                std::vector<double> syncTimesUs;
                syncTimesUs.reserve(NUM_REPS);
                for (int r = 0; r < NUM_REPS; ++r) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    if (bName == "VecAdd") tc.vecAddSync(u_a, u_b, u_c, N);
                    else if (bName == "SAXPY") tc.saxpySync(f_x, f_y, 2.0f, N);
                    else if (bName == "Classify") tc.classifySync(u_a, u_c, N);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    syncTimesUs.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }

                // Benchmark 2: Pipelined Latency (Isolates pure GPU execution time)
                std::vector<double> pipeTimesUs;
                pipeTimesUs.reserve(10);
                for (int r = 0; r < 10; ++r) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    if (bName == "VecAdd") tc.vecAddPipe(u_a, u_b, u_c, N, PIPE_ITERS);
                    else if (bName == "SAXPY") tc.saxpyPipe(f_x, f_y, 2.0f, N, PIPE_ITERS);
                    else if (bName == "Classify") tc.classifyPipe(u_a, u_c, N, PIPE_ITERS);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    pipeTimesUs.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / PIPE_ITERS);
                }

                Stats syncStats = computeStats(syncTimesUs);
                Stats pipeStats = computeStats(pipeTimesUs);
                double pureGpuUs = pipeStats.mean;
                double overheadUs = std::max(0.0, syncStats.mean - pureGpuUs);

                std::cout << "[" << std::setw(8) << bName << "] ["
                          << std::setw(4) << sName << "] ["
                          << std::setw(12) << tc.name << "] Total: "
                          << std::fixed << std::setprecision(1) << syncStats.mean << " ± " << syncStats.stdDev
                          << " µs (Pure GPU: " << pureGpuUs << " µs, Overhead: " << overheadUs << " µs)\n";

                outJson << "      \"" << tc.name << "\": {\n"
                        << "        \"total_mean_us\": " << syncStats.mean << ",\n"
                        << "        \"total_std_us\": " << syncStats.stdDev << ",\n"
                        << "        \"pure_gpu_mean_us\": " << pureGpuUs << ",\n"
                        << "        \"pure_gpu_std_us\": " << pipeStats.stdDev << ",\n"
                        << "        \"overhead_mean_us\": " << overheadUs << "\n"
                        << "      }" << (tIdx + 1 < toolchains.size() ? "," : "") << "\n";
            }

            cudaFree(u_a);
            cudaFree(u_b);
            cudaFree(u_c);
            cudaFree(f_x);
            cudaFree(f_y);

            outJson << "    }" << (sIdx + 1 < scales.size() ? "," : "") << "\n";
        }
        outJson << "  }" << (bIdx + 1 < benchmarks.size() ? "," : "") << "\n";
    }
    outJson << "}\n";
    outJson.close();

    std::cout << "\nResults saved to: " << outJsonPath << "\n";
    return 0;
}
