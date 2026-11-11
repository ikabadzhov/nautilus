#include <cstdint>
#include <cstdio>

#if defined(__ACPP__) || defined(__HIPSYCL__)
#include <pcuda.hpp>
#else
#include <cuda_runtime.h>
#endif

// Kernel 1: Vector Addition
__global__ void vecAddKernel(const uint32_t* __restrict__ a,
                            const uint32_t* __restrict__ b,
                            uint32_t* __restrict__ c,
                            uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel 2: SAXPY
__global__ void saxpyKernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           float a,
                           uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

// Kernel 3: Multi-way Classification (Control Flow)
__global__ void classifyKernel(const uint32_t* __restrict__ in,
                              uint32_t* __restrict__ out,
                              uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t val = in[idx];
        uint32_t res = 0;
        if (val > 200) {
            res = 3;
        } else if (val > 100) {
            res = 2;
        } else if (val > 0) {
            res = 1;
        }
        out[idx] = res;
    }
}

// -----------------------------------------------------------------------------
// Synchronous API
// -----------------------------------------------------------------------------
extern "C" void runVecAddSync(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vecAddKernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

extern "C" void runSaxpySync(float* x, float* y, float a, uint32_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    saxpyKernel<<<grid, block>>>(x, y, a, n);
    cudaDeviceSynchronize();
}

extern "C" void runClassifySync(uint32_t* in, uint32_t* out, uint32_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    classifyKernel<<<grid, block>>>(in, out, n);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// Pipelined (Async) API for measuring kernel execution vs launch overhead
// -----------------------------------------------------------------------------
extern "C" void runVecAddPipelined(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t n, int iters) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    for (int i = 0; i < iters; ++i) {
        vecAddKernel<<<grid, block>>>(a, b, c, n);
    }
    cudaDeviceSynchronize();
}

extern "C" void runSaxpyPipelined(float* x, float* y, float a, uint32_t n, int iters) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    for (int i = 0; i < iters; ++i) {
        saxpyKernel<<<grid, block>>>(x, y, a, n);
    }
    cudaDeviceSynchronize();
}

extern "C" void runClassifyPipelined(uint32_t* in, uint32_t* out, uint32_t n, int iters) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    for (int i = 0; i < iters; ++i) {
        classifyKernel<<<grid, block>>>(in, out, n);
    }
    cudaDeviceSynchronize();
}
