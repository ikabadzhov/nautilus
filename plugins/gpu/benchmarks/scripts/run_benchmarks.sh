#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${BASE_DIR}/src"
BIN_DIR="${BASE_DIR}/bin"
DATA_DIR="${BASE_DIR}/data"
PLOTS_DIR="${BASE_DIR}/plots"

mkdir -p "${BIN_DIR}" "${DATA_DIR}" "${PLOTS_DIR}"

echo "================================================================================"
echo "Nautilus Cross-Device GPU / CPU Benchmark Suite"
echo "Base Directory: ${BASE_DIR}"
echo "================================================================================"

NVCC_BIN="$(which nvcc 2>/dev/null || true)"
if [[ -z "${NVCC_BIN}" ]]; then
    for cand in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        if [[ -x "${cand}" ]]; then
            NVCC_BIN="${cand}"
            break
        fi
    done
fi

ACPP_BIN="$(which acpp 2>/dev/null || true)"
if [[ -z "${ACPP_BIN}" ]]; then
    for cand in /opt/AdaptiveCpp/bin/acpp /home/lorenzop/AdaptiveCpp/bin/acpp; do
        if [[ -x "${cand}" ]]; then
            ACPP_BIN="${cand}"
            break
        fi
    done
fi

echo "Detected nvcc: ${NVCC_BIN:-'Not Found'}"
echo "Detected acpp: ${ACPP_BIN:-'Not Found'}"

# 1. Compile Kernels with Native NVCC (if available)
if [[ -n "${NVCC_BIN}" ]]; then
    echo "Compiling kernels with Native NVCC..."
    "${NVCC_BIN}" -shared -Xcompiler -fPIC -O3 -std=c++20 \
        "${SRC_DIR}/benchmark_kernels.cu" -o "${BIN_DIR}/libbench_nvcc.so"
fi

# 2. Compile Kernels with AdaptiveCpp PCUDA (if available)
if [[ -n "${ACPP_BIN}" ]]; then
    echo "Compiling kernels with AdaptiveCpp PCUDA..."
    "${ACPP_BIN}" --acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch \
        -fdeclspec -shared -fPIC -O3 -std=c++20 \
        "${SRC_DIR}/benchmark_kernels.cu" -o "${BIN_DIR}/libbench_pcuda.so"
fi

# 3. Compile Benchmark Runner
if [[ -n "${NVCC_BIN}" ]]; then
    echo "Building benchmark runner..."
    g++ -O3 -std=c++20 "${SRC_DIR}/benchmark_runner.cpp" \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -ldl \
        -o "${BIN_DIR}/benchmark_runner"

    echo "Executing benchmarks..."
    "${BIN_DIR}/benchmark_runner" "${DATA_DIR}/exec_bench_stats.json" | tee "${DATA_DIR}/raw_benchmark_stdout.txt"
fi

# 4. Generate Plots
echo "Generating visual plots..."
python3 "${SCRIPT_DIR}/generate_plots.py"

echo "================================================================================"
echo "Benchmarks completed successfully."
echo "Plots saved to: ${PLOTS_DIR}"
echo "================================================================================"
