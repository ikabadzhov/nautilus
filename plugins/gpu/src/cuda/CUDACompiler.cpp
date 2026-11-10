
#include "CUDACompiler.hpp"
#include "nautilus/common/File.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "nautilus/options.hpp"

namespace nautilus::compiler::cuda {

static std::string tryWhich(const std::string& name) {
	std::string cmd = "which " + name + " 2>/dev/null";
	FILE* fp = popen(cmd.c_str(), "r");
	if (!fp) {
		return {};
	}
	char buf[512];
	std::string result;
	while (fgets(buf, sizeof(buf), fp) != nullptr) {
		result += buf;
	}
	if (pclose(fp) == 0 && !result.empty()) {
		while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
			result.pop_back();
		}
		return result;
	}
	return {};
}

CompilerInfo CUDACompiler::findCompiler(const engine::Options& options) {
	// 1. Check explicit preference via engine::Options or environment variable
	std::string preference;
	if (options.hasOption("gpu.compiler")) {
		preference = options.getOptionOrDefault<std::string>("gpu.compiler", "");
	}
	if (preference.empty()) {
		if (const char* env = std::getenv("NAUTILUS_GPU_COMPILER")) {
			preference = env;
		}
	}

	if (preference == "acpp") {
		auto path = tryWhich("acpp");
		if (!path.empty()) {
			return {path, CompilerType::Acpp};
		}
		throw std::runtime_error("CUDACompiler: 'acpp' requested but not found in PATH.");
	}

	if (preference == "nvcc") {
		auto path = tryWhich("nvcc");
		if (path.empty()) {
			for (auto& candidate : {"/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"}) {
				if (std::filesystem::exists(candidate)) {
					path = candidate;
					break;
				}
			}
		}
		if (path.empty()) {
			if (const char* cudaHome = std::getenv("CUDA_HOME")) {
				auto candidate = std::filesystem::path(cudaHome) / "bin" / "nvcc";
				if (std::filesystem::exists(candidate)) {
					path = candidate.string();
				}
			}
		}
		if (!path.empty()) {
			return {path, CompilerType::Nvcc};
		}
		throw std::runtime_error("CUDACompiler: 'nvcc' requested but not found in PATH or standard CUDA directories.");
	}

	// 2. Auto-detection:
	// If acpp is on PATH, prefer it for cross-vendor support (AMD, Intel, NVIDIA, CPU)
	auto acppPath = tryWhich("acpp");
	if (!acppPath.empty()) {
		return {acppPath, CompilerType::Acpp};
	}

	// Fallback to nvcc on PATH, common CUDA installation paths, or CUDA_HOME
	auto nvccPath = tryWhich("nvcc");
	if (!nvccPath.empty()) {
		return {nvccPath, CompilerType::Nvcc};
	}
	for (auto& candidate : {"/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"}) {
		if (std::filesystem::exists(candidate)) {
			return {candidate, CompilerType::Nvcc};
		}
	}
	if (const char* cudaHome = std::getenv("CUDA_HOME")) {
		auto candidate = std::filesystem::path(cudaHome) / "bin" / "nvcc";
		if (std::filesystem::exists(candidate)) {
			return {candidate.string(), CompilerType::Nvcc};
		}
	}

	throw std::runtime_error(
	    "CUDACompiler: could not find 'acpp' or 'nvcc' in PATH. "
	    "Ensure AdaptiveCpp or CUDA toolkit is installed and in PATH.");
}

std::string CUDACompiler::findNvcc() {
	return findCompiler(engine::Options()).path;
}

cpp::SharedLibraryPtr CUDACompiler::compile(const std::string& identifier, const std::string& code) {
	return compile(identifier, code, engine::Options());
}

cpp::SharedLibraryPtr CUDACompiler::compile(const std::string& identifier, const std::string& code,
                                            const engine::Options& options) {
	auto tmpDir = std::filesystem::temp_directory_path();
	auto sourceFileName = (tmpDir / (identifier + ".cu")).string();
	auto libraryFileName = (tmpDir / (identifier +
#ifdef __linux__
	                                  ".so"
#elif defined(__APPLE__)
	                                  ".dylib"
#else
#error "Unknown platform"
#endif
	                                  ))
	                           .string();

	// Write CUDA source
	common::File::createFile(sourceFileName, code);

	auto compiler = findCompiler(options);

	std::ostringstream cmd;
	if (compiler.type == CompilerType::Acpp) {
		cmd << compiler.path;
		cmd << " --acpp-targets=generic";
		cmd << " --acpp-pcuda";
		cmd << " --acpp-pcuda-chevron-launch";
		cmd << " -fdeclspec";
		cmd << " -shared";
		cmd << " -fPIC";
		cmd << " -O3";
		cmd << " -std=c++20";
		cmd << " -o " << libraryFileName;
		cmd << " " << sourceFileName;
	} else {
		cmd << compiler.path;
		cmd << " -shared";
		cmd << " -Xcompiler -fPIC";
		cmd << " -O3";
		cmd << " -std=c++20";
		cmd << " -o " << libraryFileName;
		cmd << " " << sourceFileName;
	}

	runCommand(cmd.str());

	// Load the shared library
	auto sharedLibrary = cpp::SharedLibrary::load(libraryFileName);

	// Clean up
	std::filesystem::remove(sourceFileName);
	std::filesystem::remove(libraryFileName);

	return sharedLibrary;
}

void CUDACompiler::runCommand(const std::string& cmd) {
	std::string fullCmd = cmd + " 2>&1";
	FILE* fp = popen(fullCmd.c_str(), "r");
	if (fp == nullptr) {
		throw std::runtime_error("CUDACompiler: failed to run command: " + cmd);
	}

	std::ostringstream output;
	char buffer[8192];
	while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
		output << buffer;
	}

	auto ret = pclose(fp);
	if (ret != 0) {
		throw std::runtime_error("CUDACompiler: compilation failed:\n" + output.str());
	}
}

} // namespace nautilus::compiler::cuda
