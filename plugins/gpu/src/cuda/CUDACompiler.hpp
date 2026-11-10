#pragma once

#include "nautilus/compiler/backends/cpp/SharedLibrary.hpp"
#include <string>
#include "nautilus/options.hpp"

namespace nautilus::compiler::cuda {

enum class CompilerType {
	Nvcc,
	Acpp
};

struct CompilerInfo {
	std::string path;
	CompilerType type;
};

/// Compiles CUDA source code (.cu) to a shared library using nvcc or acpp.
class CUDACompiler {
public:
	/// Compiles .cu source to a shared library. Returns a SharedLibrary.
	static cpp::SharedLibraryPtr compile(const std::string& identifier, const std::string& code);
	static cpp::SharedLibraryPtr compile(const std::string& identifier, const std::string& code,
	                                     const engine::Options& options);

private:
	static CompilerInfo findCompiler(const engine::Options& options);
	static std::string findNvcc();
	static void runCommand(const std::string& cmd);
};

} // namespace nautilus::compiler::cuda
