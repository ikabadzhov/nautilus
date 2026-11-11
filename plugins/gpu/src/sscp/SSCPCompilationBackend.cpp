#include "SSCPCompilationBackend.hpp"
#include "../GPUCodegenOnlyExecutable.hpp"
#include "SSCPCompiler.hpp"
#include "SSCPExecutable.hpp"
#include "SSCPLoweringProvider.hpp"

namespace nautilus::compiler::sscp {

std::unique_ptr<Executable> SSCPCompilationBackend::compile(const std::shared_ptr<ir::IRGraph>& ir,
                                                            const DumpHandler& dumpHandler,
                                                            const engine::Options& options,
                                                            CompilationStatistics* /*statistics*/) const {
	auto code = SSCPLoweringProvider::lower(ir, options);
	dumpHandler.dump("after_sscp_generation", ".cu", [&]() { return code; });

	if (options.getOptionOrDefault("gpu.codegenOnly", false)) {
		return std::make_unique<nautilus::gpu::CodegenOnlyExecutable>();
	}

	std::string identifier = "nautilus_sscp_" + ir->getId();

	auto lib = SSCPCompiler::compile(identifier, code, options);
	return std::make_unique<SSCPExecutable>(lib);
}

} // namespace nautilus::compiler::sscp
