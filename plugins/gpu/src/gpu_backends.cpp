
#include "nautilus/gpu/backends.hpp"
#include "nautilus/gpu/config.hpp"

#ifdef ENABLE_CUDA_BACKEND
#include "cuda/CUDALoweringProvider.hpp"
#endif
#ifdef ENABLE_METAL_BACKEND
#include "metal/MetalLoweringProvider.hpp"
#endif
#ifdef ENABLE_SSCP_BACKEND
#include "sscp/SSCPLoweringProvider.hpp"
#endif

namespace nautilus::gpu {

#ifdef ENABLE_CUDA_BACKEND
std::string lowerToCUDA(std::shared_ptr<compiler::ir::IRGraph> ir, const engine::Options& options) {
	return compiler::cuda::CUDALoweringProvider::lower(std::move(ir), options);
}
#endif

#ifdef ENABLE_METAL_BACKEND
MetalOutput lowerToMetal(std::shared_ptr<compiler::ir::IRGraph> ir, const engine::Options& options) {
	auto result = compiler::metal::MetalLoweringProvider::lower(std::move(ir), options);
	return MetalOutput {.deviceCode = std::move(result.deviceCode), .hostCode = std::move(result.hostCode)};
}
#endif

#ifdef ENABLE_SSCP_BACKEND
std::string lowerToSSCP(std::shared_ptr<compiler::ir::IRGraph> ir, const engine::Options& options) {
	return compiler::sscp::SSCPLoweringProvider::lower(std::move(ir), options);
}
#endif

} // namespace nautilus::gpu
