#pragma once

#include "../GPULoweringProviderBase.hpp"

namespace nautilus::compiler::sscp {

class SSCPLoweringProvider {
public:
	static std::string lower(std::shared_ptr<ir::IRGraph> ir, const engine::Options& options);

private:
	class LoweringContext : public gpu::GPULoweringProviderBase<LoweringContext> {
	public:
		explicit LoweringContext(std::shared_ptr<ir::IRGraph> ir, const engine::Options& options);
		Code process();

		// Called by base class via CRTP
		static std::string getType(const Type& stamp);
		void processOperation(ir::Operation* operation, short block, gpu::RegisterFrame& frame);
		void processGPUOperation(ir::Operation* opt, short block, gpu::RegisterFrame& frame);

	private:
		const engine::Options& options;

		std::string pendingGridX = "1";
		std::string pendingGridY = "1";
		std::string pendingGridZ = "1";
		std::string pendingBlockX = "1";
		std::string pendingBlockY = "1";
		std::string pendingBlockZ = "1";
		bool hasLaunchConfig = false;

		void registerGPUIntrinsics();
		void processCall(ir::CallOperation* opt, short block, gpu::RegisterFrame& frame);
		void processIndirectCall(ir::IndirectCallOperation* opt, short block, gpu::RegisterFrame& frame);
		void processFunctionAddressOf(ir::FunctionAddressOfOperation* opt, short block, gpu::RegisterFrame& frame);
	};
};

} // namespace nautilus::compiler::sscp
