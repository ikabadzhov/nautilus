#pragma once

#include <nautilus/Executable.hpp>
#include <nautilus/compiler/backends/cpp/SharedLibrary.hpp>
#include <memory>

namespace nautilus::compiler::sscp {

class SSCPExecutable : public Executable {
public:
	explicit SSCPExecutable(std::shared_ptr<cpp::SharedLibrary> lib);
	void* getInvocableFunctionPtr(const std::string& member) override;
	bool hasInvocableFunctionPtr() override;

private:
	std::shared_ptr<cpp::SharedLibrary> lib_;
};

} // namespace nautilus::compiler::sscp
