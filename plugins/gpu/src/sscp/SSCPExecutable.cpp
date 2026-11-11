#include "SSCPExecutable.hpp"
#include <utility>

namespace nautilus::compiler::sscp {

SSCPExecutable::SSCPExecutable(std::shared_ptr<cpp::SharedLibrary> lib) : lib_(std::move(lib)) {
}

void* SSCPExecutable::getInvocableFunctionPtr(const std::string& member) {
	return lib_->getInvocableFunctionPtr(member);
}

bool SSCPExecutable::hasInvocableFunctionPtr() {
	return true;
}

} // namespace nautilus::compiler::sscp
