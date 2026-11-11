#pragma once

#include "nautilus/compiler/backends/cpp/SharedLibrary.hpp"
#include "nautilus/options.hpp"
#include <string>

namespace nautilus::compiler::sscp {

class SSCPCompiler {
public:
	static cpp::SharedLibraryPtr compile(const std::string& identifier, const std::string& code,
	                                     const engine::Options& options = {});

private:
	static std::string findAcpp();
	static void runCommand(const std::string& cmd);
};

} // namespace nautilus::compiler::sscp
