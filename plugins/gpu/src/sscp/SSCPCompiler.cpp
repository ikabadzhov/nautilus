#include "SSCPCompiler.hpp"
#include "nautilus/common/File.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace nautilus::compiler::sscp {

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

std::string SSCPCompiler::findAcpp() {
	// Look up acpp in PATH first (no hardcoded global paths)
	auto path = tryWhich("acpp");
	if (!path.empty()) {
		return path;
	}

	// Check optional ACPP_PATH / ACPP_DIR environment variable
	for (const char* envName : {"ACPP_PATH", "ACPP_DIR", "ADAPTIVECPP_PATH"}) {
		if (const char* envVal = std::getenv(envName)) {
			auto candidate = std::filesystem::path(envVal) / "bin" / "acpp";
			if (std::filesystem::exists(candidate)) {
				return candidate.string();
			}
		}
	}

	throw std::runtime_error("SSCPCompiler: could not find 'acpp' in PATH. Ensure AdaptiveCpp is installed and in PATH.");
}

cpp::SharedLibraryPtr SSCPCompiler::compile(const std::string& identifier, const std::string& code,
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

	common::File::createFile(sourceFileName, code);

	auto acpp = findAcpp();
	auto targets = options.getOptionOrDefault<std::string>("gpu.sscp.targets", "generic");
	auto extraFlags = options.getOptionOrDefault<std::string>("gpu.sscp.cxxFlags", "");

	std::ostringstream cmd;
	cmd << acpp;
	cmd << " --acpp-targets=" << targets;
	cmd << " --acpp-pcuda";
	cmd << " --acpp-pcuda-chevron-launch";
	cmd << " -fdeclspec";
	cmd << " -shared";
	cmd << " -fPIC";
	cmd << " -O3";
	cmd << " -std=c++20";
	if (!extraFlags.empty()) {
		cmd << " " << extraFlags;
	}
	cmd << " -o " << libraryFileName;
	cmd << " " << sourceFileName;

	runCommand(cmd.str());

	auto sharedLibrary = cpp::SharedLibrary::load(libraryFileName);

	std::filesystem::remove(sourceFileName);
	std::filesystem::remove(libraryFileName);

	return sharedLibrary;
}

void SSCPCompiler::runCommand(const std::string& cmd) {
	std::string fullCmd = cmd + " 2>&1";
	FILE* fp = popen(fullCmd.c_str(), "r");
	if (fp == nullptr) {
		throw std::runtime_error("SSCPCompiler: failed to run command: " + cmd);
	}

	std::ostringstream output;
	char buffer[8192];
	while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
		output << buffer;
	}

	auto ret = pclose(fp);
	if (ret != 0) {
		throw std::runtime_error("SSCPCompiler: compilation failed:\n" + output.str());
	}
}

} // namespace nautilus::compiler::sscp
