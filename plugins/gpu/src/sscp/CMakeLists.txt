target_sources(nautilus-gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SSCPCompilationBackend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SSCPLoweringProvider.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SSCPCompiler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SSCPExecutable.cpp
)
