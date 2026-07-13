#pragma once

#include "04_RenderCompiler/CompiledRenderPackage.h"

#include <filesystem>
#include <span>
#include <string>

namespace sge::diagnostics
{
    void WriteCompiledPackageJson(
        const compiler::CompiledRenderPackage& package,
        const std::filesystem::path& outputPath);

    [[nodiscard]] std::string DiagnosticToJson(
        const compiler::Diagnostic& diagnostic);
}
