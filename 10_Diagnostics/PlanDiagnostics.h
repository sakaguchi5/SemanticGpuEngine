#pragma once

#include "03_RenderIR/RenderIR.h"
#include "04_RenderCompiler/RenderCompiler.h"

#include <filesystem>

namespace sge::diagnostics
{
    void WriteExecutionPlan(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan,
        const std::filesystem::path& outputPath);

    void WriteDependencyGraphDot(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan,
        const std::filesystem::path& outputPath);
}
