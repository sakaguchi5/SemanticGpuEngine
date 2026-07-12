#pragma once

#include "03_RenderIR/RenderIR.h"
#include "04_RenderCompiler/RenderCompiler.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace sge::runtime
{
    class IRenderBackend
    {
    public:
        virtual ~IRenderBackend() = default;

        [[nodiscard]] virtual gpu::DeviceCapabilities Capabilities() const = 0;

        virtual void Execute(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan) = 0;

        virtual void WaitIdle() = 0;
    };

    struct RuntimeConfiguration
    {
        bool enableValidation = true;
        bool enablePlanCache = true;
        std::filesystem::path planDiagnosticsPath = "execution_plan.txt";
        std::filesystem::path graphDiagnosticsPath = "work_graph.dot";
    };

    class RenderRuntime
    {
    public:
        RenderRuntime(
            std::unique_ptr<IRenderBackend> backend,
            RuntimeConfiguration configuration = {});

        RenderRuntime(const RenderRuntime&) = delete;
        RenderRuntime& operator=(const RenderRuntime&) = delete;
        ~RenderRuntime();

        void Execute(const ir::SemanticModule& module);

        [[nodiscard]] const std::vector<std::string>& LastDiagnostics()
            const noexcept;

    private:
        std::unique_ptr<IRenderBackend> backend_;
        RuntimeConfiguration configuration_;
        compiler::RenderCompiler compiler_;
        std::optional<compiler::ExecutionPlan> cachedPlan_;
        std::vector<std::string> diagnostics_;
    };
}
