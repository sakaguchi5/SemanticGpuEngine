#include "05_RenderRuntime/RenderRuntime.h"

#include "10_Diagnostics/PlanDiagnostics.h"

#include <stdexcept>
#include <utility>

namespace sge::runtime
{
    RenderRuntime::RenderRuntime(
        std::unique_ptr<IRenderBackend> backend,
        RuntimeConfiguration configuration)
        : backend_(std::move(backend)),
          configuration_(std::move(configuration))
    {
        if (!backend_)
        {
            throw std::invalid_argument(
                "RenderRuntime requires a render backend.");
        }
    }

    RenderRuntime::~RenderRuntime()
    {
        if (backend_)
        {
            backend_->WaitIdle();
        }
    }

    void RenderRuntime::Execute(const ir::SemanticModule& module)
    {
        const auto structureHash = module.StructureHash();

        const bool mustCompile =
            !configuration_.enablePlanCache
            || !cachedPlan_
            || cachedPlan_->structureHash != structureHash;

        if (mustCompile)
        {
            auto result = compiler_.Compile(
                module,
                backend_->Capabilities());

            diagnostics_ = std::move(result.diagnostics);
            cachedPlan_ = std::move(result.plan);

            if (!configuration_.planDiagnosticsPath.empty())
            {
                diagnostics::WriteExecutionPlan(
                    module,
                    *cachedPlan_,
                    configuration_.planDiagnosticsPath);
            }

            if (!configuration_.graphDiagnosticsPath.empty())
            {
                diagnostics::WriteDependencyGraphDot(
                    module,
                    *cachedPlan_,
                    configuration_.graphDiagnosticsPath);
            }
        }

        backend_->Execute(module, *cachedPlan_);
    }

    const std::vector<std::string>& RenderRuntime::LastDiagnostics()
        const noexcept
    {
        return diagnostics_;
    }
}
