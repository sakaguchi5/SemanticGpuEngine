#include "05_RenderRuntime/RenderRuntime.h"

#include "10_Diagnostics/PlanDiagnostics.h"
#include "10_Diagnostics/StructuredDiagnostics.h"

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
        try
        {
            WaitIdle();
        }
        catch (...)
        {
        }
    }

    void RenderRuntime::Execute(const ir::SemanticModule& module)
    {
        FrameInvocation invocation;
        invocation.frameNumber = frameNumber_;
        Execute(module, invocation);
    }

    void RenderRuntime::Execute(
        const ir::SemanticModule& module,
        const FrameInvocation& invocation)
    {
        // StructureHash is pure: invalid modules are rejected by the package
        // validator rather than as a side effect of cache-key generation.
        const auto sourceHash = module.StructureHash();
        const bool mustCompile =
            !configuration_.enablePlanCache
            || !cachedPackage_
            || cachedPackage_->sourceHash != sourceHash;

        if (mustCompile)
        {
            auto result = compiler_.Compile(
                module,
                backend_->Capabilities());
            structuredDiagnostics_ = result.diagnostics;
            diagnostics_.clear();
            diagnostics_.reserve(result.diagnostics.size());
            for (const auto& diagnostic : result.diagnostics)
            {
                diagnostics_.push_back(
                    std::string(compiler::ToString(diagnostic.severity))
                    + " [" + compiler::ToString(diagnostic.code) + "] "
                    + diagnostic.message);
            }

            if (!result.Succeeded())
            {
                throw std::runtime_error(
                    diagnostics_.empty()
                        ? "CompiledRenderPackage validation failed."
                        : diagnostics_.front());
            }

            cachedPackage_ = std::move(result.package);

            if (!configuration_.planDiagnosticsPath.empty())
            {
                diagnostics::WriteExecutionPlan(
                    cachedPackage_->canonicalModule,
                    cachedPackage_->plan,
                    configuration_.planDiagnosticsPath);
            }
            if (!configuration_.graphDiagnosticsPath.empty())
            {
                diagnostics::WriteDependencyGraphDot(
                    cachedPackage_->canonicalModule,
                    cachedPackage_->plan,
                    configuration_.graphDiagnosticsPath);
            }
            if (!configuration_.packageDiagnosticsPath.empty())
            {
                diagnostics::WriteCompiledPackageJson(
                    *cachedPackage_,
                    configuration_.packageDiagnosticsPath);
            }
        }

        FrameInvocation resolved = invocation;
        if (resolved.frameNumber == 0 && frameNumber_ != 0)
        {
            resolved.frameNumber = frameNumber_;
        }
        for (const auto& resource : module.resources)
        {
            if (resource.update == gpu::ResourceUpdateClass::CpuUpdated)
            {
                resolved.dynamicResourceData[resource.id] = resource.data;
            }
        }
        backend_->Execute(*cachedPackage_, resolved);
        ++frameNumber_;
    }

    void RenderRuntime::WaitIdle()
    {
        if (backend_)
        {
            backend_->WaitIdle();
        }
    }

    const std::vector<std::string>& RenderRuntime::LastDiagnostics()
        const noexcept
    {
        return diagnostics_;
    }

    const std::vector<compiler::Diagnostic>&
        RenderRuntime::LastStructuredDiagnostics() const noexcept
    {
        return structuredDiagnostics_;
    }

    const compiler::CompiledRenderPackage*
        RenderRuntime::CachedPackage() const noexcept
    {
        return cachedPackage_ ? &*cachedPackage_ : nullptr;
    }
}
