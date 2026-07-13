#pragma once

#include "04_RenderCompiler/CompiledRenderPackage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sge::runtime
{
    struct FrameInvocation
    {
        std::uint64_t frameNumber = 0;

        // Dynamic data is deliberately separated from the static package.
        // Frontends may update CPU-visible resource contents by ResourceId
        // without forcing a package recompile.
        std::unordered_map<
            gpu::ResourceId,
            std::vector<std::byte>,
            foundation::StrongIdHash<gpu::ResourceTag>> dynamicResourceData;
    };

    class IRenderBackend
    {
    public:
        virtual ~IRenderBackend() = default;

        [[nodiscard]] virtual gpu::DeviceCapabilities Capabilities() const = 0;

        // Package execution is the primary runtime boundary. Backends may
        // override this method directly. The compatibility implementation is
        // intentionally confined here and only accepts packages proven to be
        // representable by the legacy D3D12 command path.
        virtual void Execute(
            const compiler::CompiledRenderPackage& package,
            const FrameInvocation& invocation)
        {
            (void)invocation;
            if (!package.legacyExecutable)
            {
                throw std::runtime_error(
                    "Backend requires native CompiledRenderPackage support for this package.");
            }
            auto frameModule = package.legacyModule;
            for (const auto& [resourceId, bytes] :
                 invocation.dynamicResourceData)
            {
                const auto found = std::find_if(
                    frameModule.resources.begin(),
                    frameModule.resources.end(),
                    [&](const ir::ResourceDeclaration& resource)
                    {
                        return resource.id == resourceId;
                    });
                if (found != frameModule.resources.end())
                {
                    found->data = bytes;
                }
            }
            Execute(frameModule, package.plan);
        }

        // Transitional compatibility boundary. New backends should override
        // package execution and may leave this path unused.
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
        std::filesystem::path packageDiagnosticsPath = "compiled_package.json";
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
        void Execute(
            const ir::SemanticModule& module,
            const FrameInvocation& invocation);
        void WaitIdle();

        [[nodiscard]] const std::vector<std::string>& LastDiagnostics()
            const noexcept;
        [[nodiscard]] const std::vector<compiler::Diagnostic>&
            LastStructuredDiagnostics() const noexcept;
        [[nodiscard]] const compiler::CompiledRenderPackage*
            CachedPackage() const noexcept;

    private:
        std::unique_ptr<IRenderBackend> backend_;
        RuntimeConfiguration configuration_;
        compiler::RenderPackageCompiler compiler_;
        std::optional<compiler::CompiledRenderPackage> cachedPackage_;
        std::vector<std::string> diagnostics_;
        std::vector<compiler::Diagnostic> structuredDiagnostics_;
        std::uint64_t frameNumber_ = 0;
    };
}
