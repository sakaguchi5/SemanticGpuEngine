#pragma once

#include "04_RenderCompiler/CompiledRenderPackage.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sge::runtime
{
    enum class BackendFailureKind
    {
        OrdinaryFailure,
        DeviceRemoved,
        OutOfMemory,
        InvalidPackage,
        ExternalRebindRequired
    };

    class BackendFailure : public std::runtime_error
    {
    public:
        BackendFailure(
            BackendFailureKind failureKind,
            std::string message,
            std::int64_t operationResult = 0,
            std::int64_t removedReason = 0)
            : std::runtime_error(std::move(message)),
              kind(failureKind),
              operationResult(operationResult),
              deviceRemovedReason(removedReason)
        {
        }

        BackendFailureKind kind;
        std::int64_t operationResult = 0;
        std::int64_t deviceRemovedReason = 0;
    };

    class IExternalResource
    {
    public:
        virtual ~IExternalResource() = default;
        [[nodiscard]] virtual std::string_view BackendType() const noexcept = 0;
    };

    class IQueueCompletion
    {
    public:
        virtual ~IQueueCompletion() = default;
        [[nodiscard]] virtual bool IsComplete() const noexcept = 0;
        virtual void Wait() = 0;
    };

    struct ExternalResourceBinding
    {
        std::shared_ptr<IExternalResource> resource;
        // Optional producer completion. The reference implementation waits
        // this point before importing the resource for the frame.
        std::shared_ptr<IQueueCompletion> availableAfter;
        gpu::AbstractState incomingState = gpu::AbstractState::Undefined;
        gpu::AbstractState outgoingState = gpu::AbstractState::Undefined;
    };

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

        std::unordered_map<
            gpu::ResourceId,
            ExternalResourceBinding,
            foundation::StrongIdHash<gpu::ResourceTag>> externalResources;
    };

    struct QueueCompletionPoint
    {
        gpu::QueueClass queue = gpu::QueueClass::Direct;
        std::uint64_t value = 0;
        std::shared_ptr<IQueueCompletion> completion;
    };

    struct FrameSubmission
    {
        std::uint64_t deviceEpoch = 0;
        std::array<QueueCompletionPoint, 3> queues{};
    };

    class IRenderBackend
    {
    public:
        virtual ~IRenderBackend() = default;

        [[nodiscard]] virtual gpu::DeviceCapabilities Capabilities() const = 0;

        // The backend boundary is package-only. Source SemanticModule and the
        // compiler's analysis ExecutionPlan never cross this interface.
        virtual FrameSubmission Execute(
            const compiler::CompiledRenderPackage& package,
            const FrameInvocation& invocation) = 0;

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

        FrameSubmission Execute(
            const ir::SemanticModule& module);
        FrameSubmission Execute(
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
