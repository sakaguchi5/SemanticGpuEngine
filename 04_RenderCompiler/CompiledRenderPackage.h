#pragma once

#include "04_RenderCompiler/RenderCompiler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sge::compiler
{
    enum class DiagnosticSeverity
    {
        Information,
        Warning,
        Error
    };

    enum class DiagnosticCode
    {
        None,
        LegacyProgramInterface,
        InvalidResource,
        InvalidProgram,
        InvalidWork,
        InvalidView,
        InvalidSubresourceRange,
        PayloadAccessMismatch,
        ResourceStateConflict,
        UnsupportedCapability,
        MemoryBudgetExceeded,
        QueueHandoffRequired,
        ShaderInterfaceMismatch,
        InvalidInitialContent
    };

    struct DiagnosticLocation
    {
        std::optional<gpu::ResourceId> resource;
        std::optional<gpu::ProgramId> program;
        std::optional<gpu::WorkId> work;
        std::optional<ir::ResourceView> view;
    };

    struct Diagnostic
    {
        DiagnosticCode code = DiagnosticCode::None;
        DiagnosticSeverity severity = DiagnosticSeverity::Information;
        std::string message;
        DiagnosticLocation location;
        std::vector<std::string> notes;
    };

    struct NormalizedTextureRange
    {
        std::uint16_t firstMip = 0;
        std::uint16_t mipCount = 1;
        std::uint16_t firstArrayLayer = 0;
        std::uint16_t arrayLayerCount = 1;
        std::uint8_t firstPlane = 0;
        std::uint8_t planeCount = 1;
        std::uint16_t firstDepthSlice = 0;
        std::uint16_t depthSliceCount = 1;

        auto operator<=>(const NormalizedTextureRange&) const = default;
    };

    struct NormalizedResourceView
    {
        gpu::ResourceId resource;
        gpu::ResourceKind kind = gpu::ResourceKind::Buffer;
        std::uint64_t byteOffset = 0;
        std::uint64_t byteSize = 0;
        std::uint32_t strideBytes = 0;
        NormalizedTextureRange textureRange{};
        gpu::ResourceFormat format = gpu::ResourceFormat::Unknown;

        auto operator<=>(const NormalizedResourceView&) const = default;
    };

    struct RangeStateRequirement
    {
        NormalizedResourceView view;
        gpu::AbstractState state = gpu::AbstractState::Undefined;
        std::uint32_t frameLag = 0;
        bool writeCapable = false;
    };

    // These records are compiler-report structures. Backend execution does not
    // reinterpret them; their effects are lowered into CompiledOperation.
    struct QueueHandoffPlan
    {
        std::size_t releaseScheduledWork = 0;
        std::size_t acquireScheduledWork = 0;
        gpu::ResourceId resource;
        std::uint32_t frameLag = 0;
        NormalizedResourceView releaseView;
        NormalizedResourceView acquireView;
        gpu::QueueClass releaseQueue = gpu::QueueClass::Direct;
        gpu::QueueClass acquireQueue = gpu::QueueClass::Direct;
        gpu::AbstractState releaseState = gpu::AbstractState::Undefined;
        gpu::AbstractState acquireState = gpu::AbstractState::Undefined;
        bool crossesCopyQueue = false;
    };

    struct CyclicFrameHandoffPlan
    {
        std::size_t releaseScheduledWork = 0;
        std::size_t acquireScheduledWork = 0;
        gpu::ResourceId resource;
        NormalizedResourceView releaseView;
        NormalizedResourceView acquireView;
        gpu::QueueClass releaseQueue = gpu::QueueClass::Direct;
        gpu::QueueClass acquireQueue = gpu::QueueClass::Direct;
        gpu::AbstractState releaseState = gpu::AbstractState::Undefined;
        gpu::AbstractState acquireState = gpu::AbstractState::Undefined;
        bool requiresCommonRelease = false;
    };

    struct CompiledBinding
    {
        std::uint32_t parameterIndex = 0;
        gpu::ProgramParameterKind kind =
            gpu::ProgramParameterKind::ConstantBuffer;
        NormalizedResourceView view;
        std::uint32_t frameLag = 0;
    };

    struct CompiledVertexStream
    {
        std::uint32_t inputSlot = 0;
        NormalizedResourceView view;
    };

    struct CompiledIndexStream
    {
        NormalizedResourceView view;
        ir::IndexFormat format = ir::IndexFormat::Uint32;
        std::uint32_t firstIndex = 0;
        std::int32_t baseVertex = 0;
    };

    struct CompiledRasterCommand
    {
        gpu::ProgramId program;
        ExecutableKey executable;
        std::vector<CompiledVertexStream> vertexStreams;
        std::optional<CompiledIndexStream> indexStream;
        std::vector<CompiledBinding> bindings;
        std::vector<NormalizedResourceView> colorAttachments;
        std::optional<NormalizedResourceView> depthAttachment;
        std::uint32_t vertexCount = 0;
        std::uint32_t firstVertex = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t instanceCount = 1;
        std::uint32_t firstInstance = 0;
        ir::ViewportDescription viewport{};
        ir::ScissorDescription scissor{};
        ir::ClearDescription clear{};
    };

    struct CompiledComputeCommand
    {
        gpu::ProgramId program;
        ExecutableKey executable;
        std::vector<CompiledBinding> bindings;
        std::uint32_t groupCountX = 1;
        std::uint32_t groupCountY = 1;
        std::uint32_t groupCountZ = 1;
    };

    struct CompiledCopyCommand
    {
        NormalizedResourceView source;
        NormalizedResourceView destination;
        std::uint32_t sourceFrameLag = 0;
        std::uint32_t destinationFrameLag = 0;
    };

    struct CompiledPresentCommand
    {
        gpu::ResourceId source;
    };

    using CompiledCommand = std::variant<
        CompiledRasterCommand,
        CompiledComputeCommand,
        CompiledCopyCommand,
        CompiledPresentCommand>;

    struct CompiledWork
    {
        gpu::WorkId id;
        std::string name;
        gpu::QueueClass queue = gpu::QueueClass::Direct;
        std::vector<gpu::ResourceAccess> accesses;
        std::vector<RangeStateRequirement> rangeStates;
        CompiledCommand command = CompiledPresentCommand{};
    };

    struct CompiledOptimizedClearValue
    {
        gpu::ResourceFormat format = gpu::ResourceFormat::Unknown;
        bool depthStencil = false;
        std::array<float, 4> color{};
        float depth = 1.0f;
        std::uint8_t stencil = 0;
    };

    enum class ResourceOrigin
    {
        PackageOwned,
        ExternalBorrowed,
        Presentation
    };

    enum class ResourceRebuildPolicy
    {
        RecreateFromPackage,
        RequireExternalRebind,
        BackendManaged
    };

    enum class ResourceExtentPolicy
    {
        Fixed,
        SurfaceRelative
    };

    struct NoInitialContent
    {
    };

    struct BufferInitialContent
    {
        std::vector<std::byte> bytes;
    };

    struct TextureSubresourceContent
    {
        std::uint16_t mip = 0;
        std::uint16_t arrayLayer = 0;
        std::uint8_t plane = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t depth = 1;
        std::uint64_t sourceRowPitch = 0;
        std::uint64_t sourceSlicePitch = 0;
        std::vector<std::byte> bytes;
    };

    struct TextureInitialContent
    {
        std::vector<TextureSubresourceContent> subresources;
    };

    using InitialContent = std::variant<
        NoInitialContent,
        BufferInitialContent,
        TextureInitialContent>;

    struct UploadBufferPreparation
    {
        gpu::ResourceId resource;
    };

    struct UploadTexturePreparation
    {
        gpu::ResourceId resource;
    };

    struct InitializePersistentStatePreparation
    {
        gpu::ResourceId resource;
    };

    using CompiledPreparationOperation = std::variant<
        UploadBufferPreparation,
        UploadTexturePreparation,
        InitializePersistentStatePreparation>;

    struct ExternalResourceSlot
    {
        gpu::ResourceId resource;
        gpu::ResourceKind kind = gpu::ResourceKind::Buffer;
        ir::ResourceDescription expectedDescription =
            ir::BufferDescription{};
        gpu::AbstractState firstRequiredState =
            gpu::AbstractState::Undefined;
        gpu::AbstractState lastRequiredState =
            gpu::AbstractState::Undefined;
        bool requiredEveryFrame = true;
    };

    struct CompiledResourceBlueprint
    {
        ir::ResourceDeclaration declaration;
        ResourceInstancePlan instances;
        ResourceOrigin origin = ResourceOrigin::PackageOwned;
        ResourceRebuildPolicy rebuildPolicy =
            ResourceRebuildPolicy::RecreateFromPackage;
        ResourceExtentPolicy extentPolicy = ResourceExtentPolicy::Fixed;
        InitialContent initialContent = NoInitialContent{};
        std::optional<gpu::PhysicalAllocationId> allocation;
        std::vector<gpu::AbstractState> persistentReadStates;
        std::optional<CompiledOptimizedClearValue> optimizedClear;
        bool requiresTypelessResource = false;
        std::uint64_t estimatedCommittedBytes = 0;
    };

    struct CompiledProgramBlueprint
    {
        ir::ProgramDeclaration declaration;
        std::vector<gpu::ProgramParameter> parameters;
    };

    struct BackendFeatureRequirements
    {
        bool textureSubresourceViews = false;
        bool multipleVertexStreams = false;
        bool indexedDraw = false;
        bool instancedDraw = false;
        bool explicitCopyQueueHandoffs = false;
        bool dynamicDescriptorGrowth = false;
        bool advancedRasterState = false;
        bool customViewportScissor = false;
        bool expandedResourceFormats = false;
    };

    struct PackageStatistics
    {
        std::size_t logicalResourceCount = 0;
        std::size_t physicalInstanceCount = 0;
        std::size_t workCount = 0;
        std::size_t executableCount = 0;
        std::size_t descriptorViewCount = 0;
        std::size_t barrierCount = 0;
        std::size_t queueWaitCount = 0;
        std::size_t operationCount = 0;
        std::uint64_t estimatedCommittedBytes = 0;
    };

    // Backend-ready operation stream. Queue ownership, barriers, cyclic slot
    // reuse, temporal ordering and command execution are all explicit here.
    struct BeginWorkOperation
    {
        std::size_t workIndex = 0;
        gpu::WorkId work;
        std::string name;
        gpu::QueueClass queue = gpu::QueueClass::Direct;
    };

    struct WaitForWorkOperation
    {
        gpu::QueueClass waitingQueue = gpu::QueueClass::Direct;
        std::size_t signalWorkIndex = 0;
        gpu::QueueClass signalQueue = gpu::QueueClass::Direct;
    };

    struct WaitForTemporalOperation
    {
        gpu::QueueClass waitingQueue = gpu::QueueClass::Direct;
        gpu::ResourceAccess access;
    };

    struct ActivateAliasOperation
    {
        gpu::ResourceId resource;
        std::uint32_t frameLag = 0;
    };

    struct TransitionOperation
    {
        NormalizedResourceView view;
        gpu::AbstractState state = gpu::AbstractState::Undefined;
        std::uint32_t frameLag = 0;
    };

    struct RequireCommonOperation
    {
        NormalizedResourceView view;
        std::uint32_t frameLag = 0;
        gpu::AbstractState implicitCopyState = gpu::AbstractState::Undefined;
        bool cyclicReuse = false;
    };

    struct ExecuteCommandOperation
    {
        CompiledCommand command = CompiledPresentCommand{};
    };

    struct SubmitWorkOperation
    {
        std::size_t workIndex = 0;
        gpu::QueueClass queue = gpu::QueueClass::Direct;
        std::vector<gpu::ResourceAccess> temporalAccesses;
    };

    using CompiledOperation = std::variant<
        BeginWorkOperation,
        WaitForWorkOperation,
        WaitForTemporalOperation,
        ActivateAliasOperation,
        TransitionOperation,
        RequireCommonOperation,
        ExecuteCommandOperation,
        SubmitWorkOperation>;

    struct CompiledRenderPackage
    {
        std::vector<CompiledResourceBlueprint> resources;
        std::vector<CompiledProgramBlueprint> programs;
        std::vector<ExecutableKey> executables;
        std::vector<CompiledPreparationOperation> preparationOperations;
        std::vector<ExternalResourceSlot> externalSlots;
        std::vector<CompiledOperation> operations;
        BackendFeatureRequirements requirements;
        PackageStatistics statistics;
        std::size_t sourceHash = 0;
        std::size_t packageHash = 0;

        [[nodiscard]] const CompiledResourceBlueprint& Resource(
            gpu::ResourceId id) const;
        [[nodiscard]] const CompiledProgramBlueprint& Program(
            gpu::ProgramId id) const;
    };

    // Human/source-facing compiler data is deliberately separated from the
    // backend package. The backend never receives or reads this report.
    struct CompilationReport
    {
        ir::SemanticModule canonicalModule;
        ExecutionPlan analysisPlan;
        std::vector<CompiledWork> works;
        std::vector<QueueHandoffPlan> queueHandoffs;
        std::vector<CyclicFrameHandoffPlan> cyclicFrameHandoffs;
        bool planningUsedCompatibilitySnapshot = false;
    };

    struct PackageCompileResult
    {
        CompiledRenderPackage package;
        CompilationReport report;
        std::vector<Diagnostic> diagnostics;

        [[nodiscard]] bool Succeeded() const noexcept;
    };

    class RenderPackageCompiler
    {
    public:
        RenderPackageCompiler();
        explicit RenderPackageCompiler(
            std::shared_ptr<const ISchedulingPolicy> policy);

        [[nodiscard]] PackageCompileResult Compile(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities) const;

        [[nodiscard]] static std::vector<Diagnostic> Validate(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities);

    private:
        RenderCompiler coreCompiler_;
    };

    [[nodiscard]] const char* ToString(DiagnosticCode code) noexcept;
    [[nodiscard]] const char* ToString(DiagnosticSeverity severity) noexcept;
    [[nodiscard]] const char* ToString(const CompiledOperation& operation) noexcept;
}
