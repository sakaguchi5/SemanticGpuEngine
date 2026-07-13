#pragma once

#include "04_RenderCompiler/RenderCompiler.h"

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
        PackageNotLegacyExecutable
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

    struct QueueHandoffPlan
    {
        std::size_t releaseScheduledWork = 0;
        std::size_t acquireScheduledWork = 0;
        gpu::ResourceId resource;
        std::uint32_t frameLag = 0;
        gpu::QueueClass releaseQueue = gpu::QueueClass::Direct;
        gpu::QueueClass acquireQueue = gpu::QueueClass::Direct;
        gpu::AbstractState releaseState = gpu::AbstractState::Undefined;
        gpu::AbstractState acquireState = gpu::AbstractState::Undefined;
        bool crossesCopyQueue = false;
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

    struct CompiledResourceBlueprint
    {
        ir::ResourceDeclaration declaration;
        ResourceInstancePlan instances;
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
        std::uint64_t estimatedCommittedBytes = 0;
    };

    struct CompiledRenderPackage
    {
        // canonicalModule is a self-contained immutable snapshot. Legacy source
        // fields are removed during canonicalization.
        ir::SemanticModule canonicalModule;

        // legacyModule is only used by the compatibility backend adapter. It is
        // deliberately absent from backend-native package execution.
        ir::SemanticModule legacyModule;
        ExecutionPlan plan;

        std::vector<CompiledResourceBlueprint> resources;
        std::vector<CompiledProgramBlueprint> programs;
        std::vector<CompiledWork> works;
        std::vector<QueueHandoffPlan> queueHandoffs;
        BackendFeatureRequirements requirements;
        PackageStatistics statistics;
        std::vector<Diagnostic> diagnostics;
        std::size_t sourceHash = 0;
        std::size_t packageHash = 0;
        bool legacyExecutable = true;
    };

    struct PackageCompileResult
    {
        CompiledRenderPackage package;
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
}
