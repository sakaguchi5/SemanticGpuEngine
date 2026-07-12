#pragma once

#include "00_Foundation/StrongId.h"

#include <cstdint>
#include <string>

namespace sge::gpu
{
    struct ResourceTag;
    struct WorkTag;
    struct ProgramTag;
    struct PhysicalAllocationTag;

    using ResourceId = foundation::StrongId<ResourceTag>;
    using WorkId = foundation::StrongId<WorkTag>;
    using ProgramId = foundation::StrongId<ProgramTag>;
    using PhysicalAllocationId = foundation::StrongId<PhysicalAllocationTag>;

    enum class ResourceKind
    {
        Buffer,
        Texture1D,
        Texture2D,
        Texture3D,
        Presentation
    };

    enum class ResourceLifetimeClass
    {
        Persistent,
        FrameLocal,
        Temporal,
        External
    };

    enum class ResourceUpdateClass
    {
        Immutable,
        CpuUpdated,
        GpuProduced,
        Imported
    };

    enum class InstanceSelectorKind
    {
        Persistent,
        CurrentFrameSlot,
        PreviousTemporalSlot,
        CurrentTemporalSlot,
        ExternalFrameIndex
    };

    enum class ResourceFormat
    {
        Unknown,
        Rgba8Unorm,
        Depth32Float
    };

    enum class AccessMode
    {
        Read,
        Write,
        ReadWrite
    };

    enum class ResourceRole
    {
        VertexInput,
        ConstantInput,
        ProgramInput,
        ProgramOutput,
        ColorOutput,
        DepthOutput,
        TransferSource,
        TransferDestination,
        Presentation
    };

    enum class ExecutionDomain
    {
        Raster,
        Compute,
        Copy,
        Present
    };

    enum class PrimitiveTopology
    {
        TriangleList,
        LineList
    };

    enum class CompositionMode
    {
        Replace,
        AlphaOver
    };

    enum class DepthMode
    {
        Disabled,
        ReadOnly,
        ReadWrite
    };

    enum class AbstractState
    {
        Undefined,
        VertexRead,
        ConstantRead,
        ProgramRead,
        ProgramWrite,
        ColorWrite,
        DepthRead,
        DepthWrite,
        TransferRead,
        TransferWrite,
        Present
    };

    enum class ProgramParameterKind
    {
        ConstantBuffer,
        ShaderResource,
        UnorderedAccess,
        Sampler
    };

    enum class ProgramStage
    {
        Vertex,
        Pixel,
        Compute,
        AllGraphics
    };

    enum class QueueClass
    {
        Direct,
        Compute,
        Copy
    };

    struct ResourceAccess
    {
        ResourceId resource;
        AccessMode access = AccessMode::Read;
        ResourceRole role = ResourceRole::ProgramInput;
        std::uint32_t frameLag = 0;
    };

    struct ProgramParameter
    {
        std::string name;
        ProgramParameterKind kind = ProgramParameterKind::ConstantBuffer;
        ProgramStage stage = ProgramStage::AllGraphics;
        std::uint32_t registerIndex = 0;
        std::uint32_t registerSpace = 0;

        auto operator<=>(const ProgramParameter&) const = default;
    };

    struct DeviceCapabilities
    {
        bool rasterExecution = true;
        bool computeExecution = false;
        bool copyExecution = false;
        bool concurrentCompute = false;
        bool dedicatedCopyQueue = false;
        bool resourceAliasing = false;
        bool rayExecution = false;
        std::uint32_t constantDataAlignment = 256;
        std::uint64_t localMemoryBudget = 0;
        std::uint32_t maxFramesInFlight = 3;
    };

    [[nodiscard]] AbstractState RequiredState(const ResourceAccess& access);
    [[nodiscard]] const char* ToString(AbstractState state) noexcept;
}
