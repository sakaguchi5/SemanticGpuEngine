#pragma once

#include "02_GpuSemantics/Semantics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace sge::ir
{
    enum class BufferUsage : std::uint32_t
    {
        None = 0,
        Vertex = 1u << 0u,
        Index = 1u << 1u,
        Constant = 1u << 2u,
        Storage = 1u << 3u,
        Indirect = 1u << 4u,
        CopySource = 1u << 5u,
        CopyDestination = 1u << 6u
    };

    enum class TextureUsage : std::uint32_t
    {
        None = 0,
        ColorAttachment = 1u << 0u,
        DepthAttachment = 1u << 1u,
        Sampled = 1u << 2u,
        Storage = 1u << 3u,
        CopySource = 1u << 4u,
        CopyDestination = 1u << 5u
    };

    enum class VertexElementFormat
    {
        Float,
        Float2,
        Float3,
        Float4,
        Uint,
        Uint2,
        Uint3,
        Uint4
    };

    enum class VertexInputRate
    {
        PerVertex,
        PerInstance
    };

    struct VertexInputElement
    {
        std::string semanticName;
        std::uint32_t semanticIndex = 0;
        VertexElementFormat format = VertexElementFormat::Float3;
        std::uint32_t inputSlot = 0;
        std::uint32_t alignedByteOffset = 0;
        VertexInputRate inputRate = VertexInputRate::PerVertex;
        std::uint32_t instanceStepRate = 0;

        auto operator<=>(const VertexInputElement&) const = default;
    };

    struct ProgramInterface
    {
        // New declarations should place binding parameters here. The legacy
        // ProgramDeclaration::parameters field remains as a compatibility path
        // while existing frontends are migrated incrementally.
        std::vector<gpu::ProgramParameter> parameters;

        // The default preserves the current interleaved Position/Color stream,
        // but the backend no longer hardcodes it.
        std::vector<VertexInputElement> vertexInputs{
            {"POSITION", 0, VertexElementFormat::Float3, 0, 0},
            {"COLOR", 0, VertexElementFormat::Float4, 0, 12}
        };

        std::uint32_t colorOutputCount = 1;
        bool depthAttachmentAllowed = true;
    };

    [[nodiscard]] constexpr BufferUsage operator|(
        BufferUsage left, BufferUsage right) noexcept
    {
        return static_cast<BufferUsage>(
            static_cast<std::uint32_t>(left)
            | static_cast<std::uint32_t>(right));
    }

    [[nodiscard]] constexpr TextureUsage operator|(
        TextureUsage left, TextureUsage right) noexcept
    {
        return static_cast<TextureUsage>(
            static_cast<std::uint32_t>(left)
            | static_cast<std::uint32_t>(right));
    }

    struct BufferDescription
    {
        std::uint64_t sizeBytes = 0;
        std::uint32_t strideBytes = 0;
        BufferUsage usage = BufferUsage::None;
        auto operator<=>(const BufferDescription&) const = default;
    };

    struct TextureDescription
    {
        gpu::ResourceKind dimension = gpu::ResourceKind::Texture2D;
        gpu::ResourceFormat format = gpu::ResourceFormat::Unknown;
        std::uint32_t width = 0;
        std::uint32_t height = 1;
        std::uint16_t depth = 1;
        std::uint16_t mipLevels = 1;
        TextureUsage usage = TextureUsage::None;
        auto operator<=>(const TextureDescription&) const = default;
    };

    struct PresentationDescription
    {
        gpu::ResourceFormat format = gpu::ResourceFormat::Rgba8Unorm;
        auto operator<=>(const PresentationDescription&) const = default;
    };

    using ResourceDescription = std::variant<
        BufferDescription,
        TextureDescription,
        PresentationDescription>;

    struct ResourceDeclaration
    {
        gpu::ResourceId id;
        std::string name;
        gpu::ResourceLifetimeClass lifetime =
            gpu::ResourceLifetimeClass::Persistent;
        gpu::ResourceUpdateClass update =
            gpu::ResourceUpdateClass::Immutable;
        ResourceDescription description = BufferDescription{};

        std::vector<std::byte> data;

        [[nodiscard]] gpu::ResourceKind Kind() const noexcept;
        [[nodiscard]] gpu::ResourceFormat Format() const noexcept;
        [[nodiscard]] std::uint64_t SizeBytes() const noexcept;
    };

    struct ProgramDeclaration
    {
        gpu::ProgramId id;
        std::string name;
        std::filesystem::path shaderPath;
        std::string vertexEntry;
        std::string pixelEntry;
        std::string computeEntry;
        ProgramInterface programInterface;

        // Compatibility field for modules created before ProgramInterface.
        // Declaring both this and programInterface.parameters is rejected.
        std::vector<gpu::ProgramParameter> parameters;

        [[nodiscard]] const std::vector<gpu::ProgramParameter>&
            BindingParameters() const noexcept;
    };

    struct RasterState
    {
        gpu::PrimitiveTopology topology = gpu::PrimitiveTopology::TriangleList;
        gpu::CompositionMode composition = gpu::CompositionMode::Replace;
        gpu::DepthMode depth = gpu::DepthMode::ReadWrite;

        auto operator<=>(const RasterState&) const = default;
    };

    struct ClearDescription
    {
        bool clearColor = false;
        std::array<float, 4> color{0.025f, 0.035f, 0.060f, 1.0f};
        bool clearDepth = false;
        float depth = 1.0f;
    };

    struct ResourceView
    {
        gpu::ResourceId resource;
        std::uint64_t offsetBytes = 0;
        std::uint64_t sizeBytes = 0;
        std::uint32_t strideBytes = 0;

        constexpr ResourceView() noexcept = default;
        constexpr ResourceView(
            gpu::ResourceId resourceId,
            std::uint64_t offset = 0,
            std::uint64_t size = 0,
            std::uint32_t stride = 0) noexcept
            : resource(resourceId),
              offsetBytes(offset),
              sizeBytes(size),
              strideBytes(stride)
        {
        }

        [[nodiscard]] constexpr bool IsWholeResource() const noexcept
        {
            return offsetBytes == 0 && sizeBytes == 0 && strideBytes == 0;
        }

        [[nodiscard]] constexpr bool IsValid() const noexcept
        {
            return resource.IsValid();
        }

        [[nodiscard]] constexpr std::uint32_t Value() const noexcept
        {
            return resource.Value();
        }

        [[nodiscard]] constexpr operator gpu::ResourceId() const noexcept
        {
            return resource;
        }

        friend constexpr bool operator==(
            const ResourceView& view,
            gpu::ResourceId resourceId) noexcept
        {
            return view.resource == resourceId;
        }

        friend constexpr bool operator==(
            gpu::ResourceId resourceId,
            const ResourceView& view) noexcept
        {
            return resourceId == view.resource;
        }

        bool operator==(const ResourceView&) const = default;
        auto operator<=>(const ResourceView&) const = default;
    };

    struct ResourceBinding
    {
        std::uint32_t parameterIndex = 0;
        ResourceView resource;
        std::uint32_t frameLag = 0;
    };

    struct AttachmentSet
    {
        std::vector<ResourceView> colors;
        ResourceView depth;
    };

    struct RasterWork
    {
        gpu::ProgramId program;
        ResourceView vertexResource;
        std::vector<ResourceBinding> bindings;
        AttachmentSet attachments;
        std::uint32_t vertexCount = 0;
        std::uint32_t firstVertex = 0;
        RasterState rasterState{};
        ClearDescription clear{};
    };

    struct ComputeWork
    {
        gpu::ProgramId program;
        std::vector<ResourceBinding> bindings;
        std::uint32_t groupCountX = 1;
        std::uint32_t groupCountY = 1;
        std::uint32_t groupCountZ = 1;
    };

    struct CopyWork
    {
        gpu::ResourceId source;
        gpu::ResourceId destination;
        std::uint64_t sourceOffset = 0;
        std::uint64_t destinationOffset = 0;
        std::uint64_t sizeBytes = 0;
        std::uint32_t sourceFrameLag = 0;
        std::uint32_t destinationFrameLag = 0;
    };

    struct PresentWork
    {
        gpu::ResourceId source;
    };

    using WorkPayload = std::variant<
        RasterWork,
        ComputeWork,
        CopyWork,
        PresentWork>;

    struct WorkDeclaration
    {
        gpu::WorkId id;
        std::string name;
        std::vector<gpu::ResourceAccess> accesses;
        WorkPayload payload = PresentWork{};

        [[nodiscard]] gpu::ExecutionDomain Domain() const noexcept;
    };

    struct SemanticModule
    {
        std::vector<ResourceDeclaration> resources;
        std::vector<ProgramDeclaration> programs;
        std::vector<WorkDeclaration> works;

        [[nodiscard]] const ResourceDeclaration& Resource(
            gpu::ResourceId id) const;

        [[nodiscard]] const ProgramDeclaration& Program(
            gpu::ProgramId id) const;

        [[nodiscard]] const WorkDeclaration& Work(
            gpu::WorkId id) const;

        [[nodiscard]] std::size_t StructureHash() const;
    };
}
