#pragma once

#include "02_GpuSemantics/Semantics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sge::ir
{
    struct ResourceDeclaration
    {
        gpu::ResourceId id;
        std::string name;
        gpu::ResourceKind kind = gpu::ResourceKind::Buffer;
        gpu::MemoryClass memoryClass = gpu::MemoryClass::Static;
        gpu::ResourceFormat format = gpu::ResourceFormat::Unknown;

        std::uint64_t sizeBytes = 0;
        std::uint32_t strideBytes = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;

        std::vector<std::byte> data;
    };

    struct ProgramDeclaration
    {
        gpu::ProgramId id;
        std::string name;
        std::filesystem::path shaderPath;
        std::string vertexEntry;
        std::string pixelEntry;
        std::vector<gpu::ProgramParameter> parameters;
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

    struct WorkDeclaration
    {
        gpu::WorkId id;
        std::string name;
        gpu::ExecutionDomain domain = gpu::ExecutionDomain::Raster;
        gpu::ProgramId program;

        std::vector<gpu::ResourceAccess> accesses;

        gpu::ResourceId vertexResource;
        gpu::ResourceId constantResource;
        std::uint32_t vertexCount = 0;
        std::uint32_t firstVertex = 0;

        RasterState rasterState{};
        ClearDescription clear{};
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
