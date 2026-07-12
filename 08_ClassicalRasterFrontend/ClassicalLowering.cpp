#include "08_ClassicalRasterFrontend/ClassicalLowering.h"

#include <numbers>

namespace
{
    using namespace sge;

    constexpr gpu::ResourceId Geometry{0};
    constexpr gpu::ResourceId CubeConstants{1};
    constexpr gpu::ResourceId WorldConstants{2};
    constexpr gpu::ResourceId PresentationColor{3};
    constexpr gpu::ResourceId Depth{4};
    constexpr gpu::ResourceId FullscreenGeometry{5};
    constexpr gpu::ResourceId OffscreenColor{6};
    constexpr gpu::ResourceId GeneratedVertices{7};
    constexpr gpu::ResourceId CopiedVertices{8};
    constexpr gpu::ResourceId TemporalHistory{9};

    constexpr gpu::ProgramId ColorProgram{0};
    constexpr gpu::ProgramId BlitProgram{1};
    constexpr gpu::ProgramId GenerateProgram{2};
    constexpr gpu::ProgramId TemporalProgram{3};

    constexpr std::uint64_t GeneratedVertexCount = 3;
    constexpr std::uint64_t GeneratedVertexBufferSize =
        GeneratedVertexCount * sizeof(classical::Vertex);
}

namespace sge::classical
{
    ir::SemanticModule ClassicalRasterLowering::Lower(
        const ClassicalScene& scene) const
    {
        GeometryRanges ranges{};
        const auto vertices = BuildClassicalCubeGeometry(
            scene.geometry, ranges);

        const Matrix4 view = Matrix4::LookAtLH(
            scene.cameraPosition, scene.cameraTarget, scene.cameraUp);
        const Matrix4 projection = Matrix4::PerspectiveFovLH(
            scene.verticalFieldOfViewDegrees
                * std::numbers::pi_v<float> / 180.0f,
            scene.aspectRatio,
            0.1f,
            100.0f);
        const float seconds = static_cast<float>(scene.elapsedSeconds);
        const Matrix4 cubeWorld = Matrix4::RotationY(seconds * 0.85f)
            * Matrix4::RotationX(seconds * 0.47f);
        const SceneConstants cubeData{cubeWorld * view * projection};
        const SceneConstants worldData{view * projection};

        const std::vector<Vertex> fullscreenVertices = {
            {{-1.0f, -1.0f, 0.0f}, {1, 1, 1, 1}},
            {{-1.0f,  3.0f, 0.0f}, {1, 1, 1, 1}},
            {{ 3.0f, -1.0f, 0.0f}, {1, 1, 1, 1}}
        };

        ir::SemanticModule module;
        module.resources = {
            {
                .id = Geometry,
                .name = "ClassicalGeometry",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = vertices.size() * sizeof(Vertex),
                    .strideBytes = sizeof(Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination},
                .data = ToBytes(vertices)
            },
            {
                .id = CubeConstants,
                .name = "CubeTransform",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::CpuUpdated,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(SceneConstants),
                    .usage = ir::BufferUsage::Constant},
                .data = ToBytes(cubeData)
            },
            {
                .id = WorldConstants,
                .name = "WorldTransform",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::CpuUpdated,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(SceneConstants),
                    .usage = ir::BufferUsage::Constant},
                .data = ToBytes(worldData)
            },
            {
                .id = PresentationColor,
                .name = "PresentationColor",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}
            },
            {
                .id = Depth,
                .name = "MainDepth",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::TextureDescription{
                    .format = gpu::ResourceFormat::Depth32Float,
                    .width = 0,
                    .height = 0,
                    .usage = ir::TextureUsage::DepthAttachment}
            },
            {
                .id = FullscreenGeometry,
                .name = "FullscreenTriangle",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = fullscreenVertices.size() * sizeof(Vertex),
                    .strideBytes = sizeof(Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination},
                .data = ToBytes(fullscreenVertices)
            },
            {
                .id = OffscreenColor,
                .name = "OffscreenColor",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::TextureDescription{
                    .format = gpu::ResourceFormat::Rgba8Unorm,
                    .width = 0,
                    .height = 0,
                    .usage = ir::TextureUsage::ColorAttachment
                        | ir::TextureUsage::Sampled}
            },
            {
                .id = GeneratedVertices,
                .name = "ComputeGeneratedVertices",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = GeneratedVertexBufferSize,
                    .usage = ir::BufferUsage::Storage
                        | ir::BufferUsage::CopySource}
            },
            {
                .id = CopiedVertices,
                .name = "ComputeGeneratedVertexBuffer",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = GeneratedVertexBufferSize,
                    .strideBytes = sizeof(Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination}
            },
            {
                .id = TemporalHistory,
                .name = "TemporalFrameCounter",
                .lifetime = gpu::ResourceLifetimeClass::Temporal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = 16,
                    .usage = ir::BufferUsage::Storage},
                .data = std::vector<std::byte>(16)
            }
        };

        module.programs = {
            {
                .id = ColorProgram,
                .name = "BasicColor",
                .shaderPath = "Shaders/BasicColor.hlsl",
                .vertexEntry = "VSMain",
                .pixelEntry = "PSMain",
                .parameters = {{
                    .name = "Scene",
                    .kind = gpu::ProgramParameterKind::ConstantBuffer,
                    .stage = gpu::ProgramStage::Vertex}}
            },
            {
                .id = BlitProgram,
                .name = "OffscreenBlit",
                .shaderPath = "Shaders/OffscreenBlit.hlsl",
                .vertexEntry = "VSMain",
                .pixelEntry = "PSMain",
                .parameters = {{
                    .name = "SourceTexture",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Pixel}}
            },
            {
                .id = GenerateProgram,
                .name = "GenerateTriangleVertices",
                .shaderPath = "Shaders/GenerateBuffer.hlsl",
                .computeEntry = "CSMain",
                .parameters = {{
                    .name = "OutputBuffer",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute}}
            },
            {
                .id = TemporalProgram,
                .name = "TemporalAccumulate",
                .shaderPath = "Shaders/TemporalAccumulate.hlsl",
                .computeEntry = "CSMain",
                .parameters = {
                    {.name = "PreviousHistory",
                        .kind = gpu::ProgramParameterKind::ShaderResource,
                        .stage = gpu::ProgramStage::Compute},
                    {.name = "CurrentHistory",
                        .kind = gpu::ProgramParameterKind::UnorderedAccess,
                        .stage = gpu::ProgramStage::Compute}
                }
            }
        };

        module.works = {
            {
                .id = gpu::WorkId{0},
                .name = "GenerateTriangleOnComputeQueue",
                .accesses = {{GeneratedVertices, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = GenerateProgram,
                    .bindings = {{0, GeneratedVertices}}}
            },
            {
                .id = gpu::WorkId{1},
                .name = "CopyGeneratedTriangle",
                .accesses = {
                    {GeneratedVertices, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {CopiedVertices, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{
                    .source = GeneratedVertices,
                    .destination = CopiedVertices,
                    .sizeBytes = GeneratedVertexBufferSize}
            },
            {
                .id = gpu::WorkId{2},
                .name = "DrawCube",
                .accesses = {
                    {Geometry, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {CubeConstants, gpu::AccessMode::Read, gpu::ResourceRole::ConstantInput},
                    {OffscreenColor, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput},
                    {Depth, gpu::AccessMode::Write, gpu::ResourceRole::DepthOutput}},
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = Geometry,
                    .bindings = {{0, CubeConstants}},
                    .attachments = {{OffscreenColor}, Depth},
                    .vertexCount = ranges.cubeCount,
                    .firstVertex = ranges.cubeFirst,
                    .clear = {.clearColor = true,
                        .color = {0.025f, 0.035f, 0.060f, 1.0f},
                        .clearDepth = true}}
            },
            {
                .id = gpu::WorkId{3},
                .name = "DrawCoordinatePlanes",
                .accesses = {
                    {Geometry, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {WorldConstants, gpu::AccessMode::Read, gpu::ResourceRole::ConstantInput},
                    {OffscreenColor, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput},
                    {Depth, gpu::AccessMode::Read, gpu::ResourceRole::DepthOutput}},
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = Geometry,
                    .bindings = {{0, WorldConstants}},
                    .attachments = {{OffscreenColor}, Depth},
                    .vertexCount = ranges.planesCount,
                    .firstVertex = ranges.planesFirst,
                    .rasterState = {
                        .composition = gpu::CompositionMode::AlphaOver,
                        .depth = gpu::DepthMode::ReadOnly}}
            },
            {
                .id = gpu::WorkId{4},
                .name = "DrawGridAndAxes",
                .accesses = {
                    {Geometry, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {WorldConstants, gpu::AccessMode::Read, gpu::ResourceRole::ConstantInput},
                    {OffscreenColor, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput},
                    {Depth, gpu::AccessMode::Read, gpu::ResourceRole::DepthOutput}},
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = Geometry,
                    .bindings = {{0, WorldConstants}},
                    .attachments = {{OffscreenColor}, Depth},
                    .vertexCount = ranges.linesCount,
                    .firstVertex = ranges.linesFirst,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::LineList,
                        .composition = gpu::CompositionMode::AlphaOver,
                        .depth = gpu::DepthMode::ReadOnly}}
            },
            {
                .id = gpu::WorkId{5},
                .name = "DrawComputeGeneratedTriangle",
                .accesses = {
                    {CopiedVertices, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {WorldConstants, gpu::AccessMode::Read, gpu::ResourceRole::ConstantInput},
                    {OffscreenColor, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = CopiedVertices,
                    .bindings = {{0, WorldConstants}},
                    .attachments = {{OffscreenColor}, {}},
                    .vertexCount = static_cast<std::uint32_t>(GeneratedVertexCount),
                    .rasterState = {
                        .composition = gpu::CompositionMode::AlphaOver,
                        .depth = gpu::DepthMode::Disabled}}
            },
            {
                .id = gpu::WorkId{6},
                .name = "BlitOffscreenToPresentation",
                .accesses = {
                    {FullscreenGeometry, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {OffscreenColor, gpu::AccessMode::Read, gpu::ResourceRole::ProgramInput},
                    {PresentationColor, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = BlitProgram,
                    .vertexResource = FullscreenGeometry,
                    .bindings = {{0, OffscreenColor}},
                    .attachments = {{PresentationColor}, {}},
                    .vertexCount = 3,
                    .rasterState = {.depth = gpu::DepthMode::Disabled}}
            },
            {
                .id = gpu::WorkId{7},
                .name = "Present",
                .accesses = {{PresentationColor, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{PresentationColor}
            },
            {
                .id = gpu::WorkId{8},
                .name = "AccumulatePreviousFrame",
                .accesses = {
                    {TemporalHistory, gpu::AccessMode::Read,
                        gpu::ResourceRole::ProgramInput, 1},
                    {TemporalHistory, gpu::AccessMode::Write,
                        gpu::ResourceRole::ProgramOutput, 0}},
                .payload = ir::ComputeWork{
                    .program = TemporalProgram,
                    .bindings = {
                        {0, TemporalHistory, 1},
                        {1, TemporalHistory, 0}}}
            }
        };

        return module;
    }
}
