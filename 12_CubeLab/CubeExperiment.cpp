#include "12_CubeLab/CubeExperiment.h"

#include <numbers>

namespace
{
    using namespace sge;

    constexpr gpu::ResourceId GeometryResource{0};
    constexpr gpu::ResourceId CubeConstantsResource{1};
    constexpr gpu::ResourceId WorldConstantsResource{2};
    constexpr gpu::ResourceId ColorResource{3};
    constexpr gpu::ResourceId DepthResource{4};
    constexpr gpu::ResourceId FullscreenGeometryResource{5};
    constexpr gpu::ResourceId OffscreenColorResource{6};
    constexpr gpu::ResourceId GeneratedBufferResource{7};
    constexpr gpu::ResourceId CopiedBufferResource{8};

    constexpr gpu::ProgramId ColorProgram{0};
    constexpr gpu::ProgramId BlitProgram{1};
    constexpr gpu::ProgramId GenerateProgram{2};

    constexpr gpu::WorkId GenerateWork{0};
    constexpr gpu::WorkId CopyWork{1};
    constexpr gpu::WorkId CubeWork{2};
    constexpr gpu::WorkId PlaneWork{3};
    constexpr gpu::WorkId LineWork{4};
    constexpr gpu::WorkId BlitWork{5};
    constexpr gpu::WorkId PresentWork{6};
}

namespace sge::cube_lab
{
    ComparisonExperiment::ComparisonExperiment()
        : pgaBox_(experimental::PgaBox::AxisAligned(1.0f)),
          sdfBox_{1.0f, 1.0f, 1.0f}
    {
        classical::ClassicalCubeDescription description;
        description.cubeHalfExtent = pgaBox_.HalfExtent();
        vertices_ = classical::BuildClassicalCubeGeometry(
            description,
            ranges_);
    }

    ir::SemanticModule ComparisonExperiment::Build(
        double elapsedSeconds,
        float aspectRatio) const
    {
        using classical::Float3;
        using classical::Matrix4;

        const Matrix4 view = Matrix4::LookAtLH(
            Float3{5.2f, 4.0f, -6.2f},
            Float3{0.0f, 0.0f, 0.0f},
            Float3{0.0f, 1.0f, 0.0f});

        const Matrix4 projection = Matrix4::PerspectiveFovLH(
            58.0f * std::numbers::pi_v<float> / 180.0f,
            aspectRatio,
            0.1f,
            100.0f);

        const float seconds = static_cast<float>(elapsedSeconds);
        const Matrix4 cubeWorld =
            Matrix4::RotationY(seconds * 0.85f)
            * Matrix4::RotationX(seconds * 0.47f);

        const classical::SceneConstants cubeConstants{
            .worldViewProjection = cubeWorld * view * projection
        };

        const classical::SceneConstants worldConstants{
            .worldViewProjection = view * projection
        };

        const std::vector<classical::Vertex> fullscreenVertices = {
            {{-1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
            {{-1.0f,  3.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
            {{ 3.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}}
        };

        ir::SemanticModule module;

        module.resources = {
            {
                .id = GeometryResource,
                .name = "ClassicalCubeGeometry",
                .memoryClass = gpu::MemoryClass::Static,
                .description = ir::BufferDescription{
                    .sizeBytes = static_cast<std::uint64_t>(
                        vertices_.size() * sizeof(classical::Vertex)),
                    .strideBytes = sizeof(classical::Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination
                },
                .data = classical::ToBytes(vertices_)
            },
            {
                .id = CubeConstantsResource,
                .name = "CubeTransform",
                .memoryClass = gpu::MemoryClass::DynamicPerFrame,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(classical::SceneConstants),
                    .usage = ir::BufferUsage::Constant
                },
                .data = classical::ToBytes(cubeConstants)
            },
            {
                .id = WorldConstantsResource,
                .name = "WorldTransform",
                .memoryClass = gpu::MemoryClass::DynamicPerFrame,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(classical::SceneConstants),
                    .usage = ir::BufferUsage::Constant
                },
                .data = classical::ToBytes(worldConstants)
            },
            {
                .id = ColorResource,
                .name = "PresentationColor",
                .memoryClass = gpu::MemoryClass::External,
                .description = ir::PresentationDescription{
                    .format = gpu::ResourceFormat::Rgba8Unorm
                },
                .data = {}
            },
            {
                .id = DepthResource,
                .name = "MainDepth",
                .memoryClass = gpu::MemoryClass::Transient,
                .description = ir::TextureDescription{
                    .dimension = gpu::ResourceKind::Texture2D,
                    .format = gpu::ResourceFormat::Depth32Float,
                    .width = 0,
                    .height = 0,
                    .usage = ir::TextureUsage::DepthAttachment
                },
                .data = {}
            },
            {
                .id = FullscreenGeometryResource,
                .name = "FullscreenTriangle",
                .memoryClass = gpu::MemoryClass::Static,
                .description = ir::BufferDescription{
                    .sizeBytes = static_cast<std::uint64_t>(
                        fullscreenVertices.size() * sizeof(classical::Vertex)),
                    .strideBytes = sizeof(classical::Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination
                },
                .data = classical::ToBytes(fullscreenVertices)
            },
            {
                .id = OffscreenColorResource,
                .name = "OffscreenColor",
                .memoryClass = gpu::MemoryClass::Transient,
                .description = ir::TextureDescription{
                    .dimension = gpu::ResourceKind::Texture2D,
                    .format = gpu::ResourceFormat::Rgba8Unorm,
                    .width = 0,
                    .height = 0,
                    .usage = ir::TextureUsage::ColorAttachment
                        | ir::TextureUsage::Sampled
                },
                .data = {}
            },
            {
                .id = GeneratedBufferResource,
                .name = "ComputeGeneratedBuffer",
                .memoryClass = gpu::MemoryClass::Transient,
                .description = ir::BufferDescription{
                    .sizeBytes = 256,
                    .usage = ir::BufferUsage::Storage
                        | ir::BufferUsage::CopySource
                }
            },
            {
                .id = CopiedBufferResource,
                .name = "CopiedBuffer",
                .memoryClass = gpu::MemoryClass::Transient,
                .description = ir::BufferDescription{
                    .sizeBytes = 256,
                    .usage = ir::BufferUsage::CopyDestination
                }
            }
        };

        module.programs = {
            {
                .id = ColorProgram,
                .name = "BasicColor",
                .shaderPath = "Shaders/BasicColor.hlsl",
                .vertexEntry = "VSMain",
                .pixelEntry = "PSMain",
                .parameters = {
                    {
                        .kind = gpu::ProgramParameterKind::ConstantBuffer,
                        .stage = gpu::ProgramStage::Vertex,
                        .registerIndex = 0,
                        .registerSpace = 0
                    }
                }
            },
            {
                .id = BlitProgram,
                .name = "OffscreenBlit",
                .shaderPath = "Shaders/OffscreenBlit.hlsl",
                .vertexEntry = "VSMain",
                .pixelEntry = "PSMain",
                .parameters = {
                    {
                        .name = "SourceTexture",
                        .kind = gpu::ProgramParameterKind::ShaderResource,
                        .stage = gpu::ProgramStage::Pixel,
                        .registerIndex = 0,
                        .registerSpace = 0
                    }
                }
            },
            {
                .id = GenerateProgram,
                .name = "GenerateBuffer",
                .shaderPath = "Shaders/GenerateBuffer.hlsl",
                .computeEntry = "CSMain",
                .parameters = {
                    {
                        .name = "OutputBuffer",
                        .kind = gpu::ProgramParameterKind::UnorderedAccess,
                        .stage = gpu::ProgramStage::Compute,
                        .registerIndex = 0,
                        .registerSpace = 0
                    }
                }
            }
        };

        module.works = {
            {
                .id = GenerateWork,
                .name = "GenerateBufferOnComputeQueue",
                .accesses = {{GeneratedBufferResource,
                    gpu::AccessMode::Write, gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = GenerateProgram,
                    .bindings = {{0, GeneratedBufferResource}},
                    .groupCountX = 1
                }
            },
            {
                .id = CopyWork,
                .name = "CopyGeneratedBuffer",
                .accesses = {
                    {GeneratedBufferResource, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {CopiedBufferResource, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}
                },
                .payload = ir::CopyWork{
                    .source = GeneratedBufferResource,
                    .destination = CopiedBufferResource,
                    .sizeBytes = 256
                }
            },
            {
                .id = CubeWork,
                .name = "DrawCube",
                .accesses = {
                    {
                        GeometryResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput
                    },
                    {
                        CubeConstantsResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::ConstantInput
                    },
                    {
                        OffscreenColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = GeometryResource,
                    .bindings = {{0, CubeConstantsResource}},
                    .attachments = {{OffscreenColorResource}, DepthResource},
                    .vertexCount = ranges_.cubeCount,
                    .firstVertex = ranges_.cubeFirst,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::TriangleList,
                        .composition = gpu::CompositionMode::Replace,
                        .depth = gpu::DepthMode::ReadWrite
                    },
                    .clear = {
                        .clearColor = true,
                        .color = {0.025f, 0.035f, 0.060f, 1.0f},
                        .clearDepth = true,
                        .depth = 1.0f
                    }
                }
            },
            {
                .id = PlaneWork,
                .name = "DrawCoordinatePlanes",
                .accesses = {
                    {
                        GeometryResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput
                    },
                    {
                        WorldConstantsResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::ConstantInput
                    },
                    {
                        OffscreenColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = GeometryResource,
                    .bindings = {{0, WorldConstantsResource}},
                    .attachments = {{OffscreenColorResource}, DepthResource},
                    .vertexCount = ranges_.planesCount,
                    .firstVertex = ranges_.planesFirst,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::TriangleList,
                        .composition = gpu::CompositionMode::AlphaOver,
                        .depth = gpu::DepthMode::ReadOnly
                    }
                }
            },
            {
                .id = LineWork,
                .name = "DrawGridAndAxes",
                .accesses = {
                    {
                        GeometryResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput
                    },
                    {
                        WorldConstantsResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::ConstantInput
                    },
                    {
                        OffscreenColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .payload = ir::RasterWork{
                    .program = ColorProgram,
                    .vertexResource = GeometryResource,
                    .bindings = {{0, WorldConstantsResource}},
                    .attachments = {{OffscreenColorResource}, DepthResource},
                    .vertexCount = ranges_.linesCount,
                    .firstVertex = ranges_.linesFirst,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::LineList,
                        .composition = gpu::CompositionMode::AlphaOver,
                        .depth = gpu::DepthMode::ReadOnly
                    }
                }
            },
            {
                .id = BlitWork,
                .name = "BlitOffscreenToPresentation",
                .accesses = {
                    {FullscreenGeometryResource, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {OffscreenColorResource, gpu::AccessMode::Read,
                        gpu::ResourceRole::ProgramInput},
                    {ColorResource, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}
                },
                .payload = ir::RasterWork{
                    .program = BlitProgram,
                    .vertexResource = FullscreenGeometryResource,
                    .bindings = {{0, OffscreenColorResource}},
                    .attachments = {{ColorResource}, {}},
                    .vertexCount = 3,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::TriangleList,
                        .composition = gpu::CompositionMode::Replace,
                        .depth = gpu::DepthMode::Disabled
                    }
                }
            },
            {
                .id = PresentWork,
                .name = "Present",
                .accesses = {
                    {
                        ColorResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::Presentation
                    }
                },
                .payload = ir::PresentWork{ColorResource}
            }
        };

        return module;
    }

    const experimental::PgaBox& ComparisonExperiment::PgaModel() const noexcept
    {
        return pgaBox_;
    }

    const experimental::SdfBox& ComparisonExperiment::SdfModel() const noexcept
    {
        return sdfBox_;
    }
}
