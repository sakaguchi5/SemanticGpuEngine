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

    constexpr gpu::ProgramId ColorProgram{0};

    constexpr gpu::WorkId CubeWork{0};
    constexpr gpu::WorkId PlaneWork{1};
    constexpr gpu::WorkId LineWork{2};
    constexpr gpu::WorkId PresentWork{3};
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

        ir::SemanticModule module;

        module.resources = {
            {
                .id = GeometryResource,
                .name = "ClassicalCubeGeometry",
                .kind = gpu::ResourceKind::Buffer,
                .memoryClass = gpu::MemoryClass::Static,
                .format = gpu::ResourceFormat::Unknown,
                .sizeBytes = static_cast<std::uint64_t>(
                    vertices_.size() * sizeof(classical::Vertex)),
                .strideBytes = sizeof(classical::Vertex),
                .data = classical::ToBytes(vertices_)
            },
            {
                .id = CubeConstantsResource,
                .name = "CubeTransform",
                .kind = gpu::ResourceKind::Buffer,
                .memoryClass = gpu::MemoryClass::DynamicPerFrame,
                .format = gpu::ResourceFormat::Unknown,
                .sizeBytes = sizeof(classical::SceneConstants),
                .data = classical::ToBytes(cubeConstants)
            },
            {
                .id = WorldConstantsResource,
                .name = "WorldTransform",
                .kind = gpu::ResourceKind::Buffer,
                .memoryClass = gpu::MemoryClass::DynamicPerFrame,
                .format = gpu::ResourceFormat::Unknown,
                .sizeBytes = sizeof(classical::SceneConstants),
                .data = classical::ToBytes(worldConstants)
            },
            {
                .id = ColorResource,
                .name = "PresentationColor",
                .kind = gpu::ResourceKind::Presentation,
                .memoryClass = gpu::MemoryClass::External,
                .format = gpu::ResourceFormat::Rgba8Unorm,
                .data = {}
            },
            {
                .id = DepthResource,
                .name = "MainDepth",
                .kind = gpu::ResourceKind::Texture2D,
                .memoryClass = gpu::MemoryClass::Transient,
                .format = gpu::ResourceFormat::Depth32Float,
                .data = {}
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
            }
        };

        module.works = {
            {
                .id = CubeWork,
                .name = "DrawCube",
                .domain = gpu::ExecutionDomain::Raster,
                .program = ColorProgram,
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
                        ColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .vertexResource = GeometryResource,
                .constantResource = CubeConstantsResource,
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
            },
            {
                .id = PlaneWork,
                .name = "DrawCoordinatePlanes",
                .domain = gpu::ExecutionDomain::Raster,
                .program = ColorProgram,
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
                        ColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .vertexResource = GeometryResource,
                .constantResource = WorldConstantsResource,
                .vertexCount = ranges_.planesCount,
                .firstVertex = ranges_.planesFirst,
                .rasterState = {
                    .topology = gpu::PrimitiveTopology::TriangleList,
                    .composition = gpu::CompositionMode::AlphaOver,
                    .depth = gpu::DepthMode::ReadOnly
                }
            },
            {
                .id = LineWork,
                .name = "DrawGridAndAxes",
                .domain = gpu::ExecutionDomain::Raster,
                .program = ColorProgram,
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
                        ColorResource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput
                    },
                    {
                        DepthResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::DepthOutput
                    }
                },
                .vertexResource = GeometryResource,
                .constantResource = WorldConstantsResource,
                .vertexCount = ranges_.linesCount,
                .firstVertex = ranges_.linesFirst,
                .rasterState = {
                    .topology = gpu::PrimitiveTopology::LineList,
                    .composition = gpu::CompositionMode::AlphaOver,
                    .depth = gpu::DepthMode::ReadOnly
                }
            },
            {
                .id = PresentWork,
                .name = "Present",
                .domain = gpu::ExecutionDomain::Present,
                .program = gpu::ProgramId{},
                .accesses = {
                    {
                        ColorResource,
                        gpu::AccessMode::Read,
                        gpu::ResourceRole::Presentation
                    }
                },
                .vertexResource = gpu::ResourceId{},
                .constantResource = gpu::ResourceId{}
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
