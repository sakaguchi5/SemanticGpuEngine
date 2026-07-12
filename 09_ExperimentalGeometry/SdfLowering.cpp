#include "09_ExperimentalGeometry/SdfLowering.h"

#include "08_ClassicalRasterFrontend/ClassicalRaster.h"

#include <algorithm>
#include <numbers>

namespace
{
    using namespace sge;

    constexpr gpu::ResourceId FullscreenGeometry{0};
    constexpr gpu::ResourceId RayMarchConstants{1};
    constexpr gpu::ResourceId PresentationColor{2};
    constexpr gpu::ProgramId RayMarchProgram{0};

    struct alignas(16) SdfSceneConstants
    {
        classical::Float4 cameraPosition;
        classical::Float4 cameraRight;
        classical::Float4 cameraUp;
        classical::Float4 cameraForward;
        classical::Float4 projection;
        classical::Float4 box;
        classical::Float4 rayMarch;
    };

    static_assert(sizeof(SdfSceneConstants) == 112);
}

namespace sge::experimental
{
    ir::SemanticModule SdfRayMarchLowering::Lower(
        const SdfScene& scene) const
    {
        using classical::Float4;
        using classical::Vertex;

        const auto forward = classical::Normalize(
            scene.cameraTarget - scene.cameraPosition);
        const auto right = classical::Normalize(
            classical::Cross(scene.cameraUp, forward));
        const auto up = classical::Cross(forward, right);
        const float tangentHalfFieldOfView = std::tan(
            scene.verticalFieldOfViewDegrees
                * std::numbers::pi_v<float> / 360.0f);

        const SdfSceneConstants constants{
            .cameraPosition = {scene.cameraPosition.x,
                scene.cameraPosition.y, scene.cameraPosition.z, 1.0f},
            .cameraRight = {right.x, right.y, right.z, 0.0f},
            .cameraUp = {up.x, up.y, up.z, 0.0f},
            .cameraForward = {forward.x, forward.y, forward.z, 0.0f},
            .projection = {tangentHalfFieldOfView,
                std::max(scene.aspectRatio, 0.001f),
                static_cast<float>(scene.elapsedSeconds), 0.0f},
            .box = {scene.box.halfExtentX, scene.box.halfExtentY,
                scene.box.halfExtentZ, 0.0f},
            .rayMarch = {scene.maximumDistance, scene.surfaceTolerance,
                static_cast<float>(scene.maximumSteps), 0.0f}
        };

        const std::vector<Vertex> fullscreenVertices = {
            {{-1.0f, -1.0f, 0.0f}, {1, 1, 1, 1}},
            {{-1.0f,  3.0f, 0.0f}, {1, 1, 1, 1}},
            {{ 3.0f, -1.0f, 0.0f}, {1, 1, 1, 1}}
        };

        ir::SemanticModule module;
        module.resources = {
            {
                .id = FullscreenGeometry,
                .name = "SdfFullscreenTriangle",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = fullscreenVertices.size() * sizeof(Vertex),
                    .strideBytes = sizeof(Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination},
                .data = classical::ToBytes(fullscreenVertices)
            },
            {
                .id = RayMarchConstants,
                .name = "SdfRayMarchConstants",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::CpuUpdated,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(SdfSceneConstants),
                    .usage = ir::BufferUsage::Constant},
                .data = classical::ToBytes(constants)
            },
            {
                .id = PresentationColor,
                .name = "PresentationColor",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}
            }
        };

        module.programs = {{
            .id = RayMarchProgram,
            .name = "SdfBoxRayMarch",
            .shaderPath = "Shaders/SdfRayMarch.hlsl",
            .vertexEntry = "VSMain",
            .pixelEntry = "PSMain",
            .parameters = {{
                .name = "SdfScene",
                .kind = gpu::ProgramParameterKind::ConstantBuffer,
                .stage = gpu::ProgramStage::Pixel}}
        }};

        module.works = {
            {
                .id = gpu::WorkId{0},
                .name = "RayMarchSdfBox",
                .accesses = {
                    {FullscreenGeometry, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {RayMarchConstants, gpu::AccessMode::Read,
                        gpu::ResourceRole::ConstantInput},
                    {PresentationColor, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = RayMarchProgram,
                    .vertexResource = FullscreenGeometry,
                    .bindings = {{0, RayMarchConstants}},
                    .attachments = {{PresentationColor}, {}},
                    .vertexCount = 3,
                    .rasterState = {.depth = gpu::DepthMode::Disabled},
                    .clear = {.clearColor = true,
                        .color = {0.025f, 0.035f, 0.060f, 1.0f}}}
            },
            {
                .id = gpu::WorkId{1},
                .name = "PresentSdfImage",
                .accesses = {{PresentationColor, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{PresentationColor}
            }
        };

        return module;
    }
}
