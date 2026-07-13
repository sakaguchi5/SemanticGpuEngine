#include "11_Tests/D3D12IntegrationTests.h"
#include "11_Tests/V1AcceptanceTests.h"

#include "04_RenderCompiler/CompiledRenderPackage.h"
#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"
#include "07_D3D12Backend/D3D12Backend.h"
#include "08_ClassicalRasterFrontend/ClassicalRaster.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <windows.h>

namespace
{
    bool CompileFails(
        const sge::ir::SemanticModule& module,
        sge::gpu::DeviceCapabilities capabilities)
    {
        try
        {
            return !sge::compiler::RenderPackageCompiler{}
                .Compile(module, capabilities).Succeeded();
        }
        catch (const std::runtime_error&)
        {
            return true;
        }
    }

    sge::ir::ResourceDeclaration StateValidationBuffer(
        sge::gpu::ResourceId id,
        sge::gpu::ResourceLifetimeClass lifetime)
    {
        using namespace sge;

        ir::ResourceDeclaration result{
            .id = id,
            .name = "StateValidationBuffer",
            .lifetime = lifetime,
            .update = lifetime == gpu::ResourceLifetimeClass::Persistent
                ? gpu::ResourceUpdateClass::Immutable
                : gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 64,
                .strideBytes = 4,
                .usage = ir::BufferUsage::Vertex
                    | ir::BufferUsage::Storage
                    | ir::BufferUsage::CopyDestination}
        };
        if (lifetime == gpu::ResourceLifetimeClass::Persistent)
        {
            result.data = std::vector<std::byte>(64);
        }
        return result;
    }

    sge::ir::SemanticModule BuildTwoBindingStateModule(
        sge::gpu::ProgramParameterKind firstKind,
        sge::gpu::ProgramParameterKind secondKind,
        sge::gpu::AccessMode firstAccess,
        sge::gpu::AccessMode secondAccess,
        std::uint32_t firstFrameLag = 0,
        std::uint32_t secondFrameLag = 0,
        sge::gpu::ResourceLifetimeClass lifetime =
            sge::gpu::ResourceLifetimeClass::FrameLocal)
    {
        using namespace sge;
        constexpr gpu::ResourceId data{0};

        ir::SemanticModule module;
        module.resources.push_back(StateValidationBuffer(data, lifetime));
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "StateValidationCompute",
            .computeEntry = "CSMain",
            .parameters = {
                {.name = "First",
                    .kind = firstKind,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0},
                {.name = "Second",
                    .kind = secondKind,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 1}}
        });

        const auto roleFor = [](gpu::ProgramParameterKind kind)
        {
            return kind == gpu::ProgramParameterKind::ShaderResource
                ? gpu::ResourceRole::ProgramInput
                : gpu::ResourceRole::ProgramOutput;
        };

        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "StateValidationComputeWork",
            .accesses = {
                {data, firstAccess, roleFor(firstKind), firstFrameLag},
                {data, secondAccess, roleFor(secondKind), secondFrameLag}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {
                    {0, ir::ResourceView{data, 0, 32, 4}, firstFrameLag},
                    {1, ir::ResourceView{data, 32, 32, 4}, secondFrameLag}}}
        });
        return module;
    }

    sge::ir::SemanticModule BuildSharedReadStateRasterModule(
        sge::gpu::ResourceLifetimeClass lifetime)
    {
        using namespace sge;
        constexpr gpu::ResourceId data{0};
        constexpr gpu::ResourceId presentation{1};

        auto shared = StateValidationBuffer(data, lifetime);
        shared.name = "VertexAndShaderInput";
        shared.description = ir::BufferDescription{
            .sizeBytes = 64,
            .strideBytes = 16,
            .usage = ir::BufferUsage::Vertex
                | ir::BufferUsage::Storage
                | ir::BufferUsage::CopyDestination};
        if (lifetime == gpu::ResourceLifetimeClass::Persistent)
        {
            shared.data = std::vector<std::byte>(64);
        }

        ir::SemanticModule module;
        module.resources = {
            std::move(shared),
            {.id = presentation,
                .name = "StateValidationPresentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "VertexAndShaderRead",
            .vertexEntry = "VSMain",
            .pixelEntry = "PSMain",
            .programInterface = {
                .parameters = {{.name = "Data",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Pixel}},
                .vertexInputs = {{
                    "POSITION", 0, ir::VertexElementFormat::Float4,
                    0, 0, ir::VertexInputRate::PerVertex, 0}},
                .colorOutputCount = 1,
                .depthAttachmentAllowed = false}
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "ReadOneInstanceInTwoStates",
            .accesses = {
                {data, gpu::AccessMode::Read,
                    gpu::ResourceRole::VertexInput},
                {data, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput},
                {presentation, gpu::AccessMode::Write,
                    gpu::ResourceRole::ColorOutput}},
            .payload = ir::RasterWork{
                .program = gpu::ProgramId{0},
                .vertexResource = data,
                .bindings = {{0, data}},
                .attachments = {{presentation}, {}},
                .vertexCount = 4,
                .rasterState = {.depth = gpu::DepthMode::Disabled}}
        });
        return module;
    }

    void RunResourceStateValidatorTests()
    {
        using namespace sge;

        gpu::DeviceCapabilities computeCapabilities;
        computeCapabilities.computeExecution = true;

        const auto readWrite = BuildTwoBindingStateModule(
            gpu::ProgramParameterKind::ShaderResource,
            gpu::ProgramParameterKind::UnorderedAccess,
            gpu::AccessMode::Read,
            gpu::AccessMode::Write);
        if (!CompileFails(readWrite, computeCapabilities))
        {
            throw std::runtime_error(
                "Resource-state validator accepted SRV/UAV access to one instance.");
        }

        const auto twoWrites = BuildTwoBindingStateModule(
            gpu::ProgramParameterKind::UnorderedAccess,
            gpu::ProgramParameterKind::UnorderedAccess,
            gpu::AccessMode::Write,
            gpu::AccessMode::Write);
        if (!CompileFails(twoWrites, computeCapabilities))
        {
            throw std::runtime_error(
                "Resource-state validator accepted multiple writes to one instance.");
        }

        const auto twoReads = BuildTwoBindingStateModule(
            gpu::ProgramParameterKind::ShaderResource,
            gpu::ProgramParameterKind::ShaderResource,
            gpu::AccessMode::Read,
            gpu::AccessMode::Read);
        if (CompileFails(twoReads, computeCapabilities))
        {
            throw std::runtime_error(
                "Resource-state validator rejected compatible read-only views.");
        }

        const auto temporalReadWrite = BuildTwoBindingStateModule(
            gpu::ProgramParameterKind::ShaderResource,
            gpu::ProgramParameterKind::UnorderedAccess,
            gpu::AccessMode::Read,
            gpu::AccessMode::Write,
            1,
            0,
            gpu::ResourceLifetimeClass::Temporal);
        if (CompileFails(temporalReadWrite, computeCapabilities))
        {
            throw std::runtime_error(
                "Resource-state validator merged different temporal instances.");
        }

        const auto persistentReads = BuildSharedReadStateRasterModule(
            gpu::ResourceLifetimeClass::Persistent);
        if (CompileFails(persistentReads, gpu::DeviceCapabilities{}))
        {
            throw std::runtime_error(
                "Resource-state validator rejected a persistent read envelope.");
        }

        const auto frameLocalReads = BuildSharedReadStateRasterModule(
            gpu::ResourceLifetimeClass::FrameLocal);
        if (!CompileFails(frameLocalReads, gpu::DeviceCapabilities{}))
        {
            throw std::runtime_error(
                "Resource-state validator accepted uncombined frame-local read states.");
        }
    }

    template<class T, std::size_t N>
    std::vector<std::byte> ArrayBytes(const std::array<T, N>& values)
    {
        std::vector<std::byte> result(sizeof(values));
        std::memcpy(result.data(), values.data(), sizeof(values));
        return result;
    }

    sge::ir::SemanticModule BuildAdvancedPackageModule()
    {
        using namespace sge;
        constexpr gpu::ResourceId positions{0};
        constexpr gpu::ResourceId colors{1};
        constexpr gpu::ResourceId indices{2};
        constexpr gpu::ResourceId presentation{3};

        struct Position { float x, y, z; };
        struct Color { float r, g, b, a; };
        const std::array<Position, 3> positionData{{
            {-0.24f, -0.42f, 0.0f},
            { 0.24f, -0.42f, 0.0f},
            { 0.00f,  0.42f, 0.0f}}};
        const std::array<Color, 2> colorData{{
            {1.0f, 0.2f, 0.1f, 1.0f},
            {0.1f, 0.5f, 1.0f, 1.0f}}};
        const std::array<std::uint16_t, 3> indexData{{0, 1, 2}};

        ir::SemanticModule module;
        module.resources = {
            {.id = positions, .name = "AdvancedPositions",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(positionData),
                    .strideBytes = sizeof(Position),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination},
                .data = ArrayBytes(positionData)},
            {.id = colors, .name = "AdvancedInstanceColors",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(colorData),
                    .strideBytes = sizeof(Color),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination},
                .data = ArrayBytes(colorData)},
            {.id = indices, .name = "AdvancedIndices",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(indexData),
                    .strideBytes = sizeof(std::uint16_t),
                    .usage = ir::BufferUsage::Index
                        | ir::BufferUsage::CopyDestination},
                .data = ArrayBytes(indexData)},
            {.id = presentation, .name = "AdvancedPresentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "AdvancedPackageRaster",
            .shaderPath = "Shaders/IntegrationGeneralRaster.hlsl",
            .vertexEntry = "VSMain",
            .pixelEntry = "PSMain",
            .programInterface = {
                .parameters = {},
                .vertexInputs = {
                    {"POSITION", 0, ir::VertexElementFormat::Float3,
                        0, 0, ir::VertexInputRate::PerVertex, 0},
                    {"COLOR", 0, ir::VertexElementFormat::Float4,
                        1, 0, ir::VertexInputRate::PerInstance, 1}},
                .colorOutputCount = 1,
                .depthAttachmentAllowed = false}});
        module.works = {
            {.id = gpu::WorkId{0}, .name = "AdvancedIndexedInstancedDraw",
                .accesses = {
                    {positions, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {colors, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {indices, gpu::AccessMode::Read,
                        gpu::ResourceRole::IndexInput},
                    {presentation, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = gpu::ProgramId{0},
                    .vertexResource = positions,
                    .attachments = {{presentation}, {}},
                    .vertexCount = 0,
                    .rasterState = {
                        .topology = gpu::PrimitiveTopology::TriangleList,
                        .composition = gpu::CompositionMode::Replace,
                        .depth = gpu::DepthMode::Disabled,
                        .cull = ir::CullMode::Back,
                        .fill = ir::FillMode::Solid,
                        .frontFace = ir::FrontFace::Clockwise,
                        .sampleCount = 1},
                    .clear = {
                        .clearColor = true,
                        .color = {0.01f, 0.015f, 0.025f, 1.0f},
                        .colorLoad = ir::AttachmentLoadOperation::Clear},
                    .vertexStreams = {
                        {0, ir::ResourceView{
                            positions, 0, sizeof(positionData), sizeof(Position)}},
                        {1, ir::ResourceView{
                            colors, 0, sizeof(colorData), sizeof(Color)}}},
                    .indexBinding = ir::IndexBinding{
                        .resource = ir::ResourceView{
                            indices, 0, sizeof(indexData), sizeof(std::uint16_t)},
                        .format = ir::IndexFormat::Uint16},
                    .indexCount = 3,
                    .instanceCount = 2}},
            {.id = gpu::WorkId{1}, .name = "PresentAdvancedPackage",
                .accesses = {{presentation, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{presentation}}
        };
        return module;
    }

    sge::ir::SemanticModule BuildCopyQueuePackageModule()
    {
        using namespace sge;
        constexpr gpu::ResourceId sourcePositions{0};
        constexpr gpu::ResourceId copiedPositions{1};
        constexpr gpu::ResourceId instanceColor{2};
        constexpr gpu::ResourceId presentation{3};

        struct Position { float x, y, z; };
        struct Color { float r, g, b, a; };
        const std::array<Position, 3> positions{{
            {-0.55f, -0.45f, 0.0f},
            { 0.55f, -0.45f, 0.0f},
            { 0.00f,  0.55f, 0.0f}}};
        const std::array<Color, 1> colors{{
            {0.2f, 0.9f, 0.35f, 1.0f}}};

        ir::SemanticModule module;
        module.resources = {
            {.id = sourcePositions, .name = "CopyQueueSourcePositions",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(positions),
                    .strideBytes = sizeof(Position),
                    .usage = ir::BufferUsage::CopySource},
                .data = ArrayBytes(positions)},
            {.id = copiedPositions, .name = "CopyQueueVertexDestination",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(positions),
                    .strideBytes = sizeof(Position),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::CopyDestination}},
            {.id = instanceColor, .name = "CopyQueueInstanceColor",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = sizeof(colors),
                    .strideBytes = sizeof(Color),
                    .usage = ir::BufferUsage::Vertex},
                .data = ArrayBytes(colors)},
            {.id = presentation, .name = "CopyQueuePresentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "CopyQueueRasterConsumer",
            .shaderPath = "Shaders/IntegrationGeneralRaster.hlsl",
            .vertexEntry = "VSMain",
            .pixelEntry = "PSMain",
            .programInterface = {
                .parameters = {},
                .vertexInputs = {
                    {"POSITION", 0, ir::VertexElementFormat::Float3,
                        0, 0, ir::VertexInputRate::PerVertex, 0},
                    {"COLOR", 0, ir::VertexElementFormat::Float4,
                        1, 0, ir::VertexInputRate::PerInstance, 1}},
                .colorOutputCount = 1,
                .depthAttachmentAllowed = false}});
        module.works = {
            {.id = gpu::WorkId{0}, .name = "CopyStaticPositionsOnCopyQueue",
                .accesses = {
                    {sourcePositions, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {copiedPositions, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{
                    .source = sourcePositions,
                    .destination = copiedPositions,
                    .sizeBytes = sizeof(positions)}},
            {.id = gpu::WorkId{1}, .name = "ConsumeCopiedPositionsOnDirectQueue",
                .accesses = {
                    {copiedPositions, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {instanceColor, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {presentation, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = gpu::ProgramId{0},
                    .vertexResource = copiedPositions,
                    .attachments = {{presentation}, {}},
                    .vertexCount = 3,
                    .rasterState = {.depth = gpu::DepthMode::Disabled},
                    .clear = {
                        .clearColor = true,
                        .color = {0.015f, 0.02f, 0.03f, 1.0f},
                        .colorLoad = ir::AttachmentLoadOperation::Clear},
                    .vertexStreams = {
                        {0, ir::ResourceView{
                            copiedPositions, 0, sizeof(positions),
                            sizeof(Position)}},
                        {1, ir::ResourceView{
                            instanceColor, 0, sizeof(colors), sizeof(Color)}}},
                    .instanceCount = 1}},
            {.id = gpu::WorkId{2}, .name = "PresentCopyQueuePackage",
                .accesses = {{presentation, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{presentation}}
        };
        return module;
    }

    sge::ir::SemanticModule BuildIntegrationModule()
    {
        using namespace sge;

        constexpr gpu::ResourceId fullscreen{0};
        constexpr gpu::ResourceId history{1};
        constexpr gpu::ResourceId aliasA{2};
        constexpr gpu::ResourceId aliasB{3};
        constexpr gpu::ResourceId presentation{4};

        const std::vector<classical::Vertex> fullscreenVertices = {
            {{-1.0f, -1.0f, 0.0f}, {1, 1, 1, 1}},
            {{-1.0f,  3.0f, 0.0f}, {1, 1, 1, 1}},
            {{ 3.0f, -1.0f, 0.0f}, {1, 1, 1, 1}}
        };

        ir::SemanticModule module;
        module.resources = {
            {.id = fullscreen, .name = "IntegrationFullscreen",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = fullscreenVertices.size()
                        * sizeof(classical::Vertex),
                    .strideBytes = sizeof(classical::Vertex),
                    .usage = ir::BufferUsage::Vertex
                        | ir::BufferUsage::Storage
                        | ir::BufferUsage::CopyDestination},
                .data = classical::ToBytes(fullscreenVertices)},
            {.id = history, .name = "CrossQueueTemporalHistory",
                .lifetime = gpu::ResourceLifetimeClass::Temporal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = 16,
                    .usage = ir::BufferUsage::Storage},
                .data = std::vector<std::byte>(16)},
            {.id = aliasA, .name = "AliasA",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = 84,
                    .usage = ir::BufferUsage::Storage}},
            {.id = aliasB, .name = "AliasB",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{
                    .sizeBytes = 84,
                    .usage = ir::BufferUsage::Storage}},
            {.id = presentation, .name = "IntegrationPresentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };

        module.programs = {
            {.id = gpu::ProgramId{0}, .name = "TemporalWriter",
                .shaderPath = "Shaders/IntegrationTemporalWrite.hlsl",
                .computeEntry = "CSMain",
                .parameters = {{.name = "CurrentHistory",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute}}},
            {.id = gpu::ProgramId{1}, .name = "AliasWriter",
                .shaderPath = "Shaders/GenerateBuffer.hlsl",
                .computeEntry = "CSMain",
                .parameters = {{.name = "OutputBuffer",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute}}},
            {.id = gpu::ProgramId{2}, .name = "TemporalReader",
                .shaderPath = "Shaders/IntegrationTemporalRead.hlsl",
                .vertexEntry = "VSMain",
                .pixelEntry = "PSMain",
                .programInterface = {
                    .vertexInputs = {{
                        "POSITION", 0, ir::VertexElementFormat::Float3,
                        0, 0, ir::VertexInputRate::PerVertex, 0}},
                    .colorOutputCount = 1,
                    .depthAttachmentAllowed = false},
                .parameters = {{.name = "PreviousHistory",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Pixel}}},
            {.id = gpu::ProgramId{3}, .name = "PersistentReader",
                .shaderPath = "Shaders/IntegrationPersistentRead.hlsl",
                .computeEntry = "CSMain",
                .programInterface = {.parameters = {
                    {.name = "SharedVertices",
                        .kind = gpu::ProgramParameterKind::ShaderResource,
                        .stage = gpu::ProgramStage::Compute},
                    {.name = "OutputBuffer",
                        .kind = gpu::ProgramParameterKind::UnorderedAccess,
                        .stage = gpu::ProgramStage::Compute}}}}
        };

        module.works = {
            {.id = gpu::WorkId{0}, .name = "WriteCurrentTemporalInstance",
                .accesses = {{history, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{0},
                    .bindings = {{0, history}}}},
            {.id = gpu::WorkId{1},
                .name = "ReadPersistentOnComputeAndUseAliasA",
                .accesses = {
                    {fullscreen, gpu::AccessMode::Read,
                        gpu::ResourceRole::ProgramInput},
                    {aliasA, gpu::AccessMode::Write,
                        gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{3},
                    .bindings = {
                        {0, ir::ResourceView{
                            fullscreen,
                            sizeof(classical::Vertex),
                            sizeof(classical::Vertex),
                            sizeof(classical::Vertex)}},
                        {1, aliasA}}}},
            {.id = gpu::WorkId{2}, .name = "UseAliasB",
                .accesses = {{aliasB, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{1},
                    .bindings = {{0, aliasB}}}},
            {.id = gpu::WorkId{3}, .name = "ReadPreviousOnDirectQueue",
                .accesses = {
                    {fullscreen, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {history, gpu::AccessMode::Read,
                        gpu::ResourceRole::ProgramInput, 1},
                    {presentation, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = gpu::ProgramId{2},
                    .vertexResource = fullscreen,
                    .bindings = {{0, history, 1}},
                    .attachments = {{presentation}, {}},
                    .vertexCount = 3,
                    .rasterState = {.depth = gpu::DepthMode::Disabled},
                    .clear = {.clearColor = true}}},
            {.id = gpu::WorkId{4}, .name = "PresentIntegrationFrame",
                .accesses = {{presentation, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{presentation}}
        };
        return module;
    }
}

void RunD3D12IntegrationTests()
{
    using namespace sge;

    RunV1AcceptanceTests();
    RunResourceStateValidatorTests();

    platform::Win32Application application{
        GetModuleHandleW(nullptr),
        SW_HIDE,
        {.title = L"Semantic GPU Engine WARP Integration Test",
            .width = 64,
            .height = 64}
    };

    auto backend = d3d12::CreateBackend(
        application.Surface(), {.forceWarp = true});
    const auto copyQueueModule = BuildCopyQueuePackageModule();
    const auto copyQueuePackage = compiler::RenderPackageCompiler{}.Compile(
        copyQueueModule, backend->Capabilities());
    const auto copyHandoff = std::find_if(
        copyQueuePackage.report.queueHandoffs.begin(),
        copyQueuePackage.report.queueHandoffs.end(),
        [](const compiler::QueueHandoffPlan& handoff)
        {
            return handoff.releaseQueue == gpu::QueueClass::Copy
                && handoff.acquireQueue == gpu::QueueClass::Direct
                && handoff.releaseState
                    == gpu::AbstractState::TransferWrite
                && handoff.acquireState
                    == gpu::AbstractState::VertexRead;
        });
    const auto cyclicCopyReuse = std::find_if(
        copyQueuePackage.report.cyclicFrameHandoffs.begin(),
        copyQueuePackage.report.cyclicFrameHandoffs.end(),
        [](const compiler::CyclicFrameHandoffPlan& handoff)
        {
            return handoff.resource == gpu::ResourceId{1}
                && handoff.releaseQueue == gpu::QueueClass::Direct
                && handoff.acquireQueue == gpu::QueueClass::Copy
                && handoff.requiresCommonRelease;
        });
    const bool operationStreamHasQueueWait = std::any_of(
        copyQueuePackage.package.operations.begin(),
        copyQueuePackage.package.operations.end(),
        [](const compiler::CompiledOperation& operation)
        {
            return std::holds_alternative<
                compiler::WaitForWorkOperation>(operation);
        });
    const bool operationStreamHasCyclicReuse = std::any_of(
        copyQueuePackage.package.operations.begin(),
        copyQueuePackage.package.operations.end(),
        [](const compiler::CompiledOperation& operation)
        {
            const auto* common = std::get_if<
                compiler::RequireCommonOperation>(&operation);
            return common != nullptr && common->cyclicReuse;
        });
    if (!copyQueuePackage.Succeeded()
        || copyHandoff == copyQueuePackage.report.queueHandoffs.end()
        || !copyHandoff->crossesCopyQueue
        || copyHandoff->releaseView.resource != gpu::ResourceId{1}
        || copyHandoff->acquireView.resource != gpu::ResourceId{1}
        || cyclicCopyReuse
            == copyQueuePackage.report.cyclicFrameHandoffs.end()
        || !operationStreamHasQueueWait
        || !operationStreamHasCyclicReuse)
    {
        throw std::runtime_error(
            "Copy queue package did not compile exact intra-frame and cyclic handoffs.");
    }

    const auto advancedModule = BuildAdvancedPackageModule();
    const auto advancedPackage = compiler::RenderPackageCompiler{}.Compile(
        advancedModule, backend->Capabilities());
    if (!advancedPackage.Succeeded()
        || !advancedPackage.package.requirements.multipleVertexStreams
        || !advancedPackage.package.requirements.indexedDraw
        || !advancedPackage.package.requirements.instancedDraw
        || advancedPackage.package.executables.empty()
        || advancedPackage.package.operations.empty())
    {
        throw std::runtime_error(
            "Advanced package integration module was not compiled natively.");
    }
    const auto module = BuildIntegrationModule();
    const auto compiled = compiler::RenderCompiler{}.Compile(
        module, backend->Capabilities());
    const auto allocationOf = [&](gpu::ResourceId resource)
    {
        return std::find_if(
            compiled.plan.lifetimes.begin(),
            compiled.plan.lifetimes.end(),
            [&](const compiler::ResourceLifetime& lifetime)
            {
                return lifetime.resource == resource;
            })->allocation;
    };
    const auto persistent = std::find_if(
        compiled.plan.persistentReadStates.begin(),
        compiled.plan.persistentReadStates.end(),
        [](const compiler::PersistentReadStatePlan& envelope)
        {
            return envelope.resource == gpu::ResourceId{0};
        });
    const bool persistentEnvelopeValid =
        persistent != compiled.plan.persistentReadStates.end()
        && std::find(
            persistent->states.begin(), persistent->states.end(),
            gpu::AbstractState::VertexRead) != persistent->states.end()
        && std::find(
            persistent->states.begin(), persistent->states.end(),
            gpu::AbstractState::ProgramRead) != persistent->states.end();
    if (allocationOf(gpu::ResourceId{2})
            != allocationOf(gpu::ResourceId{3})
        || compiled.plan.temporalDependencies.size() != 1
        || compiled.plan.temporalDependencies[0].producerQueue
            != gpu::QueueClass::Compute
        || compiled.plan.temporalDependencies[0].consumerQueue
            != gpu::QueueClass::Direct
        || !persistentEnvelopeValid)
    {
        throw std::runtime_error(
            "WARP integration module does not exercise the required boundaries.");
    }
    runtime::RenderRuntime runtime{
        std::move(backend),
        {.enableValidation = true,
            .enablePlanCache = true,
            .planDiagnosticsPath = {},
            .graphDiagnosticsPath = {}}
    };

    std::uint32_t copyQueueFrameCount = 0;
    std::uint32_t advancedFrameCount = 0;
    std::uint32_t frameCount = 0;
    const int result = application.Run(
        [&](const platform::FrameTime&)
        {
            if (copyQueueFrameCount < 5)
            {
                runtime.Execute(copyQueueModule);
                ++copyQueueFrameCount;
                return;
            }
            if (advancedFrameCount < 2)
            {
                runtime.Execute(advancedModule);
                ++advancedFrameCount;
                return;
            }
            runtime.Execute(module);
            ++frameCount;
            if (frameCount == 12)
            {
                PostMessageW(
                    static_cast<HWND>(application.Surface().handle),
                    WM_CLOSE,
                    0,
                    0);
            }
        });

    if (result != 0 || copyQueueFrameCount != 5
        || advancedFrameCount != 2 || frameCount != 12)
    {
        throw std::runtime_error(
            "D3D12 WARP integration test did not complete the five-frame Copy-slot reuse, native-package, and 12-frame paths.");
    }
    runtime.WaitIdle();
}
