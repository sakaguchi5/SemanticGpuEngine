#include "11_Tests/D3D12IntegrationTests.h"

#include "04_RenderCompiler/RenderCompiler.h"
#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"
#include "07_D3D12Backend/D3D12Backend.h"
#include "08_ClassicalRasterFrontend/ClassicalRaster.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <windows.h>

namespace
{
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
                .parameters = {{.name = "PreviousHistory",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Pixel}}},
            {.id = gpu::ProgramId{3}, .name = "PersistentReader",
                .shaderPath = "Shaders/IntegrationPersistentRead.hlsl",
                .computeEntry = "CSMain",
                .parameters = {
                    {.name = "SharedVertices",
                        .kind = gpu::ProgramParameterKind::ShaderResource,
                        .stage = gpu::ProgramStage::Compute},
                    {.name = "OutputBuffer",
                        .kind = gpu::ProgramParameterKind::UnorderedAccess,
                        .stage = gpu::ProgramStage::Compute}}}
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
                    .bindings = {{0, fullscreen}, {1, aliasA}}}},
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

    platform::Win32Application application{
        GetModuleHandleW(nullptr),
        SW_HIDE,
        {.title = L"Semantic GPU Engine WARP Integration Test",
            .width = 64,
            .height = 64}
    };

    auto backend = d3d12::CreateBackend(
        application.Surface(), {.forceWarp = true});
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

    std::uint32_t frameCount = 0;
    const int result = application.Run(
        [&](const platform::FrameTime&)
        {
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

    if (result != 0 || frameCount != 12)
    {
        throw std::runtime_error(
            "D3D12 WARP integration test did not complete 12 frames.");
    }
    runtime.WaitIdle();
}
