#include "11_Tests/V1AcceptanceTests.h"

#include "04_RenderCompiler/CompiledRenderPackage.h"
#include "12_CubeLab/ExperimentHarness.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace sge;

    gpu::DeviceCapabilities FullCapabilities()
    {
        gpu::DeviceCapabilities result;
        result.rasterExecution = true;
        result.computeExecution = true;
        result.copyExecution = true;
        result.concurrentCompute = true;
        result.dedicatedCopyQueue = true;
        result.resourceAliasing = true;
        result.textureSubresourceViews = true;
        result.multipleVertexStreams = true;
        result.indexedDraw = true;
        result.instancedDraw = true;
        result.explicitCopyQueueHandoffs = true;
        result.dynamicDescriptorGrowth = true;
        result.advancedRasterState = true;
        result.customViewportScissor = true;
        result.expandedResourceFormats = true;
        result.maximumVertexStreams = 8;
        result.maximumColorAttachments = 8;
        return result;
    }

    void Require(bool condition, const char* message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    ir::ResourceDeclaration StorageBuffer(
        gpu::ResourceId id,
        std::uint64_t size,
        ir::BufferUsage extra = ir::BufferUsage::None)
    {
        return {
            .id = id,
            .name = "Buffer" + std::to_string(id.Value()),
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = size,
                .strideBytes = 4,
                .usage = ir::BufferUsage::Storage
                    | ir::BufferUsage::CopySource
                    | ir::BufferUsage::CopyDestination
                    | extra}
        };
    }

    ir::SemanticModule TwoViewTextureModule(bool overlap)
    {
        constexpr gpu::ResourceId texture{0};
        ir::SemanticModule module;
        module.resources.push_back({
            .id = texture,
            .name = "MipChain",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::TextureDescription{
                .dimension = gpu::ResourceKind::Texture2D,
                .format = gpu::ResourceFormat::Rgba8Unorm,
                .width = 64,
                .height = 64,
                .depth = 1,
                .mipLevels = 2,
                .arrayLayers = 1,
                .sampleCount = 1,
                .usage = ir::TextureUsage::Sampled
                    | ir::TextureUsage::Storage}}
        );
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "MipReadWrite",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {
                {.name = "InputMip",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0},
                {.name = "OutputMip",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0}}}
        });
        const ir::TextureSubresourceRange mip0{
            .baseMip = 0, .mipCount = 1,
            .baseArrayLayer = 0, .arrayLayerCount = 1,
            .basePlane = 0, .planeCount = 1};
        const ir::TextureSubresourceRange mip1{
            .baseMip = static_cast<std::uint16_t>(overlap ? 0 : 1),
            .mipCount = 1,
            .baseArrayLayer = 0, .arrayLayerCount = 1,
            .basePlane = 0, .planeCount = 1};
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "ReadAndWriteMips",
            .accesses = {
                {texture, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput},
                {texture, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {
                    {0, ir::ResourceView{texture, mip0}},
                    {1, ir::ResourceView{texture, mip1}}}}
        });
        return module;
    }

    ir::SemanticModule TwoViewBufferModule()
    {
        constexpr gpu::ResourceId buffer{0};
        ir::SemanticModule module;
        module.resources.push_back(StorageBuffer(buffer, 64));
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "BufferReadWrite",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {
                {.name = "InputRange",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0},
                {.name = "OutputRange",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0}}}
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "ReadAndWriteBufferRanges",
            .accesses = {
                {buffer, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput},
                {buffer, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {
                    {0, ir::ResourceView{buffer, 0, 32, 4}},
                    {1, ir::ResourceView{buffer, 32, 32, 4}}}}
        });
        return module;
    }

    ir::SemanticModule ReinterpretedTextureModule(
        gpu::ResourceFormat viewFormat)
    {
        constexpr gpu::ResourceId texture{0};
        ir::SemanticModule module;
        module.resources.push_back({
            .id = texture,
            .name = "TypedTexture",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::TextureDescription{
                .dimension = gpu::ResourceKind::Texture2D,
                .format = gpu::ResourceFormat::R32Float,
                .width = 16,
                .height = 16,
                .depth = 1,
                .mipLevels = 1,
                .arrayLayers = 1,
                .sampleCount = 1,
                .usage = ir::TextureUsage::Sampled}}
        );
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "ReadReinterpretedTexture",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {{
                .name = "InputTexture",
                .kind = gpu::ProgramParameterKind::ShaderResource,
                .stage = gpu::ProgramStage::Compute}}}
        });
        ir::ResourceView view{texture};
        view.formatOverride = viewFormat;
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "ReadTypedView",
            .accesses = {{texture, gpu::AccessMode::Read,
                gpu::ResourceRole::ProgramInput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, view}}}}
        );
        return module;
    }

    ir::SemanticModule Texture3DDepthSliceModule()
    {
        constexpr gpu::ResourceId volume{0};
        ir::SemanticModule module;
        module.resources.push_back({
            .id = volume,
            .name = "Volume",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::TextureDescription{
                .dimension = gpu::ResourceKind::Texture3D,
                .format = gpu::ResourceFormat::Rgba8Unorm,
                .width = 16,
                .height = 16,
                .depth = 8,
                .mipLevels = 1,
                .arrayLayers = 1,
                .sampleCount = 1,
                .usage = ir::TextureUsage::Storage}}
        );
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "WriteVolumeSlices",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {{
                .name = "OutputVolume",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute}}}
        });
        const ir::TextureSubresourceRange slices{
            .baseMip = 0,
            .mipCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1,
            .basePlane = 0,
            .planeCount = 1,
            .baseDepthSlice = 2,
            .depthSliceCount = 3};
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "WriteSelectedDepthSlices",
            .accesses = {{volume, gpu::AccessMode::Write,
                gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, ir::ResourceView{volume, slices}}}}
        });
        return module;
    }

    ir::SemanticModule GeneralRasterModule()
    {
        constexpr gpu::ResourceId positions{0};
        constexpr gpu::ResourceId attributes{1};
        constexpr gpu::ResourceId indices{2};
        constexpr gpu::ResourceId presentation{3};
        ir::SemanticModule module;
        module.resources = {
            StorageBuffer(positions, 48, ir::BufferUsage::Vertex),
            StorageBuffer(attributes, 64, ir::BufferUsage::Vertex),
            StorageBuffer(indices, 12, ir::BufferUsage::Index),
            {.id = presentation,
                .name = "Presentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "GeneralRaster",
            .vertexEntry = "VSMain",
            .pixelEntry = "PSMain",
            .programInterface = {
                .vertexInputs = {
                    {"POSITION", 0, ir::VertexElementFormat::Float3,
                        0, 0, ir::VertexInputRate::PerVertex, 0},
                    {"COLOR", 0, ir::VertexElementFormat::Float4,
                        1, 0, ir::VertexInputRate::PerInstance, 1}},
                .colorOutputCount = 1,
                .depthAttachmentAllowed = false}
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "IndexedInstancedDraw",
            .accesses = {
                {positions, gpu::AccessMode::Read,
                    gpu::ResourceRole::VertexInput},
                {attributes, gpu::AccessMode::Read,
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
                .rasterState = {.depth = gpu::DepthMode::Disabled},
                .vertexStreams = {
                    {0, ir::ResourceView{positions, 0, 48, 12}},
                    {1, ir::ResourceView{attributes, 0, 64, 16}}},
                .indexBinding = ir::IndexBinding{
                    .resource = ir::ResourceView{indices, 0, 12, 4},
                    .format = ir::IndexFormat::Uint32},
                .indexCount = 3,
                .instanceCount = 4,
                .firstInstance = 2}
        });
        return module;
    }

    ir::SemanticModule CopyHandoffModule()
    {
        constexpr gpu::ResourceId source{0};
        constexpr gpu::ResourceId produced{1};
        ir::SemanticModule module;
        module.resources = {
            {.id = source,
                .name = "UploadSeed",
                .lifetime = gpu::ResourceLifetimeClass::Persistent,
                .update = gpu::ResourceUpdateClass::Immutable,
                .description = ir::BufferDescription{
                    .sizeBytes = 64,
                    .strideBytes = 4,
                    .usage = ir::BufferUsage::CopySource},
                .data = std::vector<std::byte>(64)},
            StorageBuffer(produced, 64)
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "ReadProduced",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {{
                .name = "Input",
                .kind = gpu::ProgramParameterKind::ShaderResource,
                .stage = gpu::ProgramStage::Compute}}}
        });
        module.works = {
            {.id = gpu::WorkId{0},
                .name = "CopySeed",
                .accesses = {
                    {source, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {produced, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{
                    .source = source,
                    .destination = produced,
                    .sizeBytes = 64}},
            {.id = gpu::WorkId{1},
                .name = "ConsumeCopiedData",
                .accesses = {{produced, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{0},
                    .bindings = {{0, produced}}}}
        };
        return module;
    }

    ir::SemanticModule SimpleComputeModule(std::size_t workCount)
    {
        constexpr gpu::ResourceId data{0};
        ir::SemanticModule module;
        module.resources.push_back(StorageBuffer(data, 256));
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "Writer",
            .computeEntry = "CSMain",
            .programInterface = {.parameters = {{
                .name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute}}}
        });
        for (std::size_t index = 0; index < workCount; ++index)
        {
            module.works.push_back({
                .id = gpu::WorkId{static_cast<std::uint32_t>(index)},
                .name = "Write" + std::to_string(index),
                .accesses = {{data, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{0},
                    .bindings = {{0, data}}}}
            );
        }
        return module;
    }

    void TestPureStructureHashAndCanonicalization()
    {
        ir::SemanticModule invalid;
        invalid.works.push_back({.name = "InvalidButHashable"});
        (void)invalid.StructureHash();

        auto module = SimpleComputeModule(1);
        module.programs[0].parameters =
            module.programs[0].programInterface.parameters;
        module.programs[0].programInterface.parameters.clear();
        auto capabilities = FullCapabilities();
        const auto result = compiler::RenderPackageCompiler{}.Compile(
            module, capabilities);
        Require(result.Succeeded(), "Legacy ProgramInterface canonicalization failed.");
        Require(result.package.canonicalModule.programs[0].parameters.empty(),
            "Canonical package retained legacy program parameters.");
        Require(!result.package.canonicalModule.programs[0]
                .programInterface.parameters.empty(),
            "Canonical package lost program parameters.");
    }

    void TestTextureRangeStateModel()
    {
        auto capabilities = FullCapabilities();
        const auto separated = compiler::RenderPackageCompiler{}.Compile(
            TwoViewTextureModule(false), capabilities);
        Require(separated.Succeeded(),
            "Non-overlapping texture mip read/write was rejected.");
        Require(separated.package.requirements.textureSubresourceViews,
            "Texture subresource requirement was not recorded.");
        Require(!separated.package.legacyExecutable,
            "Texture subresource package was marked legacy executable.");

        const auto overlapping = compiler::RenderPackageCompiler{}.Compile(
            TwoViewTextureModule(true), capabilities);
        Require(!overlapping.Succeeded(),
            "Overlapping texture mip read/write was accepted.");

        const auto bufferRanges = compiler::RenderPackageCompiler{}.Compile(
            TwoViewBufferModule(), capabilities);
        Require(!bufferRanges.Succeeded(),
            "Disjoint byte views incorrectly escaped whole-buffer state validation.");

        const auto compatibleFormat =
            compiler::RenderPackageCompiler{}.Compile(
                ReinterpretedTextureModule(gpu::ResourceFormat::R32Uint),
                capabilities);
        Require(compatibleFormat.Succeeded(),
            "Typeless-compatible texture view reinterpretation failed.");

        const auto incompatibleFormat =
            compiler::RenderPackageCompiler{}.Compile(
                ReinterpretedTextureModule(gpu::ResourceFormat::Rgba8Unorm),
                capabilities);
        Require(!incompatibleFormat.Succeeded(),
            "Incompatible same-size texture view reinterpretation was accepted.");

        const auto depthSlices = compiler::RenderPackageCompiler{}.Compile(
            Texture3DDepthSliceModule(), capabilities);
        Require(depthSlices.Succeeded(),
            "Texture3D depth-slice view compilation failed.");
        const auto& compute = std::get<compiler::CompiledComputeCommand>(
            depthSlices.package.works.front().command);
        Require(compute.bindings.front().view.textureRange.firstDepthSlice == 2
                && compute.bindings.front().view.textureRange.depthSliceCount == 3,
            "Texture3D depth-slice range was not preserved in the package.");
    }

    void TestGeneralRasterContract()
    {
        const auto result = compiler::RenderPackageCompiler{}.Compile(
            GeneralRasterModule(), FullCapabilities());
        Require(result.Succeeded(), "Generalized raster package failed.");
        Require(result.package.requirements.multipleVertexStreams,
            "Multiple vertex streams were not recorded.");
        Require(result.package.requirements.indexedDraw,
            "Indexed draw was not recorded.");
        Require(result.package.requirements.instancedDraw,
            "Instanced draw was not recorded.");
        Require(!result.package.legacyExecutable,
            "Generalized raster package was marked legacy executable.");

        auto limited = FullCapabilities();
        limited.multipleVertexStreams = false;
        const auto rejected = compiler::RenderPackageCompiler{}.Compile(
            GeneralRasterModule(), limited);
        Require(!rejected.Succeeded(),
            "Backend capability mismatch was not rejected.");
    }

    void TestCopyHandoffAndBudget()
    {
        auto capabilities = FullCapabilities();
        const auto handoff = compiler::RenderPackageCompiler{}.Compile(
            CopyHandoffModule(), capabilities);
        Require(handoff.Succeeded(), "Copy handoff package failed.");
        Require(!handoff.package.queueHandoffs.empty(),
            "Copy handoff was not materialized in the package.");
        Require(handoff.package.requirements.explicitCopyQueueHandoffs,
            "Copy handoff capability requirement was not recorded.");

        auto module = SimpleComputeModule(1);
        capabilities.localMemoryBudget = 128;
        const auto budget = compiler::RenderPackageCompiler{}.Compile(
            module, capabilities);
        Require(!budget.Succeeded(), "Memory budget overflow was accepted.");
    }

    void TestDeterminismAndExperimentHarness()
    {
        auto capabilities = FullCapabilities();
        compiler::RenderPackageCompiler packageCompiler;
        for (std::size_t count = 1; count <= 32; ++count)
        {
            const auto module = SimpleComputeModule(count);
            const auto first = packageCompiler.Compile(module, capabilities);
            const auto second = packageCompiler.Compile(module, capabilities);
            Require(first.Succeeded() && second.Succeeded(),
                "Deterministic graph package compilation failed.");
            Require(first.package.packageHash == second.package.packageHash,
                "Package hash is not deterministic.");
            Require(first.package.plan.dependencies.size()
                    == second.package.plan.dependencies.size(),
                "Dependency planning size is not deterministic.");
            for (std::size_t edge = 0;
                 edge < first.package.plan.dependencies.size(); ++edge)
            {
                const auto& left = first.package.plan.dependencies[edge];
                const auto& right = second.package.plan.dependencies[edge];
                Require(left.before == right.before
                        && left.after == right.after
                        && left.resource == right.resource,
                    "Dependency planning is not deterministic.");
            }
        }

        cube_lab::ExperimentScene scene;
        scene.elapsedSeconds = 1.0;
        const auto report = cube_lab::ExperimentHarness{}.Run(
            scene, capabilities);
        Require(report.classical.succeeded && report.sdf.succeeded,
            "Common-scene Classical/SDF experiment failed.");
        Require(report.classical.package.workCount != 0
                && report.sdf.package.workCount != 0,
            "Experiment report did not capture package metrics.");
    }
}

void RunV1AcceptanceTests()
{
    TestPureStructureHashAndCanonicalization();
    TestTextureRangeStateModel();
    TestGeneralRasterContract();
    TestCopyHandoffAndBudget();
    TestDeterminismAndExperimentHarness();
}
