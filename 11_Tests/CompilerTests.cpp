#include "02_GpuSemantics/Semantics.h"
#include "03_RenderIR/RenderIR.h"
#include "04_RenderCompiler/RenderCompiler.h"
#include "09_ExperimentalGeometry/ExperimentalGeometry.h"
#include "09_ExperimentalGeometry/SdfLowering.h"
#include "12_CubeLab/CubeExperiment.h"
#include "11_Tests/D3D12IntegrationTests.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    void TestDependencyAndTransitions()
    {
        using namespace sge;

        constexpr gpu::ResourceId data{0};

        ir::SemanticModule module;
        module.resources.push_back({
            .id = data,
            .name = "Data",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 256,
                .usage = ir::BufferUsage::Storage
            },
            .data = {}
        });
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "TestCompute",
            .shaderPath = "unused.hlsl",
            .computeEntry = "CSMain",
            .parameters = {{
                .name = "Data",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute
            }}
        });

        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "Writer",
            .accesses = {{data, gpu::AccessMode::Write,
                gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, data}},
                .groupCountX = 1
            }
        });

        module.works.push_back({
            .id = gpu::WorkId{1},
            .name = "WriterAgain",
            .accesses = {{data, gpu::AccessMode::Write,
                gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, data}},
                .groupCountX = 1
            }
        });

        compiler::RenderCompiler compiler;
        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        const auto result = compiler.Compile(module, capabilities);

        assert(result.plan.dependencies.size() == 1);
        assert(result.plan.dependencies[0].before == 0);
        assert(result.plan.dependencies[0].after == 1);
        assert(result.plan.scheduledWorks.size() == 2);
        assert(result.plan.transitions.size() == 1);
    }

    void TestUnsupportedWorkFails()
    {
        using namespace sge;
        ir::SemanticModule module;
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "UnsupportedCompute",
            .computeEntry = "CSMain"
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "MustFail",
            .payload = ir::ComputeWork{.program = gpu::ProgramId{0}}
        });

        bool failed = false;
        try
        {
            (void)compiler::RenderCompiler{}.Compile(
                module, gpu::DeviceCapabilities{});
        }
        catch (const std::runtime_error&)
        {
            failed = true;
        }
        assert(failed);
    }

    void TestPayloadAccessMismatchFails()
    {
        using namespace sge;
        constexpr gpu::ResourceId data{0};

        ir::SemanticModule module;
        module.resources.push_back({
            .id = data,
            .name = "Data",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 64,
                .usage = ir::BufferUsage::Storage
            }
        });
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "Compute",
            .computeEntry = "CSMain",
            .parameters = {{
                .name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute
            }}
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "Mismatched",
            .accesses = {{data, gpu::AccessMode::Read,
                gpu::ResourceRole::ProgramInput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, data}}
            }
        });

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        bool failed = false;
        try
        {
            (void)compiler::RenderCompiler{}.Compile(module, capabilities);
        }
        catch (const std::runtime_error&)
        {
            failed = true;
        }
        assert(failed);
    }

    void TestComputeCopyRasterAndQueues()
    {
        using namespace sge;
        constexpr gpu::ResourceId generated{0};
        constexpr gpu::ResourceId copied{1};
        constexpr gpu::ResourceId color{2};

        ir::SemanticModule module;
        module.resources = {
            {.id = generated, .name = "Generated",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{84, 0,
                    ir::BufferUsage::Storage | ir::BufferUsage::CopySource}},
            {.id = copied, .name = "Copied",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::BufferDescription{84, 28,
                    ir::BufferUsage::Vertex | ir::BufferUsage::CopyDestination}},
            {.id = color, .name = "Presentation",
                .lifetime = gpu::ResourceLifetimeClass::External,
                .update = gpu::ResourceUpdateClass::Imported,
                .description = ir::PresentationDescription{}}
        };
        module.programs = {
            {.id = gpu::ProgramId{0}, .name = "Generate",
                .computeEntry = "CSMain",
                .parameters = {{.name = "Output",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute}}},
            {.id = gpu::ProgramId{1}, .name = "Draw",
                .vertexEntry = "VSMain", .pixelEntry = "PSMain"}
        };
        module.works = {
            {
                .id = gpu::WorkId{0}, .name = "Generate",
                .accesses = {{generated, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{
                    .program = gpu::ProgramId{0},
                    .bindings = {{0, generated}}}
            },
            {
                .id = gpu::WorkId{1}, .name = "Copy",
                .accesses = {
                    {generated, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {copied, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{generated, copied, 0, 0, 84}
            },
            {
                .id = gpu::WorkId{2}, .name = "Draw",
                .accesses = {
                    {copied, gpu::AccessMode::Read,
                        gpu::ResourceRole::VertexInput},
                    {color, gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput}},
                .payload = ir::RasterWork{
                    .program = gpu::ProgramId{1},
                    .vertexResource = copied,
                    .attachments = {{color}, {}},
                    .vertexCount = 3,
                    .rasterState = {.depth = gpu::DepthMode::Disabled}}
            },
            {
                .id = gpu::WorkId{3}, .name = "Present",
                .accesses = {{color, gpu::AccessMode::Read,
                    gpu::ResourceRole::Presentation}},
                .payload = ir::PresentWork{color}
            }
        };

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.copyExecution = true;
        capabilities.concurrentCompute = true;
        capabilities.dedicatedCopyQueue = true;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);
        assert(result.plan.scheduledWorks[0].queue == gpu::QueueClass::Compute);
        assert(result.plan.scheduledWorks[1].queue == gpu::QueueClass::Copy);
        assert(result.plan.scheduledWorks[2].queue == gpu::QueueClass::Direct);
        assert(result.plan.queueSynchronizations.size() == 2);
        assert(result.plan.dependencies.size() == 3);

        assert(result.plan.frameBoundaryTransitions.empty());
        const auto copiedInstances = std::find_if(
            result.plan.resourceInstances.begin(),
            result.plan.resourceInstances.end(),
            [&](const compiler::ResourceInstancePlan& instance)
            {
                return instance.resource == copied;
            });
        assert(copiedInstances != result.plan.resourceInstances.end());
        assert(copiedInstances->lifetime
            == gpu::ResourceLifetimeClass::FrameLocal);
        assert(copiedInstances->physicalInstanceCount == 3);
        assert(copiedInstances->selector
            == gpu::InstanceSelectorKind::CurrentFrameSlot);
    }

    void TestSingleQueuePlanNeedsNoFrameBoundaryRelease()
    {
        using namespace sge;
        constexpr gpu::ResourceId data{0};

        ir::SemanticModule module;
        module.resources.push_back({
            .id = data,
            .name = "SingleQueueData",
            .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 64,
                .usage = ir::BufferUsage::Storage
            }
        });
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "SingleQueueCompute",
            .computeEntry = "CSMain",
            .parameters = {{
                .name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute
            }}
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "Write",
            .accesses = {{data, gpu::AccessMode::Write,
                gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, data}}
            }
        });

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.concurrentCompute = false;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);
        assert(result.plan.frameBoundaryTransitions.empty());
    }

    void TestTransientTextureAliasingPlan()
    {
        using namespace sge;
        ir::TextureDescription texture{
            .dimension = gpu::ResourceKind::Texture2D,
            .format = gpu::ResourceFormat::Rgba8Unorm,
            .width = 64,
            .height = 64,
            .usage = ir::TextureUsage::Storage
        };
        ir::SemanticModule module;
        module.resources = {
            {.id = gpu::ResourceId{0}, .name = "A",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = texture},
            {.id = gpu::ResourceId{1}, .name = "B",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = texture}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0}, .name = "Compute",
            .computeEntry = "CSMain",
            .parameters = {{.name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute}}});
        module.works = {
            {.id = gpu::WorkId{0}, .name = "A",
                .accesses = {{gpu::ResourceId{0}, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0},
                    .bindings = {{0, gpu::ResourceId{0}}}}},
            {.id = gpu::WorkId{1}, .name = "B",
                .accesses = {{gpu::ResourceId{1}, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0},
                    .bindings = {{0, gpu::ResourceId{1}}}}}
        };
        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.resourceAliasing = true;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);
        assert(result.plan.lifetimes.size() == 2);
        assert(result.plan.lifetimes[0].allocation
            == result.plan.lifetimes[1].allocation);
    }

    void TestAliasingChecksWholeAllocationSlot()
    {
        using namespace sge;
        constexpr gpu::ResourceId resourceA{0};
        constexpr gpu::ResourceId resourceB{1};
        constexpr gpu::ResourceId resourceC{2};

        const ir::TextureDescription description{
            .dimension = gpu::ResourceKind::Texture2D,
            .format = gpu::ResourceFormat::Rgba8Unorm,
            .width = 64,
            .height = 64,
            .usage = ir::TextureUsage::Sampled
                | ir::TextureUsage::Storage
        };

        ir::SemanticModule module;
        module.resources = {
            {.id = resourceA, .name = "A",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = description},
            {.id = resourceB, .name = "B",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = description},
            {.id = resourceC, .name = "C",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = description}
        };
        module.programs = {
            {.id = gpu::ProgramId{0}, .name = "Write",
                .computeEntry = "CSWrite",
                .parameters = {{.name = "Output",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute}}},
            {.id = gpu::ProgramId{1}, .name = "ReadWrite",
                .computeEntry = "CSReadWrite",
                .parameters = {
                    {.name = "Input",
                        .kind = gpu::ProgramParameterKind::ShaderResource,
                        .stage = gpu::ProgramStage::Compute},
                    {.name = "Output",
                        .kind = gpu::ProgramParameterKind::UnorderedAccess,
                        .stage = gpu::ProgramStage::Compute}
                }}
        };
        module.works = {
            {.id = gpu::WorkId{0}, .name = "WriteA",
                .accesses = {{resourceA, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0},
                    .bindings = {{0, resourceA}}}},
            {.id = gpu::WorkId{1}, .name = "WriteB",
                .accesses = {{resourceB, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0},
                    .bindings = {{0, resourceB}}}},
            {.id = gpu::WorkId{2}, .name = "ReadBWriteC",
                .accesses = {
                    {resourceB, gpu::AccessMode::Read,
                        gpu::ResourceRole::ProgramInput},
                    {resourceC, gpu::AccessMode::Write,
                        gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{1},
                    .bindings = {{0, resourceB}, {1, resourceC}}}}
        };

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.resourceAliasing = true;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);

        const auto allocationOf = [&](gpu::ResourceId id)
        {
            return std::find_if(
                result.plan.lifetimes.begin(), result.plan.lifetimes.end(),
                [&](const compiler::ResourceLifetime& lifetime)
                {
                    return lifetime.resource == id;
                })->allocation;
        };

        assert(allocationOf(resourceA) == allocationOf(resourceB));
        assert(allocationOf(resourceC) != allocationOf(resourceA));
    }

    void TestCrossQueueUnorderedResourcesDoNotAlias()
    {
        using namespace sge;
        constexpr gpu::ResourceId computeBuffer{0};
        constexpr gpu::ResourceId copyBuffer{1};
        constexpr gpu::ResourceId copySource{2};

        const ir::TextureDescription aliasDescription{
            .dimension = gpu::ResourceKind::Texture2D,
            .format = gpu::ResourceFormat::Rgba8Unorm,
            .width = 64,
            .height = 64,
            .usage = ir::TextureUsage::Storage
                | ir::TextureUsage::CopyDestination
        };

        ir::SemanticModule module;
        module.resources = {
            {.id = computeBuffer, .name = "ComputeTexture",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = aliasDescription},
            {.id = copyBuffer, .name = "CopyTexture",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = aliasDescription},
            {.id = copySource, .name = "CopySource",
                .lifetime = gpu::ResourceLifetimeClass::FrameLocal,
                .update = gpu::ResourceUpdateClass::GpuProduced,
                .description = ir::TextureDescription{
                    .dimension = gpu::ResourceKind::Texture2D,
                    .format = gpu::ResourceFormat::Rgba8Unorm,
                    .width = 64,
                    .height = 64,
                    .usage = ir::TextureUsage::CopySource}}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0}, .name = "Compute",
            .computeEntry = "CSMain",
            .parameters = {{.name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute}}});
        module.works = {
            {.id = gpu::WorkId{0}, .name = "ComputeA",
                .accesses = {{computeBuffer, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0},
                    .bindings = {{0, computeBuffer}}}},
            {.id = gpu::WorkId{1}, .name = "CopyB",
                .accesses = {
                    {copySource, gpu::AccessMode::Read,
                        gpu::ResourceRole::TransferSource},
                    {copyBuffer, gpu::AccessMode::Write,
                        gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{copySource, copyBuffer, 0, 0, 0}}
        };

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.copyExecution = true;
        capabilities.concurrentCompute = true;
        capabilities.dedicatedCopyQueue = true;
        capabilities.resourceAliasing = true;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);

        const auto findAllocation = [&](gpu::ResourceId id)
        {
            return std::find_if(
                result.plan.lifetimes.begin(), result.plan.lifetimes.end(),
                [&](const compiler::ResourceLifetime& lifetime)
                {
                    return lifetime.resource == id;
                })->allocation;
        };
        assert(findAllocation(computeBuffer) != findAllocation(copyBuffer));
    }

    void TestPgaAndSdfModels()
    {
        using namespace sge;

        const auto pga = experimental::PgaBox::AxisAligned(1.0f);
        assert(pga.IsConsistent());
        assert(std::abs(pga.HalfExtent() - 1.0f) < 0.0001f);

        const experimental::SdfBox sdf{1.0f, 1.0f, 1.0f};
        assert(sdf.Evaluate(0.0f, 0.0f, 0.0f) < 0.0f);
        assert(std::abs(sdf.Evaluate(1.0f, 0.0f, 0.0f)) < 0.0001f);
        assert(sdf.Evaluate(2.0f, 0.0f, 0.0f) > 0.0f);
    }

    void TestTemporalResourcePlan()
    {
        using namespace sge;
        constexpr gpu::ResourceId history{0};

        ir::SemanticModule module;
        module.resources.push_back({
            .id = history,
            .name = "TemporalHistory",
            .lifetime = gpu::ResourceLifetimeClass::Temporal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 16,
                .usage = ir::BufferUsage::Storage
            }
        });
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "TemporalCompute",
            .computeEntry = "CSMain",
            .parameters = {
                {.name = "Previous",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute},
                {.name = "Current",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0}
            }
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "AccumulateHistory",
            .accesses = {
                {history, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput, 1},
                {history, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput, 0}
            },
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, history, 1}, {1, history, 0}}
            }
        });

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.concurrentCompute = true;
        capabilities.maxFramesInFlight = 3;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);
        assert(result.plan.dependencies.empty());
        assert(result.plan.temporalDependencies.size() == 1);
        assert(result.plan.temporalDependencies[0].readLag == 1);
        assert(result.plan.resourceInstances.size() == 1);
        assert(result.plan.resourceInstances[0].physicalInstanceCount == 3);
        assert(result.plan.resourceInstances[0].maximumFrameLag == 1);
        assert(result.plan.resourceInstances[0].selector
            == gpu::InstanceSelectorKind::CurrentTemporalSlot);
        assert(result.plan.scheduledWorks[0].requiredStates.size() == 2);
    }

    void TestTemporalLagDeterminesRingCapacity()
    {
        using namespace sge;
        constexpr gpu::ResourceId history{0};

        ir::SemanticModule module;
        module.resources.push_back({
            .id = history,
            .name = "LongTemporalHistory",
            .lifetime = gpu::ResourceLifetimeClass::Temporal,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 16,
                .usage = ir::BufferUsage::Storage}
        });
        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "LongTemporalCompute",
            .computeEntry = "CSMain",
            .parameters = {
                {.name = "Previous1",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0},
                {.name = "Previous2",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 1},
                {.name = "Previous3",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 2},
                {.name = "Previous4",
                    .kind = gpu::ProgramParameterKind::ShaderResource,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 3},
                {.name = "Current",
                    .kind = gpu::ProgramParameterKind::UnorderedAccess,
                    .stage = gpu::ProgramStage::Compute,
                    .registerIndex = 0}
            }
        });
        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "ReadFourPreviousFrames",
            .accesses = {
                {history, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput, 1},
                {history, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput, 2},
                {history, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput, 3},
                {history, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput, 4},
                {history, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput, 0}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {
                    {0, history, 1}, {1, history, 2},
                    {2, history, 3}, {3, history, 4},
                    {4, history, 0}}}
        });

        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.concurrentCompute = true;
        capabilities.maxFramesInFlight = 3;
        const auto result = compiler::RenderCompiler{}.Compile(
            module, capabilities);

        assert(result.plan.resourceInstances.size() == 1);
        assert(result.plan.resourceInstances[0].maximumFrameLag == 4);
        assert(result.plan.resourceInstances[0].physicalInstanceCount == 5);
        assert(result.plan.temporalDependencies.size() == 4);
    }

    void TestPersistentResourcesAreImmutableAndReadOnly()
    {
        using namespace sge;
        constexpr gpu::ResourceId persistent{0};

        const auto expectFailure = [](const ir::SemanticModule& module)
        {
            gpu::DeviceCapabilities capabilities;
            capabilities.computeExecution = true;
            bool failed = false;
            try
            {
                (void)compiler::RenderCompiler{}.Compile(module, capabilities);
            }
            catch (const std::runtime_error&)
            {
                failed = true;
            }
            assert(failed);
        };

        ir::SemanticModule produced;
        produced.resources.push_back({
            .id = persistent,
            .name = "InvalidProducedPersistent",
            .lifetime = gpu::ResourceLifetimeClass::Persistent,
            .update = gpu::ResourceUpdateClass::GpuProduced,
            .description = ir::BufferDescription{
                .sizeBytes = 16,
                .usage = ir::BufferUsage::Storage}
        });
        expectFailure(produced);

        ir::SemanticModule written;
        written.resources.push_back({
            .id = persistent,
            .name = "InvalidWrittenPersistent",
            .lifetime = gpu::ResourceLifetimeClass::Persistent,
            .update = gpu::ResourceUpdateClass::Immutable,
            .description = ir::BufferDescription{
                .sizeBytes = 16,
                .usage = ir::BufferUsage::Storage},
            .data = std::vector<std::byte>(16)
        });
        written.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "WritePersistent",
            .computeEntry = "CSMain",
            .parameters = {{.name = "Output",
                .kind = gpu::ProgramParameterKind::UnorderedAccess,
                .stage = gpu::ProgramStage::Compute}}
        });
        written.works.push_back({
            .id = gpu::WorkId{0},
            .name = "WritePersistent",
            .accesses = {{persistent, gpu::AccessMode::Write,
                gpu::ResourceRole::ProgramOutput}},
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .bindings = {{0, persistent}}}
        });
        expectFailure(written);
    }

    void TestCubePlan()
    {
        using namespace sge;

        cube_lab::ComparisonExperiment experiment;
        const auto module = experiment.Build(0.0, 16.0f / 9.0f);

        compiler::RenderCompiler compiler;
        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.copyExecution = true;
        capabilities.concurrentCompute = true;
        capabilities.dedicatedCopyQueue = true;
        capabilities.resourceAliasing = true;
        const auto result = compiler.Compile(module, capabilities);

        assert(module.works.size() == 9);
        assert(result.plan.scheduledWorks.size() == 9);
        assert(result.plan.executables.size() == 7);
        assert(!result.plan.dependencies.empty());
        assert(!result.plan.transitions.empty());

        const auto copyToRaster = std::find_if(
            result.plan.dependencies.begin(), result.plan.dependencies.end(),
            [](const compiler::DependencyEdge& edge)
            {
                return edge.before == 1 && edge.after == 5;
            });
        assert(copyToRaster != result.plan.dependencies.end());

        assert(result.plan.resourceInstances.size() == module.resources.size());
        assert(result.plan.frameBoundaryTransitions.empty());
        assert(result.plan.temporalDependencies.size() == 1);
    }

    void TestSdfFrontendPlan()
    {
        using namespace sge;

        experimental::SdfScene scene;
        scene.elapsedSeconds = 1.0;
        scene.aspectRatio = 16.0f / 9.0f;
        scene.box = {1.0f, 0.75f, 1.25f};
        const auto module = experimental::SdfRayMarchLowering{}.Lower(scene);

        const auto result = compiler::RenderCompiler{}.Compile(
            module, gpu::DeviceCapabilities{});

        assert(module.resources.size() == 3);
        assert(module.programs.size() == 1);
        assert(module.works.size() == 2);
        assert(result.plan.scheduledWorks.size() == 2);
        assert(result.plan.executables.size() == 1);
        assert(result.plan.temporalDependencies.empty());
        assert(result.plan.resourceInstances.size() == 3);
        assert(module.resources[0].name == "SdfFullscreenTriangle");
        assert(module.resources[1].name == "SdfRayMarchConstants");

        cube_lab::ComparisonExperiment experiment;
        const auto selected = experiment.Build(
            1.0, 16.0f / 9.0f,
            cube_lab::ExperimentMode::Sdf);
        assert(selected.StructureHash() == module.StructureHash());
    }
}

int main()
{
    try
    {
        TestDependencyAndTransitions();
        TestUnsupportedWorkFails();
        TestPayloadAccessMismatchFails();
        TestComputeCopyRasterAndQueues();
        TestSingleQueuePlanNeedsNoFrameBoundaryRelease();
        TestTransientTextureAliasingPlan();
        TestAliasingChecksWholeAllocationSlot();
        TestCrossQueueUnorderedResourcesDoNotAlias();
        TestTemporalResourcePlan();
        TestTemporalLagDeterminesRingCapacity();
        TestPersistentResourcesAreImmutableAndReadOnly();
        TestPgaAndSdfModels();
        TestCubePlan();
        TestSdfFrontendPlan();
        RunD3D12IntegrationTests();
        std::cout
            << "All compiler, frontend, and D3D12 WARP tests passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
