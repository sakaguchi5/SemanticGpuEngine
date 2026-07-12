#include "02_GpuSemantics/Semantics.h"
#include "03_RenderIR/RenderIR.h"
#include "04_RenderCompiler/RenderCompiler.h"
#include "09_ExperimentalGeometry/ExperimentalGeometry.h"
#include "12_CubeLab/CubeExperiment.h"

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
            .memoryClass = gpu::MemoryClass::Transient,
            .description = ir::BufferDescription{
                .sizeBytes = 256,
                .usage = ir::BufferUsage::Storage
            },
            .data = {}
        });

        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "Writer",
            .accesses = {
                {
                    data,
                    gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput
                }
            },
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .groupCountX = 1
            }
        });

        module.works.push_back({
            .id = gpu::WorkId{1},
            .name = "Reader",
            .accesses = {
                {
                    data,
                    gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput
                }
            },
            .payload = ir::ComputeWork{
                .program = gpu::ProgramId{0},
                .groupCountX = 1
            }
        });

        module.programs.push_back({
            .id = gpu::ProgramId{0},
            .name = "TestCompute",
            .shaderPath = "unused.hlsl",
            .computeEntry = "CSMain"
        });

        compiler::RenderCompiler compiler;
        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        const auto result = compiler.Compile(module, capabilities);

        assert(result.plan.dependencies.size() == 1);
        assert(result.plan.dependencies[0].before == 0);
        assert(result.plan.dependencies[0].after == 1);
        assert(result.plan.scheduledWorks.size() == 2);
        assert(result.plan.transitions.size() == 2);
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

    void TestComputeCopyRasterAndQueues()
    {
        using namespace sge;
        constexpr gpu::ResourceId generated{0};
        constexpr gpu::ResourceId copied{1};
        constexpr gpu::ResourceId color{2};

        ir::SemanticModule module;
        module.resources = {
            {generated, "Generated", gpu::MemoryClass::Transient,
                ir::BufferDescription{256, 16,
                    ir::BufferUsage::Storage | ir::BufferUsage::CopySource}},
            {copied, "Copied", gpu::MemoryClass::Transient,
                ir::BufferDescription{256, 16,
                    ir::BufferUsage::Vertex | ir::BufferUsage::CopyDestination}},
            {color, "Presentation", gpu::MemoryClass::External,
                ir::PresentationDescription{}}
        };
        module.programs = {
            {.id = gpu::ProgramId{0}, .name = "Generate", .computeEntry = "CSMain"},
            {.id = gpu::ProgramId{1}, .name = "Draw", .vertexEntry = "VSMain",
                .pixelEntry = "PSMain"}
        };
        module.works = {
            {
                .id = gpu::WorkId{0}, .name = "Generate",
                .accesses = {{generated, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0}}
            },
            {
                .id = gpu::WorkId{1}, .name = "Copy",
                .accesses = {
                    {generated, gpu::AccessMode::Read, gpu::ResourceRole::TransferSource},
                    {copied, gpu::AccessMode::Write, gpu::ResourceRole::TransferDestination}},
                .payload = ir::CopyWork{generated, copied, 0, 0, 256}
            },
            {
                .id = gpu::WorkId{2}, .name = "Draw",
                .accesses = {
                    {copied, gpu::AccessMode::Read, gpu::ResourceRole::VertexInput},
                    {color, gpu::AccessMode::Write, gpu::ResourceRole::ColorOutput}},
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
        const auto result = compiler::RenderCompiler{}.Compile(module, capabilities);
        assert(result.plan.scheduledWorks[0].queue == gpu::QueueClass::Compute);
        assert(result.plan.scheduledWorks[1].queue == gpu::QueueClass::Copy);
        assert(result.plan.scheduledWorks[2].queue == gpu::QueueClass::Direct);
        assert(result.plan.queueSynchronizations.size() == 2);
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
            {gpu::ResourceId{0}, "A", gpu::MemoryClass::Transient, texture},
            {gpu::ResourceId{1}, "B", gpu::MemoryClass::Transient, texture}
        };
        module.programs.push_back({
            .id = gpu::ProgramId{0}, .name = "Compute", .computeEntry = "CSMain"});
        module.works = {
            {.id = gpu::WorkId{0}, .name = "A",
                .accesses = {{gpu::ResourceId{0}, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0}}},
            {.id = gpu::WorkId{1}, .name = "ReadA",
                .accesses = {{gpu::ResourceId{0}, gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0}}},
            {.id = gpu::WorkId{2}, .name = "B",
                .accesses = {{gpu::ResourceId{1}, gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput}},
                .payload = ir::ComputeWork{.program = gpu::ProgramId{0}}}
        };
        gpu::DeviceCapabilities capabilities;
        capabilities.computeExecution = true;
        capabilities.resourceAliasing = true;
        const auto result = compiler::RenderCompiler{}.Compile(module, capabilities);
        assert(result.plan.lifetimes.size() == 2);
        assert(result.plan.lifetimes[0].allocation
            == result.plan.lifetimes[1].allocation);
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
        const auto result = compiler.Compile(module, capabilities);

        assert(module.works.size() == 7);
        assert(result.plan.scheduledWorks.size() == 7);
        assert(result.plan.executables.size() == 5);
        assert(!result.plan.dependencies.empty());
        assert(!result.plan.transitions.empty());
    }
}

int main()
{
    try
    {
        TestDependencyAndTransitions();
        TestUnsupportedWorkFails();
        TestComputeCopyRasterAndQueues();
        TestTransientTextureAliasingPlan();
        TestPgaAndSdfModels();
        TestCubePlan();
        std::cout << "All compiler and frontend tests passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
