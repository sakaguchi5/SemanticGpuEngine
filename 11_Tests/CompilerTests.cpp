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
            .kind = gpu::ResourceKind::Buffer,
            .memoryClass = gpu::MemoryClass::Transient,
            .format = gpu::ResourceFormat::Unknown,
            .sizeBytes = 256,
            .data = {}
        });

        module.works.push_back({
            .id = gpu::WorkId{0},
            .name = "Writer",
            .domain = gpu::ExecutionDomain::Compute,
            .program = gpu::ProgramId{},
            .accesses = {
                {
                    data,
                    gpu::AccessMode::Write,
                    gpu::ResourceRole::ProgramOutput
                }
            },
            .vertexResource = gpu::ResourceId{},
            .constantResource = gpu::ResourceId{}
        });

        module.works.push_back({
            .id = gpu::WorkId{1},
            .name = "Reader",
            .domain = gpu::ExecutionDomain::Compute,
            .program = gpu::ProgramId{},
            .accesses = {
                {
                    data,
                    gpu::AccessMode::Read,
                    gpu::ResourceRole::ProgramInput
                }
            },
            .vertexResource = gpu::ResourceId{},
            .constantResource = gpu::ResourceId{}
        });

        compiler::RenderCompiler compiler;
        const auto result = compiler.Compile(
            module,
            gpu::DeviceCapabilities{});

        assert(result.plan.dependencies.size() == 1);
        assert(result.plan.dependencies[0].before == 0);
        assert(result.plan.dependencies[0].after == 1);
        assert(result.plan.scheduledWorks.size() == 2);
        assert(result.plan.transitions.size() == 2);
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
        const auto result = compiler.Compile(
            module,
            gpu::DeviceCapabilities{});

        assert(module.works.size() == 4);
        assert(result.plan.scheduledWorks.size() == 4);
        assert(result.plan.executables.size() == 3);
        assert(!result.plan.dependencies.empty());
        assert(!result.plan.transitions.empty());
    }
}

int main()
{
    try
    {
        TestDependencyAndTransitions();
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
