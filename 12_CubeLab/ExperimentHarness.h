#pragma once

#include "04_RenderCompiler/CompiledRenderPackage.h"
#include "12_CubeLab/CubeExperiment.h"

#include <filesystem>
#include <vector>

namespace sge::cube_lab
{
    struct ExperimentMetrics
    {
        ExperimentMode mode = ExperimentMode::Classical;
        double loweringMilliseconds = 0.0;
        double compileMilliseconds = 0.0;
        compiler::PackageStatistics package;
        std::size_t diagnosticCount = 0;
        bool succeeded = false;
    };

    struct ExperimentReport
    {
        ExperimentScene scene;
        ExperimentMetrics classical;
        ExperimentMetrics sdf;
    };

    class ExperimentHarness
    {
    public:
        [[nodiscard]] ExperimentReport Run(
            const ExperimentScene& scene,
            const gpu::DeviceCapabilities& capabilities) const;

        static void WriteCsv(
            const ExperimentReport& report,
            const std::filesystem::path& outputPath);

    private:
        [[nodiscard]] ExperimentMetrics Measure(
            const ExperimentScene& scene,
            ExperimentMode mode,
            const gpu::DeviceCapabilities& capabilities) const;

        ComparisonExperiment experiment_;
        compiler::RenderPackageCompiler compiler_;
    };
}
