#include "12_CubeLab/ExperimentHarness.h"

#include <chrono>
#include <fstream>
#include <stdexcept>

namespace sge::cube_lab
{
    ExperimentMetrics ExperimentHarness::Measure(
        const ExperimentScene& scene,
        ExperimentMode mode,
        const gpu::DeviceCapabilities& capabilities) const
    {
        using Clock = std::chrono::steady_clock;
        const auto loweringStart = Clock::now();
        const auto module = experiment_.Build(scene, mode);
        const auto loweringEnd = Clock::now();
        const auto compileStart = Clock::now();
        const auto result = compiler_.Compile(module, capabilities);
        const auto compileEnd = Clock::now();

        ExperimentMetrics metrics;
        metrics.mode = mode;
        metrics.loweringMilliseconds =
            std::chrono::duration<double, std::milli>(
                loweringEnd - loweringStart).count();
        metrics.compileMilliseconds =
            std::chrono::duration<double, std::milli>(
                compileEnd - compileStart).count();
        metrics.package = result.package.statistics;
        metrics.diagnosticCount = result.diagnostics.size();
        metrics.succeeded = result.Succeeded();
        return metrics;
    }

    ExperimentReport ExperimentHarness::Run(
        const ExperimentScene& scene,
        const gpu::DeviceCapabilities& capabilities) const
    {
        return {
            .scene = scene,
            .classical = Measure(
                scene, ExperimentMode::Classical, capabilities),
            .sdf = Measure(scene, ExperimentMode::Sdf, capabilities)
        };
    }

    void ExperimentHarness::WriteCsv(
        const ExperimentReport& report,
        const std::filesystem::path& outputPath)
    {
        std::ofstream output(outputPath, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("Could not write experiment CSV.");
        }
        output << "mode,lowering_ms,compile_ms,logical_resources,"
                  "physical_instances,works,executables,descriptor_views,"
                  "operations,barriers,queue_waits,estimated_bytes,"
                  "diagnostics,succeeded\n";
        const auto write = [&](const char* name, const ExperimentMetrics& value)
        {
            output << name << ','
                   << value.loweringMilliseconds << ','
                   << value.compileMilliseconds << ','
                   << value.package.logicalResourceCount << ','
                   << value.package.physicalInstanceCount << ','
                   << value.package.workCount << ','
                   << value.package.executableCount << ','
                   << value.package.descriptorViewCount << ','
                   << value.package.operationCount << ','
                   << value.package.barrierCount << ','
                   << value.package.queueWaitCount << ','
                   << value.package.estimatedCommittedBytes << ','
                   << value.diagnosticCount << ','
                   << (value.succeeded ? 1 : 0) << '\n';
        };
        write("Classical", report.classical);
        write("Sdf", report.sdf);
    }
}
