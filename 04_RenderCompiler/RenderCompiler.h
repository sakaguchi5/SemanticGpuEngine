#pragma once

#include "03_RenderIR/RenderIR.h"

#include <cstddef>
#include <string>
#include <vector>

namespace sge::compiler
{
    struct DependencyEdge
    {
        std::size_t before = 0;
        std::size_t after = 0;
        gpu::ResourceId resource;
    };

    struct ResourceLifetime
    {
        gpu::ResourceId resource;
        std::size_t firstUse = 0;
        std::size_t lastUse = 0;
        gpu::PhysicalAllocationId allocation;
    };

    struct StateTransition
    {
        gpu::ResourceId resource;
        std::size_t beforeScheduledWork = 0;
        gpu::AbstractState from = gpu::AbstractState::Undefined;
        gpu::AbstractState to = gpu::AbstractState::Undefined;
    };

    struct ExecutableKey
    {
        gpu::ProgramId program;
        ir::RasterState rasterState{};

        auto operator<=>(const ExecutableKey&) const = default;
    };

    struct RequiredResourceState
    {
        gpu::ResourceId resource;
        gpu::AbstractState state = gpu::AbstractState::Undefined;
    };

    struct ScheduledWork
    {
        std::size_t sourceWorkIndex = 0;
        std::vector<RequiredResourceState> requiredStates;
        ExecutableKey executable;
    };

    struct ExecutionPlan
    {
        std::size_t structureHash = 0;
        std::vector<DependencyEdge> dependencies;
        std::vector<ScheduledWork> scheduledWorks;
        std::vector<ResourceLifetime> lifetimes;
        std::vector<StateTransition> transitions;
        std::vector<ExecutableKey> executables;
    };

    struct CompileResult
    {
        ExecutionPlan plan;
        std::vector<std::string> diagnostics;
    };

    class RenderCompiler
    {
    public:
        [[nodiscard]] CompileResult Compile(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities) const;

    private:
        static void Validate(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities,
            std::vector<std::string>& diagnostics);

        static std::vector<DependencyEdge> AnalyzeDependencies(
            const ir::SemanticModule& module);

        static std::vector<std::size_t> Schedule(
            std::size_t workCount,
            const std::vector<DependencyEdge>& dependencies);

        static std::vector<ResourceLifetime> AnalyzeLifetimes(
            const ir::SemanticModule& module,
            const std::vector<std::size_t>& schedule);

        static std::vector<ScheduledWork> BuildScheduledWorks(
            const ir::SemanticModule& module,
            const std::vector<std::size_t>& schedule);

        static std::vector<StateTransition> PlanTransitions(
            const std::vector<ScheduledWork>& works);
    };
}
