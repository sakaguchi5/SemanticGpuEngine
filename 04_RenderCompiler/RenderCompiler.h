#pragma once

#include "03_RenderIR/RenderIR.h"

#include <cstddef>
#include <memory>
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
        std::uint32_t frameLag = 0;
    };

    struct ExecutableKey
    {
        gpu::ProgramId program;
        ir::RasterState rasterState{};
        bool compute = false;

        auto operator<=>(const ExecutableKey&) const = default;
    };

    struct RequiredResourceState
    {
        gpu::ResourceId resource;
        gpu::AbstractState state = gpu::AbstractState::Undefined;
        std::uint32_t frameLag = 0;
    };

    struct ResourceInstancePlan
    {
        gpu::ResourceId resource;
        gpu::ResourceLifetimeClass lifetime =
            gpu::ResourceLifetimeClass::Persistent;
        gpu::InstanceSelectorKind selector =
            gpu::InstanceSelectorKind::Persistent;
        std::uint32_t physicalInstanceCount = 1;
        std::uint32_t maximumFrameLag = 0;
    };

    // Immutable persistent resources are initialized once into the union of
    // every read state required by the compiled plan. Individual works keep
    // their logical requirements, but no longer transition the physical
    // resource between compatible read states.
    struct PersistentReadStatePlan
    {
        gpu::ResourceId resource;
        std::vector<gpu::AbstractState> states;
    };

    struct TemporalDependency
    {
        gpu::ResourceId resource;
        std::size_t consumerScheduledWork = 0;
        std::uint32_t readLag = 1;
        gpu::QueueClass producerQueue = gpu::QueueClass::Direct;
        gpu::QueueClass consumerQueue = gpu::QueueClass::Direct;
    };

    struct ScheduledWork
    {
        std::size_t sourceWorkIndex = 0;
        std::vector<RequiredResourceState> requiredStates;
        ExecutableKey executable;
        gpu::QueueClass queue = gpu::QueueClass::Direct;
    };

    struct QueueSynchronization
    {
        std::size_t signalScheduledWork = 0;
        std::size_t waitScheduledWork = 0;
        gpu::QueueClass signalQueue = gpu::QueueClass::Direct;
        gpu::QueueClass waitQueue = gpu::QueueClass::Direct;
    };

    // End-state normalization for a persistent physical instance. Frame-slot
    // reuse is guarded separately by queue completion fences, and temporal
    // data flow is represented by TemporalDependency.
    struct FrameBoundaryTransition
    {
        gpu::ResourceId resource;
        std::size_t afterScheduledWork = 0;
        gpu::QueueClass releaseQueue = gpu::QueueClass::Direct;
        gpu::AbstractState from = gpu::AbstractState::Undefined;
        gpu::AbstractState to = gpu::AbstractState::Undefined;
        gpu::QueueClass nextFrameQueue = gpu::QueueClass::Direct;
        gpu::AbstractState nextFrameState = gpu::AbstractState::Undefined;
    };

    struct ExecutionPlan
    {
        std::size_t structureHash = 0;
        std::vector<DependencyEdge> dependencies;
        std::vector<ScheduledWork> scheduledWorks;
        std::vector<ResourceLifetime> lifetimes;
        std::vector<StateTransition> transitions;
        std::vector<ExecutableKey> executables;
        std::vector<QueueSynchronization> queueSynchronizations;
        std::vector<FrameBoundaryTransition> frameBoundaryTransitions;
        std::vector<ResourceInstancePlan> resourceInstances;
        std::vector<PersistentReadStatePlan> persistentReadStates;
        std::vector<TemporalDependency> temporalDependencies;
    };

    struct CompileResult
    {
        ExecutionPlan plan;
        std::vector<std::string> diagnostics;
    };

    class ISchedulingPolicy
    {
    public:
        virtual ~ISchedulingPolicy() = default;
        [[nodiscard]] virtual std::vector<std::size_t> Schedule(
            const ir::SemanticModule& module,
            const std::vector<DependencyEdge>& dependencies) const = 0;
        [[nodiscard]] virtual const char* Name() const noexcept = 0;
    };

    class StableDeclarationOrderPolicy final : public ISchedulingPolicy
    {
    public:
        [[nodiscard]] std::vector<std::size_t> Schedule(
            const ir::SemanticModule& module,
            const std::vector<DependencyEdge>& dependencies) const override;
        [[nodiscard]] const char* Name() const noexcept override;
    };

    class RenderCompiler
    {
    public:
        RenderCompiler();
        explicit RenderCompiler(std::shared_ptr<const ISchedulingPolicy> policy);

        [[nodiscard]] CompileResult Compile(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities) const;

        static std::vector<std::size_t> StableSchedule(
            std::size_t workCount,
            const std::vector<DependencyEdge>& dependencies);

    private:
        static void Validate(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities,
            std::vector<std::string>& diagnostics);

        static std::vector<DependencyEdge> AnalyzeDependencies(
            const ir::SemanticModule& module);

        static std::vector<ResourceLifetime> AnalyzeLifetimes(
            const ir::SemanticModule& module,
            const std::vector<std::size_t>& schedule,
            const std::vector<DependencyEdge>& dependencies,
            const std::vector<ScheduledWork>& works,
            bool allowAliasing);

        static std::vector<ScheduledWork> BuildScheduledWorks(
            const ir::SemanticModule& module,
            const std::vector<std::size_t>& schedule,
            const gpu::DeviceCapabilities& capabilities);

        static std::vector<StateTransition> PlanTransitions(
            const ir::SemanticModule& module,
            const std::vector<ScheduledWork>& works);

        static std::vector<PersistentReadStatePlan> PlanPersistentReadStates(
            const ir::SemanticModule& module,
            const std::vector<ScheduledWork>& works);

        static std::vector<ResourceInstancePlan> PlanResourceInstances(
            const ir::SemanticModule& module,
            const gpu::DeviceCapabilities& capabilities);

        static std::vector<TemporalDependency> PlanTemporalDependencies(
            const ir::SemanticModule& module,
            const std::vector<std::size_t>& schedule,
            const std::vector<ScheduledWork>& works);

        static std::vector<QueueSynchronization> PlanQueueSynchronization(
            const std::vector<DependencyEdge>& dependencies,
            const std::vector<std::size_t>& schedule,
            const std::vector<ScheduledWork>& works);

        static std::vector<FrameBoundaryTransition>
            PlanFrameBoundaryTransitions(
                const ir::SemanticModule& module,
                const std::vector<ScheduledWork>& works);

        std::shared_ptr<const ISchedulingPolicy> schedulingPolicy_;
    };
}
