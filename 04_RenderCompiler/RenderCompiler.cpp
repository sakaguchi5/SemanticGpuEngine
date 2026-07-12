#include "04_RenderCompiler/RenderCompiler.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace
{
    struct ResourceHistory
    {
        std::optional<std::size_t> lastWriter;
        std::vector<std::size_t> readers;
    };

    std::uint64_t EdgeKey(std::size_t before, std::size_t after)
    {
        return (static_cast<std::uint64_t>(before) << 32u)
            | static_cast<std::uint64_t>(after);
    }
}

namespace sge::compiler
{
    RenderCompiler::RenderCompiler()
        : RenderCompiler(std::make_shared<StableDeclarationOrderPolicy>())
    {
    }

    RenderCompiler::RenderCompiler(
        std::shared_ptr<const ISchedulingPolicy> policy)
        : schedulingPolicy_(std::move(policy))
    {
        if (!schedulingPolicy_)
        {
            throw std::invalid_argument("Scheduling policy must not be null.");
        }
    }

    CompileResult RenderCompiler::Compile(
        const ir::SemanticModule& module,
        const gpu::DeviceCapabilities& capabilities) const
    {
        CompileResult result;
        result.plan.structureHash = module.StructureHash();

        Validate(module, capabilities, result.diagnostics);
        result.plan.dependencies = AnalyzeDependencies(module);

        const auto schedule = schedulingPolicy_->Schedule(
            module, result.plan.dependencies);

        result.plan.lifetimes = AnalyzeLifetimes(
            module, schedule, capabilities.resourceAliasing);
        result.plan.scheduledWorks = BuildScheduledWorks(
            module, schedule, capabilities);
        result.plan.transitions = PlanTransitions(result.plan.scheduledWorks);
        result.plan.queueSynchronizations = PlanQueueSynchronization(
            result.plan.dependencies, schedule, result.plan.scheduledWorks);

        for (const auto& scheduled : result.plan.scheduledWorks)
        {
            const auto& source = module.works.at(scheduled.sourceWorkIndex);
            if (source.Domain() != gpu::ExecutionDomain::Raster
                && source.Domain() != gpu::ExecutionDomain::Compute)
            {
                continue;
            }

            const auto found = std::find(
                result.plan.executables.begin(),
                result.plan.executables.end(),
                scheduled.executable);

            if (found == result.plan.executables.end())
            {
                result.plan.executables.push_back(scheduled.executable);
            }
        }

        result.diagnostics.push_back(
            "Compilation succeeded: "
            + std::to_string(module.works.size())
            + " works, "
            + std::to_string(result.plan.dependencies.size())
            + " dependencies, "
            + std::to_string(result.plan.transitions.size())
            + " abstract transitions.");
        result.diagnostics.push_back(
            std::string("Scheduling policy: ") + schedulingPolicy_->Name());

        return result;
    }

    void RenderCompiler::Validate(
        const ir::SemanticModule& module,
        const gpu::DeviceCapabilities& capabilities,
        std::vector<std::string>& diagnostics)
    {
        std::unordered_set<std::uint32_t> resourceIds;
        std::unordered_set<std::uint32_t> programIds;
        std::unordered_set<std::uint32_t> workIds;

        for (const auto& resource : module.resources)
        {
            if (!resource.id.IsValid()
                || !resourceIds.insert(resource.id.Value()).second)
            {
                throw std::runtime_error(
                    "Semantic validation failed: invalid or duplicate resource ID.");
            }

            if (resource.Kind() == gpu::ResourceKind::Buffer
                && resource.memoryClass != gpu::MemoryClass::DynamicPerFrame
                && resource.memoryClass != gpu::MemoryClass::External
                && resource.SizeBytes() == 0)
            {
                throw std::runtime_error(
                    "Semantic validation failed: non-dynamic buffer has zero size.");
            }

            if (resource.memoryClass == gpu::MemoryClass::Static
                && resource.data.size() != resource.SizeBytes())
            {
                throw std::runtime_error(
                    "Semantic validation failed: static resource data size mismatch.");
            }
        }

        for (const auto& program : module.programs)
        {
            if (!program.id.IsValid()
                || !programIds.insert(program.id.Value()).second)
            {
                throw std::runtime_error(
                    "Semantic validation failed: invalid or duplicate program ID.");
            }
        }

        for (const auto& work : module.works)
        {
            if (!work.id.IsValid()
                || !workIds.insert(work.id.Value()).second)
            {
                throw std::runtime_error(
                    "Semantic validation failed: invalid or duplicate work ID.");
            }

            const auto domain = work.Domain();

            if (domain == gpu::ExecutionDomain::Raster)
            {
                if (!capabilities.rasterExecution)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster execution is unavailable.");
                }

                const auto& raster = std::get<ir::RasterWork>(work.payload);
                if (!programIds.contains(raster.program.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work references an unknown program.");
                }

                if (!resourceIds.contains(raster.vertexResource.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work has no valid vertex resource.");
                }

                if (raster.attachments.colors.empty())
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work has no color attachment.");
                }
            }

            if (domain == gpu::ExecutionDomain::Compute)
            {
                if (!capabilities.computeExecution)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: compute execution is unavailable.");
                }
                const auto& compute = std::get<ir::ComputeWork>(work.payload);
                if (!programIds.contains(compute.program.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: compute work references an unknown program.");
                }
                if (compute.groupCountX == 0 || compute.groupCountY == 0
                    || compute.groupCountZ == 0)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: compute group count is zero.");
                }
            }

            if (domain == gpu::ExecutionDomain::Copy
                && !capabilities.copyExecution)
            {
                throw std::runtime_error(
                    "Semantic validation failed: copy execution is unavailable.");
            }
            if (domain == gpu::ExecutionDomain::Copy)
            {
                const auto& copy = std::get<ir::CopyWork>(work.payload);
                if (!resourceIds.contains(copy.source.Value())
                    || !resourceIds.contains(copy.destination.Value())
                    || copy.source == copy.destination)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: invalid copy resources.");
                }
            }

            for (const auto& access : work.accesses)
            {
                if (!resourceIds.contains(access.resource.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: work references an unknown resource.");
                }
            }

            const auto validateBindings = [&](const auto& bindings,
                                              gpu::ProgramId programId)
            {
                const auto& program = module.Program(programId);
                std::unordered_set<std::uint32_t> bound;
                for (const auto& binding : bindings)
                {
                    if (binding.parameterIndex >= program.parameters.size()
                        || !resourceIds.contains(binding.resource.Value())
                        || !bound.insert(binding.parameterIndex).second)
                    {
                        throw std::runtime_error(
                            "Semantic validation failed: invalid or duplicate program binding.");
                    }
                    if (program.parameters[binding.parameterIndex].kind
                        == gpu::ProgramParameterKind::Sampler)
                    {
                        throw std::runtime_error(
                            "Semantic validation failed: static samplers must not have resource bindings.");
                    }
                }
                const auto requiredBindings = static_cast<std::size_t>(std::count_if(
                    program.parameters.begin(), program.parameters.end(),
                    [](const gpu::ProgramParameter& parameter)
                    {
                        return parameter.kind
                            != gpu::ProgramParameterKind::Sampler;
                    }));
                if (bound.size() != requiredBindings)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: not all program parameters are bound.");
                }
            };

            if (domain == gpu::ExecutionDomain::Raster)
            {
                const auto& raster = std::get<ir::RasterWork>(work.payload);
                validateBindings(raster.bindings, raster.program);
            }
            else if (domain == gpu::ExecutionDomain::Compute)
            {
                const auto& compute = std::get<ir::ComputeWork>(work.payload);
                validateBindings(compute.bindings, compute.program);
            }
        }

        diagnostics.push_back("Semantic validation passed.");
    }

    std::vector<DependencyEdge> RenderCompiler::AnalyzeDependencies(
        const ir::SemanticModule& module)
    {
        std::unordered_map<
            gpu::ResourceId,
            ResourceHistory,
            foundation::StrongIdHash<gpu::ResourceTag>> histories;

        std::vector<DependencyEdge> edges;
        std::unordered_set<std::uint64_t> uniqueEdges;

        const auto addEdge = [&](std::size_t before,
                                 std::size_t after,
                                 gpu::ResourceId resource)
        {
            if (before == after)
            {
                return;
            }

            if (uniqueEdges.insert(EdgeKey(before, after)).second)
            {
                edges.push_back({
                    .before = before,
                    .after = after,
                    .resource = resource
                });
            }
        };

        for (std::size_t workIndex = 0;
             workIndex < module.works.size();
             ++workIndex)
        {
            for (const auto& access : module.works[workIndex].accesses)
            {
                auto& history = histories[access.resource];

                const bool reads =
                    access.access == gpu::AccessMode::Read
                    || access.access == gpu::AccessMode::ReadWrite;

                const bool writes =
                    access.access == gpu::AccessMode::Write
                    || access.access == gpu::AccessMode::ReadWrite;

                if (reads)
                {
                    if (history.lastWriter)
                    {
                        addEdge(*history.lastWriter, workIndex, access.resource);
                    }
                    history.readers.push_back(workIndex);
                }

                if (writes)
                {
                    if (history.lastWriter)
                    {
                        addEdge(*history.lastWriter, workIndex, access.resource);
                    }

                    for (const auto reader : history.readers)
                    {
                        addEdge(reader, workIndex, access.resource);
                    }

                    history.readers.clear();
                    history.lastWriter = workIndex;
                }
            }
        }

        return edges;
    }

    std::vector<std::size_t> RenderCompiler::StableSchedule(
        std::size_t workCount,
        const std::vector<DependencyEdge>& dependencies)
    {
        std::vector<std::vector<std::size_t>> outgoing(workCount);
        std::vector<std::size_t> indegree(workCount, 0);

        for (const auto& edge : dependencies)
        {
            outgoing.at(edge.before).push_back(edge.after);
            ++indegree.at(edge.after);
        }

        std::deque<std::size_t> ready;
        for (std::size_t index = 0; index < workCount; ++index)
        {
            if (indegree[index] == 0)
            {
                ready.push_back(index);
            }
        }

        std::vector<std::size_t> result;
        result.reserve(workCount);

        while (!ready.empty())
        {
            const auto current = ready.front();
            ready.pop_front();
            result.push_back(current);

            auto& nextWorks = outgoing[current];
            std::sort(nextWorks.begin(), nextWorks.end());

            for (const auto next : nextWorks)
            {
                if (--indegree[next] == 0)
                {
                    ready.insert(
                        std::upper_bound(ready.begin(), ready.end(), next),
                        next);
                }
            }
        }

        if (result.size() != workCount)
        {
            throw std::runtime_error(
                "RenderCompiler: cyclic work dependency detected.");
        }

        return result;
    }

    std::vector<std::size_t> StableDeclarationOrderPolicy::Schedule(
        const ir::SemanticModule& module,
        const std::vector<DependencyEdge>& dependencies) const
    {
        return RenderCompiler::StableSchedule(module.works.size(), dependencies);
    }

    const char* StableDeclarationOrderPolicy::Name() const noexcept
    {
        return "StableDeclarationOrder";
    }

    std::vector<ResourceLifetime> RenderCompiler::AnalyzeLifetimes(
        const ir::SemanticModule& module,
        const std::vector<std::size_t>& schedule,
        bool allowAliasing)
    {
        struct Range
        {
            std::size_t first = std::numeric_limits<std::size_t>::max();
            std::size_t last = 0;
        };

        std::unordered_map<
            gpu::ResourceId,
            Range,
            foundation::StrongIdHash<gpu::ResourceTag>> ranges;

        for (std::size_t scheduledIndex = 0;
             scheduledIndex < schedule.size();
             ++scheduledIndex)
        {
            const auto& work = module.works.at(schedule[scheduledIndex]);
            for (const auto& access : work.accesses)
            {
                auto& range = ranges[access.resource];
                range.first = std::min(range.first, scheduledIndex);
                range.last = std::max(range.last, scheduledIndex);
            }
        }

        std::vector<ResourceLifetime> lifetimes;
        lifetimes.reserve(ranges.size());

        std::uint32_t nextAllocation = 0;

        for (const auto& resource : module.resources)
        {
            const auto range = ranges.find(resource.id);
            if (range == ranges.end())
            {
                continue;
            }

            gpu::PhysicalAllocationId allocation{nextAllocation++};

            if (allowAliasing
                && resource.memoryClass == gpu::MemoryClass::Transient)
            {
                for (const auto& existing : lifetimes)
                {
                    const auto& existingResource =
                        module.Resource(existing.resource);

                    const bool compatible =
                        existingResource.memoryClass
                            == gpu::MemoryClass::Transient
                        && existingResource.Kind() == resource.Kind()
                        && existingResource.Format() == resource.Format()
                        && existingResource.description == resource.description;

                    const bool disjoint =
                        existing.lastUse < range->second.first
                        || range->second.last < existing.firstUse;

                    if (compatible && disjoint)
                    {
                        allocation = existing.allocation;
                        break;
                    }
                }
            }

            lifetimes.push_back({
                .resource = resource.id,
                .firstUse = range->second.first,
                .lastUse = range->second.last,
                .allocation = allocation
            });
        }

        return lifetimes;
    }

    std::vector<ScheduledWork> RenderCompiler::BuildScheduledWorks(
        const ir::SemanticModule& module,
        const std::vector<std::size_t>& schedule,
        const gpu::DeviceCapabilities& capabilities)
    {
        std::vector<ScheduledWork> result;
        result.reserve(schedule.size());

        for (const auto sourceIndex : schedule)
        {
            const auto& source = module.works.at(sourceIndex);

            ScheduledWork scheduled;
            scheduled.sourceWorkIndex = sourceIndex;

            switch (source.Domain())
            {
            case gpu::ExecutionDomain::Raster:
            {
                const auto& raster = std::get<ir::RasterWork>(source.payload);
                scheduled.executable = {
                    .program = raster.program,
                    .rasterState = raster.rasterState,
                    .compute = false
                };
                scheduled.queue = gpu::QueueClass::Direct;
                break;
            }
            case gpu::ExecutionDomain::Compute:
            {
                const auto& compute = std::get<ir::ComputeWork>(source.payload);
                scheduled.executable = {
                    .program = compute.program,
                    .rasterState = {},
                    .compute = true
                };
                scheduled.queue = capabilities.concurrentCompute
                    ? gpu::QueueClass::Compute
                    : gpu::QueueClass::Direct;
                break;
            }
            case gpu::ExecutionDomain::Copy:
                scheduled.queue = capabilities.dedicatedCopyQueue
                    ? gpu::QueueClass::Copy
                    : gpu::QueueClass::Direct;
                break;
            case gpu::ExecutionDomain::Present:
                scheduled.queue = gpu::QueueClass::Direct;
                break;
            }

            for (const auto& access : source.accesses)
            {
                const auto state = gpu::RequiredState(access);

                const auto duplicate = std::find_if(
                    scheduled.requiredStates.begin(),
                    scheduled.requiredStates.end(),
                    [&](const RequiredResourceState& value)
                    {
                        return value.resource == access.resource
                            && value.state == state;
                    });

                if (duplicate == scheduled.requiredStates.end())
                {
                    scheduled.requiredStates.push_back({
                        .resource = access.resource,
                        .state = state
                    });
                }
            }

            result.push_back(std::move(scheduled));
        }

        return result;
    }

    std::vector<StateTransition> RenderCompiler::PlanTransitions(
        const std::vector<ScheduledWork>& works)
    {
        std::unordered_map<
            gpu::ResourceId,
            gpu::AbstractState,
            foundation::StrongIdHash<gpu::ResourceTag>> currentStates;

        std::vector<StateTransition> transitions;

        for (std::size_t workIndex = 0;
             workIndex < works.size();
             ++workIndex)
        {
            for (const auto& requirement : works[workIndex].requiredStates)
            {
                const auto found = currentStates.find(requirement.resource);
                const auto current = found == currentStates.end()
                    ? gpu::AbstractState::Undefined
                    : found->second;

                if (current != requirement.state)
                {
                    transitions.push_back({
                        .resource = requirement.resource,
                        .beforeScheduledWork = workIndex,
                        .from = current,
                        .to = requirement.state
                    });

                    currentStates[requirement.resource] = requirement.state;
                }
            }
        }

        return transitions;
    }

    std::vector<QueueSynchronization> RenderCompiler::PlanQueueSynchronization(
        const std::vector<DependencyEdge>& dependencies,
        const std::vector<std::size_t>& schedule,
        const std::vector<ScheduledWork>& works)
    {
        std::vector<std::size_t> scheduledPosition(schedule.size());
        for (std::size_t index = 0; index < schedule.size(); ++index)
        {
            scheduledPosition.at(schedule[index]) = index;
        }

        std::vector<QueueSynchronization> result;
        for (const auto& dependency : dependencies)
        {
            const auto before = scheduledPosition.at(dependency.before);
            const auto after = scheduledPosition.at(dependency.after);
            if (works.at(before).queue != works.at(after).queue)
            {
                result.push_back({
                    .signalScheduledWork = before,
                    .waitScheduledWork = after,
                    .signalQueue = works.at(before).queue,
                    .waitQueue = works.at(after).queue
                });
            }
        }
        return result;
    }
}
