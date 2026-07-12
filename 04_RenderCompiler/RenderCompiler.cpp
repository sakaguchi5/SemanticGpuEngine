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
    CompileResult RenderCompiler::Compile(
        const ir::SemanticModule& module,
        const gpu::DeviceCapabilities& capabilities) const
    {
        CompileResult result;
        result.plan.structureHash = module.StructureHash();

        Validate(module, capabilities, result.diagnostics);
        result.plan.dependencies = AnalyzeDependencies(module);

        const auto schedule = Schedule(
            module.works.size(),
            result.plan.dependencies);

        result.plan.lifetimes = AnalyzeLifetimes(module, schedule);
        result.plan.scheduledWorks = BuildScheduledWorks(module, schedule);
        result.plan.transitions = PlanTransitions(result.plan.scheduledWorks);

        for (const auto& scheduled : result.plan.scheduledWorks)
        {
            const auto& source = module.works.at(scheduled.sourceWorkIndex);
            if (source.domain != gpu::ExecutionDomain::Raster)
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

            if (resource.kind == gpu::ResourceKind::Buffer
                && resource.memoryClass != gpu::MemoryClass::DynamicPerFrame
                && resource.memoryClass != gpu::MemoryClass::External
                && resource.sizeBytes == 0)
            {
                throw std::runtime_error(
                    "Semantic validation failed: non-dynamic buffer has zero size.");
            }

            if (resource.memoryClass == gpu::MemoryClass::Static
                && resource.data.size() != resource.sizeBytes)
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

            if (work.domain == gpu::ExecutionDomain::Raster)
            {
                if (!capabilities.rasterExecution)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster execution is unavailable.");
                }

                if (!programIds.contains(work.program.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work references an unknown program.");
                }

                if (!resourceIds.contains(work.vertexResource.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work has no valid vertex resource.");
                }

                if (!resourceIds.contains(work.constantResource.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: raster work has no valid constant resource.");
                }
            }

            if (work.domain == gpu::ExecutionDomain::Compute
                && !capabilities.computeExecution)
            {
                throw std::runtime_error(
                    "Semantic validation failed: compute execution is unavailable.");
            }

            for (const auto& access : work.accesses)
            {
                if (!resourceIds.contains(access.resource.Value()))
                {
                    throw std::runtime_error(
                        "Semantic validation failed: work references an unknown resource.");
                }
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

    std::vector<std::size_t> RenderCompiler::Schedule(
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

    std::vector<ResourceLifetime> RenderCompiler::AnalyzeLifetimes(
        const ir::SemanticModule& module,
        const std::vector<std::size_t>& schedule)
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

            if (resource.memoryClass == gpu::MemoryClass::Transient)
            {
                for (const auto& existing : lifetimes)
                {
                    const auto& existingResource =
                        module.Resource(existing.resource);

                    const bool compatible =
                        existingResource.memoryClass
                            == gpu::MemoryClass::Transient
                        && existingResource.kind == resource.kind
                        && existingResource.format == resource.format
                        && existingResource.sizeBytes >= resource.sizeBytes;

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
        const std::vector<std::size_t>& schedule)
    {
        std::vector<ScheduledWork> result;
        result.reserve(schedule.size());

        for (const auto sourceIndex : schedule)
        {
            const auto& source = module.works.at(sourceIndex);

            ScheduledWork scheduled;
            scheduled.sourceWorkIndex = sourceIndex;
            scheduled.executable = {
                .program = source.program,
                .rasterState = source.rasterState
            };

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
}
