#include "10_Diagnostics/PlanDiagnostics.h"

#include <fstream>
#include <stdexcept>

namespace sge::diagnostics
{
    void WriteExecutionPlan(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan,
        const std::filesystem::path& outputPath)
    {
        std::ofstream output(outputPath, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(
                "Could not write execution plan diagnostics.");
        }

        output << "Semantic GPU Engine - Execution Plan\n";
        output << "Structure hash: " << plan.structureHash << "\n\n";

        output << "[Scheduled works]\n";
        for (std::size_t index = 0;
             index < plan.scheduledWorks.size();
             ++index)
        {
            const auto& scheduled = plan.scheduledWorks[index];
            const auto& work = module.works.at(scheduled.sourceWorkIndex);

            output << index << ": " << work.name << "\n";
            for (const auto& state : scheduled.requiredStates)
            {
                output << "  resource "
                       << module.Resource(state.resource).name
                       << " -> "
                       << gpu::ToString(state.state)
                       << "\n";
            }
        }

        output << "\n[Dependencies]\n";
        for (const auto& dependency : plan.dependencies)
        {
            output << module.works.at(dependency.before).name
                   << " -> "
                   << module.works.at(dependency.after).name
                   << " ("
                   << module.Resource(dependency.resource).name
                   << ")\n";
        }

        output << "\n[Resource lifetimes]\n";
        for (const auto& lifetime : plan.lifetimes)
        {
            output << module.Resource(lifetime.resource).name
                   << ": ["
                   << lifetime.firstUse
                   << ", "
                   << lifetime.lastUse
                   << "], allocation "
                   << lifetime.allocation.Value()
                   << "\n";
        }

        output << "\n[Abstract transitions]\n";
        for (const auto& transition : plan.transitions)
        {
            output << module.Resource(transition.resource).name
                   << ": "
                   << gpu::ToString(transition.from)
                   << " -> "
                   << gpu::ToString(transition.to)
                   << " before scheduled work "
                   << transition.beforeScheduledWork
                   << "\n";
        }

        output << "\n[Frame-boundary transitions]\n";
        for (const auto& transition : plan.frameBoundaryTransitions)
        {
            output << module.Resource(transition.resource).name
                   << ": "
                   << gpu::ToString(transition.from)
                   << " -> "
                   << gpu::ToString(transition.to)
                   << " after scheduled work "
                   << transition.afterScheduledWork
                   << ", release queue "
                   << static_cast<int>(transition.releaseQueue)
                   << ", next-frame queue "
                   << static_cast<int>(transition.nextFrameQueue)
                   << " requires "
                   << gpu::ToString(transition.nextFrameState)
                   << "\n";
        }

        output << "\n[Normalized executables]\n";
        for (const auto& executable : plan.executables)
        {
            output << "program=" << executable.program.Value()
                   << ", topology="
                   << static_cast<int>(executable.rasterState.topology)
                   << ", composition="
                   << static_cast<int>(executable.rasterState.composition)
                   << ", depth="
                   << static_cast<int>(executable.rasterState.depth)
                   << "\n";
        }
    }

    void WriteDependencyGraphDot(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan,
        const std::filesystem::path& outputPath)
    {
        std::ofstream output(outputPath, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(
                "Could not write dependency graph diagnostics.");
        }

        output << "digraph SemanticGpuWork {\n";
        output << "  rankdir=LR;\n";

        for (std::size_t index = 0; index < module.works.size(); ++index)
        {
            output << "  work" << index
                   << " [label=\""
                   << module.works[index].name
                   << "\"];\n";
        }

        for (const auto& dependency : plan.dependencies)
        {
            output << "  work" << dependency.before
                   << " -> work" << dependency.after
                   << " [label=\""
                   << module.Resource(dependency.resource).name
                   << "\"];\n";
        }

        output << "}\n";
    }
}
