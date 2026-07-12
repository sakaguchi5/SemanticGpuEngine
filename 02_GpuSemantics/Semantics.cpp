#include "02_GpuSemantics/Semantics.h"

namespace sge::gpu
{
    AbstractState RequiredState(const ResourceAccess& access)
    {
        switch (access.role)
        {
        case ResourceRole::VertexInput:
            return AbstractState::VertexRead;

        case ResourceRole::ConstantInput:
            return AbstractState::ConstantRead;

        case ResourceRole::ProgramInput:
            return AbstractState::ProgramRead;

        case ResourceRole::ProgramOutput:
            return AbstractState::ProgramWrite;

        case ResourceRole::ColorOutput:
            return AbstractState::ColorWrite;

        case ResourceRole::DepthOutput:
            return access.access == AccessMode::Read
                ? AbstractState::DepthRead
                : AbstractState::DepthWrite;

        case ResourceRole::TransferSource:
            return AbstractState::TransferRead;

        case ResourceRole::TransferDestination:
            return AbstractState::TransferWrite;

        case ResourceRole::Presentation:
            return AbstractState::Present;
        }

        return AbstractState::Undefined;
    }

    const char* ToString(AbstractState state) noexcept
    {
        switch (state)
        {
        case AbstractState::Undefined: return "Undefined";
        case AbstractState::VertexRead: return "VertexRead";
        case AbstractState::ConstantRead: return "ConstantRead";
        case AbstractState::ProgramRead: return "ProgramRead";
        case AbstractState::ProgramWrite: return "ProgramWrite";
        case AbstractState::ColorWrite: return "ColorWrite";
        case AbstractState::DepthRead: return "DepthRead";
        case AbstractState::DepthWrite: return "DepthWrite";
        case AbstractState::TransferRead: return "TransferRead";
        case AbstractState::TransferWrite: return "TransferWrite";
        case AbstractState::Present: return "Present";
        }

        return "Unknown";
    }
}
