#include "02_GpuSemantics/Semantics.h"

namespace sge::gpu
{
    AbstractState RequiredState(const ResourceAccess& access)
    {
        switch (access.role)
        {
        case ResourceRole::VertexInput:
            return AbstractState::VertexRead;
        case ResourceRole::IndexInput:
            return AbstractState::IndexRead;
        case ResourceRole::IndirectInput:
            return AbstractState::IndirectRead;
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
        case AbstractState::IndexRead: return "IndexRead";
        case AbstractState::IndirectRead: return "IndirectRead";
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

    const char* ToString(ResourceFormat format) noexcept
    {
        switch (format)
        {
        case ResourceFormat::Unknown: return "Unknown";
        case ResourceFormat::R8Unorm: return "R8Unorm";
        case ResourceFormat::Rg8Unorm: return "Rg8Unorm";
        case ResourceFormat::Rgba8Unorm: return "Rgba8Unorm";
        case ResourceFormat::Bgra8Unorm: return "Bgra8Unorm";
        case ResourceFormat::R16Float: return "R16Float";
        case ResourceFormat::Rg16Float: return "Rg16Float";
        case ResourceFormat::Rgba16Float: return "Rgba16Float";
        case ResourceFormat::R32Float: return "R32Float";
        case ResourceFormat::Rg32Float: return "Rg32Float";
        case ResourceFormat::Rgba32Float: return "Rgba32Float";
        case ResourceFormat::R32Uint: return "R32Uint";
        case ResourceFormat::Rg32Uint: return "Rg32Uint";
        case ResourceFormat::Rgba32Uint: return "Rgba32Uint";
        case ResourceFormat::Depth24Stencil8: return "Depth24Stencil8";
        case ResourceFormat::Depth32Float: return "Depth32Float";
        }
        return "Unknown";
    }

    bool IsDepthFormat(ResourceFormat format) noexcept
    {
        return format == ResourceFormat::Depth24Stencil8
            || format == ResourceFormat::Depth32Float;
    }

    std::uint32_t BytesPerTexel(ResourceFormat format) noexcept
    {
        switch (format)
        {
        case ResourceFormat::R8Unorm: return 1;
        case ResourceFormat::Rg8Unorm: return 2;
        case ResourceFormat::Rgba8Unorm:
        case ResourceFormat::Bgra8Unorm:
        case ResourceFormat::R32Float:
        case ResourceFormat::R32Uint:
        case ResourceFormat::Depth24Stencil8:
        case ResourceFormat::Depth32Float:
            return 4;
        case ResourceFormat::R16Float: return 2;
        case ResourceFormat::Rg16Float: return 4;
        case ResourceFormat::Rgba16Float:
        case ResourceFormat::Rg32Float:
        case ResourceFormat::Rg32Uint:
            return 8;
        case ResourceFormat::Rgba32Float:
        case ResourceFormat::Rgba32Uint:
            return 16;
        default:
            return 0;
        }
    }
}
