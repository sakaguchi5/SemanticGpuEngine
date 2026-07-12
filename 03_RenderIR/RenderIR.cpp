#include "03_RenderIR/RenderIR.h"

#include "00_Foundation/Hash.h"

#include <stdexcept>
#include <type_traits>

namespace sge::ir
{
    gpu::ResourceKind ResourceDeclaration::Kind() const noexcept
    {
        if (std::holds_alternative<BufferDescription>(description))
        {
            return gpu::ResourceKind::Buffer;
        }
        if (const auto* texture = std::get_if<TextureDescription>(&description))
        {
            return texture->dimension;
        }
        return gpu::ResourceKind::Presentation;
    }

    gpu::ResourceFormat ResourceDeclaration::Format() const noexcept
    {
        if (const auto* texture = std::get_if<TextureDescription>(&description))
        {
            return texture->format;
        }
        if (const auto* presentation =
            std::get_if<PresentationDescription>(&description))
        {
            return presentation->format;
        }
        return gpu::ResourceFormat::Unknown;
    }

    std::uint64_t ResourceDeclaration::SizeBytes() const noexcept
    {
        if (const auto* buffer = std::get_if<BufferDescription>(&description))
        {
            return buffer->sizeBytes;
        }
        return 0;
    }

    gpu::ExecutionDomain WorkDeclaration::Domain() const noexcept
    {
        if (std::holds_alternative<RasterWork>(payload))
        {
            return gpu::ExecutionDomain::Raster;
        }
        if (std::holds_alternative<ComputeWork>(payload))
        {
            return gpu::ExecutionDomain::Compute;
        }
        if (std::holds_alternative<CopyWork>(payload))
        {
            return gpu::ExecutionDomain::Copy;
        }
        return gpu::ExecutionDomain::Present;
    }

    const ResourceDeclaration& SemanticModule::Resource(
        gpu::ResourceId id) const
    {
        for (const auto& resource : resources)
        {
            if (resource.id == id)
            {
                return resource;
            }
        }

        throw std::runtime_error("SemanticModule: resource ID not found.");
    }

    const ProgramDeclaration& SemanticModule::Program(
        gpu::ProgramId id) const
    {
        for (const auto& program : programs)
        {
            if (program.id == id)
            {
                return program;
            }
        }

        throw std::runtime_error("SemanticModule: program ID not found.");
    }

    const WorkDeclaration& SemanticModule::Work(
        gpu::WorkId id) const
    {
        for (const auto& work : works)
        {
            if (work.id == id)
            {
                return work;
            }
        }

        throw std::runtime_error("SemanticModule: work ID not found.");
    }

    std::size_t SemanticModule::StructureHash() const
    {
        using namespace foundation;

        std::size_t seed = 0;

        for (const auto& resource : resources)
        {
            HashCombine(seed, resource.id.Value());
            HashCombine(seed, HashString(resource.name));
            HashEnum(seed, resource.memoryClass);
            HashEnum(seed, resource.Kind());
            HashEnum(seed, resource.Format());

            std::visit([&](const auto& description)
            {
                using T = std::decay_t<decltype(description)>;
                if constexpr (std::is_same_v<T, BufferDescription>)
                {
                    HashCombine(seed, static_cast<std::size_t>(description.sizeBytes));
                    HashCombine(seed, description.strideBytes);
                    HashEnum(seed, description.usage);
                }
                else if constexpr (std::is_same_v<T, TextureDescription>)
                {
                    HashEnum(seed, description.dimension);
                    HashEnum(seed, description.format);
                    HashCombine(seed, description.width);
                    HashCombine(seed, description.height);
                    HashCombine(seed, description.depth);
                    HashCombine(seed, description.mipLevels);
                    HashEnum(seed, description.usage);
                }
                else
                {
                    HashEnum(seed, description.format);
                }
            }, resource.description);

            if (resource.memoryClass == gpu::MemoryClass::Static
                && !resource.data.empty())
            {
                HashCombine(
                    seed,
                    HashBytes(resource.data.data(), resource.data.size()));
            }
        }

        for (const auto& program : programs)
        {
            HashCombine(seed, program.id.Value());
            HashCombine(seed, HashString(program.name));
            HashCombine(seed, HashString(program.shaderPath.generic_string()));
            HashCombine(seed, HashString(program.vertexEntry));
            HashCombine(seed, HashString(program.pixelEntry));
            HashCombine(seed, HashString(program.computeEntry));

            for (const auto& parameter : program.parameters)
            {
                HashEnum(seed, parameter.kind);
                HashEnum(seed, parameter.stage);
                HashCombine(seed, HashString(parameter.name));
                HashCombine(seed, parameter.registerIndex);
                HashCombine(seed, parameter.registerSpace);
            }
        }

        for (const auto& work : works)
        {
            HashCombine(seed, work.id.Value());
            HashCombine(seed, HashString(work.name));
            HashEnum(seed, work.Domain());

            std::visit([&](const auto& payload)
            {
                using T = std::decay_t<decltype(payload)>;
                if constexpr (std::is_same_v<T, RasterWork>)
                {
                    HashCombine(seed, payload.program.Value());
                    HashCombine(seed, payload.vertexResource.Value());
                    HashCombine(seed, payload.vertexCount);
                    HashCombine(seed, payload.firstVertex);
                    HashEnum(seed, payload.rasterState.topology);
                    HashEnum(seed, payload.rasterState.composition);
                    HashEnum(seed, payload.rasterState.depth);
                    for (const auto& binding : payload.bindings)
                    {
                        HashCombine(seed, binding.parameterIndex);
                        HashCombine(seed, binding.resource.Value());
                    }
                    for (const auto color : payload.attachments.colors)
                    {
                        HashCombine(seed, color.Value());
                    }
                    HashCombine(seed, payload.attachments.depth.Value());
                }
                else if constexpr (std::is_same_v<T, ComputeWork>)
                {
                    HashCombine(seed, payload.program.Value());
                    HashCombine(seed, payload.groupCountX);
                    HashCombine(seed, payload.groupCountY);
                    HashCombine(seed, payload.groupCountZ);
                    for (const auto& binding : payload.bindings)
                    {
                        HashCombine(seed, binding.parameterIndex);
                        HashCombine(seed, binding.resource.Value());
                    }
                }
                else if constexpr (std::is_same_v<T, CopyWork>)
                {
                    HashCombine(seed, payload.source.Value());
                    HashCombine(seed, payload.destination.Value());
                    HashCombine(seed, static_cast<std::size_t>(payload.sourceOffset));
                    HashCombine(seed, static_cast<std::size_t>(payload.destinationOffset));
                    HashCombine(seed, static_cast<std::size_t>(payload.sizeBytes));
                }
                else
                {
                    HashCombine(seed, payload.source.Value());
                }
            }, work.payload);

            for (const auto& access : work.accesses)
            {
                HashCombine(seed, access.resource.Value());
                HashEnum(seed, access.access);
                HashEnum(seed, access.role);
            }
        }

        return seed;
    }
}
