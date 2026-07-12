#include "03_RenderIR/RenderIR.h"

#include "00_Foundation/Hash.h"

#include <stdexcept>

namespace sge::ir
{
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
            HashEnum(seed, resource.kind);
            HashEnum(seed, resource.memoryClass);
            HashEnum(seed, resource.format);
            HashCombine(seed, static_cast<std::size_t>(resource.sizeBytes));
            HashCombine(seed, resource.strideBytes);
            HashCombine(seed, resource.width);
            HashCombine(seed, resource.height);

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

            for (const auto& parameter : program.parameters)
            {
                HashEnum(seed, parameter.kind);
                HashEnum(seed, parameter.stage);
                HashCombine(seed, parameter.registerIndex);
                HashCombine(seed, parameter.registerSpace);
            }
        }

        for (const auto& work : works)
        {
            HashCombine(seed, work.id.Value());
            HashCombine(seed, HashString(work.name));
            HashEnum(seed, work.domain);
            HashCombine(seed, work.program.Value());
            HashCombine(seed, work.vertexResource.Value());
            HashCombine(seed, work.constantResource.Value());
            HashCombine(seed, work.vertexCount);
            HashCombine(seed, work.firstVertex);
            HashEnum(seed, work.rasterState.topology);
            HashEnum(seed, work.rasterState.composition);
            HashEnum(seed, work.rasterState.depth);

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
