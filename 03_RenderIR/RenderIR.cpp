#include "03_RenderIR/RenderIR.h"

#include "00_Foundation/Hash.h"

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace
{
    struct ResourceInstanceStateSummary
    {
        std::optional<sge::gpu::AbstractState> readState;
        std::size_t readAccessCount = 0;
        std::size_t writeAccessCount = 0;
    };

    std::uint64_t ResourceInstanceKey(
        sge::gpu::ResourceId resource,
        std::uint32_t frameLag) noexcept
    {
        return (static_cast<std::uint64_t>(resource.Value()) << 32u)
            | static_cast<std::uint64_t>(frameLag);
    }

    void ValidateWorkResourceInstanceStates(
        const sge::ir::SemanticModule& module,
        const sge::ir::WorkDeclaration& work)
    {
        std::unordered_map<std::uint64_t, ResourceInstanceStateSummary>
            summaries;

        for (const auto& access : work.accesses)
        {
            auto& summary = summaries[ResourceInstanceKey(
                access.resource, access.frameLag)];
            const auto state = sge::gpu::RequiredState(access);
            const bool writeCapable =
                access.access != sge::gpu::AccessMode::Read;

            if (writeCapable)
            {
                if (summary.readAccessCount != 0)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: work '" + work.name
                        + "' mixes read and write states for the same "
                          "resource instance.");
                }
                if (summary.writeAccessCount != 0)
                {
                    throw std::runtime_error(
                        "Semantic validation failed: work '" + work.name
                        + "' declares multiple write-capable accesses for "
                          "the same resource instance.");
                }
                summary.writeAccessCount = 1;
                continue;
            }

            if (summary.writeAccessCount != 0)
            {
                throw std::runtime_error(
                    "Semantic validation failed: work '" + work.name
                    + "' mixes read and write states for the same "
                      "resource instance.");
            }

            if (summary.readState && *summary.readState != state
                && module.Resource(access.resource).lifetime
                    != sge::gpu::ResourceLifetimeClass::Persistent)
            {
                throw std::runtime_error(
                    "Semantic validation failed: work '" + work.name
                    + "' requires multiple read states for a non-persistent "
                      "resource instance.");
            }

            if (!summary.readState)
            {
                summary.readState = state;
            }
            ++summary.readAccessCount;
        }
    }
}

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

    const std::vector<gpu::ProgramParameter>&
        ProgramDeclaration::BindingParameters() const noexcept
    {
        return programInterface.parameters.empty()
            ? parameters
            : programInterface.parameters;
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
            HashEnum(seed, resource.lifetime);
            HashEnum(seed, resource.update);
            HashEnum(seed, resource.Kind());
            HashEnum(seed, resource.Format());

            std::visit([&](const auto& description)
            {
                using T = std::decay_t<decltype(description)>;
                if constexpr (std::is_same_v<T, BufferDescription>)
                {
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(description.sizeBytes));
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

            if (resource.update == gpu::ResourceUpdateClass::Immutable
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

            for (const auto& parameter : program.BindingParameters())
            {
                HashEnum(seed, parameter.kind);
                HashEnum(seed, parameter.stage);
                HashCombine(seed, HashString(parameter.name));
                HashCombine(seed, parameter.registerIndex);
                HashCombine(seed, parameter.registerSpace);
            }
            for (const auto& input : program.programInterface.vertexInputs)
            {
                HashCombine(seed, HashString(input.semanticName));
                HashCombine(seed, input.semanticIndex);
                HashEnum(seed, input.format);
                HashCombine(seed, input.inputSlot);
                HashCombine(seed, input.alignedByteOffset);
                HashEnum(seed, input.inputRate);
                HashCombine(seed, input.instanceStepRate);
            }
            HashCombine(seed, program.programInterface.colorOutputCount);
            HashCombine(seed, program.programInterface.depthAttachmentAllowed);
        }

        for (const auto& work : works)
        {
            // Invalid work declarations do not have a stable compiler/cache
            // identity. Validate the physical-instance state contract before
            // incorporating the work into the structural key.
            ValidateWorkResourceInstanceStates(*this, work);

            HashCombine(seed, work.id.Value());
            HashCombine(seed, HashString(work.name));
            HashEnum(seed, work.Domain());

            std::visit([&](const auto& payload)
            {
                using T = std::decay_t<decltype(payload)>;
                if constexpr (std::is_same_v<T, RasterWork>)
                {
                    HashCombine(seed, payload.program.Value());
                    HashCombine(seed, payload.vertexResource.resource.Value());
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(
                            payload.vertexResource.offsetBytes));
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(
                            payload.vertexResource.sizeBytes));
                    HashCombine(seed, payload.vertexResource.strideBytes);
                    HashCombine(seed, payload.vertexCount);
                    HashCombine(seed, payload.firstVertex);
                    HashEnum(seed, payload.rasterState.topology);
                    HashEnum(seed, payload.rasterState.composition);
                    HashEnum(seed, payload.rasterState.depth);
                    for (const auto& binding : payload.bindings)
                    {
                        HashCombine(seed, binding.parameterIndex);
                        HashCombine(
                            seed,
                            binding.resource.resource.Value());
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(
                                binding.resource.offsetBytes));
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(
                                binding.resource.sizeBytes));
                        HashCombine(seed, binding.resource.strideBytes);
                        HashCombine(seed, binding.frameLag);
                    }
                    for (const auto& color : payload.attachments.colors)
                    {
                        HashCombine(seed, color.resource.Value());
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(color.offsetBytes));
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(color.sizeBytes));
                        HashCombine(seed, color.strideBytes);
                    }
                    HashCombine(
                        seed,
                        payload.attachments.depth.resource.Value());
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(
                            payload.attachments.depth.offsetBytes));
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(
                            payload.attachments.depth.sizeBytes));
                    HashCombine(
                        seed,
                        payload.attachments.depth.strideBytes);
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
                        HashCombine(
                            seed,
                            binding.resource.resource.Value());
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(
                                binding.resource.offsetBytes));
                        HashCombine(
                            seed,
                            static_cast<std::size_t>(
                                binding.resource.sizeBytes));
                        HashCombine(seed, binding.resource.strideBytes);
                        HashCombine(seed, binding.frameLag);
                    }
                }
                else if constexpr (std::is_same_v<T, CopyWork>)
                {
                    HashCombine(seed, payload.source.Value());
                    HashCombine(seed, payload.destination.Value());
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(payload.sourceOffset));
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(payload.destinationOffset));
                    HashCombine(
                        seed,
                        static_cast<std::size_t>(payload.sizeBytes));
                    HashCombine(seed, payload.sourceFrameLag);
                    HashCombine(seed, payload.destinationFrameLag);
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
                HashCombine(seed, access.frameLag);
            }
        }

        return seed;
    }
}
