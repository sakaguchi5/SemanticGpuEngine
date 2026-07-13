#include "03_RenderIR/RenderIR.h"

#include "00_Foundation/Hash.h"

#include <bit>
#include <stdexcept>
#include <type_traits>

namespace
{
    void HashTextureRange(
        std::size_t& seed,
        const sge::ir::TextureSubresourceRange& range)
    {
        using namespace sge::foundation;
        HashCombine(seed, range.baseMip);
        HashCombine(seed, range.mipCount);
        HashCombine(seed, range.baseArrayLayer);
        HashCombine(seed, range.arrayLayerCount);
        HashCombine(seed, range.basePlane);
        HashCombine(seed, range.planeCount);
        HashCombine(seed, range.baseDepthSlice);
        HashCombine(seed, range.depthSliceCount);
    }

    void HashView(std::size_t& seed, const sge::ir::ResourceView& view)
    {
        using namespace sge::foundation;
        HashCombine(seed, view.resource.Value());
        HashCombine(seed, static_cast<std::size_t>(view.offsetBytes));
        HashCombine(seed, static_cast<std::size_t>(view.sizeBytes));
        HashCombine(seed, view.strideBytes);
        HashTextureRange(seed, view.textureRange);
        HashEnum(seed, view.formatOverride);
    }

    void HashFloat(std::size_t& seed, float value)
    {
        sge::foundation::HashCombine(
            seed, std::bit_cast<std::uint32_t>(value));
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

    const WorkDeclaration& SemanticModule::Work(gpu::WorkId id) const
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
                    HashCombine(seed, description.arrayLayers);
                    HashCombine(seed, description.sampleCount);
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
            HashCombine(seed, work.id.Value());
            HashCombine(seed, HashString(work.name));
            HashEnum(seed, work.Domain());

            std::visit([&](const auto& payload)
            {
                using T = std::decay_t<decltype(payload)>;
                if constexpr (std::is_same_v<T, RasterWork>)
                {
                    HashCombine(seed, payload.program.Value());
                    HashView(seed, payload.vertexResource);
                    HashCombine(seed, payload.vertexCount);
                    HashCombine(seed, payload.firstVertex);
                    HashEnum(seed, payload.rasterState.topology);
                    HashEnum(seed, payload.rasterState.composition);
                    HashEnum(seed, payload.rasterState.depth);
                    HashEnum(seed, payload.rasterState.cull);
                    HashEnum(seed, payload.rasterState.fill);
                    HashEnum(seed, payload.rasterState.frontFace);
                    HashCombine(seed, payload.rasterState.sampleCount);
                    HashCombine(seed, payload.clear.clearColor);
                    for (const auto component : payload.clear.color)
                    {
                        HashFloat(seed, component);
                    }
                    HashCombine(seed, payload.clear.clearDepth);
                    HashFloat(seed, payload.clear.depth);
                    HashEnum(seed, payload.clear.colorLoad);
                    HashEnum(seed, payload.clear.colorStore);
                    HashEnum(seed, payload.clear.depthLoad);
                    HashEnum(seed, payload.clear.depthStore);

                    for (const auto& binding : payload.bindings)
                    {
                        HashCombine(seed, binding.parameterIndex);
                        HashView(seed, binding.resource);
                        HashCombine(seed, binding.frameLag);
                    }
                    for (const auto& color : payload.attachments.colors)
                    {
                        HashView(seed, color);
                    }
                    HashView(seed, payload.attachments.depth);

                    for (const auto& stream : payload.vertexStreams)
                    {
                        HashCombine(seed, stream.inputSlot);
                        HashView(seed, stream.resource);
                    }
                    HashCombine(seed, payload.indexBinding.has_value());
                    if (payload.indexBinding)
                    {
                        HashView(seed, payload.indexBinding->resource);
                        HashEnum(seed, payload.indexBinding->format);
                        HashCombine(seed, payload.indexBinding->firstIndex);
                        HashCombine(
                            seed,
                            static_cast<std::uint32_t>(
                                payload.indexBinding->baseVertex));
                    }
                    HashCombine(seed, payload.indexCount);
                    HashCombine(seed, payload.instanceCount);
                    HashCombine(seed, payload.firstInstance);
                    HashFloat(seed, payload.viewport.x);
                    HashFloat(seed, payload.viewport.y);
                    HashFloat(seed, payload.viewport.width);
                    HashFloat(seed, payload.viewport.height);
                    HashFloat(seed, payload.viewport.minimumDepth);
                    HashFloat(seed, payload.viewport.maximumDepth);
                    HashCombine(seed, static_cast<std::uint32_t>(payload.scissor.left));
                    HashCombine(seed, static_cast<std::uint32_t>(payload.scissor.top));
                    HashCombine(seed, static_cast<std::uint32_t>(payload.scissor.right));
                    HashCombine(seed, static_cast<std::uint32_t>(payload.scissor.bottom));
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
                        HashView(seed, binding.resource);
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
