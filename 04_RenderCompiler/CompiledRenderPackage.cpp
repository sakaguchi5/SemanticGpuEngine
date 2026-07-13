#include "04_RenderCompiler/CompiledRenderPackage.h"

#include "00_Foundation/Hash.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <type_traits>

namespace
{
    using namespace sge;

    constexpr std::uint32_t HasFlag(std::uint32_t value, std::uint32_t flag)
    {
        return value & flag;
    }

    bool HasUsage(ir::BufferUsage value, ir::BufferUsage flag)
    {
        return HasFlag(
            static_cast<std::uint32_t>(value),
            static_cast<std::uint32_t>(flag)) != 0;
    }

    bool HasUsage(ir::TextureUsage value, ir::TextureUsage flag)
    {
        return HasFlag(
            static_cast<std::uint32_t>(value),
            static_cast<std::uint32_t>(flag)) != 0;
    }

    std::uint32_t ViewFormatCompatibilityClass(
        gpu::ResourceFormat format) noexcept
    {
        using F = gpu::ResourceFormat;
        switch (format)
        {
        case F::R8Unorm: return 1;
        case F::Rg8Unorm: return 2;
        case F::Rgba8Unorm: return 3;
        case F::Bgra8Unorm: return 4;
        case F::R16Float: return 5;
        case F::Rg16Float: return 6;
        case F::Rgba16Float: return 7;
        case F::R32Float:
        case F::R32Uint: return 8;
        case F::Rg32Float:
        case F::Rg32Uint: return 9;
        case F::Rgba32Float:
        case F::Rgba32Uint: return 10;
        case F::Depth24Stencil8: return 11;
        case F::Depth32Float: return 12;
        default: return 0;
        }
    }

    bool ViewFormatsAreCompatible(
        gpu::ResourceFormat resourceFormat,
        gpu::ResourceFormat viewFormat) noexcept
    {
        if (viewFormat == gpu::ResourceFormat::Unknown
            || viewFormat == resourceFormat)
        {
            return true;
        }
        const auto resourceClass =
            ViewFormatCompatibilityClass(resourceFormat);
        return resourceClass != 0
            && resourceClass == ViewFormatCompatibilityClass(viewFormat);
    }

    void AddDiagnostic(
        std::vector<compiler::Diagnostic>& output,
        compiler::DiagnosticCode code,
        compiler::DiagnosticSeverity severity,
        std::string message,
        compiler::DiagnosticLocation location = {},
        std::vector<std::string> notes = {})
    {
        output.push_back({
            .code = code,
            .severity = severity,
            .message = std::move(message),
            .location = std::move(location),
            .notes = std::move(notes)
        });
    }

    bool HasErrors(const std::vector<compiler::Diagnostic>& diagnostics)
    {
        return std::any_of(
            diagnostics.begin(), diagnostics.end(),
            [](const compiler::Diagnostic& diagnostic)
            {
                return diagnostic.severity
                    == compiler::DiagnosticSeverity::Error;
            });
    }

    const ir::ResourceDeclaration* TryResource(
        const ir::SemanticModule& module,
        gpu::ResourceId id)
    {
        const auto found = std::find_if(
            module.resources.begin(), module.resources.end(),
            [&](const ir::ResourceDeclaration& value)
            {
                return value.id == id;
            });
        return found == module.resources.end() ? nullptr : &*found;
    }

    const ir::ProgramDeclaration* TryProgram(
        const ir::SemanticModule& module,
        gpu::ProgramId id)
    {
        const auto found = std::find_if(
            module.programs.begin(), module.programs.end(),
            [&](const ir::ProgramDeclaration& value)
            {
                return value.id == id;
            });
        return found == module.programs.end() ? nullptr : &*found;
    }

    std::optional<compiler::NormalizedResourceView> NormalizeView(
        const ir::SemanticModule& module,
        const ir::ResourceView& view,
        std::vector<compiler::Diagnostic>& diagnostics,
        std::optional<gpu::WorkId> work = {})
    {
        const auto* resource = TryResource(module, view.resource);
        if (resource == nullptr)
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidView,
                compiler::DiagnosticSeverity::Error,
                "ResourceView references an unknown resource.",
                {.resource = view.resource, .work = work, .view = view});
            return std::nullopt;
        }

        compiler::NormalizedResourceView result;
        result.resource = view.resource;
        result.kind = resource->Kind();
        result.format = view.formatOverride == gpu::ResourceFormat::Unknown
            ? resource->Format()
            : view.formatOverride;

        if (resource->Kind() == gpu::ResourceKind::Buffer)
        {
            if (view.IsTextureRange())
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidView,
                    compiler::DiagnosticSeverity::Error,
                    "A buffer view cannot contain a texture subresource range.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }

            const auto& buffer = std::get<ir::BufferDescription>(
                resource->description);
            if (view.offsetBytes > buffer.sizeBytes)
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidView,
                    compiler::DiagnosticSeverity::Error,
                    "Buffer view offset exceeds the resource size.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }
            const auto size = view.sizeBytes == 0
                ? buffer.sizeBytes - view.offsetBytes
                : view.sizeBytes;
            if (size == 0 || size > buffer.sizeBytes - view.offsetBytes)
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidView,
                    compiler::DiagnosticSeverity::Error,
                    "Buffer view size exceeds the resource range.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }
            const auto stride = view.strideBytes == 0
                ? buffer.strideBytes
                : view.strideBytes;
            const auto alignment = stride == 0 ? 4u : stride;
            if ((view.offsetBytes % alignment) != 0 || (size % alignment) != 0)
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidView,
                    compiler::DiagnosticSeverity::Error,
                    "Buffer view range is not aligned to its element size.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }
            result.byteOffset = view.offsetBytes;
            result.byteSize = size;
            result.strideBytes = stride;
            return result;
        }

        if (resource->Kind() == gpu::ResourceKind::Presentation)
        {
            if (!view.IsWholeResource())
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidView,
                    compiler::DiagnosticSeverity::Error,
                    "Presentation resources only support a whole-resource view.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }
            return result;
        }

        if (view.IsBufferRange())
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidView,
                compiler::DiagnosticSeverity::Error,
                "A texture view cannot contain a byte range.",
                {.resource = view.resource, .work = work, .view = view});
            return std::nullopt;
        }

        const auto& texture = std::get<ir::TextureDescription>(
            resource->description);
        const auto invalidRange = [&]()
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidSubresourceRange,
                compiler::DiagnosticSeverity::Error,
                "Texture view subresource range exceeds the texture declaration.",
                {.resource = view.resource, .work = work, .view = view});
        };

        if (view.textureRange.baseMip >= texture.mipLevels
            || view.textureRange.baseArrayLayer >= texture.arrayLayers
            || view.textureRange.basePlane >= 1)
        {
            invalidRange();
            return std::nullopt;
        }
        const auto mipCount = view.textureRange.mipCount == 0
            ? static_cast<std::uint32_t>(texture.mipLevels)
                - view.textureRange.baseMip
            : view.textureRange.mipCount;
        const auto layerCount = view.textureRange.arrayLayerCount == 0
            ? static_cast<std::uint32_t>(texture.arrayLayers)
                - view.textureRange.baseArrayLayer
            : view.textureRange.arrayLayerCount;
        const auto planeCount = view.textureRange.planeCount == 0
            ? 1u - view.textureRange.basePlane
            : view.textureRange.planeCount;
        if (mipCount == 0
            || view.textureRange.baseMip + mipCount > texture.mipLevels
            || layerCount == 0
            || view.textureRange.baseArrayLayer + layerCount
                > texture.arrayLayers
            || planeCount == 0
            || view.textureRange.basePlane + planeCount > 1)
        {
            invalidRange();
            return std::nullopt;
        }

        std::uint32_t depthSliceCount = 1;
        if (texture.dimension == gpu::ResourceKind::Texture3D)
        {
            const auto mipDepth = std::max(
                1u,
                static_cast<std::uint32_t>(texture.depth)
                    >> view.textureRange.baseMip);
            if (view.textureRange.baseDepthSlice >= mipDepth)
            {
                invalidRange();
                return std::nullopt;
            }
            depthSliceCount = view.textureRange.depthSliceCount == 0
                ? mipDepth - view.textureRange.baseDepthSlice
                : view.textureRange.depthSliceCount;
            if (depthSliceCount == 0
                || view.textureRange.baseDepthSlice + depthSliceCount
                    > mipDepth
                || (mipCount != 1
                    && (view.textureRange.baseDepthSlice != 0
                        || view.textureRange.depthSliceCount != 0)))
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::InvalidSubresourceRange,
                    compiler::DiagnosticSeverity::Error,
                    "A partial Texture3D depth-slice range selects exactly one mip level.",
                    {.resource = view.resource, .work = work, .view = view});
                return std::nullopt;
            }
        }
        else if (view.textureRange.baseDepthSlice != 0
            || (view.textureRange.depthSliceCount != 0
                && view.textureRange.depthSliceCount != 1))
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidSubresourceRange,
                compiler::DiagnosticSeverity::Error,
                "Only Texture3D views can select depth slices.",
                {.resource = view.resource, .work = work, .view = view});
            return std::nullopt;
        }

        if (!ViewFormatsAreCompatible(
                texture.format, view.formatOverride))
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidView,
                compiler::DiagnosticSeverity::Error,
                "Texture view format reinterpretation is outside the resource's typeless compatibility class.",
                {.resource = view.resource, .work = work, .view = view});
            return std::nullopt;
        }

        result.textureRange = {
            .firstMip = view.textureRange.baseMip,
            .mipCount = static_cast<std::uint16_t>(mipCount),
            .firstArrayLayer = view.textureRange.baseArrayLayer,
            .arrayLayerCount = static_cast<std::uint16_t>(layerCount),
            .firstPlane = view.textureRange.basePlane,
            .planeCount = static_cast<std::uint8_t>(planeCount),
            .firstDepthSlice = view.textureRange.baseDepthSlice,
            .depthSliceCount = static_cast<std::uint16_t>(depthSliceCount)
        };
        return result;
    }

    bool RangesOverlap(
        std::uint32_t first,
        std::uint32_t firstCount,
        std::uint32_t second,
        std::uint32_t secondCount)
    {
        return first < second + secondCount
            && second < first + firstCount;
    }

    bool ViewsOverlap(
        const compiler::NormalizedResourceView& left,
        const compiler::NormalizedResourceView& right)
    {
        if (left.resource != right.resource)
        {
            return false;
        }
        if (left.kind == gpu::ResourceKind::Buffer)
        {
            // D3D12 state tracking has no byte-range granularity for buffers.
            // Buffer views refine descriptors and IA ranges, but all views of
            // one physical buffer instance share one resource state.
            return true;
        }
        if (left.kind == gpu::ResourceKind::Presentation)
        {
            return true;
        }
        return RangesOverlap(
                left.textureRange.firstMip,
                left.textureRange.mipCount,
                right.textureRange.firstMip,
                right.textureRange.mipCount)
            && RangesOverlap(
                left.textureRange.firstArrayLayer,
                left.textureRange.arrayLayerCount,
                right.textureRange.firstArrayLayer,
                right.textureRange.arrayLayerCount)
            && RangesOverlap(
                left.textureRange.firstPlane,
                left.textureRange.planeCount,
                right.textureRange.firstPlane,
                right.textureRange.planeCount);
    }

    gpu::ResourceRole RoleForParameter(gpu::ProgramParameterKind kind)
    {
        switch (kind)
        {
        case gpu::ProgramParameterKind::ConstantBuffer:
            return gpu::ResourceRole::ConstantInput;
        case gpu::ProgramParameterKind::ShaderResource:
            return gpu::ResourceRole::ProgramInput;
        case gpu::ProgramParameterKind::UnorderedAccess:
            return gpu::ResourceRole::ProgramOutput;
        case gpu::ProgramParameterKind::Sampler:
            break;
        }
        return gpu::ResourceRole::ProgramInput;
    }

    struct ViewUse
    {
        compiler::NormalizedResourceView view;
        gpu::ResourceRole role = gpu::ResourceRole::ProgramInput;
        gpu::AccessMode access = gpu::AccessMode::Read;
        std::uint32_t frameLag = 0;
    };

    void ValidateUsageForRole(
        const ir::ResourceDeclaration& resource,
        gpu::ResourceRole role,
        std::vector<compiler::Diagnostic>& diagnostics,
        gpu::WorkId work,
        const ir::ResourceView& sourceView)
    {
        const auto* buffer = std::get_if<ir::BufferDescription>(
            &resource.description);
        const auto* texture = std::get_if<ir::TextureDescription>(
            &resource.description);
        bool valid = true;
        switch (role)
        {
        case gpu::ResourceRole::VertexInput:
            valid = buffer != nullptr
                && HasUsage(buffer->usage, ir::BufferUsage::Vertex);
            break;
        case gpu::ResourceRole::IndexInput:
            valid = buffer != nullptr
                && HasUsage(buffer->usage, ir::BufferUsage::Index);
            break;
        case gpu::ResourceRole::IndirectInput:
            valid = buffer != nullptr
                && HasUsage(buffer->usage, ir::BufferUsage::Indirect);
            break;
        case gpu::ResourceRole::ConstantInput:
            valid = buffer != nullptr
                && HasUsage(buffer->usage, ir::BufferUsage::Constant);
            break;
        case gpu::ResourceRole::ProgramInput:
            valid = (buffer != nullptr
                    && HasUsage(buffer->usage, ir::BufferUsage::Storage))
                || (texture != nullptr
                    && HasUsage(texture->usage, ir::TextureUsage::Sampled));
            break;
        case gpu::ResourceRole::ProgramOutput:
            valid = (buffer != nullptr
                    && HasUsage(buffer->usage, ir::BufferUsage::Storage))
                || (texture != nullptr
                    && HasUsage(texture->usage, ir::TextureUsage::Storage));
            break;
        case gpu::ResourceRole::ColorOutput:
            valid = resource.Kind() == gpu::ResourceKind::Presentation
                || (texture != nullptr
                    && HasUsage(
                        texture->usage,
                        ir::TextureUsage::ColorAttachment));
            break;
        case gpu::ResourceRole::DepthOutput:
            valid = texture != nullptr
                && HasUsage(texture->usage, ir::TextureUsage::DepthAttachment);
            break;
        case gpu::ResourceRole::TransferSource:
            valid = buffer != nullptr
                ? HasUsage(buffer->usage, ir::BufferUsage::CopySource)
                : texture != nullptr
                    && HasUsage(texture->usage, ir::TextureUsage::CopySource);
            break;
        case gpu::ResourceRole::TransferDestination:
            valid = buffer != nullptr
                ? HasUsage(buffer->usage, ir::BufferUsage::CopyDestination)
                : texture != nullptr
                    && HasUsage(
                        texture->usage,
                        ir::TextureUsage::CopyDestination);
            break;
        case gpu::ResourceRole::Presentation:
            valid = resource.Kind() == gpu::ResourceKind::Presentation;
            break;
        }

        if (!valid)
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::InvalidView,
                compiler::DiagnosticSeverity::Error,
                "Resource usage flags do not satisfy the requested role.",
                {.resource = resource.id, .work = work, .view = sourceView});
        }
    }

    std::vector<ViewUse> CollectViewUses(
        const ir::SemanticModule& module,
        const ir::WorkDeclaration& work,
        std::vector<compiler::Diagnostic>& diagnostics)
    {
        std::vector<ViewUse> result;
        const auto add = [&](const ir::ResourceView& view,
                             gpu::ResourceRole role,
                             gpu::AccessMode access,
                             std::uint32_t frameLag)
        {
            const auto normalized = NormalizeView(
                module, view, diagnostics, work.id);
            if (!normalized)
            {
                return;
            }
            if (const auto* resource = TryResource(module, view.resource))
            {
                ValidateUsageForRole(
                    *resource, role, diagnostics, work.id, view);
            }
            result.push_back({*normalized, role, access, frameLag});
        };

        std::visit([&](const auto& payload)
        {
            using T = std::decay_t<decltype(payload)>;
            if constexpr (std::is_same_v<T, ir::RasterWork>)
            {
                if (payload.vertexStreams.empty())
                {
                    add(payload.vertexResource,
                        gpu::ResourceRole::VertexInput,
                        gpu::AccessMode::Read, 0);
                }
                else
                {
                    for (const auto& stream : payload.vertexStreams)
                    {
                        add(stream.resource,
                            gpu::ResourceRole::VertexInput,
                            gpu::AccessMode::Read, 0);
                    }
                }
                if (payload.indexBinding)
                {
                    add(payload.indexBinding->resource,
                        gpu::ResourceRole::IndexInput,
                        gpu::AccessMode::Read, 0);
                }
                const auto* program = TryProgram(module, payload.program);
                if (program != nullptr)
                {
                    const auto& parameters = program->BindingParameters();
                    for (const auto& binding : payload.bindings)
                    {
                        if (binding.parameterIndex >= parameters.size())
                        {
                            AddDiagnostic(
                                diagnostics,
                                compiler::DiagnosticCode::InvalidProgram,
                                compiler::DiagnosticSeverity::Error,
                                "Raster binding parameter index is out of range.",
                                {.program = payload.program, .work = work.id,
                                    .view = binding.resource});
                            continue;
                        }
                        const auto kind = parameters[binding.parameterIndex].kind;
                        if (kind != gpu::ProgramParameterKind::Sampler)
                        {
                            const auto access = kind
                                    == gpu::ProgramParameterKind::UnorderedAccess
                                ? gpu::AccessMode::Write
                                : gpu::AccessMode::Read;
                            add(binding.resource, RoleForParameter(kind),
                                access, binding.frameLag);
                        }
                    }
                }
                for (const auto& color : payload.attachments.colors)
                {
                    add(color, gpu::ResourceRole::ColorOutput,
                        gpu::AccessMode::Write, 0);
                }
                if (payload.attachments.depth.IsValid())
                {
                    add(payload.attachments.depth,
                        gpu::ResourceRole::DepthOutput,
                        payload.rasterState.depth == gpu::DepthMode::ReadOnly
                            ? gpu::AccessMode::Read
                            : gpu::AccessMode::Write,
                        0);
                }
            }
            else if constexpr (std::is_same_v<T, ir::ComputeWork>)
            {
                const auto* program = TryProgram(module, payload.program);
                if (program != nullptr)
                {
                    const auto& parameters = program->BindingParameters();
                    for (const auto& binding : payload.bindings)
                    {
                        if (binding.parameterIndex >= parameters.size())
                        {
                            AddDiagnostic(
                                diagnostics,
                                compiler::DiagnosticCode::InvalidProgram,
                                compiler::DiagnosticSeverity::Error,
                                "Compute binding parameter index is out of range.",
                                {.program = payload.program, .work = work.id,
                                    .view = binding.resource});
                            continue;
                        }
                        const auto kind = parameters[binding.parameterIndex].kind;
                        if (kind != gpu::ProgramParameterKind::Sampler)
                        {
                            const auto access = kind
                                    == gpu::ProgramParameterKind::UnorderedAccess
                                ? gpu::AccessMode::Write
                                : gpu::AccessMode::Read;
                            add(binding.resource, RoleForParameter(kind),
                                access, binding.frameLag);
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<T, ir::CopyWork>)
            {
                const auto* source = TryResource(module, payload.source);
                const auto* destination = TryResource(module, payload.destination);
                const auto makeCopyView = [&](const ir::ResourceDeclaration* resource,
                                              gpu::ResourceId id,
                                              std::uint64_t offset)
                {
                    if (resource == nullptr
                        || resource->Kind() != gpu::ResourceKind::Buffer)
                    {
                        return ir::ResourceView{id};
                    }
                    const auto size = payload.sizeBytes == 0
                        ? resource->SizeBytes() - offset
                        : payload.sizeBytes;
                    return ir::ResourceView{id, offset, size};
                };
                add(makeCopyView(source, payload.source, payload.sourceOffset),
                    gpu::ResourceRole::TransferSource,
                    gpu::AccessMode::Read,
                    payload.sourceFrameLag);
                add(makeCopyView(
                        destination,
                        payload.destination,
                        payload.destinationOffset),
                    gpu::ResourceRole::TransferDestination,
                    gpu::AccessMode::Write,
                    payload.destinationFrameLag);
            }
            else
            {
                add(payload.source,
                    gpu::ResourceRole::Presentation,
                    gpu::AccessMode::Read, 0);
            }
        }, work.payload);
        return result;
    }

    void ValidateAccessContract(
        const ir::WorkDeclaration& work,
        const std::vector<ViewUse>& uses,
        std::vector<compiler::Diagnostic>& diagnostics)
    {
        std::vector<bool> consumed(work.accesses.size(), false);
        for (const auto& use : uses)
        {
            const auto found = std::find_if(
                work.accesses.begin(), work.accesses.end(),
                [&](const gpu::ResourceAccess& access)
                {
                    const auto index = static_cast<std::size_t>(
                        &access - work.accesses.data());
                    const bool accessCompatible = use.access
                            == gpu::AccessMode::Write
                        ? access.access != gpu::AccessMode::Read
                        : access.access == gpu::AccessMode::Read;
                    return !consumed[index]
                        && access.resource == use.view.resource
                        && access.role == use.role
                        && access.frameLag == use.frameLag
                        && accessCompatible;
                });
            if (found == work.accesses.end())
            {
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::PayloadAccessMismatch,
                    compiler::DiagnosticSeverity::Error,
                    "Work payload has no matching ResourceAccess declaration.",
                    {.resource = use.view.resource, .work = work.id});
                continue;
            }
            consumed[static_cast<std::size_t>(
                found - work.accesses.begin())] = true;
        }
        if (std::find(consumed.begin(), consumed.end(), false) != consumed.end())
        {
            AddDiagnostic(
                diagnostics,
                compiler::DiagnosticCode::PayloadAccessMismatch,
                compiler::DiagnosticSeverity::Error,
                "Work declares a ResourceAccess not represented by its payload.",
                {.work = work.id});
        }
    }

    void ValidateViewStateConflicts(
        const ir::SemanticModule& module,
        const ir::WorkDeclaration& work,
        const std::vector<ViewUse>& uses,
        std::vector<compiler::Diagnostic>& diagnostics)
    {
        for (std::size_t left = 0; left < uses.size(); ++left)
        {
            for (std::size_t right = left + 1; right < uses.size(); ++right)
            {
                const auto& first = uses[left];
                const auto& second = uses[right];
                if (first.frameLag != second.frameLag
                    || !ViewsOverlap(first.view, second.view))
                {
                    continue;
                }
                const bool firstWrites = first.access != gpu::AccessMode::Read;
                const bool secondWrites = second.access != gpu::AccessMode::Read;
                if (firstWrites || secondWrites)
                {
                    AddDiagnostic(
                        diagnostics,
                        compiler::DiagnosticCode::ResourceStateConflict,
                        compiler::DiagnosticSeverity::Error,
                        "Overlapping views of one physical resource instance mix read/write or multiple writes.",
                        {.resource = first.view.resource, .work = work.id},
                        {"Split the operation into separate works, or use non-overlapping subresource ranges."});
                    continue;
                }

                const auto firstState = gpu::RequiredState({
                    first.view.resource, first.access, first.role,
                    first.frameLag});
                const auto secondState = gpu::RequiredState({
                    second.view.resource, second.access, second.role,
                    second.frameLag});
                const auto* resource = TryResource(module, first.view.resource);
                if (firstState != secondState && resource != nullptr
                    && resource->lifetime
                        != gpu::ResourceLifetimeClass::Persistent)
                {
                    AddDiagnostic(
                        diagnostics,
                        compiler::DiagnosticCode::ResourceStateConflict,
                        compiler::DiagnosticSeverity::Error,
                        "Overlapping non-persistent views require incompatible read states.",
                        {.resource = first.view.resource, .work = work.id});
                }
            }
        }
    }

    ir::SemanticModule Canonicalize(
        const ir::SemanticModule& source,
        std::vector<compiler::Diagnostic>& diagnostics)
    {
        ir::SemanticModule result = source;
        for (auto& program : result.programs)
        {
            if (!program.parameters.empty()
                && program.programInterface.parameters.empty())
            {
                program.programInterface.parameters = program.parameters;
                program.parameters.clear();
                AddDiagnostic(
                    diagnostics,
                    compiler::DiagnosticCode::LegacyProgramInterface,
                    compiler::DiagnosticSeverity::Warning,
                    "Legacy ProgramDeclaration::parameters was canonicalized into ProgramInterface.",
                    {.program = program.id});
            }
        }
        return result;
    }

    ir::ResourceView LegacyView(const ir::ResourceView& view)
    {
        if (view.IsTextureRange())
        {
            return ir::ResourceView{view.resource};
        }
        return ir::ResourceView{
            view.resource,
            view.offsetBytes,
            view.sizeBytes,
            view.strideBytes};
    }

    ir::SemanticModule BuildLegacyModule(
        const ir::SemanticModule& canonical,
        bool& legacyExecutable)
    {
        ir::SemanticModule result = canonical;
        for (auto& work : result.works)
        {
            if (auto* raster = std::get_if<ir::RasterWork>(&work.payload))
            {
                for (auto& binding : raster->bindings)
                {
                    if (binding.resource.IsTextureRange())
                    {
                        legacyExecutable = false;
                    }
                    binding.resource = LegacyView(binding.resource);
                }
                for (auto& color : raster->attachments.colors)
                {
                    if (color.IsTextureRange())
                    {
                        legacyExecutable = false;
                    }
                    color = LegacyView(color);
                }
                if (raster->attachments.depth.IsTextureRange())
                {
                    legacyExecutable = false;
                }
                raster->attachments.depth = LegacyView(
                    raster->attachments.depth);

                if (!raster->vertexStreams.empty())
                {
                    legacyExecutable = legacyExecutable
                        && raster->vertexStreams.size() == 1
                        && raster->vertexStreams.front().inputSlot == 0;
                    const auto slotZero = std::find_if(
                        raster->vertexStreams.begin(),
                        raster->vertexStreams.end(),
                        [](const ir::VertexStreamBinding& stream)
                        {
                            return stream.inputSlot == 0;
                        });
                    if (slotZero != raster->vertexStreams.end())
                    {
                        raster->vertexResource = LegacyView(slotZero->resource);
                    }
                    raster->vertexStreams.clear();
                }
                if (raster->indexBinding)
                {
                    legacyExecutable = false;
                    // The core compiler still requires a non-indexed draw
                    // shape. Preserve a harmless representative count only for
                    // its scheduling/lifetime passes; package-native execution
                    // uses the compiled index command below.
                    raster->vertexCount = std::max(
                        raster->vertexCount, raster->indexCount);
                    raster->indexBinding.reset();
                    raster->indexCount = 0;
                }
                if (raster->instanceCount != 1 || raster->firstInstance != 0)
                {
                    legacyExecutable = false;
                    raster->instanceCount = 1;
                    raster->firstInstance = 0;
                }

                const auto& parameters = result.Program(
                    raster->program).BindingParameters();
                std::vector<gpu::ResourceAccess> legacyAccesses;
                legacyAccesses.push_back({
                    raster->vertexResource.resource,
                    gpu::AccessMode::Read,
                    gpu::ResourceRole::VertexInput,
                    0});
                for (const auto& binding : raster->bindings)
                {
                    if (binding.parameterIndex >= parameters.size())
                    {
                        continue;
                    }
                    const auto kind = parameters[binding.parameterIndex].kind;
                    if (kind == gpu::ProgramParameterKind::Sampler)
                    {
                        continue;
                    }
                    legacyAccesses.push_back({
                        binding.resource.resource,
                        kind == gpu::ProgramParameterKind::UnorderedAccess
                            ? gpu::AccessMode::Write
                            : gpu::AccessMode::Read,
                        RoleForParameter(kind),
                        binding.frameLag});
                }
                for (const auto& color : raster->attachments.colors)
                {
                    legacyAccesses.push_back({
                        color.resource,
                        gpu::AccessMode::Write,
                        gpu::ResourceRole::ColorOutput,
                        0});
                }
                if (raster->attachments.depth.IsValid())
                {
                    legacyAccesses.push_back({
                        raster->attachments.depth.resource,
                        raster->rasterState.depth == gpu::DepthMode::ReadOnly
                            ? gpu::AccessMode::Read
                            : gpu::AccessMode::Write,
                        gpu::ResourceRole::DepthOutput,
                        0});
                }
                work.accesses = std::move(legacyAccesses);
            }
            else if (auto* compute = std::get_if<ir::ComputeWork>(
                &work.payload))
            {
                for (auto& binding : compute->bindings)
                {
                    if (binding.resource.IsTextureRange())
                    {
                        legacyExecutable = false;
                    }
                    binding.resource = LegacyView(binding.resource);
                }
            }
        }

        // The legacy core rejects one immutable Persistent resource shared by
        // Copy and non-Copy queues because its old fixed read envelope cannot
        // express release/acquire ownership. The package layer does express
        // that handoff. Mark only the compatibility snapshot FrameLocal so the
        // proven dependency/scheduling passes can run; the package resource
        // blueprint restores the canonical Persistent instance plan below.
        for (auto& resource : result.resources)
        {
            if (resource.lifetime
                != gpu::ResourceLifetimeClass::Persistent)
            {
                continue;
            }
            bool copyUse = false;
            bool nonCopyUse = false;
            for (const auto& work : result.works)
            {
                const bool uses = std::any_of(
                    work.accesses.begin(), work.accesses.end(),
                    [&](const gpu::ResourceAccess& access)
                    {
                        return access.resource == resource.id;
                    });
                if (!uses)
                {
                    continue;
                }
                copyUse = copyUse
                    || work.Domain() == gpu::ExecutionDomain::Copy;
                nonCopyUse = nonCopyUse
                    || work.Domain() != gpu::ExecutionDomain::Copy;
            }
            if (copyUse && nonCopyUse)
            {
                resource.lifetime = gpu::ResourceLifetimeClass::FrameLocal;
                legacyExecutable = false;
            }
        }

        // The core compiler presently enforces one vertex stream. Preserve
        // one representative stream in the compatibility module even when the
        // package-native declaration starts at a non-zero input slot.
        for (auto& program : result.programs)
        {
            const auto oldSize = program.programInterface.vertexInputs.size();
            const auto hasSlotZero = std::any_of(
                program.programInterface.vertexInputs.begin(),
                program.programInterface.vertexInputs.end(),
                [](const ir::VertexInputElement& input)
                {
                    return input.inputSlot == 0;
                });
            std::uint32_t selectedSlot = 0;
            if (!hasSlotZero && !program.programInterface.vertexInputs.empty())
            {
                selectedSlot = std::min_element(
                    program.programInterface.vertexInputs.begin(),
                    program.programInterface.vertexInputs.end(),
                    [](const ir::VertexInputElement& left,
                       const ir::VertexInputElement& right)
                    {
                        return left.inputSlot < right.inputSlot;
                    })->inputSlot;
            }
            std::erase_if(
                program.programInterface.vertexInputs,
                [&](const ir::VertexInputElement& input)
                {
                    return input.inputSlot != selectedSlot;
                });
            for (auto& input : program.programInterface.vertexInputs)
            {
                input.inputSlot = 0;
            }
            if (oldSize != program.programInterface.vertexInputs.size()
                || selectedSlot != 0)
            {
                legacyExecutable = false;
            }
        }
        return result;
    }

    std::uint64_t EstimateResourceBytes(
        const ir::ResourceDeclaration& resource,
        std::uint32_t instances)
    {
        if (resource.Kind() == gpu::ResourceKind::Buffer)
        {
            return resource.SizeBytes() * instances;
        }
        if (resource.Kind() == gpu::ResourceKind::Presentation)
        {
            return 0;
        }
        const auto& texture = std::get<ir::TextureDescription>(
            resource.description);
        std::uint64_t totalTexels = 0;
        std::uint64_t width = std::max(1u, texture.width);
        std::uint64_t height = std::max(1u, texture.height);
        std::uint64_t depth = std::max<std::uint16_t>(1, texture.depth);
        for (std::uint16_t mip = 0; mip < texture.mipLevels; ++mip)
        {
            totalTexels += width * height * depth * texture.arrayLayers;
            width = std::max<std::uint64_t>(1, width / 2);
            height = std::max<std::uint64_t>(1, height / 2);
            depth = std::max<std::uint64_t>(1, depth / 2);
        }
        return totalTexels
            * gpu::BytesPerTexel(texture.format)
            * texture.sampleCount
            * instances;
    }

    compiler::ExecutableKey RasterExecutable(
        const ir::SemanticModule& module,
        const ir::RasterWork& raster)
    {
        compiler::ExecutableKey key;
        key.program = raster.program;
        key.rasterState = raster.rasterState;
        key.vertexInputs = module.Program(
            raster.program).programInterface.vertexInputs;
        for (const auto& color : raster.attachments.colors)
        {
            const auto& resource = module.Resource(color.resource);
            key.colorFormats.push_back(
                color.formatOverride == gpu::ResourceFormat::Unknown
                    ? resource.Format()
                    : color.formatOverride);
        }
        if (raster.attachments.depth.IsValid())
        {
            const auto& resource = module.Resource(
                raster.attachments.depth.resource);
            key.depthFormat =
                raster.attachments.depth.formatOverride
                    == gpu::ResourceFormat::Unknown
                ? resource.Format()
                : raster.attachments.depth.formatOverride;
        }
        return key;
    }

    std::vector<compiler::CompiledBinding> CompileBindings(
        const ir::SemanticModule& module,
        gpu::ProgramId programId,
        const std::vector<ir::ResourceBinding>& bindings,
        std::vector<compiler::Diagnostic>& diagnostics,
        gpu::WorkId work)
    {
        std::vector<compiler::CompiledBinding> result;
        const auto& parameters = module.Program(programId).BindingParameters();
        for (const auto& binding : bindings)
        {
            if (binding.parameterIndex >= parameters.size())
            {
                continue;
            }
            const auto view = NormalizeView(
                module, binding.resource, diagnostics, work);
            if (!view)
            {
                continue;
            }
            result.push_back({
                .parameterIndex = binding.parameterIndex,
                .kind = parameters[binding.parameterIndex].kind,
                .view = *view,
                .frameLag = binding.frameLag
            });
        }
        return result;
    }
}

namespace sge::compiler
{
    bool PackageCompileResult::Succeeded() const noexcept
    {
        return !HasErrors(diagnostics);
    }

    RenderPackageCompiler::RenderPackageCompiler() = default;

    RenderPackageCompiler::RenderPackageCompiler(
        std::shared_ptr<const ISchedulingPolicy> policy)
        : coreCompiler_(std::move(policy))
    {
    }

    std::vector<Diagnostic> RenderPackageCompiler::Validate(
        const ir::SemanticModule& module,
        const gpu::DeviceCapabilities& capabilities)
    {
        std::vector<Diagnostic> diagnostics;
        std::unordered_set<std::uint32_t> resourceIds;
        std::unordered_set<std::uint32_t> programIds;
        std::unordered_set<std::uint32_t> workIds;

        for (const auto& resource : module.resources)
        {
            if (!resource.id.IsValid()
                || !resourceIds.insert(resource.id.Value()).second)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidResource,
                    DiagnosticSeverity::Error,
                    "Resource ID is invalid or duplicated.",
                    {.resource = resource.id});
                continue;
            }
            if (resource.Kind() == gpu::ResourceKind::Buffer
                && resource.update != gpu::ResourceUpdateClass::CpuUpdated
                && resource.lifetime != gpu::ResourceLifetimeClass::External
                && resource.SizeBytes() == 0)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidResource,
                    DiagnosticSeverity::Error,
                    "A materialized buffer has zero size.",
                    {.resource = resource.id});
            }
            if (resource.update == gpu::ResourceUpdateClass::Immutable
                && resource.data.size() != resource.SizeBytes())
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidResource,
                    DiagnosticSeverity::Error,
                    "Immutable resource data size does not match its declaration.",
                    {.resource = resource.id});
            }
            if (resource.lifetime == gpu::ResourceLifetimeClass::External
                && resource.update != gpu::ResourceUpdateClass::Imported)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidResource,
                    DiagnosticSeverity::Error,
                    "External resources must be imported.",
                    {.resource = resource.id});
            }
            if (resource.lifetime == gpu::ResourceLifetimeClass::Persistent
                && resource.update != gpu::ResourceUpdateClass::Immutable)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidResource,
                    DiagnosticSeverity::Error,
                    "Persistent resources are immutable in V1.",
                    {.resource = resource.id});
            }
            if (resource.update == gpu::ResourceUpdateClass::CpuUpdated)
            {
                const auto* buffer = std::get_if<ir::BufferDescription>(
                    &resource.description);
                const bool constantOnly = buffer != nullptr
                    && HasUsage(buffer->usage, ir::BufferUsage::Constant)
                    && (static_cast<std::uint32_t>(buffer->usage)
                        & ~static_cast<std::uint32_t>(
                            ir::BufferUsage::Constant)) == 0;
                if (!constantOnly || resource.SizeBytes() == 0
                    || resource.data.size() != resource.SizeBytes())
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "CpuUpdated V1 resources are fully initialized constant buffers.",
                        {.resource = resource.id});
                }
            }
            if (const auto* presentation =
                    std::get_if<ir::PresentationDescription>(
                        &resource.description))
            {
                if (presentation->format
                    != gpu::ResourceFormat::Rgba8Unorm)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::UnsupportedCapability,
                        DiagnosticSeverity::Error,
                        "The V1 swap-chain presentation format is Rgba8Unorm.",
                        {.resource = resource.id});
                }
            }
            if (const auto* texture = std::get_if<ir::TextureDescription>(
                &resource.description))
            {
                if (texture->mipLevels == 0
                    || texture->arrayLayers == 0
                    || texture->sampleCount == 0
                    || texture->format == gpu::ResourceFormat::Unknown)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "Texture declaration has an invalid format, mip, layer, or sample count.",
                        {.resource = resource.id});
                }
                const bool depthFormat = gpu::IsDepthFormat(texture->format);
                const bool depthUsage = HasUsage(
                    texture->usage, ir::TextureUsage::DepthAttachment);
                const bool colorUsage = HasUsage(
                    texture->usage, ir::TextureUsage::ColorAttachment);
                const bool storageUsage = HasUsage(
                    texture->usage, ir::TextureUsage::Storage);
                const bool sampledUsage = HasUsage(
                    texture->usage, ir::TextureUsage::Sampled);
                if (depthFormat != depthUsage || (depthUsage && colorUsage))
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "Depth formats and DepthAttachment usage must agree and cannot also be color attachments.",
                        {.resource = resource.id});
                }
                if (depthFormat && (storageUsage || sampledUsage))
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::UnsupportedCapability,
                        DiagnosticSeverity::Error,
                        "V1 depth resources are attachment-only; typeless sampled depth is not enabled.",
                        {.resource = resource.id});
                }
                if (depthFormat
                    && texture->dimension != gpu::ResourceKind::Texture2D)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::UnsupportedCapability,
                        DiagnosticSeverity::Error,
                        "V1 depth attachments are Texture2D resources.",
                        {.resource = resource.id});
                }
                if (texture->dimension == gpu::ResourceKind::Texture3D
                    && texture->arrayLayers != 1)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "Texture3D cannot declare array layers.",
                        {.resource = resource.id});
                }
                if (texture->sampleCount > 1
                    && (texture->mipLevels != 1
                        || texture->dimension
                            != gpu::ResourceKind::Texture2D
                        || storageUsage))
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "Multisampled V1 textures are 2D, single-mip, and cannot be UAVs.",
                        {.resource = resource.id});
                }
            }
        }

        for (const auto& program : module.programs)
        {
            if (!program.id.IsValid()
                || !programIds.insert(program.id.Value()).second)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidProgram,
                    DiagnosticSeverity::Error,
                    "Program ID is invalid or duplicated.",
                    {.program = program.id});
            }
            if (!program.parameters.empty()
                && !program.programInterface.parameters.empty())
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidProgram,
                    DiagnosticSeverity::Error,
                    "Program declares both legacy and canonical parameters.",
                    {.program = program.id});
            }
            if (program.programInterface.colorOutputCount
                > capabilities.maximumColorAttachments)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::UnsupportedCapability,
                    DiagnosticSeverity::Error,
                    "Program color-output count exceeds device capabilities.",
                    {.program = program.id});
            }
            std::set<std::pair<std::string, std::uint32_t>> semantics;
            for (const auto& input : program.programInterface.vertexInputs)
            {
                if (input.semanticName.empty()
                    || !semantics.emplace(
                        input.semanticName, input.semanticIndex).second)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidProgram,
                        DiagnosticSeverity::Error,
                        "Vertex interface has an empty or duplicate semantic.",
                        {.program = program.id});
                }
                if (input.inputRate == ir::VertexInputRate::PerVertex
                    && input.instanceStepRate != 0)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidProgram,
                        DiagnosticSeverity::Error,
                        "Per-vertex input declares an instance step rate.",
                        {.program = program.id});
                }
            }
        }

        for (const auto& work : module.works)
        {
            if (!work.id.IsValid()
                || !workIds.insert(work.id.Value()).second)
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::InvalidWork,
                    DiagnosticSeverity::Error,
                    "Work ID is invalid or duplicated.",
                    {.work = work.id});
                continue;
            }
            if ((work.Domain() == gpu::ExecutionDomain::Raster
                    && !capabilities.rasterExecution)
                || (work.Domain() == gpu::ExecutionDomain::Compute
                    && !capabilities.computeExecution)
                || (work.Domain() == gpu::ExecutionDomain::Copy
                    && !capabilities.copyExecution))
            {
                AddDiagnostic(
                    diagnostics,
                    DiagnosticCode::UnsupportedCapability,
                    DiagnosticSeverity::Error,
                    "Work execution domain is unavailable on the device.",
                    {.work = work.id});
            }

            if (const auto* raster = std::get_if<ir::RasterWork>(
                    &work.payload))
            {
                const auto* program = TryProgram(module, raster->program);
                if (program == nullptr)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidProgram,
                        DiagnosticSeverity::Error,
                        "Raster work references an unknown program.",
                        {.program = raster->program, .work = work.id});
                }
                else
                {
                    if (raster->attachments.colors.size()
                        != program->programInterface.colorOutputCount)
                    {
                        AddDiagnostic(
                            diagnostics,
                            DiagnosticCode::InvalidWork,
                            DiagnosticSeverity::Error,
                            "Raster attachment count does not match ProgramInterface.",
                            {.program = raster->program, .work = work.id});
                    }
                    std::set<std::uint32_t> declaredSlots;
                    for (const auto& input :
                         program->programInterface.vertexInputs)
                    {
                        declaredSlots.insert(input.inputSlot);
                        if (input.inputSlot
                            >= capabilities.maximumVertexStreams)
                        {
                            AddDiagnostic(
                                diagnostics,
                                DiagnosticCode::UnsupportedCapability,
                                DiagnosticSeverity::Error,
                                "Vertex input slot exceeds device capabilities.",
                                {.program = raster->program, .work = work.id});
                        }
                    }
                    std::set<std::uint32_t> boundSlots;
                    if (raster->vertexStreams.empty())
                    {
                        boundSlots.insert(0);
                    }
                    else
                    {
                        for (const auto& stream : raster->vertexStreams)
                        {
                            if (!boundSlots.insert(stream.inputSlot).second)
                            {
                                AddDiagnostic(
                                    diagnostics,
                                    DiagnosticCode::InvalidWork,
                                    DiagnosticSeverity::Error,
                                    "Raster work binds a vertex input slot more than once.",
                                    {.work = work.id,
                                     .view = stream.resource});
                            }
                        }
                    }
                    for (const auto slot : declaredSlots)
                    {
                        if (!boundSlots.contains(slot))
                        {
                            AddDiagnostic(
                                diagnostics,
                                DiagnosticCode::InvalidWork,
                                DiagnosticSeverity::Error,
                                "Raster work does not bind every declared vertex input slot.",
                                {.program = raster->program, .work = work.id});
                        }
                    }
                }
                if (raster->instanceCount == 0)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Raster instance count is zero.",
                        {.work = work.id});
                }
                if ((raster->indexBinding.has_value())
                    != (raster->indexCount != 0))
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Indexed draw requires both an index binding and an index count.",
                        {.work = work.id});
                }
                const auto samples = raster->rasterState.sampleCount;
                if (samples != 1 && samples != 2
                    && samples != 4 && samples != 8)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Raster sample count must be 1, 2, 4, or 8.",
                        {.work = work.id});
                }

                const bool hasDepth = raster->attachments.depth.IsValid();
                if ((raster->rasterState.depth
                        == gpu::DepthMode::Disabled) == hasDepth)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Raster depth mode and depth attachment presence disagree.",
                        {.work = work.id});
                }
                if (program != nullptr && hasDepth
                    && !program->programInterface.depthAttachmentAllowed)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "ProgramInterface forbids the declared depth attachment.",
                        {.program = raster->program, .work = work.id});
                }

                const auto validateAttachmentSamples = [&](const ir::ResourceView& view)
                {
                    const auto* declaration = TryResource(module, view.resource);
                    if (declaration == nullptr)
                    {
                        return;
                    }
                    std::uint32_t attachmentSamples = 1;
                    if (const auto* texture =
                            std::get_if<ir::TextureDescription>(
                                &declaration->description))
                    {
                        attachmentSamples = texture->sampleCount;
                    }
                    if (attachmentSamples != samples)
                    {
                        AddDiagnostic(
                            diagnostics,
                            DiagnosticCode::InvalidWork,
                            DiagnosticSeverity::Error,
                            "Raster sample count does not match an attachment.",
                            {.resource = view.resource,
                             .work = work.id,
                             .view = view});
                    }
                };
                for (const auto& color : raster->attachments.colors)
                {
                    validateAttachmentSamples(color);
                }
                if (hasDepth)
                {
                    validateAttachmentSamples(raster->attachments.depth);
                }
            }

            const auto uses = CollectViewUses(module, work, diagnostics);
            for (const auto& use : uses)
            {
                if (use.view.kind == gpu::ResourceKind::Buffer
                    || use.view.kind == gpu::ResourceKind::Presentation)
                {
                    continue;
                }
                const bool singleMipRole =
                    use.role == gpu::ResourceRole::ProgramOutput
                    || use.role == gpu::ResourceRole::ColorOutput
                    || use.role == gpu::ResourceRole::DepthOutput;
                if (singleMipRole && use.view.textureRange.mipCount != 1)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidView,
                        DiagnosticSeverity::Error,
                        "UAV, RTV, and DSV texture views select exactly one mip level.",
                        {.resource = use.view.resource,
                         .work = work.id});
                }
                const auto& texture = std::get<ir::TextureDescription>(
                    module.Resource(use.view.resource).description);
                if (use.role == gpu::ResourceRole::ProgramInput
                    && texture.dimension == gpu::ResourceKind::Texture3D)
                {
                    const auto mipDepth = std::max(
                        1u,
                        static_cast<std::uint32_t>(texture.depth)
                            >> use.view.textureRange.firstMip);
                    if (use.view.textureRange.firstDepthSlice != 0
                        || use.view.textureRange.depthSliceCount != mipDepth)
                    {
                        AddDiagnostic(
                            diagnostics,
                            DiagnosticCode::InvalidView,
                            DiagnosticSeverity::Error,
                            "A Texture3D SRV covers every depth slice of its selected mips.",
                            {.resource = use.view.resource,
                             .work = work.id});
                    }
                }
                if (use.role == gpu::ResourceRole::ProgramOutput)
                {
                    if (texture.sampleCount != 1)
                    {
                        AddDiagnostic(
                            diagnostics,
                            DiagnosticCode::InvalidView,
                            DiagnosticSeverity::Error,
                            "A texture UAV cannot be multisampled.",
                            {.resource = use.view.resource,
                             .work = work.id});
                    }
                }
            }
            ValidateAccessContract(work, uses, diagnostics);
            ValidateViewStateConflicts(module, work, uses, diagnostics);

            for (const auto& access : work.accesses)
            {
                const auto* resource = TryResource(module, access.resource);
                if (resource == nullptr)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidResource,
                        DiagnosticSeverity::Error,
                        "Work access references an unknown resource.",
                        {.resource = access.resource, .work = work.id});
                    continue;
                }
                if (access.frameLag > 0
                    && resource->lifetime
                        != gpu::ResourceLifetimeClass::Temporal)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Frame-lagged access requires a Temporal resource.",
                        {.resource = access.resource, .work = work.id});
                }
                if (access.frameLag > 0
                    && access.access != gpu::AccessMode::Read)
                {
                    AddDiagnostic(
                        diagnostics,
                        DiagnosticCode::InvalidWork,
                        DiagnosticSeverity::Error,
                        "Past temporal instances are read-only.",
                        {.resource = access.resource, .work = work.id});
                }
            }
        }

        return diagnostics;
    }

    PackageCompileResult RenderPackageCompiler::Compile(
        const ir::SemanticModule& module,
        const gpu::DeviceCapabilities& capabilities) const
    {
        PackageCompileResult result;
        result.diagnostics = Validate(module, capabilities);
        if (HasErrors(result.diagnostics))
        {
            return result;
        }

        result.package.sourceHash = module.StructureHash();
        result.package.canonicalModule = Canonicalize(
            module, result.diagnostics);
        result.package.legacyExecutable = true;
        result.package.legacyModule = BuildLegacyModule(
            result.package.canonicalModule,
            result.package.legacyExecutable);

        // Reuse the proven core passes with the real queue capabilities. The
        // package layer then makes every cross-queue edge explicit as a typed
        // handoff record without changing the established execution order.
        CompileResult core;
        try
        {
            core = coreCompiler_.Compile(
                result.package.legacyModule, capabilities);
        }
        catch (const std::exception& error)
        {
            AddDiagnostic(
                result.diagnostics,
                DiagnosticCode::InvalidWork,
                DiagnosticSeverity::Error,
                std::string("Compatibility planning failed: ")
                    + error.what());
            result.package.diagnostics = result.diagnostics;
            return result;
        }
        result.package.plan = std::move(core.plan);

        // Physical instance selection belongs to the canonical resource
        // lifetime, not to the compatibility module used by the old core.
        // Rebuild every entry so generalized streams/index resources omitted
        // from the compatibility payload are still materialized correctly.
        result.package.plan.resourceInstances.clear();
        const auto frameCount = std::max(
            1u, capabilities.maxFramesInFlight);
        for (const auto& resource :
             result.package.canonicalModule.resources)
        {
            ResourceInstancePlan instance;
            instance.resource = resource.id;
            instance.lifetime = resource.lifetime;
            switch (resource.lifetime)
            {
            case gpu::ResourceLifetimeClass::Persistent:
                instance.selector = gpu::InstanceSelectorKind::Persistent;
                instance.physicalInstanceCount = 1;
                break;
            case gpu::ResourceLifetimeClass::FrameLocal:
                instance.selector =
                    gpu::InstanceSelectorKind::CurrentFrameSlot;
                instance.physicalInstanceCount = frameCount;
                break;
            case gpu::ResourceLifetimeClass::Temporal:
            {
                std::uint32_t maximumFrameLag = 0;
                for (const auto& work :
                     result.package.canonicalModule.works)
                {
                    for (const auto& access : work.accesses)
                    {
                        if (access.resource == resource.id)
                        {
                            maximumFrameLag = std::max(
                                maximumFrameLag, access.frameLag);
                        }
                    }
                }
                instance.selector =
                    gpu::InstanceSelectorKind::CurrentTemporalSlot;
                instance.maximumFrameLag = maximumFrameLag;
                instance.physicalInstanceCount = std::max({
                    2u, frameCount, maximumFrameLag + 1u});
                break;
            }
            case gpu::ResourceLifetimeClass::External:
                instance.selector =
                    gpu::InstanceSelectorKind::ExternalFrameIndex;
                instance.physicalInstanceCount = 0;
                break;
            }
            result.package.plan.resourceInstances.push_back(instance);
        }

        for (auto& text : core.diagnostics)
        {
            AddDiagnostic(
                result.diagnostics,
                DiagnosticCode::None,
                DiagnosticSeverity::Information,
                std::move(text));
        }

        std::unordered_map<std::size_t, std::size_t> scheduledPosition;
        for (std::size_t index = 0;
             index < result.package.plan.scheduledWorks.size(); ++index)
        {
            scheduledPosition.emplace(
                result.package.plan.scheduledWorks[index].sourceWorkIndex,
                index);
        }

        for (const auto& resource : result.package.canonicalModule.resources)
        {
            const auto instance = std::find_if(
                result.package.plan.resourceInstances.begin(),
                result.package.plan.resourceInstances.end(),
                [&](const ResourceInstancePlan& value)
                {
                    return value.resource == resource.id;
                });
            ResourceInstancePlan instancePlan;
            if (instance != result.package.plan.resourceInstances.end())
            {
                instancePlan = *instance;
            }
            else
            {
                instancePlan.resource = resource.id;
                instancePlan.lifetime = resource.lifetime;
            }
            const auto bytes = EstimateResourceBytes(
                resource, instancePlan.physicalInstanceCount);
            result.package.resources.push_back({
                .declaration = resource,
                .instances = instancePlan,
                .estimatedCommittedBytes = bytes
            });
            result.package.statistics.estimatedCommittedBytes += bytes;
            result.package.statistics.physicalInstanceCount +=
                instancePlan.physicalInstanceCount;
        }

        for (const auto& resource : result.package.canonicalModule.resources)
        {
            const auto format = resource.Format();
            if (format != gpu::ResourceFormat::Unknown
                && format != gpu::ResourceFormat::Rgba8Unorm
                && format != gpu::ResourceFormat::Depth32Float)
            {
                result.package.requirements.expandedResourceFormats = true;
            }
        }

        for (const auto& program : result.package.canonicalModule.programs)
        {
            result.package.programs.push_back({
                .declaration = program,
                .parameters = program.BindingParameters()
            });
        }

        result.package.works.resize(
            result.package.plan.scheduledWorks.size());
        for (std::size_t scheduledIndex = 0;
             scheduledIndex < result.package.plan.scheduledWorks.size();
             ++scheduledIndex)
        {
            const auto& scheduled =
                result.package.plan.scheduledWorks[scheduledIndex];
            const auto& source = result.package.canonicalModule.works.at(
                scheduled.sourceWorkIndex);
            auto& compiled = result.package.works[scheduledIndex];
            compiled.id = source.id;
            compiled.name = source.name;
            compiled.accesses = source.accesses;
            compiled.queue = source.Domain() == gpu::ExecutionDomain::Copy
                    && capabilities.dedicatedCopyQueue
                ? gpu::QueueClass::Copy
                : source.Domain() == gpu::ExecutionDomain::Compute
                    && capabilities.concurrentCompute
                    ? gpu::QueueClass::Compute
                    : gpu::QueueClass::Direct;

            const auto viewUses = CollectViewUses(
                result.package.canonicalModule, source,
                result.diagnostics);
            for (const auto& use : viewUses)
            {
                compiled.rangeStates.push_back({
                    .view = use.view,
                    .state = gpu::RequiredState({
                        use.view.resource, use.access, use.role,
                        use.frameLag}),
                    .frameLag = use.frameLag,
                    .writeCapable = use.access != gpu::AccessMode::Read
                });
                if (use.view.kind != gpu::ResourceKind::Buffer
                    && use.view.kind != gpu::ResourceKind::Presentation)
                {
                    const auto& declaration = result.package.canonicalModule.Resource(
                        use.view.resource);
                    const auto& texture = std::get<ir::TextureDescription>(
                        declaration.description);
                    const bool fullRange =
                        use.view.textureRange.firstMip == 0
                        && use.view.textureRange.mipCount == texture.mipLevels
                        && use.view.textureRange.firstArrayLayer == 0
                        && use.view.textureRange.arrayLayerCount
                            == texture.arrayLayers
                        && use.view.textureRange.firstPlane == 0
                        && use.view.textureRange.planeCount == 1
                        && use.view.textureRange.firstDepthSlice == 0
                        && use.view.textureRange.depthSliceCount
                            == (texture.dimension
                                    == gpu::ResourceKind::Texture3D
                                ? texture.depth
                                : 1u)
                        && use.view.format == texture.format;
                    result.package.requirements.textureSubresourceViews =
                        result.package.requirements.textureSubresourceViews
                        || !fullRange;
                }
            }

            std::visit([&](const auto& payload)
            {
                using T = std::decay_t<decltype(payload)>;
                if constexpr (std::is_same_v<T, ir::RasterWork>)
                {
                    CompiledRasterCommand command;
                    command.program = payload.program;
                    command.executable = RasterExecutable(
                        result.package.canonicalModule, payload);
                    command.bindings = CompileBindings(
                        result.package.canonicalModule,
                        payload.program, payload.bindings,
                        result.diagnostics, source.id);
                    if (payload.vertexStreams.empty())
                    {
                        if (const auto view = NormalizeView(
                            result.package.canonicalModule,
                            payload.vertexResource,
                            result.diagnostics, source.id))
                        {
                            command.vertexStreams.push_back({0, *view});
                        }
                    }
                    else
                    {
                        for (const auto& stream : payload.vertexStreams)
                        {
                            if (const auto view = NormalizeView(
                                result.package.canonicalModule,
                                stream.resource,
                                result.diagnostics, source.id))
                            {
                                command.vertexStreams.push_back({
                                    stream.inputSlot, *view});
                            }
                        }
                    }
                    if (command.vertexStreams.size() > 1)
                    {
                        result.package.requirements.multipleVertexStreams = true;
                    }
                    if (payload.indexBinding)
                    {
                        if (const auto view = NormalizeView(
                            result.package.canonicalModule,
                            payload.indexBinding->resource,
                            result.diagnostics, source.id))
                        {
                            command.indexStream = CompiledIndexStream{
                                *view,
                                payload.indexBinding->format,
                                payload.indexBinding->firstIndex,
                                payload.indexBinding->baseVertex};
                            result.package.requirements.indexedDraw = true;
                        }
                    }
                    command.vertexCount = payload.vertexCount;
                    command.firstVertex = payload.firstVertex;
                    command.indexCount = payload.indexCount;
                    command.instanceCount = payload.instanceCount;
                    command.firstInstance = payload.firstInstance;
                    command.viewport = payload.viewport;
                    command.scissor = payload.scissor;
                    command.clear = payload.clear;
                    if (payload.rasterState.cull != ir::CullMode::None
                        || payload.rasterState.fill != ir::FillMode::Solid
                        || payload.rasterState.frontFace
                            != ir::FrontFace::Clockwise
                        || payload.rasterState.sampleCount != 1
                        || payload.clear.colorLoad
                            != ir::AttachmentLoadOperation::Preserve
                        || payload.clear.colorStore
                            != ir::AttachmentStoreOperation::Store
                        || payload.clear.depthLoad
                            != ir::AttachmentLoadOperation::Preserve
                        || payload.clear.depthStore
                            != ir::AttachmentStoreOperation::Store)
                    {
                        result.package.requirements.advancedRasterState = true;
                    }
                    if (payload.viewport.width != 0.0f
                        || payload.viewport.height != 0.0f
                        || payload.viewport.x != 0.0f
                        || payload.viewport.y != 0.0f
                        || payload.viewport.minimumDepth != 0.0f
                        || payload.viewport.maximumDepth != 1.0f
                        || payload.scissor.left != 0
                        || payload.scissor.top != 0
                        || payload.scissor.right != 0
                        || payload.scissor.bottom != 0)
                    {
                        result.package.requirements.customViewportScissor = true;
                    }
                    if (payload.instanceCount != 1
                        || payload.firstInstance != 0)
                    {
                        result.package.requirements.instancedDraw = true;
                    }
                    for (const auto& color : payload.attachments.colors)
                    {
                        if (const auto view = NormalizeView(
                            result.package.canonicalModule,
                            color, result.diagnostics, source.id))
                        {
                            command.colorAttachments.push_back(*view);
                        }
                    }
                    if (payload.attachments.depth.IsValid())
                    {
                        command.depthAttachment = NormalizeView(
                            result.package.canonicalModule,
                            payload.attachments.depth,
                            result.diagnostics, source.id);
                    }
                    compiled.command = std::move(command);
                }
                else if constexpr (std::is_same_v<T, ir::ComputeWork>)
                {
                    CompiledComputeCommand command;
                    command.program = payload.program;
                    command.executable.program = payload.program;
                    command.executable.compute = true;
                    command.bindings = CompileBindings(
                        result.package.canonicalModule,
                        payload.program, payload.bindings,
                        result.diagnostics, source.id);
                    command.groupCountX = payload.groupCountX;
                    command.groupCountY = payload.groupCountY;
                    command.groupCountZ = payload.groupCountZ;
                    compiled.command = std::move(command);
                }
                else if constexpr (std::is_same_v<T, ir::CopyWork>)
                {
                    const auto* sourceResource = TryResource(
                        result.package.canonicalModule, payload.source);
                    const auto* destinationResource = TryResource(
                        result.package.canonicalModule, payload.destination);
                    const auto makeCopyView = [&](const ir::ResourceDeclaration* resource,
                                                  gpu::ResourceId id,
                                                  std::uint64_t offset)
                    {
                        if (resource == nullptr
                            || resource->Kind() != gpu::ResourceKind::Buffer)
                        {
                            return ir::ResourceView{id};
                        }
                        const auto size = payload.sizeBytes == 0
                            ? resource->SizeBytes() - offset
                            : payload.sizeBytes;
                        return ir::ResourceView{id, offset, size};
                    };
                    const auto sourceView = NormalizeView(
                        result.package.canonicalModule,
                        makeCopyView(sourceResource, payload.source,
                            payload.sourceOffset),
                        result.diagnostics, source.id);
                    const auto destinationView = NormalizeView(
                        result.package.canonicalModule,
                        makeCopyView(destinationResource, payload.destination,
                            payload.destinationOffset),
                        result.diagnostics, source.id);
                    if (sourceView && destinationView)
                    {
                        compiled.command = CompiledCopyCommand{
                            *sourceView,
                            *destinationView,
                            payload.sourceFrameLag,
                            payload.destinationFrameLag};
                    }
                }
                else
                {
                    compiled.command = CompiledPresentCommand{payload.source};
                }
            }, source.payload);
        }

        // Recompute immutable Persistent read envelopes from package-native
        // range states. Resources touching the Copy queue deliberately remain
        // outside a fixed union state and use explicit handoffs instead.
        result.package.plan.persistentReadStates.clear();
        for (const auto& resource :
             result.package.canonicalModule.resources)
        {
            if (resource.lifetime
                != gpu::ResourceLifetimeClass::Persistent)
            {
                continue;
            }
            PersistentReadStatePlan envelope;
            envelope.resource = resource.id;
            bool copyUse = false;
            bool writeUse = false;
            for (const auto& work : result.package.works)
            {
                for (const auto& requirement : work.rangeStates)
                {
                    if (requirement.view.resource != resource.id)
                    {
                        continue;
                    }
                    copyUse = copyUse
                        || work.queue == gpu::QueueClass::Copy;
                    writeUse = writeUse || requirement.writeCapable;
                    if (std::find(
                            envelope.states.begin(),
                            envelope.states.end(),
                            requirement.state) == envelope.states.end())
                    {
                        envelope.states.push_back(requirement.state);
                    }
                }
            }
            if (!copyUse && !writeUse && !envelope.states.empty())
            {
                std::sort(
                    envelope.states.begin(), envelope.states.end(),
                    [](gpu::AbstractState left, gpu::AbstractState right)
                    {
                        return static_cast<std::uint32_t>(left)
                            < static_cast<std::uint32_t>(right);
                    });
                result.package.plan.persistentReadStates.push_back(
                    std::move(envelope));
            }
        }

        for (const auto& dependency : result.package.plan.dependencies)
        {
            const auto beforeIt = scheduledPosition.find(dependency.before);
            const auto afterIt = scheduledPosition.find(dependency.after);
            if (beforeIt == scheduledPosition.end()
                || afterIt == scheduledPosition.end())
            {
                continue;
            }
            const auto before = beforeIt->second;
            const auto after = afterIt->second;
            const auto releaseQueue = result.package.works[before].queue;
            const auto acquireQueue = result.package.works[after].queue;
            if (releaseQueue == acquireQueue)
            {
                continue;
            }
            const auto releaseState = std::find_if(
                result.package.works[before].rangeStates.begin(),
                result.package.works[before].rangeStates.end(),
                [&](const RangeStateRequirement& requirement)
                {
                    return requirement.view.resource == dependency.resource;
                });
            const auto acquireState = std::find_if(
                result.package.works[after].rangeStates.begin(),
                result.package.works[after].rangeStates.end(),
                [&](const RangeStateRequirement& requirement)
                {
                    return requirement.view.resource == dependency.resource;
                });
            const bool copy = releaseQueue == gpu::QueueClass::Copy
                || acquireQueue == gpu::QueueClass::Copy;
            result.package.queueHandoffs.push_back({
                .releaseScheduledWork = before,
                .acquireScheduledWork = after,
                .resource = dependency.resource,
                .frameLag = acquireState
                        == result.package.works[after].rangeStates.end()
                    ? 0
                    : acquireState->frameLag,
                .releaseQueue = releaseQueue,
                .acquireQueue = acquireQueue,
                .releaseState = releaseState
                        == result.package.works[before].rangeStates.end()
                    ? gpu::AbstractState::Undefined
                    : releaseState->state,
                .acquireState = acquireState
                        == result.package.works[after].rangeStates.end()
                    ? gpu::AbstractState::Undefined
                    : acquireState->state,
                .crossesCopyQueue = copy
            });
            result.package.requirements.explicitCopyQueueHandoffs =
                result.package.requirements.explicitCopyQueueHandoffs || copy;
        }

        // Resource state ownership is a physical-instance property, not only
        // a RAW/WAR/WAW dependency property. Two read-only works on different
        // queues may have no data dependency, yet the first transition into a
        // read state must complete before the other queue uses that instance.
        // Persistent immutable reads are the exception: their union state is
        // initialized once before frame execution.
        struct LastQueueUse
        {
            std::size_t work = 0;
            gpu::QueueClass queue = gpu::QueueClass::Direct;
            gpu::AbstractState state = gpu::AbstractState::Undefined;
            bool writeCapable = false;
        };
        std::unordered_map<std::uint64_t, LastQueueUse> lastQueueUses;
        const auto instanceKey = [](gpu::ResourceId resource,
                                    std::uint32_t frameLag)
        {
            return (static_cast<std::uint64_t>(resource.Value()) << 32u)
                | frameLag;
        };
        for (std::size_t workIndex = 0;
             workIndex < result.package.works.size(); ++workIndex)
        {
            const auto& work = result.package.works[workIndex];
            for (const auto& requirement : work.rangeStates)
            {
                const auto key = instanceKey(
                    requirement.view.resource, requirement.frameLag);
                const auto previous = lastQueueUses.find(key);
                if (previous != lastQueueUses.end()
                    && previous->second.queue != work.queue)
                {
                    const auto& declaration =
                        result.package.canonicalModule.Resource(
                            requirement.view.resource);
                    const bool persistentCompatibleReads =
                        declaration.lifetime
                            == gpu::ResourceLifetimeClass::Persistent
                        && !previous->second.writeCapable
                        && !requirement.writeCapable;
                    const bool alreadyPlanned = std::any_of(
                        result.package.queueHandoffs.begin(),
                        result.package.queueHandoffs.end(),
                        [&](const QueueHandoffPlan& handoff)
                        {
                            return handoff.releaseScheduledWork
                                    == previous->second.work
                                && handoff.acquireScheduledWork == workIndex
                                && handoff.resource
                                    == requirement.view.resource
                                && handoff.frameLag == requirement.frameLag;
                        });
                    if (!persistentCompatibleReads && !alreadyPlanned)
                    {
                        const bool copy = previous->second.queue
                                == gpu::QueueClass::Copy
                            || work.queue == gpu::QueueClass::Copy;
                        result.package.queueHandoffs.push_back({
                            .releaseScheduledWork = previous->second.work,
                            .acquireScheduledWork = workIndex,
                            .resource = requirement.view.resource,
                            .frameLag = requirement.frameLag,
                            .releaseQueue = previous->second.queue,
                            .acquireQueue = work.queue,
                            .releaseState = previous->second.state,
                            .acquireState = requirement.state,
                            .crossesCopyQueue = copy});
                        result.package.requirements.explicitCopyQueueHandoffs =
                            result.package.requirements
                                .explicitCopyQueueHandoffs || copy;
                    }
                }
                lastQueueUses[key] = {
                    workIndex,
                    work.queue,
                    requirement.state,
                    requirement.writeCapable};
            }
        }

        result.package.requirements.dynamicDescriptorGrowth =
            result.package.requirements.textureSubresourceViews
            || result.package.works.size() > 128;

        const auto& requirements = result.package.requirements;
        const bool unsupported =
            (requirements.textureSubresourceViews
                && !capabilities.textureSubresourceViews)
            || (requirements.multipleVertexStreams
                && !capabilities.multipleVertexStreams)
            || (requirements.indexedDraw && !capabilities.indexedDraw)
            || (requirements.instancedDraw && !capabilities.instancedDraw)
            || (requirements.explicitCopyQueueHandoffs
                && !capabilities.explicitCopyQueueHandoffs)
            || (requirements.dynamicDescriptorGrowth
                && !capabilities.dynamicDescriptorGrowth)
            || (requirements.advancedRasterState
                && !capabilities.advancedRasterState)
            || (requirements.customViewportScissor
                && !capabilities.customViewportScissor)
            || (requirements.expandedResourceFormats
                && !capabilities.expandedResourceFormats);
        if (unsupported)
        {
            result.package.legacyExecutable = false;
            AddDiagnostic(
                result.diagnostics,
                DiagnosticCode::UnsupportedCapability,
                DiagnosticSeverity::Error,
                "The active backend cannot execute one or more required V1 package features.");
        }

        result.package.statistics.logicalResourceCount =
            result.package.resources.size();
        result.package.statistics.workCount = result.package.works.size();
        std::vector<ExecutableKey> packageExecutables;
        std::size_t rangeBarrierUpperBound = 0;
        for (const auto& work : result.package.works)
        {
            rangeBarrierUpperBound += work.rangeStates.size();
            std::visit([&](const auto& command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<T, CompiledRasterCommand>
                    || std::is_same_v<T, CompiledComputeCommand>)
                {
                    if (std::find(
                            packageExecutables.begin(),
                            packageExecutables.end(),
                            command.executable) == packageExecutables.end())
                    {
                        packageExecutables.push_back(command.executable);
                    }
                }
            }, work.command);
        }
        result.package.statistics.executableCount =
            packageExecutables.size();
        result.package.statistics.barrierCount =
            rangeBarrierUpperBound + result.package.queueHandoffs.size();
        result.package.statistics.queueWaitCount =
            result.package.queueHandoffs.size();
        for (const auto& work : result.package.works)
        {
            result.package.statistics.descriptorViewCount +=
                std::visit([](const auto& command) -> std::size_t
                {
                    using T = std::decay_t<decltype(command)>;
                    if constexpr (std::is_same_v<T, CompiledRasterCommand>)
                    {
                        return command.bindings.size()
                            + command.colorAttachments.size()
                            + (command.depthAttachment ? 1u : 0u);
                    }
                    else if constexpr (
                        std::is_same_v<T, CompiledComputeCommand>)
                    {
                        return command.bindings.size();
                    }
                    return 0;
                }, work.command);
        }

        if (capabilities.localMemoryBudget != 0
            && result.package.statistics.estimatedCommittedBytes
                > capabilities.localMemoryBudget)
        {
            AddDiagnostic(
                result.diagnostics,
                DiagnosticCode::MemoryBudgetExceeded,
                DiagnosticSeverity::Error,
                "Estimated physical resource memory exceeds the device budget.",
                {},
                {"Estimated bytes: " + std::to_string(
                    result.package.statistics.estimatedCommittedBytes),
                 "Budget bytes: " + std::to_string(
                    capabilities.localMemoryBudget)});
        }

        std::size_t packageHash =
            result.package.canonicalModule.StructureHash();
        foundation::HashCombine(packageHash, capabilities.concurrentCompute);
        foundation::HashCombine(packageHash, capabilities.dedicatedCopyQueue);
        foundation::HashCombine(packageHash, capabilities.maxFramesInFlight);
        foundation::HashCombine(packageHash, capabilities.resourceAliasing);
        const auto hashView = [&](const NormalizedResourceView& view)
        {
            foundation::HashCombine(packageHash, view.resource.Value());
            foundation::HashEnum(packageHash, view.kind);
            foundation::HashCombine(packageHash,
                static_cast<std::size_t>(view.byteOffset));
            foundation::HashCombine(packageHash,
                static_cast<std::size_t>(view.byteSize));
            foundation::HashCombine(packageHash, view.strideBytes);
            foundation::HashCombine(packageHash, view.textureRange.firstMip);
            foundation::HashCombine(packageHash, view.textureRange.mipCount);
            foundation::HashCombine(
                packageHash, view.textureRange.firstArrayLayer);
            foundation::HashCombine(
                packageHash, view.textureRange.arrayLayerCount);
            foundation::HashCombine(packageHash, view.textureRange.firstPlane);
            foundation::HashCombine(packageHash, view.textureRange.planeCount);
            foundation::HashCombine(
                packageHash, view.textureRange.firstDepthSlice);
            foundation::HashCombine(
                packageHash, view.textureRange.depthSliceCount);
            foundation::HashEnum(packageHash, view.format);
        };
        for (const auto& work : result.package.works)
        {
            foundation::HashCombine(packageHash, work.id.Value());
            foundation::HashEnum(packageHash, work.queue);
            for (const auto& range : work.rangeStates)
            {
                hashView(range.view);
                foundation::HashEnum(packageHash, range.state);
                foundation::HashCombine(packageHash, range.frameLag);
                foundation::HashCombine(packageHash, range.writeCapable);
            }
            std::visit([&](const auto& command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<T, CompiledRasterCommand>)
                {
                    foundation::HashCombine(packageHash, command.program.Value());
                    foundation::HashCombine(packageHash, command.vertexCount);
                    foundation::HashCombine(packageHash, command.indexCount);
                    foundation::HashCombine(packageHash, command.instanceCount);
                    foundation::HashCombine(packageHash, command.firstInstance);
                    for (const auto& stream : command.vertexStreams)
                    {
                        foundation::HashCombine(packageHash, stream.inputSlot);
                        hashView(stream.view);
                    }
                    if (command.indexStream)
                    {
                        hashView(command.indexStream->view);
                        foundation::HashEnum(
                            packageHash, command.indexStream->format);
                        foundation::HashCombine(
                            packageHash, command.indexStream->firstIndex);
                        foundation::HashCombine(
                            packageHash,
                            static_cast<std::size_t>(
                                command.indexStream->baseVertex));
                    }
                }
                else if constexpr (std::is_same_v<T, CompiledComputeCommand>)
                {
                    foundation::HashCombine(packageHash, command.program.Value());
                    foundation::HashCombine(packageHash, command.groupCountX);
                    foundation::HashCombine(packageHash, command.groupCountY);
                    foundation::HashCombine(packageHash, command.groupCountZ);
                }
                else if constexpr (std::is_same_v<T, CompiledCopyCommand>)
                {
                    hashView(command.source);
                    hashView(command.destination);
                    foundation::HashCombine(
                        packageHash, command.sourceFrameLag);
                    foundation::HashCombine(
                        packageHash, command.destinationFrameLag);
                }
                else
                {
                    foundation::HashCombine(packageHash, command.source.Value());
                }
            }, work.command);
        }
        for (const auto& handoff : result.package.queueHandoffs)
        {
            foundation::HashCombine(
                packageHash, handoff.releaseScheduledWork);
            foundation::HashCombine(
                packageHash, handoff.acquireScheduledWork);
            foundation::HashCombine(packageHash, handoff.resource.Value());
            foundation::HashCombine(packageHash, handoff.frameLag);
            foundation::HashEnum(packageHash, handoff.releaseQueue);
            foundation::HashEnum(packageHash, handoff.acquireQueue);
        }
        result.package.packageHash = packageHash;
        result.package.plan.structureHash = packageHash;
        result.package.diagnostics = result.diagnostics;
        return result;
    }

    const char* ToString(DiagnosticCode code) noexcept
    {
        switch (code)
        {
        case DiagnosticCode::None: return "None";
        case DiagnosticCode::LegacyProgramInterface:
            return "LegacyProgramInterface";
        case DiagnosticCode::InvalidResource: return "InvalidResource";
        case DiagnosticCode::InvalidProgram: return "InvalidProgram";
        case DiagnosticCode::InvalidWork: return "InvalidWork";
        case DiagnosticCode::InvalidView: return "InvalidView";
        case DiagnosticCode::InvalidSubresourceRange:
            return "InvalidSubresourceRange";
        case DiagnosticCode::PayloadAccessMismatch:
            return "PayloadAccessMismatch";
        case DiagnosticCode::ResourceStateConflict:
            return "ResourceStateConflict";
        case DiagnosticCode::UnsupportedCapability:
            return "UnsupportedCapability";
        case DiagnosticCode::MemoryBudgetExceeded:
            return "MemoryBudgetExceeded";
        case DiagnosticCode::QueueHandoffRequired:
            return "QueueHandoffRequired";
        case DiagnosticCode::ShaderInterfaceMismatch:
            return "ShaderInterfaceMismatch";
        case DiagnosticCode::PackageNotLegacyExecutable:
            return "PackageNotLegacyExecutable";
        }
        return "Unknown";
    }

    const char* ToString(DiagnosticSeverity severity) noexcept
    {
        switch (severity)
        {
        case DiagnosticSeverity::Information: return "information";
        case DiagnosticSeverity::Warning: return "warning";
        case DiagnosticSeverity::Error: return "error";
        }
        return "unknown";
    }
}
