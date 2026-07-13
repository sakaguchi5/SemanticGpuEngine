#include "07_D3D12Backend/D3D12BackendInternal.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

using Microsoft::WRL::ComPtr;

namespace
{
    const sge::ir::ResourceDeclaration& PackageResource(
        const sge::compiler::CompiledRenderPackage& package,
        sge::gpu::ResourceId id)
    {
        return package.Resource(id).declaration;
    }

    bool IsTextureKind(sge::gpu::ResourceKind kind) noexcept
    {
        return kind == sge::gpu::ResourceKind::Texture1D
            || kind == sge::gpu::ResourceKind::Texture2D
            || kind == sge::gpu::ResourceKind::Texture3D;
    }

    DXGI_FORMAT NativeTypelessFormat(
        sge::gpu::ResourceFormat format) noexcept
    {
        using F = sge::gpu::ResourceFormat;
        switch (format)
        {
        case F::R8Unorm: return DXGI_FORMAT_R8_TYPELESS;
        case F::Rg8Unorm: return DXGI_FORMAT_R8G8_TYPELESS;
        case F::Rgba8Unorm: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case F::Bgra8Unorm: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        case F::R16Float: return DXGI_FORMAT_R16_TYPELESS;
        case F::Rg16Float: return DXGI_FORMAT_R16G16_TYPELESS;
        case F::Rgba16Float: return DXGI_FORMAT_R16G16B16A16_TYPELESS;
        case F::R32Float:
        case F::R32Uint: return DXGI_FORMAT_R32_TYPELESS;
        case F::Rg32Float:
        case F::Rg32Uint: return DXGI_FORMAT_R32G32_TYPELESS;
        case F::Rgba32Float:
        case F::Rgba32Uint: return DXGI_FORMAT_R32G32B32A32_TYPELESS;
        case F::Depth24Stencil8: return DXGI_FORMAT_R24G8_TYPELESS;
        case F::Depth32Float: return DXGI_FORMAT_R32_TYPELESS;
        default: return sge::d3d12::detail::NativeFormat(format);
        }
    }

    bool IsReadOnlyState(sge::gpu::AbstractState state) noexcept
    {
        using S = sge::gpu::AbstractState;
        switch (state)
        {
        case S::VertexRead:
        case S::IndexRead:
        case S::ConstantRead:
        case S::IndirectRead:
        case S::ProgramRead:
        case S::DepthRead:
        case S::TransferRead:
            return true;
        default:
            return false;
        }
    }

    std::uint32_t PlaneCount(
        ID3D12Device* device,
        DXGI_FORMAT format) noexcept
    {
        D3D12_FEATURE_DATA_FORMAT_INFO information{};
        information.Format = format;
        if (device != nullptr
            && SUCCEEDED(device->CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_INFO,
                &information,
                sizeof(information)))
            && information.PlaneCount != 0)
        {
            return information.PlaneCount;
        }
        return 1;
    }

    bool IsFullTextureRange(
        const sge::compiler::NormalizedResourceView& view,
        const sge::ir::TextureDescription& texture) noexcept
    {
        return view.textureRange.firstMip == 0
            && view.textureRange.mipCount == texture.mipLevels
            && view.textureRange.firstArrayLayer == 0
            && view.textureRange.arrayLayerCount
                == (texture.dimension == sge::gpu::ResourceKind::Texture3D
                    ? 1u : texture.arrayLayers)
            && view.textureRange.firstPlane == 0
            && view.textureRange.planeCount == 1
            && view.textureRange.firstDepthSlice == 0
            && view.textureRange.depthSliceCount
                == (texture.dimension == sge::gpu::ResourceKind::Texture3D
                    ? texture.depth
                    : 1u)
            && view.format == texture.format;
    }

    constexpr UINT SubresourceIndex(
        UINT mip,
        UINT arrayLayer,
        UINT plane,
        UINT mipLevels,
        UINT arraySize) noexcept
    {
        return mip + arrayLayer * mipLevels
            + plane * mipLevels * arraySize;
    }

    UINT CheckedUint(std::uint64_t value, const char* message)
    {
        if (value > std::numeric_limits<UINT>::max())
        {
            throw std::runtime_error(message);
        }
        return static_cast<UINT>(value);
    }
}

namespace sge::d3d12::detail
{
    void Backend::EnsurePackageDescriptorCapacity(
        const compiler::CompiledRenderPackage& package)
    {
        std::uint64_t physicalInstances = 0;
        std::uint64_t maximumInstancesPerResource = 1;
        for (const auto& resource : package.resources)
        {
            const auto count = std::max(
                1u, resource.instances.physicalInstanceCount);
            physicalInstances += count;
            maximumInstancesPerResource = std::max<std::uint64_t>(
                maximumInstancesPerResource, count);
        }
        const auto worstCaseViewInstances =
            package.statistics.descriptorViewCount
            * maximumInstancesPerResource;

        const auto requestedRtv = static_cast<UINT>(std::min<std::uint64_t>(
            std::numeric_limits<UINT>::max(),
            FrameCount + 32ull + physicalInstances
                + worstCaseViewInstances));
        const auto requestedDsv = static_cast<UINT>(std::min<std::uint64_t>(
            std::numeric_limits<UINT>::max(),
            16ull + physicalInstances
                + worstCaseViewInstances));
        const auto requestedShader = static_cast<UINT>(std::min<std::uint64_t>(
            std::numeric_limits<UINT>::max(),
            128ull + physicalInstances * 4ull
                + worstCaseViewInstances * 2ull));

        const UINT newRtv = std::max(rtvDescriptorCapacity_, requestedRtv);
        const UINT newDsv = std::max(dsvDescriptorCapacity_, requestedDsv);
        const UINT newShader = std::max(
            shaderDescriptorCapacity_, requestedShader);
        if (newRtv == rtvDescriptorCapacity_
            && newDsv == dsvDescriptorCapacity_
            && newShader == shaderDescriptorCapacity_)
        {
            return;
        }

        WaitIdle();
        rtvHeap_.Reset();
        dsvHeap_.Reset();
        shaderHeap_.Reset();

        D3D12_DESCRIPTOR_HEAP_DESC description{};
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        description.NumDescriptors = newRtv;
        ThrowIfFailed(device_->CreateDescriptorHeap(
            &description, IID_PPV_ARGS(&rtvHeap_)),
            "Grow RTV descriptor heap");

        description = {};
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        description.NumDescriptors = newDsv;
        ThrowIfFailed(device_->CreateDescriptorHeap(
            &description, IID_PPV_ARGS(&dsvHeap_)),
            "Grow DSV descriptor heap");

        description = {};
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        description.NumDescriptors = newShader;
        description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(device_->CreateDescriptorHeap(
            &description, IID_PPV_ARGS(&shaderHeap_)),
            "Grow shader descriptor heap");

        rtvDescriptorCapacity_ = newRtv;
        dsvDescriptorCapacity_ = newDsv;
        shaderDescriptorCapacity_ = newShader;
        rtvIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        dsvIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        shaderIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        nextRtvDescriptor_ = FrameCount;
        nextDsvDescriptor_ = 0;
        nextShaderDescriptor_ = 0;

        for (auto& buffer : backBuffers_) buffer.Reset();
        CreateRenderTargets();
        compiledStructureHash_.reset();
        compiledPackageHash_.reset();
    }

    void Backend::EnsureUploadCapacity(UINT64 requiredBytes)
    {
        const auto required = std::max<UINT64>(requiredBytes, 256);
        if (required <= uploadArenaCapacity_) return;

        WaitIdle();
        uploadArenaCapacity_ = AlignUp(required, 64ull * 1024ull);
        const auto properties = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        const auto description = BufferDescription(uploadArenaCapacity_);
        for (auto& frame : frames_)
        {
            if (frame.uploadArena && frame.mappedUpload != nullptr)
            {
                frame.uploadArena->Unmap(0, nullptr);
                frame.mappedUpload = nullptr;
            }
            frame.uploadArena.Reset();
            ThrowIfFailed(device_->CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&frame.uploadArena)),
                "Grow frame upload arena");
            ThrowIfFailed(frame.uploadArena->Map(
                0,
                nullptr,
                reinterpret_cast<void**>(&frame.mappedUpload)),
                "Map grown frame upload arena");
            frame.uploadOffset = 0;
        }
    }

    void Backend::UploadPackageInitialBufferData(
        const compiler::CompiledRenderPackage& package)
    {
        std::vector<const compiler::CompiledResourceBlueprint*> uploadsToMake;
        for (const auto& blueprint : package.resources)
        {
            const auto* content = std::get_if<
                compiler::BufferInitialContent>(&blueprint.initialContent);
            if (blueprint.declaration.Kind() == gpu::ResourceKind::Buffer
                && blueprint.declaration.lifetime
                    != gpu::ResourceLifetimeClass::External
                && blueprint.declaration.update
                    != gpu::ResourceUpdateClass::CpuUpdated
                && content != nullptr
                && !content->bytes.empty())
            {
                uploadsToMake.push_back(&blueprint);
            }
        }
        if (uploadsToMake.empty())
        {
            return;
        }

        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        ThrowIfFailed(device_->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)),
            "Create package initialization allocator");
        ThrowIfFailed(device_->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&list)),
            "Create package initialization list");

        std::vector<ComPtr<ID3D12Resource>> uploadResources;
        uploadResources.reserve(uploadsToMake.size());
        const auto uploadProperties = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        for (const auto* blueprint : uploadsToMake)
        {
            const auto& bytes = std::get<compiler::BufferInitialContent>(
                blueprint->initialContent).bytes;
            const auto byteCount = bytes.size();
            const auto uploadDescription = BufferDescription(byteCount);
            ComPtr<ID3D12Resource> upload;
            ThrowIfFailed(device_->CreateCommittedResource(
                &uploadProperties,
                D3D12_HEAP_FLAG_NONE,
                &uploadDescription,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&upload)),
                "Create package initialization upload buffer");

            void* mapped = nullptr;
            ThrowIfFailed(upload->Map(0, nullptr, &mapped),
                "Map package initialization upload buffer");
            std::memcpy(
                mapped,
                bytes.data(),
                byteCount);
            upload->Unmap(0, nullptr);

            auto found = resources_.find(blueprint->declaration.id);
            if (found == resources_.end() || found->second.empty())
            {
                throw std::runtime_error(
                    "Package initial data has no materialized buffer.");
            }
            for (auto& record : found->second)
            {
                list->CopyBufferRegion(
                    record.resource.Get(),
                    0,
                    upload.Get(),
                    0,
                    byteCount);
                record.state = D3D12_RESOURCE_STATE_COMMON;
                record.subresourceStates.clear();
            }
            uploadResources.push_back(std::move(upload));
        }

        ThrowIfFailed(list->Close(),
            "Close package initialization list");
        ID3D12CommandList* lists[] = {list.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const UINT64 completion = SignalQueue(gpu::QueueClass::Direct);
        WaitForCpuQueueValue(gpu::QueueClass::Direct, completion);
        ValidateDebugLayer();
    }

    void Backend::UploadPackageInitialTextureData(
        const compiler::CompiledRenderPackage& package)
    {
        struct PendingUpload
        {
            const compiler::CompiledResourceBlueprint* blueprint = nullptr;
            const compiler::TextureInitialContent* content = nullptr;
        };
        std::vector<PendingUpload> pending;
        for (const auto& blueprint : package.resources)
        {
            const auto* content = std::get_if<
                compiler::TextureInitialContent>(&blueprint.initialContent);
            if (content != nullptr && !content->subresources.empty())
            {
                pending.push_back({&blueprint, content});
            }
        }
        if (pending.empty()) return;

        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        ThrowIfFailed(device_->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
            "Create texture initialization allocator");
        ThrowIfFailed(device_->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
            IID_PPV_ARGS(&list)), "Create texture initialization list");

        const auto uploadProperties = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
        std::vector<ComPtr<ID3D12Resource>> uploadResources;
        for (const auto& upload : pending)
        {
            auto found = resources_.find(upload.blueprint->declaration.id);
            if (found == resources_.end() || found->second.empty())
            {
                throw std::runtime_error(
                    "Texture initial data has no materialized resource.");
            }
            const auto& texture = std::get<ir::TextureDescription>(
                upload.blueprint->declaration.description);
            const UINT arrays = texture.dimension == gpu::ResourceKind::Texture3D
                ? 1u : texture.arrayLayers;

            for (auto& record : found->second)
            {
                const auto native = record.resource->GetDesc();
                for (const auto& source : upload.content->subresources)
                {
                    const UINT subresource = source.mip
                        + source.arrayLayer * native.MipLevels
                        + source.plane * native.MipLevels * arrays;
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
                    UINT rows = 0;
                    UINT64 rowBytes = 0;
                    UINT64 totalBytes = 0;
                    device_->GetCopyableFootprints(
                        &native, subresource, 1, 0,
                        &footprint, &rows, &rowBytes, &totalBytes);

                    ComPtr<ID3D12Resource> staging;
                    const auto stagingDescription = BufferDescription(totalBytes);
                    ThrowIfFailed(device_->CreateCommittedResource(
                        &uploadProperties,
                        D3D12_HEAP_FLAG_NONE,
                        &stagingDescription,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                        nullptr,
                        IID_PPV_ARGS(&staging)),
                        "Create texture initialization upload buffer");
                    std::byte* mapped = nullptr;
                    ThrowIfFailed(staging->Map(
                        0, nullptr, reinterpret_cast<void**>(&mapped)),
                        "Map texture initialization upload buffer");
                    const UINT depth = footprint.Footprint.Depth;
                    for (UINT z = 0; z < depth; ++z)
                    {
                        for (UINT y = 0; y < rows; ++y)
                        {
                            const auto sourceOffset =
                                static_cast<UINT64>(z) * source.sourceSlicePitch
                                + static_cast<UINT64>(y) * source.sourceRowPitch;
                            const auto destinationOffset = footprint.Offset
                                + static_cast<UINT64>(z) * rows
                                    * footprint.Footprint.RowPitch
                                + static_cast<UINT64>(y)
                                    * footprint.Footprint.RowPitch;
                            std::memcpy(
                                mapped + destinationOffset,
                                source.bytes.data() + sourceOffset,
                                static_cast<std::size_t>(rowBytes));
                        }
                    }
                    staging->Unmap(0, nullptr);

                    D3D12_RESOURCE_BARRIER toCopy{};
                    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    toCopy.Transition.pResource = record.resource.Get();
                    toCopy.Transition.Subresource = subresource;
                    toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                    toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                    list->ResourceBarrier(1, &toCopy);

                    D3D12_TEXTURE_COPY_LOCATION destination{};
                    destination.pResource = record.resource.Get();
                    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    destination.SubresourceIndex = subresource;
                    D3D12_TEXTURE_COPY_LOCATION origin{};
                    origin.pResource = staging.Get();
                    origin.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    origin.PlacedFootprint = footprint;
                    list->CopyTextureRegion(
                        &destination, 0, 0, 0, &origin, nullptr);

                    std::swap(
                        toCopy.Transition.StateBefore,
                        toCopy.Transition.StateAfter);
                    list->ResourceBarrier(1, &toCopy);
                    uploadResources.push_back(std::move(staging));
                }
                record.state = D3D12_RESOURCE_STATE_COMMON;
                record.subresourceStates.clear();
            }
        }

        ThrowIfFailed(list->Close(), "Close texture initialization list");
        ID3D12CommandList* lists[] = {list.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const UINT64 completion = SignalQueue(gpu::QueueClass::Direct);
        WaitForCpuQueueValue(gpu::QueueClass::Direct, completion);
        ValidateDebugLayer();
    }

    void Backend::InitializePackagePersistentReadStates(
        const compiler::CompiledRenderPackage& package)
    {
        persistentReadStates_.clear();
        for (const auto& blueprint : package.resources)
        {
            if (blueprint.persistentReadStates.empty())
            {
                continue;
            }

            UINT nativeBits = 0;
            for (const auto state : blueprint.persistentReadStates)
            {
                nativeBits |= static_cast<UINT>(NativeState(state));
            }
            if (nativeBits == 0
                || !persistentReadStates_.emplace(
                    blueprint.declaration.id,
                    static_cast<D3D12_RESOURCE_STATES>(nativeBits)).second)
            {
                throw std::runtime_error(
                    "Invalid package persistent read-state blueprint.");
            }
        }
        InitializePersistentReadStates();
    }


    void Backend::EnsurePackageCompiled(
        const compiler::CompiledRenderPackage& package)
    {
        if (compiledPackageHash_
            && *compiledPackageHash_ == package.packageHash
            && compiledStructureHash_
            && *compiledStructureHash_ == package.packageHash
            && preparedWidth_ == width_
            && preparedHeight_ == height_
            && preparedDeviceEpoch_ == deviceEpoch_)
        {
            return;
        }

        EnsurePackageDescriptorCapacity(package);
        UINT64 dynamicBytes = 0;
        for (const auto& resource : package.resources)
        {
            if (resource.declaration.update
                == gpu::ResourceUpdateClass::CpuUpdated)
            {
                dynamicBytes += AlignUp(
                    resource.declaration.SizeBytes(),
                    D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            }
        }
        EnsureUploadCapacity(dynamicBytes);
        WaitIdle();

        // Package preparation starts from a clean backend object graph. No
        // SemanticModule or ExecutionPlan is reconstructed or interpreted.
        resources_.clear();
        resourceInstancePlans_.clear();
        persistentReadStates_.clear();
        customShaderViews_.clear();
        packageShaderViews_.clear();
        packageAttachmentViews_.clear();
        physicalAllocations_.clear();
        allocationHeaps_.clear();
        activeAliasedResources_.clear();
        temporalInstanceUsages_.clear();
        programs_.clear();
        pipelines_.clear();
        presentationResource_.reset();
        nextRtvDescriptor_ = FrameCount;
        nextDsvDescriptor_ = 0;
        nextShaderDescriptor_ = 0;

        struct HeapRequirement
        {
            UINT64 size = 0;
            UINT64 alignment = 0;
            D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
            bool assigned = false;
        };
        std::unordered_map<
            gpu::PhysicalAllocationId,
            HeapRequirement,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>>
            heapRequirements;

        for (const auto& blueprint : package.resources)
        {
            resourceInstancePlans_.emplace(
                blueprint.declaration.id, blueprint.instances);
            if (blueprint.instances.lifetime
                == gpu::ResourceLifetimeClass::Temporal)
            {
                temporalInstanceUsages_[blueprint.declaration.id].assign(
                    blueprint.instances.physicalInstanceCount, {});
            }

            if (!blueprint.allocation)
            {
                continue;
            }

            D3D12_RESOURCE_DESC native{};
            D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
            if (blueprint.declaration.Kind() == gpu::ResourceKind::Buffer)
            {
                native = BufferDescription(blueprint.declaration.SizeBytes());
                const auto& buffer = std::get<ir::BufferDescription>(
                    blueprint.declaration.description);
                if ((static_cast<std::uint32_t>(buffer.usage)
                    & static_cast<std::uint32_t>(
                        ir::BufferUsage::Storage)) != 0)
                {
                    native.Flags |=
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                }
                flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            }
            else if (IsTextureKind(blueprint.declaration.Kind()))
            {
                const auto& texture = std::get<ir::TextureDescription>(
                    blueprint.declaration.description);
                native = TextureDescription(texture, width_, height_);
                if (blueprint.requiresTypelessResource)
                {
                    native.Format = NativeTypelessFormat(texture.format);
                }
                const auto usage = static_cast<std::uint32_t>(texture.usage);
                flags = (usage
                    & (static_cast<std::uint32_t>(
                            ir::TextureUsage::ColorAttachment)
                        | static_cast<std::uint32_t>(
                            ir::TextureUsage::DepthAttachment))) != 0
                    ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES
                    : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            }
            else
            {
                throw std::runtime_error(
                    "Package allocation blueprint has an unsupported resource kind.");
            }

            const auto information = device_->GetResourceAllocationInfo(
                0, 1, &native);
            auto& requirement = heapRequirements[*blueprint.allocation];
            if (requirement.assigned && requirement.flags != flags)
            {
                throw std::runtime_error(
                    "Aliased package resources require incompatible D3D12 heap flags.");
            }
            requirement.assigned = true;
            requirement.size = std::max(
                requirement.size, information.SizeInBytes);
            requirement.alignment = std::max(
                requirement.alignment, information.Alignment);
            requirement.flags = flags;
            physicalAllocations_[blueprint.declaration.id] =
                *blueprint.allocation;
        }

        for (const auto& [allocation, requirement] : heapRequirements)
        {
            D3D12_HEAP_DESC description{};
            description.Alignment = requirement.alignment;
            description.SizeInBytes = AlignUp(
                requirement.size,
                std::max<UINT64>(
                    requirement.alignment,
                    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT));
            description.Properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
            description.Flags = requirement.flags;

            std::vector<ComPtr<ID3D12Heap>> heaps(FrameCount);
            for (auto& heap : heaps)
            {
                ThrowIfFailed(device_->CreateHeap(
                    &description, IID_PPV_ARGS(&heap)),
                    "Create package alias heap");
            }
            allocationHeaps_.emplace(allocation, std::move(heaps));
            activeAliasedResources_[allocation].resize(FrameCount);
        }

        for (const auto& blueprint : package.resources)
        {
            const auto& declaration = blueprint.declaration;
            if (declaration.lifetime
                == gpu::ResourceLifetimeClass::External)
            {
                if (declaration.Kind() == gpu::ResourceKind::Presentation)
                {
                    presentationResource_ = declaration.id;
                }
                continue;
            }
            if (declaration.update == gpu::ResourceUpdateClass::CpuUpdated)
            {
                continue;
            }

            auto& instances = resources_[declaration.id];
            instances.reserve(blueprint.instances.physicalInstanceCount);
            for (std::uint32_t instance = 0;
                 instance < blueprint.instances.physicalInstanceCount;
                 ++instance)
            {
                ID3D12Heap* heap = nullptr;
                if (blueprint.allocation)
                {
                    heap = allocationHeaps_.at(*blueprint.allocation)
                        .at(instance % FrameCount).Get();
                }

                if (declaration.Kind() == gpu::ResourceKind::Buffer)
                {
                    auto materialized = declaration;
                    materialized.data.clear();
                    instances.push_back(
                        CreateStaticBuffer(materialized, heap));
                    instances.back().state = D3D12_RESOURCE_STATE_COMMON;
                    instances.back().subresourceStates.clear();
                    continue;
                }

                if (!IsTextureKind(declaration.Kind()))
                {
                    throw std::runtime_error(
                        "Package resource blueprint has an unsupported materialized kind.");
                }

                const auto& texture = std::get<ir::TextureDescription>(
                    declaration.description);
                auto native = TextureDescription(texture, width_, height_);
                if (blueprint.requiresTypelessResource)
                {
                    native.Format = NativeTypelessFormat(texture.format);
                }

                D3D12_CLEAR_VALUE clear{};
                const D3D12_CLEAR_VALUE* clearPointer = nullptr;
                if (blueprint.optimizedClear)
                {
                    clear.Format = NativeFormat(
                        blueprint.optimizedClear->format);
                    if (blueprint.optimizedClear->depthStencil)
                    {
                        clear.DepthStencil.Depth =
                            blueprint.optimizedClear->depth;
                        clear.DepthStencil.Stencil =
                            blueprint.optimizedClear->stencil;
                    }
                    else
                    {
                        std::copy(
                            blueprint.optimizedClear->color.begin(),
                            blueprint.optimizedClear->color.end(),
                            std::begin(clear.Color));
                    }
                    clearPointer = &clear;
                }

                const auto properties = HeapProperties(
                    D3D12_HEAP_TYPE_DEFAULT);
                ResourceRecord record;
                const HRESULT createResult = heap != nullptr
                    ? device_->CreatePlacedResource(
                        heap,
                        0,
                        &native,
                        D3D12_RESOURCE_STATE_COMMON,
                        clearPointer,
                        IID_PPV_ARGS(&record.resource))
                    : device_->CreateCommittedResource(
                        &properties,
                        D3D12_HEAP_FLAG_NONE,
                        &native,
                        D3D12_RESOURCE_STATE_COMMON,
                        clearPointer,
                        IID_PPV_ARGS(&record.resource));
                ThrowIfFailed(
                    createResult, "Create package texture resource");
                record.state = D3D12_RESOURCE_STATE_COMMON;
                instances.push_back(std::move(record));
            }
        }

        UploadPackageInitialBufferData(package);
        UploadPackageInitialTextureData(package);
        InitializePackagePersistentReadStates(package);

        for (const auto& blueprint : package.programs)
        {
            programs_.emplace(
                blueprint.declaration.id,
                CreateProgram(blueprint.declaration));
        }
        for (const auto& executable : package.executables)
        {
            pipelines_.emplace(
                executable, CreatePackagePipeline(executable));
        }

        // ResizeIfNeeded resets compiledStructureHash_; keeping the prepared
        // package identity in both slots forces dimension-dependent resources
        // to be rematerialized after a surface resize.
        compiledStructureHash_ = package.packageHash;
        compiledPackageHash_ = package.packageHash;
        preparedWidth_ = width_;
        preparedHeight_ = height_;
        preparedDeviceEpoch_ = deviceEpoch_;
    }


    ComPtr<ID3D12PipelineState> Backend::CreatePackagePipeline(
        const compiler::ExecutableKey& executable)
    {
        const auto program = programs_.find(executable.program);
        if (program == programs_.end())
        {
            throw std::runtime_error(
                "Package pipeline references an unknown program.");
        }
        if (executable.compute)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
            description.pRootSignature = program->second.rootSignature.Get();
            description.CS = {
                program->second.compute.bytecode->GetBufferPointer(),
                program->second.compute.bytecode->GetBufferSize()};
            ComPtr<ID3D12PipelineState> pipeline;
            ThrowIfFailed(device_->CreateComputePipelineState(
                &description, IID_PPV_ARGS(&pipeline)),
                "Create package compute pipeline");
            return pipeline;
        }

        if (activePackage_ == nullptr)
        {
            throw std::runtime_error(
                "Package graphics pipeline has no active ProgramInterface.");
        }
        shaderCompiler_.ValidateRasterInterface(
            activePackage_->Program(
                executable.program).declaration.programInterface,
            program->second.vertex,
            program->second.pixel);

        if (executable.colorFormats.empty()
            || executable.colorFormats.size() > 8)
        {
            throw std::runtime_error(
                "Package graphics executable requires one to eight color targets.");
        }
        std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout;
        inputLayout.reserve(executable.vertexInputs.size());
        for (const auto& input : executable.vertexInputs)
        {
            inputLayout.push_back({
                input.semanticName.c_str(),
                input.semanticIndex,
                NativeVertexFormat(input.format),
                input.inputSlot,
                input.alignedByteOffset,
                input.inputRate == ir::VertexInputRate::PerInstance
                    ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                    : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                input.instanceStepRate});
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC description{};
        description.pRootSignature = program->second.rootSignature.Get();
        description.VS = {
            program->second.vertex.bytecode->GetBufferPointer(),
            program->second.vertex.bytecode->GetBufferSize()};
        description.PS = {
            program->second.pixel.bytecode->GetBufferPointer(),
            program->second.pixel.bytecode->GetBufferSize()};
        description.BlendState = BlendDescription(
            executable.rasterState.composition);
        description.SampleMask = UINT_MAX;
        description.RasterizerState = RasterizerDescription(
            executable.rasterState);
        description.DepthStencilState = DepthDescription(
            executable.rasterState.depth);
        description.InputLayout = {
            inputLayout.data(), static_cast<UINT>(inputLayout.size())};
        description.PrimitiveTopologyType = NativeTopologyType(
            executable.rasterState.topology);
        description.NumRenderTargets = static_cast<UINT>(
            executable.colorFormats.size());
        for (std::size_t index = 0;
             index < executable.colorFormats.size(); ++index)
        {
            description.RTVFormats[index] = NativeFormat(
                executable.colorFormats[index]);
            if (index > 0)
            {
                description.BlendState.RenderTarget[index] =
                    description.BlendState.RenderTarget[0];
            }
        }
        description.DSVFormat = NativeFormat(executable.depthFormat);
        description.SampleDesc.Count = std::max<UINT>(
            1, executable.rasterState.sampleCount);

        ComPtr<ID3D12PipelineState> pipeline;
        ThrowIfFailed(device_->CreateGraphicsPipelineState(
            &description, IID_PPV_ARGS(&pipeline)),
            "Create package graphics pipeline");
        return pipeline;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE Backend::ResolvePackageShaderDescriptor(
        const compiler::NormalizedResourceView& view,
        gpu::ProgramParameterKind kind,
        std::uint32_t frameLag)
    {
        if (activePackage_ == nullptr)
            throw std::runtime_error("No active package for descriptor resolution.");
        if (kind != gpu::ProgramParameterKind::ShaderResource
            && kind != gpu::ProgramParameterKind::UnorderedAccess)
        {
            throw std::runtime_error(
                "Package descriptor is neither SRV nor UAV.");
        }
        auto* record = ResolveResource(view.resource, frameLag);
        if (record == nullptr)
            throw std::runtime_error("Package view references no physical resource.");
        const auto& declaration = PackageResource(*activePackage_, view.resource);

        bool canonicalWhole = false;
        if (view.kind == gpu::ResourceKind::Buffer)
        {
            const auto& buffer = std::get<ir::BufferDescription>(
                declaration.description);
            canonicalWhole = view.byteOffset == 0
                && view.byteSize == buffer.sizeBytes
                && (view.strideBytes == 0
                    || view.strideBytes == buffer.strideBytes);
        }
        else if (IsTextureKind(view.kind))
        {
            const auto& texture = std::get<ir::TextureDescription>(
                declaration.description);
            canonicalWhole = IsFullTextureRange(view, texture)
                && texture.arrayLayers == 1
                && texture.sampleCount == 1;
        }
        if (canonicalWhole)
        {
            if (kind == gpu::ProgramParameterKind::ShaderResource
                && record->hasSrv)
                return record->srvDescriptor;
            if (kind == gpu::ProgramParameterKind::UnorderedAccess
                && record->hasUav)
                return record->uavDescriptor;
        }

        const PackageShaderViewKey key{view, kind};
        auto& cached = packageShaderViews_[key];
        const auto instanceCount = resources_.at(view.resource).size();
        if (cached.instances.size() != instanceCount)
            cached.instances.resize(instanceCount);
        const auto instance = ResolveInstanceIndex(view.resource, frameLag);
        if (cached.instances.at(instance).ptr != 0)
            return cached.instances.at(instance);
        if (nextShaderDescriptor_ >= shaderDescriptorCapacity_)
            throw std::runtime_error("Package shader descriptor heap exhausted.");

        auto cpu = shaderHeap_->GetCPUDescriptorHandleForHeapStart();
        auto gpu = shaderHeap_->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(nextShaderDescriptor_) * shaderIncrement_;
        gpu.ptr += static_cast<UINT64>(nextShaderDescriptor_++) * shaderIncrement_;

        if (view.kind == gpu::ResourceKind::Buffer)
        {
            const auto elementSize = view.strideBytes == 0
                ? 4u : view.strideBytes;
            const UINT64 firstElement = view.byteOffset / elementSize;
            const UINT elementCount = CheckedUint(
                view.byteSize / elementSize,
                "Package buffer view contains too many elements.");
            if (kind == gpu::ProgramParameterKind::ShaderResource)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.Format = view.strideBytes == 0
                    ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                native.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                native.Shader4ComponentMapping =
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                native.Buffer.FirstElement = firstElement;
                native.Buffer.NumElements = elementCount;
                native.Buffer.StructureByteStride = view.strideBytes;
                native.Buffer.Flags = view.strideBytes == 0
                    ? D3D12_BUFFER_SRV_FLAG_RAW
                    : D3D12_BUFFER_SRV_FLAG_NONE;
                device_->CreateShaderResourceView(
                    record->resource.Get(), &native, cpu);
            }
            else
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
                native.Format = view.strideBytes == 0
                    ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
                native.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                native.Buffer.FirstElement = firstElement;
                native.Buffer.NumElements = elementCount;
                native.Buffer.StructureByteStride = view.strideBytes;
                native.Buffer.Flags = view.strideBytes == 0
                    ? D3D12_BUFFER_UAV_FLAG_RAW
                    : D3D12_BUFFER_UAV_FLAG_NONE;
                device_->CreateUnorderedAccessView(
                    record->resource.Get(), nullptr, &native, cpu);
            }
        }
        else
        {
            const auto& texture = std::get<ir::TextureDescription>(
                declaration.description);
            const bool array = texture.dimension
                    != gpu::ResourceKind::Texture3D
                && texture.arrayLayers > 1;
            const bool multisampled = texture.sampleCount > 1;
            if (kind == gpu::ProgramParameterKind::ShaderResource)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC native{};
                native.Format = NativeFormat(view.format);
                native.Shader4ComponentMapping =
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                if (texture.dimension == gpu::ResourceKind::Texture1D)
                {
                    if (array)
                    {
                        native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                        native.Texture1DArray.MostDetailedMip = view.textureRange.firstMip;
                        native.Texture1DArray.MipLevels = view.textureRange.mipCount;
                        native.Texture1DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                        native.Texture1DArray.ArraySize = view.textureRange.arrayLayerCount;
                    }
                    else
                    {
                        native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                        native.Texture1D.MostDetailedMip = view.textureRange.firstMip;
                        native.Texture1D.MipLevels = view.textureRange.mipCount;
                    }
                }
                else if (texture.dimension == gpu::ResourceKind::Texture3D)
                {
                    native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                    native.Texture3D.MostDetailedMip = view.textureRange.firstMip;
                    native.Texture3D.MipLevels = view.textureRange.mipCount;
                }
                else if (multisampled)
                {
                    if (array)
                    {
                        native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                        native.Texture2DMSArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                        native.Texture2DMSArray.ArraySize = view.textureRange.arrayLayerCount;
                    }
                    else native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                }
                else if (array)
                {
                    native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                    native.Texture2DArray.MostDetailedMip = view.textureRange.firstMip;
                    native.Texture2DArray.MipLevels = view.textureRange.mipCount;
                    native.Texture2DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                    native.Texture2DArray.ArraySize = view.textureRange.arrayLayerCount;
                    native.Texture2DArray.PlaneSlice = view.textureRange.firstPlane;
                }
                else
                {
                    native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    native.Texture2D.MostDetailedMip = view.textureRange.firstMip;
                    native.Texture2D.MipLevels = view.textureRange.mipCount;
                    native.Texture2D.PlaneSlice = view.textureRange.firstPlane;
                }
                device_->CreateShaderResourceView(
                    record->resource.Get(), &native, cpu);
            }
            else
            {
                if (multisampled || view.textureRange.mipCount != 1)
                    throw std::runtime_error(
                        "A texture UAV selects exactly one non-MSAA mip level.");
                D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
                native.Format = NativeFormat(view.format);
                if (texture.dimension == gpu::ResourceKind::Texture1D)
                {
                    if (array)
                    {
                        native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                        native.Texture1DArray.MipSlice = view.textureRange.firstMip;
                        native.Texture1DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                        native.Texture1DArray.ArraySize = view.textureRange.arrayLayerCount;
                    }
                    else
                    {
                        native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                        native.Texture1D.MipSlice = view.textureRange.firstMip;
                    }
                }
                else if (texture.dimension == gpu::ResourceKind::Texture3D)
                {
                    native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                    native.Texture3D.MipSlice = view.textureRange.firstMip;
                    native.Texture3D.FirstWSlice =
                        view.textureRange.firstDepthSlice;
                    native.Texture3D.WSize =
                        view.textureRange.depthSliceCount;
                }
                else if (array)
                {
                    native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                    native.Texture2DArray.MipSlice = view.textureRange.firstMip;
                    native.Texture2DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                    native.Texture2DArray.ArraySize = view.textureRange.arrayLayerCount;
                    native.Texture2DArray.PlaneSlice = view.textureRange.firstPlane;
                }
                else
                {
                    native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    native.Texture2D.MipSlice = view.textureRange.firstMip;
                    native.Texture2D.PlaneSlice = view.textureRange.firstPlane;
                }
                device_->CreateUnorderedAccessView(
                    record->resource.Get(), nullptr, &native, cpu);
            }
        }
        cached.instances.at(instance) = gpu;
        return gpu;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE Backend::ResolvePackageAttachmentDescriptor(
        const compiler::NormalizedResourceView& view,
        bool depth,
        std::uint32_t frameLag)
    {
        if (activePackage_ == nullptr)
            throw std::runtime_error("No active package for attachment resolution.");
        if (view.kind == gpu::ResourceKind::Presentation)
        {
            if (depth) throw std::runtime_error("Presentation cannot be a depth target.");
            auto handle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<SIZE_T>(frameIndex_) * rtvIncrement_;
            return handle;
        }
        auto* record = ResolveResource(view.resource, frameLag);
        if (record == nullptr)
            throw std::runtime_error("Attachment view references no physical resource.");
        const auto& declaration = PackageResource(*activePackage_, view.resource);
        const auto& texture = std::get<ir::TextureDescription>(declaration.description);
        if (IsFullTextureRange(view, texture)
            && texture.arrayLayers == 1
            && texture.sampleCount == 1)
        {
            if (depth && record->hasDsv) return record->dsv;
            if (!depth && record->hasRtv) return record->rtv;
        }

        const PackageAttachmentViewKey key{view, depth};
        auto& cached = packageAttachmentViews_[key];
        const auto instanceCount = resources_.at(view.resource).size();
        if (cached.instances.size() != instanceCount)
            cached.instances.resize(instanceCount);
        const auto instance = ResolveInstanceIndex(view.resource, frameLag);
        if (cached.instances.at(instance).ptr != 0)
            return cached.instances.at(instance);

        D3D12_CPU_DESCRIPTOR_HANDLE handle{};
        if (depth)
        {
            if (nextDsvDescriptor_ >= dsvDescriptorCapacity_)
                throw std::runtime_error("Package DSV heap exhausted.");
            handle = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<SIZE_T>(nextDsvDescriptor_++) * dsvIncrement_;
            D3D12_DEPTH_STENCIL_VIEW_DESC native{};
            native.Format = NativeFormat(view.format);
            const bool array = texture.arrayLayers > 1;
            const bool msaa = texture.sampleCount > 1;
            if (msaa && array)
            {
                native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
                native.Texture2DMSArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                native.Texture2DMSArray.ArraySize = view.textureRange.arrayLayerCount;
            }
            else if (msaa) native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
            else if (array)
            {
                native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                native.Texture2DArray.MipSlice = view.textureRange.firstMip;
                native.Texture2DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                native.Texture2DArray.ArraySize = view.textureRange.arrayLayerCount;
            }
            else
            {
                native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                native.Texture2D.MipSlice = view.textureRange.firstMip;
            }
            device_->CreateDepthStencilView(record->resource.Get(), &native, handle);
        }
        else
        {
            if (nextRtvDescriptor_ >= rtvDescriptorCapacity_)
                throw std::runtime_error("Package RTV heap exhausted.");
            handle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<SIZE_T>(nextRtvDescriptor_++) * rtvIncrement_;
            D3D12_RENDER_TARGET_VIEW_DESC native{};
            native.Format = NativeFormat(view.format);
            const bool array = texture.dimension != gpu::ResourceKind::Texture3D
                && texture.arrayLayers > 1;
            const bool msaa = texture.sampleCount > 1;
            if (texture.dimension == gpu::ResourceKind::Texture1D)
            {
                if (array)
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
                    native.Texture1DArray.MipSlice = view.textureRange.firstMip;
                    native.Texture1DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                    native.Texture1DArray.ArraySize = view.textureRange.arrayLayerCount;
                }
                else
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
                    native.Texture1D.MipSlice = view.textureRange.firstMip;
                }
            }
            else if (texture.dimension == gpu::ResourceKind::Texture3D)
            {
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
                native.Texture3D.MipSlice = view.textureRange.firstMip;
                native.Texture3D.FirstWSlice =
                    view.textureRange.firstDepthSlice;
                native.Texture3D.WSize =
                    view.textureRange.depthSliceCount;
            }
            else if (msaa && array)
            {
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                native.Texture2DMSArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                native.Texture2DMSArray.ArraySize = view.textureRange.arrayLayerCount;
            }
            else if (msaa) native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            else if (array)
            {
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                native.Texture2DArray.MipSlice = view.textureRange.firstMip;
                native.Texture2DArray.FirstArraySlice = view.textureRange.firstArrayLayer;
                native.Texture2DArray.ArraySize = view.textureRange.arrayLayerCount;
                native.Texture2DArray.PlaneSlice = view.textureRange.firstPlane;
            }
            else
            {
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                native.Texture2D.MipSlice = view.textureRange.firstMip;
                native.Texture2D.PlaneSlice = view.textureRange.firstPlane;
            }
            device_->CreateRenderTargetView(record->resource.Get(), &native, handle);
        }
        cached.instances.at(instance) = handle;
        return handle;
    }

    D3D12_VERTEX_BUFFER_VIEW Backend::BuildPackageVertexView(
        const compiler::NormalizedResourceView& view,
        std::uint32_t frameLag)
    {
        const auto* record = ResolveResource(view.resource, frameLag);
        if (record == nullptr || view.kind != gpu::ResourceKind::Buffer)
            throw std::runtime_error("Vertex stream is not a materialized buffer.");
        D3D12_VERTEX_BUFFER_VIEW native{};
        native.BufferLocation = record->resource->GetGPUVirtualAddress()
            + view.byteOffset;
        native.SizeInBytes = CheckedUint(
            view.byteSize, "Vertex stream exceeds D3D12 view limits.");
        native.StrideInBytes = view.strideBytes;
        return native;
    }

    D3D12_INDEX_BUFFER_VIEW Backend::BuildPackageIndexView(
        const compiler::CompiledIndexStream& stream,
        std::uint32_t frameLag)
    {
        const auto* record = ResolveResource(stream.view.resource, frameLag);
        if (record == nullptr || stream.view.kind != gpu::ResourceKind::Buffer)
            throw std::runtime_error("Index stream is not a materialized buffer.");
        D3D12_INDEX_BUFFER_VIEW native{};
        native.BufferLocation = record->resource->GetGPUVirtualAddress()
            + stream.view.byteOffset;
        native.SizeInBytes = CheckedUint(
            stream.view.byteSize, "Index stream exceeds D3D12 view limits.");
        native.Format = NativeIndexFormat(stream.format);
        return native;
    }

    void Backend::BindPackageResources(
        gpu::ProgramId programId,
        const std::vector<compiler::CompiledBinding>& bindings,
        FrameContext& frame,
        bool compute)
    {
        if (activePackage_ == nullptr || activeInvocation_ == nullptr)
            throw std::runtime_error("No active package invocation for resource binding.");
        const auto program = programs_.find(programId);
        if (program == programs_.end())
            throw std::runtime_error("Package binding references an unknown program.");

        for (const auto& binding : bindings)
        {
            if (binding.parameterIndex >= program->second.rootParameterIndices.size())
                throw std::runtime_error("Package binding parameter index is invalid.");
            const UINT root = program->second.rootParameterIndices[binding.parameterIndex];
            if (root == UINT_MAX) continue;
            if (binding.kind == gpu::ProgramParameterKind::ConstantBuffer)
            {
                auto declaration = PackageResource(
                    *activePackage_, binding.view.resource);
                const auto dynamic = activeInvocation_->dynamicResourceData.find(
                    binding.view.resource);
                if (dynamic != activeInvocation_->dynamicResourceData.end())
                {
                    if (dynamic->second.size() != declaration.SizeBytes())
                    {
                        throw std::runtime_error(
                            "FrameInvocation constant data size does not match its resource blueprint.");
                    }
                    declaration.data = dynamic->second;
                }
                const auto address = UploadConstants(frame, declaration);
                if (compute) commandList_->SetComputeRootConstantBufferView(root, address);
                else commandList_->SetGraphicsRootConstantBufferView(root, address);
            }
            else if (binding.kind == gpu::ProgramParameterKind::ShaderResource
                || binding.kind == gpu::ProgramParameterKind::UnorderedAccess)
            {
                const auto descriptor = ResolvePackageShaderDescriptor(
                    binding.view, binding.kind, binding.frameLag);
                if (compute) commandList_->SetComputeRootDescriptorTable(root, descriptor);
                else commandList_->SetGraphicsRootDescriptorTable(root, descriptor);
            }
        }
    }

    void Backend::TransitionPackageRange(
        const compiler::NormalizedResourceView& view,
        gpu::AbstractState abstractState,
        std::uint32_t frameLag)
    {
        if (activeQueue_ == gpu::QueueClass::Copy)
        {
            throw std::runtime_error(
                "Package state transitions must not be recorded on a Copy command list.");
        }
        if (view.kind == gpu::ResourceKind::Buffer
            || view.kind == gpu::ResourceKind::Presentation)
        {
            Transition(view.resource, abstractState, frameLag);
            return;
        }
        auto* record = ResolveResource(view.resource, frameLag);
        if (record == nullptr)
            throw std::runtime_error("Range transition references no physical texture.");
        const auto description = record->resource->GetDesc();
        const UINT mipLevels = description.MipLevels;
        const UINT arraySize = description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1u : description.DepthOrArraySize;
        const UINT planes = PlaneCount(device_.Get(), description.Format);
        const std::size_t count = static_cast<std::size_t>(mipLevels)
            * arraySize * planes;
        if (record->subresourceStates.empty())
            record->subresourceStates.assign(count, record->state);

        D3D12_RESOURCE_STATES target =
            abstractState == gpu::AbstractState::ProgramRead
                && activeQueue_ == gpu::QueueClass::Compute
            ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            : NativeState(abstractState);
        if (activePackage_ != nullptr
            && PackageResource(*activePackage_, view.resource).lifetime
                == gpu::ResourceLifetimeClass::Persistent
            && IsReadOnlyState(abstractState))
        {
            const auto envelope = persistentReadStates_.find(view.resource);
            if (envelope != persistentReadStates_.end()) target = envelope->second;
        }

        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        for (UINT plane = view.textureRange.firstPlane;
             plane < static_cast<UINT>(view.textureRange.firstPlane
                 + view.textureRange.planeCount); ++plane)
        {
            for (UINT layer = view.textureRange.firstArrayLayer;
                 layer < static_cast<UINT>(view.textureRange.firstArrayLayer
                     + view.textureRange.arrayLayerCount); ++layer)
            {
                for (UINT mip = view.textureRange.firstMip;
                     mip < static_cast<UINT>(view.textureRange.firstMip
                         + view.textureRange.mipCount); ++mip)
                {
                    const UINT subresource = SubresourceIndex(
                        mip, layer, plane, mipLevels, arraySize);
                    auto& current = record->subresourceStates.at(subresource);
                    if (current == target) continue;
                    D3D12_RESOURCE_BARRIER barrier{};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrier.Transition.pResource = record->resource.Get();
                    barrier.Transition.Subresource = subresource;
                    barrier.Transition.StateBefore = current;
                    barrier.Transition.StateAfter = target;
                    barriers.push_back(barrier);
                    current = target;
                }
            }
        }
        if (!barriers.empty())
        {
            commandList_->ResourceBarrier(
                static_cast<UINT>(barriers.size()), barriers.data());
        }
        const bool uniform = std::all_of(
            record->subresourceStates.begin(),
            record->subresourceStates.end(),
            [&](D3D12_RESOURCE_STATES state){ return state == target; });
        if (uniform) record->state = target;
    }

    bool Backend::PackageRangeIsCommon(
        const compiler::NormalizedResourceView& view,
        std::uint32_t frameLag) const
    {
        if (view.kind == gpu::ResourceKind::Presentation)
        {
            return false;
        }
        const auto* record = ResolveResource(view.resource, frameLag);
        if (record == nullptr)
        {
            return false;
        }
        if (view.kind == gpu::ResourceKind::Buffer)
        {
            return record->state == D3D12_RESOURCE_STATE_COMMON;
        }

        const auto description = record->resource->GetDesc();
        const UINT mipLevels = description.MipLevels;
        const UINT arraySize = description.Dimension
                == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1u : description.DepthOrArraySize;
        const auto stateAt = [&](UINT subresource)
        {
            return record->subresourceStates.empty()
                ? record->state
                : record->subresourceStates.at(subresource);
        };
        for (UINT plane = view.textureRange.firstPlane;
             plane < static_cast<UINT>(view.textureRange.firstPlane
                 + view.textureRange.planeCount); ++plane)
        {
            for (UINT layer = view.textureRange.firstArrayLayer;
                 layer < static_cast<UINT>(view.textureRange.firstArrayLayer
                     + view.textureRange.arrayLayerCount); ++layer)
            {
                for (UINT mip = view.textureRange.firstMip;
                     mip < static_cast<UINT>(view.textureRange.firstMip
                         + view.textureRange.mipCount); ++mip)
                {
                    if (stateAt(SubresourceIndex(
                            mip, layer, plane, mipLevels, arraySize))
                        != D3D12_RESOURCE_STATE_COMMON)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void Backend::ValidateCopyQueueRequirement(
        const compiler::NormalizedResourceView& view,
        gpu::AbstractState state,
        std::uint32_t frameLag)
    {
        if (activeQueue_ != gpu::QueueClass::Copy)
        {
            throw std::runtime_error(
                "Copy requirement validation ran on a non-Copy queue.");
        }
        if (state != gpu::AbstractState::TransferRead
            && state != gpu::AbstractState::TransferWrite)
        {
            throw std::runtime_error(
                "Copy operation requires a non-copy resource state.");
        }
        if (!PackageRangeIsCommon(view, frameLag))
        {
            throw std::runtime_error(
                "Copy queue resource did not arrive in COMMON. The compiled "
                "operation stream must release the exact view before Copy "
                "queue execution.");
        }
        // COPY_SOURCE/COPY_DEST are obtained through implicit promotion from
        // COMMON. Copy-list completion decays the resource back to COMMON, so
        // no transition barrier or tracked-state mutation is emitted here.
    }


    void Backend::ExecutePackageRaster(
        const compiler::CompiledRasterCommand& command,
        FrameContext& frame)
    {
        const auto pipeline = pipelines_.find(command.executable);
        const auto program = programs_.find(command.program);
        if (pipeline == pipelines_.end() || program == programs_.end())
            throw std::runtime_error("Package raster command is not prepared.");
        commandList_->SetPipelineState(pipeline->second.Get());
        commandList_->SetGraphicsRootSignature(program->second.rootSignature.Get());
        commandList_->IASetPrimitiveTopology(
            NativeTopology(command.executable.rasterState.topology));

        const D3D12_VIEWPORT viewport{
            command.viewport.width == 0.0f ? 0.0f : command.viewport.x,
            command.viewport.height == 0.0f ? 0.0f : command.viewport.y,
            command.viewport.width == 0.0f ? static_cast<float>(width_) : command.viewport.width,
            command.viewport.height == 0.0f ? static_cast<float>(height_) : command.viewport.height,
            command.viewport.minimumDepth,
            command.viewport.maximumDepth};
        const D3D12_RECT scissor{
            command.scissor.right == 0 ? 0 : command.scissor.left,
            command.scissor.bottom == 0 ? 0 : command.scissor.top,
            command.scissor.right == 0 ? static_cast<LONG>(width_) : command.scissor.right,
            command.scissor.bottom == 0 ? static_cast<LONG>(height_) : command.scissor.bottom};
        commandList_->RSSetViewports(1, &viewport);
        commandList_->RSSetScissorRects(1, &scissor);

        for (const auto& stream : command.vertexStreams)
        {
            const auto native = BuildPackageVertexView(stream.view);
            commandList_->IASetVertexBuffers(stream.inputSlot, 1, &native);
        }
        if (command.indexStream)
        {
            const auto native = BuildPackageIndexView(*command.indexStream);
            commandList_->IASetIndexBuffer(&native);
        }
        BindPackageResources(command.program, command.bindings, frame, false);

        std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> colors;
        colors.reserve(command.colorAttachments.size());
        for (const auto& color : command.colorAttachments)
            colors.push_back(ResolvePackageAttachmentDescriptor(color, false));
        D3D12_CPU_DESCRIPTOR_HANDLE depth{};
        const D3D12_CPU_DESCRIPTOR_HANDLE* depthPointer = nullptr;
        if (command.depthAttachment)
        {
            depth = ResolvePackageAttachmentDescriptor(
                *command.depthAttachment, true);
            depthPointer = &depth;
        }
        commandList_->OMSetRenderTargets(
            static_cast<UINT>(colors.size()),
            colors.data(), FALSE, depthPointer);

        if (command.clear.clearColor
            || command.clear.colorLoad == ir::AttachmentLoadOperation::Clear)
        {
            for (const auto handle : colors)
                commandList_->ClearRenderTargetView(
                    handle, command.clear.color.data(), 0, nullptr);
        }
        if (depthPointer != nullptr
            && (command.clear.clearDepth
                || command.clear.depthLoad
                    == ir::AttachmentLoadOperation::Clear))
        {
            commandList_->ClearDepthStencilView(
                depth,
                D3D12_CLEAR_FLAG_DEPTH,
                command.clear.depth,
                0,
                0,
                nullptr);
        }

        if (command.indexStream && command.indexCount != 0)
        {
            commandList_->DrawIndexedInstanced(
                command.indexCount,
                command.instanceCount,
                command.indexStream->firstIndex,
                command.indexStream->baseVertex,
                command.firstInstance);
        }
        else
        {
            commandList_->DrawInstanced(
                command.vertexCount,
                command.instanceCount,
                command.firstVertex,
                command.firstInstance);
        }

        const auto discardView = [&](const compiler::NormalizedResourceView& view)
        {
            if (view.kind == gpu::ResourceKind::Presentation)
            {
                return;
            }
            auto* record = ResolveResource(view.resource);
            if (record == nullptr)
            {
                return;
            }
            const auto description = record->resource->GetDesc();
            const UINT mipLevels = description.MipLevels;
            const UINT arraySize = description.Dimension
                    == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                ? 1u : description.DepthOrArraySize;
            for (UINT plane = view.textureRange.firstPlane;
                 plane < static_cast<UINT>(view.textureRange.firstPlane
                     + view.textureRange.planeCount); ++plane)
            {
                for (UINT layer = view.textureRange.firstArrayLayer;
                     layer < static_cast<UINT>(view.textureRange.firstArrayLayer
                         + view.textureRange.arrayLayerCount); ++layer)
                {
                    for (UINT mip = view.textureRange.firstMip;
                         mip < static_cast<UINT>(view.textureRange.firstMip
                             + view.textureRange.mipCount); ++mip)
                    {
                        D3D12_DISCARD_REGION region{};
                        region.FirstSubresource = SubresourceIndex(
                            mip, layer, plane, mipLevels, arraySize);
                        region.NumSubresources = 1;
                        commandList_->DiscardResource(
                            record->resource.Get(), &region);
                    }
                }
            }
        };
        if (command.clear.colorStore == ir::AttachmentStoreOperation::Discard)
        {
            for (const auto& color : command.colorAttachments)
            {
                discardView(color);
            }
        }
        if (command.depthAttachment
            && command.clear.depthStore
                == ir::AttachmentStoreOperation::Discard)
        {
            discardView(*command.depthAttachment);
        }
    }

    void Backend::ExecutePackageCompute(
        const compiler::CompiledComputeCommand& command,
        FrameContext& frame)
    {
        const auto pipeline = pipelines_.find(command.executable);
        const auto program = programs_.find(command.program);
        if (pipeline == pipelines_.end() || program == programs_.end())
            throw std::runtime_error("Package compute command is not prepared.");
        commandList_->SetPipelineState(pipeline->second.Get());
        commandList_->SetComputeRootSignature(program->second.rootSignature.Get());
        BindPackageResources(command.program, command.bindings, frame, true);
        commandList_->Dispatch(
            command.groupCountX, command.groupCountY, command.groupCountZ);
    }

    void Backend::ExecutePackageCopy(
        const compiler::CompiledCopyCommand& command)
    {
        auto* source = ResolveResource(
            command.source.resource, command.sourceFrameLag);
        auto* destination = ResolveResource(
            command.destination.resource, command.destinationFrameLag);
        if (source == nullptr || destination == nullptr)
            throw std::runtime_error("Package copy references no physical resource.");

        if (command.source.kind == gpu::ResourceKind::Buffer
            && command.destination.kind == gpu::ResourceKind::Buffer)
        {
            commandList_->CopyBufferRegion(
                destination->resource.Get(),
                command.destination.byteOffset,
                source->resource.Get(),
                command.source.byteOffset,
                std::min(command.source.byteSize, command.destination.byteSize));
            return;
        }
        if (!IsTextureKind(command.source.kind)
            || !IsTextureKind(command.destination.kind))
        {
            throw std::runtime_error(
                "Buffer/texture conversion copies are not represented in V1.");
        }

        const auto sourceDesc = source->resource->GetDesc();
        const auto destinationDesc = destination->resource->GetDesc();
        const UINT sourceArrays = sourceDesc.Dimension
                == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1u : sourceDesc.DepthOrArraySize;
        const UINT destinationArrays = destinationDesc.Dimension
                == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? 1u : destinationDesc.DepthOrArraySize;
        const auto& sr = command.source.textureRange;
        const auto& dr = command.destination.textureRange;
        if (sr.mipCount != dr.mipCount
            || sr.arrayLayerCount != dr.arrayLayerCount
            || sr.planeCount != dr.planeCount)
            throw std::runtime_error("Texture copy ranges have different shapes.");
        for (UINT plane = 0; plane < sr.planeCount; ++plane)
        {
            for (UINT layer = 0; layer < sr.arrayLayerCount; ++layer)
            {
                for (UINT mip = 0; mip < sr.mipCount; ++mip)
                {
                    D3D12_TEXTURE_COPY_LOCATION from{};
                    from.pResource = source->resource.Get();
                    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    from.SubresourceIndex = SubresourceIndex(
                        sr.firstMip + mip,
                        sr.firstArrayLayer + layer,
                        sr.firstPlane + plane,
                        sourceDesc.MipLevels,
                        sourceArrays);
                    D3D12_TEXTURE_COPY_LOCATION to{};
                    to.pResource = destination->resource.Get();
                    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    to.SubresourceIndex = SubresourceIndex(
                        dr.firstMip + mip,
                        dr.firstArrayLayer + layer,
                        dr.firstPlane + plane,
                        destinationDesc.MipLevels,
                        destinationArrays);
                    commandList_->CopyTextureRegion(
                        &to, 0, 0, 0, &from, nullptr);
                }
            }
        }
    }

    void Backend::BindExternalResources(
        const compiler::CompiledRenderPackage& package,
        const runtime::FrameInvocation& invocation)
    {
        for (const auto& slot : package.externalSlots)
        {
            if (slot.kind == gpu::ResourceKind::Presentation) continue;
            const auto binding = invocation.externalResources.find(
                slot.resource);
            if (binding == invocation.externalResources.end()
                || binding->second.resource == nullptr)
            {
                throw runtime::BackendFailure(
                    runtime::BackendFailureKind::ExternalRebindRequired,
                    "A required external resource was not rebound.");
            }
            auto native = std::dynamic_pointer_cast<D3D12ExternalResource>(
                binding->second.resource);
            if (!native || native->resource == nullptr)
            {
                throw runtime::BackendFailure(
                    runtime::BackendFailureKind::InvalidPackage,
                    "External resource backend type is not D3D12.");
            }
            if (binding->second.availableAfter)
            {
                binding->second.availableAfter->Wait();
            }
            ComPtr<ID3D12Device> owner;
            ThrowIfFailed(native->resource->GetDevice(IID_PPV_ARGS(&owner)),
                "Query external resource device");
            if (owner.Get() != device_.Get())
            {
                throw runtime::BackendFailure(
                    runtime::BackendFailureKind::ExternalRebindRequired,
                    "External resource belongs to a different device epoch.");
            }

            const auto description = native->resource->GetDesc();
            if (slot.kind == gpu::ResourceKind::Buffer)
            {
                const auto& expected = std::get<ir::BufferDescription>(
                    slot.expectedDescription);
                if (description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                    || description.Width < expected.byteSize)
                {
                    throw runtime::BackendFailure(
                        runtime::BackendFailureKind::InvalidPackage,
                        "External buffer does not satisfy its package slot.");
                }
            }
            else
            {
                const auto& expected = std::get<ir::TextureDescription>(
                    slot.expectedDescription);
                const auto expectedNative = TextureDescription(
                    expected, width_, height_);
                const auto typeless = NativeTypelessFormat(expected.format);
                if (description.Dimension != expectedNative.Dimension
                    || description.Width != expectedNative.Width
                    || description.Height != expectedNative.Height
                    || description.DepthOrArraySize
                        != expectedNative.DepthOrArraySize
                    || description.MipLevels != expectedNative.MipLevels
                    || description.SampleDesc.Count
                        != expectedNative.SampleDesc.Count
                    || (description.Format != expectedNative.Format
                        && description.Format != typeless))
                {
                    throw runtime::BackendFailure(
                        runtime::BackendFailureKind::InvalidPackage,
                        "External texture does not satisfy its package slot.");
                }
            }

            auto& instances = resources_[slot.resource];
            if (instances.size() != FrameCount) instances.resize(FrameCount);
            auto& record = instances.at(frameIndex_);
            if (record.resource
                && record.resource.Get() != native->resource.Get())
            {
                // Descriptor contents may still be consumed by older frames.
                // V1 permits rebinding, but performs the safe conservative
                // recycle until a dedicated frame-local descriptor arena is
                // introduced.
                WaitIdle();
                packageShaderViews_.clear();
                packageAttachmentViews_.clear();
                nextRtvDescriptor_ = FrameCount;
                nextDsvDescriptor_ = 0;
                nextShaderDescriptor_ = 0;
            }
            record = {};
            record.resource = native->resource;
            const auto incoming = binding->second.incomingState
                    == gpu::AbstractState::Undefined
                ? slot.firstRequiredState
                : binding->second.incomingState;
            record.state = incoming == gpu::AbstractState::Undefined
                ? D3D12_RESOURCE_STATE_COMMON
                : NativeState(incoming);
        }
    }

    void Backend::ReleaseExternalResources(
        const compiler::CompiledRenderPackage& package,
        const runtime::FrameInvocation& invocation,
        FrameContext& frame,
        std::array<UINT64, QueueClassCount>& finalSignals)
    {
        bool hasRelease = false;
        for (const auto& slot : package.externalSlots)
        {
            if (slot.kind == gpu::ResourceKind::Presentation) continue;
            const auto binding = invocation.externalResources.find(slot.resource);
            if (binding == invocation.externalResources.end()) continue;
            const auto target = binding->second.outgoingState
                    == gpu::AbstractState::Undefined
                ? slot.lastRequiredState
                : binding->second.outgoingState;
            if (target != gpu::AbstractState::Undefined) hasRelease = true;
        }
        if (!hasRelease) return;

        for (const auto queue : {gpu::QueueClass::Compute,
                                 gpu::QueueClass::Copy})
        {
            const auto value = finalSignals[QueueIndex(queue)];
            if (value != 0)
                WaitForQueueValue(gpu::QueueClass::Direct, queue, value);
        }
        commandList_ = AcquireCommandList(frame, gpu::QueueClass::Direct);
        activeQueue_ = gpu::QueueClass::Direct;
        for (const auto& slot : package.externalSlots)
        {
            if (slot.kind == gpu::ResourceKind::Presentation) continue;
            const auto binding = invocation.externalResources.find(slot.resource);
            if (binding == invocation.externalResources.end()) continue;
            const auto target = binding->second.outgoingState
                    == gpu::AbstractState::Undefined
                ? slot.lastRequiredState
                : binding->second.outgoingState;
            if (target != gpu::AbstractState::Undefined)
                Transition(slot.resource, target);
        }
        ThrowIfFailed(commandList_->Close(),
            "Close external resource release list");
        ID3D12CommandList* lists[] = {commandList_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const UINT64 value = SignalQueue(gpu::QueueClass::Direct);
        finalSignals[QueueIndex(gpu::QueueClass::Direct)] = value;
        frame.queues[QueueIndex(gpu::QueueClass::Direct)]
            .completionFenceValue = value;
    }

    runtime::FrameSubmission Backend::ExecutePackageFrame(
        const compiler::CompiledRenderPackage& package,
        const runtime::FrameInvocation& invocation)
    {
        activePackage_ = &package;
        activeInvocation_ = &invocation;
        try
        {
            ResizeIfNeeded();
            if (width_ == 0 || height_ == 0)
            {
                activePackage_ = nullptr;
                activeInvocation_ = nullptr;
                return {.deviceEpoch = deviceEpoch_};
            }

            EnsurePackageCompiled(package);
            UINT64 invocationUploadBytes = 0;
            for (const auto& [resource, bytes] :
                 invocation.dynamicResourceData)
            {
                (void)resource;
                invocationUploadBytes += AlignUp(
                    bytes.size(),
                    D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            }
            EnsureUploadCapacity(invocationUploadBytes);

            activeFrameSlot_ = static_cast<std::uint32_t>(
                frameNumber_ % FrameCount);
            BindExternalResources(package, invocation);
            FrameContext& frame = frames_[activeFrameSlot_];
            BeginFrame(frame);
            ValidateDebugLayer();

            struct WorkSignal
            {
                gpu::QueueClass queue = gpu::QueueClass::Direct;
                UINT64 value = 0;
            };
            std::vector<WorkSignal> signals(
                package.statistics.workCount);
            std::array<UINT64, QueueClassCount> finalSignals{};

            bool commandListOpen = false;
            std::size_t currentWorkIndex = 0;
            gpu::QueueClass currentQueue = gpu::QueueClass::Direct;
            std::string currentWorkName;

            const auto requireOpen = [&](const char* operation)
            {
                if (!commandListOpen)
                {
                    throw std::runtime_error(
                        std::string("Compiled operation '") + operation
                        + "' requires an active work command list.");
                }
            };
            const auto requireClosed = [&](const char* operation)
            {
                if (commandListOpen)
                {
                    throw std::runtime_error(
                        std::string("Compiled operation '") + operation
                        + "' was emitted before the current work was submitted.");
                }
            };

            for (std::size_t operationIndex = 0;
                 operationIndex < package.operations.size();
                 ++operationIndex)
            {
                lastOperationIndex_ = operationIndex;
                const auto& operation = package.operations[operationIndex];
                std::visit([&](const auto& value)
                {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<
                        T, compiler::WaitForWorkOperation>)
                    {
                        requireClosed("WaitForWork");
                        if (value.signalWorkIndex >= signals.size())
                        {
                            throw std::runtime_error(
                                "Compiled wait references an invalid work index.");
                        }
                        const auto& signal = signals[value.signalWorkIndex];
                        if (signal.value == 0
                            || signal.queue != value.signalQueue)
                        {
                            throw std::runtime_error(
                                "Compiled wait references an unsignaled work.");
                        }
                        WaitForQueueValue(
                            value.waitingQueue,
                            value.signalQueue,
                            signal.value);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::WaitForTemporalOperation>)
                    {
                        requireClosed("WaitForTemporal");
                        WaitForTemporalAccess(
                            value.access, value.waitingQueue);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::BeginWorkOperation>)
                    {
                        requireClosed("BeginWork");
                        if (value.workIndex >= signals.size())
                        {
                            throw std::runtime_error(
                                "Compiled BeginWork index exceeds package work count.");
                        }
                        currentWorkIndex = value.workIndex;
                        currentQueue = value.queue;
                        currentWorkName = value.name;
                        lastWorkName_ = value.name;
                        commandList_ = AcquireCommandList(frame, currentQueue);
                        activeQueue_ = currentQueue;
                        if (currentQueue != gpu::QueueClass::Copy)
                        {
                            ID3D12DescriptorHeap* heaps[] = {
                                shaderHeap_.Get()};
                            commandList_->SetDescriptorHeaps(1, heaps);
                        }
                        commandListOpen = true;
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::ActivateAliasOperation>)
                    {
                        requireOpen("ActivateAlias");
                        ActivateAliasedResource(
                            value.resource, value.frameLag);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::TransitionOperation>)
                    {
                        requireOpen("Transition");
                        if (currentQueue == gpu::QueueClass::Copy)
                        {
                            throw std::runtime_error(
                                "Compiled stream attempted a transition on a Copy command list.");
                        }
                        TransitionPackageRange(
                            value.view, value.state, value.frameLag);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::RequireCommonOperation>)
                    {
                        if (value.cyclicReuse)
                        {
                            requireClosed("CyclicRequireCommon");
                            if (!PackageRangeIsCommon(
                                    value.view, value.frameLag))
                            {
                                throw std::runtime_error(
                                    "FrameLocal physical instance was recycled before its previous cycle released the exact state cell to COMMON.");
                            }
                            return;
                        }

                        requireOpen("RequireCommon");
                        if (currentQueue != gpu::QueueClass::Copy)
                        {
                            throw std::runtime_error(
                                "A non-cyclic RequireCommon operation must execute inside a Copy work.");
                        }
                        ValidateCopyQueueRequirement(
                            value.view,
                            value.implicitCopyState,
                            value.frameLag);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::ExecuteCommandOperation>)
                    {
                        requireOpen("ExecuteCommand");
                        std::visit([&](const auto& command)
                        {
                            using C = std::decay_t<decltype(command)>;
                            if constexpr (std::is_same_v<
                                C, compiler::CompiledRasterCommand>)
                            {
                                if (currentQueue != gpu::QueueClass::Direct)
                                {
                                    throw std::runtime_error(
                                        "Raster command was assigned to a non-Direct queue.");
                                }
                                ExecutePackageRaster(command, frame);
                            }
                            else if constexpr (std::is_same_v<
                                C, compiler::CompiledComputeCommand>)
                            {
                                if (currentQueue == gpu::QueueClass::Copy)
                                {
                                    throw std::runtime_error(
                                        "Compute command was assigned to the Copy queue.");
                                }
                                ExecutePackageCompute(command, frame);
                            }
                            else if constexpr (std::is_same_v<
                                C, compiler::CompiledCopyCommand>)
                            {
                                ExecutePackageCopy(command);
                            }
                            else
                            {
                                // Presentation is committed once after the
                                // operation stream has submitted all works.
                            }
                        }, value.command);
                    }
                    else if constexpr (std::is_same_v<
                        T, compiler::SubmitWorkOperation>)
                    {
                        requireOpen("SubmitWork");
                        if (value.workIndex != currentWorkIndex
                            || value.queue != currentQueue)
                        {
                            throw std::runtime_error(
                                "Compiled SubmitWork does not match the active work.");
                        }

                        const HRESULT closeResult = commandList_->Close();
                        if (FAILED(closeResult))
                        {
                            throw std::runtime_error(
                                "Close compiled command list for work '"
                                + currentWorkName + "' failed, HRESULT="
                                + std::to_string(
                                    static_cast<unsigned long>(closeResult)));
                        }
                        ID3D12CommandList* lists[] = {commandList_.Get()};
                        QueueFor(currentQueue)->ExecuteCommandLists(1, lists);
                        const UINT64 signalValue = SignalQueue(currentQueue);
                        finalSignals[QueueIndex(currentQueue)] = signalValue;
                        signals[currentWorkIndex] = {
                            currentQueue, signalValue};
                        frame.queues[QueueIndex(currentQueue)]
                            .completionFenceValue = signalValue;
                        for (const auto& access : value.temporalAccesses)
                        {
                            RecordTemporalAccess(
                                access, currentQueue, signalValue);
                        }
                        commandListOpen = false;
                        currentWorkName.clear();
                    }
                }, operation);
            }

            if (commandListOpen)
            {
                throw std::runtime_error(
                    "Compiled operation stream ended with an open work.");
            }

            ReleaseExternalResources(
                package, invocation, frame, finalSignals);

            ThrowIfFailed(swapChain_->Present(1, 0),
                "Present compiled operation stream");
            frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
            ++frameNumber_;
            activePackage_ = nullptr;
            activeInvocation_ = nullptr;
            runtime::FrameSubmission submission;
            submission.deviceEpoch = deviceEpoch_;
            constexpr std::array queues{
                gpu::QueueClass::Direct,
                gpu::QueueClass::Compute,
                gpu::QueueClass::Copy};
            for (const auto queue : queues)
            {
                const auto index = QueueIndex(queue);
                submission.queues[index].queue = queue;
                submission.queues[index].value = finalSignals[index];
                if (finalSignals[index] != 0)
                {
                    submission.queues[index].completion =
                        std::make_shared<D3D12QueueCompletion>(
                            SyncFor(queue).fence, finalSignals[index]);
                }
            }
            return submission;
        }
        catch (...)
        {
            activePackage_ = nullptr;
            activeInvocation_ = nullptr;
            throw;
        }
    }

    runtime::FrameSubmission Backend::Execute(
        const compiler::CompiledRenderPackage& package,
        const runtime::FrameInvocation& invocation)
    {
        try
        {
            return ExecutePackageFrame(package, invocation);
        }
        catch (const runtime::BackendFailure&)
        {
            throw;
        }
        catch (const std::exception& error)
        {
            const HRESULT removedReason = device_ == nullptr
                ? DXGI_ERROR_DEVICE_REMOVED
                : device_->GetDeviceRemovedReason();
            if (SUCCEEDED(removedReason)) throw;

            WriteDeviceRemovedDiagnostics(
                DXGI_ERROR_DEVICE_REMOVED, removedReason, error.what());
            if (!configuration_.enableDeviceRecovery)
            {
                throw runtime::BackendFailure(
                    runtime::BackendFailureKind::DeviceRemoved,
                    "D3D12 device was removed.",
                    DXGI_ERROR_DEVICE_REMOVED,
                    removedReason);
            }

            RecreateDeviceObjects();
            const bool needsExternalRebind = std::any_of(
                package.externalSlots.begin(),
                package.externalSlots.end(),
                [](const compiler::ExternalResourceSlot& slot)
                {
                    return slot.kind != gpu::ResourceKind::Presentation;
                });
            if (needsExternalRebind)
            {
                throw runtime::BackendFailure(
                    runtime::BackendFailureKind::ExternalRebindRequired,
                    "The device was recreated; external resources must be rebound.",
                    DXGI_ERROR_DEVICE_REMOVED,
                    removedReason);
            }
            return ExecutePackageFrame(package, invocation);
        }
    }

}
