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
        return package.canonicalModule.Resource(id);
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
            if (blueprint.declaration.Kind() == gpu::ResourceKind::Buffer
                && blueprint.declaration.lifetime
                    != gpu::ResourceLifetimeClass::External
                && blueprint.declaration.update
                    != gpu::ResourceUpdateClass::CpuUpdated
                && !blueprint.declaration.data.empty())
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
            const auto byteCount = blueprint->declaration.data.size();
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
                blueprint->declaration.data.data(),
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

    void Backend::InitializePackagePersistentReadStates(
        const compiler::CompiledRenderPackage& package)
    {
        persistentReadStates_.clear();
        for (const auto& envelope : package.plan.persistentReadStates)
        {
            UINT nativeBits = 0;
            for (const auto state : envelope.states)
            {
                nativeBits |= static_cast<UINT>(NativeState(state));
            }
            if (nativeBits == 0
                || !persistentReadStates_.emplace(
                    envelope.resource,
                    static_cast<D3D12_RESOURCE_STATES>(nativeBits)).second)
            {
                throw std::runtime_error(
                    "Invalid package persistent read-state envelope.");
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
            && *compiledStructureHash_ == package.packageHash)
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

        auto plan = package.plan;
        plan.structureHash = package.packageHash;
        // Package materialization starts every non-external resource in COMMON.
        // Persistent read envelopes are installed only after canonical initial
        // data has been uploaded.
        plan.persistentReadStates.clear();

        // Materialization still reuses the established EnsureCompiled path,
        // but its PSO warm-up must use the package-native ProgramInterface.
        // A compatibility executable may intentionally contain only one
        // representative vertex stream and would fail shader-signature
        // validation for a generalized program before we can replace it.
        plan.executables.clear();
        for (const auto& work : package.works)
        {
            std::visit([&](const auto& command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<T,
                    compiler::CompiledRasterCommand>
                    || std::is_same_v<T,
                        compiler::CompiledComputeCommand>)
                {
                    if (std::find(
                            plan.executables.begin(),
                            plan.executables.end(),
                            command.executable) == plan.executables.end())
                    {
                        plan.executables.push_back(command.executable);
                    }
                }
            }, work.command);
        }
        std::unordered_set<gpu::ResourceId,
            foundation::StrongIdHash<gpu::ResourceTag>> typelessResources;
        for (const auto& work : package.works)
        {
            for (const auto& requirement : work.rangeStates)
            {
                if (!IsTextureKind(requirement.view.kind))
                {
                    continue;
                }
                const auto& declaration = PackageResource(
                    package, requirement.view.resource);
                if (requirement.view.format != declaration.Format())
                {
                    typelessResources.insert(requirement.view.resource);
                }
            }
        }

        struct OptimizedClearCandidate
        {
            bool assigned = false;
            bool conflict = false;
            bool depth = false;
            D3D12_CLEAR_VALUE value{};
        };
        std::unordered_map<gpu::ResourceId, OptimizedClearCandidate,
            foundation::StrongIdHash<gpu::ResourceTag>> optimizedClears;
        const auto sameClear = [](const OptimizedClearCandidate& candidate,
                                  const D3D12_CLEAR_VALUE& value,
                                  bool depth)
        {
            if (!candidate.assigned || candidate.depth != depth
                || candidate.value.Format != value.Format)
            {
                return false;
            }
            if (depth)
            {
                return candidate.value.DepthStencil.Depth
                        == value.DepthStencil.Depth
                    && candidate.value.DepthStencil.Stencil
                        == value.DepthStencil.Stencil;
            }
            return std::equal(
                std::begin(candidate.value.Color),
                std::end(candidate.value.Color),
                std::begin(value.Color));
        };
        const auto recordOptimizedClear = [&](
            const compiler::NormalizedResourceView& view,
            const ir::ClearDescription& clear,
            bool depth)
        {
            const auto& declaration = PackageResource(package, view.resource);
            if (declaration.lifetime
                    == gpu::ResourceLifetimeClass::External
                || typelessResources.contains(view.resource))
            {
                return;
            }

            D3D12_CLEAR_VALUE value{};
            value.Format = NativeFormat(view.format);
            if (depth)
            {
                value.DepthStencil.Depth = clear.depth;
                value.DepthStencil.Stencil = 0;
            }
            else
            {
                std::copy(
                    clear.color.begin(),
                    clear.color.end(),
                    std::begin(value.Color));
            }

            auto& candidate = optimizedClears[view.resource];
            if (!candidate.assigned)
            {
                candidate.assigned = true;
                candidate.depth = depth;
                candidate.value = value;
            }
            else if (!sameClear(candidate, value, depth))
            {
                candidate.conflict = true;
            }
        };
        for (const auto& work : package.works)
        {
            const auto* raster = std::get_if<
                compiler::CompiledRasterCommand>(&work.command);
            if (raster == nullptr)
            {
                continue;
            }
            if (raster->clear.clearColor
                || raster->clear.colorLoad
                    == ir::AttachmentLoadOperation::Clear)
            {
                for (const auto& color : raster->colorAttachments)
                {
                    recordOptimizedClear(color, raster->clear, false);
                }
            }
            if (raster->depthAttachment
                && (raster->clear.clearDepth
                    || raster->clear.depthLoad
                        == ir::AttachmentLoadOperation::Clear))
            {
                recordOptimizedClear(
                    *raster->depthAttachment, raster->clear, true);
            }
        }

        auto materializationModule = package.canonicalModule;
        for (auto& resource : materializationModule.resources)
        {
            if (resource.Kind() == gpu::ResourceKind::Buffer
                && resource.lifetime
                    != gpu::ResourceLifetimeClass::External
                && resource.update
                    != gpu::ResourceUpdateClass::CpuUpdated
                && !resource.data.empty())
            {
                // The legacy materializer otherwise uploads into COPY_DEST and
                // leaves a guessed Vertex/Constant state. Package execution
                // owns initialization and keeps buffers in COMMON afterward.
                resource.data.clear();
            }

            auto* texture = std::get_if<ir::TextureDescription>(
                &resource.description);
            if (texture != nullptr
                && (texture->arrayLayers > 1
                    || texture->sampleCount > 1
                    || typelessResources.contains(resource.id)))
            {
                // The established CreateTexture path creates legacy whole-
                // resource descriptors with dimensions that are not valid for
                // every generalized texture. Package descriptors are created
                // lazily from normalized views instead.
                texture->usage = ir::TextureUsage::None;
            }
        }
        EnsureCompiled(materializationModule, plan);

        // Recreate every package texture from its canonical description. This
        // removes legacy whole-resource descriptors and, importantly, does not
        // attach an optimized clear value when the IR has no stable resource-
        // level clear contract. Placed resources keep their alias allocation.
        for (const auto& blueprint : package.resources)
        {
            const auto* texture = std::get_if<ir::TextureDescription>(
                &blueprint.declaration.description);
            if (texture == nullptr)
            {
                continue;
            }

            auto found = resources_.find(blueprint.declaration.id);
            if (found == resources_.end())
            {
                throw std::runtime_error(
                    "Package texture recreation found no materialized record.");
            }

            auto nativeDescription = TextureDescription(
                *texture, width_, height_);
            if (typelessResources.contains(blueprint.declaration.id))
            {
                nativeDescription.Format = NativeTypelessFormat(
                    texture->format);
            }
            const auto properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
            const auto allocation = physicalAllocations_.find(
                blueprint.declaration.id);

            for (std::size_t instance = 0;
                 instance < found->second.size(); ++instance)
            {
                ID3D12Heap* heap = nullptr;
                if (allocation != physicalAllocations_.end())
                {
                    heap = allocationHeaps_.at(allocation->second)
                        .at(instance % FrameCount).Get();
                }

                ResourceRecord replacement;
                const auto clear = optimizedClears.find(
                    blueprint.declaration.id);
                const D3D12_CLEAR_VALUE* clearValue =
                    clear != optimizedClears.end()
                        && clear->second.assigned
                        && !clear->second.conflict
                        && !typelessResources.contains(
                            blueprint.declaration.id)
                    ? &clear->second.value
                    : nullptr;
                const HRESULT createResult = heap != nullptr
                    ? device_->CreatePlacedResource(
                        heap,
                        0,
                        &nativeDescription,
                        D3D12_RESOURCE_STATE_COMMON,
                        clearValue,
                        IID_PPV_ARGS(&replacement.resource))
                    : device_->CreateCommittedResource(
                        &properties,
                        D3D12_HEAP_FLAG_NONE,
                        &nativeDescription,
                        D3D12_RESOURCE_STATE_COMMON,
                        clearValue,
                        IID_PPV_ARGS(&replacement.resource));
                ThrowIfFailed(
                    createResult, "Create canonical package texture");
                found->second[instance] = std::move(replacement);
            }
        }

        UploadPackageInitialBufferData(package);
        InitializePackagePersistentReadStates(package);

        packageShaderViews_.clear();
        packageAttachmentViews_.clear();

        for (const auto& work : package.works)
        {
            std::visit([&](const auto& command)
            {
                using T = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<T,
                    compiler::CompiledRasterCommand>
                    || std::is_same_v<T,
                        compiler::CompiledComputeCommand>)
                {
                    pipelines_[command.executable] =
                        CreatePackagePipeline(command.executable);
                }
            }, work.command);
        }
        compiledPackageHash_ = package.packageHash;
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
            activePackage_->canonicalModule.Program(
                executable.program).programInterface,
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
        const compiler::RangeStateRequirement& requirement)
    {
        if (activeQueue_ != gpu::QueueClass::Copy)
        {
            throw std::runtime_error(
                "Copy requirement validation ran on a non-Copy queue.");
        }
        if (requirement.state != gpu::AbstractState::TransferRead
            && requirement.state != gpu::AbstractState::TransferWrite)
        {
            throw std::runtime_error(
                "Copy work requires a non-copy resource state.");
        }
        if (!PackageRangeIsCommon(
                requirement.view, requirement.frameLag))
        {
            throw std::runtime_error(
                "Copy queue resource did not arrive in COMMON. The producing "
                "Direct/Compute queue must release the exact view before the "
                "Copy queue waits and executes.");
        }
        // COPY_SOURCE/COPY_DEST are obtained through implicit promotion from
        // COMMON. Resources used on a Copy queue decay to COMMON when this
        // command list finishes execution, so the tracked state deliberately
        // remains COMMON and no transition barrier is recorded here.
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

    void Backend::Execute(
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
                return;
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
            FrameContext& frame = frames_[activeFrameSlot_];
            BeginFrame(frame);
            ValidateDebugLayer();

            struct WorkSignal
            {
                gpu::QueueClass queue = gpu::QueueClass::Direct;
                UINT64 value = 0;
            };
            std::vector<WorkSignal> signals(package.works.size());
            for (std::size_t workIndex = 0;
                 workIndex < package.works.size(); ++workIndex)
            {
                const auto& work = package.works[workIndex];
                for (const auto& access : work.accesses)
                    WaitForTemporalAccess(access, work.queue);
                for (const auto& handoff : package.queueHandoffs)
                {
                    if (handoff.acquireScheduledWork != workIndex) continue;
                    if (handoff.acquireQueue != work.queue)
                    {
                        throw std::runtime_error(
                            "Package queue handoff acquire queue does not match its work.");
                    }
                    const auto& signal = signals.at(handoff.releaseScheduledWork);
                    if (signal.value == 0
                        || signal.queue != handoff.releaseQueue)
                    {
                        throw std::runtime_error(
                            "Package queue handoff references an unsignaled release.");
                    }
                    WaitForQueueValue(work.queue, signal.queue, signal.value);
                }

                // BeginFrame has waited for every queue that previously used
                // this physical frame slot. A cyclic handoff therefore needs
                // no new GPU fence, but a Copy-first reuse must observe the
                // COMMON release emitted by the previous cycle's last user.
                for (const auto& handoff : package.cyclicFrameHandoffs)
                {
                    if (handoff.acquireScheduledWork != workIndex) continue;
                    if (handoff.acquireQueue != work.queue)
                    {
                        throw std::runtime_error(
                            "Cyclic frame handoff acquire queue does not match its work.");
                    }
                    if (handoff.requiresCommonRelease
                        && !PackageRangeIsCommon(handoff.acquireView))
                    {
                        throw std::runtime_error(
                            "FrameLocal physical instance was recycled before its "
                            "previous cycle released the Copy-first state cell to COMMON.");
                    }
                }

                commandList_ = AcquireCommandList(frame, work.queue);
                activeQueue_ = work.queue;
                if (work.queue != gpu::QueueClass::Copy)
                {
                    ID3D12DescriptorHeap* heaps[] = {shaderHeap_.Get()};
                    commandList_->SetDescriptorHeaps(1, heaps);
                }
                for (const auto& requirement : work.rangeStates)
                {
                    ActivateAliasedResource(
                        requirement.view.resource, requirement.frameLag);
                    if (work.queue == gpu::QueueClass::Copy)
                    {
                        ValidateCopyQueueRequirement(requirement);
                    }
                    else
                    {
                        TransitionPackageRange(
                            requirement.view,
                            requirement.state,
                            requirement.frameLag);
                    }
                }

                std::visit([&](const auto& command)
                {
                    using T = std::decay_t<decltype(command)>;
                    if constexpr (std::is_same_v<T,
                        compiler::CompiledRasterCommand>)
                        ExecutePackageRaster(command, frame);
                    else if constexpr (std::is_same_v<T,
                        compiler::CompiledComputeCommand>)
                        ExecutePackageCompute(command, frame);
                    else if constexpr (std::is_same_v<T,
                        compiler::CompiledCopyCommand>)
                        ExecutePackageCopy(command);
                }, work.command);

                for (const auto& handoff : package.queueHandoffs)
                {
                    if (handoff.releaseScheduledWork != workIndex) continue;
                    if (handoff.releaseQueue != work.queue)
                    {
                        throw std::runtime_error(
                            "Package queue handoff release queue does not match its work.");
                    }
                    if (work.queue == gpu::QueueClass::Copy)
                    {
                        if (!PackageRangeIsCommon(
                                handoff.releaseView, handoff.frameLag))
                        {
                            throw std::runtime_error(
                                "Copy queue handoff lost its implicit COMMON state.");
                        }
                        continue;
                    }
                    TransitionPackageRange(
                        handoff.releaseView,
                        gpu::AbstractState::Undefined,
                        handoff.frameLag);
                }
                for (const auto& handoff : package.cyclicFrameHandoffs)
                {
                    if (handoff.releaseScheduledWork != workIndex) continue;
                    if (handoff.releaseQueue != work.queue)
                    {
                        throw std::runtime_error(
                            "Cyclic frame handoff release queue does not match its work.");
                    }
                    if (!handoff.requiresCommonRelease) continue;
                    if (work.queue == gpu::QueueClass::Copy)
                    {
                        if (!PackageRangeIsCommon(handoff.releaseView))
                        {
                            throw std::runtime_error(
                                "Copy-final FrameLocal state cell did not decay to COMMON.");
                        }
                        continue;
                    }
                    TransitionPackageRange(
                        handoff.releaseView,
                        gpu::AbstractState::Undefined);
                }
                for (const auto& requirement : work.rangeStates)
                {
                    const auto& declaration = PackageResource(
                        package, requirement.view.resource);
                    if (declaration.lifetime
                            == gpu::ResourceLifetimeClass::Temporal
                        && work.queue != gpu::QueueClass::Copy)
                    {
                        TransitionPackageRange(
                            requirement.view,
                            gpu::AbstractState::Undefined,
                            requirement.frameLag);
                    }
                }

                ThrowIfFailed(commandList_->Close(),
                    "Close package command list");
                ID3D12CommandList* lists[] = {commandList_.Get()};
                QueueFor(work.queue)->ExecuteCommandLists(1, lists);
                signals[workIndex] = {
                    work.queue, SignalQueue(work.queue)};
                frame.queues[QueueIndex(work.queue)].completionFenceValue =
                    signals[workIndex].value;
                for (const auto& access : work.accesses)
                {
                    RecordTemporalAccess(
                        access, work.queue, signals[workIndex].value);
                }
            }

            ThrowIfFailed(swapChain_->Present(1, 0),
                "Present compiled package");
            frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
            ++frameNumber_;
            activePackage_ = nullptr;
            activeInvocation_ = nullptr;
        }
        catch (...)
        {
            activePackage_ = nullptr;
            activeInvocation_ = nullptr;
            throw;
        }
    }
}
