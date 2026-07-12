#include "07_D3D12Backend/D3D12BackendInternal.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace sge::d3d12::detail
{
    void Backend::EnsureCompiled(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan)
    {
        if (compiledStructureHash_
            && *compiledStructureHash_ == plan.structureHash)
        {
            return;
        }

        WaitIdle();

        resources_.clear();
        resourceInstancePlans_.clear();
        physicalAllocations_.clear();
        allocationHeaps_.clear();
        activeAliasedResources_.clear();
        temporalCompletions_.clear();
        programs_.clear();
        pipelines_.clear();
        presentationResource_.reset();
        nextRtvDescriptor_ = FrameCount;
        nextDsvDescriptor_ = 0;
        nextShaderDescriptor_ = 0;

        for (const auto& instance : plan.resourceInstances)
        {
            resourceInstancePlans_.emplace(instance.resource, instance);
            if (instance.lifetime == gpu::ResourceLifetimeClass::Temporal)
            {
                temporalCompletions_[instance.resource].assign(
                    instance.physicalInstanceCount, {});
            }
        }

        struct HeapRequirement
        {
            UINT64 size = 0;
            UINT64 alignment = 0;
            D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
        };
        std::unordered_map<
            gpu::PhysicalAllocationId,
            HeapRequirement,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>> requirements;

        for (const auto& lifetime : plan.lifetimes)
        {
            const auto& resource = module.Resource(lifetime.resource);
            if (resource.lifetime != gpu::ResourceLifetimeClass::FrameLocal
                || resource.update != gpu::ResourceUpdateClass::GpuProduced
                || (resource.Kind() != gpu::ResourceKind::Buffer
                    && resource.Kind() != gpu::ResourceKind::Texture1D
                    && resource.Kind() != gpu::ResourceKind::Texture2D
                    && resource.Kind() != gpu::ResourceKind::Texture3D))
            {
                continue;
            }
            D3D12_RESOURCE_DESC native{};
            D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
            if (resource.Kind() == gpu::ResourceKind::Buffer)
            {
                native = BufferDescription(resource.SizeBytes());
                const auto& buffer = std::get<ir::BufferDescription>(
                    resource.description);
                if ((static_cast<std::uint32_t>(buffer.usage)
                    & static_cast<std::uint32_t>(ir::BufferUsage::Storage)) != 0)
                {
                    native.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                }
                heapFlags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
            }
            else
            {
                native = TextureDescription(
                    std::get<ir::TextureDescription>(resource.description),
                    width_, height_);
                const auto& texture = std::get<ir::TextureDescription>(
                    resource.description);
                const auto usage = static_cast<std::uint32_t>(texture.usage);
                heapFlags = (usage
                    & (static_cast<std::uint32_t>(ir::TextureUsage::ColorAttachment)
                        | static_cast<std::uint32_t>(ir::TextureUsage::DepthAttachment))) != 0
                    ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES
                    : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            }
            const auto information = device_->GetResourceAllocationInfo(
                0, 1, &native);
            auto& requirement = requirements[lifetime.allocation];
            requirement.size = std::max(requirement.size, information.SizeInBytes);
            requirement.alignment = std::max(
                requirement.alignment, information.Alignment);
            requirement.flags = heapFlags;
            physicalAllocations_[resource.id] = lifetime.allocation;
        }

        for (const auto& [allocation, requirement] : requirements)
        {
            D3D12_HEAP_DESC description{};
            description.SizeInBytes = requirement.size;
            description.Alignment = requirement.alignment;
            description.Properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
            description.Flags = requirement.flags;

            std::vector<ComPtr<ID3D12Heap>> heaps(FrameCount);
            for (auto& heap : heaps)
            {
                ThrowIfFailed(device_->CreateHeap(
                    &description, IID_PPV_ARGS(&heap)), "Create alias heap");
            }
            allocationHeaps_.emplace(allocation, std::move(heaps));
            activeAliasedResources_[allocation].resize(FrameCount);
        }

        for (const auto& resource : module.resources)
        {
            if (resource.lifetime == gpu::ResourceLifetimeClass::External)
            {
                if (resource.Kind() == gpu::ResourceKind::Presentation)
                {
                    presentationResource_ = resource.id;
                }
                continue;
            }

            if (resource.update == gpu::ResourceUpdateClass::CpuUpdated)
            {
                continue;
            }

            const auto instancePlan = resourceInstancePlans_.find(resource.id);
            if (instancePlan == resourceInstancePlans_.end())
            {
                throw std::runtime_error(
                    "Resource has no compiled instance plan.");
            }
            auto& instances = resources_[resource.id];
            instances.reserve(instancePlan->second.physicalInstanceCount);

            for (std::uint32_t instance = 0;
                 instance < instancePlan->second.physicalInstanceCount;
                 ++instance)
            {
                if (resource.Kind() == gpu::ResourceKind::Texture1D
                    || resource.Kind() == gpu::ResourceKind::Texture2D
                    || resource.Kind() == gpu::ResourceKind::Texture3D)
                {
                    ID3D12Heap* heap = nullptr;
                    const auto allocation = physicalAllocations_.find(resource.id);
                    if (allocation != physicalAllocations_.end())
                    {
                        heap = allocationHeaps_.at(allocation->second)
                            .at(instance % FrameCount).Get();
                    }
                    instances.push_back(CreateTexture(resource, heap));
                }
                else if (resource.Kind() == gpu::ResourceKind::Buffer)
                {
                    ID3D12Heap* heap = nullptr;
                    const auto allocation = physicalAllocations_.find(resource.id);
                    if (allocation != physicalAllocations_.end())
                    {
                        heap = allocationHeaps_.at(allocation->second)
                            .at(instance % FrameCount).Get();
                    }
                    instances.push_back(CreateStaticBuffer(resource, heap));
                }
            }
        }

        for (const auto& program : module.programs)
        {
            programs_.emplace(program.id, CreateProgram(program));
        }

        for (const auto& executable : plan.executables)
        {
            pipelines_.emplace(
                executable,
                CreatePipeline(executable));
        }

        compiledStructureHash_ = plan.structureHash;
    }

    ResourceRecord Backend::CreateTexture(
        const ir::ResourceDeclaration& declaration,
        ID3D12Heap* aliasHeap)
    {
        const auto& texture =
            std::get<ir::TextureDescription>(declaration.description);
        const auto description = TextureDescription(texture, width_, height_);
        const auto properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);

        D3D12_CLEAR_VALUE clear{};
        D3D12_CLEAR_VALUE* clearPointer = nullptr;
        const auto usage = static_cast<std::uint32_t>(texture.usage);
        if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::ColorAttachment)) != 0)
        {
            clear.Format = description.Format;
            clear.Color[3] = 1.0f;
            clearPointer = &clear;
        }
        else if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::DepthAttachment)) != 0)
        {
            clear.Format = description.Format;
            clear.DepthStencil.Depth = 1.0f;
            clearPointer = &clear;
        }

        ResourceRecord record;
        const HRESULT createResult = aliasHeap != nullptr
            ? device_->CreatePlacedResource(
                aliasHeap,
                0,
                &description,
                D3D12_RESOURCE_STATE_COMMON,
                clearPointer,
                IID_PPV_ARGS(&record.resource))
            : device_->CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                D3D12_RESOURCE_STATE_COMMON,
                clearPointer,
                IID_PPV_ARGS(&record.resource));
        ThrowIfFailed(createResult, "Create texture resource");

        if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::ColorAttachment)) != 0)
        {
            record.rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
            record.rtv.ptr += static_cast<SIZE_T>(nextRtvDescriptor_++)
                * rtvIncrement_;
            device_->CreateRenderTargetView(
                record.resource.Get(), nullptr, record.rtv);
            record.hasRtv = true;
        }

        if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::DepthAttachment)) != 0)
        {
            record.dsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
            record.dsv.ptr += static_cast<SIZE_T>(nextDsvDescriptor_++)
                * dsvIncrement_;
            D3D12_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format = description.Format;
            view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            device_->CreateDepthStencilView(
                record.resource.Get(), &view, record.dsv);
            record.hasDsv = true;
        }

        const auto allocateShaderDescriptor = [&]()
        {
            auto cpu = shaderHeap_->GetCPUDescriptorHandleForHeapStart();
            auto gpu = shaderHeap_->GetGPUDescriptorHandleForHeapStart();
            cpu.ptr += static_cast<SIZE_T>(nextShaderDescriptor_)
                * shaderIncrement_;
            gpu.ptr += static_cast<UINT64>(nextShaderDescriptor_++)
                * shaderIncrement_;
            return std::pair{cpu, gpu};
        };

        if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::Sampled)) != 0)
        {
            const auto [cpu, gpu] = allocateShaderDescriptor();
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = description.Format;
            view.ViewDimension = texture.dimension == gpu::ResourceKind::Texture1D
                ? D3D12_SRV_DIMENSION_TEXTURE1D
                : texture.dimension == gpu::ResourceKind::Texture3D
                    ? D3D12_SRV_DIMENSION_TEXTURE3D
                    : D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (texture.dimension == gpu::ResourceKind::Texture1D)
            {
                view.Texture1D.MipLevels = texture.mipLevels;
            }
            else if (texture.dimension == gpu::ResourceKind::Texture3D)
            {
                view.Texture3D.MipLevels = texture.mipLevels;
            }
            else
            {
                view.Texture2D.MipLevels = texture.mipLevels;
            }
            device_->CreateShaderResourceView(
                record.resource.Get(), &view, cpu);
            record.srvDescriptor = gpu;
            record.hasSrv = true;
        }

        if ((usage & static_cast<std::uint32_t>(
            ir::TextureUsage::Storage)) != 0)
        {
            const auto [cpu, gpu] = allocateShaderDescriptor();
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.Format = description.Format;
            view.ViewDimension = texture.dimension == gpu::ResourceKind::Texture1D
                ? D3D12_UAV_DIMENSION_TEXTURE1D
                : texture.dimension == gpu::ResourceKind::Texture3D
                    ? D3D12_UAV_DIMENSION_TEXTURE3D
                    : D3D12_UAV_DIMENSION_TEXTURE2D;
            if (texture.dimension == gpu::ResourceKind::Texture3D)
            {
                view.Texture3D.WSize = texture.depth;
            }
            device_->CreateUnorderedAccessView(
                record.resource.Get(), nullptr, &view, cpu);
            record.uavDescriptor = gpu;
            record.hasUav = true;
        }

        return record;
    }

    ResourceRecord Backend::CreateStaticBuffer(
        const ir::ResourceDeclaration& declaration,
        ID3D12Heap* aliasHeap)
    {
        ResourceRecord record;

        const auto defaultProperties = HeapProperties(
            D3D12_HEAP_TYPE_DEFAULT);
        const auto uploadProperties = HeapProperties(
            D3D12_HEAP_TYPE_UPLOAD);

        auto description = BufferDescription(declaration.SizeBytes());
        const auto& bufferDescription =
            std::get<ir::BufferDescription>(declaration.description);
        if ((static_cast<std::uint32_t>(bufferDescription.usage)
            & static_cast<std::uint32_t>(ir::BufferUsage::Storage)) != 0)
        {
            description.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        const auto initialState = declaration.data.empty()
            ? D3D12_RESOURCE_STATE_COMMON
            : D3D12_RESOURCE_STATE_COPY_DEST;
        const HRESULT createResult = aliasHeap != nullptr
            ? device_->CreatePlacedResource(
                aliasHeap,
                0,
                &description,
                initialState,
                nullptr,
                IID_PPV_ARGS(&record.resource))
            : device_->CreateCommittedResource(
                &defaultProperties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                initialState,
                nullptr,
                IID_PPV_ARGS(&record.resource));
        ThrowIfFailed(createResult, "Create buffer resource");

        if (!declaration.data.empty())
        {
            ComPtr<ID3D12Resource> upload;
            const auto uploadDescription = BufferDescription(
                declaration.SizeBytes());
            ThrowIfFailed(
                device_->CreateCommittedResource(
                    &uploadProperties,
                    D3D12_HEAP_FLAG_NONE,
                    &uploadDescription,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    nullptr,
                    IID_PPV_ARGS(&upload)),
                "Create static upload buffer");

            void* mapped = nullptr;
            ThrowIfFailed(
                upload->Map(0, nullptr, &mapped),
                "Map static upload buffer");
            std::memcpy(
                mapped,
                declaration.data.data(),
                declaration.data.size());
            upload->Unmap(0, nullptr);

            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> list;
            ThrowIfFailed(
                device_->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&allocator)),
                "Create upload command allocator");
            ThrowIfFailed(
                device_->CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    allocator.Get(),
                    nullptr,
                    IID_PPV_ARGS(&list)),
                "Create upload command list");

            list->CopyBufferRegion(
                record.resource.Get(),
                0,
                upload.Get(),
                0,
                declaration.SizeBytes());

            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = record.resource.Get();
            barrier.Transition.Subresource =
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter =
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
            list->ResourceBarrier(1, &barrier);

            ThrowIfFailed(list->Close(), "Close upload command list");
            ID3D12CommandList* lists[] = {list.Get()};
            queue_->ExecuteCommandLists(1, lists);
            WaitIdle();
            record.state =
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        }

        record.vertexView.BufferLocation =
            record.resource->GetGPUVirtualAddress();
        record.vertexView.SizeInBytes =
            static_cast<UINT>(declaration.SizeBytes());
        record.vertexView.StrideInBytes = bufferDescription.strideBytes;

        const auto usage = static_cast<std::uint32_t>(bufferDescription.usage);
        if ((usage & static_cast<std::uint32_t>(ir::BufferUsage::Storage)) != 0)
        {
            const auto allocateShaderDescriptor = [&]()
            {
                auto cpu = shaderHeap_->GetCPUDescriptorHandleForHeapStart();
                auto gpu = shaderHeap_->GetGPUDescriptorHandleForHeapStart();
                cpu.ptr += static_cast<SIZE_T>(nextShaderDescriptor_)
                    * shaderIncrement_;
                gpu.ptr += static_cast<UINT64>(nextShaderDescriptor_++)
                    * shaderIncrement_;
                return std::pair{cpu, gpu};
            };

            const UINT elementCount = bufferDescription.strideBytes == 0
                ? static_cast<UINT>(bufferDescription.sizeBytes / 4)
                : static_cast<UINT>(bufferDescription.sizeBytes
                    / bufferDescription.strideBytes);

            {
                const auto [cpu, gpu] = allocateShaderDescriptor();
                D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                view.Format = bufferDescription.strideBytes == 0
                    ? DXGI_FORMAT_R32_TYPELESS
                    : DXGI_FORMAT_UNKNOWN;
                view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                view.Shader4ComponentMapping =
                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                view.Buffer.NumElements = elementCount;
                view.Buffer.StructureByteStride = bufferDescription.strideBytes;
                view.Buffer.Flags = bufferDescription.strideBytes == 0
                    ? D3D12_BUFFER_SRV_FLAG_RAW
                    : D3D12_BUFFER_SRV_FLAG_NONE;
                device_->CreateShaderResourceView(
                    record.resource.Get(), &view, cpu);
                record.srvDescriptor = gpu;
                record.hasSrv = true;
            }

            {
                const auto [cpu, gpu] = allocateShaderDescriptor();
                D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                view.Format = bufferDescription.strideBytes == 0
                    ? DXGI_FORMAT_R32_TYPELESS
                    : DXGI_FORMAT_UNKNOWN;
                view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                view.Buffer.NumElements = elementCount;
                view.Buffer.StructureByteStride = bufferDescription.strideBytes;
                view.Buffer.Flags = bufferDescription.strideBytes == 0
                    ? D3D12_BUFFER_UAV_FLAG_RAW
                    : D3D12_BUFFER_UAV_FLAG_NONE;
                device_->CreateUnorderedAccessView(
                    record.resource.Get(), nullptr, &view, cpu);
                record.uavDescriptor = gpu;
                record.hasUav = true;
            }
        }

        return record;
    }

    ProgramRecord Backend::CreateProgram(
        const ir::ProgramDeclaration& declaration)
    {
        ProgramRecord record;
        std::vector<shader::ReflectedBinding> reflected;

        if (!declaration.computeEntry.empty())
        {
            record.compute = shaderCompiler_.Compile(
                declaration.shaderPath, declaration.computeEntry, "cs_5_1");
            reflected = shaderCompiler_.ReflectBindings(
                record.compute, gpu::ProgramStage::Compute);
        }
        else
        {
            record.vertex = shaderCompiler_.Compile(
                declaration.shaderPath, declaration.vertexEntry, "vs_5_1");
            record.pixel = shaderCompiler_.Compile(
                declaration.shaderPath, declaration.pixelEntry, "ps_5_1");
            reflected = shaderCompiler_.ReflectBindings(
                record.vertex, gpu::ProgramStage::Vertex);
            auto pixelBindings = shaderCompiler_.ReflectBindings(
                record.pixel, gpu::ProgramStage::Pixel);
            reflected.insert(
                reflected.end(), pixelBindings.begin(), pixelBindings.end());
        }

        shaderCompiler_.ValidateInterface(
            declaration.parameters,
            reflected);

        record.rootSignature = CreateRootSignature(
            declaration, record.rootParameterIndices);
        return record;
    }

    ComPtr<ID3D12RootSignature> Backend::CreateRootSignature(
        const ir::ProgramDeclaration& declaration,
        std::vector<UINT>& rootParameterIndices)
    {
        std::vector<D3D12_ROOT_PARAMETER> parameters;
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges(declaration.parameters.size());
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        parameters.reserve(declaration.parameters.size());
        rootParameterIndices.assign(declaration.parameters.size(), UINT_MAX);

        for (std::size_t parameterIndex = 0;
             parameterIndex < declaration.parameters.size();
             ++parameterIndex)
        {
            const auto& parameter = declaration.parameters[parameterIndex];
            D3D12_ROOT_PARAMETER native{};
            if (parameter.kind == gpu::ProgramParameterKind::ConstantBuffer)
            {
                native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                native.Descriptor.ShaderRegister = parameter.registerIndex;
                native.Descriptor.RegisterSpace = parameter.registerSpace;
            }
            else if (parameter.kind == gpu::ProgramParameterKind::Sampler)
            {
                D3D12_STATIC_SAMPLER_DESC sampler{};
                sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                sampler.MipLODBias = 0.0f;
                sampler.MaxAnisotropy = 1;
                sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
                sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
                sampler.MinLOD = 0.0f;
                sampler.MaxLOD = D3D12_FLOAT32_MAX;
                sampler.ShaderRegister = parameter.registerIndex;
                sampler.RegisterSpace = parameter.registerSpace;
                sampler.ShaderVisibility =
                    parameter.stage == gpu::ProgramStage::Vertex
                        ? D3D12_SHADER_VISIBILITY_VERTEX
                        : parameter.stage == gpu::ProgramStage::Pixel
                            ? D3D12_SHADER_VISIBILITY_PIXEL
                            : D3D12_SHADER_VISIBILITY_ALL;
                samplers.push_back(sampler);
                continue;
            }
            else
            {
                auto& range = ranges[parameters.size()];
                range.RangeType =
                    parameter.kind == gpu::ProgramParameterKind::ShaderResource
                        ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                        : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                range.NumDescriptors = 1;
                range.BaseShaderRegister = parameter.registerIndex;
                range.RegisterSpace = parameter.registerSpace;
                range.OffsetInDescriptorsFromTableStart = 0;
                native.ParameterType =
                    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                native.DescriptorTable.NumDescriptorRanges = 1;
                native.DescriptorTable.pDescriptorRanges = &range;
            }

            switch (parameter.stage)
            {
            case gpu::ProgramStage::Vertex:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
                break;
            case gpu::ProgramStage::Pixel:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                break;
            case gpu::ProgramStage::Compute:
            case gpu::ProgramStage::AllGraphics:
            default:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                break;
            }

            parameters.push_back(native);
            rootParameterIndices[parameterIndex] =
                static_cast<UINT>(parameters.size() - 1);
        }

        D3D12_ROOT_SIGNATURE_DESC description{};
        description.NumParameters = static_cast<UINT>(parameters.size());
        description.pParameters = parameters.data();
        description.NumStaticSamplers = static_cast<UINT>(samplers.size());
        description.pStaticSamplers = samplers.data();
        description.Flags =
            (declaration.computeEntry.empty()
                ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                : D3D12_ROOT_SIGNATURE_FLAG_NONE)
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        ComPtr<ID3DBlob> serialized;
        ComPtr<ID3DBlob> errors;
        const HRESULT result = D3D12SerializeRootSignature(
            &description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &serialized,
            &errors);
        if (FAILED(result))
        {
            std::string message =
                "D3D12 root signature serialization failed.";
            if (errors)
            {
                message += "\n";
                message.append(
                    static_cast<const char*>(errors->GetBufferPointer()),
                    errors->GetBufferSize());
            }
            throw std::runtime_error(message);
        }

        ComPtr<ID3D12RootSignature> rootSignature;
        ThrowIfFailed(
            device_->CreateRootSignature(
                0,
                serialized->GetBufferPointer(),
                serialized->GetBufferSize(),
                IID_PPV_ARGS(&rootSignature)),
            "CreateRootSignature");
        return rootSignature;
    }

    ComPtr<ID3D12PipelineState> Backend::CreatePipeline(
        const compiler::ExecutableKey& executable)
    {
        const auto program = programs_.find(executable.program);
        if (program == programs_.end())
        {
            throw std::runtime_error(
                "Pipeline references an unknown compiled program.");
        }

        if (executable.compute)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
            description.pRootSignature = program->second.rootSignature.Get();
            description.CS = {
                program->second.compute.bytecode->GetBufferPointer(),
                program->second.compute.bytecode->GetBufferSize()
            };
            ComPtr<ID3D12PipelineState> pipeline;
            ThrowIfFailed(device_->CreateComputePipelineState(
                &description, IID_PPV_ARGS(&pipeline)),
                "CreateComputePipelineState");
            return pipeline;
        }

        D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
            {
                "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,
                0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0
            },
            {
                "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT,
                0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0
            }
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC description{};
        description.pRootSignature = program->second.rootSignature.Get();
        description.VS = {
            program->second.vertex.bytecode->GetBufferPointer(),
            program->second.vertex.bytecode->GetBufferSize()
        };
        description.PS = {
            program->second.pixel.bytecode->GetBufferPointer(),
            program->second.pixel.bytecode->GetBufferSize()
        };
        description.BlendState = BlendDescription(
            executable.rasterState.composition);
        description.SampleMask = UINT_MAX;
        description.RasterizerState = RasterizerDescription();
        description.DepthStencilState = DepthDescription(
            executable.rasterState.depth);
        description.InputLayout = {
            inputLayout,
            static_cast<UINT>(std::size(inputLayout))
        };
        description.PrimitiveTopologyType = NativeTopologyType(
            executable.rasterState.topology);
        description.NumRenderTargets = 1;
        description.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.DSVFormat = executable.rasterState.depth
                == gpu::DepthMode::Disabled
            ? DXGI_FORMAT_UNKNOWN
            : DXGI_FORMAT_D32_FLOAT;
        description.SampleDesc.Count = 1;

        ComPtr<ID3D12PipelineState> pipeline;
        ThrowIfFailed(
            device_->CreateGraphicsPipelineState(
                &description,
                IID_PPV_ARGS(&pipeline)),
            "CreateGraphicsPipelineState");
        return pipeline;
    }

    D3D12_GPU_VIRTUAL_ADDRESS Backend::UploadConstants(
        FrameContext& frame,
        const ir::ResourceDeclaration& declaration)
    {
        const UINT64 alignedOffset = AlignUp(
            frame.uploadOffset,
            D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        const UINT64 alignedSize = AlignUp(
            declaration.data.size(),
            D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

        if (alignedOffset + alignedSize > UploadArenaSize)
        {
            throw std::runtime_error(
                "Per-frame upload arena exhausted.");
        }

        std::memcpy(
            frame.mappedUpload + alignedOffset,
            declaration.data.data(),
            declaration.data.size());
        frame.uploadOffset = alignedOffset + alignedSize;
        return frame.uploadArena->GetGPUVirtualAddress() + alignedOffset;
    }

    void Backend::BindResources(
        const ir::SemanticModule& module,
        gpu::ProgramId programId,
        const std::vector<ir::ResourceBinding>& bindings,
        FrameContext& frame,
        bool compute)
    {
        const auto& declaration = module.Program(programId);
        const auto compiledProgram = programs_.find(programId);
        if (compiledProgram == programs_.end())
        {
            throw std::runtime_error("Cannot bind an uncompiled program.");
        }

        for (const auto& binding : bindings)
        {
            const auto& parameter = declaration.parameters.at(
                binding.parameterIndex);
            if (parameter.kind == gpu::ProgramParameterKind::Sampler)
            {
                throw std::runtime_error(
                    "Static samplers do not take resource bindings.");
            }

            const UINT rootIndex =
                compiledProgram->second.rootParameterIndices.at(
                    binding.parameterIndex);
            if (parameter.kind == gpu::ProgramParameterKind::ConstantBuffer)
            {
                const auto address = UploadConstants(
                    frame, module.Resource(binding.resource));
                if (compute)
                {
                    commandList_->SetComputeRootConstantBufferView(
                        rootIndex, address);
                }
                else
                {
                    commandList_->SetGraphicsRootConstantBufferView(
                        rootIndex, address);
                }
                continue;
            }

            const auto* resource = ResolveResource(
                binding.resource, binding.frameLag);
            if (resource == nullptr)
            {
                throw std::runtime_error(
                    "Program binding references an unmaterialized resource.");
            }

            D3D12_GPU_DESCRIPTOR_HANDLE descriptor{};
            if (parameter.kind == gpu::ProgramParameterKind::ShaderResource)
            {
                if (!resource->hasSrv)
                {
                    throw std::runtime_error(
                        "Shader-resource binding has no SRV descriptor.");
                }
                descriptor = resource->srvDescriptor;
            }
            else
            {
                if (!resource->hasUav)
                {
                    throw std::runtime_error(
                        "Unordered-access binding has no UAV descriptor.");
                }
                descriptor = resource->uavDescriptor;
            }

            if (compute)
            {
                commandList_->SetComputeRootDescriptorTable(
                    rootIndex, descriptor);
            }
            else
            {
                commandList_->SetGraphicsRootDescriptorTable(
                    rootIndex, descriptor);
            }
        }
    }

    void Backend::ExecuteRasterWork(
        const ir::SemanticModule& module,
        const ir::WorkDeclaration& work,
        FrameContext& frame)
    {
        const auto& raster = std::get<ir::RasterWork>(work.payload);
        const compiler::ExecutableKey key{
            .program = raster.program,
            .rasterState = raster.rasterState,
            .compute = false
        };

        const auto pipeline = pipelines_.find(key);
        if (pipeline == pipelines_.end())
        {
            throw std::runtime_error(
                "Raster work has no compiled pipeline.");
        }

        const auto program = programs_.find(raster.program);
        if (program == programs_.end())
        {
            throw std::runtime_error(
                "Raster work has no compiled program.");
        }

        const auto* geometry = ResolveResource(raster.vertexResource);
        if (geometry == nullptr)
        {
            throw std::runtime_error(
                "Raster work has no static vertex resource.");
        }

        D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
        if (raster.attachments.colors.empty())
        {
            throw std::runtime_error("Raster work has no color attachment.");
        }
        const auto color = raster.attachments.colors.front();
        if (presentationResource_ && color == *presentationResource_)
        {
            rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
            rtv.ptr += static_cast<SIZE_T>(frameIndex_) * rtvIncrement_;
        }
        else
        {
            const auto* target = ResolveResource(color);
            if (target == nullptr || !target->hasRtv)
            {
                throw std::runtime_error(
                    "Raster color attachment has no RTV.");
            }
            rtv = target->rtv;
        }

        const bool hasDepth = raster.attachments.depth.IsValid();
        D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
        if (hasDepth)
        {
            const auto* depth = ResolveResource(raster.attachments.depth);
            if (depth == nullptr || !depth->hasDsv)
            {
                throw std::runtime_error(
                    "Raster depth attachment has no DSV.");
            }
            dsv = depth->dsv;
        }
        commandList_->OMSetRenderTargets(
            1, &rtv, FALSE, hasDepth ? &dsv : nullptr);

        if (raster.clear.clearColor)
        {
            commandList_->ClearRenderTargetView(
                rtv, raster.clear.color.data(), 0, nullptr);
        }
        if (raster.clear.clearDepth && hasDepth)
        {
            commandList_->ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                raster.clear.depth,
                0,
                0,
                nullptr);
        }

        commandList_->SetPipelineState(pipeline->second.Get());
        commandList_->SetGraphicsRootSignature(
            program->second.rootSignature.Get());
        BindResources(module, raster.program, raster.bindings, frame, false);
        commandList_->IASetPrimitiveTopology(
            NativeTopology(raster.rasterState.topology));
        commandList_->IASetVertexBuffers(
            0, 1, &geometry->vertexView);
        commandList_->DrawInstanced(
            raster.vertexCount,
            1,
            raster.firstVertex,
            0);
    }

    void Backend::ExecuteComputeWork(
        const ir::SemanticModule& module,
        const ir::WorkDeclaration& work,
        FrameContext& frame)
    {
        const auto& compute = std::get<ir::ComputeWork>(work.payload);
        const compiler::ExecutableKey key{
            .program = compute.program,
            .compute = true
        };
        const auto pipeline = pipelines_.find(key);
        const auto program = programs_.find(compute.program);
        if (pipeline == pipelines_.end() || program == programs_.end())
        {
            throw std::runtime_error(
                "Compute work has no compiled executable.");
        }

        commandList_->SetPipelineState(pipeline->second.Get());
        commandList_->SetComputeRootSignature(
            program->second.rootSignature.Get());
        BindResources(module, compute.program, compute.bindings, frame, true);
        commandList_->Dispatch(
            compute.groupCountX,
            compute.groupCountY,
            compute.groupCountZ);

        for (const auto& access : work.accesses)
        {
            if (access.access != gpu::AccessMode::Read)
            {
                Transition(
                    access.resource,
                    gpu::AbstractState::Undefined,
                    access.frameLag);
            }
        }
    }

    void Backend::ExecuteCopyWork(
        const ir::SemanticModule&,
        const ir::WorkDeclaration& work)
    {
        const auto& copy = std::get<ir::CopyWork>(work.payload);
        auto* source = ResolveResource(copy.source, copy.sourceFrameLag);
        auto* destination = ResolveResource(
            copy.destination, copy.destinationFrameLag);
        if (source == nullptr || destination == nullptr)
        {
            throw std::runtime_error(
                "Copy work references an unmaterialized resource.");
        }

        if (copy.sizeBytes == 0)
        {
            commandList_->CopyResource(
                destination->resource.Get(),
                source->resource.Get());
        }
        else
        {
            commandList_->CopyBufferRegion(
                destination->resource.Get(), copy.destinationOffset,
                source->resource.Get(), copy.sourceOffset,
                copy.sizeBytes);
        }

        // COPY queue resources remain in COPY_SOURCE/COPY_DEST. A later
        // consumer performs the legal transition after the queue fence wait.
    }

    void Backend::Transition(
        gpu::ResourceId resource,
        gpu::AbstractState abstractState,
        std::uint32_t frameLag)
    {
        const auto target = abstractState == gpu::AbstractState::ProgramRead
            && activeQueue_ == gpu::QueueClass::Compute
            ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            : NativeState(abstractState);

        ID3D12Resource* nativeResource = nullptr;
        D3D12_RESOURCE_STATES* currentState = nullptr;

        if (presentationResource_ && resource == *presentationResource_)
        {
            nativeResource = backBuffers_[frameIndex_].Get();
            currentState = &backBufferStates_[frameIndex_];
        }
        else
        {
            auto* found = ResolveResource(resource, frameLag);
            if (found != nullptr)
            {
                nativeResource = found->resource.Get();
                currentState = &found->state;
            }
            else
            {
                return;
            }
        }

        if (nativeResource == nullptr
            || currentState == nullptr
            || *currentState == target)
        {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = nativeResource;
        barrier.Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = *currentState;
        barrier.Transition.StateAfter = target;
        commandList_->ResourceBarrier(1, &barrier);
        *currentState = target;
    }

    std::size_t Backend::ResolveInstanceIndex(
        gpu::ResourceId resource,
        std::uint32_t frameLag) const
    {
        const auto plan = resourceInstancePlans_.find(resource);
        if (plan == resourceInstancePlans_.end())
        {
            return 0;
        }
        const auto count = std::max<std::uint32_t>(
            1, plan->second.physicalInstanceCount);
        switch (plan->second.lifetime)
        {
        case gpu::ResourceLifetimeClass::Persistent:
            return 0;
        case gpu::ResourceLifetimeClass::FrameLocal:
            return activeFrameSlot_ % count;
        case gpu::ResourceLifetimeClass::Temporal:
            return static_cast<std::size_t>(
                (frameNumber_ + count - (frameLag % count)) % count);
        case gpu::ResourceLifetimeClass::External:
            return frameIndex_;
        }
        return 0;
    }

    ResourceRecord* Backend::ResolveResource(
        gpu::ResourceId resource,
        std::uint32_t frameLag)
    {
        const auto found = resources_.find(resource);
        if (found == resources_.end() || found->second.empty())
        {
            return nullptr;
        }
        return &found->second.at(
            ResolveInstanceIndex(resource, frameLag));
    }

    const ResourceRecord* Backend::ResolveResource(
        gpu::ResourceId resource,
        std::uint32_t frameLag) const
    {
        const auto found = resources_.find(resource);
        if (found == resources_.end() || found->second.empty())
        {
            return nullptr;
        }
        return &found->second.at(
            ResolveInstanceIndex(resource, frameLag));
    }

    void Backend::ActivateAliasedResource(
        gpu::ResourceId resource,
        std::uint32_t frameLag)
    {
        const auto allocation = physicalAllocations_.find(resource);
        if (allocation == physicalAllocations_.end())
        {
            return;
        }

        auto& active = activeAliasedResources_[allocation->second];
        const auto instance = ResolveInstanceIndex(resource, frameLag);
        if (active.size() <= instance)
        {
            active.resize(instance + 1);
        }
        const auto previous = active[instance];
        if (previous && *previous != resource)
        {
            const auto* before = ResolveResource(*previous, frameLag);
            const auto* after = ResolveResource(resource, frameLag);
            if (before != nullptr && after != nullptr)
            {
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                barrier.Aliasing.pResourceBefore =
                    before->resource.Get();
                barrier.Aliasing.pResourceAfter =
                    after->resource.Get();
                commandList_->ResourceBarrier(1, &barrier);
            }
        }
        active[instance] = resource;
    }
}
