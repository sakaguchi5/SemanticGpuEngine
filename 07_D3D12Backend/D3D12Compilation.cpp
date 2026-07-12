#include "07_D3D12Backend/D3D12BackendInternal.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
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

        staticResources_.clear();
        programs_.clear();
        pipelines_.clear();
        presentationResource_.reset();
        depthResource_.reset();

        for (const auto& resource : module.resources)
        {
            if (resource.kind == gpu::ResourceKind::Presentation)
            {
                presentationResource_ = resource.id;
                continue;
            }

            if (resource.kind == gpu::ResourceKind::Texture2D
                && resource.format == gpu::ResourceFormat::Depth32Float)
            {
                depthResource_ = resource.id;
                continue;
            }

            if (resource.kind == gpu::ResourceKind::Buffer
                && resource.memoryClass == gpu::MemoryClass::Static)
            {
                staticResources_.emplace(
                    resource.id,
                    CreateStaticBuffer(resource));
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

    ResourceRecord Backend::CreateStaticBuffer(
        const ir::ResourceDeclaration& declaration)
    {
        if (declaration.data.empty())
        {
            throw std::runtime_error(
                "Static buffer requires initial data.");
        }

        ResourceRecord record;

        const auto defaultProperties = HeapProperties(
            D3D12_HEAP_TYPE_DEFAULT);

        const auto uploadProperties = HeapProperties(
            D3D12_HEAP_TYPE_UPLOAD);

        const auto description = BufferDescription(declaration.sizeBytes);

        ThrowIfFailed(
            device_->CreateCommittedResource(
                &defaultProperties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&record.resource)),
            "Create static default buffer");

        ComPtr<ID3D12Resource> upload;
        ThrowIfFailed(
            device_->CreateCommittedResource(
                &uploadProperties,
                D3D12_HEAP_FLAG_NONE,
                &description,
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
            declaration.sizeBytes);

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

        record.vertexView.BufferLocation =
            record.resource->GetGPUVirtualAddress();

        record.vertexView.SizeInBytes =
            static_cast<UINT>(declaration.sizeBytes);

        record.vertexView.StrideInBytes = declaration.strideBytes;

        return record;
    }

    ProgramRecord Backend::CreateProgram(
        const ir::ProgramDeclaration& declaration)
    {
        ProgramRecord record;

        record.vertex = shaderCompiler_.Compile(
            declaration.shaderPath,
            declaration.vertexEntry,
            "vs_5_1");

        record.pixel = shaderCompiler_.Compile(
            declaration.shaderPath,
            declaration.pixelEntry,
            "ps_5_1");

        auto reflected = shaderCompiler_.ReflectBindings(
            record.vertex,
            gpu::ProgramStage::Vertex);

        auto pixelBindings = shaderCompiler_.ReflectBindings(
            record.pixel,
            gpu::ProgramStage::Pixel);

        reflected.insert(
            reflected.end(),
            pixelBindings.begin(),
            pixelBindings.end());

        shaderCompiler_.ValidateInterface(
            declaration.parameters,
            reflected);

        record.rootSignature = CreateRootSignature(declaration);
        return record;
    }

    ComPtr<ID3D12RootSignature> Backend::CreateRootSignature(
        const ir::ProgramDeclaration& declaration)
    {
        std::vector<D3D12_ROOT_PARAMETER> parameters;
        parameters.reserve(declaration.parameters.size());

        for (const auto& parameter : declaration.parameters)
        {
            if (parameter.kind
                != gpu::ProgramParameterKind::ConstantBuffer)
            {
                throw std::runtime_error(
                    "The reference D3D12 backend currently supports only "
                    "root CBV program parameters.");
            }

            D3D12_ROOT_PARAMETER native{};
            native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            native.Descriptor.ShaderRegister = parameter.registerIndex;
            native.Descriptor.RegisterSpace = parameter.registerSpace;

            switch (parameter.stage)
            {
            case gpu::ProgramStage::Vertex:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
                break;

            case gpu::ProgramStage::Pixel:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
                break;

            case gpu::ProgramStage::AllGraphics:
            default:
                native.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                break;
            }

            parameters.push_back(native);
        }

        D3D12_ROOT_SIGNATURE_DESC description{};
        description.NumParameters =
            static_cast<UINT>(parameters.size());
        description.pParameters = parameters.data();
        description.Flags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
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

        D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
            {
                "POSITION",
                0,
                DXGI_FORMAT_R32G32B32_FLOAT,
                0,
                0,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                0
            },
            {
                "COLOR",
                0,
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                0,
                12,
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                0
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
        description.DSVFormat = DXGI_FORMAT_D32_FLOAT;
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

        return frame.uploadArena->GetGPUVirtualAddress()
            + alignedOffset;
    }

    void Backend::ExecuteRasterWork(
        const ir::SemanticModule& module,
        const ir::WorkDeclaration& work,
        FrameContext& frame)
    {
        const compiler::ExecutableKey key{
            .program = work.program,
            .rasterState = work.rasterState
        };

        const auto pipeline = pipelines_.find(key);
        if (pipeline == pipelines_.end())
        {
            throw std::runtime_error(
                "Raster work has no compiled pipeline.");
        }

        const auto program = programs_.find(work.program);
        if (program == programs_.end())
        {
            throw std::runtime_error(
                "Raster work has no compiled program.");
        }

        const auto geometry = staticResources_.find(work.vertexResource);
        if (geometry == staticResources_.end())
        {
            throw std::runtime_error(
                "Raster work has no static vertex resource.");
        }

        const auto& constants = module.Resource(work.constantResource);
        const auto constantAddress = UploadConstants(frame, constants);

        auto rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += static_cast<SIZE_T>(frameIndex_) * rtvIncrement_;

        const auto dsv =
            dsvHeap_->GetCPUDescriptorHandleForHeapStart();

        commandList_->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

        if (work.clear.clearColor)
        {
            commandList_->ClearRenderTargetView(
                rtv,
                work.clear.color.data(),
                0,
                nullptr);
        }

        if (work.clear.clearDepth)
        {
            commandList_->ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAG_DEPTH,
                work.clear.depth,
                0,
                0,
                nullptr);
        }

        commandList_->SetPipelineState(pipeline->second.Get());
        commandList_->SetGraphicsRootSignature(
            program->second.rootSignature.Get());
        commandList_->SetGraphicsRootConstantBufferView(
            0,
            constantAddress);
        commandList_->IASetPrimitiveTopology(
            NativeTopology(work.rasterState.topology));
        commandList_->IASetVertexBuffers(
            0,
            1,
            &geometry->second.vertexView);
        commandList_->DrawInstanced(
            work.vertexCount,
            1,
            work.firstVertex,
            0);
    }

    void Backend::Transition(
        gpu::ResourceId resource,
        gpu::AbstractState abstractState)
    {
        const auto target = NativeState(abstractState);

        ID3D12Resource* nativeResource = nullptr;
        D3D12_RESOURCE_STATES* currentState = nullptr;

        if (presentationResource_
            && resource == *presentationResource_)
        {
            nativeResource = backBuffers_[frameIndex_].Get();
            currentState = &backBufferStates_[frameIndex_];
        }
        else if (depthResource_ && resource == *depthResource_)
        {
            nativeResource = depthBuffer_.Get();
            currentState = &depthState_;
        }
        else
        {
            const auto found = staticResources_.find(resource);
            if (found != staticResources_.end())
            {
                nativeResource = found->second.resource.Get();
                currentState = &found->second.state;
            }
            else
            {
                // Dynamic constants live in persistently mapped GENERIC_READ
                // memory and require no explicit transition.
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
}
