#include "07_D3D12Backend/D3D12Backend.h"
#include "07_D3D12Backend/D3D12BackendInternal.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

using Microsoft::WRL::ComPtr;

namespace sge::d3d12::detail
{
    Backend::Backend(platform::NativeSurface surface)
        : window_(static_cast<HWND>(surface.handle)),
          width_(surface.width),
          height_(surface.height)
    {
        if (window_ == nullptr)
        {
            throw std::invalid_argument(
                "D3D12 backend requires a Win32 window.");
        }

        EnableDebugLayer();
        CreateDeviceAndQueue();
        CreateSwapChain();
        CreateDescriptorHeaps();
        CreateFrameContexts();
        CreateFence();
        CreateRenderTargets();
        CreateDepthBuffer();
    }

    Backend::~Backend()
    {
        try
        {
            WaitIdle();
        }
        catch (...)
        {
        }

        for (auto& frame : frames_)
        {
            if (frame.uploadArena && frame.mappedUpload != nullptr)
            {
                frame.uploadArena->Unmap(0, nullptr);
                frame.mappedUpload = nullptr;
            }
        }

        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }
    }

    gpu::DeviceCapabilities Backend::Capabilities() const
    {
        gpu::DeviceCapabilities capabilities;
        capabilities.rasterExecution = true;
        capabilities.computeExecution = true;
        capabilities.constantDataAlignment =
            D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;

        ComPtr<IDXGIAdapter3> adapter3;
        if (adapter_ && SUCCEEDED(adapter_.As(&adapter3)))
        {
            DXGI_QUERY_VIDEO_MEMORY_INFO information{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                0,
                DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                &information)))
            {
                capabilities.localMemoryBudget = information.Budget;
            }
        }

        return capabilities;
    }

    void Backend::Execute(
        const ir::SemanticModule& module,
        const compiler::ExecutionPlan& plan)
    {
        ResizeIfNeeded();
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        EnsureCompiled(module, plan);

        FrameContext& frame = frames_[frameIndex_];
        WaitForFrame(frame);
        frame.uploadOffset = 0;

        ThrowIfFailed(
            frame.allocator->Reset(),
            "CommandAllocator::Reset");

        ThrowIfFailed(
            commandList_->Reset(frame.allocator.Get(), nullptr),
            "GraphicsCommandList::Reset");

        const D3D12_VIEWPORT viewport{
            0.0f,
            0.0f,
            static_cast<float>(width_),
            static_cast<float>(height_),
            0.0f,
            1.0f
        };

        const D3D12_RECT scissor{
            0,
            0,
            static_cast<LONG>(width_),
            static_cast<LONG>(height_)
        };

        commandList_->RSSetViewports(1, &viewport);
        commandList_->RSSetScissorRects(1, &scissor);

        for (const auto& scheduled : plan.scheduledWorks)
        {
            const auto& work = module.works.at(
                scheduled.sourceWorkIndex);

            for (const auto& requirement : scheduled.requiredStates)
            {
                Transition(requirement.resource, requirement.state);
            }

            if (work.domain == gpu::ExecutionDomain::Raster)
            {
                ExecuteRasterWork(module, work, frame);
            }
        }

        ThrowIfFailed(
            commandList_->Close(),
            "GraphicsCommandList::Close");

        ID3D12CommandList* lists[] = {commandList_.Get()};
        queue_->ExecuteCommandLists(1, lists);

        ThrowIfFailed(
            swapChain_->Present(1, 0),
            "SwapChain::Present");

        const UINT64 signalValue = nextFenceValue_++;
        ThrowIfFailed(
            queue_->Signal(fence_.Get(), signalValue),
            "CommandQueue::Signal");

        frame.fenceValue = signalValue;
        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
    }

    void Backend::WaitIdle()
    {
        if (!queue_ || !fence_)
        {
            return;
        }

        const UINT64 signalValue = nextFenceValue_++;
        ThrowIfFailed(
            queue_->Signal(fence_.Get(), signalValue),
            "CommandQueue::Signal");

        if (fence_->GetCompletedValue() < signalValue)
        {
            ThrowIfFailed(
                fence_->SetEventOnCompletion(signalValue, fenceEvent_),
                "Fence::SetEventOnCompletion");

            WaitForSingleObject(fenceEvent_, INFINITE);
        }
    }

    void Backend::EnableDebugLayer()
    {
#if defined(_DEBUG)
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        {
            debug->EnableDebugLayer();
        }
#endif
    }

    void Backend::CreateDeviceAndQueue()
    {
        UINT factoryFlags = 0;
#if defined(_DEBUG)
        factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

        HRESULT factoryResult = CreateDXGIFactory2(
            factoryFlags,
            IID_PPV_ARGS(&factory_));

#if defined(_DEBUG)
        if (FAILED(factoryResult))
        {
            factoryResult = CreateDXGIFactory2(
                0,
                IID_PPV_ARGS(&factory_));
        }
#endif

        ThrowIfFailed(factoryResult, "CreateDXGIFactory2");

        for (UINT index = 0; ; ++index)
        {
            ComPtr<IDXGIAdapter1> candidate;

            if (factory_->EnumAdapterByGpuPreference(
                index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
            {
                break;
            }

            DXGI_ADAPTER_DESC1 description{};
            candidate->GetDesc1(&description);

            if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
            {
                continue;
            }

            if (SUCCEEDED(D3D12CreateDevice(
                candidate.Get(),
                D3D_FEATURE_LEVEL_12_0,
                __uuidof(ID3D12Device),
                nullptr)))
            {
                adapter_ = candidate;
                break;
            }
        }

        if (!adapter_)
        {
            ThrowIfFailed(
                factory_->EnumWarpAdapter(IID_PPV_ARGS(&adapter_)),
                "EnumWarpAdapter");
        }

        ThrowIfFailed(
            D3D12CreateDevice(
                adapter_.Get(),
                D3D_FEATURE_LEVEL_12_0,
                IID_PPV_ARGS(&device_)),
            "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC description{};
        description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        ThrowIfFailed(
            device_->CreateCommandQueue(
                &description,
                IID_PPV_ARGS(&queue_)),
            "CreateCommandQueue");
    }

    void Backend::CreateSwapChain()
    {
        DXGI_SWAP_CHAIN_DESC1 description{};
        description.Width = std::max(width_, 1u);
        description.Height = std::max(height_, 1u);
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = FrameCount;
        description.Scaling = DXGI_SCALING_STRETCH;
        description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        description.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

        ComPtr<IDXGISwapChain1> swapChain1;
        ThrowIfFailed(
            factory_->CreateSwapChainForHwnd(
                queue_.Get(),
                window_,
                &description,
                nullptr,
                nullptr,
                &swapChain1),
            "CreateSwapChainForHwnd");

        ThrowIfFailed(
            factory_->MakeWindowAssociation(
                window_,
                DXGI_MWA_NO_ALT_ENTER),
            "MakeWindowAssociation");

        ThrowIfFailed(
            swapChain1.As(&swapChain_),
            "Query IDXGISwapChain3");

        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
    }

    void Backend::CreateDescriptorHeaps()
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescription{};
        rtvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvDescription.NumDescriptors = FrameCount;

        ThrowIfFailed(
            device_->CreateDescriptorHeap(
                &rtvDescription,
                IID_PPV_ARGS(&rtvHeap_)),
            "Create RTV descriptor heap");

        D3D12_DESCRIPTOR_HEAP_DESC dsvDescription{};
        dsvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvDescription.NumDescriptors = 1;

        ThrowIfFailed(
            device_->CreateDescriptorHeap(
                &dsvDescription,
                IID_PPV_ARGS(&dsvHeap_)),
            "Create DSV descriptor heap");

        rtvIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    }

    void Backend::CreateFrameContexts()
    {
        for (auto& frame : frames_)
        {
            ThrowIfFailed(
                device_->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&frame.allocator)),
                "CreateCommandAllocator");

            const auto uploadProperties = HeapProperties(
                D3D12_HEAP_TYPE_UPLOAD);

            const auto uploadDescription = BufferDescription(
                UploadArenaSize);

            ThrowIfFailed(
                device_->CreateCommittedResource(
                    &uploadProperties,
                    D3D12_HEAP_FLAG_NONE,
                    &uploadDescription,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    nullptr,
                    IID_PPV_ARGS(&frame.uploadArena)),
                "Create frame upload arena");

            ThrowIfFailed(
                frame.uploadArena->Map(
                    0,
                    nullptr,
                    reinterpret_cast<void**>(&frame.mappedUpload)),
                "Map frame upload arena");
        }

        ThrowIfFailed(
            device_->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                frames_[0].allocator.Get(),
                nullptr,
                IID_PPV_ARGS(&commandList_)),
            "CreateGraphicsCommandList");

        ThrowIfFailed(
            commandList_->Close(),
            "Close initial command list");
    }

    void Backend::CreateFence()
    {
        ThrowIfFailed(
            device_->CreateFence(
                0,
                D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&fence_)),
            "CreateFence");

        fenceEvent_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error(
                "CreateEventW failed for the D3D12 fence.");
        }
    }

    void Backend::CreateRenderTargets()
    {
        auto handle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();

        for (UINT index = 0; index < FrameCount; ++index)
        {
            ThrowIfFailed(
                swapChain_->GetBuffer(
                    index,
                    IID_PPV_ARGS(&backBuffers_[index])),
                "SwapChain::GetBuffer");

            device_->CreateRenderTargetView(
                backBuffers_[index].Get(),
                nullptr,
                handle);

            backBufferStates_[index] = D3D12_RESOURCE_STATE_PRESENT;
            handle.ptr += rtvIncrement_;
        }
    }

    void Backend::CreateDepthBuffer()
    {
        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        D3D12_CLEAR_VALUE clearValue{};
        clearValue.Format = DXGI_FORMAT_D32_FLOAT;
        clearValue.DepthStencil.Depth = 1.0f;

        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        description.Width = width_;
        description.Height = height_;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.Format = DXGI_FORMAT_D32_FLOAT;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        description.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        const auto properties = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);

        ThrowIfFailed(
            device_->CreateCommittedResource(
                &properties,
                D3D12_HEAP_FLAG_NONE,
                &description,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                &clearValue,
                IID_PPV_ARGS(&depthBuffer_)),
            "Create depth buffer");

        D3D12_DEPTH_STENCIL_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_D32_FLOAT;
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;

        device_->CreateDepthStencilView(
            depthBuffer_.Get(),
            &view,
            dsvHeap_->GetCPUDescriptorHandleForHeapStart());

        depthState_ = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }

    void Backend::ResizeIfNeeded()
    {
        RECT client{};
        if (!GetClientRect(window_, &client))
        {
            return;
        }

        const auto newWidth =
            static_cast<UINT>(client.right - client.left);
        const auto newHeight =
            static_cast<UINT>(client.bottom - client.top);

        if (newWidth == width_ && newHeight == height_)
        {
            return;
        }

        width_ = newWidth;
        height_ = newHeight;

        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        WaitIdle();

        for (auto& buffer : backBuffers_)
        {
            buffer.Reset();
        }
        depthBuffer_.Reset();

        DXGI_SWAP_CHAIN_DESC description{};
        ThrowIfFailed(
            swapChain_->GetDesc(&description),
            "SwapChain::GetDesc");

        ThrowIfFailed(
            swapChain_->ResizeBuffers(
                FrameCount,
                width_,
                height_,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                description.Flags),
            "SwapChain::ResizeBuffers");

        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
        CreateRenderTargets();
        CreateDepthBuffer();
    }

    void Backend::WaitForFrame(FrameContext& frame)
    {
        if (frame.fenceValue == 0
            || fence_->GetCompletedValue() >= frame.fenceValue)
        {
            return;
        }

        ThrowIfFailed(
            fence_->SetEventOnCompletion(
                frame.fenceValue,
                fenceEvent_),
            "Fence::SetEventOnCompletion");

        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

namespace sge::d3d12
{
    std::unique_ptr<runtime::IRenderBackend> CreateBackend(
        platform::NativeSurface surface)
    {
        return std::make_unique<detail::Backend>(surface);
    }
}
