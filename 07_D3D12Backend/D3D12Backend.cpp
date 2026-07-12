#include "07_D3D12Backend/D3D12Backend.h"
#include "07_D3D12Backend/D3D12BackendInternal.h"

#include <algorithm>
#include <array>
#include <string>
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

        for (auto& state : queueSyncStates_)
        {
            if (state.event != nullptr)
            {
                CloseHandle(state.event);
                state.event = nullptr;
            }
        }
    }

    gpu::DeviceCapabilities Backend::Capabilities() const
    {
        gpu::DeviceCapabilities capabilities;
        capabilities.rasterExecution = true;
        capabilities.computeExecution = true;
        capabilities.copyExecution = true;
        capabilities.resourceAliasing = true;
        capabilities.concurrentCompute = true;
        capabilities.dedicatedCopyQueue = true;
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
        activeFrameSlot_ = static_cast<std::uint32_t>(
            frameNumber_ % FrameCount);
        FrameContext& frame = frames_[activeFrameSlot_];
        BeginFrame(frame);

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

        struct WorkSignal
        {
            gpu::QueueClass queue = gpu::QueueClass::Direct;
            UINT64 value = 0;
        };

        std::vector<WorkSignal> workSignals(plan.scheduledWorks.size());
        for (std::size_t scheduledIndex = 0;
             scheduledIndex < plan.scheduledWorks.size();
             ++scheduledIndex)
        {
            const auto& scheduled = plan.scheduledWorks[scheduledIndex];
            const auto& work = module.works.at(scheduled.sourceWorkIndex);

            auto* nativeQueue = QueueFor(scheduled.queue);

            for (const auto& temporal : plan.temporalDependencies)
            {
                if (temporal.consumerScheduledWork != scheduledIndex)
                {
                    continue;
                }
                const auto completions = temporalCompletions_.find(
                    temporal.resource);
                if (completions == temporalCompletions_.end()
                    || completions->second.empty())
                {
                    continue;
                }
                const auto count = completions->second.size();
                const auto slot = static_cast<std::size_t>(
                    (frameNumber_ + count - (temporal.readLag % count))
                    % count);
                const auto point = completions->second[slot];
                WaitForQueueValue(
                    scheduled.queue, point.queue, point.value);
            }

            for (const auto& synchronization : plan.queueSynchronizations)
            {
                if (synchronization.waitScheduledWork == scheduledIndex)
                {
                    const auto& signal = workSignals.at(
                        synchronization.signalScheduledWork);
                    if (signal.value == 0
                        || signal.queue != synchronization.signalQueue)
                    {
                        throw std::runtime_error(
                            "Queue synchronization references an unsignaled work.");
                    }
                    WaitForQueueValue(
                        scheduled.queue,
                        synchronization.signalQueue,
                        signal.value);
                }
            }

            commandList_ = AcquireCommandList(frame, scheduled.queue);
            activeQueue_ = scheduled.queue;

            if (scheduled.queue != gpu::QueueClass::Copy)
            {
                ID3D12DescriptorHeap* descriptorHeaps[] = {shaderHeap_.Get()};
                commandList_->SetDescriptorHeaps(1, descriptorHeaps);
            }
            if (scheduled.queue == gpu::QueueClass::Direct)
            {
                commandList_->RSSetViewports(1, &viewport);
                commandList_->RSSetScissorRects(1, &scissor);
            }

            for (const auto& requirement : scheduled.requiredStates)
            {
                ActivateAliasedResource(
                    requirement.resource, requirement.frameLag);
                if (scheduled.queue == gpu::QueueClass::Copy
                    && (requirement.state == gpu::AbstractState::TransferRead
                        || requirement.state
                            == gpu::AbstractState::TransferWrite))
                {
                    // COPY lists use implicit COMMON promotion/decay. Queue
                    // handoff planning guarantees COMMON before submission.
                    continue;
                }
                Transition(
                    requirement.resource,
                    requirement.state,
                    requirement.frameLag);
            }

            if (work.Domain() == gpu::ExecutionDomain::Raster)
            {
                ExecuteRasterWork(module, work, frame);
            }
            else if (work.Domain() == gpu::ExecutionDomain::Compute)
            {
                ExecuteComputeWork(module, work, frame);
            }
            else if (work.Domain() == gpu::ExecutionDomain::Copy)
            {
                ExecuteCopyWork(module, work);
            }
            else if (work.Domain() != gpu::ExecutionDomain::Present)
            {
                throw std::runtime_error(
                    "D3D12 backend received an unsupported work domain.");
            }

            const bool hasOutgoingQueueHandoff = std::any_of(
                plan.queueSynchronizations.begin(),
                plan.queueSynchronizations.end(),
                [&](const compiler::QueueSynchronization& synchronization)
                {
                    return synchronization.signalScheduledWork
                        == scheduledIndex;
                });
            if (hasOutgoingQueueHandoff
                && scheduled.queue != gpu::QueueClass::Copy)
            {
                for (const auto& requirement : scheduled.requiredStates)
                {
                    Transition(
                        requirement.resource,
                        gpu::AbstractState::Undefined,
                        requirement.frameLag);
                }
            }

            for (const auto& boundary : plan.frameBoundaryTransitions)
            {
                if (boundary.afterScheduledWork != scheduledIndex)
                {
                    continue;
                }
                if (boundary.releaseQueue != scheduled.queue)
                {
                    throw std::runtime_error(
                        "Frame-boundary transition is assigned to the wrong queue.");
                }
                const auto instancePlan = resourceInstancePlans_.find(
                    boundary.resource);
                if (instancePlan != resourceInstancePlans_.end()
                    && instancePlan->second.lifetime
                        == gpu::ResourceLifetimeClass::Persistent)
                {
                    Transition(boundary.resource, boundary.to);
                }
            }

            const HRESULT closeResult = commandList_->Close();
            if (FAILED(closeResult))
            {
                std::string message = "Close frame command list for work '"
                    + work.name + "' on queue "
                    + std::to_string(static_cast<int>(scheduled.queue))
                    + ", HRESULT="
                    + std::to_string(static_cast<unsigned long>(closeResult));
#if defined(_DEBUG)
                ComPtr<ID3D12InfoQueue> information;
                if (SUCCEEDED(device_.As(&information)))
                {
                    const auto count = information->GetNumStoredMessages();
                    const auto first = count > 8 ? count - 8 : 0;
                    for (UINT64 index = first; index < count; ++index)
                    {
                        SIZE_T bytes = 0;
                        information->GetMessage(index, nullptr, &bytes);
                        std::vector<std::byte> storage(bytes);
                        auto* debugMessage = reinterpret_cast<
                            D3D12_MESSAGE*>(storage.data());
                        if (SUCCEEDED(information->GetMessage(
                            index, debugMessage, &bytes)))
                        {
                            message += "\n";
                            message.append(
                                debugMessage->pDescription,
                                debugMessage->DescriptionByteLength);
                        }
                    }
                }
#endif
                throw std::runtime_error(message);
            }
            ID3D12CommandList* lists[] = {commandList_.Get()};
            nativeQueue->ExecuteCommandLists(1, lists);

            workSignals[scheduledIndex] = {
                .queue = scheduled.queue,
                .value = SignalQueue(scheduled.queue)
            };
            frame.queues[QueueIndex(scheduled.queue)].completionFenceValue =
                workSignals[scheduledIndex].value;

            for (const auto& access : work.accesses)
            {
                if (access.frameLag != 0
                    || access.access == gpu::AccessMode::Read)
                {
                    continue;
                }
                const auto planFound = resourceInstancePlans_.find(
                    access.resource);
                if (planFound == resourceInstancePlans_.end()
                    || planFound->second.lifetime
                        != gpu::ResourceLifetimeClass::Temporal)
                {
                    continue;
                }
                auto& completions = temporalCompletions_[access.resource];
                if (completions.size()
                    != planFound->second.physicalInstanceCount)
                {
                    completions.resize(
                        planFound->second.physicalInstanceCount);
                }
                completions[ResolveInstanceIndex(access.resource)] = {
                    scheduled.queue,
                    workSignals[scheduledIndex].value
                };
            }
        }

        ThrowIfFailed(
            swapChain_->Present(1, 0),
            "SwapChain::Present");

        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
        ++frameNumber_;
    }

    void Backend::WaitIdle()
    {
        if (!queue_)
        {
            return;
        }

        constexpr std::array queues{
            gpu::QueueClass::Direct,
            gpu::QueueClass::Compute,
            gpu::QueueClass::Copy
        };

        for (const auto queue : queues)
        {
            if (QueueFor(queue) == nullptr || !SyncFor(queue).fence)
            {
                continue;
            }

            const UINT64 value = SignalQueue(queue);
            auto& state = SyncFor(queue);
            if (state.fence->GetCompletedValue() < value)
            {
                ThrowIfFailed(state.fence->SetEventOnCompletion(
                    value, state.event), "Fence::SetEventOnCompletion");
                WaitForSingleObject(state.event, INFINITE);
            }
        }
    }

    ID3D12CommandQueue* Backend::QueueFor(
        gpu::QueueClass queue) const noexcept
    {
        switch (queue)
        {
        case gpu::QueueClass::Compute: return computeQueue_.Get();
        case gpu::QueueClass::Copy: return copyQueue_.Get();
        default: return queue_.Get();
        }
    }

    D3D12_COMMAND_LIST_TYPE Backend::CommandListType(
        gpu::QueueClass queue) noexcept
    {
        switch (queue)
        {
        case gpu::QueueClass::Compute: return D3D12_COMMAND_LIST_TYPE_COMPUTE;
        case gpu::QueueClass::Copy: return D3D12_COMMAND_LIST_TYPE_COPY;
        default: return D3D12_COMMAND_LIST_TYPE_DIRECT;
        }
    }

    std::size_t Backend::QueueIndex(gpu::QueueClass queue) noexcept
    {
        switch (queue)
        {
        case gpu::QueueClass::Compute: return 1;
        case gpu::QueueClass::Copy: return 2;
        default: return 0;
        }
    }

    QueueSyncState& Backend::SyncFor(gpu::QueueClass queue) noexcept
    {
        return queueSyncStates_[QueueIndex(queue)];
    }

    const QueueSyncState& Backend::SyncFor(
        gpu::QueueClass queue) const noexcept
    {
        return queueSyncStates_[QueueIndex(queue)];
    }

    UINT64 Backend::SignalQueue(gpu::QueueClass queue)
    {
        auto& state = SyncFor(queue);
        const UINT64 value = state.nextSignalValue++;
        ThrowIfFailed(
            QueueFor(queue)->Signal(state.fence.Get(), value),
            "Signal queue fence");
        return value;
    }

    void Backend::WaitForQueueValue(
        gpu::QueueClass waitingQueue,
        gpu::QueueClass sourceQueue,
        UINT64 value)
    {
        if (waitingQueue == sourceQueue || value == 0)
        {
            return;
        }
        ThrowIfFailed(
            QueueFor(waitingQueue)->Wait(SyncFor(sourceQueue).fence.Get(), value),
            "Cross-queue wait");
    }

    void Backend::WaitForCpuQueueValue(
        gpu::QueueClass queue,
        UINT64 value)
    {
        if (value == 0)
        {
            return;
        }
        auto& state = SyncFor(queue);
        if (state.fence->GetCompletedValue() >= value)
        {
            return;
        }
        ThrowIfFailed(state.fence->SetEventOnCompletion(value, state.event),
            "Fence::SetEventOnCompletion");
        WaitForSingleObject(state.event, INFINITE);
    }

    void Backend::BeginFrame(FrameContext& frame)
    {
        constexpr std::array queues{
            gpu::QueueClass::Direct,
            gpu::QueueClass::Compute,
            gpu::QueueClass::Copy
        };
        for (const auto queue : queues)
        {
            auto& context = frame.queues[QueueIndex(queue)];
            WaitForCpuQueueValue(queue, context.completionFenceValue);
            context.completionFenceValue = 0;
            context.usedCommandLists = 0;
            ThrowIfFailed(context.allocator->Reset(),
                "Reset frame command allocator");
        }
        frame.uploadOffset = 0;
        for (auto& [allocation, active] : activeAliasedResources_)
        {
            if (frame.slot < active.size())
            {
                active[frame.slot].reset();
            }
        }
    }

    ID3D12GraphicsCommandList* Backend::AcquireCommandList(
        FrameContext& frame,
        gpu::QueueClass queue)
    {
        auto& context = frame.queues[QueueIndex(queue)];
        const auto listType = CommandListType(queue);
        if (context.usedCommandLists == context.commandLists.size())
        {
            ComPtr<ID3D12GraphicsCommandList> list;
            ThrowIfFailed(device_->CreateCommandList(
                0, listType, context.allocator.Get(), nullptr,
                IID_PPV_ARGS(&list)), "Create frame command list");
            context.commandLists.push_back(std::move(list));
        }
        else
        {
            ThrowIfFailed(context.commandLists[context.usedCommandLists]->Reset(
                context.allocator.Get(), nullptr), "Reset frame command list");
        }
        return context.commandLists[context.usedCommandLists++].Get();
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

        description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        ThrowIfFailed(device_->CreateCommandQueue(
            &description, IID_PPV_ARGS(&computeQueue_)),
            "Create compute command queue");

        description.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        ThrowIfFailed(device_->CreateCommandQueue(
            &description, IID_PPV_ARGS(&copyQueue_)),
            "Create copy command queue");
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
        rtvDescription.NumDescriptors = 64;

        ThrowIfFailed(
            device_->CreateDescriptorHeap(
                &rtvDescription,
                IID_PPV_ARGS(&rtvHeap_)),
            "Create RTV descriptor heap");

        D3D12_DESCRIPTOR_HEAP_DESC dsvDescription{};
        dsvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvDescription.NumDescriptors = 16;

        ThrowIfFailed(
            device_->CreateDescriptorHeap(
                &dsvDescription,
                IID_PPV_ARGS(&dsvHeap_)),
            "Create DSV descriptor heap");

        D3D12_DESCRIPTOR_HEAP_DESC shaderDescription{};
        shaderDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        shaderDescription.NumDescriptors = 128;
        shaderDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(
            device_->CreateDescriptorHeap(
                &shaderDescription,
                IID_PPV_ARGS(&shaderHeap_)),
            "Create shader descriptor heap");

        rtvIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        dsvIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        shaderIncrement_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    void Backend::CreateFrameContexts()
    {
        for (std::uint32_t slot = 0; slot < FrameCount; ++slot)
        {
            auto& frame = frames_[slot];
            frame.slot = slot;
            constexpr std::array queues{
                gpu::QueueClass::Direct,
                gpu::QueueClass::Compute,
                gpu::QueueClass::Copy
            };
            for (const auto queue : queues)
            {
                ThrowIfFailed(device_->CreateCommandAllocator(
                    CommandListType(queue),
                    IID_PPV_ARGS(&frame.queues[QueueIndex(queue)].allocator)),
                    "Create frame command allocator");
            }

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
    }

    void Backend::CreateFence()
    {
        constexpr std::array queues{
            gpu::QueueClass::Direct,
            gpu::QueueClass::Compute,
            gpu::QueueClass::Copy
        };

        for (const auto queue : queues)
        {
            auto& state = SyncFor(queue);
            ThrowIfFailed(
                device_->CreateFence(
                    0,
                    D3D12_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&state.fence)),
                "Create queue fence");

            state.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (state.event == nullptr)
            {
                throw std::runtime_error(
                    "CreateEventW failed for a D3D12 queue fence.");
            }
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
        compiledStructureHash_.reset();

        if (width_ == 0 || height_ == 0)
        {
            return;
        }

        WaitIdle();

        for (auto& buffer : backBuffers_)
        {
            buffer.Reset();
        }

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
