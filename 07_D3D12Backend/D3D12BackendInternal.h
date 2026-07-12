#pragma once

#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"
#include "06_ShaderSystem/ShaderCompiler.h"
#include "07_D3D12Backend/D3D12Helpers.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sge::d3d12::detail
{
    constexpr UINT FrameCount = 3;
    constexpr UINT64 UploadArenaSize = 1024ull * 1024ull;
    constexpr std::size_t QueueClassCount = 3;

    struct ProgramRecord
    {
        shader::CompiledShader vertex;
        shader::CompiledShader pixel;
        shader::CompiledShader compute;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
        std::vector<UINT> rootParameterIndices;
    };

    struct ResourceRecord
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
        D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
        D3D12_GPU_DESCRIPTOR_HANDLE srvDescriptor{};
        D3D12_GPU_DESCRIPTOR_HANDLE uavDescriptor{};
        bool hasRtv = false;
        bool hasDsv = false;
        bool hasSrv = false;
        bool hasUav = false;
    };

    struct FrameContext
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12Resource> uploadArena;
        std::byte* mappedUpload = nullptr;
        UINT64 uploadOffset = 0;
        UINT64 fenceValue = 0;
    };

    struct QueueSyncState
    {
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        HANDLE event = nullptr;
        UINT64 nextSignalValue = 1;
    };

    class Backend final : public runtime::IRenderBackend
    {
    public:
        explicit Backend(platform::NativeSurface surface);
        ~Backend() override;

        [[nodiscard]] gpu::DeviceCapabilities Capabilities() const override;

        void Execute(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan) override;

        void WaitIdle() override;

    private:
        void EnableDebugLayer();
        void CreateDeviceAndQueue();
        void CreateSwapChain();
        void CreateDescriptorHeaps();
        void CreateFrameContexts();
        void CreateFence();
        void CreateRenderTargets();
        void CreateDepthBuffer();
        void ResizeIfNeeded();
        [[nodiscard]] ID3D12CommandQueue* QueueFor(
            gpu::QueueClass queue) const noexcept;
        [[nodiscard]] static D3D12_COMMAND_LIST_TYPE CommandListType(
            gpu::QueueClass queue) noexcept;
        [[nodiscard]] static std::size_t QueueIndex(
            gpu::QueueClass queue) noexcept;
        [[nodiscard]] QueueSyncState& SyncFor(gpu::QueueClass queue) noexcept;
        [[nodiscard]] const QueueSyncState& SyncFor(
            gpu::QueueClass queue) const noexcept;
        [[nodiscard]] UINT64 SignalQueue(gpu::QueueClass queue);
        void WaitForQueueValue(
            gpu::QueueClass waitingQueue,
            gpu::QueueClass sourceQueue,
            UINT64 value);

        void EnsureCompiled(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan);

        [[nodiscard]] ResourceRecord CreateStaticBuffer(
            const ir::ResourceDeclaration& declaration);

        [[nodiscard]] ResourceRecord CreateTexture(
            const ir::ResourceDeclaration& declaration,
            ID3D12Heap* aliasHeap = nullptr);

        [[nodiscard]] ProgramRecord CreateProgram(
            const ir::ProgramDeclaration& declaration);

        [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12RootSignature>
            CreateRootSignature(
                const ir::ProgramDeclaration& declaration,
                std::vector<UINT>& rootParameterIndices);

        [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12PipelineState>
            CreatePipeline(const compiler::ExecutableKey& executable);

        void WaitForFrame(FrameContext& frame);

        [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS UploadConstants(
            FrameContext& frame,
            const ir::ResourceDeclaration& declaration);

        void ExecuteRasterWork(
            const ir::SemanticModule& module,
            const ir::WorkDeclaration& work,
            FrameContext& frame);

        void ExecuteComputeWork(
            const ir::SemanticModule& module,
            const ir::WorkDeclaration& work,
            FrameContext& frame);

        void ExecuteCopyWork(
            const ir::SemanticModule& module,
            const ir::WorkDeclaration& work);

        void BindResources(
            const ir::SemanticModule& module,
            gpu::ProgramId program,
            const std::vector<ir::ResourceBinding>& bindings,
            FrameContext& frame,
            bool compute);

        void Transition(
            gpu::ResourceId resource,
            gpu::AbstractState abstractState);

        void ActivateAliasedResource(gpu::ResourceId resource);

        HWND window_ = nullptr;
        UINT width_ = 0;
        UINT height_ = 0;
        UINT frameIndex_ = 0;
        UINT rtvIncrement_ = 0;

        Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_;
        Microsoft::WRL::ComPtr<ID3D12Device> device_;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> computeQueue_;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> copyQueue_;
        Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap_;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> shaderHeap_;
        UINT dsvIncrement_ = 0;
        UINT shaderIncrement_ = 0;
        UINT nextRtvDescriptor_ = FrameCount;
        UINT nextDsvDescriptor_ = 1;
        UINT nextShaderDescriptor_ = 0;

        std::array<
            Microsoft::WRL::ComPtr<ID3D12Resource>,
            FrameCount> backBuffers_;

        std::array<
            D3D12_RESOURCE_STATES,
            FrameCount> backBufferStates_{};

        Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer_;
        D3D12_RESOURCE_STATES depthState_ =
            D3D12_RESOURCE_STATE_DEPTH_WRITE;

        std::array<FrameContext, FrameCount> frames_;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;

        std::array<QueueSyncState, QueueClassCount> queueSyncStates_;
        std::vector<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>>
            inFlightAllocators_;
        std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
            inFlightLists_;

        shader::ShaderCompiler shaderCompiler_;

        std::unordered_map<
            gpu::ResourceId,
            ResourceRecord,
            foundation::StrongIdHash<gpu::ResourceTag>> staticResources_;

        std::unordered_map<
            gpu::ResourceId,
            gpu::PhysicalAllocationId,
            foundation::StrongIdHash<gpu::ResourceTag>> physicalAllocations_;

        std::unordered_map<
            gpu::PhysicalAllocationId,
            Microsoft::WRL::ComPtr<ID3D12Heap>,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>> allocationHeaps_;

        std::unordered_map<
            gpu::PhysicalAllocationId,
            gpu::ResourceId,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>> activeAliasedResources_;

        std::unordered_map<
            gpu::ProgramId,
            ProgramRecord,
            foundation::StrongIdHash<gpu::ProgramTag>> programs_;

        std::unordered_map<
            compiler::ExecutableKey,
            Microsoft::WRL::ComPtr<ID3D12PipelineState>,
            ExecutableKeyHash> pipelines_;

        std::optional<gpu::ResourceId> presentationResource_;
        std::optional<gpu::ResourceId> depthResource_;
        std::optional<std::size_t> compiledStructureHash_;
    };
}
