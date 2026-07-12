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

namespace sge::d3d12::detail
{
    constexpr UINT FrameCount = 3;
    constexpr UINT64 UploadArenaSize = 1024ull * 1024ull;

    struct ProgramRecord
    {
        shader::CompiledShader vertex;
        shader::CompiledShader pixel;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature;
    };

    struct ResourceRecord
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
    };

    struct FrameContext
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12Resource> uploadArena;
        std::byte* mappedUpload = nullptr;
        UINT64 uploadOffset = 0;
        UINT64 fenceValue = 0;
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

        void EnsureCompiled(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan);

        [[nodiscard]] ResourceRecord CreateStaticBuffer(
            const ir::ResourceDeclaration& declaration);

        [[nodiscard]] ProgramRecord CreateProgram(
            const ir::ProgramDeclaration& declaration);

        [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12RootSignature>
            CreateRootSignature(
                const ir::ProgramDeclaration& declaration);

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

        void Transition(
            gpu::ResourceId resource,
            gpu::AbstractState abstractState);

        HWND window_ = nullptr;
        UINT width_ = 0;
        UINT height_ = 0;
        UINT frameIndex_ = 0;
        UINT rtvIncrement_ = 0;

        Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_;
        Microsoft::WRL::ComPtr<ID3D12Device> device_;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
        Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap_;

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

        Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
        HANDLE fenceEvent_ = nullptr;
        UINT64 nextFenceValue_ = 1;

        shader::ShaderCompiler shaderCompiler_;

        std::unordered_map<
            gpu::ResourceId,
            ResourceRecord,
            foundation::StrongIdHash<gpu::ResourceTag>> staticResources_;

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
