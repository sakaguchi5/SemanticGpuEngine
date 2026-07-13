#pragma once

#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"
#include "06_ShaderSystem/ShaderCompiler.h"
#include "07_D3D12Backend/D3D12Backend.h"
#include "07_D3D12Backend/D3D12Helpers.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
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
        std::vector<D3D12_RESOURCE_STATES> subresourceStates;
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

    struct ShaderViewKey
    {
        ir::ResourceView view;
        gpu::ProgramParameterKind kind =
            gpu::ProgramParameterKind::ShaderResource;
        auto operator<=>(const ShaderViewKey&) const = default;
    };

    struct CachedShaderView
    {
        std::vector<D3D12_GPU_DESCRIPTOR_HANDLE> instances;
    };

    struct PackageShaderViewKey
    {
        compiler::NormalizedResourceView view;
        gpu::ProgramParameterKind kind =
            gpu::ProgramParameterKind::ShaderResource;
        auto operator<=>(const PackageShaderViewKey&) const = default;
    };

    struct PackageAttachmentViewKey
    {
        compiler::NormalizedResourceView view;
        bool depth = false;
        auto operator<=>(const PackageAttachmentViewKey&) const = default;
    };

    struct CachedPackageShaderView
    {
        std::vector<D3D12_GPU_DESCRIPTOR_HANDLE> instances;
    };

    struct CachedPackageAttachmentView
    {
        std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> instances;
    };

    struct QueueFrameContext
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
            commandLists;
        std::size_t usedCommandLists = 0;
        UINT64 completionFenceValue = 0;
    };

    struct FrameContext
    {
        std::uint32_t slot = 0;
        std::array<QueueFrameContext, QueueClassCount> queues;
        Microsoft::WRL::ComPtr<ID3D12Resource> uploadArena;
        std::byte* mappedUpload = nullptr;
        UINT64 uploadOffset = 0;
    };

    struct QueueTimelinePoint
    {
        gpu::QueueClass queue = gpu::QueueClass::Direct;
        UINT64 value = 0;
    };

    struct PhysicalInstanceUsage
    {
        QueueTimelinePoint lastWriter;
        std::array<UINT64, QueueClassCount> lastReaders{};
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
        Backend(
            platform::NativeSurface surface,
            BackendConfiguration configuration);
        ~Backend() override;

        [[nodiscard]] gpu::DeviceCapabilities Capabilities() const override;

        void Execute(
            const compiler::CompiledRenderPackage& package,
            const runtime::FrameInvocation& invocation) override;

        void Execute(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan) override;

        void WaitIdle() override;

    private:
        void EnableDebugLayer();
        void ValidateDebugLayer();
        void CreateDeviceAndQueue();
        void CreateSwapChain();
        void CreateDescriptorHeaps();
        void CreateFrameContexts();
        void CreateFence();
        void CreateRenderTargets();
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
        void WaitForCpuQueueValue(gpu::QueueClass queue, UINT64 value);
        void WaitForQueueValue(
            gpu::QueueClass waitingQueue,
            gpu::QueueClass sourceQueue,
            UINT64 value);
        void WaitForTemporalAccess(
            const gpu::ResourceAccess& access,
            gpu::QueueClass queue);
        void RecordTemporalAccess(
            const gpu::ResourceAccess& access,
            gpu::QueueClass queue,
            UINT64 completionValue);
        void BeginFrame(FrameContext& frame);
        [[nodiscard]] ID3D12GraphicsCommandList* AcquireCommandList(
            FrameContext& frame, gpu::QueueClass queue);

        void EnsureCompiled(
            const ir::SemanticModule& module,
            const compiler::ExecutionPlan& plan);
        void EnsurePackageCompiled(
            const compiler::CompiledRenderPackage& package);
        void EnsurePackageDescriptorCapacity(
            const compiler::CompiledRenderPackage& package);
        void EnsureUploadCapacity(UINT64 requiredBytes);
        void UploadPackageInitialBufferData(
            const compiler::CompiledRenderPackage& package);
        void InitializePackagePersistentReadStates(
            const compiler::CompiledRenderPackage& package);
        void InitializePersistentReadStates();

        [[nodiscard]] ResourceRecord CreateStaticBuffer(
            const ir::ResourceDeclaration& declaration,
            ID3D12Heap* aliasHeap = nullptr);
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
        [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12PipelineState>
            CreatePackagePipeline(const compiler::ExecutableKey& executable);

        [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS UploadConstants(
            FrameContext& frame,
            const ir::ResourceDeclaration& declaration);
        [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE ResolveShaderDescriptor(
            const ir::SemanticModule& module,
            const ir::ResourceView& view,
            gpu::ProgramParameterKind kind,
            std::uint32_t frameLag);
        [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE
            ResolvePackageShaderDescriptor(
                const compiler::NormalizedResourceView& view,
                gpu::ProgramParameterKind kind,
                std::uint32_t frameLag);
        [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE
            ResolvePackageAttachmentDescriptor(
                const compiler::NormalizedResourceView& view,
                bool depth,
                std::uint32_t frameLag = 0);
        [[nodiscard]] D3D12_VERTEX_BUFFER_VIEW BuildVertexView(
            const ir::SemanticModule& module,
            const ir::ResourceView& view);
        [[nodiscard]] D3D12_VERTEX_BUFFER_VIEW BuildPackageVertexView(
            const compiler::NormalizedResourceView& view,
            std::uint32_t frameLag = 0);
        [[nodiscard]] D3D12_INDEX_BUFFER_VIEW BuildPackageIndexView(
            const compiler::CompiledIndexStream& stream,
            std::uint32_t frameLag = 0);

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
        void ExecutePackageRaster(
            const compiler::CompiledRasterCommand& command,
            FrameContext& frame);
        void ExecutePackageCompute(
            const compiler::CompiledComputeCommand& command,
            FrameContext& frame);
        void ExecutePackageCopy(
            const compiler::CompiledCopyCommand& command);

        void BindResources(
            const ir::SemanticModule& module,
            gpu::ProgramId program,
            const std::vector<ir::ResourceBinding>& bindings,
            FrameContext& frame,
            bool compute);
        void BindPackageResources(
            gpu::ProgramId program,
            const std::vector<compiler::CompiledBinding>& bindings,
            FrameContext& frame,
            bool compute);

        void Transition(
            gpu::ResourceId resource,
            gpu::AbstractState abstractState,
            std::uint32_t frameLag = 0);
        void TransitionPackageRange(
            const compiler::NormalizedResourceView& view,
            gpu::AbstractState abstractState,
            std::uint32_t frameLag = 0);
        void ValidateCopyQueueRequirement(
            const compiler::RangeStateRequirement& requirement);
        [[nodiscard]] bool PackageRangeIsCommon(
            const compiler::NormalizedResourceView& view,
            std::uint32_t frameLag = 0) const;
        void ActivateAliasedResource(
            gpu::ResourceId resource,
            std::uint32_t frameLag = 0);

        [[nodiscard]] std::size_t ResolveInstanceIndex(
            gpu::ResourceId resource,
            std::uint32_t frameLag = 0) const;
        [[nodiscard]] ResourceRecord* ResolveResource(
            gpu::ResourceId resource,
            std::uint32_t frameLag = 0);
        [[nodiscard]] const ResourceRecord* ResolveResource(
            gpu::ResourceId resource,
            std::uint32_t frameLag = 0) const;

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
        UINT rtvDescriptorCapacity_ = 64;
        UINT dsvDescriptorCapacity_ = 16;
        UINT shaderDescriptorCapacity_ = 128;
        UINT64 uploadArenaCapacity_ = UploadArenaSize;

        std::array<
            Microsoft::WRL::ComPtr<ID3D12Resource>,
            FrameCount> backBuffers_;
        std::array<
            D3D12_RESOURCE_STATES,
            FrameCount> backBufferStates_{};
        std::array<FrameContext, FrameCount> frames_;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
        std::uint64_t frameNumber_ = 0;
        std::uint32_t activeFrameSlot_ = 0;
        gpu::QueueClass activeQueue_ = gpu::QueueClass::Direct;
        BackendConfiguration configuration_;

        std::array<QueueSyncState, QueueClassCount> queueSyncStates_;
        shader::ShaderCompiler shaderCompiler_;
        std::unordered_map<
            gpu::ResourceId,
            std::vector<ResourceRecord>,
            foundation::StrongIdHash<gpu::ResourceTag>> resources_;
        std::unordered_map<
            gpu::ResourceId,
            compiler::ResourceInstancePlan,
            foundation::StrongIdHash<gpu::ResourceTag>> resourceInstancePlans_;
        std::unordered_map<
            gpu::ResourceId,
            D3D12_RESOURCE_STATES,
            foundation::StrongIdHash<gpu::ResourceTag>> persistentReadStates_;
        std::map<ShaderViewKey, CachedShaderView> customShaderViews_;
        std::map<PackageShaderViewKey, CachedPackageShaderView>
            packageShaderViews_;
        std::map<PackageAttachmentViewKey, CachedPackageAttachmentView>
            packageAttachmentViews_;
        std::unordered_map<
            gpu::ResourceId,
            gpu::PhysicalAllocationId,
            foundation::StrongIdHash<gpu::ResourceTag>> physicalAllocations_;
        std::unordered_map<
            gpu::PhysicalAllocationId,
            std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>>,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>> allocationHeaps_;
        std::unordered_map<
            gpu::PhysicalAllocationId,
            std::vector<std::optional<gpu::ResourceId>>,
            foundation::StrongIdHash<gpu::PhysicalAllocationTag>>
            activeAliasedResources_;
        std::unordered_map<
            gpu::ResourceId,
            std::vector<PhysicalInstanceUsage>,
            foundation::StrongIdHash<gpu::ResourceTag>> temporalInstanceUsages_;
        std::unordered_map<
            gpu::ProgramId,
            ProgramRecord,
            foundation::StrongIdHash<gpu::ProgramTag>> programs_;
        std::unordered_map<
            compiler::ExecutableKey,
            Microsoft::WRL::ComPtr<ID3D12PipelineState>,
            ExecutableKeyHash> pipelines_;
        std::optional<gpu::ResourceId> presentationResource_;
        std::optional<std::size_t> compiledStructureHash_;
        std::optional<std::size_t> compiledPackageHash_;
        const compiler::CompiledRenderPackage* activePackage_ = nullptr;
        const runtime::FrameInvocation* activeInvocation_ = nullptr;
    };
}

// Existing compilation units use this symbolic limit inside Backend member
// functions. Redirect it to the package-managed capacity without duplicating
// the established upload/binding implementation.
#define UploadArenaSize uploadArenaCapacity_
