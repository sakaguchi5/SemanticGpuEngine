#pragma once

#include "00_Foundation/Hash.h"
#include "04_RenderCompiler/CompiledRenderPackage.h"

#include <d3d12.h>
#include <cstdint>

namespace sge::d3d12::detail
{
    void ThrowIfFailed(HRESULT result, const char* operation);
    [[nodiscard]] UINT64 AlignUp(UINT64 value, UINT64 alignment) noexcept;
    [[nodiscard]] D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept;
    [[nodiscard]] D3D12_RESOURCE_DESC BufferDescription(UINT64 size) noexcept;
    [[nodiscard]] DXGI_FORMAT NativeFormat(gpu::ResourceFormat format) noexcept;
    [[nodiscard]] D3D12_RESOURCE_DESC TextureDescription(
        const ir::TextureDescription& texture,
        UINT fallbackWidth,
        UINT fallbackHeight) noexcept;
    [[nodiscard]] D3D12_RASTERIZER_DESC RasterizerDescription() noexcept;
    [[nodiscard]] D3D12_RASTERIZER_DESC RasterizerDescription(
        const ir::RasterState& state) noexcept;
    [[nodiscard]] D3D12_BLEND_DESC BlendDescription(
        gpu::CompositionMode composition) noexcept;
    [[nodiscard]] D3D12_DEPTH_STENCIL_DESC DepthDescription(
        gpu::DepthMode mode) noexcept;
    [[nodiscard]] D3D12_RESOURCE_STATES NativeState(
        gpu::AbstractState state) noexcept;
    [[nodiscard]] DXGI_FORMAT NativeVertexFormat(
        ir::VertexElementFormat format) noexcept;
    [[nodiscard]] DXGI_FORMAT NativeIndexFormat(
        ir::IndexFormat format) noexcept;
    [[nodiscard]] D3D_PRIMITIVE_TOPOLOGY NativeTopology(
        gpu::PrimitiveTopology topology) noexcept;
    [[nodiscard]] D3D12_PRIMITIVE_TOPOLOGY_TYPE NativeTopologyType(
        gpu::PrimitiveTopology topology) noexcept;

    struct ExecutableKeyHash
    {
        [[nodiscard]] std::size_t operator()(
            const compiler::ExecutableKey& key) const noexcept;
    };
}
