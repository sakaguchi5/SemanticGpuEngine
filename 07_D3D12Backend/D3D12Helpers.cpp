#include "07_D3D12Backend/D3D12Helpers.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace sge::d3d12::detail
{
    void ThrowIfFailed(HRESULT result, const char* operation)
    {
        if (SUCCEEDED(result)) return;
        char hexadecimal[16]{};
        sprintf_s(hexadecimal, "%08X", static_cast<unsigned>(result));
        throw std::runtime_error(std::string(operation) + " failed. HRESULT=0x" + hexadecimal);
    }

    UINT64 AlignUp(UINT64 value, UINT64 alignment) noexcept
    {
        return alignment == 0 ? value : (value + alignment - 1ull) & ~(alignment - 1ull);
    }

    D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) noexcept
    {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = type;
        properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        properties.CreationNodeMask = 1;
        properties.VisibleNodeMask = 1;
        return properties;
    }

    D3D12_RESOURCE_DESC BufferDescription(UINT64 size) noexcept
    {
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = size;
        description.Height = 1;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.Format = DXGI_FORMAT_UNKNOWN;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return description;
    }

    DXGI_FORMAT NativeFormat(gpu::ResourceFormat format) noexcept
    {
        using F = gpu::ResourceFormat;
        switch (format)
        {
        case F::R8Unorm: return DXGI_FORMAT_R8_UNORM;
        case F::Rg8Unorm: return DXGI_FORMAT_R8G8_UNORM;
        case F::Rgba8Unorm: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case F::Bgra8Unorm: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case F::R16Float: return DXGI_FORMAT_R16_FLOAT;
        case F::Rg16Float: return DXGI_FORMAT_R16G16_FLOAT;
        case F::Rgba16Float: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case F::R32Float: return DXGI_FORMAT_R32_FLOAT;
        case F::Rg32Float: return DXGI_FORMAT_R32G32_FLOAT;
        case F::Rgba32Float: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case F::R32Uint: return DXGI_FORMAT_R32_UINT;
        case F::Rg32Uint: return DXGI_FORMAT_R32G32_UINT;
        case F::Rgba32Uint: return DXGI_FORMAT_R32G32B32A32_UINT;
        case F::Depth24Stencil8: return DXGI_FORMAT_D24_UNORM_S8_UINT;
        case F::Depth32Float: return DXGI_FORMAT_D32_FLOAT;
        default: return DXGI_FORMAT_UNKNOWN;
        }
    }

    D3D12_RESOURCE_DESC TextureDescription(
        const ir::TextureDescription& texture,
        UINT fallbackWidth,
        UINT fallbackHeight) noexcept
    {
        D3D12_RESOURCE_DESC description{};
        description.Dimension = texture.dimension == gpu::ResourceKind::Texture3D
            ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
            : texture.dimension == gpu::ResourceKind::Texture1D
                ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
                : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        description.Width = texture.width == 0 ? fallbackWidth : texture.width;
        description.Height = texture.dimension == gpu::ResourceKind::Texture1D
            ? 1u : (texture.height == 0 ? fallbackHeight : texture.height);
        description.DepthOrArraySize = texture.dimension == gpu::ResourceKind::Texture3D
            ? texture.depth : texture.arrayLayers;
        description.MipLevels = texture.mipLevels;
        description.Format = NativeFormat(texture.format);
        description.SampleDesc.Count = texture.sampleCount;
        description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        const auto usage = static_cast<std::uint32_t>(texture.usage);
        if ((usage & static_cast<std::uint32_t>(ir::TextureUsage::ColorAttachment)) != 0)
            description.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if ((usage & static_cast<std::uint32_t>(ir::TextureUsage::DepthAttachment)) != 0)
            description.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if ((usage & static_cast<std::uint32_t>(ir::TextureUsage::Storage)) != 0)
            description.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        return description;
    }

    D3D12_RASTERIZER_DESC RasterizerDescription() noexcept
    {
        return RasterizerDescription(ir::RasterState{});
    }

    D3D12_RASTERIZER_DESC RasterizerDescription(const ir::RasterState& state) noexcept
    {
        D3D12_RASTERIZER_DESC description{};
        description.FillMode = state.fill == ir::FillMode::Wireframe
            ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
        switch (state.cull)
        {
        case ir::CullMode::Front: description.CullMode = D3D12_CULL_MODE_FRONT; break;
        case ir::CullMode::Back: description.CullMode = D3D12_CULL_MODE_BACK; break;
        default: description.CullMode = D3D12_CULL_MODE_NONE; break;
        }
        description.FrontCounterClockwise =
            state.frontFace == ir::FrontFace::CounterClockwise;
        description.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        description.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        description.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        description.DepthClipEnable = TRUE;
        description.MultisampleEnable = state.sampleCount > 1;
        description.AntialiasedLineEnable = FALSE;
        description.ForcedSampleCount = 0;
        description.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
        return description;
    }

    D3D12_BLEND_DESC BlendDescription(gpu::CompositionMode composition) noexcept
    {
        D3D12_BLEND_DESC description{};
        auto& target = description.RenderTarget[0];
        target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        target.LogicOp = D3D12_LOGIC_OP_NOOP;
        if (composition == gpu::CompositionMode::AlphaOver)
        {
            target.BlendEnable = TRUE;
            target.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            target.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
            target.BlendOp = D3D12_BLEND_OP_ADD;
            target.SrcBlendAlpha = D3D12_BLEND_ONE;
            target.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
            target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        }
        else
        {
            target.SrcBlend = D3D12_BLEND_ONE;
            target.DestBlend = D3D12_BLEND_ZERO;
            target.BlendOp = D3D12_BLEND_OP_ADD;
            target.SrcBlendAlpha = D3D12_BLEND_ONE;
            target.DestBlendAlpha = D3D12_BLEND_ZERO;
            target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        }
        return description;
    }

    D3D12_DEPTH_STENCIL_DESC DepthDescription(gpu::DepthMode mode) noexcept
    {
        D3D12_DEPTH_STENCIL_DESC description{};
        description.DepthEnable = mode != gpu::DepthMode::Disabled;
        description.DepthWriteMask = mode == gpu::DepthMode::ReadWrite
            ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        description.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        return description;
    }

    D3D12_RESOURCE_STATES NativeState(gpu::AbstractState state) noexcept
    {
        using State = gpu::AbstractState;
        switch (state)
        {
        case State::VertexRead:
        case State::ConstantRead:
            return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        case State::IndexRead: return D3D12_RESOURCE_STATE_INDEX_BUFFER;
        case State::IndirectRead: return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        case State::ProgramRead:
            return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case State::ProgramWrite: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case State::ColorWrite: return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case State::DepthRead: return D3D12_RESOURCE_STATE_DEPTH_READ;
        case State::DepthWrite: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case State::TransferRead: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case State::TransferWrite: return D3D12_RESOURCE_STATE_COPY_DEST;
        case State::Present: return D3D12_RESOURCE_STATE_PRESENT;
        default: return D3D12_RESOURCE_STATE_COMMON;
        }
    }

    DXGI_FORMAT NativeVertexFormat(ir::VertexElementFormat format) noexcept
    {
        using F = ir::VertexElementFormat;
        switch (format)
        {
        case F::Float: return DXGI_FORMAT_R32_FLOAT;
        case F::Float2: return DXGI_FORMAT_R32G32_FLOAT;
        case F::Float3: return DXGI_FORMAT_R32G32B32_FLOAT;
        case F::Float4: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case F::Uint: return DXGI_FORMAT_R32_UINT;
        case F::Uint2: return DXGI_FORMAT_R32G32_UINT;
        case F::Uint3: return DXGI_FORMAT_R32G32B32_UINT;
        case F::Uint4: return DXGI_FORMAT_R32G32B32A32_UINT;
        }
        return DXGI_FORMAT_UNKNOWN;
    }

    DXGI_FORMAT NativeIndexFormat(ir::IndexFormat format) noexcept
    {
        return format == ir::IndexFormat::Uint16
            ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
    }

    D3D_PRIMITIVE_TOPOLOGY NativeTopology(gpu::PrimitiveTopology topology) noexcept
    {
        return topology == gpu::PrimitiveTopology::LineList
            ? D3D_PRIMITIVE_TOPOLOGY_LINELIST : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }

    D3D12_PRIMITIVE_TOPOLOGY_TYPE NativeTopologyType(
        gpu::PrimitiveTopology topology) noexcept
    {
        return topology == gpu::PrimitiveTopology::LineList
            ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }

    std::size_t ExecutableKeyHash::operator()(
        const compiler::ExecutableKey& key) const noexcept
    {
        std::size_t seed = key.program.Value();
        foundation::HashEnum(seed, key.rasterState.topology);
        foundation::HashEnum(seed, key.rasterState.composition);
        foundation::HashEnum(seed, key.rasterState.depth);
        foundation::HashEnum(seed, key.rasterState.cull);
        foundation::HashEnum(seed, key.rasterState.fill);
        foundation::HashEnum(seed, key.rasterState.frontFace);
        foundation::HashCombine(seed, key.rasterState.sampleCount);
        for (const auto& input : key.vertexInputs)
        {
            foundation::HashCombine(seed, foundation::HashString(input.semanticName));
            foundation::HashCombine(seed, input.semanticIndex);
            foundation::HashEnum(seed, input.format);
            foundation::HashCombine(seed, input.inputSlot);
            foundation::HashCombine(seed, input.alignedByteOffset);
            foundation::HashEnum(seed, input.inputRate);
            foundation::HashCombine(seed, input.instanceStepRate);
        }
        for (const auto format : key.colorFormats) foundation::HashEnum(seed, format);
        foundation::HashEnum(seed, key.depthFormat);
        foundation::HashCombine(seed, key.compute);
        return seed;
    }
}
