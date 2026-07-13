#include "06_ShaderSystem/ShaderCompiler.h"

#include <d3d11shader.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
    sge::gpu::ProgramParameterKind ParameterKind(D3D_SHADER_INPUT_TYPE type)
    {
        using Kind = sge::gpu::ProgramParameterKind;
        switch (type)
        {
        case D3D_SIT_CBUFFER: return Kind::ConstantBuffer;
        case D3D_SIT_TEXTURE:
        case D3D_SIT_STRUCTURED:
        case D3D_SIT_BYTEADDRESS:
        case D3D_SIT_TBUFFER: return Kind::ShaderResource;
        case D3D_SIT_UAV_RWTYPED:
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_RWBYTEADDRESS:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
            return Kind::UnorderedAccess;
        case D3D_SIT_SAMPLER: return Kind::Sampler;
        default:
            throw std::runtime_error(
                "Shader reflection found an unsupported binding kind.");
        }
    }

    Microsoft::WRL::ComPtr<ID3D11ShaderReflection> Reflect(
        const sge::shader::CompiledShader& shader)
    {
        Microsoft::WRL::ComPtr<ID3D11ShaderReflection> reflection;
        const HRESULT result = D3DReflect(
            shader.bytecode->GetBufferPointer(),
            shader.bytecode->GetBufferSize(),
            IID_PPV_ARGS(&reflection));
        if (FAILED(result)) throw std::runtime_error("D3DReflect failed.");
        return reflection;
    }

    bool EqualSemantic(std::string_view left, const char* right) noexcept
    {
        if (right == nullptr || left.size() != std::char_traits<char>::length(right))
            return false;
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            const auto a = static_cast<unsigned char>(left[index]);
            const auto b = static_cast<unsigned char>(right[index]);
            if (std::toupper(a) != std::toupper(b)) return false;
        }
        return true;
    }

    UINT ComponentCount(BYTE mask) noexcept
    {
        UINT count = 0;
        for (UINT bit = 0; bit < 4; ++bit)
            count += (mask & (1u << bit)) != 0;
        return count;
    }

    bool FormatMatches(
        sge::ir::VertexElementFormat format,
        D3D_REGISTER_COMPONENT_TYPE component,
        BYTE mask) noexcept
    {
        using F = sge::ir::VertexElementFormat;
        const UINT expectedCount =
            format == F::Float || format == F::Uint ? 1u
            : format == F::Float2 || format == F::Uint2 ? 2u
            : format == F::Float3 || format == F::Uint3 ? 3u : 4u;
        const bool expectedUint = format == F::Uint || format == F::Uint2
            || format == F::Uint3 || format == F::Uint4;
        return ComponentCount(mask) == expectedCount
            && (expectedUint
                ? component == D3D_REGISTER_COMPONENT_UINT32
                : component == D3D_REGISTER_COMPONENT_FLOAT32);
    }
}

namespace sge::shader
{
    CompiledShader ShaderCompiler::Compile(
        const std::filesystem::path& path,
        const std::string& entryPoint,
        const std::string& target) const
    {
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
        Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
        Microsoft::WRL::ComPtr<ID3DBlob> errors;
        const HRESULT result = D3DCompileFromFile(
            path.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint.c_str(), target.c_str(), flags, 0,
            &bytecode, &errors);
        if (FAILED(result))
        {
            std::string message = "Shader compilation failed: "
                + path.string() + " [" + entryPoint + "]";
            if (errors)
            {
                message += "\n";
                message.append(
                    static_cast<const char*>(errors->GetBufferPointer()),
                    errors->GetBufferSize());
            }
            throw std::runtime_error(message);
        }
        return CompiledShader{.bytecode = std::move(bytecode)};
    }

    std::vector<ReflectedBinding> ShaderCompiler::ReflectBindings(
        const CompiledShader& shader,
        gpu::ProgramStage stage) const
    {
        const auto reflection = Reflect(shader);
        D3D11_SHADER_DESC description{};
        if (FAILED(reflection->GetDesc(&description)))
            throw std::runtime_error("Shader reflection GetDesc failed.");
        std::vector<ReflectedBinding> bindings;
        bindings.reserve(description.BoundResources);
        for (UINT index = 0; index < description.BoundResources; ++index)
        {
            D3D11_SHADER_INPUT_BIND_DESC binding{};
            if (FAILED(reflection->GetResourceBindingDesc(index, &binding)))
                throw std::runtime_error(
                    "Shader reflection binding query failed.");
            bindings.push_back({
                .kind = ParameterKind(binding.Type),
                .stage = stage,
                .registerIndex = binding.BindPoint,
                .registerSpace = 0,
                .name = binding.Name == nullptr ? "" : binding.Name});
        }
        return bindings;
    }

    void ShaderCompiler::ValidateInterface(
        const std::vector<gpu::ProgramParameter>& declared,
        const std::vector<ReflectedBinding>& reflected) const
    {
        for (const auto& binding : reflected)
        {
            const auto found = std::find_if(
                declared.begin(), declared.end(),
                [&](const gpu::ProgramParameter& parameter)
                {
                    const bool stageMatches =
                        parameter.stage == gpu::ProgramStage::AllGraphics
                        || parameter.stage == binding.stage;
                    return stageMatches
                        && parameter.kind == binding.kind
                        && parameter.registerIndex == binding.registerIndex
                        && parameter.registerSpace == binding.registerSpace;
                });
            if (found == declared.end())
                throw std::runtime_error(
                    "Shader interface does not match ProgramDeclaration: "
                    + binding.name);
        }
        for (const auto& parameter : declared)
        {
            const auto found = std::find_if(
                reflected.begin(), reflected.end(),
                [&](const ReflectedBinding& binding)
                {
                    const bool stageMatches =
                        parameter.stage == gpu::ProgramStage::AllGraphics
                        || parameter.stage == binding.stage;
                    return stageMatches
                        && parameter.kind == binding.kind
                        && parameter.registerIndex == binding.registerIndex
                        && parameter.registerSpace == binding.registerSpace;
                });
            if (found == reflected.end())
                throw std::runtime_error(
                    "Declared program parameter is not used by the shader.");
        }
    }

    void ShaderCompiler::ValidateRasterInterface(
        const ir::ProgramInterface& declared,
        const CompiledShader& vertex,
        const CompiledShader& pixel) const
    {
        const auto vertexReflection = Reflect(vertex);
        D3D11_SHADER_DESC vertexDescription{};
        if (FAILED(vertexReflection->GetDesc(&vertexDescription)))
            throw std::runtime_error("Vertex signature reflection failed.");

        std::size_t reflectedUserInputs = 0;
        for (UINT index = 0; index < vertexDescription.InputParameters; ++index)
        {
            D3D11_SIGNATURE_PARAMETER_DESC input{};
            if (FAILED(vertexReflection->GetInputParameterDesc(index, &input)))
                throw std::runtime_error("Vertex input reflection failed.");
            if (input.SystemValueType != D3D_NAME_UNDEFINED) continue;
            ++reflectedUserInputs;
            const auto found = std::find_if(
                declared.vertexInputs.begin(), declared.vertexInputs.end(),
                [&](const ir::VertexInputElement& element)
                {
                    return element.semanticIndex == input.SemanticIndex
                        && EqualSemantic(element.semanticName, input.SemanticName)
                        && FormatMatches(
                            element.format, input.ComponentType, input.Mask);
                });
            if (found == declared.vertexInputs.end())
                throw std::runtime_error(
                    "Vertex shader input signature does not match ProgramInterface.");
        }
        if (reflectedUserInputs != declared.vertexInputs.size())
            throw std::runtime_error(
                "ProgramInterface declares a vertex input unused by the shader.");

        const auto pixelReflection = Reflect(pixel);
        D3D11_SHADER_DESC pixelDescription{};
        if (FAILED(pixelReflection->GetDesc(&pixelDescription)))
            throw std::runtime_error("Pixel signature reflection failed.");
        std::uint32_t colorCount = 0;
        bool writesDepth = false;
        for (UINT index = 0; index < pixelDescription.OutputParameters; ++index)
        {
            D3D11_SIGNATURE_PARAMETER_DESC output{};
            if (FAILED(pixelReflection->GetOutputParameterDesc(index, &output)))
                throw std::runtime_error("Pixel output reflection failed.");
            if (output.SystemValueType == D3D_NAME_TARGET
                || EqualSemantic("SV_Target", output.SemanticName))
            {
                colorCount = std::max(
                    colorCount, output.SemanticIndex + 1u);
            }
            else if (output.SystemValueType == D3D_NAME_DEPTH
                || output.SystemValueType == D3D_NAME_DEPTH_GREATER_EQUAL
                || output.SystemValueType == D3D_NAME_DEPTH_LESS_EQUAL)
            {
                writesDepth = true;
            }
        }
        if (colorCount != declared.colorOutputCount)
            throw std::runtime_error(
                "Pixel shader color outputs do not match ProgramInterface.");
        if (writesDepth && !declared.depthAttachmentAllowed)
            throw std::runtime_error(
                "Pixel shader writes depth but ProgramInterface forbids it.");
    }
}
