#include "06_ShaderSystem/ShaderCompiler.h"

#include <d3d11shader.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace
{
    sge::gpu::ProgramParameterKind ParameterKind(
        D3D_SHADER_INPUT_TYPE type)
    {
        using Kind = sge::gpu::ProgramParameterKind;

        switch (type)
        {
        case D3D_SIT_CBUFFER:
            return Kind::ConstantBuffer;

        case D3D_SIT_TEXTURE:
        case D3D_SIT_STRUCTURED:
        case D3D_SIT_BYTEADDRESS:
        case D3D_SIT_TBUFFER:
            return Kind::ShaderResource;

        case D3D_SIT_UAV_RWTYPED:
        case D3D_SIT_UAV_RWSTRUCTURED:
        case D3D_SIT_UAV_RWBYTEADDRESS:
        case D3D_SIT_UAV_APPEND_STRUCTURED:
        case D3D_SIT_UAV_CONSUME_STRUCTURED:
        case D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER:
            return Kind::UnorderedAccess;

        case D3D_SIT_SAMPLER:
            return Kind::Sampler;

        default:
            throw std::runtime_error(
                "Shader reflection found an unsupported binding kind.");
        }
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
            path.c_str(),
            nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint.c_str(),
            target.c_str(),
            flags,
            0,
            &bytecode,
            &errors);

        if (FAILED(result))
        {
            std::string message =
                "Shader compilation failed: "
                + path.string()
                + " ["
                + entryPoint
                + "]";

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
        Microsoft::WRL::ComPtr<ID3D11ShaderReflection> reflection;

        const HRESULT result = D3DReflect(
            shader.bytecode->GetBufferPointer(),
            shader.bytecode->GetBufferSize(),
            IID_PPV_ARGS(&reflection));

        if (FAILED(result))
        {
            throw std::runtime_error("D3DReflect failed.");
        }

        D3D11_SHADER_DESC shaderDescription{};
        if (FAILED(reflection->GetDesc(&shaderDescription)))
        {
            throw std::runtime_error("Shader reflection GetDesc failed.");
        }

        std::vector<ReflectedBinding> bindings;
        bindings.reserve(shaderDescription.BoundResources);

        for (UINT index = 0;
             index < shaderDescription.BoundResources;
             ++index)
        {
            D3D11_SHADER_INPUT_BIND_DESC binding{};
            if (FAILED(reflection->GetResourceBindingDesc(index, &binding)))
            {
                throw std::runtime_error(
                    "Shader reflection binding query failed.");
            }

            bindings.push_back({
                .kind = ParameterKind(binding.Type),
                .stage = stage,
                .registerIndex = binding.BindPoint,
                .registerSpace = 0,
                .name = binding.Name == nullptr ? "" : binding.Name
            });
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
                declared.begin(),
                declared.end(),
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
            {
                throw std::runtime_error(
                    "Shader interface does not match ProgramDeclaration: "
                    + binding.name);
            }
        }

        for (const auto& parameter : declared)
        {
            const auto found = std::find_if(
                reflected.begin(),
                reflected.end(),
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
            {
                throw std::runtime_error(
                    "Declared program parameter is not used by the shader.");
            }
        }
    }
}
