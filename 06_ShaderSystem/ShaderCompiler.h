#pragma once

#include "03_RenderIR/RenderIR.h"

#include <d3dcompiler.h>
#include <wrl/client.h>

#include <filesystem>
#include <string>
#include <vector>

namespace sge::shader
{
    struct CompiledShader
    {
        Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    };

    struct ReflectedBinding
    {
        gpu::ProgramParameterKind kind =
            gpu::ProgramParameterKind::ConstantBuffer;
        gpu::ProgramStage stage = gpu::ProgramStage::AllGraphics;
        std::uint32_t registerIndex = 0;
        std::uint32_t registerSpace = 0;
        std::string name;
    };

    class ShaderCompiler
    {
    public:
        [[nodiscard]] CompiledShader Compile(
            const std::filesystem::path& path,
            const std::string& entryPoint,
            const std::string& target) const;

        [[nodiscard]] std::vector<ReflectedBinding> ReflectBindings(
            const CompiledShader& shader,
            gpu::ProgramStage stage) const;

        void ValidateInterface(
            const std::vector<gpu::ProgramParameter>& declared,
            const std::vector<ReflectedBinding>& reflected) const;

        void ValidateRasterInterface(
            const ir::ProgramInterface& declared,
            const CompiledShader& vertex,
            const CompiledShader& pixel) const;
    };
}
