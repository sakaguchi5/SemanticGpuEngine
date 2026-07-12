#pragma once

#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"

#include <memory>

namespace sge::d3d12
{
    struct BackendConfiguration
    {
        bool forceWarp = false;
    };

    [[nodiscard]] std::unique_ptr<runtime::IRenderBackend> CreateBackend(
        platform::NativeSurface surface,
        BackendConfiguration configuration = {});
}
