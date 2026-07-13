#pragma once

#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"

#include <memory>

namespace sge::d3d12
{
    struct BackendConfiguration
    {
        bool forceWarp = false;
        bool enableDeviceRecovery = true;
    };

    [[nodiscard]] std::unique_ptr<runtime::IRenderBackend> CreateBackend(
        platform::NativeSurface surface,
        BackendConfiguration configuration = {});

    // D3D12-specific interop remains outside the API-independent runtime.
    [[nodiscard]] std::shared_ptr<runtime::IExternalResource>
        WrapExternalResource(void* nativeResource);
    [[nodiscard]] void* NativeDevice(runtime::IRenderBackend& backend);
    [[nodiscard]] bool RecreateDeviceForTesting(
        runtime::IRenderBackend& backend);
}
