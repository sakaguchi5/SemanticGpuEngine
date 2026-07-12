#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <windows.h>

// WinGDI defines DeviceCapabilities as an A/W macro. The engine uses the
// same descriptive name for its API-independent target model.
#ifdef DeviceCapabilities
#undef DeviceCapabilities
#endif

namespace sge::platform
{
    struct NativeSurface
    {
        void* handle = nullptr;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    struct FrameTime
    {
        double deltaSeconds = 0.0;
        double elapsedSeconds = 0.0;
    };

    struct WindowDescription
    {
        std::wstring title = L"Semantic GPU Engine";
        std::uint32_t width = 1280;
        std::uint32_t height = 720;
    };

    class Win32Application
    {
    public:
        Win32Application(
            HINSTANCE instance,
            int showCommand,
            WindowDescription description);

        Win32Application(const Win32Application&) = delete;
        Win32Application& operator=(const Win32Application&) = delete;
        ~Win32Application();

        int Run(const std::function<void(const FrameTime&)>& frame);

        [[nodiscard]] NativeSurface Surface() const noexcept;

    private:
        static LRESULT CALLBACK WindowProcedure(
            HWND window,
            UINT message,
            WPARAM wParam,
            LPARAM lParam);

        LRESULT HandleMessage(
            HWND window,
            UINT message,
            WPARAM wParam,
            LPARAM lParam);

        void RegisterWindowClass();
        void CreateMainWindow(int showCommand);
        void UpdateClientSize() noexcept;

        HINSTANCE instance_ = nullptr;
        HWND window_ = nullptr;
        WindowDescription description_;
        std::uint32_t clientWidth_ = 0;
        std::uint32_t clientHeight_ = 0;
        bool minimized_ = false;
        const wchar_t* className_ = L"SemanticGpuEngineWindow";
    };
}
