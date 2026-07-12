#include "01_Platform/Platform.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace sge::platform
{
    Win32Application::Win32Application(
        HINSTANCE instance,
        int showCommand,
        WindowDescription description)
        : instance_(instance),
          description_(std::move(description))
    {
        RegisterWindowClass();
        CreateMainWindow(showCommand);
    }

    Win32Application::~Win32Application()
    {
        if (window_ != nullptr)
        {
            DestroyWindow(window_);
            window_ = nullptr;
        }

        UnregisterClassW(className_, instance_);
    }

    void Win32Application::RegisterWindowClass()
    {
        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.style = CS_HREDRAW | CS_VREDRAW;
        windowClass.lpfnWndProc = &Win32Application::WindowProcedure;
        windowClass.hInstance = instance_;
        windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        windowClass.hbrBackground = nullptr;
        windowClass.lpszClassName = className_;

        if (RegisterClassExW(&windowClass) == 0)
        {
            throw std::runtime_error("RegisterClassExW failed.");
        }
    }

    void Win32Application::CreateMainWindow(int showCommand)
    {
        RECT area{
            0,
            0,
            static_cast<LONG>(description_.width),
            static_cast<LONG>(description_.height)
        };

        if (!AdjustWindowRect(&area, WS_OVERLAPPEDWINDOW, FALSE))
        {
            throw std::runtime_error("AdjustWindowRect failed.");
        }

        window_ = CreateWindowExW(
            0,
            className_,
            description_.title.c_str(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            area.right - area.left,
            area.bottom - area.top,
            nullptr,
            nullptr,
            instance_,
            this);

        if (window_ == nullptr)
        {
            throw std::runtime_error("CreateWindowExW failed.");
        }

        ShowWindow(window_, showCommand);
        UpdateWindow(window_);
        UpdateClientSize();
    }

    int Win32Application::Run(
        const std::function<void(const FrameTime&)>& frame)
    {
        using Clock = std::chrono::steady_clock;

        const auto start = Clock::now();
        auto previous = start;
        MSG message{};

        while (message.message != WM_QUIT)
        {
            if (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&message);
                DispatchMessageW(&message);
                continue;
            }

            if (minimized_ || clientWidth_ == 0 || clientHeight_ == 0)
            {
                WaitMessage();
                continue;
            }

            const auto now = Clock::now();
            const FrameTime time{
                .deltaSeconds =
                    std::chrono::duration<double>(now - previous).count(),
                .elapsedSeconds =
                    std::chrono::duration<double>(now - start).count()
            };

            previous = now;
            frame(time);
        }

        return static_cast<int>(message.wParam);
    }

    NativeSurface Win32Application::Surface() const noexcept
    {
        return NativeSurface{
            .handle = window_,
            .width = clientWidth_,
            .height = clientHeight_
        };
    }

    void Win32Application::UpdateClientSize() noexcept
    {
        RECT client{};
        if (GetClientRect(window_, &client))
        {
            clientWidth_ =
                static_cast<std::uint32_t>(client.right - client.left);
            clientHeight_ =
                static_cast<std::uint32_t>(client.bottom - client.top);
        }
    }

    LRESULT CALLBACK Win32Application::WindowProcedure(
        HWND window,
        UINT message,
        WPARAM wParam,
        LPARAM lParam)
    {
        Win32Application* application = nullptr;

        if (message == WM_NCCREATE)
        {
            const auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            application = static_cast<Win32Application*>(
                create->lpCreateParams);

            SetWindowLongPtrW(
                window,
                GWLP_USERDATA,
                reinterpret_cast<LONG_PTR>(application));

            application->window_ = window;
        }
        else
        {
            application = reinterpret_cast<Win32Application*>(
                GetWindowLongPtrW(window, GWLP_USERDATA));
        }

        if (application != nullptr)
        {
            return application->HandleMessage(
                window,
                message,
                wParam,
                lParam);
        }

        return DefWindowProcW(window, message, wParam, lParam);
    }

    LRESULT Win32Application::HandleMessage(
        HWND window,
        UINT message,
        WPARAM wParam,
        LPARAM lParam)
    {
        switch (message)
        {
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE)
            {
                DestroyWindow(window);
                return 0;
            }
            break;

        case WM_SIZE:
            clientWidth_ = static_cast<std::uint32_t>(LOWORD(lParam));
            clientHeight_ = static_cast<std::uint32_t>(HIWORD(lParam));
            minimized_ = (wParam == SIZE_MINIMIZED);
            return 0;

        case WM_DESTROY:
            window_ = nullptr;
            PostQuitMessage(0);
            return 0;

        default:
            break;
        }

        return DefWindowProcW(window, message, wParam, lParam);
    }
}
