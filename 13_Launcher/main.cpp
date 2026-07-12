#include "01_Platform/Platform.h"
#include "05_RenderRuntime/RenderRuntime.h"
#include "07_D3D12Backend/D3D12Backend.h"
#include "12_CubeLab/CubeExperiment.h"

#include <exception>
#include <utility>
#include <windows.h>

int WINAPI wWinMain(
    HINSTANCE instance,
    HINSTANCE,
    PWSTR,
    int showCommand)
{
    try
    {
        sge::platform::Win32Application application{
            instance,
            showCommand,
            {
                .title = L"Semantic GPU Engine - D3D12",
                .width = 1280,
                .height = 720
            }
        };

        auto backend = sge::d3d12::CreateBackend(
            application.Surface());

        sge::runtime::RenderRuntime runtime{
            std::move(backend),
            {
                .enableValidation = true,
                .enablePlanCache = true,
                .planDiagnosticsPath = "execution_plan.txt",
                .graphDiagnosticsPath = "work_graph.dot"
            }
        };

        sge::cube_lab::ComparisonExperiment experiment;

        return application.Run(
            [&](const sge::platform::FrameTime& time)
            {
                const auto surface = application.Surface();
                if (surface.width == 0 || surface.height == 0)
                {
                    return;
                }

                const float aspectRatio =
                    static_cast<float>(surface.width)
                    / static_cast<float>(surface.height);

                runtime.Execute(
                    experiment.Build(
                        time.elapsedSeconds,
                        aspectRatio));
            });
    }
    catch (const std::exception& error)
    {
        MessageBoxA(
            nullptr,
            error.what(),
            "Semantic GPU Engine Error",
            MB_OK | MB_ICONERROR);

        return -1;
    }
}
