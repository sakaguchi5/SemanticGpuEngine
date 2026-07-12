#pragma once

#include "03_RenderIR/RenderIR.h"
#include "08_ClassicalRasterFrontend/ClassicalMath.h"
#include "09_ExperimentalGeometry/ExperimentalGeometry.h"

namespace sge::experimental
{
    struct SdfScene
    {
        double elapsedSeconds = 0.0;
        float aspectRatio = 16.0f / 9.0f;
        SdfBox box{};
        classical::Float3 cameraPosition{4.5f, 3.2f, -6.5f};
        classical::Float3 cameraTarget{0.0f, 0.0f, 0.0f};
        classical::Float3 cameraUp{0.0f, 1.0f, 0.0f};
        float verticalFieldOfViewDegrees = 60.0f;
        float maximumDistance = 50.0f;
        float surfaceTolerance = 0.001f;
        std::uint32_t maximumSteps = 128;
    };

    class SdfRayMarchLowering
    {
    public:
        [[nodiscard]] ir::SemanticModule Lower(
            const SdfScene& scene) const;
    };
}
