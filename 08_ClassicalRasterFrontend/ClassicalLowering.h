#pragma once

#include "03_RenderIR/RenderIR.h"
#include "08_ClassicalRasterFrontend/ClassicalRaster.h"

namespace sge::classical
{
    struct ClassicalScene
    {
        double elapsedSeconds = 0.0;
        float aspectRatio = 1.0f;
        ClassicalCubeDescription geometry{};
        Float3 cameraPosition{5.2f, 4.0f, -6.2f};
        Float3 cameraTarget{0.0f, 0.0f, 0.0f};
        Float3 cameraUp{0.0f, 1.0f, 0.0f};
        float verticalFieldOfViewDegrees = 58.0f;
    };

    class ClassicalRasterLowering
    {
    public:
        [[nodiscard]] ir::SemanticModule Lower(
            const ClassicalScene& scene) const;
    };
}
