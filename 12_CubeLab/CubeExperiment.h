#pragma once

#include "03_RenderIR/RenderIR.h"
#include "08_ClassicalRasterFrontend/ClassicalLowering.h"
#include "09_ExperimentalGeometry/ExperimentalGeometry.h"
#include "09_ExperimentalGeometry/SdfLowering.h"

namespace sge::cube_lab
{
    enum class ExperimentMode
    {
        Classical,
        Sdf
    };

    // A representation-neutral experiment scene. Every geometry frontend must
    // lower the same camera, object extent, time, and quality target.
    struct ExperimentScene
    {
        double elapsedSeconds = 0.0;
        float aspectRatio = 16.0f / 9.0f;
        float boxHalfExtent = 1.0f;
        classical::Float3 cameraPosition{5.0f, 3.5f, -6.0f};
        classical::Float3 cameraTarget{0.0f, 0.0f, 0.0f};
        classical::Float3 cameraUp{0.0f, 1.0f, 0.0f};
        float verticalFieldOfViewDegrees = 59.0f;
        float maximumDistance = 50.0f;
        float surfaceTolerance = 0.001f;
        std::uint32_t maximumSteps = 128;
    };

    class ComparisonExperiment
    {
    public:
        ComparisonExperiment();

        [[nodiscard]] ir::SemanticModule Build(
            double elapsedSeconds,
            float aspectRatio,
            ExperimentMode mode = ExperimentMode::Classical) const;

        [[nodiscard]] ir::SemanticModule Build(
            const ExperimentScene& scene,
            ExperimentMode mode) const;

        [[nodiscard]] const experimental::PgaBox& PgaModel() const noexcept;
        [[nodiscard]] const experimental::SdfBox& SdfModel() const noexcept;

    private:
        experimental::PgaBox pgaBox_;
        experimental::SdfBox sdfBox_;
        classical::ClassicalRasterLowering classicalLowering_;
        experimental::SdfRayMarchLowering sdfLowering_;
    };
}
