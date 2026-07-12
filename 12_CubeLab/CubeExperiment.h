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

    class ComparisonExperiment
    {
    public:
        ComparisonExperiment();

        [[nodiscard]] ir::SemanticModule Build(
            double elapsedSeconds,
            float aspectRatio,
            ExperimentMode mode = ExperimentMode::Classical) const;

        [[nodiscard]] const experimental::PgaBox& PgaModel() const noexcept;
        [[nodiscard]] const experimental::SdfBox& SdfModel() const noexcept;

    private:
        experimental::PgaBox pgaBox_;
        experimental::SdfBox sdfBox_;
        classical::ClassicalRasterLowering classicalLowering_;
        experimental::SdfRayMarchLowering sdfLowering_;
    };
}
