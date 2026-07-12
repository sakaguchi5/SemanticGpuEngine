#pragma once

#include "03_RenderIR/RenderIR.h"
#include "08_ClassicalRasterFrontend/ClassicalRaster.h"
#include "09_ExperimentalGeometry/ExperimentalGeometry.h"

namespace sge::cube_lab
{
    class ComparisonExperiment
    {
    public:
        ComparisonExperiment();

        [[nodiscard]] ir::SemanticModule Build(
            double elapsedSeconds,
            float aspectRatio) const;

        [[nodiscard]] const experimental::PgaBox& PgaModel() const noexcept;
        [[nodiscard]] const experimental::SdfBox& SdfModel() const noexcept;

    private:
        experimental::PgaBox pgaBox_;
        experimental::SdfBox sdfBox_;
        classical::GeometryRanges ranges_{};
        std::vector<classical::Vertex> vertices_;
    };
}
