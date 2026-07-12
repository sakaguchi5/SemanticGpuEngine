#include "12_CubeLab/CubeExperiment.h"

namespace sge::cube_lab
{
    ComparisonExperiment::ComparisonExperiment()
        : pgaBox_(experimental::PgaBox::AxisAligned(1.0f)),
          sdfBox_{1.0f, 1.0f, 1.0f}
    {
    }

    ir::SemanticModule ComparisonExperiment::Build(
        double elapsedSeconds,
        float aspectRatio,
        ExperimentMode mode) const
    {
        if (mode == ExperimentMode::Sdf)
        {
            experimental::SdfScene scene;
            scene.elapsedSeconds = elapsedSeconds;
            scene.aspectRatio = aspectRatio;
            scene.box = sdfBox_;
            return sdfLowering_.Lower(scene);
        }

        classical::ClassicalScene scene;
        scene.elapsedSeconds = elapsedSeconds;
        scene.aspectRatio = aspectRatio;
        scene.geometry.cubeHalfExtent = pgaBox_.HalfExtent();
        return classicalLowering_.Lower(scene);
    }

    const experimental::PgaBox& ComparisonExperiment::PgaModel() const noexcept
    {
        return pgaBox_;
    }

    const experimental::SdfBox& ComparisonExperiment::SdfModel() const noexcept
    {
        return sdfBox_;
    }
}
