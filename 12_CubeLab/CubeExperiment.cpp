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
        ExperimentScene scene;
        scene.elapsedSeconds = elapsedSeconds;
        scene.aspectRatio = aspectRatio;
        scene.boxHalfExtent = pgaBox_.HalfExtent();
        return Build(scene, mode);
    }

    ir::SemanticModule ComparisonExperiment::Build(
        const ExperimentScene& scene,
        ExperimentMode mode) const
    {
        if (mode == ExperimentMode::Sdf)
        {
            experimental::SdfScene lowered;
            lowered.elapsedSeconds = scene.elapsedSeconds;
            lowered.aspectRatio = scene.aspectRatio;
            lowered.box = {
                scene.boxHalfExtent,
                scene.boxHalfExtent,
                scene.boxHalfExtent};
            lowered.cameraPosition = scene.cameraPosition;
            lowered.cameraTarget = scene.cameraTarget;
            lowered.cameraUp = scene.cameraUp;
            lowered.verticalFieldOfViewDegrees =
                scene.verticalFieldOfViewDegrees;
            lowered.maximumDistance = scene.maximumDistance;
            lowered.surfaceTolerance = scene.surfaceTolerance;
            lowered.maximumSteps = scene.maximumSteps;
            return sdfLowering_.Lower(lowered);
        }

        classical::ClassicalScene lowered;
        lowered.elapsedSeconds = scene.elapsedSeconds;
        lowered.aspectRatio = scene.aspectRatio;
        lowered.geometry.cubeHalfExtent = scene.boxHalfExtent;
        lowered.cameraPosition = scene.cameraPosition;
        lowered.cameraTarget = scene.cameraTarget;
        lowered.cameraUp = scene.cameraUp;
        lowered.verticalFieldOfViewDegrees =
            scene.verticalFieldOfViewDegrees;
        return classicalLowering_.Lower(lowered);
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
