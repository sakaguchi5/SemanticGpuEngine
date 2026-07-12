#pragma once

#include <array>
#include <cmath>

namespace sge::experimental
{
    struct Plane
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;
    };

    struct PgaBox
    {
        std::array<Plane, 6> boundaryPlanes{};

        [[nodiscard]] static PgaBox AxisAligned(float halfExtent) noexcept;
        [[nodiscard]] bool IsConsistent(float tolerance = 0.0001f) const noexcept;
        [[nodiscard]] float HalfExtent() const noexcept;
    };

    struct SdfBox
    {
        float halfExtentX = 1.0f;
        float halfExtentY = 1.0f;
        float halfExtentZ = 1.0f;

        [[nodiscard]] float Evaluate(
            float x,
            float y,
            float z) const noexcept;
    };
}
