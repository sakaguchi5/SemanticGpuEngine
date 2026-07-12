#include "09_ExperimentalGeometry/ExperimentalGeometry.h"

#include <algorithm>

namespace sge::experimental
{
    PgaBox PgaBox::AxisAligned(float halfExtent) noexcept
    {
        return PgaBox{
            .boundaryPlanes = {
                Plane{ 1.0f,  0.0f,  0.0f, -halfExtent},
                Plane{-1.0f,  0.0f,  0.0f, -halfExtent},
                Plane{ 0.0f,  1.0f,  0.0f, -halfExtent},
                Plane{ 0.0f, -1.0f,  0.0f, -halfExtent},
                Plane{ 0.0f,  0.0f,  1.0f, -halfExtent},
                Plane{ 0.0f,  0.0f, -1.0f, -halfExtent}
            }
        };
    }

    bool PgaBox::IsConsistent(float tolerance) const noexcept
    {
        const float extent = HalfExtent();
        if (extent <= 0.0f)
        {
            return false;
        }

        for (const auto& plane : boundaryPlanes)
        {
            const float normalLength = std::sqrt(
                plane.x * plane.x
                + plane.y * plane.y
                + plane.z * plane.z);

            if (std::abs(normalLength - 1.0f) > tolerance)
            {
                return false;
            }

            if (std::abs((-plane.w) - extent) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    float PgaBox::HalfExtent() const noexcept
    {
        return -boundaryPlanes[0].w;
    }

    float SdfBox::Evaluate(float x, float y, float z) const noexcept
    {
        const float qx = std::abs(x) - halfExtentX;
        const float qy = std::abs(y) - halfExtentY;
        const float qz = std::abs(z) - halfExtentZ;

        const float outsideX = std::max(qx, 0.0f);
        const float outsideY = std::max(qy, 0.0f);
        const float outsideZ = std::max(qz, 0.0f);

        const float outside = std::sqrt(
            outsideX * outsideX
            + outsideY * outsideY
            + outsideZ * outsideZ);

        const float inside = std::min(
            std::max(qx, std::max(qy, qz)),
            0.0f);

        return outside + inside;
    }
}
