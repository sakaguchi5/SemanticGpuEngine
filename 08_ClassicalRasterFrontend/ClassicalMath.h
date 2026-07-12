#pragma once

#include <cmath>
#include <cstddef>

namespace sge::classical
{
    struct Float3
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct Float4
    {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 0.0f;
    };

    struct Matrix4
    {
        float m[4][4]{};

        [[nodiscard]] static Matrix4 Identity() noexcept;
        [[nodiscard]] static Matrix4 RotationX(float radians) noexcept;
        [[nodiscard]] static Matrix4 RotationY(float radians) noexcept;
        [[nodiscard]] static Matrix4 LookAtLH(
            Float3 eye,
            Float3 target,
            Float3 up) noexcept;
        [[nodiscard]] static Matrix4 PerspectiveFovLH(
            float verticalFieldOfViewRadians,
            float aspectRatio,
            float nearPlane,
            float farPlane) noexcept;
    };

    [[nodiscard]] Matrix4 operator*(
        const Matrix4& left,
        const Matrix4& right) noexcept;

    [[nodiscard]] Float3 operator-(Float3 left, Float3 right) noexcept;
    [[nodiscard]] Float3 Cross(Float3 left, Float3 right) noexcept;
    [[nodiscard]] float Dot(Float3 left, Float3 right) noexcept;
    [[nodiscard]] Float3 Normalize(Float3 value) noexcept;
}
