#include "08_ClassicalRasterFrontend/ClassicalMath.h"

namespace sge::classical
{
    Matrix4 Matrix4::Identity() noexcept
    {
        Matrix4 result{};
        result.m[0][0] = 1.0f;
        result.m[1][1] = 1.0f;
        result.m[2][2] = 1.0f;
        result.m[3][3] = 1.0f;
        return result;
    }

    Matrix4 Matrix4::RotationX(float radians) noexcept
    {
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);

        Matrix4 result = Identity();
        result.m[1][1] = cosine;
        result.m[1][2] = sine;
        result.m[2][1] = -sine;
        result.m[2][2] = cosine;
        return result;
    }

    Matrix4 Matrix4::RotationY(float radians) noexcept
    {
        const float cosine = std::cos(radians);
        const float sine = std::sin(radians);

        Matrix4 result = Identity();
        result.m[0][0] = cosine;
        result.m[0][2] = -sine;
        result.m[2][0] = sine;
        result.m[2][2] = cosine;
        return result;
    }

    Matrix4 Matrix4::LookAtLH(
        Float3 eye,
        Float3 target,
        Float3 up) noexcept
    {
        const Float3 forward = Normalize(target - eye);
        const Float3 right = Normalize(Cross(up, forward));
        const Float3 correctedUp = Cross(forward, right);

        Matrix4 result{};
        result.m[0][0] = right.x;
        result.m[0][1] = correctedUp.x;
        result.m[0][2] = forward.x;

        result.m[1][0] = right.y;
        result.m[1][1] = correctedUp.y;
        result.m[1][2] = forward.y;

        result.m[2][0] = right.z;
        result.m[2][1] = correctedUp.z;
        result.m[2][2] = forward.z;

        result.m[3][0] = -Dot(right, eye);
        result.m[3][1] = -Dot(correctedUp, eye);
        result.m[3][2] = -Dot(forward, eye);
        result.m[3][3] = 1.0f;
        return result;
    }

    Matrix4 Matrix4::PerspectiveFovLH(
        float verticalFieldOfViewRadians,
        float aspectRatio,
        float nearPlane,
        float farPlane) noexcept
    {
        const float yScale = 1.0f
            / std::tan(verticalFieldOfViewRadians * 0.5f);
        const float xScale = yScale / aspectRatio;
        const float depthScale = farPlane / (farPlane - nearPlane);

        Matrix4 result{};
        result.m[0][0] = xScale;
        result.m[1][1] = yScale;
        result.m[2][2] = depthScale;
        result.m[2][3] = 1.0f;
        result.m[3][2] = -nearPlane * depthScale;
        return result;
    }

    Matrix4 operator*(
        const Matrix4& left,
        const Matrix4& right) noexcept
    {
        Matrix4 result{};

        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t column = 0; column < 4; ++column)
            {
                for (std::size_t inner = 0; inner < 4; ++inner)
                {
                    result.m[row][column] +=
                        left.m[row][inner] * right.m[inner][column];
                }
            }
        }

        return result;
    }

    Float3 operator-(Float3 left, Float3 right) noexcept
    {
        return {
            left.x - right.x,
            left.y - right.y,
            left.z - right.z
        };
    }

    Float3 Cross(Float3 left, Float3 right) noexcept
    {
        return {
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x
        };
    }

    float Dot(Float3 left, Float3 right) noexcept
    {
        return left.x * right.x
            + left.y * right.y
            + left.z * right.z;
    }

    Float3 Normalize(Float3 value) noexcept
    {
        const float lengthSquared = Dot(value, value);
        if (lengthSquared <= 0.0f)
        {
            return {};
        }

        const float inverseLength = 1.0f / std::sqrt(lengthSquared);
        return {
            value.x * inverseLength,
            value.y * inverseLength,
            value.z * inverseLength
        };
    }
}
