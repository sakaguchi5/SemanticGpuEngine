#pragma once

#include "08_ClassicalRasterFrontend/ClassicalMath.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sge::classical
{
    struct Vertex
    {
        Float3 position;
        Float4 color;
    };

    static_assert(sizeof(Vertex) == 28);

    struct GeometryRanges
    {
        std::uint32_t cubeFirst = 0;
        std::uint32_t cubeCount = 0;
        std::uint32_t planesFirst = 0;
        std::uint32_t planesCount = 0;
        std::uint32_t linesFirst = 0;
        std::uint32_t linesCount = 0;
    };

    struct SceneConstants
    {
        Matrix4 worldViewProjection;
    };

    struct ClassicalCubeDescription
    {
        float cubeHalfExtent = 1.0f;
        float planeExtent = 3.5f;
        float gridSpacing = 0.5f;
    };

    [[nodiscard]] std::vector<Vertex> BuildClassicalCubeGeometry(
        const ClassicalCubeDescription& description,
        GeometryRanges& ranges);

    template<class T>
    [[nodiscard]] std::vector<std::byte> ToBytes(const T& value)
    {
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        return std::vector<std::byte>(begin, begin + sizeof(T));
    }

    template<class T>
    [[nodiscard]] std::vector<std::byte> ToBytes(
        const std::vector<T>& values)
    {
        const auto* begin =
            reinterpret_cast<const std::byte*>(values.data());

        return std::vector<std::byte>(
            begin,
            begin + values.size() * sizeof(T));
    }
}
