#include "08_ClassicalRasterFrontend/ClassicalRaster.h"

#include <cmath>

namespace sge::classical
{
    std::vector<Vertex> BuildClassicalCubeGeometry(
        const ClassicalCubeDescription& description,
        GeometryRanges& ranges)
    {
        std::vector<Vertex> vertices;

        const auto addTriangle = [&](Vertex first,
                                     Vertex second,
                                     Vertex third)
        {
            vertices.push_back(first);
            vertices.push_back(second);
            vertices.push_back(third);
        };

        const auto addQuad = [&](Float3 a,
                                 Float3 b,
                                 Float3 c,
                                 Float3 d,
                                 Float4 color)
        {
            addTriangle({a, color}, {b, color}, {c, color});
            addTriangle({a, color}, {c, color}, {d, color});
        };

        const auto addLine = [&](Float3 a, Float3 b, Float4 color)
        {
            vertices.push_back({a, color});
            vertices.push_back({b, color});
        };

        const float size = description.cubeHalfExtent;
        ranges.cubeFirst = static_cast<std::uint32_t>(vertices.size());

        addQuad(
            {-size, -size, size},
            { size, -size, size},
            { size,  size, size},
            {-size,  size, size},
            {0.95f, 0.20f, 0.20f, 1.0f});

        addQuad(
            { size, -size, -size},
            {-size, -size, -size},
            {-size,  size, -size},
            { size,  size, -size},
            {0.55f, 0.12f, 0.12f, 1.0f});

        addQuad(
            {size, -size,  size},
            {size, -size, -size},
            {size,  size, -size},
            {size,  size,  size},
            {0.20f, 0.90f, 0.30f, 1.0f});

        addQuad(
            {-size, -size, -size},
            {-size, -size,  size},
            {-size,  size,  size},
            {-size,  size, -size},
            {0.10f, 0.48f, 0.18f, 1.0f});

        addQuad(
            {-size, size,  size},
            { size, size,  size},
            { size, size, -size},
            {-size, size, -size},
            {0.20f, 0.35f, 1.00f, 1.0f});

        addQuad(
            {-size, -size, -size},
            { size, -size, -size},
            { size, -size,  size},
            {-size, -size,  size},
            {0.10f, 0.20f, 0.62f, 1.0f});

        ranges.cubeCount =
            static_cast<std::uint32_t>(vertices.size()) - ranges.cubeFirst;

        const float plane = description.planeExtent;
        ranges.planesFirst = static_cast<std::uint32_t>(vertices.size());

        addQuad(
            {-plane, -plane, 0.0f},
            { plane, -plane, 0.0f},
            { plane,  plane, 0.0f},
            {-plane,  plane, 0.0f},
            {0.20f, 0.35f, 1.00f, 0.16f});

        addQuad(
            {0.0f, -plane, -plane},
            {0.0f, -plane,  plane},
            {0.0f,  plane,  plane},
            {0.0f,  plane, -plane},
            {1.00f, 0.22f, 0.22f, 0.16f});

        addQuad(
            {-plane, 0.0f, -plane},
            { plane, 0.0f, -plane},
            { plane, 0.0f,  plane},
            {-plane, 0.0f,  plane},
            {0.20f, 1.00f, 0.32f, 0.16f});

        ranges.planesCount =
            static_cast<std::uint32_t>(vertices.size()) - ranges.planesFirst;

        ranges.linesFirst = static_cast<std::uint32_t>(vertices.size());

        const Float4 xyColor{0.30f, 0.45f, 1.00f, 0.55f};
        const Float4 yzColor{1.00f, 0.32f, 0.32f, 0.55f};
        const Float4 zxColor{0.30f, 1.00f, 0.42f, 0.55f};

        const int steps = static_cast<int>(
            std::round((2.0f * plane) / description.gridSpacing));

        for (int index = 0; index <= steps; ++index)
        {
            const float value =
                -plane + static_cast<float>(index) * description.gridSpacing;

            addLine({-plane, value, 0.0f}, {plane, value, 0.0f}, xyColor);
            addLine({value, -plane, 0.0f}, {value, plane, 0.0f}, xyColor);

            addLine({0.0f, -plane, value}, {0.0f, plane, value}, yzColor);
            addLine({0.0f, value, -plane}, {0.0f, value, plane}, yzColor);

            addLine({-plane, 0.0f, value}, {plane, 0.0f, value}, zxColor);
            addLine({value, 0.0f, -plane}, {value, 0.0f, plane}, zxColor);
        }

        addLine(
            {-plane, 0.0f, 0.0f},
            { plane, 0.0f, 0.0f},
            {1.0f, 0.05f, 0.05f, 1.0f});

        addLine(
            {0.0f, -plane, 0.0f},
            {0.0f,  plane, 0.0f},
            {0.05f, 1.0f, 0.05f, 1.0f});

        addLine(
            {0.0f, 0.0f, -plane},
            {0.0f, 0.0f,  plane},
            {0.05f, 0.35f, 1.0f, 1.0f});

        ranges.linesCount =
            static_cast<std::uint32_t>(vertices.size()) - ranges.linesFirst;

        return vertices;
    }
}
