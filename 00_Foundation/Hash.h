#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

namespace sge::foundation
{
    inline void HashCombine(std::size_t& seed, std::size_t value) noexcept
    {
        seed ^= value
            + static_cast<std::size_t>(0x9e3779b97f4a7c15ull)
            + (seed << 6u)
            + (seed >> 2u);
    }

    template<class T>
    inline void HashValue(std::size_t& seed, const T& value)
    {
        HashCombine(seed, std::hash<T>{}(value));
    }

    template<class Enum>
        requires std::is_enum_v<Enum>
    inline void HashEnum(std::size_t& seed, Enum value)
    {
        using Underlying = std::underlying_type_t<Enum>;
        HashCombine(
            seed,
            std::hash<Underlying>{}(static_cast<Underlying>(value)));
    }

    inline std::size_t HashBytes(const void* data, std::size_t size) noexcept
    {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        std::size_t hash = static_cast<std::size_t>(1469598103934665603ull);

        for (std::size_t index = 0; index < size; ++index)
        {
            hash ^= bytes[index];
            hash *= static_cast<std::size_t>(1099511628211ull);
        }

        return hash;
    }

    inline std::size_t HashString(std::string_view value) noexcept
    {
        return HashBytes(value.data(), value.size());
    }
}
