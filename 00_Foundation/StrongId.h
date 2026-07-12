#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <limits>

namespace sge::foundation
{
    template<class Tag>
    class StrongId
    {
    public:
        using value_type = std::uint32_t;
        static constexpr value_type invalid_value =
            std::numeric_limits<value_type>::max();

        constexpr StrongId() noexcept = default;
        explicit constexpr StrongId(value_type value) noexcept
            : value_(value)
        {
        }

        [[nodiscard]] constexpr value_type Value() const noexcept
        {
            return value_;
        }

        [[nodiscard]] constexpr bool IsValid() const noexcept
        {
            return value_ != invalid_value;
        }

        constexpr auto operator<=>(const StrongId&) const noexcept = default;

    private:
        value_type value_ = invalid_value;
    };

    template<class Tag>
    struct StrongIdHash
    {
        [[nodiscard]] std::size_t operator()(StrongId<Tag> id) const noexcept
        {
            return std::hash<typename StrongId<Tag>::value_type>{}(id.Value());
        }
    };
}
