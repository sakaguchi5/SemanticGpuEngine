#pragma once

#include <stdexcept>
#include <string_view>

namespace sge::foundation
{
    inline void Require(bool condition, std::string_view message)
    {
        if (!condition)
        {
            throw std::logic_error(std::string(message));
        }
    }
}
