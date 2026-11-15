#pragma once

#include <atl/types/concepts.hpp>
#include <climits>

namespace atl {
	/**
	 * @brief The number of bits in the integer type @p T (`sizeof(T) * CHAR_BIT`).
	 */
	template<atl::integral T>
	constexpr inline auto num_bits = sizeof(T) * CHAR_BIT;
}
