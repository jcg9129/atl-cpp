#pragma once

#include <atl/types/concepts.hpp>

namespace atl {
	/**
	 * @brief Return a word with only the lowest one bit of x set (0 for x == 0).
	 * @tparam T An unsigned integer recognized by `atl::unsigned_integral`.
	 * @param x The word to inspect; every value is valid.
	 * @return The least significant set bit, or zero when @p x is zero.
	 * @complexity O(1) time and space.
	 */
	template<unsigned_integral T>
	constexpr T lowest_one(T x) noexcept {
		return T(x & (~x + 1));
	}
}
