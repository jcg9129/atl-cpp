#pragma once
#include <atl/types/alias.hpp>
#include <atl/types/detail/int256.hpp>
#include <cstddef>

namespace atl::geometry::detail {
	/**
	 * @brief Exact comparison of signed 128x128 products using 256-bit unsigned
	 * magnitudes. Reuses the portable limb multiplier (including software
	 * int128); no signed product or difference is materialized in int128.
	 */
	[[nodiscard]] constexpr int compare_wide_products(int128 const a, int128 const b, int128 const c, int128 const d) {
		auto left = atl::detail::int256::product(a, b);
		auto right = atl::detail::int256::product(c, d);
		int const left_sign = left.sign(), right_sign = right.sign();
		if (left_sign != right_sign) return left_sign < right_sign ? -1 : 1;
		if (left_sign == 0) return 0;
		if (left_sign < 0) {
			left = -left;
			right = -right;
		}
		for (std::size_t i = 4; i != 0; --i)
			if (left.limb[i - 1] != right.limb[i - 1])
				return (left.limb[i - 1] < right.limb[i - 1] ? -1 : 1) * left_sign;
		return 0;
	}

	/**
	 * @brief Exact sign of `a*b + c*d + e*f`, including software int128.
	 * @pre Intermediate and final sums fit signed int256.
	 */
	[[nodiscard]] constexpr int sum_three_wide_products_sign(int128 const a, int128 const b,
															 int128 const c, int128 const d, int128 const e, int128 const f) {
		return atl::detail::sum_three_products_sign(a, b, c, d, e, f);
	}

}
