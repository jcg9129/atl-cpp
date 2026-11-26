#pragma once

/** @file
 * @brief Exact floating half-plane coefficient arithmetic.
 */
#include <atl/geometry/halfplane_core.hpp>
#include <atl/geometry/predicates_floating.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>

namespace atl::geometry::detail {
	/**
	 * @brief EXACT sign of `r - p x - q y` for `double` values: a static
	 *        filter on the rounded value, then the exact five-component
	 *        expansion of the two error-free products.
	 * @pre No overflow or underflow in the products `p x`, `q y` (the
	 *      floating precondition of the predicates): the error-free
	 *      product tails are exact only then. All expansion intermediates and
	 *      the absolute-value sum `|r| + |p x| + |q y|` must be finite.
	 */
	[[nodiscard]] constexpr int residual_sign(float64 const p, float64 const q, float64 const r, float64 const x,
											  float64 const y) {
		float64 const px = rounded_binary64(p * x), qy = rounded_binary64(q * y);
		float64 const value = rounded_binary64(rounded_binary64(r - px) - qy);
		// The bound covers two products and two subtractions.
		float64 const magnitude = rounded_binary64(rounded_binary64((r < 0 ? -r : r) + (px < 0 ? -px : px)) + (qy < 0 ? -qy : qy));
		atl_assert(is_finite(magnitude) && is_finite(value));
		float64 const bound = (4.0 + 32.0 * epsilon) * epsilon * magnitude;
		if (value > bound || -value > bound) return sign_of(value);
		float64 px_err = 0, qy_err = 0;
		float64 const px_hi = two_product(-p, x, px_err);
		float64 const qy_hi = two_product(-q, y, qy_err);
		float64 const products[2][2] = {{px_err, px_hi}, {qy_err, qy_hi}};
		float64 sum[4] = {}, total[5] = {};
		int const n = expansion_sum(2, products[0], 2, products[1], sum);
		float64 const offset[1] = {r};
		int const m = expansion_sum(n, sum, 1, offset, total);
		atl_assert(is_finite(total[m - 1]));
		return sign_of(total[m - 1]);
	}

	template<std::floating_point T>
	struct halfplane_arithmetic<T> {
		static constexpr float64 product(float64 a, float64 b, float64 &tail) { return two_product(a, b, tail); }

		static constexpr int residual(float64 p, float64 q, float64 r, float64 x, float64 y) { return residual_sign(p, q, r, x, y); }
	};
}
