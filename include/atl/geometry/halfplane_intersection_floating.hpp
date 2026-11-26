#pragma once

/** @file
 * @brief Adaptive floating half-plane intersection determinants.
 */
#include <atl/geometry/halfplane_floating.hpp>
#include <atl/geometry/halfplane_intersection_core.hpp>
#include <atl/geometry/predicates_floating.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cmath>

namespace atl::geometry::detail {
	/**
	 * @brief EXACT sign of `a b - c d` for `double` values. Rounding is
	 *        monotone, so different rounded products already order the
	 *        exact ones; only equal ones need the error-free product tails,
	 *        whose difference then is the exact value.
	 * @pre No overflow or underflow in the products.
	 */
	[[nodiscard]] constexpr int product_difference_sign(float64 const a, float64 const b, float64 const c,
														float64 const d) {
		float64 const left = rounded_binary64(a * b), right = rounded_binary64(c * d);
		if (left != right) return left > right ? 1 : -1;
		float64 left_err = 0, right_err = 0;
		two_product(a, b, left_err);
		two_product(c, d, right_err);
		return int(left_err > right_err) - int(left_err < right_err);
	}

	/**
	 * @brief `a b - c d` for `double` values as an exact four-component
	 *        expansion `out[0..3]`, least significant first.
	 */
	constexpr void product_difference_expansion(float64 const a, float64 const b, float64 const c,
												float64 const d, float64 *out) {
		float64 left_err = 0, right_err = 0;
		float64 const left = two_product(a, b, left_err), right = two_product(c, d, right_err);
		two_two_diff(left, left_err, right, right_err, out[3], out[2], out[1], out[0]);
	}

	/** @brief `a b - c d` for `double` values, rounded from its exact expansion. */
	[[nodiscard]] constexpr float64 product_difference_value(float64 const a, float64 const b, float64 const c,
															 float64 const d) {
		float64 parts[4] = {};
		product_difference_expansion(a, b, c, d, parts);
		return ((parts[0] + parts[1]) + parts[2]) + parts[3];
	}

	/**
	 * @brief `a b - c d` for `double` values with relative error at most
	 *        `2 epsilon` OF THE RESULT, however much the products cancel:
	 *        Kahan's algorithm, where one fma recovers the rounding error of
	 *        `c d` exactly and a second forms `a b - fl(c d)` with one
	 *        rounding. Constant evaluation sums the exact expansion instead
	 *        (fma is not constant-evaluable). `std::fma` even where it is a
	 *        library call (x86-64 at plain `-O2`): a Dekker-split expansion
	 *        was raced there and lost (see the experimental
	 *        `dekker_determinant`).
	 * @pre No overflow or underflow in the products.
	 */
	[[nodiscard]] constexpr float64 accurate_product_difference(float64 const a, float64 const b, float64 const c,
																float64 const d) {
		if consteval {
			return product_difference_value(a, b, c, d);
		} else {
			float64 const rounded = rounded_binary64(c * d);
			float64 const rounding_error = std::fma(-c, d, rounded);
			return std::fma(a, b, -rounded) + rounding_error;
		}
	}

	/**
	 * @name Filter bound of the 3x3 coefficient determinant
	 * @brief With the minors from `accurate_product_difference`
	 *        (relative error `2 epsilon` each), the rounded `r_i d_jk +
	 *        r_j d_ki + r_k d_ij` is within `5 epsilon + O(epsilon^2)` of
	 *        `M = |r_i d_jk| + |r_j d_ki| + |r_k d_ij|`: `3 epsilon` per
	 *        term (minor and product), `2 epsilon` for the two additions.
	 *        `6 epsilon` also covers the rounding of `M` itself. Unlike a
	 *        bound on the products inside the minors, `M` shrinks with
	 *        cancelling minors, so nearly parallel constraints (tangents
	 *        of one circle, say) still certify here.
	 * @{
	 */
	inline constexpr float64 halfplane_determinant_bound = (6.0 + 64.0 * epsilon) * epsilon;

	/** @} */
	/**
	 * @brief EXACT sign of the determinant `r_i d_jk + r_j d_ki + r_k
	 *        d_ij` of three `double` constraints: the three minors as
	 *        exact expansions, scaled and summed (at most 24 components).
	 */
	[[nodiscard]] constexpr int halfplane_determinant_sign_exact(halfplane_constraint<float64> const &i,
																 halfplane_constraint<float64> const &j,
																 halfplane_constraint<float64> const &k) {
		float64 minor_jk[4] = {}, minor_ki[4] = {}, minor_ij[4] = {};
		product_difference_expansion(j.p, k.q, k.p, j.q, minor_jk);
		product_difference_expansion(k.p, i.q, i.p, k.q, minor_ki);
		product_difference_expansion(i.p, j.q, j.p, i.q, minor_ij);
		float64 term_i[8] = {}, term_j[8] = {}, term_k[8] = {}, partial[16] = {}, total[24] = {};
		int const ni = scale_expansion(4, minor_jk, i.r, term_i);
		int const nj = scale_expansion(4, minor_ki, j.r, term_j);
		int const nk = scale_expansion(4, minor_ij, k.r, term_k);
		int const np = expansion_sum(ni, term_i, nj, term_j, partial);
		int const nt = expansion_sum(np, partial, nk, term_k, total);
		return sign_of(total[nt - 1]);
	}

	template<>
	struct halfplane_intersection_arithmetic<float64> {
		static constexpr int product_difference_sign(float64 a, float64 b, float64 c, float64 d) { return detail::product_difference_sign(a, b, c, d); }

		static constexpr float64 difference(float64 a, float64 b, float64 c, float64 d) { return accurate_product_difference(a, b, c, d); }

		static constexpr int perturbation_sign(halfplane_constraint<float64> const &i, halfplane_constraint<float64> const &j, halfplane_constraint<float64> const &k) {
			float64 minor_jk[4] = {}, minor_ki[4] = {}, minor_ij[4] = {};
			product_difference_expansion(j.p, k.q, k.p, j.q, minor_jk);
			product_difference_expansion(k.p, i.q, i.p, k.q, minor_ki);
			product_difference_expansion(i.p, j.q, j.p, i.q, minor_ij);
			float64 partial[8] = {}, total[12] = {};
			int const np = expansion_sum(4, minor_jk, 4, minor_ki, partial);
			int const nt = expansion_sum(np, partial, 4, minor_ij, total);
			return sign_of(total[nt - 1]);
		}

		static constexpr int halfplane_determinant_sign(halfplane_constraint<float64> const &i, halfplane_constraint<float64> const &j, halfplane_constraint<float64> const &k) {
			float64 const minor_jk = accurate_product_difference(j.p, k.q, k.p, j.q);
			float64 const minor_ki = accurate_product_difference(k.p, i.q, i.p, k.q);
			float64 const minor_ij = accurate_product_difference(i.p, j.q, j.p, i.q);
			float64 const term_i = i.r * minor_jk, term_j = j.r * minor_ki, term_k = k.r * minor_ij;
			float64 const value = term_i + term_j + term_k;
			float64 const permanent = magnitude_of(term_i) + magnitude_of(term_j) + magnitude_of(term_k);
			float64 const bound = halfplane_determinant_bound * permanent;
			if (value > bound || -value > bound) return sign_of(value);
			return halfplane_determinant_sign_exact(i, j, k);
		}
	};
}
