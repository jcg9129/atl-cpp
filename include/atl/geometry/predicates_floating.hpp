#pragma once

/** @file
 * @brief Adaptive exact floating predicates, installed before floating policy instantiation.
 */

#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cfloat>
#include <cmath>
#include <concepts>
#include <limits>

namespace atl::geometry::detail {
	/**
	 * @name Error-free transformations and expansion arithmetic
	 * @brief Shewchuk, "Adaptive Precision Floating-Point Arithmetic and
	 *        Fast Robust Geometric Predicates", 1997. Each two_* helper
	 *        returns the correctly rounded result and stores the EXACT
	 *        rounding error, so (result, err) sums to the true real
	 *        value; an "expansion" is a sum of such non-overlapping
	 *        components, least significant first.
	 * @pre Round-to-nearest and no overflow/underflow in intermediates —
	 *      coordinates of "reasonable" magnitude (|c| = 0 or roughly
	 *      1e-146 < |c| < 1e153), a documented precondition of the
	 *      floating predicates.
	 * @{
	 */

	/** @brief Force binary64 rounding on targets with excess evaluation precision. */
	constexpr float64 rounded_binary64(float64 const value) {
#if FLT_EVAL_METHOD != 0
		if consteval {
			return value;
		} else {
			float64 const volatile stored = value;
			return stored;
		}
#else
		return value;
#endif
	}

	/** @brief Knuth two-sum: exact for any argument magnitudes. */
	constexpr float64 two_sum(float64 const a, float64 const b, float64 &err) {
		float64 const s = rounded_binary64(a + b);
		float64 const bv = rounded_binary64(s - a);
		float64 const av = rounded_binary64(s - bv);
		err = rounded_binary64(rounded_binary64(a - av) + rounded_binary64(b - bv));
		return s;
	}

	/** @brief Exact error of the already computed difference `s = fl(a - b)`. */
	constexpr float64 two_diff_tail(float64 const a, float64 const b, float64 const s) {
		float64 const bv = rounded_binary64(a - s);
		float64 const av = rounded_binary64(s + bv);
		return rounded_binary64(rounded_binary64(a - av) + rounded_binary64(bv - b));
	}

	/** @brief Exact difference: returns `fl(a - b)` and stores its rounding error in @p err. */
	constexpr float64 two_diff(float64 const a, float64 const b, float64 &err) {
		float64 const s = rounded_binary64(a - b);
		err = two_diff_tail(a, b, s);
		return s;
	}

	/**
	 * @brief Exact product: result + err == a * b. Runtime uses one fma;
	 * constant evaluation uses the Veltkamp/Dekker split (fma is not
	 * constant-evaluable).
	 */
	constexpr float64 two_product(float64 const a, float64 const b, float64 &err) {
		float64 const p = rounded_binary64(a * b);
		if consteval {
			constexpr float64 splitter = 0x1p27 + 1.0;
			auto const split = [](float64 const v, float64 &lo) {
				float64 const c = splitter * v;
				float64 const hi = c - (c - v);
				lo = v - hi;
				return hi;
			};
			float64 alo = 0, blo = 0;
			float64 const ahi = split(a, alo);
			float64 const bhi = split(b, blo);
			err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo;
		} else {
			err = rounded_binary64(std::fma(a, b, -p));
		}
		return p;
	}

	/** @brief Two-component expansion minus a double -> three components. */
	constexpr void two_one_diff(float64 const a1, float64 const a0, float64 const b, float64 &x2,
								float64 &x1, float64 &x0) {
		float64 mid = 0;
		mid = two_diff(a0, b, x0);
		x2 = two_sum(a1, mid, x1);
	}

	/** @brief (a1, a0) - (b1, b0) -> four-component expansion x3..x0. */
	constexpr void two_two_diff(float64 const a1, float64 const a0, float64 const b1,
								float64 const b0, float64 &x3, float64 &x2, float64 &x1,
								float64 &x0) {
		float64 high = 0, mid = 0;
		two_one_diff(a1, a0, b0, high, mid, x0);
		two_one_diff(high, mid, b1, x3, x2, x1);
	}

	/**
	 * @brief Sum of two expansions (each non-overlapping, components in
	 * increasing magnitude), zero components dropped. Returns the
	 * length written to @p h; the sign of the true value is the sign of
	 * the LAST component (0 only if the value is exactly zero).
	 */
	constexpr int expansion_sum(int const elen, float64 const *e, int const flen,
								float64 const *f, float64 *h) {
		int ei = 0, fi = 0, hn = 0;
		// merge by ascending magnitude; the double comparison is
		// Shewchuk's branch-cheap |e[ei]| <= |f[fi]| test
		auto const take_e = [&] {
			return ei < elen && (fi >= flen || ((f[fi] > e[ei]) == (f[fi] > -e[ei])));
		};
		float64 q = take_e() ? e[ei++] : f[fi++];
		while (ei < elen || fi < flen) {
			float64 const next = take_e() ? e[ei++] : f[fi++];
			float64 err = 0;
			q = two_sum(q, next, err);
			if (err != 0) h[hn++] = err;
		}
		if (q != 0 || hn == 0) h[hn++] = q;
		return hn;
	}

	/**
	 * @brief Exact sum when `|a| >= |b|` — one operation cheaper than
	 * `two_sum` (Dekker).
	 */
	constexpr float64 fast_two_sum(float64 const a, float64 const b, float64 &err) {
		float64 const s = rounded_binary64(a + b);
		err = rounded_binary64(b - rounded_binary64(s - a));
		return s;
	}

	/**
	 * @brief The exact product of an expansion and one double: `h = e * b`,
	 * zero components dropped (Shewchuk's scale_expansion_zeroelim).
	 * @p e non-overlapping, increasing magnitude; @p h likewise.
	 */
	constexpr int scale_expansion(int const elen, float64 const *e, float64 const b,
								  float64 *h) {
		int hn = 0;
		float64 low = 0;
		float64 q = two_product(e[0], b, low);
		if (low != 0) h[hn++] = low;
		for (int i = 1; i < elen; ++i) {
			float64 p0 = 0, sum_err = 0, carry_err = 0;
			float64 const p1 = two_product(e[i], b, p0);
			float64 const sum = two_sum(q, p0, sum_err);
			if (sum_err != 0) h[hn++] = sum_err;
			q = fast_two_sum(p1, sum, carry_err);
			if (carry_err != 0) h[hn++] = carry_err;
		}
		if (q != 0 || hn == 0) h[hn++] = q;
		return hn;
	}

	/**
	 * @brief The exact product of two expansions: repeated scale + merge.
	 * @p h must hold `2 * elen * flen` components; @p scratch the same;
	 * @p scaled must hold `2 * elen`.
	 */
	constexpr int expansion_product(int const elen, float64 const *e, int const flen,
									float64 const *f, float64 *h, float64 *scratch,
									float64 *scaled) {
		int hn = 0;
		for (int j = 0; j < flen; ++j) {
			int const slen = scale_expansion(elen, e, f[j], scaled);
			int const merged = expansion_sum(hn, h, slen, scaled, scratch);
			for (int k = 0; k < merged; ++k) h[k] = scratch[k];
			hn = merged;
		}
		return hn;
	}

	/** @} */

	/**
	 * @name Static filter bounds
	 * @brief In units of the magnitude sum of the two determinant terms;
	 *        `epsilon` is half an ulp of 1 (2^-53).
	 * @{
	 */
	inline constexpr float64 epsilon = 0x1p-53;
	inline constexpr float64 orient_bound_a = (3.0 + 16.0 * epsilon) * epsilon;
	inline constexpr float64 orient_bound_b = (2.0 + 12.0 * epsilon) * epsilon;
	inline constexpr float64 orient_bound_c = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
	inline constexpr float64 result_bound = (3.0 + 8.0 * epsilon) * epsilon;

	/** @} */

	/**
	 * @brief Adaptive stages of `adaptive_cross_sign`, entered only when the static
	 * filter could not certify the sign. Successively sharper
	 * approximations, each with its own certified error bound; the
	 * final stage is the exact 16-component expansion of the
	 * determinant, whose leading component carries the true sign. The
	 * four points are INDEPENDENT: Shewchuk's analysis depends only on
	 * the (x1-x2)(y3-y4) - (y1-y2)(x3-x4) shape, not on shared origins.
	 */
	constexpr int adaptive_cross_tail(point<float64> const a, point<float64> const b,
									  point<float64> const c, point<float64> const d,
									  float64 const magnitude_sum) {
		float64 const acx = rounded_binary64(b.x - a.x), bcx = rounded_binary64(d.x - c.x);
		float64 const acy = rounded_binary64(b.y - a.y), bcy = rounded_binary64(d.y - c.y);

		// exact determinant OF THE ROUNDED DIFFERENCES
		float64 left_err = 0, right_err = 0;
		float64 const left = two_product(acx, bcy, left_err);
		float64 const right = two_product(acy, bcx, right_err);
		float64 base_expansion[4] = {};
		two_two_diff(left, left_err, right, right_err, base_expansion[3], base_expansion[2], base_expansion[1], base_expansion[0]);
		float64 det = base_expansion[0] + base_expansion[1] + base_expansion[2] + base_expansion[3];
		float64 bound = orient_bound_b * magnitude_sum;
		if (det >= bound || -det >= bound) return sign_of(det);

		// the differences themselves rounded: recover their exact tails
		float64 const acx_tail = two_diff_tail(b.x, a.x, acx);
		float64 const bcx_tail = two_diff_tail(d.x, c.x, bcx);
		float64 const acy_tail = two_diff_tail(b.y, a.y, acy);
		float64 const bcy_tail = two_diff_tail(d.y, c.y, bcy);
		if (acx_tail == 0 && acy_tail == 0 && bcx_tail == 0 && bcy_tail == 0)
			return sign_of(det); // differences were exact: det already is too

		// second-order correction, still just an estimate
		bound = orient_bound_c * magnitude_sum + result_bound * (det >= 0 ? det : -det);
		det += (acx * bcy_tail + bcy * acx_tail) - (acy * bcx_tail + bcx * acy_tail);
		if (det >= bound || -det >= bound) return sign_of(det);

		// exact: fold the three remaining cross-term expansions in
		float64 u[4] = {}, c1[8] = {}, c2[12] = {}, final_expansion[16] = {};
		float64 s1 = 0, s0 = 0, t1 = 0, t0 = 0;

		s1 = two_product(acx_tail, bcy, s0);
		t1 = two_product(acy_tail, bcx, t0);
		two_two_diff(s1, s0, t1, t0, u[3], u[2], u[1], u[0]);
		int const c1n = expansion_sum(4, base_expansion, 4, u, c1);

		s1 = two_product(acx, bcy_tail, s0);
		t1 = two_product(acy, bcx_tail, t0);
		two_two_diff(s1, s0, t1, t0, u[3], u[2], u[1], u[0]);
		int const c2n = expansion_sum(c1n, c1, 4, u, c2);

		s1 = two_product(acx_tail, bcy_tail, s0);
		t1 = two_product(acy_tail, bcx_tail, t0);
		two_two_diff(s1, s0, t1, t0, u[3], u[2], u[1], u[0]);
		int const dn = expansion_sum(c2n, c2, 4, u, final_expansion);

		return sign_of(final_expansion[dn - 1]);
	}

	/**
	 * @brief EXACT sign of `(b - a) x (d - c)` for `double` points —
	 *        four INDEPENDENT points (edge-direction comparisons): the
	 *        static filter certifies the common case from one extra
	 *        comparison, near-degenerate inputs fall through to
	 *        `adaptive_cross_tail`. The rounded differences keep the exact
	 *        signs of the true ones, which is what makes the quick
	 *        opposite-sign returns sound.
	 */
	constexpr int adaptive_cross_sign(point<float64> const a, point<float64> const b,
									  point<float64> const c, point<float64> const d) {
		float64 const left = (b.x - a.x) * (d.y - c.y);
		float64 const right = (b.y - a.y) * (d.x - c.x);
		// an infinite or NaN product would pass the filter below with a
		// wrong sign (the floating @pre: no overflow, no NaN)
		[[maybe_unused]] constexpr float64 largest = (std::numeric_limits<float64>::max)();
		atl_assert(-largest <= left && left <= largest && -largest <= right && right <= largest);
		float64 const det = left - right;
		float64 magnitude_sum = 0;
		if (left > 0) {
			if (right <= 0) return sign_of(det);
			magnitude_sum = left + right;
		} else if (left < 0) {
			if (right >= 0) return sign_of(det);
			magnitude_sum = -left - right;
		} else {
			// left == 0 exactly: det == -right exactly, unless the product
			// underflowed or a coordinate is NaN (the floating @pre)
			atl_assert(b.x == a.x || d.y == c.y);
			return sign_of(det);
		}
		float64 const bound = orient_bound_a * magnitude_sum;
		if (det >= bound || -det >= bound) return sign_of(det);
		atl_assert(magnitude_sum <= largest); // finite products can still sum past it
		return adaptive_cross_tail(a, b, c, d, magnitude_sum);
	}

	/**
	 * @brief Orientation as the shared-origin special case:
	 * `(pa - pc) x (pb - pc)`.
	 */
	constexpr int adaptive_orientation(point<float64> const a, point<float64> const b,
									   point<float64> const c) {
		return adaptive_cross_sign(c, a, c, b);
	}

	template<std::floating_point T>
		requires coordinate<T>
	struct exact_predicates<T> {
		static constexpr int orientation(point<T> o, point<T> a, point<T> b) {
			return adaptive_orientation(to_float64(a), to_float64(b), to_float64(o));
		}

		static constexpr int cross_sign(point<T> a, point<T> b, point<T> c, point<T> d) {
			return adaptive_cross_sign(to_float64(a), to_float64(b), to_float64(c), to_float64(d));
		}
	};
}
