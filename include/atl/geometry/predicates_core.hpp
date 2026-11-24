#pragma once

/** @file
 * @brief Sign policies and integral exact predicates; predicates_floating.hpp adds exact floating predicates.
 */
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>

namespace atl::geometry {
	namespace detail {
		/**
		 * @brief Whether @p value is finite (neither infinite nor NaN), usable in constant evaluation: `std::isfinite`
		 *        is not constexpr on every supported standard library. `v - v` is exactly zero for finite `v` and NaN
		 *        otherwise.
		 */
		[[nodiscard]] constexpr bool is_finite(float64 const value) { return value - value == 0.0; }

		/**
		 * @brief `(b - a) x (d - c)` for integral coordinates, EXACT: the
		 *        differences and products are widened, so nothing overflows.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T>
			requires std::signed_integral<T>
		constexpr difference_product_t<T> cross_determinant(point<T> const a, point<T> const b,
															point<T> const c, point<T> const d) {
			using D = product_t<T>;
			using W = difference_product_t<T>;
			check_difference_bounds(a, b, c, d);
			D const abx = widened_difference<D>(b.x, a.x), aby = widened_difference<D>(b.y, a.y);
			D const cdx = widened_difference<D>(d.x, c.x), cdy = widened_difference<D>(d.y, c.y);
			return W(W(abx) * W(cdy) - W(aby) * W(cdx));
		}
	}

	namespace detail {
		template<coordinate T>
		struct exact_predicates;

		template<coordinate T>
			requires std::signed_integral<T>
		struct exact_predicates<T> {
			static constexpr int orientation(point<T> o, point<T> a, point<T> b) {
				return sign_of(cross(o, a, b));
			}

			static constexpr int cross_sign(point<T> a, point<T> b, point<T> c, point<T> d) {
				return sign_of(cross_determinant(a, b, c, d));
			}
		};
	}

	/**
	 * @brief The default sign policy: every sign is EXACT.
	 *
	 * Integral coordinates use widened differences and products (see the
	 * three-point @ref atl::geometry::cross). Floating coordinates use the filtered adaptive
	 * predicates: the sign returned is the sign of the true real determinant
	 * of the given values, so near-degenerate inputs cost more but are never
	 * misclassified. Point equality is plain `==`, which for floating input
	 * is exact value identity (every `double` is an exact rational).
	 */
	struct exact {
		/**
		 * @brief Exact sign of `(a - o) x (b - o)`: +1 for a left turn, -1 for
		 * a right turn, 0 for collinear (see @ref atl::geometry::orientation).
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted). For
		 *      floating ones: finite, and no product of coordinate
		 *      differences overflows or underflows (overflow, NaN and the
		 *      zero-product underflow are asserted).
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int orientation(point<T> const o, point<T> const a, point<T> const b) const {
			return detail::exact_predicates<T>::orientation(o, a, b);
		}

		/**
		 * @brief Exact sign of `(b - a) x (d - c)`, the edge-direction
		 * comparison of four independent points (orientation is the
		 * shared-origin special case).
		 * @pre As @ref atl::geometry::orientation.
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int cross_sign(point<T> const a, point<T> const b, point<T> const c,
											   point<T> const d) const {
			return detail::exact_predicates<T>::cross_sign(a, b, c, d);
		}

		/** @brief Exact sign of @p v: -1, 0 or +1. */
		template<typename V>
			requires std::constructible_from<V, int> && std::totally_ordered<V>
		[[nodiscard]] constexpr int sign(V const v) const {
			return detail::sign_of(v);
		}

		/** @brief Exact point identity, `a == b`. */
		template<coordinate T>
		[[nodiscard]] constexpr bool equal(point<T> const a, point<T> const b) const {
			return a == b;
		}
	};

	/**
	 * @brief The loose sign policy, opted into by naming it: floating
	 *        orientation is the raw IEEE evaluation (no filter, no exact
	 *        fallback), so near-degenerate inputs may be misclassified and the
	 *        result on exact ties may depend on floating contraction
	 *        (`-ffp-contract`). Integral coordinates are identical to
	 *        @ref atl::geometry::exact. Use it only where the input is known non-degenerate or
	 *        a wrong tie sign is harmless.
	 */
	struct fast {
		/**
		 * @brief Sign of `(a - o) x (b - o)`: the rounded `double` determinant
		 * for floating coordinates, exact for integral ones.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int orientation(point<T> const o, point<T> const a, point<T> const b) const {
			if constexpr (std::floating_point<T>)
				return detail::sign_of(cross(to_float64(o), to_float64(a), to_float64(b)));
			else
				return exact{}.orientation(o, a, b);
		}

		/**
		 * @brief Sign of `(b - a) x (d - c)`: the rounded `double` determinant
		 * for floating coordinates, exact for integral ones.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int cross_sign(point<T> const a, point<T> const b, point<T> const c,
											   point<T> const d) const {
			if constexpr (std::floating_point<T>) {
				float64 const det = (float64(b.x) - float64(a.x)) * (float64(d.y) - float64(c.y))
								  - (float64(b.y) - float64(a.y)) * (float64(d.x) - float64(c.x));
				return detail::sign_of(det);
			} else
				return exact{}.cross_sign(a, b, c, d);
		}

		/** @brief Exact sign of @p v: -1, 0 or +1. */
		template<typename V>
			requires std::constructible_from<V, int> && std::totally_ordered<V>
		[[nodiscard]] constexpr int sign(V const v) const {
			return detail::sign_of(v);
		}

		/** @brief Exact point identity, `a == b`. */
		template<coordinate T>
		[[nodiscard]] constexpr bool equal(point<T> const a, point<T> const b) const {
			return a == b;
		}
	};

	/**
	 * @brief Dead-zone comparisons for CONSTRUCTED floating values:
	 *        quantities that were already rounded (intersection points,
	 *        lengths, rotated coordinates), where no exact predicate can
	 *        recover the lost information. The epsilon is a member, visible
	 *        at every call site; there is no default and no global.
	 *
	 * Every sign is the dead-zone sign of a value converted to `double`, so
	 * mind its units: orientation and cross values are products of two
	 * coordinate differences, `equal` compares a distance. For integral
	 * coordinates the determinant itself is exact (widened) and only its
	 * conversion is rounded, so `orientation` and `cross_sign` agree.
	 *
	 * `equal` is by euclidean distance (rotation-invariant "same point") and
	 * is NOT transitive: use it for boundary and output decisions, never as a
	 * sort predicate, map key or deduplication relation. To order doubles,
	 * use their exact `<`.
	 */
	struct approx {
		float64 epsilon; ///< Half-width of the dead zone, in the units of the compared value.

		/**
		 * @brief Dead-zone sign of `(a - o) x (b - o)`, in cross-product units:
		 * the three-point @ref atl::geometry::cross converted to `double`.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int orientation(point<T> const o, point<T> const a, point<T> const b) const {
			return sign(float64(cross(o, a, b)));
		}

		/**
		 * @brief Dead-zone sign of `(b - a) x (d - c)`, in cross-product units.
		 * Integral coordinates take the exact widened determinant and apply
		 * the dead zone to its `double` conversion, as @ref atl::geometry::orientation does;
		 * floating ones evaluate it in `double` directly.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T>
		[[nodiscard]] constexpr int cross_sign(point<T> const a, point<T> const b, point<T> const c,
											   point<T> const d) const {
			if constexpr (std::floating_point<T>) {
				float64 const det = (float64(b.x) - float64(a.x)) * (float64(d.y) - float64(c.y))
								  - (float64(b.y) - float64(a.y)) * (float64(d.x) - float64(c.x));
				return sign(det);
			} else
				return sign(detail::cross_determinant(a, b, c, d));
		}

		/**
		 * @brief Sign with dead zone: 0 whenever `|v| <= epsilon`, @p v
		 * compared as `double`.
		 */
		template<typename V>
			requires std::constructible_from<float64, V>
		[[nodiscard]] constexpr int sign(V const v) const {
			float64 const d = float64(v);
			return int(d > epsilon) - int(d < -epsilon);
		}

		/** @brief Whether the euclidean distance of @p a and @p b is at most `epsilon`. */
		template<coordinate T>
		[[nodiscard]] constexpr bool equal(point<T> const a, point<T> const b) const {
			auto const delta = detail::float64_difference(a, b);
			float64 const dx = delta.x, dy = delta.y;
			return dx * dx + dy * dy <= epsilon * epsilon;
		}

		/** @brief Whether `|a - b| <= epsilon`. */
		[[nodiscard]] constexpr bool equal(float64 const a, float64 const b) const {
			float64 const d = a - b;
			return -epsilon <= d && d <= epsilon;
		}
	};

	/**
	 * @brief What a sign policy must provide to drive the geometric
	 *        predicates over points of coordinate @p T: a three-point
	 *        orientation sign, a four-point edge-direction sign
	 *        `cross_sign(a, b, c, d)` of `(b - a) x (d - c)`, a scalar sign
	 *        of both widened product types (@ref atl::geometry::product_t and
	 *        @ref atl::geometry::difference_product_t values), and point equality, all on a
	 *        const policy. @ref atl::geometry::exact (the default everywhere), @ref atl::geometry::fast and
	 *        @ref atl::geometry::approx model it; user policies can too.
	 */
	template<typename Policy, typename T>
	concept sign_policy = coordinate<T> && std::copyable<Policy>
					   && requires(Policy const p, point<T> const q, product_t<T> const w,
								   difference_product_t<T> const v) {
							  { p.orientation(q, q, q) } -> std::same_as<int>;
							  { p.cross_sign(q, q, q, q) } -> std::same_as<int>;
							  { p.sign(w) } -> std::same_as<int>;
							  { p.sign(v) } -> std::same_as<int>;
							  { p.equal(q, q) } -> std::same_as<bool>;
						  };

	namespace detail {
		/**
		 * @brief Angular class of the direction `from -> to` in the [0, 2pi)
		 * sweep order (0 = +x axis, rotating counterclockwise). Signs of
		 * coordinate differences are exact even for doubles. The hull Minkowski
		 * merge and rotating calipers share this order.
		 */
		template<coordinate T>
		constexpr int sweep_group(point<T> const from, point<T> const to) {
			if (to.y > from.y) return 1;  // (0, pi)
			if (to.y < from.y) return 3;  // (pi, 2pi)
			return to.x > from.x ? 0 : 2; // +x axis : -x axis
		}

		/**
		 * @brief Whether direction `a -> b` is strictly angularly before
		 * `c -> d` in the [0, 2pi) sweep: groups first, one cross sign within
		 * a group.
		 */
		template<coordinate T, sign_policy<T> Policy>
		constexpr bool sweep_before(point<T> const a, point<T> const b, point<T> const c,
									point<T> const d, Policy const p) {
			int const ga = sweep_group(a, b), gc = sweep_group(c, d);
			if (ga != gc) return ga < gc;
			return p.cross_sign(a, b, c, d) > 0;
		}
	}

	/**
	 * @brief Orientation of the triple `(o, a, b)`: +1 if `b` lies
	 *        counterclockwise of `a` as seen from `o` (left turn), -1 if
	 *        clockwise, 0 if collinear; the sign of `cross(a - o, b - o)` as
	 *        certified by the policy. With @ref atl::geometry::exact (the default) the sign
	 *        is exact for integral and floating coordinates; duplicate points
	 *        give 0.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted). For
	 *      floating ones: finite values and no overflow or underflow in the
	 *      products (roughly `|c| = 0` or `1e-146 < |c| < 1e153`).
	 * @complexity O(1); the exact floating path is adaptive (one extra
	 *             comparison when the static filter certifies the sign)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr int orientation(point<T> const o, point<T> const a, point<T> const b, Policy const p = {}) {
		return p.orientation(o, a, b);
	}
}
