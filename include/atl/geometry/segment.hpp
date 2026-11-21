#pragma once
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <algorithm>
#include <cmath>
#include <concepts>

namespace atl::geometry {
	/**
	 * @brief A closed line segment between two points. Aggregate;
	 *        `segment{p, q}` deduces `T`. No invariant is enforced: the
	 *        degenerate `a == b` (a point) is legal and every operation
	 *        handles it.
	 */
	template<coordinate T>
	struct segment {
		/** @brief The coordinate type @p T. */
		using coordinate_type = T;

		point<T> a{}; ///< First endpoint.
		point<T> b{}; ///< Second endpoint.

		/** @brief Representational equality: the same endpoints in the same order. */
		friend constexpr bool operator==(segment const &, segment const &) = default;
	};

	/**
	 * @brief The vector `b - a`, in `T` (the overflow caveat of `point`
	 * arithmetic applies; the predicates below widen internally instead).
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<T> direction(segment<T> const s) {
		return s.b - s.a;
	}

	/**
	 * @brief Exact squared length (for integral coordinates), see
	 * @ref atl::geometry::squared_distance.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 */
	template<coordinate T>
	[[nodiscard]] constexpr difference_product_t<T> squared_length(segment<T> const s) {
		return squared_distance(s.a, s.b);
	}

	/**
	 * @brief Length as `double`, a construction.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 length(segment<T> const s) {
		return distance(s.a, s.b);
	}

	namespace detail {
		/**
		 * @brief Sign of the dot product `(b - a) . (d - c)` under policy
		 *        @p p, the counterpart of the policies' `cross_sign(a, b, c, d)`:
		 *        positive when the two directions point the same way, zero when
		 *        they are perpendicular, negative when opposed. Exact with the
		 *        default policy.
		 *
		 * Integral coordinates take the dot product in the widened types and
		 * sign it through the policy; floating ones go through the policy's
		 * `cross_sign` on the perpendiculars.
		 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
		 */
		template<coordinate T, sign_policy<T> Policy = exact>
		constexpr int dot_sign(point<T> const a, point<T> const b, point<T> const c, point<T> const d,
							   Policy const p = {}) {
			if constexpr (std::floating_point<T>) {
				// dot(u, w) = -cross(perp(u), w), and negating a float is exact
				return -p.cross_sign(perp(a), perp(b), c, d);
			} else {
				using Difference = product_t<T>;
				using Product = difference_product_t<T>;
				check_difference_bounds(a, b, c, d);
				Difference const ux = widened_difference<Difference>(b.x, a.x), uy = widened_difference<Difference>(b.y, a.y);
				Difference const wx = widened_difference<Difference>(d.x, c.x), wy = widened_difference<Difference>(d.y, c.y);
				return p.sign(Product(Product(ux) * Product(wx) + Product(uy) * Product(wy)));
			}
		}
	}

	/**
	 * @brief Distance from @p q to the closest point of the CLOSED segment
	 *        @p s, a `double` construction. Which part of the segment is
	 *        closest (an endpoint or the interior) is decided exactly; the
	 *        value is then one rounded endpoint distance, or the cross product
	 *        over the rounded length. A point on the segment (exact
	 *        predicate) returns exactly 0, for floating coordinates too.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted); for
	 *      floating ones the floating @ref atl::geometry::orientation precondition.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(segment<T> const s, point<T> const q) {
		// q projects before a: a is closest; past b: b is; otherwise the interior
		if (s.a == s.b || detail::dot_sign(s.a, s.b, s.a, q) <= 0) return distance(s.a, q);
		if (detail::dot_sign(s.b, s.a, s.b, q) <= 0) return distance(s.b, q);
		// a rounded floating cross of collinear points need not be 0
		if constexpr (std::floating_point<T>)
			if (exact{}.orientation(s.a, s.b, q) == 0) return 0.0;
		float64 const doubled = float64(cross(s.a, s.b, q));
		return (doubled < 0 ? -doubled : doubled) / distance(s.a, s.b);
	}

	/**
	 * @brief Distance between two CLOSED segments, a `double` construction:
	 *        exactly 0 when they intersect (the exact predicate decides
	 *        first). Disjoint segments attain their minimum at an endpoint
	 *        of one of them, so the value is the least of the four
	 *        endpoint-to-segment distances. Degenerate segments are points.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted); for
	 *      floating ones the floating @ref atl::geometry::orientation precondition.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(segment<T> const s, segment<T> const t) {
		if (intersects(s, t)) return 0.0;
		float64 best = distance(s, t.a);
		best = (std::min)(best, distance(s, t.b));
		best = (std::min)(best, distance(t, s.a));
		return (std::min)(best, distance(t, s.b));
	}

	namespace detail {
		/**
		 * @brief Whether @p q lies in the closed bounding box of @p s, which for
		 * a point already known collinear with the segment is exactly the
		 * on-segment test. Differences are widened (raw `T` subtraction could
		 * overflow) and signed through the policy, so `exact` and `fast` are
		 * plain exact comparisons while `approx{eps}` inflates the box by its
		 * dead zone (coordinate units).
		 */
		template<coordinate T, sign_policy<T> Policy>
		constexpr bool in_box(segment<T> const s, point<T> const q, Policy const p) {
			using D = product_t<T>;
			auto const [xlo, xhi] = std::minmax(s.a.x, s.b.x);
			auto const [ylo, yhi] = std::minmax(s.a.y, s.b.y);
			return p.sign(widened_difference<D>(q.x, xlo)) >= 0 && p.sign(widened_difference<D>(xhi, q.x)) >= 0
				&& p.sign(widened_difference<D>(q.y, ylo)) >= 0 && p.sign(widened_difference<D>(yhi, q.y)) >= 0;
		}
	}

	/**
	 * @brief Whether @p q lies on the CLOSED segment @p s (endpoints count).
	 *        Collinearity is the policy's orientation; the position along the
	 *        line is an exact bounding-box test, so with the default policy
	 *        the answer is exact for integral and floating coordinates alike.
	 *        With `approx{eps}` both tests are tolerant: collinearity within
	 *        the dead zone (cross-product units) and the box inflated by eps
	 *        (coordinate units).
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted); for
	 *      floating ones the floating @ref atl::geometry::orientation precondition.
	 * @complexity O(1)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool contains(segment<T> const s, point<T> const q, Policy const p = {}) {
		return p.orientation(s.a, s.b, q) == 0 && detail::in_box(s, q, p);
	}

	/**
	 * @brief Whether the CLOSED segments @p s and @p t share at least one
	 *        point: touching at endpoints and collinear overlap count. A
	 *        proper crossing is two strict straddles; the remaining cases
	 *        reduce to an endpoint lying on the other segment. Exact under the
	 *        default policy for integral and floating coordinates; degenerate
	 *        (point) segments are handled.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted); for
	 *      floating ones the floating @ref atl::geometry::orientation precondition.
	 * @complexity O(1)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(segment<T> const s, segment<T> const t, Policy const p = {}) {
		int const d1 = p.orientation(t.a, t.b, s.a);
		int const d2 = p.orientation(t.a, t.b, s.b);
		// both ends strictly on one side of t's line: no shared point. Only
		// for the sharp policies; approx's dead zone plus its inflated box
		// may still answer true through d3 or d4
		if constexpr (std::same_as<Policy, exact> || std::same_as<Policy, fast>)
			if (d1 * d2 > 0) return false;
		int const d3 = p.orientation(s.a, s.b, t.a);
		int const d4 = p.orientation(s.a, s.b, t.b);
		if (d1 * d2 < 0 && d3 * d4 < 0) return true; // proper crossing
		return (d1 == 0 && detail::in_box(t, s.a, p)) || (d2 == 0 && detail::in_box(t, s.b, p))
			|| (d3 == 0 && detail::in_box(s, t.a, p)) || (d4 == 0 && detail::in_box(s, t.b, p));
	}
}
