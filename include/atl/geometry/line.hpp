#pragma once
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/geometry/segment.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>

namespace atl::geometry {
	/**
	 * @brief An infinite DIRECTED line through two distinct points,
	 *        `a -> b`. The direction is meaningful: `side_of` is signed
	 *        relative to it, so a directed line also describes the closed
	 *        half-plane on its left. Aggregate; `==` is representational (the
	 *        same two points in the same order); geometric coincidence is
	 *        @ref atl::geometry::same_line.
	 * @pre `a != b` for every operation below (asserted there).
	 */
	template<coordinate T>
	struct line {
		/** @brief The coordinate type @p T. */
		using coordinate_type = T;

		point<T> a{}; ///< A point on the line, where the direction starts.
		point<T> b{}; ///< A second point on the line, where the direction points.

		/** @brief Representational equality: the same two points in the same order. */
		friend constexpr bool operator==(line const &, line const &) = default;
	};

	/**
	 * @brief The vector `b - a`, in `T` (the overflow caveat of `point`
	 * arithmetic applies; the predicates below widen internally instead).
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<T> direction(line<T> const l) {
		return l.b - l.a;
	}

	/**
	 * @brief Side of a point relative to a directed line; the values follow the
	 * sign convention (`int(s)` is the orientation sign).
	 */
	enum class side : int { right = -1, ///< Clockwise of the direction (negative orientation).
							on = 0,		///< On the line.
							left = 1 }; ///< Counterclockwise of the direction (positive orientation).

	/**
	 * @brief Side of @p q relative to the directed line: @ref atl::geometry::orientation of
	 * `(l.a, l.b, q)` as a classification.
	 * @pre `l.a != l.b` (asserted); for `int64` coordinates `|x|, |y| < 2^62`
	 *      (asserted); for floating ones the floating @ref atl::geometry::orientation
	 *      precondition. The same holds for every predicate below.
	 * @complexity O(1)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr side side_of(line<T> const l, point<T> const q, Policy const p = {}) {
		atl_assert(l.a != l.b);
		return side(p.orientation(l.a, l.b, q));
	}

	/** @brief Whether @p q lies on the line. @pre As @ref atl::geometry::side_of. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool contains(line<T> const l, point<T> const q, Policy const p = {}) {
		return side_of(l, q, p) == side::on;
	}

	/**
	 * @brief Whether the two lines have the same or opposite direction (this
	 * includes coinciding lines).
	 * @pre As @ref atl::geometry::side_of, for both lines.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool parallel(line<T> const l, line<T> const m, Policy const p = {}) {
		atl_assert(l.a != l.b && m.a != m.b);
		return p.cross_sign(l.a, l.b, m.a, m.b) == 0;
	}

	/**
	 * @brief Whether the two lines coincide as point sets (either direction) —
	 * the geometric equality that the representational `==` is not.
	 * @pre As @ref atl::geometry::side_of, for both lines.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool same_line(line<T> const l, line<T> const m, Policy const p = {}) {
		return parallel(l, m, p) && side_of(l, m.a, p) == side::on;
	}

	/**
	 * @brief Whether the two lines share at least one point: non-parallel, or
	 * coinciding.
	 * @pre As @ref atl::geometry::side_of, for both lines.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(line<T> const l, line<T> const m, Policy const p = {}) {
		return !parallel(l, m, p) || side_of(l, m.a, p) == side::on;
	}

	/**
	 * @brief Whether the line meets the CLOSED segment: the endpoints lie on
	 * opposite sides, or either touches the line. A degenerate segment is a
	 * point.
	 * @pre As @ref atl::geometry::side_of.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(line<T> const l, segment<T> const s, Policy const p = {}) {
		side const at_a = side_of(l, s.a, p), at_b = side_of(l, s.b, p);
		return at_a == side::on || at_b == side::on || at_a != at_b;
	}

	/** @brief `intersects(l, s, p)`, the segment-first argument order. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(segment<T> const s, line<T> const l, Policy const p = {}) {
		return intersects(l, s, p);
	}

	/**
	 * @brief Side of @p q relative to the directed line SPANNED by the segment —
	 * the polygon-edge side test without spelling the line out.
	 * @pre `s.a != s.b` (asserted); otherwise as @ref atl::geometry::side_of.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr side side_of(segment<T> const s, point<T> const q, Policy const p = {}) {
		return side_of(line<T>{s.a, s.b}, q, p);
	}

	/**
	 * @brief The coefficients `(A, B, C)` of the directed line as
	 *        `A x + B y + C = 0`, exact for integral coordinates (rounded for
	 *        floating ones), oriented so that
	 *        `side_of(l, q) == sign(A q.x + B q.y + C)` (`side::left` is the
	 *        positive side). `A = a.y - b.y`, `B = b.x - a.x`.
	 * @pre `l.a != l.b` (asserted); for `int64` coordinates `|x|, |y| < 2^62`
	 *      (asserted), which keeps `|C| < 2^126`.
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr std::array<difference_product_t<T>, 3> line_coefficients(line<T> const l) {
		atl_assert(l.a != l.b);
		detail::check_difference_bounds(l.a, l.b);
		using D = product_t<T>;
		using P2 = difference_product_t<T>;
		D const direction_x = detail::widened_difference<D>(l.b.x, l.a.x);
		D const direction_y = detail::widened_difference<D>(l.b.y, l.a.y);
		P2 const a = P2(-P2(direction_y)), b = P2(direction_x);
		return {a, b, P2(-(a * P2(l.a.x) + b * P2(l.a.y)))};
	}

	/**
	 * @brief The directed line of `A x + B y + C = 0` (`A`, `B` not both
	 *        zero — asserted), oriented so `side::left` is the positive
	 *        side, anchored at the foot of the perpendicular from the
	 *        origin. A `double` construction: an integer-coefficient line
	 *        need not contain any integer point (`2x + 2y - 1 = 0`).
	 *
	 * Takes the array @ref atl::geometry::line_coefficients returns, so the two round-trip:
	 * @p Coefficient may be any signed integer (`atl::int128` included) or
	 * floating type, each coefficient converted to `double` once.
	 * @pre All formed squares and pairwise products of converted coefficients
	 *      are finite and normal, except products with a zero factor (asserted
	 *      for the norm and anchor products); the norm and endpoints are finite.
	 * @complexity O(1)
	 */
	template<typename Coefficient>
		requires atl::signed_integral<Coefficient> || std::floating_point<Coefficient>
	[[nodiscard]] constexpr line<float64> line_from_coefficients(std::array<Coefficient, 3> const &coefficients) {
		auto const [a, b, c] = coefficients;
		atl_assert(a != Coefficient(0) || b != Coefficient(0));
		float64 const af = float64(a), bf = float64(b), cf = float64(c);
		atl_assert(detail::normal_product(af, af) && detail::normal_product(bf, bf));
		atl_assert(detail::normal_product(af, cf) && detail::normal_product(bf, cf));
		float64 const n2 = af * af + bf * bf;
		atl_assert(n2 <= (std::numeric_limits<float64>::max)());
		point<float64> const anchor{-af * cf / n2, -bf * cf / n2};
		// A tangent smaller than an ulp of the anchor would collapse the line.
		float64 const anchor_scale = (std::max)({1.0, anchor.x < 0 ? -anchor.x : anchor.x, anchor.y < 0 ? -anchor.y : anchor.y});
		float64 const tangent_scale = (std::max)(af < 0 ? -af : af, bf < 0 ? -bf : bf);
		float64 const displacement = (std::max)(tangent_scale, anchor_scale * 0x1p-51);
		point<float64> const endpoint = anchor + point<float64>{(bf / tangent_scale) * displacement, (-af / tangent_scale) * displacement};
		atl_assert(anchor != endpoint);
		atl_assert(endpoint.x <= (std::numeric_limits<float64>::max)() && endpoint.x >= -(std::numeric_limits<float64>::max)());
		atl_assert(endpoint.y <= (std::numeric_limits<float64>::max)() && endpoint.y >= -(std::numeric_limits<float64>::max)());
		return {anchor, endpoint};
	}

	/** @brief `line_from_coefficients({a, b, c})` for three coordinate values. */
	template<coordinate T>
	[[nodiscard]] constexpr line<float64> line_from_coefficients(T const a, T const b, T const c) {
		return line_from_coefficients(std::array<T, 3>{a, b, c});
	}

	/**
	 * @brief Distance between two lines, a `double` construction: exactly 0
	 *        unless they are parallel and distinct (the exact predicate
	 *        decides), then the point-to-line distance of `m.a`.
	 * @pre As @ref atl::geometry::side_of, for both lines.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(line<T> const l, line<T> const m) {
		if (intersects(l, m)) return 0.0;
		return distance(l, m.a);
	}

	/**
	 * @brief Distance between a line and a CLOSED segment, a `double`
	 *        construction: exactly 0 when they meet (exact predicate first); a
	 *        disjoint segment lies on one side, so the gap is at the nearer
	 *        endpoint.
	 * @pre As @ref atl::geometry::side_of.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(line<T> const l, segment<T> const s) {
		if (intersects(l, s)) return 0.0;
		return (std::min)(distance(l, s.a), distance(l, s.b));
	}

	/**
	 * @brief `distance(l, s)`, the segment-first argument order.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(segment<T> const s, line<T> const l) {
		return distance(l, s);
	}

	/**
	 * @brief Shape of an intersection. The operations of this header produce
	 * the first three; `along_ray` is reserved for half-infinite overlaps,
	 * which no operation here can produce.
	 */
	enum class intersection_kind { empty,		  ///< No common point.
								   at_point,	  ///< Exactly one common point.
								   along_segment, ///< A collinear overlap of positive length.
								   along_ray };	  ///< A half-infinite collinear overlap.

	/**
	 * @brief Result of the segment/segment and line/segment intersections:
	 *        a kind plus the field it makes meaningful (the other field is
	 *        value-initialized).
	 */
	template<coordinate T>
	struct segment_intersection {
		intersection_kind kind{intersection_kind::empty}; ///< Which field is meaningful.
		/**
		 * @brief The single common point when `kind == at_point`: a `double`
		 * construction, exact when it is an input point with coordinates
		 * exactly representable in `double` (integers within `2^53`).
		 */
		point<float64> at{};
		/**
		 * @brief The overlap when `kind == along_segment`: EXACT input
		 * points, lexicographically ordered (`along.a < along.b`).
		 */
		segment<T> along{};

		/** @brief Memberwise equality, including the unused field. */
		friend constexpr bool operator==(segment_intersection const &,
										 segment_intersection const &) = default;
	};

	/**
	 * @brief The intersection of two CLOSED segments: empty, one point (a
	 *        proper crossing, constructed, or a touch at an input point,
	 *        exact), or the exact collinear overlap segment. Degenerate
	 *        segments are points.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted); for
	 *      floating ones the floating @ref atl::geometry::orientation precondition.
	 * @complexity O(1)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr segment_intersection<T> intersection(segment<T> const s, segment<T> const t,
																 Policy const p = {}) {
		segment_intersection<T> out;
		if (!intersects(s, t, p)) return out;
		auto const as_single = [&](point<T> const q) {
			out.kind = intersection_kind::at_point;
			out.at = to_float64(q);
		};
		if (p.equal(s.a, s.b)) { // degenerate segments: the shared point
			as_single(s.a);
			return out;
		}
		if (p.equal(t.a, t.b)) {
			as_single(t.a);
			return out;
		}
		int const d1 = p.orientation(t.a, t.b, s.a), d2 = p.orientation(t.a, t.b, s.b);
		int const d3 = p.orientation(s.a, s.b, t.a), d4 = p.orientation(s.a, s.b, t.b);
		if (d1 * d2 < 0 && d3 * d4 < 0) { // proper crossing: a construction
			out.kind = intersection_kind::at_point;
			out.at = intersection(line<T>{s.a, s.b}, line<T>{t.a, t.b});
			return out;
		}
		if (d3 == 0 && d4 == 0) {
			// same line: overlap endpoints are input points; on a common
			// line the lexicographic order IS the order along it
			point<T> const s_lo = s.a < s.b ? s.a : s.b, s_hi = s.a < s.b ? s.b : s.a;
			point<T> const t_lo = t.a < t.b ? t.a : t.b, t_hi = t.a < t.b ? t.b : t.a;
			point<T> const lo = s_lo < t_lo ? t_lo : s_lo;
			point<T> const hi = s_hi < t_hi ? s_hi : t_hi;
			if (hi < lo) return out; // inflated boxes may meet across a gap
			if (p.equal(lo, hi)) {
				as_single(lo);
				return out;
			}
			out.kind = intersection_kind::along_segment;
			out.along = {lo, hi};
			return out;
		}
		// endpoint touch: the unique common point is an input vertex
		if (d1 == 0 && detail::in_box(t, s.a, p))
			as_single(s.a);
		else if (d2 == 0 && detail::in_box(t, s.b, p))
			as_single(s.b);
		else if (d3 == 0)
			as_single(t.a);
		else
			as_single(t.b);
		return out;
	}

	/**
	 * @brief The intersection of a line and a CLOSED segment: empty, one
	 *        point (an endpoint on the line, exact, or a crossing,
	 *        constructed), or the whole segment when it lies on the line
	 *        (exact, endpoints lexicographically ordered).
	 * @pre As @ref atl::geometry::side_of.
	 * @complexity O(1)
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr segment_intersection<T> intersection(line<T> const l, segment<T> const s,
																 Policy const p = {}) {
		segment_intersection<T> out;
		side const at_a = side_of(l, s.a, p), at_b = side_of(l, s.b, p);
		auto const as_single = [&](point<T> const q) {
			out.kind = intersection_kind::at_point;
			out.at = to_float64(q);
		};
		if (at_a == side::on && at_b == side::on) {
			if (p.equal(s.a, s.b)) {
				as_single(s.a);
				return out;
			}
			out.kind = intersection_kind::along_segment;
			out.along = s.a < s.b ? s : segment<T>{s.b, s.a};
			return out;
		}
		if (at_a == side::on) {
			as_single(s.a);
			return out;
		}
		if (at_b == side::on) {
			as_single(s.b);
			return out;
		}
		if (at_a == at_b) return out; // strictly one side: empty
		out.kind = intersection_kind::at_point;
		out.at = intersection(l, line<T>{s.a, s.b});
		return out;
	}

	/** @brief `intersection(l, s, p)`, the segment-first argument order. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr segment_intersection<T> intersection(segment<T> const s, line<T> const l,
																 Policy const p = {}) {
		return intersection(l, s, p);
	}

	/**
	 * @brief Distance from @p q to the line, a `double` construction: exactly
	 * 0 for points on the line (exact predicate), else the cross value over
	 * the rounded length.
	 * @pre As @ref atl::geometry::side_of.
	 * @note Not constexpr (`std::sqrt`).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(line<T> const l, point<T> const q) {
		atl_assert(l.a != l.b);
		// a rounded floating cross of collinear points need not be 0
		if constexpr (std::floating_point<T>)
			if (exact{}.orientation(l.a, l.b, q) == 0) return 0.0;
		float64 const doubled = float64(cross(l.a, l.b, q));
		return (doubled < 0 ? -doubled : doubled) / distance(l.a, l.b);
	}

	/**
	 * @brief Foot of the perpendicular from @p q onto the line, a `double`
	 * construction from exactly differenced coordinates.
	 * @pre `l.a != l.b` (asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<float64> projection(line<T> const l, point<T> const q) {
		atl_assert(l.a != l.b);
		auto const d = detail::float64_difference(l.a, l.b), w = detail::float64_difference(l.a, q);
		float64 const t = (w.x * d.x + w.y * d.y) / (d.x * d.x + d.y * d.y);
		return {float64(l.a.x) + t * d.x, float64(l.a.y) + t * d.y};
	}

	/**
	 * @brief The intersection point of two NON-PARALLEL lines, a `double`
	 *        construction (integer-line intersections are rationals): the
	 *        integer determinants are exact and converted only for the final
	 *        division. Floating determinants use double: with `u = 2^-53`, their absolute
	 *        error is at most `4u / (1 - 4u)` times the sum of the exact
	 *        product magnitudes (including the differencing errors); division
	 *        amplifies this error near parallel lines, and cancellation in
	 *        the final coordinate sum can further increase relative error.
	 * @pre `!parallel(l, m)` under the exact policy (asserted), which
	 *      implies `l.a != l.b` and `m.a != m.b`; for `int64` coordinates
	 *      `|x|, |y| < 2^62` (asserted). Floating products must be finite
	 *      and normal (or have a zero factor), and the rounded direction
	 *      determinant must be nonzero (asserted).
	 * @complexity O(1)
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<float64> intersection(line<T> const l, line<T> const m) {
		atl_assert(exact{}.cross_sign(l.a, l.b, m.a, m.b) != 0);
		auto const u = detail::float64_difference(l.a, l.b), v = detail::float64_difference(m.a, m.b);
		auto const w = detail::float64_difference(l.a, m.a);
		float64 t = 0;
		if constexpr (std::signed_integral<T>) {
			auto const numerator = detail::cross_determinant(l.a, m.a, m.a, m.b);
			auto const denominator = detail::cross_determinant(l.a, l.b, m.a, m.b);
			t = float64(numerator) / float64(denominator);
		} else {
			float64 const denominator = u.x * v.y - u.y * v.x;
			atl_assert(denominator != 0);
			t = (w.x * v.y - w.y * v.x) / denominator;
		}
		return {float64(l.a.x) + t * u.x, float64(l.a.y) + t * u.y};
	}
}
