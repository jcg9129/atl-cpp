#pragma once

/** @file
 * @brief Shared ray implementation without adaptive floating predicates.
 */
#include <atl/ds/static_vector.hpp>
#include <atl/geometry/circle_core.hpp>
#include <atl/geometry/line_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>

namespace atl::geometry {
	/**
	 * @defgroup ray_geometry Ray geometry
	 * @brief Closed rays with exact sign predicates and rounded constructions.
	 * @pre Rays have distinct defining points; lines have distinct endpoints.
	 *      Integral predicates: full `int16`/`int32` range, `|c| < 2^62` for `int64`.
	 *      Circle intersections require `|c|, radius < 2^30` for `int32`/`int64`,
	 *      `< 2^14` for `int16`; their floating counts use rounded circle predicates.
	 *      Floating predicates inherit the orientation product preconditions.
	 * @complexity O(1) time and space; no recursion.
	 * @{
	 */
	/**
	 * @brief A half-infinite ray: starts at @p origin, passes through
	 *        @p through, and continues beyond it. Aggregate; `==` is
	 *        representational.
	 * @pre `origin != through` for geometric queries (asserted); `direction` is
	 *      point arithmetic and requires a representable difference in `T`.
	 */
	template<coordinate T>
	struct ray {
		using coordinate_type = T; ///< Coordinate type.

		point<T> origin{};	///< Included starting point.
		point<T> through{}; ///< Distinct point specifying the forward direction.

		/** @brief Representational equality of all stored fields. */
		[[nodiscard]] friend constexpr bool operator==(ray const &, ray const &) = default;
	};

	/**
	 * @brief The vector `through - origin`, in `T` (same overflow caveat as
	 * `point` arithmetic; the predicates below widen internally).
	 */
	template<coordinate T>
	[[nodiscard]] constexpr point<T> direction(ray<T> const r) {
		return r.through - r.origin;
	}

	/** @brief Side of @p q relative to the directed line SPANNED by the ray. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr side side_of(ray<T> const r, point<T> const q, Policy const p = {}) {
		return side_of(line<T>{r.origin, r.through}, q, p);
	}

	/**
	 * @brief Whether @p q lies on the ray: on the spanning line, at or forward of
	 * the origin.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool contains(ray<T> const r, point<T> const q, Policy const p = {}) {
		atl_assert(r.origin != r.through);
		return p.orientation(r.origin, r.through, q) == 0 && detail::dot_sign(r.origin, r.through, r.origin, q, p) >= 0;
	}

	/**
	 * @brief Whether the ray meets the line: the origin's side is `on`, the
	 * through-point changes side, or the ray heads toward the line (the
	 * side value shrinks from origin to through).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(ray<T> const r, line<T> const l, Policy const p = {}) {
		atl_assert(r.origin != r.through);
		side const at_origin = side_of(l, r.origin, p);
		if (at_origin == side::on) return true;
		if (side_of(l, r.through, p) != at_origin) return true;
		return p.cross_sign(l.a, l.b, r.through, r.origin) == int(at_origin);
	}

	/** @brief Symmetric line-ray predicate; same bounds as the ray-line overload. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(line<T> const l, ray<T> const r, Policy const p = {}) {
		return intersects(r, l, p);
	}

	/**
	 * @brief Whether the ray meets the CLOSED segment — pure sign casework: the
	 * endpoints' sides of the spanning line, and for a proper straddle the
	 * sign of the ray parameter at the crossing.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(ray<T> const r, segment<T> const s, Policy const p = {}) {
		atl_assert(r.origin != r.through);
		int const at_a = p.orientation(r.origin, r.through, s.a);
		int const at_b = p.orientation(r.origin, r.through, s.b);
		if (at_a == 0 && at_b == 0)
			return detail::dot_sign(r.origin, r.through, r.origin, s.a, p) >= 0 || detail::dot_sign(r.origin, r.through, r.origin, s.b, p) >= 0;
		if (at_a == 0) return detail::dot_sign(r.origin, r.through, r.origin, s.a, p) >= 0;
		if (at_b == 0) return detail::dot_sign(r.origin, r.through, r.origin, s.b, p) >= 0;
		if (at_a == at_b) return false;
		// crossing parameter t = ((a - origin) x s) / (direction x s) >= 0
		int const denominator = p.cross_sign(r.origin, r.through, s.a, s.b);
		int const numerator = p.orientation(r.origin, s.a, s.b);
		return numerator == 0 || numerator == denominator;
	}

	/** @brief Symmetric closed-segment-ray predicate. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(segment<T> const s, ray<T> const r, Policy const p = {}) {
		return intersects(r, s, p);
	}

	/**
	 * @brief Whether the two rays share at least one point — collinear cases via
	 * mutual origin containment, the rest via both crossing parameters
	 * being non-negative.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr bool intersects(ray<T> const a, ray<T> const b, Policy const p = {}) {
		atl_assert(a.origin != a.through && b.origin != b.through);
		int const o1 = p.orientation(a.origin, a.through, b.origin);
		int const o2 = p.orientation(a.origin, a.through, b.through);
		if (o1 == 0 && o2 == 0) return contains(a, b.origin, p) || contains(b, a.origin, p);
		int const denominator = p.cross_sign(a.origin, a.through, b.origin, b.through);
		if (denominator == 0) return false; // parallel, distinct lines
		int const t_numerator = p.cross_sign(a.origin, b.origin, b.origin, b.through);
		int const u_numerator = p.cross_sign(a.origin, b.origin, a.origin, a.through);
		return (t_numerator == 0 || t_numerator == denominator)
			&& (u_numerator == 0 || u_numerator == denominator);
	}

	/**
	 * @brief Result of the ray intersections — the segment vocabulary plus
	 *        `along_ray` for the unbounded outcomes (ray on its line, or
	 *        collinear same-direction rays): `beam` is then the EXACT
	 *        overlap ray. As with @ref atl::geometry::segment_intersection, only the field
	 *        named by `kind` is meaningful.
	 */
	template<coordinate T>
	struct ray_intersection {
		intersection_kind kind{intersection_kind::empty}; ///< Selects the meaningful result field.
		point<float64> at{};							  ///< Rounded crossing; input vertices convert exactly only within `2^53`.
		segment<T> along{};								  ///< Exact overlap endpoints, lexicographically ordered.
		ray<T> beam{};									  ///< Exact unbounded overlap.

		/** @brief Representational equality of all stored fields. */
		[[nodiscard]] friend constexpr bool operator==(ray_intersection const &,
													   ray_intersection const &) = default;
	};

	/**
	 * @brief The intersection of a ray and a line: empty, one point, or the whole
	 * ray (exact) when it lies on the line.
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr ray_intersection<T> intersection(ray<T> const r, line<T> const l, Policy const p = {}) {
		atl_assert(r.origin != r.through);
		ray_intersection<T> out;
		if (contains(l, r.origin, p) && contains(l, r.through, p)) {
			out.kind = intersection_kind::along_ray;
			out.beam = r;
			return out;
		}
		if (!intersects(r, l, p)) return out;
		out.kind = intersection_kind::at_point;
		if (contains(l, r.origin, p))
			out.at = to_float64(r.origin);
		else
			out.at = intersection(line<T>{r.origin, r.through}, l);
		return out;
	}

	/** @brief Symmetric line-ray intersection, with the same result fields and bounds. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr ray_intersection<T> intersection(line<T> const l, ray<T> const r, Policy const p = {}) {
		return intersection(r, l, p);
	}

	/**
	 * @brief The intersection of a ray and a CLOSED segment: empty, one point, or
	 * an exact sub-segment (collinear overlap, clamped at the origin).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr ray_intersection<T> intersection(ray<T> const r, segment<T> const s, Policy const p = {}) {
		atl_assert(r.origin != r.through);
		ray_intersection<T> out;
		auto const as_single = [&](point<T> const q) {
			out.kind = intersection_kind::at_point;
			out.at = to_float64(q);
		};
		int const at_a = p.orientation(r.origin, r.through, s.a);
		int const at_b = p.orientation(r.origin, r.through, s.b);
		if (at_a == 0 && at_b == 0) {
			int const fa = detail::dot_sign(r.origin, r.through, r.origin, s.a, p), fb = detail::dot_sign(r.origin, r.through, r.origin, s.b, p);
			if (fa < 0 && fb < 0) return out;
			point<T> lo{}, hi{};
			if (fa >= 0 && fb >= 0) {
				lo = s.a;
				hi = s.b;
			} else { // one endpoint behind: the overlap starts at the origin
				lo = r.origin;
				hi = fa >= 0 ? s.a : s.b;
			}
			if (p.equal(lo, hi)) {
				as_single(lo);
				return out;
			}
			out.kind = intersection_kind::along_segment;
			out.along = lo < hi ? segment<T>{lo, hi} : segment<T>{hi, lo};
			return out;
		}
		if (!intersects(r, s, p)) return out;
		if (at_a == 0)
			as_single(s.a);
		else if (at_b == 0)
			as_single(s.b);
		else {
			out.kind = intersection_kind::at_point;
			out.at = intersection(line<T>{r.origin, r.through}, line<T>{s.a, s.b});
		}
		return out;
	}

	/** @brief Symmetric segment-ray intersection, with the same result fields and bounds. */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr ray_intersection<T> intersection(segment<T> const s, ray<T> const r, Policy const p = {}) {
		return intersection(r, s, p);
	}

	/**
	 * @brief The intersection of two rays: empty, one point, an exact segment
	 * (collinear facing rays: between the origins), or an exact ray
	 * (collinear same-direction rays: the later origin's ray).
	 */
	template<coordinate T, sign_policy<T> Policy = exact>
	[[nodiscard]] constexpr ray_intersection<T> intersection(ray<T> const a, ray<T> const b, Policy const p = {}) {
		atl_assert(a.origin != a.through && b.origin != b.through);
		ray_intersection<T> out;
		auto const as_single = [&](point<T> const q) {
			out.kind = intersection_kind::at_point;
			out.at = to_float64(q);
		};
		int const o1 = p.orientation(a.origin, a.through, b.origin);
		int const o2 = p.orientation(a.origin, a.through, b.through);
		if (o1 == 0 && o2 == 0) {
			// same direction iff dot(direction(a), direction(b)) > 0
			int const same = detail::dot_sign(a.origin, a.through, b.origin, b.through, p);
			if (same > 0) {
				if (contains(a, b.origin, p))
					out.beam = b;
				else if (contains(b, a.origin, p))
					out.beam = a;
				else
					return out; // An approximate policy may classify inconsistent collinearities.
				out.kind = intersection_kind::along_ray;
				return out;
			}
			if (!contains(a, b.origin, p)) return out; // opposite, backs turned
			if (p.equal(a.origin, b.origin)) {
				as_single(a.origin);
				return out;
			}
			out.kind = intersection_kind::along_segment;
			out.along = a.origin < b.origin ? segment<T>{a.origin, b.origin}
											: segment<T>{b.origin, a.origin};
			return out;
		}
		if (!intersects(a, b, p)) return out;
		if (p.cross_sign(a.origin, b.origin, b.origin, b.through) == 0)
			as_single(a.origin); // t == 0: crossing at a's origin, exact
		else if (p.cross_sign(a.origin, b.origin, a.origin, a.through) == 0)
			as_single(b.origin); // u == 0: crossing at b's origin, exact
		else {
			out.kind = intersection_kind::at_point;
			out.at = intersection(line<T>{a.origin, a.through}, line<T>{b.origin, b.through});
		}
		return out;
	}

	/**
	 * @brief Intersection points of the circle's BOUNDARY and the ray, 0 to
	 *        2, ordered along the ray. The COUNT is exact for integral
	 *        coordinates (quadratic root-counting on `[0, inf)`: the line
	 *        tangency, the origin's containment, and the vertex-parameter
	 *        sign); the points are `double` constructions.
	 * @pre As the circle-line relation for integral exactness.
	 *
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(circle<T> const c, ray<T> const r) {
		atl_assert(r.origin != r.through);
		static_vector<point<float64>, 2> out;
		line<T> const spanned{r.origin, r.through};
		auto const line_rel = relation(c, spanned);
		if (line_rel == tangency::disjoint) return out;
		exact const ex{};
		int const vertex_forward = detail::dot_sign(r.origin, r.through, r.origin, c.center, ex); // sign of t*
		std::size_t count = 0;
		if (line_rel == tangency::tangent)
			count = vertex_forward >= 0 ? 1 : 0;
		else {
			int const origin_gap = -int(contains(c, r.origin)); // sign of f(0)
			if (origin_gap < 0)
				count = 1; // origin inside: exactly the outgoing root
			else if (origin_gap == 0)
				count = 1 + std::size_t(vertex_forward > 0);
			else
				count = vertex_forward > 0 ? 2 : 0;
		}
		if (count == 0) return out;
		auto const on_line = intersection(c, spanned);
		// forward roots are the suffix of the ascending-parameter list
		for (std::size_t k = on_line.size() - count; k < on_line.size(); ++k)
			out.push_back(on_line[k]);
		return out;
	}

	/** @brief Symmetric ray-circle boundary intersection.
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] static_vector<point<float64>, 2> intersection(ray<T> const r, circle<T> const c) {
		return intersection(c, r);
	}

	/**
	 * @brief Distance between a ray and a line — exactly 0 when they meet
	 *        (EXACT predicate first); otherwise the ray is parallel to the
	 *        line or heading away, and both cases attain the gap at the
	 *        origin.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 *
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(ray<T> const r, line<T> const l) {
		if (intersects(r, l)) return 0.0;
		return distance(l, r.origin);
	}

	/** @brief Symmetric line-ray distance.
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(line<T> const l, ray<T> const r) {
		return distance(r, l);
	}

	/**
	 * @brief Distance between a ray and a CLOSED segment — exactly 0 when
	 *        they meet (EXACT predicate first); disjoint, the minimum is at
	 *        a segment endpoint or at the ray's origin (an interior-interior
	 *        minimum would force a crossing).
	 * @pre As the ray-line distance.
	 *
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(ray<T> const r, segment<T> const s) {
		if (intersects(r, s)) return 0.0;
		float64 best = distance(r, s.a);
		best = (std::min)(best, distance(r, s.b));
		return (std::min)(best, distance(s, r.origin));
	}

	/** @brief Symmetric segment-ray distance.
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(segment<T> const s, ray<T> const r) {
		return distance(r, s);
	}

	/**
	 * @brief Distance between two rays — exactly 0 when they meet (EXACT
	 *        predicate first); disjoint, the minimum involves one of the two
	 *        origins against the other ray.
	 * @pre As the ray-line distance.
	 *
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(ray<T> const a, ray<T> const b) {
		if (intersects(a, b)) return 0.0;
		return (std::min)(distance(a, b.origin), distance(b, a.origin));
	}

	/**
	 * @brief Distance from @p q to the closest point of the ray — a `double`
	 * CONSTRUCTION (projection clamped at the origin), exactly 0 for
	 * points on the ray.
	 * @pre For `int64` coordinates: `|x|, |y| < 2^62` (asserted).
	 *
	 * @note Not constexpr (uses runtime square root).
	 */
	template<coordinate T>
	[[nodiscard]] float64 distance(ray<T> const r, point<T> const q) {
		atl_assert(r.origin != r.through);
		if (contains(r, q)) return 0.0;
		if (detail::dot_sign(r.origin, r.through, r.origin, q, exact{}) < 0) return distance(r.origin, q);
		return distance(line<T>{r.origin, r.through}, q);
	}

	/** @} */
}
