#pragma once

/** @file
 * @brief Minkowski-based overlap and boundary contact without caliper views.
 */
#include <atl/geometry/line_core.hpp>
#include <atl/geometry/minkowski_difference_core.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/geometry/segment_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <vector>

namespace atl::geometry::detail {
	/**
	 * @brief Recover the contact set of two hulls whose interiors do not overlap
	 * but whose distance is zero. The Minkowski-difference feature
	 * through the origin gives the contact normal; each hull's EXACT
	 * extreme face along it (one edge or one vertex — strict hulls) is
	 * scanned directly, and the contact is the faces' overlap.
	 */
	template<coordinate T, sign_policy<T> Policy>
	[[nodiscard]] segment_intersection<T> boundary_contact(minkowski_difference<T> const &d, Policy const p) {
		point<T> const origin{};
		// the feature direction: a boundary edge through the origin, or
		// any incident edge when the origin is a vertex
		point<T> direction{};
		std::size_t const k = d.boundary.size();
		if (k == 1) { // point-vs-point difference: vertex contact
			segment_intersection<T> out;
			out.kind = intersection_kind::at_point;
			out.at = to_float64(d.a_hull[0]);
			return out;
		}
		for (std::size_t i = 0, j = k - 1; i < k; j = i++) {
			segment<T> const edge{d.boundary[j], d.boundary[i]};
			if (edge.a != edge.b && contains(edge, origin, p)) {
				if constexpr (std::signed_integral<T>) {
					// A supporting input edge is parallel to this difference edge.
					// Its magnitude stays below 2^62 (the predicate coordinate bound),
					// even when the difference edge approaches 2^63.
					auto const from = d.supports[j], to = d.supports[i];
					direction = from[0] != to[0] ? d.a_hull[to[0]] - d.a_hull[from[0]]
												 : d.negated_b_hull[to[1]] - d.negated_b_hull[from[1]];
				} else {
					// Floating coordinates keep the difference edge itself: a tolerant
					// policy's ties depend on the direction's length, and the shorter input
					// edge would tie faces the difference edge separates.
					direction = edge.b - edge.a;
				}
				break;
			}
		}
		atl_assert(direction.x != T(0) || direction.y != T(0));
		point<T> const outward{direction.y, T(-direction.x)}; // CCW: right side
		// the exact extreme face of a hull along `u`: an edge or vertex
		auto const face_along = [&](std::vector<point<T>> const &hull, point<T> const u) {
			point<T> const anchor{};
			std::size_t best = 0;
			for (std::size_t v = 1; v < hull.size(); ++v)
				if (-p.cross_sign(perp(anchor), perp(u), hull[best], hull[v]) > 0) best = v;
			std::size_t other = best; // a strict hull's face has <= 2 vertices
			for (std::size_t v = 0; v < hull.size(); ++v)
				if (v != best
					&& -p.cross_sign(perp(anchor), perp(u), hull[best], hull[v]) == 0)
					other = v;
			return segment<T>{hull[best], hull[other]};
		};
		segment<T> const a_face = face_along(d.a_hull, outward);
		segment<T> const negated_b_face = face_along(d.negated_b_hull, outward);
		segment<T> const b_face{-negated_b_face.a, -negated_b_face.b};
		return intersection(a_face, b_face, p);
	}

	/**
	 * @brief Classify the hulls' overlap through the Minkowski
	 * difference: outside = disjoint, boundary = zero-measure contact,
	 * inside = interiors properly overlap. Integral decisions are exact under
	 * exact/fast; floating differences and non-exact policies are construction-grade.
	 */
	template<coordinate T, sign_policy<T> Policy>
	[[nodiscard]] containment overlap_of(minkowski_difference<T> const &d, Policy const p) {
		point<T> const origin{};
		if (d.boundary.size() >= 3) return contains(d.boundary, origin, p);
		return contains(segment<T>{d.boundary.front(), d.boundary.back()}, origin, p)
				 ? containment::boundary
				 : containment::outside;
	}

}
