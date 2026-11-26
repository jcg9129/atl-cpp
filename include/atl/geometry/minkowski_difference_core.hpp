#pragma once

/** @file
 * @brief Minkowski differences and their supporting hull vertices without caliper views.
 */
#include <atl/geometry/convex_hull_core.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/polygon_core.hpp>
#include <atl/geometry/predicates_core.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <vector>

namespace atl::geometry {
	namespace detail {
		/**
		 * @brief The Minkowski difference `A - B` of two hulled point sets: the
		 * boundary vertices from the parallel dual sweep, each with the
		 * supporting vertex indices into `a_hull` / `b_hull` (the hull of
		 * the NEGATED second set) that produced it.
		 */
		template<coordinate T>
		struct minkowski_difference {
			std::vector<point<T>> a_hull, negated_b_hull;
			std::vector<point<T>> boundary;
			std::vector<std::array<vertex_index, 2>> supports;
		};

		/**
		 * @brief Hulls both point sets (the second negated) and walks the
		 *        parallel dual sweep to produce their Minkowski difference.
		 * @pre Both inputs non-empty (asserted).
		 */
		template<vertex_range RA, vertex_range RB, sign_policy<vertex_coordinate_t<RA>> Policy>
			requires std::same_as<vertex_coordinate_t<RA>, vertex_coordinate_t<RB>>
		[[nodiscard]] minkowski_difference<vertex_coordinate_t<RA>> difference_of(RA &&a_points, RB &&b_points,
																				  Policy const p) {
			using T = vertex_coordinate_t<RA>;
			atl_assert(std::ranges::size(a_points) >= 1 && std::ranges::size(b_points) >= 1);
			if constexpr (std::signed_integral<T> && sizeof(T) == 8) {
				[[maybe_unused]] auto const fits = [](point<T> const q) {
					constexpr T bound = T(1) << 61;
					return -bound < q.x && q.x < bound && -bound < q.y && q.y < bound;
				};
				atl_assert(std::ranges::all_of(a_points, fits) && std::ranges::all_of(b_points, fits));
			}
			minkowski_difference<T> d;
			d.a_hull = convex_hull(a_points, p);
			std::vector<point<T>> b_negated;
			for (auto const q : std::ranges::subrange(std::ranges::begin(b_points),
													  std::ranges::end(b_points)))
				b_negated.push_back(-point<T>(q));
			d.negated_b_hull = convex_hull(b_negated, p);
			auto const &a = d.a_hull;
			auto const &b = d.negated_b_hull;
			atl_assert(a.size() < (uint64(1) << 32) && b.size() < (uint64(1) << 32));
			atl_assert(a.size() < 3 || [&] {
				auto const c = convexity_of(a, p);
				return c.is_convex && c.is_strict && c.direction == winding::counterclockwise;
			}());
			atl_assert(b.size() < 3 || [&] {
				auto const c = convexity_of(b, p);
				return c.is_convex && c.is_strict && c.direction == winding::counterclockwise;
			}());
			atl_assert(a.size() != 2 || a[0] != a[1]);
			atl_assert(b.size() != 2 || b[0] != b[1]);

			auto const start = [](auto const &hull) {
				vertex_index best = 0;
				for (vertex_index i = 1; i < hull.size(); ++i)
					if (hull[i].y != hull[best].y ? hull[i].y < hull[best].y : hull[i].x < hull[best].x) best = i;
				return best;
			};
			auto const next = [](vertex_index const i, std::size_t const n) { return i + 1 == n ? 0 : i + 1; };
			vertex_index i = start(a), j = start(b);
			std::size_t ai = 0, bi = 0;
			std::size_t const an = a.size() < 2 ? 0 : a.size(), bn = b.size() < 2 ? 0 : b.size();
			while (ai < an || bi < bn) {
				d.boundary.push_back(a[i] + b[j]);
				d.supports.push_back({i, j});
				// Separate equal directions, with A first, just as the support sweep.
				if (ai < an && (bi == bn || !sweep_before(b[j], b[next(j, b.size())], a[i], a[next(i, a.size())], p))) {
					i = next(i, a.size());
					++ai;
				} else {
					j = next(j, b.size());
					++bi;
				}
			}
			if (d.boundary.empty()) { // point vs point
				d.boundary.push_back(d.a_hull[0] + d.negated_b_hull[0]);
				d.supports.push_back({0, 0});
			}
			return d;
		}
	}
}
