#pragma once
#include <atl/ds/dsu.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

/**
 * @file
 * @brief The contender raced against @ref atl::geometry::manhattan_mst, whose
 *        octant sweeps keep their active points in an ordered map: a Fenwick
 *        prefix minimum over ranked x.
 */
namespace atl::experimental {
	/**
	 * @brief @ref atl::geometry::manhattan_mst with the same octant
	 *        candidate set, collected by a Fenwick prefix minimum over
	 *        rank-compressed x instead of an ordered map: per sweep the
	 *        points arrive by decreasing `y - x` (ties by decreasing x, so a
	 *        point on the octant's 45-degree boundary is already in the
	 *        tree), and each queries the smallest `x + y` among the arrived
	 *        points with `x' >= x`, its nearest neighbor in the closed octant.
	 *
	 * Lost the race (Xeon Platinum 8280, GCC 15.2, 2026-10-15, int64, plain
	 * `-O2` / `-march=native`): 182 / 178 ms against 174 / 175 for the
	 * library on 1e5 uniform points, 1134 / 1109 against 976 / 966 on 4e5
	 * uniform points over a 1e9 span, a tie on 1e5 points of a 300 by 300
	 * grid (131 / 128 against 129 / 129). It wins only a staircase with
	 * every x distinct, 117 / 114 against 128 / 125. The two extra sorts per
	 * sweep (the x ranks, then the arrival order) cost what the flat tree
	 * saves.
	 *
	 * Exactness, tie behaviour and edge order as @ref atl::geometry::manhattan_mst.
	 * @pre `1 <= size(points) < 2^32` (asserted). Floating coordinate
	 *      squares and pairwise products must be finite and normal unless a
	 *      factor is zero (squares asserted before sorting).
	 * @note Not constexpr (as the library function).
	 * @complexity O(n log n) time, O(n) memory
	 */
	template<geometry::vertex_range R>
	[[nodiscard]] std::vector<std::array<geometry::vertex_index, 2>> manhattan_mst_by_fenwick_prefix_minimum(R &&points) {
		using T = geometry::vertex_coordinate_t<R>;
		using S = geometry::product_t<T>; // sums and differences of coordinates, exact for integers
		using geometry::vertex_index;
		atl_assert(std::ranges::size(points) >= 1);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);
		for (uint32 i = 0; i < n; ++i) geometry::detail::check_floating_magnitude(geometry::point<T>(first[i]));

		struct candidate {
			S length;
			vertex_index a, b;
		};

		std::vector<candidate> candidates;
		candidates.reserve(std::size_t(4) * n);
		std::vector<S> xs(n), ys(n);
		std::vector<uint32> order(n), rank(n), tree_index(n);
		std::vector<S> tree_value(n);
		constexpr uint32 none = ~uint32(0);
		for (int sweep = 0; sweep < 4; ++sweep) {
			for (uint32 i = 0; i < n; ++i) {
				S sx = S(first[i].x), sy = S(first[i].y);
				if (sweep & 1) std::swap(sx, sy);
				if (sweep & 2) sx = S(-sx);
				xs[i] = sx;
				ys[i] = sy;
			}
			// dense ranks of x in DECREASING order: "x' >= x" is a prefix
			for (uint32 i = 0; i < n; ++i) order[i] = i;
			std::ranges::sort(order, [&](uint32 const a, uint32 const b) { return xs[a] > xs[b]; });
			uint32 distinct = 0;
			for (uint32 k = 0; k < n; ++k) {
				if (k > 0 && xs[order[k]] != xs[order[k - 1]]) ++distinct;
				rank[order[k]] = distinct;
			}
			++distinct;
			std::ranges::sort(order, [&](uint32 const a, uint32 const b) {
				S const ka = S(ys[a] - xs[a]), kb = S(ys[b] - xs[b]);
				return ka != kb ? ka > kb : xs[a] > xs[b];
			});
			std::ranges::fill_n(tree_index.begin(), distinct, none);
			for (uint32 const i : order) {
				S const sum = S(xs[i] + ys[i]);
				uint32 best = none;
				S best_value{};
				for (uint32 at = rank[i] + 1; at > 0; at &= at - 1)
					if (tree_index[at - 1] != none && (best == none || tree_value[at - 1] < best_value)) {
						best = tree_index[at - 1];
						best_value = tree_value[at - 1];
					}
				if (best != none) candidates.push_back({S(best_value - sum), i, best});
				for (uint32 at = rank[i] + 1; at <= distinct;) {
					if (tree_index[at - 1] == none || sum < tree_value[at - 1]) {
						tree_index[at - 1] = i;
						tree_value[at - 1] = sum;
					}
					uint32 const increment = at & (~at + 1);
					if (increment > distinct - at) break;
					at += increment;
				}
			}
		}
		std::ranges::sort(candidates, {}, &candidate::length);
		std::vector<std::array<vertex_index, 2>> out;
		out.reserve(n - 1);
		dsu components(n);
		for (auto const &e : candidates)
			if (components.unite(e.a, e.b)) out.push_back({e.a, e.b});
		atl_assert(out.size() + 1 == n);
		return out;
	}
}
