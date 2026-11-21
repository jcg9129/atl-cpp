#pragma once
#include <atl/ds/dsu.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief A minimum spanning tree of @p points under the MANHATTAN (L1)
	 *        distance `|dx| + |dy|`: `n - 1` edges as index pairs into
	 *        @p points, spanning every index (equal points join through
	 *        edges of length 0), in nondecreasing length order.
	 *
	 * Kruskal over at most `4 n` candidate edges: one sweep per octant of
	 * the upper half plane (the lower four follow by edge symmetry) gives
	 * every point an edge to a nearest neighbor in that octant, a set that
	 * contains some L1 minimum spanning tree.
	 *
	 * All sweep keys and lengths are sums and differences of coordinates in
	 * the widened @ref product_t, so for integral coordinates the tree is
	 * EXACTLY minimal over the full range of the type, with no magnitude
	 * bound. For floating coordinates the keys are rounded `double` values
	 * (exact for `float` coordinates whose magnitudes are close enough for
	 * their sums to fit 53 bits), so near-ties may be broken either way.
	 * Which of several minimum spanning trees is returned, and the order of
	 * the two indices of an edge, is unspecified.
	 * @pre `1 <= size(points) < 2^32` (asserted). Floating coordinate
	 *      squares and pairwise products must be finite and normal unless a
	 *      factor is zero (squares asserted before sorting).
	 * @note Not constexpr (`std::map`).
	 * @complexity O(n log n) time, O(n) memory
	 */
	template<vertex_range R>
	[[nodiscard]] std::vector<std::array<vertex_index, 2>> manhattan_mst(R &&points) {
		using T = vertex_coordinate_t<R>;
		using S = product_t<T>; // sums and differences of coordinates, exact for integers
		atl_assert(std::ranges::size(points) >= 1);
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);
		for (uint32 i = 0; i < n; ++i) detail::check_floating_magnitude(point<T>(first[i]));

		struct candidate {
			S length;
			vertex_index a, b;
		};

		std::vector<candidate> candidates;
		candidates.reserve(std::size_t(4) * n);
		std::vector<S> xs(n), ys(n);
		std::vector<vertex_index> order(n);
		for (int sweep = 0; sweep < 4; ++sweep) {
			// the four upper octants by swapping x and y and negating x; the
			// canonical octant is up and to the right, at least 45 degrees steep
			for (uint32 i = 0; i < n; ++i) {
				S sx = S(first[i].x), sy = S(first[i].y);
				if (sweep & 1) std::swap(sx, sy);
				if (sweep & 2) sx = S(-sx);
				xs[i] = sx;
				ys[i] = sy;
			}
			for (uint32 i = 0; i < n; ++i) order[i] = i;
			std::ranges::sort(order, [&](vertex_index const a, vertex_index const b) {
				return xs[a] + ys[a] < xs[b] + ys[b];
			});
			// keys by decreasing x; along them x - y strictly decreases, so
			// the scan for points in the new arrival's octant stops at the
			// first that is not
			std::map<S, vertex_index, std::greater<>> active;
			for (vertex_index const i : order) {
				auto it = active.lower_bound(xs[i]); // the greatest key <= x
				while (it != active.end()) {
					vertex_index const j = it->second;
					// i is in j's octant iff x_i >= x_j (the key bound) and
					// dy >= dx, that is x_j - y_j >= x_i - y_i
					if (xs[j] - ys[j] < xs[i] - ys[i]) break;
					candidates.push_back({S((xs[i] + ys[i]) - (xs[j] + ys[j])), j, i});
					// the newly matched i is at least as close to every later
					// arrival, so j leaves: each point enters and leaves once
					it = active.erase(it);
				}
				active.insert_or_assign(xs[i], i);
			}
		}

		std::ranges::sort(candidates, [](candidate const &a, candidate const &b) { return a.length < b.length; });
		std::vector<std::array<vertex_index, 2>> out;
		out.reserve(n - 1);
		dsu components(n);
		for (auto const &e : candidates)
			if (components.unite(e.a, e.b)) out.push_back({e.a, e.b});
		atl_assert(out.size() + 1 == n);
		return out;
	}
}
