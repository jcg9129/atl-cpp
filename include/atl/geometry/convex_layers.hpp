#pragma once
#include <atl/geometry/convex_hull.hpp>
#include <atl/geometry/point.hpp>
#include <atl/geometry/predicates.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <ranges>
#include <utility>
#include <vector>

namespace atl::geometry {
	/**
	 * @brief CONVEX LAYERS of @p points (onion peeling): layer 0 is every
	 *        point on the BOUNDARY of the convex hull of the whole set
	 *        (collinear boundary points included — the boundary-of-the-set
	 *        semantics), layer 1 the boundary of what remains, and so on
	 *        until nothing is left. Every input index appears in exactly
	 *        one layer; points sharing a position (exact `==`, whatever the
	 *        policy) peel together.
	 *
	 * Each layer walks CCW from its lexicographically smallest point (the
	 * `convex_hull` convention); indices at a shared position appear
	 * consecutively at that vertex, smallest first, so a layer with
	 * duplicate positions repeats those positions in the walk. A fully
	 * collinear remainder is one final layer in lexicographic order (its
	 * boundary is the whole segment). Exactness follows the sign policy —
	 * integral and `double` coordinates are exact under @ref atl::geometry::exact.
	 *
	 * The peeler sorts and deduplicates ONCE, then re-runs the monotone
	 * chain over the surviving sorted subsequence per layer: O(n log n +
	 * sum of layer-suffix sizes) = O(n log n + n L) for L layers. Random
	 * clouds have L ~ n^(2/3); adversarial nested rings reach L = n/3 —
	 * see the benchmark for both regimes.
	 *
	 * @pre Coordinate bounds as @ref atl::geometry::convex_hull (int64: `|x|, |y| <
	 *      2^62` for exact cross products).
	 * @param points Input point cloud; not modified.
	 * @param p Predicate policy; duplicates always use raw equality.
	 * @return Layers of original indices, outermost first, partitioning the input.
	 * @pre Fewer than 2^32 input points (asserted).
	 * @complexity O(n log n + n L) time, O(n) auxiliary memory, no recursion.
	 */
	template<vertex_range R, sign_policy<vertex_coordinate_t<R>> Policy = exact>
	[[nodiscard]] constexpr std::vector<std::vector<vertex_index>> convex_layers(R &&points, Policy const p = {}) {
		using T = vertex_coordinate_t<R>;
		atl_assert(std::ranges::size(points) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(points));
		auto const first = std::ranges::begin(points);
		std::vector<detail::hull_entry<T>> all;
		all.reserve(n);
		for (uint32 i = 0; i < n; ++i) {
			point<T> const q = first[i];
			detail::check_difference_bounds(q);
			all.push_back({q, i});
		}
		std::ranges::sort(all, [](detail::hull_entry<T> const &a, detail::hull_entry<T> const &b) {
			return a.position != b.position ? a.position < b.position : a.index < b.index;
		});

		// one entry per distinct position ("slot"); a slot remembers its run
		// of coincident input points, which all peel with it. Raw == matches
		// the exact sort (a policy's equal may not be transitive)
		std::vector<detail::hull_entry<T>> alive;
		std::vector<std::pair<uint32, uint32>> span;
		for (uint32 i = 0; i < n;) {
			uint32 j = i;
			while (j < n && all[j].position == all[i].position) ++j;
			alive.push_back({all[i].position, uint32(span.size())});
			span.emplace_back(i, j);
			i = j;
		}

		std::vector<std::vector<vertex_index>> layers;
		std::vector<char> peeled(span.size(), 0);
		std::vector<detail::hull_entry<T>> next;
		while (!alive.empty()) {
			auto &layer = layers.emplace_back();
			for (vertex_index const s : detail::chain_hull<false>(alive, p)) {
				// a non-transitive policy (approx) can walk a near-coincident
				// slot twice; peeling it once keeps the output a partition
				if (peeled[s]) continue;
				peeled[s] = 1;
				for (uint32 k = span[s].first; k < span[s].second; ++k)
					layer.push_back(all[k].index);
			}
			next.clear();
			for (auto const &e : alive)
				if (!peeled[e.index]) next.push_back(e);
			alive.swap(next);
		}
		return layers;
	}
}
