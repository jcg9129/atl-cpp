#pragma once
#include <atl/geometry/box.hpp>
#include <atl/geometry/point.hpp>
#include <atl/types/alias.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief The contender raced against @ref atl::geometry::union_area: the
 *        recursive top-down Klee tree.
 */
namespace atl::experimental {
	/**
	 * @brief @ref atl::geometry::union_area over a recursive top-down Klee
	 *        tree on one interleaved node array (4 times the strip count),
	 *        the covered length held in the unsigned difference type and
	 *        widened only at the area multiply. It differs from the library
	 *        in the tree update: a recursive descent that recomputes each
	 *        visited node on the way back, reading node lengths from the y
	 *        values, instead of the iterative bottom-up update with two
	 *        leaf-to-root re-pulls over a precomputed length array.
	 *
	 * Lost the race on every shape (Xeon Platinum 8358, GCC 15.2,
	 * 2026-10-15, plain `-O2` / `-march=native`): 2587 / 2561 ms against
	 * 2001 / 1995 ms for the library on 1e6 small int64 boxes, 2743 / 2716
	 * against 2145 / 2143 ms on 1e6 nested int64 boxes, 2206 / 2205 against
	 * 1603 / 1577 ms on 1e6 small int32 boxes. A top-down tree over two
	 * parallel arrays with widened lengths costs extra memory traffic.
	 * @pre as @ref atl::geometry::union_area
	 * @complexity O(n log n) time, O(n) memory; recursion depth
	 *             O(log n)
	 */
	template<geometry::box_range R>
	[[nodiscard]] constexpr geometry::difference_product_t<typename std::ranges::range_value_t<R>::coordinate_type>
	union_area_by_top_down_tree(R &&boxes) {
		using T = typename std::ranges::range_value_t<R>::coordinate_type;
		using W = geometry::difference_product_t<T>;
		using L = geometry::detail::klee_length_t<T>;
		if (std::ranges::size(boxes) == 0) return {};
		// plain references, not a structured binding: the update lambda reads
		// ys, and AppleClang rejects a captured binding in constant evaluation
		auto const sweep = geometry::detail::klee_events_of(boxes);
		auto const &ys = sweep.ys;
		auto const &events = sweep.events;
		if (events.empty()) return {};
		uint32 const slots = uint32(ys.size()) - 1;
		std::vector<geometry::detail::klee_node<T>> tree(4 * std::size_t(slots), {L{}, 0});
		auto const update = [&](auto &&self, uint32 const node, uint32 const lo, uint32 const hi, uint32 const l,
								uint32 const r, int const delta) -> void {
			if (r <= lo || hi <= l) return;
			if (l <= lo && hi <= r)
				tree[node].cover += delta;
			else {
				uint32 const mid = (lo + hi) / 2;
				self(self, 2 * node, lo, mid, l, r, delta);
				self(self, 2 * node + 1, mid, hi, l, r, delta);
			}
			if (tree[node].cover > 0)
				tree[node].covered = L(L(ys[hi]) - L(ys[lo])); // unsigned: wraps to the true difference
			else if (hi - lo == 1)
				tree[node].covered = L{};
			else
				tree[node].covered = L(tree[2 * node].covered + tree[2 * node + 1].covered);
		};
		W area = {};
		T prev_x = events.front().x;
		for (auto const &e : events) {
			area += W(tree[1].covered) * (W(e.x) - W(prev_x));
			prev_x = e.x;
			update(update, 1, 0, slots, e.y_lo, e.y_hi, e.delta);
		}
		return area;
	}
}
