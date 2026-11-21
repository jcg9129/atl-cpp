#pragma once

/**
 * @file
 * @brief @ref atl::experimental::tree_transport_cost, a worked example of the
 *        dual slope trick (experimental, not part of the library): the least
 *        cost of moving units along the edges of a tree, `cost[v]` per unit
 *        across the edge above `v`, so that every vertex ends with at least
 *        its demand. The dual of @ref atl::experimental::balance_leaf_paths,
 *        solved on (slope, length) segments in @ref atl::meldable_heap_forest.
 */

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace atl::experimental {
	using atl::int128;
	using atl::int64;
	using atl::uint32;

	/**
	 * @brief A piece of a convex function of the flow `k` across an edge: a
	 *        run of `length` units at one slope, stored as the heap key: the
	 *        slope itself on the right side of `k = 0`, minus the slope on
	 *        the left side, so both sides are min-heaps and `+= c` is the
	 *        edge's `c|k|`.
	 *
	 * The key is `int128`: a reverse segment's key is minus a key of the
	 * other side, and every edge above adds its cost again, so keys reach
	 * about twice the sum of the costs and a sum of two of them (a swap's
	 * gain) about four times, past `int64` although the answer fits.
	 */
	struct transport_segment {
		/** @brief The slope (right side) or minus the slope (left side). */
		int128 key = 0;
		/** @brief The units the piece covers, positive. */
		int64 length = 0;

		/** @brief The lazy add of an edge: `key += shift`, which is `+ shift * |k|` on either side. */
		constexpr transport_segment &operator+=(int64 shift) {
			key += shift;
			return *this;
		}
	};

	/** @brief The heap order on segments: `top` is the least key. */
	struct transport_segment_order {
		/** @brief Whether `a` ranks below `b` (`a.key > b.key`). */
		[[nodiscard]] constexpr bool operator()(transport_segment const &a, transport_segment const &b) const { return b.key < a.key; }
	};

	/**
	 * @brief A heap forest @ref atl::experimental::tree_transport_cost_on can
	 *        run on: the interface of @ref atl::meldable_heap_forest over
	 *        @ref atl::experimental::transport_segment with an `int64` lazy add
	 *        (the experimental heap contenders share it). `top` may return a
	 *        proxy: it is read only through its conversion to
	 *        `transport_segment const &`.
	 */
	template<typename HeapForest>
	concept transport_heap_forest =
		std::constructible_from<HeapForest, std::size_t, transport_segment_order, int64>
		&& requires(HeapForest &forest, uint32 heap, transport_segment segment, int64 shift) {
			   forest.reserve(heap);
			   forest.push(heap, segment);
			   { forest.top(heap) } -> std::convertible_to<transport_segment const &>;
			   { forest.pop(heap) } -> std::convertible_to<transport_segment>;
			   { forest.empty(heap) } -> std::convertible_to<bool>;
			   forest.meld(heap, heap);
			   forest.add_to_all(heap, shift);
		   };

	/** @brief The engine of @ref atl::experimental::tree_transport_cost: the library's pairing heaps with a lazy add. */
	using transport_heap_forest_default =
		atl::meldable_heap_forest<transport_segment, transport_segment_order, int64>;

	/** @brief The checked inputs of a transport run. */
	struct transport_plan {
		/** @brief Every vertex, children before parents, the root last. */
		std::vector<uint32> order;
		/** @brief The price of a unit of unmet demand, `sum of costs + 1` (fits: the costs sum below `INT64_MAX`). */
		int64 penalty = 0;
	};

	/**
	 * @brief Check the inputs of @ref atl::experimental::tree_transport_cost
	 *        (every asserted precondition) and order the vertices children
	 *        first, root last (Kahn's order, no recursion); shared with the
	 *        experimental contenders.
	 * @param parent, cost, supply, demand As @ref atl::experimental::tree_transport_cost.
	 * @pre as @ref atl::experimental::tree_transport_cost (asserted)
	 * @complexity O(n)
	 */
	[[nodiscard]] constexpr transport_plan plan_transport(std::span<uint32 const> parent, std::span<int64 const> cost,
														  std::span<int64 const> supply, std::span<int64 const> demand) {
		std::size_t const n = parent.size();
		atl_assert(n >= 1 && cost.size() == n && supply.size() == n && demand.size() == n);
		atl_assert(n < std::numeric_limits<uint32>::max() / 2);

		int64 const max_value = std::numeric_limits<int64>::max();
		int64 cost_sum = 0, supply_sum = 0, demand_sum = 0;
		std::vector<uint32> pending(n, 0); // children not yet processed
		for (std::size_t v = 0; v < n; ++v) {
			atl_assert(supply[v] >= 0 && supply[v] <= max_value - supply_sum);
			atl_assert(demand[v] >= 0 && demand[v] <= max_value - demand_sum);
			supply_sum += supply[v];
			demand_sum += demand[v];
			if (v == 0) continue;
			atl_assert(parent[v] < n);
			atl_assert(cost[v] >= 0 && cost[v] <= max_value - 1 - cost_sum);
			cost_sum += cost[v];
			++pending[parent[v]];
		}
		atl_assert(supply_sum >= demand_sum);
		// a unit of unmet demand is priced above every path, so an optimum meets all of it
		int64 const penalty = cost_sum + 1;

		// children before parents: Kahn's order from the leaves up
		std::vector<uint32> order;
		order.reserve(n);
		for (std::size_t v = 1; v < n; ++v)
			if (pending[v] == 0) order.push_back(uint32(v));
		for (std::size_t head = 0; head < order.size(); ++head) {
			uint32 const p = parent[order[head]];
			if (--pending[p] == 0 && p != 0) order.push_back(p);
		}
		atl_assert(order.size() == n - 1); // a vertex on a cycle never runs out of pending children
		order.push_back(0);
		return {std::move(order), penalty};
	}

	/**
	 * @brief @ref atl::experimental::tree_transport_cost on the heap forest
	 *        `HeapForest` (the race's entry point; the result does not depend
	 *        on the engine).
	 * @tparam HeapForest An @ref atl::experimental::transport_heap_forest.
	 * @param parent, cost, supply, demand As @ref atl::experimental::tree_transport_cost.
	 * @pre as @ref atl::experimental::tree_transport_cost (asserted)
	 * @complexity as @ref atl::experimental::tree_transport_cost, with the engine's heap bounds
	 */
	template<transport_heap_forest HeapForest>
	[[nodiscard]] constexpr int64 tree_transport_cost_on(std::span<uint32 const> parent, std::span<int64 const> cost,
														 std::span<int64 const> supply, std::span<int64 const> demand) {
		auto const [order, penalty] = plan_transport(parent, cost, supply, demand);
		std::size_t const n = parent.size();

		// heap 2v: the right side of f_v (k >= 0), keyed by slope; heap 2v + 1:
		// the left side (k <= 0), keyed by minus the slope
		HeapForest forest(2 * n, transport_segment_order{}, int64{0});
		forest.reserve(2 * n);
		// the sum of f_v(0) over the subtrees processed so far: at most `penalty * sum
		// demand` < 2^126 and never negative (every cost is), so int128 holds it
		int128 total = 0;
		auto const top_key = [&forest](uint32 heap) -> int128 {
			return static_cast<transport_segment const &>(forest.top(heap)).key;
		};

		// one swap of a re-centering: the two keys it paired and the units moved
		struct swap_record {
			int128 l_key, r_key;
			int64 moved;
		};

		std::vector<swap_record> reverse;
		for (uint32 const v : order) {
			uint32 const right = 2 * v, left = right + 1; // n < 2^31 (asserted): the heap ids fit
			int64 const surplus = supply[v] - demand[v];
			if (surplus > 0) forest.push(right, {0, surplus});
			if (surplus < 0) {
				forest.push(left, {-penalty, -surplus});
				total += int128(penalty) * -surplus;
			}
			// the children's sides were melded in: swap units across k = 0 while a
			// left slope beats a right one, each swap lowering f_v(0) by its gain.
			// The segment being eaten on each side is held out of its heap (lengths
			// are not keys, but a heap top is read-only), and the swapped units'
			// reverse segments are pushed after the loop: a reverse key is at least
			// the opposite side's current top, so none of them can swap here (and
			// at the root they are never needed)
			transport_segment l{}, r{};
			bool has_l = false, has_r = false;
			for (;;) {
				if ((!has_l && forest.empty(left)) || (!has_r && forest.empty(right))) break;
				int128 const gain = (has_l ? l.key : top_key(left)) + (has_r ? r.key : top_key(right));
				if (!(gain < int128(0))) break;
				if (!has_l) l = forest.pop(left);
				if (!has_r) r = forest.pop(right);
				int64 const moved = l.length < r.length ? l.length : r.length;
				total += int128(moved) * gain;
				if (v != 0) reverse.push_back({l.key, r.key, moved});
				l.length -= moved;
				r.length -= moved;
				has_l = l.length != 0;
				has_r = r.length != 0;
			}
			if (v == 0) break;
			if (has_l) forest.push(left, l);
			if (has_r) forest.push(right, r);
			for (auto const &[l_key, r_key, moved] : reverse) {
				forest.push(left, {-r_key, moved});
				forest.push(right, {-l_key, moved});
			}
			reverse.clear();
			forest.add_to_all(right, cost[v]);
			forest.add_to_all(left, cost[v]);
			forest.meld(2 * parent[v], right);
			forest.meld(2 * parent[v] + 1, left);
		}
		atl_assert(!(int128(std::numeric_limits<int64>::max()) < total)); // the answer fits int64
		return int64(total);
	}

	/**
	 * @brief The least total cost of moving units along a tree's edges, at
	 *        `cost[v]` per unit across the edge between `v` and `parent[v]`
	 *        (either way), so that every vertex `v`, starting with
	 *        `supply[v]` units, ends with at least `demand[v]`.
	 *
	 * The tree is rooted at vertex 0 and given by `parent[v]` for `v >= 1`
	 * (`parent[0]` and `cost[0]` are ignored). Bottom-up, `f_v(k)` is the
	 * least cost inside `v`'s subtree when `k` units leave it upward (`k <
	 * 0`: enter it): convex in `k`. Children combine with `v`'s own `0 if k
	 * <= supply - demand` by INFIMAL CONVOLUTION (the flows add), and the
	 * edge above `v` ADDS `cost[v] * |k|`: the dual of
	 * @ref atl::experimental::balance_leaf_paths, where children add and an edge is an infimal
	 * convolution. An infimal convolution of convex functions merges their
	 * pieces sorted by slope, so `f_v` is kept as (slope, length) segments
	 * on each side of `k = 0` in two min-heaps (the left one keyed by minus
	 * the slope), melded up the tree; `+ cost * |k|` is then one lazy add of
	 * `cost` to both heaps, and re-centering the merged function at `k = 0`
	 * swaps units across while the best left slope beats the best right
	 * one, which lowers `f_v(0)` by the slope difference per unit. Unmet
	 * demand is priced `sum of costs + 1` per unit, above any path, so the
	 * domain stays the whole line and the optimum meets every demand.
	 *
	 * The order is Kahn's, children before parents, and the loop over it is
	 * flat: no recursion, so a path of 10^6 vertices needs no stack depth.
	 *
	 * @param parent `parent[v] < n` for `v >= 1`, forming a tree rooted at 0.
	 * @param cost The cost per unit across the edge above `v`, `cost[0]` ignored.
	 * @param supply The units vertex `v` starts with.
	 * @param demand The units vertex `v` must end with, at least.
	 * Keys, reverse records, gains and the running total are `int128`, so
	 * no intermediate overflows: a reverse key is minus an opposite key,
	 * every edge above adds its cost to it again, and the running total
	 * starts at the penalty times the unmet demand. Only the answer has to
	 * fit `int64` (asserted).
	 *
	 * @pre the four spans have one size `n >= 1` and describe a tree (a
	 *      cycle or an unreachable vertex is caught); costs, supplies and
	 *      demands nonnegative with sums that fit int64, `sum supply >= sum
	 *      demand`, and the least total cost fits int64 (all asserted, the
	 *      last on the result)
	 * @complexity O(n + s log n) time, `s` the number of swaps (no bound
	 *      better than the number of augmentations is claimed), O(n + s)
	 *      memory
	 */
	[[nodiscard]] constexpr int64 tree_transport_cost(std::span<uint32 const> parent, std::span<int64 const> cost,
													  std::span<int64 const> supply, std::span<int64 const> demand) {
		return tree_transport_cost_on<transport_heap_forest_default>(parent, cost, supply, demand);
	}
}
