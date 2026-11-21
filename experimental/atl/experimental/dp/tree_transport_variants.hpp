#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::experimental::tree_transport_cost (the dual
 *        slope trick over a tree).
 *
 * - @ref atl::experimental::tree_transport_cost_by_skew,
 *   @ref atl::experimental::tree_transport_cost_by_rank_leftist,
 *   @ref atl::experimental::tree_transport_cost_by_binary: the example's loop
 *   on the other engines of the meldable heap race (iterative skew heaps,
 *   rank-leftist heaps, `std::vector` binary heaps melded smaller-into-larger
 *   with one offset per heap).
 * - @ref atl::experimental::tree_transport_cost_by_pop_and_push: the loop
 *   without the example's two savings: each swap pops both tops, pushes the
 *   longer one's rest back and pushes the two reverse segments at once, also
 *   at the root.
 * - @ref atl::experimental::tree_transport_cost_by_length_records: that loop
 *   with the segment lengths in a side array, so a partial swap shortens the
 *   longer top in place instead of popping it and pushing the rest back.
 *
 * Race verdict (benchmarks/dp/tree_transport.cpp, records of 2026-10-24, Xeon
 * Platinum 8280, GCC 15.2, ms, plain -O2 / -march=native; n = 2.5e5, costs
 * uniform in [1, 1e6]; mixed = supply and demand uniform in [0, 7] on every
 * vertex, split = supplies on the first half, demands on the second):
 *
 * | instance | example (pairing) | by pop and push | by length records | by skew | by rank leftist | by binary |
 * |---|---|---|---|---|---|---|
 * | mixed path | 41.6 / 44.7 | 47.2 / 43.6 | 72.3 / 72.7 | 37.1 / 39.7 | 59.4 / 59.5 | 108 / 108 |
 * | mixed star | 74.3 / 77.6 | 107 / 103 | 129 / 130 | 77.5 / 80.0 | 99.8 / 135 | 51.6 / 52.0 |
 * | mixed caterpillar | 37.7 / 39.5 | 44.2 / 41.7 | 73.1 / 73.3 | 37.9 / 40.1 | 58.7 / 59.1 | 97.0 / 96.9 |
 * | mixed random | 69.9 / 71.6 | 71.6 / 69.8 | 85.9 / 90.0 | 62.0 / 66.3 | 77.6 / 88.1 | 153 / 153 |
 * | split path | 16.9 / 18.2 | 18.7 / 17.5 | 17.6 / 17.8 | 12.7 / 14.0 | 17.8 / 17.8 | 48.5 / 48.9 |
 * | split star | 71.6 / 75.2 | 104 / 102 | 134 / 130 | 75.2 / 75.4 | 99.2 / 131 | 50.3 / 49.6 |
 * | split caterpillar | 18.0 / 18.9 | 20.1 / 18.9 | 19.5 / 20.0 | 17.9 / 19.0 | 22.2 / 22.6 | 52.2 / 52.1 |
 * | split random | 70.4 / 69.9 | 71.4 / 71.7 | 87.2 / 89.6 | 62.9 / 67.5 | 76.2 / 84.0 | 153 / 153 |
 *
 * - by pop and push: the example beats it 1.44x / 1.33x on the mixed star,
 *   1.45x / 1.36x on the split star and 1.01x to 1.17x elsewhere under plain
 *   -O2 (0.96x to 1.06x with -march=native).
 * - by length records: 1.04x to 1.94x slower than the example except on the
 *   native split path (17.8 against 18.2); the indirection to the side array
 *   costs more than the saved pops.
 * - by skew: NOT a clean loss. Faster than pairing on the paths (1.12x to
 *   1.33x) and the random trees (1.04x to 1.13x), within 2% on the
 *   caterpillars, 1.00x to 1.05x slower on the stars. The example keeps the
 *   library's pairing heaps (skew lost the meldable heap race); switching
 *   remains an open choice.
 * - by binary: wins the stars 1.42x to 1.52x (each child brings one segment
 *   into a contiguous array) and loses every other shape 2.14x to 2.90x.
 * - by rank leftist: slower everywhere except the native split path (17.8
 *   against 18.2).
 *
 * Every contender keeps keys, gains and the running total in `int128`, as
 * the example does, and returns the same `int64` answer.
 */

#include <atl/ds/meldable_heap_forest.hpp>
#include <atl/experimental/dp/tree_transport.hpp>
#include <atl/experimental/ds/meldable_heap_variants.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace atl::experimental {
	/** @brief @ref atl::experimental::tree_transport_cost on iterative rank-leftist heaps; same contract. */
	[[nodiscard]] constexpr int64 tree_transport_cost_by_rank_leftist(std::span<uint32 const> parent, std::span<int64 const> cost,
																	  std::span<int64 const> supply, std::span<int64 const> demand) {
		return tree_transport_cost_on<meldable_heap_forest_by_rank_leftist<transport_segment, transport_segment_order, int64>>(
			parent, cost, supply, demand);
	}

	/** @brief @ref atl::experimental::tree_transport_cost on iterative skew heaps; same contract. */
	[[nodiscard]] constexpr int64 tree_transport_cost_by_skew(std::span<uint32 const> parent, std::span<int64 const> cost,
															  std::span<int64 const> supply, std::span<int64 const> demand) {
		return tree_transport_cost_on<meldable_heap_forest_by_skew<transport_segment, transport_segment_order, int64>>(
			parent, cost, supply, demand);
	}

	/** @brief @ref atl::experimental::tree_transport_cost on `std::vector` binary heaps melded smaller-into-larger; same contract. */
	[[nodiscard]] constexpr int64 tree_transport_cost_by_binary(std::span<uint32 const> parent, std::span<int64 const> cost,
																std::span<int64 const> supply, std::span<int64 const> demand) {
		return tree_transport_cost_on<meldable_heap_forest_by_binary<transport_segment, transport_segment_order, int64>>(
			parent, cost, supply, demand);
	}

	namespace detail {
		/** @brief A heap entry of `tree_transport_cost_by_length_records`: the key and the record holding its length. */
		struct transport_entry {
			int128 key = 0;
			uint32 record = 0;

			constexpr transport_entry &operator+=(int64 shift) {
				key += shift;
				return *this;
			}
		};

		/** @brief The heap order on entries: `top` is the least key. */
		struct transport_entry_order {
			constexpr bool operator()(transport_entry const &a, transport_entry const &b) const { return b.key < a.key; }
		};
	}

	/**
	 * @brief @ref atl::experimental::tree_transport_cost where each swap pops
	 *        both tops, pushes the longer one's rest back and pushes the two
	 *        reverse segments at once (also at the root); same contract.
	 */
	[[nodiscard]] constexpr int64 tree_transport_cost_by_pop_and_push(std::span<uint32 const> parent, std::span<int64 const> cost,
																	  std::span<int64 const> supply, std::span<int64 const> demand) {
		auto const [order, penalty] = atl::experimental::plan_transport(parent, cost, supply, demand);
		std::size_t const n = parent.size();
		atl::experimental::transport_heap_forest_default forest(2 * n, atl::experimental::transport_segment_order{}, int64{0});
		forest.reserve(2 * n);
		int128 total = 0;
		for (uint32 const v : order) {
			uint32 const right = 2 * v, left = right + 1; // n < 2^31 (asserted): the heap ids fit
			int64 const surplus = supply[v] - demand[v];
			if (surplus > 0) forest.push(right, {0, surplus});
			if (surplus < 0) {
				forest.push(left, {-penalty, -surplus});
				total += int128(penalty) * -surplus;
			}
			while (!forest.empty(left) && !forest.empty(right)) {
				atl::experimental::transport_segment const l = forest.top(left), r = forest.top(right);
				int128 const gain = l.key + r.key;
				if (!(gain < int128(0))) break;
				forest.pop(left);
				forest.pop(right);
				int64 const moved = l.length < r.length ? l.length : r.length;
				total += int128(moved) * gain;
				if (l.length > moved) forest.push(left, {l.key, l.length - moved});
				if (r.length > moved) forest.push(right, {r.key, r.length - moved});
				forest.push(left, {-r.key, moved});
				forest.push(right, {-l.key, moved});
			}
			if (v == 0) break;
			forest.add_to_all(right, cost[v]);
			forest.add_to_all(left, cost[v]);
			forest.meld(2 * parent[v], right);
			forest.meld(2 * parent[v] + 1, left);
		}
		atl_assert(!(int128(std::numeric_limits<int64>::max()) < total));
		return int64(total);
	}

	/**
	 * @brief @ref atl::experimental::tree_transport_cost_by_pop_and_push with
	 *        the segment lengths in a side array: a partial swap shortens the
	 *        longer top in place through its record (one pop and two pushes per
	 *        swap instead of two pops and three pushes); same contract.
	 */
	[[nodiscard]] constexpr int64 tree_transport_cost_by_length_records(std::span<uint32 const> parent, std::span<int64 const> cost,
																		std::span<int64 const> supply, std::span<int64 const> demand) {
		auto const [order, penalty] = atl::experimental::plan_transport(parent, cost, supply, demand);
		std::size_t const n = parent.size();
		atl::meldable_heap_forest<detail::transport_entry, detail::transport_entry_order, int64> forest(
			2 * n, detail::transport_entry_order{}, int64{0});
		forest.reserve(2 * n);
		std::vector<int64> length;
		length.reserve(2 * n);
		auto const push = [&](uint32 heap, int128 key, int64 units) {
			atl_assert(length.size() < std::numeric_limits<uint32>::max());
			forest.push(heap, {key, uint32(length.size())});
			length.push_back(units);
		};
		int128 total = 0;
		for (uint32 const v : order) {
			uint32 const right = 2 * v, left = right + 1; // n < 2^31 (asserted): the heap ids fit
			int64 const surplus = supply[v] - demand[v];
			if (surplus > 0) push(right, 0, surplus);
			if (surplus < 0) {
				push(left, -penalty, -surplus);
				total += int128(penalty) * -surplus;
			}
			while (!forest.empty(left) && !forest.empty(right)) {
				detail::transport_entry const l = forest.top(left), r = forest.top(right);
				int128 const gain = l.key + r.key;
				if (!(gain < int128(0))) break;
				int64 &l_length = length[l.record], &r_length = length[r.record];
				int64 const moved = l_length < r_length ? l_length : r_length;
				total += int128(moved) * gain;
				l_length -= moved;
				r_length -= moved;
				if (l_length == 0) forest.pop(left);
				if (r_length == 0) forest.pop(right);
				push(left, -r.key, moved);
				push(right, -l.key, moved);
			}
			if (v == 0) break;
			forest.add_to_all(right, cost[v]);
			forest.add_to_all(left, cost[v]);
			forest.meld(2 * parent[v], right);
			forest.meld(2 * parent[v] + 1, left);
		}
		atl_assert(!(int128(std::numeric_limits<int64>::max()) < total));
		return int64(total);
	}
}
