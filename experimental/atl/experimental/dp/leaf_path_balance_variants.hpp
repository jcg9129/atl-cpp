#pragma once

/**
 * @file
 * @brief Contenders for @ref atl::experimental::balance_leaf_paths (the
 *        weighted slope trick over a tree).
 *
 * - @ref atl::experimental::balance_leaf_paths_by_ordered_map: each vertex's
 *   function as a `std::map` from position to weight with one lazy offset,
 *   children merged smaller-into-larger (O(n log^2 n) without a bound on
 *   the new costs), clips erasing from
 *   the two ends of the map: the textbook form, with direct access to both
 *   ends where the example keeps a min-heap and a max-heap per function over
 *   shared weight records. Same interface, same reconstruction, same ties.
 *
 * Race verdict (benchmarks/dp/leaf_path_balance.cpp, records of 2026-10-24,
 * Xeon Platinum 8280, GCC 15.2, ms, plain -O2 / -march=native; n = 2e5,
 * costs uniform in [1, 1e6]; free-cost weighted: weights uniform in [1, 1e6],
 * any new costs; nonnegative unit-weight: unit weights, new costs >= 0):
 *
 * | shape | example (heap forest) | by ordered map |
 * |---|---|---|
 * | free-cost weighted path | 5.9 / 5.9 | 7.4 / 7.9 |
 * | free-cost weighted star | 44.4 / 43.7 | 109 / 106 |
 * | free-cost weighted caterpillar | 11.5 / 12.1 | 16.0 / 16.4 |
 * | free-cost weighted random | 53.6 / 54.6 | 62.3 / 63.0 |
 * | nonnegative unit-weight path | 7.1 / 6.9 | 7.3 / 7.6 |
 * | nonnegative unit-weight star | 44.8 / 44.9 | 105 / 106 |
 * | nonnegative unit-weight caterpillar | 19.9 / 21.1 | 48.3 / 50.6 |
 * | nonnegative unit-weight random | 58.9 / 57.9 | 127 / 132 |
 *
 * The map loses every shape: 1.15x to 2.46x on free-cost weighted balancing
 * (it pays a logarithm per small-to-large reinsertion where the heaps meld in
 * O(1)) and 1.03x to 2.43x on nonnegative unit-weight balancing (the path is
 * the near tie). Its root eats from the left: eating the lighter end, which
 * the example does to avoid popping the stale entries of its min-heaps, gains
 * a map nothing. Kept as the differential partner of the tests.
 *
 * - @ref atl::experimental::balance_leaf_paths_by_fewer_entries: the
 *   example on the library's breakpoint pool with each bounded edge's segment
 *   inserted by @ref atl::experimental::insert_segment_by_fewer_entries
 *   (the side with fewer entries to the crossing moved, not the side of less
 *   weight), so the bounded path whose light side fragments into many pieces
 *   is no longer quadratic. Not measured yet; the race in
 *   benchmarks/dp/leaf_path_balance.cpp includes it.
 */

#include <atl/dp/convex_piecewise_linear.hpp>
#include <atl/dp/convex_piecewise_linear_forest.hpp>
#include <atl/experimental/dp/convex_piecewise_linear_forest_variants.hpp>
#include <atl/experimental/dp/leaf_path_balance.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace atl::experimental {
	namespace detail {
		/**
		 * @brief One vertex's function: breakpoints as stored position ->
		 *        weight (real position = stored + offset) and the end slopes.
		 *
		 * The stored positions and the offset are `int128`: melding swaps a
		 * child's map into its parent, offset included, and the parent's own
		 * breakpoints are then stored relative to that offset, which can be
		 * as far from them as two `int64` ranges (`H = INT64_MAX`, `parent =
		 * [0, 0, 0, 1, 1]`, `cost = [0, H, 1 - H, -H, 1 - H]`, `weight = [0,
		 * 2, 1, 1, 1]` stores `(1 - H) - H`). Real positions fit `int64`
		 * (documented), so they are narrowed when read.
		 */
		struct ordered_map_function {
			/** @brief Stored position -> weight; the real position is stored + `offset`. */
			std::map<int128, int64> breakpoints;
			/** @brief The lazy translation of every breakpoint. */
			int128 offset = 0;
			/** @brief The slopes at minus and plus infinity. */
			int64 leftmost = 0, rightmost = 0;

			/** @brief Add weight at a real position. */
			void add(int64 position, int64 weight) {
				if (weight != 0) breakpoints[int128(position) - offset] += weight;
			}

			/** @brief The real position of a stored one. */
			[[nodiscard]] int64 real(int128 const &stored) const { return int64(stored + offset); }

			/**
			 * @brief Eat `amount > 0` weight from the left (`is_from_left`) or
			 *        right end, appending the eaten real (position, weight)
			 *        pieces to `eaten`; returns the last position touched.
			 */
			int64 eat(bool is_from_left, int64 amount, std::vector<std::pair<int64, int64>> *eaten) {
				for (;;) {
					atl_assert(!breakpoints.empty());
					auto it = is_from_left ? breakpoints.begin() : std::prev(breakpoints.end());
					int64 const position = real(it->first);
					int64 const taken = std::min(it->second, amount);
					it->second -= taken;
					amount -= taken;
					if (eaten != nullptr) eaten->emplace_back(position, taken);
					if (it->second == 0) breakpoints.erase(it);
					if (amount == 0) return position;
				}
			}
		};
	}

	/**
	 * @brief @ref atl::experimental::balance_leaf_paths over `std::map`
	 *        functions merged smaller-into-larger; same contract and result.
	 * @param parent, cost, weight, min_cost As @ref atl::experimental::balance_leaf_paths.
	 * @pre as @ref atl::experimental::balance_leaf_paths (same asserts but the
	 *      pool's record limit: the maps hold no records)
	 * @complexity Without `min_cost`, O(n log^2 n). With `min_cost`, every
	 *      bounded edge also erases and reinserts the pieces it moves, the
	 *      `M` of @ref atl::experimental::balance_leaf_paths (the lighter side
	 *      by weight, as there), and the moved entries take part in later
	 *      merges: O((n + M) log^2 (n + M)) time and O(n + M) memory, with
	 *      `M` of order `n^2` on the bounded path of that contract.
	 * @note Not constexpr (`std::map` is not constexpr in C++23).
	 */
	[[nodiscard]] inline atl::experimental::leaf_path_balance balance_leaf_paths_by_ordered_map(std::span<uint32 const> parent,
																								std::span<int64 const> cost,
																								std::span<int64 const> weight,
																								std::optional<int64> min_cost = std::nullopt) {
		std::size_t const n = parent.size();
		detail::leaf_path_plan const plan = detail::plan_leaf_paths(parent, cost, weight, min_cost);
		std::vector<detail::ordered_map_function> function(n);
		std::vector<slope_clip_interval<int64>> range(n);
		std::vector<std::pair<int64, int64>> moved;
		bool const is_zero_bound = min_cost && *min_cost == 0;
		int64 const wall = is_zero_bound ? plan.max_weight : 2 * plan.weight_sum + 1;
		for (std::size_t i = n; i-- > 1;) {
			uint32 const v = plan.order[i];
			int64 const c = cost[v], d = weight[v];
			auto &f = function[v];
			if (plan.is_leaf(v)) {
				range[v] = {0, 0};
				if (min_cost) {
					f.add(*min_cost, wall - d);
					f.leftmost -= wall - d;
				}
				f.add(c, 2 * d);
				f.leftmost -= d;
				f.rightmost += d;
			} else if (min_cost) {
				if (d < f.rightmost) {
					range[v].upper = f.eat(false, f.rightmost - d, nullptr);
					f.rightmost = d;
				}
				int64 const length = c - *min_cost;
				if (-d <= f.leftmost) {
					f.offset += length;
				} else if (-d == f.rightmost) {
					range[v].lower = f.real(std::prev(f.breakpoints.end())->first);
				} else {
					moved.clear();
					if (f.rightmost + d <= -d - f.leftmost) {
						range[v].lower = f.eat(false, f.rightmost + d, &moved);
						for (auto &piece : moved) piece.first += length;
					} else {
						range[v].lower = f.eat(true, -d - f.leftmost, &moved);
						f.offset += length;
					}
					for (auto const &[position, w] : moved) f.add(position, w);
				}
				f.offset += *min_cost;
			} else {
				if (f.leftmost < -d) {
					range[v].lower = f.eat(true, -d - f.leftmost, nullptr);
					f.leftmost = -d;
				}
				if (d < f.rightmost) {
					range[v].upper = f.eat(false, f.rightmost - d, nullptr);
					f.rightmost = d;
				}
				f.offset += c;
			}
			auto &into = function[parent[v]];
			if (into.breakpoints.size() < f.breakpoints.size()) {
				std::swap(into.breakpoints, f.breakpoints);
				std::swap(into.offset, f.offset);
			}
			for (auto const &[stored, w] : f.breakpoints) into.add(f.real(stored), w);
			into.leftmost += f.leftmost;
			into.rightmost += f.rightmost;
			f = {};
		}

		int64 root_total = 0;
		if (n > 1) {
			auto &root = function[0];
			if (root.leftmost < 0)
				root_total = root.eat(true, -root.leftmost, nullptr);
			else if (root.rightmost > 0)
				root_total = root.eat(false, root.rightmost, nullptr);
		}
		return detail::settle_leaf_paths(parent, cost, weight, min_cost, plan, range, root_total);
	}

	/**
	 * @brief @ref atl::experimental::balance_leaf_paths with each bounded
	 *        edge's segment inserted by
	 *        @ref atl::experimental::insert_segment_by_fewer_entries: the
	 *        functions are sets of the library's breakpoint pool with their
	 *        end slopes beside them, and every operation but the segment is
	 *        the forest's. Same contract; the same total cost and path total
	 *        (the new costs may differ where a piece of slope exactly
	 *        `-weight[v]` makes two reconstructions optimal).
	 *
	 * @param parent, cost, weight, min_cost As @ref atl::experimental::balance_leaf_paths.
	 * @pre as @ref atl::experimental::balance_leaf_paths, the record limit
	 *      counting this run's records (`2n + M'` below)
	 * @complexity Without `min_cost`, as the example. With `min_cost`, each
	 *      bounded edge walks and re-pushes `m'_v = O(min(k_left, k_right) +
	 *      1)` pieces, `k` the live entries between each end of the function
	 *      and the crossing: O((n + M') log n) time and O(n + M') memory for
	 *      `M'` the sum of the `m'_v`, never more than the example's `M` plus
	 *      `O(n)`, and O(n) on the bounded path where the example's `M` is
	 *      of order `n^2`. Not measured yet.
	 */
	[[nodiscard]] constexpr atl::experimental::leaf_path_balance balance_leaf_paths_by_fewer_entries(
		std::span<uint32 const> parent, std::span<int64 const> cost, std::span<int64 const> weight,
		std::optional<int64> min_cost = std::nullopt) {
		/** @brief A function's slopes at minus and plus infinity. */
		struct end_slopes {
			int64 leftmost = 0, rightmost = 0;
		};

		std::size_t const n = parent.size();
		detail::leaf_path_plan const plan = detail::plan_leaf_paths(parent, cost, weight, min_cost);
		atl::detail::breakpoint_pool<int64, int64> pool(n);
		pool.reserve(n);
		std::vector<end_slopes> slopes(n);
		fewer_entries_scratch<int64, int64> scratch;
		std::vector<slope_clip_interval<int64>> range(n);
		auto const ignore = [](int64 const &, int64 const &) {};
		auto const push = [&pool](uint32 v, int64 position, int64 w) {
			if (w != 0) pool.push(v, position, w);
		};
		bool const is_zero_bound = min_cost && *min_cost == 0;
		int64 const wall = is_zero_bound ? plan.max_weight : 2 * plan.weight_sum + 1;
		for (std::size_t i = n; i-- > 1;) {
			uint32 const v = plan.order[i];
			int64 const c = cost[v], d = weight[v];
			end_slopes &s = slopes[v];
			if (plan.is_leaf(v)) {
				range[v] = {0, 0};
				if (min_cost) {
					s.leftmost -= wall - d;
					push(v, *min_cost, wall - d);
				}
				s.leftmost -= d;
				s.rightmost += d;
				push(v, c, 2 * d);
			} else if (min_cost) {
				// clip_slopes(leftmost, d), the segment, then translate(min_cost)
				if (d < s.rightmost) {
					range[v].upper = pool.template eat<false>(v, s.rightmost - d, ignore);
					s.rightmost = d;
				}
				range[v].lower = insert_segment_by_fewer_entries(pool, v, s.leftmost, s.rightmost, -d, c - *min_cost, scratch);
				pool.translate(v, *min_cost);
			} else {
				if (s.leftmost < -d) {
					range[v].lower = pool.template eat<true>(v, -d - s.leftmost, ignore);
					s.leftmost = -d;
				}
				if (d < s.rightmost) {
					range[v].upper = pool.template eat<false>(v, s.rightmost - d, ignore);
					s.rightmost = d;
				}
				pool.translate(v, c);
			}
			slopes[parent[v]].leftmost += s.leftmost;
			slopes[parent[v]].rightmost += s.rightmost;
			s = {};
			pool.meld(parent[v], v);
		}

		// the root as in the example: its lighter end eaten, else the near end read
		int64 root_total = 0;
		if (n > 1) {
			int64 const left = -slopes[0].leftmost, right = slopes[0].rightmost;
			if (left > 0 && left <= right) {
				root_total = pool.template eat<true>(0, left, ignore);
			} else if (left > 0) {
				if (right > 0) (void)pool.template eat<false>(0, right, ignore);
				if (pool.template has_live<false>(0)) root_total = pool.template top<false>(0).position;
			} else if (right > 0 && pool.template has_live<true>(0)) {
				root_total = pool.template top<true>(0).position;
			}
		}
		return detail::settle_leaf_paths(parent, cost, weight, min_cost, plan, range, root_total);
	}
}
