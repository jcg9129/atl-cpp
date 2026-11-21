#pragma once

/**
 * @file
 * @brief Contenders for the exactly-k partition DP (@ref
 *        atl::optimal_partition, adaptive per layer between divide and
 *        conquer and Knuth's layered bound).
 *
 * Shared output contract: @ref atl::optimal_partition's curve, element `k -
 * 1` the best cost of exactly `k` segments, so every contender must agree
 * with the library exactly.
 *
 * - @ref atl::experimental::optimal_partition_by_divide_and_conquer -- every
 *   layer, all rows `k .. n`, by the monotone divide and conquer.
 * - @ref atl::experimental::optimal_partition_by_knuth_bound -- every layer by
 *   Knuth's bound `opt_{k-1}[i] <= opt_k[i] <= opt_k[i+1]`, O(n^2) in all.
 * - @ref atl::experimental::optimal_partition_by_floored_divide_and_conquer --
 *   the library's adaptive solver with its divide-and-conquer rows also
 *   floored at the previous layer's argmin.
 * - @ref atl::experimental::optimal_partition_by_smawk (opt-in only): every
 *   layer but the last by SMAWK, O(n) evaluations per layer, at the price
 *   of evaluating the cost out of order.
 *
 * Race verdicts (benchmarks/dp/optimal_partition.cpp, Xeon Platinum 8280,
 * ms, plain -O2 / -march=native; cost of `[j, i)` over prefix sums S:
 * "cheap" = (S_i - S_j)^2, "pow1.5" = |S_i - S_j - L|^1.5).
 *
 * Knuth's bound and divide and conquer on every layer ("optimal_partition
 * race", 2026-10-23; "-" = not run, past ~2e9 evaluations):
 *
 * | shape | library | by Knuth bound | by divide and conquer |
 * |---|---|---|---|
 * | 5e3 K=2 cheap | 0.012 / 0.011 | 0.012 / 0.011 | 0.111 / 0.112 |
 * | 5e3 K=20 cheap | 2.0 / 2.0 | 14.2 / 10.8 | 2.1 / 2.2 |
 * | 5e3 K=20 pow1.5 | 3.1 / 3.1 | 26.6 / 26.7 | 3.2 / 3.1 |
 * | 5e3 K=100 cheap | 6.6 / 6.4 | 18.9 / 15.2 | 10.4 / 10.7 |
 * | 5e3 K=500 pow1.5 | 33.9 / 36.4 | 59.0 / 58.9 | 70.3 / 69.8 |
 * | 5e3 K=n cheap | 85.3 / 79.6 | 100 / 82.3 | 211 / 219 |
 * | 5e3 K=n pow1.5 | 116 / 134 | 148 / 142 | 318 / 314 |
 * | 2e4 K=20 cheap | 8.7 / 9.0 | 216 / 163 | 9.2 / 9.4 |
 * | 2e4 K=100 pow1.5 | 49.5 / 51.4 | 454 / 459 | 72.1 / 72.1 |
 * | 2e4 K=2000 cheap | 388 / 375 | 605 / 506 | 822 / 839 |
 * | 2e4 K=n cheap | 1544 / 1488 | 1784 / 1544 | - |
 * | 2e4 K=n pow1.5 | 2056 / 2288 | 2605 / 2580 | - |
 *
 * The adaptive solver is the library's: it wins or ties every shape of
 * that race (n = 1e3, 5e3, 2e4; K = 2, 20, 100, n/10, n; both costs). The
 * last-layer single row alone makes K = 2 about ten times cheaper than divide
 * and conquer.
 *
 * The floored divide and conquer and SMAWK ("optimal_partition layer
 * engines", 2026-11-01, same shapes): the floor ties the library within
 * 0.92-1.16x on every shape, with no consistent direction (n = 2e4, K = n,
 * cheap: 1473 / 1460 against 1538 / 1521; n = 2e4, K = 20, cheap: 8.8 /
 * 8.9 against 8.7 / 7.7); SMAWK ties at K = 2, where both run the last row
 * alone, and is 1.35-5.6x slower from K = 20 up (n = 2e4, K = 2000, cheap:
 * 1344 / 1474 against 385 / 384).
 *
 * Every contender forms and compares its candidate totals in the library's
 * wide type (`atl::detail::dp_total_t`), as the library does, so they agree
 * wherever a candidate overflows the cost's type but the `dp` values fit.
 */

#include <atl/dp/concepts.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/dp/optimal_partition.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace atl::experimental {
	/**
	 * @brief @ref atl::optimal_partition by the divide-and-conquer
	 *        optimization alone: each layer `k` is the row minima of the
	 *        staircase matrix over all rows `k .. n`, the last layer
	 *        included.
	 *
	 * It lost to the adaptive solver, or tied within noise at K = 20, on
	 * every shape of the race (2026-10-23, Xeon Platinum 8280; see the
	 * file's table): 1.6-2.6x slower from K = 100 up (n = 2e4, K = 2000,
	 * cheap: 822 / 839 vs 388 / 375 ms), and 10x at K = 2. Its O(n log n)
	 * per layer never drops, while the adaptive solver's later layers fall
	 * to Knuth windows of about `n / k^2` per row, and its last layer is one
	 * row.
	 *
	 * @pre the leftmost best `j` of each layer is nondecreasing in `i` (the
	 *      quadrangle inequality implies it)
	 * @complexity O(segments * n log n) evaluations of `cost`, O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && atl::detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition_by_divide_and_conquer(std::size_t n, std::size_t segments,
																									Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(atl::detail::fits_index(n) && atl::detail::fits_index(n + 1));
		atl_assert(segments <= n);
		std::vector<value_type> best;
		best.reserve(segments);
		if (segments == 0) return best;
		uint32 const last = uint32(n);
		std::vector<value_type> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= last; ++i) previous[i] = cost(0, i);
		best.push_back(previous[last]);
		using total = atl::detail::dp_total_t<value_type, Compare>;
		atl::detail::total_order<value_type, Compare> const better{comp};
		auto const below = [](uint32 i) { return i - 1; };
		for (uint32 k = 2; k <= segments; ++k) {
			auto layer = [&](uint32 i, uint32 j) -> total { return total(total(previous[j]) + total(cost(j, i))); };
			auto store = [&current](uint32 i, uint32, total &&value) { current[i] = value_type(std::move(value)); };
			atl::detail::monotone_minima_recurse(k, last + 1, k - 1, last - 1, layer, better, below, store);
			std::swap(previous, current);
			best.push_back(previous[last]);
		}
		return best;
	}

	/**
	 * @brief @ref atl::optimal_partition by Knuth's layered bound alone:
	 *        every layer's rows descend, each scanning `j` over `[opt_{k-1}[i],
	 *        opt_k[i+1]]`; two rolling `uint32` argmin rows.
	 *
	 * The windows telescope along each diagonal `i - k` to at most `n`, so
	 * all layers together cost about `(n^2 - segments^2) / 2 + segments * n`
	 * evaluations; but on balanced costs layer `k` alone costs about `n^2 /
	 * (2 k^2)`, so any `segments >= 3` pays about `n^2 / 4` for layer 2.
	 * Lost on every shape of the race except K = 2, where both run the last
	 * layer alone (2026-10-23, Xeon Platinum 8280; see the file's table).
	 * At K = 20 it is 5.4-25x slower (n = 2e4, cheap: 216 / 163 vs 8.7 /
	 * 9.0 ms). Even at K = n it is up to 1.28x slower than the
	 * adaptive solver, tying only at n = 1e3 with -march=native (n = 2e4,
	 * cheap: 1784 / 1544 vs 1544 / 1488 ms),
	 * which runs its wide early layers by divide and conquer.
	 *
	 * @pre the quadrangle inequality (as @ref atl::optimal_partition)
	 * @complexity O(n^2 + segments * n) evaluations of `cost`, O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && atl::detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition_by_knuth_bound(std::size_t n, std::size_t segments, Cost &&cost,
																							 Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(atl::detail::fits_index(n) && atl::detail::fits_index(n + 1));
		atl_assert(segments <= n);
		std::vector<value_type> best;
		best.reserve(segments);
		if (segments == 0) return best;
		uint32 const last = uint32(n);
		std::vector<value_type> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= last; ++i) previous[i] = cost(0, i);
		best.push_back(previous[last]);
		std::vector<uint32> argmins(2 * (n + 1));
		auto argmin_row = [&argmins, row = n + 1](uint32 k) { return argmins.data() + (k & 1) * row; };
		auto after_layer = [&best, last](std::vector<value_type> const &layer) { best.push_back(layer[last]); };
		auto const always_knuth = [](uint64) { return std::numeric_limits<uint64>::max(); };
		atl::detail::partition_layers(last, uint32(segments), true, cost, comp, previous, current, argmin_row,
									  after_layer, always_knuth);
		return best;
	}

	namespace detail {
		/**
		 * @brief `atl::detail::monotone_minima_recurse` with a per-row
		 *        floor as well: row `i` scans `[max(col_lo, first(i)),
		 *        min(col_hi, last(i))]`, the upper end raised to the lower so
		 *        a cost breaking the precondition still gets a non-empty scan.
		 *
		 * @pre the leftmost best columns are nondecreasing over the rows, row
		 *      `i`'s at least `first(i)`; `first(i) <= last(i)` and `last(i)
		 *      >= col_lo` on every row
		 * @complexity O((rows + columns) log rows) evaluations, fewer when the
		 *             floors cut into the column ranges
		 */
		template<typename Cost, typename Compare, atl::detail::column_bound First, atl::detail::column_bound Last,
				 atl::detail::minimum_sink<cost_value_t<Cost>> Store>
			requires matrix_cost<Cost, Compare>
		constexpr void floored_minima_recurse(uint32 row_lo, uint32 row_hi, uint32 col_lo, uint32 col_hi, Cost &cost,
											  Compare const &comp, First const &first, Last const &last, Store &store) {
			using Value = cost_value_t<Cost>;
			while (row_lo < row_hi) {
				uint32 const mid = row_lo + (row_hi - row_lo) / 2;
				uint32 const lo = (std::max)(col_lo, uint32(first(mid)));
				uint32 const hi = (std::max)(lo, (std::min)(col_hi, uint32(last(mid))));
				uint32 best = lo;
				Value best_value = cost(mid, lo);
				for (uint32 c = lo + 1; c <= hi; ++c) {
					Value v = cost(mid, c);
					if (std::invoke(comp, v, best_value)) {
						best_value = std::move(v);
						best = c;
					}
				}
				store(mid, best, std::move(best_value));
				if (mid - row_lo < row_hi - mid - 1) {
					floored_minima_recurse(row_lo, mid, col_lo, best, cost, comp, first, last, store);
					row_lo = mid + 1;
					col_lo = best;
				} else {
					floored_minima_recurse(mid + 1, row_hi, best, col_hi, cost, comp, first, last, store);
					row_hi = mid;
					col_hi = best;
				}
			}
		}

		/**
		 * @brief `atl::detail::partition_divide_layer` with each row `i`
		 *        floored at the lower end of its `atl::detail::knuth_window`,
		 *        `max(opt_{k-1}[i], k - 1)` (`below` as there).
		 */
		template<typename Cost, typename Compare>
			requires additive_cost<Cost, Compare>
		constexpr void floored_divide_layer(uint32 k, uint32 stop, uint32 col_hi, uint32 const *below, uint32 below_top,
											cost_value_t<Cost> const *previous, cost_value_t<Cost> *current,
											uint32 *argmin, Cost &cost, Compare const &comp) {
			using value_type = cost_value_t<Cost>;
			using total = atl::detail::dp_total_t<value_type, Compare>;
			atl::detail::total_order<value_type, Compare> const better{comp};
			auto layer = [&](uint32 i, uint32 j) -> total { return total(total(previous[j]) + total(cost(j, i))); };
			auto store = [current, argmin](uint32 i, uint32 j, total &&value) {
				current[i] = value_type(std::move(value));
				argmin[i] = j;
			};
			auto const first = [k, below, below_top](uint32 i) {
				return below ? (std::max)(below[(std::min)(i, below_top)], k - 1) : k - 1;
			};
			auto const last = [](uint32 i) { return i - 1; };
			floored_minima_recurse(k, stop, k - 1, col_hi, layer, better, first, last, store);
		}

		/**
		 * @brief `atl::detail::partition_layers` for the whole curve, its
		 *        divide-and-conquer layers by `floored_divide_layer`; the
		 *        engine choice, budgets and Knuth layers are the library's.
		 */
		template<typename Cost, typename Compare, atl::detail::partition_layer_choice Choose>
			requires additive_cost<Cost, Compare>
		constexpr void floored_partition_layers(uint32 n, uint32 segments, Cost &cost, Compare const &comp,
												std::vector<cost_value_t<Cost>> &previous,
												std::vector<cost_value_t<Cost>> &current, std::vector<uint32> &argmins,
												std::vector<cost_value_t<Cost>> &best, Choose const &choose) {
			constexpr uint64 unlimited = std::numeric_limits<uint64>::max();
			auto const budget_of = [&](uint32 k) {
				uint64 const rows = n - k + 1;
				return uint64(choose(rows * uint64(std::bit_width(rows))));
			};
			uint32 const *below = nullptr;
			uint32 below_top = n;
			uint64 knuth_estimate = unlimited, budget = segments > 2 ? budget_of(2) : 0;
			for (uint32 k = 2; k <= segments; ++k) {
				uint32 *const argmin = argmins.data() + (k & 1) * (n + 1);
				uint32 const top = n, first = k == segments ? n : k;
				bool is_whole_knuth = false;
				if (first == top || knuth_estimate <= budget) {
					auto const outcome = atl::detail::partition_knuth_layer(
						k, first, top, below, below_top, previous.data(), current.data(), argmin, cost, comp,
						first == top ? unlimited : budget);
					knuth_estimate = outcome.evaluations;
					is_whole_knuth = outcome.stop == first;
					if (!is_whole_knuth)
						floored_divide_layer(k, outcome.stop, argmin[outcome.stop], below, below_top, previous.data(),
											 current.data(), argmin, cost, comp);
				} else {
					floored_divide_layer(k, top + 1, top - 1, below, below_top, previous.data(), current.data(),
										 argmin, cost, comp);
				}
				if (k + 1 < segments) {
					budget = budget_of(k + 1);
					if (!is_whole_knuth) {
						knuth_estimate = 0;
						for (uint32 i = top; i >= k && knuth_estimate <= budget; --i) {
							atl::detail::knuth_window const window(k, i, below, below_top,
																   i == top ? top : argmin[i + 1]);
							knuth_estimate += window.hi - window.lo + 1;
						}
					}
				}
				std::swap(previous, current);
				best.push_back(previous[n]);
				below = argmin;
				below_top = top;
			}
		}
	}

	/**
	 * @brief @ref atl::optimal_partition with every divide-and-conquer row
	 *        floored at the previous layer's argmin, `opt_{k-1}[i] <=
	 *        opt_k[i]`.
	 *
	 * The library's divide and conquer only narrows a row's columns by its
	 * neighbours in the same layer, `[opt_k[lo], opt_k[hi]]`; the floor
	 * from the layer below is one load per row (the argmins are already
	 * kept for Knuth's windows) and cuts the left end of every scan. Knuth
	 * layers, the switch rule and the budgets are the library's, so the
	 * evaluation order stays two-pointer friendly.
	 *
	 * Not adopted: it tied the library within 0.92-1.16x on every shape of
	 * the race (2026-11-01, Xeon Platinum 8280; see the file's notes), with
	 * no consistent direction, so the floor does not pay for its extra code.
	 *
	 * @pre as @ref atl::optimal_partition (the quadrangle inequality)
	 * @complexity as @ref atl::optimal_partition, O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && atl::detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition_by_floored_divide_and_conquer(std::size_t n, std::size_t segments,
																											Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(atl::detail::fits_index(n) && atl::detail::fits_index(n + 1));
		atl_assert(segments <= n);
		std::vector<value_type> best;
		best.reserve(segments);
		if (segments == 0) return best;
		uint32 const last = uint32(n);
		std::vector<value_type> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= last; ++i) previous[i] = cost(0, i);
		best.push_back(previous[last]);
		std::vector<uint32> argmins(2 * (n + 1));
		detail::floored_partition_layers(last, uint32(segments), cost, comp, previous, current, argmins, best,
										 atl::detail::adaptive_partition_choice{});
		return best;
	}

	/**
	 * @brief @ref atl::optimal_partition with every layer but the last by
	 *        SMAWK: O(n) evaluations per layer, O(segments * n) in all
	 *        (OPT-IN only).
	 *
	 * Layer `k` is the row minima of the staircase matrix (row `i` in `[k,
	 * n]`, column `j` in `[k - 1, i - 1]`). The entries right of the
	 * staircase (`j >= i`) are never evaluated: SMAWK compares them as a
	 * value worse than every real one (a flag next to the value, so no
	 * infinity is formed), and a Monge matrix padded that way on its upper
	 * right is still totally monotone, which is all SMAWK needs. The last
	 * layer only needs `dp_k[n]`: one Knuth window, as in the library.
	 *
	 * Why opt-in, never the default: SMAWK evaluates the entries in no
	 * locality-friendly order (REDUCE walks the columns against a stack of
	 * rows, then the recursion halves the rows), so the library's guarantee
	 * that a two-pointer cost (a sliding window moved to each asked
	 * segment) moves O(evaluations) steps is lost; such a cost may move
	 * O(n) per evaluation here. Use it only with an O(1) random-access cost.
	 * Even then it lost the race 1.35-5.6x from K = 20 up (2026-11-01, Xeon
	 * Platinum 8280, n = 2e4, K = 2000, cheap: 1344 / 1474 vs 385 / 384 ms):
	 * its fewer evaluations do not make up for SMAWK's constant factor.
	 *
	 * Breaking the precondition never forms an undefined entry nor an
	 * empty scan: every value returned is a real partition's cost, just
	 * not necessarily the best (tested).
	 *
	 * @pre the quadrangle inequality, as @ref atl::optimal_partition (not
	 *      checkable; documented); SMAWK needs the strict form of total
	 *      monotonicity it implies
	 * @complexity O(segments * n) evaluations of `cost`, O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && atl::detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition_by_smawk(std::size_t n, std::size_t segments,
																					   Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		using total = atl::detail::dp_total_t<value_type, Compare>;

		/** @brief A staircase entry: `value` is meaningful only when `is_defined`. */
		struct entry {
			total value{};
			bool is_defined = false;
		};

		atl_assert(atl::detail::fits_index(n) && atl::detail::fits_index(n + 1));
		atl_assert(segments <= n);
		std::vector<value_type> best;
		best.reserve(segments);
		if (segments == 0) return best;
		uint32 const last = uint32(n);
		std::vector<value_type> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= last; ++i) previous[i] = cost(0, i);
		best.push_back(previous[last]);
		std::vector<uint32> argmins(2 * (n + 1));
		std::vector<entry> values;
		atl::detail::total_order<value_type, Compare> const total_better{comp};
		auto const better = [&total_better](entry const &a, entry const &b) {
			return a.is_defined && (!b.is_defined || total_better(a.value, b.value));
		};
		for (uint32 k = 2; k <= uint32(segments); ++k) {
			uint32 *const argmin = argmins.data() + (k & 1) * (n + 1);
			uint32 const *const below = k == 2 ? nullptr : argmins.data() + ((k - 1) & 1) * (n + 1);
			if (k == uint32(segments)) {
				atl::detail::partition_knuth_layer(k, last, last, below, last, previous.data(), current.data(), argmin,
												   cost, comp, std::numeric_limits<uint64>::max());
			} else {
				uint32 const rows = last - k + 1;
				auto eval = [&, k](uint32 row, uint32 j) -> entry {
					uint32 const i = k + row;
					if (j >= i) return {};
					return {total(total(previous[j]) + total(cost(j, i))), true};
				};
				std::vector<uint32> columns(rows);
				for (uint32 c = 0; c < rows; ++c) columns[c] = k - 1 + c;
				values.resize(rows);
				atl::detail::smawk<decltype(eval), decltype(better)> solver(eval, better, argmin + k, values.data());
				solver.run(rows, std::move(columns));
				for (uint32 row = 0; row < rows; ++row) current[k + row] = value_type(std::move(values[row].value));
			}
			std::swap(previous, current);
			best.push_back(previous[last]);
		}
		return best;
	}
}
