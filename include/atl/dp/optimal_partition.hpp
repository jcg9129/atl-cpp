#pragma once

/**
 * @file
 * @brief @ref atl::optimal_partition -- cut `[0, n)` into exactly `k` segments for
 *        every `k` up to a bound, solving the layered DP `dp_k[i] = min_j
 *        dp_{k-1}[j] + w(j, i)` one layer at a time by divide and conquer or
 *        by Knuth's layered bound, whichever the previous layer says is
 *        cheaper.
 */

#include <atl/dp/concepts.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief Knuth's window for row `i` of layer `k`: the leftmost best
		 *        `j` lies in `[lo, hi]`, from `opt_{k-1}[i] <= opt_k[i] <=
		 *        opt_k[i+1]`.
		 *
		 * `below` is layer `k-1`'s argmin row (null for `k = 2`, whose
		 * argmins are all 0), filled up to row `below_top`; past it the last
		 * filled argmin is still a lower bound, as argmins rise with `i`.
		 * `upper` is `opt_k[i+1]`, or anything `>= i - 1` on the layer's top
		 * row. `hi` is raised to `lo` so a cost breaking the precondition
		 * still gets a real partition, never an empty scan.
		 */
		struct knuth_window {
			uint32 lo, hi;

			constexpr knuth_window(uint32 k, uint32 i, uint32 const *below, uint32 below_top, uint32 upper)
				: lo(below ? (std::max)(below[(std::min)(i, below_top)], k - 1) : k - 1),
				  hi((std::max)(lo, (std::min)(upper, i - 1))) {}
		};

		/** @brief What `partition_knuth_layer` did: rows `first .. stop - 1` are left over. */
		struct knuth_layer_outcome {
			uint32 stop;
			uint64 evaluations;
		};

		/**
		 * @brief Rows `top` down to `first` of layer `k` by Knuth's layered
		 *        bound: row `i` scans `j` upward over its `knuth_window`,
		 *        keeping the leftmost best. Stops after the row that takes
		 *        the evaluations past `budget`, leaving the rows below it.
		 */
		template<typename Cost, typename Compare>
			requires additive_cost<Cost, Compare>
		constexpr knuth_layer_outcome partition_knuth_layer(uint32 k, uint32 first, uint32 top, uint32 const *below,
															uint32 below_top, cost_value_t<Cost> const *previous,
															cost_value_t<Cost> *current, uint32 *argmin, Cost &cost,
															Compare const &comp, uint64 budget) {
			using value_type = cost_value_t<Cost>;
			using total = dp_total_t<value_type, Compare>;
			total_order<value_type, Compare> const better{comp};
			uint64 evaluations = 0;
			// the previous row's argmin stays in a register: reloading it from
			// `argmin` would chain every row behind a store-to-load forward
			uint32 upper = top;
			for (uint32 i = top; i >= first; --i) {
				knuth_window const window(k, i, below, below_top, upper);
				evaluations += window.hi - window.lo + 1;
				uint32 best = window.lo;
				auto best_value = total(total(previous[best]) + total(cost(best, i)));
				for (uint32 j = window.lo + 1; j <= window.hi; ++j) {
					auto v = total(total(previous[j]) + total(cost(j, i)));
					if (better(v, best_value)) {
						best_value = std::move(v);
						best = j;
					}
				}
				current[i] = value_type(std::move(best_value));
				argmin[i] = upper = best;
				if (evaluations > budget) return {i, evaluations};
			}
			return {first, evaluations};
		}

		/**
		 * @brief Rows `k .. stop - 1` of layer `k`, their argmins at most
		 *        `col_hi`, as the row minima of the staircase matrix (row
		 *        `i`, column `j < i`) by `monotone_minima_recurse`.
		 */
		template<typename Cost, typename Compare>
			requires additive_cost<Cost, Compare>
		constexpr void partition_divide_layer(uint32 k, uint32 stop, uint32 col_hi, cost_value_t<Cost> const *previous,
											  cost_value_t<Cost> *current, uint32 *argmin, Cost &cost,
											  Compare const &comp) {
			using value_type = cost_value_t<Cost>;
			using total = dp_total_t<value_type, Compare>;
			total_order<value_type, Compare> const better{comp};
			auto layer = [&](uint32 i, uint32 j) -> total { return total(total(previous[j]) + total(cost(j, i))); };
			auto store = [current, argmin](uint32 i, uint32 j, total &&value) {
				current[i] = value_type(std::move(value));
				argmin[i] = j;
			};
			auto const below = [](uint32 i) { return i - 1; };
			monotone_minima_recurse(k, stop, k - 1, col_hi, layer, better, below, store);
		}

		/**
		 * @brief The engine choice for one layer: `choose(divide_estimate)`
		 *        is the evaluation budget for Knuth's bound on a layer whose
		 *        divide and conquer would cost about `divide_estimate =
		 *        rows * bit_width(rows)` (an upper-bound estimate: the real
		 *        recursion evaluates somewhat less).
		 *        Knuth's bound runs when the previous layer's windows total
		 *        at most the budget: 0 never picks it, UINT64_MAX always does.
		 */
		template<typename Choose>
		concept partition_layer_choice = std::regular_invocable<Choose const &, uint64>
									  && std::convertible_to<std::invoke_result_t<Choose const &, uint64>, uint64>;

		/**
		 * @brief The adaptive choice @ref atl::optimal_partition uses: a
		 *        budget of `numerator / denominator` times the
		 *        divide-and-conquer estimate.
		 *
		 * Measured on x86-64 (benchmarks/dp/optimal_partition.cpp, "switch
		 * calibration", Xeon Platinum 8280): 1 and 2 tie, while 1/2 and 1/4
		 * switch too late. On
		 * balanced costs layer `k`'s windows total about `n^2 / (2 k^2)`, so
		 * the previous layer's total overestimates and the switch comes at
		 * most a layer late; the budget caps the layer where it does not.
		 */
		struct adaptive_partition_choice {
			uint64 numerator = 1, denominator = 1;

			[[nodiscard]] constexpr uint64 operator()(uint64 divide_estimate) const {
				return divide_estimate * numerator / denominator;
			}
		};

		/**
		 * @brief Layers `2 .. segments` of the partition DP over `previous`
		 *        (layer 1 on entry, the last layer on exit; `current` is
		 *        scratch of the same size `n + 1`).
		 *
		 * Layer `k` fills rows `k .. top`: `top = n` when every layer's
		 * `dp_k[n]` is wanted (`is_whole_curve`), else `n - (segments - k)`,
		 * as the remaining layers need one item each. The last layer fills
		 * row `n` alone, a single Knuth window. Every other layer goes to
		 * the engine `choose` picks; a Knuth layer that runs over its budget
		 * hands its remaining lower rows to divide and conquer, capped by
		 * the argmin of the lowest row it finished. Whichever runs, the
		 * layer's argmins land in `argmin_row(k)`, the next layer's lower
		 * bound.
		 *
		 * @param argmin_row `argmin_row(k)` for `k >= 2`: a pointer whose
		 *        slots `k .. top` are valid, disjoint from those of
		 *        `argmin_row(k - 1)`.
		 * @param after_layer called with `previous` once each layer is in it.
		 */
		template<typename Cost, typename Compare, typename ArgminRow, typename AfterLayer,
				 partition_layer_choice Choose>
			requires additive_cost<Cost, Compare> && std::is_invocable_r_v<uint32 *, ArgminRow &, uint32>
				  && std::invocable<AfterLayer &, std::vector<cost_value_t<Cost>> const &>
		constexpr void partition_layers(uint32 n, uint32 segments, bool is_whole_curve, Cost &cost,
										Compare const &comp, std::vector<cost_value_t<Cost>> &previous,
										std::vector<cost_value_t<Cost>> &current, ArgminRow &argmin_row,
										AfterLayer &after_layer, Choose const &choose) {
			constexpr uint64 unlimited = std::numeric_limits<uint64>::max();
			auto const top_of = [=](uint32 k) { return is_whole_curve || k == segments ? n : n - (segments - k); };
			auto const budget_of = [&](uint32 k) {
				uint64 const rows = top_of(k) - k + 1;
				return uint64(choose(rows * uint64(std::bit_width(rows))));
			};
			uint32 const *below = nullptr;
			uint32 below_top = n;
			uint64 knuth_estimate = unlimited, budget = segments > 2 ? budget_of(2) : 0;
			for (uint32 k = 2; k <= segments; ++k) {
				uint32 *const argmin = argmin_row(k);
				uint32 const top = top_of(k), first = k == segments ? n : k;
				bool is_whole_knuth = false;
				if (first == top || knuth_estimate <= budget) {
					auto const outcome = partition_knuth_layer(k, first, top, below, below_top, previous.data(),
															   current.data(), argmin, cost, comp,
															   first == top ? unlimited : budget);
					knuth_estimate = outcome.evaluations;
					is_whole_knuth = outcome.stop == first;
					if (!is_whole_knuth)
						partition_divide_layer(k, outcome.stop, argmin[outcome.stop], previous.data(), current.data(),
											   argmin, cost, comp);
				} else {
					partition_divide_layer(k, top + 1, top - 1, previous.data(), current.data(), argmin, cost, comp);
				}
				if (k + 1 < segments) {
					budget = budget_of(k + 1);
					// only "within the budget or not" matters: early layers, far
					// over it, stop the count after a few rows
					if (!is_whole_knuth) {
						knuth_estimate = 0;
						for (uint32 i = top; i >= k && knuth_estimate <= budget; --i) {
							knuth_window const window(k, i, below, below_top, i == top ? top : argmin[i + 1]);
							knuth_estimate += window.hi - window.lo + 1;
						}
					}
				}
				std::swap(previous, current);
				after_layer(std::as_const(previous));
				below = argmin;
				below_top = top;
			}
		}
	}

	/**
	 * @brief The best cost of cutting `[0, n)` into exactly `k` non-empty
	 *        segments, for every `k = 1 .. segments`: element `k - 1` of the
	 *        result, where `cost(j, i)` prices the segment `[j, i)`.
	 *
	 * The layered DP `dp_k[i] = min over k-1 <= j < i of dp_{k-1}[j] +
	 * cost(j, i)` (`dp_1[i] = cost(0, i)`), with the leftmost best `j` as
	 * `opt_k[i]`. Each layer is solved by one of two engines, both of which
	 * find the leftmost best `j` (so the values, and the cuts of
	 * @ref atl::optimal_partition_cuts, do not depend on which engine ran):
	 *
	 * - divide and conquer: the row minima of the staircase matrix (row `i`,
	 *   column `j < i`) by @ref atl::monotone_row_minima's recursion, O(n log n)
	 *   evaluations whatever the layer;
	 * - Knuth's layered bound `opt_{k-1}[i] <= opt_k[i] <= opt_k[i+1]`: row
	 *   `i`, descending, scans only that window. The windows shrink as `k`
	 *   grows (about `n / k^2` per row on balanced costs).
	 *
	 * After each layer the solver knows what Knuth's windows added up to,
	 * and switches to Knuth's bound once that is at most `rows *
	 * bit_width(rows)`, about what a divide-and-conquer layer evaluates
	 * (and back, should the windows grow). A Knuth layer gets that much as a
	 * budget: past it, its lower rows go to divide and conquer, so no layer
	 * costs more than about two divide-and-conquer layers. The last layer
	 * only needs `dp_k[n]`: one window. The staircase is handled by capping
	 * each row's column range, so no infinity is ever formed: every `dp`
	 * value is a real partition's cost.
	 *
	 * Every layer's best total comes out of the same run, so the whole curve
	 * `k -> best(k)` costs no more than its last point -- the oracle for a
	 * Lagrangian (WQS) solver, whose precondition is exactly that
	 * this curve is convex.
	 *
	 * Order of evaluation within a layer: rows are settled one at a time,
	 * each scanning its candidate `j` upward over a window, so a two-pointer
	 * cost (a sliding window that moves its ends to the asked segment, as
	 * for "distinct values in a segment") moves O(evaluations) steps.
	 *
	 * With `Compare = std::greater<>` this maximizes, and the precondition
	 * flips to the reverse inequality.
	 *
	 * Overflow: for an integral cost ordered by `std::less` or `std::greater`
	 * (the `std::ranges` forms included), every candidate `dp_{k-1}[j] +
	 * cost(j, i)` is formed and compared in a wider integer (`int128` for
	 * `int64`, `int64` for `int32`, `int32` below), so only the `dp` values
	 * need to fit the cost's type. A 128-bit cost, a floating cost or any
	 * other comparator has no wider type: there the candidates must fit too.
	 *
	 * @param n The number of items (`< 2^32 - 2`, asserted).
	 * @param segments The largest segment count wanted (`segments <= n`,
	 *        asserted: each segment is non-empty). `segments = 0` returns an
	 *        empty curve.
	 * @param cost `cost(j, i)` for `0 <= j < i <= n`: the price of `[j, i)`.
	 * @param comp The order: `comp(a, b)` is "a is better than b".
	 * @pre the quadrangle inequality: `cost(a, c) + cost(b, d)` no worse
	 *      than `cost(a, d) + cost(b, c)` for `a <= b < c <= d` (not
	 *      checkable; documented). It gives both monotonicities the engines
	 *      rest on, for the leftmost argmins: `opt_k[i] <= opt_k[i+1]` and
	 *      `opt_{k-1}[i] <= opt_k[i]`. The cost's value type holds every
	 *      `dp_k[i]`, and, without a wider type (see Overflow), every
	 *      candidate `dp_{k-1}[j] + cost(j, i)` (costs may be negative or
	 *      zero; no infinity is formed); not `bool` (rejected: the layers are
	 *      indexed as raw arrays). Recursion depth O(log n) (divide and
	 *      conquer's smaller half only).
	 * @complexity O(min(segments * n log n, n^2 + segments * n)) evaluations
	 *      of `cost` (Knuth's windows telescope along each diagonal `i - k`
	 *      to at most `n`); O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr std::vector<cost_value_t<Cost>> optimal_partition(std::size_t n, std::size_t segments, Cost &&cost,
																			  Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(detail::fits_index(n) && detail::fits_index(n + 1));
		atl_assert(segments <= n);
		std::vector<value_type> best;
		best.reserve(segments);
		if (segments == 0) return best;
		uint32 const last = uint32(n);
		std::vector<value_type> previous(n + 1), current(n + 1);
		for (uint32 i = 1; i <= last; ++i) previous[i] = cost(0, i);
		best.push_back(previous[last]);
		// two rolling argmin rows: layer k writes the one layer k - 1 did not
		std::vector<uint32> argmins(2 * (n + 1));
		auto argmin_row = [&argmins, row = n + 1](uint32 k) { return argmins.data() + (k & 1) * row; };
		auto after_layer = [&best, last](std::vector<value_type> const &layer) { best.push_back(layer[last]); };
		detail::partition_layers(last, uint32(segments), true, cost, comp, previous, current, argmin_row, after_layer,
								 detail::adaptive_partition_choice{});
		return best;
	}

	/**
	 * @brief What @ref atl::optimal_partition_cuts computes: the best cost of
	 *        cutting `[0, n)` into exactly `k` segments and the boundaries of
	 *        one optimum, `0 = b_0 < b_1 < ... < b_k = n` (segment `t` is
	 *        `[b_t, b_{t+1})`).
	 */
	template<std::semiregular Value>
	struct partition_cuts_result {
		/** @brief The best cost of partitioning all items into the requested number of segments. */
		Value value{};
		/** @brief Increasing endpoints of one optimal partition, including `0` and `n`. */
		std::vector<uint32> boundaries;
	};

	/**
	 * @brief @ref atl::optimal_partition for one segment count, with the cuts: the
	 *        best cost of exactly `segments` non-empty segments and the
	 *        boundaries of one optimum.
	 *
	 * Ties: the boundaries are read back from the leftmost argmins,
	 * `b_{k-1} = opt_k[n]`, then `b_{t-1} = opt_t[b_t]` down to `t = 2`:
	 * the last cut is the smallest one of any optimum, and each earlier cut
	 * is the smallest best one for the prefix it ends.
	 *
	 * Runs the same layers, keeping each layer's argmins. With one segment
	 * count to reach, layer `k` only fills rows `k .. n - (segments - k)`,
	 * and only those are stored: `(segments - 1) * (n - segments + 1) +
	 * segments` `uint32`s, 4 bytes each (k = 100, n = 1e5 is about 40 MB;
	 * when that is too much, @ref atl::exact_count_optimum over
	 * @ref atl::optimal_segmentation needs O(n), see its docs for the cuts).
	 *
	 * @param n The number of items (`< 2^32 - 2`, asserted).
	 * @param segments The segment count, `1 <= segments <= n` (asserted;
	 *        the argmin table's size must fit `std::size_t`, asserted, which
	 *        binds on 32-bit targets).
	 * @param cost `cost(j, i)` for `0 <= j < i <= n`: the price of `[j, i)`.
	 * @param comp The order: `comp(a, b)` is "a is better than b".
	 * @pre as @ref atl::optimal_partition
	 * @complexity as @ref atl::optimal_partition, O(segments * n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr partition_cuts_result<cost_value_t<Cost>> optimal_partition_cuts(std::size_t n, std::size_t segments,
																							 Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		atl_assert(detail::fits_index(n) && detail::fits_index(n + 1));
		atl_assert(1 <= segments && segments <= n);
		uint32 const last = uint32(n), count = uint32(segments);
		std::vector<value_type> previous(n + 1), current(n + 1);
		// layer k >= 2 fills rows k .. k + window - 1 only, at (k - 2) * window + i: the layers
		// leave one slot between them and the last row ends at (segments - 1) * window + segments - 1
		std::size_t const window = n - segments + 1;
		atl_assert(segments - 1 <= (std::numeric_limits<std::size_t>::max() - segments) / window);
		std::vector<uint32> choice((segments - 1) * window + segments);
		for (uint32 i = 1; i <= last - (count - 1); ++i) previous[i] = cost(0, i);
		auto argmin_row = [&choice, window](uint32 k) { return choice.data() + (k - 2) * window; };
		auto after_layer = [](std::vector<value_type> const &) {};
		detail::partition_layers(last, count, false, cost, comp, previous, current, argmin_row, after_layer,
								 detail::adaptive_partition_choice{});
		partition_cuts_result<value_type> result{std::move(previous[last]), std::vector<uint32>(segments + 1)};
		uint32 i = last;
		result.boundaries[segments] = i;
		for (std::size_t k = segments; k >= 2; --k) {
			i = choice[(k - 2) * window + i];
			result.boundaries[k - 1] = i;
		}
		return result;
	}
}
