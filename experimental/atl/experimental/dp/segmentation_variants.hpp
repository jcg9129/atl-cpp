#pragma once

/**
 * @file
 * @brief Contenders for the decision-monotone 1D/1D DP
 *        (@ref atl::optimal_segmentation, the monotone decision queue).
 *
 * Shared output contract: a @ref atl::segmentation_result with the smallest
 * best decision per boundary, so every contender must agree with the
 * library exactly, decisions included.
 *
 * - @ref atl::experimental::optimal_segmentation_by_halving_divide_and_conquer
 *   / @ref atl::experimental::optimal_segmentation_by_halving_smawk -- divide
 *   and conquer over the boundaries: finish the left half, push its rows
 *   into the right half as one rectangular row-minima problem, recurse
 *   right. O(n log^2 n) evaluations with the monotone divide and conquer,
 *   O(n log n) with SMAWK.
 * - @ref atl::experimental::optimal_segmentation_by_smawk_blocks -- the online
 *   linear-evaluation method (Galil and Park, 1990; in the block-doubling
 *   form of Feng Zhe, 2017), SMAWK on growing blocks.
 *
 * Race verdict (benchmarks/dp/segmentation.cpp, 2026-10-15, Xeon
 * Platinum 8280, ms, plain -O2 / -march=native; cost of [j, i) "cheap" =
 * (S_i - S_j - L)^2, "pow1.5" = |S_i - S_j - L|^1.5):
 *
 * | shape | library (decision queue) | by halving divide and conquer | by halving SMAWK | by SMAWK blocks |
 * |---|---|---|---|---|
 * | 1e5 cheap | 4.3 / 4.4 | 20.9 / 21.6 | 25.6 / 26.2 | 7.7 / 7.1 |
 * | 1e5 pow1.5 | 7.9 / 7.7 | 29.4 / 28.4 | 30.0 / 30.0 | 8.3 / 8.2 |
 * | 1e6 cheap | 50.2 / 52.4 | 276 / 282 | 295 / 302 | 81.9 / 74.7 |
 * | 1e6 pow1.5 | 93.7 / 91.0 | 390 / 379 | 349 / 344 | 87.7 / 86.4 |
 *
 * The decision queue is the library's: fastest on three shapes of four by
 * 1.05-1.8x; the SMAWK blocks' 5-6% win on the 1e6 pow1.5 shape is their
 * fewer evaluations (11 per boundary) paying off once an evaluation is
 * dear.
 *
 * Every contender forms and compares its candidate totals in the library's
 * wide type (`atl::detail::dp_total_t`), so the contenders agree with the
 * library wherever a candidate overflows the cost's type but the `dp`
 * values fit.
 */

#include <atl/dp/concepts.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/dp/segmentation.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace atl::experimental {
	namespace detail {
		/**
		 * @brief The halving driver: finishes boundaries `[lo, hi]` left to
		 *        right. On entry `r.value[i]` / `r.decision[i]` hold the best
		 *        candidate from the rows `< lo` for each `i` in `[lo, hi]`
		 *        (`has_best[i]` says whether there is one yet).
		 * @pre the cost satisfies the quadrangle inequality; `r.value[lo]`
		 *      is final when `lo == 0` or `has_best[lo]`
		 * @complexity O((hi - lo) log(hi - lo)) row-minima rows, times the
		 *             engine's factor; recursion depth log2(hi - lo) + 1
		 */
		template<bool ShouldUseSmawk, typename Cost, typename Compare>
			requires additive_cost<Cost, Compare>
		constexpr void halving_segmentation(uint32 lo, uint32 hi, Cost &cost, Compare const &comp,
											segmentation_result<cost_value_t<Cost>> &r, std::vector<uint8> &has_best) {
			using value_type = cost_value_t<Cost>;
			using total = atl::detail::dp_total_t<value_type, Compare>;
			if (lo == hi) return;
			uint32 const mid = lo + (hi - lo) / 2;
			halving_segmentation<ShouldUseSmawk>(lo, mid, cost, comp, r, has_best);
			atl::detail::total_order<value_type, Compare> const better{comp};
			// rows = targets mid+1 .. hi, columns = decisions lo .. mid
			auto const cross = [&](uint32 t, uint32 d) -> total {
				return total(total(r.value[lo + d]) + total(cost(lo + d, mid + 1 + t)));
			};
			auto block = [&] {
				if constexpr (ShouldUseSmawk)
					return totally_monotone_row_minima(hi - mid, mid - lo + 1, cross, better);
				else
					return monotone_row_minima(hi - mid, mid - lo + 1, cross, better);
			}();
			for (uint32 t = 0; t < hi - mid; ++t) {
				uint32 const i = mid + 1 + t;
				// an older (smaller) decision keeps a tie
				if (!has_best[i] || better(block.value[t], total(r.value[i]))) {
					r.value[i] = value_type(std::move(block.value[t]));
					r.decision[i] = lo + block.column[t];
					has_best[i] = 1;
				}
			}
			halving_segmentation<ShouldUseSmawk>(mid + 1, hi, cost, comp, r, has_best);
		}

		/**
		 * @brief `halving_segmentation`'s entry point; @p ShouldUseSmawk
		 *        picks SMAWK over the monotone divide and conquer as the
		 *        cross-half engine.
		 */
		template<bool ShouldUseSmawk, typename Cost, typename Compare>
			requires additive_cost<Cost, Compare>
		constexpr segmentation_result<cost_value_t<Cost>> halving_entry(std::size_t n, Cost &cost, Compare const &comp) {
			atl_assert(atl::detail::fits_index(n) && atl::detail::fits_index(n + 1));
			segmentation_result<cost_value_t<Cost>> r;
			r.value.assign(n + 1, cost_value_t<Cost>{});
			r.decision.assign(n + 1, 0);
			std::vector<uint8> has_best(n + 1, 0);
			halving_segmentation<ShouldUseSmawk>(0, uint32(n), cost, comp, r, has_best);
			return r;
		}

		/**
		 * @brief A value of the online method's augmented matrix: a real
		 *        entry, or the infinity that stands for "rows below the
		 *        window are strictly beaten here".
		 */
		template<std::semiregular Value>
		struct extended_value {
			Value value{};
			bool is_infinite = false;
		};
	}

	/**
	 * @brief @ref atl::optimal_segmentation by halving divide and conquer,
	 *        with @ref atl::monotone_row_minima as the cross-half engine.
	 *        Lost the race 3.7-5.5x to the decision queue on every shape
	 *        (its O(n log^2 n) evaluations; see the file's table).
	 * @pre as @ref atl::optimal_segmentation
	 * @complexity O(n log^2 n) evaluations
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare>
	[[nodiscard]] constexpr segmentation_result<cost_value_t<Cost>> optimal_segmentation_by_halving_divide_and_conquer(std::size_t n, Cost &&cost,
																													   Compare comp = {}) {
		return detail::halving_entry<false>(n, cost, comp);
	}

	/**
	 * @brief @ref atl::optimal_segmentation by halving divide and conquer,
	 *        with @ref atl::totally_monotone_row_minima (SMAWK) as the
	 *        cross-half engine. Lost the race 3.7-6.0x to the decision queue
	 *        (SMAWK's constant factor on small blocks eats its log-factor
	 *        saving; see the file's table).
	 * @pre as @ref atl::optimal_segmentation; not a `bool` cost (rejected,
	 *      as by @ref atl::totally_monotone_row_minima)
	 * @complexity O(n log n) evaluations
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && atl::detail::contiguously_storable<cost_value_t<Cost>>
	[[nodiscard]] constexpr segmentation_result<cost_value_t<Cost>> optimal_segmentation_by_halving_smawk(std::size_t n, Cost &&cost,
																										  Compare comp = {}) {
		return detail::halving_entry<true>(n, cost, comp);
	}

	/**
	 * @brief @ref atl::optimal_segmentation by the online linear-evaluation
	 *        method: SMAWK over blocks that double against the undecided
	 *        window (Galil and Park, 1990, "A linear-time algorithm for
	 *        concave one-dimensional dynamic programming"; the block form
	 *        follows Feng Zhe's 2017 Chinese national training team paper,
	 *        section 5.2). Lost the race to the decision queue on cheap
	 *        costs (1.4-1.8x slower) and on the 1e5 pow1.5 shape
	 *        (1.05-1.06x), and won by 5-6% only on the 1e6 pow1.5 shape (see the file's
	 *        table): kept as the candidate for costs whose evaluation is
	 *        expensive.
	 *
	 * State: boundaries `< c` are final; rows `[r, c)` are live; the rows
	 * below `r` are summarized per future boundary `x` by `E[x]`, their
	 * leftmost best row, valid for `x <= e_end` -- past `e_end` some live
	 * row STRICTLY beats every row below `r` (so the summary row reads as
	 * infinity there, which keeps the augmented matrix totally monotone).
	 *
	 * One step takes the block `x in [c, p]`, `p = min(2c - r, n)`, solves
	 * the `(c - r + 1)`-column matrix (summary + live rows) by SMAWK into
	 * `G`, then walks `j = c, c+1, ..`: rows `c .. j-2` cannot win at `j`,
	 * so `dp[j]` is the better of `G[j]` and row `j - 1`.
	 * - row `j-1` strictly wins at `j`: it strictly beats every older row at
	 *   all later boundaries; window becomes `[j-1, j+1)`, no summary.
	 * - it loses at `j` but strictly wins at `p`: same window, summary
	 *   `E = G` on `(j, p]`.
	 * - the walk passes `p`: `c = p + 1`; if `G[p]` is a live row, every row
	 *   below it is strictly beaten from `p` on, so `r = G[p]`, no summary;
	 *   otherwise the summary still holds and `r` stays.
	 *
	 * Against the paper's text: it sets `r = pos(p)` in the last case
	 * unconditionally and keeps stale `E` entries behind a sentinel without
	 * saying when they are valid. When `pos(p)` comes from the summary, the
	 * literal rule moves `r` backwards; it stays correct (100000 random
	 * Monge cases) but evaluated 36 entries per boundary instead of 11 on a
	 * convex-cost shape at n = 1e6. The `e_end` bookkeeping above is what
	 * the total-monotonicity proof of the augmented matrix needs; never
	 * invalidating the summary passed the same random cases, so it is not
	 * shown wrong, only unproven.
	 *
	 * @pre as @ref atl::optimal_segmentation
	 * @complexity O(n) evaluations of `cost`
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare>
	[[nodiscard]] constexpr segmentation_result<cost_value_t<Cost>> optimal_segmentation_by_smawk_blocks(std::size_t n, Cost &&cost,
																										 Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		using total = atl::detail::dp_total_t<value_type, Compare>;
		using extended = detail::extended_value<total>;
		// tighter than the family's bound: the block end 2c - r must fit uint32
		atl_assert(n < (std::size_t(1) << 31) - 1);
		segmentation_result<value_type> res;
		res.value.assign(n + 1, value_type{});
		res.decision.assign(n + 1, 0);
		if (n == 0) return res;
		auto const at = [&](uint32 j, uint32 i) -> total { return total(total(res.value[j]) + total(cost(j, i))); };
		uint32 const last = uint32(n);
		std::vector<uint32> summary(n + 1, 0), g_slot(n + 1), g_row(n + 1);
		std::vector<extended> g_value(n + 1);
		std::vector<uint32> columns;
		uint32 c = 1, r = 0, e_end = 0;
		atl::detail::total_order<value_type, Compare> const total_better{comp};
		auto const better = [&total_better](extended const &a, extended const &b) {
			return !a.is_infinite && (b.is_infinite || total_better(a.value, b.value));
		};
		while (c <= last) {
			uint32 const p = (std::min)(2 * c - r, last);
			bool const has_summary = e_end >= c;
			columns.clear();
			if (has_summary) columns.push_back(0);
			for (uint32 s = 1; s <= c - r; ++s) columns.push_back(s);
			auto eval = [&](uint32 pos, uint32 slot) -> extended {
				uint32 const x = c + pos;
				if (slot != 0) return {at(r + slot - 1, x), false};
				if (x > e_end) return {total{}, true};
				return {at(summary[x], x), false};
			};
			atl::detail::smawk<decltype(eval), decltype(better)> solver(eval, better, g_slot.data(), g_value.data());
			solver.run(p - c + 1, columns);
			for (uint32 pos = 0; pos <= p - c; ++pos)
				g_row[pos] = g_slot[pos] == 0 ? summary[c + pos] : r + g_slot[pos] - 1;
			res.value[c] = value_type(std::move(g_value[0].value));
			res.decision[c] = g_row[0];
			uint32 const c0 = c;
			bool is_settled = false;
			for (uint32 j = c0 + 1; j <= p; ++j) {
				total candidate = at(j - 1, j);
				if (total_better(candidate, g_value[j - c0].value)) {
					res.value[j] = value_type(std::move(candidate));
					res.decision[j] = j - 1;
					r = j - 1;
					c = j + 1;
					e_end = 0;
					is_settled = true;
					break;
				}
				res.value[j] = value_type(g_value[j - c0].value);
				res.decision[j] = g_row[j - c0];
				if (total_better(at(j - 1, p), g_value[p - c0].value)) {
					for (uint32 x = j + 1; x <= p; ++x) summary[x] = g_row[x - c0];
					r = j - 1;
					c = j + 1;
					e_end = p;
					is_settled = true;
					break;
				}
			}
			if (is_settled) continue;
			c = p + 1;
			if (g_slot[p - c0] != 0) {
				r = g_row[p - c0];
				e_end = 0;
			}
		}
		return res;
	}
}
