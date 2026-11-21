#pragma once

/**
 * @file
 * @brief @ref atl::optimal_interval_merge -- the interval DP
 *        `dp[l][r] = min_{l < m < r} dp[l][m] + dp[m][r] + w(l, r)` in
 *        O(n^2) by Knuth's optimization (monotone split points).
 */

#include <atl/dp/concepts.hpp>
#include <atl/dp/detail/wide.hpp>
#include <atl/dp/row_minima.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief What @ref atl::optimal_interval_merge computes: the best cost of
	 *        every segment `[l, r)`, `0 <= l < r <= items`, read with
	 *        `value`, and the order it was computed under. Single items cost
	 *        `Value{}`.
	 *
	 * `table` stores the segments by left end, row `l` holding `r = l+1 ..
	 * items` (`items * (items + 1) / 2` values in all).
	 *
	 * @tparam Value   The cost type.
	 * @tparam Compare The order the table was built with (`std::less<>` by
	 *                 default), kept so that @ref atl::optimal_merge_order
	 *                 reads the tree off under the same order.
	 */
	template<std::semiregular Value, typename Compare = std::less<>>
		requires std::copy_constructible<Compare> && std::strict_weak_order<Compare const &, Value const &, Value const &>
	struct interval_merge_result {
		/** @brief The number of items `n`. */
		uint32 items = 0;
		/** @brief The best cost of every segment, row by row (see `index`). */
		std::vector<Value> table;
		/** @brief The order `table` was built with: `comp(a, b)` is "a is better than b". */
		Compare comp{};

		/**
		 * @brief The flat index of segment `[l, r)` in `table`.
		 * @pre `l < r <= items` (asserted)
		 */
		[[nodiscard]] constexpr std::size_t index(std::size_t l, std::size_t r) const {
			atl_assert(l < r && r <= items);
			return l * (2 * std::size_t(items) - l + 1) / 2 + (r - l - 1);
		}

		/**
		 * @brief The best cost of merging segment `[l, r)` into one piece.
		 * @pre `l < r <= items` (asserted)
		 */
		[[nodiscard]] constexpr Value const &value(std::size_t l, std::size_t r) const { return table[index(l, r)]; }
	};

	/**
	 * @brief The optimal merge tree of every segment: `dp[l][l+1] = 0`,
	 *        `dp[l][r] = min over l < m < r of dp[l][m] + dp[m][r] +
	 *        cost(l, r)` -- merging `[l, r)` costs `cost(l, r)` on top of
	 *        the two halves it is merged from (stone merging, optimal
	 *        binary search trees, optimal alphabetic codes).
	 *
	 * Knuth's optimization: the leftmost best split `opt[l][r]` satisfies
	 * `opt[l][r-1] <= opt[l][r] <= opt[l+1][r]`, so each segment scans only
	 * that window and the windows telescope to O(n) per left end. Rows are
	 * filled from `l = items-1` down with `r` increasing, so only two rows
	 * of split points are kept.
	 *
	 * Boundaries are half-open, as in the rest of the family: segment `[l,
	 * r)` holds items `l .. r-1`, and the split `m` gives `[l, m)` and `[m,
	 * r)`. The closed form `f[i][j] = min_{i <= k < j} f[i][k] + f[k+1][j]
	 * + w(i, j)` is `dp[i][j+1]` with `cost(l, r) = w(l, r-1)`.
	 *
	 * A cost that breaks the precondition is not detected: the windows can
	 * never be empty (`opt[l][r-1] <= opt[l+1][r-1] <= opt[l+1][r]` holds by
	 * construction, whatever the cost), so the scan silently returns a
	 * non-optimal table.
	 *
	 * With `Compare = std::greater<>` this maximizes, with every inequality
	 * of the precondition reversed.
	 *
	 * Overflow: for an integral cost ordered by `std::less` or `std::greater`
	 * (the `std::ranges` forms included), every candidate `dp[l][m] +
	 * dp[m][r]` and its sum with `cost(l, r)` is formed and compared in a
	 * wider integer (`int32` for `int8` and `int16`, `int64` for `int32`,
	 * `int128` for `int64`), so only the table's values -- each segment's
	 * best cost -- need to fit `Value`. A 128-bit cost, a floating cost or
	 * any other comparator has no wider type: there `Value` must hold every
	 * candidate too.
	 *
	 * @param items The number of items `n` (`< 2^32 - 2`, asserted; the
	 *        table's `n (n + 1) / 2` values must fit `std::size_t`,
	 *        asserted, which binds on 32-bit targets).
	 * @param cost `cost(l, r)` for `0 <= l`, `l + 2 <= r <= n`, queried once
	 *        each, `l` descending and `r` ascending within an `l`.
	 * @param comp The order: `comp(a, b)` is "a is better than b"; stored in
	 *        the result.
	 * @pre for `a <= b < c <= d`: `cost(a, c) + cost(b, d)` no worse than
	 *      `cost(a, d) + cost(b, c)` (quadrangle inequality) and `cost(b, c)`
	 *      no worse than `cost(a, d)` (monotone on inclusion) -- with the
	 *      unqueried single-item costs chosen to fit (not checkable;
	 *      documented). `Value` holds every segment's best cost, and, without
	 *      a wider type (see Overflow), every candidate total. `Value` is not
	 *      `bool` (rejected: the table is indexed as a raw array).
	 * @complexity O(n^2) time and evaluations, O(n^2) memory for the table
	 *
	 * The split points are not returned: @ref atl::optimal_merge_order
	 * recovers a tree from the table alone.
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare> && detail::contiguously_storable<cost_value_t<Cost>>
			  && std::copy_constructible<Compare>
	[[nodiscard]] constexpr interval_merge_result<cost_value_t<Cost>, Compare>
	optimal_interval_merge(std::size_t items, Cost &&cost, Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		using total = detail::dp_total_t<value_type, Compare>;
		atl_assert(detail::fits_index(items) && detail::fits_index(items + 1));
		atl_assert(items == 0 || items + 1 <= std::numeric_limits<std::size_t>::max() / items);
		uint32 const n = uint32(items);
		interval_merge_result<value_type, Compare> result{n, std::vector<value_type>(std::size_t(n) * (std::size_t(n) + 1) / 2),
														  comp};
		if (n == 0) return result;
		detail::total_order<value_type, Compare> const better{result.comp};
		value_type *const table = result.table.data();
		// start[x] - x - 1 is row x's base: segment [x, r) sits at start[x] + r - x - 1
		std::vector<std::size_t> start(n);
		for (uint32 x = 0; x < n; ++x) start[x] = result.index(x, x + 1);
		// opt rows indexed by r: `upper` = row l+1, `lower` = row l being filled;
		// opt[x][x+1] = x makes the r = l+2 window the single split l+1
		std::vector<uint32> upper(n + 1), lower(n + 1);
		for (uint32 l = n; l-- > 0;) {
			std::size_t const row = start[l];
			lower[l + 1] = l;
			for (uint32 r = l + 2; r <= n; ++r) {
				uint32 const lo = (std::max)(lower[r - 1], l + 1), hi = (std::min)(upper[r], r - 1);
				uint32 best = lo;
				// the children's sums stay wide until the merge cost is added
				auto best_value = total(total(table[row + (lo - l - 1)]) + total(table[start[lo] + (r - lo - 1)]));
				for (uint32 m = lo + 1; m <= hi; ++m) {
					auto v = total(total(table[row + (m - l - 1)]) + total(table[start[m] + (r - m - 1)]));
					if (better(v, best_value)) {
						best_value = std::move(v);
						best = m;
					}
				}
				lower[r] = best;
				table[row + (r - l - 1)] = value_type(total(best_value + total(cost(l, r))));
			}
			std::swap(upper, lower);
		}
		return result;
	}

	/** @brief One merge of an optimal merge tree: `[left, split)` and `[split, right)` become `[left, right)`. */
	struct merge_step {
		uint32 left = 0;  ///< The left end of the merged segment.
		uint32 split = 0; ///< Where the two merged halves meet.
		uint32 right = 0; ///< The right end (exclusive) of the merged segment.
	};

	/**
	 * @brief The merges of one optimal merge tree of all `items`, read off
	 *        the table @ref atl::optimal_interval_merge returned: `items - 1` steps
	 *        in an executable order (both halves of a step are single items or
	 *        results of earlier steps; the last step makes `[0, items)`).
	 *
	 * The merge cost `cost(l, r)` does not depend on the split, so a best
	 * split of `[l, r)` is any `m` minimizing `value(l, m) + value(m, r)`:
	 * the table alone recovers the tree, and the solver needs no extra
	 * memory for it. The leftmost `m` minimizing `value(l, m) + value(m, r)`
	 * (under the order stored in the table, `result.comp`) is taken; the sums
	 * are formed and compared in the same wider type as the solver's.
	 *
	 * @param result The table of @ref atl::optimal_interval_merge.
	 * @complexity O(sum over the tree's nodes of their lengths): O(n log n)
	 *      for a balanced tree, O(n^2) at worst; an explicit stack, no
	 *      recursion
	 */
	template<std::semiregular Value, typename Compare>
		requires std::strict_weak_order<Compare const &, Value const &, Value const &>
			  && requires(Value const a) {
					 { a + a } -> std::convertible_to<Value>;
				 }
	[[nodiscard]] constexpr std::vector<merge_step> optimal_merge_order(interval_merge_result<Value, Compare> const &result) {
		using total = detail::dp_total_t<Value, Compare>;
		detail::total_order<Value, Compare> const better{result.comp};
		std::vector<merge_step> steps;
		if (result.items < 2) return steps;
		steps.reserve(result.items - 1);
		// preorder through an explicit stack; reversed, every step follows its halves
		std::vector<merge_step> pending{{0, 0, result.items}};
		while (!pending.empty()) {
			merge_step step = pending.back();
			pending.pop_back();
			step.split = step.left + 1;
			auto best = total(total(result.value(step.left, step.split)) + total(result.value(step.split, step.right)));
			for (uint32 m = step.left + 2; m < step.right; ++m) {
				auto candidate = total(total(result.value(step.left, m)) + total(result.value(m, step.right)));
				if (better(candidate, best)) {
					best = std::move(candidate);
					step.split = m;
				}
			}
			steps.push_back(step);
			if (step.split - step.left >= 2) pending.push_back({step.left, 0, step.split});
			if (step.right - step.split >= 2) pending.push_back({step.split, 0, step.right});
		}
		std::ranges::reverse(steps);
		return steps;
	}
}
