#pragma once

/**
 * @file
 * @brief @ref atl::optimal_segmentation -- the decision-monotone 1D/1D DP
 *        `dp[i] = min_{j < i} dp[j] + w(j, i)` under the quadrangle
 *        inequality, by the monotone decision queue.
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
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief What @ref atl::optimal_segmentation computes, over the boundaries
	 *        `0 .. n`: `value[i]` is the best cost of segmenting the prefix
	 *        `[0, i)` (`value[0] = Value{}`, the empty segmentation), and
	 *        `decision[i]` the start of the last segment in it -- the
	 *        smallest best `j` (`decision[0] = 0`). Walking `i =
	 *        decision[i]` from `n` down to `0` lists the cuts.
	 */
	template<std::semiregular Value>
	struct segmentation_result {
		/** @brief `value[i]`: the best cost of segmenting `[0, i)`, `value[0] = Value{}`. */
		std::vector<Value> value;
		/** @brief `decision[i]`: the smallest best start of the last segment of `[0, i)`, `decision[0] = 0`. */
		std::vector<uint32> decision;

		/**
		 * @brief The boundaries of the optimal segmentation of the whole
		 *        `[0, n)` that `decision` encodes: `0 = b_0 < b_1 < ... <
		 *        b_k = n`, segment `t` being `[b_t, b_{t+1})` (`{0}` for `n =
		 *        0`, empty for a default-constructed result).
		 * @complexity O(k) for `k` segments
		 */
		[[nodiscard]] constexpr std::vector<uint32> boundaries() const {
			std::vector<uint32> cuts;
			if (decision.empty()) return cuts;
			auto i = uint32(decision.size() - 1);
			cuts.push_back(i);
			while (i != 0) {
				i = decision[i];
				cuts.push_back(i);
			}
			std::ranges::reverse(cuts);
			return cuts;
		}
	};

	/**
	 * @brief The decision-monotone 1D/1D DP: `dp[0] = 0`, `dp[i] = min over
	 *        0 <= j < i of dp[j] + cost(j, i)` for `i = 1 .. n` -- the best
	 *        way to cut `[0, n)` into segments when `cost(j, i)` prices the
	 *        segment `[j, i)`.
	 *
	 * Monotone decision queue: the candidates `j` sit in a queue, each owning
	 * the run of future boundaries where it is best; a new candidate pops
	 * the runs it wins outright from the back and binary-searches its
	 * takeover point inside the next one. Unlike divide and conquer this is
	 * ONLINE: `dp[i]` is final before any boundary past `i` is asked about,
	 * which the recurrence needs (row `j` of the matrix exists only once
	 * `dp[j]` is known).
	 *
	 * Ties go to the smallest `j` (strict comparisons throughout), so
	 * `decision` is deterministic: `decision[i]` is the smallest `j`
	 * attaining `dp[i]`, and @ref atl::segmentation_result::boundaries follows
	 * those decisions back from `n`.
	 *
	 * With `Compare = std::greater<>` this maximizes, and the precondition
	 * flips to the reverse inequality.
	 *
	 * Overflow: for an integral cost ordered by `std::less` or `std::greater`
	 * (the `std::ranges` forms included), every candidate `dp[j] + cost(j,
	 * i)` is formed and compared in a wider integer (`int128` for `int64`,
	 * `int64` for `int32`, `int32` below), so only the `dp` values need to
	 * fit `Value`. A 128-bit cost or any other comparator has no wider type:
	 * there `Value` holds every `dp` value plus one cost.
	 *
	 * Floating-point costs: the takeover search relies on the rounded
	 * comparisons `dp[j] + cost(j, i)` against `dp[j'] + cost(j', i)` crossing
	 * once along `i`, as the exact ones do under the quadrangle inequality.
	 * Rounding can break that near ties (`n = 4`, costs about `2^54` that
	 * satisfy the inequality with equality make a later candidate tie at
	 * boundaries 2 and 4 after rounding but win at 3, and the search skips
	 * it), so a `double` cost must make those comparisons exact -- every
	 * `dp` value plus cost representable, e.g. integers below `2^53` -- or
	 * keep them single-crossing after rounding.
	 *
	 * @param n The number of items (boundaries `0 .. n`; `< 2^32 - 2`,
	 *        asserted).
	 * @param cost `cost(j, i)` for `0 <= j < i <= n`: the price of the
	 *        segment `[j, i)`. Evaluated O(n log n) times in no
	 *        locality-friendly order. Values may be negative or zero.
	 * @param comp The order: `comp(a, b)` is "a is better than b".
	 * @pre `cost` satisfies the quadrangle inequality on its domain: for
	 *      `a <= b < c <= d`, `cost(a, c) + cost(b, d)` is no worse than
	 *      `cost(a, d) + cost(b, c)` (not checkable; documented). `Value`
	 *      holds every `dp` value, and, without a wider type (see Overflow),
	 *      every `dp` value plus one cost; no infinity is ever formed. A
	 *      floating `Value` keeps the candidate comparisons exact or
	 *      single-crossing after rounding (see above; documented).
	 * @complexity O(n log n) evaluations of `cost`, O(n) memory
	 */
	template<typename Cost, typename Compare = std::less<>>
		requires additive_cost<Cost, Compare>
	[[nodiscard]] constexpr segmentation_result<cost_value_t<Cost>> optimal_segmentation(std::size_t n, Cost &&cost,
																						 Compare comp = {}) {
		using value_type = cost_value_t<Cost>;
		using total = detail::dp_total_t<value_type, Compare>;
		atl_assert(detail::fits_index(n) && detail::fits_index(n + 1));
		segmentation_result<value_type> r;
		r.value.assign(n + 1, value_type{});
		r.decision.assign(n + 1, 0);
		if (n == 0) return r;
		detail::total_order<value_type, Compare> const better{comp};
		auto const at = [&](uint32 j, uint32 i) -> total { return total(total(r.value[j]) + total(cost(j, i))); };
		uint32 const last = uint32(n);
		// the queue: candidate[k] owns boundaries [start[k], start[k + 1])
		std::vector<uint32> candidate(n + 1), start(n + 1);
		uint32 head = 0, tail = 1;
		candidate[0] = 0;
		start[0] = 1;
		for (uint32 i = 1; i <= last; ++i) {
			while (tail - head > 1 && start[head + 1] <= i) ++head;
			r.decision[i] = candidate[head];
			r.value[i] = value_type(at(candidate[head], i));
			if (i == last) break;
			// i competes for boundaries i+1 .. n; it wins a suffix of any
			// older candidate's run (quadrangle inequality), so a run it wins
			// at its first boundary is won whole
			uint32 from = i + 1;
			while (tail > head) {
				from = (std::max)(start[tail - 1], i + 1);
				if (!better(at(i, from), at(candidate[tail - 1], from))) break;
				--tail;
			}
			if (tail == head) {
				candidate[tail] = i;
				start[tail++] = i + 1;
				continue;
			}
			// first boundary in (from, n] where i is strictly better
			uint32 lo = from + 1, hi = last + 1;
			uint32 const rival = candidate[tail - 1];
			while (lo < hi) {
				uint32 const mid = lo + (hi - lo) / 2;
				if (better(at(i, mid), at(rival, mid)))
					hi = mid;
				else
					lo = mid + 1;
			}
			if (lo <= last) {
				candidate[tail] = i;
				start[tail++] = lo;
			}
		}
		return r;
	}
}
