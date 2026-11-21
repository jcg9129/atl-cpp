#pragma once
#include <atl/ds/direct_rmq.hpp>
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/suffix_array.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace atl {
	/**
	 * @brief The Lyndon factorization of @p text (Duval's algorithm): the
	 *        unique split into factors `w1 >= w2 >= ... >= wk` where each
	 *        factor is Lyndon (strictly smaller than all its proper
	 *        suffixes). Returned as the boundary positions
	 *        `0 = b[0] < b[1] < ... < b[k] = n` — factor `i` is
	 *        `[b[i], b[i+1])`; `{0}` for the empty text.
	 * @pre size(text) < 2^32 (asserted).
	 * @complexity O(n), comparisons amortized linear
	 * @tparam R A sized random-access range with totally ordered projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param text Symbols read in place; no references are retained.
	 * @param proj Stable projection whose order agrees with equality.
	 * @return The Lyndon-factor boundary positions, including zero and n.
	 */
	template<typename R, typename Proj = std::identity>
		requires sortable_text<R, Proj>
	[[nodiscard]] constexpr std::vector<uint32> lyndon_factorization(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		auto const symbol = [&](uint32 i) -> decltype(auto) { return std::invoke(proj, first[i]); };
		atl_assert(std::ranges::size(text) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(text));
		std::vector<uint32> bounds{0};
		uint32 i = 0;
		while (i < n) {
			// grow the current run of equal Lyndon factors: k trails j by a
			// whole number of factor lengths
			uint32 j = i + 1, k = i;
			while (j < n && !(symbol(j) < symbol(k))) {
				k = symbol(k) < symbol(j) ? i : k + 1;
				++j;
			}
			for (; i <= k; i += j - k) bounds.push_back(i + (j - k));
		}
		return bounds;
	}

	/**
	 * @brief Start index of the lexicographically smallest rotation of
	 *        @p text (0 for the empty text) — the classic two-candidate
	 *        elimination over the doubled string.
	 * @pre `size(text) < 2^31`, so doubled-text indices fit uint32.
	 * @complexity O(n)
	 * @tparam R A sized random-access range with totally ordered projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param text Symbols read in place; no references are retained.
	 * @param proj Stable projection whose order agrees with equality.
	 * @return A least-rotation start position, or zero for an empty text.
	 */
	template<typename R, typename Proj = std::identity>
		requires sortable_text<R, Proj>
	[[nodiscard]] constexpr uint32 minimal_rotation_index(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		// indices into the doubled text reach about 2n, so 2n must fit uint32
		atl_assert(std::ranges::size(text) < (uint64(1) << 31));
		uint32 const n = uint32(std::ranges::size(text));
		auto const symbol = [&](uint32 i) -> decltype(auto) { return std::invoke(proj, first[i < n ? i : i - n]); };
		uint32 i = 0, j = 1, k = 0;
		while (i < n && j < n && k < n) {
			auto const &a = symbol(i + k);
			auto const &b = symbol(j + k);
			if (!(a < b) && !(b < a)) {
				++k;
				continue;
			}
			(b < a ? i : j) += k + 1; // the greater candidate cannot start a
			if (i == j) ++j;		  // minimal rotation anywhere in that span
			k = 0;
		}
		return (std::min)(i, j);
	}

	/**
	 * @brief One maximal repetition ("run"): `[l, r)` has smallest period
	 * `period` with `r - l >= 2 * period`, and neither `[l-1, r)` nor
	 * `[l, r+1)` keeps that period.
	 */
	struct run {
		uint32 period; /**< @brief Smallest positive period. */
		uint32 l;	   /**< @brief Inclusive start position. */
		uint32 r;	   /**< @brief Exclusive end position. */
		/** @brief Lexicographic comparison by (period, l, r). */
		constexpr friend auto operator<=>(run const &, run const &) = default;
	};

	/**
	 * @brief All runs of @p text, sorted by `(period, l, r)` — the runs
	 *        theorem (Bannai et al.): every run's Lyndon root is a
	 *        longest-Lyndon-word-from-position under the normal or the
	 *        inverted alphabet order; roots come from next-smaller-suffix
	 *        sweeps over the two rank arrays, and each candidate extends in
	 *        both directions by LCP queries — @ref atl::suffix_array of the text
	 *        and of its reverse, with @ref atl::direct_rmq over their
	 *        @ref atl::suffix_array::lcp_lengths.
	 * @complexity O(n) time for integral symbols no wider than uint64 whose
	 * span is at most min(n, 2^32 - 3); O(n log n) otherwise. O(n) storage.
	 * Suffix sorting recurses only on geometrically smaller reduced texts;
	 * other passes are iterative, with no recursion per character.
	 * @pre `size(text) < 2^32 - 2`.
	 * @tparam R A sized random-access range with totally ordered projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param text Symbols read in place; no references are retained.
	 * @param proj Stable projection whose order agrees with equality.
	 * @return All unique maximal repetitions sorted by (period, l, r).
	 */
	template<typename R, typename Proj = std::identity>
		requires sortable_text<R, Proj>
	[[nodiscard]] constexpr std::vector<run> runs(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		// The counting passes allocate n + 2 buckets.
		atl_assert(std::ranges::size(text) < (uint64(1) << 32) - 2);
		uint32 const n = uint32(std::ranges::size(text));
		if (n == 0) return {};
		// integer keys whose order can be INVERTED by a subtraction: integral
		// symbols need no mapping at all (offsets do), anything else
		// rank-compresses with one sort
		std::vector<uint32> mapped(n);
		using symbol = text_symbol_t<R, Proj>;
		auto const compress = [&] { // rank-compress, held as INVERTED ranks
			std::vector<uint32> order(n);
			for (uint32 k = 0; k < n; ++k) order[k] = k;
			std::ranges::sort(order, [&](uint32 a, uint32 b) {
				return std::invoke(proj, first[a]) < std::invoke(proj, first[b]);
			});
			uint32 name = 0;
			for (uint32 k = 0; k < n; ++k) {
				if (k > 0 && std::invoke(proj, first[order[k - 1]]) < std::invoke(proj, first[order[k]]))
					++name;
				mapped[order[k]] = name;
			}
			for (auto &v : mapped) v = name - v + 1;
		};
		bool is_offset_done = false;
		// wider-than-64-bit integers (int128) would truncate in the uint64 span
		if constexpr (std::integral<symbol> && sizeof(symbol) <= sizeof(uint64)) {
			auto lo = std::invoke(proj, first[0]), hi = lo;
			for (uint32 i = 1; i < n; ++i) {
				auto const v = std::invoke(proj, first[i]);
				lo = (std::min)(lo, v);
				hi = (std::max)(hi, v);
			}
			// the inverted keys hi - v + 1 and the later top = max + 1 must
			// both fit uint32; limiting span to n also bounds bucket storage
			if (uint64(hi) - uint64(lo) <= (std::min)(uint64(n), (uint64(1) << 32) - 3)) {
				for (uint32 i = 0; i < n; ++i)
					mapped[i] = uint32(uint64(hi) - uint64(std::invoke(proj, first[i])) + 1);
				is_offset_done = true;
			}
		}
		if (!is_offset_done) compress();

		// nss[i] = the next j > i whose suffix is smaller under an order —
		// [i, nss[i]) is the longest Lyndon word starting at i under it
		auto const next_smaller = [n]<length_range Ranks>(Ranks const &ranks) {
			std::vector<uint32> stack;
			std::vector<uint32> nss(n, n);
			for (uint32 i = n; i-- > 0;) {
				while (!stack.empty() && ranks[stack.back()] > ranks[i]) stack.pop_back();
				nss[i] = stack.empty() ? n : stack.back();
				stack.push_back(i);
			}
			return nss;
		};
		uint32 const top = *std::ranges::max_element(mapped) + 1;
		// the inverted order needs ranks only: sort, invert, and release the
		// arrays while the normal-order structures are built
		std::vector<uint32> nss_inverted;
		{
			std::vector<uint32> s(n + 1, 0), order(n + 1), rank(n);
			std::ranges::copy(mapped, s.begin()); // mapped symbols are >= 1: the sentinel 0 is unique
			detail::suffix_sort(s, order, top);
			for (uint32 r = 1; r <= n; ++r) rank[order[r]] = r;
			nss_inverted = next_smaller(rank);
		}
		for (auto &v : mapped) v = top - v;
		suffix_array const sa(mapped);
		std::vector<uint32> reversed(mapped.rbegin(), mapped.rend());
		suffix_array const sa_rev(reversed);

		auto const lcp_arr = sa.lcp_lengths();
		auto const lcs_arr = sa_rev.lcp_lengths();
		direct_rmq const lcp_rmq(lcp_arr);
		direct_rmq const lcs_rmq(lcs_arr);
		// longest common prefix of suffixes i, j of the text (i, j < n)
		auto const lcp = [&](uint32 i, uint32 j) {
			uint32 lo = sa.rank(i), hi = sa.rank(j);
			if (lo > hi) std::swap(lo, hi);
			return lcp_rmq.fold(lo, hi);
		};
		// longest common suffix of prefixes [0, i) and [0, j) (i, j > 0)
		auto const lcs = [&](uint32 i, uint32 j) {
			uint32 lo = sa_rev.rank(n - i), hi = sa_rev.rank(n - j);
			if (lo > hi) std::swap(lo, hi);
			return lcs_rmq.fold(lo, hi);
		};

		std::vector<run> found;
		for (int pass = 0; pass < 2; ++pass) {
			auto const nss = pass == 0 ? next_smaller(sa.ranks()) : std::move(nss_inverted);
			for (uint32 i = 0; i < n; ++i) {
				uint32 const j = nss[i];
				uint32 const period = j - i;
				uint32 const left = i == 0 ? 0 : lcs(i, j);
				uint32 const right = j == n ? 0 : lcp(i, j);
				if (right + left >= period) // total length (j + right) - (i - left) >= 2 * period
					found.push_back({period, i - left, j + right});
			}
		}
		// (period, l) determines a run, so a two-pass counting radix gives the
		// (period, l, r) order and dedup adjacency in O(n)
		{
			std::vector<run> tmp(found.size());
			std::vector<std::size_t> count(n + 2, 0);
			for (auto const &x : found) ++count[x.l + 1];
			for (uint32 i = 1; i < uint32(count.size()); ++i) count[i] += count[i - 1];
			for (auto const &x : found) tmp[count[x.l]++] = x;
			count.assign(n + 2, 0);
			for (auto const &x : tmp) ++count[x.period + 1];
			for (uint32 i = 1; i < uint32(count.size()); ++i) count[i] += count[i - 1];
			for (auto const &x : tmp) found[count[x.period]++] = x;
		}
		auto const dead = std::ranges::unique(found);
		found.erase(dead.begin(), dead.end());
		return found;
	}
}
