#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief SA-IS induced sorting (Nong–Zhang–Chan), one recursion level:
		 *        sorts all suffixes of @p s into @p sa. @p s ends with a
		 *        unique smallest sentinel 0 and has values in `[0, sigma)`;
		 *        @p sa includes the sentinel suffix (at rank 0).
		 * @pre `s` is nonempty, `s.size() == sa.size() < 2^32` (so every
		 *      position stays below the `nil` marker), `s.back() == 0`.
		 *
		 * The reduced problem halves each level (at most every other position
		 * is LMS), so total work and scratch are linear. Buckets are
		 * recomputed from a shared count array instead of being stored twice.
		 */
		constexpr void suffix_sort(std::span<uint32 const> s, std::span<uint32> sa, uint32 sigma) {
			atl_assert(!s.empty() && s.size() == sa.size() && s.back() == 0);
			atl_assert(s.size() < (uint64(1) << 32));
			uint32 const n = uint32(s.size());
			if (n == 1) {
				sa[0] = 0;
				return;
			}
			constexpr uint32 nil = uint32(-1);
			// types: S-type iff smaller suffix follows; the sentinel is S
			std::vector<bool> s_type(n);
			s_type[n - 1] = true;
			for (uint32 i = n - 1; i-- > 0;)
				s_type[i] = s[i] < s[i + 1] || (s[i] == s[i + 1] && s_type[i + 1]);
			auto const is_lms = [&](uint32 i) { return i > 0 && s_type[i] && !s_type[i - 1]; };

			std::vector<uint32> count(sigma, 0);
			for (uint32 const c : s) ++count[c];
			std::vector<uint32> bucket(sigma);
			auto const bucket_tails = [&] {
				uint32 sum = 0;
				for (uint32 c = 0; c < sigma; ++c) bucket[c] = sum += count[c];
			};
			auto const bucket_heads = [&] {
				uint32 sum = 0;
				for (uint32 c = 0; c < sigma; ++c) {
					bucket[c] = sum;
					sum += count[c];
				}
			};

			// induced sort: L-types left-to-right from bucket heads, then
			// S-types right-to-left from bucket tails
			auto const induce = [&] {
				bucket_heads();
				for (uint32 i = 0; i < n; ++i) {
					uint32 const j = sa[i];
					if (j != nil && j > 0 && !s_type[j - 1]) sa[bucket[s[j - 1]]++] = j - 1;
				}
				bucket_tails();
				for (uint32 i = n; i-- > 0;) {
					uint32 const j = sa[i];
					if (j != nil && j > 0 && s_type[j - 1]) sa[--bucket[s[j - 1]]] = j - 1;
				}
			};

			// pass 1: place LMS positions at their bucket tails (arbitrary
			// order) and induce — this sorts the LMS *substrings*
			std::ranges::fill(sa, nil);
			bucket_tails();
			for (uint32 i = 1; i < n; ++i)
				if (is_lms(i)) sa[--bucket[s[i]]] = i;
			induce();

			std::vector<uint32> lms;
			for (uint32 i = 1; i < n; ++i)
				if (is_lms(i)) lms.push_back(i);
			uint32 const m = uint32(lms.size());
			std::vector<uint32> lms_index(n, nil);
			for (uint32 k = 0; k < m; ++k) lms_index[lms[k]] = k;
			std::vector<uint32> name_of(m);
			uint32 names = 0;
			uint32 preceding = nil;
			for (uint32 i = 0; i < n; ++i) {
				uint32 const j = sa[i];
				if (j == nil || !is_lms(j)) continue;
				if (preceding != nil) {
					// equal iff the whole LMS substrings (through the next
					// LMS position) match symbol by symbol
					uint32 const end_j = lms_index[j] + 1 < m ? lms[lms_index[j] + 1] : n - 1;
					uint32 const end_p = lms_index[preceding] + 1 < m ? lms[lms_index[preceding] + 1] : n - 1;
					bool is_equal = end_j - j == end_p - preceding;
					for (uint32 d = 0; is_equal && d <= end_j - j; ++d) is_equal = s[j + d] == s[preceding + d];
					names += !is_equal;
				}
				name_of[lms_index[j]] = names;
				preceding = j;
			}

			// order the LMS suffixes: directly if names are unique, else by
			// recursing on the named sequence (its sentinel is the text
			// sentinel's LMS, named 0)
			std::vector<uint32> lms_order(m);
			if (names + 1 == m) {
				for (uint32 k = 0; k < m; ++k) lms_order[name_of[k]] = k;
			} else {
				suffix_sort(name_of, lms_order, names + 1);
			}

			// pass 2: place LMS suffixes at bucket tails in sorted order and
			// induce again — now everything lands exactly
			std::ranges::fill(sa, nil);
			bucket_tails();
			for (uint32 k = m; k-- > 0;) {
				uint32 const p = lms[lms_order[k]];
				sa[--bucket[s[p]]] = p;
			}
			induce();
		}
	}

	/**
	 * @brief The suffix array of a text, with inverse ranks and the LCP array
	 *        — all built once at construction (SA-IS, O(n) for integer-mapped
	 *        alphabets) and stored as flat 32-bit arrays; the text itself is
	 *        not retained.
	 *
	 * Symbols are seen through a `std::ranges` projection and mapped to a
	 * dense integer alphabet for the linear construction: integral symbols
	 * whose value range is below `8 n` take a direct offset (no sorting
	 * anywhere; buckets use O(n) scratch with this bound); anything else — huge
	 * ranges, non-integral ordered types — is rank-compressed with one sort.
	 * Either way the build then runs on one flat `uint32` buffer.
	 *
	 * @ref atl::suffix_array::lcp_lengths entry `r` is the longest-common-prefix length of the
	 * rank-`r` and rank-`r+1` suffixes (Kasai; size `n - 1`). Classic
	 * follow-ups are one step away: LCP of arbitrary suffixes = a range-min
	 * over `lcp_lengths` between their ranks (feed it to `direct_rmq`);
	 * distinct substrings = `n(n+1)/2 - sum(lcp_lengths)`; pattern search =
	 * binary search over @ref atl::suffix_array::positions against the caller's text.
	 */
	class suffix_array {
		std::vector<uint32> sa_;
		std::vector<uint32> rank_;
		std::vector<uint32> lcp_;

	public:
		/** @brief The empty text. */
		constexpr suffix_array() = default;

		/**
		 * @brief Build from @p text via @p proj. @pre `size(text) < 2^32 - 1`.
		 * @param text Sized random-access text; no input references are retained.
		 * @param proj Stable projection to totally ordered symbols.
		 * @note SA-IS recursion has O(log n) depth; accessor spans survive until assignment or destruction.
		 * @complexity O(n) following the alphabet mapping (which is O(n) for
		 * compact-range integral symbols, one O(n log n) sort otherwise)
		 */
		template<typename R, typename Proj = std::identity>
			requires sortable_text<R, Proj>
		constexpr explicit suffix_array(R &&text, Proj proj = {}) {
			auto const first = std::ranges::begin(text);
			// 32-bit indexed throughout, including the sentinel slot at n
			atl_assert(std::ranges::size(text) < (uint64(1) << 32) - 1);
			uint32 const n = uint32(std::ranges::size(text));
			if (n == 0) return;

			std::vector<uint32> s(n + 1);
			s[n] = 0;
			uint32 sigma = 0;
			using symbol = text_symbol_t<R, Proj>;
			// wider-than-64-bit integers (int128) would truncate in the uint64
			// range computation, so they take compression
			if constexpr (std::integral<symbol> && sizeof(symbol) <= sizeof(uint64)) {
				auto lo = std::invoke(proj, first[0]), hi = lo;
				for (uint32 i = 1; i < n; ++i) {
					auto const v = std::invoke(proj, first[i]);
					lo = (std::min)(lo, v);
					hi = (std::max)(hi, v);
				}
				// Both the offset map and its alphabet bound must fit uint32: the mapped keys and
				// sigma = range + 2 must fit uint32, and 8n alone does not
				// imply that once n exceeds 2^29
				if (uint64 const range = uint64(hi) - uint64(lo);
					range < 8 * uint64(n) && range <= (uint64(1) << 32) - 3) {
					for (uint32 i = 0; i < n; ++i)
						s[i] = uint32(uint64(std::invoke(proj, first[i])) - uint64(lo) + 1);
					sigma = uint32(uint64(hi) - uint64(lo) + 2);
				} else {
					sigma = compress(text, proj, s);
				}
			} else {
				sigma = compress(text, proj, s);
			}

			std::vector<uint32> sa(n + 1);
			detail::suffix_sort(s, sa, sigma);
			sa.erase(sa.begin()); // drop the sentinel suffix in place: no second buffer
			sa_ = std::move(sa);

			rank_.resize(n);
			for (uint32 r = 0; r < n; ++r) rank_[sa_[r]] = r;

			// Kasai over the mapped buffer (mapping preserves comparisons).
			// The phi (permuted-LCP) form raced it: faster on random texts
			// (1e6 random26 16.7 vs 21.2 ms, 5e6 154 vs 226) but slower on
			// periodic ones (1e6 all-a 2.9 vs 1.8, 5e6 Fibonacci 118 vs 91),
			// so Kasai stays (x86-64 -O2, record
			// benchmarks/strings/suffix_array.cpp 2026-10-15).
			lcp_.assign(n - 1, 0);
			for (uint32 i = 0, k = 0; i < n; ++i) {
				if (rank_[i] + 1 == n) {
					k = 0;
					continue;
				}
				uint32 const j = sa_[rank_[i] + 1];
				while (k < n - i && k < n - j && s[i + k] == s[j + k]) ++k;
				lcp_[rank_[i]] = k;
				if (k > 0) --k;
			}
		}

		/**
		 * @brief The suffix array: entry `r` is the start position of the rank-`r`
		 * (r-th lexicographically smallest) suffix.
		 */
		[[nodiscard]] constexpr std::span<uint32 const> positions() const { return sa_; }

		/** @brief The inverse: entry `p` is the rank of the suffix starting at `p`. */
		[[nodiscard]] constexpr std::span<uint32 const> ranks() const { return rank_; }

		/**
		 * @brief Longest-common-prefix lengths of rank-adjacent suffixes: entry `r`
		 * covers ranks `r` and `r + 1` (size `size() == 0 ? 0 : size() - 1`).
		 */
		[[nodiscard]] constexpr std::span<uint32 const> lcp_lengths() const { return lcp_; }

		/** @brief Start position of the rank-@p r suffix, bounds-asserted. */
		[[nodiscard]] constexpr uint32 position(uint32 r) const {
			atl_assert(r < sa_.size());
			return sa_[r];
		}

		/** @brief Rank of the suffix starting at @p p, bounds-asserted. */
		[[nodiscard]] constexpr uint32 rank(uint32 p) const {
			atl_assert(p < rank_.size());
			return rank_[p];
		}

		/** @brief Number of nonempty suffixes. @complexity O(1) */
		constexpr std::size_t size() const { return sa_.size(); }

		/** @brief Whether the text is empty. @complexity O(1) */
		constexpr bool empty() const { return sa_.empty(); }

	private:
		/**
		 * @brief Rank-compress arbitrary ordered symbols into @p s (one sort).
		 *        Integral symbols are sorted as materialized `(symbol, index)`
		 *        pairs, so compares read no text; other symbols, possibly heavy,
		 *        sort an index permutation.
		 */
		template<typename R, typename Proj>
			requires sortable_text<R, Proj>
		static constexpr uint32 compress(R &&text, Proj &proj, std::vector<uint32> &s) {
			auto const first = std::ranges::begin(text);
			uint32 const n = uint32(std::ranges::size(text));
			using symbol = text_symbol_t<R, Proj>;
			if constexpr (std::integral<symbol>) {
				std::vector<std::pair<symbol, uint32>> keyed(n);
				for (uint32 i = 0; i < n; ++i) keyed[i] = {std::invoke(proj, first[i]), i};
				std::ranges::sort(keyed);
				uint32 name = 0;
				for (uint32 k = 0; k < n; ++k) {
					if (k > 0 && keyed[k - 1].first < keyed[k].first) ++name;
					s[keyed[k].second] = name + 1;
				}
				return name + 2;
			} else {
				std::vector<uint32> order(n);
				for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
				std::ranges::sort(order, [&](uint32 a, uint32 b) {
					return std::invoke(proj, first[a]) < std::invoke(proj, first[b]);
				});
				uint32 name = 0;
				for (uint32 k = 0; k < n; ++k) {
					if (k > 0 && std::invoke(proj, first[order[k - 1]]) < std::invoke(proj, first[order[k]]))
						++name;
					s[order[k]] = name + 1;
				}
				return name + 2;
			}
		}
	};
}
