#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <vector>

namespace atl {
	/**
	 * @brief Build-once palindromic-radius oracle over a text (Manacher's
	 *        algorithm, O(n) build): the longest palindrome around every
	 *        center — @ref atl::palindrome_lengths::odd_length / @ref atl::palindrome_lengths::even_length — and the O(1)
	 *        @ref atl::palindrome_lengths::is_palindrome check for any substring. Aggregates (the
	 *        longest palindromic substring, total palindrome counts) are
	 *        one-line caller folds over the accessors, deliberately not API.
	 *
	 * This is the lightweight counterpart to @ref atl::palindromic_tree: one flat uint32
	 * radius array over the `2n + 1` virtual centers of the `#`-interleaved
	 * text (~8 B per symbol, no nodes, text not retained), built in a single
	 * pass, giving per-center geometry where the palindromic_tree gives the
	 * distinct-palindrome structure (counts, links, online build). Symbols
	 * only need equality under the projection, so any value type works and
	 * there is no alphabet bound.
	 */
	class palindrome_lengths {
		/**
		 * @brief One radius per center of the virtual `#`-interleaved text
		 * `#s0#s1#...#`: entry `j` covers the real center `j/2` (a symbol at
		 * odd `j`, the gap preceding it at even `j`), and its value is exactly
		 * the REAL length of the longest palindrome there — `2n + 1` entries,
		 * one pass, no parity split anywhere.
		 */
		std::vector<uint32> length_;

	public:
		/** @brief The empty text. */
		palindrome_lengths() = default;

		/**
		 * @brief Build from @p text via @p proj, read in place — nothing is copied.
		 * @tparam R A sized random-access range with equality-comparable projected symbols.
		 * @tparam Proj A projection invocable on range references.
		 * @param text Symbols read in place; no references are retained.
		 * @param proj Stable projection defining an equivalence relation.
		 * @pre size(text) < 2^31 (asserted).
		 * @complexity O(n) time and storage; iterative, with O(1) auxiliary stack space.
		 */
		template<typename R, typename Proj = std::identity>
			requires comparable_text<R, Proj>
		constexpr explicit palindrome_lengths(R &&text, Proj proj = {}) {
			auto const first = std::ranges::begin(text);
			auto const symbol = [&](uint32 i) -> decltype(auto) { return std::invoke(proj, first[i]); };
			// Virtual centers occupy 2n + 1 uint32-indexed entries.
			atl_assert(std::ranges::size(text) < (uint64(1) << 31));
			uint32 const n = uint32(std::ranges::size(text));
			if (n == 0) return;
			uint32 const m = 2 * n + 1;
			length_.assign(m, 0);
			// mirror the radius inside the rightmost known palindrome [l, r]
			// (virtual coordinates), then expand; separator positions (even
			// virtual index) compare equal without touching the symbols
			uint32 l = 0, r = 0;
			for (uint32 j = 1; j + 1 < m; ++j) {
				// mirror of j in [l, r], associated so no intermediate exceeds r
				uint32 k = j < r ? (std::min)(length_[l + (r - j)], r - j) : 0;
				while (k < j && k < m - j - 1) {
					uint32 const left = j - k - 1;
					if (left % 2 != 0 && symbol(left / 2) != symbol((j + k + 1) / 2)) break;
					++k;
				}
				length_[j] = k;
				if (j + k > r) {
					l = j - k;
					r = j + k;
				}
			}
		}

		/**
		 * @brief Length of the longest palindrome centered at position @p center
		 * (odd, >= 1).
		 * @param center Symbol position.
		 * @pre center < size() (asserted).
		 * @complexity O(1).
		 */
		[[nodiscard]] constexpr uint32 odd_length(uint32 center) const {
			atl_assert(center < size());
			return length_[2 * center + 1];
		}

		/**
		 * @brief Length of the longest palindrome centered between positions
		 * `gap - 1` and @p gap (even, >= 0).
		 * @param gap Interior gap position.
		 * @pre 1 <= gap < size() (asserted).
		 * @complexity O(1).
		 */
		[[nodiscard]] constexpr uint32 even_length(uint32 gap) const {
			atl_assert(gap >= 1 && gap < size());
			return length_[std::size_t(2) * gap];
		}

		/**
		 * @brief Whether the substring `[l, r)` is a palindrome; the empty substring is.
		 * @param l Inclusive substring start.
		 * @param r Exclusive substring end.
		 * @pre l <= r <= size() (asserted).
		 * @complexity O(1)
		 */
		[[nodiscard]] constexpr bool is_palindrome(uint32 l, uint32 r) const {
			atl_assert(l <= r && r <= size());
			return r - l <= 1 || length_[l + r] >= r - l; // virtual center of [l, r) is l + r
		}

		/** @brief Number of symbols in the indexed text. */
		constexpr std::size_t size() const { return length_.size() / 2; }

		/** @brief Whether the indexed text has no symbols. */
		constexpr bool empty() const { return length_.empty(); }
	};
}
