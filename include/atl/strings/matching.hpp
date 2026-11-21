#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <vector>

namespace atl {
	/**
	 * @brief Longest proper border length for every nonempty prefix; entry zero is zero.
	 * @tparam R A sized random-access range with equality-comparable projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param text Symbols read in place; no references are retained.
	 * @param proj Stable projection defining symbol equality.
	 * @pre The text length is below 2^32 (asserted). Projected equality is an equivalence relation.
	 * @return One uint32 length per input symbol, or an empty vector.
	 * @complexity O(n) comparisons and projection calls, O(n) result storage, O(1) auxiliary space.
	 */
	template<typename R, typename Proj = std::identity>
		requires comparable_text<R, Proj>
	[[nodiscard]] constexpr std::vector<uint32> border_lengths(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		auto const symbol = [&](uint32 i) -> decltype(auto) { return atl::detail::symbol_at(proj, first, i); };
		atl_assert(std::ranges::size(text) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(text));
		std::vector<uint32> border(n);
		for (uint32 i = 1; i < n; ++i) {
			uint32 k = border[i - 1];
			while (k > 0 && symbol(i) != symbol(k)) k = border[k - 1];
			if (symbol(i) == symbol(k)) ++k;
			border[i] = k;
		}
		return border;
	}

	/**
	 * @brief Common-prefix lengths of the text and each suffix; entry zero is the text length.
	 * @tparam R A sized random-access range with equality-comparable projected symbols.
	 * @tparam Proj A projection invocable on range references.
	 * @param text Symbols read in place; no references are retained.
	 * @param proj Stable projection defining symbol equality.
	 * @pre The text length is below 2^32 (asserted). Projected equality is an equivalence relation.
	 * @return One uint32 length per input symbol, or an empty vector.
	 * @complexity O(n) comparisons and projection calls, O(n) result storage, O(1) auxiliary space.
	 */
	template<typename R, typename Proj = std::identity>
		requires comparable_text<R, Proj>
	[[nodiscard]] constexpr std::vector<uint32> common_prefix_lengths(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		auto const symbol = [&](uint32 i) -> decltype(auto) { return atl::detail::symbol_at(proj, first, i); };
		atl_assert(std::ranges::size(text) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(text));
		std::vector<uint32> lcp(n);
		if (n == 0) return lcp;
		lcp[0] = n;
		// reuse the match inside the rightmost known prefix occurrence [l, r)
		for (uint32 i = 1, l = 0, r = 1; i < n; ++i) {
			uint32 k = i < r ? (std::min)(lcp[i - l], r - i) : 0;
			// strictly inside the box the mirrored value is exact: skip the
			// compare that is bound to fail
			if (i + k < r) {
				lcp[i] = k;
				continue;
			}
			while (i + k < n && symbol(k) == symbol(i + k)) ++k;
			lcp[i] = k;
			if (i + k > r) {
				l = i;
				r = i + k;
			}
		}
		return lcp;
	}

	/**
	 * @brief Convert a valid Z-array to its border-length array without the text.
	 * @tparam R A sized random-access range of uint32 lengths.
	 * @param common_prefix A valid result of atl::common_prefix_lengths.
	 * @pre Size is below 2^32; positive entries at i > 0 are at most n - i (asserted).
	 * @pre All entries describe one text's Z-array (documented only).
	 * @return The corresponding border lengths.
	 * @complexity O(n) time and result storage, O(1) auxiliary space.
	 */
	template<typename R>
		requires length_range<R>
	[[nodiscard]] constexpr std::vector<uint32> to_border_lengths(R &&common_prefix) {
		auto const first = std::ranges::begin(common_prefix);
		atl_assert(std::ranges::size(common_prefix) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(common_prefix));
		std::vector<uint32> border(n, 0);
		if (n == 0) return border;
		for (uint32 i = 1; i < n; ++i)
			if (uint32 const z = first[i]; z > 0) {
				atl_assert(z <= n - i); // a prefix occurrence cannot run past the end
				uint32 const last = i + z - 1;
				border[last] = (std::max)(border[last], z);
			}
		for (uint32 i = n - 1; i-- > 0;)
			if (border[i + 1] > 0) border[i] = (std::max)(border[i], border[i + 1] - 1);
		return border;
	}

	/**
	 * @brief Convert valid border lengths to their Z-array through a canonical integer text.
	 * @tparam R A sized random-access range of uint32 lengths.
	 * @param border A valid result of atl::border_lengths.
	 * @pre Size is below 2^32 and entry i is at most i (asserted).
	 * @pre All entries describe one text's prefix function (documented only).
	 * @return The corresponding common-prefix lengths.
	 * @complexity O(n) time and storage.
	 */
	template<typename R>
		requires length_range<R>
	[[nodiscard]] constexpr std::vector<uint32> to_common_prefix_lengths(R &&border) {
		auto const first = std::ranges::begin(border);
		atl_assert(std::ranges::size(border) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(border));
		std::vector<uint32> text(n);
		uint32 fresh = 0;
		for (uint32 i = 0; i < n; ++i) {
			atl_assert(first[i] <= i); // a proper border of text[0..i] is at most i long
			text[i] = first[i] > 0 ? text[first[i] - 1] : fresh++;
		}
		return common_prefix_lengths(text);
	}

	/**
	 * @brief Ascending start positions of all pattern matches, including overlaps; an empty pattern matches nowhere.
	 * @tparam Text A comparable_text under Proj.
	 * @tparam Pattern A comparable_text with symbols equality-comparable with Text's.
	 * @tparam Borders A length_range of uint32 entries.
	 * @tparam Proj A projection invocable on both ranges' references.
	 * @param text Text read in place.
	 * @param pattern Pattern read in place.
	 * @param borders The result of atl::border_lengths(pattern, proj).
	 * @param proj Stable projection defining symbol equality.
	 * @pre Text and pattern lengths are below 2^32; borders has the pattern's size, and each
	 *      entry i the scan reads is at most i (asserted).
	 * @pre Border values are correct for the projected pattern; projected equality is an equivalence relation.
	 * @return Matching uint32 positions; no input references are retained.
	 * @complexity O(text size + pattern size) time, O(matches) result storage, O(1) auxiliary space.
	 */
	template<typename Text, typename Pattern, typename Borders, typename Proj = std::identity>
		requires comparable_text<Text, Proj> && comparable_text<Pattern, Proj> && length_range<Borders> && std::equality_comparable_with<text_symbol_t<Text, Proj>, text_symbol_t<Pattern, Proj>>
	[[nodiscard]] constexpr std::vector<uint32> occurrences(Text &&text, Pattern &&pattern, Borders &&borders, Proj proj = {}) {
		auto const text_first = std::ranges::begin(text);
		auto const pattern_first = std::ranges::begin(pattern);
		atl_assert(std::ranges::size(text) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(text));
		atl_assert(std::ranges::size(pattern) < (uint64(1) << 32));
		uint32 const m = uint32(std::ranges::size(pattern));
		auto const border_first = std::ranges::begin(borders);
		atl_assert(std::ranges::size(borders) == m);
		std::vector<uint32> starts;
		if (m == 0 || m > n) return starts;
		// the longest proper border of the matched prefix; a border entry past its own index would
		// index the pattern out of bounds
		auto const fall_back = [&](uint32 const length) {
			uint32 const border = border_first[length - 1];
			atl_assert(border < length);
			return border;
		};
		uint32 matched = 0;
		for (uint32 i = 0; i < n; ++i) {
			auto &&c = atl::detail::symbol_at(proj, text_first, i);
			// matched < m on entry: a full match falls back at once, so each pattern read is in bounds
			while (true) {
				if (atl::detail::symbol_equal(std::invoke(proj, pattern_first[matched]), c)) {
					if (++matched == m) {
						starts.push_back(i + 1 - m);
						matched = fall_back(matched); // keep scanning past a match
					}
					break;
				}
				if (matched == 0) break;
				matched = fall_back(matched);
			}
		}
		return starts;
	}
}
