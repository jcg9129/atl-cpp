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

/** @brief Benchmark-local algorithm contenders. */
namespace atl::baseline {
	/**
	 * @brief Repeated projection computes the longest proper border length for every nonempty prefix; entry zero is zero.
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
	[[nodiscard]] constexpr std::vector<uint32> border_lengths_by_repeated_projection(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		auto const symbol = [&](uint32 i) -> decltype(auto) { return std::invoke(proj, first[i]); };
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
	 * @brief Box expansion, including known mismatches, computes common-prefix lengths of the text and each suffix; entry zero is the text length.
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
	[[nodiscard]] constexpr std::vector<uint32> common_prefix_lengths_by_box_expansion(R &&text, Proj proj = {}) {
		auto const first = std::ranges::begin(text);
		auto const symbol = [&](uint32 i) -> decltype(auto) { return std::invoke(proj, first[i]); };
		atl_assert(std::ranges::size(text) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(text));
		std::vector<uint32> lcp(n);
		if (n == 0) return lcp;
		lcp[0] = n;
		// reuse the match inside the rightmost known prefix occurrence [l, r)
		for (uint32 i = 1, l = 0, r = 1; i < n; ++i) {
			uint32 k = i < r ? (std::min)(lcp[i - l], r - i) : 0;
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
	 * @param common_prefix A valid result of common_prefix_lengths_by_box_expansion.
	 * @pre Size is below 2^32; positive entries at i > 0 are at most n - i (asserted).
	 * @pre All entries describe one text's Z-array (documented only).
	 * @return The corresponding border lengths.
	 * @complexity O(n) time and result storage, O(1) auxiliary space.
	 */
	template<typename R>
		requires length_range<R>
	[[nodiscard]] constexpr std::vector<uint32> to_border_lengths_by_backward_sweep(R &&common_prefix) {
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
	 * @param border A valid result of border_lengths_by_repeated_projection.
	 * @pre Size is below 2^32 and entry i is at most i (asserted).
	 * @pre All entries describe one text's prefix function (documented only).
	 * @return The corresponding common-prefix lengths.
	 * @complexity O(n) time and storage.
	 */
	template<typename R>
		requires length_range<R>
	[[nodiscard]] constexpr std::vector<uint32> to_common_prefix_lengths_by_canonical_text(R &&border) {
		auto const first = std::ranges::begin(border);
		atl_assert(std::ranges::size(border) < (uint64(1) << 32));
		uint32 const n = uint32(std::ranges::size(border));
		std::vector<uint32> text(n);
		uint32 fresh = 0;
		for (uint32 i = 0; i < n; ++i) {
			atl_assert(first[i] <= i); // a proper border of text[0..i] is at most i long
			text[i] = first[i] > 0 ? text[first[i] - 1] : fresh++;
		}
		return common_prefix_lengths_by_box_expansion(text);
	}

	/**
	 * @brief Ascending start positions of all pattern matches, including overlaps; an empty pattern matches nowhere.
	 * @tparam Text A comparable_text under Proj.
	 * @tparam Pattern A comparable_text with symbols equality-comparable with Text's.
	 * @tparam Borders A length_range of uint32 entries.
	 * @tparam Proj A projection invocable on both ranges' references.
	 * @param text Text read in place.
	 * @param pattern Pattern read in place.
	 * @param borders The result of border_lengths_by_repeated_projection(pattern, proj).
	 * @param proj Stable projection defining symbol equality.
	 * @pre Text and pattern lengths are below 2^32; borders has the pattern's size (asserted).
	 * @pre Border values are correct for the projected pattern; projected equality is an equivalence relation.
	 * @return Matching uint32 positions; no input references are retained.
	 * @complexity O(text size + pattern size) time, O(matches) result storage, O(1) auxiliary space.
	 */
	template<typename Text, typename Pattern, typename Borders, typename Proj = std::identity>
		requires comparable_text<Text, Proj> && comparable_text<Pattern, Proj> && length_range<Borders> && std::equality_comparable_with<text_symbol_t<Text, Proj>, text_symbol_t<Pattern, Proj>>
	[[nodiscard]] constexpr std::vector<uint32> occurrences_by_border_scan(Text &&text, Pattern &&pattern, Borders &&borders, Proj proj = {}) {
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
		uint32 matched = 0;
		for (uint32 i = 0; i < n; ++i) {
			auto &&c = std::invoke(proj, text_first[i]);
			if (matched == m) matched = border_first[matched - 1]; // keep scanning past a match
			while (matched > 0 && !atl::detail::symbol_equal(std::invoke(proj, pattern_first[matched]), c)) matched = border_first[matched - 1];
			if (atl::detail::symbol_equal(std::invoke(proj, pattern_first[matched]), c)) ++matched;
			if (matched == m) starts.push_back(i + 1 - m);
		}
		return starts;
	}
}
