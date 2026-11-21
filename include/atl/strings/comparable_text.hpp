#pragma once
#include <atl/strings/detail/symbols.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace atl {
	/** @brief The projected symbol type of a text range under `Proj`. */
	template<typename R, typename Proj = std::identity>
	using text_symbol_t = std::remove_cvref_t<std::invoke_result_t<Proj &, std::ranges::range_reference_t<R>>>;

	/**
	 * @brief The projected symbol type of the words in a word collection `R`
	 *        under `Proj`: @ref atl::text_symbol_t of `R`'s element type.
	 */
	template<typename R, typename Proj = std::identity>
	using word_symbol_t = text_symbol_t<std::ranges::range_reference_t<R>, Proj>;

	/**
	 * @brief A text usable by the comparison-based string algorithms
	 * (palindrome_lengths, border_lengths, common_prefix_lengths,
	 * occurrences): a sized random-access range whose projected symbols are
	 * equality-comparable — no integral alphabet required, since these
	 * algorithms only compare. Random access lets them read the text in
	 * place, with no symbol buffer.
	 */
	template<typename R, typename Proj = std::identity>
	concept comparable_text = std::ranges::random_access_range<R> && std::ranges::sized_range<R> && std::invocable<Proj &, std::ranges::range_reference_t<R>> && std::equality_comparable<text_symbol_t<R, Proj>>;

	/**
	 * @brief A text usable by the order-based string algorithms (suffix_array,
	 * lyndon_factorization, minimal_rotation_index, runs, sorted_dictionary,
	 * and per word lexicographic_order): as @ref atl::comparable_text, but the
	 * projected symbols must be totally ordered — these algorithms need `<`,
	 * not just `==`. Still no alphabet bound: suffix_array maps symbols to a
	 * dense integer alphabet internally.
	 */
	template<typename R, typename Proj = std::identity>
	concept sortable_text = comparable_text<R, Proj> && std::totally_ordered<text_symbol_t<R, Proj>>;

	/**
	 * @brief A word collection usable by @ref atl::lexicographic_order and
	 * @ref atl::sorted_dictionary: an input range whose elements are themselves
	 * @ref atl::sortable_text s under `Proj`. Each word being sized lets the
	 * consumers append its symbols in one bulk copy.
	 */
	template<typename R, typename Proj = std::identity>
	concept sortable_word_range =
		std::ranges::input_range<R> && sortable_text<std::ranges::range_reference_t<R>, Proj>;

	namespace detail {
		/**
		 * @brief Sum the symbol counts of a multi-pass word range without
		 *        evaluating its projection, for a single symbol-buffer reserve.
		 * @pre The total fits size_t (asserted).
		 * @complexity O(n) word sizes; no symbol reads.
		 */
		template<std::ranges::forward_range R>
			requires std::ranges::sized_range<std::ranges::range_reference_t<R>>
		[[nodiscard]] constexpr std::size_t total_word_size(R &&words) {
			std::size_t total = 0;
			for (auto &&word : atl::detail::input_elements(words)) {
				auto const length = std::ranges::size(word);
				atl_assert(length <= (std::numeric_limits<std::size_t>::max)() - total);
				total += std::size_t(length);
			}
			return total;
		}

		/**
		 * @brief Append the projected symbols of @p word to @p out in one
		 *        bulk step: a pointer-range insert (a memmove) for a
		 *        contiguous word under the identity projection, else one
		 *        resize plus a transform (one reserve plus emplace_backs for a
		 *        symbol that is not default constructible or not move
		 *        assignable, which resize and assignment would need). An rvalue word that
		 *        owns its elements (not a view) and holds non-trivially
		 *        copyable symbols under the identity projection has them
		 *        moved out, not copied.
		 * @complexity O(|word|)
		 */
		template<typename Symbol, typename Word, typename Proj>
			requires sortable_text<Word, Proj> && std::same_as<text_symbol_t<Word, Proj>, Symbol>
		constexpr void append_projected(std::vector<Symbol> &out, Word &&word, Proj &proj) {
			auto const count = std::ranges::size(word);
			atl_assert(count <= out.max_size() - out.size());
			auto const length = std::size_t(count);
			if (length == 0) return;
			constexpr bool is_movable = std::same_as<Proj, std::identity> && !std::is_lvalue_reference_v<Word>
									 && !std::ranges::view<std::remove_cvref_t<Word>>
									 && !std::is_trivially_copyable_v<Symbol>
									 && std::is_lvalue_reference_v<std::ranges::range_reference_t<Word>>
									 && !std::is_const_v<std::remove_reference_t<std::ranges::range_reference_t<Word>>>;
			if constexpr (is_movable) {
				out.reserve(out.size() + length);
				for (auto &e : atl::detail::input_elements(word)) out.push_back(std::move(e));
			} else if constexpr (std::same_as<Proj, std::identity> && std::ranges::contiguous_range<Word>
								 && std::is_copy_assignable_v<Symbol>) {
				auto const *const first = std::ranges::data(word);
				out.insert(out.end(), first, first + length);
			} else if constexpr (!std::default_initializable<Symbol> || !std::is_move_assignable_v<Symbol>) {
				out.reserve(out.size() + length);
				for (auto &&e : atl::detail::input_elements(word)) out.emplace_back(std::invoke(proj, std::forward<decltype(e)>(e)));
			} else {
				auto const offset = out.size();
				out.resize(offset + length);
				// A counted endpoint also accepts non-common random-access ranges.
				auto const first = std::ranges::begin(word);
				std::ranges::transform(first, first + std::ranges::range_difference_t<Word>(length),
									   out.begin() + std::ptrdiff_t(offset),
									   [&](auto &&e) -> Symbol { return std::invoke(proj, std::forward<decltype(e)>(e)); });
			}
		}
	}

	/**
	 * @brief A sequence of uint32 lengths — the shape of @ref atl::border_lengths /
	 *        @ref atl::common_prefix_lengths results, consumed by @ref atl::occurrences
	 *        and the array conversions: any sized random-access range of
	 *        `uint32`.
	 */
	template<typename R>
	concept length_range = std::ranges::random_access_range<R> && std::ranges::sized_range<R> && std::same_as<std::ranges::range_value_t<R>, uint32>;

	/**
	 * @brief A half-open interval `[begin, end)` of word positions, as
	 *        returned by @ref atl::static_trie::word_interval and
	 *        @ref atl::sorted_dictionary::prefix_interval. Empty when
	 *        `begin == end`.
	 */
	struct index_interval {
		uint32 begin; /**< @brief The first position in the interval. */
		uint32 end;	  /**< @brief One past the last position. */

		/** @brief Componentwise equality. */
		[[nodiscard]] constexpr bool operator==(index_interval const &) const = default;
	};
}
