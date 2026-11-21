#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

namespace atl {
	namespace detail {
		/**
		 * @brief The MSD radix word argsort over materialized byte-sized
		 *        integral symbols: stable, `<` on the symbols.
		 *
		 * Each slice of words sharing their first `depth` symbols first skips
		 * its common prefix (one sequential mismatch scan per word), then is
		 * distributed by its next symbol, ended words first, in a stable
		 * 257-bucket counting pass. Slices enter in input order and every
		 * pass is stable, so equal words keep their input order. Slices below
		 * 32 words finish with a stable comparison sort resuming at their
		 * depth.
		 * @param sym all words' symbols, back to back
		 * @param word_end n + 1 offsets into @p sym
		 * @complexity O(L + 257 · branching slices) plus the compact-slice sorts,
		 *             L the total symbol count
		 */
		template<typename Symbol>
			requires(std::integral<Symbol> && sizeof(Symbol) == 1)
		std::vector<uint32> radix_word_order(std::vector<Symbol> const &sym, std::vector<std::size_t> const &word_end) {
			if constexpr (std::same_as<Symbol, bool>) {
				std::vector<uint8> bytes(sym.size());
				for (std::size_t i = 0; i < sym.size(); ++i) bytes[i] = uint8(sym[i]);
				return radix_word_order(bytes, word_end);
			} else {
				uint32 const n = uint32(word_end.size() - 1);
				std::vector<uint32> order(n);
				for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
				auto const length = [&](uint32 w) { return word_end[w + 1] - word_end[w]; };
				constexpr uint32 small_slice = 32;

				/** @brief A pending partition step: `order[lo, hi)` share their first `depth` symbols. */
				struct slice {
					uint32 lo, hi;
					std::size_t depth;
				};

				std::vector<slice> stack;
				std::vector<uint32> scratch(n);
				if (sym.empty()) return order;
				if (n > 1) stack.push_back({0, n, 0});
				while (!stack.empty()) {
					auto const [lo, hi, slice_depth] = stack.back();
					stack.pop_back();
					auto const first = order.begin() + lo, last = order.begin() + hi;
					if (hi - lo < small_slice) {
						std::ranges::stable_sort(first, last, [&](uint32 a, uint32 b) {
							return std::lexicographical_compare(sym.data() + word_end[a] + slice_depth, sym.data() + word_end[a + 1],
																sym.data() + word_end[b] + slice_depth, sym.data() + word_end[b + 1]);
						});
						continue;
					}
					// skip the slice's common prefix: shared-prefix collections
					// would otherwise pay one partition per level
					std::size_t depth = slice_depth;
					{
						uint32 const w0 = order[lo];
						std::size_t common = length(w0) - depth;
						for (uint32 i = lo + 1; i < hi && common > 0; ++i) {
							auto const *const a = sym.data() + word_end[w0] + depth;
							auto const *const b = sym.data() + word_end[order[i]] + depth;
							std::size_t const limit = (std::min)(common, length(order[i]) - depth);
							common = std::size_t(std::mismatch(a, a + limit, b).first - a);
						}
						depth += common;
					}
					// bucket 0: ended words; 1 + the byte, sign bit flipped for
					// signed symbols, so bucket order is `<` order
					auto const bucket = [&](uint32 w) -> uint32 {
						if (length(w) == depth) return 0;
						auto const byte = uint8(sym[word_end[w] + depth]);
						return 1 + uint32(std::is_signed_v<Symbol> ? byte ^ 0x80u : byte);
					};
					std::array<uint32, 258> start{};
					for (uint32 i = lo; i < hi; ++i) ++start[bucket(order[i]) + 1];
					std::partial_sum(start.begin(), start.end(), start.begin());
					auto next = start;
					for (uint32 i = lo; i < hi; ++i) scratch[lo + next[bucket(order[i])]++] = order[i];
					std::ranges::copy(scratch.begin() + lo, scratch.begin() + hi, first);
					for (uint32 b = 1; b < 257; ++b)
						if (start[b + 1] - start[b] > 1) stack.push_back({lo + start[b], lo + start[b + 1], depth + 1});
				}
				return order;
			}
		}
	}

	/**
	 * @brief Indices of @p words in lexicographic order of the words — the
	 *        argsort of a string collection. STABLE: equal words keep their
	 *        input order.
	 *
	 * The order is lexicographic by `<` over the *projected symbols* — the
	 * same convention as every other atl string tool. Note for `char` texts
	 * that this follows the platform's `char` signedness, which on
	 * signed-char targets (x86) differs from `std::string::operator<`
	 * (always unsigned, memcmp semantics) on non-ASCII bytes; project to
	 * `unsigned char` to pin byte order.
	 *
	 * The projected symbols are materialized once into one contiguous
	 * buffer, each word appended in bulk. Byte-sized integral symbols are
	 * then sorted by an MSD radix partition (`detail::radix_word_order`);
	 * every other symbol type by a stable sort of word compares. Raced on
	 * x86-64 (benchmarks/strings/lexicographic_order.cpp, record of
	 * 2026-10-15): for `char` words the radix partition beat the stable
	 * sort (random26 13.3 vs 44.4 ms, prefix500 11.7 vs 82.9 ms); for `int` and `uint64` words it lost on
	 * random words, as did an LCP-aware mergesort, so wide symbols keep the
	 * stable sort (losers in experimental/strings/lexicographic_order_variants.hpp).
	 * @param words An input range of sized random-access words.
	 * @param proj Stable projection to totally ordered symbols; evaluated once per symbol.
	 * @pre Fewer than 2^32 words; the symbol buffer and offsets fit size_t and vector limits.
	 * @return Stable permutation of word indices; no references are retained.
	 * @note Not constexpr (uses std::ranges::stable_sort).
	 * 	 * @complexity byte symbols: O(L) plus the per-slice bucket and
	 *             compact-slice work; otherwise O(compare volume · log n),
	 *             L log n with stable_sort buffer allocation; its low-memory fallback
	 *             can use O(n log-squared n) word comparisons
	 */
	template<typename R, typename Proj = std::identity>
		requires sortable_word_range<R, Proj>
	[[nodiscard]] std::vector<uint32> lexicographic_order(R &&words, Proj proj = {}) {
		using symbol = word_symbol_t<R, Proj>;

		std::vector<symbol> sym;
		std::vector<std::size_t> word_end{0};
		if constexpr (std::ranges::sized_range<R>) {
			atl_assert(std::ranges::size(words) < (uint64(1) << 32));
			atl_assert(std::ranges::size(words) < word_end.max_size());
			word_end.reserve(std::size_t(std::ranges::size(words)) + 1);
		}
		if constexpr (std::is_trivially_copyable_v<symbol> && std::ranges::forward_range<R>)
			sym.reserve(detail::total_word_size(words));
		for (auto &&w : atl::detail::input_elements(words)) {
			atl_assert(word_end.size() < (uint64(1) << 32));
			detail::append_projected(sym, w, proj);
			word_end.push_back(sym.size());
		}
		atl_assert(word_end.size() <= (uint64(1) << 32)); // word ids are uint32
		if constexpr (std::integral<symbol> && sizeof(symbol) == 1) {
			return detail::radix_word_order(sym, word_end);
		} else {
			uint32 const n = uint32(word_end.size() - 1);
			std::vector<uint32> order(n);
			for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
			// the iterator-pair spelling on purpose: std::ranges::lexicographical_compare
			// ranges compare took 123 vs 82.9 ms on prefix500 char words (GCC 15.2, x86-64, -O2, 2026-10-15)
			if (sym.empty()) return order;
			std::ranges::stable_sort(order, [&](uint32 a, uint32 b) {
				return std::lexicographical_compare(sym.data() + word_end[a], sym.data() + word_end[a + 1],
													sym.data() + word_end[b], sym.data() + word_end[b + 1]);
			});
			return order;
		}
	}
}
