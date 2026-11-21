#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/lexicographic_order.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Word argsorts raced against the library atl::lexicographic_order:
 *        the stable MSD partition for wide symbols and LCP-aware mergesort.
 *        Each docstring explains its measured tradeoffs.
 */
/** @brief Alternative string algorithms retained for comparison and certification. */
namespace atl::experimental {
	namespace detail {
		/**
		 * @brief The symbols of a word collection, materialized exactly as the
		 *        library atl::lexicographic_order does, so a race times only
		 *        the sort.
		 */
		template<std::totally_ordered Symbol>
		struct word_buffer {
			std::vector<Symbol> sym;			  /**< @brief All words' symbols, back to back. */
			std::vector<std::size_t> word_end{0}; /**< @brief n + 1 offsets into `sym`. */

			/** @brief Number of words. */
			uint32 size() const { return uint32(word_end.size() - 1); }

			/** @brief Length of word @p w. */
			[[nodiscard]] std::size_t length(uint32 w) const { return word_end[w + 1] - word_end[w]; }
		};

		/** @brief Materialize @p words under @p proj into a `word_buffer`. */
		template<typename R, typename Proj>
			requires atl::sortable_word_range<R, Proj>
		[[nodiscard]] auto materialize(R &&words, Proj &proj) {
			word_buffer<atl::word_symbol_t<R, Proj>> buf;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(words) < (uint64(1) << 32));
				atl_assert(std::ranges::size(words) < buf.word_end.max_size());
				buf.word_end.reserve(std::size_t(std::ranges::size(words)) + 1);
			}
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(buf.word_end.size() < (uint64(1) << 32));
				atl::detail::append_projected(buf.sym, w, proj);
				buf.word_end.push_back(buf.sym.size());
			}
			atl_assert(buf.word_end.size() <= (uint64(1) << 32)); // word ids are uint32
			return buf;
		}
	}

	/**
	 * @brief The MSD partition word argsort for any totally ordered symbol:
	 *        stable, same order as the library atl::lexicographic_order.
	 *
	 * Byte-sized integral symbols take the library radix partition. Other
	 * symbols run the same slice scheme with a stable sort keyed on
	 * (ended, next symbol) in place of the counting pass: each slice skips its
	 * common prefix, then partitions; slices below 32 words finish with a
	 * stable comparison sort.
	 *
	 * The wide-symbol path LOST to the library stable sort on x86-64
	 * (benchmarks/strings/lexicographic_order.cpp, record of 2026-10-15,
	 * Xeon Platinum 8358, plain -O2 and -march=native): 76.3 vs
	 * 51.7 ms on 2e5 random int words and 92.5 vs 63.3 ms as uint64, 63.5
	 * vs 49.4 ms on heavy int duplicates, though it won 67.7 vs 155 ms on
	 * 2e4 int words sharing a 500-symbol prefix (122 vs 243 as uint64). A
	 * loss on random words keeps the stable sort. Retained for comparison.
	 * @param words Input range of sized random-access words.
	 * @param proj Stable projection to ordered symbols.
	 * @return Stable word permutation.
	 * @note Not constexpr (uses std::ranges::stable_sort).
	 * @pre fewer than 2^32 words
	 * @complexity O(L log n) symbol compares with stable-sort scratch;
	 * its low-memory fallback can use O(n log-squared n) word compares
	 */
	template<typename R, typename Proj = std::identity>
		requires atl::sortable_word_range<R, Proj>
	[[nodiscard]] std::vector<uint32> lexicographic_order_by_msd_partition(R &&words, Proj proj = {}) {
		using symbol = atl::word_symbol_t<R, Proj>;
		auto const buf = detail::materialize(words, proj);
		if (buf.sym.empty()) {
			std::vector<uint32> order(buf.size());
			for (uint32 i = 0; i < buf.size(); ++i) order[i] = i;
			return order;
		}
		if constexpr (std::integral<symbol> && sizeof(symbol) == 1) {
			return atl::detail::radix_word_order(buf.sym, buf.word_end);
		} else {
			uint32 const n = buf.size();
			std::vector<uint32> order(n);
			for (std::size_t index = 0; index < order.size(); ++index) order[index] = uint32(index);
			auto const symbol_at = [&](uint32 w, std::size_t d) -> symbol const & { return buf.sym[buf.word_end[w] + d]; };
			constexpr uint32 small_slice = 32;

			/** @brief A pending partition step: `order[lo, hi)` share their first `depth` symbols. */
			struct slice {
				uint32 lo, hi;
				std::size_t depth;
			};

			std::vector<slice> stack;
			if (n > 1) stack.push_back({0, n, 0});
			while (!stack.empty()) {
				auto const [lo, hi, slice_depth] = stack.back();
				stack.pop_back();
				auto const first = order.begin() + lo, last = order.begin() + hi;
				if (hi - lo < small_slice) {
					std::ranges::stable_sort(first, last, [&](uint32 a, uint32 b) {
						return std::lexicographical_compare(buf.sym.data() + buf.word_end[a] + slice_depth, buf.sym.data() + buf.word_end[a + 1],
															buf.sym.data() + buf.word_end[b] + slice_depth, buf.sym.data() + buf.word_end[b + 1]);
					});
					continue;
				}
				// skip the slice's common prefix in one sequential pass per word,
				// so shared-prefix collections do not pay a partition per level
				std::size_t depth = slice_depth;
				{
					uint32 const w0 = order[lo];
					std::size_t common = buf.length(w0) - depth;
					for (uint32 i = lo + 1; i < hi && common > 0; ++i) {
						uint32 const w = order[i];
						auto const *const a = buf.sym.data() + buf.word_end[w0] + depth;
						auto const *const b = buf.sym.data() + buf.word_end[w] + depth;
						std::size_t const limit = (std::min)(common, buf.length(w) - depth);
						common = std::size_t(std::mismatch(a, a + limit, b).first - a);
					}
					depth += common;
				}
				{
					auto const key_less = [&](uint32 a, uint32 b) {
						bool const has_a_ended = buf.length(a) == depth, has_b_ended = buf.length(b) == depth;
						if (has_a_ended || has_b_ended) return has_a_ended && !has_b_ended;
						return symbol_at(a, depth) < symbol_at(b, depth);
					};
					std::ranges::stable_sort(first, last, key_less);
					uint32 run = lo;
					while (run < hi && buf.length(order[run]) == depth) ++run;
					while (run < hi) {
						uint32 next = run + 1;
						while (next < hi && !key_less(order[run], order[next])) ++next;
						if (next - run > 1) stack.push_back({run, next, depth + 1});
						run = next;
					}
				}
			}
			return order;
		}
	}

	/**
	 * @brief The LCP-aware stable mergesort word argsort (Ng and Kakehi's
	 *        string mergesort): same order as the library
	 *        atl::lexicographic_order.
	 *
	 * Each sorted run carries, per word, its LCP with its run predecessor.
	 * A merge step compares the two heads' LCPs with the last output word
	 * first, and reads symbols only when they tie, resuming at that LCP;
	 * so symbol reads are bounded by O(L + n log n) instead of
	 * O(L log n). Bottom-up, ping-ponging between two (word, lcp) buffers.
	 *
	 * Raced against the library stable sort for wide symbols
	 * (benchmarks/strings/lexicographic_order.cpp, record of 2026-10-15,
	 * x86-64 Xeon Platinum 8358, plain -O2 and -march=native): it
	 * LOST on random words, 58.1 vs 51.7 ms for 2e5 int words and 74.2 vs
	 * 63.3 as uint64, and won on heavy duplicates (39.6 vs 49.4) and shared
	 * 500-symbol prefixes (72.1 vs 155 int, 127 vs 243 uint64). For `char`
	 * words it trails the library radix partition everywhere (48.1 vs 13.3
	 * random26, 16.8 vs 11.7 prefix500). Retained for comparison.
	 * @param words Input range of sized random-access words.
	 * @param proj Stable projection to ordered symbols.
	 * @return Stable word permutation.
	 * @note Not constexpr (shares the runtime-only materialization interface).
	 * @pre fewer than 2^32 words, fewer than 2^32 symbols
	 * @complexity O(L + n log n) symbol reads and O(n log n) moves
	 */
	template<typename R, typename Proj = std::identity>
		requires atl::sortable_word_range<R, Proj>
	[[nodiscard]] std::vector<uint32> lexicographic_order_by_lcp_merge(R &&words, Proj proj = {}) {
		auto const buf = detail::materialize(words, proj);
		if (buf.sym.empty()) {
			std::vector<uint32> order(buf.size());
			for (uint32 i = 0; i < buf.size(); ++i) order[i] = i;
			return order;
		}
		atl_assert(buf.sym.size() < (uint64(1) << 32)); // lcps are uint32
		uint32 const n = buf.size();

		/** @brief A word in a sorted run and its LCP with its run predecessor. */
		struct entry {
			uint32 word, lcp;
		};

		std::vector<entry> from(n), to(n);
		for (uint32 i = 0; i < n; ++i) from[i] = {i, 0};

		// lcp of words a and b from h on, and whether a <= b
		auto const compare_from = [&](uint32 a, uint32 b, uint32 h) -> std::pair<uint32, bool> {
			// Indexed reads: a Boolean symbol buffer is a bit-packed vector with no data pointer.
			auto const offset_a = buf.word_end[a], offset_b = buf.word_end[b];
			auto const la = uint32(buf.length(a)), lb = uint32(buf.length(b));
			uint32 const stop = (std::min)(la, lb);
			while (h < stop && !(buf.sym[offset_a + h] < buf.sym[offset_b + h]) && !(buf.sym[offset_b + h] < buf.sym[offset_a + h])) ++h;
			return {h, h == la || (h < lb && buf.sym[offset_a + h] < buf.sym[offset_b + h])};
		};

		for (uint64 width = 1; width < n; width *= 2) {
			for (uint64 offset = 0; offset < n; offset += 2 * width) {
				uint32 const lo = uint32(offset);
				uint32 const mid = uint32((std::min)(uint64(n), offset + width));
				uint32 const hi = uint32((std::min)(uint64(n), offset + 2 * width));
				uint32 i = lo, j = mid, out = lo;
				// heads' lcps with the last output word; at the start the
				// "last output" is undefined, forcing a full compare
				uint32 li = 0, lj = 0;
				while (i < mid && j < hi) {
					bool is_left = false;
					uint32 kept = 0; // lcp with the new last output of the head not taken
					if (li > lj) {
						is_left = true;
						kept = lj;
					} else if (li < lj) {
						is_left = false;
						kept = li;
					} else {
						auto const [h, is_left_first] = compare_from(from[i].word, from[j].word, li);
						is_left = is_left_first;
						kept = h;
					}
					if (is_left) {
						to[out++] = {from[i].word, li};
						lj = kept;
						if (++i < mid) li = from[i].lcp;
					} else {
						to[out++] = {from[j].word, lj};
						li = kept;
						if (++j < hi) lj = from[j].lcp;
					}
				}
				if (i < mid) {
					to[out++] = {from[i].word, li};
					while (++i < mid) to[out++] = from[i];
				}
				if (j < hi) {
					to[out++] = {from[j].word, lj};
					while (++j < hi) to[out++] = from[j];
				}
			}
			std::swap(from, to);
		}
		std::vector<uint32> order(n);
		for (uint32 i = 0; i < n; ++i) order[i] = from[i].word;
		return order;
	}
}
