#pragma once
#include <atl/ds/direct_rmq.hpp>
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
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

namespace atl {
	/**
	 * @brief Prefix search over a SORTED word list: the half-open
	 *        @ref atl::index_interval `[begin, end)` of words having a query as a prefix, in
	 *        O(|query| + log n) per query — the Manber–Myers boundary search.
	 *
	 * The words must be in lexicographic order of their projected symbols
	 * (asserted during the build under ATL_ASSERT); sort arbitrary input with
	 * @ref atl::lexicographic_order first. The build materializes the symbols once
	 * and computes every neighboring pair's LCP (O(L) total — an adjacent LCP
	 * never exceeds the shorter word); a @ref atl::direct_rmq over those answers
	 * the LCP of *any* two words in O(1), since sortedness makes it the range
	 * minimum. Each binary-search probe then resumes comparing at the
	 * boundary's already-matched length instead of rescanning — O(m + log n)
	 * per query, against O(m·log(degree + 1)) for a trie walk
	 * (@ref atl::static_trie::word_interval) and O(m·log n) for naive binary
	 * search.
	 *
	 * The fixed lower-bound descent uses two LCP arrays (8n extra bytes);
	 * range-min remains for the upper boundary. Symbols are reserved up front
	 * for every Symbol on a forward word range.
	 *
	 * Copying is disabled (the internal range-min views the LCP array);
	 * moving transfers the internal views. A moved-from object may only be
	 * assigned or destroyed. Projected symbols that borrow storage require their
	 * referents to remain valid and unchanged for the dictionary lifetime.
	 *
	 * @tparam Symbol The projected symbol type; deduced via CTAD from the
	 *         constructor (`sorted_dictionary dict(words, proj);`).
	 */
	template<std::totally_ordered Symbol>
	class sorted_dictionary {
		std::vector<Symbol> sym_;
		std::vector<uint32> word_end_{0}; // Offsets include the final end position.
		std::vector<uint32> neighbor_lcp_;
		direct_rmq<uint32> rmq_; // Borrows neighbor_lcp_.

		/** @brief Descent-interval LCPs: each midpoint belongs to one fixed descent interval. */
		std::vector<uint32> left_lcp_, right_lcp_;

		/**
		 * @brief Whether symbol scans advance on `==` and settle the side with
		 *        one `<` at the mismatch: measured faster for wide symbols
		 *        (uint64, tokens), slower for byte symbols, which keep the
		 *        two-`<` scan (benchmarks/strings/sorted_dictionary.cpp).
		 */
		static constexpr bool is_equality_scan = !(std::integral<Symbol> && sizeof(Symbol) == 1);

		[[nodiscard]] static constexpr bool is_same(Symbol const &a, Symbol const &b) {
			if constexpr (is_equality_scan)
				return a == b;
			else
				return !(a < b) && !(b < a);
		}

		/**
		 * @brief The fixed descent-tree LCPs:
		 *        every interval (l, r) the lower-boundary descent can visit,
		 *        from (0, n - 1), stores its midpoint's two boundary LCPs; each
		 *        index is the midpoint of at most one interval. O(n).
		 */
		constexpr void build_descent_lcp() {
			uint32 const n = uint32(word_end_.size() - 1);
			if (n < 3) return;
			left_lcp_.assign(n, 0);
			right_lcp_.assign(n, 0);
			std::vector<std::pair<uint32, uint32>> stack{{0, n - 1}};
			while (!stack.empty()) {
				auto const [l, r] = stack.back();
				stack.pop_back();
				if (r - l <= 1) continue;
				uint32 const mid = l + (r - l) / 2;
				left_lcp_[mid] = rmq_.fold(l, mid);
				right_lcp_[mid] = rmq_.fold(mid, r);
				stack.emplace_back(l, mid);
				stack.emplace_back(mid, r);
			}
		}

	public:
		/**
		 * @brief Build over @p words, lexicographically sorted under @p proj.
		 *
		 * Passing an rvalue container of words (not a view) with the identity
		 * projection moves their symbols into the dictionary, leaving the
		 * moved-from words behind; an lvalue range is only read.
		 *
		 * @param words The words, in order (asserted under ATL_ASSERT).
		 * @param proj Maps a word's element to its symbol.
		 * @pre @p words sorted by their projected symbols; fewer than `2^32`
		 *      words and symbols in total (asserted)
		 * @complexity O(L + n) over L total symbols
		 */
		template<typename R, typename Proj = std::identity>
			requires sortable_word_range<R, Proj> && std::same_as<word_symbol_t<R, Proj>, Symbol>
		constexpr explicit sorted_dictionary(R &&words, Proj proj = {}) {
			// the words of an rvalue container are ours: their symbols are moved, not copied
			constexpr bool owns_words = !std::is_lvalue_reference_v<R> && !std::ranges::view<std::remove_cvref_t<R>>;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(words) < (uint64(1) << 32));
				atl_assert(std::ranges::size(words) < word_end_.max_size());
				word_end_.reserve(std::size_t(std::ranges::size(words)) + 1);
			}
			if constexpr (std::ranges::forward_range<R>)
				sym_.reserve(detail::total_word_size(words));
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(word_end_.size() < (uint64(1) << 32));
				if constexpr (owns_words)
					detail::append_projected(sym_, std::move(w), proj);
				else
					detail::append_projected(sym_, w, proj);
				atl_assert(sym_.size() < (uint64(1) << 32)); // offsets, lcps and match lengths are uint32
				word_end_.push_back(uint32(sym_.size()));
			}
			atl_assert(word_end_.size() <= (uint64(1) << 32)); // word ids are uint32
			uint32 const n = uint32(word_end_.size() - 1);
			neighbor_lcp_.reserve(n == 0 ? 0 : n - 1);
			for (uint32 i = 0; i + 1 < n; ++i) {
				std::size_t const a = word_end_[i], b = word_end_[i + 1];
				std::size_t const la = b - a, lb = word_end_[i + 2] - b, limit = (std::min)(la, lb);
				std::size_t h = 0;
				while (h < limit && is_same(sym_[a + h], sym_[b + h])) ++h;
				// sortedness: word i must not exceed word i+1 where they differ
				atl_assert(h == la || (h < lb && sym_[a + h] < sym_[b + h]));
				neighbor_lcp_.push_back(uint32(h));
			}
			rmq_ = direct_rmq<uint32>(neighbor_lcp_);
			build_descent_lcp();
		}

		/** @brief Copying is disabled because the RMQ borrows the LCP buffer. */
		sorted_dictionary(sorted_dictionary const &) = delete;
		/** @brief Copy assignment is disabled. */
		sorted_dictionary &operator=(sorted_dictionary const &) = delete;
		/** @brief Transfer buffers and their RMQ view; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary(sorted_dictionary &&) = default;
		/** @brief Transfer ownership; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary &operator=(sorted_dictionary &&) = default;

		/**
		 * @brief The words having @p x as a prefix: the half-open index
		 *        interval `[begin, end)` (empty when `begin == end`; the
		 *        empty query matches every word).
		 * @param x The query text.
		 * @param proj Maps an element of @p x to the symbol compared against the words'.
		 * @pre `x.size() < 2^32` (asserted)
		 * @complexity O(m + log n) — every query symbol is read O(1) times;
		 *             the upper end is found from the LCP minima alone, with
		 *             no symbol reads
		 */
		template<typename Q, typename QProj = std::identity>
			requires sortable_text<Q, QProj> && std::totally_ordered_with<text_symbol_t<Q, QProj>, Symbol>
		[[nodiscard]] constexpr index_interval prefix_interval(Q &&x, QProj proj = {}) const {
			auto const first = std::ranges::begin(x);
			atl_assert(std::ranges::size(x) < (uint64(1) << 32)); // 32-bit match lengths
			uint32 const m = uint32(std::ranges::size(x));
			uint32 const n = uint32(word_end_.size() - 1);
			if (n == 0) return {0, 0};

			// resume comparing x against word i from matched length h;
			// side: -1 word < x (x not a prefix), 0 x prefix of word, +1 word > x
			auto const extend = [&](uint32 i, uint32 h) -> std::pair<uint32, int> {
				std::size_t const off = word_end_[i];
				uint32 const len = word_end_[i + 1] - word_end_[i];
				uint32 const limit = (std::min)(m, len);
				if constexpr (is_equality_scan) {
					while (h < limit && atl::detail::symbol_equal(sym_[off + h], std::invoke(proj, first[h]))) ++h;
					if (h == m) return {h, 0};
					if (h == len) return {h, -1}; // shorter word compares less than x
					return {h, atl::detail::symbol_less(sym_[off + h], std::invoke(proj, first[h])) ? -1 : +1};
				} else {
					for (; h < limit; ++h) {
						auto &&q = atl::detail::symbol_at(proj, first, h); // a reference: tokens are not copied
						if (atl::detail::symbol_less(sym_[off + h], q)) return {h, -1};
						if (atl::detail::symbol_less(q, sym_[off + h])) return {h, +1};
					}
					return h == m ? std::pair{h, 0} : std::pair{h, -1}; // shorter word compares less than x
				}
			};
			auto const e0 = extend(0, 0);
			auto const en = n > 1 ? extend(n - 1, 0) : e0;
			if (en.second < 0) return {n, n}; // every word compares less than x

			// begin: the first word with side >= 0 (side is nondecreasing over the
			// sorted words), and lcp(x, word begin), exact at every step; out: the
			// first word seen to sort following every x-prefixed word, n while none is
			uint32 begin = 0, begin_lcp = e0.first, out = en.second > 0 ? n - 1 : n;
			if (e0.second < 0) {
				uint32 l = 0, r = n - 1;				 // side(l) < 0 <= side(r)
				uint32 llcp = e0.first, rlcp = en.first; // lcp(x, boundary words)
				while (r - l > 1) {
					uint32 const mid = l + (r - l) / 2;
					if (llcp >= rlcp) {
						// lcp(word_l, word_mid): below llcp the divergence decides
						// against x's matched run; above it, mid sides with l
						uint32 const h = left_lcp_[mid];
						if (h < llcp) {
							r = mid;
							rlcp = h;
						} else if (h > llcp) {
							l = mid;
						} else if (auto const [lc, s] = extend(mid, llcp); s >= 0) {
							r = mid;
							rlcp = lc;
						} else {
							l = mid;
							llcp = lc;
						}
					} else {
						uint32 const h = right_lcp_[mid];
						if (h < rlcp) {
							l = mid;
							llcp = h;
						} else if (h > rlcp) {
							r = mid;
						} else if (auto const [lc, s] = extend(mid, rlcp); s >= 0) {
							r = mid;
							rlcp = lc;
						} else {
							l = mid;
							llcp = lc;
						}
					}
					if (rlcp < m) out = r; // word r diverges from x on the side following it
				}
				begin = r;
				begin_lcp = rlcp;
			}
			if (begin_lcp < m) return {begin, begin}; // a miss: no symbol reads beyond
			if (en.second == 0) return {begin, n};	  // x prefixes the last word
			// the interval is the run of words whose lcp with word `begin` is
			// >= m, ending at or below `out`: a short gallop serves narrow
			// intervals in a few folds, then bisection, reading no symbols
			uint32 lo = begin, hi = out; // word lo is in; hi == n or word hi is out
			for (uint32 step = 1; step <= 16 && step < hi - begin; step *= 2) {
				if (rmq_.fold(begin, begin + step) < m) {
					hi = begin + step;
					break;
				}
				lo = begin + step;
			}
			while (hi - lo > 1) {
				uint32 const mid = lo + (hi - lo) / 2;
				if (rmq_.fold(begin, mid) >= m)
					lo = mid;
				else
					hi = mid;
			}
			return {begin, hi};
		}

		/**
		 * @brief The number of words.
		 * @complexity O(1)
		 */
		constexpr std::size_t size() const { return word_end_.size() - 1; }

		/** @brief Whether there are no words. */
		constexpr bool empty() const { return size() == 0; }
	};

	/** @brief Deduces the symbol type from the words and the projection. */
	template<typename R, typename Proj = std::identity>
		requires sortable_word_range<R, Proj>
	sorted_dictionary(R &&, Proj = {})
		-> sorted_dictionary<word_symbol_t<R, Proj>>;
}
