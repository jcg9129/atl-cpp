#pragma once
#include <atl/ds/direct_rmq.hpp>
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/sorted_dictionary.hpp>
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
 * @brief Sorted prefix-search contenders: parts-built arrays with RMQ or fixed
 * descent LCPs, and member-built arrays with galloping or linear upper search.
 * Intel Xeon Platinum 8168 @ 2.70 GHz, GCC 15.2.0, RelWithDebInfo, assertions off,
 * 2026-11-03, -O2 -g -DNDEBUG -std=c++23, optionally -march=native.
 * Char 1e6 build: parts/RMQ 82.2/90.6 ms, parts/novector 82.2/90.9 ms,
 * parts/descent LCP 201/204 ms; library 97.7/97.1 ms.
 * The parts/RMQ configuration is the certifying oracle. Full tables are in
 * docs/sorted-dictionary.md.
 */
/** @brief Alternative string algorithms retained for comparison and certification. */
namespace atl::experimental {

	/**
	 * @brief The knobs of @ref atl::experimental::sorted_dictionary_by_parts_build; all off preserves the
	 *        parts-build parts build and RMQ descent (the test oracle).
	 */
	struct sorted_dictionary_options {
		/** @brief Disable GCC early-break vectorization in neighbor-LCP scans.
		 * Char 1e6 parts build is 82.2/90.6 ms versus novector 82.2/90.9 ms
		 * at -O2/native on Xeon Platinum 8168, GCC 15.2.0, 2026-11-03; no gain.
		 * The directive is a no-op on other compilers. */
		bool is_novector = false;
		/** @brief Reserve all symbols on forward word ranges. */
		bool is_sym_reserved = false;
		/** @brief Scan up to 16 adjacent LCP entries, then bisect.
		 * No consistent win: inplace char misses are 219/232 ms with gallops
		 * versus 226/240 ms with the scan (-O2/native, Xeon Platinum 8168,
		 * GCC 15.2.0, 2026-11-03), despite lower hit times. */
		bool is_linear_upper_scan = false;
		/** @brief Fixed-descent boundary LCP arrays, costing 8n bytes.
		 * Char 1e6 parts build costs 201/204 ms versus RMQ descent 82.2/90.6 ms
		 * (-O2/native, Xeon Platinum 8168, GCC 15.2.0, 2026-11-03). */
		bool is_descent_lcp = false;

		/** @brief Knob-wise equality. */
		constexpr bool operator==(sorted_dictionary_options const &) const = default;
	};

	/**
	 * @brief An experimental copy of atl::sorted_dictionary with the alternative search
	 *        knobs of @p Options; same contract, same results for every knob.
	 *
	 * @tparam Symbol The projected symbol type.
	 * @tparam Options The knobs applied (@ref atl::experimental::sorted_dictionary_options).
	 */
	template<std::totally_ordered Symbol, sorted_dictionary_options Options = {}>
	class sorted_dictionary_by_parts_build {
		std::vector<Symbol> sym_;
		std::vector<uint32> word_end_;	   // n + 1 offsets into sym_ (the build asserts they fit)
		std::vector<uint32> neighbor_lcp_; // n - 1 entries
		direct_rmq<uint32> rmq_;		   // views neighbor_lcp_
		/**
		 * With is_descent_lcp: lcp(word l, word mid) and lcp(word mid, word r) for the one
		 * descent interval (l, r) whose midpoint is mid.
		 */
		std::vector<uint32> left_lcp_, right_lcp_;

		/** @brief As the library class: `==` scans except for byte symbols. */
		static constexpr bool is_equality_scan = !(std::integral<Symbol> && sizeof(Symbol) == 1);

		[[nodiscard]] static constexpr bool is_same(Symbol const &a, Symbol const &b) {
			if constexpr (is_equality_scan)
				return a == b;
			else
				return !(a < b) && !(b < a);
		}

		struct parts {
			std::vector<Symbol> sym;
			std::vector<uint32> end{0};
			std::vector<uint32> lcp;
		};

		/**
		 * @brief The parts build, writing into @p sym, @p end (holding the
		 *        leading 0) and @p lcp; asserts sortedness.
		 */
		template<typename R, typename Proj>
			requires sortable_word_range<R, Proj> && std::same_as<word_symbol_t<R, Proj>, Symbol>
		static constexpr void fill(R &&words, Proj &proj, std::vector<Symbol> &sym, std::vector<uint32> &end,
								   std::vector<uint32> &lcp) {
			constexpr bool owns_words = !std::is_lvalue_reference_v<R> && !std::ranges::view<std::remove_cvref_t<R>>;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(words) < (uint64(1) << 32));
				atl_assert(std::ranges::size(words) < end.max_size());
				end.reserve(std::size_t(std::ranges::size(words)) + 1);
			}
			if constexpr (Options.is_sym_reserved && std::ranges::forward_range<R>)
				sym.reserve(atl::detail::total_word_size(words));
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(end.size() < (uint64(1) << 32));
				if constexpr (owns_words)
					atl::detail::append_projected(sym, std::move(w), proj);
				else
					atl::detail::append_projected(sym, w, proj);
				atl_assert(sym.size() < (uint64(1) << 32)); // offsets, lcps and match lengths are uint32
				end.push_back(uint32(sym.size()));
			}
			atl_assert(end.size() <= (uint64(1) << 32)); // word ids are uint32
			uint32 const n = uint32(end.size() - 1);
			lcp.reserve(n == 0 ? 0 : n - 1);
			for (uint32 i = 0; i + 1 < n; ++i) {
				std::size_t const a = end[i], b = end[i + 1];
				std::size_t const la = b - a, lb = end[i + 2] - b, limit = (std::min)(la, lb);
				std::size_t h = 0;
				if constexpr (Options.is_novector) {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#pragma GCC novector
#endif
					while (h < limit && is_same(sym[a + h], sym[b + h])) ++h;
				} else {
					while (h < limit && is_same(sym[a + h], sym[b + h])) ++h;
				}
				atl_assert(h == la || (h < lb && sym[a + h] < sym[b + h]));
				lcp.push_back(uint32(h));
			}
		}

		template<typename R, typename Proj>
			requires sortable_word_range<R, Proj> && std::same_as<word_symbol_t<R, Proj>, Symbol>
		[[nodiscard]] static constexpr parts build(R &&words, Proj &proj) {
			parts p;
			fill(std::forward<R>(words), proj, p.sym, p.end, p.lcp);
			return p;
		}

		/**
		 * @brief The descent-tree LCPs for @ref atl::experimental::sorted_dictionary_options::is_descent_lcp:
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

		constexpr explicit sorted_dictionary_by_parts_build(parts &&p)
			: sym_(std::move(p.sym)),
			  word_end_(std::move(p.end)),
			  neighbor_lcp_(std::move(p.lcp)),
			  rmq_(neighbor_lcp_) {
			if constexpr (Options.is_descent_lcp) build_descent_lcp();
		}

	public:
		/**
		 * @brief Build over @p words, lexicographically sorted under @p proj
		 *        (as atl::sorted_dictionary's constructor).
		 * @pre @p words sorted by their projected symbols; fewer than `2^32`
		 *      words and symbols in total (asserted)
		 * @complexity O(L + n) over L total symbols
		 */
		template<typename R, typename Proj = std::identity>
			requires sortable_word_range<R, Proj> && std::same_as<word_symbol_t<R, Proj>, Symbol>
		constexpr explicit sorted_dictionary_by_parts_build(R &&words, Proj proj = {})
			: sorted_dictionary_by_parts_build(build(std::forward<R>(words), proj)) {}

		/** @brief Copying is disabled because the RMQ borrows the LCP buffer. */
		sorted_dictionary_by_parts_build(sorted_dictionary_by_parts_build const &) = delete;
		/** @brief Copy assignment is disabled. */
		sorted_dictionary_by_parts_build &operator=(sorted_dictionary_by_parts_build const &) = delete;
		/** @brief Transfer buffers and their RMQ view; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary_by_parts_build(sorted_dictionary_by_parts_build &&) = default;
		/** @brief Transfer ownership; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary_by_parts_build &operator=(sorted_dictionary_by_parts_build &&) = default;

		/**
		 * @brief The words having @p x as a prefix (as
		 *        atl::sorted_dictionary::prefix_interval).
		 * @pre `x.size() < 2^32` (asserted)
		 * @complexity O(m + log n)
		 */
		template<typename Q, typename QProj = std::identity>
			requires sortable_text<Q, QProj> && std::totally_ordered_with<text_symbol_t<Q, QProj>, Symbol>
		[[nodiscard]] constexpr index_interval prefix_interval(Q &&x, QProj proj = {}) const {
			auto const first = std::ranges::begin(x);
			atl_assert(std::ranges::size(x) < (uint64(1) << 32)); // 32-bit match lengths
			uint32 const m = uint32(std::ranges::size(x));
			uint32 const n = uint32(word_end_.size() - 1);
			if (n == 0) return {0, 0};

			auto const extend = [&](uint32 i, uint32 h) -> std::pair<uint32, int> {
				std::size_t const off = word_end_[i];
				uint32 const len = word_end_[i + 1] - word_end_[i];
				uint32 const limit = (std::min)(m, len);
				if constexpr (is_equality_scan) {
					while (h < limit && atl::detail::symbol_equal(sym_[off + h], std::invoke(proj, first[h]))) ++h;
					if (h == m) return {h, 0};
					if (h == len) return {h, -1};
					return {h, atl::detail::symbol_less(sym_[off + h], std::invoke(proj, first[h])) ? -1 : +1};
				} else {
					for (; h < limit; ++h) {
						auto &&q = std::invoke(proj, first[h]);
						if (atl::detail::symbol_less(sym_[off + h], q)) return {h, -1};
						if (atl::detail::symbol_less(q, sym_[off + h])) return {h, +1};
					}
					return h == m ? std::pair{h, 0} : std::pair{h, -1};
				}
			};
			auto const e0 = extend(0, 0);
			auto const en = n > 1 ? extend(n - 1, 0) : e0;
			if (en.second < 0) return {n, n};

			uint32 begin = 0, begin_lcp = e0.first, out = en.second > 0 ? n - 1 : n;
			if (e0.second < 0) {
				uint32 l = 0, r = n - 1;
				uint32 llcp = e0.first, rlcp = en.first;
				while (r - l > 1) {
					uint32 const mid = l + (r - l) / 2;
					if (llcp >= rlcp) {
						uint32 h = 0;
						if constexpr (Options.is_descent_lcp)
							h = left_lcp_[mid];
						else
							h = rmq_.fold(l, mid);
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
						uint32 h = 0;
						if constexpr (Options.is_descent_lcp)
							h = right_lcp_[mid];
						else
							h = rmq_.fold(mid, r);
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
					if (rlcp < m) out = r;
				}
				begin = r;
				begin_lcp = rlcp;
			}
			if (begin_lcp < m) return {begin, begin};
			if (en.second == 0) return {begin, n};
			uint32 lo = begin, hi = out; // word lo is in; hi == n or word hi is out
			if constexpr (Options.is_linear_upper_scan) {
				// word begin + k + 1 is in iff every neighbor lcp from begin to begin + k is >= m
				uint32 const stop = hi - begin > 17 ? begin + 17 : hi;
				for (; lo + 1 < stop; ++lo)
					if (neighbor_lcp_[lo] < m) return {begin, lo + 1};
			} else {
				for (uint32 step = 1; step <= 16 && step < hi - begin; step *= 2) {
					if (rmq_.fold(begin, begin + step) < m) {
						hi = begin + step;
						break;
					}
					lo = begin + step;
				}
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

	/** @brief The knobs of @ref atl::experimental::sorted_dictionary_by_inplace_build; the default is the library class. */
	struct sorted_dictionary_inplace_options {
		/** @brief Scan up to 16 adjacent LCP entries, then bisect.
		 * No consistent win: inplace char misses are 219/232 ms with gallops
		 * versus 226/240 ms with the scan (-O2/native, Xeon Platinum 8168,
		 * GCC 15.2.0, 2026-11-03), despite lower hit times. */
		bool is_linear_upper_scan = false;

		/** @brief Knob-wise equality. */
		constexpr bool operator==(sorted_dictionary_inplace_options const &) const = default;
	};

	/** @brief Member-built dictionary with an optional linear upper search.
	 * @tparam Symbol Totally ordered projected symbol type.
	 * @tparam Options Upper-search policy.
	 * @note The default matches the library algorithm; independent code layout
	 * can change timings. See the file contract for the measurement environment. */
	template<std::totally_ordered Symbol, sorted_dictionary_inplace_options Options = {}>
	class sorted_dictionary_by_inplace_build {
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
		constexpr explicit sorted_dictionary_by_inplace_build(R &&words, Proj proj = {}) {
			// the words of an rvalue container are ours: their symbols are moved, not copied
			constexpr bool owns_words = !std::is_lvalue_reference_v<R> && !std::ranges::view<std::remove_cvref_t<R>>;
			if constexpr (std::ranges::sized_range<R>) {
				atl_assert(std::ranges::size(words) < (uint64(1) << 32));
				atl_assert(std::ranges::size(words) < word_end_.max_size());
				word_end_.reserve(std::size_t(std::ranges::size(words)) + 1);
			}
			if constexpr (std::ranges::forward_range<R>)
				sym_.reserve(atl::detail::total_word_size(words));
			for (auto &&w : atl::detail::input_elements(words)) {
				atl_assert(word_end_.size() < (uint64(1) << 32));
				if constexpr (owns_words)
					atl::detail::append_projected(sym_, std::move(w), proj);
				else
					atl::detail::append_projected(sym_, w, proj);
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
		sorted_dictionary_by_inplace_build(sorted_dictionary_by_inplace_build const &) = delete;
		/** @brief Copy assignment is disabled. */
		sorted_dictionary_by_inplace_build &operator=(sorted_dictionary_by_inplace_build const &) = delete;
		/** @brief Transfer buffers and their RMQ view; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary_by_inplace_build(sorted_dictionary_by_inplace_build &&) = default;
		/** @brief Transfer ownership; the source may only be assigned or destroyed. */
		constexpr sorted_dictionary_by_inplace_build &operator=(sorted_dictionary_by_inplace_build &&) = default;

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
						auto &&q = std::invoke(proj, first[h]); // a reference: tokens are not copied
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
			if constexpr (Options.is_linear_upper_scan) {
				// word begin + k + 1 is in iff every neighbor lcp from begin to begin + k is >= m
				uint32 const stop = hi - begin > 17 ? begin + 17 : hi;
				for (; lo + 1 < stop; ++lo)
					if (neighbor_lcp_[lo] < m) return {begin, lo + 1};
			} else {
				for (uint32 step = 1; step <= 16 && step < hi - begin; step *= 2) {
					if (rmq_.fold(begin, begin + step) < m) {
						hi = begin + step;
						break;
					}
					lo = begin + step;
				}
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
}
