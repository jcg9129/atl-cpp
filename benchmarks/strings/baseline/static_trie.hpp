#pragma once
#include <atl/strings/comparable_text.hpp>
#include <atl/strings/detail/symbols.hpp>
#include <atl/strings/trie.hpp>
#include <atl/types/alias.hpp>
#include <atl/utils/assert.hpp>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

/** @brief Benchmark-local frozen trie contenders. */
namespace atl::baseline {
	/** @brief Frozen trie sorting word-id slices with an indirect symbol comparator.
	 * @tparam Alphabet Exclusive symbol bound, at most 2^32.
	 * @note Contracts and query layout match atl::static_trie. Unlike integer-key
	 *       partitioning, this sorts every live slice and emits one level per pass.
	 *       Prefix500 build takes 598/602 ms versus library 82.5/84.3 ms,
	 *       Xeon Platinum 8280, GCC 15.2.0, RelWithDebInfo, assertions off,
	 *       -O2/native, 2026-10-15 (benchmarks/strings/static_trie.cpp).
	 */
	template<std::size_t Alphabet>
		requires(Alphabet <= (uint64(1) << 32))
	class static_trie_by_slice_comparison {
		static constexpr uint32 nil = uint32(-1);

	public:
		/** @brief One child edge: the symbol and the node it leads to. @ref atl::baseline::static_trie_by_slice_comparison::children
		 * spans are sorted ascending by `symbol`. */
		struct edge {
			uint32 symbol; ///< Numeric edge label.
			uint32 target; ///< Child node id.
		};

		/**
		 * @brief A handle to a node (@ref atl::baseline::static_trie_by_slice_comparison::root, @ref atl::baseline::static_trie_by_slice_comparison::walk, @ref atl::baseline::static_trie_by_slice_comparison::children). Ids are
		 * dense `[0, node_count())` in lexicographic preorder: the root is 0
		 * and every subtree occupies the contiguous id interval
		 * `[n, subtree_end(n))`.
		 */
		using node_id = uint32;

	private:
		std::vector<uint32> offset_;	   // node -> first edge; node_count()+1 entries
		std::vector<edge> edges_;		   // all child edges, per-node runs sorted by symbol
		std::vector<uint32> word_count_;   // occurrences ending exactly at the node
		std::vector<uint32> prefix_count_; // occurrences whose word passes through the node
		std::vector<uint32> first_word_;   // first word (in lexicographic order) in the subtree
		std::vector<uint32> subtree_end_;  // one past the last preorder id in the subtree
		std::vector<uint32> word_order_;   // input index of each lexicographic word position

		[[nodiscard]] constexpr uint32 step(uint32 node, uint32 sym) const {
			if (offset_[node] == offset_[node + 1]) return nil;
			edge const *lo = edges_.data() + offset_[node];
			edge const *hi = edges_.data() + offset_[node + 1];
			if (hi - lo <= 8) {
				for (; lo != hi; ++lo)
					if (lo->symbol == sym) return lo->target;
				return nil;
			}
			while (lo < hi) {
				edge const *mid = lo + (hi - lo) / 2;
				if (mid->symbol < sym)
					lo = mid + 1;
				else
					hi = mid;
			}
			return lo != edges_.data() + offset_[node + 1] && lo->symbol == sym ? lo->target : nil;
		}

		template<typename R, typename Proj>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr uint32 node_of(R &&w, Proj &proj) const {
			uint32 cur = 0;
			for (auto &&e : atl::detail::input_elements(w)) {
				auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
				if (!atl::detail::symbol_below(s, Alphabet)) return nil;
				cur = step(cur, uint32(s));
				if (cur == nil) return nil;
			}
			return cur;
		}

	public:
		/** @brief The empty dictionary: just a root, every count 0. */
		constexpr static_trie_by_slice_comparison()
			: offset_{0, 0}, word_count_{0}, prefix_count_{0}, first_word_{0}, subtree_end_{1} {}

		/**
		 * @brief Build from @p words, each element a word whose symbols are seen
		 *        through @p proj (as in @ref atl::trie::insert). Duplicate words are
		 *        counted with multiplicity.
		 * @complexity O(n + L log(1 + n)) time for n words and L total symbols
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word_range<R, Proj>
		constexpr explicit static_trie_by_slice_comparison(R &&words, Proj proj = {}) {
			// materialize the symbol sequences once: the level passes below
			// re-read each word once per depth, which arbitrary input ranges
			// (and per-symbol projections) should not pay repeatedly
			std::vector<uint32> sym;
			std::vector<uint32> word_end{0};
			if constexpr (std::ranges::sized_range<R>) {
				auto const count = std::ranges::size(words);
				atl_assert(count <= nil && count < word_end.max_size());
				word_end.reserve(std::size_t(count) + 1);
			}
			for (auto &&w : atl::detail::input_elements(words)) {
				for (auto &&e : atl::detail::input_elements(w)) {
					auto const s = std::invoke(proj, std::forward<decltype(e)>(e));
					atl_assert(atl::detail::symbol_below(s, Alphabet));
					atl_assert(sym.size() < uint64(nil) - 1);
					sym.push_back(uint32(s));
				}
				atl_assert(word_end.size() <= nil && word_end.size() < word_end.max_size());
				word_end.push_back(uint32(sym.size()));
			}
			uint32 const n = uint32(word_end.size() - 1);

			// DFS radix partition. A frame is a slice of `ids` = the words whose
			// prefix reaches this node; the node's number is assigned at pop
			// (preorder), so the edge that led here is patched then. The
			// finished `ids` is the lexicographic word order, kept as
			// word_order_.
			std::vector<uint32> &ids = word_order_;
			ids.resize(n);
			for (uint32 i = 0; i < n; ++i) ids[i] = i;

			struct frame {
				uint32 lo, hi, depth;
				uint32 parent_edge; // index into edges_, nil at the root
			};

			std::vector<frame> stack{{0, n, 0, nil}};
			auto const symbol_at = [&](uint32 w, uint32 d) { return sym[word_end[w] + d]; };
			auto const length = [&](uint32 w) { return word_end[w + 1] - word_end[w]; };
			while (!stack.empty()) {
				auto const [lo, hi, depth, parent_edge] = stack.back();
				stack.pop_back();
				uint32 const id = uint32(word_count_.size());
				if (parent_edge != nil) edges_[parent_edge].target = id;
				offset_.push_back(uint32(edges_.size()));
				prefix_count_.push_back(hi - lo);
				first_word_.push_back(lo); // the node's slice [lo, hi) IS its word interval
				auto const rest = std::ranges::partition(ids.begin() + lo, ids.begin() + hi,
														 [&](uint32 w) { return length(w) == depth; });
				uint32 const live = uint32(rest.begin() - ids.begin());
				word_count_.push_back(live - lo);
				if (live == hi) continue;
				std::ranges::sort(ids.begin() + live, ids.begin() + hi,
								  [&](uint32 a, uint32 b) { return symbol_at(a, depth) < symbol_at(b, depth); });
				uint32 const first_edge = uint32(edges_.size());
				for (uint32 a = live; a < hi;) {
					uint32 const s = symbol_at(ids[a], depth);
					uint32 b = a + 1;
					while (b < hi && symbol_at(ids[b], depth) == s) ++b;
					edges_.push_back({s, nil}); // target patched when the child pops
					a = b;
				}
				// push runs in reverse so the smallest symbol pops first: node
				// ids come out in lexicographic preorder
				uint32 run_hi = hi;
				for (uint32 e = uint32(edges_.size()); e-- > first_edge;) {
					uint32 run_lo = run_hi;
					while (run_lo > live && symbol_at(ids[run_lo - 1], depth) == edges_[e].symbol) --run_lo;
					stack.push_back({run_lo, run_hi, depth + 1, e});
					run_hi = run_lo;
				}
			}
			offset_.push_back(uint32(edges_.size()));
			// a node's subtree ends where its last (largest-symbol, so
			// largest-id) child's does; children have larger ids, so one
			// reverse preorder pass settles every node
			uint32 const nodes = uint32(word_count_.size());
			subtree_end_.resize(nodes);
			for (uint32 v = nodes; v-- > 0;)
				subtree_end_[v] = offset_[v] == offset_[v + 1] ? v + 1 : subtree_end_[edges_[offset_[v + 1] - 1].target];
		}

		/** @brief Multiplicity of the exact word @p word. @complexity O(len·log degree) */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::size_t count(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v == nil ? 0 : word_count_[v];
		}

		/** @brief Whether @p word occurs at least once. */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr bool contains(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v != nil && word_count_[v] != 0;
		}

		/**
		 * @brief The node spelling @p word, or `std::nullopt` if its path is absent.
		 *        The empty word always returns root(), even for an empty dictionary.
		 * @complexity O(len·log degree)
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::optional<node_id> find(R &&word, Proj proj = {}) const {
			uint32 const v = node_of(word, proj);
			return v == nil ? std::nullopt : std::optional<node_id>(v);
		}

		/**
		 * @brief Number of stored occurrences whose word begins with @p prefix.
		 * @complexity O(len·log degree)
		 */
		template<typename R, typename Proj = std::identity>
			requires trie_word<R, Proj>
		[[nodiscard]] constexpr std::size_t count_with_prefix(R &&prefix, Proj proj = {}) const {
			uint32 const v = node_of(prefix, proj);
			return v == nil ? 0 : prefix_count_[v];
		}

		/** @brief The root handle — the empty prefix (id 0). */
		[[nodiscard]] constexpr node_id root() const { return 0; }

		/** @brief The child of @p n along @p sym, or `std::nullopt`. */
		template<std::integral Sym>
		[[nodiscard]] constexpr std::optional<node_id> walk(node_id n, Sym sym) const {
			atl_assert(n < word_count_.size());
			if (!atl::detail::symbol_below(sym, Alphabet)) return std::nullopt;
			uint32 const c = step(n, uint32(sym));
			return c == nil ? std::nullopt : std::optional<node_id>(c);
		}

		/**
		 * @brief @p n's child edges, **sorted ascending by symbol** — the ordered
		 * traversal the dynamic trie cannot offer (k-th word, completions in
		 * lexicographic order, merges over two tries).
		 */
		[[nodiscard]] constexpr std::span<edge const> children(node_id n) const {
			atl_assert(n < word_count_.size());
			return std::span<edge const>(edges_).subspan(offset_[n], offset_[n + 1] - offset_[n]);
		}

		/** @brief Occurrences of the word ending exactly at @p n. */
		[[nodiscard]] constexpr std::size_t word_count(node_id n) const {
			atl_assert(n < word_count_.size());
			return word_count_[n];
		}

		/**
		 * @brief The words carrying @p n's prefix, as the half-open interval
		 *        `[begin, end)` of positions in the LEXICOGRAPHIC word order
		 *        (the multiset order, duplicates counted; `end - begin ==`
		 *        @ref atl::baseline::static_trie_by_slice_comparison::prefix_count). @ref atl::baseline::static_trie_by_slice_comparison::word_order maps a position back to
		 *        its input index. Walking a query prefix with @ref atl::baseline::static_trie_by_slice_comparison::walk and
		 *        reading its interval answers "which dictionary words start
		 *        with x" in O(len·log degree) total.
		 */
		[[nodiscard]] constexpr index_interval word_interval(node_id n) const {
			atl_assert(n < first_word_.size());
			return {first_word_[n], first_word_[n] + prefix_count_[n]};
		}

		/**
		 * @brief The input indices of the words in lexicographic order: entry
		 *        `i` is the input index of the word at position `i` of every
		 *        @ref atl::baseline::static_trie_by_slice_comparison::word_interval. Equal words come in unspecified order
		 *        (the build's partition is not stable).
		 */
		[[nodiscard]] constexpr std::span<uint32 const> word_order() const { return word_order_; }

		/**
		 * @brief One past the last node id in @p n's subtree: the subtree is
		 *        exactly the preorder ids `[n, subtree_end(n))`, the interval
		 *        Euler-tour style range structures over node ids index.
		 */
		[[nodiscard]] constexpr node_id subtree_end(node_id n) const {
			atl_assert(n < subtree_end_.size());
			return subtree_end_[n];
		}

		/** @brief Occurrences whose word passes through @p n (i.e. carries its prefix). */
		[[nodiscard]] constexpr std::size_t prefix_count(node_id n) const {
			atl_assert(n < prefix_count_.size());
			return prefix_count_[n];
		}

		/** @brief Number of stored words counted with multiplicity. */
		constexpr std::size_t size() const { return prefix_count_[0]; }

		/** @brief Whether no input word was supplied. */
		constexpr bool empty() const { return size() == 0; }

		/** @brief Number of trie nodes, root included. */
		[[nodiscard]] constexpr std::size_t node_count() const { return word_count_.size(); }

		/** @brief Number of edges (= node_count() - 1 on a nonempty dictionary). */
		[[nodiscard]] constexpr std::size_t edge_count() const { return edges_.size(); }
	};
}
